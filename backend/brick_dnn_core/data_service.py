"""Data preparation (the Data tab): imported files, preparations, previews, exports.

The work is done by :mod:`brickdnn.data` (pandas); this module stores files, checks who may
use what and turns preparations into training datasets. Files::

    DATA_PREP_DIR/sources/<id>/original.csv   the uploaded or downloaded file, unchanged
                                              (original.xlsx, original.json.gz ... for others)
    DATA_PREP_DIR/sources/<id>/table/         the typed table (brickdnn.data.store)

pandas is only imported by the functions that need it, so the server starts (and code
generation and training keep working) without it.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit, urlunsplit

from django.conf import settings
from django.db import transaction
from django.db.models.signals import post_delete
from django.dispatch import receiver
from django.utils.text import slugify

from brickdnn.data import (
    COMPRESSED_SUFFIXES,
    FORMAT_SUFFIXES,
    MISSING_DEPENDENCIES,
    dependencies_installed,
    file_types,
)
from brickdnn.training import DatasetError, DatasetInfo, inspect_csv

from . import budgets, url_import
from . import training_service as training
from .models import DataPrep, Dataset, DataSource
from .retention import touch

if TYPE_CHECKING:
    from brickdnn.data.pipeline import PipelineResult
    from brickdnn.data.table import Table

logger = logging.getLogger(__name__)

DISABLED_REASON = "Data preparation is disabled on this server."
WRONG_FILE_MESSAGE = (
    "Upload a CSV, Excel, OpenDocument, JSON, Parquet or Feather file (.csv, .tsv, .txt, "
    ".xlsx, .xls, .ods, .json, .jsonl, .parquet, .feather), also packed as .gz, .bz2, .xz "
    "or .zip."
)
URL_DISABLED_MESSAGE = "Importing from links is switched off on this server."
MISSING_FILES_MESSAGE = "The imported data of this source is missing. Import the file again."
GONE_MESSAGE = "This table does not exist (any more)."
ORIGINAL_FILE = "original.csv"
TABLE_DIR = "table"
# Form fields of an import, see brickdnn.data.io.ImportOptions.
OPTION_FIELDS = ("delimiter", "decimal", "header", "encoding", "missing_values", "sheet")
MIN_SAMPLE_ROWS = 10
DOWNLOAD_FORMATS = ("csv", "recipe", "python")


class DataInputError(ValueError):
    """Invalid request data or data that cannot be used. The message is shown to the user.

    ``issues`` are the readiness issues when data is not ready for training.
    """

    def __init__(
        self,
        message: str,
        issues: list[dict[str, Any]] | None = None,
        sheets: list[str] | None = None,
    ):
        super().__init__(message)
        self.issues = issues
        # The sheets of a workbook whose (chosen) sheet could not be imported.
        self.sheets = sheets


@contextlib.contextmanager
def _user_errors() -> Iterator[None]:
    """Turn the errors of brickdnn.data / brickdnn.training into :class:`DataInputError`."""
    from brickdnn.data.table import DataError

    try:
        yield
    except (DataError, DatasetError) as exc:
        raise DataInputError(str(exc), sheets=getattr(exc, "sheets", None)) from None


# --------------------------------------------------------------------------- #
# Status and paths
# --------------------------------------------------------------------------- #


def unavailable_reason() -> str | None:
    """Why data preparation cannot be used right now (None if it can)."""
    if not settings.DATA_PREP_ENABLED:
        return DISABLED_REASON
    if not dependencies_installed():
        return MISSING_DEPENDENCIES
    return None


def status() -> dict[str, Any]:
    """``DataStatus`` (frontend/src/api/types.ts): what the UI needs before offering the tab."""
    reason = unavailable_reason()
    return {
        "enabled": bool(settings.DATA_PREP_ENABLED),
        "available": reason is None,
        "reason": reason,
        "max_upload_mb": settings.DATA_MAX_UPLOAD_MB,
        "max_rows": settings.DATA_MAX_ROWS,
        "max_columns": settings.DATA_MAX_COLUMNS,
        "default_sample_rows": settings.DATA_SAMPLE_ROWS,
        "max_sample_rows": settings.DATA_MAX_SAMPLE_ROWS,
        "login_required": True,
        # File name endings the import reads (formats whose library is missing left out).
        "file_types": file_types(),
        "url_import": bool(settings.DATA_URL_IMPORT_ENABLED),
    }


def step_catalog() -> dict[str, Any]:
    from brickdnn.data.steps import catalog

    return catalog()


def prep_dir() -> Path:
    return Path(settings.DATA_PREP_DIR)


def source_dir(source_id: int) -> Path:
    return prep_dir() / "sources" / str(source_id)


def max_upload_bytes() -> int:
    return int(float(settings.DATA_MAX_UPLOAD_MB) * 1024 * 1024)


# --------------------------------------------------------------------------- #
# Imports (data sources)
# --------------------------------------------------------------------------- #


def import_source(user, upload, fields: dict[str, Any]) -> DataSource:
    """Store and read an uploaded data file. ``fields`` are the form fields (``name`` and the
    import options). Raises :class:`DataInputError`, or
    :class:`budgets.StorageQuotaExceeded` if it does not fit into the account's storage."""
    filename = Path(upload.name or "").name
    budgets.check_storage(user, int(upload.size or 0))

    def save(work: Path) -> tuple[Path, str]:
        if not _known_suffix(filename):
            raise DataInputError(WRONG_FILE_MESSAGE)
        original = work / _original_name(filename)
        with original.open("wb") as out:
            for chunk in upload.chunks():
                out.write(chunk)
        return original, filename

    return _import(user, save, fields)


def import_url(user, url: Any, fields: dict[str, Any]) -> DataSource:
    """Download the file at ``url`` (see :mod:`url_import` for the safety checks) and read it
    like an upload. Raises :class:`DataInputError`."""
    if not settings.DATA_URL_IMPORT_ENABLED:
        raise DataInputError(URL_DISABLED_MESSAGE)
    budgets.check_storage(user)  # the size is known after the download
    try:
        given = url_import.check_url(url)
    except url_import.LinkError as exc:
        raise DataInputError(str(exc)) from None

    def save(work: Path) -> tuple[Path, str]:
        try:
            download = url_import.fetch(
                given,
                work,
                max_bytes=max_upload_bytes(),
                allow_private=bool(settings.DATA_URL_ALLOW_PRIVATE),
            )
        except url_import.LinkError as exc:
            raise DataInputError(str(exc)) from None
        original = work / _original_name(download.filename)
        os.replace(download.path, original)
        return original, download.filename

    return _import(user, save, fields, url=given)


def _shown_url(url: str) -> str:
    """The link as stored and shown, e.g. to people a preparation is shared with: without
    its query and fragment, which can hold secrets (signed links, access tokens)."""
    parts = urlsplit(url)
    hidden = "?…" if parts.query else ""
    return urlunsplit(parts._replace(query="", fragment="")) + hidden


def _known_suffix(filename: str) -> bool:
    return Path(filename).suffix.lower() in {*FORMAT_SUFFIXES, *COMPRESSED_SUFFIXES}


def original_file(source_id: int) -> Path | None:
    """The stored original of a source (``original.csv``, ``original.xlsx`` ...), if any."""
    directory = source_dir(source_id)
    if not directory.is_dir():
        return None
    found = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and (path.name == "original" or path.name.startswith("original."))
    )
    return found[0] if found else None


def _original_name(filename: str) -> str:
    """``original`` plus the known endings of ``filename`` (``original.csv.gz``)."""
    suffixes = [s.lower() for s in Path(filename).suffixes[-2:]]
    if suffixes and suffixes[-1] in COMPRESSED_SUFFIXES:
        packed = suffixes[-1]
        inner = suffixes[-2] if len(suffixes) == 2 and suffixes[-2] in FORMAT_SUFFIXES else ""
        return f"original{inner}{packed}"
    if suffixes and suffixes[-1] in FORMAT_SUFFIXES:
        return f"original{suffixes[-1]}"
    return "original"


def _import(user, save, fields: dict[str, Any], url: str | None = None) -> DataSource:
    """Store the file ``save(work_folder)`` writes, read it and create the source."""
    from brickdnn.data.io import ImportOptions

    requested = {key: fields[key] for key in OPTION_FIELDS if fields.get(key) not in (None, "")}
    with _user_errors():
        options = ImportOptions.from_dict(requested)

    # Work in a temporary folder, so that a half-imported file is never used.
    root = prep_dir() / "sources"
    root.mkdir(parents=True, exist_ok=True)
    _remove_abandoned(root, ".import-*")
    work = root / f".import-{uuid.uuid4().hex}"
    work.mkdir()
    source = None
    try:
        original, filename = save(work)
        budgets.check_storage(user, original.stat().st_size)
        name = fields.get("name")
        name = (str(name or "").strip() or _stem(filename) or "Data")[:120]
        read = {
            "path": str(original),
            "filename": filename,
            "options": options.to_dict(),
            "out": str(work / TABLE_DIR),
            "max_rows": settings.DATA_MAX_ROWS,
            "max_columns": settings.DATA_MAX_COLUMNS,
            "max_bytes": max_upload_bytes(),
        }
        if original.stat().st_size >= settings.DATA_IMPORT_PROCESS_MB * 1024 * 1024:
            answer = _read_in_process(read)
        else:
            from brickdnn.data.import_job import run as read_here

            answer = read_here(read)
        if not answer["ok"]:
            raise DataInputError(answer["error"], sheets=answer.get("sheets"))
        n_rows, columns = answer["n_rows"], answer["columns"]
        stored_options = {key: options.to_dict().get(key) for key in OPTION_FIELDS}
        stored_report = answer["report"]
        if url:
            stored_report["url"] = _shown_url(url)
        source = DataSource.objects.create(
            owner=user,
            name=name,
            filename=filename[:255],
            size_bytes=original.stat().st_size,
            n_rows=n_rows,
            n_columns=len(columns),
            columns=columns,
            import_options=stored_options,
            report=stored_report,
        )
        final = source_dir(source.pk)
        shutil.rmtree(final, ignore_errors=True)  # left over from an earlier database
        os.replace(work, final)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        if source is not None:
            source.delete()
        raise
    return source


TOO_LITTLE_MEMORY_MESSAGE = (
    "Reading this file needs more memory than the server has. "
    "Import a smaller file (fewer rows or columns, or split it and append the parts in a "
    "preparation), or ask the operator to raise BRICKDNN_WEB_MEMORY."
)
IMPORT_TIMEOUT_MESSAGE = (
    "Reading this file took longer than {seconds} seconds. Import a smaller file or "
    "split it and append the parts in a preparation."
)


def _read_in_process(job: dict[str, Any]) -> dict[str, Any]:
    """Read a large file in a process of its own (brickdnn.data.import_job): if it needs
    more memory than there is, the kernel kills that process and not the web server."""
    timeout = settings.DATA_IMPORT_TIMEOUT
    try:
        result = subprocess.run(
            [sys.executable, "-m", "brickdnn.data.import_job"],
            input=json.dumps(job),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=training._subprocess_env(),
            cwd=str(settings.BASE_DIR),
        )
    except subprocess.TimeoutExpired:
        raise DataInputError(IMPORT_TIMEOUT_MESSAGE.format(seconds=timeout)) from None
    if result.returncode in (-signal.SIGKILL, 137) or "MemoryError" in result.stderr:
        raise DataInputError(TOO_LITTLE_MEMORY_MESSAGE)
    for line in reversed(result.stdout.strip().splitlines()):
        with contextlib.suppress(ValueError):
            answer = json.loads(line)
            if isinstance(answer, dict) and "ok" in answer:
                return answer
    logger.error("Import process failed (%s): %s", result.returncode, result.stderr[-2000:])
    raise DataInputError("The file could not be read (the import stopped unexpectedly).")


def _stem(filename: str) -> str:
    """The file name without its known endings: ``sales`` for ``sales.csv.gz``."""
    path = Path(filename)
    while path.suffix.lower() in {*FORMAT_SUFFIXES, *COMPRESSED_SUFFIXES}:
        path = path.with_suffix("")
    return path.name.strip()


def _remove_abandoned(folder: Path, pattern: str) -> None:
    """Work folders/files of imports or downloads that were interrupted (e.g. a restart)."""
    cutoff = time.time() - 6 * 3600
    for path in folder.glob(pattern):
        with contextlib.suppress(OSError):
            if path.stat().st_mtime < cutoff:
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    path.unlink()


def _load_table(source: DataSource) -> Table:
    """The stored table of ``source``. Raises DataError (shown in a step or as a 400)."""
    from brickdnn.data.store import load_table_cached
    from brickdnn.data.table import DataError

    directory = source_dir(source.pk) / TABLE_DIR
    if not (directory / "table.json").is_file():
        raise DataError(MISSING_FILES_MESSAGE)
    touch(source)
    return load_table_cached(directory)


def source_table(source: DataSource) -> Table:
    """The stored table of ``source`` (counts as a use). Raises :class:`DataInputError`."""
    with _user_errors():
        return _load_table(source)


def source_info(source: DataSource) -> dict[str, Any]:
    """How to read the file of ``source`` again (``SourceInfo`` in brickdnn.data.export)."""
    from brickdnn.data.io import DEFAULT_MISSING
    from brickdnn.data.readers import SPREADSHEET_ERRORS, WORKBOOK_FORMATS

    report = source.report if isinstance(source.report, dict) else {}
    options = source.import_options if isinstance(source.import_options, dict) else {}
    missing = options.get("missing_values")
    missing = list(missing) if isinstance(missing, list) else list(DEFAULT_MISSING)
    fmt = report.get("format") or "csv"
    if fmt in WORKBOOK_FORMATS:
        missing += [token for token in SPREADSHEET_ERRORS if token not in missing]
    info = {
        "id": source.pk,
        "name": source.name,
        "filename": source.filename,
        "format": fmt,
        "delimiter": report.get("delimiter", ","),
        "decimal": report.get("decimal", "."),
        "header": bool(report.get("header", True)),
        "encoding": report.get("encoding", "utf-8"),
        "missing_values": missing,
        "columns": [
            {"name": c.get("name"), "kind": c.get("kind")}
            for c in (source.columns or [])
            if isinstance(c, dict)
        ],
    }
    for key in ("sheet", "json_path", "compression", "member", "url"):
        if report.get(key):
            info[key] = report[key]
    return info


def preps_using(source: DataSource) -> list[DataPrep]:
    """Preparations that use ``source`` as their main source (it cannot be deleted then)."""
    return list(source.preps.order_by("name", "id"))


def in_use_message(preps: list[DataPrep]) -> str:
    names = ", ".join(f"“{prep.name}”" for prep in preps[:5]) + (" …" if len(preps) > 5 else "")
    if len(preps) == 1:
        return f"Used by the preparation {names}; delete it first."
    return f"Used by the preparations {names}; delete them first."


# --------------------------------------------------------------------------- #
# Preparations
# --------------------------------------------------------------------------- #


def clean_steps(value: Any) -> list[dict[str, Any]]:
    from brickdnn.data.pipeline import clean_steps as check

    with _user_errors():
        return check(value)


def clean_settings(value: Any) -> dict[str, Any]:
    from brickdnn.data.pipeline import clean_settings as check

    with _user_errors():
        return check(value)


class _Sources:
    """The data sources a preparation may use: its main source and other sources of the same
    owner (join, append). Sources of other users are never found."""

    def __init__(self, main: DataSource):
        self.owner_id = main.owner_id
        self.found: dict[int, DataSource] = {main.pk: main}

    def get(self, source_id: Any) -> DataSource | None:
        if isinstance(source_id, str) and source_id.isdigit():
            source_id = int(source_id)
        if not isinstance(source_id, int) or isinstance(source_id, bool):
            return None
        if source_id not in self.found:
            source = DataSource.objects.filter(owner_id=self.owner_id, pk=source_id).first()
            if source is None:
                return None
            self.found[source_id] = source
        return self.found[source_id]

    # Callbacks for brickdnn.data.pipeline.run_pipeline (they raise DataError).

    def load(self, source_id: int) -> Table:
        return _load_table(self._existing(source_id))

    def name(self, source_id: int) -> str:
        source = self.get(source_id)
        return source.name if source is not None else f"source {source_id}"

    def info(self, source_id: int) -> dict[str, Any]:
        return source_info(self._existing(source_id))

    def _existing(self, source_id: int) -> DataSource:
        from brickdnn.data.table import DataError

        source = self.get(source_id)
        if source is None:
            raise DataError(GONE_MESSAGE)
        return source


@dataclass
class _Run:
    """A pipeline run of a preparation."""

    result: PipelineResult
    steps: list[dict[str, Any]]
    settings: dict[str, Any]
    sources: _Sources


def _run(
    prep: DataPrep,
    steps: Any = None,
    prep_settings: Any = None,
    upto: str | None = None,
) -> _Run:
    """Run the (given or saved) steps of ``prep``. Other sources (join, append) are only
    loaded if they belong to the owner of the preparation. Raises :class:`DataInputError`."""
    from brickdnn.data.pipeline import run_pipeline

    steps = clean_steps(prep.steps if steps is None else steps)
    prep_settings = clean_settings(prep.settings if prep_settings is None else prep_settings)
    if upto is not None and upto not in {step["id"] for step in steps}:
        raise DataInputError(f"There is no step '{upto}' (any more).")
    sources = _Sources(prep.source)
    with _user_errors():
        result = run_pipeline(
            _load_table(prep.source),
            steps,
            prep_settings,
            load_source=sources.load,
            source_name=sources.name,
            source_info=sources.info,
            upto=upto,
            source_id=prep.source_id,
        )
    return _Run(result, steps, prep_settings, sources)


def _target_args(table: Table, prep_settings: dict[str, Any], task: str | None) -> dict[str, Any]:
    """``target``/``task`` for profile_table, if the target exists in ``table``."""
    target = prep_settings.get("target")
    if target and task and target in table.kinds:
        return {"target": target, "task": task}
    return {}


# Cells (rows × columns) in a preview sample at most.
MAX_SAMPLE_CELLS = 1_000_000


def _sample_rows(value: Any) -> int:
    if value is None:
        return int(settings.DATA_SAMPLE_ROWS)
    whole = isinstance(value, int) or (isinstance(value, float) and value.is_integer())
    if isinstance(value, bool) or not whole:
        raise DataInputError("sample_rows must be a whole number.")
    return max(MIN_SAMPLE_ROWS, min(int(value), int(settings.DATA_MAX_SAMPLE_ROWS)))


def preview(prep: DataPrep, body: Any) -> dict[str, Any]:
    """``PreviewResult`` for a ``PreviewRequest`` (see frontend/src/api/types.ts)."""
    from brickdnn.data.pipeline import readiness
    from brickdnn.data.profile import profile_table, sample_table
    from brickdnn.data.relations import column_relations

    started = time.perf_counter()
    if body is None:
        body = {}
    if not isinstance(body, dict):
        raise DataInputError("Send the preview settings as a JSON object.")
    upto = body.get("upto")
    if upto is not None and (not isinstance(upto, str) or not upto):
        raise DataInputError("upto must be the id of a step.")
    sample_rows = _sample_rows(body.get("sample_rows"))
    compare = body.get("compare") is True

    run = _run(prep, body.get("steps"), body.get("settings"), upto)
    result, prep_settings = run.result, run.settings
    with _user_errors():
        # How the number columns relate (training rows): for the profile and the readiness.
        relations = column_relations(result.table, prep_settings.get("target"))
        profile = profile_table(
            result.table,
            **_target_args(result.table, prep_settings, result.task),
            relations=relations,
        )
        before = (
            profile_table(
                result.before,
                **_target_args(result.before, prep_settings, result.task),
                relations=False,
            )
            if compare
            else None
        )
        # Wide tables (e.g. 785 pixel columns) get fewer sample rows, so the response and the
        # table in the browser stay small.
        columns = max(1, len(result.table.columns))
        rows = max(MIN_SAMPLE_ROWS, min(sample_rows, MAX_SAMPLE_CELLS // columns))
        sample = sample_table(result.table, rows, seed=int(prep_settings["split"]["seed"]))
        ready = readiness(
            result.table, prep_settings, result.task, split=result.split, relations=relations
        )
    failed = _failed_step(run)
    if failed:
        # The data shown is the data before that step; it cannot be exported like this.
        ready["ready"] = False
        ready["issues"] = [
            {
                "code": "step_failed",
                "severity": "error",
                "column": None,
                "columns": [],
                "message": failed,
                "fix": None,
            },
            *ready.get("issues", []),
        ]
    if _oversampled(result.reports):
        # Oversampling copies rows on purpose; removing duplicates would undo it.
        profile["warnings"] = [w for w in profile["warnings"] if w.get("code") != "duplicates"]
        ready["issues"] = _without_duplicate_fixes(ready.get("issues", []))
    return {
        "steps": result.reports,
        "task": result.task,
        "profile": profile,
        "before": before,
        "sample": sample,
        "readiness": ready,
        "split": result.split,
        "rows": _row_summary(result, prep_settings) if body.get("rows") is True else None,
        "warnings": list(result.warnings),
        "duration_ms": round((time.perf_counter() - started) * 1000, 1),
    }


def _row_summary(result: PipelineResult, prep_settings: dict[str, Any]) -> dict[str, Any] | None:
    """A compact summary of the row analysis (MCP previews), on the same rows as the
    readiness check (so it comes from the cache), or None if no number column fits."""
    from brickdnn.data import rows
    from brickdnn.data.checks import STRUCTURE_ROWS, STRUCTURE_WORK
    from brickdnn.data.table import DataError

    table = result.table
    target = prep_settings.get("target")
    group_by = prep_settings["split"].get("group_by")
    try:
        stats = rows.cached_compute(
            table,
            target=target if target in table.kinds else None,
            exclude=[group_by] if group_by else [],
            max_rows=STRUCTURE_ROWS,
            max_work=STRUCTURE_WORK,
            with_projection=False,
        )
    except DataError:
        return None
    return rows.summary(stats, table)


def _without_duplicate_fixes(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Readiness issues after oversampling: no note about duplicate rows (they are the
    copies), and no "Remove duplicates" fix for validation/test rows that copy training rows
    (added at the end, it would undo the oversampling)."""
    result = []
    for issue in issues:
        if issue.get("code") == "duplicate_rows":
            continue
        if issue.get("code") == "split_overlap" and issue.get("fix"):
            issue = {
                **issue,
                "message": issue["message"] + " Remove duplicates before balancing the classes.",
                "fix": None,
            }
        result.append(issue)
    return result


def _oversampled(reports: list[dict[str, Any]]) -> bool:
    return any(r.get("type") == "balance_classes" and r.get("status") == "ok" for r in reports)


def _failed_step(run: _Run) -> str | None:
    """Message for the first step that failed, if one did."""
    from brickdnn.data.steps import REGISTRY

    for number, report in enumerate(run.result.reports, start=1):
        if report.get("status") == "error":
            step = REGISTRY.get(report.get("type"))
            label = step.label if step is not None else report.get("type")
            return f"Step {number} ({label}) failed: {report.get('error') or 'unknown error'}"
    return None


# --------------------------------------------------------------------------- #
# Row analysis (Analysis → Rows)
# --------------------------------------------------------------------------- #


def _row_options(body: dict[str, Any]) -> dict[str, Any]:
    """The feature choice of a row analysis request: ``{"features": {"mode", "columns"}}``."""
    features = body.get("features")
    if features is None:
        features = {}
    if not isinstance(features, dict):
        raise DataInputError('features must be an object: {"mode", "columns"}.')
    mode = features.get("mode", "auto")
    columns = features.get("columns")
    if mode not in ("auto", "all", "partnered", "columns"):
        raise DataInputError("features.mode must be auto, all, partnered or columns.")
    if columns is not None and (
        not isinstance(columns, list) or not all(isinstance(c, str) for c in columns)
    ):
        raise DataInputError("features.columns must be a list of column names.")
    if mode == "columns" and len(columns or []) > 5_000:
        raise DataInputError("Choose at most 5,000 columns.")
    return {"mode": mode, "columns": list(columns or []) if mode == "columns" else None}


def _threshold(body: dict[str, Any], key: str) -> float | None:
    value = body.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 100:
        raise DataInputError(f"{key} must be a number from 0 to 100.")
    return float(value)


def _row_stats(prep: DataPrep, body: Any):
    """Run the pipeline of a row analysis request and compute (or reuse) its row
    statistics. Returns ``(run, stats, settings)``."""
    from brickdnn.data import rows

    if body is None:
        body = {}
    if not isinstance(body, dict):
        raise DataInputError("Send the row analysis settings as a JSON object.")
    upto = body.get("upto")
    if upto is not None and (not isinstance(upto, str) or not upto):
        raise DataInputError("upto must be the id of a step.")
    options = _row_options(body)
    run = _run(prep, body.get("steps"), body.get("settings"), upto)
    result = run.result
    target = run.settings.get("target")
    target = target if target in result.table.kinds else None
    group_by = run.settings["split"].get("group_by")
    with _user_errors():
        stats = rows.cached_compute(
            result.table,
            target=target,
            exclude=[group_by] if group_by else [],
            seed=int(run.settings["split"]["seed"]),
            **options,
        )
    return run, stats, target


def row_analysis(prep: DataPrep, body: Any) -> dict[str, Any]:
    """``RowAnalysis`` for a ``RowAnalysisRequest`` (see frontend/src/api/types.ts)."""
    from brickdnn.data import rows

    started = time.perf_counter()
    run, stats, target = _row_stats(prep, body)
    body = body or {}
    with _user_errors():
        view = rows.analysis_view(
            stats,
            run.result.table,
            target=target,
            task=run.result.task if target else None,
            twin_threshold=_threshold(body, "twin_threshold"),
            atypical_threshold=_threshold(body, "atypical_threshold"),
        )
    view["split_group_by"] = run.settings["split"].get("group_by")
    view["duration_ms"] = round((time.perf_counter() - started) * 1000, 1)
    view["stats_ms"] = stats.duration_ms
    return view


def similar_rows(prep: DataPrep, body: Any) -> dict[str, Any]:
    """``SimilarRows``: the nearest rows of one analysed row (``row``: its index in the
    analysis) and a column comparison with ``other`` (default: its nearest row)."""
    from brickdnn.data import rows

    run, stats, target = _row_stats(prep, body)
    body = body or {}

    def index(key: str) -> int | None:
        value = body.get(key)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise DataInputError(f"{key} must be the index of a row of the analysis.")
        return value

    row = index("row")
    if row is None:
        raise DataInputError("row: choose a row of the analysis.")
    with _user_errors():
        return rows.similar_view(
            stats,
            run.result.table,
            row,
            other=index("other"),
            target=target,
            task=run.result.task if target else None,
        )


# --------------------------------------------------------------------------- #
# Export into a training dataset
# --------------------------------------------------------------------------- #


def export_dataset(prep: DataPrep, name: Any = None) -> Dataset:
    """Run the saved preparation and store the result as a training dataset (kind "csv")
    with its split. Raises :class:`DataInputError` (with ``issues`` if not ready)."""
    from brickdnn.data.export import DATA_FILE, write_training_dataset
    from brickdnn.data.pipeline import readiness

    if name is not None and not isinstance(name, str):
        raise DataInputError("The name must be text.")
    name = (name or "").strip() or prep.name
    if len(name) > 120:
        raise DataInputError("The name can be at most 120 characters long.")

    budgets.check_storage(prep.owner)  # the dataset belongs to the owner of the preparation
    run = _run(prep)
    if (failed := _failed_step(run)) is not None:
        raise DataInputError(f"{failed} Fix it before creating a training dataset.")
    result = run.result
    with _user_errors():
        ready = readiness(result.table, run.settings, result.task, checks=False, split=result.split)
    if not ready.get("ready"):
        issues = ready.get("issues") or []
        errors = [i for i in issues if i.get("severity") == "error"]
        message = errors[0]["message"] if errors else "The data is not ready for training."
        raise DataInputError(message, issues=issues)

    datasets = training.training_dir() / "datasets"
    datasets.mkdir(parents=True, exist_ok=True)
    _remove_abandoned(datasets, ".upload-*")
    # Same prefix as uploads: abandoned work folders are removed by both.
    work = datasets / f".upload-{uuid.uuid4().hex}"
    work.mkdir()
    dataset = None
    try:
        with _user_errors():
            written = write_training_dataset(result.table, run.settings, result.task, work)
            info = inspect_csv(work / DATA_FILE, name=name)
        _check_written(info, written)
        info.predefined_split = [int(n) for n in written["split_counts"]]
        class_names = written.get("class_names")
        info.class_names = [str(c) for c in class_names] if class_names else None
        _freeze(prep, run, work)
        size = budgets.folder_size(work)
        budgets.check_storage(prep.owner, size)
        dataset = Dataset.objects.create(
            owner=prep.owner,
            name=name,
            kind="csv",
            size_bytes=size,
            info=info.to_dict(),
            prepared_from=prep,
        )
        final = training.dataset_dir(dataset.pk)
        shutil.rmtree(final, ignore_errors=True)  # left over from an earlier database
        os.replace(work, final)
        # Shared preparation: its projects can train on the result as well.
        dataset.projects.set(prep.projects.all())
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        if dataset is not None:
            dataset.delete()
        raise
    return dataset


def _freeze(prep: DataPrep, run: _Run, work: Path) -> None:
    """Keep the preparation with the values it learned next to the dataset, so that runs
    trained on it can predict from raw rows (brickdnn.data.frozen). Without it the dataset
    still works; only raw predictions are not possible."""
    from brickdnn.data.frozen import FROZEN_DIR, freeze

    try:
        freeze(
            work / FROZEN_DIR,
            result=run.result,
            steps=run.steps,
            source=_load_table(prep.source),
            source_info=source_info(prep.source),
            prep={"id": prep.pk, "name": prep.name},
            load_source=run.sources.load,
            source_name=run.sources.name,
        )
    except Exception:
        logger.exception("Could not store the preparation %s for raw predictions", prep.pk)
        shutil.rmtree(work / FROZEN_DIR, ignore_errors=True)


def _check_written(info: DatasetInfo, written: dict[str, Any]) -> None:
    """The training engine reads data.csv with brickdnn.training.datasets.open_csv: make
    sure it sees the rows and columns that were written (split.bin has one byte per row)."""
    columns = [str(c).strip() for c in [*written.get("features", []), written.get("target")]]
    if info.has_header is not True or info.num_samples != written.get("rows"):
        raise DataInputError(
            "The column names cannot all be numbers. Rename the columns (step “Rename "
            "columns”) and try again."
        )
    if info.delimiter != "," or info.columns != columns:
        raise DataInputError(
            "The prepared data could not be read back as a training dataset. Rename columns "
            "that contain commas, semicolons or tabs and try again."
        )
    counts = written.get("split_counts") or []
    if len(counts) != 3 or sum(counts) != info.num_samples:
        raise DataInputError("The split of the prepared data does not match its rows.")


# --------------------------------------------------------------------------- #
# Downloads
# --------------------------------------------------------------------------- #


@dataclass
class Download:
    """A file for the browser: ``content`` (bytes) or ``handle`` (an open file to stream)."""

    filename: str
    content_type: str
    content: bytes | None = None
    handle: Any = None


def download(prep: DataPrep, fmt: Any) -> Download:
    """The prepared data (``csv``), the recipe (``recipe``, JSON) or a pandas script
    (``python``) of the saved preparation."""
    from brickdnn.data import export

    if fmt not in DOWNLOAD_FORMATS:
        raise DataInputError("Choose the format csv, recipe or python.")
    run = _run(prep)
    if (failed := _failed_step(run)) is not None:
        raise DataInputError(f"{failed} Fix it before downloading.")
    slug = slugify(prep.name) or "prepared-data"
    if fmt == "csv":
        return Download(f"{slug}.csv", "text/csv; charset=utf-8", handle=_csv_file(run))
    for step in run.steps:  # the other sources of combine steps (own ones only)
        if _is_combine(step):
            run.sources.get((step.get("params") or {}).get("source"))
    sources = {source_id: source_info(source) for source_id, source in run.sources.found.items()}
    with _user_errors():
        if fmt == "recipe":
            document = export.recipe(sources, prep.source_id, run.steps, run.settings, run.result)
            content = json.dumps(document, indent=2, ensure_ascii=False).encode()
            return Download(f"{slug}.recipe.json", "application/json", content)
        script = export.pandas_script(
            sources, prep.source_id, run.steps, run.settings, run.result, name=prep.name
        )
    filename = slug.replace("-", "_") + "_prepare.py"
    return Download(filename, "text/x-python; charset=utf-8", script.encode())


def _is_combine(step: dict[str, Any]) -> bool:
    from brickdnn.data.steps import REGISTRY

    step_type = REGISTRY.get(step.get("type"))
    return step_type is not None and step_type.combine


def _csv_file(run: _Run):
    """Write the prepared table into a temporary file and return it opened for reading
    (the file itself is removed right away where the system allows it)."""
    from brickdnn.data.export import write_processed_csv

    folder = prep_dir() / "tmp"
    folder.mkdir(parents=True, exist_ok=True)
    _remove_abandoned(folder, "*.csv")
    fd, name = tempfile.mkstemp(suffix=".csv", dir=folder)
    os.close(fd)
    path = Path(name)
    try:
        with _user_errors():
            write_processed_csv(run.result.table, path)
        handle = path.open("rb")
    finally:
        with contextlib.suppress(OSError):
            path.unlink()  # POSIX: the open handle keeps the data readable
    return handle


# --------------------------------------------------------------------------- #
# Clean-up of files (also when a user and with it all their objects are deleted)
# --------------------------------------------------------------------------- #


@receiver(post_delete, sender=DataSource, dispatch_uid="brickdnn_data_source_files")
def _delete_source_files(sender, instance: DataSource, **kwargs) -> None:
    directory = source_dir(instance.pk)

    def remove() -> None:
        shutil.rmtree(directory, ignore_errors=True)
        if dependencies_installed():
            from brickdnn.data.store import clear_cache

            clear_cache(directory / TABLE_DIR)  # a new source may get the same id later

    transaction.on_commit(remove)
