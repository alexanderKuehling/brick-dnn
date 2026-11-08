"""Data preparation (the Data tab): imported CSV files, preparations, previews, exports.

The work is done by :mod:`brickdnn.data` (pandas); this module stores files, checks who may
use what and turns preparations into training datasets. Files::

    DATA_PREP_DIR/sources/<id>/original.csv   the uploaded file, unchanged
    DATA_PREP_DIR/sources/<id>/table/         the typed table (brickdnn.data.store)

pandas is only imported by the functions that need it, so the server starts (and code
generation and training keep working) without it.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import tempfile
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from django.conf import settings
from django.db import transaction
from django.db.models.signals import post_delete
from django.dispatch import receiver
from django.utils.text import slugify

from brickdnn.data import MISSING_DEPENDENCIES, dependencies_installed
from brickdnn.training import DatasetError, DatasetInfo, inspect_csv

from . import training_service as training
from .models import DataPrep, Dataset, DataSource

if TYPE_CHECKING:
    from brickdnn.data.pipeline import PipelineResult
    from brickdnn.data.table import Table

DISABLED_REASON = "Data preparation is disabled on this server."
WRONG_FILE_MESSAGE = "Upload a CSV file (.csv, .tsv or .txt)."
MISSING_FILES_MESSAGE = "The imported data of this source is missing. Import the file again."
GONE_MESSAGE = "This table does not exist (any more)."
SUFFIXES = (".csv", ".tsv", ".txt")
ORIGINAL_FILE = "original.csv"
TABLE_DIR = "table"
# Form fields of an import, see brickdnn.data.io.ImportOptions.
OPTION_FIELDS = ("delimiter", "decimal", "header", "encoding", "missing_values")
MIN_SAMPLE_ROWS = 10
DOWNLOAD_FORMATS = ("csv", "recipe", "python")


class DataInputError(ValueError):
    """Invalid request data or data that cannot be used. The message is shown to the user.

    ``issues`` are the readiness issues when data is not ready for training.
    """

    def __init__(self, message: str, issues: list[dict[str, Any]] | None = None):
        super().__init__(message)
        self.issues = issues


@contextlib.contextmanager
def _user_errors() -> Iterator[None]:
    """Turn the errors of brickdnn.data / brickdnn.training into :class:`DataInputError`."""
    from brickdnn.data.table import DataError

    try:
        yield
    except (DataError, DatasetError) as exc:
        raise DataInputError(str(exc)) from None


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
    """Store and read an uploaded CSV file. ``fields`` are the form fields (``name`` and the
    import options). Raises :class:`DataInputError`."""
    from brickdnn.data.io import ImportOptions, read_csv
    from brickdnn.data.store import save_table

    filename = Path(upload.name or "").name
    if Path(filename).suffix.lower() not in SUFFIXES:
        raise DataInputError(WRONG_FILE_MESSAGE)
    name = fields.get("name")
    name = (str(name or "").strip() or Path(filename).stem.strip() or "Data")[:120]
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
        original = work / ORIGINAL_FILE
        with original.open("wb") as out:
            for chunk in upload.chunks():
                out.write(chunk)
        with _user_errors():
            table, report = read_csv(
                original,
                options,
                max_rows=settings.DATA_MAX_ROWS,
                max_columns=settings.DATA_MAX_COLUMNS,
            )
            save_table(table, work / TABLE_DIR)
        missing = table.df.isna().sum()
        columns = [
            {"name": column, "kind": table.kinds[column], "missing": int(missing[column])}
            for column in table.columns
        ]
        stored_options = {key: options.to_dict().get(key) for key in OPTION_FIELDS}
        source = DataSource.objects.create(
            owner=user,
            name=name,
            filename=filename[:255],
            size_bytes=original.stat().st_size,
            n_rows=table.n_rows,
            n_columns=len(columns),
            columns=columns,
            import_options=stored_options,
            report=report.to_dict(),
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
    return load_table_cached(directory)


def source_info(source: DataSource) -> dict[str, Any]:
    """How to read the file of ``source`` again (``SourceInfo`` in brickdnn.data.export)."""
    from brickdnn.data.io import DEFAULT_MISSING

    report = source.report if isinstance(source.report, dict) else {}
    options = source.import_options if isinstance(source.import_options, dict) else {}
    missing = options.get("missing_values")
    return {
        "id": source.pk,
        "name": source.name,
        "filename": source.filename,
        "delimiter": report.get("delimiter", ","),
        "decimal": report.get("decimal", "."),
        "header": bool(report.get("header", True)),
        "encoding": report.get("encoding", "utf-8"),
        "missing_values": list(missing) if isinstance(missing, list) else list(DEFAULT_MISSING),
        "columns": [
            {"name": c.get("name"), "kind": c.get("kind")}
            for c in (source.columns or [])
            if isinstance(c, dict)
        ],
    }


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
        profile = profile_table(
            result.table, **_target_args(result.table, prep_settings, result.task)
        )
        before = (
            profile_table(result.before, **_target_args(result.before, prep_settings, result.task))
            if compare
            else None
        )
        # Wide tables (e.g. 785 pixel columns) get fewer sample rows, so the response and the
        # table in the browser stay small.
        columns = max(1, len(result.table.columns))
        rows = max(MIN_SAMPLE_ROWS, min(sample_rows, MAX_SAMPLE_CELLS // columns))
        sample = sample_table(result.table, rows, seed=int(prep_settings["split"]["seed"]))
        ready = readiness(result.table, prep_settings, result.task)
    failed = _failed_step(run)
    if failed:
        # The data shown is the data before that step; it cannot be exported like this.
        ready["ready"] = False
        ready["issues"] = [
            {"severity": "error", "column": None, "message": failed, "fix": None},
            *ready.get("issues", []),
        ]
    if _oversampled(result.reports):
        # Oversampling copies rows on purpose; removing duplicates would undo it.
        profile["warnings"] = [w for w in profile["warnings"] if w.get("code") != "duplicates"]
    return {
        "steps": result.reports,
        "task": result.task,
        "profile": profile,
        "before": before,
        "sample": sample,
        "readiness": ready,
        "warnings": list(result.warnings),
        "duration_ms": round((time.perf_counter() - started) * 1000, 1),
    }


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

    run = _run(prep)
    if (failed := _failed_step(run)) is not None:
        raise DataInputError(f"{failed} Fix it before creating a training dataset.")
    result = run.result
    with _user_errors():
        ready = readiness(result.table, run.settings, result.task)
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
        size = sum(p.stat().st_size for p in work.iterdir() if p.is_file())
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
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        if dataset is not None:
            dataset.delete()
        raise
    return dataset


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
