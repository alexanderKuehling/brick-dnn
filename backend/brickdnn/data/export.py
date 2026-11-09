"""Turning a prepared table into files: a training dataset, a cleaned CSV file, a recipe and
a pandas script that prepares new data the same way.

**Training dataset** (``write_training_dataset``), read by the existing training engine::

    data.csv    header line, "," separated, "." decimals; the feature columns (all columns
                except the target, in table order) and the target as the last column
    split.bin   one byte per data row: 0 train, 1 validation, 2 test

Classification targets are written as class indices 0 … K-1; ``class_names`` returns the
original labels in index order (numeric labels sorted numerically, others alphabetically).
Regression targets are written as numbers. The API stores ``class_names`` and the split
counts in the dataset info (``DatasetInfo.predefined_split``), and the training engine uses
``split.bin`` instead of its own random split.

**SourceInfo** (how to read a source again, for the pandas script)::

    {"id": 3, "name": "Customers", "filename": "customers.csv", "delimiter": ";",
     "decimal": ",", "header": True, "encoding": "utf-8",
     "missing_values": ["", "NA", ...], "columns": [{"name", "kind"}, ...]}

**The pandas script** works on DataFrames in the storage format of
:mod:`brickdnn.data.table` (numbers and booleans as float64, dates as datetime64, strings as
objects with NaN for missing values), so the code of every step (``Step.to_pandas``) runs
unchanged on the training data and on new data.
"""

from __future__ import annotations

import inspect
import logging
import pprint
import re
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .pipeline import (
    PipelineResult,
    class_labels,
    clean_settings,
    json_safe,
    make_context,
    readiness,
)
from .steps import REGISTRY
from .table import SPLIT_NAMES, DataError, Table

logger = logging.getLogger(__name__)

SPLIT_FILE = "split.bin"
DATA_FILE = "data.csv"
RECIPE_FORMAT = "brickdnn-data-recipe"
RECIPE_VERSION = 1
# Whole numbers up to this size are written without ".0".
MAX_EXACT_INTEGER = 2**53

# Used by the generated reader when brickdnn.data.io is not importable.
_DEFAULT_MISSING = ("", "NA", "N/A", "n/a", "na", "NaN", "nan", "null", "NULL", "None", "-", "?")
_DEFAULT_TRUE = ("true", "yes", "y", "t", "ja", "wahr")
_DEFAULT_FALSE = ("false", "no", "n", "f", "nein", "falsch")
_DEFAULT_DATE_FORMATS = {
    "iso": ("ISO8601", "%Y-%m-%d", "%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"),
    "ymd": ("%Y/%m/%d", "%Y/%m/%d %H:%M", "%Y/%m/%d %H:%M:%S"),
    "dmy_dot": ("%d.%m.%Y", "%d.%m.%Y %H:%M", "%d.%m.%Y %H:%M:%S"),
    "mdy": ("%m/%d/%Y", "%m/%d/%Y %H:%M", "%m/%d/%Y %H:%M:%S"),
    "dmy": ("%d/%m/%Y", "%d/%m/%Y %H:%M", "%d/%m/%Y %H:%M:%S"),
    "dmy_dash": ("%d-%m-%Y", "%d-%m-%Y %H:%M", "%d-%m-%Y %H:%M:%S"),
}


# --------------------------------------------------------------------------- #
# Training dataset and CSV
# --------------------------------------------------------------------------- #


def _compact(series: pd.Series) -> pd.Series:
    """Whole-number float columns as int64 (smaller files, no ".0"); others unchanged."""
    values = series.to_numpy(dtype=float, na_value=np.nan)
    if (
        len(values)
        and np.isfinite(values).all()
        and np.all(np.floor(values) == values)
        and np.abs(values).max() < MAX_EXACT_INTEGER
    ):
        return pd.Series(values.astype(np.int64), name=series.name)
    return pd.Series(values, name=series.name)


def write_training_dataset(
    table: Table, settings: dict[str, Any], task: str | None, directory: str | Path
) -> dict[str, Any]:
    """Write ``data.csv`` and ``split.bin`` into ``directory``.

    Raises :class:`DataError` with the first readiness error if the table is not ready.
    Returns ``{"rows", "split_counts": [train, val, test], "features": [...], "target",
    "task", "class_names": [...] | None}``.
    """
    settings = clean_settings(settings)
    check = readiness(table, settings, task, checks=False)
    if not check["ready"]:
        first = next(i for i in check["issues"] if i["severity"] == "error")
        raise DataError(first["message"])
    assert table.split is not None
    target = settings["target"]
    features = [c for c in table.columns if c != target]
    columns = {name: _compact(table.df[name]) for name in features}
    class_names = None
    if task == "classification":
        class_names, codes = class_labels(table.df[target], table.kinds[target])
        columns[target] = pd.Series(codes, name=target)
    else:
        columns[target] = _compact(table.df[target])
    frame = pd.DataFrame(columns, columns=[*features, target])
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(directory / DATA_FILE, index=False, lineterminator="\n", encoding="utf-8")
    (directory / SPLIT_FILE).write_bytes(table.split.astype(np.uint8).tobytes())
    return {
        "rows": table.n_rows,
        "split_counts": table.split_counts(),
        "features": features,
        "target": target,
        "task": task,
        "class_names": class_names,
    }


def to_csv_frame(df):
    """The table as written to CSV files: whole numbers without ".0" (nullable integers),
    everything else as is; missing values become empty cells."""
    columns = {}
    for name in df.columns:
        series = df[name]
        if pd.api.types.is_float_dtype(series):
            values = series.to_numpy(dtype=float)
            present = values[~np.isnan(values)]
            if (
                len(present)
                and np.isfinite(present).all()
                and np.all(np.floor(present) == present)
                and np.abs(present).max() < 2**53
            ):
                series = series.astype("Int64")
        columns[name] = series
    return pd.DataFrame(columns, index=df.index)


def write_processed_csv(table: Table, path: str | Path, *, include_split: bool = True) -> None:
    """The prepared table as a CSV file ("," separated, "." decimals, ISO datetimes, empty
    cells for missing values), with a last column ``split`` (train / val / test) if the
    table has a split and ``include_split`` is set."""
    frame = to_csv_frame(table.df)
    if include_split and table.split is not None:
        name, n = "split", 2
        while name in frame.columns:
            name, n = f"split_{n}", n + 1
        frame[name] = np.asarray(SPLIT_NAMES, dtype=object)[table.split.astype(np.int64)]
    frame.to_csv(path, index=False, na_rep="", lineterminator="\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Recipe
# --------------------------------------------------------------------------- #


def _source_info(sources: dict[Any, dict[str, Any]], source_id: int) -> dict[str, Any]:
    for key, info in sources.items():
        if str(key) == str(source_id):
            return dict(info)
    return {
        "id": source_id,
        "name": f"source {source_id}",
        "filename": f"source_{source_id}.csv",
        "delimiter": ",",
        "decimal": ".",
        "header": True,
        "encoding": "utf-8",
        "missing_values": None,
        "columns": None,
    }


def _used_sources(steps: list[dict[str, Any]]) -> list[int]:
    used = []
    for step in steps:
        cls = REGISTRY.get(step.get("type"))
        source = (step.get("params") or {}).get("source")
        if cls is not None and cls.combine and isinstance(source, int) and source not in used:
            used.append(source)
    return used


def recipe(
    sources: dict[int, dict[str, Any]],
    source_id: int,
    steps: list[dict[str, Any]],
    settings: dict[str, Any],
    result: PipelineResult,
) -> dict[str, Any]:
    """Everything needed to reproduce the preparation, as JSON::

    {"format": "brickdnn-data-recipe", "version": 1, "source": SourceInfo,
     "sources": {id: SourceInfo, ...}, "settings": {...},
     "steps": [{"id", "type", "enabled", "params", "fitted"}, ...],
     "result": {"rows", "columns": [{"name", "kind"}], "split_counts", "task"}}
    """
    others = {str(i): _source_info(sources, i) for i in _used_sources(steps) if i != source_id}
    return json_safe(
        {
            "format": RECIPE_FORMAT,
            "version": RECIPE_VERSION,
            "source": _source_info(sources, source_id),
            "sources": others,
            "settings": result.settings or clean_settings(settings),
            "steps": [
                {
                    "id": step["id"],
                    "type": step["type"],
                    "enabled": step.get("enabled", True),
                    "params": step.get("params") or {},
                    "fitted": result.fitted.get(step["id"]),
                }
                for step in steps
            ],
            "result": {
                "rows": result.table.n_rows,
                "columns": result.table.column_info(),
                "split_counts": result.table.split_counts(),
                "task": result.task,
            },
        }
    )


# --------------------------------------------------------------------------- #
# pandas script
# --------------------------------------------------------------------------- #

# The reader in the generated script. Placeholders: __TRUE__, __FALSE__, __FORMATS__.
_READER = r'''
BOOLEANS = {**dict.fromkeys(__TRUE__, 1.0), **dict.fromkeys(__FALSE__, 0.0)}
DATE_FORMATS = __FORMATS__
DATE_PATTERNS = {
    "iso": r"\d{4}-\d{1,2}-\d{1,2}" + TIME + r"(?:Z|[+-]\d{2}(?::?\d{2})?)?",
    "ymd": r"\d{4}/\d{1,2}/\d{1,2}" + TIME,
    "dmy_dot": r"\d{1,2}\.\d{1,2}\.\d{4}" + TIME,
    "mdy": r"\d{1,2}/\d{1,2}/\d{4}" + TIME,
    "dmy": r"\d{1,2}/\d{1,2}/\d{4}" + TIME,
    "dmy_dash": r"\d{1,2}-\d{1,2}-\d{4}" + TIME,
}
# Decimal comma with "." thousands groups (1.234,5) / decimal point with "," groups (1,234.5)
COMMA_NUMBER = r"[+-]?(?:\d{1,3}(?:\.\d{3})+(?:,\d*)?|\d+(?:,\d*)?|,\d+)(?:[eE][+-]?\d+)?"
GROUPED_DOT_NUMBER = r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d*)?(?:[eE][+-]?\d+)?"


def parse_numbers(text, decimal):
    """Strings to float64 numbers like the import (decimal mark, thousands groups)."""
    text = text.astype(object)
    if decimal == ",":
        dotted = text.str.contains(".", regex=False, na=False)
        grouped = text.str.fullmatch(COMMA_NUMBER, na=False)
        text = text.mask(dotted & grouped, text.str.replace(".", "", regex=False))
        text = text.mask(dotted & ~grouped, None).str.replace(",", ".", regex=False)
    else:
        grouped = text.str.fullmatch(GROUPED_DOT_NUMBER, na=False)
        text = text.mask(grouped, text.str.replace(",", "", regex=False))
    text = text.mask(text.str.contains("_", regex=False, na=False), None)
    # to_numeric finds the numbers; float() converts them exactly (to_numeric may be off
    # in the last digit)
    numbers = text.where(pd.to_numeric(text, errors="coerce").notna()).astype("float64")
    return numbers.replace([np.inf, -np.inf], np.nan)


def parse_dates(text):
    """Strings to dates: the date family that fits the values, formats tried in order."""
    present = text.dropna()
    families = [
        f for f, pattern in DATE_PATTERNS.items() if present.str.fullmatch(pattern).all()
    ] or list(DATE_FORMATS)
    best = None
    for family in families:
        result = pd.Series(pd.NaT, index=text.index, dtype="datetime64[ns]")
        for fmt in DATE_FORMATS[family]:
            todo = result.isna() & text.notna()
            if not todo.any():
                break
            if fmt == "ISO8601":
                parsed = pd.to_datetime(text[todo], format=fmt, errors="coerce", utc=True)
                parsed = parsed.dt.tz_localize(None)
            else:
                parsed = pd.to_datetime(text[todo], format=fmt, errors="coerce")
            result[todo] = parsed.astype("datetime64[ns]")
        if best is None or result.notna().sum() > best.notna().sum():
            best = result
    return best


def read_cells(path, info):
    """All cells of a CSV file as strings (no header handling yet), like the import."""
    encoding = {"utf-8": "utf-8-sig"}.get(info["encoding"], info["encoding"])
    with open(path, encoding=encoding, newline="") as f:
        first = f.readline()
    # Excel's "sep=;" hint line
    skip = 1 if re.fullmatch(r"\s*sep=.\s*", first.rstrip("\r\n"), re.IGNORECASE) else 0
    options = {
        "sep": info["delimiter"],
        "header": None,
        "skiprows": skip,
        "dtype": str,
        "keep_default_na": False,
        "na_filter": False,
        "skip_blank_lines": True,
        "encoding": encoding,
    }
    try:
        cells = pd.read_csv(path, **options)
    except pd.errors.ParserError:  # a line has more values than the first one
        with open(path, encoding=encoding, newline="") as f:
            rows = csv.reader(f, delimiter=info["delimiter"])
            widest = max((len(row) for row in rows), default=1)
        cells = pd.read_csv(path, names=list(range(widest)), **options).fillna("")
    # Lines that end with a delimiter give empty columns at the end: drop them.
    while cells.shape[1] > 1 and (cells.iloc[:, -1].str.strip() == "").all():
        cells = cells.iloc[:, :-1]
    return cells


def read_source(path, info):
    """Read a CSV file like the Brick DNN import did: same delimiter, decimal mark,
    encoding, missing-value markers and column types; rows without any value are skipped."""
    cells = read_cells(path, info)
    columns = info["columns"]
    if cells.shape[1] != len(columns):
        sys.exit(f"{path}: expected {len(columns)} columns, found {cells.shape[1]}.")
    if info["header"]:
        cells = cells.iloc[1:]
    missing = set(info["missing_values"]) | {""}
    text = {}
    for i, name in enumerate(columns):
        values = cells.iloc[:, i].str.strip()
        text[name] = values.mask(values.isin(missing), None)
    df = pd.DataFrame(text).reset_index(drop=True)
    df = df[df.notna().any(axis=1)].reset_index(drop=True)
    for name, kind in columns.items():
        if kind == "numeric":
            df[name] = parse_numbers(df[name], info["decimal"])
        elif kind == "boolean":
            df[name] = df[name].str.lower().map(BOOLEANS).astype(float)
        elif kind == "datetime":
            df[name] = parse_dates(df[name])
    return df
'''


def _reader_source() -> str:
    try:
        from . import io as data_io

        true_values = tuple(data_io.TRUE_VALUES)
        false_values = tuple(data_io.FALSE_VALUES)
        formats = {k: tuple(v) for k, v in data_io.DATE_FORMATS.items()}
        if set(formats) != set(_DEFAULT_DATE_FORMATS):
            formats = _DEFAULT_DATE_FORMATS
    except (ImportError, AttributeError):
        true_values, false_values, formats = _DEFAULT_TRUE, _DEFAULT_FALSE, _DEFAULT_DATE_FORMATS
    text = _READER.replace("__TRUE__", repr(true_values)).replace("__FALSE__", repr(false_values))
    text = text.replace("__FORMATS__", pprint.pformat(formats, width=96, sort_dicts=False))
    time = r'TIME = r"(?:[ T]\d{1,2}:\d{2}(?::\d{2}(?:[.,]\d+)?)?)?"' + "\n"
    return time + text.strip("\n") + "\n"


def _default_missing() -> list[str]:
    try:
        from .io import DEFAULT_MISSING

        return list(DEFAULT_MISSING)
    except ImportError:
        return list(_DEFAULT_MISSING)


def _script_info(info: dict[str, Any], fallback_columns: list[dict[str, str]]) -> dict[str, Any]:
    """The part of a SourceInfo the script needs, with columns as {name: kind}."""
    columns = info.get("columns") or fallback_columns
    missing = info.get("missing_values")
    return {
        "name": str(info.get("name") or ""),
        "filename": str(info.get("filename") or "data.csv"),
        "delimiter": str(info.get("delimiter") or ","),
        "decimal": str(info.get("decimal") or "."),
        "header": bool(info.get("header", True)),
        "encoding": str(info.get("encoding") or "utf-8"),
        "missing_values": [
            str(v) for v in (missing if missing is not None else _default_missing())
        ],
        "columns": {str(c["name"]): str(c["kind"]) for c in columns},
    }


def comment(text: Any) -> str:
    """Text for a ``#`` comment: one line, no control characters."""
    return " ".join(str(text).split())


def _doc(text: Any) -> str:
    """Text for the docstring of the script: one line without double quotes or backslashes,
    so that names can never end the docstring or start an escape sequence."""
    return comment(text).replace("\\", "/").replace('"', "'")


_STATUS_NOTES = {
    "disabled": "disabled, not applied",
    "error": "failed in the preview, left out",
    "skipped": "did not run (an earlier step failed), left out",
    "not_run": "not run in the preview, left out",
}


def _step_lines(
    step: dict[str, Any], number: int, result: PipelineResult, reports: dict[str, Any], ctx: Any
) -> tuple[list[str], bool]:
    """Code lines for one step and whether it is complete (False: left out)."""
    cls = REGISTRY.get(step["type"])
    label = cls.label if cls else step["type"]
    title = f"# {number}. {comment(label)}"
    report = reports.get(step["id"])
    status = report["status"] if report else "not_run"
    if cls is None or status != "ok":
        return [f"{title}: {_STATUS_NOTES.get(status, 'left out')}."], status == "disabled"
    if cls.training_only:
        return [f"{title}: only changes the training rows; not applied to new data."], True
    try:
        lines = cls(step.get("params") or {}).to_pandas(
            result.params.get(step["id"], {}), result.fitted.get(step["id"]), ctx
        )
    except Exception as exc:  # a missing to_pandas must not break the download
        logger.exception("to_pandas of step %s (%s) failed", step["id"], step["type"])
        return [f"{title}: cannot be written as pandas code ({comment(exc)})."], False
    header = [title + (" (values learned from the training rows)" if cls.fitted else "")]
    summary = comment(report.get("summary") or "")
    if summary:
        header += ["#    " + line for line in textwrap.wrap(summary, 88)]
    body: list[str] = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("#"):  # comments stay one line, whatever text they quote
            body.append(line[: len(line) - len(stripped)] + "# " + comment(stripped[1:]))
        else:
            body += line.split("\n")
    return header + body, True


def _hoist(lines: list[str], helpers: dict[str, list[str]]) -> list[str]:
    """Move top-level function definitions out of step code into ``helpers`` (by name,
    first one wins), so ``prepare`` stays readable. Returns the remaining lines."""
    rest: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("def ") and "(" in line:
            name = line[4 : line.index("(")].strip()
            block = [line]
            i += 1
            while i < len(lines) and (not lines[i].strip() or lines[i][0].isspace()):
                block.append(lines[i])
                i += 1
            while block and not block[-1].strip():
                block.pop()
            helpers.setdefault(name, block)
            continue
        rest.append(line)
        i += 1
    return rest


def _indent(lines: list[str]) -> list[str]:
    return [("    " + line) if line.strip() else "" for line in lines]


def pandas_script(
    sources: dict[int, dict[str, Any]],
    source_id: int,
    steps: list[dict[str, Any]],
    settings: dict[str, Any],
    result: PipelineResult,
    *,
    name: str = "data preparation",
) -> str:
    """A standalone Python script (pandas + numpy) that reads the source file(s) and
    applies every enabled step that ran, with the fitted values as constants.

    It defines ``prepare(df) -> df`` (all steps except combine steps and training-only
    steps, usable on new data) and a ``__main__`` block that reads the CSV file(s), runs
    the combine steps and ``prepare``, and writes ``prepared.csv``. The target column and
    the feature columns are listed at the top as ``TARGET`` and ``FEATURES``.
    """
    settings = result.settings or clean_settings(settings)
    target = settings.get("target")
    main_info = _source_info(sources, source_id)
    others = [i for i in _used_sources(steps) if i != source_id]
    infos = {source_id: main_info, **{i: _source_info(sources, i) for i in others}}
    ctx = make_context(
        result.table,
        settings,
        result.task,
        source_name=lambda i: str(infos.get(i, {}).get("name") or f"source {i}"),
        source_info=lambda i: infos.get(i, {}),
        source_id=source_id,
    )
    reports = {r["id"]: r for r in result.reports}
    combine_lines: list[str] = []
    prepare_lines: list[str] = []
    helpers: dict[str, list[str]] = {}
    incomplete = []
    for number, step in enumerate(steps, start=1):
        cls = REGISTRY.get(step["type"])
        lines, complete = _step_lines(step, number, result, reports, ctx)
        lines = _hoist(lines, helpers)
        (combine_lines if cls is not None and cls.combine else prepare_lines).extend(lines + [""])
        if not complete:
            incomplete.append(number)

    features = [c for c in result.table.columns if c != target]
    class_names = None
    if result.task == "classification" and target in result.table.kinds:
        names, _ = class_labels(result.table.df[target], result.table.kinds[target])
        class_names = names
    # Without stored columns, the main source is the input of the steps (if nothing was combined).
    fallback = result.before.column_info() if not others else []
    script_infos = {
        i: _script_info(info, fallback if i == source_id else []) for i, info in infos.items()
    }
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    main_name = _doc(main_info.get("name") or "data")
    main_file = _doc(main_info.get("filename") or "data.csv")
    slug = re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_") or "data"
    doc = [
        f'"""Prepare data like the Brick DNN data preparation "{_doc(name)}".',
        "",
        f"Generated by Brick DNN on {stamp} from {main_name} ({main_file}).",
    ]
    if target:
        task = f", {result.task}" if result.task else ""
        doc.append(f"Target: {_doc(target)}{task}; {len(features):,} feature columns.")
    doc += [
        "",
        "Every value the steps learned (means, bounds, categories ...) was learned from the",
        "training rows and is written into this file, so new data is prepared exactly like the",
        "training data. Steps that only change the training rows (class balancing) are left out.",
    ]
    if incomplete:
        shown = ", ".join(str(n) for n in incomplete)
        doc += ["", f"Note: step(s) {shown} could not be included, see the comments in prepare()."]
    doc += [
        "",
        "Usage:",
        f"    python {slug}_prepare.py --input {main_file} --output prepared.csv",
        "",
        "Needs Python 3.10+ with pandas 2 and numpy.",
        '"""',
    ]

    out = [*doc, ""]
    out += [
        "from __future__ import annotations",
        "",
        "import argparse",
        "import csv",
        "import re",
        "import sys",
        "",
        "import numpy as np",
        "import pandas as pd",
        "",
        f"TARGET = {target!r}",
        f"FEATURES = {pprint.pformat(features, width=96, compact=True)}",
        f"TASK = {result.task!r}",
    ]
    if class_names is not None:
        out.append("# Class index -> label, as in the exported training dataset.")
        out.append(f"CLASS_NAMES = {pprint.pformat(class_names, width=96, compact=True)}")
    out += [
        "",
        "# How the data files were imported: delimiter, decimal mark, encoding, missing-value",
        "# markers and the type of every column.",
        "SOURCES = {",
        *(
            f"    {i!r}: {pprint.pformat(info, width=92, compact=True, sort_dicts=False)},".replace(
                "\n", "\n    "
            )
            for i, info in script_infos.items()
        ),
        "}",
        f"MAIN_SOURCE = {source_id!r}",
        "",
        "",
        *_reader_source().splitlines(),
        "",
        "",
        *[line for block in helpers.values() for line in [*block, "", ""]],
        "def combine(df, load_source):",
        '    """The steps that combine tables (join, append)."""',
        *_indent(combine_lines),
        "    return df",
        "",
        "",
        "def prepare(df):",
        '    """Apply the preparation steps to a table read with read_source (also new data)."""',
        "    df = df.copy()",
        *_indent(prepare_lines),
        "    return df",
        "",
        "",
        *textwrap.dedent(inspect.getsource(to_csv_frame)).splitlines(),
        "",
        "",
        "def main(argv=None):",
        "    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])",
        "    parser.add_argument(",
        '        "--input",',
        '        default=SOURCES[MAIN_SOURCE]["filename"],',
        '        help="CSV file of " + repr(SOURCES[MAIN_SOURCE]["name"]) + " (default: %(default)s)",',
        "    )",
    ]
    for i in others:
        out += [
            "    parser.add_argument(",
            f'        "--source-{i}",',
            f'        default=SOURCES[{i!r}]["filename"],',
            f'        help="CSV file of " + repr(SOURCES[{i!r}]["name"]) + " (default: %(default)s)",',
            "    )",
        ]
    out += [
        '    parser.add_argument("--output", default="prepared.csv", help="default: %(default)s")',
        "    args = parser.parse_args(argv)",
        "    paths = {MAIN_SOURCE: args.input}",
        *(f"    paths[{i!r}] = args.source_{i}" for i in others),
        "",
        "    def load_source(source_id):",
        "        return read_source(paths[source_id], SOURCES[source_id])",
        "",
        "    df = prepare(combine(load_source(MAIN_SOURCE), load_source))",
        "    to_csv_frame(df).to_csv(args.output, index=False)",
        '    print(f"Wrote {len(df):,} rows and {len(df.columns):,} columns to {args.output}.")',
        "",
        "",
        'if __name__ == "__main__":',
        "    main()",
        "",
    ]
    script = "\n".join(out)
    try:
        compile(script, f"{slug}_prepare.py", "exec")
    except SyntaxError as exc:
        logger.error("Generated pandas script does not compile: %s", exc)
        raise DataError(
            "The Python script could not be created (a step produced invalid code). "
            "Download the recipe instead."
        ) from exc
    return script
