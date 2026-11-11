"""A frozen preparation: the steps of a data preparation with the values they learned, kept
with an exported training dataset so that new **raw** rows (in the columns of the imported
source table) can be prepared exactly like the training rows, e.g. for predictions.

:func:`freeze` writes it when a dataset is exported (``<dataset>/prepare/``)::

    recipe.json     what the steps are and what they learned (see below)
    sources/<id>/   tables joined by the preparation, as they were at the export

:func:`prepare_rows` applies it to new rows. It uses the code of the pandas script that the
Data tab exports (``Step.to_pandas`` with the learned values written in as constants, see
:mod:`.export`), which reproduces the training table. So nothing is learned from the new
rows: means, categories, bounds, KNN reference rows … are the ones of the training rows.

What happens to the steps:

* **join**: the new rows are looked up in the stored copy of the other table (a left join:
  a row without a match keeps missing values, and the first match counts if a key is
  repeated);
* **append**: no rows are added; only the columns it creates are (the table name column …);
* **steps that only change rows** (drop duplicates, sort, class balancing, outlier removal
  from the training rows) are skipped; every new row is predicted;
* **row filters** (filter rows, drop rows with missing values, outlier removal from all
  rows) do not remove a new row either: they add a warning that the training data had no
  rows like it;
* all other steps run as in the script.

``recipe.json``::

    {"format": "brickdnn-frozen-preparation", "version": 1,
     "prep": {"id", "name"}, "source": {"id", "name", "decimal", "missing_values"},
     "settings": {...}, "task": "regression", "target": "sales",
     "features": [...],                  # the columns of the training table, in order
     "columns": [{"name", "kind", "categories"?, "more_categories"?, "min"?, "max"?,
                  "missing"?}, ...],     # the raw input columns (source columns, no target)
     "samples": [{"values": {column: value}, "target": value}, ...],  # validation rows
     "steps": [{"id", "type", "label", "status", "params", "fitted"}, ...],
     "sources": {"<id>": {"name", "table": "sources/<id>"}}}
"""

from __future__ import annotations

import json
import logging
import math
import re
import shutil
import textwrap
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import io as data_io
from .pipeline import PipelineResult, json_safe, make_context
from .steps import REGISTRY
from .table import KINDS as REGISTRY_KINDS
from .table import NUMERIC_KINDS, STRING_KINDS, TRAIN, VAL, Table

logger = logging.getLogger(__name__)

FROZEN_DIR = "prepare"
RECIPE_FILE = "recipe.json"
FORMAT = "brickdnn-frozen-preparation"
VERSION = 1
# Categories listed per column (for the form and to recognise unknown values).
MAX_CATEGORIES = 1000
# Validation rows kept as examples.
SAMPLE_ROWS = 5
# Messages per answer.
MAX_MESSAGES = 20

# Steps that only remove, add or reorder rows: skipped for new rows.
SKIPPED_STEPS = ("drop_duplicates", "sort_rows")
# Steps that remove rows by a condition: new rows are kept, with a warning.
FILTER_STEPS = ("filter_rows", "drop_missing", "outliers")


class RawInputError(ValueError):
    """New rows that cannot be prepared; ``messages`` say why (per row and column)."""

    def __init__(self, messages: list[str]):
        self.messages = _capped(messages)
        super().__init__(self.messages[0] if self.messages else "The rows are invalid.")


@dataclass
class PreparedRows:
    """New rows after the preparation: one ``{feature: number}`` per row, in input order."""

    rows: list[dict[str, float]]
    warnings: list[str] = field(default_factory=list)


def _capped(messages: list[str]) -> list[str]:
    if len(messages) <= MAX_MESSAGES:
        return list(messages)
    return [*messages[:MAX_MESSAGES], f"… and {len(messages) - MAX_MESSAGES:,} more."]


# --------------------------------------------------------------------------- #
# Freezing at the export
# --------------------------------------------------------------------------- #


def _jsonable_cell(value: Any, kind: str) -> Any:
    """A cell of a stored table as a JSON value (dates as ISO text, booleans true/false)."""
    if value is None or (isinstance(value, float) and math.isnan(value)) or value is pd.NaT:
        return None
    if kind == "datetime":
        stamp = pd.Timestamp(value)
        if pd.isna(stamp):
            return None
        text = stamp.strftime("%Y-%m-%dT%H:%M:%S")
        return text[:10] if text.endswith("T00:00:00") else text
    if kind == "boolean":
        return bool(value)
    if kind == "numeric":
        number = float(value)
        if not math.isfinite(number):
            return None
        return int(number) if number.is_integer() and abs(number) < 2**53 else number
    return str(value)


def _column_facts(name: str, kind: str, values: pd.Series) -> dict[str, Any]:
    """What the form needs to know about a raw input column, from its training values."""
    facts: dict[str, Any] = {"name": name, "kind": kind}
    present = values.dropna()
    if len(present) < len(values):
        facts["missing"] = True
    if kind == "numeric" and len(present):
        numbers = present.to_numpy(dtype=float)
        facts["min"] = _jsonable_cell(float(numbers.min()), "numeric")
        facts["max"] = _jsonable_cell(float(numbers.max()), "numeric")
        facts["integer"] = bool(np.all(np.floor(numbers) == numbers))
    elif kind == "datetime" and len(present):
        facts["min"] = _jsonable_cell(present.min(), "datetime")
        facts["max"] = _jsonable_cell(present.max(), "datetime")
    elif kind in STRING_KINDS:
        counts = present.astype(str).value_counts(sort=True)
        names = [str(v) for v in counts.index[:MAX_CATEGORIES]]
        facts["categories"] = names
        if len(counts) > MAX_CATEGORIES:
            facts["more_categories"] = True
    return facts


def freeze(
    directory: str | Path,
    *,
    result: PipelineResult,
    steps: list[dict[str, Any]],
    source: Table,
    source_info: dict[str, Any],
    prep: dict[str, Any],
    load_source: Any = None,
    source_name: Any = None,
) -> int:
    """Write the frozen preparation of ``result`` (the pipeline run that was exported) into
    ``directory``. ``source`` is the imported main table, ``load_source`` loads the other
    sources of join steps (they are stored as they are now). Returns the bytes written."""
    from .store import save_table

    directory = Path(directory)
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    settings = result.settings
    target = settings.get("target")
    reports = {r["id"]: r for r in result.reports}
    before = result.before  # after the combine steps, with the split of every row
    train = before.train_mask()

    unused = _unused_columns(steps, result, [c for c in source.columns if c != target])
    columns = []
    for name in source.columns:
        if name == target:
            continue
        kind = source.kinds[name]
        values = before.df[name][train] if name in before.kinds else source.df[name]
        facts = _column_facts(name, kind, values)
        if name in unused:
            facts["used"] = False
        columns.append(facts)

    samples = []
    rows = np.flatnonzero(before.split == VAL) if before.split is not None else []
    if len(rows) == 0:
        rows = np.flatnonzero(before.split != TRAIN) if before.split is not None else []
    for row in list(rows)[:SAMPLE_ROWS]:
        values = {
            c["name"]: _jsonable_cell(before.df[c["name"]].iloc[row], c["kind"])
            for c in columns
            if c["name"] in before.kinds
        }
        sample: dict[str, Any] = {"values": values}
        if target and target in before.kinds:
            sample["target"] = _jsonable_cell(before.df[target].iloc[row], before.kinds[target])
        samples.append(sample)

    frozen_steps = []
    sources: dict[str, Any] = {}
    written = 0
    for step in steps:
        cls = REGISTRY.get(step.get("type"))
        report = reports.get(step["id"]) or {}
        status = report.get("status") or "not_run"
        entry = {
            "id": step["id"],
            "type": step["type"],
            "label": cls.label if cls else step["type"],
            "status": status,
            "params": result.params.get(step["id"]),
            "fitted": result.fitted.get(step["id"]),
        }
        frozen_steps.append(entry)
        if status != "ok" or cls is None or step["type"] != "join":
            continue
        source_id = (entry["params"] or {}).get("source")
        if source_id is None or str(source_id) in sources or load_source is None:
            continue
        other = load_source(source_id)
        path = f"sources/{int(source_id)}"
        written += save_table(other, directory / path)
        name = source_name(source_id) if source_name is not None else f"source {source_id}"
        sources[str(source_id)] = {"name": str(name), "table": path}

    features = [c for c in result.table.columns if c != target]
    document = json_safe(
        {
            "format": FORMAT,
            "version": VERSION,
            "prep": prep,
            "source": {
                "id": source_info.get("id"),
                "name": source_info.get("name"),
                "decimal": source_info.get("decimal") or ".",
                "missing_values": source_info.get("missing_values"),
            },
            "settings": settings,
            "task": result.task,
            "target": target,
            "target_kind": source.kinds.get(target) if target else None,
            "features": features,
            "columns": columns,
            "samples": samples,
            "steps": frozen_steps,
            "sources": sources,
        }
    )
    text = json.dumps(document, ensure_ascii=False, allow_nan=False)
    (directory / RECIPE_FILE).write_text(text, encoding="utf-8")
    return written + len(text.encode())


# Steps that remove columns without reading them.
DROP_STEPS = ("drop_columns", "keep_columns", "drop_sparse_columns", "feature_selection")


def _unused_columns(
    steps: list[dict[str, Any]], result: PipelineResult, names: list[str]
) -> set[str]:
    """Source columns that the preparation removes before any step reads them: they do not
    change the prepared rows. (A column counts as read when a step's settings or learned
    values mention its name, which may also count some columns that are not read.)"""
    reports = {r["id"]: r for r in result.reports}
    untouched = set(names)
    unused: set[str] = set()
    for step in steps:
        report = reports.get(step["id"]) or {}
        if report.get("status") != "ok" or not untouched:
            continue
        if _keeps_values(step["type"], result.params.get(step["id"]) or {}):
            continue  # changes no values of the rows it keeps
        removed = set(report.get("removed") or [])
        if step["type"] in DROP_STEPS:
            unused |= untouched & removed
            untouched -= removed
            continue
        params = result.params.get(step["id"]) or {}
        if params.get("columns") == []:  # "all suitable columns"
            untouched.clear()
            continue
        fitted = result.fitted.get(step["id"]) or {}
        text = json.dumps(
            [params, {k: v for k, v in fitted.items() if k[:1] != "_"}], ensure_ascii=False
        )
        expression = str(params.get("expression") or "")
        untouched -= {
            c
            for c in untouched
            if c in removed or json.dumps(c, ensure_ascii=False) in text or c in expression
        }
    return unused


def _keeps_values(step_type: str, params: dict[str, Any]) -> bool:
    """Whether a step only removes, adds or reorders rows (no values change)."""
    cls = REGISTRY.get(step_type)
    if cls is None or cls.training_only or step_type in (*SKIPPED_STEPS, "filter_rows"):
        return True
    if step_type == "drop_missing":
        return True
    return step_type == "outliers" and params.get("action") == "remove"


def read_recipe(directory: str | Path) -> dict[str, Any] | None:
    """The recipe of a frozen preparation, None if there is none (or it is unreadable)."""
    path = Path(directory) / RECIPE_FILE
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(document, dict) or document.get("format") != FORMAT:
        return None
    if document.get("version") != VERSION:
        return None
    return document


def describe(recipe: dict[str, Any]) -> dict[str, Any]:
    """The raw input columns and example rows (for the prediction form and the API)."""
    return {
        "prep": recipe.get("prep"),
        "source": (recipe.get("source") or {}).get("name"),
        "target": recipe.get("target"),
        "columns": recipe.get("columns") or [],
        "samples": recipe.get("samples") or [],
        "skipped_steps": [
            f"{s['label']}"
            for s in recipe.get("steps") or []
            if s.get("status") == "ok" and _skipped(s)
        ],
        "joined": [s.get("name") for s in (recipe.get("sources") or {}).values()],
    }


def _skipped(step: dict[str, Any]) -> bool:
    cls = REGISTRY.get(step.get("type"))
    return cls is not None and (cls.training_only or step["type"] in SKIPPED_STEPS)


# --------------------------------------------------------------------------- #
# Reading new rows
# --------------------------------------------------------------------------- #


def _blank(value: Any, missing: set[str]) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, str) and value.strip() in missing


def _number_cell(value: Any, decimal: str) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, str):
        text = value.strip()
        for mark in dict.fromkeys((decimal, ".")):
            number = float(data_io.parse_numbers([text], mark)[0])
            if math.isfinite(number):
                return number
    return None


def _boolean_cell(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and value in (0, 1):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if text in ("0", "1"):
            return float(text)
        parsed = float(data_io.parse_booleans([text])[0])
        return None if math.isnan(parsed) else parsed
    return None


def _date_cell(value: Any) -> np.datetime64 | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    families = [f for f, pattern in data_io._DATE_PATTERNS.items() if pattern.fullmatch(text)]
    for family in families:
        parsed = data_io.parse_dates([text], family)[0]
        if not np.isnat(parsed):
            return parsed
    return None


def _string_cell(value: Any) -> str | None:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and math.isfinite(value):
        number = float(value)
        return str(int(number)) if number.is_integer() and abs(number) < 1e15 else repr(number)
    return None


_WANTED = {
    "numeric": "a number",
    "boolean": "true or false",
    "datetime": "a date such as 2024-05-31 (or 2024-05-31T14:30)",
    "categorical": "a text value",
    "text": "a text value",
}


def _label(number: int, count: int) -> str:
    return f"Row {number}" if count > 1 else "The row"


def _read_rows(recipe: dict[str, Any], rows: Any) -> tuple[pd.DataFrame, list[str], list[set]]:
    """The new rows as a DataFrame in the storage format of the source table, warnings,
    and per row the raw columns without a value. Raises :class:`RawInputError`."""
    columns = [c for c in recipe.get("columns") or [] if isinstance(c, dict)]
    names = [str(c["name"]) for c in columns]
    target = recipe.get("target")
    source = recipe.get("source") or {}
    decimal = source.get("decimal") or "."
    missing = source.get("missing_values")
    tokens = {str(t).strip() for t in (missing if isinstance(missing, list) else ())}
    tokens.add("")
    wanted = f"an object {{column: value}} or a list of {len(names)} values ({', '.join(names)})"
    if isinstance(rows, dict) or (
        isinstance(rows, list) and rows and not isinstance(rows[0], (list, dict))
    ):
        rows = [rows]
    if not isinstance(rows, list) or not rows:
        raise RawInputError([f"'inputs' must be a list of rows, each {wanted}."])
    count = len(rows)
    errors: list[str] = []
    warnings: list[str] = []
    cells: dict[str, list[Any]] = {name: [] for name in names}
    empty: list[set] = []
    not_given: dict[str, None] = {}
    known = {
        c["name"]: set(c["categories"])
        for c in columns
        if isinstance(c.get("categories"), list)
        and not c.get("more_categories")
        and c.get("used", True)
    }
    used = {c["name"] for c in columns if c.get("used", True)}
    for number, row in enumerate(rows, start=1):
        label = _label(number, count)
        if isinstance(row, list):
            if len(row) != len(names):
                errors.append(
                    f"{label} has {len(row)} values, but {len(names)} are needed: {wanted}."
                )
                row = {}
            else:
                row = dict(zip(names, row, strict=True))
        if not isinstance(row, dict):
            errors.append(f"{label} must be {wanted}.")
            row = {}
        unknown = [str(k) for k in row if k not in cells and k != target]
        if unknown:
            errors.append(
                f"{label}: unknown column '{unknown[0][:60]}'. The columns of "
                f"'{source.get('name') or 'the source table'}' are: {', '.join(names)}."
            )
        blank: set = set()
        for column in columns:
            name, kind = column["name"], column["kind"]
            if name not in row and name in used:
                not_given[name] = None
            value = row.get(name)
            if _blank(value, tokens):
                cells[name].append(None)
                blank.add(name)
                continue
            if kind == "numeric":
                parsed: Any = _number_cell(value, decimal)
            elif kind == "boolean":
                parsed = _boolean_cell(value)
            elif kind == "datetime":
                parsed = _date_cell(value)
            else:
                parsed = _string_cell(value)
            if parsed is None:
                shown = repr(value)[:40]
                errors.append(f"{label}, '{name}': {shown} is not {_WANTED.get(kind, 'valid')}.")
            elif name in known and parsed not in known[name]:
                warnings.append(
                    f"{label}, '{name}': '{str(parsed)[:40]}' does not occur in the training "
                    "data; the model treats it as an unknown category."
                )
            cells[name].append(parsed)
            if parsed is None:
                blank.add(name)
        empty.append(blank)
    if errors:
        raise RawInputError(errors)
    if not_given:
        shown = ", ".join(list(not_given)[:10]) + (" …" if len(not_given) > 10 else "")
        warnings.insert(0, f"Columns not given, treated as missing values: {shown}.")
    frame = {}
    for column in columns:
        name, kind = column["name"], column["kind"]
        if kind in NUMERIC_KINDS:
            frame[name] = pd.Series(
                [np.nan if v is None else v for v in cells[name]], dtype="float64"
            )
        elif kind == "datetime":
            frame[name] = pd.Series(
                [np.datetime64("NaT") if v is None else v for v in cells[name]],
                dtype="datetime64[ns]",
            )
        else:
            frame[name] = pd.Series(cells[name], dtype=object)
    df = pd.DataFrame(frame, index=pd.RangeIndex(count))
    return df, warnings, empty


# --------------------------------------------------------------------------- #
# The code of the steps
# --------------------------------------------------------------------------- #

_IMPORTS = """\
import bz2
import csv
import datetime
import gzip
import io
import json
import lzma
import math
import re
import sys
import zipfile
from xml.etree.ElementTree import iterparse

import numpy as np
import pandas as pd
"""


@dataclass
class _Program:
    """The step functions of a frozen preparation (``step_<n>(df) -> df``)."""

    namespace: dict[str, Any]
    functions: dict[str, str]  # step id -> function name


_PROGRAMS: OrderedDict[tuple[str, float], _Program] = OrderedDict()
_PROGRAMS_LOCK = threading.Lock()
MAX_PROGRAMS = 8


def _step_code(step: dict[str, Any], recipe: dict[str, Any]) -> list[str]:
    from .export import comment

    cls = REGISTRY[step["type"]]
    ctx = make_context(
        None,  # type: ignore[arg-type]  # the table is not needed for the code
        recipe.get("settings") or {},
        recipe.get("task"),
        source_id=(recipe.get("source") or {}).get("id"),
    )
    params = step.get("params") or {}
    lines = cls(params).to_pandas(params, step.get("fitted"), ctx)
    body: list[str] = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("#"):
            body.append(line[: len(line) - len(stripped)] + "# " + comment(stripped[1:]))
        else:
            body += line.split("\n")
    return body


def _program(recipe: dict[str, Any], key: tuple[str, float] | None) -> _Program:
    """Compile the code of the steps that run on new rows (cached per recipe file)."""
    from .export import _hoist, _indent, _reader_source

    if key is not None:
        with _PROGRAMS_LOCK:
            if key in _PROGRAMS:
                _PROGRAMS.move_to_end(key)
                return _PROGRAMS[key]
    helpers: dict[str, list[str]] = {}
    blocks: list[str] = []
    functions: dict[str, str] = {}
    for number, step in enumerate(recipe.get("steps") or [], start=1):
        if not _runs_as_code(step):
            continue
        body = _hoist(_step_code(step, recipe), helpers)
        name = f"step_{number}"
        functions[step["id"]] = name
        blocks += [f"def {name}(df):", *_indent(body), "    return df", "", ""]
    source = "\n".join(
        [
            _IMPORTS,
            _reader_source(),
            "",
            *[line for block in helpers.values() for line in [*block, "", ""]],
            *blocks,
        ]
    )
    namespace: dict[str, Any] = {"__name__": "brickdnn_frozen_preparation"}
    exec(compile(source, "<frozen preparation>", "exec"), namespace)  # noqa: S102 (our code)
    program = _Program(namespace, functions)
    if key is not None:
        with _PROGRAMS_LOCK:
            _PROGRAMS[key] = program
            while len(_PROGRAMS) > MAX_PROGRAMS:
                _PROGRAMS.popitem(last=False)
    return program


def _runs_as_code(step: dict[str, Any]) -> bool:
    cls = REGISTRY.get(step.get("type"))
    if cls is None or step.get("status") != "ok":
        return False
    if cls.combine or cls.training_only or step["type"] in SKIPPED_STEPS:
        return False
    if step["type"] == "outliers" and (step.get("params") or {}).get("action") == "remove":
        return False
    return step["type"] not in ("filter_rows", "drop_missing")


# --------------------------------------------------------------------------- #
# Applying the steps
# --------------------------------------------------------------------------- #


def _join(
    df: pd.DataFrame, step: dict[str, Any], other: Table, name: str
) -> tuple[pd.DataFrame, list[int]]:
    """A left join with the stored table (first match per key), like the join's code.
    Returns the frame and the rows (0-based) without a match."""
    plan = (step.get("fitted") or {}).get("_plan") or {}
    params = step.get("params") or {}
    temps, left_on = list(plan.get("temps") or []), list(params.get("left_on") or [])
    right = other.df.rename(columns=plan.get("rename") or {})
    right = right[right[temps].notna().all(axis=1)].drop_duplicates(subset=temps, keep="first")
    flag = "__brickdnn_match__"
    merged = df.merge(
        right, how="left", left_on=left_on, right_on=temps, sort=False, indicator=flag
    )
    keyed = df[left_on].notna().all(axis=1).to_numpy()
    alone = (merged[flag].to_numpy() == "left_only") & keyed
    unmatched = [int(i) for i in np.flatnonzero(alone)]
    merged = merged.drop(columns=[*temps, flag])
    for column in merged.columns:
        if column not in df.columns and other.kinds.get(_original(plan, column)) in STRING_KINDS:
            merged[column] = merged[column].where(merged[column].notna(), None)
    merged.index = df.index
    return merged, unmatched


def _original(plan: dict[str, Any], column: str) -> str:
    for old, new in (plan.get("rename") or {}).items():
        if new == column:
            return old
    return column


def _append(df: pd.DataFrame, step: dict[str, Any]) -> pd.DataFrame:
    """The columns an append step creates in the rows of the main table (no rows added)."""
    from .steps.columns import values_as_text
    from .steps.combine import _missing_series

    plan = (step.get("fitted") or {}).get("_plan") or {}
    df = df.copy()
    for column, kind in (plan.get("text_here") or {}).items():
        if column in df.columns:
            df[column] = values_as_text(df[column], kind)
    for column, kind in (plan.get("add_here") or {}).items():
        df[column] = _missing_series(df.index, kind)
    column = plan.get("column")
    if column and column not in df.columns:
        df[column] = plan.get("label_here")
    return df


def _outside_rows(
    df: pd.DataFrame, step: dict[str, Any], target: str | None, unused: set[str]
) -> list[str] | None:
    """Per row: the condition text if the step would have removed it from the training data
    (None: it would keep every row)."""
    from .steps.outliers import _code_outside
    from .steps.rows import _condition_text, _filter_code, _typed

    params = step.get("params") or {}
    kind = step["type"]
    namespace = {"df": df, "np": np, "pd": pd}
    if kind == "filter_rows":
        if params.get("column") == target or params.get("column") not in df.columns:
            return None
        typed = _typed(params)
        keep = eval(_filter_code(typed), namespace)  # noqa: S307 (code of our own step)
        text = f"rows where {_condition_text(params['column'], typed)}"
        return [None if bool(k) else text for k in keep.to_numpy(dtype=bool)]
    if kind == "drop_missing":
        columns = [
            c
            for c in (params.get("columns") or list(df.columns))
            if c != target and c not in unused
        ]
        columns = [c for c in columns if c in df.columns]
        if not columns:
            return None
        missing = df[columns].isna()
        drop = missing.any(axis=1) if params.get("how", "any") == "any" else missing.all(axis=1)
        result = []
        for i, dropped in enumerate(drop.to_numpy(dtype=bool)):
            names = [c for c in columns if missing[c].iloc[i]]
            result.append(f"rows with a value in {', '.join(names[:3])}" if dropped else None)
        return result
    if (
        kind == "outliers"
        and params.get("action") == "remove"
        and params.get("remove_from") == "all"
    ):
        bounds = {
            c: b
            for c, b in ((step.get("fitted") or {}).get("bounds") or {}).items()
            if c != target and c in df.columns
        }
        result = [None] * len(df)
        for column, (lower, upper) in bounds.items():
            outside = eval(_code_outside(column, lower, upper), namespace)  # noqa: S307
            for i in np.flatnonzero(outside.to_numpy(dtype=bool)):
                result[i] = result[i] or f"values of {column} within {lower:g} … {upper:g}"
        return result
    return None


def prepare_rows(
    recipe: dict[str, Any],
    rows: Any,
    *,
    directory: str | Path | None = None,
    max_rows: int | None = None,
) -> PreparedRows:
    """Prepare new raw rows (``[{column: value}, ...]`` in the columns of the source table;
    a target value is ignored) with the frozen preparation ``recipe``. ``directory`` holds
    its stored tables (joins). Raises :class:`RawInputError`."""
    from .store import load_table_cached

    if max_rows is not None and isinstance(rows, list) and len(rows) > max_rows:
        raise RawInputError(
            [f"{len(rows)} rows are too many: predict at most {max_rows} at a time."]
        )
    df, warnings, empty = _read_rows(recipe, rows)
    count = len(df)
    target = recipe.get("target")
    source_kinds = {c["name"]: c["kind"] for c in recipe.get("columns") or []}
    unused = {c["name"] for c in recipe.get("columns") or [] if not c.get("used", True)}
    key = None
    if directory is not None:
        path = Path(directory) / RECIPE_FILE
        try:
            key = (str(path.resolve()), path.stat().st_mtime)
        except OSError:
            key = None
    program = _program(recipe, key)
    labels = {}
    misses: list[list[str]] = [[] for _ in range(count)]  # per row: joins without a match
    # The target is unknown for new rows: a column of missing values, where the source had it.
    target_kind = recipe.get("target_kind") if target else None
    if target and target_kind in REGISTRY_KINDS and target not in df.columns:
        from .steps.combine import _missing_series

        df[target] = _missing_series(df.index, target_kind)

    for number, step in enumerate(recipe.get("steps") or [], start=1):
        cls = REGISTRY.get(step.get("type"))
        if cls is None or step.get("status") != "ok":
            continue
        labels[step["id"]] = f"step {number} ({step.get('label') or cls.label})"
        if step["type"] == "join":
            source_id = str((step.get("params") or {}).get("source"))
            stored = (recipe.get("sources") or {}).get(source_id)
            if stored is None or directory is None:
                raise RawInputError(
                    ["The table joined by the preparation was not stored with the dataset."]
                )
            other = load_table_cached(Path(directory) / stored["table"])
            df, unmatched = _join(df, step, other, stored.get("name") or "")
            how = (step.get("params") or {}).get("how")
            for i in unmatched:
                misses[i].append(f"no match in '{stored.get('name')}'")
                note = (
                    "the training data only has rows with a match"
                    if how in ("inner", "right")
                    else "its columns are missing"
                )
                warnings.append(
                    f"{_label(i + 1, count)}: no match in '{stored.get('name')}' "
                    f"({labels[step['id']]}); {note}."
                )
            continue
        if step["type"] == "append":
            df = _append(df, step)
            continue
        if step["type"] in FILTER_STEPS and not _runs_as_code(step):
            outside = _outside_rows(df, step, target, unused)
            for i, text in enumerate(outside or []):
                if text:
                    warnings.append(
                        f"{_label(i + 1, count)}: the training data only has {text} "
                        f"({labels[step['id']]}), so the model never saw rows like this one."
                    )
            continue
        function = program.functions.get(step["id"])
        if function is None:
            continue
        try:
            df = program.namespace[function](df)
        except Exception as exc:  # the user's values may not fit, e.g. a formula
            logger.info("Frozen step %s failed on new rows: %s", step["id"], exc)
            raise RawInputError(
                [f"The preparation failed in {labels[step['id']]}: {_short(exc)}"]
            ) from None
        if not isinstance(df, pd.DataFrame) or len(df) != count:
            raise RawInputError(
                [f"{labels[step['id']].capitalize()} changes the rows; it cannot prepare new rows."]
            )
        df.index = pd.RangeIndex(count)

    features = [str(c) for c in recipe.get("features") or []]
    absent = [c for c in features if c not in df.columns]
    if absent:
        raise RawInputError(
            [f"The preparation did not create the column '{absent[0]}' for the new rows."]
        )
    errors: list[str] = []
    prepared: list[dict[str, float]] = []
    values = {}
    for name in features:
        series = df[name]
        if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
            values[name] = series.to_numpy(dtype=float, na_value=np.nan)
        else:
            values[name] = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    for i in range(count):
        row = {}
        gaps = []
        for name in features:
            x = float(values[name][i])
            if not math.isfinite(x):
                gaps.append(name)
            row[name] = x
        if gaps:
            # Raw columns without a value that the empty columns come from (by name), and
            # joins without a match.
            blank = [c for c in source_kinds if c in empty[i] and any(c in g for g in gaps)]
            causes = [f"no value given for {', '.join(blank[:5])}"] if blank else []
            causes += misses[i]
            hint = f" ({'; '.join(causes)})" if causes else ""
            errors.append(
                f"{_label(i + 1, count)}: after the preparation, "
                f"{_quoted(gaps)} {'has' if len(gaps) == 1 else 'have'} no value{hint}. "
                "The model needs a number in every feature column."
            )
        prepared.append(row)
    if errors:
        raise RawInputError(errors)
    return PreparedRows(prepared, _capped(warnings))


def _quoted(names: list[str]) -> str:
    shown = ", ".join(f"'{n}'" for n in names[:3])
    return shown + (f" and {len(names) - 3} more" if len(names) > 3 else "")


def _short(exc: BaseException) -> str:
    text = re.sub(r"\s+", " ", str(exc)).strip() or type(exc).__name__
    return textwrap.shorten(text, 200)
