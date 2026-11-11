"""Data preparation (the Data tab) for the MCP server: compact answers and step editing.

The data tools in :mod:`brick_dnn_core.mcp` do their work with the same functions as the REST
API (:mod:`data_service`, the serializers, :mod:`shared_data`). This module turns the results
into small answers an AI assistant can read (rows, columns and lists capped, dates as ISO
text, numbers rounded, a ``fix_id`` for every readiness fix) and edits step lists like the
web app does (ids ``s1``, ``s2`` …, the defaults of new steps, where a new step goes).

pandas is only imported by the functions that need it, like in data_service.
"""

from __future__ import annotations

import base64
import binascii
import csv
import hashlib
import io
import json
import math
import re
from datetime import datetime, timezone
from typing import Any

from django.conf import settings

from . import data_service as data
from .models import DataPrep, DataSource

# What answers show at most.
MAX_NAMES = 100  # column names in error messages and lists of names
COLUMN_PAGE = 50  # columns (name: kind) per answer; columns_offset pages
MAX_PROFILES = 20  # column profiles in a preview
MAX_SAMPLE_COLUMNS = 20  # columns of sample rows
DEFAULT_SAMPLE_ROWS = 5
MAX_SAMPLE_ROWS = 50
MAX_CLASSES = 30  # class names and counts
MAX_ISSUE_COLUMNS = 10
MAX_FIX_PARAMS = 600  # characters of the params of a fix shown in full
MAX_TEXT = 200  # characters of a cell value
MAX_DOWNLOAD_BYTES = 300_000  # pandas script, recipe and CSV text
DEFAULT_CSV_ROWS = 100
MAX_CSV_ROWS = 1000

STEP_ID = re.compile(r"s(\d+)")


class DataToolError(ValueError):
    """A request an assistant can correct; the message says how."""


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _number(value: Any) -> Any:
    """Floats with 6 significant digits (ints and others unchanged)."""
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        rounded = float(f"{value:.6g}")
        return int(rounded) if rounded.is_integer() and abs(rounded) < 1e15 else rounded
    return value


def _iso(ms: Any) -> str | None:
    """Milliseconds since the epoch (how profiles and samples carry dates) as ISO text."""
    if not isinstance(ms, (int, float)) or isinstance(ms, bool) or not math.isfinite(ms):
        return None
    try:
        moment = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None
    text = moment.replace(tzinfo=None).isoformat()
    return text[:10] if text.endswith("T00:00:00") else text


def _cell(value: Any, kind: str | None = None) -> Any:
    if kind == "datetime" and isinstance(value, (int, float)) and not isinstance(value, bool):
        return _iso(value)
    if kind == "datetime" and isinstance(value, str) and value.endswith("T00:00:00"):
        return value[:10]
    if isinstance(value, str) and len(value) > MAX_TEXT:
        return value[: MAX_TEXT - 1] + "…"
    return _number(value)


def _capped(names: list[Any], limit: int = MAX_NAMES) -> list[Any]:
    """``names`` with at most ``limit`` entries; a last entry says how many were left out."""
    if len(names) <= limit:
        return list(names)
    return [*names[:limit], f"… and {len(names) - limit:,} more"]


def _drop_empty(data: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in data.items() if v not in (None, "", [], {})}


def names_text(names: list[str], limit: int = 30) -> str:
    shown = ", ".join(names[:limit])
    return shown + (f" and {len(names) - limit:,} more" if len(names) > limit else "")


def serializer_message(errors: Any) -> str:
    """The message of DRF validation errors, as the REST API shows it, plus a hint."""
    from .data_views import _first_message

    message = _first_message(errors)
    if "unknown step type" in message:
        message += " list_prep_steps shows the step types."
    elif isinstance(errors, dict) and "source" in errors:
        message += " list_data_sources shows the user's data sources."
    return message


# --------------------------------------------------------------------------- #
# Data sources
# --------------------------------------------------------------------------- #


def import_limit() -> int:
    """Largest file import_data takes as content, in bytes: BRICKDNN_MCP_UPLOAD_MB, at most
    the import limit of the Data tab (BRICKDNN_DATA_MAX_UPLOAD_MB)."""
    limit = int(float(settings.MCP_UPLOAD_MAX_MB) * 1024 * 1024)
    return max(0, min(limit, data.max_upload_bytes()))


def import_info() -> dict[str, Any]:
    """What import_data accepts on this server."""
    status = data.status()
    return {
        "max_mb": round(import_limit() / 1024**2, 2),
        "file_types": status["file_types"],
        "links": status["url_import"],
        "max_rows": status["max_rows"],
    }


def decode_content(args: dict[str, Any]) -> tuple[bytes, str]:
    """The file bytes and file name of an import_data request with ``content`` (text) or
    ``content_base64``. Raises :class:`DataToolError`."""
    filename = args.get("filename")
    if filename is not None and (not isinstance(filename, str) or not filename.strip()):
        raise DataToolError("'filename' must be a file name such as 'sales.csv'.")
    if args.get("content") is not None:
        text = args["content"]
        if not isinstance(text, str) or not text.strip():
            raise DataToolError("'content' must be the content of the file as text.")
        if args.get("encoding") not in (None, "", "auto"):
            raise DataToolError(
                "'encoding' only applies to content_base64 and url: text in 'content' is "
                "already decoded. Leave it out."
            )
        return text.encode("utf-8"), (filename or "data.csv").strip()
    encoded = args["content_base64"]
    if not isinstance(encoded, str) or not encoded.strip():
        raise DataToolError("'content_base64' must be the file encoded as base64.")
    if not filename:
        raise DataToolError(
            "Binary files need 'filename' with the right ending (e.g. 'sales.xlsx' or "
            "'events.parquet'): it tells the import which format to read."
        )
    try:
        content = base64.b64decode("".join(encoded.split()), validate=True)
    except (binascii.Error, ValueError):
        raise DataToolError("'content_base64' is not valid base64.") from None
    if not content:
        raise DataToolError("'content_base64' holds an empty file.")
    return content, filename.strip()


def _report(source: DataSource) -> dict[str, Any]:
    report = source.report if isinstance(source.report, dict) else {}
    keys = ("format", "delimiter", "decimal", "header", "encoding", "sheet", "sheets")
    shown = {key: report.get(key) for key in keys}
    shown.update(
        compression=report.get("compression"),
        file_in_zip=report.get("member"),
        json_path=report.get("json_path"),
        url=report.get("url"),
        warnings=report.get("warnings"),
    )
    if shown.get("delimiter") == "\t":
        shown["delimiter"] = "tab"
    return {k: v for k, v in shown.items() if v not in (None, "", [])}


def source_summary(source: DataSource) -> dict[str, Any]:
    report = source.report if isinstance(source.report, dict) else {}
    return {
        "id": source.pk,
        "name": source.name,
        "filename": source.filename,
        "format": report.get("format"),
        "rows": source.n_rows,
        "columns": source.n_columns,
        "created_at": source.created_at.isoformat(),
    }


def _column_text(column: dict[str, Any]) -> str:
    missing = column.get("missing") or 0
    return f"{column.get('kind')} ({missing:,} missing)" if missing else str(column.get("kind"))


def page_note(offset: int, shown: int, total: int, what: str = "columns") -> str | None:
    """How to see more of a list that is shown a page at a time."""
    if offset == 0 and shown >= total:
        return None
    note = f"{what.capitalize()} {offset + 1}–{offset + shown} of {total:,}"
    if offset + shown < total:
        note += f"; columns_offset={offset + shown} shows the next ones"
    return note + "."


def offset_arg(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DataToolError("'columns_offset' must be a whole number of at least 0.")
    return value


def source_detail(
    source: DataSource, user, sample_rows: int = DEFAULT_SAMPLE_ROWS, offset: int = 0
) -> dict:
    """A source with a page of its columns (name: kind), the import report, its first rows
    (the first columns of the page) and its preparations."""
    from .shared_data import visible_preps

    columns = [c for c in (source.columns or []) if isinstance(c, dict)]
    page = columns[offset : offset + COLUMN_PAGE]
    preps = list(
        source.preps.filter(pk__in=visible_preps(user).values("pk"))
        .order_by("name", "id")
        .values("id", "name")
    )
    detail: dict[str, Any] = {
        **source_summary(source),
        "size_bytes": source.size_bytes,
        "mine": source.owner_id == user.pk,
        "import": _report(source),
        "columns": {str(c.get("name")): _column_text(c) for c in page},
    }
    if note := page_note(offset, len(page), len(columns)):
        detail["columns_note"] = note
    if sample_rows > 0 and page:
        table = data.source_table(source)
        names = [c.get("name") for c in page if c.get("name") in table.kinds]
        names = names[:MAX_SAMPLE_COLUMNS]
        head = table.df[names].head(sample_rows)
        detail["first_rows"] = [
            {name: _cell(_plain(row[name]), table.kinds[name]) for name in names}
            for _, row in head.iterrows()
        ]
        if len(names) < len(table.columns):
            detail["first_rows_note"] = (
                f"{len(names)} of {len(table.columns):,} columns (the first of this page)."
            )
    detail["preparations"] = [{"id": p["id"], "name": p["name"]} for p in preps]
    numbers = sum(c.get("kind") in ("numeric", "boolean") for c in columns)
    if numbers >= 3:
        from brickdnn.data.relations import column_relations, to_json

        table = data.source_table(source)
        relations = relations_view(to_json(column_relations(table, cluster=False)))
        if relations:
            detail["column_relations"] = relations
    return detail


def _plain(value: Any) -> Any:
    """A cell of a stored table as a JSON value (dates as ISO text)."""
    from brickdnn.data.pipeline import json_safe

    return json_safe(value)


# --------------------------------------------------------------------------- #
# Preparations
# --------------------------------------------------------------------------- #


def prep_summary(prep: DataPrep, user) -> dict[str, Any]:
    from .shared_data import prep_role

    prep_settings = prep.settings if isinstance(prep.settings, dict) else {}
    steps = prep.steps if isinstance(prep.steps, list) else []
    return {
        "id": prep.pk,
        "name": prep.name,
        "source": {"id": prep.source_id, "name": prep.source.name},
        "target": prep_settings.get("target"),
        "task": prep_settings.get("task"),
        "steps": len(steps),
        "owner": prep.owner.get_username(),
        "role": prep_role(user, prep),
        "updated_at": prep.updated_at.isoformat(),
    }


def prep_detail(prep: DataPrep, request) -> dict[str, Any]:
    """A preparation as the REST API shows it, with training references for its datasets."""
    from .serializers import DataPrepSerializer

    serialized = DataPrepSerializer(prep, context={"request": request}).data
    return {
        "id": serialized["id"],
        "name": serialized["name"],
        "owner": serialized.get("owner"),
        "role": serialized.get("role"),
        "shared_in": serialized.get("shared_in") or [],
        "source": {
            "id": prep.source_id,
            "name": prep.source.name,
            "rows": prep.source.n_rows,
            "columns": prep.source.n_columns,
        },
        "settings": serialized["settings"],
        "steps": serialized["steps"],
        "datasets": [
            {**dataset, "reference": {"kind": "upload", "id": dataset["id"]}}
            for dataset in serialized["datasets"]
        ],
        "updated_at": serialized["updated_at"],
    }


def settings_from(args: dict[str, Any], current: dict[str, Any] | None = None) -> dict | None:
    """Settings of create_data_prep / update_data_prep: ``settings`` (all of them) and/or
    ``target``, ``task``, ``split`` (changed on top). None: nothing to change."""
    given = args.get("settings")
    keys = [key for key in ("target", "task", "split") if key in args]
    if given is None and not keys:
        return None
    if given is not None and not isinstance(given, dict):
        raise DataToolError("'settings' must be an object: {target, task, split}.")
    result = dict(given if given is not None else (current or {}))
    for key in keys:
        value = args[key]
        if key == "split" and isinstance(value, dict) and isinstance(result.get("split"), dict):
            value = {**result["split"], **value}  # change single split settings
        result[key] = value
    return result


# --------------------------------------------------------------------------- #
# Step catalogue
# --------------------------------------------------------------------------- #

_KIND_TEXT = {
    "column": "a column name",
    "columns": "a list of column names",
    "source": "the id of another of the user's data sources (list_data_sources)",
    "source_columns": "a list of column names of that source",
    "number": "a number",
    "int": "a whole number",
    "bool": "true or false",
    "string": "a text",
    "value": "a single value (text, number, true/false or null)",
    "list": "a list of values",
    "mapping": "an object {old: new}",
}


def _param_text(param: dict[str, Any]) -> str:
    """One line per parameter, e.g. ``one of mean | median …; default "median"``."""
    kind = param["kind"]
    options = [o["value"] for o in param.get("options") or []]
    if kind == "select":
        text = "one of " + " | ".join(map(str, options))
    elif kind == "multiselect":
        text = "a list of " + " | ".join(map(str, options))
    else:
        text = _KIND_TEXT.get(kind, kind)
    if param.get("column_kinds"):
        text += f" ({' or '.join(param['column_kinds'])} columns)"
    if kind == "mapping" and param.get("key_kind") == "column":
        text += ", keys are column names"
    low, high = param.get("min"), param.get("max")
    if low is not None and high is not None:
        text += f" from {low:g} to {high:g}"
    elif low is not None:
        text += f" ≥ {low:g}"
    elif high is not None:
        text += f" ≤ {high:g}"
    default = param.get("default")
    if default not in (None, "", []):
        text += f"; default {json.dumps(default, ensure_ascii=False)}"
    if param.get("empty_means"):
        text += f"; empty: {param['empty_means']}"
    elif param.get("optional"):
        text += "; optional"
    condition = param.get("visible_if")
    if condition:
        values = " | ".join(v if isinstance(v, str) else json.dumps(v) for v in condition["values"])
        text += f"; only used if {condition['param']} is {values}"
    return text


STEP_FORMAT = (
    'A step is {"type": "<type>", "params": {...}} (plus "id" and "enabled" once saved). '
    "Parameters left out get their defaults; column names must match the columns of the data "
    "at that point of the pipeline. Combine steps (join, append) come first; the "
    "train/validation/test split is assigned after them, and steps that learn values "
    "(means, categories …) learn them from the training rows only."
)


def step_catalog(step_type: Any = None) -> dict[str, Any]:
    """All step types with one line per parameter, or one type with its full description."""
    catalog = data.step_catalog()
    groups = {g["id"]: g["label"] for g in catalog["groups"]}
    if step_type is not None:
        spec = next((s for s in catalog["steps"] if s["type"] == step_type), None)
        if spec is None:
            known = ", ".join(s["type"] for s in catalog["steps"])
            raise DataToolError(f"Unknown step type '{step_type}'. The step types: {known}.")
        params = []
        for param in spec["params"]:
            entry = {
                "name": param["name"],
                "label": param["label"],
                "accepts": _param_text(param),
                "help": param.get("help"),
            }
            if param.get("options"):
                entry["options"] = {o["value"]: o["label"] for o in param["options"]}
            params.append(_drop_empty(entry))
        return {
            "type": spec["type"],
            "label": spec["label"],
            "group": groups.get(spec["group"], spec["group"]),
            "description": spec["description"],
            "learns_from_training_rows": spec["fitted"],
            "combine": spec["combine"],
            "training_only": spec["training_only"],
            "params": params,
            "example": {"type": spec["type"], "params": _example(spec)},
            "format": STEP_FORMAT,
        }
    return {
        "format": STEP_FORMAT,
        "details": "list_prep_steps with type gives the help texts of one step type.",
        "steps": [
            {
                "type": spec["type"],
                "group": groups.get(spec["group"], spec["group"]),
                "label": spec["label"],
                "description": spec["description"],
                "params": {p["name"]: _param_text(p) for p in spec["params"]},
            }
            for spec in catalog["steps"]
        ],
    }


def _example(spec: dict[str, Any]) -> dict[str, Any]:
    """The defaults of a step type, without the ones of hidden parameters."""
    example = {}
    for param in spec["params"]:
        condition = param.get("visible_if")
        if condition and not any(
            other["name"] == condition["param"] and other.get("default") in condition["values"]
            for other in spec["params"]
        ):
            continue
        example[param["name"]] = param.get("default")
    return example


# --------------------------------------------------------------------------- #
# Editing step lists (like frontend/src/data/steps.ts)
# --------------------------------------------------------------------------- #


def _step_class(step_type: Any):
    from brickdnn.data.steps import REGISTRY

    cls = REGISTRY.get(step_type) if isinstance(step_type, str) else None
    if cls is None:
        raise DataToolError(f"Unknown step type '{step_type}'. list_prep_steps shows the types.")
    return cls


def new_step_id(steps: list[Any], used: list[Any] = ()) -> str:
    """``s<n>`` (like the web app's newStepId): higher than the number of ``steps`` and
    than every generated id among ``steps`` and ``used``."""
    highest = len(steps)
    for step in [*steps, *used]:
        step_id = step.get("id") if isinstance(step, dict) else step
        match = STEP_ID.fullmatch(str(step_id or ""))
        if match:
            highest = max(highest, int(match.group(1)))
    return f"s{highest + 1}"


def new_step(step: Any, steps: list[Any], used: list[Any] = ()) -> dict[str, Any]:
    """A complete new step: a new id, enabled, the defaults of its type overridden by
    ``step["params"]`` (like the web app's createStep)."""
    if not isinstance(step, dict):
        raise DataToolError('A step must be an object: {"type": "impute", "params": {...}}.')
    cls = _step_class(step.get("type"))
    params = step.get("params")
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise DataToolError(f"The params of a '{cls.type}' step must be an object.")
    enabled = step.get("enabled", True)
    return {
        "id": new_step_id(steps, used),
        "type": cls.type,
        "enabled": True if enabled is None else enabled,
        "params": {**cls.defaults(), **params},
    }


def complete_steps(steps: Any) -> Any:
    """Steps of create_data_prep / update_data_prep: steps without an id are new (an id and
    the defaults are added, as when a step is added in the web app); the others are kept as
    they are. Anything but a list is left to the validation."""
    if not isinstance(steps, list):
        return steps
    given = [s.get("id") for s in steps if isinstance(s, dict) and s.get("id") not in (None, "")]
    result: list[Any] = []
    for step in steps:
        if isinstance(step, dict) and step.get("id") in (None, ""):
            result.append(new_step(step, result, given))
        else:
            result.append(step)
    return result


def _index(steps: list[dict[str, Any]], step_id: Any) -> int:
    for i, step in enumerate(steps):
        if step.get("id") == step_id:
            return i
    listing = ", ".join(f"{s.get('id')} ({s.get('type')})" for s in steps) or "none"
    raise DataToolError(f"There is no step '{step_id}'. The steps: {listing}.")


def _combine_end(steps: list[dict[str, Any]]) -> int:
    from brickdnn.data.steps import REGISTRY

    end = 0
    for i, step in enumerate(steps):
        cls = REGISTRY.get(step.get("type"))
        if cls is not None and cls.combine:
            end = i + 1
    return end


def _first_failed(steps: list[dict[str, Any]], reports: list[dict[str, Any]] | None) -> int:
    by_id = {r.get("id"): r for r in reports or []}
    for i, step in enumerate(steps):
        report = by_id.get(step.get("id"))
        if (
            step.get("enabled", True)
            and report
            and report.get("type") == step.get("type")
            and report.get("status") == "error"
        ):
            return i
    return -1


def insert_step(
    steps: list[dict[str, Any]],
    step: Any,
    after: Any = None,
    reports: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Add a step like the web app (insertStep): combine steps after the last combine step,
    all others at the end or right after ``after`` (a step id), but in front of the first
    step that failed in ``reports`` and never in front of a combine step."""
    created = new_step(step, steps)
    end = _combine_end(steps)
    if _step_class(created["type"]).combine:
        position = end
    else:
        position = len(steps) if after in (None, "") else _index(steps, after) + 1
        failed = _first_failed(steps, reports)
        if failed >= 0:
            position = min(position, failed)
        position = max(position, end)
    return [*steps[:position], created, *steps[position:]], created


def edit_steps(steps: list[dict[str, Any]], args: dict[str, Any]) -> list[dict[str, Any]]:
    """Apply remove_steps, edit_steps, move_steps and add_steps (in this order)."""
    steps = [dict(s) if isinstance(s, dict) else s for s in steps]
    removed = args.get("remove_steps")
    if removed is not None:
        if not isinstance(removed, list):
            raise DataToolError("'remove_steps' must be a list of step ids.")
        for step_id in removed:
            steps.pop(_index(steps, step_id))
    changes = args.get("edit_steps")
    if changes is not None:
        if not isinstance(changes, list):
            raise DataToolError("'edit_steps' must be a list of {id, params?, enabled?}.")
        for change in changes:
            if not isinstance(change, dict):
                raise DataToolError("Each entry of 'edit_steps' must be {id, params?, enabled?}.")
            unknown = set(change) - {"id", "params", "enabled"}
            if unknown:
                raise DataToolError(
                    f"edit_steps: unknown key '{sorted(unknown)[0]}'; use id, params, enabled."
                )
            i = _index(steps, change.get("id"))
            step = steps[i]
            if "params" in change:
                params = change["params"]
                if not isinstance(params, dict):
                    raise DataToolError(
                        f"edit_steps: the params of '{step['id']}' must be an object."
                    )
                merged = dict(step.get("params") or {})
                for key, value in params.items():
                    if value is None:
                        merged.pop(key, None)  # back to the default
                    else:
                        merged[key] = value
                step["params"] = merged
            if "enabled" in change:
                step["enabled"] = change["enabled"]
    moves = args.get("move_steps")
    if moves is not None:
        if not isinstance(moves, list):
            raise DataToolError("'move_steps' must be a list of {id, after}.")
        for move in moves:
            if not isinstance(move, dict) or "id" not in move:
                raise DataToolError(
                    "Each entry of 'move_steps' must be {id, after}: after a step id, or null "
                    "for the top."
                )
            step = steps.pop(_index(steps, move["id"]))
            after = move.get("after")
            position = 0 if after in (None, "") else _index(steps, after) + 1
            steps.insert(position, step)
    added = args.get("add_steps")
    if added is not None:
        if not isinstance(added, list):
            raise DataToolError("'add_steps' must be a list of {type, params?, after?}.")
        for step in added:
            if isinstance(step, dict):
                unknown = set(step) - {"type", "params", "enabled", "after"}
                if unknown:
                    raise DataToolError(
                        f"add_steps: unknown key '{sorted(unknown)[0]}'; use type, params, "
                        "enabled, after."
                    )
            steps, _ = insert_step(
                steps, step, step.get("after") if isinstance(step, dict) else None
            )
    return steps


# --------------------------------------------------------------------------- #
# Previews and readiness
# --------------------------------------------------------------------------- #


def _fix_key(fix: dict[str, Any]) -> str:
    """The step a fix adds, as text (like fixKey in the web app's readiness panel)."""
    return json.dumps([fix.get("type"), fix.get("params")], sort_keys=True, ensure_ascii=False)


def fix_id(issue: dict[str, Any]) -> str | None:
    """A short id for the fix of a readiness issue: its code and a hash of the step it adds,
    so the same fix keeps its id while the data and the steps stay the same."""
    fix = issue.get("fix")
    if not fix:
        return None
    digest = hashlib.sha1(_fix_key(fix).encode(), usedforsecurity=False).hexdigest()[:6]
    return f"{issue.get('code')}-{digest}"


def fixes_of(readiness: dict[str, Any]) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    """``fix_id -> (fix, issue)`` for every readiness issue with a fix."""
    found: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for issue in readiness.get("issues") or []:
        key = fix_id(issue)
        if key is not None and key not in found:
            found[key] = (issue["fix"], issue)
    return found


def error_fixes(readiness: dict[str, Any]) -> list[tuple[str, tuple[dict, dict]]]:
    """The fixes of all errors, each step once (what "Fix all" in the web app adds)."""
    chosen: dict[str, tuple[str, tuple[dict, dict]]] = {}
    for key, (fix, issue) in fixes_of(readiness).items():
        if issue.get("severity") == "error":
            chosen.setdefault(_fix_key(fix), (key, (fix, issue)))
    return list(chosen.values())


def _issue_view(issue: dict[str, Any]) -> dict[str, Any]:
    view: dict[str, Any] = {"code": issue.get("code"), "message": issue.get("message")}
    columns = issue.get("columns") or []
    if columns:
        view["columns"] = _capped(columns, MAX_ISSUE_COLUMNS)
    fix = issue.get("fix")
    if fix:
        view["fix_id"] = fix_id(issue)
        shown = {"label": fix.get("label"), "type": fix.get("type")}
        params = fix.get("params") or {}
        if len(json.dumps(params, ensure_ascii=False)) <= MAX_FIX_PARAMS:
            shown["params"] = params
        else:
            shown["params"] = "(long; apply it with apply_readiness_fix)"
        view["fix"] = shown
    return view


def readiness_view(ready: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    issues = ready.get("issues") or []
    by_severity = {
        severity: [_issue_view(i) for i in issues if i.get("severity") == severity]
        for severity in ("error", "warning", "info")
    }
    split = ready.get("split_counts")
    view: dict[str, Any] = {
        "ready": bool(ready.get("ready")),
        "target": ready.get("target"),
        "task": ready.get("task"),
        "rows": ready.get("rows"),
        "features": ready.get("num_features"),
        "classes": ready.get("num_classes"),
        "class_names": _capped(ready.get("class_names") or [], MAX_CLASSES),
        "split": (
            {"train": split[0], "validation": split[1], "test": split[2]}
            if isinstance(split, list) and len(split) == 3
            else None
        ),
        "errors": by_severity["error"],
        "warnings": by_severity["warning"],
        "suggestions": by_severity["info"],
    }
    if any(i.get("code") in ("no_target", "target_not_found") for i in issues):
        view["choose_target_from"] = _capped(columns)
    if any(issue.get("fix_id") for issue in by_severity["error"]):
        view["fix_all"] = (
            "apply_readiness_fix with fix_id 'all' adds the fixes of all errors at once."
        )
    return {k: v for k, v in view.items() if v not in (None, [])}


def _step_view(report: dict[str, Any]) -> dict[str, Any]:
    from brickdnn.data.steps import REGISTRY

    cls = REGISTRY.get(report.get("type"))
    view: dict[str, Any] = {
        "id": report.get("id"),
        "type": report.get("type"),
        "status": report.get("status"),
    }
    if report.get("status") == "ok":
        view["summary"] = report.get("summary")
        if report.get("rows_before") != report.get("rows_after"):
            view["rows"] = f"{report.get('rows_before'):,} → {report.get('rows_after'):,}"
        if report.get("added"):
            view["added"] = _capped(report["added"], 30)
        if report.get("removed"):
            view["removed"] = _capped(report["removed"], 30)
    if report.get("warnings"):
        view["warnings"] = report["warnings"]
    if report.get("status") == "error":
        label = cls.label if cls is not None else report.get("type")
        view["error"] = f"{label}: {report.get('error')}"
        # The columns the step saw: what its column parameters may name.
        view["columns_before_step"] = _capped([c.get("name") for c in report.get("columns") or []])
        view["help"] = f"list_prep_steps with type '{report.get('type')}' explains its params."
    return view


def _column_profile(column: dict[str, Any]) -> dict[str, Any]:
    kind = column.get("kind")
    view: dict[str, Any] = {
        "name": column.get("name"),
        "kind": kind,
        "missing": column.get("missing"),
        "unique": column.get("unique"),
    }
    stats = column.get("stats")
    if stats and kind == "datetime":
        view.update(min=_iso(stats.get("min")), max=_iso(stats.get("max")))
    elif stats and kind != "boolean":
        for key in ("min", "max", "mean", "median", "std"):
            view[key] = _number(stats.get(key))
        if stats.get("iqr_outliers"):
            view["iqr_outliers"] = stats["iqr_outliers"]
        if stats.get("skew") is not None and abs(stats["skew"]) > 1:
            view["skew"] = _number(stats["skew"])
    if column.get("top"):  # the most frequent values: {value: count}
        view["top"] = {str(t["value"]): t["count"] for t in column["top"][:5]}
    association = column.get("target_association")
    if isinstance(association, dict) and association.get("value") is not None:
        # |r| for numbers, the correlation ratio or Cramer's V for categories
        view["target_association"] = _number(association["value"])
    return {k: v for k, v in view.items() if v is not None}


MAX_WEAK_COLUMNS = 15  # columns without a strong partner listed by name
RELATION_BINS = 10  # bins of the strongest-partner histogram


def relations_view(relations: dict[str, Any] | None) -> dict[str, Any] | None:
    """A short summary of how the number columns relate (``ColumnRelations`` of a profile):
    counts, a coarse histogram of every column's strongest |ρ| and the weakest columns."""
    if not relations or len(relations.get("columns") or []) < 3:
        return None
    names = relations["columns"]
    strength = [abs(v) if v is not None else 0.0 for v in relations["best_rho"]]
    level = relations.get("suggested_level") or relations.get("weak_level") or 0.15
    strong = relations.get("partner_level") or 0.3
    bins = [0] * RELATION_BINS
    for value in strength:
        bins[min(int(value * RELATION_BINS), RELATION_BINS - 1)] += 1
    width = 1 / RELATION_BINS
    weakest = sorted(zip(strength, names, strict=True))[:MAX_WEAK_COLUMNS]
    weak = sum(value < level for value in strength)
    view: dict[str, Any] = {
        "about": (
            "Strongest partner of every number feature: its largest |Spearman ρ| with any "
            f"other feature, on {relations['rows']:,} "
            + ("training rows" if relations.get("training_rows") else "rows")
            + f". Unrelated columns reach about {relations.get('chance')} by chance; columns "
            "without a strong partner in a table of related columns are often noise."
        ),
        "columns": len(names),
        "without_partner": {"threshold": level, "columns": weak},
        "with_partner": {"threshold": strong, "columns": sum(v >= strong for v in strength)},
        "histogram": {
            f"{i * width:.1f}–{(i + 1) * width:.1f}": count for i, count in enumerate(bins) if count
        },
        "weakest": {name: _number(round(value, 3)) for value, name in weakest},
    }
    if relations.get("groups"):
        view["groups"] = len({g for g in relations["groups"] if g >= 0})
    if relations.get("sampled_columns"):
        view["note"] = f"A sample of the {relations['total']:,} number columns."
    if weak:
        view["drop_them"] = (
            'add_steps [{"type": "feature_selection", "params": {"method": "no_partner", '
            f'"min_partner": {level}}}}}] drops the {weak:,} columns without a strong partner '
            "(fitted on the training rows)."
        )
    return view


def _target_view(target: dict[str, Any] | None) -> dict[str, Any] | None:
    if not target:
        return None
    view: dict[str, Any] = {"column": target.get("column"), "task": target.get("task")}
    if target.get("classes"):
        view["classes"] = _capped(
            [{"value": c["value"], "count": c["count"]} for c in target["classes"]], MAX_CLASSES
        )
        if target.get("imbalance_ratio"):
            view["imbalance_ratio"] = _number(target["imbalance_ratio"])
    if target.get("stats"):
        stats = target["stats"]
        view.update({k: _number(stats.get(k)) for k in ("min", "max", "mean", "median", "std")})
    return view


def _sample_view(
    sample: dict[str, Any], rows: int, columns: list[str], total: int
) -> dict[str, Any]:
    by_name = {c["name"]: c for c in sample.get("columns") or []}
    names = [name for name in columns if name in by_name][:MAX_SAMPLE_COLUMNS]
    count = min(rows, int(sample.get("rows") or 0))
    view: dict[str, Any] = {
        "rows": [
            {name: _cell(by_name[name]["values"][i], by_name[name]["kind"]) for name in names}
            for i in range(count)
        ],
    }
    if sample.get("split"):
        view["split"] = sample["split"][:count]
    if len(names) < total:
        view["note"] = (
            f"{len(names)} of {total:,} columns; 'columns' or 'columns_offset' choose others."
        )
    return view


def preview_view(
    result: dict[str, Any],
    *,
    detail: bool,
    sample_rows: int = DEFAULT_SAMPLE_ROWS,
    columns: list[str] | None = None,
    offset: int = 0,
) -> dict[str, Any]:
    """A compact PreviewResult: step reports, the readiness with fix ids and a page of the
    columns after the pipeline (name: kind); with ``detail`` also profiles and sample rows
    of the first columns of that page (or of ``columns``) and the target."""
    profile = result.get("profile") or {}
    profiles = profile.get("columns") or []
    names = [c.get("name") for c in profiles]
    page = profiles[offset : offset + COLUMN_PAGE]
    view: dict[str, Any] = {
        "rows": profile.get("n_rows"),
        "task": result.get("task"),
        "steps": [_step_view(r) for r in result.get("steps") or []],
        "readiness": readiness_view(result.get("readiness") or {}, names),
        "columns": {c.get("name"): c.get("kind") for c in page},
    }
    if note := page_note(offset, len(page), len(profiles)):
        view["columns_note"] = note
    if result.get("warnings"):
        view["warnings"] = result["warnings"]
    if not detail:
        return view
    if columns is None:
        chosen = [c.get("name") for c in page]
    else:
        chosen = [n for n in names if n in set(columns)]
    shown = [c for c in profiles if c.get("name") in set(chosen[:MAX_PROFILES])]
    view["profiles"] = [_column_profile(c) for c in shown]
    if len(chosen) > len(shown) or len(shown) < len(names):
        view["profiles_note"] = (
            f"Profiles of {len(shown)} of {len(names):,} columns; 'columns' (names) or "
            "'columns_offset' choose others."
        )
    if profile.get("duplicate_rows"):
        view["duplicate_rows"] = profile["duplicate_rows"]
    target = _target_view(profile.get("target"))
    if target:
        view["target"] = target
    relations = relations_view(profile.get("relations"))
    if relations:
        view["column_relations"] = relations
    if sample_rows > 0:
        view["sample"] = _sample_view(result.get("sample") or {}, sample_rows, chosen, len(names))
    return view


def check_columns(columns: Any) -> list[str] | None:
    if columns is None:
        return None
    if not isinstance(columns, list) or not all(isinstance(c, str) for c in columns):
        raise DataToolError("'columns' must be a list of column names.")
    return columns


def unknown_columns(columns: list[str] | None, result: dict[str, Any]) -> str | None:
    """An error message if ``columns`` names columns the prepared data does not have."""
    if not columns:
        return None
    names = [c.get("name") for c in (result.get("profile") or {}).get("columns") or []]
    missing = [c for c in columns if c not in set(names)]
    if not missing:
        return None
    return (
        f"There is no column '{missing[0]}' in the prepared data. Its columns: "
        f"{names_text(names, MAX_NAMES)}."
    )


def sample_rows_arg(value: Any, default: int = DEFAULT_SAMPLE_ROWS) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DataToolError(f"'sample_rows' must be a whole number from 0 to {MAX_SAMPLE_ROWS}.")
    return min(value, MAX_SAMPLE_ROWS)


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #


def export_answer(dataset, user) -> dict[str, Any]:
    """An exported training dataset with what training and predictions need to know."""
    from .serializers import DatasetSerializer

    serialized = dict(DatasetSerializer(dataset, context={"user": user}).data)
    info = dict(serialized.get("info") or {})
    columns = [str(c) for c in info.pop("columns", None) or []]
    serialized["info"] = info
    features, target = columns[:-1], (columns[-1] if columns else None)
    classes = info.get("class_names")
    split = info.get("predefined_split") or [None, None, None]
    outputs = len(classes) if classes else 1
    task = "classification" if classes else "regression"
    return {
        "dataset": serialized,
        "reference": {"kind": "upload", "id": dataset.pk},
        "task": task,
        "target": target,
        "features": _capped(features, 200),
        "class_names": _capped(classes, MAX_CLASSES) if classes else None,
        "split": {"train": split[0], "validation": split[1], "test": split[2]},
        "next": (
            "Pass reference as dataset to check_training and start_training. A network for "
            f"it starts with an Input of shape [{len(features)}] and ends with {outputs} "
            f"output{'s' if outputs > 1 else ''} ({task}); check_training suggests fixes. "
            "Training uses the split of the preparation. predict takes the prepared feature "
            "columns listed in features (after the steps, e.g. rows of the sample of "
            "preview_data_prep), not the raw columns of the file."
        ),
    }


# --------------------------------------------------------------------------- #
# Downloads
# --------------------------------------------------------------------------- #


def csv_text(handle: Any, max_rows: int) -> dict[str, Any]:
    """The header and the first ``max_rows`` records of the prepared CSV file (fewer if the
    text gets too long); ``truncated`` if there are more."""
    reader = csv.reader(io.TextIOWrapper(handle, encoding="utf-8", newline=""))
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    shown, truncated = 0, False
    for number, record in enumerate(reader):
        if number and (shown >= max_rows or out.tell() >= MAX_DOWNLOAD_BYTES):
            truncated = True
            break
        writer.writerow(record)
        shown += 1 if number else 0
    return {"rows_shown": shown, "truncated": truncated, "csv": out.getvalue()}


def too_large(content: bytes) -> bool:
    return len(content) > MAX_DOWNLOAD_BYTES
