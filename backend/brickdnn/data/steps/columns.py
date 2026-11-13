"""Preparation steps that work on whole columns: drop, keep, rename, change the type,
compute a new column with a formula, replace values and extract parts of text.

The first section holds small helpers that the other basic step modules (combine, rows,
missing) use as well: number formatting for summaries, converting user-entered values to
the storage format of a column (see :mod:`brickdnn.data.table`) and writing them as code.
"""

from __future__ import annotations

import ast
import inspect
import math
import re
import textwrap
import warnings
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..table import KINDS, NUMERIC_KINDS, STRING_KINDS, DataError, Table, normalize
from .base import Param, Step, StepContext, StepResult, register
from .text import fmt_count, fmt_number, fmt_share, plural  # noqa: F401

try:
    import re._parser as _sre  # Python 3.11+
except ImportError:  # pragma: no cover - Python 3.10
    import sre_parse as _sre  # type: ignore[no-redef]

# --------------------------------------------------------------------------- #
# Helpers shared by the basic step modules
# --------------------------------------------------------------------------- #

TRUE_WORDS = ("true", "yes", "y", "t", "1", "on")
FALSE_WORDS = ("false", "no", "n", "f", "0", "off")
BOOL_WORDS = {**{w: 1.0 for w in TRUE_WORDS}, **{w: 0.0 for w in FALSE_WORDS}}
KIND_NAMES = {
    "numeric": "numbers",
    "boolean": "true/false",
    "datetime": "dates",
    "categorical": "categories",
    "text": "text",
}


def quoted(names: Sequence[Any], limit: int = 3) -> str:
    """``'a'``, ``'a' and 'b'``, ``'a', 'b', 'c' and 4 more``."""
    items = [f"'{n}'" for n in names]
    if len(items) <= limit:
        return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]
    return ", ".join(items[:limit]) + f" and {len(items) - limit:,} more"


def chosen(
    table: Table, columns: Sequence[str], ctx: StepContext, kinds: Iterable[str] | None = None
) -> list[str]:
    """The listed columns, or (empty list) all columns of ``kinds`` (default: all kinds)
    except the target."""
    if columns:
        return list(columns)
    exclude = [ctx.target] if ctx.target else []
    return table.columns_of(kinds or KINDS, exclude=exclude)


def as_number(value: Any) -> float | None:
    """A finite float from a number or a numeric string (``"1,5"`` is 1.5), else None."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            number = float(value)
        except OverflowError:
            return None
        return number if math.isfinite(number) else None
    if isinstance(value, str):
        text = value.strip()
        if text.count(",") == 1 and "." not in text:
            text = text.replace(",", ".")
        try:
            number = float(text)
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    return None


def as_label(value: Any) -> str:
    """A cell value as the string stored in categorical / text columns."""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        if number.is_integer() and abs(number) < 1e15:
            return str(int(number))
        return repr(number)
    return str(value)


def as_timestamp(value: Any) -> pd.Timestamp | None:
    """A date from a string (or seconds since 1970), else None."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        if isinstance(value, str):
            stamp = pd.Timestamp(value.strip())
        elif isinstance(value, (int, float)):
            stamp = pd.Timestamp(float(value), unit="s")
        else:
            return None
    except (ValueError, TypeError, OverflowError, pd.errors.OutOfBoundsDatetime):
        return None
    if pd.isna(stamp):
        return None
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return stamp


def cell_value(value: Any, kind: str, column: str, label: str) -> Any:
    """Convert a value entered by the user to the storage format of a ``kind`` column
    (float, Timestamp or str; None stays None). Raises :class:`DataError`."""
    if value is None:
        return None
    if kind in NUMERIC_KINDS:
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, str) and value.strip().lower() in BOOL_WORDS and kind == "boolean":
            return BOOL_WORDS[value.strip().lower()]
        number = as_number(value)
        if number is None:
            what = "numbers" if kind == "numeric" else "true/false values"
            raise DataError(f"{label}: '{value}' is not a number, but '{column}' holds {what}.")
        if kind == "boolean" and number not in (0.0, 1.0):
            raise DataError(f"{label}: '{column}' is a true/false column; use 0 or 1.")
        return number
    if kind == "datetime":
        stamp = as_timestamp(value)
        if stamp is None:
            raise DataError(
                f"{label}: '{value}' is not a date, but '{column}' holds dates "
                "(use e.g. 2024-01-31)."
            )
        return stamp
    return as_label(value)


def code_value(value: Any) -> str:
    """A Python literal for a value in generated code."""
    if value is None:
        return "None"
    if isinstance(value, pd.Timestamp):
        return f"pd.Timestamp({value.isoformat()!r})"
    if isinstance(value, (bool, np.bool_)):
        return repr(bool(value))
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        return repr(number) if math.isfinite(number) else "np.nan"
    return repr(value)


def from_fitted(value: Any, kind: str) -> Any:
    """A value stored in ``fitted`` (JSON: dates as ISO strings) in storage format."""
    if value is None:
        return None
    if kind == "datetime":
        return pd.Timestamp(value)
    if kind in NUMERIC_KINDS:
        return float(value)
    return str(value)


def to_fitted(value: Any) -> Any:
    """A storage-format value as a JSON-safe ``fitted`` value."""
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return str(value)


def string_storage(series: pd.Series) -> pd.Series:
    """A string column in storage format (``str`` values, ``None`` for missing), fast when
    the values are strings already."""
    kind = pd.api.types.infer_dtype(series, skipna=True)
    if kind in ("string", "empty"):
        result = series.astype(object).where(series.notna(), None)
        result.name = series.name
        return result
    return normalize(series, "categorical")


def storage(series: pd.Series, kind: str) -> pd.Series:
    """``series`` in the storage format of ``kind`` (see :func:`table.normalize`)."""
    if kind in STRING_KINDS:
        return string_storage(series)
    return normalize(series, kind)


def set_columns(
    table: Table,
    columns: dict[str, tuple[pd.Series, str]],
    *,
    after: dict[str, str] | None = None,
) -> Table:
    """Replace or add columns (values already in storage format). New columns go to the
    end, or right after ``after[name]``."""
    df = table.df.copy(deep=False)
    kinds = dict(table.kinds)
    for name, (values, kind) in columns.items():
        values = values.reset_index(drop=True) if isinstance(values, pd.Series) else values
        anchor = (after or {}).get(name)
        if name in df.columns or anchor not in df.columns:
            df[name] = values
        else:
            df.insert(df.columns.get_loc(anchor) + 1, name, values)
        kinds[name] = kind
    return Table(df, kinds, table.split.copy() if table.split is not None else None)


def values_as_text(series: pd.Series, kind: str) -> pd.Series:
    """Any column as strings (``3.0`` -> ``"3"``, dates ISO); see :func:`text_code`."""
    if kind in STRING_KINDS:
        return series
    if kind == "datetime":
        text = series.dt.strftime("%Y-%m-%d %H:%M:%S").str.replace(" 00:00:00", "", regex=False)
        return text.astype(object).where(series.notna(), None)
    values = series.to_numpy(dtype=float, na_value=np.nan)
    out = np.full(len(values), None, dtype=object)
    finite = np.isfinite(values)
    whole = finite & (np.floor(values) == values) & (np.abs(values) < 1e15)
    out[whole] = values[whole].astype(np.int64).astype(str).tolist()
    rest = finite & ~whole
    out[rest] = [repr(v) for v in values[rest].tolist()]
    return pd.Series(out, index=series.index, name=series.name)


def text_code(ref: str, kind: str) -> str:
    """Code that does what :func:`values_as_text` does to the column ``ref``."""
    if kind in STRING_KINDS:
        return ref
    if kind == "datetime":
        return (
            f'{ref}.dt.strftime("%Y-%m-%d %H:%M:%S").str.replace(" 00:00:00", "", regex=False)'
            f".astype(object).where({ref}.notna(), None)"
        )
    return (
        f"{ref}.map(lambda v: None if pd.isna(v) else str(int(v)) "
        "if float(v).is_integer() and abs(v) < 1e15 else repr(float(v)))"
    )


def comment(text: Any) -> str:
    """Text for a ``#`` comment in generated code: one line, no control characters."""
    return " ".join(str(text).split())


def examples(values: pd.Series, limit: int = 3) -> str:
    """``'n.a.', 'free'`` – a few distinct values for a message."""
    shown = []
    for value in values.dropna().astype(str).unique()[:limit]:
        shown.append(f"'{value[:30]}'")
    return ", ".join(shown)


# --------------------------------------------------------------------------- #
# Drop, keep, rename
# --------------------------------------------------------------------------- #


def _required_columns(params: dict[str, Any], what: str = "Columns") -> None:
    if not params["columns"]:
        raise DataError(f"{what}: choose at least one column.")


def _target_warning(ctx: StepContext, columns: Iterable[str], verb: str) -> list[str]:
    if ctx.target and ctx.target in columns:
        return [
            f"This step {verb} the target column '{ctx.target}'. Choose another target in "
            "“Target & split” or change the step."
        ]
    return []


@register
class DropColumns(Step):
    type = "drop_columns"
    label = "Drop columns"
    group = "columns"
    description = "Remove columns that should not be used, e.g. IDs or free text."
    params = (Param("columns", "columns", "Columns", default=[]),)

    def clean(self, table: Table, ctx: StepContext) -> dict[str, Any]:
        params = super().clean(table, ctx)
        _required_columns(params)
        return params

    def run(self, table: Table, params: dict[str, Any], ctx: StepContext) -> StepResult:
        columns = params["columns"]
        return StepResult(
            table.drop(columns),
            f"Dropped {plural(len(columns), 'column')}: {quoted(columns, 5)}.",
            warnings=_target_warning(ctx, columns, "drops"),
        )

    def to_pandas(
        self, params: dict[str, Any], fitted: dict[str, Any] | None, ctx: StepContext
    ) -> list[str]:
        return [f"df = df.drop(columns={list(params['columns'])!r}, errors='ignore')"]


@register
class KeepColumns(Step):
    type = "keep_columns"
    label = "Keep only some columns"
    group = "columns"
    description = "Keep the chosen columns (and the target) and drop all others."
    params = (
        Param(
            "columns",
            "columns",
            "Columns to keep",
            default=[],
            help="The target column is always kept.",
        ),
    )

    def clean(self, table: Table, ctx: StepContext) -> dict[str, Any]:
        params = super().clean(table, ctx)
        _required_columns(params, "Columns to keep")
        return params

    def run(self, table: Table, params: dict[str, Any], ctx: StepContext) -> StepResult:
        keep = set(params["columns"])
        target_kept = bool(ctx.target and ctx.target in table.kinds and ctx.target not in keep)
        if target_kept:
            keep.add(ctx.target)
        dropped = [c for c in table.columns if c not in keep]
        result = table.drop(dropped)
        kept = len(params["columns"])
        summary = f"Kept {plural(kept, 'column')}"
        if target_kept:
            summary += f" and the target '{ctx.target}'"
        summary += f"; dropped {fmt_count(len(dropped))}."
        return StepResult(result, summary, fitted=None)

    def to_pandas(
        self, params: dict[str, Any], fitted: dict[str, Any] | None, ctx: StepContext
    ) -> list[str]:
        keep = list(params["columns"])
        if ctx.target and ctx.target not in keep:
            keep.append(ctx.target)
        return [f"df = df[[c for c in df.columns if c in {keep!r}]]"]


@register
class RenameColumns(Step):
    type = "rename_columns"
    label = "Rename columns"
    group = "columns"
    description = "Give columns new names."
    params = (
        Param(
            "mapping",
            "mapping",
            "New names",
            default={},
            help="Current name → new name.",
            key_kind="column",
        ),
    )

    def clean(self, table: Table, ctx: StepContext) -> dict[str, Any]:
        params = super().clean(table, ctx)
        mapping: dict[str, str] = {}
        for old, new in params["mapping"].items():
            if old not in table.kinds:
                raise DataError(f"New names: there is no column '{old}'.")
            if not isinstance(new, str) or not new.strip():
                raise DataError(f"New names: enter a new name for '{old}'.")
            if new.strip() != old:
                mapping[old] = new.strip()
        if not mapping and not params["mapping"]:
            raise DataError("New names: add at least one column to rename.")
        final = [mapping.get(c, c) for c in table.columns]
        seen: set[str] = set()
        for name in final:
            if name in seen:
                raise DataError(f"New names: two columns would be called '{name}'.")
            seen.add(name)
        params["mapping"] = mapping
        return params

    def run(self, table: Table, params: dict[str, Any], ctx: StepContext) -> StepResult:
        mapping = params["mapping"]
        warnings = []
        if ctx.target in mapping:
            warnings.append(
                f"The target '{ctx.target}' is now called '{mapping[ctx.target]}'. Change the "
                "target in “Target & split” to the new name."
            )
        details = [{"column": new, "text": f"was {old}"} for old, new in mapping.items()]
        if not mapping:
            return StepResult(table, "Nothing to rename: the new names equal the old ones.")
        return StepResult(
            table.rename(mapping),
            f"Renamed {plural(len(mapping), 'column')}.",
            details=details,
            warnings=warnings,
        )

    def to_pandas(
        self, params: dict[str, Any], fitted: dict[str, Any] | None, ctx: StepContext
    ) -> list[str]:
        return [f"df = df.rename(columns={dict(params['mapping'])!r})"]


# --------------------------------------------------------------------------- #
# Change the type
# --------------------------------------------------------------------------- #

# Tried in order when no date format is given; the one that reads the most values wins.
DATE_FORMATS = (
    "ISO8601",
    "%d.%m.%Y",
    "%d.%m.%Y %H:%M",
    "%d.%m.%Y %H:%M:%S",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%Y/%m/%d",
    "%d-%m-%Y",
    "%Y%m%d",
)
# datetime64[ns] covers the years 1677 … 2262: about ±9.2e9 seconds around 1970.
MAX_EPOCH_SECONDS = 9.2e9


def parse_dates(series: pd.Series, fmt: str) -> pd.Series:
    """Strings to dates with the format ``fmt`` (unreadable values become missing)."""
    result = pd.to_datetime(series, format=fmt, errors="coerce", utc=True)
    return result.dt.tz_localize(None).astype("datetime64[ns]")


def _date_format(series: pd.Series) -> str:
    present = series.dropna()
    if len(present) == 0:
        return "ISO8601"
    if len(present) > 20_000:
        present = present.sample(20_000, random_state=0)
    best, best_count = DATE_FORMATS[0], -1
    for fmt in DATE_FORMATS:
        count = int(parse_dates(present, fmt).notna().sum())
        if count > best_count:
            best, best_count = fmt, count
        if count == len(present):
            break
    return best


# pd.to_numeric may be off by one unit in the last digit, Python's float() is exact: use
# to_numeric only to find the numbers, then convert those exactly.
EXACT_NUMBERS_CODE = (
    "{text}.where(pd.to_numeric({text}, errors='coerce').notna()).astype('float64')"
    ".replace([np.inf, -np.inf], np.nan)"
)


def exact_numbers(text: pd.Series) -> pd.Series:
    """Strings to float64 (NaN where not a number); see ``EXACT_NUMBERS_CODE``."""
    numbers = pd.to_numeric(text, errors="coerce")
    try:
        exact = text.where(numbers.notna()).astype("float64")
    except (TypeError, ValueError):  # a string to_numeric accepts but float() does not
        exact = numbers.astype("float64")
    return exact.replace([np.inf, -np.inf], np.nan)


# The number in a value with text around it ("$1,234.50", "12 kg", "45 %", "ca. 3"): text
# without digits before and after it, digits with "." "," "'" separators, an exponent.
NUMBER_PATTERN = r"[+-]?(?:\d[\d.,']*\d|\d|[.,]\d+)(?:[eE][+-]?\d+)?"
UNIT_NUMBER = rf"^[^\d+\-]*?({NUMBER_PATTERN})\D*$"


def number_text(text: pd.Series, decimal_comma: bool, strip_units: bool) -> pd.Series:
    """Stripped strings as ``exact_numbers`` reads them: with ``strip_units`` only the number
    in the value (see ``UNIT_NUMBER``) without thousands separators; with ``decimal_comma``
    "1.234,5" becomes "1234.5". Keep in sync with ``_number_text_code``."""
    if strip_units:
        text = text.str.extract(UNIT_NUMBER, expand=False).str.replace("'", "", regex=False)
        if not decimal_comma:
            text = text.str.replace(",", "", regex=False)
    if decimal_comma:
        text = text.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return text


def _number_text_code(text: str, decimal_comma: bool, strip_units: bool) -> str:
    if strip_units:
        text += f".str.extract({UNIT_NUMBER!r}, expand=False)"
        text += ".str.replace(\"'\", '', regex=False)"
        if not decimal_comma:
            text += ".str.replace(',', '', regex=False)"
    if decimal_comma:
        text += '.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)'
    return text


def _cast(
    series: pd.Series, source: str, to: str, params: dict[str, Any], column: str
) -> tuple[pd.Series, str | None]:
    """The column converted from kind ``source`` to ``to``; also the date format used."""
    if source == to:
        return series, None
    if to == "numeric":
        if source == "boolean":
            return series, None
        if source == "datetime":
            seconds = (series - pd.Timestamp("1970-01-01")).dt.total_seconds()
            return seconds.astype(float), None
        text = number_text(
            series.str.strip(), params["decimal_comma"], bool(params.get("strip_units"))
        )
        return exact_numbers(text), None
    if to == "boolean":
        if source == "datetime":
            raise DataError(f"'{column}' holds dates, which cannot be turned into true/false.")
        if source == "numeric":
            return series.where(series.isin([0.0, 1.0])), None
        return series.str.strip().str.lower().map(BOOL_WORDS).astype(float), None
    if to == "datetime":
        if source == "boolean":
            raise DataError(f"'{column}' holds true/false values, which cannot be dates.")
        if source == "numeric":
            limited = series.where(series.abs() < MAX_EPOCH_SECONDS)
            return pd.to_datetime(limited, unit="s").astype("datetime64[ns]"), None
        fmt = params["datetime_format"] or _date_format(series)
        return parse_dates(series, fmt), fmt
    return values_as_text(series, source), None


def _cast_code(column: str, source: str, to: str, params: dict[str, Any], fmt: str | None) -> str:
    ref = f"df[{column!r}]"
    if source == to or (to == "numeric" and source == "boolean"):
        return f"# {comment(column)}: the values are numbers already"
    if source in STRING_KINDS and to in STRING_KINDS:
        return f"# {comment(column)}: {source} -> {to}, the values stay the same"
    if to == "numeric":
        if source == "datetime":
            return f"{ref} = ({ref} - pd.Timestamp('1970-01-01')).dt.total_seconds().astype(float)"
        text = _number_text_code(
            f"{ref}.str.strip()", params["decimal_comma"], bool(params.get("strip_units"))
        )
        return f"_text = {text}\n{ref} = {EXACT_NUMBERS_CODE.format(text='_text')}"
    if to == "boolean":
        if source == "numeric":
            return f"{ref} = {ref}.where({ref}.isin([0.0, 1.0]))"
        return f"{ref} = {ref}.str.strip().str.lower().map({BOOL_WORDS!r}).astype(float)"
    if to == "datetime":
        if source == "numeric":
            return (
                f"{ref} = pd.to_datetime({ref}.where({ref}.abs() < {MAX_EPOCH_SECONDS!r}), "
                "unit='s').astype('datetime64[ns]')"
            )
        return (
            f"{ref} = pd.to_datetime({ref}, format={fmt!r}, errors='coerce', utc=True)"
            ".dt.tz_localize(None).astype('datetime64[ns]')"
        )
    return f"{ref} = {text_code(ref, source)}"


@register
class Cast(Step):
    type = "cast"
    label = "Change type"
    group = "columns"
    description = "Change the type of columns, e.g. read text as numbers or dates."
    params = (
        Param("columns", "columns", "Columns", default=[]),
        Param(
            "to",
            "select",
            "New type",
            default="numeric",
            options=(
                ("numeric", "Numbers"),
                ("categorical", "Categories"),
                ("boolean", "True/false"),
                ("datetime", "Dates"),
                ("text", "Free text"),
            ),
            help="Values that cannot be converted become missing; the step says how many. "
            "Dates become seconds since 1970 as numbers, and vice versa.",
        ),
        Param(
            "decimal_comma",
            "bool",
            "Decimal comma (1.234,5)",
            default=False,
            visible_if={"param": "to", "values": ["numeric"]},
        ),
        Param(
            "strip_units",
            "bool",
            "Ignore units and thousands separators",
            default=False,
            visible_if={"param": "to", "values": ["numeric"]},
            help="Reads the number in values like '$1,200', '12 kg' or '45 %' (1200, 12 and "
            "45): text before and after the number and thousands separators are ignored.",
        ),
        Param(
            "datetime_format",
            "string",
            "Date format",
            default="",
            optional=True,
            help="e.g. %d.%m.%Y for 31.01.2024. Empty: detect the format.",
            visible_if={"param": "to", "values": ["datetime"]},
        ),
    )

    def clean(self, table: Table, ctx: StepContext) -> dict[str, Any]:
        params = super().clean(table, ctx)
        _required_columns(params)
        params["datetime_format"] = params["datetime_format"] or ""
        fmt = params["datetime_format"]
        if fmt and fmt != "ISO8601" and "%" not in fmt:
            raise DataError("Date format: use codes like %Y-%m-%d or %d.%m.%Y %H:%M.")
        return params

    def run(self, table: Table, params: dict[str, Any], ctx: StepContext) -> StepResult:
        to = params["to"]
        updates: dict[str, tuple[pd.Series, str]] = {}
        details = []
        formats: dict[str, str] = {}
        lost_total = 0
        failed_examples: list[str] = []
        unchanged = []
        for column in params["columns"]:
            source = table.kinds[column]
            if source == to:
                unchanged.append(column)
                continue
            series = table.df[column]
            try:
                new, fmt = _cast(series, source, to, params, column)
            except (ValueError, TypeError) as exc:
                raise DataError(
                    f"'{column}' cannot be converted to {KIND_NAMES[to]}: {exc}"
                ) from exc
            if fmt:
                formats[column] = fmt
            lost = series.notna() & new.isna()
            n_lost = int(lost.sum())
            lost_total += n_lost
            text = f"{source} → {to}"
            if n_lost:
                text += f", {plural(n_lost, 'value')} became missing"
                if source in STRING_KINDS:
                    failed_examples.append(examples(series[lost]))
            if fmt:
                text += f" (format {fmt})"
            details.append({"column": column, "text": text})
            updates[column] = (storage(new, to), to)
        result = set_columns(table, updates) if updates else table
        changed = len(updates)
        summary = (
            f"Changed {plural(changed, 'column')} to {KIND_NAMES[to]}."
            if changed
            else f"Nothing to change: the columns already hold {KIND_NAMES[to]}."
        )
        warnings = []
        if lost_total:
            shown = ", ".join(e for e in failed_examples if e)
            summary = summary[:-1] + (
                f"; {plural(lost_total, 'value')} could not be converted and "
                f"{'is' if lost_total == 1 else 'are'} missing now"
                + (f" (e.g. {shown})." if shown else ".")
            )
        if unchanged and changed:
            warnings.append(f"Already {KIND_NAMES[to]}: {quoted(unchanged)}.")
        return StepResult(
            result,
            summary,
            details=details,
            fitted={"kinds": {c: table.kinds[c] for c in updates}, "formats": formats},
            warnings=warnings,
        )

    def to_pandas(
        self, params: dict[str, Any], fitted: dict[str, Any] | None, ctx: StepContext
    ) -> list[str]:
        fitted = fitted or {}
        kinds = fitted.get("kinds", {})
        formats = fitted.get("formats", {})
        return [
            _cast_code(column, kind, params["to"], params, formats.get(column))
            for column, kind in kinds.items()
        ] or ["# Nothing to change."]


# --------------------------------------------------------------------------- #
# Regular expressions (Formula and Extract from text)
# --------------------------------------------------------------------------- #

# Longer patterns are refused: Python's re has no time limit, so patterns stay small.
MAX_PATTERN_LENGTH = 200
_REPEATS = tuple(
    getattr(_sre, name) for name in ("MAX_REPEAT", "MIN_REPEAT") if hasattr(_sre, name)
)
_POSSESSIVE = getattr(_sre, "POSSESSIVE_REPEAT", None)
_ATOMIC = getattr(_sre, "ATOMIC_GROUP", None)


def _nested_repetition(items: Any, inside: bool = False) -> bool:
    """Whether a parsed pattern repeats something that repeats itself, like (a+)+ or
    (\\w+\\s?)*: such patterns can take very long on text that almost matches."""
    for op, av in items:
        if op in _REPEATS or (_POSSESSIVE is not None and op is _POSSESSIVE):
            low, high, sub = av
            many = high > 1
            if inside and low < high and op is not _POSSESSIVE:
                return True
            if _nested_repetition(sub, inside or many):
                return True
            continue
        flag = False if _ATOMIC is not None and op is _ATOMIC else inside
        children = av if isinstance(av, (tuple, list)) else (av,)
        for child in children:
            parts = child if isinstance(child, list) else [child]
            for part in parts:
                if isinstance(part, _sre.SubPattern) and _nested_repetition(part, flag):
                    return True
    return False


def check_pattern(pattern: str, label: str) -> None:
    """Raise :class:`DataError` unless ``pattern`` is a regular expression of at most
    ``MAX_PATTERN_LENGTH`` characters that compiles and has no nested repetition."""
    if len(pattern) > MAX_PATTERN_LENGTH:
        raise DataError(f"{label}: at most {MAX_PATTERN_LENGTH} characters.")
    try:
        re.compile(pattern)
        nested = _nested_repetition(_sre.parse(pattern))
    except (re.error, OverflowError, RecursionError, ValueError) as exc:
        raise DataError(f"{label}: this is not a valid regular expression ({exc}).") from None
    if nested:
        raise DataError(
            f"{label}: a repeated part that repeats something itself, like (a+)+ or "
            r"(\w+\s?)*, can take very long; write it without the nested repetition."
        )


# --------------------------------------------------------------------------- #
# Formula
# --------------------------------------------------------------------------- #

# Characters of a formula without the dictionaries of map(…), and with them.
MAX_FORMULA_LENGTH = 2_000
MAX_FORMULA_TEXT = 20_000
# Nodes of the syntax tree, without the entries of map(…) dictionaries.
MAX_FORMULA_NODES = 300
MAX_FORMULA_DEPTH = 100
# Characters of the generated code: conditions appear in it twice (the value and where it is
# missing), so nested conditions double it.
MAX_FORMULA_CODE = 30_000
MAX_MAP_ENTRIES = 500
# Characters of a literal text argument (a text to find, a replacement, a separator).
MAX_TEXT_ARGUMENT = 200
MAX_TEXT_LENGTH = 10_000  # left, right, mid
MAX_PART_INDEX = 1000  # split_part
# name: (min arguments, max arguments)
FORMULA_FUNCTIONS = {
    "log": (1, 1),
    "log1p": (1, 1),
    "exp": (1, 1),
    "sqrt": (1, 1),
    "abs": (1, 1),
    "round": (1, 2),
    "floor": (1, 1),
    "ceil": (1, 1),
    "min": (2, 2),
    "max": (2, 2),
    "clip": (3, 3),
    "where": (3, 3),
    "isnull": (1, 1),
    "left": (2, 2),
    "right": (2, 2),
    "mid": (3, 3),
    "upper": (1, 1),
    "lower": (1, 1),
    "strip": (1, 1),
    "len": (1, 1),
    "contains": (2, 3),
    "startswith": (2, 3),
    "endswith": (2, 3),
    "replace": (3, 3),
    "extract": (2, 2),
    "split_part": (3, 3),
    "concat": (2, 50),
    "to_number": (1, 1),
    "to_text": (1, 1),
    "map": (2, 3),
    "coalesce": (2, 50),
}
# How the text functions are called, for messages.
TEXT_FUNCTIONS = {
    "left": "left(x, n)",
    "right": "right(x, n)",
    "mid": "mid(x, start, length)",
    "upper": "upper(x)",
    "lower": "lower(x)",
    "strip": "strip(x)",
    "len": "len(x)",
    "contains": 'contains(x, "text")',
    "startswith": 'startswith(x, "text")',
    "endswith": 'endswith(x, "text")',
    "replace": 'replace(x, "old", "new")',
    "extract": 'extract(x, "pattern")',
    "split_part": 'split_part(x, "separator", k)',
    "concat": "concat(a, b, …)",
    "to_number": "to_number(x)",
    "to_text": "to_text(x)",
    "map": 'map(x, {"a": 1, …}, default)',
    "coalesce": "coalesce(a, b, …)",
}
# Keyword arguments (all others are refused).
_KEYWORDS = {name: ("ignore_case",) for name in ("contains", "startswith", "endswith")}
_NUMPY_FUNCTIONS = {
    "log": "np.log",
    "log1p": "np.log1p",
    "exp": "np.exp",
    "sqrt": "np.sqrt",
    "abs": "np.abs",
    "floor": "np.floor",
    "ceil": "np.ceil",
    "min": "np.minimum",
    "max": "np.maximum",
    "clip": "np.clip",
}
_BINARY = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.FloorDiv: "//",
    ast.Mod: "%",
    ast.Pow: "**",
}
_COMPARE = {
    ast.Eq: "==",
    ast.NotEq: "!=",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
}
_ALLOWED_TEXT = (
    "Allowed are numbers, text in quotes, column names (`my column` in backticks if "
    "needed), + - * / // % **, comparisons, and, or, not, and the functions "
    + ", ".join(FORMULA_FUNCTIONS)
    + "."
)
_NOT_ALLOWED = {
    ast.Attribute: "Attribute access (a.b) is not allowed in formulas.",
    ast.Subscript: "Indexing (a[0]) is not allowed in formulas.",
    ast.Lambda: "Functions (lambda) cannot be defined in formulas.",
    ast.IfExp: "Use where(condition, a, b) instead of 'a if condition else b'.",
    ast.JoinedStr: "f-strings are not allowed in formulas; use concat(a, b, …).",
    ast.ListComp: "Comprehensions ([… for …]) are not allowed in formulas.",
    ast.SetComp: "Comprehensions ({… for …}) are not allowed in formulas.",
    ast.DictComp: "Comprehensions ({…: … for …}) are not allowed in formulas.",
    ast.GeneratorExp: "Comprehensions ((… for …)) are not allowed in formulas.",
    ast.NamedExpr: "Assignments (:=) are not allowed in formulas.",
    ast.Starred: "Starred arguments (*a) are not allowed in formulas.",
    ast.Dict: "A dictionary {…} is not allowed here; it can only be the second argument of "
    'map(x, {"a": 1, …}).',
}
_EVAL_GLOBALS: dict[str, Any] = {"__builtins__": {}}
# Not a literal (see FormulaCompiler._literal).
_NO_LITERAL = object()


# The functions behind the text functions of formulas. They work value by value and are
# copied into the pandas script (and run for raw-row predictions), so all three give the
# same values. Text results are str or None: missing values and empty texts give None.


def formula_text(values):
    """Values as text: text stays, numbers are written like “Change type” writes them
    (3.0 -> "3", 2.5 -> "2.5")."""
    result = []
    for value in values:
        if isinstance(value, str):
            result.append(value or None)
        elif value is None or value != value or value in (float("inf"), float("-inf")):
            result.append(None)
        elif float(value).is_integer() and abs(value) < 1e15:
            result.append(str(int(value)))
        else:
            result.append(repr(float(value)))
    return pd.Series(result, index=values.index, dtype=object)


def formula_number(values):
    """Text as numbers, read like “Change type” reads them; missing if it is no number."""
    text = pd.Series(
        [value.strip() if isinstance(value, str) else None for value in values],
        index=values.index,
        dtype=object,
    )
    numbers = pd.to_numeric(text, errors="coerce")
    try:
        exact = text.where(numbers.notna()).astype("float64")
    except (TypeError, ValueError):
        exact = numbers.astype("float64")
    return exact.replace([np.inf, -np.inf], np.nan)


def formula_left(values, n):
    """The first n characters of every text."""
    result = [(value[:n] or None) if isinstance(value, str) else None for value in values]
    return pd.Series(result, index=values.index, dtype=object)


def formula_right(values, n):
    """The last n characters of every text."""
    result = [
        (value[max(len(value) - n, 0) :] or None) if isinstance(value, str) else None
        for value in values
    ]
    return pd.Series(result, index=values.index, dtype=object)


def formula_mid(values, start, length):
    """length characters of every text from character number start (1 is the first)."""
    result = [
        (value[start - 1 : start - 1 + length] or None) if isinstance(value, str) else None
        for value in values
    ]
    return pd.Series(result, index=values.index, dtype=object)


def formula_upper(values):
    """Every text in upper case."""
    result = [(value.upper() or None) if isinstance(value, str) else None for value in values]
    return pd.Series(result, index=values.index, dtype=object)


def formula_lower(values):
    """Every text in lower case."""
    result = [(value.lower() or None) if isinstance(value, str) else None for value in values]
    return pd.Series(result, index=values.index, dtype=object)


def formula_strip(values):
    """Every text without spaces at the start and the end."""
    result = [(value.strip() or None) if isinstance(value, str) else None for value in values]
    return pd.Series(result, index=values.index, dtype=object)


def formula_len(values):
    """The number of characters of every text."""
    result = [len(value) if isinstance(value, str) else None for value in values]
    return pd.Series(result, index=values.index, dtype="float64")


def formula_contains(values, text, ignore_case=False):
    """1 if a text contains text (plain text, not a pattern), else 0."""
    if ignore_case:
        text = text.lower()
    result = []
    for value in values:
        if not isinstance(value, str):
            result.append(None)
        else:
            result.append(float(text in (value.lower() if ignore_case else value)))
    return pd.Series(result, index=values.index, dtype="float64")


def formula_startswith(values, text, ignore_case=False):
    """1 if a text starts with text, else 0."""
    if ignore_case:
        text = text.lower()
    result = []
    for value in values:
        if not isinstance(value, str):
            result.append(None)
        else:
            result.append(float((value.lower() if ignore_case else value).startswith(text)))
    return pd.Series(result, index=values.index, dtype="float64")


def formula_endswith(values, text, ignore_case=False):
    """1 if a text ends with text, else 0."""
    if ignore_case:
        text = text.lower()
    result = []
    for value in values:
        if not isinstance(value, str):
            result.append(None)
        else:
            result.append(float((value.lower() if ignore_case else value).endswith(text)))
    return pd.Series(result, index=values.index, dtype="float64")


def formula_replace(values, old, new):
    """Every text with old replaced by new (plain text, not a pattern)."""
    result = [
        (value.replace(old, new) or None) if isinstance(value, str) else None for value in values
    ]
    return pd.Series(result, index=values.index, dtype=object)


def formula_extract(values, pattern):
    """The first group (or the whole match) of the regular expression pattern in every
    text; missing if it does not match."""
    regex = re.compile(pattern)
    result = []
    for value in values:
        match = regex.search(value) if isinstance(value, str) else None
        result.append(None if match is None else match.group(1 if regex.groups else 0) or None)
    return pd.Series(result, index=values.index, dtype=object)


def formula_split_part(values, separator, k):
    """Part number k of every text split at separator (1 is the first part, -1 the last);
    missing if there is no such part."""
    number = k - 1 if k > 0 else k
    result = []
    for value in values:
        if not isinstance(value, str):
            result.append(None)
            continue
        parts = value.split(separator)
        result.append((parts[number] or None) if -len(parts) <= number < len(parts) else None)
    return pd.Series(result, index=values.index, dtype=object)


def formula_concat(*parts):
    """The texts (Series or single texts) joined; missing if one of them is missing."""
    index = next(part.index for part in parts if isinstance(part, pd.Series))
    columns = [
        part.tolist() if isinstance(part, pd.Series) else [part] * len(index) for part in parts
    ]
    result = []
    for row in zip(*columns, strict=True):
        result.append("".join(row) if all(isinstance(v, str) for v in row) else None)
    return pd.Series(result, index=index, dtype=object)


def formula_map(values, mapping, default=None):
    """Every value looked up in mapping; values not in it give default (None: missing)."""
    result = []
    for value in values:
        if value is None or value != value:
            result.append(None)
        else:
            result.append(mapping.get(value, default))
    return pd.Series(result, index=values.index, dtype=object)


def formula_coalesce(first, *others):
    """The first value that is not missing, from left to right."""
    result = first
    for other in others:
        result = result.where(result.notna(), other)
    return result


FORMULA_HELPERS = {
    function.__name__: function
    for function in (
        formula_text,
        formula_number,
        formula_left,
        formula_right,
        formula_mid,
        formula_upper,
        formula_lower,
        formula_strip,
        formula_len,
        formula_contains,
        formula_startswith,
        formula_endswith,
        formula_replace,
        formula_extract,
        formula_split_part,
        formula_concat,
        formula_map,
        formula_coalesce,
    )
}


def formula_helpers(code: str) -> list[Any]:
    """The helper functions that the formula code calls."""
    return [f for name, f in FORMULA_HELPERS.items() if re.search(rf"\b{name}\(", code)]


class _Expr:
    """A compiled sub-expression: its code, value type and whether it depends on a column.

    ``kind`` is "num", "bool" (0/1, also a number) or "str". Constant sub-expressions
    (``value`` set) are computed while compiling, with numpy semantics; a missing text
    constant has the value None.
    """

    __slots__ = ("code", "kind", "series", "value")

    def __init__(self, code: str, kind: str, series: bool, value: Any = None):
        self.code = code
        self.kind = kind
        self.series = series
        self.value = value

    @classmethod
    def constant(cls, value: Any, kind: str = "num") -> _Expr:
        if isinstance(value, str):
            return cls(repr(value), "str", False, value)
        if value is None:
            return cls("None", "str", False, None)
        number = float(value)
        return cls(repr(number) if math.isfinite(number) else "np.nan", kind, False, number)


def _replace_backticks(text: str) -> tuple[str, dict[str, str]]:
    """Replace `quoted column names` by placeholder identifiers (outside string literals)."""
    out: list[str] = []
    names: dict[str, str] = {}
    quote = None
    i = 0
    while i < len(text):
        char = text[i]
        if quote:
            out.append(char)
            if char == "\\" and i + 1 < len(text):
                out.append(text[i + 1])
                i += 2
                continue
            if char == quote:
                quote = None
        elif char in "'\"":
            quote = char
            out.append(char)
        elif char == "`":
            end = text.find("`", i + 1)
            if end < 0:
                raise DataError("Expression: a backtick (`) is not closed.")
            name = text[i + 1 : end]
            if not name:
                raise DataError("Expression: there is no column name between the backticks.")
            key = f"__col{len(names)}__"
            names[key] = name
            out.append(f" {key} ")
            i = end + 1
            continue
        else:
            out.append(char)
        i += 1
    return "".join(out), names


_LEAVES = (ast.expr_context, ast.operator, ast.unaryop, ast.cmpop, ast.boolop)


def _size(tree: ast.AST, text: str) -> tuple[int, int, int]:
    """The number of nodes and the characters of ``text`` (both without the entries of
    dictionaries), and the nesting depth."""
    nodes = depth = 0
    length = len(text)
    stack = [(tree, 1)]
    while stack:
        node, level = stack.pop()
        nodes += 1
        if not isinstance(node, _LEAVES):
            depth = max(depth, level)
        if isinstance(node, ast.Dict):
            # Entries are literals, checked by map(…); a dictionary holds no column names,
            # so its text is the same as in the formula as written.
            length -= len(ast.get_source_segment(text, node) or "")
            continue
        stack.extend((child, level + 1) for child in ast.iter_child_nodes(node))
    return nodes, length, depth


class FormulaCompiler:
    """Checks a formula against a whitelist and turns it into pandas code.

    The generated code only contains ``df[<column name>]``, number and string literals
    (also in dictionaries), operators and calls of fixed numpy/pandas functions and of the
    ``formula_*`` helpers above, so evaluating it is safe; the step and the exported script
    run the very same code.
    """

    def __init__(self, kinds: dict[str, str]):
        self.kinds = kinds
        self.names: dict[str, str] = {}
        self.columns: list[str] = []

    def compile(self, expression: str) -> _Expr:
        too_long = (
            f"Expression: at most {MAX_FORMULA_LENGTH:,} characters (map(…) dictionaries "
            f"not counted, up to {MAX_FORMULA_TEXT:,} with them)."
        )
        if len(expression) > MAX_FORMULA_TEXT:
            raise DataError(too_long)
        text, self.names = _replace_backticks(expression)
        text = text.strip()
        try:
            with warnings.catch_warnings():
                # "\d" in a text is kept as backslash and d, as in Python.
                warnings.simplefilter("ignore")
                tree = ast.parse(text, mode="eval")
        except SyntaxError as exc:
            if "nested" in str(exc.msg):  # too many nested parentheses
                raise DataError(
                    "Expression: the formula is nested too deeply; split it into several steps."
                ) from None
            where = f" (at character {exc.offset})" if exc.offset else ""
            raise DataError(f"Expression: this is not a valid formula{where}.") from None
        except (RecursionError, MemoryError, ValueError):
            raise DataError("Expression: the formula is too complex.") from None
        nodes, length, depth = _size(tree, text)
        # Column names count as written (`Name`), not as the placeholders that replace them.
        length += sum(len(name) + 2 - len(f" {key} ") for key, name in self.names.items())
        if length > MAX_FORMULA_LENGTH:
            raise DataError(too_long)
        if nodes > MAX_FORMULA_NODES:
            raise DataError("Expression: the formula is too long; split it into several steps.")
        if depth > MAX_FORMULA_DEPTH:
            raise DataError(
                f"Expression: the formula is nested too deeply (at most {MAX_FORMULA_DEPTH} "
                "levels); split it into several steps."
            )
        try:
            return self.visit(tree.body)
        except RecursionError:
            raise DataError("Expression: the formula is too complex.") from None

    # ---------------------------------------------------------------- nodes

    def visit(self, node: ast.AST) -> _Expr:
        expr = self._visit(node)
        if len(expr.code) > MAX_FORMULA_CODE:
            raise DataError("Expression: the formula is too complex; split it into several steps.")
        return expr

    def _visit(self, node: ast.AST) -> _Expr:
        if isinstance(node, ast.Constant):
            return self._constant(node.value)
        if isinstance(node, ast.Name):
            return self._name(node.id)
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            return self._binary(node)
        if isinstance(node, ast.UnaryOp):
            return self._unary(node)
        if isinstance(node, ast.Compare):
            return self._compare_chain(node)
        if isinstance(node, ast.BoolOp):
            values = [self._number(self.visit(v), "and/or") for v in node.values]
            return self._logic(values, "&" if isinstance(node.op, ast.And) else "|")
        if isinstance(node, ast.Call):
            return self._call(node)
        for kind, message in _NOT_ALLOWED.items():
            if isinstance(node, kind):
                raise DataError(f"Expression: {message}")
        what = type(node).__name__
        raise DataError(f"Expression: '{what}' is not allowed in formulas. {_ALLOWED_TEXT}")

    def _constant(self, value: Any) -> _Expr:
        if isinstance(value, bool):
            return _Expr.constant(float(value))
        if isinstance(value, (int, float)):
            try:
                number = float(value)
            except OverflowError:
                raise DataError("Expression: a number is too large.") from None
            return _Expr.constant(number)
        if isinstance(value, str):
            return _Expr.constant(value)
        raise DataError(f"Expression: the value {value!r} is not allowed. {_ALLOWED_TEXT}")

    def _name(self, identifier: str) -> _Expr:
        name = self.names.get(identifier, identifier)
        if identifier.startswith("__") and identifier not in self.names:
            raise DataError(f"Expression: '{identifier}' is not allowed.")
        if name not in self.kinds:
            if name in FORMULA_FUNCTIONS:
                raise DataError(f"Expression: {name} is a function; write {name}(…).")
            hint = (
                " Put column names with spaces or symbols in backticks, e.g. `my column`."
                if identifier in self.names or not identifier.isidentifier()
                else ""
            )
            raise DataError(f"Expression: there is no column '{name}'.{hint}")
        kind = self.kinds[name]
        if kind == "datetime":
            raise DataError(
                f"Expression: '{name}' holds dates, which cannot be used in formulas. Split it "
                "into numbers with “Date parts” or change its type first."
            )
        if name not in self.columns:
            self.columns.append(name)
        value_kind = "str" if kind in STRING_KINDS else ("bool" if kind == "boolean" else "num")
        return _Expr(f"df[{name!r}]", value_kind, True)

    @staticmethod
    def _number(expr: _Expr, what: str) -> _Expr:
        if expr.kind == "str":
            raise DataError(
                f"Expression: text ({expr.code}) cannot be used with {what}; only == and != "
                "work with text (concat(a, b) joins texts, to_number(x) reads numbers)."
            )
        return expr

    @staticmethod
    def _fold(code: str, *args: _Expr) -> _Expr:
        """Compute a sub-expression of constants now (numpy semantics: 1/0 is inf)."""
        namespace = {"np": np, "pd": pd}
        namespace.update({f"_a{i}": np.float64(a.value) for i, a in enumerate(args)})
        with np.errstate(all="ignore"):
            value = eval(code, _EVAL_GLOBALS, namespace)  # noqa: S307 (fixed code, no user input)
        return _Expr.constant(float(value))

    def _binary(self, node: ast.BinOp) -> _Expr:
        op = _BINARY[type(node.op)]
        left = self._number(self.visit(node.left), f"'{op}'")
        right = self._number(self.visit(node.right), f"'{op}'")
        if not left.series and not right.series:
            return self._fold(f"_a0 {op} _a1", left, right)
        return _Expr(f"({left.code} {op} {right.code})", "num", True)

    def _unary(self, node: ast.UnaryOp) -> _Expr:
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.Not):
            return self._not(self._number(operand, "not"))
        if isinstance(node.op, (ast.USub, ast.UAdd)):
            operand = self._number(operand, "a sign")
            if isinstance(node.op, ast.UAdd):
                return operand
            if not operand.series:
                return self._fold("-_a0", operand)
            return _Expr(f"(-{operand.code})", "num", True)
        raise DataError(f"Expression: '~' is not allowed. {_ALLOWED_TEXT}")

    @staticmethod
    def _masked(code: str, *parts: _Expr) -> str:
        """``code`` (a boolean Series) as 1.0 / 0.0, missing where an input is missing."""
        masks = [f"{p.code}.notna()" for p in parts if p.series]
        return f"({code}).astype('float64').where({' & '.join(masks)})"

    def _compare(self, left: _Expr, op: str, right: _Expr) -> _Expr:
        if (left.kind == "str") != (right.kind == "str"):
            raise DataError(
                f"Expression: cannot compare text with numbers ({left.code} {op} {right.code}); "
                "use to_number(x) or to_text(x)."
            )
        if left.kind == "str" and op not in ("==", "!="):
            raise DataError(f"Expression: text can only be compared with == and !=, not {op}.")
        if not left.series and not right.series:
            if left.value is None or right.value is None:
                return _Expr.constant(float("nan"), "bool")
            return _Expr.constant(float(_PY_COMPARE[op](left.value, right.value)), "bool")
        return _Expr(self._masked(f"{left.code} {op} {right.code}", left, right), "bool", True)

    def _compare_chain(self, node: ast.Compare) -> _Expr:
        operands = [self.visit(node.left), *(self.visit(c) for c in node.comparators)]
        parts = []
        for i, op_node in enumerate(node.ops):
            op = _COMPARE.get(type(op_node))
            if op is None:
                raise DataError(
                    "Expression: 'in' and 'is' are not allowed; use contains(x, \"text\") or "
                    f"isnull(x). {_ALLOWED_TEXT}"
                )
            parts.append(self._compare(operands[i], op, operands[i + 1]))
        return parts[0] if len(parts) == 1 else self._logic(parts, "&")

    def _logic(self, values: list[_Expr], op: str) -> _Expr:
        if not any(v.series for v in values):
            truth = [bool(v.value) and not math.isnan(v.value) for v in values]
            return _Expr.constant(float(all(truth) if op == "&" else any(truth)), "bool")
        code = f" {op} ".join(f"({v.code} != 0)" for v in values)
        return _Expr(self._masked(code, *values), "bool", True)

    def _not(self, value: _Expr) -> _Expr:
        if not value.series:
            return _Expr.constant(float(value.value == 0), "bool")
        return _Expr(self._masked(f"{value.code} == 0", value), "bool", True)

    def _call(self, node: ast.Call) -> _Expr:
        if not isinstance(node.func, ast.Name) or node.func.id not in FORMULA_FUNCTIONS:
            for kind, message in _NOT_ALLOWED.items():
                if isinstance(node.func, kind):  # e.g. "a".format(…), (lambda: 1)()
                    raise DataError(f"Expression: {message}")
            name = node.func.id if isinstance(node.func, ast.Name) else "this function"
            raise DataError(f"Expression: {name}(…) is not allowed. {_ALLOWED_TEXT}")
        name = node.func.id
        allowed = _KEYWORDS.get(name, ())
        if any(isinstance(a, ast.Starred) for a in node.args) or any(
            k.arg not in allowed for k in node.keywords
        ):
            extra = f" (and {allowed[0]}=1)" if allowed else ""
            raise DataError(f"Expression: {name}(…) takes plain arguments only{extra}.")
        low, high = FORMULA_FUNCTIONS[name]
        if not low <= len(node.args) + len(node.keywords) <= high:
            count = (
                f"{low}"
                if low == high
                else f"{low} or {high}"
                if high == low + 1
                else f"{low} to {high}"
            )
            raise DataError(f"Expression: {name}(…) takes {count} argument(s).")
        if name in TEXT_FUNCTIONS:
            return self._text_call(name, node)
        if name == "where":
            return self._where(node)
        if name == "isnull":
            value = self.visit(node.args[0])
            if not value.series:
                return _Expr.constant(
                    float(value.value is None or value.value != value.value), "bool"
                )
            return _Expr(f"{value.code}.isna().astype('float64')", "bool", True)
        if name == "round":
            value = self._number(self.visit(node.args[0]), "round")
            digits = 0
            if len(node.args) == 2:
                arg = node.args[1]
                if (
                    not isinstance(arg, ast.Constant)
                    or not isinstance(arg.value, int)
                    or isinstance(arg.value, bool)
                    or not 0 <= arg.value <= 15
                ):
                    raise DataError("Expression: round(x, digits) needs digits from 0 to 15.")
                digits = arg.value
            if not value.series:
                return self._fold(f"np.round(_a0, {digits})", value)
            return _Expr(f"np.round({value.code}, {digits})", "num", True)
        args = [self._number(self.visit(a), f"{name}(…)") for a in node.args]
        function = _NUMPY_FUNCTIONS[name]
        if not any(a.series for a in args):
            call = ", ".join(f"_a{i}" for i in range(len(args)))
            return self._fold(f"{function}({call})", *args)
        return _Expr(f"{function}({', '.join(a.code for a in args)})", "num", True)

    def _where(self, node: ast.Call) -> _Expr:
        condition = self._number(self.visit(node.args[0]), "where(condition, …)")
        a, b = self.visit(node.args[1]), self.visit(node.args[2])
        if (a.kind == "str") != (b.kind == "str"):
            raise DataError(
                "Expression: where(…, a, b) needs a and b of the same type (both text or both "
                "numbers); use to_text(x) or to_number(x)."
            )
        kind = "str" if a.kind == "str" else ("bool" if a.kind == b.kind == "bool" else "num")
        if not condition.series:
            chosen_value = a if condition.value and not math.isnan(condition.value) else b
            if chosen_value.kind != kind:
                return _Expr(chosen_value.code, kind, chosen_value.series, chosen_value.value)
            return chosen_value
        code = f"pd.Series(np.where({condition.code} != 0, {a.code}, {b.code}), index=df.index)"
        code += f".where({condition.code}.notna())"
        return _Expr(code, kind, True)

    # ------------------------------------------------------- text functions

    @staticmethod
    def _literal(node: ast.AST) -> Any:
        """The value of a literal text or number (also negative), else ``_NO_LITERAL``."""
        sign = 1
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            sign = -1 if isinstance(node.op, ast.USub) else 1
            node = node.operand
            if not isinstance(node, ast.Constant) or isinstance(node.value, (str, bool)):
                return _NO_LITERAL
        if not isinstance(node, ast.Constant):
            return _NO_LITERAL
        value = node.value
        if isinstance(value, str):
            return value
        if isinstance(value, (bool, int, float)):
            try:
                number = sign * float(value)
            except OverflowError:
                return _NO_LITERAL
            return number if math.isfinite(number) else _NO_LITERAL
        return _NO_LITERAL

    def _whole(self, name: str, node: ast.AST, what: str, low: int, high: int) -> int:
        value = self._literal(node)
        if isinstance(value, float) and value.is_integer() and low <= value <= high:
            return int(value)
        raise DataError(
            f"Expression: {TEXT_FUNCTIONS[name]} needs {what} as a whole number from "
            f"{low:,} to {high:,} written in the formula (not a column)."
        )

    def _text_literal(self, name: str, node: ast.AST, what: str, empty: bool = False) -> str:
        value = self._literal(node)
        if value == "" and not empty:
            raise DataError(f"Expression: {TEXT_FUNCTIONS[name]}: {what} is empty.")
        if not isinstance(value, str):
            raise DataError(
                f"Expression: {TEXT_FUNCTIONS[name]} needs {what} as a text in quotes written "
                "in the formula (not a column)."
            )
        if len(value) > MAX_TEXT_ARGUMENT:
            raise DataError(
                f"Expression: {TEXT_FUNCTIONS[name]}: {what} has at most {MAX_TEXT_ARGUMENT} "
                "characters."
            )
        return value

    def _flag(self, name: str, node: ast.AST) -> bool:
        value = self._literal(node)
        if isinstance(value, str) or value not in (0.0, 1.0):
            raise DataError(
                f'Expression: {name}(x, "text", ignore_case=1): ignore_case is 1 (ignore upper '
                "and lower case) or 0."
            )
        return bool(value)

    @staticmethod
    def _as_text(expr: _Expr) -> _Expr:
        """``expr`` as text: numbers are written like to_text(x)."""
        if expr.kind == "str":
            return expr
        if not expr.series:
            value = formula_text(pd.Series([expr.value], dtype=object)).iloc[0]
            return _Expr.constant(value)
        return _Expr(f"formula_text({expr.code})", "str", True)

    @staticmethod
    def _series(expr: _Expr) -> str:
        """Code of ``expr`` as a Series (a constant is repeated for every row)."""
        if expr.series:
            return expr.code
        dtype = "object" if expr.kind == "str" else "float64"
        return f"pd.Series({expr.code}, index=df.index, dtype={dtype!r})"

    def _text_call(self, name: str, node: ast.Call) -> _Expr:
        args = node.args
        if name in ("concat", "coalesce"):
            return getattr(self, f"_{name}")([self.visit(a) for a in args])
        if name == "map":
            return self._map(node)
        x = self.visit(args[0])
        if name == "to_number":
            if x.kind != "str":
                return _Expr(x.code, "num", x.series, x.value)
            if not x.series:
                value = formula_number(pd.Series([x.value], dtype=object)).iloc[0]
                return _Expr.constant(float(value))
            return _Expr(f"formula_number({x.code})", "num", True)
        x = self._as_text(x)
        if name == "to_text":
            return x
        values = self._series(x)
        extra: list[str] = []
        kind = "str"
        if name in ("left", "right"):
            extra.append(repr(self._whole(name, args[1], "n", 0, MAX_TEXT_LENGTH)))
        elif name == "mid":
            extra.append(repr(self._whole(name, args[1], "start", 1, MAX_TEXT_LENGTH)))
            extra.append(repr(self._whole(name, args[2], "length", 0, MAX_TEXT_LENGTH)))
        elif name == "len":
            kind = "num"
        elif name in ("contains", "startswith", "endswith"):
            extra.append(repr(self._text_literal(name, args[1], "the text to look for")))
            flags = [*args[2:], *(k.value for k in node.keywords)]
            if flags and self._flag(name, flags[0]):
                extra.append("ignore_case=True")
            kind = "bool"
        elif name == "replace":
            extra.append(repr(self._text_literal(name, args[1], "the text to replace")))
            extra.append(repr(self._text_literal(name, args[2], "the new text", empty=True)))
        elif name == "extract":
            pattern = self._literal(args[1])
            if not isinstance(pattern, str) or not pattern:
                raise DataError(
                    'Expression: extract(x, "pattern") needs the pattern (a regular expression) '
                    'as a text in quotes written in the formula, e.g. extract(`Name`, "(\\w+)\\.").'
                )
            check_pattern(pattern, 'Expression: the pattern of extract(x, "pattern")')
            extra.append(repr(pattern))
        elif name == "split_part":
            extra.append(repr(self._text_literal(name, args[1], "the separator")))
            k = self._whole(name, args[2], "k", -MAX_PART_INDEX, MAX_PART_INDEX)
            if k == 0:
                raise DataError(
                    f"Expression: {TEXT_FUNCTIONS[name]}: k is 1 for the first part, 2 for the "
                    "second, -1 for the last."
                )
            extra.append(repr(k))
        return _Expr(f"formula_{name}({', '.join([values, *extra])})", kind, True)

    def _concat(self, args: list[_Expr]) -> _Expr:
        texts = [self._as_text(a) for a in args]
        if not any(t.series for t in texts):
            if any(t.value is None for t in texts):
                return _Expr.constant(None)
            return _Expr.constant("".join(t.value for t in texts) or None)
        return _Expr(f"formula_concat({', '.join(t.code for t in texts)})", "str", True)

    def _coalesce(self, args: list[_Expr]) -> _Expr:
        if len({a.kind == "str" for a in args}) > 1:
            raise DataError(
                "Expression: coalesce(a, b, …) needs values of the same type (all text or all "
                "numbers); use to_text(x) or to_number(x)."
            )
        kind = (
            "str"
            if args[0].kind == "str"
            else "bool"
            if all(a.kind == "bool" for a in args)
            else "num"
        )
        if not args[0].series:
            missing = args[0].value is None or args[0].value != args[0].value
            if not missing or len(args) == 1:
                return _Expr(args[0].code, kind, False, args[0].value)
            return self._coalesce(args[1:])
        if len(args) == 1:
            return args[0]
        return _Expr(f"formula_coalesce({', '.join(a.code for a in args)})", kind, True)

    def _map(self, node: ast.Call) -> _Expr:
        usage = TEXT_FUNCTIONS["map"]
        x = self.visit(node.args[0])
        table = node.args[1]
        if not isinstance(table, ast.Dict):
            raise DataError(
                f"Expression: {usage} needs a dictionary of values written in the formula as "
                'its second argument, e.g. map(`Embarked`, {"S": 1, "C": 2, "Q": 3}).'
            )
        if len(table.keys) > MAX_MAP_ENTRIES:
            raise DataError(f"Expression: {usage}: at most {MAX_MAP_ENTRIES} entries.")
        if not table.keys:
            raise DataError(f"Expression: {usage}: the dictionary is empty.")
        pairs = []
        for key_node, value_node in zip(table.keys, table.values, strict=True):
            key = self._literal(key_node) if key_node is not None else _NO_LITERAL
            value = self._literal(value_node)
            if key is _NO_LITERAL or value is _NO_LITERAL:
                raise DataError(
                    f"Expression: {usage}: keys and values are texts in quotes or numbers "
                    "written in the formula (no columns or calculations)."
                )
            pairs.append((key, value))
        default: Any = None
        if len(node.args) == 3:
            default = self._literal(node.args[2])
            if default is _NO_LITERAL:
                raise DataError(
                    f"Expression: {usage}: the default is a text in quotes or a number written "
                    "in the formula."
                )
        mapping: dict[Any, Any] = {}
        for key, value in pairs:
            if x.kind == "str":
                key = as_label(key)
            elif isinstance(key, str):
                number = as_number(key)
                if number is None:
                    raise DataError(
                        f"Expression: {usage}: the key '{key}' is a text, but x holds numbers; "
                        "use to_text(x) to map texts."
                    )
                key = number
            if key in mapping:
                raise DataError(f"Expression: {usage}: the key '{as_label(key)}' is listed twice.")
            mapping[key] = value
        results = [*mapping.values(), *([] if default is None else [default])]
        numeric = all(not isinstance(v, str) for v in results)
        if not numeric:
            mapping = {k: as_label(v) or None for k, v in mapping.items()}
            default = None if default is None else as_label(default) or None
        literal = "{" + ", ".join(f"{code_value(k)}: {code_value(v)}" for k, v in mapping.items())
        literal += "}"
        call = f"formula_map({self._series(x)}, {literal}"
        call += f", {code_value(default)})" if default is not None else ")"
        if numeric:
            return _Expr(f"{call}.astype('float64')", "num", True)
        return _Expr(call, "str", True)


_PY_COMPARE = {
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
}


def formula_code(expr: _Expr) -> str:
    """The right-hand side that turns a compiled formula into a column of ``df``."""
    if expr.kind == "str":
        return f"pd.Series({expr.code}, index=df.index, dtype='object')"
    return (
        f"pd.Series({expr.code}, index=df.index, dtype='float64')"
        ".replace([np.inf, -np.inf], np.nan)"
    )


def evaluate_formula(code: str, df: pd.DataFrame) -> pd.Series:
    """Evaluate code made by :func:`formula_code` (only names: df, np, pd and the
    ``formula_*`` helpers)."""
    namespace = {"df": df, "np": np, "pd": pd, **FORMULA_HELPERS}
    with np.errstate(all="ignore"):
        return eval(code, _EVAL_GLOBALS, namespace)  # noqa: S307


FORMULA_HELP = (
    "Columns by name (`my column` in backticks), numbers, text in quotes, + - * / // % **, "
    "comparisons (give 1/0; text with == and !=), and, or, not, and log, log1p, exp, sqrt, "
    "abs, round, floor, ceil, min(a, b), max(a, b), clip(x, lo, hi), where(cond, a, b), "
    "isnull(x). Text: left(x, n), right(x, n), mid(x, start, length), upper, lower, strip, "
    'len, contains(x, "text") (also startswith, endswith; ignore_case=1), '
    'replace(x, "old", "new"), extract(x, "pattern"), split_part(x, " ", k), '
    'concat(a, b, …), to_number, to_text, map(x, {"A": 1, …}, default), coalesce(a, b, …). '
    "A missing input gives a missing result."
)


@register
class Formula(Step):
    type = "formula"
    label = "Formula"
    group = "columns"
    description = (
        "Compute a new column from other columns, e.g. weight / height ** 2, or from text, "
        'e.g. the deck of a cabin: map(upper(left(`Cabin`, 1)), {"A": 1, …}).'
    )
    # The deck of a Titanic cabin as a number: C85 → C → 3; a missing cabin stays missing.
    example = {
        "name": "Deck",
        "expression": 'map(upper(left(`Cabin`, 1)), {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, '
        '"F": 6, "G": 7, "T": 8})',
    }
    params = (
        Param("name", "string", "New column", default="", help="An existing column is replaced."),
        Param("expression", "string", "Formula", default="", help=FORMULA_HELP),
    )

    def clean(self, table: Table, ctx: StepContext) -> dict[str, Any]:
        params = super().clean(table, ctx)
        compiler = FormulaCompiler(table.kinds)
        expr = compiler.compile(params["expression"])
        code = formula_code(expr)
        try:
            compile(code, "<formula>", "eval")
        except (SyntaxError, RecursionError, MemoryError, ValueError):
            raise DataError("Expression: the formula is too complex.") from None
        params["_code"] = code
        params["_kind"] = {"str": "categorical", "bool": "boolean"}.get(expr.kind, "numeric")
        params["_columns"] = compiler.columns
        return params

    def run(self, table: Table, params: dict[str, Any], ctx: StepContext) -> StepResult:
        name, kind = params["name"], params["_kind"]
        values = evaluate_formula(params["_code"], table.df)
        values = storage(values, kind)
        replaced = name in table.kinds
        result = set_columns(table, {name: (values, kind)})
        missing = int(values.isna().sum())
        inputs = params["_columns"]
        input_missing = (
            table.df[inputs].isna().any(axis=1).to_numpy()
            if inputs
            else np.zeros(table.n_rows, bool)
        )
        invalid = int((values.isna().to_numpy() & ~input_missing).sum())
        verb = "Replaced" if replaced else "Added"
        summary = f"{verb} '{name}' ({KIND_NAMES[kind]})"
        notes = []
        if missing - invalid:
            notes.append(
                f"{plural(missing - invalid, 'value')} missing because an input is missing"
            )
        if invalid:
            reasons = (
                "no match, no such part, a value not in the mapping or an empty text"
                if formula_helpers(params["_code"])
                else "division by zero or log of a negative number"
            )
            notes.append(f"{plural(invalid, 'value')} could not be computed (e.g. {reasons})")
        summary += ": " + "; ".join(notes) + "." if notes else "."
        warnings = []
        if kind == "numeric" and missing < table.n_rows:
            present = values.dropna()
            if len(present) and present.nunique() == 1:
                warnings.append(f"Every value of '{name}' is {fmt_number(present.iloc[0])}.")
        elif missing == table.n_rows and table.n_rows:
            warnings.append(f"Every value of '{name}' is missing.")
        if replaced and name == ctx.target:
            warnings.append(f"This step replaces the target column '{name}'.")
        return StepResult(result, summary, warnings=warnings)

    def to_pandas(
        self, params: dict[str, Any], fitted: dict[str, Any] | None, ctx: StepContext
    ) -> list[str]:
        lines = []
        for helper in formula_helpers(params["_code"]):
            lines += [*_code_lines(helper), ""]
        return [
            *lines,
            f"# {comment(params['name'])} = {comment(params['expression'])}",
            'with np.errstate(all="ignore"):  # 1 / 0 gives inf -> missing',
            f"    df[{params['name']!r}] = {params['_code']}",
        ]


# --------------------------------------------------------------------------- #
# Replace values
# --------------------------------------------------------------------------- #


def replace_mapping(column: str, kind: str, mapping: dict[str, Any], label: str) -> dict[Any, Any]:
    """``{"old": new}`` from the UI as a mapping in the storage format of ``kind``. Old
    values that cannot occur in the column (text in a numeric column) are left out."""
    result: dict[Any, Any] = {}
    for old, new in mapping.items():
        if kind in STRING_KINDS:
            key: Any = old
        else:
            try:
                key = cell_value(old, kind, column, label)
            except DataError:
                continue
        result[key] = cell_value(new, kind, column, label) if new is not None else None
    if not result:
        raise DataError(
            f"{label}: none of the old values can occur in '{column}', which holds "
            f"{KIND_NAMES[kind]}."
        )
    return result


@register
class ReplaceValues(Step):
    type = "replace_values"
    label = "Replace values"
    group = "columns"
    description = "Replace values with others, e.g. fix spellings or merge categories."
    params = (
        Param("columns", "columns", "Columns", default=[]),
        Param(
            "mapping",
            "mapping",
            "Replacements",
            default={},
            help="Old value → new value; leave the new value empty to mark it as missing. "
            "Numbers are compared as numbers in numeric columns.",
            key_kind="value",
        ),
    )

    def clean(self, table: Table, ctx: StepContext) -> dict[str, Any]:
        params = super().clean(table, ctx)
        _required_columns(params)
        if not params["mapping"]:
            raise DataError("Replacements: add at least one value to replace.")
        for column in params["columns"]:
            replace_mapping(column, table.kinds[column], params["mapping"], "Replacements")
        return params

    def run(self, table: Table, params: dict[str, Any], ctx: StepContext) -> StepResult:
        updates: dict[str, tuple[pd.Series, str]] = {}
        details = []
        found: set[str] = set()
        total = 0
        numeric_now = []
        for column in params["columns"]:
            kind = table.kinds[column]
            mapping = replace_mapping(column, kind, params["mapping"], "Replacements")
            series = table.df[column]
            hits = series.isin(list(mapping))
            count = int(hits.sum())
            if not count:
                continue
            present = set(series[hits].unique().tolist())
            for old in params["mapping"]:
                try:
                    key = old if kind in STRING_KINDS else cell_value(old, kind, column, "")
                except DataError:
                    continue
                if key in present:
                    found.add(old)
            new = storage(series.mask(hits, series.map(mapping)), kind)
            updates[column] = (new, kind)
            total += count
            details.append({"column": column, "text": f"{plural(count, 'value')} replaced"})
            if kind in STRING_KINDS:
                distinct = pd.Series(new.dropna().unique(), dtype=object)
                if len(distinct) and pd.to_numeric(distinct, errors="coerce").notna().all():
                    numeric_now.append(column)
        result = set_columns(table, updates) if updates else table
        warnings = []
        unused = [k for k in params["mapping"] if k not in found]
        if unused:
            warnings.append(f"Not found: {quoted(unused)}.")
        if numeric_now:
            warnings.append(
                f"{quoted(numeric_now)} only hold numbers now; use “Change type” to turn "
                f"{'it' if len(numeric_now) == 1 else 'them'} into numbers."
            )
        summary = (
            f"Replaced {plural(total, 'value')} in {plural(len(updates), 'column')}."
            if total
            else "None of the values were found; nothing changed."
        )
        return StepResult(
            result,
            summary,
            details=details,
            # all chosen columns: new data may contain the values even if these rows do not
            fitted={"kinds": {c: table.kinds[c] for c in params["columns"]}},
            warnings=warnings,
        )

    def to_pandas(
        self, params: dict[str, Any], fitted: dict[str, Any] | None, ctx: StepContext
    ) -> list[str]:
        lines = []
        for column, kind in (fitted or {}).get("kinds", {}).items():
            mapping = replace_mapping(column, kind, params["mapping"], "Replacements")
            literal = (
                "{"
                + ", ".join(f"{code_value(k)}: {code_value(v)}" for k, v in mapping.items())
                + "}"
            )
            ref = f"df[{column!r}]"
            lines.append(f"{ref} = {ref}.mask({ref}.isin(list({literal})), {ref}.map({literal}))")
        return lines or ["# Nothing to replace."]


# --------------------------------------------------------------------------- #
# Extract from text
# --------------------------------------------------------------------------- #

MAX_EXTRACT_LENGTH = 10_000
MAX_WORD_INDEX = 1000
EXTRACT_PARTS = (
    ("first", "The first characters"),
    ("last", "The last characters"),
    ("pattern", "The match of a pattern (regular expression)"),
    ("word", "A word"),
)
EXTRACT_CASES = (
    ("keep", "Keep as it is"),
    ("upper", "UPPER CASE"),
    ("lower", "lower case"),
)


def extract_text(values, part, length=1, pattern=None, index=1, strip=True, case="upper"):
    """A part of every text value: the first or last `length` characters, the first group
    (or the whole match) of the regular expression `pattern`, or word number `index` (1 is
    the first word, -1 the last; words are split by spaces). Missing values, values without
    such a part and empty parts give None."""
    regex = re.compile(pattern) if part == "pattern" else None
    result = []
    for value in values:
        if not isinstance(value, str):
            result.append(None)
            continue
        text = value.strip() if strip else value
        if part == "first":
            piece = text[:length]
        elif part == "last":
            piece = text[-length:]
        elif part == "pattern":
            match = regex.search(text)
            piece = None if match is None else match.group(1 if regex.groups else 0)
        else:
            words = text.split()
            number = index - 1 if index > 0 else index
            piece = words[number] if -len(words) <= number < len(words) else None
        if not piece:
            result.append(None)
        elif case == "upper":
            result.append(piece.upper())
        elif case == "lower":
            result.append(piece.lower())
        else:
            result.append(piece)
    return pd.Series(result, index=values.index, dtype=object)


def extract_mapping(mapping: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """The new values of an extract mapping in storage format and the kind of the new
    column: numbers if every new value is a number, else categories."""
    present = [value for value in mapping.values() if value is not None]
    if present and all(as_number(value) is not None for value in present):
        numbers = {k: None if v is None else as_number(v) for k, v in mapping.items()}
        return numbers, "numeric"
    labels = {k: None if v is None else as_label(v) for k, v in mapping.items()}
    return labels, "categorical"


def _ordinal(n: int) -> str:
    if n == 1:
        return "first"
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _part_text(params: dict[str, Any]) -> str:
    """``the first character``, ``the last 3 characters``, ``the 2nd word`` …"""
    part = params["part"]
    if part in ("first", "last"):
        length = params["length"]
        return f"the {part} {'character' if length == 1 else f'{length:,} characters'}"
    if part == "pattern":
        pattern = params["pattern"]
        shown = pattern if len(pattern) <= 40 else pattern[:39] + "…"
        return f"the part matching '{shown}'"
    index = params["index"]
    if index > 0:
        return f"the {_ordinal(index)} word"
    return "the last word" if index == -1 else f"the {_ordinal(-index)} last word"


def _code_lines(function: Any) -> list[str]:
    return textwrap.dedent(inspect.getsource(function)).rstrip().splitlines()


@register
class ExtractText(Step):
    type = "extract_text"
    label = "Extract from text"
    group = "columns"
    description = (
        "Take a part of a text column into a new column, e.g. the deck letter of a cabin "
        "(C85 → C), and optionally map it to numbers (A → 1, B → 2 …)."
    )
    # The deck of a Titanic cabin as a number: C85 → C → 3; a missing cabin stays missing.
    example = {
        "column": "Cabin",
        "name": "Deck",
        "part": "first",
        "length": 1,
        "strip": True,
        "case": "upper",
        "mapping": {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "T": 8},
    }
    params = (
        Param("column", "column", "Text column", default=None, column_kinds=STRING_KINDS),
        Param("name", "string", "New column", default="", help="An existing column is replaced."),
        Param(
            "part",
            "select",
            "Take",
            default="first",
            options=EXTRACT_PARTS,
            help="Values without such a part (too few words, no match) become missing.",
        ),
        Param(
            "length",
            "int",
            "Characters",
            default=1,
            min=1,
            max=MAX_EXTRACT_LENGTH,
            help="How many characters to take.",
            visible_if={"param": "part", "values": ["first", "last"]},
        ),
        Param(
            "pattern",
            "string",
            "Pattern",
            default="",
            help=(
                r"A regular expression (Python), e.g. [A-Z] for the first capital letter or \d+ "
                "for the first number. With a group in brackets the first group is taken, "
                f"else the whole match. At most {MAX_PATTERN_LENGTH} characters."
            ),
            visible_if={"param": "part", "values": ["pattern"]},
        ),
        Param(
            "index",
            "int",
            "Word number",
            default=1,
            min=-MAX_WORD_INDEX,
            max=MAX_WORD_INDEX,
            help="1 is the first word, 2 the second; -1 is the last word. Words are split by "
            "spaces.",
            visible_if={"param": "part", "values": ["word"]},
        ),
        Param(
            "strip",
            "bool",
            "Trim spaces first",
            default=True,
            help="Remove spaces at the start and the end of every value before taking the part.",
        ),
        Param(
            "case",
            "select",
            "Case",
            default="upper",
            options=EXTRACT_CASES,
            help="Of the part taken, before the mapping.",
        ),
        Param(
            "mapping",
            "mapping",
            "Map to new values",
            default={},
            optional=True,
            help=(
                "Optional: part → new value, e.g. A → 1, B → 2. If every new value is a "
                "number, the new column holds numbers. Parts not listed become missing. "
                "Empty: the parts themselves, as categories."
            ),
            key_kind="value",
        ),
    )

    def clean(self, table: Table, ctx: StepContext) -> dict[str, Any]:
        raw = self.raw_params if isinstance(self.raw_params, dict) else {}
        column = raw.get("column")
        if isinstance(column, str) and table.kinds.get(column) not in (None, *STRING_KINDS):
            kind = table.kinds[column]
            raise DataError(
                f"Text column: '{column}' holds {KIND_NAMES[kind]}, not text. Choose a text "
                "or category column, or turn it into categories with “Change type” first."
            )
        params = super().clean(table, ctx)
        if params["part"] == "pattern":
            # Spaces count in a pattern: use it as typed (the generic check trims it).
            pattern = raw.get("pattern")
            pattern = pattern if isinstance(pattern, str) and pattern.strip() else ""
            check_pattern(pattern, "Pattern")
            params["pattern"] = pattern
        if params["part"] == "word" and params["index"] == 0:
            raise DataError("Word number: use 1 for the first word or -1 for the last one.")
        target = ctx.target
        if target and params["column"] == target and params["name"] != target:
            raise DataError(
                f"Text column: '{target}' is the target. A column taken from the target gives "
                "the model the answer during training and cannot be computed for new rows. "
                "Choose another column."
            )
        return params

    def run(self, table: Table, params: dict[str, Any], ctx: StepContext) -> StepResult:
        column, name, mapping = params["column"], params["name"], params["mapping"]
        kind = extract_mapping(mapping)[1] if mapping else "categorical"
        # Run the code of the pandas script, so the script and raw-row predictions
        # (frozen.py) give the very same values.
        namespace: dict[str, Any] = {"np": np, "pd": pd, "re": re, "df": table.df[[column]].copy()}
        code = "\n".join(self.to_pandas(params, None, ctx))
        exec(compile(code, "<extract step>", "exec"), namespace)  # noqa: S102 (our code)
        values = storage(namespace["df"][name], kind)
        parts = namespace.get("_extracted", values)
        replaced = name in table.kinds
        result = set_columns(table, {name: (values, kind)})

        present = int(values.notna().sum())
        missing = table.n_rows - present
        summary = f"Took {_part_text(params)} of '{column}' into '{name}': "
        summary += plural(present, "value")
        if not mapping:
            summary += f" in {plural(int(values.nunique()), 'category', 'categories')}"
        if missing:
            summary += f", {fmt_count(missing)} missing"
        if mapping and present:
            if kind == "numeric":
                low, high = float(values.min()), float(values.max())
                summary += (
                    f"; mapped to the number {fmt_number(low)}"
                    if low == high
                    else f"; mapped to numbers {fmt_number(low)}–{fmt_number(high)}"
                )
            else:
                count = int(values.nunique())
                summary += f"; mapped to {plural(count, 'category', 'categories')}"
        summary += "."

        warnings = []
        if mapping:
            unmapped = parts.notna() & ~parts.isin(list(mapping))
            count = int(unmapped.sum())
            if count:
                shown = quoted(parts[unmapped].value_counts().index.tolist(), 5)
                warnings.append(
                    f"{plural(count, 'value')} not in the mapping became missing: {shown}."
                )
            warnings += _case_warnings(params["case"], list(mapping))
        if not present and table.n_rows:
            warnings.append(f"Every value of '{name}' is missing.")
        if replaced and name == ctx.target:
            warnings.append(f"This step replaces the target column '{name}'.")
        details = []
        counts = parts.value_counts()
        if len(counts):
            top = ", ".join(f"'{value[:30]}' ({fmt_count(n)})" for value, n in counts[:5].items())
            more = f" and {fmt_count(len(counts) - 5)} more" if len(counts) > 5 else ""
            details.append({"column": name, "text": f"parts taken: {top}{more}"})
        return StepResult(result, summary, details=details, warnings=warnings)

    def to_pandas(
        self, params: dict[str, Any], fitted: dict[str, Any] | None, ctx: StepContext
    ) -> list[str]:
        params = {**self.defaults(), **params}
        part = params["part"]
        args = [f"df[{params['column']!r}]", repr(part)]
        if part in ("first", "last"):
            args.append(f"length={int(params['length'])!r}")
        elif part == "pattern":
            args.append(f"pattern={params['pattern']!r}")
        else:
            args.append(f"index={int(params['index'])!r}")
        args += [f"strip={bool(params['strip'])!r}", f"case={params['case']!r}"]
        call = f"extract_text({', '.join(args)})"
        target = f"df[{params['name']!r}]"
        lines = [*_code_lines(extract_text), ""]
        if not params["mapping"]:
            return [*lines, f"{target} = {call}"]
        mapping, kind = extract_mapping(params["mapping"])
        pairs = ", ".join(f"{code_value(k)}: {code_value(v)}" for k, v in mapping.items())
        convert = ".astype('float64')" if kind == "numeric" else ""
        return [*lines, f"_extracted = {call}", f"{target} = _extracted.map({{{pairs}}}){convert}"]


def _case_warnings(case: str, keys: list[str]) -> list[str]:
    """A warning for mapping keys that cannot match because of the case of the parts."""
    if case == "keep":
        return []
    change = str.upper if case == "upper" else str.lower
    odd = [key for key in keys if change(key) != key]
    if not odd:
        return []
    verb = "matches" if len(odd) == 1 else "match"
    return [
        f"The parts are in {case} case, so {quoted(odd)} in the mapping never {verb}. "
        "Change the mapping or the case."
    ]
