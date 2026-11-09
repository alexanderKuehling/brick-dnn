"""Preparation steps that work on whole columns: drop, keep, rename, change the type,
compute a new column with a formula and replace values.

The first section holds small helpers that the other basic step modules (combine, rows,
missing) use as well: number formatting for summaries, converting user-entered values to
the storage format of a column (see :mod:`brickdnn.data.table`) and writing them as code.
"""

from __future__ import annotations

import ast
import math
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..table import KINDS, NUMERIC_KINDS, STRING_KINDS, DataError, Table, normalize
from .base import Param, Step, StepContext, StepResult, register
from .text import fmt_count, fmt_number, fmt_share, plural  # noqa: F401

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
# Formula
# --------------------------------------------------------------------------- #

MAX_FORMULA_LENGTH = 1000
MAX_FORMULA_NODES = 300
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
}
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
    "Allowed are numbers, column names (`my column` in backticks if needed), + - * / // % **, "
    "comparisons, and, or, not, and the functions " + ", ".join(FORMULA_FUNCTIONS) + "."
)
_NOT_ALLOWED = {
    ast.Attribute: "Attribute access (a.b) is not allowed in formulas.",
    ast.Subscript: "Indexing (a[0]) is not allowed in formulas.",
    ast.Lambda: "Functions (lambda) cannot be defined in formulas.",
    ast.IfExp: "Use where(condition, a, b) instead of 'a if condition else b'.",
    ast.JoinedStr: "f-strings are not allowed in formulas.",
}
_EVAL_GLOBALS: dict[str, Any] = {"__builtins__": {}}


class _Expr:
    """A compiled sub-expression: its code, value type and whether it depends on a column.

    ``kind`` is "num", "bool" (0/1, also a number) or "str". Constant sub-expressions
    (``value`` set) are computed while compiling, with numpy semantics.
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


class FormulaCompiler:
    """Checks a formula against a whitelist and turns it into pandas code.

    The generated code only contains ``df[<column name>]``, number and string literals,
    operators and calls of fixed numpy/pandas functions, so evaluating it is safe; the step
    and the exported script run the very same code.
    """

    def __init__(self, kinds: dict[str, str]):
        self.kinds = kinds
        self.names: dict[str, str] = {}
        self.columns: list[str] = []

    def compile(self, expression: str) -> _Expr:
        if len(expression) > MAX_FORMULA_LENGTH:
            raise DataError(f"Expression: at most {MAX_FORMULA_LENGTH:,} characters.")
        text, self.names = _replace_backticks(expression)
        try:
            tree = ast.parse(text.strip(), mode="eval")
        except SyntaxError as exc:
            where = f" (at character {exc.offset})" if exc.offset else ""
            raise DataError(f"Expression: this is not a valid formula{where}.") from None
        except (RecursionError, MemoryError, ValueError):
            raise DataError("Expression: the formula is too complex.") from None
        if sum(1 for _ in ast.walk(tree)) > MAX_FORMULA_NODES:
            raise DataError("Expression: the formula is too long; split it into several steps.")
        try:
            return self.visit(tree.body)
        except RecursionError:
            raise DataError("Expression: the formula is too complex.") from None

    # ---------------------------------------------------------------- nodes

    def visit(self, node: ast.AST) -> _Expr:
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
                "work with text."
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
                f"Expression: cannot compare text with numbers ({left.code} {op} {right.code})."
            )
        if left.kind == "str" and op not in ("==", "!="):
            raise DataError(f"Expression: text can only be compared with == and !=, not {op}.")
        if not left.series and not right.series:
            return _Expr.constant(float(_PY_COMPARE[op](left.value, right.value)), "bool")
        return _Expr(self._masked(f"{left.code} {op} {right.code}", left, right), "bool", True)

    def _compare_chain(self, node: ast.Compare) -> _Expr:
        operands = [self.visit(node.left), *(self.visit(c) for c in node.comparators)]
        parts = []
        for i, op_node in enumerate(node.ops):
            op = _COMPARE.get(type(op_node))
            if op is None:
                raise DataError(f"Expression: 'in' and 'is' are not allowed. {_ALLOWED_TEXT}")
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
            name = node.func.id if isinstance(node.func, ast.Name) else "this function"
            raise DataError(f"Expression: {name}(…) is not allowed. {_ALLOWED_TEXT}")
        name = node.func.id
        if node.keywords or any(isinstance(a, ast.Starred) for a in node.args):
            raise DataError(f"Expression: {name}(…) takes plain arguments only.")
        low, high = FORMULA_FUNCTIONS[name]
        if not low <= len(node.args) <= high:
            count = f"{low}" if low == high else f"{low} or {high}"
            raise DataError(f"Expression: {name}(…) takes {count} argument(s).")
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
            raise DataError("Expression: where(…, a, b) needs a and b of the same type.")
        kind = "str" if a.kind == "str" else ("bool" if a.kind == b.kind == "bool" else "num")
        if not condition.series:
            chosen_value = a if condition.value and not math.isnan(condition.value) else b
            if chosen_value.kind != kind:
                return _Expr(chosen_value.code, kind, chosen_value.series, chosen_value.value)
            return chosen_value
        code = f"pd.Series(np.where({condition.code} != 0, {a.code}, {b.code}), index=df.index)"
        code += f".where({condition.code}.notna())"
        return _Expr(code, kind, True)


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
    """Evaluate code made by :func:`formula_code` (only names: df, np, pd)."""
    with np.errstate(all="ignore"):
        return eval(code, _EVAL_GLOBALS, {"df": df, "np": np, "pd": pd})  # noqa: S307


@register
class Formula(Step):
    type = "formula"
    label = "Formula"
    group = "columns"
    description = "Compute a new column from other columns, e.g. weight / height ** 2."
    params = (
        Param("name", "string", "New column", default="", help="An existing column is replaced."),
        Param(
            "expression",
            "string",
            "Formula",
            default="",
            help="Columns by name (`my column` in backticks), numbers, + - * / // % **, "
            "comparisons (give 1/0), and, or, not, and log, log1p, exp, sqrt, abs, round, "
            "floor, ceil, min(a, b), max(a, b), clip(x, lo, hi), where(cond, a, b), isnull(x). "
            "A missing input gives a missing result.",
        ),
    )

    def clean(self, table: Table, ctx: StepContext) -> dict[str, Any]:
        params = super().clean(table, ctx)
        compiler = FormulaCompiler(table.kinds)
        expr = compiler.compile(params["expression"])
        params["_code"] = formula_code(expr)
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
            notes.append(
                f"{plural(invalid, 'value')} could not be computed (e.g. division by zero or "
                "log of a negative number)"
            )
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
        return [
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
