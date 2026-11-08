"""Preparation steps that remove or reorder rows: duplicates, filters, rows with missing
values and sorting.

They work on all rows, whatever their split: every row keeps its split label (``Table.take``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..table import STRING_KINDS, DataError, Table
from .base import Param, Step, StepContext, StepResult, register
from .columns import (
    as_label,
    cell_value,
    code_value,
    comment,
    fmt_count,
    fmt_number,
    fmt_share,
    from_fitted,
    plural,
    quoted,
    to_fitted,
)

# --------------------------------------------------------------------------- #
# Duplicates
# --------------------------------------------------------------------------- #


@register
class DropDuplicates(Step):
    type = "drop_duplicates"
    label = "Remove duplicates"
    group = "rows"
    description = "Remove rows that repeat another row (in all or in some columns)."
    params = (
        Param(
            "columns",
            "columns",
            "Compare",
            default=[],
            optional=True,
            empty_means="All columns",
            help="Rows count as duplicates if they agree in these columns.",
        ),
        Param(
            "keep",
            "select",
            "Keep",
            default="first",
            options=(("first", "The first occurrence"), ("last", "The last occurrence")),
        ),
    )

    def run(self, table: Table, params: dict[str, Any], ctx: StepContext) -> StepResult:
        columns = params["columns"] or None
        duplicated = table.df.duplicated(subset=columns, keep=params["keep"]).to_numpy()
        removed = int(duplicated.sum())
        if not removed:
            return StepResult(table, "No duplicate rows found.")
        where = f" (same values in {quoted(columns)})" if columns else ""
        return StepResult(
            table.take(~duplicated),
            f"Removed {plural(removed, 'duplicate row')}{where}, "
            f"{fmt_share(removed, table.n_rows)} of the rows.",
        )

    def to_pandas(
        self, params: dict[str, Any], fitted: dict[str, Any] | None, ctx: StepContext
    ) -> list[str]:
        subset = list(params["columns"]) or None
        return [
            f"df = df.drop_duplicates(subset={subset!r}, keep={params['keep']!r})"
            ".reset_index(drop=True)"
        ]


# --------------------------------------------------------------------------- #
# Filter
# --------------------------------------------------------------------------- #

OPERATORS = (
    ("==", "equals"),
    ("!=", "does not equal"),
    (">", "is greater than"),
    (">=", "is at least"),
    ("<", "is less than"),
    ("<=", "is at most"),
    ("between", "is between"),
    ("in", "is one of"),
    ("not_in", "is not one of"),
    ("contains", "contains (ignoring case)"),
    ("is_missing", "is missing"),
    ("not_missing", "is not missing"),
)
SINGLE_VALUE = ("==", "!=", ">", ">=", "<", "<=", "between", "contains")
ORDERING = (">", ">=", "<", "<=", "between")
SYMBOLS = {"==": "=", "!=": "≠", ">": ">", ">=": "≥", "<": "<", "<=": "≤"}


def _typed(params: dict[str, Any]) -> dict[str, Any]:
    """The comparison values of cleaned params in the storage format of the column."""
    kind = params["_kind"]
    return {
        **params,
        "value": from_fitted(params["value"], kind),
        "value2": from_fitted(params["value2"], kind),
        "values": [from_fitted(v, kind) for v in params["values"]],
    }


def _shown(value: Any) -> str:
    if isinstance(value, pd.Timestamp):
        text = value.strftime("%Y-%m-%d %H:%M:%S")
        return text[:-9] if text.endswith(" 00:00:00") else text
    if isinstance(value, float):
        return fmt_number(value)
    return f"'{value}'"


def _condition_text(column: str, params: dict[str, Any]) -> str:
    op = params["operator"]
    if op in SYMBOLS:
        return f"{column} {SYMBOLS[op]} {_shown(params['value'])}"
    if op == "between":
        return f"{column} is between {_shown(params['value'])} and {_shown(params['value2'])}"
    if op in ("in", "not_in"):
        values = ", ".join(_shown(v) for v in params["values"][:5])
        more = (
            f" … ({fmt_count(len(params['values']))} values)" if len(params["values"]) > 5 else ""
        )
        return f"{column} is {'not ' if op == 'not_in' else ''}one of {values}{more}"
    if op == "contains":
        return f"{column} contains {_shown(params['value'])}"
    return f"{column} is {'not ' if op == 'not_missing' else ''}missing"


@register
class FilterRows(Step):
    type = "filter_rows"
    label = "Filter rows"
    group = "rows"
    description = "Keep only the rows that meet a condition."
    params = (
        Param("column", "column", "Column", default=None),
        Param(
            "operator",
            "select",
            "Keep rows where the value",
            default=">=",
            options=OPERATORS,
            help="Rows with a missing value only pass “is missing”.",
        ),
        Param(
            "value",
            "value",
            "Value",
            default=None,
            optional=True,
            visible_if={"param": "operator", "values": list(SINGLE_VALUE)},
        ),
        Param(
            "value2",
            "value",
            "and",
            default=None,
            optional=True,
            visible_if={"param": "operator", "values": ["between"]},
        ),
        Param(
            "values",
            "list",
            "Values",
            default=[],
            optional=True,
            visible_if={"param": "operator", "values": ["in", "not_in"]},
        ),
    )

    def clean(self, table: Table, ctx: StepContext) -> dict[str, Any]:
        params = super().clean(table, ctx)
        column, op = params["column"], params["operator"]
        kind = table.kinds[column]
        strings = kind in STRING_KINDS
        if op == "contains" and not strings:
            raise DataError(
                f"“contains” works on text and category columns; '{column}' holds {kind} values."
            )
        if op in ORDERING and strings:
            raise DataError(
                f"'{column}' holds text; use equals, does not equal, is one of or contains."
            )

        def convert(value: Any, label: str) -> Any:
            # JSON-safe (dates as ISO strings); see _typed for the storage format
            return as_label(value) if strings else to_fitted(cell_value(value, kind, column, label))

        if op in SINGLE_VALUE:
            if params["value"] is None or params["value"] == "":
                raise DataError("Value: enter a value to compare with.")
            params["value"] = convert(params["value"], "Value")
        if op == "between":
            if params["value2"] is None or params["value2"] == "":
                raise DataError("and: enter the upper end of the range.")
            params["value2"] = convert(params["value2"], "and")
            if from_fitted(params["value"], kind) > from_fitted(params["value2"], kind):
                raise DataError("Value: the first value must not be larger than the second.")
        if op in ("in", "not_in"):
            if not params["values"]:
                raise DataError("Values: add at least one value.")
            params["values"] = [convert(v, "Values") for v in params["values"]]
        params["_kind"] = kind
        return params

    def run(self, table: Table, params: dict[str, Any], ctx: StepContext) -> StepResult:
        column = params["column"]
        series = table.df[column]
        params = _typed(params)
        keep = _filter_mask(series, params).to_numpy(dtype=bool)
        kept = int(keep.sum())
        condition = _condition_text(column, params)
        if kept == 0 and table.n_rows:
            raise DataError(
                f"No row meets the condition ({condition}); the filter would remove all "
                f"{fmt_count(table.n_rows)} rows."
            )
        removed = table.n_rows - kept
        return StepResult(
            table.take(keep) if removed else table,
            f"Kept {fmt_count(kept)} of {fmt_count(table.n_rows)} rows where {condition} "
            f"(removed {fmt_count(removed)}).",
        )

    def to_pandas(
        self, params: dict[str, Any], fitted: dict[str, Any] | None, ctx: StepContext
    ) -> list[str]:
        params = _typed(params)
        return [
            f"# Keep rows where {comment(_condition_text(params['column'], params))}",
            f"df = df[{_filter_code(params)}].reset_index(drop=True)",
        ]


def _filter_mask(series: pd.Series, params: dict[str, Any]) -> pd.Series:
    op, value = params["operator"], params["value"]
    if op == "==":
        return series == value
    if op == "!=":
        return (series != value) & series.notna()
    if op == ">":
        return series > value
    if op == ">=":
        return series >= value
    if op == "<":
        return series < value
    if op == "<=":
        return series <= value
    if op == "between":
        return series.between(value, params["value2"])
    if op == "in":
        return series.isin(params["values"])
    if op == "not_in":
        return ~series.isin(params["values"]) & series.notna()
    if op == "contains":
        return series.str.contains(value, case=False, regex=False, na=False)
    if op == "is_missing":
        return series.isna()
    return series.notna()


def _filter_code(params: dict[str, Any]) -> str:
    ref = f"df[{params['column']!r}]"
    op = params["operator"]
    value = code_value(params["value"])
    if op in ("==", ">", ">=", "<", "<="):
        return f"{ref} {op} {value}"
    if op == "!=":
        return f"({ref} != {value}) & {ref}.notna()"
    if op == "between":
        return f"{ref}.between({value}, {code_value(params['value2'])})"
    if op in ("in", "not_in"):
        values = "[" + ", ".join(code_value(v) for v in params["values"]) + "]"
        if op == "in":
            return f"{ref}.isin({values})"
        return f"~{ref}.isin({values}) & {ref}.notna()"
    if op == "contains":
        return f"{ref}.str.contains({value}, case=False, regex=False, na=False)"
    if op == "is_missing":
        return f"{ref}.isna()"
    return f"{ref}.notna()"


# --------------------------------------------------------------------------- #
# Rows with missing values
# --------------------------------------------------------------------------- #


@register
class DropMissing(Step):
    type = "drop_missing"
    label = "Drop rows with missing values"
    group = "rows"
    description = "Remove rows that have missing values in some or all columns."
    params = (
        Param(
            "columns",
            "columns",
            "Columns",
            default=[],
            optional=True,
            empty_means="All columns",
        ),
        Param(
            "how",
            "select",
            "Drop a row if",
            default="any",
            options=(("any", "Any of the columns is missing"), ("all", "All of them are missing")),
        ),
    )

    def run(self, table: Table, params: dict[str, Any], ctx: StepContext) -> StepResult:
        columns = params["columns"] or table.columns
        if not columns:
            return StepResult(table, "There are no columns; nothing changed.")
        missing = table.df[columns].isna()
        drop = (missing.any(axis=1) if params["how"] == "any" else missing.all(axis=1)).to_numpy()
        removed = int(drop.sum())
        if params["columns"] and len(columns) == 1:
            where = f"a missing {columns[0]}"
        elif params["how"] == "any":
            where = f"a missing value in any of {plural(len(columns), 'column')}"
        else:
            where = f"all of {plural(len(columns), 'column')} missing"
        if removed and removed == table.n_rows:
            raise DataError(
                f"Every row has {where}, so no row would be left. Choose fewer columns or fill "
                "the missing values first."
            )
        if not removed:
            return StepResult(table, f"No row has {where}; nothing removed.")
        return StepResult(
            table.take(~drop),
            f"Removed {plural(removed, 'row')} ({fmt_share(removed, table.n_rows)}) with {where}.",
        )

    def to_pandas(
        self, params: dict[str, Any], fitted: dict[str, Any] | None, ctx: StepContext
    ) -> list[str]:
        subset = list(params["columns"]) or None
        return [f"df = df.dropna(subset={subset!r}, how={params['how']!r}).reset_index(drop=True)"]


# --------------------------------------------------------------------------- #
# Sort
# --------------------------------------------------------------------------- #


@register
class SortRows(Step):
    type = "sort_rows"
    label = "Sort rows"
    group = "rows"
    description = (
        "Sort the rows by a column, e.g. by date before filling gaps with the previous value."
    )
    params = (
        Param("column", "column", "Sort by", default=None),
        Param("descending", "bool", "Largest / newest first", default=False),
    )

    def run(self, table: Table, params: dict[str, Any], ctx: StepContext) -> StepResult:
        column = params["column"]
        series = table.df[column]
        order = series.sort_values(
            ascending=not params["descending"], kind="stable", na_position="last"
        ).index.to_numpy()
        kind = table.kinds[column]
        if kind == "datetime":
            direction = "newest first" if params["descending"] else "oldest first"
        elif kind in STRING_KINDS:
            direction = "Z to A" if params["descending"] else "A to Z"
        else:
            direction = "largest first" if params["descending"] else "smallest first"
        changed = not np.array_equal(order, np.arange(table.n_rows))
        summary = f"Sorted {plural(table.n_rows, 'row')} by {column} ({direction})"
        summary += "; missing values last." if series.isna().any() else "."
        if not changed:
            summary += " The order did not change."
        return StepResult(table.take(order) if changed else table, summary)

    def to_pandas(
        self, params: dict[str, Any], fitted: dict[str, Any] | None, ctx: StepContext
    ) -> list[str]:
        return [
            f"df = df.sort_values({params['column']!r}, ascending={not params['descending']!r}, "
            "kind='stable', na_position='last').reset_index(drop=True)"
        ]
