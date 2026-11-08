"""Preparation steps: dates and times."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from ..table import DataError, Table
from .base import Param, Step, StepContext, StepResult, register
from .outliers import check_new_names, chosen_columns, insert_after, name_list, plural

PART_OPTIONS = (
    ("year", "Year"),
    ("quarter", "Quarter"),
    ("month", "Month"),
    ("day", "Day"),
    ("weekday", "Weekday"),
    ("dayofyear", "Day of year"),
    ("hour", "Hour"),
    ("minute", "Minute"),
    ("is_weekend", "Weekend (0/1)"),
    ("timestamp", "Timestamp (seconds since 1970)"),
)
PART_NAMES = {"dayofyear": "day of year", "is_weekend": "weekend"}
DT_PARTS = ("year", "quarter", "month", "day", "weekday", "dayofyear", "hour", "minute")
# Cyclical parts: value -> angle 2π · (value − offset) / period.
CYCLES = {"month": (1, 12), "weekday": (0, 7), "hour": (0, 24), "dayofyear": (1, 365.25)}


def _part(series: pd.Series, part: str) -> pd.Series:
    """The same operations as :func:`_code_part`."""
    if part in DT_PARTS:
        return getattr(series.dt, part).astype(float)
    if part == "is_weekend":
        return (series.dt.weekday >= 5).astype(float).where(series.notna())
    return (series - pd.Timestamp(0)).dt.total_seconds()


def _code_part(ref: str, part: str) -> str:
    if part in DT_PARTS:
        return f"{ref}.dt.{part}.astype(float)"
    if part == "is_weekend":
        return f"({ref}.dt.weekday >= 5).astype(float).where({ref}.notna())"
    return f"({ref} - pd.Timestamp(0)).dt.total_seconds()"


def _cycle(series: pd.Series, part: str, func: Callable[[Any], Any]) -> pd.Series:
    offset, period = CYCLES[part]
    return func(2 * np.pi * (getattr(series.dt, part) - offset) / period)


def _code_cycle(ref: str, part: str, func: str) -> str:
    offset, period = CYCLES[part]
    shifted = f"({ref}.dt.{part} - {offset})" if offset else f"{ref}.dt.{part}"
    return f"np.{func}(2 * np.pi * {shifted} / {period})"


def _new_parts(params: dict[str, Any]) -> list[tuple[str, str, str]]:
    """``(suffix, part, how)`` of the new columns of every date column; ``how`` is "value",
    "sin" or "cos"."""
    result = []
    for part in params["parts"]:
        if params["cyclical"] and part in CYCLES:
            result += [(f"_{part}_sin", part, "sin"), (f"_{part}_cos", part, "cos")]
        else:
            result.append((f"_{part}", part, "value"))
    return result


@register
class DatetimeFeaturesStep(Step):
    type = "datetime_features"
    label = "Date parts"
    group = "datetime"
    description = "Split dates into year, month, weekday … columns."
    params = (
        Param(
            "columns",
            "columns",
            "Columns",
            default=[],
            optional=True,
            column_kinds=("datetime",),
            empty_means="All date columns",
        ),
        Param(
            "parts",
            "multiselect",
            "Parts",
            default=["year", "month", "weekday"],
            options=PART_OPTIONS,
            help=(
                "Each part becomes a number column '<column>_<part>'. Weekday: Monday = 0 … "
                "Sunday = 6."
            ),
        ),
        Param(
            "cyclical",
            "bool",
            "Cyclical encoding (sin/cos)",
            default=False,
            help=(
                "Month, weekday, hour and day of year become a sin and a cos column instead "
                "of one number, so December is next to January and Sunday next to Monday."
            ),
        ),
        Param("drop_original", "bool", "Drop the original column", default=True),
    )

    def clean(self, table: Table, ctx: StepContext) -> dict[str, Any]:
        params = super().clean(table, ctx)
        if not params["parts"]:
            raise DataError("Parts: choose at least one part.")
        return params

    def run(self, table: Table, params: dict[str, Any], ctx: StepContext) -> StepResult:
        columns = chosen_columns(table, params["columns"], ctx, ("datetime",))
        if not columns:
            return StepResult(table, "There are no date columns; nothing changed.")
        drop = params["drop_original"]
        parts = _new_parts(params)
        new_names = [c + suffix for c in columns for suffix, _, _ in parts]
        check_new_names(table, new_names)
        result = table
        details: list[dict[str, Any]] = []
        for column in columns:
            series = table.df[column]
            new = []
            for suffix, part, how in parts:
                name = column + suffix
                if how == "value":
                    kind = "boolean" if part == "is_weekend" else "numeric"
                    new.append((name, _part(series, part), kind))
                else:
                    new.append((name, _cycle(series, part, getattr(np, how)), "numeric"))
            result = insert_after(result, column, new, drop_after=drop)
            missing = int(series.isna().sum())
            text = f"{plural(len(new), 'new column')}"
            if missing:
                text += f" · {plural(missing, 'missing date')}, their parts are missing too"
            details.append({"column": column, "text": text})
        originals = "dropped the originals" if drop else "kept the originals"
        if len(columns) == 1:
            originals = "dropped the original" if drop else "kept the original"
        part_names = ", ".join(PART_NAMES.get(p, p) for p in params["parts"])
        summary = (
            f"Added {plural(len(new_names), 'column')} ({part_names}) from "
            f"{plural(len(columns), 'date column')} ({name_list(columns)}) and {originals}."
        )
        return StepResult(result, summary, details)

    def to_pandas(
        self, params: dict[str, Any], fitted: dict[str, Any] | None, ctx: StepContext
    ) -> list[str]:
        params = {**self.defaults(), **params}
        if params["columns"]:
            return [line for c in params["columns"] for line in _code_lines(c, params)]
        # All date columns except the target, as in the preparation.
        names = "df.select_dtypes('datetime').columns"
        if ctx.target:
            names = f"{names}.drop({ctx.target!r}, errors='ignore')"
        lines = [f"for name in list({names}):"]
        return lines + ["    " + line for line in _code_lines(None, params)]


def _code_lines(column: str | None, params: dict[str, Any]) -> list[str]:
    """Code for one date column; ``None``: for the column named by the variable ``name``."""
    col = repr(column) if column is not None else "name"
    ref = f"df[{col}]"
    lines = []
    for i, (suffix, part, how) in enumerate(_new_parts(params), start=1):
        expr = _code_part(ref, part) if how == "value" else _code_cycle(ref, part, how)
        new_name = repr(column + suffix) if column is not None else f"name + {suffix!r}"
        lines.append(f"df.insert(df.columns.get_loc({col}) + {i}, {new_name}, {expr})")
    if params["drop_original"]:
        lines.append(f"df = df.drop(columns=[{col}])")
    return lines
