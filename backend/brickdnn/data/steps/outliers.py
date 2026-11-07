"""Preparation steps: finding and handling outliers, clipping values to a range.

This module also holds small helpers (number formatting, column selection, adding columns)
used by the other modules of the advanced steps (transform, encode, dates, target,
selection).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..table import DataError, Table, normalize
from .base import Param, Step, StepContext, StepResult, register
from .text import fmt_count, fmt_number, fmt_share, plural  # noqa: F401

# --------------------------------------------------------------------------- #
# Helpers shared by the step modules
# --------------------------------------------------------------------------- #


def percentile_name(q: float) -> str:
    """Quantile ``0.01`` -> ``"1st"``, ``0.99`` -> ``"99th"``, ``0.025`` -> ``"2.5th"``."""
    p = round(q * 100, 6)
    if not float(p).is_integer():
        return f"{fmt_number(p)}th"
    p = int(p)
    suffix = "th" if 10 <= p % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(p % 10, "th")
    return f"{p}{suffix}"


def name_list(names: Sequence[str], limit: int = 3) -> str:
    """``'a', 'b' and 'c'`` or ``'a', 'b', 'c' and 4 more``."""
    quoted = [f"'{n}'" for n in names]
    if len(quoted) <= limit:
        if len(quoted) <= 1:
            return "".join(quoted)
        return ", ".join(quoted[:-1]) + " and " + quoted[-1]
    return ", ".join(quoted[:limit]) + f" and {len(quoted) - limit} more"


def chosen_columns(
    table: Table, columns: Sequence[str], ctx: StepContext, kinds: Iterable[str]
) -> list[str]:
    """The columns a step works on: the listed ones, or (empty list) all columns of
    ``kinds`` except the target."""
    if columns:
        return list(columns)
    return table.columns_of(kinds, exclude=[ctx.target] if ctx.target else [])


def train_values(table: Table, column: str) -> np.ndarray:
    """The non-missing values of a numeric column in the training rows, as float64."""
    values = table.df[column].to_numpy(dtype=float)[table.train_mask()]
    return values[np.isfinite(values)]


def plain(x: Any) -> float:
    """A finite Python float for ``fitted`` (no numpy types, no NaN)."""
    value = float(x)
    if not math.isfinite(value):
        raise DataError("Internal error: a learned value is not a finite number.")
    return value


def code_float(x: float | None) -> str:
    """A float literal for generated code."""
    return "None" if x is None else repr(float(x))


def check_new_names(table: Table, names: Iterable[str], replaced: Iterable[str] = ()) -> None:
    """Raise a :class:`DataError` if a new column name is already taken."""
    free = set(replaced)
    seen: set[str] = set()
    for name in names:
        if (name in table.kinds and name not in free) or name in seen:
            raise DataError(
                f"There is already a column '{name}'. Rename or drop it first, or choose "
                "another name."
            )
        seen.add(name)


def insert_after(
    table: Table, after: str, columns: Sequence[tuple[str, Any, str]], *, drop_after: bool = False
) -> Table:
    """Add ``(name, values, kind)`` columns right after ``after``, in order; with
    ``drop_after`` the column ``after`` is removed (the new columns take its place).

    All new columns are added with one ``concat`` (inserting them one by one fragments the
    frame and gets slow with hundreds of one-hot columns)."""
    names = [name for name, _, _ in columns]
    clash = any(name in table.kinds and not (drop_after and name == after) for name in names)
    if clash or len(set(names)) != len(names) or after not in table.kinds:
        anchor = after
        for name, values, kind in columns:
            table = table.with_column(name, values, kind, after=anchor)
            anchor = name
        return table.drop([after]) if drop_after else table
    block: dict[str, pd.Series] = {}
    kinds = dict(table.kinds)
    for name, values, kind in columns:
        series = values if isinstance(values, pd.Series) else pd.Series(values)
        series = series.reset_index(drop=True)
        if len(series) != table.n_rows:
            raise DataError("Internal error: a new column does not match the number of rows.")
        block[name] = normalize(series.rename(name), kind)
        kinds[name] = kind
    position = table.columns.index(after) + 1
    left = (
        table.columns[: position - 1] if drop_after and after in names else table.columns[:position]
    )
    parts = [
        table.df[left],
        pd.DataFrame(block, index=table.df.index),
        table.df[table.columns[position:]],
    ]
    df = pd.concat(parts, axis=1)
    if drop_after and after not in names:
        df = df.drop(columns=[after])
        kinds.pop(after, None)
    kinds = {c: kinds[c] for c in df.columns}
    return Table(df, kinds, table.split.copy() if table.split is not None else None)


def code_insert_after(after: str, columns: Sequence[tuple[str, str]]) -> list[str]:
    """Code lines that insert ``(name, expression)`` columns right after ``after``."""
    return [
        f"df.insert(df.columns.get_loc({after!r}) + {i}, {name!r}, {expr})"
        for i, (name, expr) in enumerate(columns, start=1)
    ]


def no_columns_result(table: Table, what: str) -> StepResult:
    return StepResult(table, f"There are no {what} to work on; nothing changed.")


# --------------------------------------------------------------------------- #
# Outliers
# --------------------------------------------------------------------------- #

NUMERIC = ("numeric",)
COLUMNS_PARAM = Param(
    "columns",
    "columns",
    "Columns",
    default=[],
    optional=True,
    column_kinds=NUMERIC,
    empty_means="All numeric columns",
)
LOWER_Q_HELP = "Share of the training values below the lower bound, e.g. 0.01 = 1st percentile."
UPPER_Q_HELP = "Share of the training values below the upper bound, e.g. 0.99 = 99th percentile."
# Modified z-score: 0.6745 · (x − median) / MAD (Iglewicz & Hoaglin). If the MAD is 0, the
# mean absolute deviation · 1.253314 is used instead.
MAD_FACTOR = 0.6745
MEAN_AD_FACTOR = 1.253314


def _clean_quantiles(params: dict[str, Any]) -> None:
    if params["lower_q"] >= params["upper_q"]:
        raise DataError("The lower quantile must be smaller than the upper quantile.")


def outlier_bounds(values: np.ndarray, params: dict[str, Any]) -> tuple[float, float, str | None]:
    """``(lower, upper, note)`` for the outlier rule in ``params`` (``method`` and its
    setting), learned from ``values``. ``note`` explains degenerate cases."""
    method = params["method"]
    note = None
    if method == "iqr":
        q1, q3 = np.quantile(values, [0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            note = "the IQR is 0 (most values are equal), so every other value is an outlier"
        k = params["iqr_k"]
        return q1 - k * iqr, q3 + k * iqr, note
    if method == "zscore":
        mean, std = values.mean(), values.std()
        return mean - params["z"] * std, mean + params["z"] * std, note
    if method == "mad":
        median = np.median(values)
        deviation = np.abs(values - median)
        mad = np.median(deviation)
        if mad > 0:
            scale = mad / MAD_FACTOR
        else:
            scale = MEAN_AD_FACTOR * deviation.mean()
            note = "the MAD is 0, so the mean absolute deviation is used"
        return median - params["mad_z"] * scale, median + params["mad_z"] * scale, note
    lower, upper = np.quantile(values, [params["lower_q"], params["upper_q"]])
    return lower, upper, note


def rule_text(params: dict[str, Any]) -> str:
    method = params["method"]
    if method == "iqr":
        return f"IQR rule, k = {fmt_number(params['iqr_k'])}"
    if method == "zscore":
        return f"|z| > {fmt_number(params['z'])}"
    if method == "mad":
        return f"modified z-score > {fmt_number(params['mad_z'])}"
    return (
        f"outside the {percentile_name(params['lower_q'])}–"
        f"{percentile_name(params['upper_q'])} percentile"
    )


def _outside(series: pd.Series, lower: float, upper: float) -> pd.Series:
    return (series < lower) | (series > upper)


def _code_outside(column: str, lower: float, upper: float) -> str:
    ref = f"df[{column!r}]"
    return f"({ref} < {code_float(lower)}) | ({ref} > {code_float(upper)})"


@register
class OutliersStep(Step):
    type = "outliers"
    label = "Handle outliers"
    group = "outliers"
    description = (
        "Find unusual values with the IQR rule, z-scores, MAD or quantiles of the training "
        "rows, then clip, remove, blank or flag them."
    )
    fitted = True
    params = (
        COLUMNS_PARAM,
        Param(
            "method",
            "select",
            "Find outliers with",
            default="iqr",
            options=(
                ("iqr", "IQR rule (Tukey fences)"),
                ("zscore", "Z-score"),
                ("mad", "Median absolute deviation (MAD)"),
                ("quantile", "Quantiles"),
            ),
            help=(
                "IQR: outside Q1 − k·IQR … Q3 + k·IQR, robust and the usual choice. Z-score: "
                "more than z standard deviations from the mean, for roughly normal data. MAD: "
                "robust z-score around the median. Quantiles: a fixed share at each end."
            ),
        ),
        Param(
            "iqr_k",
            "number",
            "k (IQR multiplier)",
            default=1.5,
            min=0,
            max=100,
            step=0.5,
            visible_if={"param": "method", "values": ["iqr"]},
            help=(
                "Values below Q1 − k·IQR or above Q3 + k·IQR are outliers (IQR = Q3 − Q1, the "
                "range of the middle half). 1.5 is Tukey's rule, 3 finds only extreme values."
            ),
        ),
        Param(
            "z",
            "number",
            "Z-score limit",
            default=3,
            min=0,
            max=100,
            step=0.5,
            visible_if={"param": "method", "values": ["zscore"]},
            help="Values more than this many standard deviations from the mean are outliers.",
        ),
        Param(
            "mad_z",
            "number",
            "Modified z-score limit",
            default=3.5,
            min=0,
            max=100,
            step=0.5,
            visible_if={"param": "method", "values": ["mad"]},
            help=(
                "Modified z-score = 0.6745 · (x − median) / MAD, where MAD is the median "
                "distance to the median. Outliers do not inflate it; 3.5 is the usual limit."
            ),
        ),
        Param(
            "lower_q",
            "number",
            "Lower quantile",
            default=0.01,
            min=0,
            max=0.5,
            step=0.01,
            visible_if={"param": "method", "values": ["quantile"]},
            help=LOWER_Q_HELP,
        ),
        Param(
            "upper_q",
            "number",
            "Upper quantile",
            default=0.99,
            min=0.5,
            max=1,
            step=0.01,
            visible_if={"param": "method", "values": ["quantile"]},
            help=UPPER_Q_HELP,
        ),
        Param(
            "action",
            "select",
            "Then",
            default="clip",
            options=(
                ("clip", "Clip them to the bounds"),
                ("remove", "Remove their rows"),
                ("set_missing", "Set them to missing"),
                ("flag", "Add a column that marks them"),
            ),
            help=(
                "Flag adds a 0/1 column '<column>_outlier' and keeps the values. Set to "
                "missing lets a later 'Fill missing values' step replace them."
            ),
        ),
        Param(
            "remove_from",
            "select",
            "Remove from",
            default="train",
            options=(("train", "Training rows only"), ("all", "All rows")),
            visible_if={"param": "action", "values": ["remove"]},
            help=(
                "Removing rows from the validation and test data makes the results look "
                "better than they will be on new data. Only do it for values that are errors."
            ),
        ),
    )

    def clean(self, table: Table, ctx: StepContext) -> dict[str, Any]:
        params = super().clean(table, ctx)
        if params["method"] == "quantile":
            _clean_quantiles(params)
        return params

    def run(self, table: Table, params: dict[str, Any], ctx: StepContext) -> StepResult:
        columns = chosen_columns(table, params["columns"], ctx, NUMERIC)
        if not columns:
            return no_columns_result(table, "numeric columns")
        action = params["action"]
        rule = rule_text(params)
        train = table.train_mask()
        bounds: dict[str, list[float]] = {}
        details: list[dict[str, Any]] = []
        warnings: list[str] = []
        masks: dict[str, pd.Series] = {}
        for column in columns:
            values = train_values(table, column)
            if len(values) == 0:
                warnings.append(f"'{column}' has no values in the training rows; left unchanged.")
                continue
            lower, upper, note = outlier_bounds(values, params)
            lower, upper = plain(lower), plain(upper)
            bounds[column] = [lower, upper]
            mask = _outside(table.df[column], lower, upper)
            masks[column] = mask
            train_only = action == "remove" and params["remove_from"] == "train"
            count = int(mask[train].sum()) if train_only else int(mask.sum())
            details.append(
                {
                    "column": column,
                    "text": f"bounds {fmt_number(lower)} … {fmt_number(upper)} · "
                    f"{plural(count, 'outlier')}",
                }
            )
            if note:
                warnings.append(f"'{column}': {note}.")

        fitted = {"bounds": bounds}
        n_cols = plural(len(bounds), "column")
        cells = sum(int(table.df[c].notna().sum()) for c in bounds)
        total = sum(int(m.sum()) for m in masks.values())

        if action == "remove":
            any_outlier = np.zeros(table.n_rows, dtype=bool)
            for mask in masks.values():
                any_outlier |= mask.to_numpy()
            if params["remove_from"] == "train":
                any_outlier &= train
                base, where = int(train.sum()), "training row"
            else:
                base, where = table.n_rows, "row"
                warnings.append(
                    "Rows were also removed from the validation and test data, so the "
                    "evaluation no longer shows how the model does on unusual values."
                )
            removed = int(any_outlier.sum())
            summary = (
                f"Removed {plural(removed, where)} ({fmt_share(removed, base)}) with "
                f"outliers in {n_cols} ({rule})."
            )
            if removed == 0:
                summary = f"No outliers in {n_cols} ({rule}); no rows removed."
            return StepResult(table.take(~any_outlier), summary, details, fitted, warnings)

        result = table
        for column, (lower, upper) in bounds.items():
            series = table.df[column]
            if action == "clip":
                result = result.with_column(column, series.clip(lower, upper), "numeric")
            elif action == "set_missing":
                result = result.with_column(column, series.mask(masks[column]), "numeric")
        if action == "flag":
            new = [f"{c}_outlier" for c in bounds]
            check_new_names(table, new)
            for column in bounds:
                flag = masks[column].astype(float)
                result = insert_after(result, column, [(f"{column}_outlier", flag, "boolean")])
        what = f"{plural(total, 'outlier')} ({fmt_share(total, cells)})"
        if total == 0:
            summary = f"No outliers in {n_cols} ({rule})."
        elif action == "clip":
            summary = f"Clipped {what} in {n_cols} to the bounds ({rule})."
        elif action == "set_missing":
            summary = f"Set {what} in {n_cols} to missing ({rule})."
        else:
            summary = f"Flagged {what} in {plural(len(bounds), 'new column')} ({rule})."
        return StepResult(result, summary, details, fitted, warnings)

    def to_pandas(
        self, params: dict[str, Any], fitted: dict[str, Any] | None, ctx: StepContext
    ) -> list[str]:
        params = {**self.defaults(), **params}
        bounds = (fitted or {}).get("bounds", {})
        action = params["action"]
        lines: list[str] = []
        if action == "remove":
            if params["remove_from"] == "train" or not bounds:
                return ["# Rows with outliers were only removed from the training rows."]
            condition = " | ".join(
                f"({_code_outside(c, lo, hi)})" for c, (lo, hi) in bounds.items()
            )
            lines.append(f"df = df[~({condition})].reset_index(drop=True)")
            return lines
        for column, (lower, upper) in bounds.items():
            ref = f"df[{column!r}]"
            if action == "clip":
                lines.append(f"{ref} = {ref}.clip({code_float(lower)}, {code_float(upper)})")
            elif action == "set_missing":
                lines.append(f"{ref} = {ref}.mask({_code_outside(column, lower, upper)})")
            else:
                expr = f"({_code_outside(column, lower, upper)}).astype(float)"
                lines += code_insert_after(column, [(f"{column}_outlier", expr)])
        return lines or ["# Nothing to do: no numeric columns."]


# --------------------------------------------------------------------------- #
# Clip
# --------------------------------------------------------------------------- #


@register
class ClipStep(Step):
    type = "clip"
    label = "Clip values"
    group = "outliers"
    description = "Limit values to a range, e.g. the 1st to 99th percentile of the training rows."
    fitted = True
    params = (
        COLUMNS_PARAM,
        Param(
            "mode",
            "select",
            "Bounds",
            default="quantiles",
            options=(
                ("quantiles", "Quantiles"),
                ("values", "Fixed values"),
                ("std", "Mean ± k standard deviations"),
            ),
            help=(
                "Values below the lower bound are set to the lower bound, values above the "
                "upper bound to the upper bound (winsorising). Rows are kept."
            ),
        ),
        Param(
            "lower_q",
            "number",
            "Lower quantile",
            default=0.01,
            min=0,
            max=0.5,
            step=0.01,
            visible_if={"param": "mode", "values": ["quantiles"]},
            help=LOWER_Q_HELP,
        ),
        Param(
            "upper_q",
            "number",
            "Upper quantile",
            default=0.99,
            min=0.5,
            max=1,
            step=0.01,
            visible_if={"param": "mode", "values": ["quantiles"]},
            help=UPPER_Q_HELP,
        ),
        Param(
            "lower",
            "number",
            "Lower bound",
            default=None,
            optional=True,
            visible_if={"param": "mode", "values": ["values"]},
            help="Leave empty for no lower bound.",
        ),
        Param(
            "upper",
            "number",
            "Upper bound",
            default=None,
            optional=True,
            visible_if={"param": "mode", "values": ["values"]},
            help="Leave empty for no upper bound.",
        ),
        Param(
            "k",
            "number",
            "k (standard deviations)",
            default=3,
            min=0,
            max=100,
            step=0.5,
            visible_if={"param": "mode", "values": ["std"]},
            help="Bounds are mean − k·std and mean + k·std of the training values.",
        ),
    )

    def clean(self, table: Table, ctx: StepContext) -> dict[str, Any]:
        params = super().clean(table, ctx)
        if params["mode"] == "quantiles":
            _clean_quantiles(params)
        elif params["mode"] == "values":
            lower, upper = params["lower"], params["upper"]
            if lower is None and upper is None:
                raise DataError("Enter a lower bound, an upper bound or both.")
            if lower is not None and upper is not None and lower > upper:
                raise DataError("The lower bound must not be larger than the upper bound.")
        return params

    @staticmethod
    def _range_text(params: dict[str, Any]) -> str:
        mode = params["mode"]
        if mode == "quantiles":
            return (
                f"the {percentile_name(params['lower_q'])}–"
                f"{percentile_name(params['upper_q'])} percentile"
            )
        if mode == "std":
            return f"mean ± {fmt_number(params['k'])} standard deviations"
        lower, upper = params["lower"], params["upper"]
        if lower is None:
            return f"at most {fmt_number(upper)}"
        if upper is None:
            return f"at least {fmt_number(lower)}"
        return f"{fmt_number(lower)} … {fmt_number(upper)}"

    def run(self, table: Table, params: dict[str, Any], ctx: StepContext) -> StepResult:
        columns = chosen_columns(table, params["columns"], ctx, NUMERIC)
        if not columns:
            return no_columns_result(table, "numeric columns")
        mode = params["mode"]
        bounds: dict[str, list[float | None]] = {}
        details: list[dict[str, Any]] = []
        warnings: list[str] = []
        result = table
        clipped = cells = 0
        for column in columns:
            if mode == "values":
                lower, upper = params["lower"], params["upper"]
            else:
                values = train_values(table, column)
                if len(values) == 0:
                    warnings.append(
                        f"'{column}' has no values in the training rows; left unchanged."
                    )
                    continue
                if mode == "quantiles":
                    lower, upper = np.quantile(values, [params["lower_q"], params["upper_q"]])
                else:
                    mean, std = values.mean(), values.std()
                    lower, upper = mean - params["k"] * std, mean + params["k"] * std
                lower, upper = plain(lower), plain(upper)
            series = table.df[column]
            outside = ((series < lower) if lower is not None else False) | (
                (series > upper) if upper is not None else False
            )
            count = int(np.sum(outside))
            clipped += count
            cells += int(series.notna().sum())
            bounds[column] = [lower, upper]
            result = result.with_column(column, series.clip(lower, upper), "numeric")
            range_text = (
                f"{fmt_number(lower) if lower is not None else '-∞'} … "
                f"{fmt_number(upper) if upper is not None else '∞'}"
            )
            details.append(
                {"column": column, "text": f"{range_text} · {plural(count, 'value')} clipped"}
            )
        n_cols = plural(len(bounds), "column")
        range_text = self._range_text(params)
        if clipped:
            summary = (
                f"Clipped {plural(clipped, 'value')} ({fmt_share(clipped, cells)}) in "
                f"{n_cols} to {range_text}."
            )
        else:
            summary = f"No values to clip in {n_cols} ({range_text})."
        # Fixed values are not learned, but the bounds per column are what the script needs.
        return StepResult(result, summary, details, {"bounds": bounds}, warnings)

    def to_pandas(
        self, params: dict[str, Any], fitted: dict[str, Any] | None, ctx: StepContext
    ) -> list[str]:
        lines = []
        for column, (lower, upper) in (fitted or {}).get("bounds", {}).items():
            ref = f"df[{column!r}]"
            lines.append(f"{ref} = {ref}.clip({code_float(lower)}, {code_float(upper)})")
        return lines or ["# Nothing to clip."]
