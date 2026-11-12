"""Preparation steps that build new features from several columns: row-wise combinations
(sum, mean, count of flags set …), pairwise interactions (a × b, a / b …) and group
statistics (the mean amount of a merchant category, learned from the training rows).

Each step writes its pandas code first and computes its columns by running exactly that
code, so the exported script and the predictions from raw rows (:mod:`..frozen`) give the
same values bit for bit. The helpers ``combine_rows``, ``group_keys`` and ``group_lookup``
are copied into that code (``inspect.getsource``) and only use ``np`` and ``pd``.
"""

from __future__ import annotations

import inspect
import textwrap
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..table import MAX_NEW_COLUMNS, NUMERIC_KINDS, DataError, Table
from .base import Param, Step, StepContext, StepResult, register
from .columns import comment
from .outliers import check_new_names, fmt_count, fmt_number, name_list, plain, plural

NUMBERS = NUMERIC_KINDS  # numbers and yes/no (0/1) columns
GROUP_KINDS = ("categorical", "boolean")
BOOLEAN_KEYS = {0: "0", 1: "1"}

# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #


def _leak_error(ctx: StepContext, columns: Sequence[str], role: str) -> None:
    """Raise if the target is among ``columns``: a feature computed from the target hands
    the model the answer (target leakage), and new rows have no target to compute it."""
    if ctx.target and ctx.target in columns:
        raise DataError(
            f"{role}: '{ctx.target}' is the target. A feature computed from the target "
            "gives the model the answer during training (target leakage) and cannot be "
            "computed for new rows, which have no target. Choose other columns."
        )


def _execute(lines: list[str], df: pd.DataFrame) -> pd.DataFrame:
    """Run generated step code on ``df`` (as the exported script does) and return ``df``."""
    namespace: dict[str, Any] = {"np": np, "pd": pd, "df": df}
    exec(compile("\n".join(lines), "<feature step>", "exec"), namespace)  # noqa: S102 (our code)
    return namespace["df"]


def _add_numeric(table: Table, new: dict[str, pd.Series]) -> Table:
    """``table`` with the new number columns at the end (one concat, not one by one)."""
    block = pd.DataFrame(
        {name: pd.Series(values, dtype="float64").to_numpy() for name, values in new.items()},
        index=table.df.index,
    )
    block = block.where(np.isfinite(block), np.nan) if len(block.columns) else block
    df = pd.concat([table.df, block], axis=1)
    kinds = {**table.kinds, **dict.fromkeys(new, "numeric")}
    return Table(df, kinds, table.split.copy() if table.split is not None else None)


def _concat_lines(header: list[str], assignments: list[str]) -> list[str]:
    """Code that collects the new columns in ``new`` and adds them to ``df`` at once (inside
    ``np.errstate``: a division by zero gives a missing value, not a warning)."""
    return [
        *header,
        'with np.errstate(all="ignore"):  # 1 / 0 and overflows -> missing',
        "    new = {}",
        *[f"    {line}" for line in assignments],
        "    new = pd.DataFrame(new, index=df.index).astype(float)",
        "    new = new.where(np.isfinite(new), np.nan)",
        "df = pd.concat([df, new], axis=1)",
    ]


def _constant_warnings(new: dict[str, pd.Series]) -> list[str]:
    """Warnings for new columns with a single value or no value at all."""
    constant, empty = [], []
    for name, values in new.items():
        present = values.dropna()
        if len(values) and present.empty:
            empty.append(name)
        elif len(present) and present.nunique() == 1:
            constant.append((name, present.iloc[0]))
    warnings = []
    if len(constant) == 1:
        name, value = constant[0]
        warnings.append(
            f"Every value of '{name}' is {fmt_number(value)}: it tells the model nothing."
        )
    elif constant:
        warnings.append(
            f"{name_list([n for n, _ in constant])} have the same value in every row: they "
            "tell the model nothing."
        )
    if empty:
        warnings.append(f"Every value of {name_list(empty)} is missing.")
    return warnings


# --------------------------------------------------------------------------- #
# Combine columns (row-wise)
# --------------------------------------------------------------------------- #


def combine_rows(df, columns, operation, skip_missing=True):
    """Row-wise statistic of number columns: sum, mean, min, max, std (population),
    count (values present), nonzero (values present and not 0) or product. Missing values
    are left out (a row without any value: missing, counts 0); with skip_missing=False a
    row with a missing value gets a missing result."""
    values = [df[c].to_numpy(dtype=float, na_value=np.nan) for c in columns]
    present = [~np.isnan(v) for v in values]
    count = np.zeros(len(df))
    for p in present:
        count = count + p
    with np.errstate(all="ignore"):
        if operation == "count":
            result = count
        elif operation == "nonzero":
            result = np.zeros(len(df))
            for v, p in zip(values, present, strict=True):
                result = result + (p & (v != 0))
        elif operation in ("sum", "mean", "std"):
            total = np.zeros(len(df))
            for v, p in zip(values, present, strict=True):
                total = total + np.where(p, v, 0.0)
            result = total
            if operation != "sum":
                mean = total / count
                result = mean
                if operation == "std":
                    squares = np.zeros(len(df))
                    for v, p in zip(values, present, strict=True):
                        squares = squares + np.where(p, (v - mean) ** 2, 0.0)
                    result = np.sqrt(squares / count)
            result = np.where(count > 0, result, np.nan)
        elif operation == "product":
            result = np.ones(len(df))
            for v, p in zip(values, present, strict=True):
                result = result * np.where(p, v, 1.0)
            result = np.where(count > 0, result, np.nan)
        elif operation in ("min", "max"):
            result = np.full(len(df), np.nan)
            for v in values:
                result = np.fmin(result, v) if operation == "min" else np.fmax(result, v)
        else:
            raise ValueError(f"unknown operation {operation!r}")
    if not skip_missing:
        result = np.where(count == len(columns), result, np.nan)
    result = np.where(np.isfinite(result), result, np.nan)
    return pd.Series(result, index=df.index, dtype="float64")


COMBINE_OPERATIONS = (
    ("sum", "Sum"),
    ("mean", "Mean"),
    ("min", "Minimum"),
    ("max", "Maximum"),
    ("std", "Standard deviation"),
    ("count", "Count of values present"),
    ("nonzero", "Count of non-zero values (e.g. flags set)"),
    ("product", "Product"),
)
COMBINE_WORDS = {
    "sum": "sum",
    "mean": "mean",
    "min": "minimum",
    "max": "maximum",
    "std": "standard deviation",
    "count": "count of values",
    "nonzero": "count of non-zero values",
    "product": "product",
}


def combine_names(columns: Sequence[str], operations: Sequence[str], name: str) -> list[str]:
    """The new column names: ``name`` itself for one operation, else ``<name>_<op>``;
    without a name ``<columns joined by _>_<op>`` (``row_<op>`` for more than 3 columns or
    long names)."""
    if name and len(operations) == 1:
        return [name]
    prefix = name or "_".join(columns)
    if not name and (len(columns) > 3 or len(prefix) > 40):
        prefix = "row"
    return [f"{prefix}_{op}" for op in operations]


@register
class CombineColumns(Step):
    type = "combine_columns"
    label = "Combine columns (row-wise)"
    group = "features"
    description = (
        "New columns from several number or yes/no columns per row: sum, mean, minimum, "
        "maximum, standard deviation, product, or how many have a value or are not 0 (e.g. "
        "the number of risk flags set)."
    )
    params = (
        Param(
            "columns",
            "columns",
            "Columns",
            default=[],
            column_kinds=NUMBERS,
            help="At least two number or yes/no columns; yes/no counts as 1/0.",
        ),
        Param(
            "operations",
            "multiselect",
            "Operations",
            default=["sum"],
            options=COMBINE_OPERATIONS,
            help=(
                "One new column per operation. The standard deviation is the population one "
                "(0 for a single value). Non-zero counts the values that are present and not 0."
            ),
        ),
        Param(
            "name",
            "string",
            "New column name",
            default="",
            optional=True,
            help=(
                "With one operation the name of the new column, with several the start of "
                "every name (<name>_sum, <name>_mean …). Empty: the column names joined by "
                "'_' (row_sum … for more than 3 columns)."
            ),
        ),
        Param(
            "missing",
            "select",
            "Missing values",
            default="skip",
            options=(
                ("skip", "Leave them out"),
                ("any", "Missing if any value is missing"),
            ),
            help=(
                "Leave them out: the result of the values present (a row without any value "
                "gets a missing result; its counts are 0). Missing if any value is missing: "
                "a row with a missing input gets a missing result in every new column."
            ),
        ),
    )

    def clean(self, table: Table, ctx: StepContext) -> dict[str, Any]:
        params = super().clean(table, ctx)
        if len(params["columns"]) < 2:
            raise DataError("Columns: choose at least two columns to combine.")
        if not params["operations"]:
            raise DataError("Operations: choose at least one operation.")
        _leak_error(ctx, params["columns"], "Columns")
        params["name"] = params["name"] or ""
        names = combine_names(params["columns"], params["operations"], params["name"])
        check_new_names(table, names)
        return params

    def run(self, table: Table, params: dict[str, Any], ctx: StepContext) -> StepResult:
        columns, operations = params["columns"], params["operations"]
        names = combine_names(columns, operations, params["name"])
        out = _execute(self.to_pandas(params, None, ctx), table.df[columns].copy())
        new = {name: out[name] for name in names}
        result = _add_numeric(table, new)
        inputs = table.df[columns].isna()
        some_missing = inputs.any(axis=1).to_numpy()
        all_missing = inputs.all(axis=1).to_numpy()
        what = ", ".join(COMBINE_WORDS[op] for op in operations)
        summary = f"Added {name_list(names)} (row-wise {what} of {plural(len(columns), 'column')})"
        rows = int(some_missing.sum())
        if rows and params["missing"] == "skip":
            summary += (
                f"; {plural(rows, 'row')} with missing inputs {'uses' if rows == 1 else 'use'} "
                "the values present"
            )
            if all_missing.any():
                summary += f" ({fmt_count(all_missing.sum())} without any value)"
        elif rows:
            summary += f"; {plural(rows, 'row')} with a missing input {_get(rows)} missing results"
        overflow = sum(
            int((values.isna().to_numpy() & ~some_missing).sum()) for values in new.values()
        )
        if overflow:
            summary += f"; {plural(overflow, 'value')} too large for a number, set missing"
        warnings = _constant_warnings(new)
        details = [{"column": name, "text": _range_text(values)} for name, values in new.items()]
        return StepResult(result, summary + ".", details, warnings=warnings)

    def to_pandas(
        self, params: dict[str, Any], fitted: dict[str, Any] | None, ctx: StepContext
    ) -> list[str]:
        params = {**self.defaults(), **params}
        columns, operations = params["columns"], params["operations"]
        names = combine_names(columns, operations, params["name"] or "")
        skip = params["missing"] == "skip"
        assignments = [
            f"new[{name!r}] = combine_rows(df, {list(columns)!r}, {op!r}, skip_missing={skip!r})"
            for name, op in zip(names, operations, strict=True)
        ]
        return [*_source(combine_rows), "", *_concat_lines([], assignments)]


def _source(function: Any) -> list[str]:
    return textwrap.dedent(inspect.getsource(function)).rstrip().splitlines()


def _range_text(values: pd.Series) -> str:
    present = values.dropna()
    if present.empty:
        return "all missing"
    text = f"{fmt_number(present.min())} … {fmt_number(present.max())}"
    missing = len(values) - len(present)
    return text + (f", {fmt_count(missing)} missing" if missing else "")


# --------------------------------------------------------------------------- #
# Pairwise interactions
# --------------------------------------------------------------------------- #

INTERACTIONS = (
    ("multiply", "a × b"),
    ("divide", "a / b"),
    ("subtract", "a − b"),
    ("add", "a + b"),
)
SYMBOLS = {"multiply": "*", "divide": "/", "subtract": "-", "add": "+"}
ORDERED = ("divide", "subtract")  # a / b ≠ b / a
DEFAULT_MAX_INTERACTIONS = 200


def interaction_pairs(first: Sequence[str], second: Sequence[str]) -> list[tuple[str, str]]:
    """Every pair of two different columns of ``first`` (in list order), or with
    ``second`` every column of ``first`` with every column of ``second``; a pair of the same
    two columns only once."""
    if second:
        candidates = [(a, b) for a in first for b in second if a != b]
    else:
        candidates = [(a, b) for i, a in enumerate(first) for b in first[i + 1 :]]
    seen: set[frozenset[str]] = set()
    pairs = []
    for a, b in candidates:
        key = frozenset((a, b))
        if key not in seen:
            seen.add(key)
            pairs.append((a, b))
    return pairs


def interaction_plan(params: dict[str, Any]) -> list[tuple[str, str, str, str]]:
    """``(name, operation, a, b)`` of every new column, in order."""
    plan = []
    for a, b in interaction_pairs(params["columns"], params["columns_b"] or []):
        for op in params["operations"]:
            orders = [(a, b), (b, a)] if op in ORDERED and params["both_directions"] else [(a, b)]
            for x, y in orders:
                plan.append((f"{x}{SYMBOLS[op]}{y}", op, x, y))
    return plan


def _interaction_code(op: str, a: str, b: str) -> str:
    x, y = f"df[{a!r}]", f"df[{b!r}]"
    if op == "divide":  # 0 or missing in the denominator -> missing
        return f"{x} / {y}.where({y} != 0)"
    return f"{x} {SYMBOLS[op]} {y}"


@register
class Interactions(Step):
    type = "interactions"
    label = "Pairwise interactions"
    group = "features"
    description = (
        "New columns for pairs of number columns: a × b, a / b, a − b and a + b, e.g. "
        "amount / account balance."
    )
    params = (
        Param(
            "columns",
            "columns",
            "Columns",
            default=[],
            column_kinds=NUMBERS,
            help="Number or yes/no columns. Every pair of them is combined.",
        ),
        Param(
            "columns_b",
            "columns",
            "Combine with",
            default=[],
            optional=True,
            column_kinds=NUMBERS,
            empty_means="Pairs within the first list",
            help="If given, every column of the first list is combined with every column of "
            "this list instead.",
        ),
        Param(
            "operations",
            "multiselect",
            "Operations",
            default=["multiply"],
            options=INTERACTIONS,
            help=(
                "Names: a*b, a/b, a-b, a+b. Division: a 0 or missing denominator gives a "
                "missing value (as do results too large for a number)."
            ),
        ),
        Param(
            "both_directions",
            "bool",
            "Both directions",
            default=False,
            help="Also b / a and b − a (division and subtraction depend on the order).",
        ),
        Param(
            "max_columns",
            "int",
            "Most new columns",
            default=DEFAULT_MAX_INTERACTIONS,
            min=1,
            max=MAX_NEW_COLUMNS,
            help="The step stops with an error instead of adding more columns than this: "
            "n columns have n·(n − 1)/2 pairs.",
        ),
    )

    def clean(self, table: Table, ctx: StepContext) -> dict[str, Any]:
        params = super().clean(table, ctx)
        if not params["columns"]:
            raise DataError("Columns: choose the columns to combine.")
        if not params["operations"]:
            raise DataError("Operations: choose at least one operation.")
        _leak_error(ctx, params["columns"], "Columns")
        _leak_error(ctx, params["columns_b"], "Combine with")
        plan = interaction_plan(params)
        if not plan:
            raise DataError(
                "Columns: choose at least two columns (or other columns under “Combine with”)."
            )
        if len(plan) > params["max_columns"]:
            pairs = len(interaction_pairs(params["columns"], params["columns_b"]))
            raise DataError(
                f"Most new columns: these settings would add {len(plan):,} columns "
                f"({plural(pairs, 'pair')} of columns), more than the limit of "
                f"{params['max_columns']:,}. Choose fewer columns or operations, or raise the "
                f"limit (at most {MAX_NEW_COLUMNS:,})."
            )
        check_new_names(table, [name for name, _, _, _ in plan])
        return params

    def run(self, table: Table, params: dict[str, Any], ctx: StepContext) -> StepResult:
        plan = interaction_plan(params)
        inputs = list(dict.fromkeys([*params["columns"], *params["columns_b"]]))
        out = _execute(self.to_pandas(params, None, ctx), table.df[inputs].copy())
        new = {name: out[name] for name, _, _, _ in plan}
        result = _add_numeric(table, new)
        caused = 0
        zero = 0
        overflow = 0
        for name, op, a, b in plan:
            missing = new[name].isna().to_numpy()
            from_inputs = (table.df[a].isna() | table.df[b].isna()).to_numpy()
            caused += int((missing & from_inputs).sum())
            rest = missing & ~from_inputs
            if op == "divide":
                by_zero = rest & (table.df[b].to_numpy(dtype=float) == 0)
                zero += int(by_zero.sum())
                rest = rest & ~by_zero
            overflow += int(rest.sum())
        pairs = len(interaction_pairs(params["columns"], params["columns_b"]))
        ops = ", ".join(dict(INTERACTIONS)[op] for op in params["operations"])
        summary = (
            f"Added {plural(len(plan), 'column')} from {plural(pairs, 'pair')} of columns ({ops})"
        )
        if len(plan) <= 3:
            summary += f": {name_list(list(new))}"
        notes = []
        if caused:
            notes.append(f"{plural(caused, 'value')} missing because an input is missing")
        if zero:
            notes.append(f"{plural(zero, 'division')} by zero gave missing values")
        if overflow:
            notes.append(f"{plural(overflow, 'value')} too large for a number, set missing")
        summary += ("; " + "; ".join(notes) if notes else "") + "."
        return StepResult(result, summary, warnings=_constant_warnings(new))

    def to_pandas(
        self, params: dict[str, Any], fitted: dict[str, Any] | None, ctx: StepContext
    ) -> list[str]:
        params = {**self.defaults(), **params}
        plan = interaction_plan(params)
        assignments = [f"new[{name!r}] = {_interaction_code(op, a, b)}" for name, op, a, b in plan]
        return _concat_lines([], assignments)


# --------------------------------------------------------------------------- #
# Group statistics
# --------------------------------------------------------------------------- #


def group_keys(df, columns, booleans=()):
    """The group of every row: the value of the column, or a tuple of the values of
    several columns (yes/no columns as "0"/"1")."""
    parts = [df[c].map({0: "0", 1: "1"}) if c in booleans else df[c] for c in columns]
    if len(parts) == 1:
        return parts[0].tolist()
    return list(zip(*(part.tolist() for part in parts), strict=True))


def group_lookup(keys, mapping, default=np.nan):
    """The value of every row's group in ``mapping``; ``default`` for groups that are not
    in it and for rows with a missing group value."""
    return np.array([mapping.get(key, default) for key in keys], dtype=float)


GROUP_STATS = (
    ("mean", "Mean"),
    ("median", "Median"),
    ("std", "Standard deviation"),
    ("min", "Minimum"),
    ("max", "Maximum"),
    ("count", "Count of values"),
    ("rows", "Rows per group (frequency)"),
    ("diff_mean", "Difference from the group mean"),
    ("ratio_mean", "Ratio to the group mean"),
)
# Statistics learned per group (the last two are computed from the mean).
LEARNED = ("mean", "median", "std", "min", "max", "count", "rows")
COUNTS = ("count", "rows")  # unseen groups: 0
STAT_WORDS = {
    "mean": "mean",
    "median": "median",
    "std": "standard deviation",
    "min": "minimum",
    "max": "maximum",
    "count": "count",
    "rows": "rows per group",
    "diff_mean": "difference from the mean",
    "ratio_mean": "ratio to the mean",
}
MAX_GROUPS = 20_000


def group_name(stat: str, value: str | None, group_by: Sequence[str]) -> str:
    key = "_".join(group_by)
    if stat == "rows":
        return f"rows_by_{key}"
    return f"{value}_{stat}_by_{key}"


def _learned_stats(stats: Sequence[str]) -> list[str]:
    """The statistics to learn per group for the chosen outputs."""
    wanted = set(stats)
    if wanted & {"diff_mean", "ratio_mean"}:
        wanted.add("mean")
    return [s for s in LEARNED if s in wanted]


@register
class GroupFeatures(Step):
    type = "group_features"
    label = "Group statistics"
    group = "features"
    description = (
        "Statistics of a number column per group of a category column, learned from the "
        "training rows: mean, median, spread, count, rows per group, and how far a row is "
        "from its group's mean, e.g. the amount relative to the usual amount of the "
        "merchant category."
    )
    fitted = True
    params = (
        Param(
            "group_by",
            "columns",
            "Group by",
            default=[],
            column_kinds=GROUP_KINDS,
            help="A category or yes/no column; several columns form one group per "
            "combination of their values. Change number codes to a category first.",
        ),
        Param(
            "value",
            "column",
            "Value column",
            default=None,
            optional=True,
            column_kinds=NUMBERS,
            help="The number column to describe per group (not the target). Only “Rows per "
            "group” works without one.",
        ),
        Param(
            "stats",
            "multiselect",
            "Statistics",
            default=["mean", "ratio_mean"],
            options=GROUP_STATS,
            help=(
                "One new column each, named like amount_mean_by_category (rows: "
                "rows_by_category). Standard deviation: population (0 for one value). "
                "Difference: value − group mean. Ratio: value / group mean (a group mean of 0 "
                "gives a missing value)."
            ),
        ),
        Param(
            "unseen",
            "select",
            "Groups not in the training rows",
            default="overall",
            options=(
                ("overall", "Use the overall training value"),
                ("missing", "Missing value"),
            ),
            help=(
                "For groups that do not occur in the training rows (and rows without a group "
                "value): the statistic of all training rows, or a missing value. Counts and "
                "rows per group are 0 for them."
            ),
        ),
        Param(
            "min_rows",
            "int",
            "Min. rows per group",
            default=1,
            min=1,
            max=100000,
            help="Groups with fewer training values are treated as unseen, so that a few rows "
            "do not decide a statistic.",
        ),
    )

    def clean(self, table: Table, ctx: StepContext) -> dict[str, Any]:
        params = super().clean(table, ctx)
        group_by, value = params["group_by"], params["value"]
        if not group_by:
            raise DataError("Group by: choose a category column.")
        if not params["stats"]:
            raise DataError("Statistics: choose at least one statistic.")
        if ctx.target and ctx.target == value:
            raise DataError(
                f"Value column: '{value}' is the target. Its mean per group would give the "
                "model the answer of the training rows (target leakage), and new rows have no "
                "target. To use the target per category, choose Encode categories → Target "
                "mean: it uses out-of-fold values for the training rows."
            )
        _leak_error(ctx, group_by, "Group by")
        if value is not None and value in group_by:
            raise DataError(f"Value column: '{value}' is also a group column.")
        needs_value = [s for s in params["stats"] if s != "rows"]
        if needs_value and value is None:
            raise DataError(
                f"Value column: choose the number column for the "
                f"{STAT_WORDS[needs_value[0]]}; only “Rows per group” works without one."
            )
        check_new_names(table, [group_name(s, value, group_by) for s in params["stats"]])
        return params

    def run(self, table: Table, params: dict[str, Any], ctx: StepContext) -> StepResult:
        group_by, value, stats = params["group_by"], params["value"], params["stats"]
        booleans = [c for c in group_by if table.kinds[c] == "boolean"]
        train = table.train_mask()
        groups, learned, enough, rows = _learn_groups(table, params, booleans, train)
        overall: dict[str, float] = {}
        if value:
            present = table.df[value].to_numpy(dtype=float)[train]
            present = present[np.isfinite(present)]
            if len(present):
                totals = {
                    "mean": present.mean(),
                    "median": np.median(present),
                    "std": present.std(),
                    "min": present.min(),
                    "max": present.max(),
                }
                overall = {s: plain(v) for s, v in totals.items() if s in learned}
        fitted = {
            "groups": len(groups),
            "booleans": booleans,
            "overall": overall,
            "_groups": groups,
            "_values": learned,
        }
        names = [group_name(s, value, group_by) for s in stats]
        inputs = [*group_by, *([value] if value else [])]
        out = _execute(self.to_pandas(params, fitted, ctx), table.df[inputs].copy())
        new = {name: out[name] for name in names}
        result = _add_numeric(table, new)

        # Rows of the validation/test part whose group the training rows do not have.
        keys = group_keys(table.df, group_by, booleans)
        known = {tuple(g) for g in groups}
        row_keys = [k if isinstance(k, tuple) else (k,) for k in keys]
        no_key = np.array([any(_blank(part) for part in k) for k in row_keys], bool)
        seen = np.array([tuple(k) in known for k in row_keys], bool)
        unseen = ~no_key & ~seen & ~train
        fill = "the overall training value" if params["unseen"] == "overall" else "missing values"
        if stats == ["rows"]:
            fill = "0"
        what = ", ".join(STAT_WORDS[s] for s in stats)
        of_value = f" from '{value}'" if value and stats != ["rows"] else ""
        summary = (
            f"Added {plural(len(names), 'column')}{of_value} per {name_list(group_by)} "
            f"({what}; {plural(len(groups), 'group')} in the training rows)"
        )
        notes = []
        rare = int((~enough).sum()) if value else 0
        if rare and params["min_rows"] > 1:
            notes.append(
                f"{plural(rare, 'group')} with fewer than {fmt_count(params['min_rows'])} "
                "training values count as unseen"
            )
        if unseen.any():
            notes.append(
                f"{plural(int(unseen.sum()), 'validation/test row')} with a group that is not "
                f"in the training rows {_get(int(unseen.sum()))} {fill}"
            )
        if no_key.any():
            count = int(no_key.sum())
            notes.append(f"{plural(count, 'row')} without a group value {_get(count)} {fill}")
        missing = sum(int(v.isna().sum()) for v in new.values())
        if missing:
            notes.append(f"{plural(missing, 'value')} missing in the new columns")
        summary += ("; " + "; ".join(notes) if notes else "") + "."
        warnings = _constant_warnings(new)
        singles = int((rows == 1).sum())
        if len(groups) >= 4 and singles > len(groups) / 2:
            warnings.append(
                f"{fmt_count(singles)} of {plural(len(groups), 'group')} have only one training "
                "row: their statistics are that row's own value, which the model learns by "
                "heart. Raise “Min. rows per group” or group rare categories first."
            )
        details = [{"column": name, "text": _range_text(v)} for name, v in new.items()]
        return StepResult(result, summary, details, fitted, warnings)

    def to_pandas(
        self, params: dict[str, Any], fitted: dict[str, Any] | None, ctx: StepContext
    ) -> list[str]:
        params = {**self.defaults(), **params}
        fitted = fitted or {}
        group_by, value, stats = params["group_by"], params["value"], params["stats"]
        groups = [tuple(g) for g in fitted.get("_groups") or []]
        values = fitted.get("_values") or {}
        overall = fitted.get("overall") or {}
        single = len(group_by) == 1

        def mapping(stat: str) -> str:
            pairs = {
                (g[0] if single else g): v
                for g, v in zip(groups, values.get(stat) or [], strict=False)
                if v is not None
            }
            return repr(pairs)

        def default(stat: str) -> str:
            if stat in COUNTS:
                return "0.0"
            if params["unseen"] == "overall" and overall.get(stat) is not None:
                return repr(float(overall[stat]))
            return "np.nan"

        booleans = list(fitted.get("booleans") or [])
        header = [
            *_source(group_keys),
            "",
            *_source(group_lookup),
            "",
            f"# {len(groups):,} groups of {comment(', '.join(group_by))} learned from the training "
            "rows",
            f"keys = group_keys(df, {list(group_by)!r}"
            + (f", booleans={booleans!r})" if booleans else ")"),
        ]
        assignments = []
        x = f"df[{value!r}].to_numpy(dtype=float)" if value else ""
        if {"diff_mean", "ratio_mean"} & set(stats):
            assignments.append(
                f"group_mean = group_lookup(keys, {mapping('mean')}, {default('mean')})"
            )
        for stat in stats:
            name = group_name(stat, value, group_by)
            if stat == "diff_mean":
                expr = f"{x} - group_mean"
            elif stat == "ratio_mean":
                expr = f"{x} / np.where(group_mean != 0, group_mean, np.nan)"
            elif stat == "mean" and {"diff_mean", "ratio_mean"} & set(stats):
                expr = "group_mean"
            else:
                expr = f"group_lookup(keys, {mapping(stat)}, {default(stat)})"
            assignments.append(f"new[{name!r}] = {expr}")
        return _concat_lines(header, assignments)


def _learn_groups(
    table: Table, params: dict[str, Any], booleans: list[str], train: np.ndarray
) -> tuple[list[list[str]], dict[str, list[float | None]], np.ndarray, np.ndarray]:
    """The groups of the training rows (sorted), the learned statistics per group (None:
    not usable, e.g. fewer values than ``min_rows``), which groups have enough values and
    the training rows per group."""
    group_by, value = params["group_by"], params["value"]
    keys = [f"k{i}" for i in range(len(group_by))]
    frame = pd.DataFrame(
        {
            **{
                key: table.df[c].map(BOOLEAN_KEYS) if c in booleans else table.df[c]
                for key, c in zip(keys, group_by, strict=True)
            },
            "x": table.df[value].to_numpy(dtype=float) if value else np.nan,
        }
    )
    frame = frame[train]
    frame = frame[frame[keys].notna().all(axis=1)]
    grouped = frame.groupby(keys, sort=True)["x"]
    stats = pd.DataFrame({"rows": grouped.size(), "count": grouped.count()})
    if stats.empty:
        raise DataError(
            f"Group by: {name_list(group_by)} has no values in the training rows, so there are "
            "no groups to learn."
        )
    if len(stats) > MAX_GROUPS:
        raise DataError(
            f"Group by: the training rows have {len(stats):,} groups; at most {MAX_GROUPS:,} "
            "are supported. Group rare categories first or choose a column with fewer values."
        )
    wanted = _learned_stats(params["stats"])
    if value:
        for stat in ("mean", "median", "min", "max"):
            if stat in wanted:
                stats[stat] = grouped.agg(stat)
        if "std" in wanted:
            stats["std"] = grouped.std(ddof=0)
    groups = [
        [str(part) for part in (index if isinstance(index, tuple) else (index,))]
        for index in stats.index
    ]
    enough = (stats["count"] >= params["min_rows"]).to_numpy()
    learned: dict[str, list[float | None]] = {}
    for stat in wanted:
        column = stats[stat].to_numpy(dtype=float)
        usable = np.ones(len(column), bool) if stat in COUNTS else enough
        learned[stat] = [
            plain(v) if ok and np.isfinite(v) else None
            for v, ok in zip(column.tolist(), usable, strict=True)
        ]
    return groups, learned, enough, stats["rows"].to_numpy()


def _get(n: int) -> str:
    return "gets" if n == 1 else "get"


def _blank(value: Any) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value))
