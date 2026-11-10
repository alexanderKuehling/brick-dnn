"""Preparation steps from the row analysis (Analysis → Rows): remove atypical rows, find
groups of near-duplicate rows, and drop rows picked by hand.

"Remove atypical rows" and "Find near-duplicate groups" run **before the split**
(``pre_split``): right after the combine steps, on all rows. They use no labels. Atypical
rows (e.g. filler or synthetic rows whose every value sits at its column median) are errors
rather than rare cases, so they are removed from all parts: left in the validation and test
data, their arbitrary labels would only add noise to the scores. The group column must
exist before the split, so that "Keep rows of a group together" (Target & split) can use it.

The statistics are those of :mod:`brickdnn.data.rows`.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .. import rows as rowstats
from ..table import STRING_KINDS, DataError, Table
from .base import Param, Step, StepContext, StepResult, register
from .columns import as_label, cell_value, code_value, comment, fmt_count, from_fitted, to_fitted
from .outliers import check_new_names, code_float, plain
from .text import fmt_number, fmt_share, plural

FEATURE_OPTIONS = (
    ("auto", "Automatic (partnered columns in wide tables)"),
    ("partnered", "Columns with a strong partner"),
    ("all", "All number columns"),
    ("columns", "Columns I choose"),
)
FEATURES_HELP = (
    "Distances and spreads use these number columns (never the target or ID columns). In "
    "wide tables most columns without any correlated partner are noise, and noise columns "
    "drown the differences between rows; Automatic uses the columns with a strong partner "
    "(largest |Spearman r| ≥ 0.3) in tables with more than 20 number columns."
)


def _feature_params(what: str) -> tuple[Param, Param]:
    return (
        Param(
            "features",
            "select",
            f"{what} on",
            default="auto",
            options=FEATURE_OPTIONS,
            help=FEATURES_HELP,
        ),
        Param(
            "columns",
            "columns",
            "Columns",
            default=[],
            column_kinds=("numeric", "boolean"),
            visible_if={"param": "features", "values": ["columns"]},
        ),
    )


def _features_text(choice: rowstats.FeatureChoice) -> str:
    what = {"partnered": "partnered ", "all": "", "columns": "chosen "}[choice.used]
    return plural(len(choice.columns), f"{what}number column")


# --------------------------------------------------------------------------- #
# Remove atypical rows
# --------------------------------------------------------------------------- #


@register
class RemoveAtypicalRows(Step):
    type = "remove_atypical"
    label = "Remove atypical rows"
    group = "rows"
    description = (
        "Remove rows whose values are all unusually close to the column medians (filler or "
        "synthetic rows), or unusually far from them. Runs before the split, on all rows."
    )
    fitted = True
    pre_split = True
    params = (
        *_feature_params("Measure"),
        Param(
            "side",
            "select",
            "Remove rows that are",
            default="low",
            options=(
                ("low", "Too typical: every value close to its median"),
                ("high", "Unusual: values far from the medians"),
            ),
            help=(
                "The spread of a row is the median of |robust z| over its values; ordinary "
                "rows have about 0.67. Too typical rows (near 0) are often filler or "
                "synthetic rows with made-up labels."
            ),
        ),
        Param(
            "threshold",
            "number",
            "Spread threshold",
            default=None,
            optional=True,
            min=0,
            step=0.05,
            help=(
                "Rows with a spread below (too typical) or above (unusual) this value are "
                "removed. Empty: the gap the Rows analysis finds between a small cluster and "
                "the other rows (nothing is removed if there is no clear gap)."
            ),
        ),
    )

    def run(self, table: Table, params: dict[str, Any], ctx: StepContext) -> StepResult:
        exclude = [ctx.target] if ctx.target else []
        choice, medians, scales = rowstats.fit_features(
            table, params["features"], params["columns"], exclude=exclude, seed=ctx.seed
        )
        spread = rowstats.spreads(table, choice.columns, medians, scales)
        threshold = params["threshold"]
        low = params["side"] == "low"
        warnings: list[str] = []
        automatic = threshold is None
        if automatic:
            if low:
                threshold = rowstats.suggest_threshold(spread, "atypical", len(choice.columns))
            if threshold is None:
                fitted = self._fitted(choice, medians, scales, None, params["side"])
                what = "too typical" if low else "unusual"
                return StepResult(
                    table,
                    f"No clear cluster of {what} rows found on {_features_text(choice)}; "
                    "nothing removed. Enter a spread threshold to remove rows anyway.",
                    fitted=fitted,
                )
        remove = (spread < threshold) if low else (spread > threshold)
        remove &= np.isfinite(spread)
        removed = int(remove.sum())
        if removed == table.n_rows and removed:
            raise DataError("Every row would be removed. Choose another threshold.")
        median = float(np.nanmedian(spread)) if np.isfinite(spread).any() else float("nan")
        sign = "<" if low else ">"
        rule = (
            f"spread {sign} {fmt_number(threshold)}"
            f"{' (found automatically)' if automatic else ''}; ordinary rows "
            f"{fmt_number(median)}"
        )
        if removed:
            summary = (
                f"Removed {plural(removed, 'row')} ({fmt_share(removed, table.n_rows)}) "
                f"that {'are' if removed != 1 else 'is'} {'too typical' if low else 'unusual'} "
                f"on {_features_text(choice)} ({rule})."
            )
        else:
            summary = f"No row has a {rule}; nothing removed."
        if removed > 0.2 * table.n_rows:
            warnings.append(
                f"This removes {fmt_share(removed, table.n_rows)} of the rows. Check the "
                "threshold in Analysis → Rows."
            )
        fitted = self._fitted(choice, medians, scales, threshold, params["side"])
        return StepResult(table.take(~remove) if removed else table, summary, [], fitted, warnings)

    @staticmethod
    def _fitted(
        choice: rowstats.FeatureChoice,
        medians: np.ndarray,
        scales: np.ndarray,
        threshold: float | None,
        side: str,
    ) -> dict[str, Any]:
        return {
            "threshold": plain(threshold) if threshold is not None else None,
            "side": side,
            "features": choice.used,
            "feature_count": len(choice.columns),
            "_columns": list(choice.columns),
            "_medians": [plain(v) for v in medians],
            "_scales": [plain(v) for v in scales],
        }

    def to_pandas(
        self, params: dict[str, Any], fitted: dict[str, Any] | None, ctx: StepContext
    ) -> list[str]:
        fitted = fitted or {}
        threshold = fitted.get("threshold")
        columns = fitted.get("_columns") or []
        if threshold is None or not columns:
            return ["# No clear cluster of atypical rows was found; nothing to remove."]
        sign = "<" if fitted.get("side", "low") == "low" else ">"
        return [
            "# Spread of a row: median |robust z| over its values (medians and scales learned",
            "# before the split). Remove this step's lines to keep every row of new data.",
            f"_columns = {columns!r}",
            f"_medians = np.array({[float(v) for v in fitted['_medians']]!r})",
            f"_scales = np.array({[float(v) for v in fitted['_scales']]!r})",
            "_z = ((df[_columns].astype(float) - _medians) / _scales).clip("
            f"-{rowstats.Z_CLIP!r}, {rowstats.Z_CLIP!r})",
            "_spread = _z.abs().median(axis=1, skipna=True)",
            f"df = df[~(_spread {sign} {code_float(threshold)})].reset_index(drop=True)",
        ]


# --------------------------------------------------------------------------- #
# Find near-duplicate groups
# --------------------------------------------------------------------------- #


@register
class FindGroups(Step):
    type = "find_groups"
    label = "Find near-duplicate groups"
    group = "rows"
    description = (
        "Add a column that numbers groups of rows that are near twins of each other (e.g. "
        "repeated measurements of one person). Keep the groups together in the split "
        "(Target & split). Runs before the split, on all rows."
    )
    pre_split = True
    params = (
        *_feature_params("Compare rows"),
        Param(
            "threshold",
            "number",
            "NN ratio threshold",
            default=None,
            optional=True,
            min=0,
            max=2,
            step=0.05,
            help=(
                "Two rows are near twins if one is among the 10 nearest rows of the other and "
                "their distance is below this share of the typical distance between two "
                "rows. Empty: the valley the Rows analysis finds between rows with a near "
                "twin and the others (no groups if there is none)."
            ),
        ),
        Param("name", "string", "Name of the new column", default="group"),
    )

    def run(self, table: Table, params: dict[str, Any], ctx: StepContext) -> StepResult:
        name = params["name"]
        check_new_names(table, [name])
        exclude = [ctx.target] if ctx.target else []
        choice, _, _ = rowstats.fit_features(
            table, params["features"], params["columns"], exclude=exclude, seed=ctx.seed
        )
        limit = rowstats.row_limit(len(choice.columns))
        if table.n_rows > limit:
            raise DataError(
                f"Finding near twins compares every row with every other row; this works for "
                f"up to {fmt_count(limit)} rows with {_features_text(choice)} "
                f"({fmt_count(table.n_rows)} here). Use fewer columns, or an existing column "
                "(e.g. a person id) for the groups."
            )
        stats = rowstats.compute(
            table,
            target=ctx.target,
            mode="columns",
            columns=choice.columns,
            fit="all",
            seed=ctx.seed,
            with_projection=False,
        )
        stats.features = choice
        threshold = params["threshold"]
        automatic = threshold is None
        warnings: list[str] = []
        if automatic:
            threshold = stats.suggestions()["twins"]
        if threshold is None:
            labels = np.arange(table.n_rows)
            sizes = np.ones(table.n_rows, dtype=np.int64)
            warnings.append(
                "No clear valley between rows with a near twin and the others, so every row is "
                "its own group. Look at the NN ratios in Analysis → Rows and enter a threshold."
            )
        else:
            labels, sizes = stats.groups(threshold)
        group = labels.astype(np.float64) + 1
        result = table.with_column(name, group, "numeric")
        in_groups = sizes >= 2
        count = len(np.unique(labels[in_groups]))
        fitted = {
            "threshold": plain(threshold) if threshold is not None else None,
            "features": choice.used,
            "feature_count": len(choice.columns),
            "groups": count,
            "rows_in_groups": int(in_groups.sum()),
        }
        if count:
            largest = int(sizes.max())
            smallest = int(sizes[in_groups].min())
            span = f"{smallest}–{largest}" if smallest != largest else str(largest)
            summary = (
                f"Found {plural(count, 'group')} of near twins ({span} rows each, "
                f"{fmt_count(int(in_groups.sum()))} rows in all) on {_features_text(choice)}, "
                f"NN ratio < {fmt_number(threshold)}"
                f"{' (found automatically)' if automatic else ''}; every other row is its own "
                f"group. '{name}' numbers the groups: keep them together in Target & split."
            )
            if largest > max(20, 0.1 * table.n_rows):
                warnings.append(
                    f"The largest group has {fmt_count(largest)} rows: the threshold may link "
                    "rows that are not near twins (or many rows are nearly identical)."
                )
        elif threshold is not None:
            summary = (
                f"No two rows are near twins (NN ratio < {fmt_number(threshold)}) on "
                f"{_features_text(choice)}; every row is its own group in '{name}'."
            )
        else:
            summary = f"Every row is its own group in '{name}' (no near twins found)."
        return StepResult(result, summary, [], fitted, warnings)

    def to_pandas(
        self, params: dict[str, Any], fitted: dict[str, Any] | None, ctx: StepContext
    ) -> list[str]:
        name = params.get("name") or "group"
        return [
            "# Groups of near twins only matter for the split of the training data: every new",
            "# row is its own group.",
            f"df[{name!r}] = np.arange(1, len(df) + 1, dtype=float)",
        ]


# --------------------------------------------------------------------------- #
# Drop rows picked by hand
# --------------------------------------------------------------------------- #


@register
class DropRows(Step):
    type = "drop_rows"
    label = "Drop rows"
    group = "rows"
    description = (
        "Remove rows picked by hand, e.g. in Analysis → Rows: by the values of an ID column, "
        "or by their row numbers."
    )
    params = (
        Param(
            "by",
            "select",
            "Pick rows by",
            default="column",
            options=(("column", "Values of a column (e.g. an id)"), ("position", "Row numbers")),
            help=(
                "Values of an ID column stay right when earlier steps change. Row numbers "
                "(0 = first row) count the rows as they are right before this step."
            ),
        ),
        Param(
            "column",
            "column",
            "Column",
            default=None,
            visible_if={"param": "by", "values": ["column"]},
        ),
        Param(
            "values",
            "list",
            "Values",
            default=[],
            visible_if={"param": "by", "values": ["column"]},
        ),
        Param(
            "positions",
            "list",
            "Row numbers",
            default=[],
            visible_if={"param": "by", "values": ["position"]},
        ),
    )

    def clean(self, table: Table, ctx: StepContext) -> dict[str, Any]:
        params = super().clean(table, ctx)
        if params["by"] == "column":
            column = params["column"]
            if not params["values"]:
                raise DataError("Values: add at least one value.")
            kind = table.kinds[column]
            strings = kind in STRING_KINDS
            params["values"] = [
                as_label(v) if strings else to_fitted(cell_value(v, kind, column, "Values"))
                for v in params["values"]
            ]
            params["_kind"] = kind
        else:
            if not params["positions"]:
                raise DataError("Row numbers: add at least one row number.")
            positions = []
            for value in params["positions"]:
                number = float(value) if not isinstance(value, str) else _int(value)
                if number is None or not math.isfinite(number) or not number.is_integer():
                    raise DataError(f"Row numbers: '{value}' is not a whole number.")
                if not 0 <= number < table.n_rows:
                    raise DataError(
                        f"Row numbers: there is no row {int(number):,} (the rows are numbered "
                        f"0 to {table.n_rows - 1:,} here)."
                    )
                positions.append(int(number))
            params["positions"] = sorted(set(positions))
        return params

    def run(self, table: Table, params: dict[str, Any], ctx: StepContext) -> StepResult:
        if params["by"] == "column":
            column, kind = params["column"], params["_kind"]
            values = [from_fitted(v, kind) for v in params["values"]]
            drop = table.df[column].isin(values).to_numpy()
            where = f"whose {column} is one of {plural(len(values), 'value')}"
        else:
            drop = np.zeros(table.n_rows, dtype=bool)
            drop[params["positions"]] = True
            where = "picked by row number"
        removed = int(drop.sum())
        if removed == table.n_rows and removed:
            raise DataError("This would remove every row.")
        if not removed:
            return StepResult(table, f"No row {where}; nothing removed.")
        return StepResult(
            table.take(~drop),
            f"Removed {plural(removed, 'row')} ({fmt_share(removed, table.n_rows)}) {where}.",
        )

    def to_pandas(
        self, params: dict[str, Any], fitted: dict[str, Any] | None, ctx: StepContext
    ) -> list[str]:
        if params.get("by") == "position":
            return [
                f"# {comment(plural(len(params.get('positions') or []), 'row'))} picked by row "
                "number are only removed from the prepared data, not from new data."
            ]
        kind = params.get("_kind", "categorical")
        values = "[" + ", ".join(code_value(from_fitted(v, kind)) for v in params["values"]) + "]"
        return [f"df = df[~df[{params['column']!r}].isin({values})].reset_index(drop=True)"]


def _int(text: str) -> float | None:
    try:
        return float(text.strip())
    except ValueError:
        return None
