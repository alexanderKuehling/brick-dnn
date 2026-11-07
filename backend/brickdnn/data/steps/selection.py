"""Preparation steps: dropping features that do not help (feature selection).

Statistics come from the training rows (at most ``MAX_ROWS`` of them, a seeded random
sample). Association with the target, depending on the column types: |Pearson r| (number vs
number), the correlation ratio η (number vs categories) or Cramér's V (categories vs
classes), all between 0 and 1.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from ..table import DataError, Table
from .base import Param, Step, StepContext, StepResult, register
from .outliers import fmt_number, fmt_share, name_list, plural

MAX_ROWS = 100_000
NUMBERS = ("numeric", "boolean")


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    """|Pearson r| over the rows where both are present; None if undefined."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return None
    x, y = x[mask], y[mask]
    sx, sy = x.std(), y.std()
    if sx == 0 or sy == 0:
        return None
    r = np.mean((x - x.mean()) * (y - y.mean())) / (sx * sy)
    return float(min(abs(r), 1.0))


def correlation_ratio(values: np.ndarray, groups: pd.Series) -> float | None:
    """η: the share of the spread of ``values`` explained by ``groups`` (square root)."""
    mask = np.isfinite(values) & groups.notna().to_numpy()
    if mask.sum() < 3:
        return None
    s = pd.Series(values[mask])
    g = groups[mask].reset_index(drop=True)
    total = float(((s - s.mean()) ** 2).sum())
    if total == 0 or g.nunique() < 2:
        return None
    means = s.groupby(g).transform("mean")
    between = float(((means - s.mean()) ** 2).sum())
    return math.sqrt(min(between / total, 1.0))


def cramers_v(a: pd.Series, b: pd.Series) -> float | None:
    """Cramér's V of two categorical series; None if undefined."""
    mask = a.notna() & b.notna()
    counts = pd.crosstab(a[mask], b[mask]).to_numpy(dtype=float)
    n = counts.sum()
    if n == 0 or min(counts.shape) < 2:
        return None
    expected = np.outer(counts.sum(axis=1), counts.sum(axis=0)) / n
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    return math.sqrt(min(chi2 / (n * (min(counts.shape) - 1)), 1.0))


def association(
    df: pd.DataFrame, kinds: dict[str, str], column: str, target: str, task: str | None
) -> tuple[float | None, str | None]:
    """``(value, method)`` of the association of ``column`` with ``target``; value None if
    undefined, method None for column kinds without a measure (text, datetime)."""
    kind = kinds[column]
    x = df[column]
    y = df[target]
    if kind in NUMBERS:
        if task == "regression":
            return pearson(x.to_numpy(dtype=float), y.to_numpy(dtype=float)), "|r|"
        return correlation_ratio(x.to_numpy(dtype=float), y), "η"
    if kind == "categorical":
        if task == "regression":
            return correlation_ratio(y.to_numpy(dtype=float), x), "η"
        return cramers_v(x, y), "Cramér's V"
    return None, None


def _training_frame(table: Table, columns: list[str], seed: int) -> pd.DataFrame:
    frame = table.df.loc[table.train_mask(), columns].reset_index(drop=True)
    if len(frame) > MAX_ROWS:
        rows = np.sort(np.random.default_rng(seed).choice(len(frame), MAX_ROWS, replace=False))
        frame = frame.iloc[rows].reset_index(drop=True)
    return frame


@register
class FeatureSelectionStep(Step):
    type = "feature_selection"
    label = "Select features"
    group = "selection"
    description = (
        "Drop features that are constant, near-duplicates of others, mostly missing or "
        "unrelated to the target, judged on the training rows. Never drops the target."
    )
    fitted = True
    params = (
        Param(
            "method",
            "select",
            "Drop features",
            default="correlation",
            options=(
                ("correlation", "That correlate strongly with another feature"),
                ("variance", "With (almost) no variance"),
                ("missing", "With many missing values"),
                ("target_correlation", "With little association with the target"),
            ),
            help=(
                "Of two strongly correlated features, the one less associated with the "
                "target is dropped (without a target, the later one). Association: |Pearson "
                "r|, the correlation ratio η or Cramér's V, from 0 (none) to 1."
            ),
        ),
        Param(
            "max_correlation",
            "number",
            "Max. |correlation|",
            default=0.95,
            min=0,
            max=1,
            step=0.01,
            visible_if={"param": "method", "values": ["correlation"]},
            help="Pairs of number features with |Pearson r| above this lose one feature.",
        ),
        Param(
            "min_variance",
            "number",
            "Min. variance",
            default=0,
            min=0,
            step=0.01,
            visible_if={"param": "method", "values": ["variance"]},
            help=(
                "Number features with a variance up to this are dropped, as are other "
                "features with a single value. The variance depends on the scale of the "
                "column; 0 drops only constant features."
            ),
        ),
        Param(
            "max_missing",
            "number",
            "Max. share missing",
            default=0.5,
            min=0,
            max=1,
            step=0.05,
            visible_if={"param": "method", "values": ["missing"]},
            help="Features missing in a larger share of the training rows (0.5 = 50 %) are dropped.",
        ),
        Param(
            "min_association",
            "number",
            "Min. association",
            default=0.05,
            min=0,
            max=1,
            step=0.01,
            visible_if={"param": "method", "values": ["target_correlation"]},
            help=(
                "Number and category features less associated with the target are dropped. "
                "Text and date columns are kept."
            ),
        ),
    )

    def clean(self, table: Table, ctx: StepContext) -> dict[str, Any]:
        params = super().clean(table, ctx)
        if params["method"] == "target_correlation":
            if not ctx.target:
                raise DataError(
                    "Selecting by association with the target needs a target. Choose the "
                    "target column under 'Target & split' first."
                )
            if ctx.target not in table.kinds:
                raise DataError(f"The target column '{ctx.target}' does not exist (any more).")
        return params

    def run(self, table: Table, params: dict[str, Any], ctx: StepContext) -> StepResult:
        target = ctx.target if ctx.target in table.kinds else None
        features = [c for c in table.columns if c != target]
        method = params["method"]
        if not features:
            return StepResult(table, "There are no feature columns; nothing changed.")
        task = None
        if target:
            # Only number targets can be regression targets.
            numeric_target = table.kinds[target] in NUMBERS
            task = (ctx.task or "regression") if numeric_target else "classification"
        dropped: dict[str, str] = {}
        if method == "variance":
            summary_rule = f"with a variance of at most {fmt_number(params['min_variance'])}"
            none_text = f"no feature has a variance of at most {fmt_number(params['min_variance'])}"
            frame = _training_frame(table, features, ctx.seed)
            for column in features:
                series = frame[column]
                if table.kinds[column] in NUMBERS:
                    values = series.to_numpy(dtype=float)
                    values = values[np.isfinite(values)]
                    variance = float(values.var()) if len(values) else 0.0
                    if variance <= params["min_variance"]:
                        dropped[column] = f"variance {fmt_number(variance)}"
                elif series.nunique() <= 1:
                    dropped[column] = "a single value"
        elif method == "missing":
            limit = params["max_missing"]
            summary_rule = f"missing in more than {fmt_share(limit, 1)} of the training rows"
            none_text = f"no feature is {summary_rule}"
            frame = table.df.loc[table.train_mask(), features]
            for column in features:
                share = float(frame[column].isna().mean()) if len(frame) else 0.0
                if share > limit:
                    dropped[column] = f"{fmt_share(share, 1)} missing"
        elif method == "correlation":
            limit = params["max_correlation"]
            summary_rule = f"correlated with another feature at |r| > {fmt_number(limit)}"
            none_text = f"no pair of features has |r| > {fmt_number(limit)}"
            numeric = [c for c in features if table.kinds[c] in NUMBERS]
            columns = numeric + ([target] if target else [])
            frame = _training_frame(table, columns, ctx.seed)
            dropped = _correlated(frame, table.kinds, numeric, target, task, limit)
        else:
            limit = params["min_association"]
            summary_rule = f"with an association below {fmt_number(limit)} with '{target}'"
            none_text = f"all have an association of at least {fmt_number(limit)} with '{target}'"
            frame = _training_frame(table, [*features, target], ctx.seed)
            for column in features:
                value, measure = association(frame, table.kinds, column, target, task)
                if measure is None:
                    continue
                if value is None or value < limit:
                    text = f"{measure} {fmt_number(value)}" if value is not None else "constant"
                    dropped[column] = text

        drop = [c for c in features if c in dropped]
        details = [{"column": c, "text": dropped[c]} for c in drop]
        if drop:
            summary = (
                f"Dropped {len(drop)} of {plural(len(features), 'feature')} {summary_rule}: "
                f"{name_list(drop)}."
            )
        else:
            summary = f"Kept all {plural(len(features), 'feature')}: {none_text}."
        return StepResult(table.drop(drop), summary, details, {"drop": drop})

    def to_pandas(
        self, params: dict[str, Any], fitted: dict[str, Any] | None, ctx: StepContext
    ) -> list[str]:
        drop = (fitted or {}).get("drop", [])
        if not drop:
            return ["# No feature dropped."]
        return [f"df = df.drop(columns={drop!r})"]


def _correlated(
    frame: pd.DataFrame,
    kinds: dict[str, str],
    numeric: list[str],
    target: str | None,
    task: str | None,
    limit: float,
) -> dict[str, str]:
    """Greedy: strongest pairs first; of each pair still complete, drop the feature less
    associated with the target (or the later one)."""
    if len(numeric) < 2:
        return {}
    corr = frame[numeric].corr().to_numpy()
    pairs = [
        (abs(corr[i, j]), i, j)
        for i in range(len(numeric))
        for j in range(i + 1, len(numeric))
        if np.isfinite(corr[i, j]) and abs(corr[i, j]) > limit
    ]
    pairs.sort(key=lambda p: (-p[0], p[1], p[2]))
    scores: dict[str, float | None] = {}
    if target and pairs:
        for column in {numeric[i] for _, i, _ in pairs} | {numeric[j] for _, _, j in pairs}:
            scores[column] = association(frame, kinds, column, target, task)[0]
    dropped: dict[str, str] = {}
    for r, i, j in pairs:
        a, b = numeric[i], numeric[j]
        if a in dropped or b in dropped:
            continue
        sa, sb = scores.get(a), scores.get(b)
        drop, keep = (b, a)
        if sa is not None and sb is not None and sa < sb:
            drop, keep = (a, b)
        dropped[drop] = f"|r| = {fmt_number(r)} with '{keep}'"
    return dropped
