"""Preparation steps: the target column (class balancing)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..table import DataError, Table
from .base import Param, Step, StepContext, StepResult, register
from .encode import class_labels, class_name
from .outliers import fmt_count, fmt_share, plural

# Oversampling refuses to add more rows than this.
MAX_ADDED_ROWS = 5_000_000


def classification_target(table: Table, ctx: StepContext, what: str) -> str:
    """The target column for a step that needs classes; raises a helpful DataError."""
    target = ctx.target
    if not target:
        raise DataError(
            f"{what} needs a target. Choose the target column under 'Target & split' first."
        )
    if target not in table.kinds:
        raise DataError(f"The target column '{target}' does not exist (any more).")
    if ctx.task == "regression":
        raise DataError(f"{what} only works for classification; '{target}' is a regression target.")
    return target


@register
class BalanceClassesStep(Step):
    type = "balance_classes"
    label = "Balance classes"
    group = "target"
    description = (
        "Even out the class counts of the training rows by repeating rows of rare classes or "
        "dropping rows of common ones. Validation and test rows stay as they are."
    )
    training_only = True
    params = (
        Param(
            "method",
            "select",
            "Method",
            default="oversample",
            options=(
                ("oversample", "Oversample (repeat rows of smaller classes)"),
                ("undersample", "Undersample (drop rows of larger classes)"),
            ),
            help=(
                "Oversampling draws random copies of training rows until every class has as "
                "many rows as the largest one. Undersampling keeps a random subset of every "
                "class as large as the smallest one. Both are seeded with the split seed."
            ),
        ),
    )

    def run(self, table: Table, params: dict[str, Any], ctx: StepContext) -> StepResult:
        target = classification_target(table, ctx, "Balancing")
        y = table.df[target]
        train = table.train_mask()
        labelled = train & y.notna().to_numpy()
        classes = class_labels(y[labelled])
        if len(classes) < 2:
            raise DataError(f"The training rows have only one class of '{target}'.")
        positions = {c: np.flatnonzero(labelled & (y == c).to_numpy()) for c in classes}
        counts = {c: len(p) for c, p in positions.items()}
        rng = np.random.default_rng(ctx.seed)
        warnings = []
        unlabelled = int(train.sum() - labelled.sum())
        if unlabelled:
            warnings.append(
                f"Training rows without a target value ({fmt_count(unlabelled)}) were left as "
                "they are."
            )
        if len(set(counts.values())) == 1:
            size = next(iter(counts.values()))
            summary = (
                f"The classes are already balanced ({plural(size, 'training row')} each); "
                "nothing changed."
            )
            return StepResult(table, summary, warnings=warnings)

        if params["method"] == "oversample":
            size = max(counts.values())
            added = sum(size - n for n in counts.values())
            if added > MAX_ADDED_ROWS:
                raise DataError(
                    f"Oversampling would add {fmt_count(added)} rows. Undersample instead, or "
                    "merge small classes first."
                )
            extra = [rng.choice(p, size - len(p), replace=True) for p in positions.values()]
            rows = np.concatenate([np.arange(table.n_rows), *extra])
            changed = int(rows.size - table.n_rows)
            summary = (
                f"Oversampled the training rows to {fmt_count(size)} per class: added "
                f"{plural(changed, 'copy', 'copies')} of rows of "
                f"{plural(sum(n < size for n in counts.values()), 'class', 'classes')}."
            )
        else:
            size = min(counts.values())
            keep = np.ones(table.n_rows, dtype=bool)
            for p in positions.values():
                keep[p] = False
                keep[rng.choice(p, size, replace=False)] = True
            rows = np.flatnonzero(keep)
            changed = int(table.n_rows - rows.size)
            n_train = int(train.sum())
            summary = (
                f"Undersampled the training rows to {fmt_count(size)} per class: removed "
                f"{plural(changed, 'row')} ({fmt_share(changed, n_train)} of the training rows)."
            )
            if changed > n_train / 2:
                warnings.append(
                    "More than half of the training rows were removed; oversampling keeps "
                    "all of them."
                )
        details = [
            {
                "column": target,
                "text": f"{class_name(c)}: {fmt_count(counts[c])} → {fmt_count(size)} rows",
            }
            for c in classes[:50]
        ]
        return StepResult(table.take(rows), summary, details, None, warnings)

    def to_pandas(
        self, params: dict[str, Any], fitted: dict[str, Any] | None, ctx: StepContext
    ) -> list[str]:
        how = "oversampled" if params.get("method") != "undersample" else "undersampled"
        return [f"# The training rows were {how}; new data is not changed."]
