"""The objective metric of a hyperparameter search: what ranks its trials (no PyTorch or
NumPy needed to read the definitions).

Every search ranks its trials by one metric on the rows that compare them: the validation
rows, or with cross-validation (classical models) the out-of-fold predictions on the
training rows. The default is the loss (the cross-entropy of the predicted probabilities,
i.e. the log loss, or the mean squared error of the standardised target), which is lower
for better models. The other metrics need a single class output (classification) or a
single value output (regression), like the final evaluation (see evaluation.py), and their
direction follows from the metric: accuracy, balanced accuracy, macro F1, ROC AUC, PR AUC
and R² are maximised, RMSE and MAE minimised. ROC AUC and PR AUC need two classes.

Within a neural network's trial, the kept epoch is still the one with the lowest
validation loss (early stopping and *Keep the best weights* work as before); the metric
only ranks the trials.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .config import ConfigError


@dataclass(frozen=True)
class Metric:
    name: str
    label: str  # "ROC AUC", in sentences and table headers
    maximize: bool
    task: str | None  # classification | regression | None (any task)
    binary: bool = False  # needs exactly two classes
    percent: bool = False  # shown as a percentage

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "direction": "maximize" if self.maximize else "minimize",
            "task": self.task,
            "binary": self.binary,
        }


DEFAULT_METRIC = "loss"
METRICS: dict[str, Metric] = {
    metric.name: metric
    for metric in (
        Metric("loss", "loss", False, None),
        Metric("accuracy", "accuracy", True, "classification", percent=True),
        Metric("balanced_accuracy", "balanced accuracy", True, "classification", percent=True),
        Metric("macro_f1", "macro F1", True, "classification"),
        Metric("roc_auc", "ROC AUC", True, "classification", binary=True),
        Metric("pr_auc", "PR AUC", True, "classification", binary=True),
        Metric("rmse", "RMSE", False, "regression"),
        Metric("mae", "MAE", False, "regression"),
        Metric("r2", "R²", True, "regression"),
    )
}
# Metrics that the history of a neural network already has for every epoch (no extra pass).
HISTORY_KEYS = {
    "loss": "val_loss",
    "accuracy": "val_accuracy",
    "rmse": "val_rmse",
    "r2": "val_r2",
}


def parse_metric(value: Any) -> str:
    """The metric name of a request (default: the loss). Raises :class:`ConfigError`."""
    if value is None or value == "":
        return DEFAULT_METRIC
    name = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if name in ("log_loss", "val_loss"):
        name = "loss"
    if name not in METRICS:
        raise ConfigError(
            f"Hyperparameter search: the metric '{value}' is unknown. Possible: "
            f"{', '.join(METRICS)}."
        )
    return name


def label(name: str) -> str:
    return METRICS[name].label if name in METRICS else name


def evaluation_task(plan: Any) -> str | None:
    """``classification`` or ``regression`` if the run has a single class output or a single
    value (the runs with a final evaluation), otherwise None (segmentation, several outputs).
    ``plan``: a :class:`~.compat.TrainingPlan` or its dict."""
    targets = plan.get("targets") if isinstance(plan, dict) else getattr(plan, "targets", None)
    if not isinstance(targets, list) or len(targets) != 1 or not isinstance(targets[0], dict):
        return None
    target = targets[0]
    if target.get("kind") == "class":
        return "classification"
    if target.get("kind") == "value" and math.prod(target.get("shape") or [1]) == 1:
        return "regression"
    return None


def _num_classes(plan: Any) -> int | None:
    if isinstance(plan, dict):
        return plan.get("num_classes")
    return getattr(plan, "num_classes", None)


def available(plan: Any) -> list[str]:
    """The metrics a search on ``plan`` can rank its trials by."""
    task = evaluation_task(plan)
    binary = _num_classes(plan) == 2
    return [
        name
        for name, metric in METRICS.items()
        if metric.task is None or (metric.task == task and (binary or not metric.binary))
    ]


def metric_error(name: str, plan: Any) -> str | None:
    """Why ``name`` cannot rank the trials of a search on ``plan`` (None: it can)."""
    if name in available(plan):
        return None
    metric = METRICS[name]
    task = evaluation_task(plan)
    possible = ", ".join(available(plan))
    if metric.binary and task == "classification":
        why = f"needs two classes, this data has {_num_classes(plan)}"
    elif task is None:
        why = "needs a network with a single class or value output (not a segmentation)"
    else:
        why = f"is for {metric.task}, and this is a {task}"
    return f"The objective metric {metric.label} {why}. Possible: {possible}."


def better(value: float | None, best: float | None, name: str) -> bool:
    """True if ``value`` beats ``best`` (None, e.g. an undefined R², never wins against a
    number)."""
    if value is None or not math.isfinite(value):
        return False
    if best is None or not math.isfinite(best):
        return True
    return value > best if METRICS[name].maximize else value < best


def history_value(name: str, entry: dict[str, Any]) -> float | None:
    """The metric from a history entry of a neural network (None if it is not there)."""
    key = HISTORY_KEYS.get(name)
    value = entry.get(key) if key is not None else None
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def split_value(name: str, split: Any) -> float | None:
    """The metric from a split of the final evaluation (see evaluation.py): a classification
    split (with ``binary_scores`` for ROC and PR AUC) or a regression split."""
    if name in ("roc_auc", "pr_auc"):
        scores = getattr(split, "binary_scores", None)
        if scores is None:
            return None
        from .evaluation import _Ranked

        ranked = _Ranked(*scores)
        return ranked.roc_auc() if name == "roc_auc" else ranked.average_precision()
    value = split.get(name) if isinstance(split, dict) else None
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def format_value(name: str, value: float | None) -> str:
    """``"93.1%"``, ``"0.912"`` or ``"–"``."""
    if value is None:
        return "–"
    metric = METRICS.get(name)
    if metric is not None and metric.percent:
        return f"{value * 100:.1f}%"
    return f"{value:.4g}"


def capabilities() -> dict[str, Any]:
    """The metrics for the training options of the API."""
    return {
        "default_metric": DEFAULT_METRIC,
        "metrics": [metric.to_dict() for metric in METRICS.values()],
    }
