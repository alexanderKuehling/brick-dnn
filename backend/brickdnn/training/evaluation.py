"""The final evaluation of a run: metrics of the kept weights (or the fitted model) on the
validation and the test rows (without either: on the training rows), computed once at the
end (state.json key ``evaluation``, see state.py). Needs NumPy, not PyTorch or scikit-learn: both engines hand over plain arrays.

Classification (one class output)
    a confusion matrix (rows: true class, columns: predicted class; counts), per class the
    precision, recall, F1 score and support, and overall the accuracy, the balanced accuracy
    (mean recall over the classes that occur), the macro and the weighted F1 score, the log
    loss and the top-k accuracy (only for more than k classes). Undefined values follow
    scikit-learn with ``zero_division=0`` in the averages: a class without predictions has no
    precision (stored as null, counted as 0), a class that does not occur in the split has no
    recall. Macro averages run over the classes that occur in the true or the predicted
    labels, like scikit-learn does without ``labels``. For more than
    :data:`MAX_MATRIX_CLASSES` classes the matrix is not stored (``confusion_matrix`` is null);
    the most confused pairs are stored for every run.

Regression (one value output)
    MAE, MSE, RMSE, R², MAPE (only if no target is close to 0, see :func:`mape`), the median
    and the largest absolute error, quantiles of the residuals (predicted − actual) and a
    sample of at most :data:`MAX_SAMPLE` predicted and actual values (for a scatter plot),
    all in the units of the target column.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

# Above this many classes the confusion matrix is not stored (a 50 × 50 matrix is 2,500
# numbers per split; the per-class metrics are kept for any number of classes).
MAX_MATRIX_CLASSES = 50
# Top-k accuracies, each only with more than k classes.
TOP_K = (3, 5)
# The most confused pairs (true class, predicted class) kept per split.
MAX_CONFUSED = 10
# Points of the predicted-vs-actual sample of a regression, per split.
MAX_SAMPLE = 500
# MAPE is left out when an actual value is closer to 0 than this share of the mean absolute
# actual value: the relative error of such a row would dominate the mean.
MAPE_MIN_SHARE = 0.01
# Probabilities are clipped to this before their logarithm is taken (log loss).
EPSILON = 1e-15
RESIDUAL_QUANTILES = {"q05": 0.05, "q25": 0.25, "median": 0.5, "q75": 0.75, "q95": 0.95}


def _float(value: Any) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _short(value: float | None, digits: int = 6) -> float | None:
    """``value`` with ``digits`` significant digits (keeps lists in state.json small)."""
    if value is None or not math.isfinite(value):
        return None
    return float(f"{value:.{digits}g}")


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #


class ClassificationCounter:
    """The confusion matrix, the log loss and the top-k hits of one split, added batch by
    batch (so no probabilities need to be kept)."""

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.log_loss_sum = 0.0
        self.top_k = {k: 0 for k in TOP_K if num_classes > k}
        self.count = 0

    def add(self, probabilities: np.ndarray, targets: np.ndarray) -> None:
        """One batch: the predicted probabilities (rows × classes) and the true classes."""
        proba = np.asarray(probabilities, dtype=np.float64).reshape(len(targets), -1)
        true = np.asarray(targets, dtype=np.int64).reshape(-1)
        if len(true) == 0:
            return
        predicted = np.argmax(proba, axis=1)
        k = self.num_classes
        self.matrix += np.bincount(true * k + predicted, minlength=k * k).reshape(k, k)
        p_true = proba[np.arange(len(true)), true]
        self.log_loss_sum += float(-np.log(np.clip(p_true, EPSILON, 1.0)).sum())
        if self.top_k:
            # How many classes are more probable than the true one (ties count for it).
            rank = (proba > p_true[:, None]).sum(axis=1)
            for top in self.top_k:
                self.top_k[top] += int((rank < top).sum())
        self.count += len(true)

    def result(self, loss: float | None = None) -> dict[str, Any]:
        """The metrics of the split (see :func:`classification_metrics`)."""
        result = classification_metrics(self.matrix)
        result["loss"] = _float(loss) if loss is not None else None
        result["log_loss"] = _float(self.log_loss_sum / self.count) if self.count else None
        result["top_k_accuracy"] = (
            {str(k): hits / self.count for k, hits in self.top_k.items()}
            if self.top_k and self.count
            else None
        )
        return result


def classification_metrics(matrix: np.ndarray) -> dict[str, Any]:
    """Metrics from a confusion matrix (rows: true class, columns: predicted class)::

        {"size": 120, "accuracy": 0.9, "balanced_accuracy": 0.88, "macro_f1": 0.87,
         "weighted_f1": 0.9,
         "per_class": {"precision": [...], "recall": [...], "f1": [...], "support": [...]},
         "confusion_matrix": [[...], ...],        # null above MAX_MATRIX_CLASSES classes
         "most_confused": [{"true": 2, "predicted": 1, "count": 5}, ...]}

    Per class, precision is null without predictions of the class, recall without rows of
    it, and F1 if neither exists; the averages count undefined values as 0.
    """
    matrix = np.asarray(matrix, dtype=np.int64)
    k = matrix.shape[0]
    total = int(matrix.sum())
    hits = np.diag(matrix).astype(np.float64)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    precision: list[float | None] = []
    recall: list[float | None] = []
    f1: list[float | None] = []
    for c in range(k):
        p = hits[c] / predicted[c] if predicted[c] else None
        r = hits[c] / support[c] if support[c] else None
        precision.append(p)
        recall.append(r)
        if p is None and r is None:
            f1.append(None)
        else:
            p0, r0 = p or 0.0, r or 0.0
            f1.append(2 * p0 * r0 / (p0 + r0) if p0 + r0 > 0 else 0.0)
    occurring = [c for c in range(k) if support[c] or predicted[c]]
    with_rows = [c for c in range(k) if support[c]]
    macro_f1 = sum(f1[c] or 0.0 for c in occurring) / len(occurring) if occurring else None
    weighted_f1 = (
        sum((f1[c] or 0.0) * int(support[c]) for c in with_rows) / total if total else None
    )
    balanced = sum(recall[c] or 0.0 for c in with_rows) / len(with_rows) if with_rows else None
    pairs = [
        (int(matrix[t, p]), t, p) for t in range(k) for p in range(k) if t != p and matrix[t, p] > 0
    ]
    pairs.sort(key=lambda pair: (-pair[0], pair[1], pair[2]))
    return {
        "size": total,
        "accuracy": float(hits.sum() / total) if total else None,
        "balanced_accuracy": balanced,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": {
            "precision": [_short(v) for v in precision],
            "recall": [_short(v) for v in recall],
            "f1": [_short(v) for v in f1],
            "support": [int(v) for v in support],
        },
        "confusion_matrix": matrix.tolist() if k <= MAX_MATRIX_CLASSES else None,
        "most_confused": [
            {"true": t, "predicted": p, "count": count} for count, t, p in pairs[:MAX_CONFUSED]
        ],
    }


# --------------------------------------------------------------------------- #
# Regression
# --------------------------------------------------------------------------- #


class RegressionCollector:
    """The predicted and the actual values of one split (in the units of the target),
    collected batch by batch."""

    def __init__(self) -> None:
        self.predicted: list[np.ndarray] = []
        self.actual: list[np.ndarray] = []

    def add(self, predicted: np.ndarray, actual: np.ndarray) -> None:
        self.predicted.append(np.asarray(predicted, dtype=np.float64).reshape(-1))
        self.actual.append(np.asarray(actual, dtype=np.float64).reshape(-1))

    def result(self, loss: float | None = None) -> dict[str, Any]:
        predicted = np.concatenate(self.predicted) if self.predicted else np.empty(0)
        actual = np.concatenate(self.actual) if self.actual else np.empty(0)
        result = regression_metrics(predicted, actual)
        result["loss"] = _float(loss) if loss is not None else None
        return result


def mape(predicted: np.ndarray, actual: np.ndarray) -> float | None:
    """The mean absolute percentage error as a share (0.05 = 5 %), or None when an actual
    value is 0 or closer to 0 than :data:`MAPE_MIN_SHARE` of the mean absolute actual
    value."""
    size = np.abs(actual)
    if len(actual) == 0:
        return None
    mean = float(size.mean())
    if not mean > 0 or float(size.min()) < MAPE_MIN_SHARE * mean:
        return None
    return _float(np.mean(np.abs(predicted - actual) / size))


def regression_metrics(predicted: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    """Metrics of predicted against actual values (same units)::

        {"size": 120, "mae": 1.2, "mse": 2.3, "rmse": 1.52, "r2": 0.91, "mape": 0.04,
         "median_absolute_error": 0.9, "max_error": 6.1,
         "residuals": {"q05": -2.4, "q25": -0.8, "median": 0.1, "q75": 0.9, "q95": 2.6},
         "sample": {"actual": [...], "predicted": [...]}}

    R² is null if the actual values are all equal; MAPE see :func:`mape`. Residuals are
    predicted − actual. The sample takes evenly spaced rows of the split.
    """
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    n = len(actual)
    if n == 0:
        return {"size": 0, "mae": None, "mse": None, "rmse": None, "r2": None, "mape": None}
    residuals = predicted - actual
    errors = np.abs(residuals)
    mse = float(np.mean(residuals**2))
    spread = float(np.sum((actual - actual.mean()) ** 2))
    r2 = 1 - float(np.sum(residuals**2)) / spread if spread > 0 and np.ptp(actual) > 0 else None
    if n > MAX_SAMPLE:
        rows = np.unique(np.linspace(0, n - 1, MAX_SAMPLE).round().astype(np.int64))
    else:
        rows = np.arange(n)
    quantiles = np.quantile(residuals, list(RESIDUAL_QUANTILES.values()))
    return {
        "size": n,
        "mae": _float(errors.mean()),
        "mse": _float(mse),
        "rmse": _float(math.sqrt(mse)),
        "r2": _float(r2) if r2 is not None else None,
        "mape": mape(predicted, actual),
        "median_absolute_error": _float(np.median(errors)),
        "max_error": _float(errors.max()),
        "residuals": {
            name: _float(value) for name, value in zip(RESIDUAL_QUANTILES, quantiles, strict=True)
        },
        "sample": {
            "actual": [_short(float(v)) for v in actual[rows]],
            "predicted": [_short(float(v)) for v in predicted[rows]],
        },
    }


# --------------------------------------------------------------------------- #
# The whole evaluation
# --------------------------------------------------------------------------- #


def evaluation(
    task: str,
    val: dict[str, Any] | None,
    test: dict[str, Any] | None,
    train: dict[str, Any] | None,
    *,
    class_names: list[str] | None = None,
    epoch: int | None = None,
    trial: int | None = None,
    training: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The ``evaluation`` of state.json (see state.py): ``task`` is "classification" or
    "regression", ``val`` and ``test`` the results of the splits (None if a split has no
    rows), ``train`` the loss (and accuracy) of the kept epoch on the training rows.
    ``training``: a run without validation and test rows is measured on its training rows
    instead, a split result like ``val`` (optimistic: the model has seen these rows)."""
    result: dict[str, Any] = {
        "task": task,
        "epoch": epoch,
        "trial": trial,
        "train": train,
        "val": val,
        "test": test,
        "training": training,
    }
    if task == "classification":
        result["class_names"] = [str(name) for name in class_names or []]
        result["matrix_classes_max"] = MAX_MATRIX_CLASSES
    return result


def class_names_for(names: list[str] | None, num_classes: int) -> list[str]:
    """The class names of a run, numbered where they are missing."""
    found = [str(name) for name in names or []][:num_classes]
    return found + [str(i) for i in range(len(found), num_classes)]
