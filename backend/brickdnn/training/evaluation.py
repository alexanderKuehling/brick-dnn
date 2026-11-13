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

Binary classification (two classes) additionally
    a threshold curve per split from the probability of the positive class (the second
    class, index 1): ROC AUC, PR AUC (average precision, as scikit-learn computes it) and at
    most :data:`MAX_CURVE_POINTS` operating points "predicted positive when p(positive) >=
    threshold" with their precision, recall, false-positive rate and confusion counts. The
    thresholds are chosen from one split, the validation rows (without them the test rows,
    without either the training rows; ``binary.source`` says which): a grid 0.00, 0.01 …
    1.00, the probabilities of evenly spaced positive and negative rows of that split, and
    the suggestions. The other splits get their curve at the same thresholds, so a threshold
    picked on validation can be checked on test. The suggestions come from the chosen split
    only: the threshold with the best F1 score and the highest threshold that still finds
    :data:`RECALL_TARGET` of the positives. See :func:`threshold_analysis`.

Regression (one value output)
    MAE, MSE, RMSE, R², MAPE (only if no target is close to 0, see :func:`mape`), the median
    and the largest absolute error, quantiles of the residuals (predicted − actual) and a
    sample of at most :data:`MAX_SAMPLE` predicted and actual values (for a scatter plot),
    all in the units of the target column.
"""

from __future__ import annotations

import math
import traceback
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
# Binary classification, the threshold curve: a grid of thresholds 0.00, 0.01 … 1.00, plus
# the probabilities of this many evenly spaced positive and negative rows each (of the split
# the thresholds are chosen from; where the curve changes), plus the two suggestions.
THRESHOLD_GRID = 101
CURVE_QUANTILES = 48
MAX_CURVE_POINTS = THRESHOLD_GRID + 2 * CURVE_QUANTILES + 2
# The second suggestion: the highest threshold that still finds this share of the positives.
RECALL_TARGET = 0.8
# The class whose probability is compared with the threshold (the second one).
POSITIVE_CLASS = 1
# The splits the thresholds may be chosen from, in this order.
THRESHOLD_SOURCES = ("val", "test", "training")


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


class SplitResult(dict):
    """The metrics of a split (a plain dict in state.json). A binary classification also
    carries the probabilities of the positive class and the true classes of its rows
    (``binary_scores``), from which :func:`evaluation` makes the threshold curves."""

    binary_scores: tuple[np.ndarray, np.ndarray] | None = None


class ClassificationCounter:
    """The confusion matrix, the log loss and the top-k hits of one split, added batch by
    batch (so no probabilities need to be kept; with two classes the probability of the
    positive class of every row is kept, for the threshold curve)."""

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.log_loss_sum = 0.0
        self.top_k = {k: 0 for k in TOP_K if num_classes > k}
        self.count = 0
        self.binary = num_classes == 2
        self.scores: list[np.ndarray] = []
        self.labels: list[np.ndarray] = []

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
        if self.binary:
            self.scores.append(proba[:, POSITIVE_CLASS].copy())
            self.labels.append(true.copy())
        self.count += len(true)

    def result(self, loss: float | None = None) -> SplitResult:
        """The metrics of the split (see :func:`classification_metrics`)."""
        result = SplitResult(classification_metrics(self.matrix))
        if self.binary and self.scores:
            result.binary_scores = (np.concatenate(self.scores), np.concatenate(self.labels))
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
# Binary classification: decision thresholds
# --------------------------------------------------------------------------- #


def _floor_short(value: float, digits: int = 6) -> float:
    """``value`` (0 to 1) with ``digits`` significant digits, rounded down: a threshold
    rounded like this still includes the row whose probability it was taken from."""
    value = float(value)
    if not value > 0:
        return 0.0
    short = float(f"{value:.{digits}g}")
    if short > value:
        step = 10 ** (math.floor(math.log10(value)) - digits + 1)
        short = float(f"{short - step:.{digits}g}")
    return max(short, 0.0)


class _Ranked:
    """The positive-class probabilities of one split, sorted per true class, to count the
    rows at or above any threshold. Rows without a probability (NaN from a diverged network)
    are left out."""

    def __init__(self, scores: np.ndarray, labels: np.ndarray):
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        known = np.isfinite(scores)
        self.scores = scores[known]
        self.positive = (np.asarray(labels).reshape(-1) == POSITIVE_CLASS)[known]
        self.pos = np.sort(self.scores[self.positive])
        self.neg = np.sort(self.scores[~self.positive])
        self.P = len(self.pos)
        self.N = len(self.neg)

    def counts(self, thresholds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """True and false positives per threshold, with the rule ``p >= threshold``."""
        tp = self.P - np.searchsorted(self.pos, thresholds, side="left")
        fp = self.N - np.searchsorted(self.neg, thresholds, side="left")
        return tp.astype(np.int64), fp.astype(np.int64)

    def steps(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Every distinct probability (highest first) with the true and false positives
        when it is the threshold, like scikit-learn's ``precision_recall_curve``."""
        order = np.argsort(-self.scores, kind="mergesort")
        scores = self.scores[order]
        hits = self.positive[order].astype(np.int64)
        last = np.r_[np.flatnonzero(np.diff(scores)), len(scores) - 1]
        tp = np.cumsum(hits)[last]
        return scores[last], tp, (last + 1) - tp

    def roc_auc(self) -> float | None:
        """The area under the ROC curve (ties count half); None without both classes."""
        if not self.P or not self.N:
            return None
        order = np.argsort(self.scores, kind="mergesort")
        ordered = self.scores[order]
        starts = np.r_[0, np.flatnonzero(np.diff(ordered)) + 1]
        ends = np.r_[starts[1:], len(ordered)]
        ranks = np.empty(len(ordered), dtype=np.float64)
        ranks[order] = np.repeat((starts + ends + 1) / 2.0, ends - starts)
        total = float(ranks[self.positive].sum())
        return _float((total - self.P * (self.P + 1) / 2.0) / (self.P * self.N))

    def average_precision(self) -> float | None:
        """The area under the precision-recall curve as scikit-learn's
        ``average_precision_score`` computes it; None without positive rows."""
        if not self.P:
            return None
        _, tp, fp = self.steps()
        recall = tp / self.P
        return _float(np.sum(np.diff(np.r_[0.0, recall]) * tp / (tp + fp)))

    def point(self, threshold: float) -> dict[str, Any]:
        """Precision, recall, F1, the false-positive rate and the counts at ``threshold``."""
        found = self.counts(np.array([threshold], dtype=np.float64))
        tp, fp = int(found[0][0]), int(found[1][0])
        fn, tn = self.P - tp, self.N - fp
        return {
            "threshold": threshold,
            "precision": _short(tp / (tp + fp), 4) if tp + fp else None,
            "recall": _short(tp / self.P, 4) if self.P else None,
            "f1": _short(2 * tp / (2 * tp + fp + fn), 4) if tp + fp + fn else None,
            "fpr": _short(fp / self.N, 4) if self.N else None,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        }


def suggested_thresholds(ranked: _Ranked) -> dict[str, float | None]:
    """``best_f1``: the threshold with the best F1 score (the highest of equally good ones);
    ``recall``: the highest threshold that still finds RECALL_TARGET of the positive rows.
    None without positive rows."""
    if not ranked.P:
        return {"best_f1": None, "recall": None}
    scores, tp, fp = ranked.steps()
    f1 = 2 * tp / (tp + fp + ranked.P)
    best = int(np.argmax(f1))  # the first maximum: the highest threshold
    enough = np.flatnonzero(tp >= math.ceil(RECALL_TARGET * ranked.P - 1e-9))
    return {
        "best_f1": _floor_short(scores[best]),
        "recall": _floor_short(scores[int(enough[0])]) if len(enough) else None,
    }


def curve_thresholds(ranked: _Ranked, extra: list[float | None]) -> np.ndarray:
    """The thresholds of the curves (see the module docstring), ascending and distinct."""
    picked = np.round(np.linspace(0.0, 1.0, THRESHOLD_GRID), 2).tolist()
    picked += [t for t in extra if t is not None]
    for values in (ranked.pos, ranked.neg):
        if len(values):
            rows = np.unique(np.linspace(0, len(values) - 1, CURVE_QUANTILES).round())
            picked += [_floor_short(values[int(i)]) for i in rows]
    return np.unique(np.clip(np.array(picked, dtype=np.float64), 0.0, 1.0))


def threshold_curve(ranked: _Ranked, thresholds: np.ndarray) -> dict[str, Any]:
    """The ``threshold_curve`` of a split::

        {"positive_class": 1, "positives": 20, "negatives": 1180,
         "roc_auc": 0.95, "pr_auc": 0.71,        # pr_auc: the average precision
         "points": {"threshold": [0.0, 0.01, ...],   # ascending; predicted positive:
                    "precision": [...],              #   p(positive) >= threshold
                    "recall": [...], "fpr": [...],   # null where undefined
                    "tp": [...], "fp": [...], "tn": [...], "fn": [...]}}

    The AUCs are null without rows of both classes (PR AUC: without positive rows).
    """
    tp, fp = ranked.counts(thresholds)
    P, N = ranked.P, ranked.N
    return {
        "positive_class": POSITIVE_CLASS,
        "positives": P,
        "negatives": N,
        "roc_auc": ranked.roc_auc(),
        "pr_auc": ranked.average_precision(),
        "points": {
            "threshold": [float(t) for t in thresholds],
            "precision": [
                _short(int(t) / int(t + f), 4) if t + f else None
                for t, f in zip(tp, fp, strict=True)
            ],
            "recall": [_short(int(t) / P, 4) if P else None for t in tp],
            "fpr": [_short(int(f) / N, 4) if N else None for f in fp],
            "tp": tp.tolist(),
            "fp": fp.tolist(),
            "tn": (N - fp).tolist(),
            "fn": (P - tp).tolist(),
        },
    }


def threshold_analysis(
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """The ``binary`` part of the evaluation and the threshold curve of every split in
    ``splits`` (``{"val": (probabilities of the positive class, true classes), ...}``). The
    thresholds and the suggestions come from the first split of THRESHOLD_SOURCES that is
    there (``source``)::

        {"positive_class": 1, "recall_target": 0.8, "source": "val",
         "best_f1": {"threshold": 0.31, "precision": 0.8, "recall": 0.7, "f1": 0.75,
                     "fpr": 0.003, "tp": 14, "fp": 4, "tn": 1176, "fn": 6},
         "recall": {...}}         # the highest threshold with recall >= recall_target

    Both suggestions are measured on the source split (null without positive rows in it).
    """
    source = next(key for key in THRESHOLD_SOURCES if key in splits)
    ranked = {key: _Ranked(*found) for key, found in splits.items()}
    chosen = ranked[source]
    suggested = suggested_thresholds(chosen)
    thresholds = curve_thresholds(chosen, list(suggested.values()))
    binary: dict[str, Any] = {
        "positive_class": POSITIVE_CLASS,
        "recall_target": RECALL_TARGET,
        "source": source,
    }
    for key, value in suggested.items():
        binary[key] = chosen.point(value) if value is not None else None
    curves = {key: threshold_curve(found, thresholds) for key, found in ranked.items()}
    return binary, curves


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
        scores = {
            key: found
            for key in THRESHOLD_SOURCES
            if (found := getattr(result[key], "binary_scores", None)) is not None
        }
        if scores:
            try:
                binary, curves = threshold_analysis(scores)
            except Exception:  # never lose the rest of the evaluation over the curves
                traceback.print_exc()
            else:
                result["binary"] = binary
                for key, curve in curves.items():
                    result[key] = {**result[key], "threshold_curve": curve}
    return result


def class_names_for(names: list[str] | None, num_classes: int) -> list[str]:
    """The class names of a run, numbered where they are missing."""
    found = [str(name) for name in names or []][:num_classes]
    return found + [str(i) for i in range(len(found), num_classes)]
