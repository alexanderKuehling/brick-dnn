"""Ensembles: one model from several finished runs (no PyTorch needed; NumPy, and
scikit-learn only to fit a stacking model).

Every member predicts the same rows: class probabilities (rows × classes) for a
classification, values in the units of the target (rows) for a regression. Two methods
combine them:

``average``
    The weighted average of the members' probabilities or values. The weights are not
    negative and add up to 1 (equal by default).

``stacking``
    A small meta-model on the members' predictions, fitted on the validation rows: a logistic
    regression for a classification (on the probability of the second class per member with
    two classes, on all probabilities otherwise) and a ridge regression for a regression on
    the predicted values, both with scikit-learn's defaults. It is measured on the test rows;
    on the validation rows it was fitted on, so those metrics are optimistic. The fitted
    coefficients are kept as plain numbers (:func:`fit_stacking`), so applying the model
    needs NumPy only (:func:`apply_stacking`).

The evaluation of the combined predictions is the one of a trained run (evaluation.py),
including the threshold analysis of a binary classification. :func:`ensemble_result` also
measures every member on the same rows, for the comparison table.

The training of an ensemble (ensemble_engine.py) keeps the members' predictions of the
validation and test rows in ``ensemble.npz`` (:func:`save_outputs`), so new weights are
measured again without the members (:func:`ensemble_result`).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .evaluation import (
    ClassificationCounter,
    RegressionCollector,
    binary_aucs,
    class_names_for,
    evaluation,
)

METHODS = ("average", "stacking")
METHOD_LABELS = {"average": "Weighted average", "stacking": "Stacking"}
# Splits of the evaluation, in this order. "training": a dataset without validation and test
# rows is measured on its training rows (optimistic), like a trained run.
SPLITS = ("val", "test", "training")
MIN_MEMBERS = 2
MAX_MEMBERS = 10
# The class whose probability a binary stacking model uses per member (as the threshold).
POSITIVE_CLASS = 1
STACKING_NEEDS_VALIDATION = (
    "Stacking fits its model on the validation rows, and this dataset has none. Use the "
    "weighted average, or give the split validation rows in the Data tab and export the "
    "dataset again."
)


class EnsembleError(ValueError):
    """The members cannot be combined like this. The message is for the user."""


# --------------------------------------------------------------------------- #
# Weights and methods
# --------------------------------------------------------------------------- #


def check_method(method: Any) -> str:
    if method is None:
        return "average"
    if method not in METHODS:
        raise EnsembleError(f"The method must be one of: {', '.join(METHODS)}.")
    return method


def normalized_weights(weights: Any, count: int) -> list[float]:
    """The weights of ``count`` members, scaled to add up to 1 (None: equal weights).
    Raises :class:`EnsembleError` for anything but ``count`` numbers of at least 0 with a
    sum above 0."""
    if weights is None:
        return [1.0 / count] * count
    if not isinstance(weights, list) or len(weights) != count:
        raise EnsembleError(f"Give one weight per member ({count} numbers).")
    numbers = []
    for value in weights:
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                value = None
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise EnsembleError("The weights must be numbers of at least 0.")
        numbers.append(float(value))
    total = sum(numbers)
    if not total > 0:
        raise EnsembleError("At least one weight must be above 0.")
    return [value / total for value in numbers]


# --------------------------------------------------------------------------- #
# Combining
# --------------------------------------------------------------------------- #


def softmax(scores: np.ndarray) -> np.ndarray:
    """Probabilities from the raw scores of a network (rows × classes)."""
    scores = np.asarray(scores, dtype=np.float64)
    shifted = scores - scores.max(axis=1, keepdims=True)
    exps = np.exp(shifted)
    return exps / exps.sum(axis=1, keepdims=True)


def weighted_average(outputs: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """The weighted average of the members' predictions (same shape each)."""
    stacked = np.stack([np.asarray(o, dtype=np.float64) for o in outputs])
    w = np.asarray(weights, dtype=np.float64).reshape(-1, *([1] * (stacked.ndim - 1)))
    return (stacked * w).sum(axis=0) / float(np.sum(weights))


def stacking_features(task: str, outputs: list[np.ndarray]) -> np.ndarray:
    """The inputs of the stacking model: per member the probability of the second class
    (two classes), all its probabilities (more classes) or its predicted value."""
    if task == "regression":
        return np.column_stack([np.asarray(o, dtype=np.float64).reshape(-1) for o in outputs])
    columns = []
    for output in outputs:
        proba = np.asarray(output, dtype=np.float64)
        columns.append(proba[:, [POSITIVE_CLASS]] if proba.shape[1] == 2 else proba)
    return np.hstack(columns)


def fit_stacking(
    task: str, outputs: list[np.ndarray], targets: np.ndarray, num_classes: int | None = None
) -> dict[str, Any]:
    """Fit the stacking model on the members' predictions ``outputs`` of rows with the true
    ``targets`` (scikit-learn, its defaults). Returns it as plain numbers::

        {"model": "logistic_regression", "classes": [0, 1], "coef": [[...]],
         "intercept": [...], "features": [...]}
        {"model": "ridge", "coef": [...], "intercept": 0.3, "features": [...]}

    Raises :class:`EnsembleError` without rows or with one class only."""
    x = stacking_features(task, outputs)
    y = np.asarray(targets).reshape(-1)
    if len(y) == 0:
        raise EnsembleError("Stacking needs validation rows to fit its model on.")
    if not np.isfinite(x).all():
        raise EnsembleError(
            "A member predicted no number for some validation rows (NaN), so no stacking "
            "model can be fitted on them. Use the weighted average."
        )
    if task == "regression":
        from sklearn.linear_model import Ridge

        model = Ridge().fit(x, y.astype(np.float64))
        return {
            "model": "ridge",
            "coef": [float(c) for c in np.ravel(model.coef_)],
            "intercept": float(model.intercept_),
            "rows": int(len(y)),
        }
    from sklearn.linear_model import LogisticRegression

    y = y.astype(np.int64)
    if len(np.unique(y)) < 2:
        raise EnsembleError(
            "The validation rows hold only one class, so no stacking model can be fitted on "
            "them. Use the weighted average."
        )
    model = LogisticRegression(max_iter=1000).fit(x, y)
    return {
        "model": "logistic_regression",
        "classes": [int(c) for c in model.classes_],
        "num_classes": int(num_classes or (int(y.max()) + 1)),
        "coef": [[float(c) for c in row] for row in model.coef_],
        "intercept": [float(c) for c in np.ravel(model.intercept_)],
        "rows": int(len(y)),
    }


def apply_stacking(meta: dict[str, Any], task: str, outputs: list[np.ndarray]) -> np.ndarray:
    """What the fitted stacking model ``meta`` predicts from the members' ``outputs``: the
    probabilities of all classes (classes it never saw get 0) or the values."""
    x = stacking_features(task, outputs)
    coef = np.asarray(meta["coef"], dtype=np.float64)
    if task == "regression":
        return x @ coef + float(meta["intercept"])
    intercept = np.asarray(meta["intercept"], dtype=np.float64)
    classes = [int(c) for c in meta["classes"]]
    full = np.zeros((len(x), int(meta["num_classes"])))
    scores = x @ coef.T + intercept
    if len(classes) == 2:
        positive = 1.0 / (1.0 + np.exp(-scores[:, 0]))
        full[:, classes[1]] = positive
        full[:, classes[0]] = 1.0 - positive
    else:
        full[:, classes] = softmax(scores)
    return full


def combine(
    task: str,
    method: str,
    outputs: list[np.ndarray],
    weights: list[float] | None = None,
    meta: dict[str, Any] | None = None,
) -> np.ndarray:
    """The ensemble's predictions from its members' ``outputs``."""
    if method == "stacking":
        if meta is None:
            raise EnsembleError("This ensemble has no fitted stacking model.")
        return apply_stacking(meta, task, outputs)
    return weighted_average(outputs, weights or [1.0 / len(outputs)] * len(outputs))


# --------------------------------------------------------------------------- #
# Measuring
# --------------------------------------------------------------------------- #


def split_result(task: str, predictions: np.ndarray, targets: np.ndarray, num_classes: int):
    """The metrics of one split (as evaluation.py measures a trained run)."""
    if task == "classification":
        counter = ClassificationCounter(num_classes)
        counter.add(predictions, targets)
        return counter.result()
    collector = RegressionCollector()
    collector.add(predictions, targets)
    return collector.result()


def headline(task: str, result: dict[str, Any] | None) -> dict[str, Any] | None:
    """The numbers of a split for the comparison table."""
    if result is None:
        return None
    if task == "regression":
        keys = ("size", "rmse", "mae", "r2")
        return {key: result.get(key) for key in keys}
    found = {
        key: result.get(key)
        for key in ("size", "accuracy", "balanced_accuracy", "macro_f1", "log_loss")
    }
    scores = getattr(result, "binary_scores", None)
    if scores is not None:
        found["roc_auc"], found["pr_auc"] = binary_aucs(*scores)
    return found


def _loss(task: str, result: dict[str, Any] | None) -> float | None:
    if result is None:
        return None
    return result.get("log_loss") if task == "classification" else result.get("mse")


def ensemble_result(
    task: str,
    method: str,
    outputs: dict[str, list[np.ndarray]],
    targets: dict[str, np.ndarray],
    *,
    weights: list[float] | None = None,
    num_classes: int | None = None,
    class_names: list[str] | None = None,
    member_names: list[str] | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Combine and measure. ``outputs``: per split ("val", "test", "training") the
    predictions of every member, ``targets`` the true classes or values of its rows.
    Stacking fits its model on the validation rows unless ``meta`` is given.

    Returns ``{"evaluation", "ensemble", "history", "best", "test"}``: the evaluation of
    the combined predictions (evaluation.py), what the ensemble keeps (method, weights,
    stacking model, the comparison with the members), and the metrics a trained run
    reports (one "epoch" with the validation metrics, the test metrics)."""
    method = check_method(method)
    splits = [key for key in SPLITS if key in outputs and len(targets[key])]
    if not splits:
        raise EnsembleError("The dataset has no rows to measure the ensemble on.")
    count = len(outputs[splits[0]])
    k = int(num_classes or 0)
    if method == "stacking" and meta is None:
        if "val" not in splits:
            raise EnsembleError(STACKING_NEEDS_VALIDATION)
        meta = fit_stacking(task, outputs["val"], targets["val"], k)
    if method == "average":
        weights = normalized_weights(weights, count)
    combined = {key: combine(task, method, outputs[key], weights, meta) for key in splits}
    results = {key: split_result(task, combined[key], targets[key], k) for key in splits}
    names = class_names_for(class_names, k) if task == "classification" else None
    found = evaluation(
        task,
        results.get("val"),
        results.get("test"),
        None,
        class_names=names,
        training=results.get("training"),
    )
    # Stacking measured on the rows it was fitted on (no test rows), or a dataset without
    # validation and test rows (the members were fitted on the training rows).
    optimistic = "test" not in splits if method == "stacking" else splits == ["training"]
    labels = member_names or [f"Member {i + 1}" for i in range(count)]
    rows = []
    for index, label in enumerate(labels):
        rows.append(
            {
                "name": label,
                "member": index,
                **{
                    key: headline(task, split_result(task, outputs[key][index], targets[key], k))
                    for key in splits
                },
            }
        )
    rows.append(
        {
            "name": "Ensemble",
            "member": None,
            **{key: headline(task, results[key]) for key in splits},
        }
    )
    info: dict[str, Any] = {
        "method": method,
        "weights": weights if method == "average" else None,
        "meta": meta if method == "stacking" else None,
        "fitted_on": "val" if method == "stacking" else None,
        "optimistic": optimistic,
        "splits": splits,
        "rows": {key: int(len(targets[key])) for key in splits},
        "comparison": rows,
    }
    val, test = results.get("val"), results.get("test")
    entry = {
        "epoch": 1,
        "train_loss": None,
        "train_accuracy": None,
        "val_loss": _loss(task, val),
        "val_accuracy": val.get("accuracy") if val is not None else None,
        "seconds": None,
    }
    if task == "regression":
        entry["val_rmse"] = val.get("rmse") if val is not None else None
        entry["val_r2"] = val.get("r2") if val is not None else None
    best = {key: entry[key] for key in ("epoch", "val_loss", "val_accuracy")} if val else None
    tested = None
    if test is not None:
        tested = {
            "loss": _loss(task, test),
            "accuracy": test.get("accuracy"),
            "rmse": test.get("rmse"),
            "r2": test.get("r2"),
        }
    return {
        "evaluation": found,
        "ensemble": info,
        "history": [entry],
        "best": best,
        "test": tested,
    }


def summary(task: str, result: dict[str, Any]) -> str:
    """One sentence about the ensemble's metrics, for the run's message."""
    info = result["ensemble"]
    evaluation_ = result["evaluation"]
    parts = []
    for key, label in (("val", "validation"), ("test", "test"), ("training", "training")):
        found = evaluation_.get(key)
        if not isinstance(found, dict):
            continue
        if task == "classification" and found.get("accuracy") is not None:
            parts.append(f"{label} accuracy {found['accuracy'] * 100:.1f}%")
        elif task == "regression" and found.get("rmse") is not None:
            text = f"{label} RMSE {found['rmse']:.4g}"
            if found.get("r2") is not None:
                text += f", R² {found['r2']:.3f}"
            parts.append(text)
    text = f"Combined {len(info['comparison']) - 1} runs ({METHOD_LABELS[info['method']].lower()})"
    if parts:
        text += ": " + "; ".join(parts)
    if info["optimistic"]:
        text += (
            " (optimistic: measured on the rows the stacking model was fitted on)"
            if info["method"] == "stacking"
            else " (on the training rows, optimistic)"
        )
    return text + "."


# --------------------------------------------------------------------------- #
# The members' predictions of an ensemble (ensemble.npz)
# --------------------------------------------------------------------------- #


def save_outputs(
    path: str | Path,
    outputs: dict[str, list[np.ndarray]],
    targets: dict[str, np.ndarray],
    rows: dict[str, np.ndarray] | None = None,
) -> None:
    """Keep the members' predictions and the true targets per split (NumPy's .npz)."""
    arrays: dict[str, np.ndarray] = {}
    for key, members in outputs.items():
        arrays[f"y_{key}"] = np.asarray(targets[key])
        if rows is not None and key in rows:
            arrays[f"rows_{key}"] = np.asarray(rows[key], dtype=np.int64)
        for index, output in enumerate(members):
            arrays[f"out_{index}_{key}"] = np.asarray(output, dtype=np.float64)
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def load_outputs(
    path: str | Path, count: int
) -> tuple[dict[str, list[np.ndarray]], dict[str, np.ndarray]]:
    """The members' predictions and the targets saved by :func:`save_outputs`."""
    outputs: dict[str, list[np.ndarray]] = {}
    targets: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as data:
        for key in SPLITS:
            if f"y_{key}" not in data:
                continue
            targets[key] = data[f"y_{key}"]
            outputs[key] = [data[f"out_{i}_{key}"] for i in range(count)]
    return outputs, targets
