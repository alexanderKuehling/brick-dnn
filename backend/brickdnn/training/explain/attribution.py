"""Model-agnostic explanations of tabular models (NumPy only): Shapley values of one
prediction against a baseline, and permutation feature importance on validation rows.

Both only call ``predict(rows) -> outputs`` (rows × features in, rows × outputs out), so
they work the same for neural networks and scikit-learn models.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable

import numpy as np

Predict = Callable[[np.ndarray], np.ndarray]

# Exact Shapley values evaluate the model on 2^features rows: up to this many features.
EXACT_FEATURES = 10
# Otherwise random feature orders, (features + 1) rows each, at most this many rows.
MAX_SHAPLEY_ROWS = 20_000
MAX_ORDERS = 256
# Rows per call of predict (keeps memory bounded, e.g. a Gaussian process's kernel matrix).
CHUNK = 2048


def predict_chunked(predict: Predict, rows: np.ndarray) -> np.ndarray:
    parts = [predict(rows[i : i + CHUNK]) for i in range(0, len(rows), CHUNK)]
    out = np.concatenate(parts, axis=0) if parts else np.zeros((0, 1))
    return out.reshape(len(rows), -1)


def shapley_values(
    predict: Predict,
    x: np.ndarray,
    baseline: np.ndarray,
    output: int = 0,
    seed: int = 0,
) -> dict:
    """Shapley values of ``predict(x)[output]`` against a single ``baseline`` row (the
    "baseline Shapley" of Sundararajan & Najmi, 2020): a feature that is "absent" takes its
    baseline value. The values add up exactly to ``f(x) - f(baseline)``.

    Exact (all 2^d coalitions) for up to EXACT_FEATURES features, else averaged over random
    feature orders (each order telescopes, so the sum stays exact; single values are
    estimates)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    baseline = np.asarray(baseline, dtype=np.float64).ravel()
    d = len(x)
    if d <= EXACT_FEATURES:
        masks = np.array(
            [[(m >> i) & 1 for i in range(d)] for m in range(2**d)], dtype=bool
        ).reshape(2**d, d)
        rows = np.where(masks, x, baseline)
        values = predict_chunked(predict, rows)[:, output]
        # Weight of a coalition of size s that feature i joins: s! (d - s - 1)! / d!
        weights = np.array(
            [math.factorial(s) * math.factorial(d - s - 1) / math.factorial(d) for s in range(d)]
        )
        sizes = masks.sum(axis=1)
        everything = np.arange(2**d)
        phi = np.zeros(d)
        for i in range(d):
            without = everything[~masks[:, i]]
            gains = values[without | (1 << i)] - values[without]
            phi[i] = float((weights[sizes[without]] * gains).sum())
        return {
            "values": phi,
            "base": float(values[0]),
            "prediction": float(values[-1]),
            "exact": True,
            "orders": None,
        }
    orders = max(1, min(MAX_ORDERS, MAX_SHAPLEY_ROWS // (d + 1)))
    rng = np.random.default_rng(seed)
    permutations = np.array([rng.permutation(d) for _ in range(orders)])
    rows = np.empty((orders, d + 1, d))
    for k, order in enumerate(permutations):
        current = baseline.copy()
        rows[k, 0] = current
        for step, feature in enumerate(order, start=1):
            current[feature] = x[feature]
            rows[k, step] = current
    values = predict_chunked(predict, rows.reshape(-1, d))[:, output].reshape(orders, d + 1)
    phi = np.zeros(d)
    for k, order in enumerate(permutations):
        phi[order] += np.diff(values[k])
    phi /= orders
    return {
        "values": phi,
        "base": float(values[0, 0]),
        "prediction": float(values[0, -1]),
        "exact": False,
        "orders": orders,
    }


def accuracy(outputs: np.ndarray, y: np.ndarray) -> float:
    return float((np.argmax(outputs, axis=1) == y).mean())


def r2_score(outputs: np.ndarray, y: np.ndarray) -> float | None:
    predicted = outputs[:, 0]
    spread = float(((y - y.mean()) ** 2).sum())
    if spread <= 1e-12:
        return None
    return 1.0 - float(((y - predicted) ** 2).sum()) / spread


def permutation_importance(
    predict: Predict,
    x: np.ndarray,
    y: np.ndarray,
    classification: bool,
    repeats: int = 5,
    seed: int = 0,
    deadline: float | None = None,
) -> dict:
    """How much the score (accuracy, or R² for regression) drops when the values of one
    feature are shuffled across the rows, which breaks its link to the target. Mean and
    standard deviation over ``repeats`` shuffles; fewer repeats if ``deadline``
    (time.monotonic()) comes first (at least one)."""
    x = np.asarray(x, dtype=np.float64)
    score = accuracy if classification else r2_score
    baseline = score(predict_chunked(predict, x), y)
    if baseline is None:
        raise ValueError(
            "The validation targets are all equal, so R² (and how much it drops) is undefined."
        )
    rng = np.random.default_rng(seed)
    d = x.shape[1]
    drops: list[list[float]] = [[] for _ in range(d)]
    done = 0
    for repeat in range(repeats):
        if repeat > 0 and deadline is not None and time.monotonic() > deadline:
            break
        for j in range(d):
            shuffled = x.copy()
            shuffled[:, j] = x[rng.permutation(len(x)), j]
            permuted = score(predict_chunked(predict, shuffled), y)
            drops[j].append(baseline - (permuted if permuted is not None else 0.0))
        done += 1
    means = np.array([np.mean(v) for v in drops])
    stds = np.array([np.std(v) for v in drops])
    return {"baseline": baseline, "mean": means, "std": stds, "repeats": done}
