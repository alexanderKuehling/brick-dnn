"""XGBoost, LightGBM and CatBoost in the classical engine: the CPU threads of their fits,
the validation rows for their early stopping and the trees they kept.

The bricks (layers.py) and the generated code name the libraries; this module only reaches
their estimators through scikit-learn's ``get_params``/``set_params`` and the attributes of
the fitted models, so it imports none of them (whether they are installed is checked in
compat.py). Needs NumPy.
"""

from __future__ import annotations

import contextlib
import os
from typing import Any

import numpy as np

from ..layers import BOOSTING_LIBRARIES

# LightGBM 4.7 deprecates eval_set for eval_X/eval_y, but its classifier encodes the labels
# of eval_set only: the engine passes eval_set to LGBMClassifier and drops this warning.
EVAL_SET_DEPRECATED = "The argument 'eval_set' is deprecated"


def library_of(estimator: Any) -> str | None:
    """``"xgboost"``, ``"lightgbm"`` or ``"catboost"`` for an estimator of those libraries."""
    library = type(estimator).__module__.split(".")[0]
    return library if library in BOOSTING_LIBRARIES else None


def library_estimators(pipeline: Any) -> list[Any]:
    """The (unfitted) estimators of the libraries in a pipeline: its model, or the models of
    its ensemble."""
    model = pipeline.named_steps["model"]
    members = getattr(model, "estimators", None)
    candidates = [estimator for _, estimator in members] if isinstance(members, list) else [model]
    return [estimator for estimator in candidates if library_of(estimator) is not None]


def job_threads() -> int | None:
    """The CPU threads of a training (OMP_NUM_THREADS, as for PyTorch); None if not set."""
    with contextlib.suppress(ValueError):
        threads = int(os.environ.get("OMP_NUM_THREADS", ""))
        if threads > 0:
            return threads
    return None


def limit_threads(pipeline: Any, threads: int | None) -> None:
    """Fit (and later predict) with ``threads`` CPU threads; None: the library's default (all
    CPUs). CatBoost does not read OMP_NUM_THREADS, so it is set for every library."""
    for estimator in library_estimators(pipeline):
        if library_of(estimator) == "catboost":
            estimator.set_params(thread_count=threads or -1)
        else:
            estimator.set_params(n_jobs=threads)


def stopping_rounds(pipeline: Any) -> int:
    """The early stopping rounds of the pipeline's model (0: no early stopping, or not a
    model of the libraries; an ensemble's models are written without it)."""
    model = pipeline.named_steps["model"]
    if library_of(model) is None:
        return 0
    return int(model.get_params().get("early_stopping_rounds") or 0)


def stopping_fit_params(
    pipeline: Any,
    x: np.ndarray,
    y: np.ndarray,
    x_eval: np.ndarray | None,
    y_eval: np.ndarray | None,
    classification: bool,
) -> dict[str, Any]:
    """The keyword arguments of ``pipeline.fit(x, y, …)`` that hand the model the rows its
    early stopping watches (standardised like the training rows), like the generated
    script. Without such rows (None, or none of a class the training rows have) early
    stopping is switched off and the model fits all its trees. ``{}`` for other models."""
    model = pipeline.named_steps["model"]
    library = library_of(model)
    if library is None or not stopping_rounds(pipeline):
        return {}
    if x_eval is not None and y_eval is not None and classification:
        # A class the training rows lack cannot be scored.
        known = np.isin(y_eval, np.unique(y))
        x_eval, y_eval = x_eval[known], y_eval[known]
    if x_eval is None or y_eval is None or not len(x_eval):
        model.set_params(early_stopping_rounds=None)
        return {}
    watch = pipeline[:-1].fit(x).transform(x_eval)  # the scaler of the pipeline
    if library == "xgboost":
        return {"model__eval_set": [(watch, y_eval)], "model__verbose": False}
    if library == "lightgbm":
        if classification:
            return {"model__eval_set": [(watch, y_eval)]}
        return {"model__eval_X": watch, "model__eval_y": y_eval}
    return {"model__eval_set": (watch, y_eval)}


def trees(model: Any) -> tuple[int, int]:
    """``(trees used, most trees)`` of a fitted model of the libraries: early stopping keeps
    the trees up to its best round."""
    library = library_of(model)
    if library == "catboost":
        return int(model.tree_count_), int(model.get_params().get("iterations") or 1000)
    most = int(model.get_params()["n_estimators"])
    if library == "xgboost":
        best = getattr(model, "best_iteration", None)  # only with early stopping
        return (int(best) + 1 if best is not None else most), most
    best = int(getattr(model, "best_iteration_", 0) or 0)  # 0 without early stopping
    return (best or int(getattr(model, "n_estimators_", most))), most
