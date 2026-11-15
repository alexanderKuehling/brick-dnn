"""The loss of a neural network: which losses there are, their settings and how they are
checked against the task (no PyTorch needed; the loss functions are in loss_functions.py).

The loss is chosen per task (``loss`` of the training settings, "auto" by default):

    classification  cross_entropy (default), weighted_cross_entropy (``class_weights``,
                    "balanced" by default), focal (``focal_gamma``, optional
                    ``class_weights`` as alpha); ``label_smoothing`` works with both
                    cross-entropies
    regression      mse (default), mae, huber (``huber_delta``); all of the standardised
                    target, so ``huber_delta`` is in standard deviations of the target
    segmentation    cross_entropy (default), cross_entropy_dice, weighted_cross_entropy
                    (``class_weights``, "balanced" from the pixel counts)

"auto" (or no loss) is the default of the task, so runs and settings of older versions
behave as before. Settings of older versions say ``cross_entropy`` for every task: for a
regression that still means its default, the mean squared error.

``class_weights`` is "balanced" (``n / (k · count)`` from the class counts of the training
rows, or pixels of a segmentation: ``n`` rows, ``k`` classes that occur; a class without
training rows gets 1), a list with one weight per class, or ``{class: weight}`` with class
names or indices (missing classes get 1).

The loss in the curves, in early stopping, in the kept epoch and in the default objective
of a search is the chosen loss, so losses of different kinds are not comparable; a search
over losses of a network is ranked by another metric (see :func:`search_problems`).
"""

from __future__ import annotations

import math
from typing import Any

AUTO = "auto"
LABELS = {
    "cross_entropy": "cross-entropy",
    "weighted_cross_entropy": "weighted cross-entropy",
    "focal": "focal loss",
    "cross_entropy_dice": "cross-entropy + Dice",
    "mse": "mean squared error",
    "mae": "mean absolute error",
    "huber": "Huber loss",
}
# The losses of every task; the first one is its default.
TASK_LOSSES = {
    "classification": ("cross_entropy", "weighted_cross_entropy", "focal"),
    "regression": ("mse", "mae", "huber"),
    "segmentation": ("cross_entropy", "cross_entropy_dice", "weighted_cross_entropy"),
}
LOSSES = (AUTO, *dict.fromkeys(name for names in TASK_LOSSES.values() for name in names))
# The task of a target of the plan (see compat.TrainingPlan).
KIND_TASKS = {"class": "classification", "value": "regression", "mask": "segmentation"}
# Which losses use which settings (other losses ignore them).
CLASS_WEIGHT_LOSSES = ("weighted_cross_entropy", "focal")
SMOOTHING_LOSSES = ("cross_entropy", "weighted_cross_entropy")  # classification only
PARAMS = {
    "class_weights": CLASS_WEIGHT_LOSSES,
    "focal_gamma": ("focal",),
    "label_smoothing": SMOOTHING_LOSSES,
    "huber_delta": ("huber",),
}
# Limits of the settings: (smallest, largest, smallest excluded).
FOCAL_GAMMA = (0.0, 10.0, False)
LABEL_SMOOTHING = (0.0, 0.3, False)
HUBER_DELTA = (0.0, 10.0, True)
MAX_CLASS_WEIGHT = 1000.0
MAX_WEIGHTED_CLASSES = 1000
BALANCED = "balanced"
SEARCH_LOSS_OBJECTIVE = (
    "A search over the loss cannot rank its trials by the loss: losses of different kinds "
    "are not comparable (a focal loss is smaller than the cross-entropy of the same model, "
    "a mean absolute error is not a squared one). Choose another objective metric, e.g. "
    "{metrics}."
)


class LossError(ValueError):
    """An invalid loss setting. The message is shown to the user."""


def _kinds(plan: Any) -> list[str]:
    targets = plan.get("targets") if isinstance(plan, dict) else getattr(plan, "targets", None)
    return [str(t.get("kind")) for t in targets or [] if isinstance(t, dict)]


def plan_tasks(plan: Any) -> list[str]:
    """The tasks of the outputs of ``plan`` (a TrainingPlan or its dict), without
    duplicates: one task, or for demo data with several outputs classification and
    regression."""
    return list(dict.fromkeys(KIND_TASKS[k] for k in _kinds(plan) if k in KIND_TASKS))


def task_text(tasks: list[str]) -> str:
    return " and ".join(tasks) if tasks else "this task"


def options(tasks: list[str]) -> list[str]:
    """The losses ``tasks`` can use ("auto" first)."""
    return [AUTO, *dict.fromkeys(name for task in tasks for name in TASK_LOSSES.get(task, ()))]


def loss_for(name: str, task: str) -> str:
    """The loss a ``task`` output uses with the setting ``name``: its default for "auto",
    for the old default ``cross_entropy`` of a regression, and for a loss of another task
    (demo data with outputs of both tasks: each uses the setting only if it fits)."""
    losses = TASK_LOSSES[task]
    return name if name in losses else losses[0]


def loss_error(name: str, plan: Any) -> str | None:
    """Why the loss ``name`` does not fit the task of ``plan`` (None: it does)."""
    tasks = plan_tasks(plan)
    if name == AUTO or not tasks:
        return None
    if any(name in TASK_LOSSES[task] for task in tasks):
        return None
    if name == "cross_entropy" and "regression" in tasks:
        return None  # the old default of every task
    owners = [task for task, losses in TASK_LOSSES.items() if name in losses]
    what = f"is for {task_text(owners)}" if owners else "is unknown"
    return (
        f"The loss '{name}' {what}, and this is a {task_text(tasks)}. "
        f"Possible: {', '.join(options(tasks))}."
    )


def resolved(name: str, plan: Any) -> str:
    """The loss a run on ``plan`` trains with, as it is stored with the run: "auto" becomes
    the default of the task. Data with outputs of two tasks keeps the setting."""
    tasks = plan_tasks(plan)
    if len(tasks) != 1:
        return name
    return loss_for(name, tasks[0])


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #


def _weight(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise LossError("Class weights must be numbers.")
    try:
        number = float(value)
    except ValueError:
        raise LossError(f"Class weights must be numbers, not '{value}'.") from None
    if not math.isfinite(number) or number <= 0 or number > MAX_CLASS_WEIGHT:
        raise LossError(f"Class weights must be greater than 0 and at most {MAX_CLASS_WEIGHT:g}.")
    return number


def parse_class_weights(value: Any) -> str | list[float] | dict[str, float] | None:
    """``class_weights`` of the settings: None, "balanced", a list of positive numbers or
    ``{class: weight}``. Whether they fit the classes is checked with the data
    (:func:`class_weights_error`). Raises :class:`LossError`."""
    if value is None or value == "" or value == []:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("", "none"):
            return None
        if text == BALANCED:
            return BALANCED
        raise LossError(
            'Class weights must be "balanced", a list with one weight per class or {class: weight}.'
        )
    if isinstance(value, (list, tuple)):
        if len(value) > MAX_WEIGHTED_CLASSES:
            raise LossError(f"Class weights: at most {MAX_WEIGHTED_CLASSES:,} classes.")
        return [_weight(item) for item in value]
    if isinstance(value, dict):
        if not value or len(value) > MAX_WEIGHTED_CLASSES:
            raise LossError(f"Class weights: give 1 to {MAX_WEIGHTED_CLASSES:,} classes.")
        return {str(key).strip(): _weight(item) for key, item in value.items()}
    raise LossError(
        'Class weights must be "balanced", a list with one weight per class or {class: weight}.'
    )


def _class_targets(plan: Any) -> list[int]:
    targets = plan.get("targets") if isinstance(plan, dict) else getattr(plan, "targets", None)
    return [
        int(t.get("num_classes") or 0)
        for t in targets or []
        if isinstance(t, dict) and t.get("kind") in ("class", "mask")
    ]


def _class_names(plan: Any, k: int) -> list[str]:
    names = (
        plan.get("class_names") if isinstance(plan, dict) else getattr(plan, "class_names", None)
    )
    names = [str(n) for n in names or []][:k]
    return names + [str(i) for i in range(len(names), k)]


def _shown(names: list[str], limit: int = 8) -> str:
    text = ", ".join(names[:limit])
    return text + (f" … ({len(names)})" if len(names) > limit else "")


def _class_index(key: str, names: list[str]) -> int | None:
    if key in names:
        return names.index(key)
    lowered = [name.lower() for name in names]
    if key.lower() in lowered:
        return lowered.index(key.lower())
    if key.isdigit() and int(key) < len(names):
        return int(key)
    return None


def class_weights_error(weights: Any, plan: Any) -> str | None:
    """Why explicit class weights do not fit the classes of ``plan`` (None: they do)."""
    if not isinstance(weights, (list, dict)):
        return None
    for k in _class_targets(plan):
        names = _class_names(plan, k)
        if isinstance(weights, list) and len(weights) != k:
            return (
                f"Class weights: give one weight per class ({k}: {_shown(names)}), not "
                f"{len(weights)}."
            )
        if isinstance(weights, dict):
            unknown = [key for key in weights if _class_index(key, names) is None]
            if unknown:
                return (
                    f"Class weights: '{unknown[0]}' is not a class. The classes are "
                    f"{_shown(names)} (or their indices 0 to {k - 1})."
                )
    return None


def class_weight_vector(
    weights: Any, k: int, names: list[str] | None, counts: list[float] | None
) -> list[float] | None:
    """The weight of every one of ``k`` classes (None: no weights). "balanced" needs the
    ``counts`` of the classes in the training rows (pixels)."""
    if weights is None:
        return None
    if weights == BALANCED:
        assert counts is not None
        n = float(sum(counts))
        present = sum(1 for count in counts if count > 0)
        return [n / (present * count) if count > 0 else 1.0 for count in counts]
    if isinstance(weights, list):
        return [float(w) for w in weights][:k] + [1.0] * max(0, k - len(weights))
    vector = [1.0] * k
    all_names = [str(n) for n in names or []][:k]
    all_names += [str(i) for i in range(len(all_names), k)]
    for key, weight in dict(weights).items():
        index = _class_index(str(key), all_names)
        if index is not None:
            vector[index] = float(weight)
    return vector


def uses(param: str, name: str, task: str | None = None) -> bool:
    """True if the loss ``name`` (of ``task``) uses the setting ``param``."""
    if param == "label_smoothing" and task == "segmentation":
        return False
    return name in PARAMS[param]


def class_weights_of(config: Any, name: str) -> Any:
    """The class weights the loss ``name`` uses: those of the settings; a weighted
    cross-entropy without any is balanced."""
    if name not in CLASS_WEIGHT_LOSSES:
        return None
    weights = getattr(config, "class_weights", None)
    if weights is None and name == "weighted_cross_entropy":
        return BALANCED
    return weights


def describe(config: Any, task: str, weights: list[float] | None = None) -> dict[str, Any]:
    """The loss of a ``task`` output with ``config`` as the run reports it (state.json
    ``loss``): ``{"name", "label", ...}`` with the settings this loss uses; ``weights``:
    the class weights it uses (computed for "balanced")."""
    name = loss_for(getattr(config, "loss", AUTO) or AUTO, task)
    info: dict[str, Any] = {"name": name, "label": LABELS[name], "task": task}
    given = class_weights_of(config, name)
    if given is not None:
        info["class_weights_from"] = BALANCED if given == BALANCED else "given"
        info["class_weights"] = [round(w, 4) for w in weights] if weights is not None else given
    for param in ("focal_gamma", "label_smoothing", "huber_delta"):
        if uses(param, name, task):
            info[param] = getattr(config, param)
    return info


def text(info: dict[str, Any] | None) -> str:
    """``"focal loss (gamma 2, class weights balanced)"``."""
    if not info:
        return "the loss"
    parts = []
    if "focal_gamma" in info:
        parts.append(f"gamma {info['focal_gamma']:g}")
    if info.get("label_smoothing"):
        parts.append(f"label smoothing {info['label_smoothing']:g}")
    if "huber_delta" in info:
        parts.append(f"delta {info['huber_delta']:g} standard deviations")
    if info.get("class_weights_from") == BALANCED:
        parts.append("balanced class weights")
    elif info.get("class_weights_from"):
        parts.append("class weights given")
    label = str(info.get("label") or info.get("name"))
    return f"{label} ({', '.join(parts)})" if parts else label


def config_problems(
    config: Any, plan: Any, searched: set[str] | None = None
) -> tuple[list[str], list[str]]:
    """``(errors, warnings)`` of the loss settings on ``plan``: the loss must fit the task,
    explicit class weights the classes; settings that the loss does not use are ignored
    (a warning), unless a search varies the loss (``searched``: the names of its
    space)."""
    errors: list[str] = []
    warnings: list[str] = []
    tasks = plan_tasks(plan)
    if not tasks:
        return errors, warnings
    name = getattr(config, "loss", AUTO) or AUTO
    error = loss_error(name, plan)
    if error is not None:
        return [error], warnings
    weights = getattr(config, "class_weights", None)
    error = class_weights_error(weights, plan)
    if error is not None:
        errors.append(error)
    if searched and "loss" in searched:
        return errors, warnings
    used = {task: loss_for(name, task) for task in tasks}
    given = {
        "class_weights": weights is not None,
        "label_smoothing": bool(getattr(config, "label_smoothing", 0.0)),
    }
    for param, set_ in given.items():
        if set_ and not any(uses(param, loss, task) for task, loss in used.items()):
            losses = ", ".join(sorted({LABELS[loss] for loss in used.values()}))
            what = "Class weights" if param == "class_weights" else "Label smoothing"
            works = (
                "weighted_cross_entropy and focal"
                if param == "class_weights"
                else "cross_entropy and weighted_cross_entropy of a classification"
            )
            verb, them = ("work", "them") if param == "class_weights" else ("works", "it")
            warnings.append(f"{what} only {verb} with {works}: the {losses} ignores {them}.")
    return errors, warnings


# --------------------------------------------------------------------------- #
# Hyperparameter search
# --------------------------------------------------------------------------- #

LOSS_PARAMS = ("focal_gamma", "label_smoothing", "huber_delta")


def search_problems(tuning: Any, plan: Any, config: Any) -> tuple[list[str], list[str]]:
    """``(errors, warnings)`` of a search that varies the loss or its settings (``space``
    of a network, ``neural_space`` of a hybrid network): every loss must fit the task, a
    setting of a loss needs that loss in the trials, and a network's search over losses
    cannot be ranked by the loss. A hybrid network ranks its trials by its classical model,
    whose loss does not depend on the loss of the neural layers."""
    errors: list[str] = []
    warnings: list[str] = []
    hybrid = getattr(tuning, "model", None) is not None
    space = getattr(tuning, "neural_space", None) if hybrid else getattr(tuning, "space", None)
    if not space:
        return errors, warnings
    tasks = plan_tasks(plan)
    names = [str(v) for v in space.get("loss", ())] or [getattr(config, "loss", AUTO) or AUTO]
    for name in names:
        error = loss_error(name, plan)
        if error is not None:
            return [f"Hyperparameter search: {error}"], warnings
    used = {loss_for(name, task) for name in names for task in tasks}
    for param in LOSS_PARAMS:
        if param in space and not any(uses(param, loss, t) for loss in used for t in tasks):
            errors.append(
                f"Hyperparameter search: {param} only changes the loss "
                f"{' or '.join(PARAMS[param])}, which no trial uses. Set the loss, or search it."
            )
    if errors or hybrid or getattr(tuning, "metric", "loss") != "loss":
        return errors, warnings
    if len(used) > 1:
        errors.append(SEARCH_LOSS_OBJECTIVE.format(metrics=_better_metrics(plan)))
    elif any(param in space for param in LOSS_PARAMS):
        searched = ", ".join(p for p in LOSS_PARAMS if p in space)
        warnings.append(
            f"The search varies {searched}, which changes the loss itself: the trials are "
            "ranked by losses that are not quite comparable. An objective metric such as "
            f"{_better_metrics(plan)} compares them better."
        )
    return errors, warnings


def _better_metrics(plan: Any) -> str:
    tasks = plan_tasks(plan)
    num_classes = (
        plan.get("num_classes") if isinstance(plan, dict) else getattr(plan, "num_classes", None)
    )
    if tasks == ["regression"]:
        return "rmse, mae or r2"
    if tasks == ["classification"]:
        return (
            "pr_auc, balanced_accuracy or macro_f1"
            if num_classes == 2
            else ("balanced_accuracy, macro_f1 or accuracy")
        )
    return "a metric of the task"


def capabilities() -> dict[str, Any]:
    """The losses for the training options of the API: per task its losses (the first is
    the default) and the settings with their limits."""
    return {
        "default": AUTO,
        "tasks": {task: list(names) for task, names in TASK_LOSSES.items()},
        "labels": dict(LABELS),
        "params": {
            "class_weights": {
                "losses": list(CLASS_WEIGHT_LOSSES),
                "values": [BALANCED, "[weight per class]", "{class: weight}"],
            },
            "focal_gamma": {"losses": ["focal"], "min": 0.0, "max": 10.0, "default": 2.0},
            "label_smoothing": {
                "losses": list(SMOOTHING_LOSSES),
                "min": 0.0,
                "max": 0.3,
                "default": 0.0,
            },
            "huber_delta": {
                "losses": ["huber"],
                "min": 0.0,
                "min_exclusive": True,
                "max": 10.0,
                "default": 1.0,
            },
        },
    }
