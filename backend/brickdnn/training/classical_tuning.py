"""Hyperparameter search for classical models: what can be tuned and how (no scikit-learn
needed; the search itself is in classical_search.py).

A classical model searches over the parameters of its brick. The search space maps their
names to a range or a list of values, as for neural networks (see tuning.py)::

    {"learning_rate": {"min": 0.01, "max": 0.3},           # drawn on a log scale
     "min_samples_leaf": {"min": 5, "max": 100, "scale": "linear"},   # whole numbers
     "max_depth": [3, 6, null],                            # null: no limit
     "class_weight": ["none", "balanced"]}

A range is drawn log-uniformly by default (``"scale": "linear"``: uniformly); parameters
that are whole numbers give whole numbers. A log range needs a minimum above 0. Grid
search needs lists. The SVM's ``gamma`` takes numbers (the brick's *Gamma value*) and the
names ``"scale"`` and ``"auto"``. A Gaussian process has little to tune, because its fit
optimises the kernel hyperparameters itself: ``alpha``, ``normalize_y`` and the optimizer
restarts of the regression, the restarts of the classifier.

The trials are compared on the validation rows, or by k-fold cross-validation on the
training rows (``cv_folds``; without validation rows always, with DEFAULT_FOLDS folds).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from ..layers import SVM_KERNELS
from .config import ConfigError, TrainingConfig
from .datasets import DatasetInfo
from .tabular import has_validation
from .tuning import MAX_VALUES, Range, TuningConfig, Values, _number

# Folds of a cross-validation when there are no validation rows and none were chosen.
DEFAULT_FOLDS = 5
# A cross-validation needs at least this many training rows per fold.
MIN_ROWS_PER_FOLD = 2


@dataclass(frozen=True)
class Tunable:
    """A brick parameter that a search can vary.

    ``kind``: ``float`` and ``int`` (numbers between ``low`` and ``high``; ``open_low``: low
    itself is not allowed), ``choice`` (one of ``options``), ``bool``, or ``gamma`` (a number
    above 0 or one of ``options``). ``log``: ranges are drawn on a log scale unless the
    request says otherwise. ``none``: the list may hold null (e.g. no depth limit).
    ``default``: what the editor suggests (a range or a list).
    """

    name: str
    label: str
    kind: str
    default: Any
    low: float = 0.0
    high: float = 1e6
    open_low: bool = False
    log: bool = True
    options: tuple[Any, ...] = ()
    none: bool = False
    help: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "kind": self.kind,
            "min": self.low,
            "max": self.high,
            "min_exclusive": self.open_low,
            "scale": "log" if self.log else "linear",
            "options": list(self.options),
            "none": self.none,
            "default": self.default,
            "help": self.help,
        }


_CLASS_WEIGHT = Tunable(
    "class_weight",
    "class weights",
    "choice",
    ["none", "balanced"],
    options=("none", "balanced"),
    help="balanced weights the classes inversely to their frequency.",
)
_BOOSTING = (
    Tunable(
        "learning_rate",
        "learning rate",
        "float",
        {"min": 0.01, "max": 0.3},
        high=10.0,
        open_low=True,
        help="How much each tree corrects the ones before.",
    ),
    Tunable("max_iter", "max. iterations", "int", [100, 200, 500], low=1, high=10_000),
    Tunable(
        "max_leaf_nodes", "max. leaves per tree", "int", {"min": 15, "max": 127}, low=2, high=4096
    ),
    Tunable(
        "max_depth",
        "max. depth",
        "int",
        [None, 3, 6],
        low=1,
        high=100,
        none=True,
        help="null (no limit) is allowed in a list.",
    ),
    Tunable(
        "min_samples_leaf", "min. rows per leaf", "int", {"min": 5, "max": 100}, low=1, high=1e6
    ),
    Tunable(
        "l2_regularization",
        "L2 regularisation",
        "float",
        [0, 0.1, 1, 10],
        high=1e4,
        help="A log range needs a minimum above 0; a list may hold 0.",
    ),
)
_SVM = (
    Tunable(
        "C",
        "C",
        "float",
        {"min": 0.1, "max": 100},
        high=1e6,
        open_low=True,
        help="Larger values fit the training rows more closely.",
    ),
    Tunable(
        "gamma",
        "gamma",
        "gamma",
        {"min": 0.001, "max": 1},
        high=1e4,
        open_low=True,
        options=("scale", "auto"),
        help="Numbers (the Gamma value of the brick), or scale and auto in a list. The "
        "linear kernel does not use it.",
    ),
    Tunable("kernel", "kernel", "choice", ["rbf", "linear"], options=SVM_KERNELS),
)
_RESTARTS = Tunable(
    "n_restarts_optimizer", "optimizer restarts", "int", [0, 2, 5], high=20, log=False
)

TUNABLES: dict[str, tuple[Tunable, ...]] = {
    "HistGradientBoostingRegressor": _BOOSTING,
    "HistGradientBoostingClassifier": (*_BOOSTING, _CLASS_WEIGHT),
    "SVR": (
        *_SVM,
        Tunable(
            "epsilon",
            "epsilon",
            "float",
            {"min": 0.01, "max": 1},
            high=1e6,
            help="A log range needs a minimum above 0; a list may hold 0.",
        ),
    ),
    "SVC": (*_SVM, _CLASS_WEIGHT),
    "GaussianProcessRegressor": (
        Tunable(
            "alpha",
            "alpha",
            "float",
            {"min": 1e-10, "max": 0.01},
            high=1e3,
            help="Added to the diagonal of the kernel matrix.",
        ),
        Tunable("normalize_y", "normalize targets", "bool", [True, False], options=(True, False)),
        _RESTARTS,
    ),
    "GaussianProcessClassifier": (_RESTARTS,),
}
# What a search varies when the request names nothing.
DEFAULT_SPACES: dict[str, tuple[str, ...]] = {
    "HistGradientBoostingRegressor": ("learning_rate", "max_leaf_nodes", "min_samples_leaf"),
    "HistGradientBoostingClassifier": ("learning_rate", "max_leaf_nodes", "min_samples_leaf"),
    "SVR": ("C", "gamma"),
    "SVC": ("C", "gamma"),
    "GaussianProcessRegressor": ("alpha",),
    "GaussianProcessClassifier": ("n_restarts_optimizer",),
}
NOTES = {
    "GaussianProcessRegressor": "The fit optimises the kernel hyperparameters itself; a "
    "search can only vary alpha, normalize_y and the optimizer restarts.",
    "GaussianProcessClassifier": "The fit optimises the kernel hyperparameters itself; a "
    "search can only vary the optimizer restarts.",
}
MODEL_LABELS = {
    "HistGradientBoostingRegressor": "gradient boosting regression",
    "HistGradientBoostingClassifier": "gradient boosting classification",
    "SVR": "support vector regression",
    "SVC": "support vector classification",
    "GaussianProcessRegressor": "Gaussian process regression",
    "GaussianProcessClassifier": "Gaussian process classification",
}


def tunables(model: str) -> tuple[Tunable, ...]:
    """The tunable parameters of the classical model brick ``model`` (a type)."""
    if model not in TUNABLES:
        raise ConfigError(f"Hyperparameter search: {model} has no parameters to tune.")
    return TUNABLES[model]


def tunable_label(model: str, name: str) -> str:
    for tunable in TUNABLES.get(model, ()):
        if tunable.name == name:
            return tunable.label
    return name


def default_space(model: str) -> dict[str, Any]:
    by_name = {t.name: t for t in tunables(model)}
    return {name: copy.deepcopy(by_name[name].default) for name in DEFAULT_SPACES[model]}


def capabilities() -> dict[str, Any]:
    """What a search of each classical model can vary, for the training options."""
    return {
        model: {
            "tunable": [tunable.to_dict() for tunable in tunables(model)],
            "default": default_space(model),
            "note": NOTES.get(model),
        }
        for model in TUNABLES
    }


# --------------------------------------------------------------------------- #
# Parsing the search space
# --------------------------------------------------------------------------- #


def _value(tunable: Tunable, item: Any) -> Any:
    what = f"values of the {tunable.label}"
    if tunable.kind == "choice":
        value = str(item).strip()
        if value not in tunable.options:
            raise ConfigError(
                f"Hyperparameter search: the {tunable.label} must be one of: "
                f"{', '.join(map(str, tunable.options))}."
            )
        return value
    if tunable.kind == "bool":
        if isinstance(item, str) and item.strip().lower() in ("true", "false"):
            return item.strip().lower() == "true"
        if not isinstance(item, bool):
            raise ConfigError(f"Hyperparameter search: the {what} must be true or false.")
        return item
    if tunable.kind == "gamma" and isinstance(item, str) and item.strip() in tunable.options:
        return item.strip()
    if tunable.none and (item is None or (isinstance(item, str) and item.strip() == "none")):
        return None
    whole = tunable.kind == "int"
    return _number(item, what, tunable.low, tunable.high, tunable.open_low, whole)


def _values(tunable: Tunable, raw: list[Any]) -> Values:
    if not raw:
        raise ConfigError(f"Hyperparameter search: give at least one {tunable.label}.")
    if len(raw) > MAX_VALUES:
        raise ConfigError(f"Hyperparameter search: at most {MAX_VALUES} values per parameter.")
    values = [_value(tunable, item) for item in raw]
    return tuple(dict.fromkeys(values))  # without duplicates, in the given order


def _range(tunable: Tunable, raw: dict[str, Any]) -> Range:
    label = tunable.label
    if tunable.kind in ("choice", "bool"):
        raise ConfigError(f"Hyperparameter search: give a list of values for the {label}.")
    if set(raw) - {"min", "max", "scale"} or "min" not in raw or "max" not in raw:
        raise ConfigError(f'Hyperparameter search: a range for the {label} needs "min" and "max".')
    scale = raw.get("scale") or ("log" if tunable.log else "linear")
    if scale not in ("log", "linear"):
        raise ConfigError(
            f'Hyperparameter search: the scale of a range must be "log" or "linear" ({label}).'
        )
    whole = tunable.kind == "int"
    what = f"the {label} range"
    lo = _number(raw["min"], what, tunable.low, tunable.high, tunable.open_low, whole)
    hi = _number(raw["max"], what, tunable.low, tunable.high, tunable.open_low, whole)
    if lo > hi:
        raise ConfigError(
            f"Hyperparameter search: the smallest {label} must not be larger than the largest."
        )
    if scale == "log" and lo <= 0:
        raise ConfigError(
            f"Hyperparameter search: a log range of the {label} needs a minimum above 0. "
            'Use "scale": "linear", or a list of values.'
        )
    return Range(lo, hi, log=scale == "log", integer=whole)


def parse_space(model: str, raw: Any) -> dict[str, Range | Values]:
    """Validate the search space of the classical model brick ``model`` (a type). Raises
    :class:`ConfigError` with a message for the user."""
    specs = {tunable.name: tunable for tunable in tunables(model)}
    if raw is None:
        raw = default_space(model)
    if not isinstance(raw, dict):
        raise ConfigError("Hyperparameter search: the search space must be an object.")
    unknown = [name for name in raw if name not in specs]
    if unknown:
        what = MODEL_LABELS.get(model, model)
        raise ConfigError(
            f"Hyperparameter search: '{unknown[0]}' cannot be tuned for {what}. "
            f"Possible: {', '.join(specs)}."
        )
    space: dict[str, Range | Values] = {}
    for name, tunable in specs.items():
        if name not in raw:
            continue
        value = raw[name]
        if isinstance(value, dict):
            space[name] = _range(tunable, value)
        elif isinstance(value, (list, tuple)):
            space[name] = _values(tunable, list(value))
        else:
            space[name] = _values(tunable, [value])
    if not space:
        raise ConfigError("Hyperparameter search: choose at least one parameter to tune.")
    return space


# --------------------------------------------------------------------------- #
# Trials
# --------------------------------------------------------------------------- #


def brick_params(params: dict[str, Any]) -> dict[str, Any]:
    """The brick parameters of a trial: a number for ``gamma`` is the brick's Gamma value."""
    result: dict[str, Any] = {}
    for name, value in params.items():
        if name == "gamma":
            if isinstance(value, str):
                result["gamma"] = value
            else:
                result.update(gamma="value", gamma_value=value)
        else:
            result[name] = value
    return result


def trial_graph(graph: dict[str, Any], node: str, params: dict[str, Any]) -> dict[str, Any]:
    """A copy of ``graph`` whose model brick ``node`` has the values of a trial."""
    graph = copy.deepcopy(graph)
    for entry in graph.get("nodes") or []:
        if isinstance(entry, dict) and entry.get("id") == node:
            entry["params"] = {**(entry.get("params") or {}), **brick_params(params)}
    return graph


def search_folds(tuning: TuningConfig, info: DatasetInfo, config: TrainingConfig) -> int | None:
    """The folds of the cross-validation a classical search uses (None: it compares the
    trials on the validation rows). Without validation rows it always cross-validates."""
    if tuning.cv_folds is not None:
        return tuning.cv_folds
    return None if has_validation(info, config.val_split) else DEFAULT_FOLDS
