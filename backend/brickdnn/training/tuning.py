"""Hyperparameter search: settings and the list of trials (no PyTorch needed).

A search trains the same network several times (*trials*), each time with other values
for some settings, on exactly the same training and validation data. The trial with the
best objective metric (by default the lowest validation loss, see objectives.py) wins; its
weights are kept. A test set, if there is one, is only used afterwards to measure the
winner.

The search space maps setting names to the values to try::

    {"learning_rate": {"min": 1e-4, "max": 1e-2},   # a range, sampled log-uniformly
     "batch_size": [32, 64, 128],                   # a list of values
     "optimizer": ["adam", "sgd"]}

Settings that are not in the space keep the value of the normal training settings.
``random`` search draws ``trials`` different combinations; ``grid`` search tries every
combination of the listed values (ranges are not allowed there).

A classical model (scikit-learn) searches over the parameters of its brick instead, and
may cross-validate on the training rows (``cv_folds``); see classical_tuning.py. All
settings together::

    {"method": "random", "trials": 20, "metric": "roc_auc", "cv_folds": 5,
     "space": {"C": {"min": 0.1, "max": 100}, "kernel": ["rbf", "linear"]}}

A hybrid network (neural layers that learn the features of a classical model, see
hybrid_engine.py) searches the parameters of its classical model in ``space`` and the
training settings of its neural layers in ``neural_space`` (the space of a network), one of
them or both::

    {"method": "random", "trials": 10,
     "neural_space": {"learning_rate": {"min": 1e-4, "max": 1e-2}, "batch_size": [32, 64]},
     "space": {"C": {"min": 0.1, "max": 100}}}

Without ``neural_space`` the features are learned once and only the classical model is
searched (``space`` defaults to the model's default space). With it, every trial learns the
features again with its settings (``space`` then defaults to nothing) and is ranked by the
classical model fitted on its features, on the validation rows: such a search needs
validation rows and cannot cross-validate, because the features of the training rows were
learned from those rows, which favours the settings that memorise them. In the params of a
trial the neural settings have the prefix ``neural.`` (``{"neural.learning_rate": 0.003,
"C": 3.2}``): gradient boosting has a learning rate of its own.
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass, field, replace
from typing import Any

from .config import OPTIMIZERS, ConfigError, TrainingConfig
from .losses import FOCAL_GAMMA, HUBER_DELTA, LABEL_SMOOTHING, LOSSES
from .losses import LABELS as LOSS_LABELS
from .objectives import DEFAULT_METRIC, parse_metric

METHODS = ("random", "grid")
# Settings that can be tuned, in the order they are shown. The loss and its settings (see
# losses.py) change what the loss measures: a search over losses is ranked by another
# metric than the loss.
TUNABLE = (
    "learning_rate",
    "batch_size",
    "optimizer",
    "weight_decay",
    "loss",
    "focal_gamma",
    "label_smoothing",
    "huber_delta",
)
LABELS = {
    "learning_rate": "learning rate",
    "batch_size": "batch size",
    "optimizer": "optimizer",
    "weight_decay": "weight decay",
    "loss": "loss",
    "focal_gamma": "focal gamma",
    "label_smoothing": "label smoothing",
    "huber_delta": "Huber delta",
}
# Ranges of the settings drawn evenly (not on a log scale): (smallest, largest, smallest
# excluded).
LINEAR_RANGES = {
    "focal_gamma": FOCAL_GAMMA,
    "label_smoothing": LABEL_SMOOTHING,
    "huber_delta": HUBER_DELTA,
}
OPTIMIZER_NAMES = {"adam": "Adam", "adamw": "AdamW", "sgd": "SGD"}
MAX_TRIALS = 100
MAX_VALUES = 20
DEFAULT_TRIALS = 10
DEFAULT_SPACE: dict[str, Any] = {
    "learning_rate": {"min": 1e-4, "max": 1e-2},
    "batch_size": [32, 64, 128],
}
# Cross-validation (classical models only): the number of folds of the training rows.
MIN_FOLDS = 2
MAX_FOLDS = 10
# Sampled numbers are rounded to this many significant digits (easier to read and reuse).
DIGITS = 3
# Hybrid networks: the prefix of the neural training settings in the params of a trial.
NEURAL_PREFIX = "neural."
NEURAL_CV = (
    "Hyperparameter search: cross-validation (cv_folds) is not available when the neural "
    "settings are searched: the features of the training rows were learned from them, so "
    "the trials are compared on the validation rows. Leave out cv_folds, or search only the "
    "classical model."
)
NEURAL_SPACE_FOR_NETWORK = (
    "Hyperparameter search: neural_space is for hybrid networks (neural layers before a "
    "classical model). A neural network searches its training settings in space."
)


@dataclass(frozen=True)
class Range:
    """Numbers between ``min`` and ``max``, sampled log-uniformly (``log``, the default) or
    uniformly. An ``integer`` range gives whole numbers."""

    min: float
    max: float
    log: bool = True
    integer: bool = False

    def sample(self, rng: random.Random) -> float:
        if self.min == self.max:
            return self.min
        if self.log:
            value = math.exp(rng.uniform(math.log(self.min), math.log(self.max)))
        else:
            value = rng.uniform(self.min, self.max)
        if self.integer:
            return int(min(max(round(value), self.min), self.max))
        return min(max(float(f"{value:.{DIGITS}g}"), self.min), self.max)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"min": self.min, "max": self.max}
        if not self.log:
            data["scale"] = "linear"
        return data


Values = tuple[Any, ...]


def _number(value: Any, what: str, lo: float, hi: float, open_lo: bool, whole: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ConfigError(f"Hyperparameter search: {what} must be numbers.")
    try:
        number = float(value)
    except ValueError as exc:
        raise ConfigError(f"Hyperparameter search: '{value}' is not a number ({what}).") from exc
    if not math.isfinite(number) or (whole and not number.is_integer()):
        kind = "whole numbers" if whole else "numbers"
        raise ConfigError(f"Hyperparameter search: {what} must be {kind}.")
    if number > hi or number < lo or (open_lo and number == lo):
        bound = "greater than" if open_lo else "at least"
        raise ConfigError(
            f"Hyperparameter search: {what} must be {bound} {lo:g} and at most {hi:g}."
        )
    return int(number) if whole else number


def _values(name: str, raw: list[Any]) -> Values:
    label = LABELS[name]
    if not raw:
        raise ConfigError(f"Hyperparameter search: give at least one {label}.")
    if len(raw) > MAX_VALUES:
        raise ConfigError(f"Hyperparameter search: at most {MAX_VALUES} values per setting.")
    if name == "optimizer":
        values = []
        for item in raw:
            value = str(item).strip().lower()
            if value not in OPTIMIZERS:
                raise ConfigError(
                    f"Hyperparameter search: the optimizer must be one of: {', '.join(OPTIMIZERS)}."
                )
            values.append(value)
    elif name == "loss":
        values = []
        for item in raw:
            value = str(item).strip().lower()
            if value not in LOSSES:
                raise ConfigError(
                    f"Hyperparameter search: the loss must be one of: {', '.join(LOSSES)}."
                )
            values.append(value)
    elif name == "batch_size":
        values = [_number(v, "batch sizes", 1, 8192, False, True) for v in raw]
    elif name == "learning_rate":
        values = [_number(v, "learning rates", 0.0, 10.0, True, False) for v in raw]
    elif name in LINEAR_RANGES:
        lo, hi, open_lo = LINEAR_RANGES[name]
        values = [_number(v, f"the values of the {label}", lo, hi, open_lo, False) for v in raw]
    else:
        values = [_number(v, "weight decays", 0.0, 1.0, False, False) for v in raw]
    return tuple(dict.fromkeys(values))  # without duplicates, in the given order


def _range(name: str, raw: dict[str, Any]) -> Range:
    label = LABELS[name]
    if name in LINEAR_RANGES:
        return _linear_range(name, raw)
    if name not in ("learning_rate", "weight_decay"):
        raise ConfigError(f"Hyperparameter search: give a list of values for the {label}.")
    if set(raw) - {"min", "max", "scale"} or "min" not in raw or "max" not in raw:
        raise ConfigError(f'Hyperparameter search: a range for the {label} needs "min" and "max".')
    if raw.get("scale") not in (None, "log"):
        raise ConfigError(
            f'Hyperparameter search: the {label} is drawn on a log scale; leave out "scale".'
        )
    hi = 10.0 if name == "learning_rate" else 1.0
    lo = _number(raw["min"], f"the {label} range", 0.0, hi, True, False)
    top = _number(raw["max"], f"the {label} range", 0.0, hi, True, False)
    if lo > top:
        raise ConfigError(
            f"Hyperparameter search: the smallest {label} must not be larger than the largest."
        )
    return Range(lo, top)


def _linear_range(name: str, raw: dict[str, Any]) -> Range:
    """A range of a setting of the loss, drawn evenly (``"scale": "log"`` for a log scale,
    which needs a minimum above 0)."""
    label = LABELS[name]
    if set(raw) - {"min", "max", "scale"} or "min" not in raw or "max" not in raw:
        raise ConfigError(f'Hyperparameter search: a range for the {label} needs "min" and "max".')
    scale = raw.get("scale") or "linear"
    if scale not in ("linear", "log"):
        raise ConfigError(
            f'Hyperparameter search: the scale of the {label} must be "linear" or "log".'
        )
    lo_limit, hi, open_lo = LINEAR_RANGES[name]
    what = f"the {label} range"
    lo = _number(raw["min"], what, lo_limit, hi, open_lo or scale == "log", False)
    top = _number(raw["max"], what, lo_limit, hi, open_lo or scale == "log", False)
    if lo > top:
        raise ConfigError(
            f"Hyperparameter search: the smallest {label} must not be larger than the largest."
        )
    return Range(lo, top, log=scale == "log")


def _space(raw: Any) -> dict[str, Range | Values]:
    if raw is None:
        raw = DEFAULT_SPACE
    if not isinstance(raw, dict):
        raise ConfigError("Hyperparameter search: the search space must be an object.")
    unknown = [name for name in raw if name not in TUNABLE]
    if unknown:
        raise ConfigError(
            f"Hyperparameter search: '{unknown[0]}' cannot be tuned. "
            f"Possible: {', '.join(TUNABLE)}."
        )
    space: dict[str, Range | Values] = {}
    for name in TUNABLE:
        value = raw.get(name)
        if value is None:
            continue
        if isinstance(value, dict):
            space[name] = _range(name, value)
        elif isinstance(value, (list, tuple)):
            space[name] = _values(name, list(value))
        else:
            space[name] = _values(name, [value])
    if not space:
        raise ConfigError("Hyperparameter search: choose at least one setting to tune.")
    return space


def _combinations(space: dict[str, Range | Values]) -> int | None:
    """Number of different combinations, or None if a range makes it unlimited."""
    count = 1
    for values in space.values():
        if isinstance(values, Range):
            if values.min != values.max:
                return None
        else:
            count *= len(values)
    return count


def _folds(value: Any) -> int | None:
    if value is None or value == "" or value is False:
        return None
    what = "the number of cross-validation folds"
    return int(_number(value, what, MIN_FOLDS, MAX_FOLDS, False, True))


@dataclass
class TuningConfig:
    method: str = "random"
    # Planned number of trials. Grid search: the number of combinations. Random search
    # over lists only: at most the number of combinations (every one is tried once).
    trials: int = DEFAULT_TRIALS
    space: dict[str, Range | Values] = field(default_factory=lambda: _space(None))
    # What ranks the trials (see objectives.py).
    metric: str = DEFAULT_METRIC
    # Classical models only: compare the trials by k-fold cross-validation on the training
    # rows instead of on the validation rows (None: the validation rows, and without them
    # cross-validation with DEFAULT_FOLDS of classical_tuning.py).
    cv_folds: int | None = None
    # The type of the classical model brick whose parameters are searched; None for a
    # neural network (its training settings are searched).
    model: str | None = None
    # A hybrid network: the training settings of its neural layers to search (None: the
    # features are learned once and only the classical model is searched). ``space`` may
    # then be empty.
    neural_space: dict[str, Range | Values] | None = None

    @classmethod
    def from_dict(
        cls, data: dict[str, Any] | None, model: str | None = None, hybrid: bool = False
    ) -> TuningConfig:
        """Validate the search settings. ``model``: the type of the classical model brick
        (None: a neural network); ``hybrid``: neural layers learn its features, whose
        training settings may be searched as well (``neural_space``). Raises
        :class:`ConfigError` (shown to the user)."""
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ConfigError("The hyperparameter search settings must be an object.")
        method = str(data.get("method") or "random").strip().lower()
        if method not in METHODS:
            raise ConfigError(
                f"Hyperparameter search: the method must be one of: {', '.join(METHODS)}."
            )
        metric = parse_metric(data.get("metric"))
        folds = _folds(data.get("cv_folds"))
        neural: dict[str, Range | Values] | None = None
        if data.get("neural_space") is not None and not (hybrid and model is not None):
            raise ConfigError(NEURAL_SPACE_FOR_NETWORK)
        if model is None:
            space = _space(data.get("space"))
            if folds is not None:
                raise ConfigError(
                    "Hyperparameter search: cross-validation (cv_folds) is only available for "
                    "classical models; a neural network compares its trials on the "
                    "validation data."
                )
        elif data.get("neural_space") is not None:
            neural = _space(data["neural_space"])
            if folds is not None:
                raise ConfigError(NEURAL_CV)
            space = _classical_space(model, data.get("space"), hybrid, required=False)
        else:
            space = _classical_space(model, data.get("space"), hybrid, required=True)
        merged = _merged(space, neural)
        combinations = _combinations(merged)
        if method == "grid":
            for name, values in merged.items():
                if isinstance(values, Range) and values.min != values.max:
                    raise ConfigError(
                        f"Grid search needs a list of values for the {param_label(name, model)}, "
                        "not a range."
                    )
            assert combinations is not None
            if combinations > MAX_TRIALS:
                raise ConfigError(
                    f"The grid has {combinations:,} combinations, at most {MAX_TRIALS} are "
                    "possible. Use fewer values or random search."
                )
            return cls(method, combinations, space, metric, folds, model, neural)
        trials = data.get("trials")
        if trials is None or trials == "":
            trials = DEFAULT_TRIALS
        trials = int(_number(trials, "the number of trials", 1, MAX_TRIALS, False, True))
        if combinations is not None:
            trials = min(trials, combinations)
        return cls(method, trials, space, metric, folds, model, neural)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "method": self.method,
            "trials": self.trials,
            "space": _space_dict(self.space),
            "metric": self.metric,
        }
        if self.model is not None:
            data["cv_folds"] = self.cv_folds
        if self.neural_space is not None:
            data["neural_space"] = _space_dict(self.neural_space)
        return data

    @property
    def searches_neural(self) -> bool:
        """A hybrid network's search whose trials learn the features with settings of their
        own (not only the classical model's parameters)."""
        return self.neural_space is not None

    def sample(self, seed: int) -> list[dict[str, Any]]:
        """The settings of every trial (only the tuned ones), reproducible for a seed. The
        neural settings of a hybrid network have the prefix ``neural.``."""
        space = _merged(self.space, self.neural_space)
        names = list(space)
        if self.method == "grid":
            lists = [self._as_values(space[name]) for name in names]
            return [dict(zip(names, combo, strict=True)) for combo in itertools.product(*lists)]
        rng = random.Random(seed)
        if _combinations(space) is not None:
            lists = [self._as_values(space[name]) for name in names]
            combos = list(itertools.product(*lists))
            chosen = rng.sample(combos, min(self.trials, len(combos)))
            return [dict(zip(names, combo, strict=True)) for combo in chosen]
        trials: list[dict[str, Any]] = []
        seen: set[tuple[Any, ...]] = set()
        attempts = 0
        while len(trials) < self.trials:
            params = {name: self._draw(space[name], rng) for name in names}
            key = tuple(params.values())
            attempts += 1
            # A narrow range may not have enough different (rounded) values.
            if key in seen and attempts < 50 * self.trials:
                continue
            seen.add(key)
            trials.append(params)
        return trials

    @staticmethod
    def _as_values(values: Range | Values) -> Values:
        return (values.min,) if isinstance(values, Range) else values

    @staticmethod
    def _draw(values: Range | Values, rng: random.Random) -> Any:
        return values.sample(rng) if isinstance(values, Range) else rng.choice(values)

    def describe(self) -> str:
        """``"random search, 10 trials"``."""
        method = "grid search" if self.method == "grid" else "random search"
        return f"{method}, {self.trials} trial{'s' if self.trials != 1 else ''}"


def _space_dict(space: dict[str, Range | Values]) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for name, values in space.items():
        if not isinstance(values, Range):
            data[name] = list(values)
            continue
        data[name] = values.to_dict()
        if values.log and name.removeprefix(NEURAL_PREFIX) in LINEAR_RANGES:
            data[name]["scale"] = "log"  # these are drawn evenly by default
    return data


def _merged(
    space: dict[str, Range | Values], neural: dict[str, Range | Values] | None
) -> dict[str, Range | Values]:
    """The space of the params of a trial: a hybrid network's neural settings with their
    prefix first, then the parameters of the classical model."""
    prefixed = {f"{NEURAL_PREFIX}{name}": values for name, values in (neural or {}).items()}
    return {**prefixed, **space}


def _classical_space(
    model: str, raw: Any, hybrid: bool, required: bool
) -> dict[str, Range | Values]:
    """The space of the classical model brick ``model``. ``required``: it is all that is
    searched (without ``raw``: the model's default space); otherwise the neural settings of
    a hybrid network are searched, and without ``raw`` (or with ``{}``) the model keeps the
    parameters of its brick. A hybrid network's training settings in ``raw`` are
    explained."""
    from .classical_tuning import parse_space, tunables

    if not required and (raw is None or raw == {}):
        return {}
    if hybrid and isinstance(raw, dict):
        params = {tunable.name for tunable in tunables(model)}
        settings = [name for name in raw if name in TUNABLE and name not in params]
        if settings:
            raise ConfigError(
                f"Hyperparameter search: '{settings[0]}' is a training setting of the neural "
                "layers: put it into neural_space (space holds the parameters of the "
                "classical model)."
            )
    return parse_space(model, raw)


def split_params(params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """The params of a hybrid network's trial as ``(neural settings, parameters of the
    classical model)``, the neural settings without their prefix."""
    neural = {
        name.removeprefix(NEURAL_PREFIX): value
        for name, value in params.items()
        if name.startswith(NEURAL_PREFIX)
    }
    model = {name: value for name, value in params.items() if not name.startswith(NEURAL_PREFIX)}
    return neural, model


def param_label(name: str, model: str | None = None) -> str:
    """``"learning rate"``; for a classical model (``model``: the type of its brick) the
    label of the brick parameter; ``"learning rate of the neural layers"`` for a neural
    setting of a hybrid network (``neural.learning_rate``)."""
    if name.startswith(NEURAL_PREFIX):
        setting = name.removeprefix(NEURAL_PREFIX)
        return f"{LABELS.get(setting, setting)} of the neural layers"
    if model is not None:
        from .classical_tuning import tunable_label

        return tunable_label(model, name)
    return LABELS.get(name, name)


def trial_config(base: TrainingConfig, params: dict[str, Any]) -> TrainingConfig:
    """The training settings of one trial."""
    return replace(base, **params)


def format_value(name: str, value: Any) -> str:
    name = name.removeprefix(NEURAL_PREFIX)
    if name == "optimizer":
        return OPTIMIZER_NAMES.get(value, str(value))
    if name == "loss":
        return LOSS_LABELS.get(value, str(value))
    if value is None:
        return "no limit"
    if name == "batch_size" or isinstance(value, (bool, str, int)):
        return str(value)
    return f"{value:g}"


def describe_params(params: dict[str, Any], model: str | None = None) -> str:
    """``"learning rate 0.00312, batch size 64, Adam"``; for a classical model (``model``:
    the type of its brick) ``"C 3.2, gamma 0.01"``; for a hybrid network that searches its
    neural settings ``"neural layers: learning rate 0.00312; C 3.2"``."""
    neural, rest = split_params(params)
    if neural:
        text = f"neural layers: {describe_params(neural)}"
        return f"{text}; {describe_params(rest, model)}" if rest else text
    if model is not None:
        return ", ".join(
            f"{param_label(name, model)} {format_value(name, value)}"
            for name, value in params.items()
        )
    parts = []
    for name in TUNABLE:
        if name in params:
            value = format_value(name, params[name])
            plain = name in ("optimizer", "loss")
            parts.append(value if plain else f"{LABELS[name]} {value}")
    return ", ".join(parts)
