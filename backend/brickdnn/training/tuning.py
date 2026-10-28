"""Hyperparameter search: settings and the list of trials (no PyTorch needed).

A search trains the same network several times (*trials*), each time with other values
for some settings, on exactly the same training and validation data. The trial with the
lowest validation loss after its last epoch wins; its weights are kept. A test set, if
there is one, is only used afterwards to measure the winner.

The search space maps setting names to the values to try::

    {"learning_rate": {"min": 1e-4, "max": 1e-2},   # a range, sampled log-uniformly
     "batch_size": [32, 64, 128],                   # a list of values
     "optimizer": ["adam", "sgd"]}

Settings that are not in the space keep the value of the normal training settings.
``random`` search draws ``trials`` different combinations; ``grid`` search tries every
combination of the listed values (ranges are not allowed there).
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass, field, replace
from typing import Any

from .config import OPTIMIZERS, ConfigError, TrainingConfig

METHODS = ("random", "grid")
# Settings that can be tuned, in the order they are shown.
TUNABLE = ("learning_rate", "batch_size", "optimizer", "weight_decay")
LABELS = {
    "learning_rate": "learning rate",
    "batch_size": "batch size",
    "optimizer": "optimizer",
    "weight_decay": "weight decay",
}
OPTIMIZER_NAMES = {"adam": "Adam", "adamw": "AdamW", "sgd": "SGD"}
MAX_TRIALS = 100
MAX_VALUES = 20
DEFAULT_TRIALS = 10
DEFAULT_SPACE: dict[str, Any] = {
    "learning_rate": {"min": 1e-4, "max": 1e-2},
    "batch_size": [32, 64, 128],
}
# Sampled numbers are rounded to this many significant digits (easier to read and reuse).
DIGITS = 3


@dataclass(frozen=True)
class Range:
    """Numbers between ``min`` and ``max``, sampled log-uniformly."""

    min: float
    max: float

    def sample(self, rng: random.Random) -> float:
        if self.min == self.max:
            return self.min
        value = math.exp(rng.uniform(math.log(self.min), math.log(self.max)))
        return min(max(float(f"{value:.{DIGITS}g}"), self.min), self.max)

    def to_dict(self) -> dict[str, float]:
        return {"min": self.min, "max": self.max}


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
    elif name == "batch_size":
        values = [_number(v, "batch sizes", 1, 8192, False, True) for v in raw]
    elif name == "learning_rate":
        values = [_number(v, "learning rates", 0.0, 10.0, True, False) for v in raw]
    else:
        values = [_number(v, "weight decays", 0.0, 1.0, False, False) for v in raw]
    return tuple(dict.fromkeys(values))  # without duplicates, in the given order


def _range(name: str, raw: dict[str, Any]) -> Range:
    label = LABELS[name]
    if name not in ("learning_rate", "weight_decay"):
        raise ConfigError(f"Hyperparameter search: give a list of values for the {label}.")
    if set(raw) - {"min", "max"} or "min" not in raw or "max" not in raw:
        raise ConfigError(f'Hyperparameter search: a range for the {label} needs "min" and "max".')
    hi = 10.0 if name == "learning_rate" else 1.0
    lo = _number(raw["min"], f"the {label} range", 0.0, hi, True, False)
    top = _number(raw["max"], f"the {label} range", 0.0, hi, True, False)
    if lo > top:
        raise ConfigError(
            f"Hyperparameter search: the smallest {label} must not be larger than the largest."
        )
    return Range(lo, top)


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


@dataclass
class TuningConfig:
    method: str = "random"
    # Planned number of trials. Grid search: the number of combinations. Random search
    # over lists only: at most the number of combinations (every one is tried once).
    trials: int = DEFAULT_TRIALS
    space: dict[str, Range | Values] = field(default_factory=lambda: _space(None))

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> TuningConfig:
        """Validate the search settings. Raises :class:`ConfigError` (shown to the user)."""
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ConfigError("The hyperparameter search settings must be an object.")
        method = str(data.get("method") or "random").strip().lower()
        if method not in METHODS:
            raise ConfigError(
                f"Hyperparameter search: the method must be one of: {', '.join(METHODS)}."
            )
        space = _space(data.get("space"))
        combinations = _combinations(space)
        if method == "grid":
            for name, values in space.items():
                if isinstance(values, Range) and values.min != values.max:
                    raise ConfigError(
                        f"Grid search needs a list of values for the {LABELS[name]}, not a range."
                    )
            assert combinations is not None
            if combinations > MAX_TRIALS:
                raise ConfigError(
                    f"The grid has {combinations:,} combinations, at most {MAX_TRIALS} are "
                    "possible. Use fewer values or random search."
                )
            return cls(method, combinations, space)
        trials = data.get("trials")
        if trials is None or trials == "":
            trials = DEFAULT_TRIALS
        trials = int(_number(trials, "the number of trials", 1, MAX_TRIALS, False, True))
        if combinations is not None:
            trials = min(trials, combinations)
        return cls(method, trials, space)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "trials": self.trials,
            "space": {
                name: values.to_dict() if isinstance(values, Range) else list(values)
                for name, values in self.space.items()
            },
        }

    def sample(self, seed: int) -> list[dict[str, Any]]:
        """The settings of every trial (only the tuned ones), reproducible for a seed."""
        names = list(self.space)
        if self.method == "grid":
            lists = [self._as_values(self.space[name]) for name in names]
            return [dict(zip(names, combo, strict=True)) for combo in itertools.product(*lists)]
        rng = random.Random(seed)
        if _combinations(self.space) is not None:
            lists = [self._as_values(self.space[name]) for name in names]
            combos = list(itertools.product(*lists))
            chosen = rng.sample(combos, min(self.trials, len(combos)))
            return [dict(zip(names, combo, strict=True)) for combo in chosen]
        trials: list[dict[str, Any]] = []
        seen: set[tuple[Any, ...]] = set()
        attempts = 0
        while len(trials) < self.trials:
            params = {name: self._draw(self.space[name], rng) for name in names}
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


def trial_config(base: TrainingConfig, params: dict[str, Any]) -> TrainingConfig:
    """The training settings of one trial."""
    return replace(base, **params)


def format_value(name: str, value: Any) -> str:
    if name == "optimizer":
        return OPTIMIZER_NAMES.get(value, str(value))
    if name == "batch_size":
        return str(value)
    return f"{value:g}"


def describe_params(params: dict[str, Any]) -> str:
    """``"learning rate 0.00312, batch size 64, Adam"``."""
    parts = []
    for name in TUNABLE:
        if name in params:
            value = format_value(name, params[name])
            parts.append(value if name == "optimizer" else f"{LABELS[name]} {value}")
    return ", ".join(parts)
