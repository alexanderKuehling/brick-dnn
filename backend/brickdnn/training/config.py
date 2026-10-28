"""Training settings (no PyTorch needed)."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

OPTIMIZERS = ("adam", "adamw", "sgd")
DEVICES = ("auto", "cpu", "cuda", "mps")
# Validation and test data together: at most this share of all samples.
MAX_HELD_OUT = 0.8


class ConfigError(ValueError):
    """Raised for invalid training settings. The message is shown to the user."""


def _int(data: dict[str, Any], name: str, label: str, default: int | None, lo: int, hi: int):
    value = data.get(name)
    if value is None or value == "":
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ConfigError(f"{label} must be a whole number.")
    try:
        number = float(value)
    except ValueError as exc:
        raise ConfigError(f"{label} must be a whole number.") from exc
    if not math.isfinite(number) or not number.is_integer():
        raise ConfigError(f"{label} must be a whole number.")
    number = int(number)
    if not lo <= number <= hi:
        raise ConfigError(f"{label} must be between {lo:,} and {hi:,}.")
    return number


def _float(
    data: dict[str, Any], name: str, label: str, default: float, lo: float, hi: float, open_lo: bool
) -> float:
    value = data.get(name)
    if value is None or value == "":
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ConfigError(f"{label} must be a number.")
    try:
        number = float(value)
    except ValueError as exc:
        raise ConfigError(f"{label} must be a number.") from exc
    if not math.isfinite(number) or number > hi or number < lo or (open_lo and number == lo):
        bound = "greater than" if open_lo else "at least"
        raise ConfigError(f"{label} must be {bound} {lo:g} and at most {hi:g}.")
    return number


def _choice(data: dict[str, Any], name: str, label: str, default: str, options: tuple[str, ...]):
    value = data.get(name)
    if value is None or value == "":
        return default
    value = str(value).strip().lower()
    if value not in options:
        raise ConfigError(f"{label} must be one of: {', '.join(options)}.")
    return value


@dataclass
class TrainingConfig:
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-3
    optimizer: str = "adam"
    weight_decay: float = 0.0
    # Share of the data held out for validation. Built-in datasets without a test set
    # validate on their official test split instead and ignore it.
    val_split: float = 0.2
    # Share of the data held out as a test set (0: none). The test set is only used
    # once, after training, to measure the final model. Built-in datasets use their
    # official test split as test set when this is above 0; validation data then comes
    # from their training split (``val_split``).
    test_split: float = 0.0
    device: str = "auto"
    # Train on at most this many samples (quick experiments on large datasets).
    max_samples: int | None = None
    seed: int | None = None

    @property
    def has_test(self) -> bool:
        return self.test_split > 0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> TrainingConfig:
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ConfigError("The training settings must be an object.")
        val_split = _float(data, "val_split", "Validation split", cls.val_split, 0.05, 0.5, False)
        test_split = _float(data, "test_split", "Test split", cls.test_split, 0.0, 0.5, False)
        if val_split + test_split > MAX_HELD_OUT + 1e-9:
            raise ConfigError(
                f"Validation and test split together can be at most {MAX_HELD_OUT:.0%}."
            )
        return cls(
            epochs=_int(data, "epochs", "Epochs", cls.epochs, 1, 1000),
            batch_size=_int(data, "batch_size", "Batch size", cls.batch_size, 1, 8192),
            learning_rate=_float(
                data, "learning_rate", "Learning rate", cls.learning_rate, 0.0, 10.0, True
            ),
            optimizer=_choice(data, "optimizer", "Optimizer", cls.optimizer, OPTIMIZERS),
            weight_decay=_float(
                data, "weight_decay", "Weight decay", cls.weight_decay, 0.0, 1.0, False
            ),
            val_split=val_split,
            test_split=test_split,
            device=_choice(data, "device", "Device", cls.device, DEVICES),
            max_samples=_int(data, "max_samples", "Max. samples", None, 10, 10**9),
            seed=_int(data, "seed", "Seed", None, 0, 2**31 - 1),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
