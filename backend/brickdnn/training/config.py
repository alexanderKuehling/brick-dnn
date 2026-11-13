"""Training settings (no PyTorch needed)."""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from typing import Any

OPTIMIZERS = ("adam", "adamw", "sgd")
DEVICES = ("auto", "cpu", "cuda", "mps")
# Loss of a segmentation: the cross-entropy of every pixel, or that plus a Dice term (one
# minus the overlap of the predicted and the true area of each class), which helps when a
# class covers only a few pixels. Other tasks do not use the setting.
LOSSES = ("cross_entropy", "cross_entropy_dice")
# Hybrid networks (neural layers before a classical model): how the neural layers learn the
# features before the classical model is fitted on them. "pretrain": like a network of their
# own, with a temporary linear head on the target (the normal epochs, optimizer, early
# stopping …); "frozen": not at all (random features, a quick baseline). Other networks do
# not use the setting.
FEATURE_TRAINING = ("pretrain", "frozen")
# Segmentation keeps every layer's output for a whole batch of images in memory. A batch
# may need at most this much (MB): the default leaves room for PyTorch and the data in a
# training process that may use about 1.8 GB.
SEGMENTATION_MEMORY_MB = ("BRICKDNN_SEGMENTATION_MEMORY_MB", 900)


def segmentation_memory_bytes() -> int:
    """The memory one batch of a segmentation training may need on this machine."""
    variable, default = SEGMENTATION_MEMORY_MB
    try:
        value = float(os.environ.get(variable, "").strip())
    except ValueError:
        value = default
    return int((value if value >= 16 else default) * 1024 * 1024)


# Validation and test data together: at most this share of all samples.
MAX_HELD_OUT = 0.8
# Classical models fit on at most this many training rows; larger training sets are
# subsampled. Gaussian processes need O(n²) memory and O(n³) time (a classifier once per
# class, and its Laplace approximation is much slower than a regression), support vector
# machines O(n²) memory (kernel cache) and between O(n²) and O(n³) time. Gradient boosting
# grows about linearly with the rows (its features are binned into bytes); its limit only
# keeps very long fits in bounds.
# {limit (ClassicalModel.limit): (variable, default)}
CLASSICAL_MAX_ROWS = {
    "gaussian_process": ("BRICKDNN_GP_MAX_ROWS", 5_000),
    "gaussian_process_classifier": ("BRICKDNN_GPC_MAX_ROWS", 2_000),
    "svm": ("BRICKDNN_SVM_MAX_ROWS", 50_000),
    "gradient_boosting": ("BRICKDNN_GB_MAX_ROWS", 1_000_000),
}


def classical_max_rows(limit: str) -> int:
    """The row limit ``limit`` (a key of CLASSICAL_MAX_ROWS) on this machine."""
    variable, default = CLASSICAL_MAX_ROWS[limit]
    try:
        value = int(os.environ.get(variable, "").strip())
    except ValueError:
        return default
    return value if value >= 10 else default


# Without validation data (a validation split of 0, or a prepared split without validation
# rows) a training keeps the weights of its last epoch; these need validation data.
NO_VALIDATION_PATIENCE = (
    "Early stopping needs validation data: it stops when the validation loss no longer "
    "improves. Set a validation split above 0 (for a dataset from the Data tab: a validation "
    "share above 0 there), or turn early stopping off."
)
NO_VALIDATION_SEARCH = (
    "A hyperparameter search needs validation data: it compares its trials by their "
    "validation loss. Set a validation split above 0 (for a dataset from the Data tab: a "
    "validation share above 0 there), or train without a search."
)


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


def _bool(data: dict[str, Any], name: str, label: str, default: bool) -> bool:
    value = data.get(name)
    if value is None or value == "":
        return default
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    if not isinstance(value, bool):
        raise ConfigError(f"{label} must be true or false.")
    return value


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
    # Share of the data held out for validation (built-in datasets: of their training split).
    # 0: no validation data; then the weights of the last epoch are kept, and early stopping
    # and hyperparameter searches are not possible (they compare validation losses).
    val_split: float = 0.2
    # Share of the data held out as a test set (0: none). The test set is only used
    # once, after training, to measure the kept weights. Built-in datasets do not use it:
    # their official test split always is the test set.
    test_split: float = 0.0
    device: str = "auto"
    # Train on at most this many samples (quick experiments on large datasets).
    max_samples: int | None = None
    seed: int | None = None
    # Early stopping: end the training when the validation loss has not improved for this
    # many epochs (None: always train all epochs).
    patience: int | None = None
    # Keep the weights of the epoch with the lowest validation loss (they are tested and
    # saved) instead of those of the last epoch.
    restore_best: bool = True
    # Segmentation only (see LOSSES).
    loss: str = "cross_entropy"
    # Hybrid networks only (see FEATURE_TRAINING).
    feature_training: str = "pretrain"

    @property
    def has_test(self) -> bool:
        """True if a test set is held out (built-in datasets always test on their official
        test split, whatever this says)."""
        return self.test_split > 0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> TrainingConfig:
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ConfigError("The training settings must be an object.")
        val_split = _float(data, "val_split", "Validation split", cls.val_split, 0.0, 0.5, False)
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
            patience=_int(data, "patience", "Patience", None, 1, 1000),
            restore_best=_bool(data, "restore_best", "Restore best weights", cls.restore_best),
            loss=_choice(data, "loss", "Loss", cls.loss, LOSSES),
            feature_training=_choice(
                data,
                "feature_training",
                "Feature training",
                cls.feature_training,
                FEATURE_TRAINING,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
