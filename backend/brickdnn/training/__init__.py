"""Train Brick DNN networks with PyTorch, and classical models with scikit-learn.

Everything importable from here works without PyTorch: settings, dataset metadata,
compatibility checks and the run-directory file format. The modules that need
PyTorch (``engine``, ``cli``, ``probe``) import it themselves; install it with
``pip install "brickdnn[train]"`` (or ``pip install torch torchvision``). Classical models
(``classical_engine``) need scikit-learn instead: ``pip install "brickdnn[classic]"``.
"""

from __future__ import annotations

import importlib.util

from .compat import Compatibility, TrainingPlan, check_compatibility, suggest_fixes
from .config import DEVICES, OPTIMIZERS, ConfigError, TrainingConfig
from .datasets import (
    BUILTIN_DATASETS,
    IMAGE_EXTENSIONS,
    BuiltinDataset,
    DatasetError,
    DatasetInfo,
    builtin_info,
    extract_image_zip,
    inspect_csv,
    inspect_image_folder,
    random_info,
)
from .state import ACTIVE_STATUSES, STATUSES, TERMINAL_STATUSES


def torch_installed() -> bool:
    """True if PyTorch can be imported (checked without importing it)."""
    return importlib.util.find_spec("torch") is not None


def torchvision_installed() -> bool:
    return importlib.util.find_spec("torchvision") is not None


def sklearn_installed() -> bool:
    """True if scikit-learn (for classical models) can be imported (checked without
    importing it)."""
    return importlib.util.find_spec("sklearn") is not None


__all__ = [
    "ACTIVE_STATUSES",
    "BUILTIN_DATASETS",
    "DEVICES",
    "IMAGE_EXTENSIONS",
    "OPTIMIZERS",
    "STATUSES",
    "TERMINAL_STATUSES",
    "BuiltinDataset",
    "Compatibility",
    "ConfigError",
    "DatasetError",
    "DatasetInfo",
    "TrainingConfig",
    "TrainingPlan",
    "builtin_info",
    "check_compatibility",
    "extract_image_zip",
    "inspect_csv",
    "inspect_image_folder",
    "random_info",
    "sklearn_installed",
    "suggest_fixes",
    "torch_installed",
    "torchvision_installed",
]
