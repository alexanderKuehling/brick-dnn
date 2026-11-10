"""What a training run gets and what it returns (no PyTorch needed).

:class:`TrainingJob` is the content of ``job.json``; both engines (PyTorch for neural
networks, scikit-learn for classical models) take one and return a :class:`TrainingResult`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..graph import analyze
from .config import ConfigError, TrainingConfig
from .datasets import DatasetInfo, builtin_info, random_info
from .reporting import TrainingError
from .tuning import TuningConfig


@dataclass
class TrainingJob:
    """Everything needed for one run (the content of ``job.json``)."""

    graph: dict[str, Any]
    dataset: DatasetInfo
    config: TrainingConfig = field(default_factory=TrainingConfig)
    path: str | None = None  # CSV file, image folder or segmentation folder
    data_root: str | None = None  # where torchvision datasets are downloaded
    name: str = ""
    tuning: TuningConfig | None = None  # set for a hyperparameter search

    @property
    def family(self) -> str:
        """``classical`` (scikit-learn) or ``neural`` (PyTorch), see :func:`graph_family`."""
        return graph_family(self.graph)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingJob:
        """Read a job (see the contract / state.py). Raises :class:`TrainingError`."""
        if not isinstance(data, dict):
            raise TrainingError("The training job is invalid.")
        graph = data.get("graph")
        if not isinstance(graph, dict):
            raise TrainingError("The training job contains no network.")
        dataset = data.get("dataset") or {}
        if not isinstance(dataset, dict):
            raise TrainingError("The training job describes no valid dataset.")
        info = dataset.get("info")
        try:
            if isinstance(info, dict):
                dataset_info = DatasetInfo.from_dict(info)
            elif dataset.get("kind") == "builtin":
                dataset_info = builtin_info(str(dataset.get("id")))
            elif dataset.get("kind", "random") == "random":
                dataset_info = random_info()
            else:
                raise TrainingError("The training job describes no valid dataset.")
        except (TypeError, ValueError) as exc:
            raise TrainingError("The training job describes no valid dataset.") from exc
        try:
            config = TrainingConfig.from_dict(data.get("config"))
            tuning = data.get("tuning")
            tuning = TuningConfig.from_dict(tuning) if tuning is not None else None
        except ConfigError as exc:
            raise TrainingError(str(exc)) from exc
        path = dataset.get("path")
        data_root = data.get("data_root")
        return cls(
            graph=graph,
            dataset=dataset_info,
            config=config,
            path=str(path) if path else None,
            data_root=str(data_root) if data_root else None,
            name=str(data.get("name") or ""),
            tuning=tuning,
        )


@dataclass
class TrainingResult:
    status: str  # completed | stopped | failed
    message: str
    weights: bool = False
    history: list[dict[str, Any]] = field(default_factory=list)
    best: dict[str, Any] | None = None
    # {"loss", "accuracy", "rmse", "r2"} of the final model on the test set (a
    # segmentation adds "iou", "dice", "pixel_accuracy", "class_iou", "class_dice")
    test: dict[str, Any] | None = None
    # Hyperparameter search: every trial (see state.py) and the number of the best one
    trials: list[dict[str, Any]] = field(default_factory=list)
    best_trial: int | None = None
    # The epoch whose weights were tested and saved (None: not the end of an epoch)
    weights_epoch: int | None = None
    # True if early stopping ended the training (in a search: the best trial)
    stopped_early: bool = False


def graph_family(graph: Any) -> str:
    """``classical`` if ``graph`` has a classical model or kernel bricks (it is trained with
    scikit-learn), otherwise ``neural`` (PyTorch); also for graphs that cannot be read."""
    try:
        return analyze(graph).family
    except Exception:  # pragma: no cover - analyze() reports problems instead of raising
        return "neural"
