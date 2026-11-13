"""Local training: dataset uploads, training processes and their status.

Every run is trained by its own process (``python -m brickdnn.training.runner <run_dir>``),
so the web server never imports PyTorch. The API writes ``job.json`` into the run
directory, starts the process and afterwards only looks at the files the process writes
(``state.json``, ``weights.pt``). See :mod:`brickdnn.training.state` for the file format.
Classical models (Gaussian processes, SVMs) are fitted with scikit-learn by the same
runner and saved as ``model.joblib``; they need scikit-learn, not PyTorch.
Predictions with the trained weights also run in a process of their own
(``python -m brickdnn.training.predict <run_dir>``), one per request.

Both report the CPU time they use, which is counted against the budgets of the account and of
the API token or app that started them (see budgets.py): every look at an active run counts
what it used since, and stops it when a budget is used up.

With ``BRICKDNN_TRAINING_WORKER`` the web server does not start the training processes
itself: :func:`start_run` only puts the run into the queue (training_queue.py), and a worker
(training_worker.py) takes it, starts the process with :func:`start_claimed` and watches it
with :func:`sync_run`, like the web server does without the setting. The web server then
only reads the files of the run; it stops a run through the ``STOP`` file and the database,
and the worker does the killing. Everything that reads those files goes through
:func:`read_run_state` and :func:`weights_path`, so that another store can replace the
shared directory later.
"""

from __future__ import annotations

import base64
import contextlib
import copy
import importlib.metadata
import json
import logging
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import datetime, timedelta
from datetime import timezone as dt_timezone
from pathlib import Path
from typing import Any

from django.conf import settings
from django.core.cache import cache
from django.db import transaction
from django.db.models.signals import post_delete, pre_delete
from django.dispatch import receiver
from django.utils import timezone
from django.utils.text import slugify
from rest_framework import serializers

from brickdnn import REGISTRY, Analysis, analyze
from brickdnn.layers import ClassicalModel
from brickdnn.training import (
    ACTIVE_STATUSES,
    BUILTIN_DATASETS,
    QUEUED,
    RUNNING_STATUSES,
    STATUSES,
    TERMINAL_STATUSES,
    Compatibility,
    DatasetError,
    DatasetInfo,
    TrainingConfig,
    builtin_info,
    check_compatibility,
    extract_image_zip,
    inference,
    inspect_csv,
    inspect_image_folder,
    inspect_segmentation_folder,
    is_segmentation_folder,
    random_info,
    sklearn_installed,
    torch_installed,
    torchvision_installed,
)
from brickdnn.training.compat import MAX_SEGMENTATION_SIDE
from brickdnn.training.config import (
    LOSSES,
    ConfigError,
    classical_max_rows,
    segmentation_memory_bytes,
)
from brickdnn.training.estimate import Estimate, estimate_training
from brickdnn.training.job import graph_family
from brickdnn.training.segmentation import MAX_PAIRS, MAX_SEGMENTATION_CLASSES, class_colors
from brickdnn.training.state import (
    LOG_FILE,
    PREVIEW_DIR,
    PREVIEW_INDEX_FILE,
    STATE_FILE,
    STOP_FILE,
    is_stale,
    preprocessing_path,
    read_json,
    read_state,
    request_stop,
    stop_request,
    stop_requested,
    weights_file,
    write_job,
)
from brickdnn.training.tuning import (
    MAX_TRIALS,
    MAX_VALUES,
    METHODS,
    TUNABLE,
    TuningConfig,
)

from . import budgets
from . import training_queue as queue
from .budgets import Actor
from .models import Dataset, TrainingRun
from .retention import touch
from .serializers import validate_graph_document

try:  # POSIX only: serialises run starts across server processes
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

DISABLED_REASON = "Training is disabled on this server."
NO_TORCH_REASON = (
    "PyTorch is not installed on the server. Run `pip install torch torchvision` in the "
    "backend environment (Docker: build with BRICKDNN_INSTALL_TORCH=true)."
)
NO_SKLEARN_REASON = (
    "scikit-learn is not installed on the server, which classical models (Gaussian processes, "
    "support vector machines) need. Run `pip install scikit-learn` in the backend environment."
)
NO_TUNING_MESSAGE = (
    "Hyperparameter search is not available for classical models yet. Switch it off: a "
    "Gaussian process optimises its kernel hyperparameters while it is fitted."
)
BUSY_MESSAGE = "Another training is still running. Stop it or wait until it finishes."
UNEXPECTED_MESSAGE = "The training process stopped unexpectedly."
NOT_STARTED_MESSAGE = "The training process did not start."
STOPPED_MESSAGE = "Training was stopped."
WRONG_FILE_MESSAGE = (
    "Upload a .csv file or a .zip file: with one folder of images per class, or for "
    "segmentation with the folders images/ and masks/."
)
UPLOAD_KINDS = ("images", "segmentation")
WRONG_KIND_MESSAGE = (
    "'kind' says what a .zip file holds: 'images' (one folder per class) or 'segmentation' "
    "(the folders images/ and masks/). Leave it out to let the folders decide."
)
IN_USE_MESSAGE = "This dataset is used by a training that is still running. Stop it first."
IN_USE_QUEUED_MESSAGE = (
    "This dataset is used by a training that waits in the queue. Cancel it first."
)
PREDICT_BUSY_MESSAGE = "The server is busy with other predictions. Try again in a few seconds."
# A prediction (starting Python, loading PyTorch and the weights, a forward pass on the CPU)
# that takes longer than this is stopped. It blocks a web server worker meanwhile.
PREDICT_TIMEOUT_SECONDS = 30
# CPU threads of a prediction (brickdnn.training.predict.THREADS), for counting its CPU time.
PREDICT_THREADS = 2

PROBE_CACHE_KEY = "brickdnn:training:probe"
PROBE_CACHE_SECONDS = 600
PROBE_FAILURE_CACHE_SECONDS = 30
PROBE_TIMEOUT_SECONDS = 120
# A run without any state.json this long after its start is considered dead.
START_TIMEOUT_SECONDS = 60
# A runner that has not reacted to a stop request after this long is killed.
STOP_TIMEOUT_SECONDS = 30
# A stale heartbeat must still be stale this long after it was first noticed before the
# run is given up. Wall-clock based heartbeats look stale right after a laptop wakes up.
STALE_GRACE_SECONDS = 15
# val_rmse, val_r2, test_rmse and test_r2: regression only (None otherwise)
METRIC_KEYS = ("train_loss", "train_accuracy", "val_loss", "val_accuracy", "val_rmse", "val_r2")
TEST_KEYS = {
    "test_loss": "loss",
    "test_accuracy": "accuracy",
    "test_rmse": "rmse",
    "test_r2": "r2",
}
TRIAL_KEYS = (
    "trial",
    "params",
    "val_loss",
    "val_accuracy",
    "val_rmse",
    "val_r2",
    "weights_epoch",
)
# Segmentation runs add these (other runs do not have the keys).
SEGMENTATION_METRIC_KEYS = ("val_iou", "val_dice", "val_pixel_accuracy")
SEGMENTATION_TEST_KEYS = {
    "test_iou": "iou",
    "test_dice": "dice",
    "test_pixel_accuracy": "pixel_accuracy",
}
# The preview images of a segmentation run (brickdnn.training.previews).
PREVIEW_PARTS = ("image", "truth", "prediction", "overlay")
PREVIEW_NAME = re.compile(r"[0-9]{1,3}_(?:image|truth|prediction|overlay)\.png")
# A preview image is small (see previews.py); anything larger is not served.
MAX_PREVIEW_BYTES = 1024 * 1024

# Training processes started by this server process, by run id. A daemon thread waits
# for each of them so that finished processes do not linger as zombies.
_PROCESSES: dict[str, subprocess.Popen] = {}
_START_LOCK = threading.Lock()
# The training worker this process is (TrainingWorker.pk) while it looks after its runs,
# see :func:`supervising`. None in the web server.
_SUPERVISOR = threading.local()


class TrainingInputError(ValueError):
    """Invalid request data. The message is shown to the user."""


class TrainingUnavailable(Exception):
    """Training cannot be used on this server (disabled or no PyTorch)."""


class TrainingBusy(Exception):
    """The maximum number of concurrent trainings is reached."""


class DatasetInUse(Exception):
    """A dataset cannot be deleted while a training uses it."""


class PredictionUnavailable(Exception):
    """The run cannot predict: no weights, or how its inputs were prepared is unknown."""


class PredictionFailed(Exception):
    """The prediction process failed or took too long. The message is shown to the user."""


class IncompatibleNetwork(Exception):
    """The network cannot be trained on the chosen dataset."""

    def __init__(self, compatibility: Compatibility):
        self.compatibility = compatibility
        super().__init__(compatibility.errors[0] if compatibility.errors else "Incompatible.")


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #


def training_dir() -> Path:
    return Path(settings.TRAINING_DIR)


def dataset_dir(dataset_id: int) -> Path:
    return training_dir() / "datasets" / str(dataset_id)


def run_dir(run_id: uuid.UUID | str) -> Path:
    return training_dir() / "runs" / str(run_id)


def builtin_data_dir() -> Path:
    return training_dir() / "builtin"


def dataset_path(dataset: Dataset) -> Path:
    """The path written into job.json: the CSV file or the folder that was inspected."""
    directory = dataset_dir(dataset.pk)
    return directory / "data.csv" if dataset.kind == "csv" else directory / "images"


def max_upload_bytes() -> int:
    return int(float(settings.TRAINING_MAX_UPLOAD_MB) * 1024 * 1024)


def _folder_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    base = str(settings.BASE_DIR)
    env["PYTHONPATH"] = base + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    return env


# --------------------------------------------------------------------------- #
# Capabilities
# --------------------------------------------------------------------------- #


def run_probe() -> dict[str, Any]:
    """Import PyTorch in a separate process and report its version and devices.

    Returns the probe's JSON, or ``{"error": "..."}`` if PyTorch cannot be loaded.
    """
    command = [sys.executable, "-m", "brickdnn.training.probe"]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_SECONDS,
            env=_subprocess_env(),
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        return {"error": "Loading PyTorch took too long."}
    except OSError as exc:
        return {"error": f"The PyTorch check could not be started: {exc}"}
    if result.returncode == 0:
        for line in reversed(result.stdout.strip().splitlines()):
            if line.strip().startswith("{"):
                try:
                    data = json.loads(line)
                except ValueError:
                    break
                if isinstance(data, dict):
                    return data
                break
    lines = [line.strip() for line in result.stderr.splitlines() if line.strip()]
    detail = lines[-1] if lines else f"exit code {result.returncode}"
    logger.warning("Training probe failed: %s", result.stderr[-2000:])
    return {"error": detail[:300]}


def probe() -> dict[str, Any]:
    """:func:`run_probe`, cached (importing PyTorch takes a few seconds)."""
    data = cache.get(PROBE_CACHE_KEY)
    if data is None:
        data = run_probe()
        timeout = PROBE_FAILURE_CACHE_SECONDS if "error" in data else PROBE_CACHE_SECONDS
        cache.set(PROBE_CACHE_KEY, data, timeout)
    return data


def unavailable_reason(family: str = "neural") -> str | None:
    """Why runs of neural networks (PyTorch) or ``classical`` models (scikit-learn) cannot
    be started right now (cheap: does not run the probe)."""
    if not settings.TRAINING_ENABLED:
        return DISABLED_REASON
    if family == "classical":
        return None if sklearn_installed() else NO_SKLEARN_REASON
    if not torch_installed():
        return NO_TORCH_REASON
    return None


def run_family(run: TrainingRun) -> str:
    """``classical`` for a run of a classical model (scikit-learn), else ``neural``."""
    plan = run.plan if isinstance(run.plan, dict) else {}
    return "classical" if isinstance(plan.get("classical"), dict) else "neural"


def _sklearn_version() -> str | None:
    try:
        return importlib.metadata.version("scikit-learn")
    except importlib.metadata.PackageNotFoundError:
        return None


def classical_capabilities() -> dict[str, Any]:
    """Whether classical models can be trained here (needs scikit-learn, not PyTorch)."""
    reason = unavailable_reason("classical")
    return {
        "available": reason is None,
        "reason": reason,
        "sklearn_version": _sklearn_version() if reason is None else None,
        # Training rows each classical model is fitted on at most (more are subsampled).
        "max_rows": {
            name: classical_max_rows(layer.limit)
            for name, layer in REGISTRY.items()
            if issubclass(layer, ClassicalModel)
        },
        "tuning": False,
    }


def capabilities(user=None) -> dict[str, Any]:
    """Everything the UI needs to know before offering training. ``user``: who asks (for
    what their plan means in the queue)."""
    reason = unavailable_reason()
    waiting = queue.status(user)
    has_torchvision = torchvision_installed()
    info: dict[str, Any] = {}
    if reason is None:
        info = probe()
        if "error" in info:
            reason = f"PyTorch is installed but cannot be loaded: {info['error']}"
    available = reason is None
    return {
        "enabled": bool(settings.TRAINING_ENABLED),
        "available": available,
        "reason": reason,
        "torch_version": info.get("torch_version") if available else None,
        "torchvision": bool(info.get("torchvision", has_torchvision)),
        "devices": info.get("devices", []) if available else [],
        "default_device": info.get("default_device") if available else None,
        "builtin_datasets": [
            # Generated datasets (Shapes) need no torchvision.
            {**dataset.to_dict(), "available": has_torchvision or dataset.generated}
            for dataset in BUILTIN_DATASETS.values()
        ],
        # Segmentation (a class for every pixel): what a dataset and a training may be.
        "segmentation": {
            "max_classes": MAX_SEGMENTATION_CLASSES,
            "max_images": MAX_PAIRS,
            "max_input_side": MAX_SEGMENTATION_SIDE,
            # A batch (every layer's output for all its images) may need this much memory.
            "batch_memory_mb": round(segmentation_memory_bytes() / 1024**2),
            "losses": list(LOSSES),
        },
        "max_upload_mb": settings.TRAINING_MAX_UPLOAD_MB,
        # Trainings that run at the same time (all users): the slots of the workers, or
        # BRICKDNN_TRAINING_MAX_RUNS when the web server trains itself.
        "max_concurrent_runs": (
            max(1, waiting["slots"]) if waiting is not None else settings.TRAINING_MAX_RUNS
        ),
        # With a training worker: new trainings wait in a queue (see training_queue.status);
        # null when the web server trains itself and refuses a start while it is busy.
        "queue": waiting,
        "login_required": True,
        "tuning": {
            "methods": list(METHODS),
            "tunable": list(TUNABLE),
            "max_trials": MAX_TRIALS,
            "max_values": MAX_VALUES,
            "default": TuningConfig().to_dict(),
        },
        # Classical models (scikit-learn): "available" above is about PyTorch.
        "classical": classical_capabilities(),
    }


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #


def save_upload(user, upload, name: str | None = None, kind: str | None = None) -> Dataset:
    """Store and inspect an uploaded CSV or ZIP file. Raises :class:`DatasetError`, or
    :class:`budgets.StorageQuotaExceeded` if it does not fit into the account's storage.

    A ZIP file with a folder of images and a folder of masks (``images/`` and ``masks/``)
    becomes a segmentation dataset, any other one a dataset with one folder per class;
    ``kind`` ("images" or "segmentation") says which one is meant, for clearer errors."""
    filename = Path(upload.name or "").name
    suffix = Path(filename).suffix.lower()
    if suffix not in (".csv", ".zip"):
        raise DatasetError(WRONG_FILE_MESSAGE)
    if kind in ("", None):
        kind = None
    elif kind not in UPLOAD_KINDS or suffix != ".zip":
        raise DatasetError(WRONG_KIND_MESSAGE)
    wanted = kind
    name = (str(name or "").strip() or Path(filename).stem.strip() or "Dataset")[:120]
    kind = "csv" if suffix == ".csv" else "images"
    budgets.check_storage(user, int(upload.size or 0))

    # Work in a temporary folder, so that a half-processed upload is never listed.
    datasets = training_dir() / "datasets"
    datasets.mkdir(parents=True, exist_ok=True)
    _remove_abandoned_uploads(datasets)
    work = datasets / f".upload-{uuid.uuid4().hex}"
    work.mkdir()
    dataset = None
    try:
        if kind == "csv":
            target = work / "data.csv"
            _copy_upload(upload, target)
            info = inspect_csv(target, name=name)
            size = target.stat().st_size
        else:
            source = getattr(upload, "temporary_file_path", None)
            if source is not None:
                archive = Path(source())
            else:  # small uploads are kept in memory
                archive = work / "upload.zip"
                _copy_upload(upload, archive)
            limit = max_upload_bytes()
            extract_image_zip(
                archive, work / "images", max_bytes=max(2 * limit, min(4 * limit, 4 * 1024**3))
            )
            if archive.parent == work:
                archive.unlink()
            if wanted == "segmentation" or (
                wanted is None and is_segmentation_folder(work / "images")
            ):
                info = inspect_segmentation_folder(work / "images", name=name)
                kind = "segmentation"
            else:
                info = inspect_image_folder(work / "images", name=name)
            size = _folder_size(work / "images")
            budgets.check_storage(user, size)  # unpacked, the images can take more room
        dataset = Dataset.objects.create(
            owner=user, name=name, kind=kind, size_bytes=size, info=info.to_dict()
        )
        final = dataset_dir(dataset.pk)
        shutil.rmtree(final, ignore_errors=True)  # left over from an earlier database
        os.replace(work, final)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        if dataset is not None:
            dataset.delete()
        raise
    return dataset


def _remove_abandoned_uploads(datasets: Path) -> None:
    """Work folders of uploads that were interrupted (e.g. by a server restart)."""
    cutoff = time.time() - 6 * 3600
    for path in datasets.glob(".upload-*"):
        with contextlib.suppress(OSError):
            if path.stat().st_mtime < cutoff:
                shutil.rmtree(path, ignore_errors=True)


def _copy_upload(upload, target: Path) -> None:
    with target.open("wb") as out:
        for chunk in upload.chunks():
            out.write(chunk)


def validate_graph(graph: Any) -> None:
    """Structural check of a graph document. Raises :class:`TrainingInputError`."""
    try:
        validate_graph_document(graph)
    except serializers.ValidationError as exc:
        detail = exc.detail
        message = detail[0] if isinstance(detail, list) and detail else detail
        raise TrainingInputError(str(message)) from None


@dataclass
class ResolvedDataset:
    info: DatasetInfo
    path: Path | None
    upload: Dataset | None
    summary: dict[str, Any]  # stored on the run: {"kind", "id", "name"}


def resolve_dataset(user, ref: Any) -> ResolvedDataset:
    """Turn a DatasetRef from the UI into dataset metadata. Raises :class:`TrainingInputError`."""
    if not isinstance(ref, dict):
        raise TrainingInputError("Choose a dataset.")
    kind = ref.get("kind")
    if kind == "random":
        info = random_info()
        return ResolvedDataset(info, None, None, {"kind": "random", "id": None, "name": info.name})
    if kind == "builtin":
        dataset_id = ref.get("id")
        if not isinstance(dataset_id, str):
            raise TrainingInputError("Choose one of the built-in datasets.")
        try:
            info = builtin_info(dataset_id)
        except DatasetError as exc:
            raise TrainingInputError(str(exc)) from None
        summary = {"kind": "builtin", "id": dataset_id, "name": info.name}
        return ResolvedDataset(info, None, None, summary)
    if kind == "upload":
        dataset_id = ref.get("id")
        if isinstance(dataset_id, str) and dataset_id.isdigit():
            dataset_id = int(dataset_id)
        if not isinstance(dataset_id, int) or isinstance(dataset_id, bool):
            raise TrainingInputError("Choose one of your uploaded datasets.")
        from .shared_data import visible_datasets

        # Your own datasets and the ones shared in your projects.
        dataset = visible_datasets(user).filter(pk=dataset_id).first()
        if dataset is None:
            raise TrainingInputError(
                "This dataset does not exist (any more). Upload it again or choose another one."
            )
        path = dataset_path(dataset)
        if not path.exists():
            raise TrainingInputError("The files of this dataset are missing. Upload it again.")
        touch(dataset)
        info = DatasetInfo.from_dict(dataset.info)
        summary = {"kind": "upload", "id": dataset.pk, "name": dataset.name}
        return ResolvedDataset(info, path, dataset, summary)
    raise TrainingInputError("Unknown dataset type. Use 'random', 'builtin' or 'upload'.")


def parse_config(config: Any) -> TrainingConfig | None:
    """Training settings from a request (None stays None). Raises
    :class:`TrainingInputError`."""
    if config is None:
        return None
    try:
        return TrainingConfig.from_dict(config)
    except ConfigError as exc:
        raise TrainingInputError(str(exc)) from None


def parse_tuning(tuning: Any) -> TuningConfig | None:
    """Search settings from a request (None stays None). Raises :class:`TrainingInputError`."""
    if tuning is None:
        return None
    try:
        return TuningConfig.from_dict(tuning)
    except ConfigError as exc:
        raise TrainingInputError(str(exc)) from None


def estimate_run(
    graph: Any,
    dataset: ResolvedDataset,
    config: TrainingConfig | None,
    plan: dict[str, Any] | None,
    search: TuningConfig | None = None,
) -> Estimate | None:
    """The estimated CPU time of a run (None if it cannot be estimated)."""
    analysis = graph if isinstance(graph, Analysis) else analyze(graph)
    trials = search.trials if search is not None else 1
    try:
        return estimate_training(analysis, dataset.info, config, plan, trials)
    except Exception:  # an estimate must never break a check or a start
        logger.exception("Could not estimate the CPU time of a training")
        return None


def budget_check(
    actor: Actor,
    graph: Any,
    dataset: ResolvedDataset,
    config: TrainingConfig | None,
    result: Compatibility,
    search: TuningConfig | None = None,
) -> dict[str, Any]:
    """``{"estimate", "budget"}`` for a training check: the estimated CPU time of the planned
    run and whether the budgets of ``actor`` allow it (see budgets.estimate_data)."""
    sync_runs(TrainingRun.objects.filter(owner=actor.user, status__in=ACTIVE_STATUSES))
    estimate = None
    if result.ok and result.plan is not None:
        estimate = estimate_run(graph, dataset, config, result.plan.to_dict(), search)
    data = budgets.estimate_data(actor, estimate)
    # With a training worker: where the run would wait and for how long (an estimate).
    data["queue"] = queue.preview(
        actor.user, estimate.cpu_seconds if estimate is not None else None
    )
    return data


def check(
    graph: Any,
    dataset: ResolvedDataset,
    config: TrainingConfig | None = None,
    search: TuningConfig | None = None,
) -> Compatibility:
    """check_compatibility() plus checks that depend on this server. For a graph document
    (not an :class:`Analysis`), the result suggests fixes as well. ``search``: the run is a
    hyperparameter search (it needs validation data)."""
    result = check_compatibility(graph, dataset.info, config, tuning=search is not None)
    family = graph.family if isinstance(graph, Analysis) else graph_family(graph)
    if family == "classical":
        reason = unavailable_reason("classical")
        if reason is not None and reason != DISABLED_REASON:
            result.errors.insert(0, reason)
            result.plan = None
        return result
    builtin = BUILTIN_DATASETS.get(dataset.info.builtin or "")
    generated = builtin is not None and builtin.generated
    if dataset.info.kind == "builtin" and not generated and not torchvision_installed():
        result.errors.insert(
            0,
            "torchvision is not installed on the server, so the built-in datasets cannot be "
            "used. Run `pip install torchvision` in the backend environment.",
        )
    return result


def dataset_in_use(dataset: Dataset) -> bool:
    """True if an active training uses ``dataset`` (after updating the run statuses)."""
    sync_runs(dataset.runs.filter(status__in=ACTIVE_STATUSES))
    return dataset.runs.filter(status__in=ACTIVE_STATUSES).exists()


def delete_dataset(dataset: Dataset) -> None:
    """Delete a dataset with its files. Raises :class:`DatasetInUse` while a training uses
    it. The caller checks that the user owns it."""
    if dataset_in_use(dataset):
        training = dataset.runs.filter(status__in=RUNNING_STATUSES).exists()
        raise DatasetInUse(IN_USE_MESSAGE if training else IN_USE_QUEUED_MESSAGE)
    dataset.delete()  # the files are removed by a post_delete signal


# --------------------------------------------------------------------------- #
# Starting runs
# --------------------------------------------------------------------------- #


@contextlib.contextmanager
def _start_lock():
    """Only one run is started at a time, also across server processes (POSIX)."""
    with _START_LOCK:
        root = training_dir()
        root.mkdir(parents=True, exist_ok=True)
        with open(root / ".lock", "a") as handle:
            if fcntl is not None:
                fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(handle, fcntl.LOCK_UN)


def start_lock():
    """The start lock, for the queue (claims on databases without row locks)."""
    return _start_lock()


def active_run_count() -> int:
    """Number of active runs of all users (after updating their status)."""
    sync_runs(TrainingRun.objects.filter(status__in=ACTIVE_STATUSES))
    return TrainingRun.objects.filter(status__in=ACTIVE_STATUSES).count()


def start_run(
    user,
    name: Any,
    graph: Any,
    dataset_ref: Any,
    config: Any,
    tuning: Any = None,
    project_id: Any = None,
    actor: Actor | None = None,
) -> TrainingRun:
    """Validate everything and start a training process.

    ``tuning`` (search settings, see brickdnn.training.tuning) turns the run into a
    hyperparameter search. ``actor``: who starts it (the API token or app, for their
    budgets; default: ``user`` in the web app). Raises :class:`TrainingUnavailable`,
    :class:`TrainingInputError` (also for ``ConfigError``), :class:`IncompatibleNetwork`,
    :class:`TrainingBusy` or a :class:`budgets.BudgetError`. The run's ``budget_warning``
    (not stored) says if it will probably be stopped by its limit.

    With a training worker (BRICKDNN_TRAINING_WORKER) the run is only put into the queue and
    returned as ``queued``: a worker starts it (see training_queue.py). It reserves what it
    may use of the budgets while it waits. :class:`TrainingBusy` then means that the account
    has as many trainings waiting as it may.
    """
    actor = actor or Actor(user)
    family = graph_family(graph)
    reason = unavailable_reason(family)
    if reason:
        raise TrainingUnavailable(reason)

    if name is None:
        name = ""
    if not isinstance(name, str):
        raise TrainingInputError("The name must be text.")
    name = name.strip() or "Training run"
    if len(name) > 120:
        raise TrainingInputError("The name can be at most 120 characters long.")
    validate_graph(graph)
    project = None
    if project_id is not None:
        from .shared_data import editable_projects

        if isinstance(project_id, bool) or not isinstance(project_id, int):
            raise TrainingInputError("'project' must be the id of a project.")
        project = editable_projects(user).filter(pk=project_id).first()
        if project is None:
            raise TrainingInputError("You can only train in projects you own or can edit.")
    try:
        training_config = TrainingConfig.from_dict(config)
        search = TuningConfig.from_dict(tuning) if tuning is not None else None
    except ConfigError as exc:
        raise TrainingInputError(str(exc)) from None
    if family == "classical" and search is not None:
        raise TrainingInputError(NO_TUNING_MESSAGE)
    dataset = resolve_dataset(user, dataset_ref)
    analysis = analyze(graph)
    compatibility = check(analysis, dataset, training_config, search)
    if not compatibility.ok:
        raise IncompatibleNetwork(compatibility)
    assert compatibility.plan is not None
    estimate = estimate_run(
        analysis, dataset, training_config, compatibility.plan.to_dict(), search
    )
    budgets.check_storage(user)

    def create(limit: budgets.RunLimit, **fields: Any) -> TrainingRun:
        return TrainingRun.objects.create(
            owner=user,
            project=project,
            name=name,
            graph=graph,
            dataset=dataset.summary,
            upload=dataset.upload,
            config=training_config.to_dict(),
            tuning=search.to_dict() if search is not None else None,
            plan=compatibility.plan.to_dict(),
            total_params=analysis.total_params,
            api_token=actor.token,
            connection=actor.connection,
            cpu_limit_seconds=limit.seconds,
            estimated_cpu_seconds=estimate.cpu_seconds if estimate is not None else None,
            **fields,
        )

    # Under the start lock: two starts cannot both use what is left of a budget.
    with _start_lock():
        if queue.enabled():
            # The run waits for a worker. Until then it reserves what it may use, so the
            # runs queued after it cannot count on the same rest of a budget.
            with queue.account_lock(user.pk):
                sync_runs(TrainingRun.objects.filter(owner=user, status__in=RUNNING_STATUSES))
                limit = budgets.run_limit(actor)
                refusal, warning = budgets.estimate_verdict(estimate, limit)
                if refusal is not None:
                    raise budgets.EstimateTooHigh(refusal, budgets.estimate_data(actor, estimate))
                refusal = budgets.waiting_refusal(user, queue.waiting_count(user))
                if refusal is not None:
                    raise TrainingBusy(refusal)
                run = create(
                    limit, status=QUEUED, message=queue.QUEUED_MESSAGE, queued_at=timezone.now()
                )
        else:
            active = active_run_count()  # also counts what the active runs used so far
            limit = budgets.run_limit(actor)
            refusal, warning = budgets.estimate_verdict(estimate, limit)
            if refusal is not None:
                raise budgets.EstimateTooHigh(refusal, budgets.estimate_data(actor, estimate))
            if active >= settings.TRAINING_MAX_RUNS:
                raise TrainingBusy(BUSY_MESSAGE)
            mine = TrainingRun.objects.filter(owner=user, status__in=ACTIVE_STATUSES).count()
            if (refusal := budgets.concurrency_refusal(user, mine)) is not None:
                raise TrainingBusy(refusal)  # the plan allows fewer trainings at a time
            run = create(limit, message="Starting …")
            _launch(run, _job(run, dataset.info, dataset.path, limit))
    run.budget_warning = warning  # type: ignore[attr-defined]
    return run


def _job(
    run: TrainingRun, info: DatasetInfo, path: Path | None, limit: budgets.RunLimit
) -> dict[str, Any]:
    """job.json of ``run`` (see brickdnn.training.state)."""
    return {
        "name": run.name,
        "graph": run.graph,
        "dataset": {
            "kind": info.kind,
            "path": str(path) if path is not None else None,
            "info": info.to_dict(),
        },
        "config": run.config,
        "tuning": run.tuning,
        "data_root": str(builtin_data_dir()),
        # The runner stops by itself once it has used this much CPU time.
        "limits": limit.job(),
    }


def _run_dataset(run: TrainingRun) -> tuple[DatasetInfo, Path | None]:
    """The dataset of a stored run, as :func:`resolve_dataset` found it when the run was
    created. Raises :class:`TrainingInputError` if it is gone."""
    ref = run.dataset if isinstance(run.dataset, dict) else {}
    kind = ref.get("kind")
    try:
        if kind == "random":
            return random_info(), None
        if kind == "builtin":
            return builtin_info(ref.get("id")), None
    except DatasetError as exc:
        raise TrainingInputError(str(exc)) from None
    dataset = run.upload if kind == "upload" else None
    path = dataset_path(dataset) if dataset is not None else None
    if dataset is None or path is None or not path.exists():
        raise TrainingInputError(
            "The dataset of this training was deleted before the training started."
        )
    return DatasetInfo.from_dict(dataset.info), path


def start_claimed(run: TrainingRun, limit: budgets.RunLimit) -> None:
    """Start the training process of a run that this worker took from the queue
    (training_queue.claim), with the CPU time it may use now. A run that was interrupted
    before starts from the beginning: what its earlier attempt left behind is removed."""
    directory = run_dir(run.pk)
    if run.pid:
        _kill(run, wait=3.0)  # only a process that still is the runner of this run
    family = run_family(run)
    leftovers = [STATE_FILE, STOP_FILE, weights_file(family)]
    leftovers.append(preprocessing_path(weights_file(family)).name)
    for name in leftovers:
        with contextlib.suppress(OSError):
            (directory / name).unlink()
    shutil.rmtree(directory / PREVIEW_DIR, ignore_errors=True)
    try:
        info, path = _run_dataset(run)
    except TrainingInputError as exc:
        run.status = "failed"
        run.message = f"Not started: {exc}"
        run.finished_at = timezone.now()
        run.save(update_fields=["status", "message", "finished_at"])
        return
    if run.restarts:
        with contextlib.suppress(OSError), open(directory / LOG_FILE, "ab") as log:
            log.write(f"\n--- started again (restart {run.restarts}) ---\n".encode())
    _launch(run, _job(run, info, path, limit))


def _command(configured: Any, module: str) -> list[str]:
    command = configured
    if not command:
        command = [sys.executable, "-m", module]
    elif isinstance(command, str):
        command = shlex.split(command)
    return [str(part) for part in command]


def runner_command() -> list[str]:
    return _command(settings.TRAINING_RUNNER_COMMAND, "brickdnn.training.runner")


def predict_command() -> list[str]:
    return _command(settings.TRAINING_PREDICT_COMMAND, "brickdnn.training.predict")


def _launch(run: TrainingRun, job: dict[str, Any]) -> None:
    """Write job.json and start the training process."""
    directory = run_dir(run.pk)
    command = [*runner_command(), str(directory)]
    options: dict[str, Any] = {}
    if os.name == "posix":
        # Own process group: lets us kill the runner with all its children, and Ctrl+C
        # in a terminal running the development server does not hit the training.
        options["start_new_session"] = True
    else:  # pragma: no cover - Windows
        options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    try:
        write_job(directory, job)
        builtin_data_dir().mkdir(parents=True, exist_ok=True)
        with open(directory / LOG_FILE, "ab") as log:
            process = subprocess.Popen(
                command,
                cwd=directory,
                env=_subprocess_env(),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                **options,
            )
    except OSError:
        logger.exception("Could not start the training process %s", command)
        run.status = "failed"
        run.message = "The training process could not be started. See the server log for details."
        run.finished_at = timezone.now()
        run.save(update_fields=["status", "message", "finished_at"])
        return
    _PROCESSES[str(run.pk)] = process
    threading.Thread(target=process.wait, name=f"brickdnn-reaper-{run.pk}", daemon=True).start()
    run.pid = process.pid
    run.started_at = timezone.now()
    run.save(update_fields=["pid", "started_at"])


# --------------------------------------------------------------------------- #
# Watching, stopping and killing runs
# --------------------------------------------------------------------------- #


SYNC_FIELDS = [
    "status",
    "message",
    "epochs_done",
    "metrics",
    "best",
    "trials_done",
    "best_trial",
    "has_weights",
    "total_params",
    "finished_at",
    "stop_reason",
    "size_bytes",
]
STOP_REASONS = ("user", "budget", "run_limit")


def _timestamp(value: Any) -> datetime | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return datetime.fromtimestamp(value, tz=dt_timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    return None


def _apply_state(run: TrainingRun, state: dict[str, Any]) -> None:
    """Copy the interesting parts of state.json into ``run`` (not saved)."""
    if state.get("status") in STATUSES:
        run.status = state["status"]
    if isinstance(state.get("message"), str):
        run.message = state["message"][:2000]
    history = state.get("history")
    if isinstance(history, list) and history:
        last = history[-1] if isinstance(history[-1], dict) else {}
        epoch = last.get("epoch")
        run.epochs_done = epoch if isinstance(epoch, int) and epoch >= 0 else len(history)
        # Once the weights are chosen: the metrics of the kept epoch (also what is tested).
        kept = state.get("weights_epoch")
        epochs = [e for e in history if isinstance(e, dict)]
        entry = next((e for e in epochs if kept is not None and e.get("epoch") == kept), last)
        test = state.get("test") if isinstance(state.get("test"), dict) else {}
        run.metrics = {
            **{key: entry.get(key) for key in METRIC_KEYS},
            **{key: test.get(name) for key, name in TEST_KEYS.items()},
        }
        if state.get("task") == "segmentation":
            run.metrics.update({key: entry.get(key) for key in SEGMENTATION_METRIC_KEYS})
            run.metrics.update(
                {key: test.get(name) for key, name in SEGMENTATION_TEST_KEYS.items()}
            )
    elif isinstance(history, list):
        run.epochs_done = 0  # a new trial of a search has started
    if isinstance(state.get("best"), dict) or state.get("best") is None:
        run.best = state.get("best")
    trials = state.get("trials")
    if isinstance(trials, list):
        trials = [t for t in trials if isinstance(t, dict)]
        run.trials_done = sum(1 for t in trials if t.get("status") in ("completed", "failed"))
        number = state.get("best_trial")
        best = next((t for t in trials if number is not None and t.get("trial") == number), None)
        run.best_trial = {key: best.get(key) for key in TRIAL_KEYS} if best else None
        if best and "val_iou" in best:  # segmentation
            run.best_trial["val_iou"] = best.get("val_iou")
    run.has_weights = bool(state.get("weights")) and _weights_file(run).is_file()
    total = state.get("total_params")
    if isinstance(total, int) and not isinstance(total, bool):
        run.total_params = total
    finished = _timestamp(state.get("finished_at"))
    if finished is not None:
        run.finished_at = finished
    if state.get("stop_reason") in STOP_REASONS:
        run.stop_reason = state["stop_reason"]


def _finite(value: Any) -> Any:
    """Replace NaN and infinity (not valid JSON) by None, recursively."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite(item) for item in value]
    return value


def _read_state(directory: Path) -> dict[str, Any] | None:
    state = read_state(directory)
    return _finite(state) if state is not None else None


def _log_tail(run: TrainingRun) -> str | None:
    """The last line the training process printed (often the error)."""
    try:
        with open(run_dir(run.pk) / LOG_FILE, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 4096))
            text = f.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    lines = [line.strip() for line in text.replace("\r", "\n").splitlines() if line.strip()]
    return lines[-1][:300] if lines else None


def _finish(run: TrainingRun, status: str, message: str, with_log: bool = False) -> None:
    if with_log and (tail := _log_tail(run)):
        message = f"{message} Last output: {tail}"
    run.status = status
    run.message = message
    if run.finished_at is None:
        run.finished_at = timezone.now()
    run.has_weights = run.has_weights and _weights_file(run).is_file()


def _stopped(run: TrainingRun) -> None:
    """End a run the server stopped (its process is gone or was killed after a stop
    request), saying why if it was a budget."""
    request = stop_request(run_dir(run.pk)) or {}
    why = request.get("message") if request.get("reason") in ("budget", "run_limit") else ""
    _finish(run, "stopped", f"{why} {STOPPED_MESSAGE}" if why else STOPPED_MESSAGE)
    if request.get("reason") in STOP_REASONS:
        run.stop_reason = request["reason"]
    elif not run.stop_reason:
        run.stop_reason = "user"


def _count_usage(run: TrainingRun, state: dict[str, Any] | None) -> float:
    """Count the CPU time ``run`` used since the last look into the ledger (see budgets.py);
    measure its files once it has ended. Returns the CPU seconds added."""
    final = run.status in TERMINAL_STATUSES
    measured = budgets.measured_cpu(run, state)
    added = budgets.record_run(run, measured, final=final) if measured is not None else 0.0
    if final and run.size_bytes is None:
        run.size_bytes = budgets.folder_size(run_dir(run.pk))
    return added


def _enforce_budgets(run: TrainingRun) -> None:
    """Stop the active ``run`` (like a stop request, keeping its weights) if it used the
    most one run may use or a budget it counts against is used up."""
    if run.status not in ACTIVE_STATUSES or run.stop_requested_at is not None:
        return
    verdict = budgets.over_budget(run)
    if verdict is None:
        return
    reason, message = verdict
    directory = run_dir(run.pk)
    if directory.is_dir():
        request_stop(directory, reason, message)
    now = timezone.now()
    TrainingRun.objects.filter(pk=run.pk, stop_requested_at__isnull=True).update(
        stop_requested_at=now, stop_reason=reason
    )
    run.stop_requested_at = now
    run.stop_reason = reason
    logger.info("Stopping training run %s: %s", run.pk, reason)


@contextlib.contextmanager
def supervising(worker_id: int):
    """While it lasts, this thread is the training worker ``worker_id`` (TrainingWorker.pk):
    it looks after the training processes of that worker's runs (see :func:`sync_run`)."""
    previous = getattr(_SUPERVISOR, "worker", None)
    _SUPERVISOR.worker = worker_id
    try:
        yield
    finally:
        _SUPERVISOR.worker = previous


def _supervises(run: TrainingRun) -> bool:
    """Whether this process looks after the training process of ``run``: it may judge if
    the process is alive, and kill it. Without a training worker every server process does
    (the web server started the process); with one, only the worker that took the run."""
    if not queue.enabled():
        return True
    worker = getattr(_SUPERVISOR, "worker", None)
    return worker is not None and run.worker_id == worker


def _mirror_stop(run: TrainingRun, directory: Path) -> None:
    """Pass a stop request that is only in the database on to the runner (the ``STOP``
    file): the web server could not write it, e.g. because the run was being started."""
    if run.stop_requested_at is None or not directory.is_dir() or stop_requested(directory):
        return
    if run.stop_reason in ("budget", "run_limit"):
        verdict = budgets.over_budget(run)
        request_stop(directory, run.stop_reason, verdict[1] if verdict else "")
    else:
        request_stop(directory)


def _is_own_child(run: TrainingRun) -> bool:
    return str(run.pk) in _PROCESSES


def _process_alive(run: TrainingRun) -> bool:
    process = _PROCESSES.get(str(run.pk))
    if process is not None:
        return process.poll() is None
    if not run.pid:
        # Created, but the process is being launched right now (by another request).
        return run.started_at is None and timezone.now() - run.created_at < timedelta(
            seconds=START_TIMEOUT_SECONDS
        )
    # Started by another server process (or before a restart). Zombies cannot occur
    # here: the process that started the runner reaps it, or init does.
    try:
        os.kill(run.pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _kill(run: TrainingRun, wait: float = 0.0) -> None:
    """Kill the training process (and its children), optionally waiting until it is gone."""
    process = _PROCESSES.get(str(run.pk))
    if process is not None and process.poll() is not None:
        return  # already exited and reaped: its pid may belong to another process now
    pid = process.pid if process is not None else run.pid
    if not pid:
        return
    if process is None and not _is_runner(pid, run):
        return  # gone, or the pid now belongs to an unrelated process
    try:
        if os.name == "posix":
            # The runner leads its own process group: this also kills its children.
            os.killpg(pid, signal.SIGKILL)
        elif process is not None:  # pragma: no cover - Windows
            process.kill()
        else:  # pragma: no cover - Windows
            os.kill(pid, signal.SIGTERM)
    except OSError:
        pass
    deadline = time.monotonic() + wait
    while wait and time.monotonic() < deadline and _process_alive(run):
        time.sleep(0.05)


def _command_line(pid: int) -> str | None:
    proc = Path(f"/proc/{pid}/cmdline")
    try:
        if proc.exists():  # Linux (also in Docker, where `ps` is not installed)
            return proc.read_bytes().replace(b"\0", b" ").decode(errors="replace")
        result = subprocess.run(
            ["ps", "-o", "command=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=5,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def _is_runner(pid: int, run: TrainingRun) -> bool:
    """True if ``pid`` still is the training process of ``run`` (pids get reused)."""
    return _is_runner_of(pid, run.pk)


def _is_runner_of(pid: int, run_id: Any) -> bool:
    if os.name != "posix":  # pragma: no cover - Windows
        return True
    try:
        if os.getpgid(pid) != pid:  # the runner always leads its own process group
            return False
    except OSError:
        return False
    command = _command_line(pid)
    return command is not None and str(run_dir(run_id)) in command


def sync_run(run: TrainingRun) -> TrainingRun:
    """Update an active run from its state.json and detect dead or stuck processes.

    A run that waits in the queue has no process and nothing to read. With a training
    worker, the web server only reads what the runner reports (progress, usage) and asks it
    to stop when a budget is used up; whether the process is alive is for the worker that
    started it to judge. The web server instead checks that this worker is: the runs of a
    dead worker go back into the queue (training_queue.recover)."""
    if run.status not in ACTIVE_STATUSES:
        return run
    if run.status == QUEUED:
        if not queue.enabled():
            # The queue was switched off while the run waited: nobody would start it.
            queue.fail_waiting(run, queue.MODE_OFF_MESSAGE)
        return run
    supervised = _supervises(run)
    if not supervised and not queue.is_alive(run.worker):
        queue.recover()
        with contextlib.suppress(TrainingRun.DoesNotExist):
            run.refresh_from_db()
        return run
    directory = run_dir(run.pk)
    state = read_run_state(run)
    if state is not None:
        _apply_state(run, state)

    if run.status in ACTIVE_STATUSES and supervised:
        _mirror_stop(run, directory)
        now = timezone.now()
        started = run.started_at or run.created_at
        if not _process_alive(run):
            # Read again: the runner may have written its final state just before exiting.
            state = read_run_state(run)
            if state is not None:
                _apply_state(run, state)
            if run.status in ACTIVE_STATUSES:
                if run.stop_requested_at is not None:
                    _stopped(run)
                else:
                    _finish(run, "failed", UNEXPECTED_MESSAGE, with_log=True)
        elif state is None and now - started > timedelta(seconds=START_TIMEOUT_SECONDS):
            if _is_own_child(run):
                _kill(run)
            _finish(run, "failed", NOT_STARTED_MESSAGE, with_log=True)
        elif state is not None and is_stale(state) and _stale_confirmed(run):
            if _is_own_child(run):
                _kill(run)
            _finish(run, "failed", UNEXPECTED_MESSAGE, with_log=True)
        elif run.stop_requested_at is not None and now - run.stop_requested_at > timedelta(
            seconds=STOP_TIMEOUT_SECONDS
        ):
            # The process is alive and sends heartbeats, so the pid is really the runner.
            _kill(run, wait=2.0)
            _stopped(run)

    added = _count_usage(run, state)
    # Never overwrite a final status that another request has already stored, nor a run
    # that went back into the queue or to another worker meanwhile.
    updated = _still_training(run).update(**{field: getattr(run, field) for field in SYNC_FIELDS})
    if not updated:
        with contextlib.suppress(TrainingRun.DoesNotExist):
            run.refresh_from_db()
    elif added > 0:
        _enforce_budgets(run)
    if run.status in TERMINAL_STATUSES:
        if supervised:
            _PROCESSES.pop(str(run.pk), None)
        if updated and run.worker_id is not None:
            queue.close_attempt(run.pk, run.status)
    return run


def _still_training(run: TrainingRun):
    """The row of ``run`` as long as it still trains where ``run`` was loaded from."""
    runs = TrainingRun.objects.filter(pk=run.pk, status__in=RUNNING_STATUSES)
    if run.worker_id is None:
        return runs.filter(worker__isnull=True)
    return runs.filter(worker_id=run.worker_id)


def _stale_confirmed(run: TrainingRun) -> bool:
    """True once a stale heartbeat has been seen for STALE_GRACE_SECONDS.

    The first sighting is remembered in the (shared) cache. After a laptop wakes up from
    sleep every heartbeat looks stale until the runner writes its next one, about a second
    later, so a single sighting is not enough to give up on a run.
    """
    key = f"brickdnn:training:stale:{run.pk}"
    first = cache.get(key)
    now = time.time()
    if not isinstance(first, (int, float)) or now - first > 10 * STALE_GRACE_SECONDS:
        cache.set(key, now, 10 * STALE_GRACE_SECONDS)
        return False
    return now - first >= STALE_GRACE_SECONDS


def sync_runs(runs) -> None:
    for run in runs:
        sync_run(run)


def stop_run(run: TrainingRun) -> TrainingRun:
    """Ask the runner to stop after the current batch (it saves the weights). A run that
    still waits in the queue is cancelled: it ends as stopped without having used compute.

    The request is a ``STOP`` file in the run directory and a time in the database. The
    runner follows the file; whoever looks after its process (the worker, or the web server
    without one) kills it if it does not, and passes a request it only finds in the database
    on to the runner."""
    sync_run(run)
    if run.status == QUEUED and not queue.cancel(run):
        with contextlib.suppress(TrainingRun.DoesNotExist):
            run.refresh_from_db()  # a worker took it at that moment: stop the training
    if run.status in RUNNING_STATUSES:
        directory = run_dir(run.pk)
        if directory.is_dir() and run.stop_requested_at is None:
            request_stop(directory)
        if run.stop_requested_at is None:
            run.stop_requested_at = timezone.now()
            run.stop_reason = "user"
            run.save(update_fields=["stop_requested_at", "stop_reason"])
    return run


def terminate_run(run: TrainingRun) -> None:
    """Kill the process of an active run (used before deleting it). With a training worker
    the web server cannot: the worker kills the process as soon as it sees that the run is
    gone (training_worker.py)."""
    sync_run(run)
    if run.status in RUNNING_STATUSES and _supervises(run):
        # Still active after the sync: the process is alive and its pid is trustworthy.
        _kill(run, wait=3.0)


def delete_run(run: TrainingRun) -> None:
    """Delete a run with its files, killing its process first if it is still active. The
    caller checks that the user may delete it (``shared_data.can_delete_run``)."""
    terminate_run(run)
    run.delete()  # the run directory is removed by a post_delete signal


def read_run_state(run: TrainingRun) -> dict[str, Any] | None:
    """What the training process of ``run`` last reported (state.json; NaN/infinity become
    None), or None if it has not reported yet. The one place that reads it, so that another
    store than the shared run directory can take over later."""
    return _read_state(run_dir(run.pk))


def apply_state(run: TrainingRun, state: dict[str, Any]) -> None:
    """Copy what the runner reported into ``run`` (not saved)."""
    _apply_state(run, state)


def mark_finished(run: TrainingRun, status: str, message: str) -> None:
    """End ``run`` (not saved): the server, not the runner, decided how it ends."""
    _finish(run, status, message)


def mark_stopped(run: TrainingRun) -> None:
    """End ``run`` as stopped after a stop request (not saved)."""
    _stopped(run)


def process_of(run_id: Any) -> subprocess.Popen | None:
    """The training process this server process started for the run, if any."""
    return _PROCESSES.get(str(run_id))


def is_runner(pid: int, run: TrainingRun) -> bool:
    """True if ``pid`` is the training process of ``run`` on this machine."""
    return _is_runner(pid, run)


def kill_process(run_id: Any, wait: float = 0.0, pid: int | None = None) -> None:
    """Kill the training process this server process started for the run, with its
    children (also when the run itself is gone from the database). ``pid``: the process of
    a run that another process started, killed only if it still is that run's runner."""
    process = _PROCESSES.pop(str(run_id), None)
    if process is not None:
        if process.poll() is not None:
            return
        pid = process.pid
    elif not pid or not _is_runner_of(pid, run_id):
        return
    try:
        if os.name == "posix":
            os.killpg(pid, signal.SIGKILL)
        elif process is not None:  # pragma: no cover - Windows
            process.kill()
    except OSError:
        pass
    deadline = time.monotonic() + wait
    while wait and time.monotonic() < deadline:
        if process is not None:
            if process.poll() is not None:
                break
        elif not _is_runner_of(pid, run_id):
            break
        time.sleep(0.05)


def state_of(run: TrainingRun) -> dict[str, Any] | None:
    """state.json for the API. NaN/infinity become None.

    If the API itself ended the run (crash, lost heartbeat, kill after a stop request),
    state.json still shows the last progress; the outcome from the database wins then.
    """
    if run.status == QUEUED:
        return None  # nothing was trained yet (after a restart: not in this attempt)
    state = read_run_state(run)
    if (
        state is not None
        and run.status in TERMINAL_STATUSES
        and state.get("status") not in TERMINAL_STATUSES
    ):
        finished = run.finished_at.timestamp() if run.finished_at else state.get("finished_at")
        state = {
            **state,
            "status": run.status,
            "message": run.message,
            "progress": None,
            "finished_at": finished,
        }
    return state


def _weights_file(run: TrainingRun) -> Path:
    """``weights.pt``, or ``model.joblib`` for classical models (see state.py)."""
    return run_dir(run.pk) / weights_file(run_family(run))


def weights_path(run: TrainingRun) -> Path | None:
    path = _weights_file(run)
    return path if run.has_weights and path.is_file() else None


def weights_filename(run: TrainingRun) -> str:
    """Download name: ``<run>.pt``, or ``<run>.joblib`` for a fitted classical model."""
    suffix = ".joblib" if run_family(run) == "classical" else ".pt"
    return (slugify(run.name) or ("model" if suffix == ".joblib" else "weights")) + suffix


def previews_of(run: TrainingRun, embed: bool = True) -> dict[str, Any] | None:
    """The preview images of a segmentation run (what the kept weights predict for a few
    validation images, see brickdnn.training.previews): its index, with every image as a
    data URL (``embed``) next to its file name. None if the run has none."""
    directory = run_dir(run.pk) / PREVIEW_DIR
    index = read_json(directory / PREVIEW_INDEX_FILE)
    if not isinstance(index, dict) or not isinstance(index.get("samples"), list):
        return None
    samples = []
    for sample in index["samples"]:
        if not isinstance(sample, dict):
            continue
        entry = _finite(dict(sample))
        for part in PREVIEW_PARTS:
            path = _preview_path(directory, sample.get(part))
            if path is None:
                break
            entry[part] = {"file": path.name}
            if embed:
                encoded = base64.b64encode(path.read_bytes()).decode("ascii")
                entry[part]["url"] = f"data:image/png;base64,{encoded}"
        else:
            samples.append(entry)
    return {**index, "samples": samples}


def _preview_path(directory: Path, name: Any) -> Path | None:
    """The file of a preview image, if ``name`` is one (``3_overlay.png``) and it exists."""
    if not isinstance(name, str) or not PREVIEW_NAME.fullmatch(name):
        return None
    path = directory / name
    try:
        ok = path.is_file() and not path.is_symlink() and path.stat().st_size <= MAX_PREVIEW_BYTES
    except OSError:
        return None
    return path if ok else None


def preview_file(run: TrainingRun, name: Any) -> Path | None:
    """One preview image of ``run`` by its file name (None if there is none)."""
    return _preview_path(run_dir(run.pk) / PREVIEW_DIR, name)


def no_weights_reason(run: TrainingRun) -> str:
    """Why a run has no weights to download or predict with (for the user)."""
    if run.status == QUEUED:
        return "This run still waits in the queue: it has not trained anything yet."
    if run.status in ACTIVE_STATUSES:
        return (
            f"This run is still {run.status}: its weights are saved when it finishes or is stopped."
        )
    if run.status == "failed":
        return f"This run failed before it saved weights: {run.message}"
    return "This run has no trained weights (it was stopped before the training started)."


# --------------------------------------------------------------------------- #
# The decision threshold of a binary classification
# --------------------------------------------------------------------------- #


def decision_threshold(run: TrainingRun) -> float | None:
    """The threshold the predictions of a binary classification use (the second class when
    its probability is at least this); None for every other run."""
    return float(run.decision_threshold) if inference.is_binary(run.plan) else None


def set_decision_threshold(run: TrainingRun, value: Any) -> TrainingRun:
    """Save the decision threshold of a binary classification run: a number from 0 to 1.
    Raises :class:`TrainingInputError` for other runs and other values."""
    if not inference.is_binary(run.plan):
        raise TrainingInputError(
            "Only binary classifications (two classes) have a decision threshold."
        )
    if isinstance(value, str):
        with contextlib.suppress(ValueError):
            value = float(value)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise TrainingInputError("The decision threshold must be a number from 0 to 1.")
    run.decision_threshold = float(value)
    # Only this column: a training that reports at the same time keeps its other fields.
    TrainingRun.objects.filter(pk=run.pk).update(decision_threshold=run.decision_threshold)
    return run


# --------------------------------------------------------------------------- #
# Predictions with the trained weights
# --------------------------------------------------------------------------- #


@dataclass
class Prediction:
    """Checked and prepared inputs, ready for :func:`run_prediction`."""

    run: TrainingRun
    preprocessing: dict[str, Any]
    inputs: inference.PreparedInputs
    # Raw rows (input_format "raw"): what to know about them, and the CPU time their
    # preparation took in this process.
    warnings: list[str] = dataclass_field(default_factory=list)
    prepare_seconds: float = 0.0
    mask_format: str = "png"  # segmentation: how the class maps are returned


def prepare_prediction(
    run: TrainingRun,
    inputs: Any,
    input_scale: Any = None,
    top_k: Any = None,
    image: Any = None,
    mask: Any = None,
    input_format: Any = None,
) -> Prediction:
    """Check that ``run`` can predict and prepare ``inputs`` like its training data (cheap:
    no process is started). ``input_format`` "raw": rows of the source table of a data
    preparation, prepared first (see raw_predictions.py). Image networks also take ``image``
    instead: image files as base64 text, which the prediction process decodes. ``mask``: how
    a segmentation returns its class maps (inference.MASK_FORMATS). Raises
    :class:`TrainingUnavailable`,
    :class:`PredictionUnavailable` or :class:`TrainingInputError`."""
    from . import raw_predictions

    reason = unavailable_reason(run_family(run))
    if reason:
        raise TrainingUnavailable(reason)
    if weights_path(run) is None:
        raise PredictionUnavailable(no_weights_reason(run))
    preprocessing = (state_of(run) or {}).get("preprocessing")
    warnings: list[str] = []
    started = time.thread_time()
    if raw_predictions.check_format(input_format) == "raw":
        inputs, warnings = raw_predictions.prepare_raw(run, inputs)
        input_scale = None  # the prepared rows are in the units of the training CSV file
    try:
        mask_format = inference.mask_format(mask)
        if image is not None and inputs is not None:
            raise inference.InvalidInputs("Give either 'inputs' or 'image', not both.")
        if image is not None:
            prepared = inference.prepare_image_files(run.plan, preprocessing, image, top_k)
        else:
            prepared = inference.prepare_inputs(run.plan, preprocessing, inputs, input_scale, top_k)
    except inference.UnknownPreprocessing as exc:
        raise PredictionUnavailable(str(exc)) from None
    except inference.InvalidInputs as exc:
        message = str(exc)
        if "unknown column" in message and input_format != "raw" and raw_predictions.recipe_of(run):
            message += (
                " Rows in the columns of the source table (before the data preparation) "
                "need input_format 'raw'."
            )
        raise TrainingInputError(message) from None
    spent = time.thread_time() - started if input_format == "raw" else 0.0
    return Prediction(
        run,
        preprocessing,
        prepared,
        mask_format=mask_format,
        warnings=warnings,
        prepare_seconds=spent,
    )


def describe_inputs(run: TrainingRun) -> str | None:
    """What the network of ``run`` takes, for error messages (None if unknown)."""
    from . import raw_predictions

    preprocessing = (state_of(run) or {}).get("preprocessing")
    try:
        text = inference.describe_inputs(run.plan, preprocessing)
    except inference.UnknownPreprocessing:
        return None
    raw = raw_predictions.raw_hint(run)
    return f"{text} {raw}" if raw else text


def run_prediction(prediction: Prediction, actor: Actor) -> dict[str, Any]:
    """Run the network on the prepared inputs in a separate process. Its CPU time counts
    against the budgets of ``actor`` (who asks, not who trained the run).

    Raises :class:`budgets.BudgetExceeded` (no compute left), :class:`TrainingBusy` (too
    many predictions at once) or :class:`PredictionFailed`.
    """
    run, prepared = prediction.run, prediction.inputs
    budgets.check_compute(actor)
    if prediction.prepare_seconds > 0:  # preparing raw rows (in this process)
        budgets.record_prediction(actor, run, prediction.prepare_seconds)
    request: dict[str, Any] = {"inputs": prepared.tensors, "kind": prepared.kind}
    if prepared.files is not None:
        request = {"images": prepared.files, "image": prepared.image, "kind": prepared.kind}
    segmentation = inference.is_segmentation(run.plan)
    names = inference.class_names(run.plan, prediction.preprocessing)
    if segmentation:
        target = run.plan["targets"][0]
        request["segmentation"] = {
            "binary": bool(target.get("binary")),
            "num_classes": len(names or []),
            "mask": prediction.mask_format,
        }
    with _prediction_slot():
        result = _predict_process(run, request, actor)
    try:
        if segmentation:
            predictions = inference.read_masks(
                run.plan, prediction.preprocessing, result.get("masks"), prepared.count
            )
        else:
            predictions = inference.read_outputs(
                run.plan,
                prediction.preprocessing,
                result.get("outputs"),
                prepared.count,
                prepared.top_k,
                probabilities=bool(result.get("probabilities")),
                std=result.get("std"),
                threshold=decision_threshold(run),
            )
    except ValueError as exc:
        raise PredictionFailed(str(exc)) from None
    answer: dict[str, Any] = {
        "run": {"id": str(run.pk), "name": run.name},
        "task": run.plan.get("task"),
        "predictions": predictions,
    }
    threshold = decision_threshold(run)
    if threshold is not None and names:
        answer["decision_threshold"] = threshold
        answer["positive_class"] = names[inference.BINARY_POSITIVE]
        answer["threshold_note"] = (
            f"The prediction is '{names[inference.BINARY_POSITIVE]}' when its probability "
            f"(positive_probability) is at least the run's decision threshold {threshold:g}, "
            f"else '{names[1 - inference.BINARY_POSITIVE]}'."
        )
    if segmentation:
        answer["class_names"] = names
        answer["colors"] = class_colors(len(names or []))
        answer["mask_note"] = (
            "mask_png is the predicted class of every pixel as a PNG file: the pixel values "
            "are the class indices, shown in 'colors'. mask 'rle' returns run-lengths "
            "instead (class, pixels, class, pixels … row by row), 'none' only the shares."
        )
    if (
        prepared.kind == "tabular"
        and inference.class_names(run.plan, prediction.preprocessing) is None
    ):
        answer["target"] = prediction.preprocessing.get("target")
    if prepared.kind == "demo":
        answer["note"] = "Trained on random demo data: the predictions mean nothing."
    if prediction.warnings:
        answer["warnings"] = prediction.warnings
    if result.get("std") is not None:
        answer["std_note"] = (
            "std is the Gaussian process's predictive standard deviation, in the units of "
            "the target: how uncertain each prediction is."
        )
    return answer


@contextlib.contextmanager
def _prediction_slot():
    """At most TRAINING_MAX_PREDICTIONS predictions at a time, across server processes (an
    exclusive lock on one of as many lock files; POSIX only)."""
    if fcntl is None:  # pragma: no cover - Windows
        yield
        return
    root = training_dir()
    root.mkdir(parents=True, exist_ok=True)
    for slot in range(max(1, int(settings.TRAINING_MAX_PREDICTIONS))):
        handle = open(root / f".predict-{slot}.lock", "a")  # noqa: SIM115 (closed below)
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            continue
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()
        return
    raise TrainingBusy(PREDICT_BUSY_MESSAGE)


def _predict_process(run: TrainingRun, request: dict[str, Any], actor: Actor) -> dict[str, Any]:
    """Start ``python -m brickdnn.training.predict <run dir>``, count its CPU time and return
    its answer."""
    directory = run_dir(run.pk)
    command = [*predict_command(), str(directory)]
    started = time.monotonic()
    try:
        result = subprocess.run(
            command,
            input=json.dumps(request),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=PREDICT_TIMEOUT_SECONDS,
            cwd=directory,
            env=_subprocess_env(),
        )
    except subprocess.TimeoutExpired:
        # Killed before it could report: count its wall time on all its threads.
        threads = max(1, min(PREDICT_THREADS, os.cpu_count() or 1))
        budgets.record_prediction(actor, run, (time.monotonic() - started) * threads)
        raise PredictionFailed(
            f"The prediction took longer than {PREDICT_TIMEOUT_SECONDS} seconds and was stopped."
        ) from None
    except OSError:
        logger.exception("Could not start the prediction process %s", command)
        raise PredictionFailed(
            "The prediction process could not be started. See the server log for details."
        ) from None
    answer = None
    for line in reversed(result.stdout.strip().splitlines()):
        if line.strip().startswith("{"):
            with contextlib.suppress(ValueError):
                answer = json.loads(line)
            break
    spent = answer.get("cpu_seconds") if isinstance(answer, dict) else None
    if isinstance(spent, bool) or not isinstance(spent, (int, float)) or spent < 0:
        spent = time.monotonic() - started  # it did not report: at least its wall time
    budgets.record_prediction(actor, run, float(spent))
    if isinstance(answer, dict) and isinstance(answer.get("error"), str):
        raise PredictionFailed(answer["error"][:500])
    if (
        result.returncode == 0
        and isinstance(answer, dict)
        and ("outputs" in answer or "masks" in answer)
    ):
        return answer
    lines = [line.strip() for line in result.stderr.splitlines() if line.strip()]
    detail = lines[-1][:300] if lines else f"exit code {result.returncode}"
    logger.warning("Prediction for run %s failed: %s", run.pk, result.stderr[-2000:])
    raise PredictionFailed(f"The prediction failed: {detail}")


# --------------------------------------------------------------------------- #
# Clean-up of files (also when a user and with it all their objects are deleted)
# --------------------------------------------------------------------------- #


# Files are removed once the deletion is committed, so that a rolled back transaction
# does not lose them and the database is not locked while a process is being killed.


@receiver(post_delete, sender=Dataset, dispatch_uid="brickdnn_dataset_files")
def _delete_dataset_files(sender, instance: Dataset, **kwargs) -> None:
    directory = dataset_dir(instance.pk)
    transaction.on_commit(lambda: shutil.rmtree(directory, ignore_errors=True))


@receiver(post_delete, sender=TrainingRun, dispatch_uid="brickdnn_run_files")
def _delete_run_files(sender, instance: TrainingRun, **kwargs) -> None:
    run = copy.copy(instance)  # Django clears instance.pk after the deletion

    def remove() -> None:
        # With a training worker, its worker kills the process (and removes what the
        # process wrote meanwhile) when it sees that the run is gone.
        if _supervises(run):
            if run.status in RUNNING_STATUSES:
                _kill(run, wait=3.0)  # only kills a process that still is this runner
            _PROCESSES.pop(str(run.pk), None)
        shutil.rmtree(run_dir(run.pk), ignore_errors=True)

    transaction.on_commit(remove)


@receiver(pre_delete, sender=TrainingRun, dispatch_uid="brickdnn_run_attempt")
def _close_attempt_of_deleted_run(sender, instance: TrainingRun, **kwargs) -> None:
    if instance.worker_id is not None and instance.status in RUNNING_STATUSES:
        queue.close_attempt(instance.pk, "deleted")
