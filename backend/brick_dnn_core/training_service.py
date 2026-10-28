"""Local training: dataset uploads, training processes and their status.

Every run is trained by its own process (``python -m brickdnn.training.runner <run_dir>``),
so the web server never imports PyTorch. The API writes ``job.json`` into the run
directory, starts the process and afterwards only looks at the files the process writes
(``state.json``, ``weights.pt``). See :mod:`brickdnn.training.state` for the file format.
"""

from __future__ import annotations

import contextlib
import copy
import json
import logging
import math
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from datetime import timezone as dt_timezone
from pathlib import Path
from typing import Any

from django.conf import settings
from django.core.cache import cache
from django.db import transaction
from django.db.models.signals import post_delete
from django.dispatch import receiver
from django.utils import timezone
from rest_framework import serializers

from brickdnn import analyze
from brickdnn.training import (
    ACTIVE_STATUSES,
    BUILTIN_DATASETS,
    STATUSES,
    TERMINAL_STATUSES,
    Compatibility,
    DatasetError,
    DatasetInfo,
    TrainingConfig,
    builtin_info,
    check_compatibility,
    extract_image_zip,
    inspect_csv,
    inspect_image_folder,
    random_info,
    torch_installed,
    torchvision_installed,
)
from brickdnn.training.state import (
    LOG_FILE,
    WEIGHTS_FILE,
    is_stale,
    read_state,
    request_stop,
    write_job,
)

from .models import Dataset, TrainingRun
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
BUSY_MESSAGE = "Another training is still running. Stop it or wait until it finishes."
UNEXPECTED_MESSAGE = "The training process stopped unexpectedly."
NOT_STARTED_MESSAGE = "The training process did not start."
STOPPED_MESSAGE = "Training was stopped."
WRONG_FILE_MESSAGE = "Upload a .csv file or a .zip file with one folder per class."

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
METRIC_KEYS = ("train_loss", "val_loss", "val_accuracy", "val_rmse")

# Training processes started by this server process, by run id. A daemon thread waits
# for each of them so that finished processes do not linger as zombies.
_PROCESSES: dict[str, subprocess.Popen] = {}
_START_LOCK = threading.Lock()


class TrainingInputError(ValueError):
    """Invalid request data. The message is shown to the user."""


class TrainingUnavailable(Exception):
    """Training cannot be used on this server (disabled or no PyTorch)."""


class TrainingBusy(Exception):
    """The maximum number of concurrent trainings is reached."""


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


def unavailable_reason() -> str | None:
    """Why runs cannot be started right now (cheap: does not run the probe)."""
    if not settings.TRAINING_ENABLED:
        return DISABLED_REASON
    if not torch_installed():
        return NO_TORCH_REASON
    return None


def capabilities() -> dict[str, Any]:
    """Everything the UI needs to know before offering training."""
    reason = unavailable_reason()
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
            {**dataset.to_dict(), "available": has_torchvision}
            for dataset in BUILTIN_DATASETS.values()
        ],
        "max_upload_mb": settings.TRAINING_MAX_UPLOAD_MB,
        "max_concurrent_runs": settings.TRAINING_MAX_RUNS,
        "login_required": True,
    }


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #


def save_upload(user, upload, name: str | None = None) -> Dataset:
    """Store and inspect an uploaded CSV or ZIP file. Raises :class:`DatasetError`."""
    filename = Path(upload.name or "").name
    suffix = Path(filename).suffix.lower()
    if suffix not in (".csv", ".zip"):
        raise DatasetError(WRONG_FILE_MESSAGE)
    name = (str(name or "").strip() or Path(filename).stem.strip() or "Dataset")[:120]
    kind = "csv" if suffix == ".csv" else "images"

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
            info = inspect_image_folder(work / "images", name=name)
            size = _folder_size(work / "images")
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
        dataset = Dataset.objects.filter(owner=user, pk=dataset_id).first()
        if dataset is None:
            raise TrainingInputError(
                "This dataset does not exist (any more). Upload it again or choose another one."
            )
        path = dataset_path(dataset)
        if not path.exists():
            raise TrainingInputError("The files of this dataset are missing. Upload it again.")
        info = DatasetInfo.from_dict(dataset.info)
        summary = {"kind": "upload", "id": dataset.pk, "name": dataset.name}
        return ResolvedDataset(info, path, dataset, summary)
    raise TrainingInputError("Unknown dataset type. Use 'random', 'builtin' or 'upload'.")


def check(graph: Any, dataset: ResolvedDataset) -> Compatibility:
    """check_compatibility() plus checks that depend on this server."""
    result = check_compatibility(graph, dataset.info)
    if dataset.info.kind == "builtin" and not torchvision_installed():
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


def active_run_count() -> int:
    """Number of active runs of all users (after updating their status)."""
    sync_runs(TrainingRun.objects.filter(status__in=ACTIVE_STATUSES))
    return TrainingRun.objects.filter(status__in=ACTIVE_STATUSES).count()


def start_run(user, name: Any, graph: Any, dataset_ref: Any, config: Any) -> TrainingRun:
    """Validate everything and start a training process.

    Raises :class:`TrainingUnavailable`, :class:`TrainingInputError` (also for
    ``ConfigError``), :class:`IncompatibleNetwork` or :class:`TrainingBusy`.
    """
    reason = unavailable_reason()
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
    try:
        training_config = TrainingConfig.from_dict(config)
    except ValueError as exc:  # ConfigError
        raise TrainingInputError(str(exc)) from None
    dataset = resolve_dataset(user, dataset_ref)
    analysis = analyze(graph)
    compatibility = check(analysis, dataset)
    if not compatibility.ok:
        raise IncompatibleNetwork(compatibility)
    assert compatibility.plan is not None

    with _start_lock():
        if active_run_count() >= settings.TRAINING_MAX_RUNS:
            raise TrainingBusy(BUSY_MESSAGE)
        run = TrainingRun.objects.create(
            owner=user,
            name=name,
            graph=graph,
            dataset=dataset.summary,
            upload=dataset.upload,
            config=training_config.to_dict(),
            plan=compatibility.plan.to_dict(),
            total_params=analysis.total_params,
            message="Starting …",
        )
        job = {
            "name": run.name,
            "graph": graph,
            "dataset": {
                "kind": dataset.info.kind,
                "path": str(dataset.path) if dataset.path is not None else None,
                "info": dataset.info.to_dict(),
            },
            "config": training_config.to_dict(),
            "data_root": str(builtin_data_dir()),
        }
        _launch(run, job)
    return run


def runner_command() -> list[str]:
    command = settings.TRAINING_RUNNER_COMMAND
    if not command:
        command = [sys.executable, "-m", "brickdnn.training.runner"]
    elif isinstance(command, str):
        command = shlex.split(command)
    return [str(part) for part in command]


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
    "has_weights",
    "total_params",
    "finished_at",
]


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
        run.metrics = {key: last.get(key) for key in METRIC_KEYS}
    if isinstance(state.get("best"), dict) or state.get("best") is None:
        run.best = state.get("best")
    run.has_weights = bool(state.get("weights")) and (run_dir(run.pk) / WEIGHTS_FILE).is_file()
    total = state.get("total_params")
    if isinstance(total, int) and not isinstance(total, bool):
        run.total_params = total
    finished = _timestamp(state.get("finished_at"))
    if finished is not None:
        run.finished_at = finished


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
    run.has_weights = run.has_weights and (run_dir(run.pk) / WEIGHTS_FILE).is_file()


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
    if os.name != "posix":  # pragma: no cover - Windows
        return True
    try:
        if os.getpgid(pid) != pid:  # the runner always leads its own process group
            return False
    except OSError:
        return False
    command = _command_line(pid)
    return command is not None and str(run_dir(run.pk)) in command


def sync_run(run: TrainingRun) -> TrainingRun:
    """Update an active run from its state.json and detect dead or stuck processes."""
    if run.status not in ACTIVE_STATUSES:
        return run
    directory = run_dir(run.pk)
    state = _read_state(directory)
    if state is not None:
        _apply_state(run, state)

    if run.status in ACTIVE_STATUSES:
        now = timezone.now()
        started = run.started_at or run.created_at
        if not _process_alive(run):
            # Read again: the runner may have written its final state just before exiting.
            state = _read_state(directory)
            if state is not None:
                _apply_state(run, state)
            if run.status in ACTIVE_STATUSES:
                if run.stop_requested_at is not None:
                    _finish(run, "stopped", STOPPED_MESSAGE)
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
            _finish(run, "stopped", STOPPED_MESSAGE)

    # Never overwrite a final status that another request has already stored.
    updated = TrainingRun.objects.filter(pk=run.pk, status__in=ACTIVE_STATUSES).update(
        **{field: getattr(run, field) for field in SYNC_FIELDS}
    )
    if not updated:
        with contextlib.suppress(TrainingRun.DoesNotExist):
            run.refresh_from_db()
    if run.status in TERMINAL_STATUSES:
        _PROCESSES.pop(str(run.pk), None)
    return run


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
    """Ask the runner to stop after the current batch (it saves the weights)."""
    sync_run(run)
    if run.status in ACTIVE_STATUSES:
        directory = run_dir(run.pk)
        if directory.is_dir():
            request_stop(directory)
        if run.stop_requested_at is None:
            run.stop_requested_at = timezone.now()
            run.save(update_fields=["stop_requested_at"])
    return run


def terminate_run(run: TrainingRun) -> None:
    """Kill the process of an active run (used before deleting it)."""
    sync_run(run)
    if run.status in ACTIVE_STATUSES:
        # Still active after the sync: the process is alive and its pid is trustworthy.
        _kill(run, wait=3.0)


def state_of(run: TrainingRun) -> dict[str, Any] | None:
    """state.json for the API. NaN/infinity become None.

    If the API itself ended the run (crash, lost heartbeat, kill after a stop request),
    state.json still shows the last progress; the outcome from the database wins then.
    """
    state = _read_state(run_dir(run.pk))
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


def weights_path(run: TrainingRun) -> Path | None:
    path = run_dir(run.pk) / WEIGHTS_FILE
    return path if run.has_weights and path.is_file() else None


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
        if run.status in ACTIVE_STATUSES:
            _kill(run, wait=3.0)  # only kills a process that still is this runner
        _PROCESSES.pop(str(run.pk), None)
        shutil.rmtree(run_dir(run.pk), ignore_errors=True)

    transaction.on_commit(remove)
