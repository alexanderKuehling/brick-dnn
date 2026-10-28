"""Files shared between a training process and whoever watches it (no PyTorch needed).

Every run lives in its own directory::

    job.json      what to train: graph, dataset, plan, settings   (written by the API / CLI)
    state.json    live progress, replaced atomically              (written by the runner)
    STOP          created to ask the runner to stop after the current batch
    weights.pt    model.state_dict() after training or stopping   (written by the runner)
    output.log    stdout/stderr of the runner process             (for debugging)

``state.json`` looks like this::

    {
      "status": "running",           # starting | preparing | running | completed | stopped | failed
      "message": "Epoch 3 of 10",    # current activity, or the result / error for the user
      "updated_at": 1730000000.0,    # unix time; refreshed at least every 5 s (heartbeat)
      "started_at": 1730000000.0,
      "finished_at": null,
      "device": "mps", "device_label": "Apple GPU (MPS)",
      "task": "classification",      # classification | regression | demo
      "num_classes": 10, "class_names": ["0", "1", ...],
      "train_size": 60000, "val_size": 10000,
      "test_size": null,             # null without a test set
      "epochs": 10,                  # planned number of epochs (per trial in a search)
      "steps_per_epoch": 938,
      "total_params": 101770,
      "history": [                   # one entry per finished epoch
        {"epoch": 1, "train_loss": 0.31, "val_loss": 0.15,
         "train_accuracy": 0.91, "val_accuracy": 0.95, "seconds": 12.3}
      ],                             # accuracies are null for regression; regression runs
                                     # add "val_rmse" (root mean squared error, original units)
      "progress": {"epoch": 2, "step": 120, "steps": 938, "loss": 0.21,
                   "samples_per_second": 5400.0},          # null outside of an epoch
      "best": {"epoch": 1, "val_loss": 0.15, "val_accuracy": 0.95},  # lowest val_loss
      "test": {"loss": 0.16, "accuracy": 0.95, "rmse": null},  # final model on the test
                                     # set; null without a test set or before the end
      "weights": false,              # true once weights.pt exists
      "preprocessing": {...},        # how inputs were prepared, needed to reuse the weights
      "log": ["Downloading MNIST …", "..."],               # last 200 lines

      # Hyperparameter search only (null / empty for normal runs):
      "tuning": {"method": "random", "trials": 10,        # see tuning.py
                 "space": {"learning_rate": {"min": 0.0001, "max": 0.01}}},
      "trials_total": 10,
      "trial": 2,                    # the trial that is running, null outside of trials
      "trials": [                    # one entry per started trial
        {"trial": 1, "params": {"learning_rate": 0.00312}, "status": "completed",
         "message": "", "epochs_done": 10, "train_loss": 0.2, "val_loss": 0.15,
         "val_accuracy": 0.95, "val_rmse": null, "seconds": 120.5}
      ],                             # status: running | completed | failed | stopped;
                                     # the metrics are those of the trial's last epoch
      "best_trial": 1                # lowest final val_loss so far
    }

In a search, ``history``, ``best`` and ``progress`` belong to the running trial. When the
search ends (or is stopped), they are replaced by those of the best trial, whose weights
are saved.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any

JOB_FILE = "job.json"
STATE_FILE = "state.json"
STOP_FILE = "STOP"
WEIGHTS_FILE = "weights.pt"
LOG_FILE = "output.log"

ACTIVE_STATUSES = ("starting", "preparing", "running")
TERMINAL_STATUSES = ("completed", "stopped", "failed")
STATUSES = ACTIVE_STATUSES + TERMINAL_STATUSES

HEARTBEAT_SECONDS = 5
# A run whose state has not been updated for this long is considered dead.
STALE_SECONDS = 60
MAX_LOG_LINES = 200


def write_json_atomic(path: str | Path, data: Any) -> None:
    """Write JSON so that readers never see a half-written file.

    NaN and infinity are rejected (``ValueError``): they are not valid JSON.
    """
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, allow_nan=False)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def read_json(path: str | Path) -> Any | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def initial_state(epochs: int = 0) -> dict[str, Any]:
    now = time.time()
    return {
        "status": "starting",
        "message": "Starting …",
        "updated_at": now,
        "started_at": now,
        "finished_at": None,
        "device": None,
        "device_label": None,
        "task": None,
        "num_classes": None,
        "class_names": None,
        "train_size": None,
        "val_size": None,
        "test_size": None,
        "epochs": epochs,
        "steps_per_epoch": None,
        "total_params": None,
        "history": [],
        "progress": None,
        "best": None,
        "test": None,
        "weights": False,
        "preprocessing": None,
        "log": [],
        "tuning": None,
        "trials_total": None,
        "trial": None,
        "trials": [],
        "best_trial": None,
    }


def read_state(run_dir: str | Path) -> dict[str, Any] | None:
    state = read_json(Path(run_dir) / STATE_FILE)
    return state if isinstance(state, dict) else None


def finite(data: Any) -> Any:
    """A copy of ``data`` with NaN and infinite floats replaced by ``None``."""
    if isinstance(data, float):
        return data if math.isfinite(data) else None
    if isinstance(data, dict):
        return {key: finite(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [finite(value) for value in data]
    return data


def write_state(run_dir: str | Path, state: dict[str, Any]) -> None:
    """Write state.json; a non-finite number (e.g. a NaN metric) becomes ``null``."""
    write_json_atomic(Path(run_dir) / STATE_FILE, finite(state))


def read_job(run_dir: str | Path) -> dict[str, Any] | None:
    job = read_json(Path(run_dir) / JOB_FILE)
    return job if isinstance(job, dict) else None


def write_job(run_dir: str | Path, job: dict[str, Any]) -> None:
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    write_json_atomic(Path(run_dir) / JOB_FILE, job)


def request_stop(run_dir: str | Path) -> None:
    (Path(run_dir) / STOP_FILE).touch()


def stop_requested(run_dir: str | Path) -> bool:
    return (Path(run_dir) / STOP_FILE).exists()


def is_stale(state: dict[str, Any] | None, now: float | None = None) -> bool:
    """True if an active run has stopped sending heartbeats."""
    if state is None or state.get("status") in TERMINAL_STATUSES:
        return False
    updated = state.get("updated_at")
    if not isinstance(updated, (int, float)):
        return True
    return (now if now is not None else time.time()) - updated > STALE_SECONDS
