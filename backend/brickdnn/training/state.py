"""Files shared between a training process and whoever watches it (no PyTorch needed).

Every run lives in its own directory::

    job.json      what to train: graph, dataset, plan, settings   (written by the API / CLI)
                  and optionally ``limits``: {"cpu_seconds": 1800, "reason": "run_limit",
                  "message": "..."}, the CPU time after which the runner stops by itself
                  (reason "budget" or "run_limit"; the message starts its final message)
    state.json    live progress, replaced atomically              (written by the runner)
    STOP          created to ask the runner to stop after the current batch; it holds
                  {"reason": "budget", "message": "..."} when the server stops the run for
                  another reason than the user's request
    weights.pt    model.state_dict() after training or stopping   (written by the runner)
    model.joblib  instead of weights.pt for classical models (scikit-learn): the fitted
                  pipeline saved with joblib, see brickdnn.classical for its format
    weights.preprocessing.json   ``preprocessing`` of the state, saved with the weights
                  (not for demo data): what new inputs need to be prepared the same way
                  (classical models: model.preprocessing.json)
    output.log    stdout/stderr of the runner process             (for debugging)

``state.json`` looks like this::

    {
      "status": "running",           # starting | preparing | running | completed | stopped | failed
      "message": "Epoch 3 of 10",    # current activity, or the result / error for the user
      "updated_at": 1730000000.0,    # unix time; refreshed at least every 5 s (heartbeat)
      "started_at": 1730000000.0,
      "finished_at": null,
      "device": "mps", "device_label": "Apple GPU (MPS)",
      "family": "neural",            # neural (PyTorch) | classical (scikit-learn)
      "task": "classification",      # classification | regression | demo
      "num_classes": 10, "class_names": ["0", "1", ...],
      "train_size": 48000, "val_size": 12000,
      "test_size": 10000,            # null without a test set
      "epochs": 10,                  # planned number of epochs (per trial in a search)
      "steps_per_epoch": 750,
      "total_params": 101770,
      "history": [                   # one entry per finished epoch
        {"epoch": 1, "train_loss": 0.31, "val_loss": 0.15,
         "train_accuracy": 0.91, "val_accuracy": 0.95, "seconds": 12.3}
      ],                             # accuracies are null for regression; regression runs
                                     # add "val_rmse" (root mean squared error, original units)
                                     # and "val_r2" (coefficient of determination; null if
                                     # the validation targets are all equal)
      "progress": {"epoch": 2, "step": 120, "steps": 938, "loss": 0.21,
                   "samples_per_second": 5400.0},          # null outside of an epoch
      "best": {"epoch": 1, "val_loss": 0.15, "val_accuracy": 0.95},  # lowest val_loss
      "weights_epoch": 1,            # the epoch whose weights are tested and saved: the
                                     # best one (restore_best) or the last one; null until
                                     # the epochs are over, and if the saved weights are not
                                     # those of a finished epoch (stopped without restore_best)
      "stopped_early": false,        # true if early stopping (patience) ended the training
      "test": {"loss": 0.16, "accuracy": 0.95, "rmse": null, "r2": null},  # kept weights
                                     # on the test set (rmse and r2: regression only); null
                                     # without a test set or before the end
      "weights": false,              # true once weights.pt exists
      "preprocessing": {...},        # how inputs were prepared, needed to reuse the weights:
                                     # {"kind": "image", channels, height, width, layout,
                                     #  mean, std, class_names}, {"kind": "tabular", columns,
                                     #  target, mean, std, target_mean, target_std,
                                     #  class_names} (statistics of the training part only)
                                     # or {"kind": "demo"}
      "log": ["Downloading MNIST …", "..."],               # last 200 lines
      "usage": {"cpu_seconds": 12.5,   # CPU time of the runner (all its threads) and of
                "wall_seconds": 8.1,   # its finished child processes, time since the start;
                "limit_cpu_seconds": 1800},  # refreshed with every write (null: no limit)
      "stop_reason": null,           # stopped runs: "user", "budget" (a monthly budget is
                                     # used up) or "run_limit" (the most one run may use)

      # Hyperparameter search only (null / empty for normal runs):
      # Classical models only (null for neural networks), see below
      "classical": {...},

      "tuning": {"method": "random", "trials": 10,        # see tuning.py
                 "space": {"learning_rate": {"min": 0.0001, "max": 0.01}}},
      "trials_total": 10,
      "trial": 2,                    # the trial that is running, null outside of trials
      "trials": [                    # one entry per started trial
        {"trial": 1, "params": {"learning_rate": 0.00312}, "status": "completed",
         "message": "", "epochs_done": 10, "train_loss": 0.2, "val_loss": 0.15,
         "val_accuracy": 0.95, "val_rmse": null, "val_r2": null, "weights_epoch": 8,
         "stopped_early": false, "seconds": 120.5}
      ],                             # status: running | completed | failed | stopped;
                                     # the metrics are those of the epoch whose weights the
                                     # trial keeps (weights_epoch); while it runs and for
                                     # failed or stopped trials, those of its last epoch
      "best_trial": 1                # lowest val_loss of the kept weights so far
    }

In a search, ``history``, ``best`` and ``progress`` belong to the running trial. When the
search ends (or is stopped), they are replaced by those of the best trial, whose weights
are saved; ``weights_epoch`` and ``stopped_early`` then describe the best trial.

Classical models (``family`` "classical") are fitted once: ``epochs`` and
``steps_per_epoch`` are 1, ``progress`` stays null, and when the fit is done ``history``
gets a single entry (epoch 1, the metrics of the fitted model), ``weights_epoch`` is 1
and ``device`` is "cpu".
``train_size`` counts the rows the model is fitted on (after the row limit). The losses
are the cross-entropy of the predicted class probabilities and, for regression, the mean
squared error of the standardised target (like those of a neural network), so runs of
both families can be compared. The details are in ``classical``::

      "classical": {
        "estimator": "GaussianProcessRegressor", "label": "Gaussian process regression",
        "max_rows": 5000,              # the row limit of this server
        "rows": 4000,                  # training rows the model is fitted on
        "subsampled_from": null,       # training rows before the row limit, if it applied
        "fit_seconds": 12.3,           # null until fitted
        "initial_kernel": "...",       # Gaussian processes: the kernel of the bricks
        "kernel": "...",               # ... and the fitted one (null until fitted)
        "log_marginal_likelihood": 12.5,     # Gaussian processes, null until fitted
        "support_vectors": null,       # support vector machines: their number once fitted
        "train": {"loss": 0.1, "accuracy": null, "rmse": 0.3, "r2": 0.97}  # training rows
      }
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
CLASSICAL_WEIGHTS_FILE = "model.joblib"
LOG_FILE = "output.log"

ACTIVE_STATUSES = ("starting", "preparing", "running")
TERMINAL_STATUSES = ("completed", "stopped", "failed")
STATUSES = ACTIVE_STATUSES + TERMINAL_STATUSES

HEARTBEAT_SECONDS = 5
# A run whose state has not been updated for this long is considered dead.
STALE_SECONDS = 60
MAX_LOG_LINES = 200


def weights_file(family: str | None) -> str:
    """Name of the weights file of a run: ``model.joblib`` for classical models (the fitted
    scikit-learn pipeline), ``weights.pt`` for neural networks."""
    return CLASSICAL_WEIGHTS_FILE if family == "classical" else WEIGHTS_FILE


def preprocessing_path(weights_path: str | Path) -> Path:
    """Where the preprocessing is saved next to the weights: ``x.pt`` ->
    ``x.preprocessing.json``."""
    path = Path(weights_path)
    return path.with_name(f"{path.stem}.preprocessing.json")


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


def initial_state(epochs: int = 0, family: str = "neural") -> dict[str, Any]:
    now = time.time()
    return {
        "status": "starting",
        "message": "Starting …",
        "updated_at": now,
        "started_at": now,
        "finished_at": None,
        "device": None,
        "device_label": None,
        "family": family,
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
        "weights_epoch": None,
        "stopped_early": False,
        "test": None,
        "weights": False,
        "preprocessing": None,
        "log": [],
        "tuning": None,
        "trials_total": None,
        "trial": None,
        "trials": [],
        "best_trial": None,
        "classical": None,
        "usage": {"cpu_seconds": 0.0, "wall_seconds": 0.0, "limit_cpu_seconds": None},
        "stop_reason": None,
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


def request_stop(
    run_dir: str | Path, reason: str | None = None, message: str | None = None
) -> None:
    """Ask the runner to stop. ``reason`` ("budget", "run_limit") and ``message`` (the start
    of its final message) say why when it is not the user's request."""
    path = Path(run_dir) / STOP_FILE
    if reason is None:
        path.touch()
    else:
        write_json_atomic(path, {"reason": reason, "message": message or ""})


def stop_requested(run_dir: str | Path) -> bool:
    return (Path(run_dir) / STOP_FILE).exists()


def stop_request(run_dir: str | Path) -> dict[str, str] | None:
    """Why the run was asked to stop: ``{"reason", "message"}`` (reason "user" for an empty
    STOP file), None if it was not."""
    path = Path(run_dir) / STOP_FILE
    if not path.exists():
        return None
    data = read_json(path)
    if isinstance(data, dict) and isinstance(data.get("reason"), str):
        return {"reason": data["reason"], "message": str(data.get("message") or "")}
    return {"reason": "user", "message": ""}


def cpu_seconds() -> float:
    """CPU time (user + system, all threads) of this process and its finished children."""
    times = os.times()
    return times.user + times.system + times.children_user + times.children_system


def is_stale(state: dict[str, Any] | None, now: float | None = None) -> bool:
    """True if an active run has stopped sending heartbeats."""
    if state is None or state.get("status") in TERMINAL_STATUSES:
        return False
    updated = state.get("updated_at")
    if not isinstance(updated, (int, float)):
        return True
    return (now if now is not None else time.time()) - updated > STALE_SECONDS
