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
    previews/     segmentation only: index.json and small PNG files that show what the
                  kept weights predict for a few validation images (see previews.py)
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
      "task": "classification",      # classification | regression | segmentation | demo
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
                                     # the validation targets are all equal); segmentation
                                     # runs add "val_iou" and "val_dice" (means over the
                                     # classes that occur) and "val_pixel_accuracy";
                                     # without validation data (val_size 0) every val_*
                                     # value is null
      "progress": {"epoch": 2, "step": 120, "steps": 938, "loss": 0.21,
                   "samples_per_second": 5400.0},          # null outside of an epoch
      "best": {"epoch": 1, "val_loss": 0.15, "val_accuracy": 0.95},  # lowest val_loss
                                     # (segmentation: also "val_iou" of that epoch); null
                                     # without validation data
      "weights_epoch": 1,           # the epoch whose weights are tested and saved: the
                                     # best one (restore_best) or the last one (always
                                     # without validation data); null until
                                     # the epochs are over, and if the saved weights are not
                                     # those of a finished epoch (stopped without restore_best)
      "stopped_early": false,        # true if early stopping (patience) ended the training
      "test": {"loss": 0.16, "accuracy": 0.95, "rmse": null, "r2": null},  # kept weights
                                     # on the test set (rmse and r2: regression only); null
                                     # without a test set or before the end. Segmentation
                                     # adds "iou", "dice", "pixel_accuracy" and the lists
                                     # "class_iou" and "class_dice" (see "segmentation")
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

      # Classical models only (null for neural networks), see below
      "classical": {...},
      # Segmentation only (null otherwise), set once the weights are saved, see below
      "segmentation": {...},
      # Completed runs with a single class or value output (null otherwise and in runs of
      # older versions): the kept weights measured once more at the end, see below
      "evaluation": {...},

      # Hyperparameter search only (null / empty for normal runs):

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
                                     # segmentation adds "val_iou", "val_dice" and
                                     # "val_pixel_accuracy";
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

A segmentation (``task`` "segmentation") has a class for every pixel. Its accuracies are
null; IoU and Dice are computed from all pixels of a split together, per class, and averaged
over the classes that occur in its masks or predictions. Once the weights are saved, the
results of the kept weights per class and the preview images are in ``segmentation``::

      "segmentation": {
        "loss": "cross_entropy",       # or "cross_entropy_dice"
        "binary": false,               # true: one output channel (two classes)
        "ignore_index": null,          # 255 if the masks mark pixels to ignore
        "class_names": ["background", "circle"], "colors": ["#20242c", "#e5534b"],
        "validation": {"epoch": 7, "iou": 0.81, "dice": 0.89, "pixel_accuracy": 0.97,
                       "class_iou": [0.98, 0.64], "class_dice": [0.99, 0.78]},
                                       # null if the kept weights are not those of a
                                       # finished epoch; a class entry is null for a class
                                       # that occurs neither in the masks nor the predictions
        "test": {...},                 # the same on the test set (without "epoch"), or null
        "previews": {"count": 8, "epoch": 7, "split": "validation"}   # previews/, or null;
                                       # split "test" or "training" without validation data
      }

When a run with a single class output (classification) or a single value output
(regression) completes, the kept weights (in a search: those of the best trial; classical
models: the fitted model) are evaluated once more on the validation rows and, if there is a
test set, on the test rows (see evaluation.py for the definitions); a run with neither is
evaluated on its training rows instead (``training``, optimistic). Stopped and failed runs
and segmentations get none::

      "evaluation": {
        "task": "classification",      # or "regression"
        "epoch": 7,                    # weights_epoch (classical models: 1)
        "trial": null,                 # search: the best trial, whose weights these are
        "train": {"loss": 0.21, "accuracy": 0.93},   # the kept epoch in history (neural:
                                       # the mean over its batches, in training mode;
                                       # classical: the fitted model on its training rows);
                                       # accuracy null for regression
        "val": {...},                  # the validation rows, see below (null without them)
        "test": {...},                 # the same on the test rows, null without a test set
        "training": null,              # the same on the training rows, only without
                                       # validation and test rows (optimistic: the model
                                       # has seen them)
        # classification only:
        "class_names": ["cat", "dog", "bird"],      # the order of all lists and indices
        "matrix_classes_max": 50       # above this many classes no confusion matrix is kept
      }

A classification split::

      {"size": 1200, "loss": 0.25,     # loss: the run's loss (as in history)
       "accuracy": 0.93, "balanced_accuracy": 0.91,  # balanced: mean recall per class
       "macro_f1": 0.9, "weighted_f1": 0.93,         # mean F1 (weighted by support)
       "log_loss": 0.25,               # cross-entropy of the predicted probabilities
       "top_k_accuracy": {"3": 0.99, "5": 1.0},      # only k < number of classes; or null
       "per_class": {"precision": [0.95, 0.9, null], "recall": [...], "f1": [...],
                     "support": [400, 420, 380]},    # null: undefined (no predictions of
                                       # the class / no rows of it / neither)
       "confusion_matrix": [[380, 15, 5], ...],      # counts; rows: true class, columns:
                                       # predicted class; null above matrix_classes_max
       "most_confused": [{"true": 1, "predicted": 0, "count": 15}, ...]}  # up to 10 pairs

A regression split (in the units of the target column)::

      {"size": 300, "loss": 0.08,      # loss: MSE of the standardised target (as in history)
       "mae": 1.2, "mse": 2.3, "rmse": 1.52, "r2": 0.91,
       "mape": 0.04,                   # share (0.04 = 4 %); null if a target is near 0
       "median_absolute_error": 0.9, "max_error": 6.1,
       "residuals": {"q05": -2.4, "q25": -0.8, "median": 0.1, "q75": 0.9, "q95": 2.6},
                                       # quantiles of predicted − actual
       "sample": {"actual": [...], "predicted": [...]}}   # at most 500 evenly spaced rows
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
PREVIEW_DIR = "previews"
PREVIEW_INDEX_FILE = "index.json"

# A run that waits in the queue of the server for a free worker. Only the server sets it: a
# runner starts with "starting" and never reports "queued".
QUEUED = "queued"
# A training process exists for the run (or is being started right now).
RUNNING_STATUSES = ("starting", "preparing", "running")
# Not finished yet: waiting or training.
ACTIVE_STATUSES = (QUEUED, *RUNNING_STATUSES)
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
        "segmentation": None,
        "evaluation": None,
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
