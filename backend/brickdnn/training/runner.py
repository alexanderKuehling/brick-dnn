"""Run one training job in its own process: ``python -m brickdnn.training.runner RUN_DIR``.

Reads ``RUN_DIR/job.json``, keeps ``RUN_DIR/state.json`` up to date (see state.py) and
saves ``RUN_DIR/weights.pt`` (classical models: ``RUN_DIR/model.joblib``); a segmentation
also writes ``RUN_DIR/previews/`` (see previews.py). A ``STOP`` file,
SIGTERM or SIGINT stop the run after the current batch; a classical model that is being
fitted cannot be interrupted, so the process then ends right away without saving it. The
process always ends with a terminal status in ``state.json``. It runs with a lower
priority than the web server that started it (see :func:`lower_priority`).

The runner measures its own CPU time (``usage`` in state.json), so that whoever reads the
state can count it, and stops like after a stop request once it has used the CPU time in
the ``limits`` of the job (a budget or the most one run may use), also if nobody watches it.
"""

from __future__ import annotations

import contextlib
import copy
import os
import signal
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from .job import TrainingJob, graph_family
from .reporting import Reporter, TrainingError
from .state import (
    HEARTBEAT_SECONDS,
    MAX_LOG_LINES,
    PREVIEW_DIR,
    cpu_seconds,
    initial_state,
    read_job,
    stop_request,
    stop_requested,
    weights_file,
    write_state,
)

NO_SKLEARN_MESSAGE = (
    "scikit-learn is not installed on the server, which classical models need. Install it "
    "with: pip install scikit-learn"
)

# state.json is written at most this often for progress updates (seconds).
WRITE_INTERVAL = 0.5
# The web server goes first: the training gets a lower CPU priority (nice) and is the first
# process the kernel kills when memory runs out (oom_score_adj, 1000 is the most).
NICE = 10
OOM_SCORE_ADJ = 1000


def job_limits(raw: Any) -> dict[str, Any] | None:
    """``limits`` of job.json: ``{"cpu_seconds", "reason", "message"}``, or None."""
    if not isinstance(raw, dict):
        return None
    limit = raw.get("cpu_seconds")
    if isinstance(limit, bool) or not isinstance(limit, (int, float)) or not limit > 0:
        return None
    reason = raw.get("reason")
    return {
        "cpu_seconds": float(limit),
        "reason": reason if reason in ("budget", "run_limit") else "run_limit",
        "message": str(raw.get("message") or ""),
    }


class StateReporter(Reporter):
    """Mirrors the engine's progress into ``state.json``.

    All changes and writes happen under one lock; a heartbeat thread refreshes
    ``updated_at`` and ``usage`` (and flushes throttled changes) while the run is alive.
    ``limits`` (see :func:`job_limits`): stop once this much CPU time is used.
    """

    def __init__(
        self,
        run_dir: str | Path,
        epochs: int = 0,
        heartbeat: float = 1.0,
        family: str = "neural",
        limits: dict[str, Any] | None = None,
    ):
        self.run_dir = Path(run_dir)
        self.state: dict[str, Any] = initial_state(epochs, family)
        self.stop_signal = False
        self.limits = job_limits(limits)
        self.limit_reached = False
        self._started = time.monotonic()
        self._lock = threading.Lock()
        self._last_write = 0.0
        self._dirty = False
        self._done = threading.Event()
        with self._lock:
            self._write()
        self._thread = threading.Thread(
            target=self._heartbeat, args=(heartbeat,), name="heartbeat", daemon=True
        )
        self._thread.start()

    # -- writing ------------------------------------------------------------ #

    def _write(self) -> None:
        """Write state.json (call with the lock held)."""
        now = time.time()
        self.state["updated_at"] = now
        self.state["usage"] = {
            "cpu_seconds": round(cpu_seconds(), 3),
            "wall_seconds": round(time.monotonic() - self._started, 3),
            "limit_cpu_seconds": self.limits["cpu_seconds"] if self.limits else None,
        }
        try:
            write_state(self.run_dir, self.state)
        except OSError as exc:
            print(f"Could not write the state file: {exc}", file=sys.stderr, flush=True)
            return
        self._last_write = now
        self._dirty = False

    def _update(self, force: bool = True, **fields: Any) -> None:
        with self._lock:
            self.state.update(fields)
            if force or time.time() - self._last_write >= WRITE_INTERVAL:
                self._write()
            else:
                self._dirty = True

    def _heartbeat(self, interval: float) -> None:
        refresh = min(HEARTBEAT_SECONDS / 2, 2.0)
        while not self._done.wait(interval):
            self._check_limit()
            with self._lock:
                if self._done.is_set():
                    return
                if self._dirty or time.time() - self._last_write >= refresh:
                    self._write()

    def _check_limit(self) -> bool:
        """True (and remembered) once the CPU time of ``limits`` is used up."""
        if not self.limit_reached and self.limits is not None:
            self.limit_reached = cpu_seconds() >= self.limits["cpu_seconds"]
        return self.limit_reached

    def _stop_reason(self) -> tuple[str, str]:
        """Why the run stopped: ("user" | "budget" | "run_limit", start of the message)."""
        if self.limit_reached and self.limits is not None:
            return self.limits["reason"], self.limits["message"]
        request = stop_request(self.run_dir)
        if request is not None:
            return request["reason"], request["message"]
        return "user", ""

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self.state)

    # -- Reporter ------------------------------------------------------------ #

    def status(self, status: str, message: str) -> None:
        self._update(force=status != self.state["status"], status=status, message=message)

    def log(self, line: str) -> None:
        print(line, flush=True)
        with self._lock:
            self.state["log"] = [*self.state["log"], line][-MAX_LOG_LINES:]
            self._write()

    def setup(self, info: dict[str, Any]) -> None:
        self._update(**info)

    def progress(self, progress: dict[str, Any]) -> None:
        self._update(force=False, progress=progress)

    def epoch_end(self, entry: dict[str, Any], best: dict[str, Any] | None) -> None:
        line = f"Epoch {entry['epoch']}: " + ", ".join(
            f"{k} {v:.4g}" for k, v in entry.items() if k != "epoch" and v is not None
        )
        print(line, flush=True)
        with self._lock:
            self.state["history"] = [*self.state["history"], entry]
            self.state["best"] = best
            self.state["progress"] = None
            trials = self.state["trials"]
            if self.state["trial"] is not None and trials:
                # Live values of the running trial.
                current = dict(trials[-1])
                current["epochs_done"] = entry["epoch"]
                for key in ("train_loss", "val_loss", "val_accuracy", "val_rmse", "val_r2"):
                    current[key] = entry.get(key)
                for key in ("val_iou", "val_dice", "val_pixel_accuracy"):  # segmentation
                    if key in entry:
                        current[key] = entry[key]
                self.state["trials"] = [*trials[:-1], current]
            self._write()

    def trial_start(self, trial: dict[str, Any]) -> None:
        print(f"Trial {trial['trial']} of {trial['trials']}: {trial['params']}", flush=True)
        entry = {
            "trial": trial["trial"],
            "params": trial["params"],
            "status": "running",
            "message": "",
            "epochs_done": 0,
            "train_loss": None,
            "val_loss": None,
            "val_accuracy": None,
            "val_rmse": None,
            "val_r2": None,
            "weights_epoch": None,
            "stopped_early": False,
            "seconds": None,
            "score": None,
        }
        with self._lock:
            self.state.update(
                trial=trial["trial"],
                steps_per_epoch=trial.get("steps_per_epoch"),
                history=[],
                best=None,
                progress=None,
                trials=[*self.state["trials"], entry],
            )
            self._write()

    def trial_end(self, trial: dict[str, Any], best_trial: int | None) -> None:
        print(f"Trial {trial['trial']} {trial['status']}: {trial}", flush=True)
        with self._lock:
            trials = [t for t in self.state["trials"] if t.get("trial") != trial["trial"]]
            self.state.update(trials=[*trials, trial], best_trial=best_trial, progress=None)
            self._write()

    def search_end(self, result: dict[str, Any]) -> None:
        print(f"Best trial: {result['best_trial']} {result['params']}", flush=True)
        with self._lock:
            self.state.update(
                trial=None,
                best_trial=result["best_trial"],
                history=result["history"],
                best=result["best"],
                progress=None,
            )
            self._write()

    def weights_kept(self, result: dict[str, Any]) -> None:
        print(f"Weights kept: {result}", flush=True)
        self._update(
            weights_epoch=result.get("weights_epoch"),
            stopped_early=bool(result.get("stopped_early")),
        )

    def test_end(self, result: dict[str, Any]) -> None:
        print(f"Test: {result}", flush=True)
        self._update(test=result, progress=None)

    def evaluation_end(self, result: dict[str, Any]) -> None:
        print("Final evaluation done.", flush=True)
        self._update(evaluation=result)

    def segmentation_end(self, result: dict[str, Any]) -> None:
        print(f"Segmentation: {result}", flush=True)
        self._update(segmentation=result)

    def finish(self, status: str, message: str, weights: bool) -> None:
        reason = None
        if status == "stopped":
            reason, why = self._stop_reason()
            message = f"{why} {message}" if why else message
        print(f"{status}: {message}", flush=True)
        with self._lock:
            self._done.set()
            self.state.update(
                status=status,
                message=message,
                weights=weights,
                progress=None,
                trial=None,
                finished_at=time.time(),
                stop_reason=reason,
            )
            self._write()
        self._thread.join(timeout=5)

    def should_stop(self) -> bool:
        return self.stop_signal or stop_requested(self.run_dir) or self._check_limit()

    @property
    def finished(self) -> bool:
        return self._done.is_set()


def _install_signal_handlers(reporter: StateReporter) -> None:
    def handler(signum: int, frame: Any) -> None:
        reporter.stop_signal = True

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, handler)


def run(run_dir: str | Path) -> dict[str, Any]:
    """Run the job in ``run_dir`` and return the final state.

    Neural networks are trained with PyTorch (:mod:`.engine`, saved as ``weights.pt``),
    classical models with scikit-learn (:mod:`.classical_engine`, ``model.joblib``). An
    ensemble combines finished runs (:mod:`.ensemble_engine`, ``ensemble.npz``).
    """
    run_dir = Path(run_dir)
    job_data = read_job(run_dir)
    family = graph_family(job_data.get("graph")) if job_data else "neural"
    if job_data and isinstance(job_data.get("ensemble"), dict):
        family = "ensemble"
    epochs = 1 if family in ("classical", "ensemble") else 0
    if family == "neural" and job_data and isinstance(job_data.get("config"), dict):
        value = job_data["config"].get("epochs")
        epochs = value if isinstance(value, int) else 0
    limits = job_data.get("limits") if job_data else None
    reporter = StateReporter(run_dir, epochs, family=family, limits=limits)
    _install_signal_handlers(reporter)
    try:
        if family == "ensemble":
            from . import ensemble_engine  # NumPy; PyTorch only for neural members

            ensemble_engine.run(job_data, reporter, run_dir)
            return reporter.snapshot()
        if family == "classical":
            _run_classical(run_dir, job_data, reporter)
            return reporter.snapshot()
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        try:
            from . import engine
        except ImportError:
            traceback.print_exc()
            reporter.finish(
                "failed",
                "PyTorch is not installed on the server. "
                "Install it with: pip install torch torchvision",
                False,
            )
            return reporter.snapshot()
        if job_data is None:
            raise TrainingError("The training job is missing or invalid.")
        job = TrainingJob.from_dict(job_data)
        engine.train(job, reporter, run_dir / weights_file(family), run_dir / PREVIEW_DIR)
    except TrainingError as exc:
        if not reporter.finished:
            reporter.finish("failed", str(exc), False)
    except BaseException as exc:
        traceback.print_exc()
        if not reporter.finished:
            reporter.finish("failed", f"Training failed: {type(exc).__name__}: {exc}"[:400], False)
    return reporter.snapshot()


def _run_classical(run_dir: Path, job_data: Any, reporter: StateReporter) -> None:
    """Fit a classical model with scikit-learn (PyTorch is not needed)."""
    try:
        from . import classical_engine
    except ImportError:
        traceback.print_exc()
        reporter.finish("failed", NO_SKLEARN_MESSAGE, False)
        return
    if job_data is None:
        raise TrainingError("The training job is missing or invalid.")
    job = TrainingJob.from_dict(job_data)
    # A stop request during the fit ends this process (scikit-learn cannot be interrupted).
    classical_engine.train(job, reporter, run_dir / weights_file("classical"), hard_stop=True)


def lower_priority() -> None:
    """Make way for the web server, which keeps the normal priority.

    The API gets the CPU first while a training runs, and if the memory runs out (e.g. the
    memory limit of the Docker container), the kernel kills the training, not the API. Call
    it before any thread starts: threads inherit the priority. Every process may lower its
    own priority; where a setting does not exist (Windows, macOS), it is skipped.
    """
    with contextlib.suppress(AttributeError, OSError):  # no os.nice on Windows
        os.nice(NICE)
    with contextlib.suppress(OSError):  # Linux only
        Path("/proc/self/oom_score_adj").write_text(f"{OOM_SCORE_ADJ}\n")


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m brickdnn.training.runner RUN_DIR", file=sys.stderr)
        return 2
    run_dir = Path(args[0])
    if not run_dir.is_dir():
        print(f"Run directory not found: {run_dir}", file=sys.stderr)
        return 2
    lower_priority()
    run(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
