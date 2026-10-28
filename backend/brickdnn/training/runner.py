"""Run one training job in its own process: ``python -m brickdnn.training.runner RUN_DIR``.

Reads ``RUN_DIR/job.json``, keeps ``RUN_DIR/state.json`` up to date (see state.py) and
saves ``RUN_DIR/weights.pt``. A ``STOP`` file, SIGTERM or SIGINT stop the run after the
current batch. The process always ends with a terminal status in ``state.json``.
"""

from __future__ import annotations

import copy
import os
import signal
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from .reporting import Reporter, TrainingError
from .state import (
    HEARTBEAT_SECONDS,
    MAX_LOG_LINES,
    WEIGHTS_FILE,
    initial_state,
    read_job,
    stop_requested,
    write_state,
)

# state.json is written at most this often for progress updates (seconds).
WRITE_INTERVAL = 0.5


class StateReporter(Reporter):
    """Mirrors the engine's progress into ``state.json``.

    All changes and writes happen under one lock; a heartbeat thread refreshes
    ``updated_at`` (and flushes throttled changes) while the run is alive.
    """

    def __init__(self, run_dir: str | Path, epochs: int = 0, heartbeat: float = 1.0):
        self.run_dir = Path(run_dir)
        self.state: dict[str, Any] = initial_state(epochs)
        self.stop_signal = False
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
            with self._lock:
                if self._done.is_set():
                    return
                if self._dirty or time.time() - self._last_write >= refresh:
                    self._write()

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

    def epoch_end(self, entry: dict[str, Any], best: dict[str, Any]) -> None:
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
                for key in ("train_loss", "val_loss", "val_accuracy", "val_rmse"):
                    current[key] = entry.get(key)
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
            "seconds": None,
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

    def test_end(self, result: dict[str, Any]) -> None:
        print(f"Test: {result}", flush=True)
        self._update(test=result, progress=None)

    def finish(self, status: str, message: str, weights: bool) -> None:
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
            )
            self._write()
        self._thread.join(timeout=5)

    def should_stop(self) -> bool:
        return self.stop_signal or stop_requested(self.run_dir)

    @property
    def finished(self) -> bool:
        return self._done.is_set()


def _install_signal_handlers(reporter: StateReporter) -> None:
    def handler(signum: int, frame: Any) -> None:
        reporter.stop_signal = True

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, handler)


def run(run_dir: str | Path) -> dict[str, Any]:
    """Run the job in ``run_dir`` and return the final state."""
    run_dir = Path(run_dir)
    job_data = read_job(run_dir)
    epochs = 0
    if job_data and isinstance(job_data.get("config"), dict):
        value = job_data["config"].get("epochs")
        epochs = value if isinstance(value, int) else 0
    reporter = StateReporter(run_dir, epochs)
    _install_signal_handlers(reporter)
    try:
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
        job = engine.TrainingJob.from_dict(job_data)
        engine.train(job, reporter, run_dir / WEIGHTS_FILE)
    except TrainingError as exc:
        if not reporter.finished:
            reporter.finish("failed", str(exc), False)
    except BaseException as exc:
        traceback.print_exc()
        if not reporter.finished:
            reporter.finish("failed", f"Training failed: {type(exc).__name__}: {exc}"[:400], False)
    return reporter.snapshot()


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m brickdnn.training.runner RUN_DIR", file=sys.stderr)
        return 2
    run_dir = Path(args[0])
    if not run_dir.is_dir():
        print(f"Run directory not found: {run_dir}", file=sys.stderr)
        return 2
    run(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
