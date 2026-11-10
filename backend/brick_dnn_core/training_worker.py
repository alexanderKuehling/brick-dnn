"""The training worker: takes runs from the queue and trains them.

``manage.py training_worker`` runs a :class:`Worker` in its own process (in Docker: its own
container, with its own CPU and memory limits), so trainings cannot slow the web server
down. The worker

- says every few seconds that it is alive (a :class:`TrainingWorker` row with a heartbeat);
- takes the next run of the queue whenever it has a free slot (training_queue.claim),
  starts the usual runner process for it (training_service.start_claimed) and watches it
  exactly like the web server does without a worker (training_service.sync_run): progress,
  usage and budgets, stop requests, dead and stuck processes;
- kills the process of a run that was deleted meanwhile;
- when it is asked to stop (SIGTERM: a deploy, ``docker compose stop``), takes no new runs,
  lets the running ones finish for BRICKDNN_WORKER_DRAIN_SECONDS, then kills what is left
  and puts those runs back into the queue;
- when it starts, takes over what a worker of the same name left behind: runs whose
  process still lives are watched again, the others go back into the queue.

All a worker needs is the database, the run directories and the datasets. It keeps nothing
in memory that it could not find again, and with ``--once`` it ends after one run, so the
same command can later run as a Kubernetes Job or on another server (docs/scaling.md).
"""

from __future__ import annotations

import contextlib
import math
import os
import signal
import socket
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from django.conf import settings
from django.db import DatabaseError, close_old_connections, connection
from django.utils import timezone

from brickdnn.training import QUEUED, RUNNING_STATUSES, TERMINAL_STATUSES

from . import training_queue as queue
from . import training_service as training
from .models import TrainingAttempt, TrainingRun, TrainingWorker

# The worker touches this file in every round; the health check of its container looks at
# its age (cheaper than starting Django twice a minute).
ALIVE_FILE = Path(tempfile.gettempdir()) / "brickdnn-worker.alive"
# A process that has reported its end gets this long to exit before it is killed.
EXIT_SECONDS = 10.0


def touch_alive_file() -> None:
    with contextlib.suppress(OSError):
        ALIVE_FILE.touch()


def _reconnect() -> None:
    """Drop the database connection after an error, so that the next query opens a new one.
    Never inside a transaction someone else opened (a test's): that would end it."""
    if connection.in_atomic_block:
        return
    with contextlib.suppress(Exception):
        connection.close()


def _kill_group(process: Any) -> None:
    """Kill a training process this worker started, with its children."""
    if process.poll() is not None:
        return
    with contextlib.suppress(OSError):
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:  # pragma: no cover - Windows
            process.kill()


def heartbeat_seconds() -> float:
    """How often a worker says that it is alive: three times within the timeout."""
    return max(1.0, min(10.0, settings.WORKER_TIMEOUT / 3))


def _cgroup_number(*paths: str) -> list[str] | None:
    for path in paths:
        with contextlib.suppress(OSError):
            return Path(path).read_text().split()
    return None


def available_cpus() -> float:
    """The CPUs this process may use: the limit of its container (cgroup), else the CPUs it
    may run on."""
    quota = _cgroup_number("/sys/fs/cgroup/cpu.max")  # cgroup v2: "150000 100000" or "max …"
    if quota and len(quota) == 2 and quota[0] != "max":
        with contextlib.suppress(ValueError, ZeroDivisionError):
            return max(0.1, int(quota[0]) / int(quota[1]))
    v1 = _cgroup_number("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    period = _cgroup_number("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if v1 and period:
        with contextlib.suppress(ValueError, ZeroDivisionError):
            if int(v1[0]) > 0:
                return max(0.1, int(v1[0]) / int(period[0]))
    with contextlib.suppress(AttributeError, OSError):
        return float(len(os.sched_getaffinity(0)))
    return float(os.cpu_count() or 1)


def job_threads() -> int | None:
    """The CPU threads of one training (OMP_NUM_THREADS), None if it is not set: PyTorch
    then uses every CPU."""
    with contextlib.suppress(ValueError):
        threads = int(os.environ.get("OMP_NUM_THREADS", ""))
        if threads > 0:
            return threads
    return None


def default_slots() -> int:
    """Trainings a worker runs at the same time unless it is told (``--slots``,
    BRICKDNN_WORKER_SLOTS): as many as fit on its CPUs with OMP_NUM_THREADS threads each,
    and one if that is not set (a training then uses all CPUs)."""
    threads = job_threads()
    if threads is None:
        return 1
    return max(1, int(available_cpus() // threads))


def memory_limit_mb() -> int | None:
    """The memory limit of this container in megabytes, None without one."""
    value = _cgroup_number(
        "/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"
    )
    if not value or value[0] == "max":
        return None
    with contextlib.suppress(ValueError):
        limit = int(value[0])
        # cgroup v1 reports "no limit" as a huge number.
        return limit // (1024 * 1024) if limit < 1 << 50 else None
    return None


class Worker:
    """One training worker. :meth:`run` is the whole life of ``manage.py training_worker``;
    :meth:`start`, :meth:`step` and :meth:`stop` are its parts (tests call them one by one).
    """

    def __init__(
        self,
        name: str | None = None,
        slots: int | None = None,
        once: bool = False,
        poll: float = 2.0,
        idle_exit: float | None = None,
        drain: float | None = None,
        log: Callable[[str], Any] | None = None,
    ):
        self.hostname = socket.gethostname()[:255]
        self.name = (name or settings.WORKER_NAME or self.hostname)[:120]
        self.slots = max(1, int(slots or settings.WORKER_SLOTS or default_slots()))
        self.threads = job_threads() or max(1, math.ceil(available_cpus()))
        self.once = once
        self.poll = max(0.05, poll)
        self.idle_exit = idle_exit
        self.drain = settings.WORKER_DRAIN_SECONDS if drain is None else drain
        self.log = log or (lambda line: None)
        self.row: TrainingWorker | None = None
        # The runs this worker trains, with the process of each; for a run it took over
        # from a predecessor (that process is not its child) the pid instead.
        self.jobs: dict[str, Any] = {}
        # Processes whose run has ended and that have not exited yet: (process, deadline).
        self.ending: dict[str, tuple[Any, float]] = {}
        self.claimed = 0
        self.stopping = False
        self.stop_now = False
        self._drain_until: float | None = None
        self._last_beat = 0.0
        self._last_recover = 0.0
        self._last_prune = 0.0
        self._last_activity = time.monotonic()

    # ------------------------------------------------------------------ life
    def run(self) -> int:
        """Work until asked to stop (or, with ``once``, until one run is done)."""
        previous = self._install_signal_handlers()
        try:
            if not self._wait_for_database():
                return 0
            self.start()
            try:
                while not self._finished():
                    try:
                        self.step()
                    except DatabaseError as exc:
                        self._database_trouble(exc)
                    self._sleep()
            finally:
                self.stop()
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        return 0

    def start(self) -> None:
        """Register this worker and take over what a worker of the same name left behind."""
        now = timezone.now()
        previous = list(TrainingWorker.objects.filter(name=self.name, stopped_at__isnull=True))
        self.row = TrainingWorker.objects.create(
            name=self.name,
            hostname=self.hostname,
            pid=os.getpid(),
            slots=self.slots,
            threads=self.threads,
            memory_mb=memory_limit_mb(),
            started_at=now,
            heartbeat_at=now,
        )
        self._last_beat = time.monotonic()
        with training.supervising(self.row.pk):
            for old in previous:
                self._take_over(old)
                TrainingWorker.objects.filter(pk=old.pk, stopped_at__isnull=True).update(
                    stopped_at=now
                )
            self._recover()
        memory = f", {self.row.memory_mb} MB" if self.row.memory_mb else ""
        self.log(
            f"Training worker “{self.name}” started: {self.slots} slot"
            f"{'' if self.slots == 1 else 's'}, {self.threads} CPU thread"
            f"{'' if self.threads == 1 else 's'} per training{memory}."
        )

    def _take_over(self, old: TrainingWorker) -> None:
        runs = TrainingRun.objects.filter(worker=old, status__in=RUNNING_STATUSES)
        for run in runs.select_related("owner", "worker", "api_token", "connection__client"):
            same_machine = old.hostname == self.hostname
            if same_machine and run.pid and training.is_runner(run.pid, run):
                # Its process survived the old worker: watch it again.
                TrainingRun.objects.filter(pk=run.pk, worker=old).update(worker=self.row)
                TrainingAttempt.objects.filter(run=run, ended_at__isnull=True).update(
                    worker=self.row
                )
                self.jobs[str(run.pk)] = run.pid
                self.log(f"Watching “{run.name}” again (its process is still running).")
            else:
                outcome = queue.release(run, lost=True)
                self.log(f"“{run.name}” of the previous worker: {outcome or 'already handled'}.")

    def step(self) -> None:
        """One round: heartbeat, look after the running trainings, take new ones."""
        assert self.row is not None, "call start() first"
        now = time.monotonic()
        with training.supervising(self.row.pk):
            if now - self._last_beat >= heartbeat_seconds():
                self._beat()
            self._watch()
            self._reap()
            if now - self._last_recover >= settings.WORKER_TIMEOUT / 2:
                self._recover()
            if not self.stopping and not (self.once and self.claimed):
                self._claim()
        if now - self._last_prune >= 3600:
            self._last_prune = now
            queue.prune()
        touch_alive_file()

    def stop(self) -> None:
        """End in an orderly way: the trainings that still run are killed and go back into
        the queue (in their old place), and the worker says that it is gone."""
        if self.row is None:
            return
        # First the processes (that needs no database), then what becomes of their runs.
        run_ids = list(self.jobs)
        self._abandon()
        with contextlib.suppress(DatabaseError), training.supervising(self.row.pk):
            for run_id in run_ids:
                run = self._load(run_id)
                if run is not None and run.worker_id == self.row.pk:
                    outcome = queue.release(run, lost=False)
                    self.log(f"“{run.name}”: {outcome or 'already handled'}.")
            now = timezone.now()
            TrainingWorker.objects.filter(pk=self.row.pk, stopped_at__isnull=True).update(
                stopped_at=now, heartbeat_at=now
            )
        self.log(f"Training worker “{self.name}” stopped.")

    # ------------------------------------------------------------------ rounds
    @property
    def busy(self) -> int:
        return len(self.jobs) + len(self.ending)

    def _load(self, run_id: str) -> TrainingRun | None:
        return (
            TrainingRun.objects.filter(pk=run_id)
            .select_related("owner", "worker", "api_token", "connection__client")
            .first()
        )

    def _beat(self) -> None:
        now = timezone.now()
        alive = TrainingWorker.objects.filter(pk=self.row.pk, stopped_at__isnull=True).update(
            heartbeat_at=now
        )
        self._last_beat = time.monotonic()
        if not alive:
            # Someone declared this worker dead (e.g. it could not reach the database for a
            # while) and put its runs back into the queue: let go of them and start afresh.
            self.log("This worker was considered dead: starting afresh.")
            self._abandon()
            self.row = TrainingWorker.objects.create(
                name=self.name,
                hostname=self.hostname,
                pid=os.getpid(),
                slots=self.slots,
                threads=self.threads,
                memory_mb=memory_limit_mb(),
            )

    def _abandon(self) -> None:
        """Kill the processes of all runs without touching the runs."""
        for run_id in list(self.jobs):
            self._kill(run_id, wait=5.0)
        for process, _deadline in self.ending.values():
            _kill_group(process)
        self.jobs.clear()
        self.ending.clear()

    def _kill(self, run_id: str, wait: float = 0.0) -> None:
        """Kill the training process of one of this worker's runs."""
        process = self.jobs.get(run_id)
        pid = process if isinstance(process, int) else None  # taken over: by its pid
        training.kill_process(run_id, wait=wait, pid=pid)

    def _watch(self) -> None:
        for run_id in sorted(self.jobs):
            run = self._load(run_id)
            if run is None:
                # Deleted: the web server could not kill the process, so this worker does.
                self._kill(run_id, wait=3.0)
                queue.remove_run_files(run_id)
                del self.jobs[run_id]
                self.log(f"Run {run_id} was deleted: its training was stopped.")
            elif run.worker_id != self.row.pk or run.status == QUEUED:
                # Not this worker's any more (it was considered dead meanwhile).
                self._kill(run_id, wait=3.0)
                del self.jobs[run_id]
            else:
                training.sync_run(run)
                if run.status in TERMINAL_STATUSES:
                    queue.close_attempt(run.pk, run.status)
                    process = self.jobs.pop(run_id)
                    if not isinstance(process, int | None) and process.poll() is None:
                        self.ending[run_id] = (process, time.monotonic() + EXIT_SECONDS)
                    self._last_activity = time.monotonic()
                    self.log(f"“{run.name}” {run.status}: {run.message}"[:300])

    def _reap(self) -> None:
        for run_id, (process, deadline) in list(self.ending.items()):
            if process.poll() is not None:
                del self.ending[run_id]
            elif time.monotonic() > deadline:
                _kill_group(process)
                del self.ending[run_id]

    def _recover(self) -> None:
        self._last_recover = time.monotonic()
        for run, outcome in queue.recover():
            self.log(f"“{run.name}” of a worker that is gone: {outcome}.")

    def _claim(self) -> None:
        while self.busy < self.slots and not (self.once and self.claimed):
            taken = queue.claim(self.row)
            if taken is None:
                return
            run, limit = taken
            training.start_claimed(run, limit)
            run_id = str(run.pk)
            self.jobs[run_id] = training.process_of(run_id)
            self.claimed += 1
            self._last_activity = time.monotonic()
            self.log(f"Started “{run.name}” of {run.owner.get_username()} ({run_id}).")

    # ------------------------------------------------------------------ the loop
    def _finished(self) -> bool:
        if self.stop_now:
            return True
        if self.stopping:
            if self._drain_until is None:
                self._drain_until = time.monotonic() + self.drain
                if self.jobs and self.drain > 0:
                    self.log(
                        f"Stopping: {len(self.jobs)} training(s) get {self.drain:g} s to "
                        "finish, then they go back into the queue."
                    )
            return not self.jobs or time.monotonic() >= self._drain_until
        if self.once and self.claimed and not self.busy:
            return True
        idle = time.monotonic() - self._last_activity
        return self.idle_exit is not None and not self.busy and idle >= self.idle_exit

    def _sleep(self) -> None:
        if not connection.in_atomic_block:
            close_old_connections()  # honours BRICKDNN_DB_CONN_MAX_AGE between rounds
        # Asked to stop while it waits: go on at once (and look more often while the
        # running trainings get their time to finish).
        waiting = self.stopping
        deadline = time.monotonic() + (min(self.poll, 0.5) if waiting else self.poll)
        while time.monotonic() < deadline and not self.stop_now:
            if self.stopping and not waiting:
                return
            time.sleep(min(0.2, self.poll))

    def _database_trouble(self, exc: Exception) -> None:
        self.log(f"Database error, trying again: {exc}"[:300])
        _reconnect()
        # Without a heartbeat the others take this worker for dead after the timeout and
        # start its runs again: stop training them before that happens.
        if self.jobs and time.monotonic() - self._last_beat > settings.WORKER_TIMEOUT * 0.8:
            self.log("No heartbeat for too long: stopping the trainings of this worker.")
            self._abandon()

    def _wait_for_database(self) -> bool:
        """Wait until the database is there and migrated (the web server migrates it)."""
        announced = False
        while not self.stopping:
            try:
                TrainingWorker.objects.exists()
                return True
            except DatabaseError as exc:
                if not announced:
                    self.log(f"Waiting for the database: {exc}"[:300])
                    announced = True
                _reconnect()
                touch_alive_file()  # waiting, but alive
                time.sleep(2.0)
        return False

    def _install_signal_handlers(self) -> dict[int, Any]:
        """Stop on SIGTERM and SIGINT. Returns the handlers that were there before."""

        def handler(signum: int, frame: Any) -> None:
            if self.stopping:
                self.stop_now = True  # asked twice: do not wait for the trainings
            self.stopping = True

        return {sig: signal.signal(sig, handler) for sig in (signal.SIGTERM, signal.SIGINT)}
