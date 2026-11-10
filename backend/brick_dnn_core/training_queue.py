"""The training queue: runs wait in the database until a training worker takes them.

With ``BRICKDNN_TRAINING_WORKER`` the web server does not start trainings itself. A new run
is stored as ``queued`` (after all the checks of training_service.start_run) and a worker
process, ``manage.py training_worker`` (training_worker.py), takes it when it has a free slot,
trains it and reports back through the database and the run directory. Without the setting
nothing here is used: the web server trains itself, as it always did.

**Order.** The run of the plan with the highest ``priority`` goes first (accounts without
limits, e.g. staff, come before every plan). Among runs of the same priority the accounts
take turns: an account's first waiting run comes before every account's second one, and a
training an account has running counts like a turn it already had. Within a turn the older
run goes first. A run whose account already trains as many runs as its plan allows at a time
keeps its place and is passed over until one of them ends.

**Claiming** (:func:`claim`) is safe with several workers: on PostgreSQL the run is locked
with ``SELECT … FOR UPDATE SKIP LOCKED`` and the account with an advisory lock, both inside
one transaction with the change of the status; on SQLite (one server) the start lock of
training_service serialises claims, and the status only changes if it still is ``queued``.
The budgets are checked again at that moment: the run gets what is left then, or ends with
the budget's message.

**Workers** (:class:`TrainingWorker`, one row per start) say every few seconds that they are
alive. The runs of a worker that stopped saying so for ``BRICKDNN_WORKER_TIMEOUT`` seconds go
back into the queue (:func:`recover`, :func:`release`): once after a worker died, and at most
``BRICKDNN_WORKER_MAX_RESTARTS`` times in all, then the run fails with a message that says so.
A restarted run starts from the beginning; the CPU time of the interrupted attempt stays
counted and the new attempt only adds what it uses beyond that, so nothing is counted twice.

Everything the API, the MCP server and the admin show about the queue (positions, the
estimated start, whether a worker is running, waiting times and utilisation) comes from
:func:`snapshot`, :func:`run_info`, :func:`status`, :func:`overview` and :func:`stats`.
"""

from __future__ import annotations

import contextlib
import heapq
import logging
import math
import shutil
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from django.conf import settings
from django.db import connection, transaction
from django.db.models import F, Q
from django.utils import timezone

from brickdnn.training import QUEUED, RUNNING_STATUSES, TERMINAL_STATUSES
from brickdnn.training.state import STATE_FILE, STOP_FILE

from . import budgets
from .budgets import Actor, Plan
from .models import TrainingAttempt, TrainingRun, TrainingWorker, UserProfile

logger = logging.getLogger(__name__)

QUEUED_MESSAGE = "Waiting for a free training slot."
CANCELLED_MESSAGE = "Cancelled before it started."
RESTARTED_MESSAGE = "Waiting to start again from the beginning: the training worker was restarted."
LOST_MESSAGE = (
    "Waiting to start again from the beginning: the training worker stopped unexpectedly."
)
LOST_TWICE_MESSAGE = (
    "The training worker stopped unexpectedly twice while this training ran, so it was not "
    "started again. Start it again; if this keeps happening, the training may need more "
    "memory than the server has."
)
NO_WORKER_MESSAGE = (
    "No training worker is running on this server right now. Queued trainings start as soon "
    "as one is back."
)
MODE_OFF_MESSAGE = (
    "The training queue of this server was switched off before this training started. "
    "Start it again."
)
ESTIMATE_NOTE = (
    "A rough estimate from the CPU-time estimates of the trainings ahead and the worker's "
    "slots; trainings often end earlier or later, and runs of higher plans may still come "
    "first."
)
# A run without an estimate and without a limit is assumed to use this much CPU time.
DEFAULT_RUN_SECONDS = 600.0
# A running training is assumed to need at least this much longer.
MIN_REMAINING_SECONDS = 15.0
# Rows of attempts and of workers that ended are kept this long (for the statistics).
KEEP_DAYS = 90
# Advisory locks of this module (PostgreSQL): the high half of the key.
_LOCK_SPACE = 0x42444E51  # "BDNQ"

# Large columns the queue never needs.
_DEFERRED = ("graph", "plan", "config", "tuning", "metrics", "best", "best_trial")


def enabled() -> bool:
    """Trainings wait in the queue for a worker (False: the web server trains itself)."""
    return bool(settings.TRAINING_WORKER)


# --------------------------------------------------------------------------- #
# Workers
# --------------------------------------------------------------------------- #


def _cutoff(now: datetime | None = None) -> datetime:
    return (now or timezone.now()) - timedelta(seconds=settings.WORKER_TIMEOUT)


def is_alive(worker: TrainingWorker | None, now: datetime | None = None) -> bool:
    """The worker has not stopped and sent a heartbeat recently."""
    return worker is not None and worker.stopped_at is None and worker.heartbeat_at >= _cutoff(now)


def alive_workers(now: datetime | None = None) -> list[TrainingWorker]:
    return list(
        TrainingWorker.objects.filter(
            stopped_at__isnull=True, heartbeat_at__gte=_cutoff(now)
        ).order_by("started_at", "pk")
    )


# --------------------------------------------------------------------------- #
# Accounts: priority and trainings at a time
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Account:
    """What the queue needs to know about an account."""

    priority: int
    plan: Plan
    unlimited: bool = False
    # Trainings of the account that may run at the same time (None: no limit).
    concurrent: int | None = None


def top_priority() -> int:
    """The priority of accounts without limits: above every plan."""
    return max(plan.priority for plan in budgets.plans().values()) + 1


def account_of(user, profile: UserProfile | None = None) -> Account:
    """The account's place in the queue, from its plan (``profile``: its profile if it was
    loaded already)."""
    if profile is None:
        profile = UserProfile.objects.filter(user=user).first() or UserProfile(user=user)
    plan = budgets.plan_of(user, profile)
    if budgets.is_unlimited(user, profile):
        return Account(top_priority(), plan, unlimited=True)
    return Account(plan.priority, plan, concurrent=plan.concurrent_trainings)


def _accounts(users: Iterable[Any]) -> dict[int, Account]:
    """:func:`account_of` for many users with one query."""
    users = {user.pk: user for user in users}
    profiles = {p.user_id: p for p in UserProfile.objects.filter(user__in=list(users))}
    return {
        pk: account_of(user, profiles.get(pk) or UserProfile(user=user))
        for pk, user in users.items()
    }


def priority_hint(account: Account) -> str | None:
    """One line on what the plan means for the place in the queue; None on a server whose
    plans all have the same priority."""
    plans = list(budgets.plans().values())
    levels = {plan.priority for plan in plans}
    if len(levels) < 2 or account.unlimited:
        return None
    if account.priority >= max(levels):
        return f"Your plan ({account.plan.name}) starts before the other plans."
    higher = [plan for plan in plans if plan.priority > account.priority]
    free = not (account.plan.price or 0) > 0
    if free and all((plan.price or 0) > 0 for plan in higher):
        return "Paid plans start first."
    return "Higher plans start first."


# --------------------------------------------------------------------------- #
# The order of the queue and the estimated start
# --------------------------------------------------------------------------- #


@dataclass
class Entry:
    """A waiting run (or, for :func:`preview`, one that might be added)."""

    id: str
    owner_id: int
    created_at: datetime
    account: Account
    # The wall time the run is expected to train (for the estimated start of later ones).
    seconds: float
    run: TrainingRun | None = None
    position: int = 0
    # Its account already trains as many runs as its plan allows at a time.
    blocked: bool = False
    # Estimated seconds until it starts (None: unknown, e.g. without a worker).
    wait: float | None = None


@dataclass
class Snapshot:
    """The queue at one moment: who waits in which order, who trains, which workers live."""

    now: datetime
    workers: list[TrainingWorker] = field(default_factory=list)
    running: list[TrainingRun] = field(default_factory=list)
    entries: list[Entry] = field(default_factory=list)
    accounts: dict[int, Account] = field(default_factory=dict)

    @property
    def slots(self) -> int:
        return sum(worker.slots for worker in self.workers)

    @property
    def threads(self) -> int:
        """CPU threads of one training (the smallest of the workers')."""
        return max(1, min((worker.threads for worker in self.workers), default=1))

    @property
    def online(self) -> bool:
        return bool(self.workers)

    def entry(self, run_id: Any) -> Entry | None:
        wanted = str(run_id)
        return next((entry for entry in self.entries if entry.id == wanted), None)


def _expected_cpu(run: TrainingRun) -> float:
    """The CPU time a run is expected to use: its estimate, at most its limit."""
    estimate = run.estimated_cpu_seconds or run.cpu_limit_seconds or DEFAULT_RUN_SECONDS
    if run.cpu_limit_seconds is not None:
        estimate = min(estimate, run.cpu_limit_seconds)
    return max(1.0, float(estimate))


def _remaining_cpu(run: TrainingRun) -> float:
    """The CPU time a running training probably still needs."""
    used = float(run.cpu_seconds or 0.0)
    # Past its estimate already: assume a quarter of what it used so far comes on top.
    remaining = max(_expected_cpu(run) - used, 0.25 * used, MIN_REMAINING_SECONDS)
    if run.cpu_limit_seconds is not None:
        remaining = min(remaining, max(run.cpu_limit_seconds - used, 1.0))
    return remaining


def _order(entries: list[Entry], running: Counter) -> list[Entry]:
    """``entries`` in the order they start (see the module docstring); sets ``position``
    and ``blocked``. ``running``: the trainings each account has running."""
    turns: Counter = Counter()
    keyed = []
    for entry in sorted(entries, key=lambda e: (e.created_at, e.id)):
        turn = running[entry.owner_id] + turns[entry.owner_id]
        turns[entry.owner_id] += 1
        keyed.append(((-entry.account.priority, turn, entry.created_at, entry.id), entry))
    ordered = [entry for _key, entry in sorted(keyed, key=lambda item: item[0])]
    for position, entry in enumerate(ordered, start=1):
        entry.position = position
        limit = entry.account.concurrent
        entry.blocked = limit is not None and running[entry.owner_id] >= limit
    return ordered


def _simulate(
    slots: int, busy: list[tuple[float, int]], entries: list[Entry], running: Counter
) -> None:
    """Set ``wait`` of the ordered ``entries``: when each would start if every training took
    as long as expected. ``busy``: ``(seconds left, owner)`` of the running trainings."""
    for entry in entries:
        entry.wait = None
    if slots <= 0:
        return
    ends = list(busy)
    heapq.heapify(ends)
    active = Counter(running)
    pending = list(entries)
    clock = 0.0
    while pending:
        started = None
        if len(ends) < slots:
            for entry in pending:
                limit = entry.account.concurrent
                if limit is None or active[entry.owner_id] < limit:
                    started = entry
                    break
        if started is not None:
            started.wait = clock
            heapq.heappush(ends, (clock + started.seconds, started.owner_id))
            active[started.owner_id] += 1
            pending.remove(started)
            continue
        if not ends:
            return  # nothing runs and nothing may start: no estimate for the rest
        end, owner = heapq.heappop(ends)
        clock = max(clock, end)
        active[owner] -= 1


def snapshot(now: datetime | None = None, extra: Entry | None = None) -> Snapshot:
    """The queue right now. ``extra``: a run that is not in the database (yet), to see where
    it would end up."""
    now = now or timezone.now()
    workers = alive_workers(now)
    active = list(
        TrainingRun.objects.filter(status__in=(QUEUED, *RUNNING_STATUSES))
        .select_related("owner", "worker")
        .defer(*_DEFERRED)
        .order_by("created_at", "pk")
    )
    running = [run for run in active if run.status != QUEUED]
    waiting = [run for run in active if run.status == QUEUED]
    result = Snapshot(now=now, workers=workers, running=running)
    threads = result.threads
    result.accounts = _accounts(run.owner for run in active)
    entries = [
        Entry(
            id=str(run.pk),
            owner_id=run.owner_id,
            created_at=run.created_at,
            account=result.accounts[run.owner_id],
            seconds=_expected_cpu(run) / threads,
            run=run,
        )
        for run in waiting
    ]
    if extra is not None:
        extra.seconds = extra.seconds / threads
        entries.append(extra)
    counts = Counter(run.owner_id for run in running)
    result.entries = _order(entries, counts)
    busy = [(_remaining_cpu(run) / threads, run.owner_id) for run in running]
    _simulate(result.slots, busy, result.entries, counts)
    return result


def snapshot_in(context: dict[str, Any]) -> Snapshot:
    """The snapshot of one request, made once (``context``: e.g. a serializer's)."""
    found = context.get("_training_queue")
    if found is None:
        found = snapshot()
        with contextlib.suppress(TypeError):  # an immutable context: made again next time
            context["_training_queue"] = found
    return found


def _entry_info(entry: Entry, shot: Snapshot) -> dict[str, Any]:
    wait = entry.wait if shot.online else None
    return {
        # Its place among all waiting trainings, counted from 1, and how many wait.
        "position": entry.position,
        "waiting": len(shot.entries),
        # A rough estimate (see docs/training.md); null if it cannot be told.
        "estimated_wait_seconds": round(wait) if wait is not None else None,
        "estimated_start_at": (
            budgets._iso(shot.now + timedelta(seconds=wait)) if wait is not None else None
        ),
        "worker_online": shot.online,
        # "account": it waits for a training of the same account to finish (the plan's
        # trainings at a time).
        "blocked_by": "account" if entry.blocked else None,
        "priority": entry.account.priority,
        "hint": priority_hint(entry.account),
    }


def run_info(run: TrainingRun, shot: Snapshot | None = None) -> dict[str, Any] | None:
    """``queue`` of a run in the API (``RunQueue`` in frontend/src/api/types.ts): its
    position and estimated start while it waits, else None."""
    if run.status != QUEUED or not enabled():
        return None
    shot = shot or snapshot()
    entry = shot.entry(run.pk)
    if entry is None:  # taken by a worker (or cancelled) since the run was loaded
        return None
    return _entry_info(entry, shot)


def status(user=None, shot: Snapshot | None = None) -> dict[str, Any] | None:
    """``queue`` of ``GET /api/training/`` and of the MCP tool get_training_options: whether
    a worker is running, its slots and how many trainings run and wait. None when the web
    server trains itself. With ``user``: what their plan means for their place."""
    if not enabled():
        return None
    shot = shot or snapshot()
    known = user is not None and getattr(user, "is_authenticated", False)
    return {
        "online": shot.online,
        "workers": len(shot.workers),
        "slots": shot.slots,
        "running": len(shot.running),
        "waiting": len(shot.entries),
        "message": None if shot.online else NO_WORKER_MESSAGE,
        "priority_hint": priority_hint(account_of(user)) if known else None,
    }


def preview(user, cpu_seconds: float | None = None) -> dict[str, Any] | None:
    """``queue`` of the training check: where a training that ``user`` starts now would wait
    and for how long (an estimate). None when the web server trains itself."""
    if not enabled():
        return None
    now = timezone.now()
    extra = Entry(
        id="new",
        owner_id=user.pk,
        created_at=now,
        account=account_of(user),
        seconds=float(cpu_seconds or DEFAULT_RUN_SECONDS),
    )
    shot = snapshot(now, extra=extra)
    info = _entry_info(extra, shot)
    info["waiting"] -= 1  # without the run that is being planned
    info["running"] = len(shot.running)
    info["message"] = None if shot.online else NO_WORKER_MESSAGE
    return info


# --------------------------------------------------------------------------- #
# Locks
# --------------------------------------------------------------------------- #


def _postgres() -> bool:
    return connection.vendor == "postgresql"


@contextlib.contextmanager
def account_lock(user_id: int, wait: bool = True):
    """Serialise decisions about the runs of one account (its limits and what its budgets
    still allow) across server processes and workers. Yields False if ``wait`` is off and
    someone else holds the lock.

    PostgreSQL: an advisory lock that ends with the transaction opened here. Other databases
    (SQLite: one server) rely on the start lock of training_service."""
    if not _postgres():
        yield True
        return
    key = (_LOCK_SPACE << 32) | (int(user_id) & 0xFFFFFFFF)
    with transaction.atomic():
        with connection.cursor() as cursor:
            if wait:
                cursor.execute("SELECT pg_advisory_xact_lock(%s)", [key])
                locked = True
            else:
                cursor.execute("SELECT pg_try_advisory_xact_lock(%s)", [key])
                locked = bool(cursor.fetchone()[0])
        yield locked


# --------------------------------------------------------------------------- #
# Putting runs into the queue, cancelling them
# --------------------------------------------------------------------------- #


def waiting_count(user) -> int:
    return TrainingRun.objects.filter(owner=user, status=QUEUED).count()


def cancel(run: TrainingRun) -> bool:
    """End a waiting run as stopped; it used no compute. False if it is not waiting any more
    (a worker took it meanwhile: stop it like a running training)."""
    now = timezone.now()
    updated = TrainingRun.objects.filter(pk=run.pk, status=QUEUED).update(
        status="stopped",
        message=CANCELLED_MESSAGE,
        stop_requested_at=now,
        stop_reason="user",
        finished_at=now,
        size_bytes=0,
    )
    if updated:
        run.status = "stopped"
        run.message = CANCELLED_MESSAGE
        run.stop_requested_at = run.finished_at = now
        run.stop_reason = "user"
        run.size_bytes = 0
    return bool(updated)


def fail_waiting(run: TrainingRun, message: str, stop_reason: str = "") -> bool:
    """End a waiting run that cannot start (its budget is gone, its dataset was deleted)."""
    now = timezone.now()
    updated = TrainingRun.objects.filter(pk=run.pk, status=QUEUED).update(
        status="failed", message=message[:2000], finished_at=now, stop_reason=stop_reason
    )
    if updated:
        run.status, run.message, run.finished_at = "failed", message[:2000], now
        run.stop_reason = stop_reason
    return bool(updated)


# --------------------------------------------------------------------------- #
# Claiming
# --------------------------------------------------------------------------- #


def _lock_run(run_id: Any) -> TrainingRun | None:
    """The waiting run, locked until the transaction ends; None if it does not wait any more
    or another worker is looking at it right now."""
    runs = TrainingRun.objects.filter(pk=run_id, status=QUEUED)
    if connection.features.has_select_for_update_skip_locked:
        runs = runs.select_for_update(skip_locked=True, of=("self",))
    return runs.select_related("owner", "api_token", "connection__client").first()


def _take(run_id: Any, worker: TrainingWorker) -> tuple[TrainingRun, budgets.RunLimit] | None:
    with transaction.atomic():
        run = _lock_run(run_id)
        if run is None:
            return None
        with account_lock(run.owner_id, wait=False) as mine:
            if not mine:
                return None  # another worker decides about this account right now
            account = account_of(run.owner)
            running = TrainingRun.objects.filter(
                owner_id=run.owner_id, status__in=RUNNING_STATUSES
            ).count()
            if account.concurrent is not None and running >= account.concurrent:
                return None  # keeps its place until one of the account's trainings ends
            # What its budgets allow now: it may be less than when the run was queued (or
            # more), and nothing if they were used up or lowered meanwhile.
            try:
                budgets.check_storage(run.owner)
                limit = budgets.run_limit(Actor.of_run(run), exclude=run.pk)
            except budgets.BudgetError as exc:
                reason = "budget" if exc.data.get("code") == "budget_exceeded" else ""
                fail_waiting(run, f"Not started: {exc.message}", reason)
                logger.info("Training run %s not started: %s", run.pk, exc.message)
                return None
            now = timezone.now()
            updated = TrainingRun.objects.filter(pk=run.pk, status=QUEUED).update(
                status="starting",
                message="Starting …",
                worker=worker,
                cpu_limit_seconds=limit.seconds,
                pid=None,
                started_at=None,
            )
            if not updated:  # without row locks (SQLite): someone else was faster
                return None
            TrainingAttempt.objects.create(
                run=run,
                worker=worker,
                plan=account.plan.key,
                priority=account.priority,
                queued_at=run.queued_at or run.created_at,
                started_at=now,
            )
            run.status, run.message, run.worker = "starting", "Starting …", worker
            run.cpu_limit_seconds, run.pid, run.started_at = limit.seconds, None, None
            return run, limit


def claim(worker: TrainingWorker) -> tuple[TrainingRun, budgets.RunLimit] | None:
    """Take the next run of the queue for ``worker``: ``(run, the CPU time it may use)``, or
    None if no run can start now. The run is ``starting`` and belongs to the worker, which
    then launches it (training_service.start_claimed). Safe with several workers."""
    from . import training_service as training

    # Most rounds of a worker find nothing: one small query then.
    if not TrainingRun.objects.filter(status=QUEUED).exists():
        return None
    for entry in snapshot().entries:
        if entry.blocked:
            continue
        if _postgres():
            taken = _take(entry.id, worker)
        else:
            with training.start_lock():
                taken = _take(entry.id, worker)
        if taken is not None:
            return taken
    return None


# --------------------------------------------------------------------------- #
# Attempts
# --------------------------------------------------------------------------- #


def close_attempt(run_id: Any, outcome: str, now: datetime | None = None) -> None:
    """Note how the attempt of a run that a worker was training ended."""
    TrainingAttempt.objects.filter(run_id=run_id, ended_at__isnull=True).update(
        ended_at=now or timezone.now(), outcome=outcome
    )


# --------------------------------------------------------------------------- #
# Runs of workers that restarted or died
# --------------------------------------------------------------------------- #


def _owned(run: TrainingRun):
    """The run as long as it still trains on the worker ``run`` was loaded with."""
    runs = TrainingRun.objects.filter(pk=run.pk, status__in=RUNNING_STATUSES)
    if run.worker_id is None:
        return runs.filter(worker__isnull=True)
    return runs.filter(worker_id=run.worker_id)


def release(run: TrainingRun, lost: bool = True) -> str:
    """Decide what happens to a run whose worker does not train it any more: the worker
    died (``lost``) or was asked to stop and gave the run back. Returns what it did:
    ``requeued``, an end status, or "" if the run was not this worker's any more.

    A run that had finished, or was asked to stop, ends that way. Otherwise it goes back
    into the queue, in its old place, and starts from the beginning: once after a lost
    worker, at most BRICKDNN_WORKER_MAX_RESTARTS times in all; beyond that it fails. The CPU
    time the interrupted attempt reported is counted (once: see budgets.record_run)."""
    from . import training_service as training

    directory = training.run_dir(run.pk)
    state = training.read_run_state(run)
    measured = budgets.measured_cpu(run, state) if state is not None else None
    now = timezone.now()

    def end(outcome: str) -> str:
        """Store the end that training_service worked out for ``run``."""
        if measured is not None:
            budgets.record_run(run, measured, final=True)
        if run.size_bytes is None:
            run.size_bytes = budgets.folder_size(directory)
        fields = {name: getattr(run, name) for name in training.SYNC_FIELDS}
        if not _owned(run).update(**fields):
            return ""
        close_attempt(run.pk, outcome, now)
        return run.status

    if state is not None and state.get("status") in TERMINAL_STATUSES:
        training.apply_state(run, state)  # the runner finished before its worker went away
        return end(run.status)
    if run.stop_requested_at is not None:
        training.mark_stopped(run)
        return end("stopped")
    earlier_losses = TrainingAttempt.objects.filter(run=run, outcome="lost").count()
    if (lost and earlier_losses >= 1) or run.restarts >= settings.WORKER_MAX_RESTARTS:
        if lost and earlier_losses >= 1:
            message = LOST_TWICE_MESSAGE
        else:
            message = (
                f"This training was interrupted {run.restarts + 1} times by restarts of the "
                "training worker and was not started again. Start it again."
            )
        training.mark_finished(run, "failed", message)
        return end("failed")

    if measured is not None:
        budgets.record_run(run, measured, final=True)
    message = LOST_MESSAGE if lost else RESTARTED_MESSAGE
    updated = _owned(run).update(
        status=QUEUED,
        message=message,
        worker=None,
        pid=None,
        started_at=None,
        queued_at=now,
        restarts=F("restarts") + 1,
        epochs_done=0,
        metrics=None,
        best=None,
        trials_done=0,
        best_trial=None,
        has_weights=False,
    )
    if not updated:
        return ""
    close_attempt(run.pk, "lost" if lost else "requeued", now)
    # The progress of the interrupted attempt is not shown while the run waits again; the
    # worker that takes it clears the rest of the directory (start_claimed).
    for name in (STATE_FILE, STOP_FILE):
        with contextlib.suppress(OSError):
            (directory / name).unlink()
    run.refresh_from_db()
    logger.warning("Training run %s goes back into the queue: %s", run.pk, message)
    return "requeued"


def lost_runs(now: datetime | None = None):
    """Runs that train according to the database, but on no living worker: their worker
    stopped or sends no heartbeat any more, or (after the queue was switched on) they were
    started by the web server itself before."""
    return TrainingRun.objects.filter(status__in=RUNNING_STATUSES).filter(
        Q(worker__isnull=True)
        | Q(worker__stopped_at__isnull=False)
        | Q(worker__heartbeat_at__lt=_cutoff(now))
    )


def recover(now: datetime | None = None) -> list[tuple[TrainingRun, str]]:
    """Find the workers that died and :func:`release` their runs. Called by every worker
    now and then, and by the web server when it looks at a run of a dead worker."""
    if not enabled():
        return []
    TrainingWorker.objects.filter(stopped_at__isnull=True, heartbeat_at__lt=_cutoff(now)).update(
        stopped_at=F("heartbeat_at")
    )
    done = []
    for run in lost_runs(now).select_related("owner", "worker", "api_token", "connection__client"):
        outcome = release(run, lost=True)
        if outcome:
            done.append((run, outcome))
    return done


def prune(now: datetime | None = None) -> int:
    """Delete rows of attempts and workers that ended more than KEEP_DAYS days ago (the runs
    such a worker trained stay; they only forget which worker it was)."""
    cutoff = (now or timezone.now()) - timedelta(days=KEEP_DAYS)
    attempts = TrainingAttempt.objects.filter(ended_at__lt=cutoff).delete()[1]
    workers = TrainingWorker.objects.filter(stopped_at__lt=cutoff).delete()[1]
    return attempts.get(TrainingAttempt._meta.label, 0) + workers.get(TrainingWorker._meta.label, 0)


# --------------------------------------------------------------------------- #
# For the operator: overview and statistics
# --------------------------------------------------------------------------- #


def duration(seconds: float | None) -> str:
    """``42 s``, ``3.5 min`` or ``2.1 h``; ``-`` for nothing."""
    if seconds is None:
        return "-"
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 90 * 60:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


def _percentile(values: list[float], share: float) -> float | None:
    """The value below which ``share`` of the sorted ``values`` lie (nearest rank)."""
    if not values:
        return None
    return values[min(len(values) - 1, max(0, math.ceil(share * len(values)) - 1))]


def stats(days: int = 7, now: datetime | None = None) -> dict[str, Any]:
    """How the queue did in the last ``days`` days: how long runs waited for a worker
    (median and 95th percentile, in seconds) and how busy the workers were (the time their
    slots trained / the time the slots were there)."""
    now = now or timezone.now()
    since = now - timedelta(days=days)
    started = list(
        TrainingAttempt.objects.filter(started_at__gte=since, started_at__lte=now).values_list(
            "queued_at", "started_at", "plan"
        )
    )
    waits = sorted(max(0.0, (start - queued).total_seconds()) for queued, start, _ in started)
    by_plan: dict[str, list[float]] = {}
    for queued, start, plan in started:
        by_plan.setdefault(plan, []).append(max(0.0, (start - queued).total_seconds()))

    busy = 0.0
    overlapping = TrainingAttempt.objects.filter(started_at__lt=now).filter(
        Q(ended_at__isnull=True) | Q(ended_at__gte=since)
    )
    for start, end in overlapping.values_list("started_at", "ended_at"):
        busy += max(0.0, (min(end or now, now) - max(start, since)).total_seconds())
    available = 0.0
    cutoff = _cutoff(now)
    workers = TrainingWorker.objects.filter(started_at__lt=now).filter(
        Q(stopped_at__isnull=True) | Q(stopped_at__gte=since)
    )
    for worker in workers:
        end = worker.stopped_at
        if end is None:  # still there, or dead and not noticed yet
            end = now if worker.heartbeat_at >= cutoff else worker.heartbeat_at
        seconds = (min(end, now) - max(worker.started_at, since)).total_seconds()
        available += max(0.0, seconds) * worker.slots
    plans = budgets.plans()
    return {
        "days": days,
        "started": len(waits),
        "median_wait_seconds": _percentile(waits, 0.5),
        "p95_wait_seconds": _percentile(waits, 0.95),
        "max_wait_seconds": waits[-1] if waits else None,
        "by_plan": [
            {
                "plan": plans[key].name if key in plans else (key or "-"),
                "started": len(values),
                "median_wait_seconds": _percentile(sorted(values), 0.5),
                "p95_wait_seconds": _percentile(sorted(values), 0.95),
            }
            for key, values in sorted(by_plan.items(), key=lambda item: -len(item[1]))
        ],
        "busy_slot_seconds": busy,
        "available_slot_seconds": available,
        # Above 1 for a moment is possible (a run that ends while its worker stops).
        "utilisation": min(1.0, busy / available) if available > 0 else None,
    }


def _plan_name(account: Account) -> str:
    return "unlimited" if account.unlimited else account.plan.name


def overview(now: datetime | None = None) -> dict[str, Any]:
    """Everything the operator's queue overview shows (Django admin and
    ``manage.py training_queue``)."""
    now = now or timezone.now()
    shot = snapshot(now)
    jobs: dict[int, list[TrainingRun]] = {}
    for run in shot.running:
        jobs.setdefault(run.worker_id, []).append(run)
    # The workers that are there, the ones that died unnoticed and the ones that still have
    # runs (the Django admin lists all of them, also the ones that stopped).
    known = TrainingWorker.objects.filter(
        Q(stopped_at__isnull=True) | Q(pk__in=[pk for pk in jobs if pk is not None])
    ).order_by("-started_at", "-pk")[:20]
    alive = {worker.pk for worker in shot.workers}
    return {
        "enabled": enabled(),
        "now": now,
        "online": shot.online,
        "slots": shot.slots,
        "message": None if shot.online or not enabled() else NO_WORKER_MESSAGE,
        "workers": [
            {
                "id": worker.pk,
                "name": worker.name,
                "hostname": worker.hostname,
                "state": "online"
                if worker.pk in alive
                else "stopped"
                if worker.stopped_at is not None
                else "no heartbeat",
                "alive": worker.pk in alive,
                "heartbeat_seconds": max(0.0, (now - worker.heartbeat_at).total_seconds()),
                "slots": worker.slots,
                "threads": worker.threads,
                "memory_mb": worker.memory_mb,
                "started_at": worker.started_at,
                "stopped_at": worker.stopped_at,
                "jobs": [run.name for run in jobs.get(worker.pk, [])],
            }
            for worker in known
        ],
        "running": [
            {
                "id": str(run.pk),
                "name": run.name,
                "account": run.owner.get_username(),
                "plan": _plan_name(shot.accounts[run.owner_id]),
                "priority": shot.accounts[run.owner_id].priority,
                "status": run.status,
                "worker": run.worker.name if run.worker is not None else None,
                "worker_alive": run.worker_id in alive,
                "running_seconds": max(
                    0.0, (now - (run.started_at or run.created_at)).total_seconds()
                ),
                "cpu_minutes": round((run.cpu_seconds or 0.0) / 60, 2),
                "restarts": run.restarts,
            }
            for run in shot.running
        ],
        "waiting": [
            {
                "position": entry.position,
                "id": entry.id,
                "name": entry.run.name if entry.run else "",
                "account": entry.run.owner.get_username() if entry.run else "",
                "plan": _plan_name(entry.account),
                "priority": entry.account.priority,
                "waiting_seconds": max(
                    0.0,
                    (now - (entry.run.queued_at or entry.created_at)).total_seconds()
                    if entry.run
                    else 0.0,
                ),
                "estimated_wait_seconds": entry.wait if shot.online else None,
                "blocked": entry.blocked,
                "restarts": entry.run.restarts if entry.run else 0,
            }
            for entry in shot.entries
        ],
        "stats": stats(now=now),
    }


def remove_run_files(run_id: Any) -> None:
    """Delete the directory of a run that does not exist any more (a worker, after it
    stopped the training of a deleted run)."""
    from . import training_service as training

    shutil.rmtree(training.run_dir(run_id), ignore_errors=True)
