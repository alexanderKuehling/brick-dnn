"""The training queue and the worker (training_queue.py, training_worker.py).

With ``BRICKDNN_TRAINING_WORKER`` the web server only puts runs into the queue; a worker
takes them, trains them and looks after their processes. In these tests the API client is
the web server and a :class:`Worker` whose rounds the test calls one by one is the worker
(in the same process, but only the worker's rounds look after the training processes, as in
two containers). The runner is the controlled fake runner of test_budgets: it reports the
CPU time written into ``cpu.txt``, follows stop requests and completes once ``finish``
appears. The plans are brickdnn.com's (test_plans).
"""

# ruff: noqa: F811 (fixtures are imported from other test modules)

from __future__ import annotations

import os
import signal
import sys
import threading
import time
from datetime import timedelta
from io import StringIO

import pytest
from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.db import connection
from django.utils import timezone
from rest_framework.test import APIClient

from brick_dnn_core import budgets, training_service, training_worker
from brick_dnn_core import training_queue as queue
from brick_dnn_core.models import (
    TrainingAttempt,
    TrainingRun,
    TrainingWorker,
    UsageRecord,
    UserProfile,
)
from brick_dnn_core.training_worker import Worker
from brickdnn.training.state import read_job, read_state

from .test_budgets import (  # noqa: F401 (fixtures)
    LONG,
    QUICK,
    ada,
    controlled,
    ledger,
    limits,
    run_dir,
    spend,
    start,
    started,
    token_client,
    use_cpu,
    web,
)
from .test_plans import site  # noqa: F401
from .test_training_api import (  # noqa: F401
    DEMO_GRAPH,
    RANDOM,
    iris_csv,
    training_dir,
    upload,
    wait_for_state,
)

pytestmark = pytest.mark.django_db

User = get_user_model()


# --------------------------------------------------------------------------- #
# Fixtures and helpers
# --------------------------------------------------------------------------- #


@pytest.fixture
def queued(controlled, limits):
    """Worker mode with the controlled runner; returns the training directory."""
    limits.TRAINING_WORKER = True
    limits.WORKER_NAME = ""
    limits.WORKER_SLOTS = 0
    limits.WORKER_TIMEOUT = 60
    limits.WORKER_MAX_RESTARTS = 3
    limits.WORKER_DRAIN_SECONDS = 0
    limits.QUEUE_MAX_WAITING = None
    return controlled


@pytest.fixture
def make_worker(queued):
    """Start workers; what they leave running is killed by the training_dir fixture."""

    def make(name: str = "worker-1", slots: int = 1, **options) -> Worker:
        worker = Worker(name=name, slots=slots, **options)
        worker.start()
        return worker

    return make


def account(name: str, plan: str | None = None, **fields) -> tuple[APIClient, object]:
    user = User.objects.create_user(name, password="correct-horse-battery", **fields)
    if plan is not None:
        budgets.set_plan(user, plan)
    client = APIClient()
    client.force_authenticate(user)
    return client, user


def step_until(worker: Worker, condition, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while True:
        worker.step()
        if condition():
            return
        assert time.monotonic() < deadline, "the worker did not get there"
        time.sleep(0.02)


def complete(worker: Worker, training_dir, run_id) -> None:
    """Let the controlled runner of a run finish and the worker notice it."""
    run_id = str(run_id)
    wait_for_state(training_dir, run_id)
    (run_dir(training_dir, run_id) / "finish").touch()
    step_until(worker, lambda: run_id not in worker.jobs and not worker.ending)


def names(shot: queue.Snapshot | None = None) -> list[str]:
    return [entry.run.name for entry in (shot or queue.snapshot()).entries]


def get(client, run_id) -> dict:
    return client.get(f"/api/training/runs/{run_id}/").data


def beat_long_ago(worker: Worker, seconds: int = 600) -> None:
    """As if the worker had died that long ago (it sends no heartbeat any more)."""
    TrainingWorker.objects.filter(pk=worker.row.pk).update(
        heartbeat_at=timezone.now() - timedelta(seconds=seconds)
    )


def kill(run_id) -> None:
    """Kill the training process of a run, as the end of its container would."""
    process = training_service._PROCESSES.pop(str(run_id))
    os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=5)


# --------------------------------------------------------------------------- #
# The mode switch
# --------------------------------------------------------------------------- #


def test_without_a_worker_the_web_server_trains_itself(web, ada, controlled):
    """The default: nothing is queued, a second start is refused while the first trains."""
    controlled_settings = training_service.settings
    assert controlled_settings.TRAINING_WORKER is False
    controlled_settings.TRAINING_MAX_RUNS = 1
    info = web.get("/api/training/").data
    assert info["queue"] is None and info["max_concurrent_runs"] == 1
    response = start(web)
    assert response.status_code == 201
    assert response.data["status"] == "starting" and response.data["queue"] is None
    run_id = response.data["id"]
    assert run_id in training_service._PROCESSES  # started by the web server itself
    assert start(web).status_code == 409
    run = TrainingRun.objects.get(pk=run_id)
    assert (run.worker, run.queued_at, run.restarts) == (None, None, 0)
    assert not TrainingWorker.objects.exists() and not TrainingAttempt.objects.exists()
    check = web.post(
        "/api/training/check/", {"graph": DEMO_GRAPH, "dataset": RANDOM}, format="json"
    ).data
    assert check["queue"] is None


def test_a_start_only_queues_the_run(web, ada, queued):
    response = start(web)
    assert response.status_code == 201, response.data
    data = response.data
    assert data["status"] == "queued" and data["message"] == "Waiting for a free training slot."
    assert data["state"] is None and data["has_weights"] is False
    assert not training_service._PROCESSES  # the web server starts no process
    assert not run_dir(queued, data["id"]).exists()
    run = TrainingRun.objects.get(pk=data["id"])
    assert run.queued_at is not None and run.worker is None and run.pid is None
    # No worker has ever said hello: the answer says so instead of promising a start.
    assert data["queue"] == {
        "position": 1,
        "waiting": 1,
        "estimated_wait_seconds": None,
        "estimated_start_at": None,
        "worker_online": False,
        "blocked_by": None,
        "priority": 0,
        "hint": None,
    }
    info = web.get("/api/training/").data["queue"]
    assert info["online"] is False and info["workers"] == 0 and info["waiting"] == 1
    assert info["message"].startswith("No training worker is running")
    # It is listed and shown like any other run.
    listed = web.get("/api/training/runs/").data
    assert [(r["status"], r["queue"]["position"]) for r in listed] == [("queued", 1)]
    assert get(web, data["id"])["queue"]["worker_online"] is False


def test_a_worker_trains_what_is_queued(web, ada, queued, make_worker):
    run_id = started(web)
    worker = make_worker()
    info = web.get("/api/training/").data
    assert info["queue"]["online"] is True and info["queue"]["slots"] == 1
    assert info["queue"]["message"] is None and info["max_concurrent_runs"] == 1
    assert get(web, run_id)["queue"]["estimated_wait_seconds"] == 0

    worker.step()
    assert list(worker.jobs) == [run_id]
    run = TrainingRun.objects.get(pk=run_id)
    assert run.status == "starting" and run.worker == worker.row and run.pid
    job = read_job(run_dir(queued, run_id))
    assert job["graph"] == DEMO_GRAPH and job["config"]["epochs"] == 1
    assert job["dataset"]["kind"] == "random" and job["limits"] is None
    wait_for_state(queued, run_id)
    data = use_cpu(web, queued, run_id, 12.0)  # the web server reads the progress
    assert data["status"] == "running" and data["queue"] is None
    assert data["state"]["message"] == "Epoch 1 of 1"
    assert ledger(ada) == 12.0

    complete(worker, queued, run_id)
    data = get(web, run_id)
    assert data["status"] == "completed" and data["has_weights"] is True
    attempt = TrainingAttempt.objects.get()
    assert (attempt.run_id, attempt.worker, attempt.outcome) == (run.pk, worker.row, "completed")
    assert attempt.queued_at == run.queued_at and attempt.ended_at is not None
    assert web.get(f"/api/training/runs/{run_id}/weights/").status_code == 200
    assert web.get("/api/training/").data["queue"]["running"] == 0


def test_the_web_server_does_not_judge_the_process(web, ada, queued, make_worker):
    """A process that died is the worker's to notice: the web server cannot see it."""
    run_id = started(web)
    worker = make_worker()
    worker.step()
    wait_for_state(queued, run_id)
    kill(run_id)
    assert get(web, run_id)["status"] == "running"
    worker.step()
    data = get(web, run_id)
    assert data["status"] == "failed"
    assert data["message"].startswith("The training process stopped unexpectedly.")
    assert TrainingAttempt.objects.get().outcome == "failed"


def test_switching_the_queue_on_and_off_leaves_no_run_behind(web, ada, queued, limits):
    run_id = started(web)
    # Off again while a run waits: nobody would start it.
    limits.TRAINING_WORKER = False
    data = get(web, run_id)
    assert data["status"] == "failed" and "switched off" in data["message"]
    # On while the web server trains a run itself: after the restart that the change
    # needs, the run is nobody's and goes back into the queue.
    limits.TRAINING_MAX_RUNS = 5
    own = started(web)
    wait_for_state(queued, own)
    kill(own)
    limits.TRAINING_WORKER = True
    data = get(web, own)
    assert data["status"] == "queued" and "stopped unexpectedly" in data["message"]
    assert TrainingRun.objects.get(pk=own).restarts == 1


# --------------------------------------------------------------------------- #
# The order
# --------------------------------------------------------------------------- #


def test_paid_plans_start_first_then_who_waits_longest(queued, site, make_worker):
    free_a, _ = account("free-a")
    free_b, _ = account("free-b")
    starter, _ = account("starter", "starter")
    pro, _ = account("pro", "pro")
    staff, _ = account("staff", is_staff=True)
    started(free_a, name="free a")
    started(free_b, name="free b")
    started(starter, name="starter")
    started(pro, name="pro")
    started(staff, name="staff")
    # Accounts without limits (staff) come before every plan.
    assert names() == ["staff", "pro", "starter", "free a", "free b"]
    assert [e.account.priority for e in queue.snapshot().entries] == [4, 3, 1, 0, 0]

    worker = make_worker()
    order = []
    for _ in range(5):
        worker.step()
        (run_id,) = worker.jobs
        order.append(TrainingRun.objects.get(pk=run_id).name)
        complete(worker, queued, run_id)
    assert order == ["staff", "pro", "starter", "free a", "free b"]
    assert list(TrainingAttempt.objects.order_by("started_at").values_list("plan", "priority")) == [
        ("free", 4),
        ("pro", 3),
        ("starter", 1),
        ("free", 0),
        ("free", 0),
    ]


def test_a_plan_change_moves_the_waiting_runs(queued, site):
    first, _ = account("first")
    second, user = account("second")
    started(first, name="first")
    started(second, name="second")
    assert names() == ["first", "second"]
    budgets.set_plan(user, "plus")
    assert names() == ["second", "first"]
    data = second.get("/api/training/runs/").data[0]
    assert data["queue"]["position"] == 1 and data["queue"]["priority"] == 2


def test_accounts_of_one_priority_take_turns(queued, limits, make_worker):
    """One account with many waiting runs cannot keep the others waiting."""
    busy, _ = account("busy")
    other, _ = account("other")
    late, _ = account("late")
    for number in (1, 2, 3):
        started(busy, name=f"busy {number}")
    started(other, name="other 1")
    started(other, name="other 2")
    started(late, name="late 1")
    assert names() == ["busy 1", "other 1", "late 1", "busy 2", "other 2", "busy 3"]

    # A training an account has running counts like a turn it already had.
    worker = make_worker()
    worker.step()
    assert TrainingRun.objects.get(pk=next(iter(worker.jobs))).name == "busy 1"
    assert names() == ["other 1", "late 1", "busy 2", "other 2", "busy 3"]
    newcomer, _ = account("newcomer")
    started(newcomer, name="newcomer 1")
    assert names()[:4] == ["other 1", "late 1", "newcomer 1", "busy 2"]
    positions = {
        r["name"]: r["queue"]["position"] for r in busy.get("/api/training/runs/").data[:2]
    }
    assert positions == {"busy 3": 6, "busy 2": 4}


# --------------------------------------------------------------------------- #
# Limits per account
# --------------------------------------------------------------------------- #


def test_trainings_at_a_time_are_the_plans(queued, site, make_worker):
    """A plan's trainings at a time limit what runs, not what waits: the next run of the
    account keeps its place and is passed over until the first one ends."""
    free, _ = account("free")
    pro, _ = account("pro", "pro")
    first = started(free, name="free 1")
    second = started(free, name="free 2")
    worker = make_worker(slots=3)
    worker.step()
    assert list(worker.jobs) == [first]  # two free slots, but Free trains one at a time
    waiting = get(free, second)["queue"]
    assert waiting["blocked_by"] == "account" and waiting["position"] == 1
    # Other accounts are not held up by it; Pro trains two at a time.
    for number in (1, 2, 3):
        started(pro, name=f"pro {number}")
    worker.step()
    running = TrainingRun.objects.filter(status__in=("starting", "preparing", "running"))
    assert sorted(run.name for run in running) == ["free 1", "pro 1", "pro 2"]
    assert names() == ["pro 3", "free 2"]
    complete(worker, queued, first)
    worker.step()
    assert second in worker.jobs


def test_the_number_of_waiting_trainings_is_limited(queued, site, limits):
    paying, _ = account("paying", "starter")
    for _ in range(3):  # the default on a server with plans
        assert start(paying).status_code == 201
    response = start(paying)
    assert response.status_code == 409
    assert response.data["detail"] == (
        "Your plan (Starter) lets 3 trainings of an account wait in the queue. Wait until "
        "one of yours has started, or cancel one."
    )
    # Cancelling one makes room.
    run_id = paying.get("/api/training/runs/").data[0]["id"]
    assert paying.post(f"/api/training/runs/{run_id}/stop/").data["status"] == "stopped"
    assert start(paying).status_code == 201
    # The server's default and a plan's own number.
    limits.QUEUE_MAX_WAITING = 1
    other, user = account("other")
    assert start(other).status_code == 201
    assert "lets 1 training of an account wait" in start(other).data["detail"]
    limits.PLANS = {**limits.PLANS, "pro": {**limits.PLANS["pro"], "queued_trainings": 2}}
    budgets.set_plan(user, "pro")
    assert budgets.limits_for(user).queued_trainings == 2
    assert start(other).status_code == 201
    assert start(other).status_code == 409
    # Accounts without limits, and a server of your own (no plans), have none.
    staff, _ = account("staff", is_staff=True)
    assert [start(staff).status_code for _ in range(4)] == [201] * 4
    limits.PLANS, limits.QUEUE_MAX_WAITING = None, None
    assert budgets.limits_for(user).queued_trainings is None
    limits.QUEUE_MAX_WAITING = 0
    assert budgets.limits_for(user).queued_trainings is None


# --------------------------------------------------------------------------- #
# Budgets
# --------------------------------------------------------------------------- #


def test_waiting_runs_reserve_their_budget(queued, limits):
    """Two waiting runs cannot both count on the same rest of a budget."""
    limits.BUDGET_CPU_MINUTES_PER_MONTH = 20
    limits.BUDGET_CPU_MINUTES_PER_RUN = 15
    web, ada = account("ada")
    first = TrainingRun.objects.get(pk=started(web))
    second = TrainingRun.objects.get(pk=started(web))
    assert (first.cpu_limit_seconds, second.cpu_limit_seconds) == (900, 300)
    response = start(web)
    assert response.status_code == 403 and response.data["code"] == "budget_exceeded"
    assert "is reserved by trainings that are still running" in response.data["detail"]
    usage = web.get("/api/account/usage/").data["compute"]
    assert usage["reserved_by_active_runs"] == 20
    # A cancelled run gives its share back.
    web.post(f"/api/training/runs/{first.pk}/stop/")
    assert start(web).status_code == 201


def test_the_budget_is_checked_again_when_the_run_starts(queued, limits, make_worker):
    limits.BUDGET_CPU_MINUTES_PER_MONTH = 20
    limits.BUDGET_CPU_MINUTES_PER_RUN = 15
    web, ada = account("ada")
    first, second, third = started(web), started(web), None
    assert TrainingRun.objects.get(pk=first).cpu_limit_seconds == 900
    # Other usage of the account (a prediction, another server) eats into the budget
    # while the runs wait: the first gets what is left then, less than it was promised.
    spend(ada, 600)
    worker = make_worker()
    worker.step()
    assert list(worker.jobs) == [first]
    run = TrainingRun.objects.get(pk=first)
    assert run.cpu_limit_seconds == 300  # 20 − 10 minutes, minus the 5 of the second
    limits_of_job = read_job(run_dir(queued, first))["limits"]
    assert limits_of_job["cpu_seconds"] == 300 and limits_of_job["reason"] == "budget"
    # The runner stops at that limit and the budget is used up: the second cannot start.
    wait_for_state(queued, first)
    worker.stopping = True  # takes nothing new for a moment
    use_cpu(web, queued, first, 300.0)
    step_until(worker, lambda: first not in worker.jobs and not worker.ending)
    assert get(web, first)["usage"]["stop_reason"] == "budget"
    spend(ada, 300)
    worker.stopping = False
    worker.step()
    data = get(web, second)
    assert data["status"] == "failed" and not worker.jobs
    assert data["message"].startswith(
        "Not started: The monthly compute allowance of this account is used up: 20.0 CPU "
        "minutes of 20.0 CPU minutes this month."
    )
    assert data["usage"]["stop_reason"] == "budget" and data["usage"]["cpu_minutes"] == 0
    assert TrainingAttempt.objects.count() == 1  # it never started
    assert third is None


def test_a_budget_stops_a_run_through_the_worker(queued, limits, make_worker):
    """The worker counts and enforces also when nobody looks at the run in the web app."""
    web, ada = account("ada")
    run_id = started(web)
    worker = make_worker()
    worker.step()
    wait_for_state(queued, run_id)
    UserProfile.objects.update_or_create(user=ada, defaults={"budget_cpu_minutes": 0.5})
    (run_dir(queued, run_id) / "cpu.txt").write_text("31.0")
    step_until(worker, lambda: TrainingRun.objects.get(pk=run_id).stop_requested_at is not None)
    assert ledger(ada) == 31.0
    step_until(worker, lambda: run_id not in worker.jobs)
    data = get(web, run_id)
    assert (data["status"], data["usage"]["stop_reason"]) == ("stopped", "budget")
    assert data["message"].startswith("Stopped: The monthly budget set for this account is used up")
    assert data["has_weights"] is True
    assert TrainingAttempt.objects.get().outcome == "stopped"


def test_a_budget_stop_asked_by_the_web_server_reaches_the_runner(queued, limits, make_worker):
    web, ada = account("ada")
    run_id = started(web)
    worker = make_worker()
    worker.step()
    wait_for_state(queued, run_id)
    UserProfile.objects.update_or_create(user=ada, defaults={"cpu_minutes_per_run": 0.2})
    data = use_cpu(web, queued, run_id, 13.0)  # the web server looks: over the limit
    assert data["stop_requested"] is True
    step_until(worker, lambda: run_id not in worker.jobs)
    data = get(web, run_id)
    assert (data["status"], data["usage"]["stop_reason"]) == ("stopped", "run_limit")
    assert "the most one training may use" in data["message"]


# --------------------------------------------------------------------------- #
# Claiming with several workers
# --------------------------------------------------------------------------- #


def test_two_workers_never_take_the_same_run(queued, site, make_worker):
    accounts = [account(f"user-{n}", "pro" if n % 2 else None) for n in range(4)]
    for client, _user in accounts:
        started(client)
        started(client)
    one, two = make_worker("one", slots=3), make_worker("two", slots=3)
    taken = []
    for worker in (one, two, one, two, one, two, one, two):
        claim = queue.claim(worker.row)
        if claim is not None:
            taken.append((claim[0].pk, worker.name, claim[0].owner.username))
    # Free trains one at a time and Pro two, whichever worker asks.
    assert len(taken) == len({run_id for run_id, _, _ in taken}) == 6
    per_account = {
        name: sum(1 for _, _, owner in taken if owner == name)
        for name in ("user-0", "user-1", "user-2", "user-3")
    }
    assert per_account == {"user-0": 1, "user-1": 2, "user-2": 1, "user-3": 2}
    assert TrainingRun.objects.filter(status="queued").count() == 2
    assert TrainingAttempt.objects.count() == 6
    # A worker that looked at the queue a moment too early gets nothing for a taken run.
    run_id = taken[0][0]
    assert queue._take(run_id, two.row) is None
    assert TrainingRun.objects.get(pk=run_id).worker.name == taken[0][1]


@pytest.mark.django_db(transaction=True)
def test_workers_claiming_at_the_same_time(queued, site):
    """Real concurrency: several connections claim at once (PostgreSQL: row locks with SKIP
    LOCKED and an advisory lock per account)."""
    if connection.vendor != "postgresql":
        pytest.skip("SQLite serialises claims with a file lock; see the test above")
    users = [account(f"user-{n}")[0] for n in range(6)]
    for client in users:
        started(client)
        started(client)
    rows = [TrainingWorker.objects.create(name=f"w{n}", slots=20) for n in range(4)]
    taken, errors = [], []
    barrier = threading.Barrier(len(rows))

    def work(row_id: int) -> None:
        try:
            row = TrainingWorker.objects.get(pk=row_id)
            barrier.wait(timeout=10)
            for _ in range(30):
                claim = queue.claim(row)
                if claim is not None:
                    taken.append((str(claim[0].pk), claim[0].owner_id, row_id))
                else:
                    time.sleep(0.01)
        except Exception as exc:  # noqa: BLE001 (reported below)
            errors.append(exc)
        finally:
            connection.close()

    threads = [threading.Thread(target=work, args=(row.pk,)) for row in rows]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert not errors, errors
    # Free trains one run at a time: exactly one run per account was taken, each once.
    assert len(taken) == len({run_id for run_id, _, _ in taken}) == 6
    assert len({owner for _, owner, _ in taken}) == 6
    assert TrainingRun.objects.filter(status="starting").count() == 6
    assert TrainingRun.objects.filter(status="queued").count() == 6
    assert TrainingAttempt.objects.count() == 6
    for run_id, _owner, row_id in taken:
        assert TrainingRun.objects.get(pk=run_id).worker_id == row_id


# --------------------------------------------------------------------------- #
# Stopping and deleting
# --------------------------------------------------------------------------- #


def test_cancelling_a_waiting_run(web, ada, queued, make_worker):
    first, second = started(web, name="first"), started(web, name="second")
    response = web.post(f"/api/training/runs/{first}/stop/")
    assert response.status_code == 200
    data = response.data
    assert data["status"] == "stopped" and data["message"] == "Cancelled before it started."
    assert data["usage"] == {
        "cpu_minutes": 0.0,
        "limit_cpu_minutes": None,
        "estimated_cpu_minutes": data["usage"]["estimated_cpu_minutes"],
        "stop_reason": "user",
    }
    assert data["queue"] is None and data["finished_at"] is not None
    assert ledger(ada) == 0 and not UsageRecord.objects.exists()
    assert get(web, second)["queue"]["position"] == 1
    # A worker never sees it; cancelling again changes nothing.
    worker = make_worker()
    worker.step()
    assert list(worker.jobs) == [second]
    assert web.post(f"/api/training/runs/{first}/stop/").data["status"] == "stopped"
    assert not TrainingAttempt.objects.filter(run_id=first).exists()
    # Deleting a waiting run just removes it.
    third = started(web, name="third")
    assert web.delete(f"/api/training/runs/{third}/").status_code == 204
    assert names() == []


def test_stopping_a_running_training_goes_through_the_worker(web, ada, queued, make_worker):
    run_id = started(web)
    worker = make_worker()
    worker.step()
    wait_for_state(queued, run_id)
    response = web.post(f"/api/training/runs/{run_id}/stop/")
    assert response.data["stop_requested"] is True and response.data["status"] == "running"
    assert (run_dir(queued, run_id) / "STOP").exists()  # the runner follows the file
    step_until(worker, lambda: run_id not in worker.jobs)
    data = get(web, run_id)
    assert data["status"] == "stopped" and data["has_weights"] is True
    assert data["usage"]["stop_reason"] == "user"


def test_a_stop_that_is_only_in_the_database_reaches_the_runner(web, ada, queued, make_worker):
    """E.g. requested while the worker was just starting the run (no directory yet)."""
    run_id = started(web)
    worker = make_worker()
    worker.step()
    wait_for_state(queued, run_id)
    TrainingRun.objects.filter(pk=run_id).update(
        stop_requested_at=timezone.now(), stop_reason="user"
    )
    assert not (run_dir(queued, run_id) / "STOP").exists()
    step_until(worker, lambda: run_id not in worker.jobs)
    assert get(web, run_id)["status"] == "stopped"


def test_the_worker_kills_a_runner_that_ignores_the_stop(web, ada, queued, settings, make_worker):
    settings.TRAINING_RUNNER_COMMAND = [
        sys.executable,
        str(queued.parent / "fake_runner.py"),
        "hang",
    ]
    run_id = started(web)
    worker = make_worker()
    worker.step()
    process = training_service._PROCESSES[run_id]
    wait_for_state(queued, run_id)
    web.post(f"/api/training/runs/{run_id}/stop/")
    TrainingRun.objects.filter(pk=run_id).update(
        stop_requested_at=timezone.now() - timedelta(seconds=31)
    )
    # The web server cannot kill a process in another container ...
    assert get(web, run_id)["status"] == "running"
    assert process.poll() is None
    # ... the worker does.
    worker.step()
    assert process.wait(timeout=5) == -signal.SIGKILL
    data = get(web, run_id)
    assert data["status"] == "stopped" and data["message"] == "Training was stopped."


def test_deleting_a_running_training(
    web, ada, queued, settings, make_worker, django_capture_on_commit_callbacks
):
    settings.TRAINING_RUNNER_COMMAND = [
        sys.executable,
        str(queued.parent / "fake_runner.py"),
        "hang",
    ]
    run_id = started(web)
    worker = make_worker()
    worker.step()
    process = training_service._PROCESSES[run_id]
    wait_for_state(queued, run_id)
    with django_capture_on_commit_callbacks(execute=True):
        assert web.delete(f"/api/training/runs/{run_id}/").status_code == 204
    assert process.poll() is None  # the web server could only delete the row and the files
    worker.step()
    assert process.wait(timeout=5) == -signal.SIGKILL
    assert not worker.jobs and not run_dir(queued, run_id).exists()
    attempt = TrainingAttempt.objects.get()
    assert attempt.run is None and attempt.outcome == "deleted" and attempt.ended_at
    # The slot is free again.
    other = started(web)
    worker.step()
    assert list(worker.jobs) == [other]


def test_a_dataset_cannot_be_deleted_while_a_run_waits_for_it(web, ada, queued):
    dataset = upload(web, "iris.csv", iris_csv(30)).data
    graph = {
        "nodes": [
            {"id": "in", "type": "Input", "params": {"shape": [4]}},
            {"id": "out", "type": "Linear", "params": {"out_features": 3}},
        ],
        "edges": [{"source": "in", "target": "out"}],
    }
    run_id = started(web, graph=graph, dataset={"kind": "upload", "id": dataset["id"]})
    response = web.delete(f"/api/training/datasets/{dataset['id']}/")
    assert response.status_code == 409
    assert "waits in the queue" in response.data["detail"]
    web.post(f"/api/training/runs/{run_id}/stop/")
    assert web.delete(f"/api/training/datasets/{dataset['id']}/").status_code == 204


# --------------------------------------------------------------------------- #
# Workers that restart or die
# --------------------------------------------------------------------------- #


def test_a_dead_worker_is_reported_and_its_runs_start_again_once(web, ada, queued, make_worker):
    run_id = started(web, name="long")
    later = started(web, name="later")
    worker = make_worker("first")
    worker.step()
    wait_for_state(queued, run_id)
    use_cpu(web, queued, run_id, 30.0)
    assert ledger(ada) == 30.0
    # The worker dies with its training (its container is gone).
    kill(run_id)
    beat_long_ago(worker)
    info = web.get("/api/training/").data["queue"]
    assert info["online"] is False and info["message"].startswith("No training worker")
    # The next look at the run notices it: back into the queue, before the runs after it.
    data = get(web, run_id)
    assert data["status"] == "queued" and data["queue"]["position"] == 1
    assert data["queue"]["worker_online"] is False
    assert data["message"] == (
        "Waiting to start again from the beginning: the training worker stopped unexpectedly."
    )
    assert data["state"] is None and data["epochs_done"] == 0
    run = TrainingRun.objects.get(pk=run_id)
    assert (run.restarts, run.worker, run.status) == (1, None, "queued")
    assert TrainingWorker.objects.get(pk=worker.row.pk).stopped_at is not None
    assert TrainingAttempt.objects.get().outcome == "lost"
    assert names() == ["long", "later"]

    # A new worker starts it again, from the beginning.
    second = make_worker("second")
    second.step()
    assert list(second.jobs) == [run_id]
    wait_for_state(queued, run_id)
    assert "started again (restart 1)" in (run_dir(queued, run_id) / "output.log").read_text()
    # Nothing is counted twice: the new attempt only adds what it uses beyond the first.
    use_cpu(web, queued, run_id, 20.0)
    assert ledger(ada) == 30.0
    use_cpu(web, queued, run_id, 45.0)
    assert ledger(ada) == 45.0 and UsageRecord.objects.count() == 1

    # Dead again: this time the run fails, with a message that says why.
    kill(run_id)
    beat_long_ago(second)
    third = make_worker("third")
    data = get(web, run_id)
    assert data["status"] == "failed"
    assert data["message"].startswith("The training worker stopped unexpectedly twice")
    assert ledger(ada) == 45.0
    assert [a.outcome for a in TrainingAttempt.objects.order_by("started_at")] == [
        "lost",
        "failed",
    ]
    third.step()
    assert list(third.jobs) == [later]


def test_overage_is_not_billed_twice_for_a_restarted_run(queued, site, make_worker):
    web, ada = account("ada", "starter")
    UserProfile.objects.filter(user=ada).update(overage_enabled=True)
    spend(ada, 10 * 3600 - 20)  # 20 seconds of the allowance are left
    run_id = started(web)
    worker = make_worker("first")
    worker.step()
    wait_for_state(queued, run_id)
    use_cpu(web, queued, run_id, 50.0)
    record = UsageRecord.objects.get(run_id=run_id)
    assert (record.quantity, record.overage) == (50, 30)
    kill(run_id)
    beat_long_ago(worker)
    second = make_worker("second")  # finds the run of the dead worker when it starts
    assert TrainingRun.objects.get(pk=run_id).status == "queued"
    second.step()
    wait_for_state(queued, run_id)
    use_cpu(web, queued, run_id, 40.0)
    record.refresh_from_db()
    assert (record.quantity, record.overage) == (50, 30)
    use_cpu(web, queued, run_id, 80.0)
    record.refresh_from_db()
    assert (record.quantity, record.overage) == (80, 60)
    complete(second, queued, run_id)
    record.refresh_from_db()
    assert (record.quantity, record.overage) == (80, 60)
    assert TrainingRun.objects.get(pk=run_id).cpu_seconds == 80


def test_a_worker_that_is_asked_to_stop_gives_its_runs_back(web, ada, queued, make_worker):
    """A deploy restarts the worker container: its trainings go back into the queue, in
    their old place, and the next worker starts them again."""
    run_id = started(web, name="interrupted")
    worker = make_worker("worker")
    worker.step()
    process = training_service._PROCESSES[run_id]
    wait_for_state(queued, run_id)
    use_cpu(web, queued, run_id, 10.0)
    later = started(web, name="later")

    worker.stop()
    assert process.wait(timeout=5) == -signal.SIGKILL
    row = TrainingWorker.objects.get(pk=worker.row.pk)
    assert row.stopped_at is not None
    data = get(web, run_id)
    assert data["status"] == "queued" and data["queue"]["position"] == 1
    assert data["message"] == (
        "Waiting to start again from the beginning: the training worker was restarted."
    )
    assert TrainingAttempt.objects.get().outcome == "requeued"
    assert web.get("/api/training/").data["queue"]["online"] is False

    # The new container: a worker of the same name.
    again = make_worker("worker")
    again.step()
    assert list(again.jobs) == [run_id]
    complete(again, queued, run_id)
    assert get(web, run_id)["status"] == "completed"
    assert TrainingRun.objects.get(pk=run_id).restarts == 1
    assert ledger(ada) == 10.0  # the interrupted attempt, counted once
    again.step()
    assert list(again.jobs) == [later]


def test_restarts_are_limited(web, ada, queued, limits, make_worker):
    limits.WORKER_MAX_RESTARTS = 2
    run_id = started(web)
    for restart in (1, 2):
        worker = make_worker(f"worker-{restart}")
        worker.step()
        wait_for_state(queued, run_id)
        worker.stop()
        assert TrainingRun.objects.get(pk=run_id).restarts == restart
    worker = make_worker("worker-3")
    worker.step()
    wait_for_state(queued, run_id)
    worker.stop()
    data = get(web, run_id)
    assert data["status"] == "failed"
    assert data["message"] == (
        "This training was interrupted 3 times by restarts of the training worker and was "
        "not started again. Start it again."
    )


def test_a_run_that_finished_or_was_stopped_is_not_started_again(web, ada, queued, make_worker):
    done, asked = started(web, name="done"), started(web, name="asked to stop")
    worker = make_worker(slots=2)
    worker.step()
    wait_for_state(queued, done)
    wait_for_state(queued, asked)
    # One finishes just before the worker goes away, the other was asked to stop.
    (run_dir(queued, done) / "finish").touch()
    deadline = time.monotonic() + 10
    while read_state(run_dir(queued, done))["status"] != "completed":
        assert time.monotonic() < deadline
        time.sleep(0.02)
    TrainingRun.objects.filter(pk=asked).update(
        stop_requested_at=timezone.now(), stop_reason="user"
    )
    kill(asked)
    beat_long_ago(worker)
    recovered = dict((str(run.pk), outcome) for run, outcome in queue.recover())
    assert recovered == {done: "completed", asked: "stopped"}
    assert get(web, done)["has_weights"] is True
    assert get(web, asked)["message"] == "Training was stopped."
    assert not TrainingRun.objects.filter(status="queued").exists()


def test_a_worker_takes_over_a_process_that_survived_its_predecessor(web, ada, queued, make_worker):
    """Without Docker a killed worker can leave its training process behind: a worker of
    the same name on the same machine watches it again instead of starting it anew."""
    run_id = started(web)
    worker = make_worker("worker")
    worker.step()
    wait_for_state(queued, run_id)
    process = training_service._PROCESSES.pop(run_id)  # not the new worker's child
    try:
        again = make_worker("worker")
        run = TrainingRun.objects.get(pk=run_id)
        assert run.worker == again.row and run.status in ("starting", "running")
        assert run.restarts == 0 and again.jobs == {run_id: process.pid}
        assert TrainingWorker.objects.get(pk=worker.row.pk).stopped_at is not None
        (run_dir(queued, run_id) / "finish").touch()
        process.wait(timeout=10)
        step_until(again, lambda: run_id not in again.jobs)
        assert get(web, run_id)["status"] == "completed"
        assert TrainingAttempt.objects.get().worker == again.row
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)


def test_a_worker_that_was_taken_for_dead_lets_go_of_its_runs(web, ada, queued, make_worker):
    """E.g. it could not reach the database for a while: its run was started again by
    another worker meanwhile, so it must not keep training it."""
    run_id = started(web)
    worker = make_worker("slow")
    worker.step()
    process = training_service._PROCESSES[run_id]
    wait_for_state(queued, run_id)
    beat_long_ago(worker)
    queue.recover()
    assert TrainingRun.objects.get(pk=run_id).status == "queued"
    old_row = worker.row.pk
    worker._last_beat = 0  # its next round sends a heartbeat again
    worker.stopping = True  # (and takes no new run, to keep the test simple)
    worker.step()
    assert process.wait(timeout=5) == -signal.SIGKILL
    assert not worker.jobs and worker.row.pk != old_row
    assert queue.is_alive(TrainingWorker.objects.get(pk=worker.row.pk))


def test_a_stopping_worker_lets_its_trainings_finish_for_a_while(web, ada, queued, make_worker):
    """The drain time of a deploy: no new run is taken, a run that ends in time completes."""
    run_id, waiting = started(web, name="nearly done"), started(web, name="waiting")
    worker = make_worker(drain=30)
    worker.step()
    wait_for_state(queued, run_id)
    worker.stopping = True  # SIGTERM
    assert worker._finished() is False  # it waits for the running training
    complete(worker, queued, run_id)
    assert worker._finished() is True
    assert not worker.jobs  # and did not take the next one
    worker.stop()
    assert get(web, run_id)["status"] == "completed"
    assert TrainingRun.objects.get(pk=run_id).restarts == 0
    assert get(web, waiting)["status"] == "queued"
    # Without drain time (or when it is over) it does not wait.
    hurried = make_worker("hurried", drain=0)
    hurried.step()
    hurried.stopping = True
    assert hurried._finished() is True


def test_a_worker_without_the_database_stops_training_before_it_is_taken_for_dead(
    web, ada, queued, limits, make_worker
):
    run_id = started(web)
    worker = make_worker()
    worker.step()
    process = training_service._PROCESSES[run_id]
    wait_for_state(queued, run_id)
    # A short outage: the trainings go on.
    worker._database_trouble(Exception("connection lost"))
    assert process.poll() is None and list(worker.jobs) == [run_id]
    # No heartbeat for most of the timeout: the others will start its runs again soon.
    worker._last_beat = time.monotonic() - limits.WORKER_TIMEOUT
    worker._database_trouble(Exception("connection lost"))
    assert process.wait(timeout=5) == -signal.SIGKILL
    assert not worker.jobs
    # The run itself is untouched: whoever notices the missing heartbeat puts it back.
    assert TrainingRun.objects.get(pk=run_id).status in ("starting", "running")
    beat_long_ago(worker)
    assert get(web, run_id)["status"] == "queued"


def test_a_run_whose_dataset_is_gone_does_not_start(web, ada, queued, make_worker):
    import shutil

    dataset = upload(web, "iris.csv", iris_csv(30)).data
    graph = {
        "nodes": [
            {"id": "in", "type": "Input", "params": {"shape": [4]}},
            {"id": "out", "type": "Linear", "params": {"out_features": 3}},
        ],
        "edges": [{"source": "in", "target": "out"}],
    }
    run_id = started(web, graph=graph, dataset={"kind": "upload", "id": dataset["id"]})
    shutil.rmtree(queued / "datasets" / str(dataset["id"]))
    worker = make_worker()
    worker.step()
    worker.step()
    data = get(web, run_id)
    assert data["status"] == "failed" and not worker.jobs
    assert data["message"] == (
        "Not started: The dataset of this training was deleted before the training started."
    )
    assert TrainingAttempt.objects.get().outcome == "failed"


def test_slots_and_the_worker_command(web, ada, queued, settings, monkeypatch):
    monkeypatch.setattr(training_worker, "available_cpus", lambda: 4.0)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    assert training_worker.default_slots() == 1  # a training uses every CPU
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    assert training_worker.default_slots() == 2
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    assert training_worker.default_slots() == 1
    settings.WORKER_SLOTS = 5
    assert (Worker().slots, Worker(slots=2).slots) == (5, 2)
    assert Worker(name="blue").name == "blue"

    # --once: one run, then the command ends (a Kubernetes Job).
    settings.TRAINING_RUNNER_COMMAND = [
        sys.executable,
        str(queued.parent / "fake_runner.py"),
        "complete",
    ]
    first, second = started(web, name="first"), started(web, name="second")
    before = signal.getsignal(signal.SIGTERM)
    out = StringIO()
    call_command("training_worker", "--once", "--poll", "0.05", "--name", "job", stdout=out)
    assert signal.getsignal(signal.SIGTERM) is before
    assert get(web, first)["status"] == "completed"
    assert get(web, second)["status"] == "queued"
    row = TrainingWorker.objects.get(name="job")
    assert row.stopped_at is not None and row.slots == 5
    assert "Started “first” of ada" in out.getvalue() and "stopped" in out.getvalue()
    # Nothing to do: with --idle-exit it does not wait forever.
    web.post(f"/api/training/runs/{second}/stop/")
    call_command(
        "training_worker", "--once", "--poll", "0.05", "--idle-exit", "0.1", stdout=StringIO()
    )
    # Without the setting there is nothing for a worker to do.
    settings.TRAINING_WORKER = False
    out = StringIO()
    call_command("training_worker", "--once", stdout=out)
    assert "BRICKDNN_TRAINING_WORKER is off" in out.getvalue()


# --------------------------------------------------------------------------- #
# The estimated start
# --------------------------------------------------------------------------- #


def make_run(user, name, status="queued", estimate=None, limit=None, used=0.0, minutes_ago=0):
    run = TrainingRun.objects.create(
        owner=user,
        name=name,
        status=status,
        estimated_cpu_seconds=estimate,
        cpu_limit_seconds=limit,
        cpu_seconds=used,
        queued_at=timezone.now(),
    )
    when = timezone.now() - timedelta(minutes=minutes_ago)
    TrainingRun.objects.filter(pk=run.pk).update(created_at=when, queued_at=when)
    return run


def waits(shot: queue.Snapshot | None = None) -> dict[str, float | None]:
    return {e.run.name: e.wait for e in (shot or queue.snapshot()).entries}


def test_the_estimated_start(queued, site):
    one = User.objects.create_user("one")
    two = User.objects.create_user("two")
    three = User.objects.create_user("three")
    row = TrainingWorker.objects.create(name="w", slots=1, threads=1)
    running = make_run(one, "running", "running", estimate=600, used=100, minutes_ago=9)
    TrainingRun.objects.filter(pk=running.pk).update(worker=row)
    make_run(two, "a", estimate=300, minutes_ago=5)
    make_run(three, "b", estimate=900, limit=120, minutes_ago=4)  # at most its limit
    make_run(two, "c", minutes_ago=3)  # no estimate: ten minutes
    make_run(one, "d", estimate=60, minutes_ago=2)
    # One slot: each starts when everything ahead of it has ended.
    assert names() == ["a", "b", "c", "d"]
    assert waits() == {"a": 500, "b": 800, "c": 920, "d": 1520}
    client = APIClient()
    client.force_authenticate(three)
    info = client.get("/api/training/runs/").data[0]["queue"]
    assert info["position"] == 2 and info["waiting"] == 4
    assert info["estimated_wait_seconds"] == 800 and info["worker_online"] is True
    start_at = timezone.datetime.fromisoformat(info["estimated_start_at"].replace("Z", "+00:00"))
    assert abs((start_at - timezone.now()).total_seconds() - 800) < 5

    # Two threads per training: the CPU time passes twice as fast.
    TrainingWorker.objects.filter(pk=row.pk).update(threads=2)
    assert waits() == {"a": 250, "b": 400, "c": 460, "d": 760}
    # Two slots. "d" waits for its account's running training (Free: one at a time).
    TrainingWorker.objects.filter(pk=row.pk).update(threads=1, slots=2)
    assert waits() == {"a": 0, "b": 300, "c": 420, "d": 500}
    shot = queue.snapshot()
    assert [e.blocked for e in shot.entries] == [False, False, False, True]
    # A training that is past its estimate is expected to need a quarter of that again.
    TrainingRun.objects.filter(pk=running.pk).update(cpu_seconds=800)
    TrainingWorker.objects.filter(pk=row.pk).update(slots=1)
    assert waits()["a"] == 200
    # Without a worker nothing can be promised.
    TrainingWorker.objects.filter(pk=row.pk).update(stopped_at=timezone.now())
    assert set(waits().values()) == {None}
    info = client.get("/api/training/runs/").data[0]["queue"]
    assert info["estimated_wait_seconds"] is None and info["estimated_start_at"] is None
    assert info["worker_online"] is False


def test_the_check_says_how_long_the_wait_is(queued, site):
    web, ada = account("ada")
    other = User.objects.create_user("other")
    paying, _ = account("paying", "starter")
    TrainingWorker.objects.create(name="w", slots=1, threads=1)
    make_run(other, "running", "running", estimate=300, minutes_ago=3)
    make_run(other, "waiting", estimate=120, minutes_ago=2)
    body = {"graph": DEMO_GRAPH, "dataset": RANDOM, "config": QUICK}
    info = web.post("/api/training/check/", body, format="json").data["queue"]
    assert info["position"] == 1  # "other" has a turn running already
    assert (info["waiting"], info["running"]) == (1, 1)
    assert info["estimated_wait_seconds"] == 300 and info["worker_online"] is True
    assert info["hint"] == "Paid plans start first." and info["message"] is None
    info = paying.post("/api/training/check/", body, format="json").data["queue"]
    assert info["hint"] == "Higher plans start first."
    budgets.set_plan(paying.handler._force_user, "pro")
    info = paying.post("/api/training/check/", body, format="json").data["queue"]
    assert info["hint"] == "Your plan (Pro) starts before the other plans."
    assert paying.get("/api/training/").data["queue"]["priority_hint"] == info["hint"]
    assert APIClient().get("/api/training/").data["queue"]["priority_hint"] is None


def test_no_priority_hint_when_all_plans_are_equal(queued, limits):
    web, ada = account("ada")
    run_id = started(web)
    assert get(web, run_id)["queue"]["hint"] is None
    limits.PLANS = {"a": {"name": "A"}, "b": {"name": "B", "price": 5}}
    assert get(web, run_id)["queue"]["hint"] is None


# --------------------------------------------------------------------------- #
# The statistics of the queue
# --------------------------------------------------------------------------- #


def test_the_statistics_of_the_queue(queued, site):
    now = timezone.now()
    worker = TrainingWorker.objects.create(
        name="w", slots=2, started_at=now - timedelta(hours=10), heartbeat_at=now
    )
    # A worker that ran for five hours and stopped; one that died an hour after it started.
    TrainingWorker.objects.create(
        name="old",
        slots=1,
        started_at=now - timedelta(days=2, hours=5),
        heartbeat_at=now - timedelta(days=2),
        stopped_at=now - timedelta(days=2),
    )
    TrainingWorker.objects.create(
        name="dead",
        slots=1,
        started_at=now - timedelta(days=1, hours=1),
        heartbeat_at=now - timedelta(days=1),
    )
    # Before the seven days: only what reaches into them counts.
    TrainingWorker.objects.create(
        name="ancient",
        slots=1,
        started_at=now - timedelta(days=8),
        heartbeat_at=now - timedelta(days=6, hours=22),
        stopped_at=now - timedelta(days=6, hours=22),
    )

    def attempt(waited, ran, hours_ago, plan="free", running=False):
        start = now - timedelta(hours=hours_ago)
        TrainingAttempt.objects.create(
            worker=worker,
            plan=plan,
            queued_at=start - timedelta(seconds=waited),
            started_at=start,
            ended_at=None if running else start + timedelta(seconds=ran),
            outcome="" if running else "completed",
        )

    for index in range(18):
        attempt(waited=10 * index, ran=1800, hours_ago=5)
    attempt(waited=3600, ran=3600, hours_ago=3, plan="pro")
    attempt(waited=7200, ran=0, hours_ago=1, plan="pro", running=True)
    # Started eight days ago, ended six days and 23 hours ago: one hour in the window.
    TrainingAttempt.objects.create(
        worker=worker,
        queued_at=now - timedelta(days=8),
        started_at=now - timedelta(days=8),
        ended_at=now - timedelta(days=6, hours=23),
        outcome="completed",
    )
    stats = queue.stats(now=now)
    assert stats["started"] == 20
    assert stats["median_wait_seconds"] == 90 and stats["p95_wait_seconds"] == 3600
    assert stats["max_wait_seconds"] == 7200
    assert stats["by_plan"][0] == {
        "plan": "Free",
        "started": 18,
        "median_wait_seconds": 80,
        "p95_wait_seconds": 170,
    }
    assert stats["by_plan"][1]["plan"] == "Pro" and stats["by_plan"][1]["started"] == 2
    busy = 18 * 1800 + 3600 + 3600 + 3600
    there = 2 * 10 * 3600 + 5 * 3600 + 3600 + 2 * 3600
    assert stats["busy_slot_seconds"] == pytest.approx(busy)
    assert stats["available_slot_seconds"] == pytest.approx(there)
    assert stats["utilisation"] == pytest.approx(busy / there)
    assert queue.stats(days=1, now=now)["started"] == 20
    empty = queue.stats(now=now - timedelta(days=30))
    assert (empty["started"], empty["median_wait_seconds"], empty["utilisation"]) == (0, None, None)

    # Old rows are deleted after 90 days.
    TrainingAttempt.objects.filter(plan="pro").update(ended_at=now - timedelta(days=91))
    TrainingWorker.objects.filter(name="old").update(stopped_at=now - timedelta(days=91))
    assert queue.prune(now) == 3
    assert not TrainingWorker.objects.filter(name="old").exists()
