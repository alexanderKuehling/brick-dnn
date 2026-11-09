"""Usage and budgets: measuring compute, the ledger, limits and where they are enforced.

Most tests use a *controlled* fake runner: it reports the CPU time written into ``cpu.txt``
of its run directory, stops by itself at the limit of its job, follows stop requests like
the real runner (the reason and message of the STOP file) and completes once a file
``finish`` appears. The real runner's own measuring and limit are tested with its
StateReporter, and one test trains for real.
"""

# ruff: noqa: F811 (the fixture training_dir is imported from test_training_api)

from __future__ import annotations

import io
import json
import sys
import time
import zipfile
from datetime import date, datetime, timedelta
from datetime import timezone as dt_timezone

import pytest
from django.contrib.auth import get_user_model
from django.utils import timezone
from rest_framework.test import APIClient

from brick_dnn_core import accounts, budgets, training_service
from brick_dnn_core.api_tokens import create_token
from brick_dnn_core.models import (
    ApiToken,
    OAuthClient,
    OAuthConnection,
    TrainingRun,
    UsageRecord,
    UserProfile,
)
from brick_dnn_core.oauth import _issue_tokens
from brickdnn.training import torch_installed
from brickdnn.training.state import read_job, read_state, request_stop

from .test_mcp import call
from .test_training_api import (  # noqa: F401 (training_dir is a fixture)
    DEMO_GRAPH,
    QUICK,
    RANDOM,
    iris_csv,
    training_dir,
    upload,
    use_runner,
    wait_for,
    wait_for_state,
)

pytestmark = pytest.mark.django_db

CONTROLLED_RUNNER = """
import sys
import time
from pathlib import Path

from brickdnn.training import state as st

run_dir = Path(sys.argv[-1])
job = st.read_job(run_dir)
limits = job.get("limits") or {}
state = st.initial_state(1)
state.update(status="running", message="Epoch 1 of 1", task="demo", total_params=99,
             preprocessing={"kind": "demo"})


def cpu():
    try:
        return float((run_dir / "cpu.txt").read_text())
    except (OSError, ValueError):
        return 0.0


def save(**changes):
    state.update(changes)
    state["updated_at"] = time.time()
    state["usage"] = {"cpu_seconds": cpu(), "wall_seconds": 0.0,
                      "limit_cpu_seconds": limits.get("cpu_seconds")}
    st.write_state(run_dir, state)


def finish(status, message, reason=None, why=""):
    (run_dir / st.WEIGHTS_FILE).write_bytes(b"fake weights")
    message = f"{why} {message}" if why else message
    save(status=status, message=message, weights=True, finished_at=time.time(),
         stop_reason=reason)
    sys.exit(0)


deadline = time.time() + 30
while time.time() < deadline:
    save()
    if limits.get("cpu_seconds") is not None and cpu() >= limits["cpu_seconds"]:
        finish("stopped", "Stopped after 0 of 1 epochs.", limits["reason"], limits["message"])
    request = st.stop_request(run_dir)
    if request is not None:
        finish("stopped", "Stopped after 0 of 1 epochs.", request["reason"], request["message"])
    if (run_dir / "finish").exists():
        finish("completed", "Finished after 1 epoch.")
    time.sleep(0.03)
"""

# Answers like brickdnn.training.predict, with the CPU time it used.
COUNTING_PREDICTOR = """
import json
import sys

request = json.load(sys.stdin)
tensor = request["inputs"][0]
count = tensor["shape"][0]
print(json.dumps({"outputs": [[[1.0, 0.0, 0.0]] * count], "cpu_seconds": 1.5}))
"""

LONG = {**QUICK, "epochs": 1000, "max_samples": None}
User = get_user_model()


# --------------------------------------------------------------------------- #
# Fixtures and helpers
# --------------------------------------------------------------------------- #


@pytest.fixture
def controlled(training_dir, settings, tmp_path):
    script = tmp_path / "controlled_runner.py"
    script.write_text(CONTROLLED_RUNNER)
    settings.TRAINING_RUNNER_COMMAND = [sys.executable, str(script)]
    settings.TRAINING_MAX_RUNS = 3
    return training_dir


@pytest.fixture
def limits(settings):
    """No limits unless a test sets them (as on a server of your own)."""
    settings.BUDGET_CPU_MINUTES_PER_MONTH = 0
    settings.BUDGET_CPU_MINUTES_PER_RUN = 0
    settings.BUDGET_STORAGE_MB = 0
    settings.PRICE_PER_CPU_HOUR = 0
    settings.PRICE_PER_GB_MONTH = 0
    settings.PRICE_CURRENCY = "EUR"
    return settings


@pytest.fixture
def ada(limits):
    return User.objects.create_user("ada", password="correct-horse-battery")


@pytest.fixture
def web(ada):
    client = APIClient()
    client.force_authenticate(ada)
    return client


def token_client(user, name="agent", **kwargs) -> tuple[APIClient, ApiToken]:
    token, key = create_token(user, name, **kwargs)
    client = APIClient()
    client.credentials(HTTP_AUTHORIZATION=f"Bearer {key}")
    return client, token


def app_client(user, name="Claude") -> tuple[APIClient, OAuthConnection]:
    """An app connected with OAuth: its short-lived access token, for the MCP server."""
    client_app = OAuthClient.objects.create(client_id=f"id-{name}", name=name, redirect_uris=[])
    connection = OAuthConnection.objects.create(
        owner=user,
        client=client_app,
        refresh_hash=f"hash-{name}",
        refresh_expires_at=timezone.now() + timedelta(days=1),
    )
    access = _issue_tokens(connection)["access_token"]
    client = APIClient()
    client.credentials(HTTP_AUTHORIZATION=f"Bearer {access}")
    return client, connection


def start(client, config=QUICK, name="Small net", graph=DEMO_GRAPH, dataset=RANDOM):
    return client.post(
        "/api/training/runs/",
        {"name": name, "graph": graph, "dataset": dataset, "config": config},
        format="json",
    )


def started(client, **kwargs) -> str:
    response = start(client, **kwargs)
    assert response.status_code == 201, response.data
    return response.data["id"]


def run_dir(training_dir, run_id):
    return training_dir / "runs" / str(run_id)


def use_cpu(client, training_dir, run_id, seconds: float) -> dict:
    """Let the controlled runner report ``seconds`` of CPU time, then look at the run."""
    directory = run_dir(training_dir, run_id)
    (directory / "cpu.txt").write_text(str(seconds))
    deadline = time.monotonic() + 10
    while True:
        state = read_state(directory) or {}
        if (state.get("usage") or {}).get("cpu_seconds") == seconds:
            break
        assert time.monotonic() < deadline, "the runner did not report the CPU time"
        time.sleep(0.02)
    return client.get(f"/api/training/runs/{run_id}/").data


def finish(client, training_dir, run_id) -> dict:
    (run_dir(training_dir, run_id) / "finish").touch()
    return wait_for(client, run_id)


def spend(user, seconds: float, period: date | None = None, **fields) -> UsageRecord:
    """Usage of earlier trainings, straight into the ledger."""
    return UsageRecord.objects.create(
        user=user,
        kind="training",
        quantity=seconds,
        period=period or budgets.period_of(),
        **fields,
    )


def ledger(user, **filters) -> float:
    return sum(r.quantity for r in UsageRecord.objects.filter(user=user, **filters))


# --------------------------------------------------------------------------- #
# The ledger
# --------------------------------------------------------------------------- #


def test_usage_is_counted_while_the_run_is_active(web, ada, controlled):
    run_id = started(web)
    wait_for_state(controlled, run_id)
    use_cpu(web, controlled, run_id, 10.0)
    assert ledger(ada) == 10.0
    use_cpu(web, controlled, run_id, 25.5)
    assert ledger(ada) == 25.5
    # One row per run and month, which grows.
    record = UsageRecord.objects.get(user=ada)
    assert (record.kind, record.unit, record.source, record.via) == (
        "training",
        "cpu_second",
        "web",
        "",
    )
    assert record.run_id == uuid(run_id) and record.run_name == "Small net"
    assert record.period == budgets.period_of()
    data = finish(web, controlled, run_id)
    assert data["status"] == "completed"
    assert data["usage"]["cpu_minutes"] == round(25.5 / 60, 2)
    # Looking again counts nothing twice.
    web.get(f"/api/training/runs/{run_id}/")
    web.get("/api/training/runs/")
    assert ledger(ada) == 25.5


def uuid(value):
    import uuid as uuid_module

    return uuid_module.UUID(str(value))


def test_small_amounts_wait_until_the_end(web, ada, controlled):
    run_id = started(web)
    wait_for_state(controlled, run_id)
    use_cpu(web, controlled, run_id, 1.0)
    assert ledger(ada) == 0  # less than RECORD_STEP_SECONDS
    use_cpu(web, controlled, run_id, 1.5)
    finish(web, controlled, run_id)
    assert ledger(ada) == 1.5


def test_a_run_reaped_by_another_process_is_counted(web, ada, controlled):
    run_id = started(web)
    wait_for_state(controlled, run_id)
    (run_dir(controlled, run_id) / "cpu.txt").write_text("42.0")
    # Another server process looks at it: it did not start the runner.
    process = training_service._PROCESSES.pop(str(run_id))
    (run_dir(controlled, run_id) / "finish").touch()
    process.wait(timeout=10)
    data = web.get(f"/api/training/runs/{run_id}/").data
    assert data["status"] == "completed"
    assert ledger(ada) == 42.0


def test_counting_twice_from_two_processes_counts_once(ada):
    run = TrainingRun.objects.create(owner=ada, name="Run", status="running")
    first = TrainingRun.objects.get(pk=run.pk)
    second = TrainingRun.objects.get(pk=run.pk)
    assert budgets.record_run(first, 30.0) == 30.0
    assert budgets.record_run(second, 30.0) == 0.0  # it saw 0 and lost the race
    assert second.cpu_seconds == 30.0
    assert budgets.record_run(second, 50.0) == 20.0
    assert ledger(ada) == 50.0
    assert UsageRecord.objects.count() == 1


NO_REPORT_RUNNER = """
import sys
import time
from pathlib import Path

from brickdnn.training import state as st

run_dir = Path(sys.argv[-1])
state = st.initial_state(1)
del state["usage"]  # like a runner that does not measure its CPU time
time.sleep(0.3)
(run_dir / st.WEIGHTS_FILE).write_bytes(b"fake weights")
state.update(status="completed", message="Done.", weights=True, finished_at=time.time())
st.write_state(run_dir, state)
"""


def test_runs_without_a_report_count_their_wall_time(
    web, ada, training_dir, settings, tmp_path, monkeypatch
):
    script = tmp_path / "no_report.py"
    script.write_text(NO_REPORT_RUNNER)
    settings.TRAINING_RUNNER_COMMAND = [sys.executable, str(script)]
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    run_id = started(web)
    data = wait_for(web, run_id)
    run = TrainingRun.objects.get(pk=run_id)
    wall = (run.finished_at - run.started_at).total_seconds()
    assert data["status"] == "completed"
    assert ledger(ada) == pytest.approx(2 * wall, rel=0.01)
    assert run.size_bytes > 0  # its files count against the storage quota


def test_usage_follows_the_api_token_and_the_app(ada, controlled):
    agent, token = token_client(ada, "agent")
    run_id = started(agent)
    wait_for_state(controlled, run_id)
    use_cpu(agent, controlled, run_id, 12.0)
    finish(agent, controlled, run_id)
    record = UsageRecord.objects.get(run_id=run_id)
    assert (record.source, record.api_token, record.via) == ("token", token, "agent")
    assert TrainingRun.objects.get(pk=run_id).api_token == token

    app, connection = app_client(ada, "Claude")
    result, text = call(
        app, "start_training", graph=DEMO_GRAPH, dataset=RANDOM, config=QUICK, name="App"
    )
    assert "isError" not in result, text
    app_run = result["structuredContent"]["id"]
    assert result["structuredContent"]["started_via"] == {"source": "app", "name": "Claude"}
    wait_for_state(controlled, app_run)
    (run_dir(controlled, app_run) / "cpu.txt").write_text("7.0")
    (run_dir(controlled, app_run) / "finish").touch()
    deadline = time.monotonic() + 10
    while call(app, "get_training_run", id=app_run)[0]["structuredContent"]["status"] != (
        "completed"
    ):
        assert time.monotonic() < deadline
        time.sleep(0.05)
    record = UsageRecord.objects.get(run_id=app_run)
    assert (record.source, record.connection, record.api_token) == ("app", connection, None)
    assert record.via == "Claude"
    # Deleting the token keeps its usage (for the month's total), with its name.
    token.delete()
    record = UsageRecord.objects.get(run_id=run_id)
    assert record.api_token is None and record.via == "agent"
    assert ledger(ada) == 19.0


def test_predictions_count_their_cpu_time(ada, controlled, settings, tmp_path):
    script = tmp_path / "predictor.py"
    script.write_text(COUNTING_PREDICTOR)
    settings.TRAINING_PREDICT_COMMAND = [sys.executable, str(script)]
    web = APIClient()
    web.force_authenticate(ada)
    run_id = started(web)
    wait_for_state(controlled, run_id)
    finish(web, controlled, run_id)
    agent, token = token_client(ada, "predictor")
    response = agent.post(
        f"/api/training/runs/{run_id}/predict/", {"inputs": [[0.1, 0.2, 0.3, 0.4]]}, format="json"
    )
    assert response.status_code == 200, response.data
    record = UsageRecord.objects.get(kind="prediction")
    assert (record.quantity, record.api_token, record.source) == (1.5, token, "token")
    assert record.run_id == uuid(run_id)


def test_months_and_their_end():
    assert budgets.period_of(datetime(2026, 11, 30, 23, 59, tzinfo=dt_timezone.utc)) == date(
        2026, 11, 1
    )
    # Midnight in Berlin on 1 December is still November in UTC.
    berlin = dt_timezone(timedelta(hours=1))
    assert budgets.period_of(datetime(2026, 12, 1, 0, 30, tzinfo=berlin)) == date(2026, 11, 1)
    assert budgets.period_end(date(2026, 12, 1)) == datetime(2027, 1, 1, tzinfo=dt_timezone.utc)
    assert budgets.period_end(date(2026, 2, 1)) == datetime(2026, 3, 1, tzinfo=dt_timezone.utc)


def test_budgets_reset_every_month(web, ada, limits, controlled):
    limits.BUDGET_CPU_MINUTES_PER_MONTH = 1
    last_month = budgets.period_of() - timedelta(days=1)
    spend(ada, 600, period=last_month.replace(day=1))  # ten minutes last month
    assert budgets.used_seconds(ada) == 0
    started(web)  # this month nothing is used yet
    spend(ada, 60)
    response = start(web)
    assert response.status_code == 403
    assert "resets on 1 " in response.data["detail"]


# --------------------------------------------------------------------------- #
# Limits before a training starts
# --------------------------------------------------------------------------- #


def test_monthly_allowance_blocks_new_trainings(web, ada, limits, controlled):
    limits.BUDGET_CPU_MINUTES_PER_MONTH = 2
    spend(ada, 125)
    response = start(web)
    assert response.status_code == 403
    assert response.data["code"] == "budget_exceeded"
    detail = response.data["detail"]
    assert "The monthly compute allowance of this account is used up" in detail
    assert "2.08 CPU minutes of 2 CPU minutes this month" in detail
    assert "operator of this server" in detail
    assert response.data["budget"]["limit_cpu_minutes"] == 2
    assert not TrainingRun.objects.exists()


def test_the_rest_of_the_budget_is_the_limit_of_the_run(web, ada, limits, controlled):
    limits.BUDGET_CPU_MINUTES_PER_MONTH = 10
    spend(ada, 240)
    run_id = started(web)
    job = read_job(run_dir(controlled, run_id))
    assert job["limits"]["cpu_seconds"] == 360
    assert job["limits"]["reason"] == "budget"
    assert "compute budget ran out" in job["limits"]["message"]
    data = use_cpu(web, controlled, run_id, 360.0)
    data = wait_for(web, run_id)
    assert data["status"] == "stopped"
    assert data["usage"]["stop_reason"] == "budget"
    assert data["message"].startswith("Stopped: the compute budget ran out")
    assert data["has_weights"] is True  # stopped like a user stop
    assert ledger(ada) == 600


def test_per_run_limit(web, ada, limits, controlled):
    limits.BUDGET_CPU_MINUTES_PER_RUN = 0.5
    run_id = started(web)
    job = read_job(run_dir(controlled, run_id))
    assert job["limits"] == {
        "cpu_seconds": 30.0,
        "reason": "run_limit",
        "message": "Stopped: this run used 0.5 CPU minutes, the most one training may use on "
        "this server.",
    }
    use_cpu(web, controlled, run_id, 31.0)
    data = wait_for(web, run_id)
    assert (data["status"], data["usage"]["stop_reason"]) == ("stopped", "run_limit")
    assert data["usage"]["limit_cpu_minutes"] == 0.5


def test_own_budget_and_token_budgets(ada, limits, controlled):
    web = APIClient()
    web.force_authenticate(ada)
    # The own budget: set on the account page.
    response = web.patch("/api/account/usage/", {"budget_cpu_minutes": 1}, format="json")
    assert response.status_code == 200, response.data
    assert response.data["compute"]["own_budget"] == 1
    spend(ada, 61)
    response = start(web)
    assert response.status_code == 403
    assert "The monthly budget set for this account is used up" in response.data["detail"]
    assert "account page" in response.data["detail"]
    web.patch("/api/account/usage/", {"budget_cpu_minutes": None}, format="json")
    assert start(web).status_code == 201

    # A token with a budget of its own: its usage counts, the web app's does not.
    agent, token = token_client(ada, "agent")
    token.budget_cpu_minutes = 0.5
    token.save()
    spend(ada, 20, api_token=token, source="token", via="agent")
    run_id = started(agent, name="Agent run")
    assert read_job(run_dir(controlled, run_id))["limits"]["cpu_seconds"] == 10
    TrainingRun.objects.filter(pk=run_id).update(status="completed")
    spend(ada, 11, api_token=token, source="token", via="agent")
    response = start(agent)
    assert response.status_code == 403
    assert "The monthly budget of the API token “agent” is used up" in response.data["detail"]
    assert "API tokens page" in response.data["detail"]
    # Other tokens and the web app can still train.
    other, _ = token_client(ada, "other")
    assert start(other).status_code == 201


def test_running_trainings_reserve_their_limits(ada, limits, controlled):
    limits.BUDGET_CPU_MINUTES_PER_MONTH = 1
    web = APIClient()
    web.force_authenticate(ada)
    first = started(web)
    assert read_job(run_dir(controlled, first))["limits"]["cpu_seconds"] == 60
    # The first run may use all that is left: a second one has to wait.
    response = start(web)
    assert response.status_code == 403
    assert "reserved by trainings that are still running" in response.data["detail"]
    # With a smaller per-run limit, both fit.
    limits.BUDGET_CPU_MINUTES_PER_RUN = 0.5
    web.post(f"/api/training/runs/{first}/stop/")
    wait_for(web, first)
    one = started(web)
    two = started(web)
    assert read_job(run_dir(controlled, one))["limits"]["cpu_seconds"] == 30
    assert read_job(run_dir(controlled, two))["limits"]["cpu_seconds"] == 30
    assert start(web).status_code == 403


@pytest.mark.django_db(transaction=True)
def test_concurrent_starts_cannot_both_use_the_rest(ada, limits, controlled):
    """Starts happen one at a time under the start lock: the second sees the first."""
    import threading

    limits.BUDGET_CPU_MINUTES_PER_MONTH = 1
    results = []

    def go():
        from django.db import connection

        client = APIClient()
        client.force_authenticate(ada)
        results.append(start(client).status_code)
        connection.close()

    threads = [threading.Thread(target=go) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(results) == [201, 403]


def test_runs_far_beyond_the_limit_are_refused(web, ada, limits, controlled):
    limits.BUDGET_CPU_MINUTES_PER_RUN = 1
    response = start(web, config=LONG)
    assert response.status_code == 403
    assert response.data["code"] == "estimate_too_high"
    assert "far more than it can use" in response.data["detail"]
    assert "the most one training may use" in response.data["detail"]
    assert response.data["estimate"]["cpu_minutes"] > 3
    assert not TrainingRun.objects.exists()

    # A little more than allowed: it starts with a warning.
    limits.BUDGET_CPU_MINUTES_PER_RUN = 0.1
    response = start(web, config={**QUICK, "epochs": 600})
    assert response.status_code == 201, response.data
    assert "probably be stopped" in response.data["budget_warning"]


def test_the_check_shows_the_estimate_and_the_budget(web, ada, limits, controlled):
    limits.BUDGET_CPU_MINUTES_PER_MONTH = 30
    limits.PRICE_PER_CPU_HOUR = 0.6
    spend(ada, 600)
    body = {"graph": DEMO_GRAPH, "dataset": RANDOM, "config": QUICK}
    data = web.post("/api/training/check/", body, format="json").data
    assert data["ok"] is True
    estimate = data["estimate"]
    assert 0 < estimate["low_cpu_minutes"] < estimate["cpu_minutes"] < estimate["high_cpu_minutes"]
    assert estimate["currency"] == "EUR" and estimate["cost"] > 0
    assert "samples" in estimate["basis"] and "estimate" in estimate["note"]
    budget = data["budget"]
    assert budget["allowed"] is True and budget["reason"] is None
    assert budget["run_limit_cpu_minutes"] == 20 and budget["limit_reason"] == "budget"
    # A search multiplies the estimate by its trials.
    tuning = {
        "method": "random",
        "trials": 4,
        "space": {"learning_rate": {"min": 0.001, "max": 0.1}},
    }
    longer = {**body, "config": {**QUICK, "epochs": 200}}
    single = web.post("/api/training/check/", longer, format="json").data
    searched = web.post("/api/training/check/", {**longer, "tuning": tuning}, format="json").data
    assert searched["estimate"]["details"]["trials"] == 4
    assert searched["estimate"]["cpu_minutes"] > 2 * single["estimate"]["cpu_minutes"]

    # A used-up budget: the check says so (and is not refused or counted itself).
    spend(ada, 1200)
    data = web.post("/api/training/check/", body, format="json").data
    assert data["budget"]["allowed"] is False
    assert "used up" in data["budget"]["reason"]
    assert data["budget"]["code"] == "budget_exceeded"
    assert ledger(ada) == 1800


def test_read_only_actions_are_never_counted_or_limited(web, ada, limits, controlled):
    limits.BUDGET_CPU_MINUTES_PER_MONTH = 1
    limits.BUDGET_STORAGE_MB = 0.001
    spend(ada, 100)
    assert web.post("/api/analyze/", DEMO_GRAPH, format="json").status_code == 200
    assert web.post("/api/generate/", {"graph": DEMO_GRAPH}, format="json").status_code == 200
    assert web.get("/api/training/runs/").status_code == 200
    assert web.get("/api/account/usage/").status_code == 200
    body = {"graph": DEMO_GRAPH, "dataset": RANDOM}
    assert web.post("/api/training/check/", body, format="json").status_code == 200
    assert UsageRecord.objects.count() == 1


def test_storage_quota_blocks_uploads_and_trainings(web, ada, limits, training_dir):
    limits.BUDGET_STORAGE_MB = 0.002  # about 2 kB
    small = upload(web, "iris.csv", iris_csv(12))
    assert small.status_code == 201, small.data
    response = upload(web, "big.csv", iris_csv(200))
    assert response.status_code == 403
    assert response.data["code"] == "storage_quota_exceeded"
    assert "does not fit into the storage of this account" in response.data["detail"]
    # Full: trainings (whose weights take room too) are refused as well.
    limits.BUDGET_STORAGE_MB = 0.0001
    response = start(web)
    assert response.status_code == 403
    assert "storage of this account is full" in response.data["detail"]
    # MCP
    agent, _ = token_client(ada)
    result, text = call(agent, "upload_dataset", name="Iris", csv=iris_csv(12).decode())
    assert result["isError"] is True
    assert "storage of this account is full" in text and "get_usage" in text


def test_storage_quota_blocks_imports_in_the_data_tab(web, ada, limits, settings, tmp_path):
    pytest.importorskip("pandas")
    settings.DATA_PREP_DIR = tmp_path / "prepare"
    limits.BUDGET_STORAGE_MB = 0.0001
    from django.core.files.uploadedfile import SimpleUploadedFile

    response = web.post(
        "/api/data/sources/",
        {"file": SimpleUploadedFile("data.csv", iris_csv(50))},
        format="multipart",
    )
    assert response.status_code == 403
    assert "does not fit into the storage" in response.data["detail"]


# --------------------------------------------------------------------------- #
# Limits during a training
# --------------------------------------------------------------------------- #


def test_a_used_up_budget_stops_the_run(ada, limits, controlled):
    agent, token = token_client(ada, "agent")
    run_id = started(agent)
    wait_for_state(controlled, run_id)
    # The user gives the token a budget while the run is active.
    token.budget_cpu_minutes = 0.5
    token.save()
    use_cpu(agent, controlled, run_id, 20.0)
    assert TrainingRun.objects.get(pk=run_id).stop_requested_at is None
    data = use_cpu(agent, controlled, run_id, 31.0)
    assert data["stop_requested"] is True
    data = wait_for(agent, run_id)
    assert data["status"] == "stopped"
    assert data["usage"]["stop_reason"] == "budget"
    assert data["message"].startswith(
        "Stopped: The monthly budget of the API token “agent” is used up"
    )
    assert data["has_weights"] is True


def test_the_per_run_limit_is_enforced_by_the_server_too(web, ada, limits, controlled):
    run_id = started(web)
    wait_for_state(controlled, run_id)
    # Set after the start (the runner does not know it): the server stops it.
    UserProfile.objects.update_or_create(user=ada, defaults={"cpu_minutes_per_run": 0.2})
    use_cpu(web, controlled, run_id, 13.0)
    data = wait_for(web, run_id)
    assert (data["status"], data["usage"]["stop_reason"]) == ("stopped", "run_limit")
    assert "the most one training may use" in data["message"]


def test_a_runner_that_ignores_the_budget_stop_is_killed(web, ada, limits, controlled, settings):
    settings.TRAINING_RUNNER_COMMAND = [
        sys.executable,
        str(controlled.parent / "fake_runner.py"),
        "hang",
    ]
    run_id = started(web)
    wait_for_state(controlled, run_id)
    run = TrainingRun.objects.get(pk=run_id)
    run.stop_requested_at = timezone.now() - timedelta(minutes=5)
    run.stop_reason = "budget"
    run.save()
    request_stop(run_dir(controlled, run_id), "budget", "Stopped: the budget is used up.")
    data = wait_for(web, run_id)
    assert data["status"] == "stopped"
    assert data["message"] == "Stopped: the budget is used up. Training was stopped."
    assert data["usage"]["stop_reason"] == "budget"


def test_admin_overrides(ada, limits, controlled):
    limits.BUDGET_CPU_MINUTES_PER_MONTH = 1
    spend(ada, 120)
    web = APIClient()
    web.force_authenticate(ada)
    assert start(web).status_code == 403
    # The operator gives this account more (0 = unlimited).
    UserProfile.objects.update_or_create(user=ada, defaults={"cpu_minutes_per_month": 0})
    assert start(web).status_code == 201
    assert budgets.limits_for(ada).monthly_seconds is None
    UserProfile.objects.filter(user=ada).update(cpu_minutes_per_month=5, storage_mb=10)
    assert budgets.limits_for(ada).monthly_seconds == 300
    assert budgets.limits_for(ada).storage_bytes == 10 * 1024 * 1024


# --------------------------------------------------------------------------- #
# Predictions
# --------------------------------------------------------------------------- #


def test_predictions_need_compute_left(ada, limits, controlled, settings, tmp_path):
    script = tmp_path / "predictor.py"
    script.write_text(COUNTING_PREDICTOR)
    settings.TRAINING_PREDICT_COMMAND = [sys.executable, str(script)]
    web = APIClient()
    web.force_authenticate(ada)
    run_id = started(web)
    wait_for_state(controlled, run_id)
    finish(web, controlled, run_id)

    agent, token = token_client(ada, "agent")
    token.budget_cpu_minutes = 0.02  # 1.2 s: one prediction (1.5 s) uses it up
    token.save()
    path = f"/api/training/runs/{run_id}/predict/"
    inputs = {"inputs": [[0.1, 0.2, 0.3, 0.4]]}
    assert agent.post(path, inputs, format="json").status_code == 200
    response = agent.post(path, inputs, format="json")
    assert response.status_code == 403
    assert "API token “agent” is used up" in response.data["detail"]
    result, text = call(agent, "predict", run_id=run_id, inputs=[[0.1, 0.2, 0.3, 0.4]])
    assert result["isError"] is True and "is used up" in text
    # The web app has no token budget.
    assert web.post(path, inputs, format="json").status_code == 200


# --------------------------------------------------------------------------- #
# MCP
# --------------------------------------------------------------------------- #


def test_mcp_budget_errors_and_get_usage(ada, limits, controlled):
    limits.BUDGET_CPU_MINUTES_PER_MONTH = 5
    agent, token = token_client(ada, "agent")
    token.budget_cpu_minutes = 1
    token.save()
    spend(ada, 30, api_token=token, source="token", via="agent")

    result, _ = call(agent, "get_usage")
    usage = result["structuredContent"]
    assert usage["compute"]["used"] == 0.5 and usage["compute"]["limit"] == 5
    assert usage["caller"] == {
        "source": "token",
        "id": token.pk,
        "name": "agent",
        "used_cpu_minutes": 0.5,
        "budget_cpu_minutes": 1.0,
        "remaining_cpu_minutes": 0.5,
    }
    assert usage["compute"]["available"] == 0.5

    result, _ = call(agent, "check_training", graph=DEMO_GRAPH, dataset=RANDOM, config=QUICK)
    check = result["structuredContent"]
    assert check["ok"] is True and check["budget"]["allowed"] is True
    assert check["budget"]["run_limit_cpu_minutes"] == 0.5
    assert check["estimate"]["cpu_minutes"] > 0

    spend(ada, 31, api_token=token, source="token", via="agent")
    result, text = call(agent, "start_training", graph=DEMO_GRAPH, dataset=RANDOM, config=QUICK)
    assert result["isError"] is True
    assert "The monthly budget of the API token “agent” is used up" in text
    assert "get_usage" in text
    result, _ = call(agent, "check_training", graph=DEMO_GRAPH, dataset=RANDOM)
    assert result["structuredContent"]["budget"]["allowed"] is False

    # Read-only tokens can see their usage too.
    reader, _ = token_client(ada, "reader", read_only=True)
    names = {
        tool["name"]
        for tool in reader.post(
            "/api/mcp/",
            json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}),
            content_type="application/json",
        ).json()["result"]["tools"]
    }
    assert "get_usage" in names


# --------------------------------------------------------------------------- #
# The API: usage, budgets of tokens and apps
# --------------------------------------------------------------------------- #


def test_usage_api(ada, limits, controlled):
    limits.BUDGET_CPU_MINUTES_PER_MONTH = 60
    limits.BUDGET_CPU_MINUTES_PER_RUN = 30
    limits.BUDGET_STORAGE_MB = 100
    limits.PRICE_PER_CPU_HOUR = 1.2
    limits.PRICE_PER_GB_MONTH = 0.5
    web = APIClient()
    web.force_authenticate(ada)
    agent, token = token_client(ada, "agent")
    _, connection = app_client(ada, "Claude")
    spend(ada, 120)
    spend(ada, 600, api_token=token, source="token", via="agent")
    spend(ada, 60, connection=connection, source="app", via="Claude")
    UsageRecord.objects.create(user=ada, kind="prediction", quantity=30, period=budgets.period_of())
    upload(web, "iris.csv", iris_csv(12))

    data = web.get("/api/account/usage/").data
    compute = data["compute"]
    assert compute["used"] == 13.5 and compute["limit"] == 60 and compute["remaining"] == 46.5
    assert compute["allowance"] == 60 and compute["own_budget"] is None
    assert compute["run_limit"] == 30
    assert compute["by_kind"] == {"training": 13.0, "prediction": 0.5}
    sources = {(s["source"], s["name"]): s for s in compute["by_source"]}
    assert sources[("token", "agent")]["used_cpu_minutes"] == 10
    assert sources[("token", "agent")]["id"] == token.pk
    assert sources[("app", "Claude")]["used_cpu_minutes"] == 1
    assert sources[("web", "Web app")]["used_cpu_minutes"] == 2.5
    assert data["period"]["resets_at"] == budgets.period_end(budgets.period_of())
    assert data["storage"]["limit_bytes"] == 100 * 1024 * 1024
    assert data["storage"]["used_bytes"] == data["storage"]["by_kind"]["datasets"] > 0
    prices = data["prices"]
    assert prices["currency"] == "EUR" and prices["compute_cost"] == pytest.approx(0.27)
    assert prices["compute_limit_cost"] == pytest.approx(1.2)
    assert data["caller"] is None

    # With the token: its own budget is shown; changing budgets needs the website login.
    token.budget_cpu_minutes = 20
    token.save()
    data = agent.get("/api/account/usage/").data
    assert data["caller"]["remaining_cpu_minutes"] == 10
    response = agent.patch("/api/account/usage/", {"budget_cpu_minutes": 1000}, format="json")
    assert response.status_code == 403
    response = web.patch("/api/account/usage/", {"budget_cpu_minutes": -1}, format="json")
    assert response.status_code == 400
    response = web.patch("/api/account/usage/", {"budget_cpu_minutes": "lots"}, format="json")
    assert response.status_code == 400
    response = web.patch("/api/account/usage/", {"budget_cpu_minutes": 20}, format="json")
    assert response.data["compute"]["limit"] == 20
    assert response.data["compute"]["own_budget"] == 20
    assert APIClient().get("/api/account/usage/").status_code == 401


def test_budgets_of_tokens_and_apps(ada, limits):
    web = APIClient()
    web.force_authenticate(ada)
    agent, token = token_client(ada, "agent")
    _, connection = app_client(ada, "Claude")
    spend(ada, 90, api_token=token, source="token", via="agent")
    spend(ada, 30, connection=connection, source="app", via="Claude")

    response = web.patch(
        f"/api/auth/api-tokens/{token.pk}/", {"budget_cpu_minutes": 5}, format="json"
    )
    assert response.status_code == 200, response.data
    assert response.data["budget_cpu_minutes"] == 5
    listed = web.get("/api/auth/api-tokens/").data
    assert [(t["name"], t["budget_cpu_minutes"], t["used_cpu_minutes"]) for t in listed] == [
        ("agent", 5.0, 1.5)
    ]
    response = web.patch(
        f"/api/auth/connections/{connection.pk}/", {"budget_cpu_minutes": 2.5}, format="json"
    )
    assert response.status_code == 200, response.data
    apps = web.get("/api/auth/connections/").data
    assert [(a["client_name"], a["budget_cpu_minutes"], a["used_cpu_minutes"]) for a in apps] == [
        ("Claude", 2.5, 0.5)
    ]
    web.patch(f"/api/auth/api-tokens/{token.pk}/", {"budget_cpu_minutes": None}, format="json")
    assert ApiToken.objects.get(pk=token.pk).budget_cpu_minutes is None

    # An agent cannot raise its own budget.
    response = agent.patch(
        f"/api/auth/api-tokens/{token.pk}/", {"budget_cpu_minutes": 1000}, format="json"
    )
    assert response.status_code in (401, 403)
    # Nor change other people's tokens.
    other = User.objects.create_user("eve", password="correct-horse-battery")
    eve = APIClient()
    eve.force_authenticate(other)
    response = eve.patch(
        f"/api/auth/api-tokens/{token.pk}/", {"budget_cpu_minutes": 1}, format="json"
    )
    assert response.status_code == 404


# --------------------------------------------------------------------------- #
# Privacy: export and deletion
# --------------------------------------------------------------------------- #


def test_usage_is_exported_and_deleted_with_the_account(ada, limits, training_dir):
    token, _ = create_token(ada, "agent")
    spend(ada, 90, api_token=token, source="token", via="agent")
    UserProfile.objects.update_or_create(user=ada, defaults={"budget_cpu_minutes": 30})
    buffer = io.BytesIO()
    accounts.export_account(ada, buffer)
    with zipfile.ZipFile(buffer) as archive:
        usage = json.loads(archive.read("usage.json"))
        assert "usage.json" in archive.read("README.txt").decode()
    assert usage["limits"]["own_monthly_budget_cpu_minutes"] == 30
    [record] = usage["records"]
    assert record["quantity"] == 90 and record["via"] == "agent" and record["kind"] == "training"
    assert record["month"] == budgets.period_of().isoformat()[:7]

    other = User.objects.create_user("eve", password="correct-horse-battery")
    spend(other, 10)
    accounts.delete_account(ada)
    assert list(UsageRecord.objects.values_list("user__username", flat=True)) == ["eve"]


@pytest.mark.skipif(not torch_installed(), reason="PyTorch is not installed")
def test_real_training_counts_its_cpu_time(web, ada, training_dir, settings, limits):
    settings.TRAINING_RUNNER_COMMAND = None  # the real runner
    limits.BUDGET_CPU_MINUTES_PER_RUN = 60
    run_id = started(web, config={**QUICK, "epochs": 2})
    data = wait_for(web, run_id, timeout=120)
    assert data["status"] == "completed", data["message"]
    state = read_state(run_dir(training_dir, run_id))
    assert state["usage"]["cpu_seconds"] > 0.5  # at least starting Python and PyTorch
    assert state["usage"]["limit_cpu_seconds"] == 3600
    assert ledger(ada) == pytest.approx(state["usage"]["cpu_seconds"])
    assert data["usage"]["estimated_cpu_minutes"] > 0
