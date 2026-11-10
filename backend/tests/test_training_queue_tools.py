"""The training queue as agents and the operator see it: the MCP tools and the activity log,
the queue overview in the Django admin and ``manage.py training_queue``. See
test_training_queue.py for the queue and the worker themselves (and for the fixtures).
"""

# ruff: noqa: F811 (fixtures are imported from other test modules)

from __future__ import annotations

import json
from datetime import timedelta
from io import StringIO

import pytest
from django.core.management import call_command
from django.utils import timezone

from brick_dnn_core import training_queue as queue
from brick_dnn_core.models import AgentAction, TrainingRun

from .test_budgets import controlled, limits, started, token_client  # noqa: F401 (fixtures)
from .test_mcp import call
from .test_plans import site  # noqa: F401
from .test_training_api import (  # noqa: F401
    DEMO_GRAPH,
    RANDOM,
    client,
    training_dir,
    wait_for_state,
)
from .test_training_queue import (  # noqa: F401
    User,
    account,
    beat_long_ago,
    kill,
    make_worker,
    queued,
)

pytestmark = pytest.mark.django_db


# --------------------------------------------------------------------------- #
# MCP and the activity log
# --------------------------------------------------------------------------- #


def test_mcp_tools_show_the_queue(queued, site, make_worker):
    web, ada = account("ada")
    agent, _token = token_client(ada)
    result, text = call(agent, "get_training_options")
    options = json.loads(text)
    assert options["queue"] == {
        "online": False,
        "workers": 0,
        "slots": 0,
        "running": 0,
        "waiting": 0,
        "message": queue.NO_WORKER_MESSAGE,
        "priority_hint": "Paid plans start first.",
    }
    worker = make_worker()
    _, text = call(agent, "check_training", graph=DEMO_GRAPH, dataset=RANDOM)
    check = json.loads(text)["queue"]
    assert check["position"] == 1 and check["estimated_wait_seconds"] == 0

    _, text = call(agent, "start_training", graph=DEMO_GRAPH, dataset=RANDOM, name="GP 1")
    run = json.loads(text)
    assert run["status"] == "queued" and run["queue"]["position"] == 1
    _, text = call(agent, "start_training", graph=DEMO_GRAPH, dataset=RANDOM, name="GP 2")
    second = json.loads(text)
    assert second["queue"]["position"] == 2

    _, text = call(agent, "list_training_runs", status="queued")
    listed = json.loads(text)
    assert [(r["name"], r["queue"]["position"]) for r in listed["runs"]] == [
        ("GP 2", 2),
        ("GP 1", 1),
    ]
    assert listed["runs"][1]["queue"] == {
        "position": 1,
        "waiting": 2,
        "estimated_wait_minutes": 0.0,
    }
    worker.step()  # the first one trains; Free trains one at a time
    _, text = call(agent, "get_training_run", id=second["id"])
    detail = json.loads(text)
    assert detail["status"] == "queued" and detail["queue"]["position"] == 1
    assert detail["queue"]["note"].startswith("Waits until a running training of this account")
    assert detail["queue"]["priority_hint"] == "Paid plans start first."
    assert detail["queue"]["estimated_wait_minutes"] > 0
    assert "progress" not in detail

    _, text = call(agent, "stop_training", id=second["id"])
    assert json.loads(text)["status"] == "stopped"
    _, text = call(agent, "delete_training_run", id=second["id"])
    assert json.loads(text)["killed_running_training"] is False
    _, text = call(agent, "delete_training_run", id=run["id"])
    assert json.loads(text)["killed_running_training"] is True
    worker.step()
    assert not worker.jobs

    summaries = list(
        AgentAction.objects.filter(owner=ada).order_by("id").values_list("tool", "summary")
    )
    assert (
        "start_training",
        "Queued training “GP 1” on random demo data, 10 epochs, position 1 in the queue",
    ) in summaries
    assert (
        "get_training_run",
        "Checked training “GP 2”: queued, position 1 in the queue",
    ) in summaries
    assert ("stop_training", "Stopped training “GP 2”") in summaries


# --------------------------------------------------------------------------- #
# The operator's view
# --------------------------------------------------------------------------- #


def test_the_queue_overview_in_the_admin_and_the_terminal(queued, site, make_worker, client):
    free, _ = account("free-user")
    pro, _ = account("pro-user", "pro")
    running = started(free, name="Free running")
    worker = make_worker("worker-a")
    worker.step()
    wait_for_state(queued, running)
    started(free, name="Free waiting")
    started(pro, name="Pro waiting")
    TrainingRun.objects.filter(name="Pro waiting").update(
        queued_at=timezone.now() - timedelta(minutes=5)
    )

    out = StringIO()
    call_command("training_queue", stdout=out)
    text = out.getvalue()
    assert "Workers (1 slot online)" in text
    assert "worker-a: online" in text and "1 slot(s)" in text and "Free running" in text
    assert "Running (1)" in text and "free-user (Free, priority 0)" in text
    assert "Waiting (2)" in text
    lines = text.splitlines()
    first = next(line for line in lines if line.strip().startswith("1. "))
    second = next(line for line in lines if line.strip().startswith("2. "))
    assert "Pro waiting  pro-user (Pro, priority 3)  waiting 5.0 min" in first
    assert "Free waiting" in second and "waits for a training of the same account" in second
    assert "trainings started: 1; wait for a slot: median" in text
    assert "worker utilisation:" in text

    admin = User.objects.create_superuser("root", password="correct-horse-battery")
    client.force_login(admin)
    response = client.get("/admin/brick_dnn_core/trainingworker/")
    assert response.status_code == 200
    page = response.content.decode()
    assert "Training queue and workers" in page
    for expected in (
        "Pro waiting",
        "Free waiting",
        "Free running",
        "pro-user",
        "worker-a",
        "Worker utilisation",
        "95th percentile",
        "waits for a training of the same",
    ):
        assert expected in page, expected
    assert page.index("Pro waiting") < page.index("Free waiting")
    assert client.get("/admin/brick_dnn_core/trainingattempt/").status_code == 200
    run_page = client.get(f"/admin/brick_dnn_core/trainingrun/{running}/change/")
    assert run_page.status_code == 200

    # A dead worker and its run: the command says so, and --recover puts the run back.
    kill(running)
    beat_long_ago(worker)
    out = StringIO()
    call_command("training_queue", stdout=out)
    assert "No training worker is running" in out.getvalue()
    assert "[its worker is gone]" in out.getvalue() and "no heartbeat" in out.getvalue()
    out = StringIO()
    call_command("training_queue", "--recover", "--days", "30", stdout=out)
    assert "Recovered “Free running”" in out.getvalue() and "requeued" in out.getvalue()
    assert "Last 30 days" in out.getvalue() and "restart 1" in out.getvalue()
    response = client.get("/admin/brick_dnn_core/trainingworker/")
    assert "No training worker is running" in response.content.decode()
