"""Runs that ran elsewhere: the logging API (brick_dnn_core/tracking.py), its tokens, limits
and permissions, how the runs appear in the table, MCP and the export, and the client the
generated script embeds (brickdnn/tracking.py)."""

from __future__ import annotations

import io
import json
import zipfile
from datetime import timedelta

import pytest
from django.contrib.auth import get_user_model
from django.utils import timezone
from rest_framework.test import APIClient
from rest_framework.throttling import ScopedRateThrottle

from brick_dnn_core import training_queue
from brick_dnn_core import training_service as training
from brick_dnn_core.accounts import export_account
from brick_dnn_core.api_tokens import create_token
from brick_dnn_core.models import (
    OAuthClient,
    OAuthConnection,
    Project,
    ProjectMember,
    TrainingRun,
    UsageRecord,
)
from brick_dnn_core.oauth import _issue_tokens

from .helpers import chain

pytestmark = pytest.mark.django_db

GRAPH = chain(("Input", {"shape": [4]}), ("Linear", {"out_features": 3}))


def make_user(name: str):
    return get_user_model().objects.create_user(name, email=f"{name}@example.com", password="x")


@pytest.fixture(autouse=True)
def training_dir(settings, tmp_path):
    settings.TRAINING_DIR = tmp_path / "training"
    return settings.TRAINING_DIR


@pytest.fixture
def ada():
    return make_user("ada")


def token_client(user, **kwargs) -> APIClient:
    _, key = create_token(user, "laptop", **kwargs)
    client = APIClient()
    client.credentials(HTTP_AUTHORIZATION=f"Bearer {key}")
    return client


@pytest.fixture
def script(ada):
    return token_client(ada, scope="runs:log")


def start(http, **body):
    return http.post("/api/tracking/runs/", body, format="json")


def points(n: int, start_epoch: int = 1):
    return [
        {
            "epoch": e,
            "time": e * 1.5,
            "train_loss": 1 / e,
            "val_loss": 1 / e + 0.1,
            "val_accuracy": 0.5,
        }
        for e in range(start_epoch, start_epoch + n)
    ]


def log(client, run_id, batch):
    return client.post(f"/api/tracking/runs/{run_id}/log/", {"points": batch}, format="json")


# --------------------------------------------------------------------------- #
# The life of a run
# --------------------------------------------------------------------------- #


def test_a_script_logs_a_run_that_appears_next_to_the_others(ada, script):
    project = Project.objects.create(owner=ada, name="Iris net", graph=GRAPH, updated_by=ada)
    created = start(
        script,
        name="MLP on my GPU",
        project=project.pk,
        dataset="iris.csv",
        task="classification",
        epochs=3,
        config={"learning_rate": 0.01, "batch_size": 32, "optimizer": "adam", "loss": "focal"},
        client="brickdnn-tracking/1",
    )
    assert created.status_code == 201, created.json()
    data = created.json()
    assert data["status"] == "running" and data["points"] == 0
    assert data["limits"]["max_points"] == 20000 and data["limits"]["max_batch"] == 1000
    run_id = data["id"]
    assert log(script, run_id, points(2)).json()["points"] == 2
    assert log(script, run_id, points(1, 3)).json()["points"] == 3
    patched = script.patch(
        f"/api/tracking/runs/{run_id}/",
        {"config": {"seed": 7}, "summary": {"val_f1": 0.8}},
        format="json",
    )
    assert patched.status_code == 200
    finished = script.post(
        f"/api/tracking/runs/{run_id}/finish/",
        {"status": "completed", "summary": {"test_accuracy": 0.93, "test_loss": 0.2}},
        format="json",
    )
    assert finished.json()["status"] == "completed"

    run = TrainingRun.objects.get(pk=run_id)
    assert run.kind == "external" and run.project == project and run.graph == GRAPH
    assert run.dataset == {"kind": "external", "id": None, "name": "iris.csv"}
    assert run.config == {
        "learning_rate": 0.01,
        "batch_size": 32,
        "optimizer": "adam",
        "loss": "focal",
        "epochs": 3,
        "seed": 7,
    }
    assert run.epochs_done == 3 and run.metrics["val_loss"] == pytest.approx(1 / 3 + 0.1)
    assert run.metrics["test_accuracy"] == 0.93
    assert run.cpu_seconds == 0 and not UsageRecord.objects.exists()  # no server compute
    assert run.size_bytes > 0  # its points count toward the storage quota

    web = APIClient()
    web.force_authenticate(ada)
    row = web.get("/api/training/runs/table/").json()["results"][0]
    assert row["kind"] == "external" and row["external"]["points"] == 3
    assert row["external"]["client"] == "brickdnn-tracking/1"
    assert row["metrics"]["test_accuracy"] == 0.93 and row["metrics"]["val_f1"] == 0.8
    assert row["settings"]["learning_rate"] == 0.01 and row["loss"] == "focal"
    assert web.get("/api/training/runs/table/", {"family": "external"}).json()["count"] == 1
    assert web.get("/api/training/runs/table/", {"family": "neural"}).json()["count"] == 0
    curves = web.get("/api/training/runs/curves/", {"ids": run_id}).json()["runs"][0]
    assert [p["epoch"] for p in curves["points"]] == [1, 2, 3]
    assert curves["points"][2]["time"] == 4.5 and curves["x_unit"] == "epoch"
    detail = web.get(f"/api/training/runs/{run_id}/").json()
    assert detail["external"]["points"] == 3 and detail["has_weights"] is False
    assert len(detail["state"]["history"]) == 3 and detail["state"]["test"]["accuracy"] == 0.93
    assert detail["kind"] == "external"


def test_points_by_step_and_gaps(ada, script):
    run_id = start(script).json()["id"]
    batch = [{"step": 10, "train_loss": 2.0}, {"step": 20, "train_loss": None, "custom.metric": 3}]
    assert log(script, run_id, batch).status_code == 200
    web = APIClient()
    web.force_authenticate(ada)
    curve = web.get("/api/training/runs/curves/", {"ids": run_id}).json()["runs"][0]
    assert curve["x_unit"] == "step" and [p["x"] for p in curve["points"]] == [10, 20]
    run = TrainingRun.objects.get(pk=run_id)
    assert run.name == "Run from a script" and run.dataset["name"] == "Elsewhere"


# --------------------------------------------------------------------------- #
# Who may log
# --------------------------------------------------------------------------- #


def test_only_personal_api_tokens_log(ada):
    assert start(APIClient()).status_code == 401
    web = APIClient()
    web.force_authenticate(ada)  # a website login
    assert start(web).status_code == 403
    assert start(token_client(ada, read_only=True)).status_code == 403
    client_app = OAuthClient.objects.create(client_id="id-c", name="Claude", redirect_uris=[])
    connection = OAuthConnection.objects.create(
        owner=ada,
        client=client_app,
        refresh_hash="hash",
        refresh_expires_at=timezone.now() + timedelta(days=1),
    )
    app = APIClient()
    app.credentials(HTTP_AUTHORIZATION=f"Bearer {_issue_tokens(connection)['access_token']}")
    assert start(app).status_code in (401, 403)
    assert start(token_client(ada)).status_code == 201  # a full token may log too


def test_a_logging_token_can_do_nothing_else(ada, script):
    assert start(script).status_code == 201
    assert script.get("/api/projects/").status_code == 403
    assert script.get("/api/training/runs/").status_code == 403
    mcp = script.post(
        "/api/mcp/", {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, format="json"
    )
    assert mcp.status_code in (401, 403)


def test_tokens_are_created_with_the_logging_scope(ada):
    web = APIClient()
    web.force_authenticate(ada)
    created = web.post(
        "/api/auth/api-tokens/", {"name": "GPU box", "scope": "runs:log"}, format="json"
    )
    assert created.status_code == 201 and created.json()["scope"] == "runs:log"
    assert web.get("/api/auth/api-tokens/").json()[0]["scope"] == "runs:log"
    bad = web.post("/api/auth/api-tokens/", {"name": "x", "scope": "admin"}, format="json")
    assert bad.status_code == 400


def test_runs_of_others_and_of_the_server_are_not_reachable(ada, script):
    bob = make_user("bob")
    theirs = start(token_client(bob)).json()["id"]
    assert log(script, theirs, points(1)).status_code == 404
    trained_here = TrainingRun.objects.create(owner=ada, name="here", status="running")
    assert log(script, trained_here.pk, points(1)).status_code == 404


def test_the_network_must_be_one_the_user_owns_or_edits(ada, script):
    bob = make_user("bob")
    theirs = Project.objects.create(owner=bob, name="Bob's", graph=GRAPH, updated_by=bob)
    assert start(script, project=theirs.pk).status_code == 400
    ProjectMember.objects.create(project=theirs, user=ada, role="viewer")
    assert start(script, project=theirs.pk).status_code == 400
    ProjectMember.objects.filter(project=theirs).update(role="editor")
    run = start(script, project=theirs.pk)
    assert run.status_code == 201
    # Bob, the network's owner, sees it as a run of his network.
    web = APIClient()
    web.force_authenticate(bob)
    rows = web.get("/api/training/runs/table/", {"project": theirs.pk}).json()["results"]
    assert [r["id"] for r in rows] == [run.json()["id"]]
    assert rows[0]["can_annotate"] is True and rows[0]["started_by"] == "ada"


# --------------------------------------------------------------------------- #
# Checking what arrives
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "batch",
    [
        [],
        "points",
        [{"train_loss": 1}],  # neither epoch nor step
        [{"epoch": -1}],
        [{"epoch": 1.5}],
        [{"epoch": True}],
        [{"epoch": 1, "val loss": 1}],  # not a metric name
        [{"epoch": 1, "1st": 1}],
        [{"epoch": 1, "loss": "low"}],
        [{"epoch": 1, "loss": 1e20}],
        [{"epoch": 1, "time": -3}],
        [{"epoch": 1, **{f"m{i}": i for i in range(33)}}],
        ["not an object"],
    ],
)
def test_bad_points_are_refused(script, batch):
    run_id = start(script).json()["id"]
    assert log(script, run_id, batch).status_code == 400


@pytest.mark.parametrize(
    "body",
    [
        {"task": "clustering"},
        {"epochs": 0},
        {"project": "3"},
        {"config": "lr=1"},
        {"config": {"a": {"b": {"c": {"d": {"e": 1}}}}}},
        {"name": 5},
    ],
)
def test_bad_runs_are_refused(script, body):
    assert start(script, **body).status_code == 400


def test_a_large_config_or_body_is_refused(script):
    assert start(script, config={"text": ["x" * 400] * 60}).status_code == 413
    run_id = start(script).json()["id"]
    many = [{"epoch": i, "loss": 0.5} for i in range(1001)]
    answer = log(script, run_id, many)
    assert answer.status_code == 413 and answer.json()["code"] == "too_large"
    huge = json.dumps({"points": [{"epoch": 1, "loss": 1}], "pad": "x" * 600_000})
    answer = script.post(f"/api/tracking/runs/{run_id}/log/", huge, content_type="application/json")
    assert answer.status_code == 413


# --------------------------------------------------------------------------- #
# Limits
# --------------------------------------------------------------------------- #


def test_points_per_run_are_limited(settings, script):
    settings.TRACKING_MAX_POINTS = 5
    run_id = start(script).json()["id"]
    assert log(script, run_id, points(4)).status_code == 200
    refused = log(script, run_id, points(2, 5))
    assert refused.status_code == 409 and refused.json()["code"] == "points_limit"
    assert log(script, run_id, points(1, 5)).status_code == 200


def test_runs_per_account_and_at_a_time_are_limited(settings, ada, script):
    settings.TRACKING_MAX_ACTIVE = 2
    settings.TRACKING_MAX_RUNS = 3
    first = start(script).json()["id"]
    start(script)
    refused = start(script)
    assert refused.status_code == 409 and refused.json()["code"] == "active_limit"
    script.post(f"/api/tracking/runs/{first}/finish/", {"status": "completed"}, format="json")
    assert start(script).status_code == 201
    refused = start(script)
    assert refused.status_code == 409  # 3 kept, 2 of them logging
    # Runs trained on the server do not count.
    TrainingRun.objects.filter(owner=ada, kind="external", status="running").update(
        status="completed"
    )
    assert start(script).json()["code"] == "runs_limit"


def test_the_storage_quota_counts_the_points(settings, script):
    settings.PLANS = None
    settings.STAFF_UNLIMITED = True
    run_id = start(script).json()["id"]
    settings.BUDGET_STORAGE_MB = 0.0001  # 100 bytes
    refused = log(script, run_id, points(20))
    assert refused.status_code == 403 and refused.json()["code"] == "storage_quota_exceeded"
    assert start(script).status_code in (201, 403)


def test_the_requests_are_rate_limited(monkeypatch, script):
    monkeypatch.setitem(ScopedRateThrottle.THROTTLE_RATES, "tracking", "3/minute")
    run_id = start(script).json()["id"]
    assert log(script, run_id, points(1)).status_code == 200
    assert log(script, run_id, points(1, 2)).status_code == 200
    assert log(script, run_id, points(1, 3)).status_code == 429


# --------------------------------------------------------------------------- #
# Ending a run
# --------------------------------------------------------------------------- #


def test_finished_runs_take_no_more_reports(script):
    run_id = start(script).json()["id"]
    bad = script.post(f"/api/tracking/runs/{run_id}/finish/", {"status": "great"}, format="json")
    assert bad.status_code == 400
    done = script.post(
        f"/api/tracking/runs/{run_id}/finish/",
        {"status": "failed", "message": "CUDA out of memory"},
        format="json",
    )
    assert done.json()["status"] == "failed"
    assert TrainingRun.objects.get(pk=run_id).message == "CUDA out of memory"
    again = log(script, run_id, points(1))
    assert again.status_code == 409 and again.json()["code"] == "finished"
    assert script.post(f"/api/tracking/runs/{run_id}/finish/", {}, format="json").status_code == 409


def test_stopping_it_in_the_app_ends_it(ada, script, django_capture_on_commit_callbacks):
    run_id = start(script).json()["id"]
    web = APIClient()
    web.force_authenticate(ada)
    stopped = web.post(f"/api/training/runs/{run_id}/stop/")
    assert stopped.json()["status"] == "stopped"
    assert log(script, run_id, points(1)).status_code == 409
    with django_capture_on_commit_callbacks(execute=True):
        assert web.delete(f"/api/training/runs/{run_id}/").status_code == 204
    assert not training.run_dir(run_id).exists()


def test_a_run_that_stops_reporting_fails_after_a_day(ada, script):
    run_id = start(script).json()["id"]
    run = TrainingRun.objects.get(pk=run_id)
    old = (timezone.now() - timedelta(hours=25)).isoformat()
    run.plan = {**run.plan, "external": {**run.plan["external"], "reported_at": old}}
    run.save(update_fields=["plan"])
    web = APIClient()
    web.force_authenticate(ada)
    detail = web.get(f"/api/training/runs/{run_id}/").json()
    assert detail["status"] == "failed" and "stopped reporting" in detail["message"]


def test_logged_runs_are_not_trainings_of_the_server(ada, script):
    start(script)
    assert training.active_run_count() == 0
    assert training_queue.snapshot().running == []
    assert not training_queue.lost_runs().exists()


# --------------------------------------------------------------------------- #
# MCP and the data export
# --------------------------------------------------------------------------- #


def test_the_final_numbers_win_over_the_last_point(ada, script):
    """A script whose result is not its last epoch (a classical model fitted on learned
    features) reports it at the end: those numbers are the run's metrics and its row."""
    run_id = start(script, task="classification", epochs=2).json()["id"]
    log(script, run_id, points(2))  # the temporary head: val. accuracy 0.5
    summary = {"val_accuracy": 0.9, "val_balanced_accuracy": 0.88, "fit_seconds": 1.25}
    finished = script.post(
        f"/api/tracking/runs/{run_id}/finish/",
        {"status": "completed", "summary": summary},
        format="json",
    )
    assert finished.status_code == 200
    run = TrainingRun.objects.get(pk=run_id)
    assert run.metrics["val_accuracy"] == 0.9
    assert run.metrics["val_loss"] == pytest.approx(0.6)  # not reported: the last point's
    assert "fit_seconds" not in run.metrics and "val_balanced_accuracy" not in run.metrics
    web = APIClient()
    web.force_authenticate(ada)
    row = web.get("/api/training/runs/table/").json()["results"][0]
    assert row["metrics"]["val_accuracy"] == 0.9
    assert row["metrics"]["val_balanced_accuracy"] == 0.88 and row["metrics"]["fit_seconds"] == 1.25


def mcp_call(client, tool, **arguments):
    response = client.post(
        "/api/mcp/",
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": tool, "arguments": arguments},
            }
        ),
        content_type="application/json",
        HTTP_ACCEPT="application/json, text/event-stream",
    )
    return json.loads(response.json()["result"]["content"][0]["text"])


def test_mcp_lists_and_shows_logged_runs(ada, script):
    run_id = start(script, name="On my laptop", client="brickdnn-tracking/1").json()["id"]
    log(script, run_id, points(3))
    script.post(
        f"/api/tracking/runs/{run_id}/finish/", {"summary": {"test_rmse": 1.5}}, format="json"
    )
    agent = token_client(ada)
    listed = mcp_call(agent, "list_training_runs")["runs"][0]
    assert listed["ran_elsewhere"] == "logged by a script (brickdnn-tracking/1): 3 points"
    shown = mcp_call(agent, "get_training_run", id=run_id)
    assert shown["ran_elsewhere"].startswith("logged by a script")
    assert shown["reported"] == {"test_rmse": 1.5}
    assert shown["has_weights"] is False


def test_the_export_has_the_logged_points(ada, script):
    run_id = start(script).json()["id"]
    log(script, run_id, points(2))
    out = io.BytesIO()
    export_account(ada, out)
    with zipfile.ZipFile(out) as archive:
        runs = json.loads(archive.read("training_runs.json"))
        readme = archive.read("README.txt").decode()
    assert runs[0]["kind"] == "external" and len(runs[0]["points"]) == 2
    assert runs[0]["points"][0]["epoch"] == 1
    assert "logged from your own machine" in readme


def test_the_activity_log_skips_the_points(ada, script):
    from brick_dnn_core.models import AgentAction

    run_id = start(script).json()["id"]
    for e in range(1, 4):
        log(script, run_id, points(1, e))
    script.post(f"/api/tracking/runs/{run_id}/finish/", {}, format="json")
    tools = list(AgentAction.objects.values_list("tool", flat=True))
    assert "tracking-run-log" not in tools
    assert {"tracking-runs", "tracking-run-finish"} <= set(tools)
