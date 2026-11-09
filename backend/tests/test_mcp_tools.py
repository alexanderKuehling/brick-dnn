"""MCP tools that delete, upload, download and predict, with the roles of shared projects."""

# ruff: noqa: F811 (the fixture training_dir is imported from test_training_api)

from __future__ import annotations

import signal
import time

import pytest
from django.contrib.auth import get_user_model

from brick_dnn_core import live, training_service
from brick_dnn_core.models import Dataset, Project, ProjectMember, TrainingRun

from .test_mcp import call, mcp_client, rpc
from .test_training_api import (  # noqa: F401 (training_dir is a fixture)
    DEMO_GRAPH,
    QUICK,
    RANDOM,
    iris_csv,
    training_dir,
    upload,
    use_runner,
    wait_for_state,
)

pytestmark = pytest.mark.django_db

IRIS = iris_csv().decode()

DESTRUCTIVE_TOOLS = {"delete_project", "delete_training_run", "delete_dataset"}
WRITING_TOOLS = {*DESTRUCTIVE_TOOLS, "upload_dataset"}


@pytest.fixture
def team():
    """ada owns the project, bob edits it, cy views it, eve is a stranger."""
    users = {
        name: get_user_model().objects.create_user(name, password="correct-horse-1")
        for name in ("ada", "bob", "cy", "eve")
    }
    project = Project.objects.create(
        owner=users["ada"], name="Shared", graph=DEMO_GRAPH, updated_by=users["ada"]
    )
    ProjectMember.objects.create(project=project, user=users["bob"], role="editor")
    ProjectMember.objects.create(project=project, user=users["cy"], role="viewer")
    return project, users


@pytest.fixture
def mcps(team):
    _, users = team
    return {name: mcp_client(user) for name, user in users.items()}


def error_of(client, tool, **arguments) -> str:
    result, text = call(client, tool, **arguments)
    assert result.get("isError") is True, text
    return text


def ok(client, tool, **arguments) -> dict:
    result, text = call(client, tool, **arguments)
    assert "isError" not in result, text
    return result["structuredContent"]


def start_run(client, project_id=None, name="Run") -> str:
    arguments = {"graph": DEMO_GRAPH, "dataset": RANDOM, "config": QUICK, "name": name}
    if project_id is not None:
        arguments["project_id"] = project_id
    return ok(client, "start_training", **arguments)["id"]


def wait_until_done(client, run_id: str) -> dict:
    deadline = time.monotonic() + 15
    while True:
        run = ok(client, "get_training_run", id=run_id)
        if run["status"] in ("completed", "stopped", "failed"):
            return run
        assert time.monotonic() < deadline, run
        time.sleep(0.05)


# --------------------------------------------------------------------------- #
# Read-only access
# --------------------------------------------------------------------------- #


def test_read_only_tokens_do_not_get_the_writing_tools(team):
    _, users = team
    client = mcp_client(users["ada"], read_only=True)
    tools = {t["name"]: t for t in rpc(client, "tools/list").json()["result"]["tools"]}
    assert not set(tools) & WRITING_TOOLS
    full = {
        t["name"]: t for t in rpc(mcp_client(users["ada"]), "tools/list").json()["result"]["tools"]
    }
    for name in WRITING_TOOLS:
        assert full[name]["annotations"]["destructiveHint"] is (name in DESTRUCTIVE_TOOLS)
        assert full[name]["annotations"]["readOnlyHint"] is False
    project, _ = team
    assert "read-only" in error_of(client, "delete_project", id=project.pk)
    assert Project.objects.filter(pk=project.pk).exists()
    assert "read-only" in error_of(client, "upload_dataset", name="Iris", csv=IRIS)
    assert not Dataset.objects.exists()


# --------------------------------------------------------------------------- #
# delete_project
# --------------------------------------------------------------------------- #


def test_only_the_owner_deletes_a_project(team, mcps, monkeypatch):
    project, _ = team
    notified = []
    monkeypatch.setattr(live, "notify_deleted", notified.append)

    text = error_of(mcps["bob"], "delete_project", id=project.pk)
    assert "Only its owner (ada)" in text and "an editor" in text and "Leave" in text
    assert "a viewer" in error_of(mcps["cy"], "delete_project", id=project.pk)
    assert "No such project" in error_of(mcps["eve"], "delete_project", id=project.pk)
    assert "number" in error_of(mcps["ada"], "delete_project", id="first")
    assert Project.objects.filter(pk=project.pk).exists() and notified == []

    result = ok(mcps["ada"], "delete_project", id=project.pk)
    assert result["deleted"] is True and result["name"] == "Shared"
    assert not Project.objects.filter(pk=project.pk).exists()
    assert notified == [project.pk]  # live editors are told, as with the REST API
    assert "No such project" in error_of(mcps["ada"], "delete_project", id=project.pk)


def test_deleting_a_project_keeps_its_runs(team, mcps, training_dir, settings):
    settings.TRAINING_MAX_RUNS = 5
    project, users = team
    run_id = start_run(mcps["bob"], project.pk)
    wait_until_done(mcps["bob"], run_id)
    ok(mcps["ada"], "delete_project", id=project.pk)
    run = TrainingRun.objects.get(pk=run_id)
    assert run.project is None and run.owner == users["bob"]
    assert "No such training run" in error_of(mcps["ada"], "get_training_run", id=run_id)


# --------------------------------------------------------------------------- #
# delete_training_run
# --------------------------------------------------------------------------- #


def test_delete_training_run_permissions(
    team,
    mcps,
    training_dir,
    settings,
    django_capture_on_commit_callbacks,
):
    settings.TRAINING_MAX_RUNS = 5
    project, _ = team
    first = start_run(mcps["bob"], project.pk, "First")
    second = start_run(mcps["bob"], project.pk, "Second")
    for run_id in (first, second):
        wait_until_done(mcps["bob"], run_id)

    # Viewers and other editors can't, strangers don't even see it.
    assert "Only who started this run (bob)" in error_of(
        mcps["cy"], "delete_training_run", id=first
    )
    assert "No such training run" in error_of(mcps["eve"], "delete_training_run", id=first)
    assert "No such training run" in error_of(mcps["ada"], "delete_training_run", id="nope")

    # Who started it and the project's owner can.
    with django_capture_on_commit_callbacks(execute=True):
        result = ok(mcps["bob"], "delete_training_run", id=first)
    assert result == {
        "deleted": True,
        "id": first,
        "name": "First",
        "killed_running_training": False,
    }
    assert not (training_dir / "runs" / first).exists()
    with django_capture_on_commit_callbacks(execute=True):
        ok(mcps["ada"], "delete_training_run", id=second)
    assert not TrainingRun.objects.exists()


def test_delete_training_run_kills_an_active_run(
    team,
    mcps,
    training_dir,
    settings,
    django_capture_on_commit_callbacks,
):
    use_runner(settings, "hang")
    run_id = start_run(mcps["ada"])
    process = training_service._PROCESSES[run_id]
    wait_for_state(training_dir, run_id)
    with django_capture_on_commit_callbacks(execute=True):
        result = ok(mcps["ada"], "delete_training_run", id=run_id)
    assert result["killed_running_training"] is True
    assert process.wait(timeout=5) == -signal.SIGKILL
    assert not (training_dir / "runs" / run_id).exists()


# --------------------------------------------------------------------------- #
# delete_dataset
# --------------------------------------------------------------------------- #


def test_delete_dataset(team, mcps, training_dir, settings, django_capture_on_commit_callbacks):
    from rest_framework.test import APIClient

    project, users = team
    client = APIClient()
    client.force_authenticate(users["bob"])
    dataset_id = upload(client, "iris.csv", iris_csv()).data["id"]
    Dataset.objects.get(pk=dataset_id).projects.add(project)

    # Members see it but only its owner deletes it.
    assert "Only its owner (bob)" in error_of(mcps["ada"], "delete_dataset", id=dataset_id)
    assert "No such dataset" in error_of(mcps["eve"], "delete_dataset", id=dataset_id)

    # Not while a training uses it.
    use_runner(settings, "until_stop")
    run_id = ok(
        mcps["bob"],
        "start_training",
        graph=DEMO_GRAPH,
        dataset={"kind": "upload", "id": dataset_id},
        config=QUICK,
    )["id"]
    text = error_of(mcps["bob"], "delete_dataset", id=dataset_id)
    assert "still running" in text and "stop_training" in text
    ok(mcps["bob"], "stop_training", id=run_id)
    wait_until_done(mcps["bob"], run_id)

    with django_capture_on_commit_callbacks(execute=True):
        result = ok(mcps["bob"], "delete_dataset", id=dataset_id)
    assert result == {"deleted": True, "id": dataset_id, "name": "iris"}
    assert not Dataset.objects.exists()
    assert not (training_dir / "datasets" / str(dataset_id)).exists()
    assert TrainingRun.objects.get(pk=run_id).upload is None


def test_rest_and_mcp_share_the_delete_code(team, training_dir, monkeypatch):
    """The REST API deletes projects through the same function (it notifies live editors)."""
    from rest_framework.test import APIClient

    project, users = team
    notified = []
    monkeypatch.setattr(live, "notify_deleted", notified.append)
    client = APIClient()
    client.force_authenticate(users["ada"])
    assert client.delete(f"/api/projects/{project.pk}/").status_code == 204
    assert notified == [project.pk]


# --------------------------------------------------------------------------- #
# upload_dataset
# --------------------------------------------------------------------------- #


def test_upload_dataset(team, mcps, training_dir):
    result = ok(mcps["ada"], "upload_dataset", name="Iris flowers", csv=IRIS)
    dataset = result["dataset"]
    assert result["reference"] == {"kind": "upload", "id": dataset["id"]}
    assert dataset["name"] == "Iris flowers" and dataset["kind"] == "csv"
    assert dataset["mine"] is True and dataset["shared_in"] == []
    assert dataset["info"]["num_features"] == 4
    assert dataset["info"]["class_names"] == ["setosa", "versicolor", "virginica"]
    assert (training_dir / "datasets" / str(dataset["id"]) / "data.csv").read_text() == IRIS

    # It is listed like an upload in the web app and can be trained on.
    options = ok(mcps["ada"], "get_training_options")
    assert [d["id"] for d in options["your_datasets"]] == [dataset["id"]]
    assert options["upload_dataset_max_mb"] == 5
    graph = {
        "nodes": [
            {"id": "in", "type": "Input", "params": {"shape": [4]}},
            {"id": "fc", "type": "Linear", "params": {"out_features": 3}},
        ],
        "edges": [{"source": "in", "target": "fc"}],
    }
    check = ok(mcps["ada"], "check_training", graph=graph, dataset=result["reference"])
    assert check["ok"] is True and check["plan"]["task"] == "classification"


def test_upload_dataset_into_a_project(team, mcps, training_dir):
    project, _ = team
    result = ok(mcps["bob"], "upload_dataset", name="Iris", csv=IRIS, project_id=project.pk)
    assert result["dataset"]["shared_in"] == [{"id": project.pk, "name": "Shared"}]
    # Members of the project see it.
    seen = ok(mcps["cy"], "get_training_options")["your_datasets"]
    assert [(d["name"], d["owner"], d["mine"]) for d in seen] == [("Iris", "bob", False)]

    # Viewers and strangers cannot share in it; nothing is stored then.
    for name in ("cy", "eve"):
        text = error_of(mcps[name], "upload_dataset", name="X", csv=IRIS, project_id=project.pk)
        assert "projects the user owns or edits" in text
    assert "number" in error_of(mcps["bob"], "upload_dataset", name="X", csv=IRIS, project_id="1")
    assert Dataset.objects.count() == 1


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"name": "Tiny", "csv": "a,b\n1,2\n"}, "at least 4 data rows"),
        ({"name": "Text", "csv": "a,b\n1,x\nzz,2\n3,4\n5,6\n"}, "is not a number"),
        ({"name": "One column", "csv": "a\n1\n2\n3\n4\n"}, "at least two columns"),
        ({"name": "Empty", "csv": "  "}, "content of the CSV file"),
        ({"name": "Not text", "csv": 5}, "content of the CSV file"),
        ({"name": "", "csv": IRIS}, "'name'"),
        ({"name": "x" * 121, "csv": IRIS}, "120 characters"),
        ({"name": "Missing"}, "Missing argument: csv"),
    ],
)
def test_upload_dataset_rejects_bad_csv(team, mcps, training_dir, arguments, message):
    text = error_of(mcps["ada"], "upload_dataset", **arguments)
    assert message in text, text
    assert not Dataset.objects.exists()
    assert not list((training_dir / "datasets").glob(".upload-*"))


def test_upload_dataset_limits(team, mcps, training_dir, settings):
    settings.MCP_UPLOAD_MAX_MB = 0.0001  # about 100 bytes
    text = error_of(mcps["ada"], "upload_dataset", name="Iris", csv=IRIS)
    assert "at most 0.0 MB" in text and "web app" in text
    # The training upload limit applies too.
    settings.MCP_UPLOAD_MAX_MB = 5
    settings.TRAINING_MAX_UPLOAD_MB = 0.0001
    assert "at most" in error_of(mcps["ada"], "upload_dataset", name="Iris", csv=IRIS)
    assert ok(mcps["ada"], "get_training_options")["upload_dataset_max_mb"] == 0.0
    assert not Dataset.objects.exists()


def test_mcp_requests_may_carry_larger_csv_files(team, mcps, training_dir, settings):
    """Up to MCP_UPLOAD_MAX_MB (5 MB) instead of the usual 1 MB for JSON requests."""
    assert settings.DATA_UPLOAD_MAX_MEMORY_SIZE == 1024 * 1024
    rows = ["a,b,c,target"] + [f"{i},{i % 7}.5,{i % 3},{i % 2}" for i in range(100_000)]
    big = "\n".join(rows) + "\n"
    assert 1.2 * 1024**2 < len(big) < 2 * 1024**2
    result = ok(mcps["ada"], "upload_dataset", name="Big", csv=big)
    assert result["dataset"]["info"]["num_samples"] == 100_000

    # Larger requests are refused with a JSON-RPC error, before they are read.
    settings.MCP_UPLOAD_MAX_MB = 0.1
    response = rpc(
        mcps["ada"],
        "tools/call",
        {"name": "upload_dataset", "arguments": {"name": "Big", "csv": big}},
    )
    assert response.status_code == 413
    assert "at most 1.0 MB" in response.json()["error"]["message"]
    # Other requests still work.
    assert rpc(mcps["ada"], "ping").status_code == 200


def test_upload_dataset_when_training_is_disabled_or_rate_limited(
    team, mcps, training_dir, settings, monkeypatch
):
    from rest_framework.throttling import ScopedRateThrottle

    settings.TRAINING_ENABLED = False
    assert "disabled" in error_of(mcps["ada"], "upload_dataset", name="Iris", csv=IRIS)
    settings.TRAINING_ENABLED = True
    monkeypatch.setitem(ScopedRateThrottle.THROTTLE_RATES, "uploads", "1/hour")
    ok(mcps["ada"], "upload_dataset", name="Iris", csv=IRIS)
    assert "Too many" in error_of(mcps["ada"], "upload_dataset", name="Iris", csv=IRIS)
    assert Dataset.objects.count() == 1
