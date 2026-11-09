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

WRITING_TOOLS = {"delete_project", "delete_training_run", "delete_dataset"}


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
        assert full[name]["annotations"]["destructiveHint"] is True
        assert full[name]["annotations"]["readOnlyHint"] is False
    project, _ = team
    assert "read-only" in error_of(client, "delete_project", id=project.pk)
    assert Project.objects.filter(pk=project.pk).exists()


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
