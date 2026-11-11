"""The activity log of AI agents: what is logged, redaction, the API, privacy."""

# ruff: noqa: F811 (the fixture training_dir is imported from test_training_api)

from __future__ import annotations

import base64
import dataclasses
import io
import json
import logging
import zipfile
from datetime import timedelta

import pytest
from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.management import call_command
from django.core.signals import request_finished
from django.db import close_old_connections
from django.http import HttpResponse
from django.utils import timezone
from rest_framework.test import APIClient

from brick_dnn_core import agent_activity
from brick_dnn_core.agent_activity import summarize
from brick_dnn_core.api_tokens import create_token
from brick_dnn_core.models import AgentAction, ApiToken, OAuthConnection, Project, ProjectMember

from .test_mcp import GRAPH, rpc
from .test_oauth import sign_in
from .test_training_api import (  # noqa: F401 (training_dir is a fixture)
    DEMO_GRAPH,
    QUICK,
    RANDOM,
    iris_csv,
    training_dir,
)

pytestmark = pytest.mark.django_db

User = get_user_model()
PASSWORD = "correct-horse-battery"
IRIS = iris_csv().decode()
USER_AGENT = "claude-code/2.0.14 (cli)"


# --------------------------------------------------------------------------- #
# Fixtures and helpers
# --------------------------------------------------------------------------- #


@pytest.fixture
def ada():
    return User.objects.create_user("ada", email="ada@example.com", password=PASSWORD)


@pytest.fixture
def mallory():
    return User.objects.create_user("mallory", password="whatever-password-1")


def website(username: str = "ada", password: str = PASSWORD) -> APIClient:
    """Logged in on the website (JWT)."""
    client = APIClient()
    response = client.post(
        "/api/auth/token/", {"username": username, "password": password}, format="json"
    )
    assert response.status_code == 200, response.data
    client.credentials(HTTP_AUTHORIZATION=f"Bearer {response.data['access']}")
    return client


def token_client(user, name: str = "Claude on my laptop", read_only: bool = False):
    token, key = create_token(user, name, read_only=read_only)
    client = APIClient()
    client.credentials(HTTP_AUTHORIZATION=f"Bearer {key}", HTTP_USER_AGENT=USER_AGENT)
    return client, token, key


def call(client, tool, **arguments) -> dict:
    response = rpc(client, "tools/call", {"name": tool, "arguments": arguments})
    assert response.status_code == 200, response.content
    return response.json()["result"]


def ok(client, tool, **arguments) -> dict:
    result = call(client, tool, **arguments)
    assert "isError" not in result, result
    return result.get("structuredContent") or json.loads(result["content"][0]["text"])


def log(user) -> list[AgentAction]:
    """The user's entries, oldest first."""
    return list(AgentAction.objects.filter(owner=user).order_by("id"))


def activity(client, **params) -> dict:
    response = client.get("/api/agent-activity/", params)
    assert response.status_code == 200, response.data
    return response.data


def stored_text(user) -> str:
    """Everything stored about the user's agent actions, as one text."""
    return json.dumps(agent_activity.export_entries(user), default=str)


def entry(user, **fields) -> AgentAction:
    values = {
        "source": "mcp",
        "agent_kind": "token",
        "agent_id": 1,
        "agent_name": "script",
        "tool": "list_projects",
        "summary": "Listed the saved networks",
        **fields,
    }
    return AgentAction.objects.create(owner=user, **values)


# --------------------------------------------------------------------------- #
# MCP tool calls
# --------------------------------------------------------------------------- #


def test_mcp_tool_calls_are_logged_with_who_what_and_how_long(ada):
    client, token, _ = token_client(ada)
    ok(client, "list_layer_types")
    created = ok(client, "save_project", name="House prices", graph=GRAPH)
    ok(client, "save_project", id=created["id"], version=created["version"], graph=DEMO_GRAPH)
    ok(client, "get_project", id=created["id"])

    layers, create, update, opened = log(ada)
    assert layers.source == "mcp" and layers.tool == "list_layer_types"
    assert layers.summary == "Looked up the layer types"
    assert layers.outcome == "ok" and layers.error == "" and layers.writes is False
    assert (layers.agent_kind, layers.agent_id) == ("token", token.pk)
    assert layers.agent_name == "Claude on my laptop"
    assert layers.token_prefix == token.prefix
    assert layers.user_agent == USER_AGENT
    assert 0 <= layers.duration_ms < 10_000
    assert layers.targets == [] and layers.project_id is None

    project = Project.objects.get()
    ref = {"kind": "project", "id": project.pk, "name": "House prices"}
    assert create.writes is True
    assert create.summary == f"Created network “House prices” (#{project.pk}) with 2 layers"
    assert create.targets == [ref] and create.project_id == project.pk
    assert create.arguments == {
        "name": "House prices",
        "graph": "2 layers (Input, Linear), 1 connection",
    }
    assert update.summary == (
        f"Saved network “House prices” (#{project.pk}) with 4 layers, now version 2"
    )
    assert update.arguments["graph"] == "4 layers (Input, Linear ×2, Activation), 3 connections"
    assert update.arguments["id"] == project.pk and update.arguments["version"] == 1
    assert opened.summary == f"Opened network “House prices” (#{project.pk})"
    assert opened.targets == [ref]


def test_failed_tool_calls_are_logged_with_the_message(ada):
    client, _, _ = token_client(ada)
    reader, _, _ = token_client(ada, "Read-only bot", read_only=True)
    assert call(client, "get_project", id=404)["isError"] is True
    assert call(client, "start_training", graph=GRAPH)["isError"] is True  # no dataset
    assert call(reader, "save_project", name="X", graph=GRAPH)["isError"] is True
    response = rpc(client, "tools/call", {"name": "make_coffee", "arguments": {}})
    assert response.json()["error"]["code"] == -32602

    missing, no_dataset, read_only, unknown = log(ada)
    assert missing.outcome == "error"
    assert missing.summary == "Could not open network #404"
    assert missing.error == "No such project. list_projects shows the user's projects."
    assert no_dataset.summary == "Could not start a training on a dataset"
    assert no_dataset.error == "Missing argument: dataset."
    assert no_dataset.writes is True
    assert read_only.agent_name == "Read-only bot"
    assert read_only.error.startswith("This API token is read-only.")
    assert read_only.summary == "Could not create network “X” with 2 layers"
    assert unknown.tool == "make_coffee" and unknown.outcome == "error"
    assert unknown.summary == "Could not call an unknown tool “make_coffee”"
    assert unknown.error == "Unknown tool: make_coffee"


def test_unexpected_errors_are_logged_as_internal_errors(ada, monkeypatch):
    from brick_dnn_core import mcp

    def broken(request, args):
        raise RuntimeError("database on fire")

    monkeypatch.setitem(
        mcp.TOOLS, "list_examples", dataclasses.replace(mcp.TOOLS["list_examples"], run=broken)
    )
    client, _, _ = token_client(ada)
    assert call(client, "list_examples")["isError"] is True
    (failed,) = log(ada)
    assert failed.error == "Internal error in the Brick DNN server."
    assert "fire" not in stored_text(ada)


def test_every_call_of_a_batch_is_logged(ada):
    client, _, _ = token_client(ada)
    batch = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "list_examples"}},
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "list_projects"}},
    ]
    response = client.post("/api/mcp/", json.dumps(batch), content_type="application/json")
    assert response.status_code == 200
    # Only tool calls are actions: initialize and tools/list are not logged.
    assert [e.tool for e in log(ada)] == ["list_examples", "list_projects"]
    assert log(ada)[1].summary == "Listed 0 saved networks"


def test_mcp_requests_that_never_reach_a_tool(ada, settings):
    client, _, _ = token_client(ada)
    response = client.post("/api/mcp/", "{not json", content_type="application/json")
    assert response.status_code == 400
    (broken,) = log(ada)
    assert broken.source == "mcp" and broken.tool == "mcp" and broken.outcome == "error"
    assert broken.summary == "Could not send a request to the MCP server"
    assert broken.error.startswith("Parse error")


def test_apps_connected_with_oauth_are_logged_by_connection(ada):
    _, tokens = sign_in(website(), client_name="Claude")
    app = APIClient()
    app.credentials(
        HTTP_AUTHORIZATION=f"Bearer {tokens['access_token']}", HTTP_USER_AGENT="Claude-User"
    )
    ok(app, "list_projects")
    assert call(app, "stop_training", id="not-a-run")["isError"] is True
    # App tokens only work for MCP: using one elsewhere is refused, and logged.
    assert app.get("/api/projects/").status_code == 401

    connection = OAuthConnection.objects.get()
    listed, stop, rest = log(ada)
    for item in (listed, stop, rest):
        assert (item.agent_kind, item.agent_id, item.agent_name) == ("app", connection.pk, "Claude")
        assert item.user_agent == "Claude-User"
    assert listed.outcome == "ok"
    assert stop.outcome == "error" and stop.summary == "Could not stop a training run"
    assert stop.error.startswith("No such training run")
    assert rest.source == "api" and rest.status_code == 401
    assert rest.error == "This token is only valid for the MCP server."

    # Disconnecting the app keeps its entries until they expire.
    browser = website()
    assert browser.delete(f"/api/auth/connections/{connection.pk}/").status_code == 204
    assert len(log(ada)) == 3
    summary = browser.get("/api/agent-activity/summary/").data
    assert summary["agents"] == [
        {
            "kind": "app",
            "id": connection.pk,
            "key": f"app:{connection.pk}",
            "name": "Claude",
            "prefix": None,
            "active": False,
            "actions": 3,
            "errors": 2,
            "last_at": summary["agents"][0]["last_at"],
            "last_id": rest.pk,
        }
    ]


# --------------------------------------------------------------------------- #
# REST requests
# --------------------------------------------------------------------------- #


def test_rest_requests_with_a_token_are_logged(ada):
    client, token, _ = token_client(ada, "Nightly script")
    created = client.post("/api/projects/", {"name": "Churn", "graph": GRAPH}, format="json")
    assert created.status_code == 201
    project_id = created.data["id"]
    assert client.get("/api/projects/").status_code == 200
    assert (
        client.patch(
            f"/api/projects/{project_id}/", {"name": "Churn v2"}, format="json"
        ).status_code
        == 200
    )
    assert client.post("/api/analyze/", {"graph": GRAPH}, format="json").status_code == 200
    assert client.get("/api/projects/999/").status_code == 404
    assert client.get("/api/layers/?q=1").status_code == 200
    assert client.delete(f"/api/projects/{project_id}/").status_code == 204

    create, listed, renamed, analyzed, missing, layers, deleted = log(ada)
    assert create.source == "api" and create.method == "POST" and create.tool == "project-list"
    assert create.status_code == 201 and create.writes is True
    assert create.summary == f"Created network “Churn” (#{project_id}) with 2 layers"
    assert create.targets == [{"kind": "project", "id": project_id, "name": "Churn"}]
    assert (create.agent_kind, create.agent_id) == ("token", token.pk)
    assert listed.summary == "Listed 1 saved network" and listed.writes is False
    assert renamed.summary == f"Saved network “Churn v2” (#{project_id}), now version 2"
    assert renamed.arguments == {"id": project_id, "name": "Churn v2"}
    assert analyzed.summary == "Analyzed a network of 2 layers: valid, 10 parameters"
    assert analyzed.writes is False  # a POST that changes nothing
    assert missing.outcome == "error" and missing.status_code == 404
    assert missing.summary == "Could not open network #999"
    assert layers.summary == "Looked up the layer types"
    assert layers.arguments == {"query": {"q": "1"}}
    assert deleted.summary == f"Deleted network #{project_id}"
    assert deleted.project_id == project_id  # kept after the network is gone


def test_refused_rest_requests_are_logged(ada):
    reader, _, _ = token_client(ada, "Read-only", read_only=True)
    assert (
        reader.post("/api/projects/", {"name": "X", "graph": GRAPH}, format="json").status_code
        == 403
    )
    # API tokens cannot manage tokens, nor read the activity log.
    assert reader.get("/api/auth/api-tokens/").status_code == 403
    assert reader.get("/api/agent-activity/").status_code == 403

    create, tokens, own_log = log(ada)
    assert create.outcome == "error" and create.status_code == 403
    assert create.error == "This API token is read-only."
    # Refused before the body was read: it is not read just for the log.
    assert create.summary == "Could not create a network" and create.arguments == {}
    assert tokens.summary == "Could not call GET api-token-list"
    assert own_log.error.startswith("See the agent activity after logging in on the website")


def test_website_requests_are_not_logged(ada):
    browser = website()
    created = browser.post("/api/projects/", {"name": "By hand", "graph": GRAPH}, format="json")
    assert created.status_code == 201
    browser.get("/api/projects/")
    browser.get("/api/agent-activity/")
    APIClient().get("/api/layers/")
    # Bad tokens are not anyone's agent.
    stranger = APIClient()
    stranger.credentials(HTTP_AUTHORIZATION="Bearer bdnn_" + "x" * 43)
    assert stranger.get("/api/projects/").status_code == 401
    assert AgentAction.objects.count() == 0


def test_nothing_is_logged_when_the_log_is_off(ada, settings):
    settings.AGENT_LOG_DAYS = 0
    client, _, _ = token_client(ada)
    ok(client, "list_projects")
    client.get("/api/projects/")
    assert AgentAction.objects.count() == 0
    data = activity(website())
    assert data["retention_days"] == 0 and data["results"] == []


# --------------------------------------------------------------------------- #
# Redaction
# --------------------------------------------------------------------------- #


def test_imported_files_are_stored_as_their_size_not_their_content(ada, settings, tmp_path):
    settings.DATA_PREP_DIR = tmp_path / "prepare"
    settings.DATA_PREP_ENABLED = True
    client, _, _ = token_client(ada)
    ok(client, "import_data", name="Tiny", content="city,pop\nBern,134\n")
    packed = base64.b64encode(b"city,pop\nBasel,173\n").decode()
    ok(client, "import_data", filename="basel.csv", content_base64=packed)

    text, binary = log(ada)
    assert text.arguments == {"name": "Tiny", "content": "18 B, 2 lines, 2 columns"}
    assert binary.arguments == {"filename": "basel.csv", "content_base64": "file, about 21 B"}
    stored = stored_text(ada)
    assert "Bern" not in stored and "Basel" not in stored and packed not in stored


def test_uploads_are_stored_as_their_size_not_their_content(ada, training_dir):
    client, _, _ = token_client(ada)
    result = ok(client, "upload_dataset", name="Iris flowers", csv=IRIS)
    dataset_id = result["dataset"]["id"]
    assert call(client, "upload_dataset", name="Broken", csv="a,b\n1")["isError"] is True

    upload, broken = log(ada)
    size = agent_activity.size_text(len(IRIS.encode()))
    assert upload.arguments == {"name": "Iris flowers", "csv": f"{size}, 13 lines, 5 columns"}
    assert upload.summary == (
        f"Uploaded dataset “Iris flowers” (#{dataset_id}): {size}, 13 lines, 5 columns"
    )
    assert upload.targets == [{"kind": "dataset", "id": dataset_id, "name": "Iris flowers"}]
    assert broken.summary == "Could not upload dataset “Broken”: 5 B, 2 lines, 2 columns"
    assert "setosa" not in stored_text(ada) and "sepal_length" not in stored_text(ada)


def test_prediction_inputs_are_counted_not_stored(ada):
    client, _, _ = token_client(ada)
    inputs = [[1234.5678, 3.456, 1.4, 0.2], [9876.54321, 3.0, 4.8, 1.8]]
    assert call(client, "predict", run_id="00000000-0000-0000-0000-000000000000", inputs=inputs)[
        "isError"
    ]
    (predict,) = log(ada)
    assert predict.arguments["inputs"] == "2 samples with 4 values"
    assert predict.summary == "Could not predict 2 samples with a training run"
    assert "1234.5678" not in stored_text(ada) and "9876.54321" not in stored_text(ada)


DESCRIPTION = "A long description of the network " * 10


def test_secrets_never_reach_the_log(ada):
    client, _, key = token_client(ada)
    other_key = create_token(ada, "other")[1]
    # A token pasted into a name, a password in a body, a signed link, a link with a key.
    ok(client, "save_project", name=f"Net {other_key}", graph=GRAPH)
    client.post(
        "/api/projects/", {"name": "N", "graph": GRAPH, "password": PASSWORD}, format="json"
    )
    client.get("/api/layers/", {"sig": "secret-signature", "token": "abc"})
    stored = stored_text(ada)
    for secret in (key, other_key, PASSWORD, "secret-signature"):
        assert secret not in stored
    assert "[redacted]" in log(ada)[0].summary
    assert log(ada)[1].arguments["password"] == "[redacted]"
    assert log(ada)[2].arguments["query"] == {"sig": "[redacted]", "token": "[redacted]"}

    summary = summarize(
        {
            "url": "https://user:pw@example.com/data.csv?X-Amz-Signature=abc#top",
            "client_secret": "s3cr3t",
            "refresh": "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.sig",
            "description": DESCRIPTION,
            "text": "Comment text",
            "user": "bob@example.com",
            "file": SimpleUploadedFile("houses.csv", b"a,b\n1,2\n"),
            "config": {"epochs": 20, "learning_rate": 0.001, "optimizer": "adam", "api_key": "k"},
            "shape": [3, 28, 28],
            "inputs": [{"x": 1, "y": 2}],
            "anything": "x" * 300,
            "nan": float("nan"),
        }
    )
    assert summary == {
        "url": "https://example.com/data.csv",
        "client_secret": "[redacted]",
        "refresh": "[redacted]",
        "description": f"{len(DESCRIPTION)} characters",
        "text": "12 characters",
        "user": "15 characters",
        "file": "houses.csv, 8 B",
        "config": {
            "epochs": 20,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "api_key": "[redacted]",
        },
        "shape": [3, 28, 28],
        "inputs": "1 sample with 2 columns",
        "anything": "300 characters",
        "nan": "nan",
    }


def test_lengths_are_capped(ada):
    client, _, _ = token_client(ada)
    call(client, "get_project", id="x" * 5000)
    many = {f"key{i}": i for i in range(50)}
    (opened,) = log(ada)
    assert len(opened.arguments["id"]) <= agent_activity.MAX_SHORT
    assert len(summarize(many)) == agent_activity.MAX_KEYS + 1
    assert len(agent_activity.clean_text("y" * 1000, 120)) == 120
    assert agent_activity.clean_text("a\x00b\nc​", 50) == "a b c​"


# --------------------------------------------------------------------------- #
# What an action touched
# --------------------------------------------------------------------------- #


def test_trainings_name_their_run_dataset_and_network(ada, training_dir):
    project = Project.objects.create(owner=ada, name="Shared", graph=DEMO_GRAPH)
    client, _, _ = token_client(ada)
    dataset = ok(client, "upload_dataset", name="houses.csv", csv=IRIS)["dataset"]
    graph = {
        "nodes": [
            {"id": "in", "type": "Input", "params": {"shape": [4]}},
            {"id": "fc", "type": "Linear", "params": {"out_features": 3}},
        ],
        "edges": [{"source": "in", "target": "fc"}],
    }
    reference = {"kind": "upload", "id": dataset["id"]}
    ok(client, "check_training", graph=graph, dataset=reference)
    run = ok(
        client,
        "start_training",
        name="GP 1",
        graph=graph,
        dataset=reference,
        config=QUICK,
        project_id=project.pk,
    )
    ok(client, "get_training_run", id=run["id"])
    ok(client, "stop_training", id=run["id"])

    _, check, start, status, stop = log(ada)
    dataset_ref = {"kind": "dataset", "id": dataset["id"], "name": "houses.csv"}
    run_ref = {"kind": "run", "id": run["id"], "name": "GP 1"}
    project_ref = {"kind": "project", "id": project.pk, "name": "Shared"}
    assert check.summary == (
        f"Checked a network of 2 layers against dataset “houses.csv” (#{dataset['id']}): it fits"
    )
    assert check.targets == [dataset_ref]
    assert start.summary == (
        f"Started training “GP 1” on dataset “houses.csv” (#{dataset['id']}) in network "
        f"“Shared” (#{project.pk}), 1 epoch"
    )
    assert start.targets == [run_ref, dataset_ref, project_ref]
    assert start.project_id == project.pk
    assert start.arguments["config"] == QUICK
    assert start.arguments["dataset"] == {"kind": "upload", "id": dataset["id"]}
    assert status.summary.startswith("Checked training “GP 1”: ")
    assert status.targets == [run_ref] and status.project_id == project.pk
    assert stop.summary == "Stopped training “GP 1”" and stop.writes is True
    assert agent_activity.training_data(ada, RANDOM) == (None, "random demo data")
    assert agent_activity.training_data(ada, {"kind": "builtin", "id": "mnist"}) == (
        None,
        "the built-in dataset “MNIST”",
    )

    # Filtering by the network finds everything about it.
    found = activity(website(), project=project.pk)["results"]
    assert [e["id"] for e in found] == [stop.pk, status.pk, start.pk]


def test_checks_of_a_run_say_how_far_it_is():
    progress = agent_activity._run_progress
    running = {"status": "running", "epochs": 20, "epochs_done": 3}
    assert progress(running) == ": running, epoch 3 of 20"
    search = {"status": "running", "tuning": {"trials": 8}, "trials_done": 2, "epochs": 5}
    assert progress(search) == ": running, trial 2 of 8"
    metrics = {"val_accuracy": 0.9612, "test_accuracy": 0.96667}
    assert progress({"status": "completed", "metrics": metrics}) == (
        ": completed, test accuracy 96.7 %"
    )
    assert progress({"status": "stopped", "metrics": {"val_r2": 0.87654}}) == (
        ": stopped, validation R² 0.877"
    )
    assert progress({"status": "failed", "metrics": None}) == ": failed"


def test_names_are_only_looked_up_where_the_user_can_see_them(ada, mallory):
    secret = Project.objects.create(owner=mallory, name="Mallory's secret", graph=GRAPH)
    client, _, _ = token_client(ada)
    call(client, "get_project", id=secret.pk)
    (opened,) = log(ada)
    assert opened.summary == f"Could not open network #{secret.pk}"
    assert "secret" not in stored_text(ada)

    ProjectMember.objects.create(project=secret, user=ada, role="viewer")
    call(client, "delete_project", id=secret.pk)
    assert log(ada)[1].summary == f"Could not delete network “Mallory's secret” (#{secret.pk})"


# --------------------------------------------------------------------------- #
# The API
# --------------------------------------------------------------------------- #


def test_pages_filters_and_polling(ada):
    now = timezone.now()
    first = [entry(ada, created_at=now - timedelta(hours=3)) for _ in range(3)]
    error = entry(ada, tool="save_project", outcome="error", error="Conflict", agent_id=2)
    app = entry(ada, agent_kind="app", agent_id=7, agent_name="Claude", project_id=12)
    browser = website()

    page = activity(browser, limit=2)
    assert [e["id"] for e in page["results"]] == [app.pk, error.pk]
    assert page["has_more"] is True and page["latest_id"] == app.pk
    assert page["retention_days"] == 90
    older = activity(browser, limit=2, before=error.pk)
    assert [e["id"] for e in older["results"]] == [first[2].pk, first[1].pk]
    assert activity(browser, before=first[0].pk)["results"] == []

    # Polling: only what is newer than the newest one shown.
    assert activity(browser, after=app.pk)["results"] == []
    new = entry(ada, tool="get_training_run")
    polled = activity(browser, after=app.pk)
    assert [e["id"] for e in polled["results"]] == [new.pk] and polled["latest_id"] == new.pk

    def ids(**params) -> list[int]:
        return [e["id"] for e in activity(browser, **params)["results"]]

    assert ids(agent="app:7") == [app.pk]
    assert ids(agent="token:2") == [error.pk]
    assert ids(tool="save_project") == [error.pk]
    assert ids(outcome="error") == [error.pk]
    assert ids(project=12) == [app.pk]
    assert ids(since=(now - timedelta(hours=1)).isoformat()) == [new.pk, app.pk, error.pk]
    assert ids(source="api") == []
    assert ids(agent="app:7", outcome="error") == []
    for bad in ({"agent": "me"}, {"outcome": "maybe"}, {"limit": 0}, {"since": "soon"}):
        assert browser.get("/api/agent-activity/", bad).status_code == 400

    item = activity(browser, limit=1)["results"][0]
    assert set(item) == {
        "id",
        "created_at",
        "source",
        "agent",
        "user_agent",
        "tool",
        "method",
        "writes",
        "outcome",
        "error",
        "status_code",
        "duration_ms",
        "summary",
        "arguments",
        "targets",
        "project",
        "approval",
    }
    assert item["agent"] == {
        "kind": "token",
        "id": 1,
        "key": "token:1",
        "name": "script",
        "prefix": None,
    }


def test_summary_names_the_agents_and_tools(ada):
    client, token, _ = token_client(ada, "Laptop")
    ok(client, "list_projects")
    call(client, "get_project", id=1)
    browser = website()
    summary = browser.get("/api/agent-activity/summary/").data
    (agent,) = summary["agents"]
    assert agent["key"] == f"token:{token.pk}" and agent["name"] == "Laptop"
    assert agent["prefix"] == token.prefix and agent["active"] is True
    assert (agent["actions"], agent["errors"]) == (2, 1)
    assert summary["tools"] == [
        {"tool": "get_project", "actions": 1},
        {"tool": "list_projects", "actions": 1},
    ]
    assert summary["total"] == 2 and summary["latest_id"] == log(ada)[-1].pk

    # Revoking the token keeps its entries.
    assert browser.delete(f"/api/auth/api-tokens/{token.pk}/").status_code == 204
    summary = browser.get("/api/agent-activity/summary/").data
    assert summary["agents"][0]["active"] is False and summary["total"] == 2


def test_everyone_sees_only_their_own_agents(ada, mallory):
    project = Project.objects.create(owner=ada, name="Shared", graph=GRAPH)
    ProjectMember.objects.create(project=project, user=mallory, role="editor")
    client, _, _ = token_client(mallory, "Mallory's bot")
    ok(client, "get_project", id=project.pk)
    entry(ada)

    # Not even the owner of the network sees what another member's agent did in it.
    assert [e["agent"]["name"] for e in activity(website())["results"]] == ["script"]
    assert activity(website(), project=project.pk)["results"] == []
    theirs = activity(website("mallory", "whatever-password-1"))["results"]
    assert [e["agent"]["name"] for e in theirs] == ["Mallory's bot"]
    assert APIClient().get("/api/agent-activity/").status_code == 401
    assert APIClient().get("/api/agent-activity/summary/").status_code == 401


# --------------------------------------------------------------------------- #
# Privacy: export, deleting the account, purge
# --------------------------------------------------------------------------- #


def test_the_export_holds_the_log(ada, mallory):
    client, _, _ = token_client(ada)
    ok(client, "list_projects")
    entry(mallory)
    browser = website()
    assert browser.get("/api/auth/account/").data["stored"]["agent_actions"] == 1
    response = browser.get("/api/auth/account/export/")
    archive = zipfile.ZipFile(io.BytesIO(b"".join(response.streaming_content)))
    assert "agent_activity.json" in archive.read("README.txt").decode()
    (exported,) = json.loads(archive.read("agent_activity.json"))
    assert exported["summary"] == "Listed 0 saved networks"
    assert exported["agent"]["name"] == "Claude on my laptop"


def test_deleting_the_account_deletes_the_log(ada, mallory):
    client, _, _ = token_client(ada)
    ok(client, "list_projects")
    entry(mallory)
    response = website().post(
        "/api/auth/account/delete/", {"password": PASSWORD, "username": "ada"}, format="json"
    )
    assert response.status_code == 204
    assert [e.owner.username for e in AgentAction.objects.all()] == ["mallory"]


def purge_command(*args: str) -> str:
    out = io.StringIO()
    call_command("purge_old_data", *args, stdout=out)
    return out.getvalue()


def test_purge_deletes_old_entries(ada, mallory, settings):
    now = timezone.now()
    old = entry(ada, created_at=now - timedelta(days=91))
    recent = entry(ada, created_at=now - timedelta(days=89))
    theirs = entry(mallory, created_at=now - timedelta(days=120))

    output = purge_command("--dry-run")
    assert "Would delete 2 entries of the agent activity log older than 90 days" in output
    assert AgentAction.objects.count() == 3
    output = purge_command()
    assert "Agent activity log: kept for 90 days." in output
    assert "2 entries of the agent activity log" in output
    assert list(AgentAction.objects.all()) == [recent]
    assert not AgentAction.objects.filter(pk__in=[old.pk, theirs.pk]).exists()

    settings.AGENT_LOG_DAYS = 30
    purge_command()
    assert AgentAction.objects.count() == 0


def test_purge_keeps_the_newest_entries_of_busy_accounts(ada, mallory, monkeypatch):
    monkeypatch.setattr(agent_activity, "MAX_ENTRIES_PER_USER", 3)
    mine = [entry(ada) for _ in range(5)]
    theirs = [entry(mallory) for _ in range(2)]
    purge_command()
    assert [e.pk for e in log(ada)] == [e.pk for e in mine[2:]]
    assert [e.pk for e in log(mallory)] == [e.pk for e in theirs]


def test_purge_deletes_everything_when_the_log_is_off(ada, settings):
    entry(ada)
    assert "Agent activity log: off" in purge_command("--agent-log-days", "0")
    assert AgentAction.objects.count() == 0


# --------------------------------------------------------------------------- #
# Logging never gets in the way
# --------------------------------------------------------------------------- #


def test_failures_of_the_log_do_not_break_the_action(ada, monkeypatch, caplog):
    def broken(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(AgentAction.objects, "bulk_create", broken)
    client, _, _ = token_client(ada)
    with caplog.at_level(logging.ERROR, logger="brick_dnn_core.agent_activity"):
        created = ok(client, "save_project", name="Still saved", graph=GRAPH)
        assert client.get("/api/projects/").status_code == 200
    assert Project.objects.get(pk=created["id"]).name == "Still saved"
    assert "Could not write 1 entries of the agent activity log" in caplog.text
    monkeypatch.undo()

    # Describing an action can fail too: that entry is left out, the others are written.
    monkeypatch.setattr(agent_activity, "describe_tool", broken)
    batch = [
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "list_projects"}},
    ]
    response = client.post("/api/mcp/", json.dumps(batch), content_type="application/json")
    assert response.status_code == 200 and "isError" not in response.json()[0]["result"]
    assert client.get("/api/auth/me/").status_code == 200
    assert [e.tool for e in log(ada)] == ["me"]

    monkeypatch.setattr(agent_activity, "_record_request", broken)
    assert client.get("/api/layers/").status_code == 200


def test_entries_are_written_after_the_response_was_sent(ada):
    token, _ = create_token(ada, "script")
    token = ApiToken.objects.select_related("owner").get(pk=token.pk)
    call_ = agent_activity._Call(
        token=token,
        user_agent="",
        source="mcp",
        tool="list_projects",
        method="",
        writes=False,
        error=None,
        status_code=None,
        duration_ms=3,
        finished_at=timezone.now(),
        describe=lambda user: (agent_activity.Described("list the saved networks"), {}),
    )
    response = HttpResponse("done")
    agent_activity._defer(response, [call_])
    assert AgentAction.objects.count() == 0
    # Closing a response also gives the database connection back (request_finished), which
    # would end the transaction this test runs in: left out here, as in Django's test client.
    request_finished.disconnect(close_old_connections)
    try:
        response.close()  # what the server does once the response is out
    finally:
        request_finished.connect(close_old_connections)
    (written,) = log(ada)
    assert written.summary == "Listed the saved networks" and written.duration_ms == 3


@pytest.mark.django_db
def test_every_mcp_tool_has_a_description(ada):
    from brick_dnn_core import mcp

    missing = [name for name in mcp.TOOLS if name not in agent_activity.TOOL_DESCRIPTIONS]
    assert missing == []
    for name in mcp.TOOLS:  # also with unexpected arguments and answers
        text = agent_activity.describe_tool(ada, name, {"id": "x"}, {"oops": 1}).phrase
        assert text and "unknown tool" not in text


@pytest.mark.django_db
def test_data_tools_name_what_they_used(ada):
    from brick_dnn_core.models import DataSource

    source = DataSource.objects.create(owner=ada, name="sales.csv")
    described = agent_activity.describe_tool(ada, "delete_data_source", {"id": source.pk}, None)
    assert described.phrase == f"delete data file “sales.csv” (#{source.pk})"
    described = agent_activity.describe_tool(ada, "import_data", {"url": "https://x"}, None)
    assert described.phrase == "import a data file from a link"
