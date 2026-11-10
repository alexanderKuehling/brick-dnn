"""The MCP server for AI assistants (JSON-RPC over HTTP, logged in with an API token)."""

from __future__ import annotations

import json
import time

import pytest
from django.contrib.auth import get_user_model
from rest_framework.test import APIClient

from brick_dnn_core import training_service
from brick_dnn_core.api_tokens import create_token
from brick_dnn_core.mcp import PROTOCOL_VERSIONS, TOOLS
from brick_dnn_core.models import Project, TrainingRun
from brickdnn.training import TERMINAL_STATUSES
from brickdnn.training.state import read_state, write_state

from .helpers import chain
from .test_training_api import (  # noqa: F401 (training_dir is a fixture)
    DEMO_GRAPH,
    QUICK,
    RANDOM,
    training_dir,
    use_runner,
    wait_for_state,
)

pytestmark = pytest.mark.django_db

GRAPH = chain(("Input", {"shape": [4]}), ("Linear", {"out_features": 2}))
BROKEN = chain(("Input", {"shape": [4]}), ("Linear", {"in_features": 3, "out_features": 2}))


@pytest.fixture
def user():
    return get_user_model().objects.create_user("ada", password="correct-horse-battery")


def mcp_client(user, read_only: bool = False) -> APIClient:
    _, key = create_token(user, "assistant", read_only=read_only)
    client = APIClient()
    client.credentials(HTTP_AUTHORIZATION=f"Bearer {key}")
    return client


@pytest.fixture
def mcp(user):
    return mcp_client(user)


def rpc(client, method, params=None, message_id=1, **headers):
    message = {"jsonrpc": "2.0", "id": message_id, "method": method}
    if params is not None:
        message["params"] = params
    return client.post(
        "/api/mcp/",
        json.dumps(message),
        content_type="application/json",
        HTTP_ACCEPT="application/json, text/event-stream",
        **headers,
    )


def call(client, tool, **arguments):
    response = rpc(client, "tools/call", {"name": tool, "arguments": arguments})
    assert response.status_code == 200, response.content
    result = response.json()["result"]
    return result, result["content"][0]["text"]


def test_initialize_and_list_tools(mcp):
    response = rpc(
        mcp,
        "initialize",
        {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "1"},
        },
    )
    assert response.status_code == 200
    assert response["Content-Type"] == "application/json"
    result = response.json()["result"]
    assert result["protocolVersion"] == "2025-06-18"
    assert result["serverInfo"]["name"] == "brick-dnn"
    assert "tools" in result["capabilities"]
    assert "Input" in result["instructions"]

    # An unknown version is answered with the newest one.
    response = rpc(mcp, "initialize", {"protocolVersion": "1999-01-01"})
    assert response.json()["result"]["protocolVersion"] == PROTOCOL_VERSIONS[0]

    tools = rpc(mcp, "tools/list").json()["result"]["tools"]
    assert {tool["name"] for tool in tools} == set(TOOLS)
    analyze = next(tool for tool in tools if tool["name"] == "analyze_network")
    assert analyze["inputSchema"]["required"] == ["graph"]
    assert analyze["annotations"]["readOnlyHint"] is True


def test_notifications_get_202(mcp):
    response = mcp.post(
        "/api/mcp/",
        json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        content_type="application/json",
    )
    assert response.status_code == 202
    assert response.content == b""


def test_ping_batch_and_errors(mcp):
    assert rpc(mcp, "ping").json() == {"jsonrpc": "2.0", "id": 1, "result": {}}
    error = rpc(mcp, "resources/list").json()["error"]
    assert error["code"] == -32601

    batch = [
        {"jsonrpc": "2.0", "id": 1, "method": "ping"},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "nope"},
    ]
    answers = mcp.post("/api/mcp/", json.dumps(batch), content_type="application/json").json()
    assert [a["id"] for a in answers] == [1, 2]
    assert "error" in answers[1]

    response = mcp.post("/api/mcp/", "{not json", content_type="application/json")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == -32700
    assert rpc(mcp, "tools/call", {"name": "nope"}).json()["error"]["code"] == -32602
    response = rpc(mcp, "ping", **{"HTTP_MCP_PROTOCOL_VERSION": "1999-01-01"})
    assert response.status_code == 400


def test_url_without_slash(mcp):
    response = mcp.post(
        "/api/mcp",
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}),
        content_type="application/json",
    )
    assert response.status_code == 200


def test_needs_an_api_token(user):
    response = rpc(APIClient(), "tools/list")
    assert response.status_code == 401
    assert response["WWW-Authenticate"].startswith("Bearer")

    # A website login (JWT) is not enough: assistants get their own revocable token.
    client = APIClient()
    login = client.post(
        "/api/auth/token/", {"username": "ada", "password": "correct-horse-battery"}, format="json"
    )
    client.credentials(HTTP_AUTHORIZATION=f"Bearer {login.data['access']}")
    assert rpc(client, "tools/list").status_code == 401


def test_foreign_origins_are_rejected(mcp):
    response = rpc(mcp, "ping", HTTP_ORIGIN="https://evil.example")
    assert response.status_code == 403
    assert rpc(mcp, "ping", HTTP_ORIGIN="http://testserver").status_code == 200


def test_get_is_not_supported(mcp):
    assert mcp.get("/api/mcp/").status_code == 405


def test_layers_examples_analyze_and_code(mcp):
    result, text = call(mcp, "list_layer_types")
    assert any(layer["type"] == "Linear" for layer in json.loads(text)["layers"])
    assert "structuredContent" in result

    result, _ = call(mcp, "list_examples")
    first = result["structuredContent"]["examples"][0]
    assert "graph" not in first and first["name"]
    result, _ = call(mcp, "list_examples", id=first["id"])
    assert result["structuredContent"]["graph"]["nodes"]
    result, text = call(mcp, "list_examples", id="nope")
    assert result["isError"] is True and first["id"] in text

    result, text = call(mcp, "analyze_network", graph=GRAPH)
    assert result["structuredContent"]["valid"] is True
    assert result["structuredContent"]["total_params"] == 10
    result, _ = call(mcp, "analyze_network", graph=BROKEN)
    assert result["structuredContent"]["valid"] is False

    result, text = call(mcp, "generate_pytorch_code", graph=GRAPH, class_name="Tiny")
    assert "class Tiny(nn.Module)" in text
    assert "isError" not in result
    result, text = call(mcp, "generate_pytorch_code", graph=BROKEN)
    assert result["isError"] is True
    assert "in_features" in text or "Input features" in text


def test_missing_arguments_are_tool_errors(mcp):
    result, text = call(mcp, "analyze_network")
    assert result["isError"] is True
    assert "graph" in text


def test_save_list_and_get_projects(mcp, user):
    result, _ = call(mcp, "save_project", name="Tiny", graph=GRAPH, description="test")
    project_id = result["structuredContent"]["id"]
    assert Project.objects.get(pk=project_id).owner == user

    _, text = call(mcp, "list_projects")
    listing = json.loads(text)
    assert [p["name"] for p in listing] == ["Tiny"]
    assert "graph" not in listing[0]

    call(mcp, "save_project", id=project_id, name="Renamed")
    result, _ = call(mcp, "get_project", id=project_id)
    assert result["structuredContent"]["name"] == "Renamed"
    assert result["structuredContent"]["graph"]["nodes"]

    result, text = call(mcp, "save_project", name="No graph")
    assert result["isError"] is True

    other = get_user_model().objects.create_user("mallory", password="whatever-password-1")
    theirs = Project.objects.create(owner=other, name="Theirs", graph=GRAPH)
    result, _ = call(mcp, "get_project", id=theirs.pk)
    assert result["isError"] is True
    result, _ = call(mcp, "save_project", id=theirs.pk, name="Mine now")
    assert result["isError"] is True
    theirs.refresh_from_db()
    assert theirs.name == "Theirs"


def test_read_only_tokens_only_get_reading_tools(user):
    client = mcp_client(user, read_only=True)
    names = {tool["name"] for tool in rpc(client, "tools/list").json()["result"]["tools"]}
    assert "analyze_network" in names and "list_projects" in names
    assert not names & {"save_project", "start_training", "stop_training"}

    result, text = call(client, "save_project", name="Tiny", graph=GRAPH)
    assert result["isError"] is True and "read-only" in text
    assert not Project.objects.exists()
    result, _ = call(client, "analyze_network", graph=GRAPH)
    assert result["structuredContent"]["valid"] is True


def test_training_disabled(mcp, settings):
    settings.TRAINING_ENABLED = False
    result, text = call(mcp, "start_training", graph=DEMO_GRAPH, dataset=RANDOM)
    assert result["isError"] is True and "disabled" in text


def test_training(mcp, user, training_dir):  # noqa: F811
    result, text = call(mcp, "get_training_options")
    options = result["structuredContent"]
    assert "available" in options  # depends on whether PyTorch is installed
    assert options["config_defaults"]["epochs"] == 10
    assert "dataset_reference" in options

    result, _ = call(mcp, "check_training", graph=DEMO_GRAPH, dataset=RANDOM)
    assert result["structuredContent"]["ok"] is True

    result, text = call(
        mcp, "start_training", graph=DEMO_GRAPH, dataset=RANDOM, config=QUICK, name="MCP run"
    )
    assert "isError" not in result, text
    run_id = result["structuredContent"]["id"]
    assert TrainingRun.objects.get(pk=run_id).owner == user

    deadline = time.monotonic() + 10
    while True:
        result, _ = call(mcp, "get_training_run", id=run_id)
        status = result["structuredContent"]["status"]
        if status in TERMINAL_STATUSES or time.monotonic() > deadline:
            break
        time.sleep(0.05)
    assert status == "completed"
    result, _ = call(mcp, "list_training_runs")
    listing = result["structuredContent"]
    assert listing["total"] == 1 and [run["name"] for run in listing["runs"]] == ["MCP run"]

    result, text = call(mcp, "get_training_run", id="not-a-uuid")
    assert result["isError"] is True

    result, text = call(mcp, "start_training", graph=DEMO_GRAPH, dataset={"kind": "builtin"})
    assert result["isError"] is True


def test_regression_metrics_reach_rest_and_mcp(mcp, user, training_dir, settings):  # noqa: F811
    """The R² of a neural regression is in the run's metrics, its best trial and its state."""
    use_runner(settings, "wait")  # does not overwrite the state written below
    result, text = call(mcp, "start_training", graph=DEMO_GRAPH, dataset=RANDOM, config=QUICK)
    assert "isError" not in result, text
    run_id = result["structuredContent"]["id"]
    wait_for_state(training_dir, run_id)
    process = training_service._PROCESSES[run_id]
    process.kill()
    process.wait(timeout=5)
    directory = training_dir / "runs" / run_id
    state = read_state(directory)
    entry = {"epoch": 1, "train_loss": 0.4, "val_loss": 0.3, "train_accuracy": None}
    entry.update(val_accuracy=None, val_rmse=12.5, val_r2=0.81, seconds=0.1)
    trial = {"trial": 1, "params": {"learning_rate": 0.01}, "status": "completed"}
    trial.update(message="", epochs_done=1, train_loss=0.4, val_loss=0.3, val_accuracy=None)
    trial.update(val_rmse=12.5, val_r2=0.81, weights_epoch=1, stopped_early=False, seconds=1.0)
    state.update(
        status="completed",
        message="Finished: validation loss 0.3, RMSE 12.5, R² 0.810; test loss 0.35, …",
        task="regression",
        history=[entry],
        weights_epoch=1,
        test={"loss": 0.35, "accuracy": None, "rmse": 13.4, "r2": 0.78},
        trials=[trial],
        best_trial=1,
        finished_at=time.time(),
    )
    write_state(directory, state)

    expected = {"val_rmse": 12.5, "val_r2": 0.81, "test_rmse": 13.4, "test_r2": 0.78}
    result, _ = call(mcp, "get_training_run", id=run_id, include=["history"])
    run = result["structuredContent"]
    assert run["status"] == "completed"
    assert {key: run["metrics"][key] for key in expected} == expected
    assert run["tuning"]["best_trial"]["val_r2"] == 0.81
    assert run["history"][0]["val_r2"] == 0.81 and run["test"]["r2"] == 0.78
    _, text = call(mcp, "list_training_runs")
    listed = json.loads(text)["runs"][0]
    assert listed["metrics"]["val_r2"] == 0.81 and listed["metrics"]["test_r2"] == 0.78
    rest = APIClient()
    rest.force_authenticate(user)
    detail = rest.get(f"/api/training/runs/{run_id}/").data
    assert {key: detail["metrics"][key] for key in expected} == expected
    assert detail["best_trial"]["val_r2"] == 0.81
