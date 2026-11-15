"""Selectable losses through the REST API, the MCP tools and the command line: the checks
per task, the loss a run stores and reports, searches over losses and the generated script."""

# ruff: noqa: F811 (the fixtures are imported from test_training_api)

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from brick_dnn_core.models import TrainingRun  # noqa: E402
from brickdnn.training import cli  # noqa: E402
from brickdnn.training import state as st  # noqa: E402

from .test_classical_api import upload_csv  # noqa: E402
from .test_losses import fraud_csv, mlp, prices_csv  # noqa: E402
from .test_mcp import call, mcp_client  # noqa: E402
from .test_training_api import (  # noqa: E402, F401 (fixtures)
    auth_client,
    training_dir,
    user,
    wait_for,
)

QUICK = {"epochs": 2, "batch_size": 16, "seed": 1, "device": "cpu"}


def start(client, graph, dataset, config, tuning=None):
    body = {"name": "Losses", "graph": graph, "dataset": dataset, "config": config}
    if tuning is not None:
        body["tuning"] = tuning
    return client.post("/api/training/runs/", body, format="json")


@pytest.mark.django_db
def test_rest_checks_the_loss_and_stores_the_one_a_run_trains_with(
    auth_client, training_dir, tmp_path
):
    fraud = upload_csv(auth_client, fraud_csv(tmp_path / "fraud.csv")[0])
    prices = upload_csv(auth_client, prices_csv(tmp_path / "prices.csv"))
    checked = auth_client.post(
        "/api/training/check/",
        {"graph": mlp(2, 2), "dataset": fraud, "config": {"loss": "mae"}},
        format="json",
    ).data
    assert checked["ok"] is False
    assert checked["errors"] == [
        "The loss 'mae' is for regression, and this is a classification. "
        "Possible: auto, cross_entropy, weighted_cross_entropy, focal."
    ]
    response = start(auth_client, mlp(2, 2), fraud, {**QUICK, "loss": "huber"})
    assert response.status_code == 400
    assert "The loss 'huber' is for regression" in str(response.data)

    # "auto" (or nothing) is stored as the default of the task.
    response = start(auth_client, mlp(3, 1), prices, QUICK)
    assert response.status_code == 201, response.data
    assert TrainingRun.objects.get(pk=response.data["id"]).config["loss"] == "mse"
    wait_for(auth_client, response.data["id"])
    response = start(auth_client, mlp(2, 2), fraud, {**QUICK, "loss": "auto"})
    assert response.status_code == 201, response.data
    assert response.data["config"]["loss"] == "cross_entropy"
    wait_for(auth_client, response.data["id"])
    config = {**QUICK, "loss": "focal", "focal_gamma": 1.5, "class_weights": {"fraud": 20}}
    response = start(auth_client, mlp(2, 2), fraud, config)
    assert response.status_code == 201, response.data
    stored = response.data["config"]
    assert stored["loss"] == "focal" and stored["focal_gamma"] == 1.5
    assert stored["class_weights"] == {"fraud": 20.0}
    wait_for(auth_client, response.data["id"])


@pytest.mark.django_db
def test_rest_search_over_losses_needs_another_metric(auth_client, training_dir, tmp_path):
    fraud = upload_csv(auth_client, fraud_csv(tmp_path / "fraud.csv")[0])
    tuning = {"method": "grid", "space": {"loss": ["cross_entropy", "focal"]}}
    response = start(auth_client, mlp(2, 2), fraud, QUICK, tuning)
    assert response.status_code == 400
    assert "A search over the loss cannot rank its trials by the loss" in str(response.data)
    checked = auth_client.post(
        "/api/training/check/",
        {"graph": mlp(2, 2), "dataset": fraud, "config": QUICK, "tuning": tuning},
        format="json",
    ).data
    assert checked["ok"] is False and "e.g. pr_auc" in checked["errors"][0]
    response = start(auth_client, mlp(2, 2), fraud, QUICK, {**tuning, "metric": "pr_auc"})
    assert response.status_code == 201, response.data
    assert response.data["tuning"]["space"] == {"loss": ["cross_entropy", "focal"]}
    wait_for(auth_client, response.data["id"])


@pytest.mark.django_db
def test_a_run_reports_its_loss_over_rest_and_mcp(
    auth_client, user, training_dir, settings, tmp_path
):
    settings.TRAINING_RUNNER_COMMAND = None  # the real runner
    fraud = upload_csv(auth_client, fraud_csv(tmp_path / "fraud.csv")[0])
    response = start(auth_client, mlp(2, 2), fraud, {**QUICK, "loss": "weighted_cross_entropy"})
    assert response.status_code == 201, response.data
    data = wait_for(auth_client, response.data["id"], timeout=120, interval=0.2)
    assert data["status"] == "completed", data["message"]
    loss = data["state"]["loss"]
    assert loss["name"] == "weighted_cross_entropy" and loss["class_weights_from"] == "balanced"
    assert (
        len(loss["class_weights"]) == 2 and loss["class_weights"][0] > 1 > loss["class_weights"][1]
    )
    # The final evaluation keeps the log loss next to the run's loss.
    evaluation = data["state"]["evaluation"]["val"]
    assert evaluation["loss"] != pytest.approx(evaluation["log_loss"])

    mcp = mcp_client(user)
    result, text = call(mcp, "get_training_run", id=data["id"])
    detail = result["structuredContent"]
    assert detail["loss"] == loss
    assert detail["config"]["loss"] == "weighted_cross_entropy"
    assert "focal_gamma" not in detail["config"]  # the settings of other losses


@pytest.mark.django_db
def test_mcp_lists_and_checks_the_losses(user, training_dir, tmp_path):
    mcp = mcp_client(user)
    result, _ = call(mcp, "get_training_options")
    options = result["structuredContent"]
    losses = options["losses"]
    assert losses["classification"] == "cross_entropy*, weighted_cross_entropy, focal"
    assert losses["regression"] == "mse*, mae, huber"
    assert "focal_gamma 0-10 (2)" in losses["settings"]
    assert "huber_delta >0-10 (1)" in losses["settings"]
    assert options["config_defaults"]["loss"] == "auto"
    assert "focal_gamma" not in options["config_defaults"]
    assert "focal_gamma" in options["tuning"]["tunable"]

    path = prices_csv(tmp_path / "prices.csv")
    result, _ = call(mcp, "upload_dataset", name="Prices", csv=path.read_text())
    dataset = result["structuredContent"]["reference"]
    result, text = call(
        mcp, "check_training", graph=mlp(3, 1), dataset=dataset, config={"loss": "focal"}
    )
    checked = result["structuredContent"]
    assert checked["ok"] is False and checked["errors"][0].startswith("The loss 'focal' is for")
    result, _ = call(
        mcp,
        "check_training",
        graph=mlp(3, 1),
        dataset=dataset,
        config={"loss": "huber", "huber_delta": 0.5},
        tuning={"method": "grid", "space": {"loss": ["mse", "huber"]}, "metric": "rmse"},
    )
    assert result["structuredContent"]["ok"] is True

    # The run's loss from its settings before the run reports it (and for older runs).
    response = start_over_mcp(mcp, dataset)
    run = TrainingRun.objects.get(pk=response["id"])
    run.config = {**run.config, "loss": "cross_entropy"}  # settings of an older version
    run.save(update_fields=["config"])
    directory = training_dir / "runs" / str(run.pk)
    state = st.read_state(directory) or st.initial_state(2)
    state["loss"] = None
    st.write_state(directory, state)
    result, _ = call(mcp, "get_training_run", id=str(run.pk))
    assert result["structuredContent"]["loss"] == {
        "name": "mse",
        "label": "mean squared error",
        "task": "regression",
    }


def start_over_mcp(mcp, dataset):
    result, text = call(
        mcp, "start_training", graph=mlp(3, 1), dataset=dataset, config={"epochs": 1}
    )
    assert "isError" not in result, text
    return result["structuredContent"]


@pytest.mark.django_db
def test_generated_code_with_a_loss_over_rest_and_mcp(auth_client, user):
    response = auth_client.post(
        "/api/generate/",
        {"graph": mlp(2, 2), "options": {"loss": "focal", "focal_gamma": 1}},
        format="json",
    )
    assert response.status_code == 200, response.data
    assert (
        'LOSS = "focal"' in response.data["code"] and "FOCAL_GAMMA = 1.0" in response.data["code"]
    )
    response = auth_client.post(
        "/api/generate/", {"graph": mlp(2, 2), "options": {"loss": "mae"}}, format="json"
    )
    assert response.status_code == 400
    assert "The loss 'mae' does not fit the training code" in response.data["detail"]
    result, text = call(
        mcp_client(user), "generate_pytorch_code", graph=mlp(3, 1), loss="huber", huber_delta=2
    )
    assert "isError" not in result, text
    assert "HUBER_DELTA = 2.0" in text


def test_cli_options_of_the_loss(tmp_path, capsys):
    path, _ = fraud_csv(tmp_path / "fraud.csv")
    graph = tmp_path / "net.json"
    graph.write_text(json.dumps(mlp(2, 2)))
    common = [str(graph), "--data", str(path), "--epochs", "2", "--device", "cpu", "--seed", "0"]
    common += ["--save", str(tmp_path / "net.pt")]
    options = ["--loss", "focal", "--focal-gamma", "1", "--class-weights", "fraud=10,ok=1"]
    assert cli.main([*common, *options]) == 0
    out = capsys.readouterr().out
    assert "Loss: focal loss (gamma 1, class weights given)." in out
    assert cli.main([*common, "--loss", "weighted_cross_entropy", "--label-smoothing", "0.1"]) == 0
    assert "Loss: weighted cross-entropy (label smoothing 0.1, balanced class weights)" in (
        capsys.readouterr().out
    )
    assert cli.main([*common, "--loss", "mae"]) == 1
    assert "error: The loss 'mae' is for regression" in capsys.readouterr().err
    assert cli.main([*common, "--loss", "focal", "--class-weights", "1,2,3"]) == 1
    assert "Class weights: give one weight per class (2: fraud, ok), not 3." in (
        capsys.readouterr().err
    )
    # A search over losses: by another metric than the loss.
    assert cli.main([*common, "--tune-loss", "cross_entropy,focal"]) == 1
    assert "cannot rank its trials by the loss" in capsys.readouterr().err
    search = [*common, "--tune-loss", "cross_entropy,focal", "--metric", "pr_auc", "--epochs", "1"]
    assert cli.main(search) == 0
    out = capsys.readouterr().out
    assert "Trial 1/2: " in out and "focal loss" in out
    prices = prices_csv(tmp_path / "prices.csv")
    regression = [str(tmp_path / "reg.json"), "--data", str(prices), "--epochs", "1"]
    (tmp_path / "reg.json").write_text(json.dumps(mlp(3, 1)))
    regression += ["--device", "cpu", "--save", str(tmp_path / "reg.pt")]
    assert cli.main([*regression, "--loss", "huber", "--huber-delta", "0.5"]) == 0
    assert "Loss: Huber loss (delta 0.5 standard deviations)." in capsys.readouterr().out


def test_cli_parses_class_weights():
    assert cli.parse_class_weights("balanced") == "balanced"
    assert cli.parse_class_weights("1,10") == [1.0, 10.0]
    assert cli.parse_class_weights("3") == [3.0]
    assert cli.parse_class_weights("fraud=10, ok=1") == {"fraud": 10.0, "ok": 1.0}
    assert cli.parse_class_weights(None) is None
