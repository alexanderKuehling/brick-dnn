"""Classical models (scikit-learn) through the REST API and the MCP tools, with the real
runner and prediction processes. PyTorch is not needed."""

# ruff: noqa: F811 (the fixtures are imported from test_training_api)

from __future__ import annotations

import subprocess
import sys
import time
from urllib.parse import urlsplit

import pytest

pytest.importorskip("sklearn")

import joblib  # noqa: E402
import numpy as np  # noqa: E402
from rest_framework.test import APIClient  # noqa: E402

from brick_dnn_core import training_service  # noqa: E402
from brickdnn import EXAMPLES  # noqa: E402

from .test_classical_training import (  # noqa: E402
    classification_csv,
    model_graph,
    regression_csv,
)
from .test_mcp import call, mcp_client  # noqa: E402
from .test_training_api import (  # noqa: E402, F401 (fixtures)
    TERMINAL_STATUSES,
    auth_client,
    training_dir,
    upload,
    user,
    wait_for,
)


@pytest.fixture
def real_processes(training_dir, settings, monkeypatch):
    settings.TRAINING_RUNNER_COMMAND = None
    settings.TRAINING_PREDICT_COMMAND = None
    # Classical models need no PyTorch: training and predictions work without it.
    monkeypatch.setattr(training_service, "torch_installed", lambda: False)
    return training_dir


def upload_csv(client, path):
    response = upload(client, path.name, path.read_bytes())
    assert response.status_code == 201, response.data
    return {"kind": "upload", "id": response.data["id"]}


@pytest.mark.django_db
def test_capabilities_list_classical_models(auth_client, training_dir, monkeypatch):
    monkeypatch.setattr(training_service, "torch_installed", lambda: False)
    monkeypatch.setenv("BRICKDNN_GP_MAX_ROWS", "1234")
    data = auth_client.get("/api/training/").data
    assert data["available"] is False  # PyTorch
    classical = data["classical"]
    assert classical["available"] is True and classical["reason"] is None
    assert classical["sklearn_version"]
    tuning = classical["tuning"]
    assert tuning["cv_folds"] == {"min": 2, "max": 10, "default": 5}
    assert sorted(tuning["models"]) == sorted(classical["max_rows"])
    svc = tuning["models"]["SVC"]
    assert [t["name"] for t in svc["tunable"]] == ["C", "gamma", "kernel", "class_weight"]
    assert svc["default"] == {"C": {"min": 0.1, "max": 100}, "gamma": {"min": 0.001, "max": 1}}
    assert "kernel hyperparameters" in tuning["models"]["GaussianProcessRegressor"]["note"]
    assert {m["name"]: m["direction"] for m in data["tuning"]["metrics"]}["roc_auc"] == "maximize"
    assert classical["max_rows"] == {
        "GaussianProcessRegressor": 1234,
        "GaussianProcessClassifier": 2_000,
        "SVR": 50_000,
        "SVC": 50_000,
        "HistGradientBoostingRegressor": 1_000_000,
        "HistGradientBoostingClassifier": 1_000_000,
    }
    monkeypatch.setattr(training_service, "sklearn_installed", lambda: False)
    classical = auth_client.get("/api/training/").data["classical"]
    assert classical["available"] is False and "pip install scikit-learn" in classical["reason"]
    response = auth_client.post(
        "/api/training/runs/",
        {"graph": model_graph("SVR"), "dataset": {"kind": "random"}, "config": {}},
        format="json",
    )
    assert response.status_code == 503 and "scikit-learn" in response.data["detail"]


@pytest.mark.django_db
def test_check_and_start_refusals(auth_client, real_processes, tmp_path, monkeypatch):
    dataset = upload_csv(auth_client, regression_csv(tmp_path / "houses.csv"))
    graph = model_graph("GaussianProcessRegressor")
    monkeypatch.setenv("BRICKDNN_GP_MAX_ROWS", "50")
    body = {"graph": graph, "dataset": dataset}
    data = auth_client.post("/api/training/check/", body, format="json").data
    assert data["ok"] is True and "BRICKDNN_GP_MAX_ROWS" in data["warnings"][0]
    assert data["plan"]["classical"]["max_rows"] == 50
    body["config"] = {"max_samples": 60}
    data = auth_client.post("/api/training/check/", body, format="json").data
    assert data["ok"] is False and "Max. samples is 60" in data["errors"][0]
    response = auth_client.post(
        "/api/training/check/", {**body, "config": {"epochs": 0}}, format="json"
    )
    assert response.status_code == 400 and "Epochs" in response.data["detail"]

    # A search of the network settings is not one of the brick parameters.
    response = auth_client.post(
        "/api/training/runs/",
        {
            "graph": graph,
            "dataset": dataset,
            "config": {},
            "tuning": {"space": {"batch_size": [32]}},
        },
        format="json",
    )
    assert response.status_code == 400
    assert "'batch_size' cannot be tuned for Gaussian process regression" in str(
        response.data["detail"]
    )
    response = auth_client.post(
        "/api/training/runs/",
        {"graph": graph, "dataset": {"kind": "builtin", "id": "mnist"}, "config": {}},
        format="json",
    )
    assert response.status_code == 400 and "not on images" in response.data["detail"]


@pytest.mark.django_db
def test_classical_run_end_to_end(auth_client, user, real_processes, tmp_path):
    dataset = upload_csv(auth_client, classification_csv(tmp_path / "mail.csv"))
    response = auth_client.post(
        "/api/training/runs/",
        {
            "name": "Mail SVM",
            "graph": EXAMPLES["svm_classifier"]["graph"],
            "dataset": dataset,
            "config": {"seed": 1, "test_split": 0.1},
        },
        format="json",
    )
    assert response.status_code == 201, response.data
    run_id = response.data["id"]
    assert response.data["family"] == "classical" and response.data["epochs"] == 1
    data = wait_for(auth_client, run_id, timeout=120, interval=0.25)
    assert data["status"] == "completed", data["message"]
    assert data["has_weights"] is True and data["epochs_done"] == 1
    assert data["metrics"]["val_accuracy"] > 0.5 and data["metrics"]["test_loss"] is not None
    assert data["plan"]["classical"]["estimator"] == "SVC"
    state = data["state"]
    assert state["family"] == "classical" and state["classical"]["support_vectors"] > 0

    # The weights: the fitted pipeline as a joblib file.
    response = auth_client.get(f"/api/training/runs/{run_id}/weights/")
    assert response.status_code == 200
    assert response["Content-Disposition"] == 'attachment; filename="mail-svm.joblib"'
    (tmp_path / "mail-svm.joblib").write_bytes(b"".join(response.streaming_content))
    assert joblib.load(tmp_path / "mail-svm.joblib")["class_names"] == ["eggs", "ham", "spam"]

    # Predictions over REST: classes with their probabilities.
    response = auth_client.post(
        f"/api/training/runs/{run_id}/predict/",
        {"inputs": [{"a": 2, "b": 0, "c": 0, "d": 0}, [-2, 0, 0, 0]], "top_k": 2},
        format="json",
    )
    assert response.status_code == 200, response.data
    predictions = response.data["predictions"]
    assert [p["class"] for p in predictions] == ["spam", "ham"]
    assert all(len(p["top"]) == 2 and p["probability"] > 0.4 for p in predictions)

    # The matching script predicts with the downloaded file.
    code = auth_client.post(
        "/api/generate/",
        {"graph": data["graph"], "options": {"include_training": False}},
        format="json",
    ).data["code"]
    (tmp_path / "mail_svm.py").write_text(code)
    (tmp_path / "new.csv").write_text("a,b,c,d\n2,0,0,0\n")
    result = subprocess.run(
        [sys.executable, "mail_svm.py", "--predict", "new.csv", "--save", "mail-svm.joblib"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[1].startswith("spam,")


@pytest.mark.django_db
def test_gradient_boosting_end_to_end(auth_client, user, real_processes, tmp_path):
    """An imbalanced classification (a rare class) with balanced class weights: trained by
    the runner, evaluated, predicted over REST and MCP, and downloaded."""
    rng = np.random.default_rng(7)
    x = rng.normal(size=(600, 4))
    fraud = x[:, 0] + x[:, 1] > 2.2  # about 6 % of the rows
    lines = ["a,b,c,d,label"] + [
        ",".join([*(f"{v:.4f}" for v in row), "fraud" if bad else "ok"])
        for row, bad in zip(x, fraud, strict=True)
    ]
    (tmp_path / "payments.csv").write_text("\n".join(lines) + "\n")
    dataset = upload_csv(auth_client, tmp_path / "payments.csv")
    graph = model_graph(
        "HistGradientBoostingClassifier",
        params={"class_weight": "balanced", "max_iter": 60, "min_samples_leaf": 10},
    )
    checked = auth_client.post(
        "/api/training/check/", {"graph": graph, "dataset": dataset}, format="json"
    ).data
    assert checked["ok"] is True and checked["estimate"]["details"]["trees"] == 60
    response = auth_client.post(
        "/api/training/runs/",
        {"name": "Fraud trees", "graph": graph, "dataset": dataset, "config": {"seed": 1}},
        format="json",
    )
    assert response.status_code == 201, response.data
    run_id = response.data["id"]
    data = wait_for(auth_client, run_id, timeout=120, interval=0.25)
    assert data["status"] == "completed", data["message"]
    assert data["plan"]["classical"]["estimator"] == "HistGradientBoostingClassifier"
    state = data["state"]
    assert state["classical"]["iterations"] == state["classical"]["max_iterations"] == 60
    evaluation = state["evaluation"]
    assert evaluation["val"]["confusion_matrix"] and evaluation["val"]["balanced_accuracy"] > 0.7
    assert data["metrics"]["val_accuracy"] > 0.85

    response = auth_client.post(
        f"/api/training/runs/{run_id}/predict/",
        {"inputs": [{"a": 2.5, "b": 2.5, "c": 0, "d": 0}, [-2, -2, 0, 0]]},
        format="json",
    )
    assert response.status_code == 200, response.data
    assert [p["class"] for p in response.data["predictions"]] == ["fraud", "ok"]

    mcp = mcp_client(user)
    result, text = call(mcp, "get_training_run", id=run_id)
    assert result["structuredContent"]["classical"]["iterations"] == 60, text
    response = auth_client.get(f"/api/training/runs/{run_id}/weights/")
    assert response.status_code == 200
    (tmp_path / "trees.joblib").write_bytes(b"".join(response.streaming_content))
    bundle = joblib.load(tmp_path / "trees.joblib")
    assert bundle["class_names"] == ["fraud", "ok"]
    assert bundle["model"].named_steps["model"].class_weight == "balanced"


@pytest.mark.django_db
def test_mcp_flow_with_a_gaussian_process(user, real_processes, tmp_path):
    mcp = mcp_client(user)
    graph = EXAMPLES["gp_regression"]["graph"]
    analysis, _ = call(mcp, "analyze_network", graph=graph)
    analysis = analysis["structuredContent"]
    assert analysis["valid"] and analysis["family"] == "classical"
    assert analysis["nodes"]["kernel"]["output_kind"] == "kernel"
    code, _ = call(mcp, "generate_pytorch_code", graph=graph)
    assert "def make_model() -> Pipeline:" in code["content"][0]["text"]

    options, _ = call(mcp, "get_training_options")
    assert options["structuredContent"]["classical"]["available"] is True
    csv = regression_csv(tmp_path / "houses.csv", features=8).read_text()
    uploaded, _ = call(mcp, "upload_dataset", name="Houses", csv=csv)
    dataset = uploaded["structuredContent"]["reference"]
    checked, text = call(mcp, "check_training", graph=graph, dataset=dataset, config={"seed": 1})
    assert checked["structuredContent"]["ok"] is True, text
    started, text = call(mcp, "start_training", graph=graph, dataset=dataset, name="GP")
    assert "isError" not in started, text
    run_id = started["structuredContent"]["id"]

    deadline = time.monotonic() + 120
    while True:
        result, _ = call(mcp, "get_training_run", id=run_id)
        run = result["structuredContent"]
        if run["status"] in TERMINAL_STATUSES:
            break
        assert time.monotonic() < deadline, run["message"]
        time.sleep(0.25)
    assert run["status"] == "completed", run["message"]
    assert run["classical"]["kernel"] and run["metrics"]["val_r2"] is not None

    result, text = call(
        mcp, "predict", run_id=run_id, inputs=[dict.fromkeys([f"f{i}" for i in range(8)], 0)]
    )
    assert "isError" not in result, text
    answer = result["structuredContent"]
    (prediction,) = answer["predictions"]
    assert prediction["value"] == pytest.approx(100, abs=2) and prediction["std"] > 0
    assert answer["target"] == "price" and "std_note" in answer

    info, _ = call(mcp, "get_weights_download", id=run_id)
    info = info["structuredContent"]
    assert info["filename"] == "gp.joblib" and "joblib" in info["format"]
    assert "def make_model() -> Pipeline:" in info["model_code"]
    url = urlsplit(info["url"])
    weights = APIClient().get(f"{url.path}?{url.query}")
    assert weights.status_code == 200
    (tmp_path / info["filename"]).write_bytes(b"".join(weights.streaming_content))
    script = subprocess.run(
        [sys.executable, "-c", info["how_to_use"]],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert script.returncode == 0, script.stderr


# --------------------------------------------------------------------------- #
# Hyperparameter search
# --------------------------------------------------------------------------- #


@pytest.mark.django_db
def test_check_a_classical_search(auth_client, real_processes, tmp_path):
    dataset = upload_csv(auth_client, classification_csv(tmp_path / "mail.csv"))
    graph = model_graph("SVC")

    def check(tuning, config=None):
        body = {"graph": graph, "dataset": dataset, "config": config or {}, "tuning": tuning}
        return auth_client.post("/api/training/check/", body, format="json")

    single = check(None).data["estimate"]
    data = check({"trials": 4, "space": {"C": {"min": 0.1, "max": 10}}}).data
    assert data["ok"] is True and data["estimate"]["details"]["fits"] == 4
    assert "fits" not in single["details"] and data["estimate"]["basis"].endswith("4 trials")
    data = check({"trials": 4, "cv_folds": 3}).data
    assert data["estimate"]["details"]["fits"] == 13 and "3 folds" in data["estimate"]["basis"]
    # Without validation rows the search cross-validates (5 folds).
    data = check({"trials": 2}, {"val_split": 0}).data
    assert data["ok"] is True and data["estimate"]["details"]["fits"] == 11
    assert any("5-fold cross-validation" in warning for warning in data["warnings"])
    # A metric the data does not have (three classes): an error of the check.
    data = check({"metric": "roc_auc"}).data
    assert data["ok"] is False
    assert "ROC AUC needs two classes, this data has 3" in data["errors"][0]
    for tuning, message in [
        ({"space": {"degree": [2]}}, "'degree' cannot be tuned for support vector classification"),
        ({"space": {"C": {"min": 5, "max": 1}}}, "smallest C must not be larger"),
        (
            {"method": "grid", "space": {"C": list(range(1, 12)), "gamma": list(range(1, 11))}},
            "The grid has 110 combinations",
        ),
        ({"metric": "speed"}, "the metric 'speed' is unknown"),
    ]:
        response = check(tuning)
        assert response.status_code == 400 and message in response.data["detail"], tuning


@pytest.mark.django_db
def test_classical_search_end_to_end(auth_client, user, real_processes, tmp_path):
    dataset = upload_csv(auth_client, classification_csv(tmp_path / "mail.csv", n=150))
    tuning = {
        "method": "grid",
        "metric": "balanced_accuracy",
        "cv_folds": 3,
        "space": {"C": [0.3, 3], "kernel": ["rbf", "linear"]},
    }
    response = auth_client.post(
        "/api/training/runs/",
        {
            "name": "SVC search",
            "graph": model_graph("SVC"),
            "dataset": dataset,
            "config": {"seed": 1, "test_split": 0.2},
            "tuning": tuning,
        },
        format="json",
    )
    assert response.status_code == 201, response.data
    assert response.data["tuning"] == {**tuning, "trials": 4}
    run_id = response.data["id"]
    data = wait_for(auth_client, run_id, timeout=120, interval=0.25)
    assert data["status"] == "completed", data["message"]
    assert data["trials_done"] == 4 and data["has_weights"] is True
    best = data["best_trial"]
    trials = data["state"]["trials"]
    assert best["score"] == max(t["score"] for t in trials)
    assert best["params"] == trials[best["trial"] - 1]["params"]
    assert data["state"]["evaluation"]["trial"] == best["trial"]
    assert data["metrics"]["test_accuracy"] is not None

    mcp = mcp_client(user)
    result, text = call(mcp, "get_training_run", id=run_id, include=["trials"])
    run = result["structuredContent"]
    assert run["tuning"]["metric"] == "balanced_accuracy", text
    assert run["tuning"]["direction"] == "maximize" and run["tuning"]["cv_folds"] == 3
    assert run["tuning"]["best_trial"]["score"] == best["score"]
    assert len(run["trials"]) == 4 and all(t["folds"] == 3 for t in run["trials"])


@pytest.mark.django_db
def test_mcp_classical_search(user, real_processes, tmp_path):
    mcp = mcp_client(user)
    options, _ = call(mcp, "get_training_options")
    options = options["structuredContent"]
    assert options["classical"]["tuning"]["models"]["SVR"] == "C*, gamma*, kernel, epsilon"
    assert "r2 (max; regression)" in options["tuning"]["metrics"]
    csv = regression_csv(tmp_path / "houses.csv", n=100).read_text()
    uploaded, _ = call(mcp, "upload_dataset", name="Houses", csv=csv)
    dataset = uploaded["structuredContent"]["reference"]
    graph = model_graph("SVR")
    tuning = {"trials": 3, "metric": "r2", "space": {"C": {"min": 0.1, "max": 10}}}
    checked, text = call(mcp, "check_training", graph=graph, dataset=dataset, tuning=tuning)
    assert checked["structuredContent"]["ok"] is True, text
    assert checked["structuredContent"]["estimate"]["details"]["trials"] == 3
    bad, text = call(
        mcp, "check_training", graph=graph, dataset=dataset, tuning={"metric": "accuracy"}
    )
    assert bad["structuredContent"]["ok"] is False and "is for classification" in text
    refused, text = call(
        mcp, "start_training", graph=graph, dataset=dataset, tuning={"space": {"gamma": [-1]}}
    )
    assert refused["isError"] is True and "gamma must be greater than 0" in text
    started, text = call(mcp, "start_training", graph=graph, dataset=dataset, tuning=tuning)
    assert "isError" not in started, text
    run_id = started["structuredContent"]["id"]
    deadline = time.monotonic() + 120
    while True:
        result, _ = call(mcp, "get_training_run", id=run_id)
        run = result["structuredContent"]
        if run["status"] in TERMINAL_STATUSES:
            break
        assert time.monotonic() < deadline, run["message"]
        time.sleep(0.25)
    assert run["status"] == "completed", run["message"]
    assert run["tuning"]["metric"] == "r2" and run["tuning"]["trials_done"] == 3
    assert "cv_folds" not in run["tuning"]  # the validation rows compared the trials
    assert "validation R²" in run["message"]
