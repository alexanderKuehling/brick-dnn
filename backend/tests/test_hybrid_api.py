"""Hybrid networks through the REST API and the MCP tools, with the real runner and
prediction processes: a run end to end (curves, the classical results, predictions, the
download as a ZIP file, explanations), raw rows of a prepared dataset and a hybrid run as a
member of an ensemble of runs."""

# ruff: noqa: F811 (the fixtures are imported from test_training_api)

from __future__ import annotations

import io
import json
import subprocess
import sys
import zipfile

import pytest

pytest.importorskip("sklearn")
pytest.importorskip("torch")

import joblib  # noqa: E402
import numpy as np  # noqa: E402
from django.contrib.auth import get_user_model  # noqa: E402

from brick_dnn_core.models import Dataset  # noqa: E402
from brickdnn import EXAMPLES  # noqa: E402

from .test_classical_api import upload_csv  # noqa: E402
from .test_classical_training import classification_csv  # noqa: E402
from .test_ensembles import client_for, create, payments_csv  # noqa: E402
from .test_hybrid import hybrid_graph  # noqa: E402
from .test_mcp import call, mcp_client  # noqa: E402
from .test_training_api import (  # noqa: E402, F401 (fixtures)
    auth_client,
    training_dir,
    user,
    wait_for,
)


@pytest.fixture
def real_processes(training_dir, settings):
    settings.TRAINING_RUNNER_COMMAND = None
    settings.TRAINING_PREDICT_COMMAND = None
    settings.TRAINING_MAX_RUNS = 5
    return training_dir


@pytest.mark.django_db
def test_a_hybrid_run_end_to_end(auth_client, user, real_processes, tmp_path):
    dataset = upload_csv(auth_client, classification_csv(tmp_path / "mail.csv", n=150))
    graph = hybrid_graph("SVC")
    checked = auth_client.post(
        "/api/training/check/",
        {"graph": graph, "dataset": dataset, "config": {"epochs": 6}},
        format="json",
    ).data
    assert checked["ok"] is True, checked
    assert checked["plan"]["hybrid"]["features"] == 6
    assert checked["estimate"]["details"]["family"] == "hybrid"
    response = auth_client.post(
        "/api/training/runs/",
        {
            "name": "Mail features",
            "graph": graph,
            "dataset": dataset,
            "config": {"seed": 1, "epochs": 6, "test_split": 0.2, "device": "cpu"},
        },
        format="json",
    )
    assert response.status_code == 201, response.data
    assert response.data["family"] == "hybrid" and response.data["epochs"] == 6
    run_id = response.data["id"]
    data = wait_for(auth_client, run_id, timeout=180, interval=0.25)
    assert data["status"] == "completed", data["message"]
    assert data["has_weights"] is True and data["epochs_done"] == 6
    state = data["state"]
    assert state["family"] == "hybrid" and len(state["history"]) == 6
    # The run's metrics are those of the classical model, not of the temporary head.
    classical = state["classical"]
    assert data["metrics"]["val_accuracy"] == pytest.approx(classical["validation"]["accuracy"])
    assert data["metrics"]["train_loss"] == pytest.approx(classical["train"]["loss"])
    assert data["metrics"]["test_accuracy"] == pytest.approx(state["test"]["accuracy"])
    assert state["evaluation"]["test"]["size"] == 30

    # Predictions over REST: the neural layers compute the features, SVC answers.
    response = auth_client.post(
        f"/api/training/runs/{run_id}/predict/",
        {"inputs": [{"a": 2, "b": 0, "c": 0, "d": 0}, [-2, 0, 0, 0]], "top_k": 2},
        format="json",
    )
    assert response.status_code == 200, response.data
    predictions = response.data["predictions"]
    assert all(p["class"] in ("eggs", "ham", "spam") for p in predictions)
    assert all(len(p["top"]) == 2 for p in predictions)

    # The download: a ZIP file with both parts, the preprocessing and the script.
    response = auth_client.get(f"/api/training/runs/{run_id}/weights/")
    assert response.status_code == 200
    assert response["Content-Disposition"] == 'attachment; filename="mail-features.zip"'
    archive = zipfile.ZipFile(io.BytesIO(b"".join(response.streaming_content)))
    assert sorted(archive.namelist()) == [
        "README.txt",
        "mail-features.features.pt",
        "mail-features.joblib",
        "mail-features.preprocessing.json",
        "mail_features.py",
    ]
    archive.extractall(tmp_path / "download")
    folder = tmp_path / "download"
    bundle = joblib.load(folder / "mail-features.joblib")
    assert bundle["class_names"] == ["eggs", "ham", "spam"] and bundle["hybrid"]["features"] == 6
    preprocessing = json.loads((folder / "mail-features.preprocessing.json").read_text())
    assert preprocessing["columns"] == ["a", "b", "c", "d"]
    (folder / "new.csv").write_text("a,b,c,d\n2,0,0,0\n-2,0,0,0\n")
    result = subprocess.run(
        [
            sys.executable,
            "mail_features.py",
            "--predict",
            "new.csv",
            "--weights",
            "mail-features.features.pt",
            "--save",
            "mail-features.joblib",
        ],
        cwd=folder,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    printed = [line.split(",")[0] for line in result.stdout.strip().splitlines()[1:]]
    assert printed == [p["class"] for p in predictions]

    # Explanations: permutation importance through the whole pipeline; the activations of
    # the neural layers; no Grad-CAM through the classical model.
    methods = auth_client.get(f"/api/training/runs/{run_id}/explain/").data["methods"]
    available = {m["id"]: m["available"] for m in methods}
    assert available["permutation"] and available["activations"] and available["weights"]
    response = auth_client.post(
        f"/api/training/runs/{run_id}/explain/", {"method": "permutation"}, format="json"
    )
    assert response.status_code == 200, response.data
    assert {f["name"] for f in response.data["features"]} == {"a", "b", "c", "d"}
    response = auth_client.post(
        f"/api/training/runs/{run_id}/explain/",
        {"method": "activations", "sample": 0},
        format="json",
    )
    assert response.status_code == 200, response.data
    assert "fc2" in {node["id"] for node in response.data["nodes"]}
    assert "m" not in {node["id"] for node in response.data["nodes"]}

    # MCP: the summary says what the features and the classical model did.
    mcp = mcp_client(user)
    result, text = call(mcp, "get_training_run", id=run_id)
    summary = result["structuredContent"]
    assert summary["family"] == "hybrid", text
    assert summary["classical"]["hybrid"]["features"] == 6
    assert summary["epochs"] == {"done": 6, "total": 6}
    result, text = call(mcp, "get_weights_download", id=run_id)
    info = result["structuredContent"]
    assert info["filename"] == "mail-features.zip" and "ZIP file" in info["format"]
    assert "FeatureExtractor" in info["how_to_use"]


@pytest.mark.django_db
def test_mcp_analyzes_checks_and_starts_a_hybrid_network(user, real_processes):
    mcp = mcp_client(user)
    graph = EXAMPLES["hybrid_mlp_gradient_boosting"]["graph"]
    analysis, _ = call(mcp, "analyze_network", graph=graph)
    analysis = analysis["structuredContent"]
    assert analysis["valid"] and analysis["family"] == "hybrid"
    assert analysis["hybrid"]["boundary"] == "fc2" and analysis["hybrid"]["features"] == 32
    code, _ = call(mcp, "generate_pytorch_code", graph=graph)
    assert "class FeatureExtractor(nn.Module):" in code["content"][0]["text"]
    examples, text = call(mcp, "list_examples")
    assert "hybrid_mlp_gradient_boosting" in text
    checked, text = call(
        mcp,
        "check_training",
        graph=graph,
        dataset={"kind": "random"},
        config={"epochs": 2, "feature_training": "frozen"},
    )
    checked = checked["structuredContent"]
    assert checked["ok"] is True, text
    assert checked["plan"]["hybrid"]["feature_training"] == "frozen"
    started, text = call(
        mcp,
        "start_training",
        graph=graph,
        dataset={"kind": "random"},
        config={"epochs": 2, "seed": 1, "device": "cpu"},
        tuning={"method": "random", "trials": 2, "space": {"learning_rate": [0.05, 0.2]}},
    )
    run_id = started["structuredContent"]["id"]
    client = client_for(user)
    data = wait_for(client, run_id, timeout=180, interval=0.25)
    assert data["status"] == "completed", data["message"]
    assert data["trials_done"] == 2 and len(data["state"]["history"]) == 2
    result, text = call(mcp, "get_training_run", id=run_id)
    assert result["structuredContent"]["tuning"]["trials_done"] == 2, text


@pytest.mark.django_db
def test_raw_rows_and_a_hybrid_member_of_an_ensemble(real_processes, settings, tmp_path):
    from .test_data_api import import_csv

    settings.DATA_PREP_DIR = tmp_path / "prepare"
    settings.DATA_PREP_ENABLED = True
    user = get_user_model().objects.create_user("ada", password="correct-horse-1")
    client = client_for(user)
    source = import_csv(client, "payments.csv", payments_csv()).data
    body = {
        "name": "Payments",
        "source": source["id"],
        "steps": [
            {
                "id": "onehot",
                "type": "encode",
                "params": {"columns": ["merchant"], "method": "onehot"},
            }
        ],
        "settings": {
            "target": "label",
            "task": "classification",
            "split": {"val": 0.25, "test": 0.25, "seed": 1},
        },
    }
    prep = client.post("/api/data/preps/", body, format="json")
    assert prep.status_code == 201, prep.data
    exported = client.post(f"/api/data/preps/{prep.data['id']}/export/", {}, format="json")
    assert exported.status_code == 201, exported.data
    dataset = Dataset.objects.get(pk=exported.data["id"])
    features = len(dataset.info["columns"]) - 1
    ref = {"kind": "upload", "id": dataset.pk}

    hybrid = hybrid_graph(features=features, width=4)
    boosting = {
        "version": 1,
        "nodes": [
            {"id": "in", "type": "Input", "params": {"shape": [features]}},
            {
                "id": "m",
                "type": "HistGradientBoostingClassifier",
                "params": {"max_iter": 30, "min_samples_leaf": 5, "early_stopping": "off"},
            },
        ],
        "edges": [{"source": "in", "target": "m"}],
    }
    ids = []
    for name, graph, config in (
        ("Hybrid", hybrid, {"epochs": 10, "batch_size": 16, "device": "cpu", "seed": 1}),
        ("Boosting", boosting, {"seed": 1}),
    ):
        response = client.post(
            "/api/training/runs/",
            {"name": name, "graph": graph, "dataset": ref, "config": config},
            format="json",
        )
        assert response.status_code == 201, response.data
        data = wait_for(client, response.data["id"], timeout=180, interval=0.25)
        assert data["status"] == "completed", data["message"]
        ids.append(data["id"])
    hybrid_run = client.get(f"/api/training/runs/{ids[0]}/").data
    own = hybrid_run["state"]["evaluation"]
    assert own["binary"] is not None  # two classes: the threshold analysis

    raw = [
        {"amount": "250.00", "merchant": "travel", "hour": 13},
        {"amount": "12.50", "merchant": "shop", "hour": 10},
    ]
    response = client.post(
        f"/api/training/runs/{ids[0]}/predict/",
        {"inputs": raw, "input_format": "raw"},
        format="json",
    )
    assert response.status_code == 200, response.data
    alone = [p["positive_probability"] for p in response.data["predictions"]]
    assert all(0 <= p <= 1 for p in alone)

    response = create(client, ids, name="Hybrid + trees")
    assert response.status_code == 201, response.data
    run_id = response.data["id"]
    data = wait_for(client, run_id, timeout=180, interval=0.25)
    assert data["status"] == "completed", data["message"]
    members = data["ensemble"]["members"]
    assert members[0]["family"] == "hybrid"
    assert members[0]["model"] == "Gradient boosting classification on learned features"
    comparison = {row["name"]: row for row in data["state"]["ensemble"]["comparison"]}
    # The hybrid member measured on the same rows as in its own evaluation.
    assert comparison["Hybrid"]["val"]["accuracy"] == pytest.approx(own["val"]["accuracy"])
    response = client.post(
        f"/api/training/runs/{run_id}/predict/",
        {"inputs": raw, "input_format": "raw", "members": True},
        format="json",
    )
    assert response.status_code == 200, response.data
    first = response.data["members"][0]["predictions"]
    np.testing.assert_allclose([p["positive_probability"] for p in first], alone, atol=1e-6)
