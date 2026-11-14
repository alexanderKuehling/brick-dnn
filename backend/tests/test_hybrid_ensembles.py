"""Ensemble bricks after neural layers: an MLP learns the features, gradient boosting and a
support vector machine are fitted on them and combined by soft voting (or stacking for a
regression). The analysis, both phases of the training on a CSV file, each model measured
on its own, predictions and the generated script."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

pytest.importorskip("sklearn")
torch = pytest.importorskip("torch")

import joblib  # noqa: E402
import numpy as np  # noqa: E402

from brickdnn import analyze, generate_code  # noqa: E402
from brickdnn.hybrid import split_graph  # noqa: E402
from brickdnn.training import (  # noqa: E402
    check_compatibility,
    inference,
    inspect_csv,
    predict,
    runner,
)
from brickdnn.training import state as st  # noqa: E402

from .test_classical_training import classification_csv, regression_csv  # noqa: E402
from .test_hybrid import run_job  # noqa: E402

QUICK_BOOSTING = {"max_iter": 30, "min_samples_leaf": 5, "early_stopping": "off"}


def ensemble_hybrid(task="classification", ensemble=None, params=None):
    """Input → Linear(16) → ReLU → Linear(8) → gradient boosting and an SVM → the ensemble."""
    classification = task == "classification"
    ensemble = ensemble or ("VotingClassifier" if classification else "StackingRegressor")
    if params is None:
        params = {"voting": "soft"} if ensemble == "VotingClassifier" else {"cv": 3}
    boosting = "HistGradientBoosting" + ("Classifier" if classification else "Regressor")
    nodes = [
        ("in", "Input", {"shape": [4]}),
        ("fc", "Linear", {"out_features": 16}),
        ("act", "Activation", {}),
        ("fc2", "Linear", {"out_features": 8}),
        ("gb", boosting, QUICK_BOOSTING),
        ("svm", "SVC" if classification else "SVR", {"C": 10.0}),
        ("ens", ensemble, params),
    ]
    edges = [
        ("in", "fc"),
        ("fc", "act"),
        ("act", "fc2"),
        ("fc2", "gb"),
        ("fc2", "svm"),
        ("gb", "ens"),
        ("svm", "ens"),
    ]
    return {
        "version": 1,
        "nodes": [{"id": i, "type": t, "params": p} for i, t, p in nodes],
        "edges": [{"source": a, "target": b} for a, b in edges],
    }


@pytest.mark.parametrize(
    "task,ensemble", [("classification", "VotingClassifier"), ("regression", "StackingRegressor")]
)
def test_the_analysis_and_the_plan_of_an_ensemble_on_learned_features(task, ensemble, tmp_path):
    graph = ensemble_hybrid(task)
    analysis = analyze(graph)
    assert analysis.valid, analysis.errors
    assert analysis.family == "hybrid" and analysis.boundary.id == "fc2"
    data = analysis.to_dict()
    assert data["hybrid"]["features"] == 8
    assert data["classical"]["type"] == ensemble and data["classical"]["task"] == task
    assert [m["name"] for m in data["classical"]["members"]] == ["gb", "svm"]
    # Both models take the 8 features of the boundary.
    parts = split_graph(graph)
    assert parts.classical_graph["nodes"][0] == {
        "id": "fc2",
        "type": "Input",
        "params": {"shape": [8]},
    }
    assert {
        e["source"] for e in parts.classical_graph["edges"] if e["target"] in ("gb", "svm")
    } == {"fc2"}
    make = classification_csv if task == "classification" else regression_csv
    compat = check_compatibility(analysis, inspect_csv(make(tmp_path / "data.csv")))
    assert compat.ok, compat.errors
    assert compat.plan.hybrid["features"] == 8
    assert [m["name"] for m in compat.plan.classical["members"]] == ["gb", "svm"]
    assert any("ensemble fits each of its models" in note for note in compat.plan.notes)


def test_soft_voting_on_learned_features(tmp_path):
    path = classification_csv(tmp_path / "mail.csv", n=150)
    result, reporter = run_job(
        ensemble_hybrid(), inspect_csv(path), path, weights=tmp_path / "m.joblib", test_split=0.2
    )
    assert result.status == "completed", result.message
    assert len(result.history) == 4
    assert result.message.startswith("Finished: 8 features learned in 4 epochs; VotingClassifier")
    assert any("Fitting the voting ensemble of 2 models" in m for _, m in reporter.statuses)
    classical = reporter.info["classical"]
    assert classical["estimator"] == "VotingClassifier" and classical["ensemble"] == "voting"
    assert classical["hybrid"]["features"] == 8
    # Every model measured on its own on the validation rows, beside the ensemble.
    members = classical["members"]
    assert [m["name"] for m in members] == ["gb", "svm"]
    assert all(0 <= m["metrics"]["accuracy"] <= 1 for m in members)
    assert classical["validation"]["accuracy"] is not None
    assert reporter.test["accuracy"] is not None
    assert reporter.evaluation["test"]["size"] == 30
    bundle = joblib.load(tmp_path / "m.joblib")
    voting = bundle["model"].named_steps["model"]
    assert type(voting).__name__ == "VotingClassifier" and voting.voting == "soft"
    assert list(voting.named_estimators_) == ["gb", "svm"]
    assert bundle["columns"] == [f"feature_{i}" for i in range(1, 9)]


def test_stacking_regression_on_learned_features(tmp_path):
    path = regression_csv(tmp_path / "houses.csv", n=150)
    result, reporter = run_job(
        ensemble_hybrid("regression"),
        inspect_csv(path),
        path,
        weights=tmp_path / "m.joblib",
        test_split=0.2,
    )
    assert result.status == "completed", result.message
    classical = reporter.info["classical"]
    assert classical["estimator"] == "StackingRegressor" and classical["ensemble"] == "stacking"
    assert [m["name"] for m in classical["members"]] == ["gb", "svm"]
    assert all(m["metrics"]["rmse"] is not None for m in classical["members"])
    # In the units of the target (around 100), like a single model on the features.
    assert classical["validation"]["rmse"] < 15
    assert reporter.evaluation["task"] == "regression"
    stacking = joblib.load(tmp_path / "m.joblib")["model"].named_steps["model"]
    assert type(stacking).__name__ == "StackingRegressor" and stacking.cv == 3


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_predictions_of_an_ensemble_on_learned_features(task, tmp_path):
    make = classification_csv if task == "classification" else regression_csv
    path = make(tmp_path / "data.csv", n=150)
    info = inspect_csv(path)
    graph = ensemble_hybrid(task)
    run_dir = tmp_path / "run"
    st.write_job(
        run_dir,
        {
            "graph": graph,
            "dataset": {"info": info.to_dict(), "path": str(path)},
            "config": {"seed": 1, "epochs": 6},
        },
    )
    state = runner.run(run_dir)
    assert state["status"] == "completed", state["message"]
    assert state["classical"]["members"][1]["name"] == "svm"
    plan = check_compatibility(graph, info).plan.to_dict()
    preprocessing = state["preprocessing"]
    columns = preprocessing["columns"]
    rows = [dict(zip(columns, [2, 0, 0, 0], strict=True)), [-2, 0, 0, 0]]
    prepared = inference.prepare_inputs(plan, preprocessing, rows)
    answer = predict.predict(run_dir, {"inputs": prepared.tensors, "kind": prepared.kind})
    predictions = inference.read_outputs(
        plan,
        preprocessing,
        answer["outputs"],
        2,
        probabilities=bool(answer.get("probabilities")),
    )
    # The same as the neural layers and the ensemble one after the other.
    from brickdnn.training.hybrid_engine import features_of, load_extractor

    extractor, saved = load_extractor(run_dir, graph)
    raw = np.array([[2, 0, 0, 0], [-2, 0, 0, 0]], dtype=np.float64)
    x = torch.tensor((raw - saved["mean"]) / saved["std"], dtype=torch.float32)
    model = joblib.load(run_dir / "model.joblib")["model"]
    expected = model.predict(features_of(extractor, [x]))
    if task == "classification":
        assert answer["probabilities"] is True
        assert [p["class"] for p in predictions] == [plan["class_names"][c] for c in expected]
    else:
        assert [p["value"] for p in predictions] == pytest.approx(list(expected), rel=1e-4)


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_the_generated_script_trains_the_ensemble_on_learned_features(task, tmp_path):
    graph = ensemble_hybrid(task)
    code = generate_code(graph)
    ensemble = "VotingClassifier" if task == "classification" else "StackingRegressor"
    assert "class FeatureExtractor(nn.Module):" in code and f"{ensemble}(" in code
    (tmp_path / "hybrid.py").write_text(code)
    make = classification_csv if task == "classification" else regression_csv
    make(tmp_path / "data.csv", n=150)

    def script(*args):
        # One OpenMP thread: gradient boosting fitted again and again on tiny data (stacking)
        # stalls with more threads when the CPUs are busy.
        return subprocess.run(
            [sys.executable, "hybrid.py", *args],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=300,
            env={**os.environ, "OMP_NUM_THREADS": "1"},
        )

    run = script("--csv", "data.csv", "--epochs", "3")
    assert run.returncode == 0, run.stderr
    assert "Saved the neural layers to features.pt" in run.stdout
    header = (tmp_path / "data.csv").read_text().splitlines()[:3]
    (tmp_path / "new.csv").write_text("\n".join(header) + "\n")
    run = script("--predict", "new.csv")
    assert run.returncode == 0, run.stderr
    assert len(run.stdout.strip().splitlines()) == 3  # the header and two rows
