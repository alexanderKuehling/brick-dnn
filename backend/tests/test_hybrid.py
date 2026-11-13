"""Hybrid networks: neural layers learn the features, a classical model is fitted on them.
The analysis and its errors, the check against data, both phases of the training, the
evaluation, predictions, the estimate and the generated script. The REST API, downloads,
raw rows, MCP and ensembles are tested in test_hybrid_api.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("sklearn")
torch = pytest.importorskip("torch")

import joblib  # noqa: E402
import numpy as np  # noqa: E402

from brickdnn import EXAMPLES, analyze, generate_code  # noqa: E402
from brickdnn.hybrid import HEAD_NODE, head_graph, split_graph  # noqa: E402
from brickdnn.training import (  # noqa: E402
    TrainingConfig,
    builtin_info,
    check_compatibility,
    inspect_csv,
    predict,
    random_info,
    runner,
)
from brickdnn.training import state as st  # noqa: E402
from brickdnn.training.estimate import estimate_training  # noqa: E402
from brickdnn.training.hybrid_engine import train  # noqa: E402
from brickdnn.training.job import TrainingJob  # noqa: E402
from brickdnn.training.tuning import TuningConfig  # noqa: E402

from .test_classical_training import Recorder, classification_csv, regression_csv  # noqa: E402

QUICK_BOOSTING = {"max_iter": 30, "min_samples_leaf": 5, "early_stopping": "off"}


def hybrid_graph(model="HistGradientBoostingClassifier", params=None, features=4, width=6):
    """Input → Linear(16) → ReLU → Linear(width) → model."""
    if params is None and model.startswith("HistGradientBoosting"):
        params = QUICK_BOOSTING
    nodes = [
        {"id": "in", "type": "Input", "params": {"shape": [features]}},
        {"id": "fc", "type": "Linear", "params": {"out_features": 16}},
        {"id": "act", "type": "Activation", "params": {}},
        {"id": "fc2", "type": "Linear", "params": {"out_features": width}},
        {"id": "m", "type": model, "params": params or {}},
    ]
    edges = [("in", "fc"), ("fc", "act"), ("act", "fc2"), ("fc2", "m")]
    return {
        "version": 1,
        "nodes": nodes,
        "edges": [{"source": a, "target": b} for a, b in edges],
    }


def graph(nodes, edges):
    return {
        "version": 1,
        "nodes": [{"id": i, "type": t, "params": p} for i, t, p in nodes],
        "edges": [{"source": a, "target": b} for a, b in edges],
    }


def binary_csv(path, n=160, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 4))
    positive = x[:, 0] + 0.5 * x[:, 1] > 0.3
    lines = ["a,b,c,d,label"] + [
        ",".join([*(f"{v:.5f}" for v in row), "yes" if p else "no"])
        for row, p in zip(x, positive, strict=True)
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


def run_job(graph_, info, path=None, tuning=None, weights=None, **config):
    job = TrainingJob(
        graph=graph_,
        dataset=info,
        config=TrainingConfig.from_dict({"seed": 1, "epochs": 4, **config}),
        path=str(path) if path else None,
        tuning=tuning,
    )
    reporter = HybridRecorder()
    result = train(job, reporter, weights)
    return result, reporter


class HybridRecorder(Recorder):
    """Also keeps what phase 2 reports beside the history."""

    def __init__(self):
        super().__init__()
        self.evaluation = None
        self.trials = []

    def evaluation_end(self, result):
        self.evaluation = result

    def trial_end(self, trial, best_trial):
        self.trials.append(trial)


# --------------------------------------------------------------------------- #
# The analysis
# --------------------------------------------------------------------------- #


def test_a_hybrid_network_is_analysed_with_its_boundary():
    analysis = analyze(hybrid_graph())
    assert analysis.valid, analysis.errors
    assert analysis.family == "hybrid"
    data = analysis.to_dict()
    assert data["hybrid"] == {"boundary": "fc2", "features": 6, "neural_params": 80 + 102}
    assert data["classical"]["task"] == "classification"
    assert data["total_params"] is None  # the trees are only known after the fit
    # The parts: the feature graph ends at the boundary, the classical graph starts there.
    parts = split_graph(hybrid_graph())
    assert [n["id"] for n in parts.feature_graph["nodes"]] == ["in", "fc", "act", "fc2"]
    assert parts.classical_graph["nodes"][0] == {
        "id": "fc2",
        "type": "Input",
        "params": {"shape": [6]},
    }
    assert analyze(parts.classical_graph).family == "classical"
    head = analyze(head_graph(parts, 3))
    assert head.family == "neural" and head.outputs == [HEAD_NODE]
    # Images: a CNN that ends in a flat vector, and an ensemble on the same features.
    cnn = graph(
        [
            ("in", "Input", {"shape": [1, 8, 8]}),
            ("conv", "Conv2d", {"out_channels": 4}),
            ("flat", "Flatten", {}),
            ("fc", "Linear", {"out_features": 8}),
            ("svc", "SVC", {}),
            ("gb", "HistGradientBoostingClassifier", {}),
            ("vote", "VotingClassifier", {}),
        ],
        [
            ("in", "conv"),
            ("conv", "flat"),
            ("flat", "fc"),
            ("fc", "svc"),
            ("fc", "gb"),
            ("svc", "vote"),
            ("gb", "vote"),
        ],
    )
    analysis = analyze(cnn)
    assert analysis.valid, analysis.errors
    assert analysis.family == "hybrid" and analysis.boundary.id == "fc"


def errors(graph_):
    return [(issue.node, issue.message) for issue in analyze(graph_).errors]


@pytest.mark.parametrize(
    "graph_,node,message",
    [
        (
            graph(
                [("in", "Input", {"shape": [1, 8, 8]}), ("c", "Conv2d", {}), ("m", "SVC", {})],
                [("in", "c"), ("c", "m")],
            ),
            "m",
            "needs a flat feature vector, but Conv2d outputs (32, 8, 8). Add Flatten (or a "
            "Linear layer) before Support vector classification",
        ),
        (
            graph(
                [
                    ("in", "Input", {"shape": [4]}),
                    ("fc", "Linear", {}),
                    ("x", "Linear", {}),
                    ("m", "SVC", {}),
                ],
                [("in", "fc"), ("fc", "m"), ("fc", "x")],
            ),
            "x",
            "Linear does not lead to Support vector classification. In a network with a "
            "classical model, the neural layers turn the input into its features, and Linear "
            "'fc' passes them on",
        ),
        (
            graph(
                [("in", "Input", {"shape": [4]}), ("m", "SVR", {}), ("fc", "Linear", {})],
                [("in", "m"), ("m", "fc")],
            ),
            "m",
            "Nothing can follow Support vector regression: a classical model ends the network",
        ),
        (
            graph(
                [
                    ("in", "Input", {"shape": [4]}),
                    ("in2", "Input", {"shape": [4]}),
                    ("cat", "Concat", {}),
                    ("fc", "Linear", {}),
                    ("m", "SVC", {}),
                ],
                [("in", "cat"), ("in2", "cat"), ("cat", "fc"), ("fc", "m")],
            ),
            "m",
            "from one Input layer: remove the other Input layers",
        ),
        (
            graph(
                [
                    ("in", "Input", {"shape": [4]}),
                    ("fc", "Linear", {}),
                    ("fc2", "Linear", {}),
                    ("a", "SVC", {}),
                    ("b", "SVC", {}),
                    ("v", "VotingClassifier", {}),
                ],
                [("in", "fc"), ("fc", "fc2"), ("fc", "a"), ("fc2", "b"), ("a", "v"), ("b", "v")],
            ),
            "v",
            "take their features from the same Input layer (or the same last neural layer)",
        ),
        (
            graph(
                [
                    ("in", "Input", {"shape": [4]}),
                    ("fc", "Linear", {}),
                    ("k", "RBF", {}),
                    ("m", "GaussianProcessRegressor", {}),
                    ("x", "Linear", {}),
                ],
                [("in", "fc"), ("fc", "m"), ("k", "m"), ("k", "x")],
            ),
            "k",
            "A kernel brick can only be connected once",
        ),
    ],
)
def test_invalid_hybrid_networks(graph_, node, message):
    found = errors(graph_)
    assert any(n == node and message in text for n, text in found), found


def test_kernels_still_go_into_the_gaussian_process_of_a_hybrid_network():
    g = graph(
        [
            ("in", "Input", {"shape": [4]}),
            ("fc", "Linear", {"out_features": 3}),
            ("rbf", "RBF", {}),
            ("gp", "GaussianProcessRegressor", {}),
        ],
        [("in", "fc"), ("fc", "gp"), ("rbf", "gp")],
    )
    analysis = analyze(g)
    assert analysis.valid and analysis.family == "hybrid"
    assert analysis.to_dict()["classical"]["kernel"] == "RBF(length_scale=1.0)"
    pipeline_code = generate_code(g, {"include_main": False})
    assert "kernel = RBF(length_scale=1.0)" in pipeline_code
    assert "class FeatureExtractor(nn.Module):" in pipeline_code


# --------------------------------------------------------------------------- #
# The check against data
# --------------------------------------------------------------------------- #


def test_the_plan_has_the_head_the_features_and_the_classical_model(tmp_path):
    info = inspect_csv(classification_csv(tmp_path / "mail.csv"))
    compat = check_compatibility(hybrid_graph(), info)
    assert compat.ok, compat.errors
    plan = compat.plan
    assert plan.task == "classification" and plan.class_names == ["eggs", "ham", "spam"]
    assert plan.targets == [{"kind": "class", "num_classes": 3}]
    assert plan.hybrid == {
        "boundary": "fc2",
        "features": 6,
        "head_outputs": 3,
        "feature_training": "pretrain",
        "neural_params": 182,
    }
    assert plan.classical["estimator"] == "HistGradientBoostingClassifier"
    assert plan.classical["class_targets"] == "indices"
    assert "neural layers learn 6 features" in plan.summary
    assert any("temporary linear head" in note for note in plan.notes)
    frozen = check_compatibility(
        hybrid_graph(), info, TrainingConfig.from_dict({"feature_training": "frozen"})
    )
    assert any("Frozen features" in note for note in frozen.plan.notes)
    regression = check_compatibility(
        hybrid_graph("SVR"), inspect_csv(regression_csv(tmp_path / "h.csv"))
    )
    assert regression.ok and regression.plan.targets == [{"kind": "value", "shape": [1]}]


def test_the_check_explains_what_does_not_fit(tmp_path):
    houses = inspect_csv(regression_csv(tmp_path / "houses.csv"))
    mail = inspect_csv(classification_csv(tmp_path / "mail.csv"))
    compat = check_compatibility(hybrid_graph(), houses)
    assert "decimal numbers, which is a regression task" in compat.errors[0]
    compat = check_compatibility(hybrid_graph("SVR"), mail)
    assert "contains text labels (3 classes), so this is a classification task" in compat.errors[0]
    compat = check_compatibility(hybrid_graph("SVR"), builtin_info("mnist"))
    assert "predicts a number, but MNIST has classes" in compat.errors[0]
    compat = check_compatibility(hybrid_graph(features=5), mail)
    assert "Set the Input shape to 4" in compat.errors[0]
    # Early stopping still needs validation data; a search of the classical model does not.
    no_val = TrainingConfig.from_dict({"val_split": 0, "patience": 3})
    assert not check_compatibility(hybrid_graph(), mail, no_val).ok
    no_val = TrainingConfig.from_dict({"val_split": 0})
    assert check_compatibility(hybrid_graph(), mail, no_val, tuning=True).ok
    # The fix for another dataset: only the Input (the head follows the data).
    fixes = check_compatibility(hybrid_graph(features=5), mail).fixes
    assert fixes == [
        {
            "node": "in",
            "type": "Input",
            "params": {"shape": [4]},
            "previous": {"shape": [5]},
            "summary": fixes[0]["summary"],
        }
    ]


# --------------------------------------------------------------------------- #
# Training: both phases
# --------------------------------------------------------------------------- #


def test_a_classifier_learns_features_then_fits_the_model(tmp_path):
    path = classification_csv(tmp_path / "mail.csv", n=150)
    info = inspect_csv(path)
    result, reporter = run_job(
        hybrid_graph(), info, path, weights=tmp_path / "model.joblib", test_split=0.2
    )
    assert result.status == "completed", result.message
    assert result.weights is True
    assert result.message.startswith("Finished: 6 features learned in 4 epochs; ")
    assert "HistGradientBoostingClassifier on them: validation accuracy" in result.message
    # Phase 1: the curves of the features (the head on the target).
    assert [entry["epoch"] for entry in reporter.history] == [1, 2, 3, 4]
    assert len(result.history) == 4 and result.weights_epoch is not None
    # Phase 2: the classical model, measured like any classical model.
    classical = reporter.info["classical"]
    assert classical["estimator"] == "HistGradientBoostingClassifier"
    assert classical["hybrid"] == {
        "boundary": "fc2",
        "features": 6,
        "feature_training": "pretrain",
        "epochs": 4,
    }
    assert classical["validation"]["accuracy"] > 0.4
    assert reporter.test is not None and reporter.test["accuracy"] is not None
    evaluation = reporter.evaluation
    assert evaluation["task"] == "classification" and evaluation["test"]["size"] == 30
    assert evaluation["class_names"] == ["eggs", "ham", "spam"]
    assert reporter.info["preprocessing"]["hybrid"] == {"boundary": "fc2", "features": 6}
    assert reporter.info["preprocessing"]["target_std"] is None
    # The files: the neural layers with their preprocessing, the classical model.
    checkpoint = torch.load(tmp_path / "weights.pt", weights_only=True)
    assert sorted(checkpoint) == ["preprocessing", "state_dict"]
    assert "linear_1.weight" in checkpoint["state_dict"]
    assert not any(key.startswith("head") for key in checkpoint["state_dict"])
    assert checkpoint["preprocessing"]["columns"] == ["a", "b", "c", "d"]
    assert (tmp_path / "weights.preprocessing.json").is_file()
    bundle = joblib.load(tmp_path / "model.joblib")
    assert bundle["columns"] == [f"feature_{i}" for i in range(1, 7)]
    assert bundle["hybrid"]["input_columns"] == ["a", "b", "c", "d"]
    assert bundle["class_names"] == ["eggs", "ham", "spam"]


def test_a_regressor_and_its_values_in_the_units_of_the_target(tmp_path):
    path = regression_csv(tmp_path / "houses.csv", n=120)
    result, reporter = run_job(
        hybrid_graph("SVR", {"C": 10.0}), inspect_csv(path), path, weights=tmp_path / "m.joblib"
    )
    assert result.status == "completed", result.message
    validation = reporter.info["classical"]["validation"]
    # The target is around 100: the classical model predicts in its units.
    assert validation["rmse"] < 10 and validation["r2"] > 0.3
    bundle = joblib.load(tmp_path / "m.joblib")
    checkpoint = torch.load(tmp_path / "weights.pt", weights_only=True)
    extractor = _extractor(hybrid_graph("SVR"), checkpoint)
    x = (np.array([[0.5, 0.0, 0.0, 0.0]]) - checkpoint["preprocessing"]["mean"]) / checkpoint[
        "preprocessing"
    ]["std"]
    with torch.no_grad():
        features = extractor(torch.tensor(x, dtype=torch.float32)).double().numpy()
    assert 95 < float(bundle["model"].predict(features)[0]) < 110


def _extractor(graph_, checkpoint):
    from brickdnn.training.engine import build_model

    model = build_model(split_graph(graph_).feature_graph)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()


def test_frozen_random_features_train_nothing(tmp_path):
    path = classification_csv(tmp_path / "mail.csv")
    result, reporter = run_job(
        hybrid_graph(),
        inspect_csv(path),
        path,
        weights=tmp_path / "m.joblib",
        feature_training="frozen",
    )
    assert result.status == "completed", result.message
    assert result.history == [] and reporter.history == []
    assert reporter.info["epochs"] == 0
    assert result.message.startswith("Finished: 6 frozen random features; ")
    assert any("Frozen features" in line for line in reporter.logs)
    # The same seed gives the same random features.
    first = torch.load(tmp_path / "weights.pt", weights_only=True)["state_dict"]
    run_job(
        hybrid_graph(),
        inspect_csv(path),
        path,
        weights=tmp_path / "m.joblib",
        feature_training="frozen",
    )
    second = torch.load(tmp_path / "weights.pt", weights_only=True)["state_dict"]
    assert all(torch.equal(first[k], second[k]) for k in first)


def test_two_classes_get_the_threshold_analysis(tmp_path):
    path = binary_csv(tmp_path / "binary.csv")
    result, reporter = run_job(
        hybrid_graph("SVC"), inspect_csv(path), path, weights=tmp_path / "m.joblib"
    )
    assert result.status == "completed", result.message
    binary = reporter.evaluation["binary"]
    assert binary["positive_class"] == 1 and binary["source"] == "val"
    curve = reporter.evaluation["val"]["threshold_curve"]
    assert curve["roc_auc"] > 0.6 and len(curve["points"]["threshold"]) > 10


def test_a_search_varies_the_classical_model_on_the_learned_features(tmp_path):
    path = classification_csv(tmp_path / "mail.csv")
    model = "HistGradientBoostingClassifier"
    tuning = TuningConfig.from_dict(
        {"method": "grid", "space": {"learning_rate": [0.05, 0.2]}, "cv_folds": 3}, model
    )
    result, reporter = run_job(
        hybrid_graph(), inspect_csv(path), path, tuning=tuning, weights=tmp_path / "m.joblib"
    )
    assert result.status == "completed", result.message
    assert [t["params"] for t in reporter.trials] == [
        {"learning_rate": 0.05},
        {"learning_rate": 0.2},
    ]
    assert result.best_trial in (1, 2) and len(result.history) == 4  # the feature curves
    assert reporter.evaluation["trial"] == result.best_trial
    assert "on them: finished 2 trials. Best: trial" in result.message


def test_demo_data_and_the_runner(tmp_path):
    run_dir = tmp_path / "run"
    st.write_job(
        run_dir,
        {
            "graph": hybrid_graph(),
            "dataset": {"kind": "random"},
            "config": {"seed": 2, "epochs": 2, "batch_size": 64},
        },
    )
    state = runner.run(run_dir)
    assert state["status"] == "completed", state["message"]
    assert state["family"] == "hybrid" and state["weights"] is True
    assert len(state["history"]) == 2 and state["weights_epoch"] in (1, 2)
    assert state["classical"]["hybrid"]["features"] == 6
    assert state["num_classes"] == 3
    assert sorted(p.name for p in run_dir.iterdir()) == [
        "job.json",
        "model.joblib",
        "state.json",
        "weights.pt",
    ]
    # Predictions: the neural layers compute the features, the model answers.
    answer = predict.predict(
        run_dir, {"inputs": [{"shape": [2, 4], "values": [0.1] * 8}], "kind": "demo"}
    )
    assert answer["probabilities"] is True and len(answer["outputs"][0]) == 2
    assert sum(answer["outputs"][0][0]) == pytest.approx(1.0)


def test_stopping_while_the_features_are_learned_keeps_nothing(tmp_path):
    path = classification_csv(tmp_path / "mail.csv")
    job = TrainingJob(
        graph=hybrid_graph(),
        dataset=inspect_csv(path),
        config=TrainingConfig.from_dict({"seed": 1, "epochs": 50}),
        path=str(path),
    )

    class StopInEpoch(HybridRecorder):
        def should_stop(self):
            return bool(self.history)

    reporter = StopInEpoch()
    result = train(job, reporter, tmp_path / "m.joblib")
    assert result.status == "stopped" and result.weights is False
    assert "nothing was saved" in result.message
    assert reporter.finished[0] == "stopped"
    assert not (tmp_path / "m.joblib").exists()


# --------------------------------------------------------------------------- #
# Predictions, the estimate and the generated script
# --------------------------------------------------------------------------- #


def test_predictions_of_a_tabular_run(tmp_path):
    path = classification_csv(tmp_path / "mail.csv")
    run_dir = tmp_path / "run"
    info = inspect_csv(path)
    st.write_job(
        run_dir,
        {
            "graph": hybrid_graph("SVC"),
            "dataset": {"info": info.to_dict(), "path": str(path)},
            "config": {"seed": 1, "epochs": 8},
        },
    )
    state = runner.run(run_dir)
    assert state["status"] == "completed", state["message"]
    from brickdnn.training import inference

    plan = check_compatibility(hybrid_graph("SVC"), info).plan.to_dict()
    prepared = inference.prepare_inputs(
        plan, state["preprocessing"], [{"a": 2, "b": 0, "c": 0, "d": 0}, [-2, 0, 0, 0]]
    )
    answer = predict.predict(run_dir, {"inputs": prepared.tensors, "kind": prepared.kind})
    predictions = inference.read_outputs(
        plan, state["preprocessing"], answer["outputs"], 2, probabilities=True
    )
    assert [p["class"] for p in predictions] == ["spam", "ham"]
    snippet = inference.usage_snippet(plan, state["preprocessing"], "mail", "mail.zip")
    assert 'torch.load("mail.features.pt"' in snippet and 'joblib.load("mail.joblib")' in snippet
    assert "model.predict_proba(features)" in snippet


def test_the_estimate_adds_both_phases(tmp_path):
    info = inspect_csv(classification_csv(tmp_path / "mail.csv", n=900))
    analysis = analyze(hybrid_graph())
    plan = check_compatibility(analysis, info).plan.to_dict()
    config = TrainingConfig.from_dict({"epochs": 20})
    estimate = estimate_training(analysis, info, config, plan)
    details = estimate.details
    assert details["family"] == "hybrid" and details["features"] == 6
    assert details["epochs"] == 20 and details["neural_cpu_seconds"] > 0
    assert details["classical"]["trees"] == 30 * 3
    assert "neural layers (20 epochs" in estimate.basis and "gradient boosting" in estimate.basis
    frozen = estimate_training(
        analysis, info, TrainingConfig.from_dict({"epochs": 20, "feature_training": "frozen"}), plan
    )
    assert frozen.details["epochs"] == 0
    assert frozen.details["neural_cpu_seconds"] < details["neural_cpu_seconds"]
    # A search multiplies the classical part only.
    searched = estimate_training(analysis, info, config, plan, trials=10, folds=None)
    assert searched.details["neural_cpu_seconds"] == details["neural_cpu_seconds"]
    assert searched.details["classical_cpu_seconds"] > details["classical_cpu_seconds"]
    random = estimate_training(analysis, random_info(), config, None)
    assert random is not None and random.cpu_seconds > 0


@pytest.mark.parametrize("model", ["HistGradientBoostingClassifier", "SVR"])
def test_the_generated_script_trains_both_phases_on_a_csv_file(model, tmp_path):
    graph_ = hybrid_graph(model)
    code = generate_code(graph_)
    assert "class FeatureExtractor(nn.Module):" in code
    assert "def learn_features(" in code and "def make_model() -> Pipeline:" in code
    script = tmp_path / "hybrid.py"
    script.write_text(code)
    make = classification_csv if model.endswith("Classifier") else regression_csv
    make(tmp_path / "data.csv")
    run = subprocess.run(
        [sys.executable, str(script), "--csv", "data.csv", "--epochs", "3"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert run.returncode == 0, run.stderr
    assert "Epoch   3/3" in run.stdout and "Saved the neural layers to features.pt" in run.stdout
    assert ("Validation accuracy" if model.endswith("Classifier") else "Validation RMSE") in (
        run.stdout
    )
    header = (tmp_path / "data.csv").read_text().splitlines()[:3]
    (tmp_path / "new.csv").write_text("\n".join(header) + "\n")
    run = subprocess.run(
        [sys.executable, str(script), "--predict", "new.csv"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert run.returncode == 0, run.stderr
    assert len(run.stdout.strip().splitlines()) == 3  # the target name and two rows
    frozen = subprocess.run(
        [sys.executable, str(script), "--csv", "data.csv", "--frozen", "--save", "f.joblib"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert frozen.returncode == 0 and "Frozen random features" in frozen.stdout


def test_the_script_predicts_with_the_files_of_a_trained_run(tmp_path):
    """weights.pt and model.joblib of a run are the script's --weights and --save."""
    path = classification_csv(tmp_path / "mail.csv")
    result, _ = run_job(hybrid_graph(), inspect_csv(path), path, weights=tmp_path / "m.joblib")
    assert result.status == "completed"
    code = generate_code(hybrid_graph(), {"include_training": False})
    assert "def learn_features(" not in code
    (tmp_path / "mail.py").write_text(code)
    (tmp_path / "new.csv").write_text("a,b,c,d\n2,0,0,0\n-2,0,0,0\n")
    run = subprocess.run(
        [
            sys.executable,
            "mail.py",
            "--predict",
            "new.csv",
            "--weights",
            "weights.pt",
            "--save",
            "m.joblib",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert run.returncode == 0, run.stderr
    lines = run.stdout.strip().splitlines()
    assert lines[0] == "label,probability" and len(lines) == 3


@pytest.mark.skipif(__import__("shutil").which("ruff") is None, reason="ruff is not installed")
@pytest.mark.parametrize("options", [{}, {"include_main": False}, {"include_training": False}])
def test_the_example_script_is_lint_clean(options, tmp_path):
    import shutil

    target = tmp_path / "model.py"
    target.write_text(generate_code(EXAMPLES["hybrid_mlp_gradient_boosting"]["graph"], options))
    ruff = shutil.which("ruff")
    for args in (
        ["check", "--isolated", "--select", "E,F,W,I", "--line-length", "100"],
        ["format", "--isolated", "--line-length", "100", "--check"],
    ):
        result = subprocess.run([ruff, *args, str(target)], capture_output=True, text=True)
        assert result.returncode == 0, result.stdout


def test_the_command_line_trains_a_hybrid_network(tmp_path, capsys):
    from brickdnn.training.cli import main

    path = classification_csv(tmp_path / "mail.csv")
    (tmp_path / "net.json").write_text(json.dumps(hybrid_graph()))
    save = tmp_path / "net.joblib"
    code = main(
        [
            str(tmp_path / "net.json"),
            "--data",
            str(path),
            "--epochs",
            "2",
            "--seed",
            "1",
            "--save",
            str(save),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0, out
    assert save.is_file() and (tmp_path / "net.features.pt").is_file()
    assert "Saved the neural layers (features) to" in out
    assert Path(tmp_path / "net.features.preprocessing.json").is_file()
