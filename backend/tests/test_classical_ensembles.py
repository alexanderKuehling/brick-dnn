"""Ensembles of classical models (voting and stacking): the analysis rules, the generated
script, training with the classical engine, predictions, the estimate, explanations and the
MCP tools. Single classical models are tested in test_classical.py and
test_classical_training.py."""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys

import pytest

from brickdnn import EXAMPLES, analyze, generate_code, layer_specs
from brickdnn.layers import SVC

try:
    import sklearn
except ImportError:  # pragma: no cover - scikit-learn is optional for the package
    sklearn = None

needs_sklearn = pytest.mark.skipif(sklearn is None, reason="scikit-learn is not installed")

ENSEMBLES = ["VotingClassifier", "VotingRegressor", "StackingClassifier", "StackingRegressor"]
# Gradient boosting on the small test files: fewer iterations, smaller leaves.
QUICK_BOOSTING = {"max_iter": 30, "min_samples_leaf": 5}
CLASSIFIERS = {
    "gb": ("HistGradientBoostingClassifier", QUICK_BOOSTING),
    "svc": ("SVC", {}),
    "gp": ("GaussianProcessClassifier", {}),
}
REGRESSORS = {
    "gb": ("HistGradientBoostingRegressor", QUICK_BOOSTING),
    "svr": ("SVR", {}),
    "gp": ("GaussianProcessRegressor", {}),
}


def ensemble_graph(ensemble="VotingClassifier", params=None, members=None, features=4):
    """Input -> every model -> the ensemble. ``members``: {id: (type, params)}."""
    if members is None:
        members = CLASSIFIERS if ensemble.endswith("Classifier") else REGRESSORS
    nodes = [{"id": "in", "type": "Input", "params": {"shape": [features]}}]
    edges = []
    for node_id, (layer_type, layer_params) in members.items():
        nodes.append({"id": node_id, "type": layer_type, "params": layer_params})
        edges.append({"source": "in", "target": node_id})
    nodes.append({"id": "ens", "type": ensemble, "params": params or {}})
    edges += [{"source": node_id, "target": "ens"} for node_id in members]
    return {"version": 1, "nodes": nodes, "edges": edges}


def with_nodes(graph, nodes=(), edges=()):
    return {
        **graph,
        "nodes": [*graph["nodes"], *({"id": i, "type": t, "params": p} for i, t, p in nodes)],
        "edges": [*graph["edges"], *({"source": s, "target": t} for s, t in edges)],
    }


def errors(graph):
    return [(e.node, e.message) for e in analyze(graph).errors]


# --------------------------------------------------------------------------- #
# Catalogue and analysis
# --------------------------------------------------------------------------- #


def test_ensemble_bricks_in_the_catalogue():
    specs = {spec["type"]: spec for spec in layer_specs()}
    for ensemble in ENSEMBLES:
        spec = specs[ensemble]
        assert spec["category"] == "Ensembles" and spec["family"] == "classical"
        assert spec["output_kind"] == "estimator"
        assert spec["input_slots"] == [
            {"name": "models", "label": "Models", "kind": "estimator", "min": 2, "max": None}
        ]
    params = {name: [p["name"] for p in specs[name]["params"]] for name in ENSEMBLES}
    assert params == {
        "VotingClassifier": ["voting", "weights"],
        "VotingRegressor": ["weights"],
        "StackingClassifier": ["cv", "passthrough"],
        "StackingRegressor": ["cv", "passthrough"],
    }
    voting = specs["VotingClassifier"]["params"]
    assert voting[0]["options"] == ["soft", "hard"] and voting[0]["default"] == "soft"
    assert voting[1]["kind"] == "ints" and voting[1]["auto"] is True
    assert specs["StackingRegressor"]["params"][0]["default"] == 5


@pytest.mark.parametrize("ensemble", ENSEMBLES)
def test_an_ensemble_lists_its_models(ensemble):
    result = analyze(ensemble_graph(ensemble))
    assert result.valid, result.errors
    assert result.model.id == "ens" and result.outputs == ["ens"]
    data = result.to_dict()
    classification = ensemble.endswith("Classifier")
    names = ["gb", "svc", "gp"] if classification else ["gb", "svr", "gp"]
    members = data["nodes"]["ens"]["members"]
    assert [m["name"] for m in members] == names and [m["node"] for m in members] == names
    assert data["classical"]["members"] == members
    assert data["classical"]["type"] == ensemble and data["classical"]["kernel"] is None
    assert data["nodes"]["ens"]["task"] == ("classification" if classification else "regression")
    assert data["nodes"]["ens"]["output_shape"] == (None if classification else [1])
    # The models pass themselves on: their connections carry an estimator.
    assert {data["nodes"][n]["output_kind"] for n in names} == {"estimator"}
    assert "members" not in data["nodes"]["gb"]
    assert data["total_params"] is None and data["family"] == "classical"


def test_names_of_the_models():
    members = {
        "a": ("SVC", {}),
        "b": ("SVC", {}),
        "c": ("SVC", {}),
        "d": ("GaussianProcessClassifier", {}),
        "e": ("HistGradientBoostingClassifier", QUICK_BOOSTING),
        "f": ("SVC", {}),
    }
    graph = ensemble_graph(members=members)
    named = {"a": "Boosting 1", "b": "boosting-1", "c": "1st", "d": "model", "e": "class"}
    for node in graph["nodes"]:
        if node["id"] in named:
            node["name"] = named[node["id"]]
    result = analyze(graph)
    assert result.valid, result.errors
    names = [m.name for m in result.model.layer.members]
    # Lower-case identifiers, unique, and none that make_model() or scikit-learn use.
    assert names == ["boosting_1", "boosting_1_2", "svc_1st", "model_gp", "class_gb", "f"]


@pytest.mark.parametrize(
    "graph,node,message",
    [
        (
            ensemble_graph(members={"svc": ("SVC", {})}),
            "ens",
            "Voting classifier combines at least 2 models, but only one is connected.",
        ),
        (
            ensemble_graph(members={"svc": ("SVC", {}), "svr": ("SVR", {})}),
            "ens",
            "Voting classifier combines classifiers, but Support vector regression is a "
            "regression model",
        ),
        (
            ensemble_graph("StackingRegressor", members={"a": ("SVR", {}), "b": ("SVC", {})}),
            "ens",
            "Stacking regression combines regression models, but Support vector "
            "classification is a classifier",
        ),
        (
            with_nodes(ensemble_graph(), edges=[("in", "ens")]),
            "ens",
            "takes no data (Input): connect model bricks",
        ),
        (
            with_nodes(ensemble_graph(), [("k", "RBF", {})], [("k", "ens")]),
            "ens",
            "takes no kernel (RBF kernel)",
        ),
        (
            with_nodes(
                ensemble_graph(),
                [("svc2", "SVC", {}), ("outer", "VotingClassifier", {})],
                [("in", "svc2"), ("ens", "outer"), ("svc2", "outer")],
            ),
            "outer",
            "Ensembles can't be nested yet",
        ),
        (
            with_nodes(
                ensemble_graph(members={"svc": ("SVC", {})}),
                [("in2", "Input", {"shape": [4]}), ("svc2", "SVC", {})],
                [("in2", "svc2"), ("svc2", "ens")],
            ),
            "ens",
            "take their features from the same Input layer (or the same last neural layer)",
        ),
        (
            ensemble_graph(params={"weights": [1, 2]}),
            "ens",
            "Weights has 2 values, but 3 models are connected (in this order: gb, svc, gp).",
        ),
        (
            ensemble_graph("VotingRegressor", params={"weights": [0, 0, 0]}),
            "ens",
            "At least one weight must be greater than 0.",
        ),
        (
            with_nodes(ensemble_graph(), [("lonely", "SVC", {})], [("in", "lonely")]),
            "lonely",
            "Support vector classification is not connected to Voting classifier",
        ),
        (
            with_nodes(ensemble_graph(), [("fc", "Linear", {})], [("svc", "fc")]),
            "svc",
            "can only be connected to Voting classifier",
        ),
        (
            with_nodes(ensemble_graph(), [("fc", "Linear", {})], [("in", "fc")]),
            "fc",
            "Linear does not lead to Voting classifier, which takes its features straight "
            "from the Input layer",
        ),
        (
            with_nodes(ensemble_graph(), [("fc", "Linear", {})], [("ens", "fc")]),
            "ens",
            "Nothing can follow Voting classifier",
        ),
        (
            with_nodes(
                ensemble_graph(),
                [("svc2", "SVC", {}), ("svc3", "SVC", {}), ("other", "StackingClassifier", {})],
                [("in", "svc2"), ("in", "svc3"), ("svc2", "other"), ("svc3", "other")],
            ),
            "other",
            "Only one ensemble per network",
        ),
        (
            {
                "nodes": [
                    {"id": "in", "type": "Input", "params": {"shape": [4]}},
                    {"id": "a", "type": "SVC", "params": {}},
                    {"id": "b", "type": "SVC", "params": {}},
                ],
                "edges": [{"source": "in", "target": "a"}, {"source": "a", "target": "b"}],
            },
            "b",
            "takes its features from the Input layer (or from neural layers), not the "
            "predictions of Support vector classification. To combine models, connect each "
            "of them to an ensemble",
        ),
    ],
)
def test_invalid_ensembles(graph, node, message):
    found = errors(graph)
    assert any(n == node and message in text for n, text in found), found


def test_a_bad_model_is_reported_on_the_model():
    graph = ensemble_graph(members={"svc": ("SVC", {"C": 0}), "gp": ("SVC", {})})
    assert errors(graph) == [("svc", "C must be greater than 0.")]


def test_soft_voting_needs_probabilities(monkeypatch):
    # All classifier bricks predict probabilities (SVC with Platt scaling): pretend one does
    # not.
    monkeypatch.setattr(SVC, "probabilities", property(lambda self: False))
    assert errors(ensemble_graph()) == [
        (
            "ens",
            "Soft voting averages class probabilities, but Support vector classification "
            "predicts none. Choose hard voting, or another model.",
        )
    ]
    assert analyze(ensemble_graph(params={"voting": "hard"})).valid
    assert analyze(ensemble_graph("StackingClassifier")).valid is False


def test_check_on_the_command_line(capsys):
    from brickdnn.__main__ import main

    assert main(["--example", "voting_ensemble_classifier", "--check"]) == 0
    out = capsys.readouterr().out
    assert "-> estimator" in out and "OK - a voting ensemble of 3 classical models" in out


# --------------------------------------------------------------------------- #
# The generated script
# --------------------------------------------------------------------------- #


def test_script_builds_the_ensemble():
    code = generate_code(ensemble_graph(params={"weights": [2, 1, 1]}))
    ast.parse(code)
    assert "from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier" in code
    assert "from sklearn.calibration import CalibratedClassifierCV" in code
    assert "from sklearn.gaussian_process.kernels import RBF, ConstantKernel" in code
    assert '    svc_svm = SVC(kernel="rbf", C=1.0, gamma="scale")\n' in code
    assert "    svc = CalibratedClassifierCV(svc_svm, ensemble=False)\n" in code
    assert "    gp_kernel = ConstantKernel(constant_value=1.0) * RBF(length_scale=1.0)\n" in code
    assert "    gp = GaussianProcessClassifier(kernel=gp_kernel)\n" in code
    assert (
        "    model = VotingClassifier(\n"
        '        estimators=[("gb", gb), ("svc", svc), ("gp", gp)],\n'
        '        voting="soft",\n'
        "        weights=[2, 1, 1],\n"
        "    )\n"
    ) in code
    assert "weighted 2:1:1." in code
    assert 'print(f"Models: {' in code

    code = generate_code(ensemble_graph("StackingClassifier", {"cv": 3, "passthrough": True}))
    assert "from sklearn.linear_model import LogisticRegression" in code
    assert (
        "        final_estimator=LogisticRegression(),\n        cv=3,\n        passthrough=True,\n"
    ) in code
    assert 'print(f"Final estimator: {ensemble.final_estimator_}")' in code
    code = generate_code(ensemble_graph("StackingRegressor"))
    assert "from sklearn.linear_model import RidgeCV" in code
    assert "final_estimator=RidgeCV()" in code and "passthrough=False" in code
    code = generate_code(ensemble_graph("VotingRegressor"))
    assert "weights" not in code.split("def make_model")[1].split("return Pipeline")[0]
    # Hard voting predicts classes, but no probabilities.
    code = generate_code(ensemble_graph(params={"voting": "hard"}))
    assert "for label in model.predict(features):" in code
    assert "model.predict_proba(features)" not in code


def test_long_lists_of_models_are_wrapped():
    members = {f"support_vector_machine_{i}": ("SVR", {}) for i in range(4)}
    code = generate_code(ensemble_graph("VotingRegressor", members=members))
    assert (
        "        estimators=[\n"
        '            ("support_vector_machine_0", support_vector_machine_0),\n'
    ) in code


@needs_sklearn
def test_kernel_variables_do_not_take_the_names_of_models():
    """A Gaussian process writes its kernel (and, if it is long, a variable per kernel
    brick) into variables of its own, which must not overwrite another model."""
    long_kernel = {
        "hyperparameters": "custom bounds",
        "length_scale_min": 0.001,
        "length_scale_max": 1000,
    }
    graph = ensemble_graph(
        members={
            "gp_kernel": ("SVC", {}),
            "gp_rbf": ("SVC", {}),
            "gp": ("GaussianProcessClassifier", {}),
        }
    )
    graph = with_nodes(
        graph,
        [("r", "RBF", long_kernel), ("m", "Matern", long_kernel), ("s", "KernelSum", {})],
        [("r", "s"), ("m", "s"), ("s", "gp")],
    )
    code = generate_code(graph, {"include_main": False})
    assert "    gp_kernel_ = gp_rbf_k + gp_matern\n" in code
    namespace: dict = {"__name__": "generated"}
    exec(compile(code, "<generated>", "exec"), namespace)
    ensemble = namespace["make_model"]().named_steps["model"]
    kinds = {name: type(model).__name__ for name, model in ensemble.estimators}
    assert kinds == {
        "gp_kernel": "CalibratedClassifierCV",
        "gp_rbf": "CalibratedClassifierCV",
        "gp": "GaussianProcessClassifier",
    }
    assert len(ensemble.estimators[2][1].kernel.hyperparameters) == 2


def write_csv(path, regression, n=90):
    import numpy as np

    rng = np.random.default_rng(4)
    x = rng.normal(size=(n, 4))
    if regression:
        target = [f"{v:.5f}" for v in 2 * x[:, 0] - x[:, 1] + 0.05 * rng.normal(size=n)]
        header = "a,b,c,d,price"
    else:
        target = np.where(x[:, 0] + x[:, 2] > 0, "spam", "ham")
        header = "a,b,c,d,label"
    lines = [header] + [
        ",".join([*(f"{v:.5f}" for v in row), str(t)]) for row, t in zip(x, target, strict=True)
    ]
    path.write_text("\n".join(lines) + "\n")


@needs_sklearn
@pytest.mark.parametrize(
    "ensemble,params",
    [
        ("VotingClassifier", {"weights": [2, 1, 1]}),
        ("VotingClassifier", {"voting": "hard"}),
        ("StackingClassifier", {"cv": 3}),
        ("VotingRegressor", {}),
        ("StackingRegressor", {"passthrough": True}),
    ],
)
def test_script_fits_and_predicts(ensemble, params, tmp_path):
    import joblib

    regression = ensemble.endswith("Regressor")
    write_csv(tmp_path / "data.csv", regression)
    code = generate_code(ensemble_graph(ensemble, params))
    script = tmp_path / "model.py"
    script.write_text(code)
    ruff = shutil.which("ruff")
    if ruff:
        check = [ruff, "check", "--isolated", "--select", "E,F,W,I", "--line-length", "100"]
        lint = subprocess.run([*check, str(script)], capture_output=True, text=True)
        assert lint.returncode == 0, lint.stdout
        form = [ruff, "format", "--isolated", "--line-length", "100", "--check", str(script)]
        assert subprocess.run(form, capture_output=True).returncode == 0

    def run(*args):
        # One OpenMP thread: gradient boosting fitted again and again on tiny data (stacking)
        # stalls with more threads when the CPUs are busy.
        return subprocess.run(
            [sys.executable, str(script), *args],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            timeout=300,
            env={**os.environ, "OMP_NUM_THREADS": "1"},
        )

    result = run("--csv", "data.csv")
    assert result.returncode == 0, result.stderr
    assert "Fitting on 72 rows, 18 for validation" in result.stdout
    assert f"Models: gb, {'svr' if regression else 'svc'}, gp" in result.stdout
    assert ("Final estimator:" in result.stdout) == ensemble.startswith("Stacking")
    bundle = joblib.load(tmp_path / "model.joblib")
    assert type(bundle["model"].named_steps["model"]).__name__ == ensemble
    (tmp_path / "new.csv").write_text("a,b,c,d\n2,0,2,0\n-2,0,-2,0\n")
    result = run("--predict", "new.csv")
    assert result.returncode == 0, result.stderr
    lines = result.stdout.split()
    if regression:
        assert lines[0] == "price" and float(lines[1]) > 1 > -1 > float(lines[2])
    elif params.get("voting") == "hard":
        assert lines == ["label", "spam", "ham"]
    else:
        assert lines[0] == "label,probability"
        assert lines[1].startswith("spam,") and lines[2].startswith("ham,")
    result = run()  # the demo: 40 random rows
    assert result.returncode == 0, result.stderr
    assert "Predictions for 2 random rows:" in result.stdout


# --------------------------------------------------------------------------- #
# Training, predictions, estimate and explanations
# --------------------------------------------------------------------------- #


@pytest.fixture
def training():
    pytest.importorskip("sklearn")
    from . import test_classical_training as helpers

    return helpers


@pytest.mark.parametrize(
    "ensemble,params",
    [
        ("VotingClassifier", {}),
        ("VotingClassifier", {"voting": "hard", "weights": [1, 1, 2]}),
        ("StackingClassifier", {}),
        ("VotingRegressor", {"weights": [1, 2, 1]}),
        ("StackingRegressor", {"cv": 3}),
    ],
)
def test_training_an_ensemble(ensemble, params, training, tmp_path):
    import joblib
    import numpy as np

    from brickdnn.training import inspect_csv
    from brickdnn.training.classical_engine import class_probabilities

    regression = ensemble.endswith("Regressor")
    make = training.regression_csv if regression else training.classification_csv
    path = make(tmp_path / "data.csv")
    graph = ensemble_graph(ensemble, params)
    result, rec = training.run_job(graph, inspect_csv(path), tmp_path, path, test_split=0.1)
    assert result.status == "completed", result.message
    combines = ensemble[: -len("Classifier" if not regression else "Regressor")].lower()
    assert any(f"Fitting the {combines} ensemble of 3 models" in m for _, m in rec.statuses)
    details = rec.info["classical"]
    assert details["estimator"] == ensemble and details["ensemble"] == combines
    assert details["kernel"] is None and details["iterations"] is None
    members = details["members"]
    assert [m["name"] for m in members] == ["gb", "svr" if regression else "svc", "gp"]
    (entry,) = rec.history
    for member in members:
        # Every model measured on its own, on the validation rows.
        metrics = member["metrics"]
        assert set(metrics) == {"loss", "accuracy", "rmse", "r2"}
        if regression:
            assert metrics["r2"] > 0.5
        else:
            assert 0.5 <= metrics["accuracy"] <= 1
    if regression:
        assert entry["val_r2"] > 0.8 and rec.test["r2"] is not None
    else:
        assert entry["val_accuracy"] >= 0.6 and entry["val_loss"] > 0
    bundle = joblib.load(tmp_path / "model.joblib")
    pipeline = bundle["model"]
    assert type(pipeline.named_steps["model"]).__name__ == ensemble
    assert rec.info["preprocessing"]["estimator"] == ensemble
    if params.get("voting") == "hard":
        # The share of the weighted votes stands in for the probabilities.
        x = np.random.default_rng(0).normal(size=(20, 4))
        shares = class_probabilities(pipeline, x)
        assert np.allclose(shares.sum(axis=1), 1)
        assert set(np.round(shares.ravel() * 4).astype(int)) <= {0, 1, 2, 3, 4}
        assert (pipeline.classes_[shares.argmax(axis=1)] == pipeline.predict(x)).all()


def test_the_strictest_row_limit_applies(training, tmp_path, monkeypatch):
    from brickdnn.training import check_compatibility, inspect_csv

    monkeypatch.setenv("BRICKDNN_GPC_MAX_ROWS", "40")
    path = training.classification_csv(tmp_path / "data.csv")
    graph = ensemble_graph()
    compat = check_compatibility(graph, inspect_csv(path))
    assert compat.ok and compat.plan.classical["max_rows"] == 40
    assert "Gaussian process classifiers are fitted on at most 40" in compat.warnings[0]
    assert [m["name"] for m in compat.plan.classical["members"]] == ["gb", "svc", "gp"]
    assert any("ensemble fits each of its models" in note for note in compat.plan.notes)
    result, rec = training.run_job(graph, inspect_csv(path), tmp_path, path)
    assert result.status == "completed", result.message
    assert rec.info["classical"]["rows"] == 40 and rec.info["classical"]["max_rows"] == 40
    assert rec.info["classical"]["subsampled_from"] == 72


def test_stacking_needs_rows_for_its_folds(training, tmp_path):
    from brickdnn.training import inspect_csv

    path = training.classification_csv(tmp_path / "data.csv", n=40)
    graph = ensemble_graph("StackingClassifier", {"cv": 20})
    result, _ = training.run_job(graph, inspect_csv(path), tmp_path, path)
    assert result.status == "failed" and "Lower Cross-validation folds" in result.message


def test_ensembles_on_demo_data(training, tmp_path):
    from brickdnn.training import random_info

    graph = ensemble_graph("VotingRegressor")
    result, rec = training.run_job(graph, random_info(), tmp_path)
    assert result.status == "completed", result.message
    # The demo rows of the slowest model (the Gaussian process).
    assert rec.info["train_size"] == 400 and rec.history[0]["val_r2"] > 0.5


@pytest.mark.parametrize("voting", ["soft", "hard"])
def test_predictions_of_an_ensemble(voting, training, tmp_path):
    import joblib
    import numpy as np

    from brickdnn.training import inference, predict

    path = training.classification_csv(tmp_path / "mail.csv")
    graph = ensemble_graph(params={"voting": voting})
    run_dir, plan, preprocessing = training.trained_run(tmp_path, graph, path)
    assert plan["classical"]["members"][0]["type"] == "HistGradientBoostingClassifier"
    rows = [{"a": 2, "b": 0, "c": 0, "d": 0}, [-2, 0, 0, 0], [0.5, 2, 0, 0]]
    prepared = inference.prepare_inputs(plan, preprocessing, rows, top_k=3)
    answer = predict.predict(run_dir, {"inputs": prepared.tensors, "kind": prepared.kind})
    assert answer["probabilities"] is True
    predictions = inference.read_outputs(
        plan, preprocessing, answer["outputs"], 3, 3, probabilities=True
    )
    pipeline = joblib.load(run_dir / "model.joblib")["model"]
    raw = np.array([[2, 0, 0, 0], [-2, 0, 0, 0], [0.5, 2, 0, 0]], dtype=float)
    names = ["eggs", "ham", "spam"]
    assert [p["class"] for p in predictions] == [names[c] for c in pipeline.predict(raw)]
    if voting == "soft":
        expected = pipeline.predict_proba(raw).max(axis=1)
        assert [p["probability"] for p in predictions] == pytest.approx(expected, abs=1e-4)
    else:
        probabilities = [round(p["probability"], 3) for p in predictions]
        assert set(probabilities) <= {0.667, 1.0}, predictions

    # The usage snippet of the run's download.
    snippet = inference.usage_snippet(plan, preprocessing, "m", "model.joblib")
    result = subprocess.run(
        [sys.executable, "-c", snippet], cwd=run_dir, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.replace("'", '"'))[0] in names
    assert ("model.predict(rows)" in snippet) == (voting == "hard")


def test_estimate_of_an_ensemble(training):
    from brickdnn.training import TrainingConfig, check_compatibility
    from brickdnn.training.estimate import CLASSICAL_STARTUP_CPU_SECONDS, estimate_training

    info = training.csv_info(num_samples=2000, **training.NUMBER)

    def estimate(graph):
        plan = check_compatibility(graph, info).plan.to_dict()
        return estimate_training(analyze(graph), info, TrainingConfig(), plan)

    singles = [
        estimate(training.model_graph(model, params=params)).cpu_seconds
        - CLASSICAL_STARTUP_CPU_SECONDS
        for model, params in REGRESSORS.values()
    ]
    voting = estimate(ensemble_graph("VotingRegressor"))
    assert voting.cpu_seconds == pytest.approx(CLASSICAL_STARTUP_CPU_SECONDS + sum(singles))
    assert voting.details["fits"] == 1 and len(voting.details["members"]) == 3
    assert voting.basis.startswith("voting ensemble of 3 models (gradient boosting with up to")
    stacking = estimate(ensemble_graph("StackingRegressor", {"cv": 4}))
    assert stacking.cpu_seconds == pytest.approx(CLASSICAL_STARTUP_CPU_SECONDS + 5 * sum(singles))
    assert stacking.details["fits"] == 5 and "each fitted 5 times" in stacking.basis
    assert stacking.details["max_rows"] == 5_000  # the Gaussian process's row limit


def test_explanations_of_an_ensemble(tmp_path):
    """Permutation importance and Shapley values only need predictions: they work for
    ensembles as for single models, also for hard voting (the shares of the votes)."""
    pytest.importorskip("sklearn")
    import math

    import numpy as np

    from brickdnn.training.classical_engine import _ClassicalRun
    from brickdnn.training.compat import check_compatibility
    from brickdnn.training.datasets import inspect_csv
    from brickdnn.training.job import TrainingJob
    from brickdnn.training.reporting import Reporter

    from .test_explain import ask, run_dir

    rng = np.random.default_rng(3)
    rows = rng.uniform(-2, 2, size=(160, 2))
    lines = ["a,b,y"] + [f"{a:.4f},{b:.4f},{'yes' if a > 0.2 else 'no'}" for a, b in rows]
    csv = tmp_path / "line.csv"
    csv.write_text("\n".join(lines) + "\n")
    info = inspect_csv(csv)
    graph = ensemble_graph(params={"voting": "hard"}, features=2)
    directory = run_dir(tmp_path, graph, info, csv, {"seed": 1})
    job = TrainingJob.from_dict(json.loads((directory / "job.json").read_text()))
    fitted = _ClassicalRun(job, Reporter(), directory / "model.joblib", False)
    fitted._execute()
    plan = check_compatibility(graph, info, job.config).plan.to_dict()
    preprocessing = fitted._preprocessing()

    answer = ask(directory, plan, preprocessing, method="permutation")
    assert answer["features"][0]["name"] == "a" and answer["features"][0]["mean"] > 0.2
    answer = ask(directory, plan, preprocessing, method="tabular", sample={"index": 0})
    total = sum(f["shapley"] for f in answer["features"])
    assert math.isclose(total, answer["value"] - answer["base_value"], abs_tol=2e-3)
    answer = ask(directory, plan, preprocessing, method="boundary")
    assert answer["classes"] == ["no", "yes"]


# --------------------------------------------------------------------------- #
# MCP
# --------------------------------------------------------------------------- #


@pytest.mark.django_db
def test_mcp_lists_examples_and_analyzes_ensembles():
    from django.contrib.auth import get_user_model

    from .test_mcp import call, mcp_client

    client = mcp_client(get_user_model().objects.create_user("ens", password="x-y-z-w-1"))
    result, _ = call(client, "list_layer_types")
    layers = {layer["type"]: layer for layer in result["structuredContent"]["layers"]}
    for ensemble in ENSEMBLES:
        assert layers[ensemble]["output_kind"] == "estimator"
    assert layers["SVC"]["output_kind"] == "estimator"
    result, _ = call(client, "list_examples")
    ids = [example["id"] for example in result["structuredContent"]["examples"]]
    assert {"voting_ensemble_classifier", "stacking_ensemble_regression"} <= set(ids)
    result, _ = call(client, "list_examples", id="voting_ensemble_classifier")
    graph = result["structuredContent"]["graph"]
    result, _ = call(client, "analyze_network", graph=graph)
    analysis = result["structuredContent"]
    assert analysis["valid"] is True
    assert [m["type"] for m in analysis["classical"]["members"]] == [
        "HistGradientBoostingClassifier",
        "SVC",
        "GaussianProcessClassifier",
    ]
    result, text = call(client, "generate_pytorch_code", graph=graph)
    assert "model = VotingClassifier(" in text and 'voting="soft"' in text


def test_the_examples():
    voting = analyze(EXAMPLES["voting_ensemble_classifier"]["graph"])
    assert voting.valid and voting.model.type == "VotingClassifier"
    assert voting.model.layer.p["voting"] == "soft"
    stacking = analyze(EXAMPLES["stacking_ensemble_regression"]["graph"])
    assert stacking.valid and stacking.model.type == "StackingRegressor"
    gp = stacking.nodes["gp"]
    assert gp.kernel.code() == "RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)"
