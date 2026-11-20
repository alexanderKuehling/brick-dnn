"""XGBoost, LightGBM and CatBoost as classical model bricks: the analysis, the generated
script, training (also in ensembles, as the head of a hybrid network and in a search),
predictions, explanations, and a server without the libraries."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys

import pytest

pytest.importorskip("sklearn")

import joblib  # noqa: E402
import numpy as np  # noqa: E402

from brickdnn import analyze, generate_code, layer_specs  # noqa: E402
from brickdnn.training import (  # noqa: E402
    TrainingConfig,
    check_compatibility,
    inference,
    inspect_csv,
    predict,
    random_info,
    torch_installed,
)
from brickdnn.training import predict_file as file_predictions  # noqa: E402
from brickdnn.training.classical_engine import _ClassicalRun, train  # noqa: E402
from brickdnn.training.estimate import estimate_training  # noqa: E402
from brickdnn.training.explain import explain  # noqa: E402
from brickdnn.training.job import TrainingJob  # noqa: E402
from brickdnn.training.reporting import Reporter  # noqa: E402
from brickdnn.training.tuning import TuningConfig  # noqa: E402

from .test_classical_training import (  # noqa: E402
    Recorder,
    classification_csv,
    regression_csv,
    trained_run,
)

LIBRARY = {
    "XGBRegressor": "xgboost",
    "XGBClassifier": "xgboost",
    "LGBMRegressor": "lightgbm",
    "LGBMClassifier": "lightgbm",
    "CatBoostRegressor": "catboost",
    "CatBoostClassifier": "catboost",
}
MODELS = list(LIBRARY)
REGRESSORS = [m for m in MODELS if m.endswith("Regressor")]
CLASSIFIERS = [m for m in MODELS if m.endswith("Classifier")]
# Few trees: the test files are small.
QUICK = {
    "xgboost": {"n_estimators": 40, "early_stopping_rounds": 10},
    "lightgbm": {"n_estimators": 40, "min_child_samples": 5, "early_stopping_rounds": 10},
    "catboost": {"iterations": 40, "early_stopping_rounds": 10},
}
TREES = {"xgboost": "n_estimators", "lightgbm": "n_estimators", "catboost": "iterations"}


def installed(model: str) -> bool:
    return importlib.util.find_spec(LIBRARY[model]) is not None


def each(models: list[str]) -> list:
    """The models as parameters, skipped where their library is not installed."""
    return [
        pytest.param(
            m, marks=pytest.mark.skipif(not installed(m), reason=f"{LIBRARY[m]} not installed")
        )
        for m in models
    ]


needs_all = pytest.mark.skipif(
    not all(installed(m) for m in MODELS), reason="XGBoost, LightGBM or CatBoost not installed"
)
needs_torch = pytest.mark.skipif(not torch_installed(), reason="PyTorch is not installed")


def quick(model: str, **params) -> dict:
    if model not in LIBRARY:  # scikit-learn's gradient boosting
        return {"max_iter": 40, "min_samples_leaf": 5, **params}
    return {**QUICK[LIBRARY[model]], **params}


def model_graph(model, params=None, features=4):
    return {
        "version": 1,
        "nodes": [
            {"id": "in", "type": "Input", "params": {"shape": [features]}},
            {"id": "m", "type": model, "params": quick(model) if params is None else params},
        ],
        "edges": [{"source": "in", "target": "m"}],
    }


def ensemble_graph(ensemble, models, params=None):
    nodes = [{"id": "in", "type": "Input", "params": {"shape": [4]}}]
    edges = []
    for i, model in enumerate(models):
        nodes.append({"id": f"m{i}", "type": model, "params": quick(model)})
        edges += [{"source": "in", "target": f"m{i}"}, {"source": f"m{i}", "target": "e"}]
    nodes.append({"id": "e", "type": ensemble, "params": params or {}})
    return {"version": 1, "nodes": nodes, "edges": edges}


def data_csv(model, path, **kwargs):
    make = classification_csv if model.endswith("Classifier") else regression_csv
    return make(path, **kwargs)


def header(model):
    """The feature columns of data_csv()."""
    return "a,b,c,d" if model.endswith("Classifier") else "f0,f1,f2,f3"


def run_job(graph, path, tuning=None, **config):
    job = TrainingJob(
        graph=graph,
        dataset=inspect_csv(path),
        config=TrainingConfig.from_dict({"seed": 1, **config}),
        path=str(path),
        tuning=tuning,
    )
    recorder = Recorder()
    return train(job, recorder, path.parent / "model.joblib"), recorder


def run_script(code, cwd, *args):
    (cwd / "model.py").write_text(code)
    return subprocess.run(
        [sys.executable, "model.py", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=300,
        # One OpenMP thread: the threads of LightGBM and XGBoost crawl on busy machines.
        env={**os.environ, "OMP_NUM_THREADS": "1"},
    )


# --------------------------------------------------------------------------- #
# The bricks and the analysis (no library needed)
# --------------------------------------------------------------------------- #


def test_the_bricks_in_the_catalogue():
    specs = {spec["type"]: spec for spec in layer_specs()}
    for model in MODELS:
        spec = specs[model]
        assert spec["category"] == "Classical models" and spec["family"] == "classical"
        assert spec["output_kind"] == "estimator"
        assert [slot["kind"] for slot in spec["input_slots"]] == ["tensor"]
        names = [p["name"] for p in spec["params"]]
        assert names[0] == TREES[LIBRARY[model]] and "learning_rate" in names
        assert "early_stopping_rounds" in names and names[-1] == "random_state"
        assert len(names) <= 12  # a short, practical list
    weights = {
        "XGBClassifier": "scale_pos_weight",
        "LGBMClassifier": "class_weight",
        "CatBoostClassifier": "auto_class_weights",
    }
    for model, name in weights.items():
        assert name in [p["name"] for p in specs[model]["params"]]


@pytest.mark.parametrize(
    "model,params,message",
    [
        ("XGBRegressor", {"learning_rate": 0}, "Learning rate must be greater than 0."),
        ("LGBMClassifier", {"subsample": 0}, "Row sample must be greater than 0."),
        ("XGBClassifier", {"colsample_bytree": 1.5}, "Column sample"),
        ("XGBClassifier", {"scale_pos_weight": 0}, "Positive class weight must be greater"),
        ("CatBoostRegressor", {"depth": 20}, "Depth"),
        ("CatBoostClassifier", {"auto_class_weights": "balanced"}, "Class weights must be"),
    ],
)
def test_the_analysis_checks_the_parameters(model, params, message):
    errors = [e.message for e in analyze(model_graph(model, params)).errors]
    assert errors and message in errors[0]


def test_the_analysis_works_and_explains_connections():
    for model in MODELS:
        analysis = analyze(model_graph(model, {}))
        assert analysis.valid and analysis.family == "classical"
        assert analysis.total_params is None  # trees: known after the fit
    g = model_graph("LGBMRegressor", {})
    g["nodes"].append({"id": "k", "type": "RBF", "params": {}})
    g["edges"].append({"source": "k", "target": "m"})
    assert "builds decision trees and takes no kernel" in analyze(g).errors[0].message
    # CatBoost predicts the classes as a column: hard voting cannot count them.
    hard = ensemble_graph("VotingClassifier", CLASSIFIERS, {"voting": "hard"})
    assert "Choose soft voting." in analyze(hard).errors[0].message
    assert analyze(ensemble_graph("VotingClassifier", CLASSIFIERS)).valid
    two = ensemble_graph("VotingClassifier", CLASSIFIERS[:2], {"voting": "hard"})
    assert analyze(two).valid


def test_the_code_of_the_bricks():
    code = generate_code(model_graph("XGBClassifier", {"scale_pos_weight": 19}))
    assert "from xgboost import XGBClassifier" in code
    assert (
        "    model = XGBClassifier(\n"
        "        n_estimators=500,\n"
        "        learning_rate=0.05,\n"
        "        max_depth=6,\n"
        "        min_child_weight=1.0,\n"
        "        subsample=1.0,\n"
        "        colsample_bytree=1.0,\n"
        "        reg_alpha=0.0,\n"
        "        reg_lambda=1.0,\n"
        "        early_stopping_rounds=50,\n"
        "        scale_pos_weight=19.0,\n"
        "        random_state=0,\n"
        "    )\n"
    ) in code
    # XGBoost learns class indices: the script numbers the labels and names them again.
    assert "class_names, target = np.unique(target, return_inverse=True)" in code
    assert "model__eval_set=[(watch, target[val_rows])]" in code
    code = generate_code(model_graph("LGBMRegressor", {"subsample": 0.8, "max_depth": 7}))
    assert "subsample=0.8,\n        subsample_freq=1," in code and "max_depth=7" in code
    assert "model__eval_X=watch" in code and "verbose=-1" in code
    assert "max_depth=-1" in generate_code(model_graph("LGBMRegressor", {}))
    code = generate_code(
        model_graph("CatBoostClassifier", {"auto_class_weights": "Balanced", "random_state": 4})
    )
    assert 'auto_class_weights="Balanced"' in code and "random_seed=4" in code
    assert "allow_writing_files=False" in code and "np.ravel(model.predict(" in code
    # Without early stopping: no validation rows for the fit, the labels stay as they are.
    code = generate_code(model_graph("LGBMClassifier", {"early_stopping_rounds": 0}))
    assert "early_stopping_rounds" not in code and "watch" not in code
    assert "np.unique(target" not in code
    # An ensemble fits its models without validation rows: no early stopping.
    code = generate_code(ensemble_graph("StackingClassifier", CLASSIFIERS))
    assert "early_stopping_rounds" not in code and "watch" not in code
    assert "from catboost import CatBoostClassifier" in code


@pytest.mark.parametrize("model", each(MODELS))
def test_the_check_and_the_estimate(model):
    result = check_compatibility(model_graph(model, {}), random_info())
    assert result.ok, result.errors
    assert any("XGBoost, LightGBM and CatBoost stop adding trees" in n for n in result.plan.notes)
    estimate = estimate_training(
        analyze(model_graph(model, {})), random_info(), TrainingConfig(), result.plan.to_dict()
    )
    assert estimate is not None and estimate.cpu_seconds > 0
    assert estimate.details["trees"] >= 500


# --------------------------------------------------------------------------- #
# Training, predictions, explanations
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model", each(MODELS))
def test_training_on_a_csv_file(model, tmp_path):
    path = data_csv(model, tmp_path / "data.csv", n=200)
    result, rec = run_job(model_graph(model), path, test_split=0.2)
    assert result.status == "completed", result.message
    details = rec.info["classical"]
    assert details["estimator"] == model and details["label"].split()[0] in (
        "XGBoost",
        "LightGBM",
        "CatBoost",
    )
    assert 1 <= details["iterations"] <= details["max_iterations"] == 40
    assert any(line.startswith("Boosting iterations:") for line in rec.logs)
    assert any(f"Fitting the {details['label'].split()[0]} model" in s[1] for s in rec.statuses)
    bundle = joblib.load(tmp_path / "model.joblib")
    estimator = bundle["model"].named_steps["model"]
    # Small data: one thread (more stall on busy CPUs), also for the predictions.
    threads = estimator.get_params()["thread_count" if "CatBoost" in model else "n_jobs"]
    assert threads == 1
    if model.endswith("Classifier"):
        assert bundle["class_names"] == ["eggs", "ham", "spam"]
        assert result.best["val_accuracy"] > 0.6
    else:
        assert rec.history[0]["val_r2"] > 0.5


@pytest.mark.parametrize("model", each(["XGBRegressor", "LGBMClassifier", "CatBoostRegressor"]))
def test_without_validation_rows_all_trees_are_fitted(model, tmp_path):
    path = data_csv(model, tmp_path / "data.csv", n=120)
    result, rec = run_job(model_graph(model), path, val_split=0, test_split=0.2)
    assert result.status == "completed", result.message
    assert any(
        "Early stopping needs validation rows (set a validation split above 0)" in s
        for s in rec.logs
    )
    details = rec.info["classical"]
    assert details["iterations"] == details["max_iterations"] == 40


@pytest.mark.skipif(not importlib.util.find_spec("xgboost"), reason="xgboost not installed")
def test_xgboost_needs_rows_of_every_class(tmp_path):
    """XGBoost learns class indices without gaps; the other libraries take any classes."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 4))
    labels = ["a" if v < 0 else "c" for v in x[:, 0]]
    labels[0] = "b"  # one row: in the validation rows for some seeds
    lines = ["p,q,r,s,label"] + [
        ",".join([*(f"{v:.4f}" for v in row), lab]) for row, lab in zip(x, labels, strict=True)
    ]
    path = tmp_path / "rare.csv"
    path.write_text("\n".join(lines) + "\n")
    messages = set()
    for seed in range(12):
        result, _ = run_job(model_graph("XGBClassifier"), path, seed=seed, val_split=0.3)
        messages.add(result.status if result.status == "completed" else result.message)
    assert "completed" in messages
    assert (
        "XGBoost needs training rows of every class up to the last one, but the class 'b' has "
        "none. Use more rows, or LightGBM or CatBoost, which take any classes."
    ) in messages


@pytest.mark.parametrize("model", each(MODELS))
def test_single_and_file_predictions(model, tmp_path):
    path = data_csv(model, tmp_path / "data.csv", n=200)
    run_dir, plan, preprocessing = trained_run(tmp_path, model_graph(model), path)
    rows = [[2, 0, 0, 0], [-2, 0, 0, 0], [0.5, 2, 0, 0]]
    prepared = inference.prepare_inputs(plan, preprocessing, rows, top_k=3)
    answer = predict.predict(run_dir, {"inputs": prepared.tensors, "kind": prepared.kind})
    pipeline = joblib.load(run_dir / "model.joblib")["model"]
    raw = np.array(rows, dtype=float)
    classification = model.endswith("Classifier")
    if classification:
        assert answer["probabilities"] is True
        predictions = inference.read_outputs(
            plan, preprocessing, answer["outputs"], 3, 3, probabilities=True
        )
        expected = pipeline.predict_proba(raw)
        for prediction, probabilities in zip(predictions, expected, strict=True):
            assert prediction["probability"] == pytest.approx(probabilities.max(), abs=1e-4)
        assert [p["class"] for p in predictions[:2]] == ["spam", "ham"]
    else:
        predictions = inference.read_outputs(plan, preprocessing, answer["outputs"], 3)
        expected = pipeline.predict(raw)
        assert [p["value"] for p in predictions] == pytest.approx(list(expected), abs=1e-4)

    # A file: every row, the columns by name.
    source = tmp_path / "new.csv"
    names = header(model).split(",")
    source.write_text(
        f"id,{names[3]},{names[2]},{names[1]},{names[0]}\n"
        + "".join(f"r{i},0,0,{r[1]},{r[0]}\n" for i, r in enumerate(rows))
    )
    summary = file_predictions.predict_file(
        {
            "input": str(source),
            "output": str(tmp_path / "result.csv"),
            "task": "classification" if classification else "regression",
            "class_names": ["eggs", "ham", "spam"] if classification else None,
            "models": [{"run_dir": str(run_dir), "name": model, "preprocessing": preprocessing}],
            "options": {"columns": ["id"], "output": "label"},
        }
    )
    assert summary["rows"] == summary["predicted"] == 3, summary
    lines = (tmp_path / "result.csv").read_text().splitlines()
    assert lines[0].split(",")[0] == "id" and len(lines) == 4
    if classification:
        assert [line.split(",")[1] for line in lines[1:3]] == ["spam", "ham"]
    else:
        assert [float(line.split(",")[1]) for line in lines[1:]] == pytest.approx(
            list(expected), rel=1e-4
        )


@pytest.mark.parametrize("model", each(["XGBRegressor", "LGBMRegressor", "CatBoostClassifier"]))
def test_permutation_importance(model, tmp_path):
    """The explanations treat the models like the other classical models."""
    from .test_explain import run_dir

    path = data_csv(model, tmp_path / "data.csv", n=200)
    info = inspect_csv(path)
    graph = model_graph(model)
    directory = run_dir(tmp_path, graph, info, path, {"seed": 1})
    job = TrainingJob.from_dict(json.loads((directory / "job.json").read_text()))
    fitted = _ClassicalRun(job, Reporter(), directory / "model.joblib", False)
    fitted._execute()
    plan = check_compatibility(graph, info, job.config).plan.to_dict()
    answer = explain(
        directory,
        {"plan": plan, "preprocessing": fitted._preprocessing(), "method": "permutation"},
    )
    # The target depends most on the first column (regression_csv, classification_csv).
    assert answer["features"][0]["name"] in ("f0", "a")
    assert answer["features"][0]["mean"] > 0.05


# --------------------------------------------------------------------------- #
# The generated script: standalone, and the same predictions as the server
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model", each(MODELS))
def test_the_script_fits_and_predicts_on_its_own(model, tmp_path):
    data_csv(model, tmp_path / "data.csv", n=200)
    code = generate_code(model_graph(model))
    result = run_script(code, tmp_path)
    assert result.returncode == 0, result.stderr
    assert "Predictions for 2 random rows:" in result.stdout
    result = run_script(code, tmp_path, "--csv", "data.csv")
    assert result.returncode == 0, result.stderr
    trees = next(line for line in result.stdout.splitlines() if line.startswith("Trees: "))
    assert trees.endswith("of at most 40")
    if model.endswith("Classifier"):
        accuracy = next(line for line in result.stdout.splitlines() if "accuracy" in line)
        assert float(accuracy.split()[-1].rstrip("%")) > 60
        bundle = joblib.load(tmp_path / "model.joblib")
        assert bundle["class_names"] == ["eggs", "ham", "spam"]
    assert not (tmp_path / "catboost_info").exists()
    (tmp_path / "new.csv").write_text(f"{header(model)}\n2,0,0,0\n-2,0,0,0\n")
    result = run_script(code, tmp_path, "--predict", "new.csv")
    assert result.returncode == 0, result.stderr
    lines = result.stdout.split()
    if model.endswith("Classifier"):
        assert lines[1].startswith("spam,") and lines[2].startswith("ham,")
    else:
        assert float(lines[1]) > float(lines[2])


@pytest.mark.parametrize("model", each(MODELS))
def test_the_script_predicts_like_the_server(model, tmp_path):
    """The downloaded model.joblib in the script gives the server's predictions."""
    path = data_csv(model, tmp_path / "data.csv", n=200)
    graph = model_graph(model)
    run_dir, plan, preprocessing = trained_run(tmp_path, graph, path)
    rows = [[2, 0, 0, 0], [-2, 0, 0, 0], [0.5, 2, 0, 0], [0.3, -1, 1, 0]]
    prepared = inference.prepare_inputs(plan, preprocessing, rows, top_k=3)
    answer = predict.predict(run_dir, {"inputs": prepared.tensors, "kind": prepared.kind})
    code = generate_code(graph, {"include_training": False})
    lines = "".join(",".join(map(str, r)) + "\n" for r in rows)
    (run_dir / "new.csv").write_text(f"{header(model)}\n{lines}")
    result = run_script(code, run_dir, "--predict", "new.csv", "--save", "model.joblib")
    assert result.returncode == 0, result.stderr
    printed = result.stdout.split()[1:]
    if model.endswith("Classifier"):
        predictions = inference.read_outputs(
            plan, preprocessing, answer["outputs"], 4, 3, probabilities=True
        )
        assert [line.split(",")[0] for line in printed] == [p["class"] for p in predictions]
        assert [float(line.split(",")[1]) for line in printed] == pytest.approx(
            [p["probability"] for p in predictions], abs=1e-4
        )
    else:
        predictions = inference.read_outputs(plan, preprocessing, answer["outputs"], 4)
        assert [float(v) for v in printed] == pytest.approx(
            [p["value"] for p in predictions], rel=1e-5
        )


# --------------------------------------------------------------------------- #
# Ensembles, hybrid networks, searches
# --------------------------------------------------------------------------- #


@needs_all
@pytest.mark.parametrize(
    "ensemble,models",
    [
        ("VotingClassifier", CLASSIFIERS),
        ("StackingClassifier", CLASSIFIERS),
        ("StackingRegressor", REGRESSORS),
        ("VotingRegressor", ["XGBRegressor", "HistGradientBoostingRegressor"]),
    ],
)
def test_members_of_ensembles(ensemble, models, tmp_path):
    graph = ensemble_graph(ensemble, models, {"cv": 3} if ensemble.startswith("Stacking") else None)
    path = data_csv(models[0], tmp_path / "data.csv", n=200)
    result, rec = run_job(graph, path)
    assert result.status == "completed", result.message
    members = rec.info["classical"]["members"]
    assert [m["type"] for m in members] == models
    assert all(m["metrics"]["loss"] is not None for m in members)
    # Their models fit all trees (an ensemble hands them no validation rows) with one thread.
    ensemble_model = joblib.load(tmp_path / "model.joblib")["model"].named_steps["model"]
    for fitted in ensemble_model.named_estimators_.values():
        params = fitted.get_params()
        assert params.get("early_stopping_rounds") is None
        assert (
            params.get("n_jobs", params.get("thread_count")) == 1 or "Hist" in type(fitted).__name__
        )
    data_csv(models[0], tmp_path / "data.csv", n=120)
    script = run_script(generate_code(graph), tmp_path, "--csv", "data.csv")
    assert script.returncode == 0, script.stderr


@needs_all
@needs_torch
@pytest.mark.parametrize("model", ["XGBClassifier", "LGBMRegressor", "CatBoostClassifier"])
def test_the_classical_head_of_a_hybrid_network(model, tmp_path):
    from .test_hybrid import hybrid_graph
    from .test_hybrid import run_job as run_hybrid

    graph = hybrid_graph(model, quick(model))
    path = data_csv(model, tmp_path / "data.csv", n=160)
    result, rec = run_hybrid(graph, inspect_csv(path), path, weights=tmp_path / "model.joblib")
    assert result.status == "completed", result.message
    classical = rec.info["classical"]
    assert classical["estimator"] == model
    assert 1 <= classical["iterations"] <= classical["max_iterations"] == 40
    # The script trains both phases on its own (XGBoost: on class indices).
    script = run_script(generate_code(graph), tmp_path, "--csv", "data.csv", "--epochs", "2")
    assert script.returncode == 0, script.stderr
    assert "Trees: " in script.stdout
    (tmp_path / "new.csv").write_text(f"{header(model)}\n2,0,0,0\n-2,0,0,0\n")
    script = run_script(generate_code(graph), tmp_path, "--predict", "new.csv")
    assert script.returncode == 0, script.stderr
    assert len(script.stdout.split()) == 3


@pytest.mark.parametrize("model", each(["LGBMRegressor", "XGBClassifier", "CatBoostRegressor"]))
def test_a_search_with_a_tiny_space(model, tmp_path):
    path = data_csv(model, tmp_path / "data.csv", n=160)
    tuning = TuningConfig.from_dict(
        {"method": "grid", "space": {"learning_rate": [0.05, 0.3]}}, model
    )
    result, rec = run_job(model_graph(model), path, tuning=tuning)
    assert result.status == "completed", result.message
    assert rec.info["trials_total"] == 2
    assert rec.info["classical"]["iterations"] <= 40
    # Cross-validation: no early stopping, all trees.
    tuning = TuningConfig.from_dict(
        {"method": "grid", "space": {"learning_rate": [0.1]}, "cv_folds": 2}, model
    )
    result, rec = run_job(model_graph(model), path, tuning=tuning)
    assert result.status == "completed", result.message
    assert any("fits all its trees in this search" in line for line in rec.logs)
    assert rec.info["classical"]["iterations"] == 40


def test_the_search_spaces_of_the_bricks():
    from brickdnn.training.classical_tuning import capabilities, parse_space

    found = capabilities()
    for model in MODELS:
        assert len(found[model]["default"]) == 3
        assert parse_space(model, None)
    with pytest.raises(Exception, match="'max_leaf_nodes' cannot be tuned for XGBoost"):
        parse_space("XGBClassifier", {"max_leaf_nodes": [31]})
    space = parse_space("LGBMClassifier", {"subsample": {"min": 0.5, "max": 1}})
    assert space["subsample"].min == 0.5 and not space["subsample"].log


# --------------------------------------------------------------------------- #
# A server without the libraries
# --------------------------------------------------------------------------- #


def test_without_the_library_the_analysis_works_and_training_explains(tmp_path, monkeypatch):
    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name, *a: None if name == "xgboost" else real(name, *a)
    )
    graph = model_graph("XGBClassifier")
    assert analyze(graph).valid and "XGBClassifier(" in generate_code(graph)
    result = check_compatibility(graph, random_info())
    assert result.errors == [
        "The library XGBoost is not installed on this server, so XGBoost classification cannot "
        "be trained here. Install it with: pip install xgboost"
    ]
    ensemble = ensemble_graph("VotingClassifier", ["LGBMClassifier", "XGBClassifier"])
    assert "XGBoost is not installed" in check_compatibility(ensemble, random_info()).errors[0]
    path = classification_csv(tmp_path / "data.csv")
    result, _ = run_job(graph, path)
    assert result.status == "failed" and result.message.startswith(
        "The library XGBoost is not installed"
    )


@pytest.mark.parametrize("model", each(["XGBRegressor", "CatBoostClassifier"]))
def test_a_missing_library_when_the_model_is_fitted_or_loaded(model, tmp_path, monkeypatch):
    path = data_csv(model, tmp_path / "data.csv", n=120)
    run_dir, plan, preprocessing = trained_run(tmp_path, model_graph(model), path)
    library = LIBRARY[model]
    name = {"xgboost": "XGBoost", "catboost": "CatBoost"}[library]
    for module in [m for m in sys.modules if m == library or m.startswith(library + ".")]:
        monkeypatch.setitem(sys.modules, module, None)
    prepared = inference.prepare_inputs(plan, preprocessing, [[0, 0, 0, 0]])
    with pytest.raises(predict.PredictionError, match=f"{name} is not installed on this server"):
        predict.predict(run_dir, {"inputs": prepared.tensors, "kind": prepared.kind})
    # Installed as far as the check knows, but the import fails: the fit says why.
    (tmp_path / "again").mkdir()
    result, _ = run_job(model_graph(model), data_csv(model, tmp_path / "again" / "d.csv"))
    assert result.status == "failed" and result.message.startswith(
        f"The library {name} is not installed"
    )
