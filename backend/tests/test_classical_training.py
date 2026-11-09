"""Training classical models with scikit-learn: compatibility, the engine, the runner (also
stopping it) and predictions. PyTorch is not needed. The REST API and the MCP tools are
tested in test_classical_api.py."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("sklearn")

import joblib  # noqa: E402
import numpy as np  # noqa: E402

from brickdnn import EXAMPLES  # noqa: E402
from brickdnn.training import (  # noqa: E402
    TrainingConfig,
    builtin_info,
    check_compatibility,
    classical_engine,
    inference,
    inspect_csv,
    predict,
    random_info,
    runner,
)
from brickdnn.training import state as st  # noqa: E402
from brickdnn.training.classical_engine import build_pipeline, train  # noqa: E402
from brickdnn.training.datasets import SPLIT_FILE, DatasetInfo  # noqa: E402
from brickdnn.training.job import TrainingJob  # noqa: E402
from brickdnn.training.reporting import Reporter  # noqa: E402

BACKEND = Path(__file__).resolve().parent.parent


def model_graph(model, features=4, params=None, kernel=None):
    nodes = [
        {"id": "in", "type": "Input", "params": {"shape": [features]}},
        {"id": "m", "type": model, "params": params or {}},
    ]
    edges = [{"source": "in", "target": "m"}]
    if kernel is not None:
        nodes.append({"id": "k", "type": kernel, "params": {}})
        edges.append({"source": "k", "target": "m"})
    return {"version": 1, "nodes": nodes, "edges": edges}


def regression_csv(path, n=80, features=4, seed=0, header=True):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, features))
    y = 3 * x[:, 0] - 2 * np.sin(x[:, 1]) + 100 + 0.05 * rng.normal(size=n)
    lines = [",".join([*(f"f{i}" for i in range(features)), "price"])] if header else []
    lines += [",".join(f"{v:.5f}" for v in [*row, t]) for row, t in zip(x, y, strict=True)]
    path.write_text("\n".join(lines) + "\n")
    return path


def classification_csv(path, n=90, labels=("ham", "spam", "eggs"), seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 4))
    index = (x[:, 0] > 0).astype(int) + (x[:, 1] > 0.6).astype(int)
    lines = ["a,b,c,d,label"]
    lines += [
        ",".join([*(f"{v:.5f}" for v in row), str(labels[i])])
        for row, i in zip(x, index, strict=True)
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


class Recorder(Reporter):
    def __init__(self, stop_when_fitting=False):
        self.statuses, self.logs, self.history = [], [], []
        self.info, self.test, self.kept, self.finished = {}, None, None, None
        self.stop_when_fitting = stop_when_fitting

    def status(self, status, message):
        self.statuses.append((status, message))

    def log(self, line):
        self.logs.append(line)

    def setup(self, info):
        self.info.update(info)

    def epoch_end(self, entry, best):
        self.history.append(entry)
        self.best = best

    def weights_kept(self, result):
        self.kept = result

    def test_end(self, result):
        self.test = result

    def finish(self, status, message, weights):
        assert self.finished is None
        self.finished = (status, message, weights)

    def should_stop(self):
        return self.stop_when_fitting and any(m.startswith("Fitting") for _, m in self.statuses)


def run_job(graph, info, tmp_path, path=None, reporter=None, tuning=None, **config):
    job = TrainingJob(
        graph=graph,
        dataset=info,
        config=TrainingConfig.from_dict({"seed": 1, **config}),
        path=str(path) if path else None,
        tuning=tuning,
    )
    reporter = reporter or Recorder()
    tmp_path.mkdir(exist_ok=True)
    return train(job, reporter, tmp_path / "model.joblib"), reporter


# --------------------------------------------------------------------------- #
# Compatibility
# --------------------------------------------------------------------------- #


def csv_info(**fields):
    base = {
        "kind": "csv",
        "name": "data",
        "num_samples": 100,
        "num_features": 4,
        "columns": ["a", "b", "c", "d", "y"],
        "has_header": True,
        "delimiter": ",",
    }
    return DatasetInfo(**{**base, **fields})


TEXT = {"target_kind": "text", "class_names": ["no", "yes"]}
NUMBER = {"target_kind": "number", "target_range": [0.5, 9.5]}
INTEGER = {"target_kind": "integer", "target_range": [1, 7], "class_names": ["1", "4", "7"]}


@pytest.mark.parametrize(
    "model,info,message",
    [
        ("SVR", builtin_info("mnist"), "classical model: it trains on tabular data"),
        ("SVC", DatasetInfo(kind="images", name="pets"), "not on images like pets"),
        (
            "SVR",
            csv_info(num_features=5, columns=list("abcdey"), **NUMBER),
            "Set the Input shape to 5",
        ),
        ("SVR", csv_info(**TEXT), "use GaussianProcessClassifier or SVC instead"),
        ("GaussianProcessRegressor", csv_info(**TEXT), "classification task"),
        ("SVC", csv_info(**NUMBER), "use GaussianProcessRegressor or SVR instead"),
        ("GaussianProcessClassifier", csv_info(**NUMBER), "regression task"),
        ("SVC", csv_info(target_kind="text", class_names=["only"]), "only one class"),
        ("SVC", csv_info(target_kind="integer", class_names=None), "more than 1,000"),
        (
            "SVC",
            csv_info(target_kind="integer", class_names=None, predefined_split=[80, 20, 0]),
            "prepared for regression",
        ),
    ],
)
def test_incompatible_data(model, info, message):
    result = check_compatibility(model_graph(model), info)
    assert not result.ok and result.plan is None
    assert message in result.errors[0], result.errors


def test_plans_of_classical_models():
    plan = check_compatibility(model_graph("SVR"), csv_info(**NUMBER)).plan
    assert plan.task == "regression" and plan.targets == [{"kind": "value", "shape": [1]}]
    assert plan.classical["estimator"] == "SVR" and plan.classical["scale"] == "svm"
    assert plan.classical["max_rows"] == 50_000
    assert any("fitted once, without epochs" in note for note in plan.notes)

    gp = model_graph("GaussianProcessClassifier")
    plan = check_compatibility(gp, csv_info(**TEXT)).plan
    assert plan.task == "classification" and plan.num_classes == 2
    assert plan.class_names == ["no", "yes"]
    assert plan.classical["class_targets"] == "labels"
    assert plan.classical["kernel"].startswith("ConstantKernel(constant_value=1.0) * RBF(")
    assert plan.classical["max_rows"] == 5_000

    plan = check_compatibility(model_graph("SVC"), csv_info(**INTEGER)).plan
    assert plan.class_names == ["1", "4", "7"] and plan.classical["class_targets"] == "values"
    prepared = csv_info(**INTEGER, predefined_split=[70, 20, 10])
    result = check_compatibility(model_graph("SVC"), prepared)
    assert result.plan.classical["class_targets"] == "indices"
    assert "Uses the split of the data preparation" in result.warnings[0]

    result = check_compatibility(model_graph("SVR"), csv_info(**INTEGER))
    assert result.ok and "predicts them as numbers" in result.warnings[0]


def test_demo_data_plans():
    plan = check_compatibility(model_graph("SVC"), random_info()).plan
    assert plan.task == "demo" and plan.num_classes == 3
    assert plan.class_names == ["class 0", "class 1", "class 2"]
    plan = check_compatibility(model_graph("GaussianProcessRegressor"), random_info()).plan
    assert plan.targets == [{"kind": "value", "shape": [1]}]
    assert plan.inputs == [{"shape": [4], "dtype": "float", "vocab": None}]


def test_row_limits(monkeypatch):
    monkeypatch.setenv("BRICKDNN_GP_MAX_ROWS", "50")
    graph = model_graph("GaussianProcessRegressor")
    result = check_compatibility(graph, csv_info(**NUMBER))  # 80 training rows
    assert result.ok
    assert "at most 50 training rows on this server (BRICKDNN_GP_MAX_ROWS" in result.warnings[0]
    assert "50 random rows of the 80 training rows" in result.warnings[0]
    config = TrainingConfig.from_dict({"max_samples": 40})
    assert not check_compatibility(graph, csv_info(**NUMBER), config).warnings
    config = TrainingConfig.from_dict({"max_samples": 60})
    result = check_compatibility(graph, csv_info(**NUMBER), config)
    assert not result.ok and "Max. samples is 60" in result.errors[0]
    monkeypatch.setenv("BRICKDNN_GP_MAX_ROWS", "not a number")
    assert check_compatibility(graph, csv_info(**NUMBER)).plan.classical["max_rows"] == 5_000


# --------------------------------------------------------------------------- #
# The engine
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model", ["GaussianProcessRegressor", "SVR"])
def test_regression_on_csv(model, tmp_path):
    path = regression_csv(tmp_path / "houses.csv")
    result, rec = run_job(model_graph(model), inspect_csv(path), tmp_path, path, test_split=0.1)
    assert result.status == "completed", result.message
    assert rec.finished[2] is True and result.weights_epoch == 1
    assert result.message.startswith("Finished: validation RMSE ")
    assert "; test RMSE " in result.message and "Fitted on 56 rows in " in result.message
    assert rec.info["family"] == "classical" and rec.info["device"] == "cpu"
    assert (rec.info["train_size"], rec.info["val_size"], rec.info["test_size"]) == (56, 16, 8)
    assert rec.info["epochs"] == 1 and rec.info["steps_per_epoch"] == 1
    (entry,) = rec.history
    assert set(entry) == {
        "epoch",
        "train_loss",
        "val_loss",
        "train_accuracy",
        "val_accuracy",
        "seconds",
        "val_rmse",
        "val_r2",
    }
    assert entry["epoch"] == 1 and entry["val_accuracy"] is None
    assert entry["val_r2"] > 0.5 and entry["val_rmse"] < 2  # the target is about 100 ± 3
    # The loss is the mean squared error of the standardised target, like in PyTorch runs.
    # Both losses divide by the variance of the training targets.
    trained = rec.info["classical"]["train"]
    variance = entry["val_rmse"] ** 2 / entry["val_loss"]
    assert trained["rmse"] ** 2 / trained["loss"] == pytest.approx(variance)
    targets = np.loadtxt(path, delimiter=",", skiprows=1)[:, -1]
    assert variance == pytest.approx(np.var(targets), rel=0.3)
    assert rec.kept == {"weights_epoch": 1, "stopped_early": False}
    assert set(rec.test) == {"loss", "accuracy", "rmse", "r2"}
    details = rec.info["classical"]
    assert details["rows"] == 56 and details["subsampled_from"] is None
    assert details["fit_seconds"] >= 0 and details["train"]["r2"] > 0.5
    if model == "SVR":
        assert details["support_vectors"] > 0 and details["kernel"] is None
    else:
        assert "RBF(length_scale=" in details["kernel"]
        assert details["initial_kernel"].startswith("ConstantKernel(constant_value=1.0)")
        assert math.isfinite(details["log_marginal_likelihood"])

    preprocessing = rec.info["preprocessing"]
    assert preprocessing["kind"] == "tabular" and preprocessing["estimator"] == model
    assert (
        preprocessing["columns"] == ["f0", "f1", "f2", "f3"] and preprocessing["target"] == "price"
    )
    assert preprocessing["target_mean"] is None and preprocessing["target_std"] is None
    saved = json.loads((tmp_path / "model.preprocessing.json").read_text())
    assert saved == preprocessing

    bundle = joblib.load(tmp_path / "model.joblib")
    assert bundle["format"] == "brickdnn-sklearn-1" and bundle["task"] == "regression"
    assert bundle["columns"] == preprocessing["columns"] and bundle["target"] == "price"
    scaler = bundle["model"].named_steps["scale"]
    assert scaler.n_samples_seen_ == 56  # fitted on the training rows only
    assert list(scaler.mean_) == pytest.approx(preprocessing["mean"])


@pytest.mark.parametrize("model", ["GaussianProcessClassifier", "SVC"])
def test_classification_on_csv(model, tmp_path):
    path = classification_csv(tmp_path / "mail.csv")
    result, rec = run_job(model_graph(model), inspect_csv(path), tmp_path, path)
    assert result.status == "completed", result.message
    assert result.message.startswith("Finished: validation accuracy ")
    assert rec.info["class_names"] == ["eggs", "ham", "spam"]  # alphabetical
    (entry,) = rec.history
    assert 0.6 <= entry["val_accuracy"] <= 1 and entry["val_loss"] > 0
    assert "val_rmse" not in entry
    bundle = joblib.load(tmp_path / "model.joblib")
    assert bundle["task"] == "classification"
    assert bundle["class_names"] == ["eggs", "ham", "spam"]
    assert bundle["class_indices"] == [0, 1, 2] and bundle["num_classes"] == 3
    if model == "GaussianProcessClassifier":
        # One Gaussian process per class (one against the rest), each with its kernel.
        assert rec.info["classical"]["kernel"].startswith("eggs: ")
    else:
        assert rec.info["classical"]["support_vectors"] > 0


def test_integer_classes_and_demo_data(tmp_path):
    path = tmp_path / "grades.csv"
    rng = np.random.default_rng(2)
    x = rng.normal(size=(60, 2))
    grades = np.where(x[:, 0] > 0, 7, 1)
    path.write_text("\n".join(f"{a:.4f};{b:.4f};{g}" for (a, b), g in zip(x, grades, strict=True)))
    result, rec = run_job(model_graph("SVC", 2), inspect_csv(path), tmp_path, path)
    assert result.status == "completed", result.message
    assert rec.info["class_names"] == ["1", "7"]
    assert rec.info["preprocessing"]["delimiter"] == ";"
    assert rec.info["preprocessing"]["columns"] == ["column_1", "column_2"]

    result, rec = run_job(
        model_graph("GaussianProcessRegressor", 3), random_info(), tmp_path / "demo"
    )
    assert result.status == "completed", result.message
    assert rec.info["task"] == "demo" and rec.info["train_size"] == 800
    assert rec.info["preprocessing"] == {"kind": "demo"}
    assert rec.history[0]["val_r2"] > 0.5  # the hidden rule can be learned
    assert (tmp_path / "demo" / "model.joblib").exists()
    assert not (tmp_path / "demo" / "model.preprocessing.json").exists()


def test_prepared_dataset_uses_its_split(tmp_path):
    path = classification_csv(tmp_path / "data.csv", n=60, labels=(0, 1, 2))
    split = bytes([0] * 40 + [1] * 15 + [2] * 5)
    (tmp_path / SPLIT_FILE).write_bytes(split)
    info = inspect_csv(path)
    info.predefined_split = [40, 15, 5]
    info.class_names = ["red", "green", "blue"]  # written as class indices 0..2
    result, rec = run_job(model_graph("SVC"), info, tmp_path, path, val_split=0.5)
    assert result.status == "completed", result.message
    assert (rec.info["train_size"], rec.info["val_size"], rec.info["test_size"]) == (40, 15, 5)
    assert rec.info["preprocessing"]["split"] == {"source": "dataset", "counts": [40, 15, 5]}
    assert rec.info["class_names"] == ["red", "green", "blue"]

    result, rec = run_job(model_graph("SVC"), info, tmp_path, path, max_samples=20)
    assert (rec.info["train_size"], rec.info["val_size"], rec.info["test_size"]) == (20, 8, 2)


def test_row_limit_subsamples_like_max_samples(tmp_path, monkeypatch):
    monkeypatch.setenv("BRICKDNN_SVM_MAX_ROWS", "40")
    path = regression_csv(tmp_path / "data.csv", n=100)
    result, rec = run_job(model_graph("SVR"), inspect_csv(path), tmp_path, path)
    assert result.status == "completed", result.message
    assert rec.info["train_size"] == 40 and rec.info["val_size"] == 10  # 40 · 0.2 / 0.8
    assert rec.info["classical"]["subsampled_from"] == 80
    assert rec.info["classical"]["max_rows"] == 40
    assert any("using 40 random rows of the 80 training rows" in line for line in rec.logs)
    # The same rows as with Max. samples 40 (the order depends only on the seed).
    monkeypatch.delenv("BRICKDNN_SVM_MAX_ROWS")
    _, limited = run_job(model_graph("SVR"), inspect_csv(path), tmp_path, path, max_samples=40)
    assert limited.info["preprocessing"]["mean"] == rec.info["preprocessing"]["mean"]


@pytest.mark.parametrize(
    "rows,graph,message",
    [
        ("a,b,c,d,y\n" + "1,2,3,4,x\n" * 10, model_graph("SVR"), "is not a number"),
        ("a,b,c,d,y\n" + "1,2,3,4,0\n" * 10, model_graph("SVC"), "only one class"),
        (
            "a,b,c,d,y\n" + "1,2,3,4,0\n2,3,4,5,1\n" * 3,
            model_graph("SVC"),
            "SVC needs at least 5 training rows",
        ),
    ],
)
def test_data_errors(rows, graph, message, tmp_path):
    path = tmp_path / "data.csv"
    path.write_text(rows)
    info = inspect_csv(path)
    if "x" in rows:
        info.target_kind, info.class_names = "number", None  # the file changed after the check
    result, rec = run_job(graph, info, tmp_path, path)
    assert result.status == "failed" and message in result.message, result.message
    assert rec.finished[2] is False and not (tmp_path / "model.joblib").exists()


def test_tuning_and_images_are_refused(tmp_path):
    from brickdnn.training.tuning import TuningConfig

    result, _ = run_job(model_graph("SVR"), random_info(), tmp_path, tuning=TuningConfig())
    assert result.status == "failed" and "not available for classical models" in result.message
    result, _ = run_job(model_graph("SVC"), builtin_info("mnist"), tmp_path)
    assert result.status == "failed" and "not on images" in result.message


def test_stop_during_the_fit_saves_nothing(tmp_path, monkeypatch):
    real = classical_engine.build_pipeline

    def slow_pipeline(graph):
        pipeline = real(graph)
        fit = pipeline.fit

        def slow_fit(*args, **kwargs):
            time.sleep(1.0)  # longer than the interval of the stop checks
            return fit(*args, **kwargs)

        pipeline.fit = slow_fit
        return pipeline

    monkeypatch.setattr(classical_engine, "build_pipeline", slow_pipeline)
    rec = Recorder(stop_when_fitting=True)
    result, _ = run_job(model_graph("SVC"), random_info(), tmp_path, reporter=rec)
    assert result.status == "stopped" and rec.finished == ("stopped", result.message, False)
    assert "nothing was saved" in result.message
    assert not (tmp_path / "model.joblib").exists() and not rec.history


def test_fit_warnings_are_logged(tmp_path):
    graph = model_graph("GaussianProcessRegressor", 3, kernel="DotProduct")
    result, rec = run_job(graph, random_info(), tmp_path)  # a linear kernel for a curve
    assert result.status == "completed", result.message
    assert any(line.startswith("ConvergenceWarning: ") for line in rec.logs), rec.logs


def test_build_pipeline_is_the_exported_make_model():
    pipeline = build_pipeline(EXAMPLES["svm_classifier"]["graph"])
    assert [name for name, _ in pipeline.steps] == ["scale", "model"]
    assert type(pipeline.named_steps["model"]).__name__ == "CalibratedClassifierCV"


# --------------------------------------------------------------------------- #
# The runner (its own process, like on the server)
# --------------------------------------------------------------------------- #


def write_job(run_dir, graph, path=None, **config):
    info = inspect_csv(path) if path else random_info()
    dataset = {"kind": info.kind, "path": str(path) if path else None, "info": info.to_dict()}
    st.write_job(run_dir, {"name": "t", "graph": graph, "dataset": dataset, "config": config})


@pytest.mark.parametrize(
    "model", ["GaussianProcessRegressor", "GaussianProcessClassifier", "SVR", "SVC"]
)
def test_runner_fits_and_saves_a_joblib_model(model, tmp_path):
    regression = model in ("GaussianProcessRegressor", "SVR")
    path = (regression_csv if regression else classification_csv)(tmp_path / "data.csv")
    run_dir = tmp_path / "run"
    write_job(run_dir, model_graph(model), path, seed=3, test_split=0.2)
    process = subprocess.run(
        [sys.executable, "-m", "brickdnn.training.runner", str(run_dir)],
        cwd=BACKEND,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert process.returncode == 0, process.stderr
    state = st.read_state(run_dir)
    assert state["status"] == "completed", state["message"]
    assert state["family"] == "classical" and state["weights"] is True
    assert state["task"] == ("regression" if regression else "classification")
    assert state["epochs"] == 1 and len(state["history"]) == 1 and state["weights_epoch"] == 1
    assert state["progress"] is None and state["preprocessing"]["kind"] == "tabular"
    details = state["classical"]
    assert details["estimator"] == model and details["fit_seconds"] is not None
    if regression:
        assert state["test"]["r2"] is not None and state["history"][0]["val_r2"] is not None
    else:
        assert state["test"]["accuracy"] is not None and state["class_names"]
    if model.startswith("Gaussian"):
        assert details["kernel"] and details["log_marginal_likelihood"] is not None
    else:
        assert details["support_vectors"] > 0
    assert sorted(p.name for p in run_dir.iterdir()) == [
        "job.json",
        "model.joblib",
        "model.preprocessing.json",
        "state.json",
    ]


def test_runner_reports_a_missing_scikit_learn(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    write_job(run_dir, model_graph("SVR"))
    import brickdnn.training

    monkeypatch.setitem(sys.modules, "brickdnn.training.classical_engine", None)
    monkeypatch.delattr(brickdnn.training, "classical_engine")
    state = runner.run(run_dir)
    assert state["status"] == "failed" and "pip install scikit-learn" in state["message"]
    assert state["family"] == "classical"


def test_stopping_a_long_fit_ends_the_runner(tmp_path):
    path = regression_csv(tmp_path / "data.csv", n=1500, features=8)
    run_dir = tmp_path / "run"
    graph = model_graph("GaussianProcessRegressor", 8, {"n_restarts_optimizer": 20})
    write_job(run_dir, graph, path, seed=1)
    process = subprocess.Popen(
        [sys.executable, "-m", "brickdnn.training.runner", str(run_dir)],
        cwd=BACKEND,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 60
        while not (st.read_state(run_dir) or {}).get("message", "").startswith("Fitting"):
            assert time.monotonic() < deadline, st.read_state(run_dir)
            assert process.poll() is None, st.read_state(run_dir)
            time.sleep(0.05)
        st.request_stop(run_dir)
        stopped = time.monotonic()
        process.wait(timeout=20)
        assert time.monotonic() - stopped < 10  # does not wait for the fit to finish
    finally:
        if process.poll() is None:
            process.kill()
    state = st.read_state(run_dir)
    assert state["status"] == "stopped" and state["weights"] is False
    assert "nothing was saved" in state["message"]
    assert not (run_dir / "model.joblib").exists()


# --------------------------------------------------------------------------- #
# Predictions
# --------------------------------------------------------------------------- #


def trained_run(tmp_path, graph, path=None, **config):
    """A finished run directory, trained in this process."""
    run_dir = tmp_path / "run"
    write_job(run_dir, graph, path, seed=2, **config)
    job = TrainingJob.from_dict(st.read_job(run_dir))
    rec = Recorder()
    result = train(job, rec, run_dir / "model.joblib")
    assert result.status == "completed", result.message
    plan = check_compatibility(graph, job.dataset).plan.to_dict()
    return run_dir, plan, rec.info["preprocessing"]


def test_predict_classes_with_probabilities(tmp_path):
    path = classification_csv(tmp_path / "mail.csv")
    run_dir, plan, preprocessing = trained_run(tmp_path, model_graph("SVC"), path)
    rows = [{"a": 2, "b": 0, "c": 0, "d": 0}, [-2, 0, 0, 0], [0.5, 2, 0, 0]]
    prepared = inference.prepare_inputs(plan, preprocessing, rows, top_k=3)
    answer = predict.predict(run_dir, {"inputs": prepared.tensors, "kind": prepared.kind})
    assert answer["probabilities"] is True and len(answer["outputs"][0]) == 3
    predictions = inference.read_outputs(
        plan, preprocessing, answer["outputs"], 3, 3, probabilities=True
    )
    # The same as the pipeline on the raw rows (the scaler is skipped for prepared inputs).
    pipeline = joblib.load(run_dir / "model.joblib")["model"]
    raw = np.array([[2, 0, 0, 0], [-2, 0, 0, 0], [0.5, 2, 0, 0]], dtype=float)
    expected = pipeline.predict_proba(raw)
    for prediction, probabilities in zip(predictions, expected, strict=True):
        best = int(np.argmax(probabilities))
        assert prediction["class"] == ["eggs", "ham", "spam"][best]
        assert prediction["probability"] == pytest.approx(probabilities[best], abs=1e-4)
        assert sum(t["probability"] for t in prediction["top"]) == pytest.approx(1, abs=1e-3)
    assert [p["class"] for p in predictions[:2]] == ["spam", "ham"]


def test_predict_gp_values_with_uncertainty(tmp_path):
    path = regression_csv(tmp_path / "houses.csv")
    run_dir, plan, preprocessing = trained_run(
        tmp_path, model_graph("GaussianProcessRegressor"), path
    )
    rows = [[0, 0, 0, 0], [1, 0, 0, 0]]
    prepared = inference.prepare_inputs(plan, preprocessing, rows)
    answer = predict.predict(run_dir, {"inputs": prepared.tensors, "kind": prepared.kind})
    assert "probabilities" not in answer and len(answer["std"]) == 2
    predictions = inference.read_outputs(
        plan, preprocessing, answer["outputs"], 2, std=answer["std"]
    )
    assert predictions[0]["value"] == pytest.approx(100, abs=1)  # in the units of the target
    assert predictions[1]["value"] - predictions[0]["value"] == pytest.approx(3, abs=0.5)
    assert all(p["std"] > 0 for p in predictions)


def test_predict_demo_runs_and_bad_requests(tmp_path):
    run_dir, plan, preprocessing = trained_run(tmp_path, model_graph("SVR", 3))
    assert preprocessing == {"kind": "demo"}
    prepared = inference.prepare_inputs(plan, preprocessing, [[0.1, 0.2, 0.3]])
    answer = predict.predict(run_dir, {"inputs": prepared.tensors, "kind": prepared.kind})
    assert len(answer["outputs"][0]) == 1 and "std" not in answer
    with pytest.raises(predict.PredictionError, match="exactly one input"):
        predict.predict(run_dir, {"inputs": []})
    with pytest.raises(predict.PredictionError, match="shape they claim"):
        predict.predict(run_dir, {"inputs": [{"shape": [1, 3], "values": [1]}]})
    (run_dir / "model.joblib").unlink()
    with pytest.raises(predict.PredictionError, match="no fitted model"):
        predict.predict(run_dir, {"inputs": prepared.tensors})


def test_inference_of_classical_runs():
    plan = {
        "task": "regression",
        "inputs": [{"shape": [2], "dtype": "float", "vocab": None}],
        "targets": [{"kind": "value", "shape": [1]}],
        "classical": {"estimator": "SVR", "type": "SVR"},
    }
    preprocessing = {
        "kind": "tabular",
        "columns": ["rooms", "area"],
        "target": "price",
        "mean": [3.0, 80.0],
        "std": [1.0, 20.0],
        "target_mean": None,
        "target_std": None,
    }
    assert inference.kind_of(plan, preprocessing) == "tabular"
    neural = {key: value for key, value in plan.items() if key != "classical"}
    with pytest.raises(inference.UnknownPreprocessing):
        inference.kind_of(neural, preprocessing)  # a neural network needs target_mean/std
    assert inference.read_outputs(plan, preprocessing, [[[250.0]]], 1) == [{"value": 250.0}]
    with pytest.raises(ValueError, match="uncertainties"):
        inference.read_outputs(plan, preprocessing, [[[1.0]]], 1, std=[1.0, 2.0])
    snippet = inference.usage_snippet(plan, preprocessing, "houses", "houses.joblib")
    assert 'joblib.load("houses.joblib")' in snippet and "model.predict(rows)" in snippet
    assert "import torch" not in snippet


@pytest.mark.parametrize("model", ["GaussianProcessRegressor", "SVC"])
def test_usage_snippet_runs(model, tmp_path):
    path = classification_csv if model == "SVC" else regression_csv
    run_dir, plan, preprocessing = trained_run(
        tmp_path, model_graph(model), path(tmp_path / "d.csv")
    )
    snippet = inference.usage_snippet(plan, preprocessing, "m", "model.joblib")
    result = subprocess.run(
        [sys.executable, "-c", snippet], cwd=run_dir, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    if model == "SVC":
        assert json.loads(result.stdout.replace("'", '"'))[0] in ("eggs", "ham", "spam")
    else:
        assert "return_std=True" in snippet and len(result.stdout.split("] [")) == 2
