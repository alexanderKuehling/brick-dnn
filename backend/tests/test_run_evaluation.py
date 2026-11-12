"""The final evaluation of a run (state.json key ``evaluation``): the metrics against
scikit-learn, the evaluation of real neural and classical runs (with and without a test set,
in a hyperparameter search), state.json, the CLI table and the compact MCP answer; old runs
without an evaluation keep working."""

# ruff: noqa: F811 (the fixture training_dir is imported from test_training_api)

from __future__ import annotations

import io
import json
import math

import pytest

np = pytest.importorskip("numpy")
sklearn_metrics = pytest.importorskip("sklearn.metrics")

from brickdnn.training import evaluation as ev  # noqa: E402

# --------------------------------------------------------------------------- #
# The metrics
# --------------------------------------------------------------------------- #


def as_zero(values):
    """null (undefined) counts as 0 in scikit-learn with zero_division=0."""
    return [0.0 if v is None else v for v in values]


def random_probabilities(rng, n, k, skew=None):
    proba = rng.dirichlet(np.ones(k), size=n)
    if skew is not None:
        proba[:, skew] *= 0.01  # this class is (almost) never predicted
        proba /= proba.sum(axis=1, keepdims=True)
    return proba


def check_against_sklearn(proba, y, result):
    k = proba.shape[1]
    labels = list(range(k))
    predicted = proba.argmax(axis=1)
    matrix = sklearn_metrics.confusion_matrix(y, predicted, labels=labels)
    assert result["confusion_matrix"] == matrix.tolist()
    precision, recall, f1, support = sklearn_metrics.precision_recall_fscore_support(
        y, predicted, labels=labels, zero_division=0
    )
    per_class = result["per_class"]
    assert as_zero(per_class["precision"]) == pytest.approx(precision, abs=1e-6)
    assert as_zero(per_class["recall"]) == pytest.approx(recall, abs=1e-6)
    assert as_zero(per_class["f1"]) == pytest.approx(f1, abs=1e-6)
    assert per_class["support"] == support.tolist()
    assert result["accuracy"] == pytest.approx(sklearn_metrics.accuracy_score(y, predicted))
    assert result["macro_f1"] == pytest.approx(
        sklearn_metrics.f1_score(y, predicted, average="macro", zero_division=0)
    )
    assert result["weighted_f1"] == pytest.approx(
        sklearn_metrics.f1_score(y, predicted, average="weighted", zero_division=0)
    )
    with _quiet():
        balanced = sklearn_metrics.balanced_accuracy_score(y, predicted)
    assert result["balanced_accuracy"] == pytest.approx(balanced)
    assert result["log_loss"] == pytest.approx(
        sklearn_metrics.log_loss(y, proba, labels=labels), rel=1e-6
    )
    for top in (3, 5):
        if k > top:
            expected = sklearn_metrics.top_k_accuracy_score(y, proba, k=top, labels=labels)
            assert result["top_k_accuracy"][str(top)] == pytest.approx(expected)
        else:
            assert str(top) not in (result["top_k_accuracy"] or {})


class _quiet:
    """Silences scikit-learn's warnings about classes that are only predicted."""

    def __enter__(self):
        import warnings

        self.context = warnings.catch_warnings()
        self.context.__enter__()
        warnings.simplefilter("ignore")

    def __exit__(self, *args):
        self.context.__exit__(*args)


@pytest.mark.parametrize("k", [2, 3, 4, 7])
def test_classification_metrics_match_scikit_learn(k):
    rng = np.random.default_rng(k)
    n = 300
    # The last class never occurs in the targets, the first is (almost) never predicted.
    y = rng.integers(0, k - 1 if k > 2 else k, size=n)
    proba = random_probabilities(rng, n, k, skew=0)
    counter = ev.ClassificationCounter(k)
    for start in range(0, n, 64):  # batch by batch, like the engine
        counter.add(proba[start : start + 64], y[start : start + 64])
    result = counter.result(loss=0.5)
    assert result["size"] == n and result["loss"] == 0.5
    check_against_sklearn(proba, y, result)
    if k > 2:
        # A class without rows has no recall; one without predictions no precision.
        assert result["per_class"]["recall"][k - 1] is None
    if k in (2, 3):
        assert result["top_k_accuracy"] is None


def test_classification_metrics_by_hand():
    # true \ predicted:  a  b  c
    matrix = np.array([[5, 1, 0], [2, 3, 0], [0, 0, 0]])  # c: neither true nor predicted
    result = ev.classification_metrics(matrix)
    assert result["size"] == 11 and result["accuracy"] == pytest.approx(8 / 11)
    per_class = result["per_class"]
    assert per_class["precision"][:2] == pytest.approx([5 / 7, 3 / 4])
    assert per_class["recall"][:2] == pytest.approx([5 / 6, 3 / 5])
    assert per_class["precision"][2] is None and per_class["recall"][2] is None
    assert per_class["f1"][2] is None and per_class["support"] == [6, 5, 0]
    f1_a = 2 * (5 / 7) * (5 / 6) / (5 / 7 + 5 / 6)
    f1_b = 2 * (3 / 4) * (3 / 5) / (3 / 4 + 3 / 5)
    assert result["macro_f1"] == pytest.approx((f1_a + f1_b) / 2)  # c does not count
    assert result["weighted_f1"] == pytest.approx((6 * f1_a + 5 * f1_b) / 11)
    assert result["balanced_accuracy"] == pytest.approx((5 / 6 + 3 / 5) / 2)
    assert result["most_confused"] == [
        {"true": 1, "predicted": 0, "count": 2},
        {"true": 0, "predicted": 1, "count": 1},
    ]


def test_many_classes_keep_their_values_but_no_matrix():
    k = ev.MAX_MATRIX_CLASSES + 10
    rng = np.random.default_rng(0)
    y = rng.integers(0, k, size=2000)
    proba = random_probabilities(rng, 2000, k)
    counter = ev.ClassificationCounter(k)
    counter.add(proba, y)
    result = counter.result()
    assert result["confusion_matrix"] is None
    assert len(result["per_class"]["f1"]) == k and sum(result["per_class"]["support"]) == 2000
    pairs = result["most_confused"]
    assert len(pairs) == ev.MAX_CONFUSED
    counts = [p["count"] for p in pairs]
    assert counts == sorted(counts, reverse=True)
    matrix = sklearn_metrics.confusion_matrix(y, proba.argmax(axis=1), labels=list(range(k)))
    np.fill_diagonal(matrix, 0)
    assert counts[0] == matrix.max()
    assert set(result["top_k_accuracy"]) == {"3", "5"}

    small = ev.ClassificationCounter(ev.MAX_MATRIX_CLASSES)
    small.add(proba[:, : ev.MAX_MATRIX_CLASSES], y % ev.MAX_MATRIX_CLASSES)
    assert len(small.result()["confusion_matrix"]) == ev.MAX_MATRIX_CLASSES


def test_regression_metrics_match_scikit_learn():
    rng = np.random.default_rng(1)
    actual = rng.uniform(50, 150, size=800)
    predicted = actual + rng.normal(0, 5, size=800)
    collector = ev.RegressionCollector()
    for start in range(0, 800, 100):
        collector.add(predicted[start : start + 100], actual[start : start + 100])
    result = collector.result(loss=0.1)
    m = sklearn_metrics
    assert result["size"] == 800 and result["loss"] == 0.1
    assert result["mae"] == pytest.approx(m.mean_absolute_error(actual, predicted))
    assert result["mse"] == pytest.approx(m.mean_squared_error(actual, predicted))
    assert result["rmse"] == pytest.approx(math.sqrt(m.mean_squared_error(actual, predicted)))
    assert result["r2"] == pytest.approx(m.r2_score(actual, predicted))
    assert result["mape"] == pytest.approx(m.mean_absolute_percentage_error(actual, predicted))
    assert result["median_absolute_error"] == pytest.approx(
        m.median_absolute_error(actual, predicted)
    )
    assert result["max_error"] == pytest.approx(m.max_error(actual, predicted))
    residuals = predicted - actual
    assert result["residuals"]["median"] == pytest.approx(np.median(residuals))
    assert result["residuals"]["q05"] == pytest.approx(np.quantile(residuals, 0.05))
    assert result["residuals"]["q95"] == pytest.approx(np.quantile(residuals, 0.95))
    sample = result["sample"]
    assert len(sample["actual"]) == len(sample["predicted"]) == ev.MAX_SAMPLE
    assert sample["actual"][0] == pytest.approx(actual[0], rel=1e-5)
    assert sample["actual"][-1] == pytest.approx(actual[-1], rel=1e-5)


@pytest.mark.parametrize(
    "actual,mape",
    [
        ([10.0, 20.0, 30.0], True),
        ([-10.0, 20.0, -30.0], True),  # the sign does not matter
        ([0.0, 20.0, 30.0], False),  # 0: the relative error is undefined
        ([0.1, 20.0, 30.0], False),  # 0.1 < 1 % of the mean absolute value (16.7)
        ([0.2, 20.0, 30.0], True),
    ],
)
def test_mape_only_without_targets_near_zero(actual, mape):
    actual = np.array(actual)
    result = ev.regression_metrics(actual * 1.1, actual)
    if mape:
        assert result["mape"] == pytest.approx(0.1)
    else:
        assert result["mape"] is None


def test_regression_edge_cases():
    same = ev.regression_metrics(np.array([1.0, 2.0, 3.0]), np.full(3, 2.0))
    assert same["r2"] is None and same["mae"] == pytest.approx(2 / 3)
    assert len(same["sample"]["actual"]) == 3  # small splits keep every row
    empty = ev.RegressionCollector().result()
    assert empty["size"] == 0 and empty["rmse"] is None


def test_class_names_are_numbered_where_missing():
    assert ev.class_names_for(None, 3) == ["0", "1", "2"]
    assert ev.class_names_for(["cat"], 2) == ["cat", "1"]
    assert ev.class_names_for(["a", "b", "c"], 2) == ["a", "b"]


# --------------------------------------------------------------------------- #
# Neural networks
# --------------------------------------------------------------------------- #

torch = pytest.importorskip("torch")

from brickdnn import EXAMPLES  # noqa: E402
from brickdnn.training import (  # noqa: E402
    cli,
    data,
    inspect_csv,
    random_info,
    runner,
)
from brickdnn.training import state as st  # noqa: E402
from brickdnn.training.engine import TrainingJob, build_model, train  # noqa: E402
from brickdnn.training.tuning import TuningConfig  # noqa: E402

from .test_training_engine import Recorder as EngineRecorder  # noqa: E402
from .test_training_engine import config, iris_like, mlp  # noqa: E402


class Recorder(EngineRecorder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.evaluation = None

    def evaluation_end(self, result):
        self.evaluation = result
        self.events.append(("evaluation_end",))


def run_neural(graph, info, tmp_path, path=None, tuning=None, reporter=None, **settings):
    rec = reporter or Recorder()
    job = TrainingJob(
        graph=graph,
        dataset=info,
        config=config(**settings),
        path=str(path) if path else None,
        tuning=TuningConfig.from_dict(tuning) if tuning else None,
    )
    return train(job, rec, tmp_path / "weights.pt"), rec


def reference_split(tmp_path, rec, n, val_split=0.2, test_split=0.0, part=1):
    """Features and targets of a split (1: validation, 2: test) of a CSV run with seed 0,
    prepared like the engine does, and the outputs of the saved weights for them."""
    split = data.split_indices(n, val_split, None, torch.Generator().manual_seed(0), test_split)
    return split[part]


def model_outputs(graph, tmp_path, x):
    model = build_model(graph)
    model.load_state_dict(torch.load(tmp_path / "weights.pt"))
    model.eval()
    with torch.no_grad():
        return model(x)


def csv_features(path, rec):
    rows = [line.split(",") for line in path.read_text().strip().splitlines()[1:]]
    x = torch.tensor([[float(v) for v in row[:-1]] for row in rows])
    pre = rec.info["preprocessing"]
    x = (x - torch.tensor(pre["mean"])) / torch.tensor(pre["std"])
    return x, [row[-1] for row in rows]


@pytest.mark.parametrize("test_split", [0.0, 0.2])
def test_neural_classification_with_text_labels(tmp_path, test_split):
    path = tmp_path / "iris.csv"
    path.write_text(iris_like(80))
    graph = mlp(4, 3)
    result, rec = run_neural(
        graph,
        inspect_csv(path),
        tmp_path,
        path,
        epochs=3,
        batch_size=8,
        learning_rate=0.02,
        test_split=test_split,
    )
    assert result.status == "completed", result.message
    found = rec.evaluation
    assert found["task"] == "classification" and found["trial"] is None
    assert found["class_names"] == ["setosa", "versicolor", "virginica"]
    assert found["epoch"] == result.weights_epoch
    kept = next(e for e in result.history if e["epoch"] == result.weights_epoch)
    assert found["train"] == {"loss": kept["train_loss"], "accuracy": kept["train_accuracy"]}
    val = found["val"]
    assert val["size"] == rec.info["val_size"]
    assert val["accuracy"] == pytest.approx(kept["val_accuracy"])
    assert val["loss"] == pytest.approx(kept["val_loss"], rel=1e-5)
    assert val["log_loss"] == pytest.approx(val["loss"], rel=1e-4)
    assert val["top_k_accuracy"] is None  # 3 classes: top-3 is always right

    # The same numbers from the saved weights, with scikit-learn.
    x, labels = csv_features(path, rec)
    index = {name: i for i, name in enumerate(found["class_names"])}
    y = np.array([index[label] for label in labels])
    parts = [("val", 1)] + ([("test", 2)] if test_split else [])
    for key, part in parts:
        rows = reference_split(tmp_path, rec, 80, test_split=test_split, part=part)
        proba = torch.softmax(model_outputs(graph, tmp_path, x[rows]).double(), dim=1).numpy()
        check_against_sklearn(proba, y[rows.numpy()], found[key])
    if test_split:
        assert found["test"]["size"] == rec.info["test_size"]
        assert found["test"]["loss"] == pytest.approx(result.test["loss"])
        assert found["test"]["accuracy"] == pytest.approx(result.test["accuracy"])
        assert rec.events.index(("test_end",)) < rec.events.index(("evaluation_end",))
    else:
        assert found["test"] is None


def test_neural_binary_classification(tmp_path):
    path = tmp_path / "spam.csv"
    path.write_text(iris_like(60, labels=("ham", "spam")))
    result, rec = run_neural(mlp(4, 2), inspect_csv(path), tmp_path, path, epochs=2, batch_size=8)
    assert result.status == "completed", result.message
    val = rec.evaluation["val"]
    assert len(val["confusion_matrix"]) == 2 and sum(map(sum, val["confusion_matrix"])) == 12
    assert rec.evaluation["class_names"] == ["ham", "spam"]


def noisy_csv(tmp_path, n=80):
    gen = torch.Generator().manual_seed(1)
    x = torch.randn(n, 3, generator=gen)
    y = 1000 + 250 * (x @ torch.tensor([1.0, -2.0, 0.5])) + 20 * torch.randn(n, generator=gen)
    lines = ["a,b,c,price"] + [
        ",".join(f"{v:.4f}" for v in row.tolist()) + f",{t:.2f}"
        for row, t in zip(x, y.tolist(), strict=True)
    ]
    path = tmp_path / "prices.csv"
    path.write_text("\n".join(lines))
    return path


@pytest.mark.parametrize("test_split", [0.0, 0.25])
def test_neural_regression_in_the_units_of_the_target(tmp_path, test_split):
    path = noisy_csv(tmp_path)
    graph = mlp(3, 1)
    result, rec = run_neural(
        graph, inspect_csv(path), tmp_path, path, epochs=4, batch_size=8, test_split=test_split
    )
    assert result.status == "completed", result.message
    found = rec.evaluation
    assert found["task"] == "regression" and "class_names" not in found
    kept = next(e for e in result.history if e["epoch"] == result.weights_epoch)
    val = found["val"]
    # Consistent with the history: RMSE = √(loss) · target std, R² of the same rows.
    assert val["rmse"] == pytest.approx(kept["val_rmse"], rel=1e-4)
    assert val["r2"] == pytest.approx(kept["val_r2"], rel=1e-4, abs=1e-6)
    assert val["loss"] == pytest.approx(kept["val_loss"], rel=1e-5)
    assert found["train"]["accuracy"] is None

    x, targets = csv_features(path, rec)
    actual = np.array([float(t) for t in targets])
    pre = rec.info["preprocessing"]
    parts = [("val", 1)] + ([("test", 2)] if test_split else [])
    for key, part in parts:
        rows = reference_split(tmp_path, rec, 80, test_split=test_split, part=part).numpy()
        out = model_outputs(graph, tmp_path, x[rows]).double().reshape(-1).numpy()
        predicted = out * pre["target_std"] + pre["target_mean"]
        m = sklearn_metrics
        split = found[key]
        assert split["mae"] == pytest.approx(m.mean_absolute_error(actual[rows], predicted), 1e-4)
        assert split["r2"] == pytest.approx(m.r2_score(actual[rows], predicted), rel=1e-3)
        assert split["max_error"] == pytest.approx(m.max_error(actual[rows], predicted), 1e-4)
        assert split["mape"] == pytest.approx(
            m.mean_absolute_percentage_error(actual[rows], predicted), rel=1e-3
        )
        # The sample is in the units of the CSV file (about 1000 ± 600), every row here.
        assert split["sample"]["actual"] == pytest.approx(list(actual[rows]), rel=1e-5)
    if not test_split:
        assert found["test"] is None


def test_search_evaluates_the_best_trial_only(tmp_path):
    path = tmp_path / "iris.csv"
    path.write_text(iris_like(60))
    tuning = {"method": "grid", "space": {"learning_rate": [0.05, 0.0001]}}
    result, rec = run_neural(
        mlp(4, 3), inspect_csv(path), tmp_path, path, tuning=tuning, epochs=2, batch_size=8
    )
    assert result.status == "completed", result.message
    assert rec.events.count(("evaluation_end",)) == 1
    assert rec.events[-2:] == [("evaluation_end",), ("finish", "completed")]
    found = rec.evaluation
    best = next(t for t in result.trials if t["trial"] == result.best_trial)
    assert found["trial"] == result.best_trial and found["epoch"] == best["weights_epoch"]
    assert found["val"]["accuracy"] == pytest.approx(best["val_accuracy"])
    assert found["val"]["loss"] == pytest.approx(best["val_loss"], rel=1e-5)


def test_stopped_and_multi_output_runs_get_no_evaluation(tmp_path, monkeypatch):
    from brickdnn.training import engine

    monkeypatch.setattr(engine, "STOP_CHECK_INTERVAL", 0.0)  # check every batch
    rec = Recorder(stop_after_checks=3)
    graph = EXAMPLES["mlp_mnist"]["graph"]
    result, _ = run_neural(graph, random_info(), tmp_path, reporter=rec, max_samples=256)
    assert result.status == "stopped" and rec.evaluation is None

    two_heads = {
        "version": 1,
        "nodes": [
            {"id": "in", "type": "Input", "params": {"shape": [6]}},
            {"id": "classes", "type": "Linear", "params": {"out_features": 4}},
            {"id": "value", "type": "Linear", "params": {"out_features": 1}},
        ],
        "edges": [{"source": "in", "target": "classes"}, {"source": "in", "target": "value"}],
    }
    result, rec = run_neural(two_heads, random_info(), tmp_path, max_samples=64)
    assert result.status == "completed", result.message
    assert rec.evaluation is None  # two outputs: no single target to measure


def test_demo_classification_numbers_the_classes(tmp_path):
    result, rec = run_neural(
        EXAMPLES["mlp_mnist"]["graph"], random_info(), tmp_path, max_samples=128
    )
    assert result.status == "completed", result.message
    found = rec.evaluation
    assert found["class_names"] == [str(i) for i in range(10)]
    assert set(found["val"]["top_k_accuracy"]) == {"3", "5"}


def test_runner_writes_the_evaluation_into_state_json(tmp_path):
    path = tmp_path / "iris.csv"
    path.write_text(iris_like(60))
    info = inspect_csv(path)
    run_dir = tmp_path / "run"
    dataset = {"kind": "upload", "path": str(path), "info": info.to_dict()}
    job = {
        "name": "t",
        "graph": mlp(4, 3),
        "dataset": dataset,
        "config": {"epochs": 2, "batch_size": 8, "device": "cpu", "seed": 0, "test_split": 0.2},
    }
    st.write_job(run_dir, job)
    state = runner.run(run_dir)
    assert state["status"] == "completed", state["message"]
    saved = st.read_state(run_dir)
    assert saved["evaluation"]["task"] == "classification"
    assert saved["evaluation"]["test"]["size"] == saved["test_size"]
    json.dumps(saved, allow_nan=False)


def test_initial_state_has_no_evaluation():
    assert st.initial_state()["evaluation"] is None


def test_cli_prints_the_evaluation():
    out = io.StringIO()
    reporter = cli.ConsoleReporter(1, st.Path("w.pt"), out)
    counter = ev.ClassificationCounter(3)
    counter.add(np.eye(3)[[0, 1, 2, 2]] * 0.8 + 0.0667, np.array([0, 1, 2, 1]))
    reporter.evaluation_end(
        ev.evaluation(
            "classification",
            counter.result(0.4),
            None,
            {"loss": 0.3, "accuracy": 0.9},
            class_names=["cat", "dog", "bird"],
            epoch=2,
        )
    )
    text = out.getvalue()
    assert "Final evaluation (of the kept weights)" in text
    assert "balanced accuracy" in text and "75.0%" in text
    assert "Per class (validation)" in text and "dog" in text
    assert "top-3" not in text  # 3 classes

    out = io.StringIO()
    reporter = cli.ConsoleReporter(1, st.Path("w.pt"), out)
    collector = ev.RegressionCollector()
    collector.add(np.array([1.0, 2.0, 4.0]), np.array([1.0, 2.0, 3.0]))
    reporter.evaluation_end(
        ev.evaluation("regression", collector.result(0.1), collector.result(0.2), {"loss": 0.1})
    )
    text = out.getvalue()
    assert "RMSE" in text and "MAPE" in text and "test" in text


# --------------------------------------------------------------------------- #
# Classical models
# --------------------------------------------------------------------------- #

from .test_classical_training import (  # noqa: E402
    Recorder as ClassicalRecorder,
)
from .test_classical_training import (  # noqa: E402
    classification_csv,
    model_graph,
    regression_csv,
)
from .test_classical_training import run_job as run_classical_job  # noqa: E402


class ClassicalRec(ClassicalRecorder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.evaluation = None

    def evaluation_end(self, result):
        self.evaluation = result


@pytest.mark.parametrize("test_split", [0.0, 0.2])
def test_classical_svc_with_text_labels(tmp_path, test_split):
    path = classification_csv(tmp_path / "mail.csv", n=120)
    rec = ClassicalRec()
    result, _ = run_classical_job(
        model_graph("SVC"), inspect_csv(path), tmp_path, path, reporter=rec, test_split=test_split
    )
    assert result.status == "completed", result.message
    found = rec.evaluation
    assert found["task"] == "classification" and found["epoch"] == 1
    assert found["class_names"] == ["eggs", "ham", "spam"]
    (entry,) = rec.history
    assert found["val"]["accuracy"] == pytest.approx(entry["val_accuracy"])
    assert found["val"]["loss"] == pytest.approx(entry["val_loss"])
    assert found["val"]["log_loss"] == pytest.approx(entry["val_loss"])
    train = rec.info["classical"]["train"]
    assert found["train"] == {"loss": train["loss"], "accuracy": train["accuracy"]}
    assert sum(found["val"]["per_class"]["support"]) == rec.info["val_size"]
    if test_split:
        assert found["test"]["accuracy"] == pytest.approx(result.test["accuracy"])
        assert sum(map(sum, found["test"]["confusion_matrix"])) == rec.info["test_size"]
    else:
        assert found["test"] is None


@pytest.mark.parametrize("test_split", [0.0, 0.1])
def test_classical_gaussian_process_regression(tmp_path, test_split):
    path = regression_csv(tmp_path / "houses.csv")
    rec = ClassicalRec()
    result, _ = run_classical_job(
        model_graph("GaussianProcessRegressor"),
        inspect_csv(path),
        tmp_path,
        path,
        reporter=rec,
        test_split=test_split,
    )
    assert result.status == "completed", result.message
    found = rec.evaluation
    (entry,) = rec.history
    val = found["val"]
    assert val["rmse"] == pytest.approx(entry["val_rmse"])
    assert val["r2"] == pytest.approx(entry["val_r2"])
    assert val["loss"] == pytest.approx(entry["val_loss"])
    # The target is about 100: MAPE exists. The sample holds every validation row (with 6
    # significant digits, so the errors of about 0.05 match to about 1 %).
    sample = val["sample"]
    actual, predicted = np.array(sample["actual"]), np.array(sample["predicted"])
    assert len(actual) == rec.info["val_size"] and 90 < actual.mean() < 110
    assert val["mape"] == pytest.approx(
        sklearn_metrics.mean_absolute_percentage_error(actual, predicted), rel=0.01
    )
    assert val["mae"] == pytest.approx(
        sklearn_metrics.mean_absolute_error(actual, predicted), rel=0.01
    )
    if test_split:
        assert found["test"]["rmse"] == pytest.approx(result.test["rmse"])
        assert found["test"]["size"] == rec.info["test_size"]
    else:
        assert found["test"] is None


# --------------------------------------------------------------------------- #
# API and MCP
# --------------------------------------------------------------------------- #

from brick_dnn_core import mcp_compact  # noqa: E402


def classification_evaluation(k, test=True):
    rng = np.random.default_rng(k)
    y = rng.integers(0, k, size=400)
    proba = random_probabilities(rng, 400, k)
    counter = ev.ClassificationCounter(k)
    counter.add(proba, y)
    split = counter.result(1.2)
    return ev.evaluation(
        "classification",
        split,
        split if test else None,
        {"loss": 1.0, "accuracy": 0.5},
        class_names=[f"class {i}" for i in range(k)],
        epoch=3,
    )


def test_mcp_summary_of_a_few_classes_has_the_matrix():
    found = mcp_compact.evaluation_summary(classification_evaluation(4))
    val = found["validation"]
    assert len(val["confusion_matrix"]) == 4
    assert set(val["per_class"]) == {"class 0", "class 1", "class 2", "class 3"}
    assert set(val["per_class"]["class 0"]) == {"precision", "recall", "f1", "support"}
    assert "most_confused" not in val and found["classes"][0] == "class 0"
    assert found["train"] == {"loss": 1.0, "accuracy": 0.5}
    assert "sample" not in json.dumps(found) and len(json.dumps(found)) < 2500


def test_mcp_summary_of_many_classes_has_the_most_confused_pairs():
    found = mcp_compact.evaluation_summary(classification_evaluation(100))
    val = found["validation"]
    assert "confusion_matrix" not in val and "per_class" not in val
    assert len(val["most_confused"]) == mcp_compact.MAX_PAIRS
    assert val["most_confused"][0]["true"].startswith("class ")
    assert len(val["lowest_f1"]) == mcp_compact.MAX_PAIRS
    assert set(val["top_k_accuracy"]) == {"3", "5"}
    assert len(json.dumps(found)) < 2500


def test_mcp_summary_of_a_regression_leaves_out_the_sample():
    collector = ev.RegressionCollector()
    collector.add(np.linspace(1, 100, 900) + 0.5, np.linspace(1, 100, 900))
    found = mcp_compact.evaluation_summary(
        ev.evaluation("regression", collector.result(0.1), None, {"loss": 0.1, "accuracy": None})
    )
    assert set(found["validation"]) >= {"mae", "rmse", "r2", "max_error", "residuals"}
    assert "sample" not in found["validation"] and "test" not in found
    assert len(json.dumps(found)) < 800


def test_mcp_summary_without_validation_and_test_rows():
    rng = np.random.default_rng(1)
    counter = ev.ClassificationCounter(3)
    counter.add(random_probabilities(rng, 50, 3), rng.integers(0, 3, size=50))
    found = mcp_compact.evaluation_summary(
        ev.evaluation(
            "classification",
            None,
            None,
            {"loss": 1.0, "accuracy": 0.5},
            class_names=["a", "b", "c"],
            epoch=4,
            training=counter.result(0.9),
        )
    )
    assert "validation" not in found and "test" not in found
    assert found["training_optimistic"]["size"] == 50
    assert len(found["training_optimistic"]["confusion_matrix"]) == 3


@pytest.mark.django_db
def test_run_without_validation_data_in_the_api_and_mcp(training_dir):
    from django.contrib.auth import get_user_model
    from rest_framework.test import APIClient

    from brick_dnn_core import training_service
    from brick_dnn_core.models import TrainingRun

    from .test_mcp import call, mcp_client
    from .test_mcp_compact import long_run

    user = get_user_model().objects.create_user("eve", password="correct-horse-1")
    client = APIClient()
    client.force_authenticate(user)
    run = long_run(user, training_dir, epochs=3)
    directory = training_service.run_dir(run.pk)
    state = st.read_state(directory)
    for entry in state["history"]:
        entry.update(val_loss=None, val_accuracy=None)
    state.update(best=None, weights_epoch=3, test=None, val_size=0, test_size=None)
    state["evaluation"] = ev.evaluation(
        "classification",
        None,
        None,
        {"loss": 0.3, "accuracy": 0.9},
        class_names=["a", "b", "c"],
        epoch=3,
        training=classification_evaluation(3)["val"],
    )
    st.write_state(directory, state)
    detail = client.get(f"/api/training/runs/{run.pk}/").json()
    assert detail["state"]["evaluation"]["val"] is None
    assert detail["state"]["evaluation"]["training"]["size"] == 400
    # What the server keeps of the state (as when it syncs a running run).
    training_service._apply_state(run, state)
    run.save()
    saved = TrainingRun.objects.get(pk=run.pk)
    assert saved.metrics["val_loss"] is None and saved.metrics["train_accuracy"] == 0.75
    assert saved.best is None
    result, text = call(client := mcp_client(user), "get_training_run", id=str(run.pk))
    assert "isError" not in result, text
    answer = result["structuredContent"]
    assert "val_loss" not in (answer.get("metrics") or {})
    assert answer["metrics"]["train_accuracy"] == 0.75
    assert answer["evaluation"]["training_optimistic"]["size"] == 400
    assert "validation" not in answer["evaluation"]
    result, text = call(client, "list_training_runs")
    assert "isError" not in result, text
    assert "training accuracy (optimistic)" in text or "train_accuracy" in text


@pytest.mark.django_db
def test_runs_with_and_without_an_evaluation_in_the_api_and_mcp(training_dir):
    from django.contrib.auth import get_user_model
    from rest_framework.test import APIClient

    from brick_dnn_core import training_service

    from .test_mcp import call, mcp_client
    from .test_mcp_compact import long_run

    user = get_user_model().objects.create_user("eve", password="correct-horse-1")
    client = APIClient()
    client.force_authenticate(user)
    mcp = mcp_client(user)

    old = long_run(user, training_dir, epochs=3)  # written without an evaluation
    directory = training_service.run_dir(old.pk)
    state = st.read_state(directory)
    del state["evaluation"]
    st.write_state(directory, state)
    detail = client.get(f"/api/training/runs/{old.pk}/").json()
    assert detail["status"] == "completed" and "evaluation" not in detail["state"]
    result, text = call(mcp, "get_training_run", id=str(old.pk))
    assert "isError" not in result and "evaluation" not in result["structuredContent"]

    new = long_run(user, training_dir, epochs=3)
    directory = training_service.run_dir(new.pk)
    state = st.read_state(directory)
    state["evaluation"] = classification_evaluation(3)
    st.write_state(directory, state)
    detail = client.get(f"/api/training/runs/{new.pk}/").json()
    assert detail["state"]["evaluation"]["val"]["confusion_matrix"]
    result, text = call(mcp, "get_training_run", id=str(new.pk))
    summary = result["structuredContent"]["evaluation"]
    assert summary["epoch"] == 3 and summary["validation"]["confusion_matrix"]


from .test_training_api import training_dir  # noqa: E402, F401 (a fixture)
