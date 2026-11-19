"""Logging to Brick DNN from the generated scripts besides the neural CSV loop (that one is in
test_tracking_client.py): a classical model or an ensemble (one run with its settings and
validation numbers), a hybrid network (the curves of its features, the classical model's
numbers as the result) and the training on a built-in text dataset (curves and the test
accuracy). Each one is inert without BRICKDNN_TOKEN (the same output with logging on and off),
logs to the fake server and into the real API with a token that may only log runs."""

from __future__ import annotations

import re

import pytest

from brickdnn import generate_code

from .helpers import chain
from .test_tracking_client import (
    FakeServer,
    _run,
    no_proxy,  # noqa: F401 (autouse: no proxy, no token)
)

QUICK_BOOSTING = {"max_iter": 30, "min_samples_leaf": 5, "early_stopping": "off"}


def model_graph(model: str, params: dict | None = None) -> dict:
    return chain(("Input", {"shape": [4]}), (model, params or {}))


def ensemble_graph(ensemble: str, params: dict | None = None) -> dict:
    """Input → SVC and gradient boosting → the ensemble (classifiers)."""
    nodes = [
        {"id": "in", "type": "Input", "params": {"shape": [4]}},
        {"id": "svc", "type": "SVC", "params": {}},
        {"id": "gb", "type": "HistGradientBoostingClassifier", "params": QUICK_BOOSTING},
        {"id": "e", "type": ensemble, "params": params or {}},
    ]
    edges = [("in", "svc"), ("in", "gb"), ("svc", "e"), ("gb", "e")]
    return {
        "version": 1,
        "nodes": nodes,
        "edges": [{"source": a, "target": b} for a, b in edges],
    }


def hybrid_graph(model: str = "HistGradientBoostingClassifier", params=None) -> dict:
    """Input → Linear(16) → ReLU → Linear(6) → model."""
    if params is None and model.startswith("HistGradientBoosting"):
        params = QUICK_BOOSTING
    return chain(
        ("Input", {"shape": [4]}),
        ("Linear", {"out_features": 16}),
        ("Activation", {}),
        ("Linear", {"out_features": 6}),
        (model, params or {}),
    )


# Sentiment: 24 token ids of 151 words, two classes.
TEXT_NETWORK = chain(
    ("Input", {"shape": [24], "dtype": "long"}),
    ("Embedding", {"num_embeddings": 151, "embedding_dim": 8}),
    ("SequencePool", {"mode": "mean"}),
    ("Linear", {"out_features": 2}),
)

CLASSICAL = {
    "svc": model_graph("SVC"),
    "boosting": model_graph("HistGradientBoostingRegressor", QUICK_BOOSTING),
    "hard_voting": ensemble_graph("VotingClassifier", {"voting": "hard"}),
    "stacking": ensemble_graph("StackingClassifier", {"cv": 3}),
}
REGRESSION = {"boosting"}


def write_csv(path, regression=False, n=100):
    np = pytest.importorskip("numpy")
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
    return path


def logged(fake) -> tuple[dict, list[dict], dict]:
    """What the script sent to the fake server: the run, its points and its end."""
    assert fake.paths()[0] == "POST /api/tracking/runs/"
    created = fake.requests[0][2]
    points = [p for r in fake.requests if r[1].endswith("/log/") for p in r[2]["points"]]
    finished = [r[2] for r in fake.requests if r[1].endswith("/finish/")]
    assert len(finished) == 1
    return created, points, finished[0]


@pytest.fixture
def fake():
    """The fake Brick DNN of test_tracking_client.py."""
    started = FakeServer()
    yield started
    started.close()


def stable(output: str) -> str:
    """The output of a script without the time of the fit, which varies."""
    return re.sub(r"Fitted in [0-9.]+ s", "Fitted in … s", output)


def fake_env(fake) -> dict:
    return {"BRICKDNN_TOKEN": "bdnn_fake", "BRICKDNN_URL": fake.url}


# --------------------------------------------------------------------------- #
# The code
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "graph",
    [CLASSICAL["svc"], CLASSICAL["stacking"], hybrid_graph(), TEXT_NETWORK],
    ids=["classical", "ensemble", "hybrid", "text"],
)
def test_every_training_script_embeds_the_client_once_but_no_token(graph):
    code = generate_code(graph, {"tracking_project": 7, "tracking_url": "https://my.fake"})
    assert code.count("class BrickDNNRun:") == 1 and "import urllib.request" in code
    assert (
        code.startswith("# Built with Brick DNN") and "from __future__ import annotations" in code
    )
    assert 'BRICKDNN_URL = "https://my.fake"' in code and "BRICKDNN_PROJECT = 7" in code
    assert "export BRICKDNN_TOKEN=<your token>" in code and "default https://my.fake)" in " ".join(
        code.split()
    )
    assert "bdnn_" not in code
    compile(code, "<generated>", "exec")
    off = generate_code(graph, {"log_to_brickdnn": False})
    for word in ("BrickDNNRun", "urllib", "BRICKDNN_TOKEN", "run.finish", "sklearn.metrics"):
        assert word not in off
    # Without the training code there is nothing to log.
    assert "BrickDNNRun" not in generate_code(graph, {"include_training": False})


def test_a_hybrid_network_of_images_has_no_training_to_log():
    images = chain(
        ("Input", {"shape": [1, 8, 8]}),
        ("Flatten", {}),
        ("Linear", {"out_features": 4}),
        ("SVC", {}),
    )
    assert "BrickDNNRun" not in generate_code(images)


# --------------------------------------------------------------------------- #
# Classical models and ensembles
# --------------------------------------------------------------------------- #


def _fit_twice(tmp_path, graph, regression):
    """Fit the script with and without logging (no token): output and model."""
    joblib = pytest.importorskip("joblib")
    np = pytest.importorskip("numpy")
    data = write_csv(tmp_path / "data.csv", regression)
    results = []
    for logging in (True, False):
        script = tmp_path / f"model_{logging}.py"
        script.write_text(generate_code(graph, {"log_to_brickdnn": logging}))
        saved = tmp_path / f"model_{logging}.joblib"
        done = _run(script, "--csv", str(data), "--seed", "3", "--save", str(saved), cwd=tmp_path)
        assert done.returncode == 0, done.stderr
        assert "Brick DNN" not in done.stderr
        features = np.loadtxt(data, delimiter=",", skiprows=1, usecols=range(4))
        predictions = joblib.load(saved)["model"].predict(features)
        results.append((stable(done.stdout).replace(str(saved), "M"), predictions))
    return results


@pytest.mark.parametrize("name", list(CLASSICAL))
def test_a_classical_script_runs_offline_with_the_same_results(tmp_path, name):
    pytest.importorskip("sklearn")
    (on, on_predictions), (off, off_predictions) = _fit_twice(
        tmp_path, CLASSICAL[name], name in REGRESSION
    )
    assert on == off and "Fitted in" in on
    assert list(on_predictions) == list(off_predictions)


@pytest.mark.parametrize("name", list(CLASSICAL))
def test_a_classical_script_logs_its_settings_and_validation_numbers(tmp_path, fake, name):
    pytest.importorskip("sklearn")
    regression = name in REGRESSION
    data = write_csv(tmp_path / "houses.csv", regression)
    script = tmp_path / "model.py"
    script.write_text(generate_code(CLASSICAL[name], {"tracking_project": 3}))
    done = _run(script, "--csv", str(data), "--seed", "2", env=fake_env(fake), cwd=tmp_path)
    assert done.returncode == 0, done.stderr
    assert "Brick DNN: logging this run" in done.stderr and "Saved the fitted model" in done.stdout
    created, points, finished = logged(fake)
    assert points == []  # fitted once: no epochs
    assert created["task"] == ("regression" if regression else "classification")
    assert created["dataset"] == "houses.csv" and created["project"] == 3
    assert "epochs" not in created
    config = created["config"]
    assert config["seed"] == 2 and config["val_split"] == 0.2
    params = config["params"]
    assert params and all(
        value is None or isinstance(value, (bool, int, float, str)) for value in params.values()
    )
    summary = finished["summary"]
    assert finished["status"] == "completed" and summary["fit_seconds"] >= 0
    if regression:
        assert config["model"] == "HistGradientBoostingRegressor"
        assert params["max_iter"] == 30 and params["min_samples_leaf"] == 5
        assert {"val_rmse", "val_mae", "val_r2"} <= set(summary)
        assert summary["val_r2"] > 0.5
    else:
        assert {
            "val_accuracy",
            "val_balanced_accuracy",
            "val_macro_f1",
            "val_weighted_f1",
        } <= set(summary)
        assert 0.5 < summary["val_accuracy"] <= 1
        # Hard voting has no probabilities: no ROC AUC.
        assert ("val_roc_auc" in summary) == (name != "hard_voting")
    if name == "svc":
        assert config["model"] == "SVC" and params["estimator"].startswith("SVC(")
        assert params["estimator__C"] == 1.0
    if name == "stacking":
        assert config["model"] == "StackingClassifier" and params["cv"] == 3


def test_numbers_that_cannot_be_computed_are_left_out_quietly(tmp_path, fake):
    """Validation rows of one class only: no ROC AUC, and no warnings of scikit-learn that
    the script without logging would not print either."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    n, seed = 60, 5
    rng = np.random.default_rng(1)
    x = rng.normal(size=(n, 4))
    labels = np.where(x[:, 0] > 0, "spam", "ham")
    labels[np.random.default_rng(seed).permutation(n)[: n // 5]] = "ham"  # the validation rows
    rows = [",".join([*(f"{v:.5f}" for v in row), label]) for row, label in zip(x, labels)]
    data = tmp_path / "data.csv"
    data.write_text("\n".join(["a,b,c,d,label", *rows]) + "\n")
    graph = model_graph("HistGradientBoostingClassifier", QUICK_BOOSTING)
    script = tmp_path / "model.py"
    script.write_text(generate_code(graph))
    done = _run(script, "--csv", str(data), "--seed", str(seed), env=fake_env(fake), cwd=tmp_path)
    assert done.returncode == 0, done.stderr
    assert "Warning" not in done.stderr and "Saved the fitted model" in done.stdout
    summary = logged(fake)[2]["summary"]
    assert {"val_accuracy", "val_macro_f1", "fit_seconds"} <= set(summary)
    assert "val_roc_auc" not in summary


def test_a_classical_script_keeps_fitting_when_the_server_fails(tmp_path, fake):
    pytest.importorskip("sklearn")
    fake.failures = [500] * 20
    data = write_csv(tmp_path / "data.csv")
    script = tmp_path / "model.py"
    script.write_text(generate_code(CLASSICAL["svc"]))
    done = _run(script, "--csv", str(data), env=fake_env(fake), cwd=tmp_path)
    assert done.returncode == 0, done.stderr
    assert done.stderr.count("could not log") == 1
    assert "Saved the fitted model" in done.stdout


# --------------------------------------------------------------------------- #
# Hybrid networks
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model", ["HistGradientBoostingClassifier", "SVR"])
def test_a_hybrid_script_runs_offline_with_the_same_results(tmp_path, model):
    torch = pytest.importorskip("torch")
    joblib = pytest.importorskip("joblib")
    pytest.importorskip("sklearn")
    regression = model == "SVR"
    data = write_csv(tmp_path / "data.csv", regression)
    outputs, weights, predictions = [], [], []
    for logging in (True, False):
        script = tmp_path / f"hybrid_{logging}.py"
        script.write_text(generate_code(hybrid_graph(model), {"log_to_brickdnn": logging}))
        files = [str(tmp_path / f"features_{logging}.pt"), str(tmp_path / f"m_{logging}.joblib")]
        done = _run(
            script,
            *("--csv", str(data), "--epochs", "3", "--weights", files[0], "--save", files[1]),
            cwd=tmp_path,
        )
        assert done.returncode == 0, done.stderr
        assert "Brick DNN" not in done.stderr
        outputs.append(stable(done.stdout).replace(files[0], "F").replace(files[1], "M"))
        weights.append(torch.load(files[0], weights_only=True)["state_dict"])
        bundle = joblib.load(files[1])
        predicted = _run(
            script, "--predict", str(data), "--weights", files[0], "--save", files[1], cwd=tmp_path
        )
        assert predicted.returncode == 0, predicted.stderr
        predictions.append(predicted.stdout)
        assert bundle["hybrid"]["features"] == 6
    assert outputs[0] == outputs[1] and "Epoch   3/3" in outputs[0]
    assert all(torch.equal(weights[0][k], weights[1][k]) for k in weights[0])
    assert predictions[0] == predictions[1]


@pytest.mark.parametrize("model", ["HistGradientBoostingClassifier", "SVR"])
def test_a_hybrid_script_logs_the_feature_curves_and_the_classical_result(tmp_path, fake, model):
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    regression = model == "SVR"
    data = write_csv(tmp_path / "data.csv", regression)
    script = tmp_path / "hybrid.py"
    script.write_text(generate_code(hybrid_graph(model)))
    done = _run(script, "--csv", str(data), "--epochs", "3", env=fake_env(fake), cwd=tmp_path)
    assert done.returncode == 0, done.stderr
    created, points, finished = logged(fake)
    assert created["epochs"] == 3 and created["dataset"] == "data.csv"
    config = created["config"]
    assert config["feature_training"] == "pretrain" and config["learned_features"] == 6
    assert config["model"] == model and config["optimizer"] == "adam"
    assert config["loss"] == ("mse" if regression else "cross_entropy")
    assert [p["epoch"] for p in points] == [1, 2, 3]
    metric = "val_rmse" if regression else "val_accuracy"
    assert {"train_loss", "val_loss", metric} <= set(points[0])
    summary = finished["summary"]
    assert finished["status"] == "completed" and summary["fit_seconds"] >= 0
    if regression:
        assert {"val_rmse", "val_mae", "val_r2"} <= set(summary)
    else:
        assert {"val_accuracy", "val_balanced_accuracy", "val_macro_f1", "val_roc_auc"} <= set(
            summary
        )


def test_frozen_features_log_no_curves(tmp_path, fake):
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    data = write_csv(tmp_path / "data.csv")
    script = tmp_path / "hybrid.py"
    script.write_text(generate_code(hybrid_graph()))
    done = _run(script, "--csv", str(data), "--frozen", env=fake_env(fake), cwd=tmp_path)
    assert done.returncode == 0, done.stderr
    created, points, finished = logged(fake)
    assert "epochs" not in created and created["config"]["feature_training"] == "frozen"
    assert created["config"]["epochs"] == 0
    assert points == [] and "val_accuracy" in finished["summary"]


# --------------------------------------------------------------------------- #
# The training on a built-in text dataset
# --------------------------------------------------------------------------- #


def test_a_text_script_runs_offline_with_the_same_results(tmp_path):
    torch = pytest.importorskip("torch")
    outputs, weights = [], []
    for logging in (True, False):
        script = tmp_path / f"text_{logging}.py"
        script.write_text(generate_code(TEXT_NETWORK, {"log_to_brickdnn": logging}))
        saved = str(tmp_path / f"text_{logging}.pt")
        done = _run(script, "--train", "--epochs", "1", "--save", saved, cwd=tmp_path)
        assert done.returncode == 0, done.stderr
        assert "Brick DNN" not in done.stderr
        outputs.append(done.stdout.replace(saved, "W"))
        weights.append(torch.load(saved, weights_only=True)["state_dict"])
    assert outputs[0] == outputs[1] and "Test accuracy:" in outputs[0]
    assert all(torch.equal(weights[0][k], weights[1][k]) for k in weights[0])


def test_a_text_script_logs_its_curves_and_the_test_accuracy(tmp_path, fake):
    pytest.importorskip("torch")
    script = tmp_path / "text.py"
    script.write_text(generate_code(TEXT_NETWORK))
    done = _run(
        script,
        *("--train", "--epochs", "2", "--save", str(tmp_path / "t.pt")),
        env=fake_env(fake),
        cwd=tmp_path,
    )
    assert done.returncode == 0, done.stderr
    created, points, finished = logged(fake)
    assert created["dataset"] == "Sentiment" and created["task"] == "classification"
    assert created["epochs"] == 2 and created["config"]["loss"] == "cross_entropy"
    assert [p["epoch"] for p in points] == [1, 2]
    assert {"train_loss", "val_loss", "val_accuracy"} <= set(points[0])
    summary = finished["summary"]
    assert finished["status"] == "completed"
    assert {"test_accuracy", "test_loss", "val_accuracy"} <= set(summary)
    assert 0 <= summary["test_accuracy"] <= 1


# --------------------------------------------------------------------------- #
# Into the real API
# --------------------------------------------------------------------------- #


@pytest.mark.django_db(transaction=True)
@pytest.mark.parametrize("kind", ["classical", "hybrid", "text"])
def test_every_script_logs_into_brick_dnn(tmp_path, live_server, settings, kind):
    """End to end: the generated script, a logging-only token and the real API."""
    pytest.importorskip("sklearn")
    if kind != "classical":
        pytest.importorskip("torch")
    from django.contrib.auth import get_user_model
    from rest_framework.test import APIClient

    from brick_dnn_core.api_tokens import create_token
    from brick_dnn_core.models import Project, TrainingRun

    graph = {"classical": CLASSICAL["svc"], "hybrid": hybrid_graph(), "text": TEXT_NETWORK}[kind]
    settings.TRAINING_DIR = tmp_path / "training"
    user = get_user_model().objects.create_user("ada", email="ada@example.com", password="x")
    project = Project.objects.create(owner=user, name="Mail", graph=graph, updated_by=user)
    _, key = create_token(user, "laptop", scope="runs:log")
    script = tmp_path / "model.py"
    script.write_text(generate_code(graph, {"tracking_project": project.pk}))
    if kind == "text":
        args = ["--train", "--epochs", "2", "--save", str(tmp_path / "t.pt")]
    else:
        args = ["--csv", str(write_csv(tmp_path / "mail.csv")), "--epochs", "2"]
        if kind == "classical":
            args = args[:2]
    done = _run(
        script,
        *args,
        env={"BRICKDNN_TOKEN": key, "BRICKDNN_URL": live_server.url},
        cwd=tmp_path,
    )
    assert done.returncode == 0, done.stderr
    assert "Brick DNN: logging this run" in done.stderr and "could not log" not in done.stderr
    run = TrainingRun.objects.get(kind="external")
    assert run.status == "completed" and run.project_id == project.pk
    assert run.api_token.scope == "runs:log"
    reported = run.plan["external"]["summary"]
    if kind == "text":
        assert run.dataset["name"] == "Sentiment" and run.epochs_done == 2
        assert run.metrics["test_accuracy"] == reported["test_accuracy"]
        result = "test_accuracy"
    else:
        assert run.dataset["name"] == "mail.csv"
        assert run.epochs_done == (0 if kind == "classical" else 2)
        # The classical model's validation accuracy is the run's, not the temporary head's.
        assert run.metrics["val_accuracy"] == reported["val_accuracy"]
        assert reported["fit_seconds"] >= 0 and "val_balanced_accuracy" in reported
        assert run.config["params"] and run.config["model"] in (
            "SVC",
            "HistGradientBoostingClassifier",
        )
        result = "val_balanced_accuracy"
    web = APIClient()
    web.force_authenticate(user)
    row = web.get("/api/training/runs/table/").json()["results"][0]
    assert row["kind"] == "external" and row["metrics"][result] == reported[result]
    detail = web.get(f"/api/training/runs/{run.pk}/").json()
    assert detail["status"] == "completed" and detail["has_weights"] is False


@pytest.mark.django_db
def test_mcp_writes_this_server_into_the_script_and_can_leave_logging_out(settings):
    from django.contrib.auth import get_user_model

    from .test_mcp import call, mcp_client

    settings.PUBLIC_URL = "https://brickdnn.example"
    mcp = mcp_client(get_user_model().objects.create_user("ada", password="x"))
    _, code = call(mcp, "generate_pytorch_code", graph=CLASSICAL["svc"], tracking_project=5)
    assert 'BRICKDNN_URL = "https://brickdnn.example"' in code and "BRICKDNN_PROJECT = 5" in code
    _, off = call(mcp, "generate_pytorch_code", graph=hybrid_graph(), log_to_brickdnn=False)
    assert "BrickDNNRun" not in off and "def learn_features(" in off
