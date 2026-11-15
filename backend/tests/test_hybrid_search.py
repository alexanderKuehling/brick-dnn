"""The hyperparameter search of hybrid networks over the training settings of their neural
layers (also together with the parameters of the classical model): the search settings,
the trials, the stop, the estimate, the runner's state and the checks of the REST API and
the MCP tools. The search of the classical model alone is tested in test_hybrid.py."""

# ruff: noqa: F811 (the fixtures are imported from test_training_api)

from __future__ import annotations

import json

import pytest

pytest.importorskip("sklearn")
torch = pytest.importorskip("torch")

import joblib  # noqa: E402

from brickdnn import analyze  # noqa: E402
from brickdnn.training import (  # noqa: E402
    TrainingConfig,
    check_compatibility,
    inspect_csv,
    runner,
)
from brickdnn.training import state as st  # noqa: E402
from brickdnn.training.classical_engine import final_estimator  # noqa: E402
from brickdnn.training.config import ConfigError  # noqa: E402
from brickdnn.training.estimate import estimate_training  # noqa: E402
from brickdnn.training.hybrid_engine import train  # noqa: E402
from brickdnn.training.job import TrainingJob, search_config  # noqa: E402
from brickdnn.training.tuning import (  # noqa: E402
    TuningConfig,
    describe_params,
    param_label,
    split_params,
)

from .test_classical_api import upload_csv  # noqa: E402
from .test_classical_training import classification_csv, regression_csv  # noqa: E402
from .test_hybrid import HybridRecorder, hybrid_graph  # noqa: E402
from .test_mcp import call, mcp_client  # noqa: E402
from .test_training_api import (  # noqa: E402, F401 (fixtures)
    auth_client,
    training_dir,
    user,
)


def tuning(data, model="SVC"):
    return TuningConfig.from_dict(data, model, hybrid=True)


def run_search(path, search, graph=None, weights=None, reporter=None, **config):
    job = TrainingJob(
        graph=graph or hybrid_graph("SVC"),
        dataset=inspect_csv(path),
        config=TrainingConfig.from_dict({"seed": 1, "epochs": 3, **config}),
        path=str(path),
        tuning=search,
    )
    reporter = reporter or HybridRecorder()
    return train(job, reporter, weights), reporter


# --------------------------------------------------------------------------- #
# The search settings
# --------------------------------------------------------------------------- #


def test_the_neural_settings_go_into_neural_space():
    search = tuning(
        {
            "method": "grid",
            "neural_space": {"learning_rate": [0.001, 0.01], "optimizer": ["adam", "sgd"]},
            "space": {"C": [1, 10]},
        }
    )
    assert search.searches_neural and search.trials == 8 and search.cv_folds is None
    params = search.sample(0)
    assert params[0] == {"neural.learning_rate": 0.001, "neural.optimizer": "adam", "C": 1.0}
    assert split_params(params[0]) == ({"learning_rate": 0.001, "optimizer": "adam"}, {"C": 1.0})
    assert describe_params(params[0], "SVC") == ("neural layers: learning rate 0.001, Adam; C 1")
    assert param_label("neural.batch_size") == "batch size of the neural layers"
    # The settings survive job.json (the runner reads them again).
    data = search.to_dict()
    assert data["neural_space"] == {"learning_rate": [0.001, 0.01], "optimizer": ["adam", "sgd"]}
    assert data["space"] == {"C": [1.0, 10.0]} and data["cv_folds"] is None
    assert search_config(data, hybrid_graph("SVC")) == search
    # Only the neural settings: the classical model keeps its brick's parameters.
    alone = tuning({"neural_space": {"learning_rate": {"min": 1e-4, "max": 1e-2}}, "trials": 4})
    assert alone.space == {} and alone.trials == 4
    assert all(set(p) == {"neural.learning_rate"} for p in alone.sample(3))
    assert search_config(alone.to_dict(), hybrid_graph("SVC")).space == {}
    # Without neural_space only the classical model is searched, as before.
    assert tuning({}).neural_space is None and set(tuning({}).space) == {"C", "gamma"}


@pytest.mark.parametrize(
    "data,model,hybrid,message",
    [
        (
            {"neural_space": {"learning_rate": [0.01]}, "cv_folds": 3},
            "SVC",
            True,
            "cross-validation (cv_folds) is not available when the neural settings are searched",
        ),
        (
            {"space": {"batch_size": [32, 64]}},
            "SVC",
            True,
            "'batch_size' is a training setting of the neural layers: put it into neural_space",
        ),
        (
            {"neural_space": {"learning_rate": [0.01]}},
            "SVC",
            False,
            "neural_space is for hybrid networks",
        ),
        (
            {"neural_space": {"learning_rate": [0.01]}},
            None,
            False,
            "neural_space is for hybrid networks",
        ),
        (
            {"neural_space": {"C": [1]}},
            "SVC",
            True,
            "'C' cannot be tuned. Possible: learning_rate, batch_size",
        ),
        (
            {"method": "grid", "neural_space": {"learning_rate": {"min": 0.001, "max": 0.1}}},
            "SVC",
            True,
            "Grid search needs a list of values for the learning rate of the neural layers",
        ),
        (
            {
                "method": "grid",
                "neural_space": {"batch_size": list(range(1, 21))},
                "space": {"C": [1, 2, 3, 4, 5, 6]},
            },
            "SVC",
            True,
            "The grid has 120 combinations, at most 100",
        ),
    ],
)
def test_search_settings_that_are_refused(data, model, hybrid, message):
    with pytest.raises(ConfigError) as error:
        TuningConfig.from_dict(data, model, hybrid=hybrid)
    assert message in str(error.value)


def test_the_learning_rate_of_gradient_boosting_stays_in_space():
    search = tuning(
        {
            "method": "grid",
            "neural_space": {"learning_rate": [0.001]},
            "space": {"learning_rate": [0.1, 0.3]},
        },
        "HistGradientBoostingClassifier",
    )
    assert [p for p in search.sample(0)] == [
        {"neural.learning_rate": 0.001, "learning_rate": 0.1},
        {"neural.learning_rate": 0.001, "learning_rate": 0.3},
    ]


# --------------------------------------------------------------------------- #
# The trials
# --------------------------------------------------------------------------- #


def test_every_trial_learns_the_features_and_fits_the_model(tmp_path):
    path = classification_csv(tmp_path / "mail.csv", n=150)
    search = tuning(
        {
            "method": "grid",
            "neural_space": {"learning_rate": [0.0001, 0.02]},
            "space": {"C": [0.5, 20]},
            "metric": "accuracy",
        }
    )
    result, reporter = run_search(path, search, weights=tmp_path / "m.joblib", test_split=0.2)
    assert result.status == "completed", result.message
    assert [t["params"] for t in reporter.trials] == [
        {"neural.learning_rate": 0.0001, "C": 0.5},
        {"neural.learning_rate": 0.0001, "C": 20.0},
        {"neural.learning_rate": 0.02, "C": 0.5},
        {"neural.learning_rate": 0.02, "C": 20.0},
    ]
    assert len(result.trials) == 4 and result.best_trial in (1, 2, 3, 4)
    best = next(t for t in result.trials if t["trial"] == result.best_trial)
    assert best["score"] == max(t["score"] for t in result.trials)
    # Each trial learned its features for 3 epochs and is ranked by the classical model.
    assert all(t["epochs_done"] == 3 and t["status"] == "completed" for t in result.trials)
    assert all(t["folds"] is None and t["score"] == t["val_accuracy"] for t in result.trials)
    # The run keeps the best trial: its curves, its features and its fitted model, which is
    # measured on the validation rows like in the trial, then tested and evaluated.
    assert len(result.history) == 3
    classical = reporter.info["classical"]
    assert classical["validation"]["accuracy"] == pytest.approx(best["val_accuracy"])
    assert reporter.test is not None and reporter.evaluation["trial"] == result.best_trial
    assert reporter.info["tuning"]["neural_space"] == {"learning_rate": [0.0001, 0.02]}
    assert reporter.info["trials_total"] == 4
    assert result.message.startswith(f"Finished 4 trials. Best: trial {result.best_trial} (")
    assert "), validation accuracy " in result.message
    assert "6 features learned in 3 epochs; SVC on them: validation accuracy" in result.message
    assert any(
        "every trial learns 6 features and fits SVC on them" in line for line in reporter.logs
    )
    assert ("running", "Trial 2 of 4: epoch 1 of 3") in reporter.statuses
    assert ("running", "Trial 2 of 4: fitting SVC …") in reporter.statuses
    # The saved files are those of the best trial.
    bundle = joblib.load(tmp_path / "m.joblib")
    assert best["params"]["C"] == final_estimator(bundle["model"]).C
    checkpoint = torch.load(tmp_path / "weights.pt", weights_only=True)
    assert "linear_1.weight" in checkpoint["state_dict"]


def test_the_neural_settings_alone_and_a_regression(tmp_path):
    path = regression_csv(tmp_path / "houses.csv", n=120)
    search = tuning(
        {"method": "grid", "neural_space": {"batch_size": [8, 64]}, "metric": "rmse"}, "SVR"
    )
    result, reporter = run_search(
        path, search, graph=hybrid_graph("SVR", {"C": 10.0}), weights=tmp_path / "m.joblib"
    )
    assert result.status == "completed", result.message
    assert [t["params"] for t in result.trials] == [
        {"neural.batch_size": 8},
        {"neural.batch_size": 64},
    ]
    assert all(t["val_rmse"] is not None and t["score"] == t["val_rmse"] for t in result.trials)
    best = next(t for t in result.trials if t["trial"] == result.best_trial)
    assert best["score"] == min(t["score"] for t in result.trials)
    assert reporter.info["classical"]["validation"]["rmse"] == pytest.approx(best["val_rmse"])
    assert "Best: trial" in result.message and "(neural layers: batch size " in result.message


def test_a_search_of_the_neural_settings_needs_validation_rows_and_learned_features(tmp_path):
    path = classification_csv(tmp_path / "mail.csv")
    search = tuning({"neural_space": {"learning_rate": [0.001, 0.01]}})
    result, _ = run_search(path, search, val_split=0)
    assert result.status == "failed"
    assert result.message.startswith(
        "Searching the training settings of the neural layers needs validation rows"
    )
    result, _ = run_search(path, search, feature_training="frozen")
    assert result.status == "failed"
    assert result.message.startswith("Frozen features are not trained")


def test_a_failed_trial_does_not_end_the_search(tmp_path):
    path = classification_csv(tmp_path / "mail.csv")
    # Batch normalization cannot train on batches of a single row.
    graph = hybrid_graph("SVC")
    graph["nodes"].insert(2, {"id": "bn", "type": "BatchNorm1d", "params": {}})
    graph["edges"][1:2] = [{"source": "fc", "target": "bn"}, {"source": "bn", "target": "act"}]
    search = tuning({"method": "grid", "neural_space": {"batch_size": [1, 16]}})
    result, reporter = run_search(path, search, graph=graph)
    assert result.status == "completed", result.message
    assert [t["status"] for t in result.trials] == ["failed", "completed"]
    assert "Batch normalization needs at least 2 samples" in result.trials[0]["message"]
    assert result.best_trial == 2 and "Finished 2 trials (1 failed)" in result.message
    assert any(line.startswith("Trial 1 failed: ") for line in reporter.logs)


class StopInTrial(HybridRecorder):
    """Asks to stop once trial ``trial`` has finished ``epochs`` epochs."""

    def __init__(self, trial, epochs):
        super().__init__()
        self.trial, self.epochs = trial, epochs
        self.started = 0
        self.done = 0

    def trial_start(self, trial):
        self.started = trial["trial"]
        self.done = 0

    def epoch_end(self, entry, best):
        super().epoch_end(entry, best)
        self.done += 1

    def should_stop(self):
        return self.started == self.trial and self.done >= self.epochs


@pytest.fixture
def _check_stop_every_batch(monkeypatch):
    from brickdnn.training import engine

    monkeypatch.setattr(engine, "STOP_CHECK_INTERVAL", 0.0)


@pytest.mark.usefixtures("_check_stop_every_batch")
def test_stopping_keeps_the_best_trial_so_far(tmp_path):
    path = classification_csv(tmp_path / "mail.csv")
    search = tuning(
        {"method": "grid", "neural_space": {"learning_rate": [0.01, 0.001]}, "space": {"C": [2]}}
    )
    reporter = StopInTrial(2, 1)
    result, _ = run_search(path, search, weights=tmp_path / "m.joblib", reporter=reporter)
    assert result.status == "stopped" and result.weights is True, result.message
    assert result.message.startswith("Stopped after 1 of 2 trials. Best: trial 1 (")
    assert "Kept the 6 features learned so far with SVC fitted on them (not tested)" in (
        result.message
    )
    assert [t["status"] for t in result.trials] == ["completed", "stopped"]
    assert result.best_trial == 1 and len(result.history) == 3
    assert reporter.test is None and reporter.evaluation is None
    assert (tmp_path / "m.joblib").is_file() and (tmp_path / "weights.pt").is_file()


@pytest.mark.usefixtures("_check_stop_every_batch")
def test_stopping_during_the_first_trial_keeps_what_it_learned(tmp_path):
    path = classification_csv(tmp_path / "mail.csv")
    search = tuning({"method": "grid", "neural_space": {"learning_rate": [0.01, 0.001]}})
    reporter = StopInTrial(1, 2)
    result, _ = run_search(
        path, search, weights=tmp_path / "m.joblib", reporter=reporter, epochs=20
    )
    assert result.status == "stopped" and result.weights is True, result.message
    assert result.message.startswith(
        "Stopped during trial 1 of 2 (neural layers: learning rate 0.01): Stopped after 2 of 20 "
        "epochs."
    )
    assert result.best_trial is None and [t["status"] for t in result.trials] == ["stopped"]
    assert len(result.history) == 2


# --------------------------------------------------------------------------- #
# The estimate, the runner and the checks
# --------------------------------------------------------------------------- #


def test_the_estimate_multiplies_both_phases(tmp_path):
    info = inspect_csv(classification_csv(tmp_path / "mail.csv", n=900))
    analysis = analyze(hybrid_graph())
    plan = check_compatibility(analysis, info).plan.to_dict()
    config = TrainingConfig.from_dict({"epochs": 20})
    single = estimate_training(analysis, info, config, plan).details
    classical = estimate_training(analysis, info, config, plan, trials=5).details
    neural = estimate_training(analysis, info, config, plan, trials=5, neural_trials=True)
    assert classical["neural_cpu_seconds"] == single["neural_cpu_seconds"]
    assert neural.details["neural_trials"] == 5 and classical["neural_trials"] == 1
    startup = 3.0
    assert neural.details["neural_cpu_seconds"] - startup == pytest.approx(
        5 * (single["neural_cpu_seconds"] - startup),
        abs=0.3,  # rounded to 0.1 s
    )
    assert neural.details["classical_cpu_seconds"] == classical["classical_cpu_seconds"]
    assert "20 epochs × 5 trials" in neural.basis


def test_the_runner_writes_the_trials_of_a_neural_search(tmp_path):
    path = classification_csv(tmp_path / "mail.csv")
    info = inspect_csv(path)
    run_dir = tmp_path / "run"
    search = tuning({"method": "grid", "neural_space": {"batch_size": [16, 32]}})
    st.write_job(
        run_dir,
        {
            "graph": hybrid_graph("SVC"),
            "dataset": {"info": info.to_dict(), "path": str(path)},
            "config": {"seed": 1, "epochs": 2},
            "tuning": search.to_dict(),
        },
    )
    state = runner.run(run_dir)
    assert state["status"] == "completed", state["message"]
    assert state["family"] == "hybrid" and state["weights"] is True
    assert state["trials_total"] == 2 and state["trial"] is None
    assert [t["params"] for t in state["trials"]] == [
        {"neural.batch_size": 16},
        {"neural.batch_size": 32},
    ]
    assert state["best_trial"] in (1, 2) and len(state["history"]) == 2
    assert state["tuning"]["neural_space"] == {"batch_size": [16, 32]}
    assert state["evaluation"]["trial"] == state["best_trial"]
    assert state["classical"]["hybrid"]["epochs"] == 2


@pytest.mark.django_db
def test_the_api_and_mcp_check_a_search_of_the_neural_settings(auth_client, user, tmp_path):
    dataset = upload_csv(auth_client, classification_csv(tmp_path / "mail.csv"))
    graph = hybrid_graph("SVC")
    search = {
        "method": "grid",
        "neural_space": {"learning_rate": [0.001, 0.01]},
        "space": {"C": [1, 10]},
    }

    def check(tuning, config=None):
        return auth_client.post(
            "/api/training/check/",
            {
                "graph": graph,
                "dataset": dataset,
                "config": config or {"epochs": 5},
                "tuning": tuning,
            },
            format="json",
        )

    response = check(search)
    assert response.status_code == 200 and response.data["ok"] is True, response.data
    details = response.data["estimate"]["details"]
    assert details["family"] == "hybrid" and details["neural_trials"] == 4
    response = check(search, {"epochs": 5, "val_split": 0})
    assert response.data["ok"] is False
    assert "needs validation rows" in response.data["errors"][0]
    response = check(search, {"epochs": 5, "feature_training": "frozen"})
    assert response.data["errors"][0].startswith("Frozen features are not trained")
    response = check({**search, "cv_folds": 3})
    assert response.status_code == 400
    assert "cv_folds) is not available when the neural settings" in response.data["detail"]
    # A network has no neural_space: its space holds the training settings.
    network = {
        "version": 1,
        "nodes": [
            {"id": "in", "type": "Input", "params": {"shape": [4]}},
            {"id": "fc", "type": "Linear", "params": {"out_features": 3}},
        ],
        "edges": [{"source": "in", "target": "fc"}],
    }
    response = auth_client.post(
        "/api/training/check/",
        {"graph": network, "dataset": dataset, "tuning": search},
        format="json",
    )
    assert response.status_code == 400
    assert "neural_space is for hybrid networks" in response.data["detail"]

    mcp = mcp_client(user)
    checked, text = call(
        mcp, "check_training", graph=graph, dataset=dataset, config={"epochs": 5}, tuning=search
    )
    assert checked["structuredContent"]["ok"] is True, text
    checked, text = call(
        mcp,
        "check_training",
        graph=graph,
        dataset=dataset,
        config={"epochs": 5},
        tuning={**search, "cv_folds": 3},
    )
    assert "is not available when the neural settings are searched" in text


def test_the_command_line_searches_both_parts(tmp_path, capsys):
    from brickdnn.training.cli import main

    path = classification_csv(tmp_path / "mail.csv")
    network = tmp_path / "net.json"
    network.write_text(json.dumps(hybrid_graph("SVC")))
    arguments = [str(network), "--data", str(path), "--epochs", "2", "--seed", "1"]
    code = main(
        [
            *arguments,
            "--search",
            "grid",
            "--tune-batch-size",
            "16,32",
            "--tune-param",
            "C=1,10",
            "--save",
            str(tmp_path / "net.joblib"),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0, out
    assert "Hyperparameter search: grid search, 4 trials of 2 epochs each." in out
    assert "Trial 4/4: neural layers: batch size 32; C 10" in out
    assert "batch size of the neural layers" in out  # the column of the table
    assert (tmp_path / "net.joblib").is_file() and (tmp_path / "net.features.pt").is_file()
    # The neural options alone, and cross-validation is refused with them.
    assert main([*arguments, "--tune-lr", "0.001,0.01", "--cv-folds", "3"]) == 1
    assert "cv_folds) is not available" in capsys.readouterr().err
