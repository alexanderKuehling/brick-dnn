"""Hyperparameter search for classical models: the search space of each model, the
objective metrics, the search itself (validation rows and cross-validation), stopping,
the estimate, the runner and the command line. PyTorch is not needed."""

from __future__ import annotations

import subprocess
import sys

import pytest

pytest.importorskip("sklearn")

import joblib  # noqa: E402

from brickdnn.graph import analyze  # noqa: E402
from brickdnn.training import (  # noqa: E402  # noqa: E402
    classical_search,
    cli,
    inspect_csv,
    objectives,
    random_info,
)
from brickdnn.training import state as st
from brickdnn.training.classical_tuning import (  # noqa: E402
    TUNABLES,
    brick_params,
    capabilities,
    parse_space,
    search_folds,
    trial_graph,
)
from brickdnn.training.compat import check_compatibility  # noqa: E402
from brickdnn.training.config import ConfigError, TrainingConfig  # noqa: E402
from brickdnn.training.estimate import estimate_training  # noqa: E402
from brickdnn.training.tuning import Range, TuningConfig  # noqa: E402

from .test_classical_training import (  # noqa: E402
    BACKEND,
    Recorder,
    classification_csv,
    model_graph,
    regression_csv,
    run_job,
)

MODELS = list(TUNABLES)


class SearchRecorder(Recorder):
    def __init__(self, stop_after_trials=None):
        super().__init__()
        self.trials, self.search, self.evaluation = [], None, None
        self.stop_after = stop_after_trials

    def trial_start(self, trial):
        pass

    def trial_end(self, trial, best_trial):
        self.trials.append((trial, best_trial))

    def search_end(self, result):
        self.search = result

    def evaluation_end(self, result):
        self.evaluation = result

    def should_stop(self):
        done = [t for t, _ in self.trials if t["status"] in ("completed", "failed")]
        return self.stop_after is not None and len(done) >= self.stop_after


def tuning(model, **data):
    return TuningConfig.from_dict(data, model)


def binary_csv(path, n=200):
    return classification_csv(path, n=n, labels=("no", "yes", "yes"))


# --------------------------------------------------------------------------- #
# The search space
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model", MODELS)
def test_every_model_has_a_valid_default_space(model):
    search = tuning(model)
    assert search.model == model and search.trials >= 1 and search.metric == "loss"
    assert search.to_dict()["cv_folds"] is None
    assert TuningConfig.from_dict(search.to_dict(), model) == search
    info = capabilities()[model]
    names = [t["name"] for t in info["tunable"]]
    assert set(info["default"]) <= set(names)
    assert parse_space(model, info["default"]) == search.space
    # Every tunable parameter is a parameter of the brick (gamma: gamma and gamma_value).
    for item in TUNABLES[model]:
        values = item.default if isinstance(item.default, list) else [item.default["min"]]
        graph = trial_graph(model_graph(model), "m", {item.name: values[-1]})
        assert analyze(graph).valid, (model, item.name)


def test_ranges_lists_and_scales():
    space = parse_space(
        "HistGradientBoostingClassifier",
        {
            "learning_rate": {"min": 0.01, "max": 0.3},
            "min_samples_leaf": {"min": "5", "max": 100, "scale": "linear"},
            "max_depth": [None, "none", 3, 6.0],
            "l2_regularization": [0, 0.1],
            "class_weight": ["balanced", "none", "balanced"],
        },
    )
    assert space["learning_rate"] == Range(0.01, 0.3)
    assert space["min_samples_leaf"] == Range(5, 100, log=False, integer=True)
    assert space["max_depth"] == (None, 3, 6)
    assert space["class_weight"] == ("balanced", "none")
    search = tuning(
        "HistGradientBoostingClassifier",
        trials=30,
        space={"learning_rate": {"min": 0.01, "max": 0.3}, "max_leaf_nodes": {"min": 8, "max": 64}},
    )
    trials = search.sample(seed=3)
    assert trials == search.sample(seed=3) and len(trials) == 30
    for params in trials:
        assert 0.01 <= params["learning_rate"] <= 0.3
        assert isinstance(params["max_leaf_nodes"], int) and 8 <= params["max_leaf_nodes"] <= 64
    assert search.to_dict()["space"]["max_leaf_nodes"] == {"min": 8, "max": 64}
    linear = tuning("SVR", space={"epsilon": {"min": 0, "max": 1, "scale": "linear"}})
    assert linear.to_dict()["space"]["epsilon"] == {"min": 0, "max": 1, "scale": "linear"}


def test_gamma_takes_numbers_and_names():
    search = tuning("SVC", method="grid", space={"gamma": ["scale", 0.1, "auto"], "C": [1]})
    assert search.trials == 3
    assert brick_params({"gamma": 0.1, "C": 1}) == {"gamma": "value", "gamma_value": 0.1, "C": 1}
    assert brick_params({"gamma": "auto"}) == {"gamma": "auto"}
    graph = trial_graph(model_graph("SVC"), "m", {"gamma": 0.1, "kernel": "poly"})
    params = analyze(graph).nodes["m"].layer.p
    assert (params["gamma"], params["gamma_value"], params["kernel"]) == ("value", 0.1, "poly")


@pytest.mark.parametrize(
    "model,data,message",
    [
        ("SVC", {"space": {"epsilon": [0.1]}}, "'epsilon' cannot be tuned for support vector"),
        ("SVC", {"space": {"learning_rate": [0.1]}}, "Possible: C, gamma, kernel, class_weight"),
        ("SVC", {"space": {"C": {"min": 10, "max": 1}}}, "smallest C must not be larger"),
        ("SVC", {"space": {"C": {"min": 0, "max": 1}}}, "C range must be greater than 0"),
        ("SVC", {"space": {"C": {"min": 1}}}, 'needs "min" and "max"'),
        ("SVC", {"space": {"C": {"min": 1, "max": 2, "scale": "cubic"}}}, '"log" or "linear"'),
        (
            "SVC",
            {"space": {"kernel": {"min": 1, "max": 2}}},
            "give a list of values for the kernel",
        ),
        ("SVC", {"space": {"kernel": ["rbf", "cubic"]}}, "kernel must be one of: linear, poly"),
        ("SVC", {"space": {"gamma": ["fast"]}}, "'fast' is not a number"),
        ("SVC", {"space": {}}, "choose at least one parameter to tune"),
        ("SVR", {"space": {"epsilon": {"min": 0, "max": 1}}}, "log range of the epsilon"),
        (
            "HistGradientBoostingRegressor",
            {"space": {"max_iter": [10.5]}},
            "max. iterations must be whole numbers",
        ),
        (
            "HistGradientBoostingRegressor",
            {"space": {"class_weight": ["balanced"]}},
            "'class_weight' cannot be tuned",
        ),
        ("GaussianProcessClassifier", {"space": {"alpha": [1e-5]}}, "Possible: n_restarts"),
        ("GaussianProcessRegressor", {"space": {"normalize_y": ["maybe"]}}, "true or false"),
        (
            "SVC",
            {"method": "grid", "space": {"C": {"min": 1, "max": 2}}},
            "Grid search needs a list of values for the C, not a range",
        ),
        (
            "HistGradientBoostingClassifier",
            {
                "method": "grid",
                "space": {"max_iter": list(range(1, 12)), "max_depth": list(range(1, 11))},
            },
            "The grid has 110 combinations, at most 100",
        ),
        ("SVC", {"metric": "f2"}, "the metric 'f2' is unknown"),
        ("SVC", {"cv_folds": 1}, "folds must be at least 2 and at most 10"),
        ("SVC", {"cv_folds": 11}, "folds must be at least 2 and at most 10"),
    ],
)
def test_space_errors(model, data, message):
    with pytest.raises(ConfigError) as error:
        TuningConfig.from_dict(data, model)
    assert message in str(error.value)


def test_cross_validation_is_for_classical_models_only():
    with pytest.raises(ConfigError, match="only available for classical models"):
        TuningConfig.from_dict({"cv_folds": 5})
    assert TuningConfig.from_dict({"cv_folds": "3"}, "SVC").cv_folds == 3


def test_folds_without_validation_rows():
    search = tuning("SVC")
    info = random_info()
    assert search_folds(search, info, TrainingConfig()) is None
    assert search_folds(search, info, TrainingConfig(val_split=0)) == 5
    assert search_folds(tuning("SVC", cv_folds=3), info, TrainingConfig()) == 3


# --------------------------------------------------------------------------- #
# Objective metrics
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "metric,maximize",
    [
        ("loss", False),
        ("accuracy", True),
        ("balanced_accuracy", True),
        ("macro_f1", True),
        ("roc_auc", True),
        ("pr_auc", True),
        ("rmse", False),
        ("mae", False),
        ("r2", True),
    ],
)
def test_metric_directions(metric, maximize):
    assert objectives.METRICS[metric].maximize is maximize
    assert objectives.better(0.9, 0.5, metric) is maximize
    assert objectives.better(0.5, 0.9, metric) is not maximize
    assert objectives.better(None, 0.5, metric) is False
    assert objectives.better(float("nan"), 0.5, metric) is False
    assert objectives.better(0.5, None, metric) is True


def test_metrics_per_task(tmp_path):
    path = binary_csv(tmp_path / "mail.csv")
    plan = check_compatibility(model_graph("SVC"), inspect_csv(path)).plan
    assert objectives.evaluation_task(plan) == "classification"
    assert objectives.metric_error("pr_auc", plan) is None
    binary = {"targets": [{"kind": "class", "num_classes": 2}], "num_classes": 2}
    three = {"targets": [{"kind": "class", "num_classes": 3}], "num_classes": 3}
    value = {"targets": [{"kind": "value", "shape": [1]}], "num_classes": None}
    masks = {"targets": [{"kind": "mask", "num_classes": 2}], "num_classes": 2}
    assert objectives.available(binary) == [
        "loss",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "roc_auc",
        "pr_auc",
    ]
    assert "roc_auc" not in objectives.available(three)
    assert objectives.available(value) == ["loss", "rmse", "mae", "r2"]
    assert objectives.available(masks) == ["loss"]
    assert "needs two classes, this data has 3" in objectives.metric_error("roc_auc", three)
    assert "is for regression, and this is a classification" in objectives.metric_error(
        "rmse", binary
    )
    assert "single class or value output" in objectives.metric_error("accuracy", masks)
    assert objectives.metric_error("loss", masks) is None
    assert objectives.parse_metric(None) == "loss" == objectives.parse_metric("log loss")
    assert objectives.parse_metric("ROC-AUC") == "roc_auc"


# --------------------------------------------------------------------------- #
# The search
# --------------------------------------------------------------------------- #


def test_random_search_of_a_classifier_on_validation_rows(tmp_path):
    path = binary_csv(tmp_path / "mail.csv")
    search = tuning(
        "HistGradientBoostingClassifier",
        trials=4,
        metric="roc_auc",
        space={"learning_rate": {"min": 0.01, "max": 0.5}, "class_weight": ["none", "balanced"]},
    )
    rec = SearchRecorder()
    result, _ = run_job(
        model_graph("HistGradientBoostingClassifier"),
        inspect_csv(path),
        tmp_path / "run",
        path,
        reporter=rec,
        tuning=search,
        test_split=0.2,
    )
    assert result.status == "completed", result.message
    assert len(result.trials) == 4 and all(t["status"] == "completed" for t in result.trials)
    assert all(t["folds"] is None and 0 <= t["score"] <= 1 for t in result.trials)
    best = max(result.trials, key=lambda t: (t["score"], -t["trial"]))
    assert result.best_trial == best["trial"] and rec.search["best_trial"] == best["trial"]
    assert rec.info["tuning"]["metric"] == "roc_auc" and rec.info["trials_total"] == 4
    assert "ranked by ROC AUC on the 40 validation rows" in rec.logs[0]
    assert result.message.startswith(f"Finished 4 trials. Best: trial {best['trial']} (")
    assert "validation ROC AUC" in result.message and "test accuracy" in result.message
    # The best trial is kept, measured and evaluated like a normal fit.
    assert rec.evaluation["trial"] == best["trial"] and rec.test is not None
    curve = rec.evaluation["val"]["threshold_curve"]
    assert curve["roc_auc"] == pytest.approx(best["score"], abs=1e-6)
    bundle = joblib.load(tmp_path / "run" / "model.joblib")
    model = bundle["model"].named_steps["model"]
    assert model.learning_rate == best["params"]["learning_rate"]
    weight = best["params"]["class_weight"]
    assert model.class_weight == (None if weight == "none" else weight)
    assert len(result.history) == 1 and rec.kept == {"weights_epoch": 1, "stopped_early": False}


def test_grid_search_of_a_regressor(tmp_path):
    path = regression_csv(tmp_path / "houses.csv", n=120)
    search = tuning(
        "SVR",
        method="grid",
        metric="mae",
        space={"C": [0.1, 10], "gamma": ["scale", 0.05], "kernel": ["rbf", "linear"]},
    )
    rec = SearchRecorder()
    result, _ = run_job(
        model_graph("SVR"), inspect_csv(path), tmp_path / "run", path, reporter=rec, tuning=search
    )
    assert result.status == "completed", result.message
    assert [t["params"] for t in result.trials][:2] == [
        {"C": 0.1, "gamma": "scale", "kernel": "rbf"},
        {"C": 0.1, "gamma": "scale", "kernel": "linear"},
    ]
    assert len(result.trials) == 8
    best = min(result.trials, key=lambda t: (t["score"], t["trial"]))
    assert result.best_trial == best["trial"]
    assert rec.evaluation["val"]["mae"] == pytest.approx(best["score"])
    assert all(t["val_rmse"] is not None and t["val_r2"] is not None for t in result.trials)
    svr = joblib.load(tmp_path / "run" / "model.joblib")["model"].named_steps["model"]
    gamma = best["params"]["gamma"]
    assert (svr.C, svr.kernel, svr.gamma) == (best["params"]["C"], best["params"]["kernel"], gamma)


def test_cross_validation_without_validation_rows(tmp_path):
    path = classification_csv(tmp_path / "mail.csv", n=150)
    search = tuning("SVC", trials=3, metric="balanced_accuracy")
    rec = SearchRecorder()
    result, _ = run_job(
        model_graph("SVC"),
        inspect_csv(path),
        tmp_path / "run",
        path,
        reporter=rec,
        tuning=search,
        val_split=0,
        test_split=0.2,
    )
    assert result.status == "completed", result.message
    assert all(t["folds"] == 5 and t["score"] is not None for t in result.trials)
    assert rec.info["tuning"]["cv_folds"] == 5
    assert (
        "5-fold cross-validation of the 120 training rows (there are no validation rows)"
        in (rec.logs[0])
    )
    folds = [m for _, m in rec.statuses if "fold" in m]
    assert folds[:5] == [f"Trial 1 of 3: fold {k} of 5 …" for k in range(1, 6)]
    best = result.trials[result.best_trial - 1]
    assert any(m.startswith(f"Fitting the best trial ({best['trial']}:") for _, m in rec.statuses)
    assert "Fitted the best trial on all 120 training rows" in " ".join(rec.logs)
    assert "cross-validated balanced accuracy" in result.message
    assert result.history[0]["val_loss"] is None and rec.test is not None
    assert rec.evaluation["test"] is not None and rec.evaluation["trial"] == best["trial"]
    svc = joblib.load(tmp_path / "run" / "model.joblib")["model"].named_steps["model"]
    assert best["params"]["C"] == svc.estimator.C


def test_chosen_cross_validation_keeps_the_validation_rows_unseen(tmp_path):
    path = regression_csv(tmp_path / "houses.csv", n=100)
    search = tuning("GaussianProcessRegressor", trials=2, cv_folds=3, metric="r2")
    rec = SearchRecorder()
    result, _ = run_job(
        model_graph("GaussianProcessRegressor"),
        inspect_csv(path),
        tmp_path / "run",
        path,
        reporter=rec,
        tuning=search,
    )
    assert result.status == "completed", result.message
    assert all(t["folds"] == 3 for t in result.trials)
    assert "3-fold cross-validation of the 80 training rows." in rec.logs[0]
    # The validation rows measure the refitted best trial only.
    assert result.history[0]["val_r2"] is not None
    assert rec.evaluation["val"]["r2"] == pytest.approx(result.history[0]["val_r2"])


def test_failed_trials_do_not_end_the_search(tmp_path, monkeypatch):
    real = classical_search.build_pipeline

    def picky(graph):
        params = next(n for n in graph["nodes"] if n["id"] == "m")["params"]
        if params.get("C") == 0.5:
            raise ValueError("C 0.5 does not work here")
        return real(graph)

    monkeypatch.setattr(classical_search, "build_pipeline", picky)
    search = tuning("SVR", method="grid", space={"C": [0.5, 2]})
    rec = SearchRecorder()
    result, _ = run_job(model_graph("SVR"), random_info(), tmp_path, reporter=rec, tuning=search)
    assert result.status == "completed", result.message
    first, second = result.trials
    assert first["status"] == "failed" and "C 0.5 does not work here" in first["message"]
    assert second["status"] == "completed" and result.best_trial == 2
    assert result.message.startswith("Finished 2 trials (1 failed). Best: trial 2 (C 2)")
    assert "Trial 1 failed: Fitting failed: ValueError: C 0.5 does not work here" in rec.logs

    search = tuning("SVR", method="grid", space={"C": [0.5]})
    result, _ = run_job(model_graph("SVR"), random_info(), tmp_path, tuning=search)
    assert result.status == "failed"
    assert result.message.startswith("All 1 trials failed. Trial 1: Fitting failed")


def test_metric_must_fit_the_task(tmp_path):
    search = tuning("SVC", metric="roc_auc")  # the demo data has three classes
    result, _ = run_job(model_graph("SVC"), random_info(), tmp_path, tuning=search)
    assert result.status == "failed" and "needs two classes, this data has 3" in result.message
    search = tuning("SVR", metric="accuracy")
    result, _ = run_job(model_graph("SVR"), random_info(), tmp_path, tuning=search)
    assert result.status == "failed" and "is for classification" in result.message


@pytest.mark.parametrize("folds", [None, 3])
def test_stopping_keeps_the_best_trial_so_far(tmp_path, folds):
    search = tuning("SVC", trials=5, cv_folds=folds)
    rec = SearchRecorder(stop_after_trials=2)
    result, _ = run_job(model_graph("SVC"), random_info(), tmp_path, reporter=rec, tuning=search)
    assert result.status == "stopped", result.message
    assert [t["status"] for t in result.trials] == ["completed", "completed"]
    best = result.trials[result.best_trial - 1]
    assert result.message.startswith(f"Stopped after 2 of 5 trials. Best: trial {best['trial']}")
    assert result.message.endswith("its model was kept (not tested).")
    assert rec.finished == ("stopped", result.message, True) and result.weights
    assert rec.search["best_trial"] == best["trial"] and len(rec.search["history"]) == 1
    assert rec.test is None and rec.evaluation is None
    svc = joblib.load(tmp_path / "model.joblib")["model"].named_steps["model"]
    assert best["params"]["C"] == svc.estimator.C
    assert rec.info["preprocessing"] == {"kind": "demo"}


def test_stopping_before_a_trial_finished_saves_nothing(tmp_path):
    rec = SearchRecorder(stop_after_trials=0)
    result, _ = run_job(
        model_graph("SVC"), random_info(), tmp_path, reporter=rec, tuning=tuning("SVC")
    )
    assert result.status == "stopped" and "nothing was saved" in result.message
    assert not result.weights and not (tmp_path / "model.joblib").exists()


# --------------------------------------------------------------------------- #
# Estimate, runner and command line
# --------------------------------------------------------------------------- #


def test_estimate_counts_trials_and_folds():
    analysis = analyze(model_graph("SVC"))
    info = random_info()
    single = estimate_training(analysis, info, None, None)
    holdout = estimate_training(analysis, info, None, None, trials=10)
    cv = estimate_training(analysis, info, None, None, trials=10, folds=5)
    startup = 2.0  # CLASSICAL_STARTUP_CPU_SECONDS
    work = single.cpu_seconds - startup
    assert holdout.cpu_seconds == pytest.approx(startup + 10 * work)
    assert cv.cpu_seconds == pytest.approx(startup + 51 * work)  # 10 × 5 folds + the refit
    assert cv.details["fits"] == 51 and cv.details["folds"] == 5
    assert cv.basis.endswith("10 trials × 5 folds + the final fit")
    assert estimate_training(analysis, info, None, None, trials=1, folds=3).details["fits"] == 4


def test_runner_runs_a_classical_search(tmp_path):
    path = binary_csv(tmp_path / "mail.csv", n=120)
    info = inspect_csv(path)
    run_dir = tmp_path / "run"
    search = tuning("SVC", trials=3, metric="pr_auc", cv_folds=3)
    st.write_job(
        run_dir,
        {
            "name": "t",
            "graph": model_graph("SVC"),
            "dataset": {"kind": info.kind, "path": str(path), "info": info.to_dict()},
            "config": {"seed": 2, "test_split": 0.2},
            "tuning": search.to_dict(),
        },
    )
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
    assert state["tuning"]["metric"] == "pr_auc" and state["tuning"]["cv_folds"] == 3
    assert state["trials_total"] == 3 and len(state["trials"]) == 3 and state["trial"] is None
    assert all(t["score"] is not None and t["folds"] == 3 for t in state["trials"])
    assert state["best_trial"] in (1, 2, 3) and state["weights"] is True
    assert state["evaluation"]["trial"] == state["best_trial"]
    assert len(state["history"]) == 1 and state["test"] is not None


def test_cli_classical_search(tmp_path, capsys):
    path = binary_csv(tmp_path / "mail.csv", n=120)
    graph = tmp_path / "svc.json"
    graph.write_text(__import__("json").dumps(model_graph("SVC")))
    save = tmp_path / "svc.joblib"
    code = cli.main(
        [str(graph), "--data", str(path), "--seed", "1", "--test-split", "0.2", "--save", str(save)]
        + ["--search", "grid", "--tune-param", "C=0.5,5", "--tune-param", "kernel=rbf,linear"]
        + ["--metric", "roc_auc"]
    )
    out = capsys.readouterr().out
    assert code == 0, out
    assert (
        "Hyperparameter search: grid search, 4 trials, ranked by ROC AUC on the validation rows."
        in out
    )
    assert "Trial 1/4: C 0.5, kernel rbf" in out
    assert "trial    C  kernel  val loss  val acc  ROC AUC" in out
    assert "Finished 4 trials. Best: trial " in out and save.exists()

    code = cli.main(
        [str(graph), "--data", str(path), "--trials", "2", "--tune-param", "C=0.1:10"]
        + ["--cv-folds", "3", "--metric", "balanced_accuracy", "--save", str(save)]
    )
    out = capsys.readouterr().out
    assert code == 0, out
    assert "ranked by balanced accuracy on 3-fold cross-validation." in out
    assert "(cross-validated)" in out


@pytest.mark.parametrize(
    "args,message",
    [
        (["--tune-lr", "0.01,0.1"], "--tune-param NAME=VALUES"),
        (["--tune-param", "C"], "--tune-param needs NAME=VALUES"),
        (["--tune-param", "degree=2,3"], "'degree' cannot be tuned for support vector"),
        (["--tune", "--metric", "rmse"], "is for regression, and this is a classification"),
    ],
)
def test_cli_classical_search_errors(tmp_path, capsys, args, message):
    path = classification_csv(tmp_path / "mail.csv")
    graph = tmp_path / "svc.json"
    graph.write_text(__import__("json").dumps(model_graph("SVC")))
    assert cli.main([str(graph), "--data", str(path), *args]) == 1
    assert message in capsys.readouterr().err


def test_cli_param_option_is_for_classical_models(capsys):
    assert cli.main(["--example", "mlp_mnist", "--tune-param", "C=1,2"]) == 1
    assert "--tune-param is for classical models" in capsys.readouterr().err
