"""The objective metric of a hyperparameter search of a neural network: the trials are
ranked by it, and the default (the loss) behaves as before."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from brickdnn import EXAMPLES  # noqa: E402
from brickdnn.training import cli, engine, runner  # noqa: E402
from brickdnn.training import state as st  # noqa: E402

from .test_training_engine import Recorder, search_job  # noqa: E402

GRAPH = EXAMPLES["mlp_mnist"]["graph"]


class EvaluationRecorder(Recorder):
    def evaluation_end(self, result):
        self.evaluation = result


def scripted(monkeypatch, values):
    """Validation results by learning rate: ``{lr: (loss, accuracy)}`` for every epoch."""
    original = engine._Run._validate

    def validate(self, epoch):
        result = original(self, epoch)
        loss, accuracy = values[self.config.learning_rate]
        return {**result, "loss": loss, "accuracy": accuracy}

    monkeypatch.setattr(engine._Run, "_validate", validate)


GRID = {"method": "grid", "space": {"learning_rate": [0.01, 0.001]}}


def test_the_loss_ranks_the_trials_by_default(tmp_path, monkeypatch):
    # Trial 1 has the lower loss, trial 2 the higher accuracy.
    scripted(monkeypatch, {0.01: (0.2, 0.6), 0.001: (0.3, 0.9)})
    result, rec = search_job(GRAPH, tmp_path, GRID, epochs=1, max_samples=64)
    assert result.status == "completed", result.message
    assert [t["score"] for t in result.trials] == [0.2, 0.3]
    assert result.best_trial == 1 and rec.info["tuning"]["metric"] == "loss"
    assert result.message.startswith("Finished 2 trials. Best: trial 1 (learning rate 0.01), ")
    assert "validation loss" not in result.message  # the default needs no mention


def test_another_metric_ranks_the_trials(tmp_path, monkeypatch):
    scripted(monkeypatch, {0.01: (0.2, 0.6), 0.001: (0.3, 0.9)})
    tuning = {**GRID, "metric": "accuracy"}
    result, rec = search_job(GRAPH, tmp_path, tuning, epochs=1, max_samples=64)
    assert result.status == "completed", result.message
    assert [t["score"] for t in result.trials] == [0.6, 0.9]
    assert result.best_trial == 2 and rec.search["best_trial"] == 2
    assert result.message.startswith(
        "Finished 2 trials. Best: trial 2 (learning rate 0.001), validation accuracy 90.0%; "
    )


def test_a_metric_from_an_extra_pass(tmp_path):
    tuning = {
        "method": "grid",
        "metric": "balanced_accuracy",
        "space": {"learning_rate": [1e-7, 3e-3]},
    }
    rec = EvaluationRecorder()
    result, _ = search_job(GRAPH, tmp_path, tuning, reporter=rec, epochs=2, max_samples=128)
    assert result.status == "completed", result.message
    scores = [t["score"] for t in result.trials]
    assert all(0 <= score <= 1 for score in scores)
    best = max(result.trials, key=lambda t: (t["score"], -t["trial"]))
    assert result.best_trial == best["trial"]
    # The final evaluation measures the same kept weights on the same rows.
    assert rec.evaluation["trial"] == best["trial"]
    assert rec.evaluation["val"]["balanced_accuracy"] == pytest.approx(best["score"])
    assert "validation balanced accuracy" in result.message


def test_a_metric_the_task_does_not_have(tmp_path):
    result, _ = search_job(GRAPH, tmp_path, {**GRID, "metric": "rmse"}, epochs=1, max_samples=64)
    assert result.status == "failed"
    assert "RMSE is for regression, and this is a classification" in result.message
    regression = EXAMPLES["tabular_regression"]["graph"]
    tuning = {**GRID, "metric": "r2"}
    result, _ = search_job(regression, tmp_path, tuning, epochs=1, max_samples=64)
    assert result.status == "completed", result.message
    assert [t["score"] for t in result.trials] == [t["val_r2"] for t in result.trials]


def test_state_reporter_keeps_the_score(tmp_path):
    reporter = runner.StateReporter(tmp_path, epochs=1, heartbeat=0.05)
    reporter.trial_start({"trial": 1, "trials": 2, "params": {"learning_rate": 0.1}})
    assert st.read_state(tmp_path)["trials"][0]["score"] is None
    reporter.trial_end(
        {**st.read_state(tmp_path)["trials"][0], "status": "completed", "score": 0.8}, 1
    )
    reporter.finish("completed", "Done.", False)
    assert st.read_state(tmp_path)["trials"][0]["score"] == 0.8


def test_cli_metric(capsys):
    code = cli.main(
        ["--example", "mlp_mnist", "--epochs", "1", "--device", "cpu", "--max-samples", "64"]
        + ["--seed", "0", "--search", "grid", "--tune-lr", "0.01,0.001"]
        + ["--metric", "macro_f1", "--save", "/dev/null"]
    )
    out = capsys.readouterr().out
    assert code == 0, out
    assert (
        "Hyperparameter search: grid search, 2 trials of 1 epoch each, ranked by macro F1." in out
    )
    assert "trial  learning rate  val loss  val acc  macro F1" in out
    assert ", macro F1 0." in out


def test_cli_metric_errors(capsys):
    assert cli.main(["--example", "mlp_mnist", "--tune", "--metric", "mae"]) == 1
    assert "MAE is for regression" in capsys.readouterr().err
    assert cli.main(["--example", "mlp_mnist", "--tune", "--cv-folds", "3"]) == 1
    assert "only available for classical models" in capsys.readouterr().err
