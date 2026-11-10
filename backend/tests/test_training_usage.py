"""What the training runner measures and how much a training is estimated to use.

The runner's StateReporter reports the CPU time of the process in state.json and stops the
run at the CPU limit of its job; the estimate (brickdnn.training.estimate) is pure
arithmetic. Neither needs a database or PyTorch. See test_budgets.py for the server side.
"""

from __future__ import annotations

import time

import pytest

from brickdnn import EXAMPLES, analyze
from brickdnn.training import builtin_info, random_info
from brickdnn.training.config import TrainingConfig
from brickdnn.training.datasets import DatasetInfo
from brickdnn.training.estimate import estimate_training, forward_flops
from brickdnn.training.runner import StateReporter
from brickdnn.training.state import read_state, request_stop

# --------------------------------------------------------------------------- #
# Measuring in the runner
# --------------------------------------------------------------------------- #


def burn(seconds: float) -> None:
    """Use CPU time (not just wall time)."""
    end = time.process_time() + seconds
    while time.process_time() < end:
        sum(i * i for i in range(1000))


def test_runner_reports_its_cpu_time(tmp_path):
    reporter = StateReporter(tmp_path, epochs=1, heartbeat=0.05)
    burn(0.2)
    reporter.progress({"epoch": 1, "step": 1, "steps": 2, "loss": 0.5})
    reporter.finish("completed", "Finished.", False)
    usage = read_state(tmp_path)["usage"]
    assert usage["cpu_seconds"] >= 0.2
    assert usage["wall_seconds"] > 0
    assert usage["limit_cpu_seconds"] is None
    assert read_state(tmp_path)["stop_reason"] is None


def test_runner_stops_by_itself_at_its_limit(tmp_path):
    limits = {"cpu_seconds": 0.05, "reason": "run_limit", "message": "Stopped: the limit."}
    reporter = StateReporter(tmp_path, epochs=3, heartbeat=0.05, limits=limits)
    burn(0.1)
    assert reporter.should_stop() is True
    reporter.finish("stopped", "Stopped after 1 of 3 epochs.", True)
    state = read_state(tmp_path)
    assert state["message"] == "Stopped: the limit. Stopped after 1 of 3 epochs."
    assert state["stop_reason"] == "run_limit"
    assert state["usage"]["limit_cpu_seconds"] == 0.05

    # Without a limit, or with one that is not reached, it goes on.
    other = tmp_path / "other"
    other.mkdir()
    reporter = StateReporter(other, heartbeat=0.05, limits={"cpu_seconds": 10**6})
    assert reporter.should_stop() is False
    reporter.finish("completed", "Done.", True)
    assert StateReporter(other, limits={"cpu_seconds": "x"}).limits is None


def test_runner_says_why_the_server_stopped_it(tmp_path):
    reporter = StateReporter(tmp_path, heartbeat=0.05)
    request_stop(tmp_path, "budget", "Stopped: the budget of the API token “agent” is used up.")
    assert reporter.should_stop() is True
    reporter.finish("stopped", "Stopped after 2 of 5 epochs.", True)
    state = read_state(tmp_path)
    assert state["stop_reason"] == "budget"
    assert state["message"].startswith("Stopped: the budget of the API token “agent”")

    user_stop = tmp_path / "user"
    user_stop.mkdir()
    reporter = StateReporter(user_stop, heartbeat=0.05)
    request_stop(user_stop)
    reporter.finish("stopped", "Stopped after 2 of 5 epochs.", True)
    assert read_state(user_stop)["stop_reason"] == "user"
    assert read_state(user_stop)["message"] == "Stopped after 2 of 5 epochs."


# --------------------------------------------------------------------------- #
# The estimate
# --------------------------------------------------------------------------- #


def test_estimate_grows_with_epochs_samples_and_the_network():
    mlp = analyze(EXAMPLES["mlp_mnist"]["graph"])
    cnn = analyze(EXAMPLES["cnn_mnist"]["graph"])
    mnist = builtin_info("mnist")

    def cpu(analysis, **config):
        return estimate_training(
            analysis, mnist, TrainingConfig.from_dict(config), None
        ).cpu_seconds

    assert cpu(mlp, epochs=10) > 3 * cpu(mlp, epochs=2)
    assert cpu(mlp, epochs=2, max_samples=1000) < cpu(mlp, epochs=2)
    assert cpu(cnn, epochs=2) > cpu(mlp, epochs=2)
    # About 2 × weights per sample for a dense network.
    assert forward_flops(mlp) == pytest.approx(2 * mlp.total_params, rel=0.05)
    estimate = estimate_training(mlp, mnist, TrainingConfig.from_dict({"epochs": 3}), None, 5)
    assert estimate.details["trials"] == 5
    assert estimate.low < estimate.cpu_seconds < estimate.high
    assert "48,000 training" in estimate.basis


def test_estimate_of_classical_models():
    gp = analyze(EXAMPLES["gp_regression"]["graph"])
    svm = analyze(EXAMPLES["svm_classifier"]["graph"])
    rows = DatasetInfo(kind="csv", name="rows", num_samples=4000, num_features=8)
    small = DatasetInfo(kind="csv", name="rows", num_samples=400, num_features=8)
    config = TrainingConfig()
    assert estimate_training(gp, rows, config, None).cpu_seconds > (
        5 * estimate_training(gp, small, config, None).cpu_seconds
    )
    assert estimate_training(svm, random_info(), config, None).details["family"] == "classical"
    broken = analyze({"nodes": [{"id": "x", "type": "Linear", "params": {}}], "edges": []})
    assert estimate_training(broken, rows, config, None) is None
