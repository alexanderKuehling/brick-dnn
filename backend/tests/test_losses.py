"""Selectable losses of neural networks: the settings and their checks per task, the loss
functions against hand calculations, trainings with every loss (also in a search, early
stopping and phase 1 of a hybrid network) and the generated script."""

from __future__ import annotations

import math
import shutil
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from brickdnn import generate_code  # noqa: E402
from brickdnn.training import TrainingConfig, inspect_csv, random_info  # noqa: E402
from brickdnn.training.compat import check_compatibility  # noqa: E402
from brickdnn.training.config import ConfigError  # noqa: E402
from brickdnn.training.data import Subset, TensorSource, class_counts  # noqa: E402
from brickdnn.training.engine import Reporter, TrainingJob, train  # noqa: E402
from brickdnn.training.loss_functions import CrossEntropy, FocalLoss, build_loss  # noqa: E402
from brickdnn.training.losses import (  # noqa: E402
    class_weight_vector,
    describe,
    loss_for,
    resolved,
)
from brickdnn.training.segmetrics import SegmentationLoss  # noqa: E402
from brickdnn.training.tuning import TuningConfig  # noqa: E402

from .helpers import chain  # noqa: E402
from .segmentation_helpers import dataset_files, flat_net, write_files  # noqa: E402


class Recorder(Reporter):
    def __init__(self):
        self.info, self.logs, self.trials = {}, [], []
        self.history, self.finished = [], None

    def setup(self, info):
        self.info.update(info)

    def log(self, line):
        self.logs.append(line)

    def epoch_end(self, entry, best):
        self.history.append(entry)

    def trial_end(self, trial, best_trial):
        self.trials.append(trial)

    def finish(self, status, message, weights):
        self.finished = (status, message, weights)


def mlp(inputs, outputs, width=16):
    return chain(
        ("Input", {"shape": [inputs]}),
        ("Linear", {"out_features": width}),
        ("Activation", {}),
        ("Linear", {"out_features": outputs}),
    )


def fraud_csv(path, n=400, seed=0):
    """Two features; about 5 % of the rows are 'fraud' (far out in both)."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 2, generator=gen)
    fraud = (x[:, 0] + x[:, 1]) > 2.3
    lines = ["a,b,label"] + [
        f"{a:.4f},{b:.4f},{'fraud' if f else 'ok'}"
        for (a, b), f in zip(x.tolist(), fraud.tolist(), strict=True)
    ]
    path.write_text("\n".join(lines) + "\n")
    return path, int(fraud.sum())


def prices_csv(path, n=200, seed=1):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, generator=gen)
    y = 100 + 20 * (x @ torch.tensor([1.0, -2.0, 0.5])) + 5 * torch.randn(n, generator=gen)
    y[:5] += 400  # outliers
    lines = ["a,b,c,price"] + [
        ",".join(f"{v:.4f}" for v in row) + f",{t:.3f}"
        for row, t in zip(x.tolist(), y.tolist(), strict=True)
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


def run(graph, path, tmp_path, tuning=None, **config):
    settings = {"epochs": 8, "batch_size": 16, "learning_rate": 0.01, "seed": 3, "device": "cpu"}
    job = TrainingJob(
        graph=graph,
        dataset=inspect_csv(path),
        config=TrainingConfig.from_dict({**settings, **config}),
        path=str(path),
        tuning=tuning,
    )
    reporter = Recorder()
    return train(job, reporter, tmp_path / "weights.pt"), reporter


# --------------------------------------------------------------------------- #
# Settings and checks
# --------------------------------------------------------------------------- #


def test_defaults_are_those_of_before():
    config = TrainingConfig()
    assert config.loss == "auto" and config.class_weights is None
    assert (config.focal_gamma, config.label_smoothing, config.huber_delta) == (2.0, 0.0, 1.0)
    # auto: the default of the task, as before.
    assert loss_for("auto", "classification") == "cross_entropy"
    assert loss_for("auto", "regression") == "mse"
    assert loss_for("auto", "segmentation") == "cross_entropy"
    # Settings of older versions said cross_entropy for every task.
    assert loss_for("cross_entropy", "regression") == "mse"
    assert (
        type(build_loss(config, {"kind": "class", "num_classes": 3}, None)) is nn.CrossEntropyLoss
    )
    assert type(build_loss(config, {"kind": "value", "shape": [1]}, None)) is nn.MSELoss
    old = TrainingConfig.from_dict({"loss": "cross_entropy"})
    assert type(build_loss(old, {"kind": "value", "shape": [1]}, None)) is nn.MSELoss
    plan = {"task": "regression", "targets": [{"kind": "value", "shape": [1]}]}
    assert resolved("auto", plan) == "mse" and resolved("cross_entropy", plan) == "mse"


def test_settings_are_parsed_and_their_ranges_checked():
    config = TrainingConfig.from_dict(
        {
            "loss": " Focal ",
            "focal_gamma": "1.5",
            "class_weights": "Balanced",
            "label_smoothing": 0.1,
            "huber_delta": "0.5",
        }
    )
    assert config.loss == "focal" and config.focal_gamma == 1.5
    assert config.class_weights == "balanced" and config.huber_delta == 0.5
    assert TrainingConfig.from_dict({"class_weights": [1, "2.5"]}).class_weights == [1.0, 2.5]
    assert TrainingConfig.from_dict({"class_weights": {"fraud": 10}}).class_weights == {
        "fraud": 10.0
    }
    assert TrainingConfig.from_dict(TrainingConfig().to_dict()) == TrainingConfig()
    for data, message in [
        ({"loss": "dice"}, "Loss must be one of: auto, cross_entropy, weighted_cross_entropy"),
        ({"focal_gamma": 11}, "Focal gamma must be at least 0 and at most 10."),
        ({"label_smoothing": 0.5}, "Label smoothing must be at least 0 and at most 0.3."),
        ({"huber_delta": 0}, "Huber delta must be greater than 0 and at most 10."),
        ({"class_weights": [1, -1]}, "Class weights must be greater than 0 and at most 1000."),
        ({"class_weights": "inverse"}, 'Class weights must be "balanced", a list'),
        ({"class_weights": ["x"]}, "Class weights must be numbers, not 'x'."),
        ({"class_weights": 3}, 'Class weights must be "balanced"'),
    ]:
        with pytest.raises(ConfigError, match=message.replace("(", r"\(").replace(".", r"\.")):
            TrainingConfig.from_dict(data)


def check(graph, info, **config):
    return check_compatibility(graph, info, TrainingConfig.from_dict(config))


def test_the_loss_must_fit_the_task(tmp_path):
    fraud = inspect_csv(fraud_csv(tmp_path / "fraud.csv")[0])
    prices = inspect_csv(prices_csv(tmp_path / "prices.csv"))
    for loss in ("auto", "cross_entropy", "weighted_cross_entropy", "focal"):
        assert check(mlp(2, 2), fraud, loss=loss).ok, loss
    for loss in ("auto", "mse", "mae", "huber", "cross_entropy"):  # cross_entropy: old default
        assert check(mlp(3, 1), prices, loss=loss).ok, loss
    result = check(mlp(2, 2), fraud, loss="mae")
    assert result.plan is None
    assert result.errors == [
        "The loss 'mae' is for regression, and this is a classification. "
        "Possible: auto, cross_entropy, weighted_cross_entropy, focal."
    ]
    assert check(mlp(3, 1), prices, loss="focal").errors == [
        "The loss 'focal' is for classification, and this is a regression. "
        "Possible: auto, mse, mae, huber."
    ]
    assert (
        check(mlp(2, 2), fraud, loss="cross_entropy_dice")
        .errors[0]
        .startswith(
            "The loss 'cross_entropy_dice' is for segmentation, and this is a classification."
        )
    )
    # Without settings (e.g. when a network is adapted) the loss is not checked.
    assert check_compatibility(mlp(2, 2), fraud).ok


def test_segmentation_losses(tmp_path):
    from brickdnn.training import builtin_info

    shapes = builtin_info("shapes")
    from .segmentation_helpers import tiny_unet

    unet = tiny_unet(classes=4, size=32, width=8)
    for loss in ("cross_entropy", "cross_entropy_dice", "weighted_cross_entropy"):
        assert check(unet, shapes, loss=loss, batch_size=4).ok, loss
    errors = check(unet, shapes, loss="focal", batch_size=4).errors
    assert errors == [
        "The loss 'focal' is for classification, and this is a segmentation. "
        "Possible: auto, cross_entropy, cross_entropy_dice, weighted_cross_entropy."
    ]
    # Label smoothing is for classification only: a segmentation ignores it.
    warnings = check(unet, shapes, label_smoothing=0.1, batch_size=4).warnings
    assert any("Label smoothing only work" in w for w in warnings), warnings


def test_class_weights_are_checked_against_the_classes(tmp_path):
    fraud = inspect_csv(fraud_csv(tmp_path / "fraud.csv")[0])
    net = mlp(2, 2)
    loss = "weighted_cross_entropy"
    assert check(net, fraud, loss=loss, class_weights=[10, 1]).ok
    assert check(net, fraud, loss=loss, class_weights={"fraud": 10}).ok
    assert check(net, fraud, loss=loss, class_weights={"FRAUD": 10, "1": 1}).ok
    assert check(net, fraud, loss=loss, class_weights=[1, 2, 3]).errors == [
        "Class weights: give one weight per class (2: fraud, ok), not 3."
    ]
    assert check(net, fraud, loss=loss, class_weights={"spam": 2}).errors == [
        "Class weights: 'spam' is not a class. The classes are fraud, ok (or their indices 0 to 1)."
    ]
    # The plain cross-entropy does not use them: a warning, not an error.
    result = check(net, fraud, loss="cross_entropy", class_weights="balanced")
    assert result.ok and result.warnings == [
        "Class weights only work with weighted_cross_entropy and focal: the cross-entropy "
        "ignores them."
    ]


def test_balanced_weights_from_the_class_counts():
    sklearn = pytest.importorskip("sklearn.utils.class_weight")
    labels = torch.tensor([0] * 90 + [1] * 8 + [2] * 2)
    source = TensorSource([torch.zeros(100, 1)], [labels])
    counts = class_counts(Subset(source, torch.arange(100)), 0, 4)
    assert counts == [90.0, 8.0, 2.0, 0.0]
    weights = class_weight_vector("balanced", 4, None, counts)
    np = pytest.importorskip("numpy")
    present = np.array([0, 1, 2])
    reference = sklearn.compute_class_weight("balanced", classes=present, y=labels.numpy())
    assert weights[:3] == pytest.approx(reference.tolist())
    assert weights[0] == pytest.approx(100 / (3 * 90)) and weights[3] == 1.0  # not in training
    # Given weights: a list, or by class name or index (others get 1).
    assert class_weight_vector([2, 3], 2, ["a", "b"], None) == [2.0, 3.0]
    assert class_weight_vector({"b": 5, "0": 2}, 3, ["a", "b", "c"], None) == [2.0, 5.0, 1.0]
    assert class_weight_vector(None, 3, None, None) is None
    # Pixels of masks, without the ignored ones (255).
    masks = torch.tensor([[[0, 0], [1, 255]], [[0, 1], [1, 1]]], dtype=torch.uint8)
    assert class_counts(TensorSource([torch.zeros(2, 1)], [masks]), 0, 2) == [3.0, 4.0]


def test_balanced_weights_of_a_run_come_from_the_training_rows(tmp_path):
    path, _ = fraud_csv(tmp_path / "fraud.csv")
    result, rec = run(mlp(2, 2), path, tmp_path, epochs=1, loss="weighted_cross_entropy")
    assert result.status == "completed", result.message
    loss = rec.info["loss"]
    assert loss["name"] == "weighted_cross_entropy" and loss["class_weights_from"] == "balanced"
    # The 320 training rows of the seeded split, not all 400 rows.
    assert rec.info["train_size"] == 320
    fraud, ok = loss["class_weights"]
    n_fraud = round(320 / (2 * fraud))
    assert ok == pytest.approx(320 / (2 * (320 - n_fraud)), abs=1e-4)
    assert fraud > 5 > 1 > ok
    assert rec.logs[0] == (
        "Loss: weighted cross-entropy (balanced class weights). The curves, early stopping "
        "and the kept epoch use it."
    )


# --------------------------------------------------------------------------- #
# The loss functions
# --------------------------------------------------------------------------- #


def test_focal_loss_against_a_hand_calculation():
    logits = torch.tensor([[2.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    target = torch.tensor([0, 0, 1])
    p = [
        math.exp(2) / (math.exp(2) + 1),  # true class 0, probability 0.881
        1 / (1 + math.e),  # true class 0, probability 0.269
        0.5,
    ]
    for gamma in (0.0, 1.0, 2.0):
        expected = sum(-((1 - q) ** gamma) * math.log(q) for q in p) / 3
        assert float(FocalLoss(gamma)(logits, target)) == pytest.approx(expected, rel=1e-6)
    assert float(FocalLoss(0.0)(logits, target)) == pytest.approx(
        float(F.cross_entropy(logits, target))
    )
    # Class weights are alpha: the loss of every sample times the weight of its class.
    alpha = torch.tensor([0.25, 4.0])
    expected = sum(
        -w * (1 - q) ** 2 * math.log(q) for q, w in zip(p, (0.25, 0.25, 4.0), strict=True)
    )
    assert float(FocalLoss(2.0, alpha)(logits, target)) == pytest.approx(expected / 3, rel=1e-6)
    # Easy samples count much less than with the cross-entropy.
    easy = torch.tensor([[6.0, 0.0]])
    assert float(FocalLoss(2.0)(easy, torch.tensor([0]))) < 1e-4 * float(
        F.cross_entropy(easy, torch.tensor([0]))
    )


def test_label_smoothing_and_class_weights_of_the_cross_entropy():
    logits = torch.tensor([[2.0, 0.0, -1.0], [0.5, 0.5, 0.0]])
    target = torch.tensor([0, 2])
    log_p = torch.log_softmax(logits, dim=1)
    eps, k = 0.1, 3
    smoothed = (1 - eps) * F.one_hot(target, k) + eps / k
    expected = float(-(smoothed * log_p).sum(dim=1).mean())
    assert float(CrossEntropy(None, eps)(logits, target)) == pytest.approx(expected, rel=1e-6)
    config = TrainingConfig.from_dict({"loss": "cross_entropy", "label_smoothing": eps})
    built = build_loss(config, {"kind": "class", "num_classes": 3}, None)
    assert float(built(logits, target)) == pytest.approx(expected, rel=1e-6)
    # Weighted: the weighted mean over the samples.
    weight = torch.tensor([1.0, 1.0, 3.0])
    nll = -log_p[torch.arange(2), target]
    expected = float((nll * weight[target]).sum() / weight[target].sum())
    config = TrainingConfig.from_dict({"loss": "weighted_cross_entropy"})
    built = build_loss(config, {"kind": "class", "num_classes": 3}, [1.0, 1.0, 3.0])
    assert float(built(logits, target)) == pytest.approx(expected, rel=1e-6)


def test_mae_and_huber():
    output = torch.tensor([[0.0], [1.0], [4.0]])
    target = torch.tensor([[0.5], [1.0], [1.0]])
    mae = build_loss(TrainingConfig.from_dict({"loss": "mae"}), {"kind": "value"}, None)
    assert float(mae(output, target)) == pytest.approx((0.5 + 0 + 3) / 3)
    config = TrainingConfig.from_dict({"loss": "huber", "huber_delta": 1.0})
    huber = build_loss(config, {"kind": "value"}, None)
    # 0.5 · e² for |e| <= delta, delta · (|e| - delta / 2) beyond.
    assert float(huber(output, target)) == pytest.approx((0.5 * 0.25 + 0 + (3 - 0.5)) / 3)
    config = TrainingConfig.from_dict({"loss": "huber", "huber_delta": 0.25})
    huber = build_loss(config, {"kind": "value"}, None)
    expected = (0.25 * (0.5 - 0.125) + 0 + 0.25 * (3 - 0.125)) / 3
    assert float(huber(output, target)) == pytest.approx(expected)
    assert describe(config, "regression") == {
        "name": "huber",
        "label": "Huber loss",
        "task": "regression",
        "huber_delta": 0.25,
    }


def test_weighted_cross_entropy_of_pixels():
    output = torch.tensor([[[[2.0, 0.0]], [[0.0, 1.0]]]])  # (1, 2 classes, 1, 2)
    target = torch.tensor([[[0, 1]]])
    weight = torch.tensor([1.0, 3.0])
    log_p = torch.log_softmax(output, dim=1)
    nll = torch.stack([-log_p[0, 0, 0, 0], -log_p[0, 1, 0, 1]])
    expected = float((nll * torch.tensor([1.0, 3.0])).sum() / 4)
    loss = SegmentationLoss(2, False, weight=weight)
    assert float(loss(output, target)) == pytest.approx(expected, rel=1e-6)
    # One output channel (two classes): the binary cross-entropy weighted the same way.
    logits = torch.tensor([[[[1.0, -1.0, 0.5]]]])
    labels = torch.tensor([[[0, 1, 255]]])
    bce = F.binary_cross_entropy_with_logits(
        torch.tensor([1.0, -1.0]), torch.tensor([0.0, 1.0]), reduction="none"
    )
    expected = float((bce * torch.tensor([1.0, 3.0])).sum() / 4)
    assert float(SegmentationLoss(2, True, weight=weight)(logits, labels)) == pytest.approx(
        expected, rel=1e-6
    )


# --------------------------------------------------------------------------- #
# Trainings
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "config",
    [
        {"loss": "weighted_cross_entropy"},
        {"loss": "weighted_cross_entropy", "class_weights": {"fraud": 8}},
        {"loss": "focal"},
        {"loss": "focal", "class_weights": "balanced", "focal_gamma": 1},
        {"loss": "cross_entropy", "label_smoothing": 0.1},
    ],
)
def test_classification_trains_with_every_loss(tmp_path, config):
    path, _ = fraud_csv(tmp_path / "fraud.csv")
    result, rec = run(mlp(2, 2), path, tmp_path, **config)
    assert result.status == "completed", result.message
    assert rec.info["loss"]["name"] == config["loss"]
    losses = [entry["train_loss"] for entry in result.history]
    assert losses[-1] < losses[0], losses
    # The final evaluation keeps its metrics, the log loss among them.
    assert result.history[-1]["val_accuracy"] is not None


@pytest.mark.parametrize("config", [{"loss": "mae"}, {"loss": "huber", "huber_delta": 0.5}])
def test_regression_trains_with_every_loss(tmp_path, config):
    path = prices_csv(tmp_path / "prices.csv")
    result, rec = run(mlp(3, 1), path, tmp_path, **config)
    assert result.status == "completed", result.message
    assert rec.info["loss"]["name"] == config["loss"]
    history = result.history
    assert history[-1]["train_loss"] < history[0]["train_loss"]
    assert history[-1]["val_loss"] < history[0]["val_loss"]
    # RMSE comes from the squared errors, not from the loss (which is no squared error).
    rmse = history[-1]["val_rmse"]
    std = rec.info["preprocessing"]["target_std"]
    assert rmse is not None and rmse != pytest.approx(math.sqrt(history[-1]["val_loss"]) * std)


def test_rmse_of_the_mean_squared_error_is_as_before(tmp_path):
    path = prices_csv(tmp_path / "prices.csv")
    result, rec = run(mlp(3, 1), path, tmp_path, epochs=2)
    assert rec.info["loss"] == {"name": "mse", "label": "mean squared error", "task": "regression"}
    entry = result.history[-1]
    std = rec.info["preprocessing"]["target_std"]
    assert entry["val_rmse"] == pytest.approx(math.sqrt(entry["val_loss"]) * std)
    assert not rec.logs or not rec.logs[0].startswith("Loss:")  # the default is not logged


def test_early_stopping_and_the_kept_epoch_use_the_chosen_loss(tmp_path):
    path, _ = fraud_csv(tmp_path / "fraud.csv")
    result, _ = run(
        mlp(2, 2, width=32),
        path,
        tmp_path,
        epochs=60,
        learning_rate=0.05,
        patience=3,
        loss="focal",
        focal_gamma=3,
    )
    assert result.status == "completed", result.message
    history = result.history
    best = min(history, key=lambda entry: entry["val_loss"])
    assert result.weights_epoch == best["epoch"] == result.best["epoch"]
    assert result.stopped_early and history[-1]["epoch"] - best["epoch"] == 3


def test_weighted_segmentation_trains(tmp_path):
    write_files(tmp_path / "data", dataset_files(24, classes=3))
    from .test_segmentation_engine import folder_job

    job = folder_job(tmp_path / "data", flat_net(classes=3), loss="weighted_cross_entropy")
    rec = Recorder()
    result = train(job, rec, tmp_path / "weights.pt")
    assert result.status == "completed", result.message
    loss = rec.info["loss"]
    assert loss["name"] == "weighted_cross_entropy" and len(loss["class_weights"]) == 3
    assert result.history[-1]["train_loss"] < result.history[0]["train_loss"]


def test_demo_data_with_outputs_of_two_tasks():
    graph = {
        "version": 1,
        "nodes": [
            {"id": "in", "type": "Input", "params": {"shape": [4]}},
            {"id": "a", "type": "Linear", "params": {"out_features": 3}},
            {"id": "b", "type": "Linear", "params": {"out_features": 1}},
        ],
        "edges": [{"source": "in", "target": "a"}, {"source": "in", "target": "b"}],
    }
    config = TrainingConfig.from_dict({"epochs": 1, "loss": "huber", "seed": 0, "device": "cpu"})
    assert check_compatibility(graph, random_info(), config).ok
    rec = Recorder()
    result = train(TrainingJob(graph=graph, dataset=random_info(), config=config), rec)
    assert result.status == "completed", result.message
    assert rec.info["loss"]["name"] == "cross_entropy"  # the first output: classes
    assert rec.info["loss"]["outputs"] == ["cross_entropy", "huber"]


# --------------------------------------------------------------------------- #
# Hyperparameter search
# --------------------------------------------------------------------------- #


def search(space, metric="loss", **extra):
    return TuningConfig.from_dict({"method": "grid", "space": space, "metric": metric, **extra})


def test_a_search_over_losses_needs_another_metric(tmp_path):
    fraud = inspect_csv(fraud_csv(tmp_path / "fraud.csv")[0])
    config = TrainingConfig()
    over_losses = search({"loss": ["cross_entropy", "focal"]})
    result = check_compatibility(mlp(2, 2), fraud, config, tuning=over_losses)
    assert result.plan is None
    assert result.errors[0].startswith(
        "A search over the loss cannot rank its trials by the loss: losses of different kinds "
        "are not comparable"
    )
    assert "e.g. pr_auc, balanced_accuracy or macro_f1." in result.errors[0]
    ranked = search({"loss": ["cross_entropy", "focal"]}, "pr_auc")
    assert check_compatibility(mlp(2, 2), fraud, config, tuning=ranked).ok
    # One loss in the list is no search over losses.
    assert check_compatibility(mlp(2, 2), fraud, config, tuning=search({"loss": ["focal"]})).ok
    # A loss of another task, and a setting that no trial's loss uses.
    wrong = search({"loss": ["mae"]}, "pr_auc")
    assert (
        check_compatibility(mlp(2, 2), fraud, config, tuning=wrong)
        .errors[0]
        .startswith("Hyperparameter search: The loss 'mae' is for regression")
    )
    unused = search({"focal_gamma": [1, 2]}, "pr_auc")
    assert check_compatibility(mlp(2, 2), fraud, config, tuning=unused).errors == [
        "Hyperparameter search: focal_gamma only changes the loss focal, which no trial uses. "
        "Set the loss, or search it."
    ]
    # Searching a setting of the loss ranked by the loss: allowed, with a warning.
    gammas = search({"focal_gamma": [1, 2]})
    result = check_compatibility(
        mlp(2, 2), fraud, TrainingConfig.from_dict({"loss": "focal"}), tuning=gammas
    )
    assert result.ok and any("changes the loss itself" in w for w in result.warnings)


def test_search_parses_losses_and_their_settings():
    tuning = TuningConfig.from_dict(
        {
            "method": "random",
            "trials": 4,
            "metric": "pr_auc",
            "space": {
                "loss": ["Focal", "cross_entropy"],
                "focal_gamma": {"min": 0, "max": 4},
                "label_smoothing": [0, 0.1],
            },
        }
    )
    space = tuning.to_dict()["space"]
    assert space == {
        "loss": ["focal", "cross_entropy"],
        "focal_gamma": {"min": 0.0, "max": 4.0, "scale": "linear"},
        "label_smoothing": [0.0, 0.1],
    }
    assert TuningConfig.from_dict(tuning.to_dict()).to_dict() == tuning.to_dict()
    log = TuningConfig.from_dict({"space": {"huber_delta": {"min": 0.1, "max": 5, "scale": "log"}}})
    assert TuningConfig.from_dict(log.to_dict()).space["huber_delta"].log is True
    for space, message in [
        ({"loss": ["dice"]}, "the loss must be one of: auto, "),
        ({"label_smoothing": [0.5]}, "must be at least 0 and at most 0.3"),
        ({"huber_delta": {"min": 0, "max": 1}}, "must be greater than 0 and at most 10"),
    ]:
        with pytest.raises(ConfigError, match=message):
            TuningConfig.from_dict({"space": space})


def test_search_over_losses_ranked_by_pr_auc(tmp_path):
    path, _ = fraud_csv(tmp_path / "fraud.csv")
    tuning = search({"loss": ["cross_entropy", "weighted_cross_entropy", "focal"]}, "pr_auc")
    result, rec = run(mlp(2, 2), path, tmp_path, tuning=tuning, epochs=3)
    assert result.status == "completed", result.message
    assert [t["params"]["loss"] for t in result.trials] == [
        "cross_entropy",
        "weighted_cross_entropy",
        "focal",
    ]
    scores = [t["score"] for t in result.trials]
    assert result.best_trial == 1 + scores.index(max(scores))
    best = result.trials[result.best_trial - 1]
    assert rec.info["loss"]["name"] == best["params"]["loss"]  # the loss of the kept trial
    assert "Best: trial" in result.message and "validation PR AUC" in result.message


def test_the_engine_refuses_a_search_over_losses_by_the_loss(tmp_path):
    path, _ = fraud_csv(tmp_path / "fraud.csv")
    tuning = search({"loss": ["cross_entropy", "focal"]})
    result, _ = run(mlp(2, 2), path, tmp_path, tuning=tuning, epochs=1)
    assert result.status == "failed"
    assert result.message.startswith("A search over the loss cannot rank its trials by the loss")


# --------------------------------------------------------------------------- #
# Hybrid networks
# --------------------------------------------------------------------------- #


def test_phase_one_of_a_hybrid_network_learns_with_the_focal_loss(tmp_path):
    pytest.importorskip("sklearn")
    from .test_hybrid import hybrid_graph

    path, _ = fraud_csv(tmp_path / "fraud.csv")
    graph = hybrid_graph("HistGradientBoostingClassifier", features=2)
    result, rec = run(graph, path, tmp_path, loss="focal", class_weights="balanced", epochs=4)
    assert result.status == "completed", result.message
    loss = rec.info["loss"]
    assert loss["name"] == "focal" and loss["class_weights_from"] == "balanced"
    assert any(line.startswith("Loss: focal loss (gamma 2") for line in rec.logs)
    losses = [entry["train_loss"] for entry in result.history]
    assert len(losses) == 4 and losses[-1] < losses[0]
    # The search of its neural settings may vary the loss and still rank by the loss: that of
    # the classical model, which does not depend on the loss of the neural layers.
    tuning = TuningConfig.from_dict(
        {"method": "grid", "neural_space": {"loss": ["cross_entropy", "focal"]}, "space": {}},
        "HistGradientBoostingClassifier",
        hybrid=True,
    )
    assert check_compatibility(graph, inspect_csv(path), TrainingConfig(), tuning=tuning).ok


# --------------------------------------------------------------------------- #
# The generated script
# --------------------------------------------------------------------------- #


def run_script(code, tmp_path, *args):
    script = tmp_path / "model.py"
    script.write_text(code)
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=300,
    )


@pytest.mark.parametrize(
    ("options", "weights"),
    [
        ({"loss": "weighted_cross_entropy"}, True),
        ({"loss": "weighted_cross_entropy", "class_weights": {"1": 5}}, True),
        ({"loss": "focal", "focal_gamma": 1.5, "class_weights": "balanced"}, True),
        ({"loss": "focal"}, False),
        ({"loss": "cross_entropy", "label_smoothing": 0.1}, False),
    ],
)
def test_script_trains_a_classifier_with_the_loss(tmp_path, options, weights):
    path, frauds = fraud_csv(tmp_path / "fraud.csv", n=120)
    rows = path.read_text().replace(",fraud", ",1").replace(",ok", ",0")
    (tmp_path / "data.csv").write_text(rows)
    code = generate_code(mlp(2, 2), options)
    assert f'LOSS = "{options["loss"]}"' in code
    assert ("def focal_loss(" in code) and ("def class_weights(" in code)
    result = run_script(code, tmp_path, "--csv", "data.csv", "--epochs", "3")
    assert result.returncode == 0, result.stderr
    assert ("Class weights: [" in result.stdout) is weights
    epochs = [line for line in result.stdout.splitlines() if line.startswith("Epoch")]
    assert len(epochs) == 3 and "val acc" in epochs[-1]
    if options.get("class_weights") == {"1": 5}:
        assert "Class weights: [1.0, 5.0]" in result.stdout
    elif weights:
        # Balanced from the 96 training rows of the script's own split.
        line = next(x for x in result.stdout.splitlines() if x.startswith("Class weights"))
        ok, fraud = (float(v) for v in line.split("[")[1].rstrip("]").split(","))
        assert fraud > ok and 96 / (2 * fraud) + 96 / (2 * ok) == pytest.approx(96, rel=0.01)
    assert frauds > 0


def test_script_trains_a_regression_with_the_huber_loss(tmp_path):
    prices_csv(tmp_path / "data.csv", n=60)
    code = generate_code(mlp(3, 1), {"loss": "huber", "huber_delta": 0.5})
    assert 'LOSS = "huber"' in code and "HUBER_DELTA = 0.5" in code
    result = run_script(code, tmp_path, "--csv", "data.csv", "--epochs", "2")
    assert result.returncode == 0, result.stderr
    assert "Saved the weights" in result.stdout
    code = generate_code(mlp(3, 1), {"loss": "mae"})
    assert run_script(code, tmp_path, "--csv", "data.csv", "--epochs", "1").returncode == 0


def test_script_defaults_and_refusals():
    plain = generate_code(mlp(2, 2))
    assert "loss_fn = nn.CrossEntropyLoss()" in plain and "LOSS =" not in plain
    assert "loss_fn = nn.MSELoss()" in generate_code(mlp(3, 1), {"loss": "auto"})
    assert "LOSS =" not in generate_code(mlp(3, 1), {"loss": "cross_entropy"})  # old default
    with pytest.raises(ValueError, match="The loss 'mae' does not fit the training code"):
        generate_code(mlp(2, 2), {"loss": "mae"})
    with pytest.raises(ValueError, match="Class weights: give one weight per class \\(2\\)"):
        generate_code(mlp(2, 2), {"loss": "focal", "class_weights": [1, 2, 3]})
    with pytest.raises(ValueError, match="'fraud' is not a class of the script"):
        generate_code(mlp(2, 2), {"loss": "focal", "class_weights": {"fraud": 2}})
    with pytest.raises(ValueError, match="Focal gamma must be"):
        generate_code(mlp(2, 2), {"loss": "focal", "focal_gamma": -1})
    # Without training code the loss does not matter.
    assert "LOSS" not in generate_code(mlp(2, 2), {"loss": "mae", "include_training": False})


@pytest.mark.skipif(shutil.which("ruff") is None, reason="ruff is not installed")
@pytest.mark.parametrize(
    ("graph", "options"),
    [
        (mlp(2, 2), {"loss": "focal", "class_weights": [1, 4]}),
        (mlp(2, 2), {"loss": "weighted_cross_entropy", "label_smoothing": 0.2}),
        (mlp(3, 1), {"loss": "huber"}),
    ],
)
def test_script_with_a_loss_is_lint_clean(tmp_path, graph, options):
    target = tmp_path / "model.py"
    target.write_text(generate_code(graph, options))
    result = subprocess.run(
        [
            shutil.which("ruff"),
            *("check", "--isolated", "--select", "E,F,W", "--line-length", "100"),
            str(target),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout
