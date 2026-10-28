"""Training with PyTorch: data preparation, the training loop, the runner, probe and CLI.

Everything runs on the CPU with tiny settings and without network access (the
torchvision datasets are replaced by fake data).
"""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from brickdnn import EXAMPLES, generate_code  # noqa: E402
from brickdnn.__main__ import main as brickdnn_main  # noqa: E402
from brickdnn.training import (  # noqa: E402
    BUILTIN_DATASETS,
    TrainingConfig,
    builtin_info,
    check_compatibility,
    cli,
    data,
    devices,
    engine,
    inspect_csv,
    inspect_image_folder,
    random_info,
    runner,
)
from brickdnn.training import state as st  # noqa: E402
from brickdnn.training.engine import Reporter, TrainingJob, build_model, train  # noqa: E402
from brickdnn.training.tuning import TuningConfig  # noqa: E402

from .helpers import build_model as build_from_code  # noqa: E402
from .helpers import chain  # noqa: E402

BACKEND = Path(__file__).resolve().parent.parent

try:
    from PIL import Image
except ImportError:  # pragma: no cover - Pillow comes with torchvision
    Image = None

needs_pil = pytest.mark.skipif(Image is None, reason="Pillow is not installed")


class Recorder(Reporter):
    """Collects everything the engine reports; can ask to stop after some steps."""

    def __init__(self, stop_after_checks: int | None = None):
        self.statuses: list[tuple[str, str]] = []
        self.logs: list[str] = []
        self.info: dict = {}
        self.history: list[dict] = []
        self.best = None
        self.finished = None
        self.checks = 0
        self.stop_after_checks = stop_after_checks
        self.steps: list[int] = []
        self.events: list[tuple] = []
        self.test = None
        self.trials_started: list[dict] = []
        self.trials_ended: list[tuple[dict, int | None]] = []
        self.search = None

    def status(self, status, message):
        self.statuses.append((status, message))

    def log(self, line):
        self.logs.append(line)

    def setup(self, info):
        self.info.update(info)

    def progress(self, progress):
        self.steps.append(progress["step"])

    def epoch_end(self, entry, best):
        self.history.append(entry)
        self.best = best
        self.events.append(("epoch_end", entry["epoch"]))

    def trial_start(self, trial):
        self.trials_started.append(trial)
        self.events.append(("trial_start", trial["trial"]))

    def trial_end(self, trial, best_trial):
        self.trials_ended.append((trial, best_trial))
        self.events.append(("trial_end", trial["trial"]))

    def search_end(self, result):
        self.search = result
        self.events.append(("search_end", result["best_trial"]))

    def test_end(self, result):
        self.test = result
        self.events.append(("test_end",))

    def finish(self, status, message, weights):
        assert self.finished is None, "finish must be called exactly once"
        self.finished = (status, message, weights)
        self.events.append(("finish", status))

    def should_stop(self):
        self.checks += 1
        return self.stop_after_checks is not None and self.checks > self.stop_after_checks


def config(**overrides):
    values = {"epochs": 1, "batch_size": 32, "device": "cpu", "seed": 0}
    values.update(overrides)
    return TrainingConfig.from_dict(values)


def run_job(graph, info, tmp_path, path=None, data_root=None, reporter=None, **settings):
    reporter = reporter or Recorder()
    job = TrainingJob(
        graph=graph,
        dataset=info,
        config=config(**settings),
        path=str(path) if path else None,
        data_root=str(data_root) if data_root else None,
    )
    result = train(job, reporter, tmp_path / "weights.pt")
    return result, reporter


@pytest.fixture(autouse=True)
def _check_stop_every_batch(monkeypatch):
    monkeypatch.setattr(engine, "STOP_CHECK_INTERVAL", 0.0)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("example_id", sorted(EXAMPLES))
def test_build_model_matches_exported_code(example_id):
    graph = EXAMPLES[example_id]["graph"]
    model = build_model(graph)
    exported = build_from_code(generate_code(graph))  # the full script incl. __main__
    assert type(model).__name__ == "NeuralNetwork"
    assert list(model.state_dict()) == list(exported.state_dict())
    exported.load_state_dict(model.state_dict())

    plan = check_compatibility(graph, random_info()).plan
    inputs, _ = data.demo_samples(
        plan, 2, torch.Generator().manual_seed(0), torch.Generator().manual_seed(1)
    )
    model.eval()
    with torch.no_grad():
        out = model(*inputs)
    outputs = out if isinstance(out, tuple) else (out,)
    assert len(outputs) == len(plan.targets)


# --------------------------------------------------------------------------- #
# Demo data
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("example_id", sorted(EXAMPLES))
def test_demo_training_completes_for_every_example(example_id, tmp_path):
    graph = EXAMPLES[example_id]["graph"]
    result, rec = run_job(graph, random_info(), tmp_path, max_samples=64, val_split=0.05)
    assert result.status == "completed", result.message
    assert rec.finished == ("completed", result.message, True)
    assert result.message.startswith("Finished: validation")
    assert (tmp_path / "weights.pt").exists()
    weights = torch.load(tmp_path / "weights.pt")
    build_from_code(generate_code(graph)).load_state_dict(weights)  # loads into the export

    assert rec.info["task"] == "demo"
    assert rec.info["device"] == "cpu" and rec.info["device_label"] == "CPU"
    assert rec.info["train_size"] == 64 and rec.info["val_size"] == 3  # 64 · 0.05 / 0.95
    assert rec.info["steps_per_epoch"] == 2
    assert rec.info["preprocessing"] == {"kind": "demo"}
    entry = rec.history[0]
    assert set(entry) == {
        "epoch",
        "train_loss",
        "val_loss",
        "train_accuracy",
        "val_accuracy",
        "seconds",
    }
    assert math.isfinite(entry["train_loss"]) and math.isfinite(entry["val_loss"])
    single_class = check_compatibility(graph, random_info()).plan.num_classes is not None
    assert (entry["val_accuracy"] is not None) == single_class
    assert rec.best == {
        "epoch": 1,
        "val_loss": entry["val_loss"],
        "val_accuracy": entry["val_accuracy"],
    }
    assert ("running", "Epoch 1 of 1") in rec.statuses


@pytest.mark.parametrize("example_id", ["mlp_mnist", "multi_input", "tabular_regression"])
def test_demo_loss_decreases(example_id, tmp_path):
    result, rec = run_job(EXAMPLES[example_id]["graph"], random_info(), tmp_path, epochs=3)
    assert result.status == "completed"
    first, last = rec.history[0], rec.history[-1]
    assert last["train_loss"] < 0.7 * first["train_loss"]
    if example_id == "mlp_mnist":
        assert last["val_accuracy"] > 0.6  # 10 classes: chance is 10 %
    else:
        assert last["val_loss"] < 0.8  # the targets have unit variance


def test_demo_data_is_seeded_and_balanced():
    plan = check_compatibility(EXAMPLES["transformer_classifier"]["graph"], random_info()).plan

    def sample():
        return data.demo_samples(
            plan, 400, torch.Generator().manual_seed(3), torch.Generator().manual_seed(4)
        )

    (tokens,), (labels,) = sample()
    (again,), _ = sample()
    assert torch.equal(tokens, again)
    assert tokens.dtype == torch.long and tokens.shape == (400, 64)
    assert int(tokens.min()) >= 0 and int(tokens.max()) < 10_000
    counts = torch.bincount(labels, minlength=4)
    assert int(counts.min()) > 60  # four roughly equal classes


def test_demo_split_follows_settings():
    plan = check_compatibility(EXAMPLES["mlp_mnist"]["graph"], random_info()).plan
    prepared = data.demo_data(plan, config(val_split=0.5, max_samples=100), torch.Generator())
    assert len(prepared.train) == 100 and len(prepared.val) == 100  # same proportion
    assert prepared.test is None
    prepared = data.demo_data(
        plan, config(val_split=0.2, test_split=0.2, max_samples=60), torch.Generator()
    )
    assert (len(prepared.train), len(prepared.val), len(prepared.test)) == (60, 20, 20)


def test_demo_data_shrinks_for_large_inputs(monkeypatch):
    plan = check_compatibility(EXAMPLES["mlp_mnist"]["graph"], random_info()).plan
    monkeypatch.setattr(data, "DEMO_BYTES", 1_000_000)  # 784 floats + label: 3,144 bytes
    rec = Recorder()
    prepared = data.demo_data(plan, config(), torch.Generator().manual_seed(0), rec)
    assert len(prepared.train) + len(prepared.val) == 318
    assert len(prepared.val) == 64
    assert rec.logs == ["The inputs are large, so the demo data has only 318 samples."]


@pytest.mark.parametrize("max_samples,steps", [(64, 2), (70, 3), (65, 2), (20, 1)])
def test_every_training_sample_is_used(max_samples, steps, tmp_path, monkeypatch):
    """Only a last batch of a single sample is skipped (BatchNorm cannot train on it)."""
    monkeypatch.setattr(engine, "PROGRESS_INTERVAL", 0.0)  # report every step
    graph = EXAMPLES["tabular_regression"]["graph"]  # contains BatchNorm1d
    result, rec = run_job(graph, random_info(), tmp_path, max_samples=max_samples)
    assert result.status == "completed", result.message
    assert rec.info["train_size"] == max_samples
    assert rec.info["steps_per_epoch"] == steps
    assert rec.steps == list(range(1, steps + 1))


def test_max_samples_shrinks_validation_in_proportion():
    gen = torch.Generator().manual_seed(0)
    train, val, test = data.split_indices(50_000, 0.2, 100, gen)
    assert (len(train), len(val), test) == (100, 25, None)
    assert set(train.tolist()).isdisjoint(val.tolist())
    train, val, _ = data.split_indices(50_000, 0.2, None, gen)
    assert (len(train), len(val)) == (40_000, 10_000)
    train, val, _ = data.split_indices(10, 0.5, 100, gen)
    assert (len(train), len(val)) == (5, 5)


def test_device_test_batch_leaves_the_model_unchanged(monkeypatch):
    graph = EXAMPLES["tabular_regression"]["graph"]  # BatchNorm1d keeps running statistics
    job = TrainingJob(graph=graph, dataset=random_info(), config=config(max_samples=64))
    rec = Recorder()
    run = engine._Run(job, rec, None)
    run._prepare()
    before = {k: v.clone() for k, v in run.model.state_dict().items()}
    run._try_mps()  # runs on the CPU here, but it is the same code path
    after = run.model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert all(p.grad is None for p in run.model.parameters())

    def unsupported(inputs):
        raise NotImplementedError("The operator 'aten::foo' is not implemented for MPS.")

    monkeypatch.setattr(run, "_forward", unsupported)
    run._try_mps()
    assert run.device.type == "cpu" and rec.info["device_label"] == "CPU"
    assert any("training on the CPU instead" in line for line in rec.logs)


def test_wrong_gradients_on_the_gpu_fall_back_to_the_cpu(monkeypatch):
    """Some Macs compute wrong MPS gradients without raising: compare with the CPU."""
    graph = EXAMPLES["mlp_mnist"]["graph"]
    job = TrainingJob(graph=graph, dataset=random_info(), config=config(max_samples=64))
    rec = Recorder()
    run = engine._Run(job, rec, None)
    run._prepare()
    before = {k: v.clone() for k, v in run.model.state_dict().items()}
    original = run._gradients
    calls = []

    def gradients(device, inputs, targets):
        calls.append(device.type)
        grads = original(device, inputs, targets)
        return grads if len(calls) == 1 else {k: g * 3 for k, g in grads.items()}

    monkeypatch.setattr(run, "_gradients", gradients)
    run._try_mps()
    assert calls[0] == "cpu" and len(calls) == 2  # CPU reference, then the GPU
    assert run.device.type == "cpu" and rec.info["device_label"] == "CPU"
    assert any("wrong gradients" in line for line in rec.logs)
    after = run.model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert run.model.training


def test_automatic_device_avoids_mps_on_intel_macs(monkeypatch):
    monkeypatch.setattr(devices.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(devices, "mps_works", lambda: True)
    monkeypatch.setattr(devices, "intel_mac", lambda: True)
    assert devices.default_device() == "cpu"
    assert devices.select_device("auto").type == "cpu"
    assert devices.available_devices()[-1] == {
        "id": "mps",
        "label": "Apple GPU (MPS, experimental on Intel Macs)",
    }
    monkeypatch.setattr(devices, "intel_mac", lambda: False)
    assert devices.default_device() == "mps"
    assert devices.available_devices()[-1] == {"id": "mps", "label": "Apple GPU (MPS)"}


# --------------------------------------------------------------------------- #
# Outcomes: stop, NaN, devices, invalid jobs
# --------------------------------------------------------------------------- #


def test_stop_saves_weights(tmp_path):
    rec = Recorder(stop_after_checks=3)
    graph = EXAMPLES["mlp_mnist"]["graph"]
    result, _ = run_job(graph, random_info(), tmp_path, reporter=rec, epochs=5)
    assert result.status == "stopped"
    assert result.message == "Stopped after 0 of 5 epochs."
    assert result.weights and (tmp_path / "weights.pt").exists()


def test_stop_before_training_has_no_weights(tmp_path):
    rec = Recorder(stop_after_checks=0)
    result, _ = run_job(EXAMPLES["mlp_mnist"]["graph"], random_info(), tmp_path, reporter=rec)
    assert result.status == "stopped"
    assert result.message == "Stopped before the training started."
    assert not result.weights and not (tmp_path / "weights.pt").exists()


def test_exploding_loss_fails(tmp_path):
    graph = EXAMPLES["tabular_regression"]["graph"]
    result, rec = run_job(
        graph, random_info(), tmp_path, epochs=3, optimizer="sgd", learning_rate=10
    )
    assert result.status == "failed"
    assert result.message.startswith("The loss became ")
    assert result.message.endswith("Try a lower learning rate.")
    assert not (tmp_path / "weights.pt").exists()
    assert rec.finished == ("failed", result.message, False)


@pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA is available here")
def test_unavailable_device_fails(tmp_path):
    result, _ = run_job(EXAMPLES["mlp_mnist"]["graph"], random_info(), tmp_path, device="cuda")
    assert result.status == "failed"
    assert result.message == "CUDA is not available on this machine."


def test_device_helpers():
    devices = engine.available_devices()
    assert devices[0] == {"id": "cpu", "label": "CPU"}
    assert engine.default_device() in {d["id"] for d in devices}
    assert engine.select_device("cpu").type == "cpu"
    assert engine.device_label("cpu") == "CPU"


def test_incompatible_job_fails_with_first_error(tmp_path):
    result, _ = run_job(EXAMPLES["lstm_sentiment"]["graph"], builtin_info("mnist"), tmp_path)
    assert result.status == "failed"
    assert "Text models can only be trained on demo data" in result.message


def test_unexpected_errors_are_reported(tmp_path, monkeypatch, capsys):
    def broken(*args, **kwargs):
        raise RuntimeError("something odd\nwith details")

    monkeypatch.setattr(engine, "prepare_data", broken)
    result, _ = run_job(EXAMPLES["mlp_mnist"]["graph"], random_info(), tmp_path)
    assert result.status == "failed"
    assert result.message == "Training failed: RuntimeError: something odd"
    assert "Traceback" in capsys.readouterr().err


def test_out_of_memory_message(tmp_path, monkeypatch):
    def oom(*args, **kwargs):
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

    monkeypatch.setattr(engine, "prepare_data", oom)
    result, _ = run_job(EXAMPLES["mlp_mnist"]["graph"], random_info(), tmp_path)
    assert result.message == (
        "Out of memory on CPU. Try a smaller batch size or a smaller network."
    )


def test_job_from_dict():
    graph = EXAMPLES["mlp_mnist"]["graph"]
    job = TrainingJob.from_dict(
        {
            "name": "MLP",
            "graph": graph,
            "dataset": {"kind": "builtin", "path": None, "info": builtin_info("mnist").to_dict()},
            "config": {"epochs": 2},
            "data_root": "/data",
        }
    )
    assert job.dataset.builtin == "mnist" and job.config.epochs == 2
    assert job.data_root == "/data" and job.path is None and job.name == "MLP"
    assert TrainingJob.from_dict({"graph": graph, "dataset": {"kind": "random"}}).dataset.kind == (
        "random"
    )
    assert job.tuning is None
    search = TrainingJob.from_dict({"graph": graph, "tuning": {"trials": 3}, "config": None})
    assert search.tuning == TuningConfig.from_dict({"trials": 3})
    for bad, message in [
        ({"dataset": {}}, "no network"),
        ({"graph": graph, "dataset": {"kind": "csv"}}, "no valid dataset"),
        ({"graph": graph, "dataset": {"info": {"x": 1}}}, "no valid dataset"),
        ({"graph": graph, "config": {"epochs": 0}}, "Epochs must be between"),
        ({"graph": graph, "tuning": {"trials": 0}}, "the number of trials must be at least 1"),
    ]:
        with pytest.raises(engine.TrainingError, match=message):
            TrainingJob.from_dict(bad)


# --------------------------------------------------------------------------- #
# CSV files
# --------------------------------------------------------------------------- #


def iris_like(n=60, labels=("setosa", "versicolor", "virginica")):
    gen = torch.Generator().manual_seed(0)
    lines = ["a,b,c,d,species"]
    for i in range(n):
        k = i % len(labels)
        x = (torch.randn(4, generator=gen) * 0.3 + k * 2).tolist()
        lines.append(",".join(f"{v:.3f}" for v in x) + f",{labels[k]}")
    return "\n".join(lines) + "\n"


def mlp(inputs, outputs):
    return chain(
        ("Input", {"shape": [inputs]}),
        ("Linear", {"out_features": 16}),
        ("Activation", {}),
        ("Linear", {"out_features": outputs}),
    )


def test_csv_classification_with_text_labels(tmp_path):
    path = tmp_path / "iris.csv"
    path.write_text(iris_like())
    info = inspect_csv(path)
    result, rec = run_job(
        mlp(4, 3), info, tmp_path, path=path, epochs=8, batch_size=8, learning_rate=0.02
    )
    assert result.status == "completed", result.message
    assert rec.info["task"] == "classification"
    assert rec.info["class_names"] == ["setosa", "versicolor", "virginica"]
    assert rec.info["train_size"] == 48 and rec.info["val_size"] == 12
    pre = rec.info["preprocessing"]
    assert pre["kind"] == "tabular"
    assert pre["columns"] == ["a", "b", "c", "d"] and pre["target"] == "species"
    assert pre["class_names"] == ["setosa", "versicolor", "virginica"]
    assert len(pre["mean"]) == 4 and len(pre["std"]) == 4
    assert pre["target_mean"] is None
    assert rec.history[-1]["val_accuracy"] > 0.9  # well separated classes
    assert "val_rmse" not in rec.history[-1]
    assert result.message.startswith("Finished: validation accuracy")


def test_csv_classification_with_integer_labels_and_semicolons(tmp_path):
    text = "x;y;label\n" + "".join(f"{i % 3},5;{i};{i % 3}\n" for i in range(30))
    path = tmp_path / "ints.csv"
    path.write_text(text)
    info = inspect_csv(path)
    result, rec = run_job(mlp(2, 3), info, tmp_path, path=path, max_samples=12)
    assert result.status == "completed", result.message
    assert rec.info["class_names"] == ["0", "1", "2"]
    assert rec.info["train_size"] == 12 and rec.info["val_size"] == 3
    assert rec.info["preprocessing"]["decimal_comma"] is True


def test_csv_regression_reports_rmse_in_original_units(tmp_path):
    gen = torch.Generator().manual_seed(1)
    x = torch.randn(80, 3, generator=gen)
    y = 1000 + 250 * (x @ torch.tensor([1.0, -2.0, 0.5]))
    lines = ["a,b,c,price"] + [
        ",".join(f"{v:.4f}" for v in row.tolist()) + f",{t:.2f}" for row, t in zip(x, y.tolist())
    ]
    path = tmp_path / "prices.csv"
    path.write_text("\n".join(lines))
    info = inspect_csv(path)
    result, rec = run_job(
        mlp(3, 1), info, tmp_path, path=path, epochs=3, batch_size=8, learning_rate=0.01
    )
    assert result.status == "completed", result.message
    assert rec.info["task"] == "regression"
    pre = rec.info["preprocessing"]
    assert pre["target_mean"] == pytest.approx(float(y.mean()), rel=0.1)
    assert pre["target_std"] > 100
    for entry in rec.history:
        assert entry["val_accuracy"] is None
        rmse = math.sqrt(entry["val_loss"]) * pre["target_std"]
        assert entry["val_rmse"] == pytest.approx(rmse)
    assert ", RMSE " in result.message


def test_csv_errors_name_line_and_column(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text(iris_like(8))
    info = inspect_csv(path)
    path.write_text(iris_like(8).replace("species\n", "species\n1,2,oops,4,setosa\n"))
    result, _ = run_job(mlp(4, 3), info, tmp_path, path=path)
    assert result.status == "failed"
    assert result.message == "Line 2, column 'c': 'oops' is not a number."


def test_csv_is_streamed_into_tensors(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "CSV_BLOCK_ROWS", 3)  # several blocks and a partial one
    lines = ["x;y;label"] + [f"{i},5;{-i};{'ab'[i % 2]}" for i in range(10)]
    path = tmp_path / "data.csv"
    path.write_text("\n".join(lines) + "\n\n")
    info = inspect_csv(path)
    plan = check_compatibility(mlp(2, 2), info).plan
    x, y, columns, delimiter = data._read_csv_tensors(path, info, plan)
    assert torch.equal(x, torch.tensor([[i + 0.5, -i] for i in range(10)]))
    assert y.tolist() == [i % 2 for i in range(10)]
    assert columns == ["x", "y", "label"] and delimiter == ";"

    for changed in (lines + ["1;2;a"], lines[:-1]):
        path.write_text("\n".join(changed) + "\n")
        with pytest.raises(engine.TrainingError, match="has changed since it was checked"):
            data._read_csv_tensors(path, info, plan)


@pytest.mark.parametrize(
    "rows,outputs,message",
    [
        (
            [f"{i},{1e39 if i == 3 else i * 2},{i % 2}" for i in range(40)],
            2,
            "Line 5, column 'b': the value is too large (numbers up to about 3.4e38 are "
            "supported).",
        ),
        (
            [f"{i},{i * 2},{(-1) ** i * 1e300}" for i in range(40)],
            1,
            "The CSV file contains values that are too large to train on.",
        ),
    ],
    ids=["feature-beyond-float32", "target-variance-overflows"],
)
def test_csv_values_that_are_too_large(rows, outputs, message, tmp_path):
    path = tmp_path / "huge.csv"
    path.write_text("a,b,y\n" + "\n".join(rows) + "\n")
    result, _ = run_job(mlp(2, outputs), inspect_csv(path), tmp_path, path=path)
    assert result.status == "failed"
    assert result.message == message


# --------------------------------------------------------------------------- #
# Image folders
# --------------------------------------------------------------------------- #


def make_png_folder(root, classes=("cats", "dogs"), per_class=6, size=(10, 8)):
    for k, name in enumerate(classes):
        folder = root / name
        folder.mkdir(parents=True)
        for i in range(per_class):
            color = (200 * k, 40 * i, 100) if k else (20, 40 * i, 220)
            Image.new("RGB", size, color).save(folder / f"{i}.png")


def small_cnn(shape, classes=2):
    return chain(
        ("Input", {"shape": shape}),
        ("Flatten", {}),
        ("Linear", {"out_features": classes}),
    )


@needs_pil
def test_image_folder_training_with_wrapper_folder(tmp_path):
    make_png_folder(tmp_path / "upload" / "pets")
    info = inspect_image_folder(tmp_path / "upload")
    assert info.subdir == "pets"
    result, rec = run_job(
        small_cnn([3, 16, 16]), info, tmp_path, path=tmp_path / "upload", epochs=2, batch_size=4
    )
    assert result.status == "completed", result.message
    pre = rec.info["preprocessing"]
    assert pre["kind"] == "image"
    assert (pre["channels"], pre["height"], pre["width"], pre["layout"]) == (3, 16, 16, "chw")
    assert pre["mean"] == [0.5] * 3 and pre["std"] == [0.5] * 3
    assert pre["class_names"] == ["cats", "dogs"]
    assert rec.info["train_size"] + rec.info["val_size"] == 12
    assert any(m.startswith("Loading images 12 / 12") for _, m in rec.statuses)


@needs_pil
def test_image_folder_grayscale_and_lazy_loading(tmp_path, monkeypatch):
    make_png_folder(tmp_path / "pets")
    info = inspect_image_folder(tmp_path / "pets")
    monkeypatch.setattr(data, "PRELOAD_BYTES", 0)
    graph = chain(
        ("Input", {"shape": [12, 12]}),
        ("LSTM", {"hidden_size": 8}),
        ("Linear", {"out_features": 2}),
    )
    result, rec = run_job(graph, info, tmp_path, path=tmp_path / "pets", batch_size=4)
    assert result.status == "completed", result.message
    assert rec.info["preprocessing"]["layout"] == "hw"
    assert any("read from disk" in line for line in rec.logs)


@needs_pil
def test_image_folder_max_samples_also_limits_validation(tmp_path):
    make_png_folder(tmp_path / "pets", per_class=15)
    info = inspect_image_folder(tmp_path / "pets")
    result, rec = run_job(
        small_cnn([1, 8, 8]), info, tmp_path, path=tmp_path / "pets", val_split=0.5, max_samples=10
    )
    assert result.status == "completed", result.message
    assert (rec.info["train_size"], rec.info["val_size"]) == (10, 10)
    assert ("preparing", "Loading images 20 / 20") in rec.statuses  # not all 30


@needs_pil
@pytest.mark.parametrize("lazy", [False, True])
def test_unreadable_image_fails_with_its_name(tmp_path, monkeypatch, lazy):
    make_png_folder(tmp_path / "pets")
    (tmp_path / "pets" / "dogs" / "broken.png").write_bytes(b"not an image")
    info = inspect_image_folder(tmp_path / "pets")
    if lazy:
        monkeypatch.setattr(data, "PRELOAD_BYTES", 0)
    result, _ = run_job(
        small_cnn([1, 8, 8]), info, tmp_path, path=tmp_path / "pets", batch_size=2, val_split=0.5
    )
    assert result.status == "failed"
    assert result.message == (
        "Cannot read the image 'dogs/broken.png'. Remove it from the dataset or replace it."
    )


@needs_pil
def test_changed_image_folder_fails(tmp_path):
    make_png_folder(tmp_path / "pets")
    info = inspect_image_folder(tmp_path / "pets")
    make_png_folder(tmp_path / "pets", classes=("birds",), per_class=2)
    result, _ = run_job(small_cnn([1, 8, 8]), info, tmp_path, path=tmp_path / "pets")
    assert result.status == "failed"
    assert "has changed" in result.message


# --------------------------------------------------------------------------- #
# Built-in datasets (fake torchvision, no downloads)
# --------------------------------------------------------------------------- #


class FakeTorchvision:
    """Stands in for ``data.open_torchvision``: small random datasets in torchvision's format."""

    def __init__(self, downloaded=True, offline=False, n=40, error=None):
        self.downloaded = downloaded
        self.offline = offline
        self.n = n
        self.error = error  # raised when opening the files
        self.calls: list[tuple[str, bool, bool]] = []

    def __call__(self, dataset_id, root, train, download):
        self.calls.append((dataset_id, train, download))
        if self.error is not None:
            raise self.error
        if not self.downloaded and not download:
            raise RuntimeError("Dataset not found. You can use download=True to download it")
        if download and self.offline:
            raise RuntimeError("Error downloading train-images-idx3-ubyte.gz")
        n = self.n if train else self.n // 2
        gen = torch.Generator().manual_seed(int(train))
        targets = torch.arange(n) % 10
        if dataset_id == "cifar10":  # numpy N×H×W×C and a list of labels, like torchvision
            images = torch.randint(0, 256, (n, 32, 32, 3), generator=gen, dtype=torch.uint8)
            return type("Fake", (), {"data": images.numpy(), "targets": targets.tolist()})
        images = torch.randint(0, 256, (n, 28, 28), generator=gen, dtype=torch.uint8)
        return type("Fake", (), {"data": images, "targets": targets})


@pytest.fixture
def fake_torchvision(monkeypatch):
    fake = FakeTorchvision()
    monkeypatch.setattr(data, "open_torchvision", fake)
    return fake


def test_builtin_mnist(tmp_path, fake_torchvision):
    result, rec = run_job(
        EXAMPLES["cnn_mnist"]["graph"], builtin_info("mnist"), tmp_path, data_root=tmp_path
    )
    assert result.status == "completed", result.message
    assert rec.info["train_size"] == 40 and rec.info["val_size"] == 20
    assert rec.info["class_names"] == [str(i) for i in range(10)]
    pre = rec.info["preprocessing"]
    assert pre["dataset"] == "mnist"
    assert pre["mean"] == list(BUILTIN_DATASETS["mnist"].mean)  # no conversion: official stats
    assert fake_torchvision.calls == [("mnist", True, False), ("mnist", False, False)]
    assert not any("Downloading" in line for line in rec.logs)


def test_builtin_download_is_logged(tmp_path, monkeypatch):
    fake = FakeTorchvision(downloaded=False)
    monkeypatch.setattr(data, "open_torchvision", fake)
    result, rec = run_job(
        EXAMPLES["mlp_mnist"]["graph"],
        builtin_info("fashion_mnist"),
        tmp_path,
        data_root=tmp_path,
        max_samples=16,
    )
    assert result.status == "completed", result.message
    assert "Downloading Fashion-MNIST (about 30 MB) …" in rec.logs
    assert ("preparing", "Downloading Fashion-MNIST …") in rec.statuses
    assert ("fashion_mnist", True, True) in fake.calls
    assert rec.info["train_size"] == 16
    assert rec.info["val_size"] == 4  # the test split shrinks with max_samples (20 %)


def test_builtin_offline(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "open_torchvision", FakeTorchvision(downloaded=False, offline=True))
    result, _ = run_job(
        EXAMPLES["mlp_mnist"]["graph"], builtin_info("mnist"), tmp_path, data_root=tmp_path
    )
    assert result.status == "failed"
    assert (
        result.message == "Could not download MNIST. Check the internet connection of the server."
    )


def test_builtin_damaged_files(tmp_path, monkeypatch, capsys):
    error = RuntimeError("shape '[60000, 28, 28]' is invalid for input of size 1000")
    fake = FakeTorchvision(error=error)
    monkeypatch.setattr(data, "open_torchvision", fake)
    result, rec = run_job(
        EXAMPLES["mlp_mnist"]["graph"], builtin_info("mnist"), tmp_path, data_root=tmp_path
    )
    assert result.status == "failed"
    assert result.message == (
        f"Could not load MNIST (RuntimeError: {error}). The downloaded files may be damaged: "
        f"delete the folder {tmp_path / 'mnist'} and try again."
    )
    assert fake.calls == [("mnist", True, False)]  # no download attempt
    assert not rec.logs
    assert "Traceback" in capsys.readouterr().err


def test_broken_torchvision_install(tmp_path, monkeypatch, capsys):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("torchvision"):
            raise RuntimeError("operator torchvision::nms does not exist")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    result, _ = run_job(
        EXAMPLES["mlp_mnist"]["graph"], builtin_info("mnist"), tmp_path, data_root=tmp_path
    )
    assert result.status == "failed"
    assert result.message.startswith(
        "torchvision cannot be loaded (RuntimeError: operator torchvision::nms does not exist)."
    )
    assert f"matches PyTorch {torch.__version__}" in result.message

    from brickdnn.training import probe

    monkeypatch.setattr(probe.importlib.util, "find_spec", lambda name: object())
    assert probe._torchvision_version() is None
    assert "torchvision cannot be imported: RuntimeError" in capsys.readouterr().err


def test_channel_stats_in_chunks(monkeypatch):
    images = torch.randint(0, 256, (50, 3, 32, 32), dtype=torch.uint8)
    transform = data.ImageTransform(
        {"channels": 1, "height": 20, "width": 20, "layout": "chw"}, [0.0], [1.0]
    )
    whole = transform.to_unit(images).double()
    monkeypatch.setattr(data, "STATS_CHUNK_BYTES", 4 * 3 * 32 * 32 * 7)  # 7 images per chunk
    mean, std = data._channel_stats(images, transform, torch.Generator())
    assert mean == pytest.approx([float(whole.mean())], rel=1e-6)
    assert std == pytest.approx([float(whole.std(unbiased=False))], rel=1e-5)


@pytest.mark.parametrize(
    "dataset,shape,expected",
    [
        ("mnist", [3, 32, 32], (4, 3, 32, 32)),  # gray -> RGB, upscaled
        ("cifar10", [1, 28, 28], (4, 1, 28, 28)),  # RGB -> gray, downscaled
        ("cifar10", [3, 16, 16], (4, 3, 16, 16)),
        ("cifar10", [3072], (4, 3072)),  # flat
        ("mnist", [28, 28], (4, 28, 28)),  # rows of pixels
        ("mnist", [784], (4, 784)),
    ],
)
def test_builtin_conversions(dataset, shape, expected, fake_torchvision):
    classes = 10
    graph = chain(
        ("Input", {"shape": shape}), ("Flatten", {}), ("Linear", {"out_features": classes})
    )
    if len(shape) == 1:
        graph = chain(("Input", {"shape": shape}), ("Linear", {"out_features": classes}))
    info = builtin_info(dataset)
    plan = check_compatibility(graph, info).plan
    prepared = data.builtin_data(
        plan, info, Path("/nowhere"), config(), Recorder(), torch.Generator().manual_seed(0)
    )
    (x,), (y,) = prepared.train.batch(torch.arange(4))
    assert tuple(x.shape) == expected
    assert x.dtype == torch.float32 and y.dtype == torch.long
    channels = shape[0] if len(shape) == 3 else BUILTIN_DATASETS[dataset].shape[0]
    pre = prepared.preprocessing
    assert len(pre["mean"]) == len(pre["std"]) == (channels if len(shape) != 2 else 1)
    if pre["mean"] == list(BUILTIN_DATASETS[dataset].mean):
        return  # unchanged images use the official statistics
    # Converted images are normalised with the statistics of the converted training set.
    (full,), _ = prepared.train.batch(torch.arange(len(prepared.train)))
    assert abs(float(full.mean())) < 0.05 and abs(float(full.std()) - 1) < 0.1


def test_image_transform_helpers():
    gray = torch.full((2, 1, 4, 4), 255, dtype=torch.uint8)
    rgb = torch.zeros((2, 3, 4, 4), dtype=torch.uint8)
    rgb[:, 1] = 255  # pure green
    to_rgb = data.ImageTransform(
        {"channels": 3, "height": 4, "width": 4, "layout": "chw"}, [0.0], [1.0]
    )
    assert torch.allclose(to_rgb(gray), torch.ones(2, 3, 4, 4))
    to_gray = data.ImageTransform(
        {"channels": 1, "height": 2, "width": 2, "layout": "flat"}, [0.0], [1.0]
    )
    out = to_gray(rgb)
    assert out.shape == (2, 4)
    assert torch.allclose(out, torch.full((2, 4), 0.587))
    assert data.apply_layout(torch.zeros(2, 1, 5, 6), "hw").shape == (2, 5, 6)


# --------------------------------------------------------------------------- #
# Test split
# --------------------------------------------------------------------------- #


def test_split_indices_with_a_test_set():
    gen = torch.Generator().manual_seed(0)
    train, val, test = data.split_indices(1000, 0.2, None, gen, test_split=0.1)
    assert (len(train), len(val), len(test)) == (700, 200, 100)
    assert len(set(train.tolist()) | set(val.tolist()) | set(test.tolist())) == 1000
    train, val, test = data.split_indices(1000, 0.2, 70, gen, test_split=0.1)
    assert (len(train), len(val), len(test)) == (70, 20, 10)  # same proportions
    train, val, test = data.split_indices(3, 0.2, None, gen, test_split=0.1)
    assert (len(train), len(val), len(test)) == (1, 1, 1)
    with pytest.raises(engine.TrainingError, match="at least 3 samples"):
        data.split_indices(2, 0.2, None, gen, test_split=0.1)


def test_without_a_test_set_the_split_is_unchanged():
    """Seeds of earlier runs still select the same training and validation samples."""
    train, val, test = data.split_indices(500, 0.2, None, torch.Generator().manual_seed(5))
    order = torch.randperm(500, generator=torch.Generator().manual_seed(5))
    assert torch.equal(val, order[:100]) and torch.equal(train, order[100:]) and test is None


def evaluate_saved_weights(graph, tmp_path, source):
    """Cross-entropy of the saved weights on a whole source (recomputed outside the engine)."""
    model = build_model(graph)
    model.load_state_dict(torch.load(tmp_path / "weights.pt"))
    model.eval()
    (x,), (y,) = source.batch(torch.arange(len(source)))
    with torch.no_grad():
        return float(torch.nn.functional.cross_entropy(model(x), y))


def test_demo_training_with_a_test_set(tmp_path):
    graph = EXAMPLES["mlp_mnist"]["graph"]
    result, rec = run_job(graph, random_info(), tmp_path, epochs=2, test_split=0.25)
    assert result.status == "completed", result.message
    sizes = (rec.info["train_size"], rec.info["val_size"], rec.info["test_size"])
    assert sizes == (1408, 512, 640)  # 55 % / 20 % / 25 % of 2,560
    assert set(rec.test) == {"loss", "accuracy", "rmse"}
    assert 0 <= rec.test["accuracy"] <= 1 and rec.test["rmse"] is None
    assert result.test == rec.test
    assert rec.events[-2:] == [("test_end",), ("finish", "completed")]
    assert ("running", "Testing the final model …") in rec.statuses
    assert result.message.startswith("Finished: validation accuracy ")
    assert "; test accuracy " in result.message
    # The test set is measured with the saved (final) weights.
    plan = check_compatibility(graph, random_info()).plan
    prepared = data.demo_data(plan, config(test_split=0.25), torch.Generator().manual_seed(0))
    loss = evaluate_saved_weights(graph, tmp_path, prepared.test)
    assert loss == pytest.approx(rec.test["loss"], rel=1e-4)


def test_no_test_set_by_default(tmp_path):
    result, rec = run_job(EXAMPLES["mlp_mnist"]["graph"], random_info(), tmp_path)
    assert result.status == "completed"
    assert rec.info["test_size"] is None and result.test is None and rec.test is None
    assert "test" not in result.message


def test_stopped_runs_are_not_tested(tmp_path):
    rec = Recorder(stop_after_checks=3)
    graph = EXAMPLES["mlp_mnist"]["graph"]
    result, _ = run_job(graph, random_info(), tmp_path, reporter=rec, test_split=0.2)
    assert result.status == "stopped" and result.test is None and rec.test is None


def test_csv_test_set_uses_the_training_statistics(tmp_path):
    gen = torch.Generator().manual_seed(1)
    x = torch.randn(80, 3, generator=gen)
    y = 1000 + 250 * (x @ torch.tensor([1.0, -2.0, 0.5]))
    lines = ["a,b,c,price"] + [
        ",".join(f"{v:.4f}" for v in row.tolist()) + f",{t:.2f}" for row, t in zip(x, y.tolist())
    ]
    path = tmp_path / "prices.csv"
    path.write_text("\n".join(lines))
    result, rec = run_job(
        mlp(3, 1), inspect_csv(path), tmp_path, path=path, epochs=3, batch_size=8, test_split=0.25
    )
    assert result.status == "completed", result.message
    assert (rec.info["train_size"], rec.info["val_size"], rec.info["test_size"]) == (44, 16, 20)
    train, _, _ = data.split_indices(80, 0.2, None, torch.Generator().manual_seed(0), 0.25)
    pre = rec.info["preprocessing"]
    assert pre["mean"] == pytest.approx(x[train].mean(dim=0).tolist(), abs=1e-3)
    assert pre["target_mean"] == pytest.approx(float(y[train].mean()), rel=1e-4)
    assert rec.test["rmse"] == pytest.approx(math.sqrt(rec.test["loss"]) * pre["target_std"])
    assert "; test loss " in result.message and result.message.count("RMSE") == 2


@needs_pil
@pytest.mark.parametrize("lazy", [False, True])
def test_image_folder_with_a_test_set(tmp_path, monkeypatch, lazy):
    if lazy:
        monkeypatch.setattr(data, "PRELOAD_BYTES", 0)
    make_png_folder(tmp_path / "pets", per_class=10)
    info = inspect_image_folder(tmp_path / "pets")
    result, rec = run_job(
        small_cnn([1, 8, 8]), info, tmp_path, path=tmp_path / "pets", batch_size=4, test_split=0.2
    )
    assert result.status == "completed", result.message
    assert (rec.info["train_size"], rec.info["val_size"], rec.info["test_size"]) == (12, 4, 4)
    assert rec.test is not None and rec.test["accuracy"] is not None
    if not lazy:
        assert ("preparing", "Loading images 20 / 20") in rec.statuses


def test_builtin_with_a_test_set_validates_on_training_images(tmp_path, fake_torchvision):
    graph = EXAMPLES["cnn_mnist"]["graph"]
    result, rec = run_job(
        graph, builtin_info("mnist"), tmp_path, data_root=tmp_path, val_split=0.25, test_split=0.1
    )
    assert result.status == "completed", result.message
    # 40 training images: 30 for training, 10 for validation; 20 official test images
    assert (rec.info["train_size"], rec.info["val_size"], rec.info["test_size"]) == (30, 10, 20)
    assert rec.test is not None
    result, rec = run_job(
        graph,
        builtin_info("mnist"),
        tmp_path,
        data_root=tmp_path,
        val_split=0.25,
        test_split=0.1,
        max_samples=12,
    )
    assert (rec.info["train_size"], rec.info["val_size"], rec.info["test_size"]) == (12, 4, 3)


def test_builtin_conversion_statistics_use_only_training_images(tmp_path, fake_torchvision):
    graph = EXAMPLES["resnet_block"]["graph"]  # 3×32×32: MNIST images are converted
    result, rec = run_job(
        graph, builtin_info("mnist"), tmp_path, data_root=tmp_path, test_split=0.2
    )
    assert result.status == "completed", result.message
    assert len(rec.info["preprocessing"]["mean"]) == 3
    assert rec.info["test_size"] == 20


# --------------------------------------------------------------------------- #
# Hyperparameter search
# --------------------------------------------------------------------------- #


def search_job(graph, tmp_path, tuning, reporter=None, **settings):
    reporter = reporter or Recorder()
    job = TrainingJob(
        graph=graph,
        dataset=random_info(),
        config=config(**settings),
        tuning=TuningConfig.from_dict(tuning),
    )
    return train(job, reporter, tmp_path / "weights.pt"), reporter


def test_search_keeps_the_best_trial(tmp_path):
    graph = EXAMPLES["mlp_mnist"]["graph"]
    tuning = {"method": "grid", "space": {"learning_rate": [1e-7, 3e-3], "batch_size": [64, 128]}}
    result, rec = search_job(graph, tmp_path, tuning, epochs=2, test_split=0.2)
    assert result.status == "completed", result.message
    assert [t["trial"] for t in result.trials] == [1, 2, 3, 4]
    assert all(t["status"] == "completed" and t["epochs_done"] == 2 for t in result.trials)
    best = min(result.trials, key=lambda t: t["val_loss"])
    assert result.best_trial == best["trial"]
    assert best["params"]["learning_rate"] == 3e-3  # a tiny learning rate learns nothing
    assert rec.search == {
        "best_trial": best["trial"],
        "params": best["params"],
        "history": result.history,
        "best": result.best,
    }
    assert result.history[-1]["val_loss"] == best["val_loss"]
    assert rec.info["tuning"] == TuningConfig.from_dict(tuning).to_dict()
    assert rec.info["trials_total"] == 4
    train_size = rec.info["train_size"]
    assert [t["steps_per_epoch"] for t in rec.trials_started] == [
        math.ceil(train_size / t["params"]["batch_size"]) for t in rec.trials_started
    ]
    events = [e for e in rec.events if e[0] != "epoch_end"]
    assert events == [
        *[(kind, n) for n in range(1, 5) for kind in ("trial_start", "trial_end")],
        ("search_end", best["trial"]),
        ("test_end",),
        ("finish", "completed"),
    ]
    ranked_so_far = [b for _, b in rec.trials_ended]
    assert ranked_so_far[0] == 1 and ranked_so_far[-1] == best["trial"]
    assert ("running", "Trial 1 of 4: epoch 1 of 2") in rec.statuses
    assert result.message.startswith(
        f"Finished 4 trials. Best: trial {best['trial']} (learning rate 0.003, batch size "
    )
    assert "; test accuracy " in result.message
    # The saved weights are those of the best trial.
    plan = check_compatibility(graph, random_info()).plan
    prepared = data.demo_data(plan, config(test_split=0.2), torch.Generator().manual_seed(0))
    loss = evaluate_saved_weights(graph, tmp_path, prepared.val)
    assert loss == pytest.approx(best["val_loss"], rel=1e-4)
    assert evaluate_saved_weights(graph, tmp_path, prepared.test) == pytest.approx(
        result.test["loss"], rel=1e-4
    )


def test_trials_start_from_the_same_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(TuningConfig, "sample", lambda self, seed: [{"learning_rate": 0.01}] * 3)
    graph = EXAMPLES["multi_input"]["graph"]
    tuning = {"space": {"learning_rate": [0.01]}}
    result, _ = search_job(graph, tmp_path, tuning, max_samples=64)
    assert result.status == "completed", result.message
    assert len(result.trials) == 3
    assert len({t["val_loss"] for t in result.trials}) == 1  # same start, same batches


@pytest.mark.parametrize(
    "error,message",
    [
        (
            engine.DivergedError("The loss became NaN in epoch 1. Try a lower learning rate."),
            "The loss became NaN in epoch 1. Try a lower learning rate.",
        ),
        (RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"), "Out of memory on CPU."),
    ],
)
def test_failed_trials_do_not_end_the_search(tmp_path, monkeypatch, error, message):
    original = engine._Run._fit

    def fit(self):
        if self.config.learning_rate == 0.5:
            raise error
        original(self)

    monkeypatch.setattr(engine._Run, "_fit", fit)
    graph = EXAMPLES["mlp_mnist"]["graph"]
    tuning = {"method": "grid", "space": {"learning_rate": [0.5, 0.01]}}
    result, rec = search_job(graph, tmp_path, tuning, max_samples=64)
    assert result.status == "completed", result.message
    failed, ok = result.trials
    assert failed["status"] == "failed" and failed["message"].startswith(message)
    assert failed["val_loss"] is None and failed["epochs_done"] == 0
    assert ok["status"] == "completed" and result.best_trial == 2
    assert result.message.startswith("Finished 2 trials (1 failed). Best: trial 2")
    assert any(line.startswith(f"Trial 1 failed: {message}") for line in rec.logs)
    assert result.weights


def test_search_fails_when_every_trial_fails(tmp_path, monkeypatch):
    def fit(self):
        raise engine.DivergedError("The loss became NaN in epoch 1. Try a lower learning rate.")

    monkeypatch.setattr(engine._Run, "_fit", fit)
    tuning = {"method": "grid", "space": {"learning_rate": [0.5, 0.1]}}
    result, _ = search_job(EXAMPLES["mlp_mnist"]["graph"], tmp_path, tuning, max_samples=64)
    assert result.status == "failed"
    assert result.message == (
        "All 2 trials failed. Trial 1: The loss became NaN in epoch 1. Try a lower learning rate."
    )
    assert not result.weights and not (tmp_path / "weights.pt").exists()


def test_unexpected_errors_end_the_search(tmp_path, monkeypatch, capsys):
    def fit(self):
        raise RuntimeError("something else broke")

    monkeypatch.setattr(engine._Run, "_fit", fit)
    tuning = {"method": "grid", "space": {"learning_rate": [0.5, 0.1]}}
    result, _ = search_job(EXAMPLES["mlp_mnist"]["graph"], tmp_path, tuning, max_samples=64)
    assert result.status == "failed"
    assert result.message == "Training failed: RuntimeError: something else broke"
    assert len(result.trials) == 1


class StopAtTrial(Recorder):
    """Asks to stop as soon as the given trial has started."""

    def __init__(self, trial):
        super().__init__()
        self.stop_at = trial
        self.stopping = False

    def trial_start(self, trial):
        super().trial_start(trial)
        self.stopping = self.stopping or trial["trial"] == self.stop_at

    def should_stop(self):
        return self.stopping


def test_stopping_a_search_keeps_the_best_finished_trial(tmp_path):
    graph = EXAMPLES["mlp_mnist"]["graph"]
    tuning = {"method": "grid", "space": {"learning_rate": [0.01, 0.001, 0.0001]}}
    result, rec = search_job(
        graph, tmp_path, tuning, reporter=StopAtTrial(2), max_samples=64, test_split=0.2
    )
    assert result.status == "stopped"
    assert [t["status"] for t in result.trials] == ["completed", "stopped"]
    assert result.best_trial == 1 and rec.search["best_trial"] == 1
    assert (
        result.message == "Stopped after 1 of 3 trials. Best so far: trial 1 (learning rate 0.01)."
    )
    assert result.weights and rec.test is None  # no test after a stop
    assert result.history == rec.search["history"] and len(result.history) == 1


def test_stopping_during_the_first_trial(tmp_path):
    tuning = {"method": "grid", "space": {"learning_rate": [0.01, 0.001]}}
    result, rec = search_job(
        EXAMPLES["mlp_mnist"]["graph"],
        tmp_path,
        tuning,
        reporter=Recorder(stop_after_checks=3),
        epochs=3,
    )
    assert result.status == "stopped"
    assert result.message == "Stopped during trial 1 of 2."
    assert result.weights and result.best_trial is None and rec.search is None


def test_state_reporter_tracks_trials(tmp_path):
    reporter = runner.StateReporter(tmp_path, epochs=2, heartbeat=0.05)
    reporter.setup({"tuning": {"method": "grid"}, "trials_total": 2})
    reporter.trial_start(
        {"trial": 1, "trials": 2, "params": {"learning_rate": 0.1}, "steps_per_epoch": 7}
    )
    state = st.read_state(tmp_path)
    assert state["trial"] == 1 and state["steps_per_epoch"] == 7
    assert state["trials"][0]["status"] == "running" and state["history"] == []
    entry = {"epoch": 1, "train_loss": 0.5, "val_loss": 0.4, "val_accuracy": 0.8, "seconds": 1}
    reporter.epoch_end(entry, {"epoch": 1, "val_loss": 0.4, "val_accuracy": 0.8})
    live = st.read_state(tmp_path)["trials"][0]
    assert live["epochs_done"] == 1 and live["val_loss"] == 0.4 and live["status"] == "running"
    done = {**live, "status": "completed", "seconds": 2.0}
    reporter.trial_end(done, 1)
    reporter.trial_start(
        {"trial": 2, "trials": 2, "params": {"learning_rate": 9}, "steps_per_epoch": 7}
    )
    state = st.read_state(tmp_path)
    assert state["history"] == [] and state["best"] is None and state["best_trial"] == 1
    assert [t["trial"] for t in state["trials"]] == [1, 2]
    reporter.trial_end({**state["trials"][1], "status": "failed", "message": "NaN"}, 1)
    reporter.search_end(
        {"best_trial": 1, "params": {"learning_rate": 0.1}, "history": [entry], "best": None}
    )
    reporter.test_end({"loss": 0.45, "accuracy": 0.79, "rmse": None})
    reporter.finish("completed", "Finished 2 trials (1 failed).", True)
    state = st.read_state(tmp_path)
    assert state["trial"] is None and state["history"] == [entry]
    assert [t["status"] for t in state["trials"]] == ["completed", "failed"]
    assert state["test"] == {"loss": 0.45, "accuracy": 0.79, "rmse": None}
    assert set(state) == set(st.initial_state())


# --------------------------------------------------------------------------- #
# Runner, probe and command line
# --------------------------------------------------------------------------- #


def subprocess_env():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(BACKEND) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    return env


def test_runner_end_to_end(tmp_path):
    run_dir = tmp_path / "run"
    job = {
        "name": "Demo",
        "graph": EXAMPLES["multi_input"]["graph"],
        "dataset": {"kind": "random", "path": None, "info": random_info().to_dict()},
        "config": {"epochs": 2, "device": "cpu", "max_samples": 64, "seed": 1, "batch_size": 16},
        "data_root": str(tmp_path / "builtin"),
    }
    st.write_job(run_dir, job)
    with open(run_dir / st.LOG_FILE, "w") as log:
        process = subprocess.run(
            [sys.executable, "-m", "brickdnn.training.runner", str(run_dir)],
            cwd=run_dir,
            env=subprocess_env(),
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            timeout=120,
        )
    output = (run_dir / st.LOG_FILE).read_text()
    assert process.returncode == 0, output
    state = st.read_state(run_dir)
    assert state["status"] == "completed", output
    assert set(state) == set(st.initial_state())
    assert state["weights"] is True and (run_dir / st.WEIGHTS_FILE).exists()
    assert state["epochs"] == 2 and len(state["history"]) == 2
    assert state["finished_at"] >= state["started_at"]
    assert state["progress"] is None
    assert state["total_params"] == 50_497
    assert state["message"].startswith("Finished: validation loss")
    assert "completed: Finished" in output


def test_runner_runs_a_search(tmp_path):
    run_dir = tmp_path / "run"
    tuning = {"method": "grid", "space": {"learning_rate": [0.01, 0.001]}}
    job = {
        "name": "Search",
        "graph": EXAMPLES["mlp_mnist"]["graph"],
        "dataset": {"kind": "random", "path": None, "info": random_info().to_dict()},
        "config": {"epochs": 1, "device": "cpu", "max_samples": 64, "seed": 1, "test_split": 0.2},
        "tuning": tuning,
        "data_root": str(tmp_path / "builtin"),
    }
    st.write_job(run_dir, job)
    process = subprocess.run(
        [sys.executable, "-m", "brickdnn.training.runner", str(run_dir)],
        cwd=run_dir,
        env=subprocess_env(),
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=120,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    state = st.read_state(run_dir)
    assert state["status"] == "completed", process.stdout
    assert set(state) == set(st.initial_state())
    assert state["tuning"] == TuningConfig.from_dict(tuning).to_dict()
    assert state["trials_total"] == 2 and state["trial"] is None
    assert [t["status"] for t in state["trials"]] == ["completed", "completed"]
    best = state["trials"][state["best_trial"] - 1]
    assert state["history"][-1]["val_loss"] == best["val_loss"]
    assert state["test_size"] == 21 and state["test"]["accuracy"] is not None
    assert state["weights"] is True and state["message"].startswith("Finished 2 trials.")


def test_runner_reports_invalid_jobs(tmp_path):
    process = subprocess.run(
        [sys.executable, "-m", "brickdnn.training.runner", str(tmp_path)],
        env=subprocess_env(),
        capture_output=True,
        timeout=120,
    )
    assert process.returncode == 0
    state = st.read_state(tmp_path)
    assert state["status"] == "failed"
    assert state["message"] == "The training job is missing or invalid."


def test_state_reporter(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "WRITE_INTERVAL", 60)
    reporter = runner.StateReporter(tmp_path, epochs=3, heartbeat=0.05)
    first = st.read_state(tmp_path)
    assert first["status"] == "starting" and first["epochs"] == 3
    reporter.status("preparing", "Loading …")
    assert st.read_state(tmp_path)["status"] == "preparing"
    reporter.progress({"epoch": 1, "step": 1, "steps": 9, "loss": 1.0, "samples_per_second": 1})
    assert st.read_state(tmp_path)["progress"] is None  # throttled …
    for _ in range(100):
        if st.read_state(tmp_path)["progress"] is not None:
            break
        time.sleep(0.02)
    assert st.read_state(tmp_path)["progress"]["step"] == 1  # … and flushed by the heartbeat
    for i in range(st.MAX_LOG_LINES + 5):
        reporter.log(f"line {i}")
    assert not reporter.should_stop()
    st.request_stop(tmp_path)
    assert reporter.should_stop()
    reporter.finish("stopped", "Stopped after 0 of 3 epochs.", False)
    state = st.read_state(tmp_path)
    assert state["status"] == "stopped" and state["finished_at"] is not None
    assert len(state["log"]) == st.MAX_LOG_LINES and state["log"][-1] == "line 204"
    assert not reporter._thread.is_alive()


def test_probe_prints_json():
    process = subprocess.run(
        [sys.executable, "-m", "brickdnn.training.probe"],
        env=subprocess_env(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert process.returncode == 0, process.stderr
    result = json.loads(process.stdout)
    assert result["torch_version"] == torch.__version__
    assert isinstance(result["torchvision"], bool)
    assert result["devices"][0] == {"id": "cpu", "label": "CPU"}
    assert result["default_device"] in {d["id"] for d in result["devices"]}


def test_cli_train(tmp_path, capsys):
    save = tmp_path / "mlp.pt"
    code = brickdnn_main(
        ["train", "--example", "mlp_mnist", "--data", "random", "--epochs", "1"]
        + ["--save", str(save), "--device", "cpu", "--max-samples", "64", "--seed", "0"]
    )
    out = capsys.readouterr().out
    assert code == 0, out
    assert save.exists()
    assert "Training on CPU: 64 training and 16 validation samples" in out
    assert "Epoch   1/1  train loss " in out and "val acc " in out
    assert f"Saved weights to {save}" in out
    model = build_from_code(generate_code(EXAMPLES["mlp_mnist"]["graph"]))
    model.load_state_dict(torch.load(save))


@needs_pil
def test_cli_train_on_zip(tmp_path, capsys, monkeypatch):
    make_png_folder(tmp_path / "src" / "pets", per_class=3)
    archive = tmp_path / "pets.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for path in (tmp_path / "src").rglob("*.png"):
            zf.write(path, path.relative_to(tmp_path / "src").as_posix())
    graph = tmp_path / "net.json"
    graph.write_text(json.dumps(small_cnn([1, 8, 8])))
    monkeypatch.chdir(tmp_path)
    code = cli.main([str(graph), "--data", str(archive), "--epochs", "1", "--device", "cpu"])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "Extracted 6 images from pets.zip." in out
    assert (tmp_path / "net.pt").exists()  # default name: graph file stem


def test_cli_reports_incompatible_data(tmp_path, capsys):
    code = cli.main(
        ["--example", "lstm_sentiment", "--data", "mnist", "--save", str(tmp_path / "x")]
    )
    assert code == 1
    assert "error: This network expects integer token ids" in capsys.readouterr().err
    assert cli.main(["--example", "mlp_mnist", "--data", str(tmp_path / "missing.csv")]) == 1
    assert "neither a dataset" in capsys.readouterr().err
    assert cli.main(["--example", "mlp_mnist", "--epochs", "0"]) == 1
    assert "Epochs must be between" in capsys.readouterr().err


def test_cli_prints_warnings_and_notes(tmp_path, capsys, fake_torchvision):
    code = cli.main(
        ["--example", "resnet_block", "--data", "mnist", "--max-samples", "10", "--epochs", "1"]
        + ["--device", "cpu", "--save", str(tmp_path / "r.pt"), "--data-root", str(tmp_path)]
    )
    out = capsys.readouterr().out
    assert code == 0, out
    assert "warning: MNIST images (28×28) are resized to 32×32." in out
    assert "note: The grayscale images are copied into 3 channels." in out


def test_cli_search_with_a_test_set(tmp_path, capsys):
    save = tmp_path / "best.pt"
    code = cli.main(
        ["--example", "mlp_mnist", "--epochs", "1", "--device", "cpu", "--max-samples", "64"]
        + ["--seed", "0", "--test-split", "0.2", "--search", "grid", "--tune-lr", "0.01,0.001"]
        + ["--tune-optimizer", "adam,sgd", "--save", str(save)]
    )
    out = capsys.readouterr().out
    assert code == 0, out
    assert "64 training, 21 validation and 21 test samples" in out
    assert "Hyperparameter search: grid search, 4 trials of 1 epoch each." in out
    assert "Trial 1/4: learning rate 0.01, Adam" in out
    assert "Trial 4/4: learning rate 0.001, SGD" in out
    assert "(best so far)" in out
    assert "Trials, best first:" in out
    assert "trial  learning rate  optimizer  val loss  val acc" in out
    assert "Best: trial " in out and "Test: loss " in out
    assert "Finished 4 trials. Best: trial " in out
    assert f"Saved weights to {save}" in out and save.exists()


def test_cli_tuning_options():
    parser = cli.build_parser()
    assert cli.tuning_from_args(parser.parse_args(["--example", "mlp_mnist"])) is None
    tuning = cli.tuning_from_args(parser.parse_args(["--trials", "5"]))
    assert tuning.trials == 5 and tuning.space == TuningConfig().space
    assert cli.tuning_from_args(parser.parse_args(["--tune"])) == TuningConfig()
    args = parser.parse_args(
        ["--tune-lr", "1e-4:1e-2", "--tune-batch-size", "16, 32", "--tune-optimizer", "adam,SGD"]
        + ["--tune-weight-decay", "0,1e-4"]
    )
    assert cli.tuning_from_args(args).to_dict()["space"] == {
        "learning_rate": {"min": 1e-4, "max": 1e-2},
        "batch_size": [16, 32],
        "optimizer": ["adam", "sgd"],
        "weight_decay": [0.0, 1e-4],
    }


@pytest.mark.parametrize(
    "args,message",
    [
        (["--tune-batch-size", "0,32"], "batch sizes must be at least 1"),
        (["--search", "grid", "--tune-lr", "1e-4:1e-2"], "Grid search needs a list of values"),
        (["--tune-lr", "fast"], "'fast' is not a number"),
        (["--trials", "0"], "the number of trials must be at least 1"),
        (["--test-split", "0.6"], "Test split must be at least 0 and at most 0.5."),
    ],
)
def test_cli_search_errors(args, message, capsys):
    assert cli.main(["--example", "mlp_mnist", *args]) == 1
    assert message in capsys.readouterr().err


def test_cli_without_torch(monkeypatch, capsys):
    monkeypatch.setattr(cli, "torch_installed", lambda: False)
    assert brickdnn_main(["train", "--example", "mlp_mnist"]) == 1
    assert 'pip install "brickdnn[train]"' in capsys.readouterr().err


@pytest.mark.skipif(
    threading.current_thread() is not threading.main_thread(),
    reason="signal handlers need the main thread",
)
def test_cli_ctrl_c_stops_gracefully(tmp_path, capsys, monkeypatch):
    original = signal.getsignal(signal.SIGINT)
    calls = {"n": 0}

    def should_stop(self):
        calls["n"] += 1
        if calls["n"] == 3:
            signal.raise_signal(signal.SIGINT)  # like pressing Ctrl+C
        return self.stop

    monkeypatch.setattr(cli.ConsoleReporter, "should_stop", should_stop)
    save = tmp_path / "stopped.pt"
    code = cli.main(
        ["--example", "mlp_mnist", "--epochs", "3", "--device", "cpu", "--save", str(save)]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "Stopping after the current batch" in out
    assert "Stopped after 0 of 3 epochs." in out and save.exists()
    assert signal.getsignal(signal.SIGINT) is original


def test_existing_cli_is_unchanged(tmp_path, capsys):
    assert brickdnn_main(["--example", "mlp_mnist", "--check"]) == 0
    assert "OK - 235,146 trainable parameters" in capsys.readouterr().out
    out = tmp_path / "model.py"
    assert brickdnn_main(["--example", "cnn_mnist", "-o", str(out)]) == 0
    assert "class NeuralNetwork(nn.Module)" in out.read_text()
