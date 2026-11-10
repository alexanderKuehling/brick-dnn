"""Explanations of trained models (brickdnn.training.explain) against cases whose answer is
known: linear models (their attributions are exact), a feature that alone decides the
target, a tiny CNN that learned a planted pattern, and decision boundaries that must equal
the model's own predictions. No database and no processes: the methods run in this one.
"""

from __future__ import annotations

import base64
import io
import json
import math
from pathlib import Path

import numpy as np
import pytest

from brickdnn import EXAMPLES
from brickdnn.training import torch_installed
from brickdnn.training.compat import check_compatibility
from brickdnn.training.datasets import inspect_csv
from brickdnn.training.explain import ExplainError, explain, render
from brickdnn.training.explain.attribution import permutation_importance, shapley_values
from brickdnn.training.explain.data import POOL_FILE, find_seed, load_pool
from brickdnn.training.job import TrainingJob
from brickdnn.training.reporting import Reporter
from brickdnn.training.state import write_json_atomic, write_state

from .helpers import NEURAL_EXAMPLES, chain

needs_torch = pytest.mark.skipif(not torch_installed(), reason="PyTorch is not installed")


# --------------------------------------------------------------------------- #
# Model-agnostic methods (NumPy only)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("features", [3, 12])
def test_shapley_values_of_a_linear_model_are_its_terms(features):
    """f(x) = w·x + b: feature i contributes exactly w_i (x_i - baseline_i), whether the
    values are exact (few features) or estimated from random orders (many)."""
    rng = np.random.default_rng(1)
    w, b = rng.normal(size=features), 0.7
    x, baseline = rng.normal(size=features), rng.normal(size=features)

    def predict(rows):
        return (rows @ w + b)[:, None]

    result = shapley_values(predict, x, baseline)
    assert result["exact"] is (features <= 10)
    np.testing.assert_allclose(result["values"], w * (x - baseline), atol=1e-9)
    assert result["base"] == pytest.approx(baseline @ w + b)
    assert result["prediction"] == pytest.approx(x @ w + b)


def test_shapley_values_share_an_interaction_and_add_up():
    """f = x0·x1 + x2 against a zero baseline: x0 and x1 share their product equally."""

    def predict(rows):
        return (rows[:, 0] * rows[:, 1] + rows[:, 2])[:, None]

    x = np.array([2.0, 3.0, 5.0])
    result = shapley_values(predict, x, np.zeros(3))
    np.testing.assert_allclose(result["values"], [3.0, 3.0, 5.0])
    assert result["values"].sum() == pytest.approx(result["prediction"] - result["base"])


def test_permutation_importance_finds_the_one_feature_that_matters():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 4))
    y = 3 * x[:, 2] + 0.1 * rng.normal(size=400)

    def predict(rows):  # the true rule: only feature 2
        return (3 * rows[:, 2])[:, None]

    result = permutation_importance(predict, x, y, classification=False, repeats=3)
    assert result["baseline"] > 0.99 and result["repeats"] == 3
    assert result["mean"][2] > 1.5  # shuffled, R² falls far below 0
    np.testing.assert_allclose(np.delete(result["mean"], 2), 0.0)
    np.testing.assert_allclose(np.delete(result["std"], 2), 0.0)

    labels = (x[:, 0] > 0).astype(int)

    def classify(rows):
        return np.stack([rows[:, 0] <= 0, rows[:, 0] > 0], axis=1).astype(float)

    result = permutation_importance(classify, x, labels, classification=True)
    assert result["baseline"] == 1.0
    assert 0.35 < result["mean"][0] < 0.65  # shuffled: about half are right by chance
    assert list(result["mean"][1:]) == [0.0, 0.0, 0.0]


def test_pictures_and_numbers():
    from PIL import Image

    gray = np.arange(12, dtype=np.uint8).reshape(3, 4) * 20
    rgba = render.heat_rgba(np.linspace(0, 1, 6).reshape(2, 3))
    for pixels, mode in (
        (gray, "L"),
        (rgba, "RGBA"),
        (render.diverging_rgb(np.zeros((2, 2))), "RGB"),
    ):
        url = render.data_url(pixels)
        image = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
        assert image.mode == mode and image.size == (pixels.shape[1], pixels.shape[0])
        assert np.array_equal(np.asarray(image), pixels)
    assert render.diverging_rgb(np.zeros(1))[0].tolist() == [247, 247, 247]  # 0 is white
    assert render.grid(np.ones((28, 28)), 14) == [[1.0] * 14] * 14
    assert render.grid(np.arange(4.0).reshape(2, 2)) == [[0.0, 1.0], [2.0, 3.0]]
    assert render.normalise(np.zeros((2, 2))).tolist() == [[0, 0], [0, 0]]
    histogram = render.histogram(np.array([0.0, 1.0, 1.0, 2.0]), bins=2)
    assert histogram["counts"] == [1, 3] and histogram["edges"] == [0.0, 1.0, 2.0]
    assert render.sig(123456.7) == 123500 and render.sig(float("nan")) is None


# --------------------------------------------------------------------------- #
# Run directories
# --------------------------------------------------------------------------- #


def run_dir(tmp_path: Path, graph, info, path=None, config=None, log=None) -> Path:
    directory = tmp_path / "run"
    directory.mkdir()
    job = {
        "graph": graph,
        "dataset": {"kind": info.kind, "path": str(path) if path else None, "info": info.to_dict()},
        "config": config or {"seed": 3},
    }
    write_json_atomic(directory / "job.json", job)
    write_state(directory, {"status": "completed", "log": log or []})
    return directory


def ask(directory: Path, plan, preprocessing, **request):
    return explain(directory, {"plan": plan, "preprocessing": preprocessing, **request})


def test_the_seed_comes_from_the_settings_or_the_log(tmp_path):
    from brickdnn.training.config import TrainingConfig

    write_state(tmp_path, {"log": ["Loading …", "Random seed: 1234", "Epoch 1"]})
    assert find_seed(tmp_path, TrainingConfig(seed=7)) == 7
    assert find_seed(tmp_path, TrainingConfig()) == 1234
    write_state(tmp_path, {"log": []})
    (tmp_path / "output.log").write_text("Random seed: 99\n")
    assert find_seed(tmp_path, TrainingConfig()) == 99
    (tmp_path / "output.log").unlink()
    assert find_seed(tmp_path, TrainingConfig()) is None


def moons_csv(path: Path, rows: int = 240) -> Path:
    rng = np.random.default_rng(5)
    lines = ["x1,x2,label"]
    for i in range(rows):
        t = rng.uniform(0, math.pi)
        c = i % 2
        x, y = (math.cos(t), math.sin(t)) if c == 0 else (1 - math.cos(t), 0.5 - math.sin(t))
        lines.append(f"{x + rng.normal(0, 0.1):.4f},{y + rng.normal(0, 0.1):.4f},{'ab'[c]}")
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.mark.parametrize("estimator", ["SVC", "GaussianProcessClassifier"])
def test_decision_boundary_of_a_classical_model_is_its_prediction(tmp_path, estimator):
    """The boundary's grid holds the model's own predictions; the validation rows are those
    of the training (same split)."""
    from brickdnn.training.classical_engine import _ClassicalRun

    csv = moons_csv(tmp_path / "moons.csv", rows=120)
    info = inspect_csv(csv)
    graph = chain(("Input", {"shape": [2]}), (estimator, {}))
    config = {"seed": 11, "val_split": 0.25}
    directory = run_dir(tmp_path, graph, info, csv, config)
    job = TrainingJob.from_dict(json.loads((directory / "job.json").read_text()))
    fitted = _ClassicalRun(job, Reporter(), directory / "model.joblib", False)
    fitted._execute()
    plan = check_compatibility(graph, info, job.config).plan.to_dict()
    preprocessing = fitted._preprocessing()

    answer = ask(directory, plan, preprocessing, method="boundary")
    assert answer["classes"] == ["a", "b"] and answer["features"] == ["x1", "x2"]
    n = answer["resolution"]
    xs = np.linspace(*answer["x"], n)[2::4]
    ys = np.linspace(*answer["y"], n)[::-1][2::4]
    grid = np.array([[x, y] for y in ys for x in xs])
    expected = fitted.pipeline.predict(grid).reshape(len(ys), len(xs))
    agree = (np.array(answer["grid"]) == expected).mean()
    assert agree > 0.97  # the corners of the range are rounded to 4 digits
    assert len(answer["points"]) == min(len(fitted.data.train), 300)

    # Validation rows: those of the training, read again.
    pool = load_pool(directory, job, preprocessing)
    assert set(pool["rows"].tolist()) == set(fitted.data.val.tolist())
    np.testing.assert_allclose(pool["val_x"], fitted.data.x[np.sort(fitted.data.val)])
    assert (directory / "explain" / POOL_FILE).is_file()

    samples = ask(directory, plan, preprocessing, method="samples")
    assert samples["columns"] == ["x1", "x2"] and samples["target"] == "label"
    first = samples["samples"][0]
    assert first["label"] in ("a", "b") and 0.5 <= first["probability"] <= 1

    answer = ask(directory, plan, preprocessing, method="tabular", sample={"index": 0})
    total = sum(f["shapley"] for f in answer["features"])
    assert total == pytest.approx(answer["value"] - answer["base_value"], abs=2e-3)
    assert "integrated_gradients" not in answer["features"][0]  # networks only

    with pytest.raises(ExplainError, match="no layers"):
        ask(directory, plan, preprocessing, method="weights")


def test_gaussian_process_regression_reports_its_uncertainty(tmp_path):
    from brickdnn.training.classical_engine import _ClassicalRun

    rng = np.random.default_rng(2)
    lines = ["a,b,y"] + [
        f"{a:.4f},{b:.4f},{math.sin(2 * a) * 3 + 0.05 * rng.normal():.4f}"
        for a, b in rng.uniform(-2, 2, size=(120, 2))
    ]
    csv = tmp_path / "wave.csv"
    csv.write_text("\n".join(lines) + "\n")
    info = inspect_csv(csv)
    graph = EXAMPLES["gp_regression"]["graph"]
    graph = {
        **graph,
        "nodes": [
            {**n, "params": {"shape": [2]}} if n["type"] == "Input" else n for n in graph["nodes"]
        ],
    }
    directory = run_dir(tmp_path, graph, info, csv, {"seed": 1})
    job = TrainingJob.from_dict(json.loads((directory / "job.json").read_text()))
    fitted = _ClassicalRun(job, Reporter(), directory / "model.joblib", False)
    fitted._execute()
    plan = check_compatibility(graph, info, job.config).plan.to_dict()
    preprocessing = fitted._preprocessing()

    answer = ask(directory, plan, preprocessing, method="boundary")
    assert answer["std"]["range"][0] >= 0 and answer["value_range"][1] > 2
    answer = ask(directory, plan, preprocessing, method="tabular", sample={"index": 0})
    assert answer["prediction"]["std"] > 0
    assert answer["features"][0]["name"] == "a"  # the target depends on a only
    answer = ask(directory, plan, preprocessing, method="permutation")
    assert answer["metric"] == "r2" and answer["features"][0]["name"] == "a"
    assert answer["features"][0]["mean"] > 0.5 and abs(answer["features"][1]["mean"]) < 0.1


# --------------------------------------------------------------------------- #
# Neural networks
# --------------------------------------------------------------------------- #


@needs_torch
@pytest.mark.parametrize("example", NEURAL_EXAMPLES)
def test_the_traced_network_computes_what_the_generated_one_does(example):
    import torch

    from brickdnn.training.engine import build_model
    from brickdnn.training.explain.neural import build_traced_model

    graph = EXAMPLES[example]["graph"]
    traced, analysis = build_traced_model(graph)
    model = build_model(graph)
    traced.load_state_dict(model.state_dict())  # the same attribute names
    model.eval()
    traced.eval()
    torch.manual_seed(0)
    inputs = []
    for nid in analysis.inputs:
        info = analysis.nodes[nid]
        if info.dtype == "long":
            inputs.append(torch.randint(0, 10, (2, *info.shape)))
        else:
            inputs.append(torch.randn(2, *info.shape))
    with torch.no_grad():
        expected, got = model(*inputs), traced(*inputs)
    expected = expected if isinstance(expected, tuple) else (expected,)
    got = got if isinstance(got, tuple) else (got,)
    for a, b in zip(expected, got, strict=True):
        torch.testing.assert_close(a, b)
    assert set(traced._brick_outputs) == set(analysis.order)
    for nid in analysis.order:
        assert tuple(traced._brick_outputs[nid].shape[1:]) == tuple(analysis.nodes[nid].shape)


def linear_network(tmp_path, weights, bias, csv_rows=20):
    """A trained regression network y = w·x + b on standardised features, with its run."""
    import torch

    from brickdnn.training.engine import build_model

    n = len(weights)
    graph = chain(("Input", {"shape": [n]}), ("Linear", {"out_features": 1}))
    header = ",".join(f"f{i}" for i in range(n)) + ",y"
    rows = [",".join(str(float(v)) for v in np.arange(n) + r) + f",{r}" for r in range(csv_rows)]
    csv = tmp_path / "linear.csv"
    csv.write_text("\n".join([header, *rows]) + "\n")
    info = inspect_csv(csv)
    directory = run_dir(tmp_path, graph, info, csv)
    model = build_model(graph)
    with torch.no_grad():
        model.state_dict()["linear_1.weight"].copy_(torch.tensor([weights]))
        model.state_dict()["linear_1.bias"].fill_(bias)
    torch.save(model.state_dict(), directory / "weights.pt")
    plan = check_compatibility(graph, info).plan.to_dict()
    preprocessing = {
        "kind": "tabular",
        "columns": [f"f{i}" for i in range(n)],
        "target": "y",
        "mean": [1.0] * n,
        "std": [2.0] * n,
        "target_mean": 10.0,
        "target_std": 4.0,
        "class_names": None,
    }
    return directory, plan, preprocessing


@needs_torch
def test_attributions_of_a_linear_network_are_exact(tmp_path):
    import torch

    from brickdnn.training.explain.neural import integrated_gradients

    weights, bias = [0.5, -2.0, 1.5], 0.25
    directory, plan, preprocessing = linear_network(tmp_path, weights, bias)
    raw = [3.0, 0.0, 1.0]
    z = (np.array(raw) - 1.0) / 2.0
    tensor = {"dtype": "float", "shape": [1, 3], "values": z.tolist()}
    answer = ask(directory, plan, preprocessing, method="tabular", sample={"tensor": tensor})
    # In the units of the target: w_i · z_i · target_std, against the training mean (z = 0).
    expected = {f"f{i}": w * zi * 4.0 for i, (w, zi) in enumerate(zip(weights, z, strict=True))}
    for feature in answer["features"]:
        assert feature["shapley"] == pytest.approx(expected[feature["name"]], rel=1e-3)
        assert feature["integrated_gradients"] == pytest.approx(expected[feature["name"]], rel=1e-3)
    assert answer["base_value"] == pytest.approx(bias * 4 + 10, rel=1e-3)
    assert answer["prediction"]["value"] == pytest.approx((z @ weights + bias) * 4 + 10, rel=1e-4)
    assert [f["name"] for f in answer["features"]] == ["f1", "f0", "f2"]  # largest first
    assert answer["sample"]["values"] == pytest.approx(raw)

    # Integrated gradients of a non-linear function add up to the difference.
    def score(rows):
        return torch.sin(rows).sum(dim=1) + rows[:, 0] * rows[:, 1]

    x, base = torch.tensor([[1.0, 2.0]]), torch.zeros(1, 2)
    attributions = integrated_gradients(score, x, base, steps=64)
    assert attributions.sum() == pytest.approx(float(score(x) - score(base)), abs=1e-3)

    weights_answer = ask(directory, plan, preprocessing, method="weights")
    layer = weights_answer["layers"][0]
    assert layer["params"][0]["histogram"]["count"] == 3 and weights_answer["filters"] is None
    activations = ask(
        directory, plan, preprocessing, method="activations", sample={"tensor": tensor}
    )
    assert [n["shape"] for n in activations["nodes"]] == [[3], [1]]
    assert activations["nodes"][1]["view"]["value"] == pytest.approx(
        float(z @ weights + bias), rel=1e-3
    )


def planted_cnn(tmp_path):
    """A tiny CNN trained to tell images with a bright 3×3 square (anywhere) from noise."""
    import torch

    from brickdnn.training.explain.neural import build_traced_model

    graph = chain(
        ("Input", {"shape": [1, 12, 12]}),
        ("Conv2d", {"out_channels": 6, "kernel_size": [3, 3]}),
        ("Activation", {"function": "relu"}),
        ("Conv2d", {"out_channels": 6, "kernel_size": [3, 3]}),
        ("Activation", {"function": "relu"}),
        ("AdaptiveAvgPool2d", {"output_size": [1, 1]}),
        ("Flatten", {}),
        ("Linear", {"out_features": 2}),
    )
    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(0)

    def batch(size):
        x = torch.rand((size, 1, 12, 12), generator=generator) * 0.3
        y = torch.randint(0, 2, (size,), generator=generator)
        for i in torch.nonzero(y).flatten().tolist():
            top, left = torch.randint(0, 10, (2,), generator=generator).tolist()
            x[i, 0, top : top + 3, left : left + 3] = 1.0
        return (x - 0.5) / 0.25, y

    model, _ = build_traced_model(graph)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    for _ in range(150):
        x, y = batch(64)
        loss = torch.nn.functional.cross_entropy(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    x, y = batch(200)
    with torch.no_grad():
        accuracy = (model(x).argmax(dim=1) == y).float().mean().item()
    assert accuracy > 0.95

    from brickdnn.training.datasets import DatasetInfo

    info = DatasetInfo(
        kind="images", name="squares", num_samples=10, class_names=["noise", "square"]
    )
    directory = run_dir(tmp_path, graph, info)
    torch.save(model.state_dict(), directory / "weights.pt")
    plan = {
        "task": "classification",
        "inputs": [{"shape": [1, 12, 12], "dtype": "float"}],
        "targets": [{"kind": "class", "num_classes": 2}],
        "class_names": ["noise", "square"],
    }
    preprocessing = {
        "kind": "image",
        "channels": 1,
        "height": 12,
        "width": 12,
        "layout": "chw",
        "mean": [0.5],
        "std": [0.25],
        "class_names": ["noise", "square"],
    }
    return directory, plan, preprocessing


def peak(grid) -> tuple[int, int]:
    array = np.asarray(grid)
    return tuple(int(v) for v in np.unravel_index(np.argmax(array), array.shape))


@needs_torch
def test_image_explanations_find_a_planted_square(tmp_path):
    directory, plan, preprocessing = planted_cnn(tmp_path)
    image = np.full((12, 12), 0.15)
    image[7:10, 2:5] = 1.0  # rows 7-9, columns 2-4
    values = ((image - 0.5) / 0.25).ravel().tolist()
    tensor = {"dtype": "float", "shape": [1, 1, 12, 12], "values": values}
    answer = ask(directory, plan, preprocessing, method="image", sample={"tensor": tensor})

    assert answer["prediction"]["class"] == "square" and answer["target"]["index"] == 1
    assert answer["size"] == [12, 12] and answer["sample"]["source"] == "input"
    gradcam = answer["gradcam"]
    assert gradcam["layer"]["id"] == "n3"  # the last convolution
    assert [layer["id"] for layer in gradcam["layers"]] == ["n1", "n2", "n3", "n4"]
    assert len(gradcam["grid"]) == 12 and len(gradcam["grid"][0]) == 12
    for method in ("saliency", "gradcam", "occlusion"):
        row, col = peak(answer[method]["grid"])
        assert 6 <= row <= 10 and 1 <= col <= 5, (method, row, col)
        assert answer[method]["png"].startswith("data:image/png;base64,")
    assert answer["occlusion"]["largest_drop"] > 0.01

    # Another layer, and the other class.
    answer = ask(
        directory,
        plan,
        preprocessing,
        method="image",
        sample={"tensor": tensor},
        layer="n1",
        target=0,
    )
    assert answer["gradcam"]["layer"]["id"] == "n1" and answer["target"]["class"] == "noise"
    with pytest.raises(ExplainError, match="feature maps"):
        ask(directory, plan, preprocessing, method="image", sample={"tensor": tensor}, layer="n6")

    activations = ask(
        directory, plan, preprocessing, method="activations", sample={"tensor": tensor}
    )
    views = {node["id"]: node["view"] for node in activations["nodes"]}
    assert views["n1"]["kind"] == "maps" and views["n1"]["channels"] == 6
    assert len(views["n1"]["channel_means"]) == 6
    assert views["n7"]["kind"] == "vector" and views["n0"]["kind"] == "maps"
    assert activations["prediction"]["class"] == "square"

    weights = ask(directory, plan, preprocessing, method="weights")
    assert weights["filters"]["shape"] == [6, 1, 3, 3] and weights["filters"]["rgb"] is False
    assert [layer["id"] for layer in weights["layers"]] == ["n1", "n3", "n7"]

    # No validation images without the dataset.
    with pytest.raises(ExplainError, match="not available"):
        ask(directory, plan, preprocessing, method="image", sample={"index": 0})


@needs_torch
def test_validation_rows_of_a_network_are_those_of_the_training(tmp_path):
    import torch

    from brickdnn.training.config import TrainingConfig
    from brickdnn.training.data import csv_data

    csv = moons_csv(tmp_path / "moons.csv", rows=300)
    info = inspect_csv(csv)
    graph = chain(("Input", {"shape": [2]}), ("Linear", {"out_features": 2}))
    config = {"val_split": 0.3, "max_samples": 150}
    directory = run_dir(tmp_path, graph, info, csv, config, log=["Random seed: 4242"])
    plan = check_compatibility(graph, info).plan
    data = csv_data(
        plan,
        info,
        csv,
        TrainingConfig.from_dict(config),
        Reporter(),
        torch.Generator().manual_seed(4242),
    )
    job = TrainingJob.from_dict(json.loads((directory / "job.json").read_text()))
    pool = load_pool(directory, job, data.preprocessing)
    assert set(pool["rows"].tolist()) == set(data.val.indices.tolist())
    mean, std = np.array(data.preprocessing["mean"]), np.array(data.preprocessing["std"])
    expected_x = data.val.source.inputs[0][torch.as_tensor(np.sort(data.val.indices.numpy()))]
    np.testing.assert_allclose((pool["val_x"] - mean) / std, expected_x.numpy(), atol=1e-5)
    assert set(pool["val_y"].tolist()) <= {0, 1}

    # Train a little and explain: everything works on the reread rows.
    from brickdnn.training.engine import build_model

    torch.manual_seed(0)
    model = build_model(graph)
    torch.save(model.state_dict(), directory / "weights.pt")
    answer = ask(directory, plan.to_dict(), data.preprocessing, method="permutation")
    assert answer["rows"] == len(data.val) and answer["metric"] == "accuracy"
    answer = ask(directory, plan.to_dict(), data.preprocessing, method="boundary")
    assert np.array(answer["grid"]).shape == (16, 16)


@needs_torch
def test_demo_runs_show_activations_of_a_random_input(tmp_path):
    import torch

    from brickdnn.training.datasets import random_info
    from brickdnn.training.engine import build_model

    graph = EXAMPLES["lstm_sentiment"]["graph"]
    directory = run_dir(tmp_path, graph, random_info())
    torch.save(build_model(graph).state_dict(), directory / "weights.pt")
    plan = check_compatibility(graph, random_info()).plan.to_dict()
    answer = ask(directory, plan, {"kind": "demo"}, method="activations")
    assert answer["sample"]["source"] == "random"
    assert any("demo data" in note for note in answer["notes"])
    kinds = {node["type"]: node["view"]["kind"] for node in answer["nodes"]}
    assert kinds["Embedding"] == "matrix" and kinds["LSTM"] == "vector"
    with pytest.raises(ExplainError, match="demo data"):
        ask(directory, plan, {"kind": "demo"}, method="permutation")
