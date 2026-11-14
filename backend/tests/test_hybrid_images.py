"""Hybrid networks on images: a small CNN (Conv → Pool → Flatten → Linear(16)) learns the
features, a support vector machine or gradient boosting is fitted on them. On MNIST in
torchvision's format (fake data, nothing is downloaded) through the runner, and on an
uploaded folder of generated images through the REST API with the real processes: both
phases, the evaluation, predictions on an image and the download."""

# ruff: noqa: F811 (the fixtures are imported from test_training_api)

from __future__ import annotations

import base64
import io
import json
import subprocess
import sys
import zipfile

import pytest

pytest.importorskip("sklearn")
torch = pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")

import joblib  # noqa: E402
import numpy as np  # noqa: E402

from brickdnn import analyze, generate_code  # noqa: E402
from brickdnn.training import check_compatibility, data, inference, predict, runner  # noqa: E402
from brickdnn.training import state as st  # noqa: E402

from .test_training_api import (  # noqa: E402, F401 (fixtures)
    auth_client,
    training_dir,
    upload,
    user,
    wait_for,
)
from .test_training_engine import FakeTorchvision  # noqa: E402

QUICK_BOOSTING = {"max_iter": 20, "min_samples_leaf": 3, "early_stopping": "off"}


def cnn_hybrid(model, shape, params=None):
    """Input → Conv2d(4) → MaxPool2d → Flatten → Linear(16) → model."""
    nodes = [
        ("in", "Input", {"shape": shape}),
        ("conv", "Conv2d", {"out_channels": 4}),
        ("pool", "MaxPool2d", {}),
        ("flat", "Flatten", {}),
        ("fc", "Linear", {"out_features": 16}),
        ("m", model, params or {}),
    ]
    ids = [node[0] for node in nodes]
    return {
        "version": 1,
        "nodes": [{"id": i, "type": t, "params": p} for i, t, p in nodes],
        "edges": [{"source": a, "target": b} for a, b in zip(ids, ids[1:], strict=False)],
    }


def png_base64(pixels):
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(pixels, dtype=np.uint8)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


@pytest.fixture
def fake_mnist(monkeypatch):
    fake = FakeTorchvision(n=120)
    monkeypatch.setattr(data, "open_torchvision", fake)
    return fake


@pytest.mark.parametrize(
    "model,params",
    [("SVC", {}), ("HistGradientBoostingClassifier", QUICK_BOOSTING)],
)
def test_a_cnn_learns_image_features_for_a_classical_model(model, params, tmp_path, fake_mnist):
    graph = cnn_hybrid(model, [1, 28, 28], params)
    analysis = analyze(graph)
    assert analysis.valid, analysis.errors
    assert analysis.family == "hybrid" and analysis.boundary.id == "fc"
    plan = check_compatibility(analysis, _mnist()).plan
    assert plan.hybrid["features"] == 16 and plan.hybrid["head_outputs"] == 10
    run_dir = tmp_path / "run"
    st.write_job(
        run_dir,
        {
            "graph": graph,
            "dataset": {"kind": "builtin", "id": "mnist"},
            "data_root": str(tmp_path / "data"),
            "config": {"seed": 1, "epochs": 2, "batch_size": 16, "device": "cpu"},
        },
    )
    state = runner.run(run_dir)
    assert state["status"] == "completed", state["message"]
    assert state["family"] == "hybrid" and state["weights"] is True
    # Phase 1: the CNN with its temporary head, on the images of MNIST.
    assert [entry["epoch"] for entry in state["history"]] == [1, 2]
    assert (state["train_size"], state["val_size"], state["test_size"]) == (96, 24, 60)
    assert fake_mnist.calls == [("mnist", True, False), ("mnist", False, False)]
    preprocessing = state["preprocessing"]
    assert preprocessing["kind"] == "image" and preprocessing["dataset"] == "mnist"
    assert preprocessing["hybrid"] == {"boundary": "fc", "features": 16}
    assert (preprocessing["channels"], preprocessing["height"], preprocessing["width"]) == (
        1,
        28,
        28,
    )
    # Phase 2: the classical model on the 16 features, measured, tested and evaluated.
    classical = state["classical"]
    assert classical["estimator"] == model and classical["rows"] == 96
    assert classical["hybrid"] == {
        "boundary": "fc",
        "features": 16,
        "feature_training": "pretrain",
        "epochs": 2,
    }
    assert classical["validation"]["accuracy"] is not None
    assert state["test"]["accuracy"] is not None
    evaluation = state["evaluation"]
    assert evaluation["task"] == "classification" and evaluation["test"]["size"] == 60
    assert evaluation["val"]["size"] == 24 and evaluation["class_names"] == [
        str(i) for i in range(10)
    ]
    assert len(evaluation["val"]["confusion_matrix"]) == 10
    assert state["message"].startswith(f"Finished: 16 features learned in 2 epochs; {model}")
    # The files: the neural layers and the model fitted on their 16 features.
    checkpoint = torch.load(run_dir / "weights.pt", weights_only=True)
    assert "conv2d_1.weight" in checkpoint["state_dict"]
    assert checkpoint["state_dict"]["conv2d_1.weight"].shape == (4, 1, 3, 3)
    bundle = joblib.load(run_dir / "model.joblib")
    assert bundle["columns"] == [f"feature_{i}" for i in range(1, 17)]
    assert bundle["hybrid"]["features"] == 16 and bundle["num_classes"] == 10

    # Predictions on image files, prepared like the training images.
    plan_dict = plan.to_dict()
    images = [png_base64(np.full((28, 28), 255)), png_base64(np.zeros((28, 28)))]
    prepared = inference.prepare_image_files(plan_dict, preprocessing, images, top_k=3)
    answer = predict.predict(
        run_dir, {"images": prepared.files, "image": prepared.image, "kind": prepared.kind}
    )
    assert answer["probabilities"] is True and len(answer["outputs"][0]) == 2
    predictions = inference.read_outputs(
        plan_dict, preprocessing, answer["outputs"], 2, 3, probabilities=True
    )
    assert all(p["class"] in [str(i) for i in range(10)] for p in predictions)
    assert all(len(p["top"]) == 3 for p in predictions)
    assert sum(answer["outputs"][0][0]) == pytest.approx(1.0)


def _mnist():
    from brickdnn.training import builtin_info

    return builtin_info("mnist")


def test_frozen_image_features_and_their_script(tmp_path, fake_mnist):
    graph = cnn_hybrid("SVC", [1, 28, 28])
    run_dir = tmp_path / "run"
    st.write_job(
        run_dir,
        {
            "graph": graph,
            "dataset": {"kind": "builtin", "id": "mnist"},
            "data_root": str(tmp_path / "data"),
            "config": {"seed": 1, "feature_training": "frozen", "device": "cpu"},
        },
    )
    state = runner.run(run_dir)
    assert state["status"] == "completed", state["message"]
    assert state["history"] == [] and state["epochs"] == 0
    assert state["message"].startswith("Finished: 16 frozen random features; SVC on them:")
    # The script of an image network has no CSV training: it prints and fits demo data.
    code = generate_code(graph)
    assert "INPUT_SHAPE = (1, 28, 28)" in code and "--csv" not in code
    (tmp_path / "net.py").write_text(code)
    result = subprocess.run(
        [sys.executable, "net.py"], cwd=tmp_path, capture_output=True, text=True, timeout=180
    )
    assert result.returncode == 0, result.stderr
    assert "FeatureExtractor(" in result.stdout


# --------------------------------------------------------------------------- #
# Through the REST API, with an uploaded folder of images
# --------------------------------------------------------------------------- #


def image_folder_zip(per_class=16, size=(12, 12)):
    """Dark and bright images (with noise) in one folder per class."""
    rng = np.random.default_rng(0)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, level in (("dark", 50), ("bright", 200)):
            for i in range(per_class):
                pixels = np.clip(level + rng.normal(0, 25, size=(*size, 3)), 0, 255)
                file = io.BytesIO()
                Image.fromarray(pixels.astype(np.uint8)).save(file, format="PNG")
                archive.writestr(f"{name}/{i}.png", file.getvalue())
    return buffer.getvalue()


@pytest.fixture
def real_processes(training_dir, settings):
    settings.TRAINING_RUNNER_COMMAND = None
    settings.TRAINING_PREDICT_COMMAND = None
    settings.TRAINING_MAX_RUNS = 5
    return training_dir


@pytest.mark.django_db
def test_an_image_hybrid_through_the_api(auth_client, real_processes, tmp_path):
    response = upload(auth_client, "brightness.zip", image_folder_zip())
    assert response.status_code == 201, response.data
    assert response.data["kind"] == "images"
    dataset = {"kind": "upload", "id": response.data["id"]}
    graph = cnn_hybrid("SVC", [3, 16, 16])
    config = {"seed": 1, "epochs": 4, "batch_size": 8, "test_split": 0.2, "device": "cpu"}
    checked = auth_client.post(
        "/api/training/check/",
        {"graph": graph, "dataset": dataset, "config": config},
        format="json",
    ).data
    assert checked["ok"] is True, checked
    assert checked["plan"]["hybrid"]["features"] == 16
    assert checked["estimate"]["details"]["family"] == "hybrid"
    response = auth_client.post(
        "/api/training/runs/",
        {"name": "Brightness", "graph": graph, "dataset": dataset, "config": config},
        format="json",
    )
    assert response.status_code == 201, response.data
    run_id = response.data["id"]
    run = wait_for(auth_client, run_id, timeout=180, interval=0.25)
    assert run["status"] == "completed", run["message"]
    assert run["has_weights"] is True and run["epochs_done"] == 4
    state = run["state"]
    assert state["preprocessing"]["kind"] == "image"
    assert state["preprocessing"]["class_names"] == ["bright", "dark"]
    assert state["evaluation"]["test"]["size"] == 6
    # Two classes: the threshold analysis of the classical model.
    assert state["evaluation"]["binary"]["positive_class"] == 1

    # A prediction on image files (they are resized like the training images).
    bright = png_base64(np.full((12, 12, 3), 210))
    dark = png_base64(np.full((12, 12, 3), 40))
    response = auth_client.post(
        f"/api/training/runs/{run_id}/predict/", {"image": [bright, dark]}, format="json"
    )
    assert response.status_code == 200, response.data
    predictions = response.data["predictions"]
    assert [p["class"] for p in predictions] == ["bright", "dark"]

    # The download: both parts, the preprocessing of the images and the script.
    response = auth_client.get(f"/api/training/runs/{run_id}/weights/")
    assert response.status_code == 200
    assert response["Content-Disposition"] == 'attachment; filename="brightness.zip"'
    archive = zipfile.ZipFile(io.BytesIO(b"".join(response.streaming_content)))
    assert sorted(archive.namelist()) == [
        "README.txt",
        "brightness.features.pt",
        "brightness.joblib",
        "brightness.preprocessing.json",
        "brightness.py",
    ]
    folder = tmp_path / "download"
    archive.extractall(folder)
    preprocessing = json.loads((folder / "brightness.preprocessing.json").read_text())
    assert preprocessing["kind"] == "image" and preprocessing["hybrid"]["features"] == 16
    assert (preprocessing["height"], preprocessing["width"]) == (16, 16)
    bundle = joblib.load(folder / "brightness.joblib")
    assert bundle["class_names"] == ["bright", "dark"] and bundle["hybrid"]["features"] == 16
    # The script's FeatureExtractor loads the neural layers; with the classical model they
    # predict a prepared image like the server did.
    use = (
        "import joblib, torch\n"
        "from brightness import FeatureExtractor\n"
        "extractor = FeatureExtractor()\n"
        "extractor.load_state_dict(torch.load('brightness.features.pt')['state_dict'])\n"
        "extractor.eval()\n"
        "model = joblib.load('brightness.joblib')['model']\n"
        "bright = torch.full((1, 3, 16, 16), (210 / 255 - 0.5) / 0.5)\n"
        "with torch.no_grad():\n"
        "    print(model.predict(extractor(bright).double().numpy())[0])\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", use], cwd=folder, capture_output=True, text=True, timeout=180
    )
    assert result.returncode == 0, result.stderr
    names = bundle["class_names"]
    assert names[int(result.stdout.strip())] == "bright"
