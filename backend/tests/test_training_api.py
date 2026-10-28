"""Local training API: capabilities, dataset uploads, compatibility checks and training runs.

Most tests use a fake runner, a tiny script that writes ``state.json`` like the real
runner does, so they are fast and do not need PyTorch. One end-to-end test at the end
trains a real network on demo data.
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import signal
import subprocess
import sys
import time
import uuid
import zipfile
from datetime import timedelta

import pytest
from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.utils import timezone
from rest_framework.parsers import MultiPartParser
from rest_framework.test import APIClient
from rest_framework.throttling import ScopedRateThrottle

from brick_dnn_core import training_service
from brick_dnn_core.models import Dataset, TrainingRun
from brickdnn.training import TERMINAL_STATUSES, torch_installed
from brickdnn.training.state import read_job, read_state

from .helpers import chain

pytestmark = pytest.mark.django_db

DEMO_GRAPH = chain(
    ("Input", {"shape": [4]}),
    ("Linear", {"out_features": 8}),
    ("Activation", {"function": "relu"}),
    ("Linear", {"out_features": 3}),
)
QUICK = {"epochs": 1, "batch_size": 16, "max_samples": 64, "device": "cpu"}
RANDOM = {"kind": "random"}
PROBE = {
    "torch_version": "2.2.2",
    "torchvision": True,
    "torchvision_version": "0.17.2",
    "devices": [{"id": "cpu", "label": "CPU"}, {"id": "mps", "label": "Apple GPU (MPS)"}],
    "default_device": "mps",
}

# Behaves like brickdnn.training.runner as far as the API can tell. The mode is the first
# argument, the run directory the last one.
FAKE_RUNNER = """
import sys
import time
from pathlib import Path

from brickdnn.training import state as st

mode, run_dir = sys.argv[1], Path(sys.argv[-1])
job = st.read_job(run_dir)
state = st.initial_state(job["config"]["epochs"])
state.update(status="running", message="Epoch 1 of 1", task="demo", total_params=99)


def save(**changes):
    state.update(changes)
    state["updated_at"] = time.time()
    st.write_state(run_dir, state)


def finish(status, message):
    (run_dir / st.WEIGHTS_FILE).write_bytes(b"fake weights")
    save(status=status, message=message, weights=True, finished_at=time.time())


if mode == "silent":  # never writes a state
    time.sleep(30)
    sys.exit(0)
save()
if mode == "complete":
    state["history"].append(
        {"epoch": 1, "train_loss": 0.5, "val_loss": 0.4, "train_accuracy": 0.8,
         "val_accuracy": 0.9, "seconds": 0.1}
    )
    state["best"] = {"epoch": 1, "val_loss": 0.4, "val_accuracy": 0.9}
    finish("completed", "Finished after 1 epoch.")
elif mode == "until_stop":
    deadline = time.time() + 30
    while not st.stop_requested(run_dir) and time.time() < deadline:
        save()
        time.sleep(0.05)
    finish("stopped", "Stopped after 0 epochs.")
elif mode == "search":  # a finished hyperparameter search with a test set
    trial = {"status": "completed", "message": "", "epochs_done": 1, "train_loss": 0.5,
             "val_rmse": None, "seconds": 0.1}
    state["tuning"] = job["tuning"]
    state["trials_total"] = 2
    state["trials"] = [
        {**trial, "trial": 1, "params": {"learning_rate": 0.01}, "val_loss": 0.6,
         "val_accuracy": 0.7},
        {**trial, "trial": 2, "params": {"learning_rate": 0.001}, "val_loss": 0.3,
         "val_accuracy": 0.95},
        {**trial, "trial": 3, "params": {"learning_rate": 9.0}, "status": "failed",
         "message": "The loss became NaN in epoch 1. Try a lower learning rate.",
         "val_loss": None, "val_accuracy": None, "epochs_done": 0},
    ]
    state["best_trial"] = 2
    state["history"].append(
        {"epoch": 1, "train_loss": 0.5, "val_loss": 0.3, "train_accuracy": 0.8,
         "val_accuracy": 0.95, "seconds": 0.1}
    )
    state["best"] = {"epoch": 1, "val_loss": 0.3, "val_accuracy": 0.95}
    state["test_size"] = 13
    state["test"] = {"loss": 0.35, "accuracy": 0.92, "rmse": None}
    finish("completed", "Finished 3 trials (1 failed).")
elif mode == "crash":  # exits without a final status
    print("RuntimeError: something broke", file=sys.stderr)
    sys.exit(1)
elif mode == "stale":  # alive, but no heartbeat
    state["updated_at"] = time.time() - 600
    st.write_state(run_dir, state)
    time.sleep(30)
elif mode == "hang":  # sends heartbeats but ignores stop requests
    deadline = time.time() + 30
    while time.time() < deadline:
        save()
        time.sleep(0.1)
"""


# --------------------------------------------------------------------------- #
# Fixtures and helpers
# --------------------------------------------------------------------------- #


@pytest.fixture
def training_dir(settings, tmp_path, monkeypatch):
    """A fresh TRAINING_DIR, a fake runner and 'PyTorch installed' for every test."""
    settings.TRAINING_DIR = tmp_path / "training"
    settings.TRAINING_ENABLED = True
    settings.TRAINING_MAX_RUNS = 1
    settings.TRAINING_MAX_UPLOAD_MB = 5
    monkeypatch.setattr(training_service, "torch_installed", lambda: True)
    script = tmp_path / "fake_runner.py"
    script.write_text(FAKE_RUNNER)
    settings.TRAINING_RUNNER_COMMAND = [sys.executable, str(script), "complete"]
    yield settings.TRAINING_DIR
    # Never leave processes behind, even when a test fails.
    for process in list(training_service._PROCESSES.values()):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                process.kill()
            process.wait(timeout=5)
    training_service._PROCESSES.clear()


def use_runner(settings, mode: str) -> None:
    settings.TRAINING_RUNNER_COMMAND = [*settings.TRAINING_RUNNER_COMMAND[:2], mode]


@pytest.fixture
def user():
    return get_user_model().objects.create_user("ada", password="correct-horse-battery")


@pytest.fixture
def other_user():
    return get_user_model().objects.create_user("mallory", password="whatever-password-1")


@pytest.fixture
def client():
    return APIClient()


@pytest.fixture
def auth_client(user):
    client = APIClient()
    client.force_authenticate(user)
    return client


def iris_csv(rows: int = 12) -> bytes:
    species = ["setosa", "versicolor", "virginica"]
    lines = ["sepal_length,sepal_width,petal_length,petal_width,species"]
    for i in range(rows):
        lines.append(f"{5 + i / 10:.1f},3.{i % 10},1.{i % 7},0.{i % 5},{species[i % 3]}")
    return ("\n".join(lines) + "\n").encode()


def image_zip(names: list[str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name in names:
            archive.writestr(name, b"\x89PNG not really an image")
    return buffer.getvalue()


def upload(client, filename: str, content: bytes, **fields):
    return client.post(
        "/api/training/datasets/",
        {"file": SimpleUploadedFile(filename, content), **fields},
        format="multipart",
    )


def start(client, graph=DEMO_GRAPH, dataset=RANDOM, config=QUICK, name="Small net"):
    return client.post(
        "/api/training/runs/",
        {"name": name, "graph": graph, "dataset": dataset, "config": config},
        format="json",
    )


def wait_for(client, run_id, statuses=TERMINAL_STATUSES, timeout=15.0, interval=0.05):
    """Poll the run until its status is one of ``statuses``."""
    deadline = time.monotonic() + timeout
    while True:
        data = client.get(f"/api/training/runs/{run_id}/").data
        if data["status"] in statuses:
            return data
        if time.monotonic() > deadline:
            raise AssertionError(f"The run is still {data['status']}: {data['message']}")
        time.sleep(interval)


def wait_for_state(training_dir, run_id, timeout=10.0):
    deadline = time.monotonic() + timeout
    while (state := read_state(training_dir / "runs" / str(run_id))) is None:
        assert time.monotonic() < deadline, "the runner did not write a state"
        time.sleep(0.02)
    return state


def entries(path):
    return sorted(p.name for p in path.iterdir()) if path.exists() else []


# --------------------------------------------------------------------------- #
# Capabilities
# --------------------------------------------------------------------------- #


def test_capabilities_without_torch(client, training_dir, monkeypatch):
    monkeypatch.setattr(training_service, "torch_installed", lambda: False)

    def no_probe():
        raise AssertionError("the probe must not run without PyTorch")

    monkeypatch.setattr(training_service, "run_probe", no_probe)
    response = client.get("/api/training/")
    assert response.status_code == 200
    data = response.data
    assert data["enabled"] is True
    assert data["available"] is False
    assert "pip install torch torchvision" in data["reason"]
    assert data["devices"] == [] and data["default_device"] is None
    assert data["torch_version"] is None
    assert data["login_required"] is True
    assert data["max_upload_mb"] == 5
    assert data["max_concurrent_runs"] == 1
    assert data["tuning"] == {
        "methods": ["random", "grid"],
        "tunable": ["learning_rate", "batch_size", "optimizer", "weight_decay"],
        "max_trials": 100,
        "max_values": 20,
        "default": {
            "method": "random",
            "trials": 10,
            "space": {"learning_rate": {"min": 1e-4, "max": 1e-2}, "batch_size": [32, 64, 128]},
        },
    }
    builtin = data["builtin_datasets"]
    assert [d["id"] for d in builtin] == ["mnist", "fashion_mnist", "cifar10"]
    assert builtin[0]["num_classes"] == 10 and "available" in builtin[0]


def test_capabilities_with_torch_are_cached(client, training_dir, monkeypatch):
    calls = []
    monkeypatch.setattr(training_service, "run_probe", lambda: calls.append(1) or dict(PROBE))
    data = client.get("/api/training/").data
    assert data["available"] is True
    assert data["reason"] is None
    assert data["torch_version"] == "2.2.2"
    assert data["torchvision"] is True
    assert data["devices"] == PROBE["devices"]
    assert data["default_device"] == "mps"
    assert client.get("/api/training/").data["available"] is True
    assert len(calls) == 1  # the second request used the cache


def test_capabilities_when_pytorch_cannot_be_loaded(client, training_dir, monkeypatch):
    monkeypatch.setattr(
        training_service, "run_probe", lambda: {"error": "OSError: libtorch_cpu.so missing"}
    )
    data = client.get("/api/training/").data
    assert data["available"] is False
    assert "libtorch_cpu.so missing" in data["reason"]


def test_capabilities_when_disabled(client, training_dir, settings):
    settings.TRAINING_ENABLED = False
    data = client.get("/api/training/").data
    assert data["enabled"] is False
    assert data["available"] is False
    assert data["reason"] == "Training is disabled on this server."


def test_run_probe_reports_errors(training_dir, monkeypatch):
    """The real probe command; brickdnn.training.probe may not exist (yet) or lack torch."""
    data = training_service.run_probe()
    assert "error" in data or "torch_version" in data


def test_disabled_training_blocks_runs_and_uploads(auth_client, training_dir, settings):
    settings.TRAINING_ENABLED = False
    response = start(auth_client)
    assert response.status_code == 503
    assert response.data["detail"] == "Training is disabled on this server."
    assert upload(auth_client, "iris.csv", iris_csv()).status_code == 503
    response = auth_client.post(
        "/api/training/check/", {"graph": DEMO_GRAPH, "dataset": RANDOM}, format="json"
    )
    assert response.status_code == 503
    assert TrainingRun.objects.count() == 0


def test_runs_need_pytorch(auth_client, training_dir, monkeypatch):
    monkeypatch.setattr(training_service, "torch_installed", lambda: False)
    response = start(auth_client)
    assert response.status_code == 503
    assert "PyTorch is not installed" in response.data["detail"]


def test_training_endpoints_require_login(client, training_dir):
    assert client.get("/api/training/datasets/").status_code == 401
    assert upload(client, "iris.csv", iris_csv()).status_code == 401
    assert client.get("/api/training/runs/").status_code == 401
    assert start(client).status_code == 401
    response = client.post(
        "/api/training/check/", {"graph": DEMO_GRAPH, "dataset": RANDOM}, format="json"
    )
    assert response.status_code == 401


def test_uploads_and_run_starts_are_rate_limited(auth_client, training_dir, monkeypatch):
    monkeypatch.setitem(ScopedRateThrottle.THROTTLE_RATES, "uploads", "2/hour")
    monkeypatch.setitem(ScopedRateThrottle.THROTTLE_RATES, "training", "1/hour")
    statuses = [upload(auth_client, "iris.csv", iris_csv()).status_code for _ in range(3)]
    assert statuses == [201, 201, 429]
    assert auth_client.get("/api/training/datasets/").status_code == 200  # only uploads count
    run_id = start(auth_client).data["id"]
    wait_for(auth_client, run_id)
    assert start(auth_client).status_code == 429


def test_api_root_lists_training_endpoints(client):
    endpoints = client.get("/api/").data["endpoints"]
    assert endpoints["training"].endswith("/api/training/")
    assert endpoints["training_runs"].endswith("/api/training/runs/")
    assert endpoints["training_datasets"].endswith("/api/training/datasets/")
    assert endpoints["training_check"].endswith("/api/training/check/")


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #


def test_csv_upload(auth_client, training_dir):
    content = iris_csv()
    response = upload(auth_client, "iris.csv", content)
    assert response.status_code == 201, response.data
    data = response.data
    assert data["name"] == "iris"
    assert data["kind"] == "csv"
    assert data["size_bytes"] == len(content)
    assert data["created_at"]
    assert data["info"]["num_features"] == 4
    assert data["info"]["num_samples"] == 12
    assert data["info"]["class_names"] == ["setosa", "versicolor", "virginica"]
    path = training_dir / "datasets" / str(data["id"]) / "data.csv"
    assert path.read_bytes() == content
    assert entries(training_dir / "datasets") == [str(data["id"])]

    assert [d["id"] for d in auth_client.get("/api/training/datasets/").data] == [data["id"]]
    assert auth_client.get(f"/api/training/datasets/{data['id']}/").data["name"] == "iris"


def test_upload_with_name_newest_first(auth_client, training_dir):
    first = upload(auth_client, "a.csv", iris_csv()).data
    second = upload(auth_client, "b.csv", iris_csv(), name="  My flowers ").data
    assert second["name"] == "My flowers"
    assert second["info"]["name"] == "My flowers"
    listing = auth_client.get("/api/training/datasets/").data
    assert [d["id"] for d in listing] == [second["id"], first["id"]]


def test_bad_csv_upload(auth_client, training_dir):
    response = upload(auth_client, "bad.csv", b"a\n1\n2\n3\n4\n5\n")
    assert response.status_code == 400
    assert "at least two columns" in response.data["detail"]
    response = upload(auth_client, "bad.csv", b"x,y\n1,a\nfoo,b\n3,c\n4,d\n")
    assert response.status_code == 400
    assert "is not a number" in response.data["detail"]
    assert Dataset.objects.count() == 0
    assert entries(training_dir / "datasets") == []


def test_upload_wrong_extension(auth_client, training_dir):
    response = upload(auth_client, "data.txt", b"1,2\n3,4\n")
    assert response.status_code == 400
    assert response.data["detail"] == (
        "Upload a .csv file or a .zip file with one folder per class."
    )
    response = auth_client.post("/api/training/datasets/", {"name": "x"}, format="multipart")
    assert response.status_code == 400
    assert response.data["detail"] == "Choose a file to upload."


def test_upload_too_large(auth_client, training_dir, settings, monkeypatch):
    settings.TRAINING_MAX_UPLOAD_MB = 0.01  # about 10 KB
    # Slightly too large: detected from the size of the parsed file.
    response = upload(auth_client, "big.csv", b"1,2\n" * 5000)
    assert response.status_code == 413
    assert "limited to 0.01 MB" in response.data["detail"]

    # Far too large: rejected from the Content-Length header, before the body is parsed.
    def no_parsing(*args, **kwargs):
        raise AssertionError("the body must not be parsed")

    monkeypatch.setattr(MultiPartParser, "parse", no_parsing)
    response = upload(auth_client, "huge.csv", b"1,2\n" * 50_000)
    assert response.status_code == 413
    assert Dataset.objects.count() == 0


def test_zip_upload(auth_client, training_dir, settings):
    settings.FILE_UPLOAD_MAX_MEMORY_SIZE = 10  # stream the upload into a temporary file
    names = [
        "flowers/roses/a.png",
        "flowers/roses/b.png",
        "flowers/tulips/a.jpg",
        "flowers/tulips/sub/b.JPG",
        "flowers/readme.txt",
        "__MACOSX/flowers/roses/._a.png",
    ]
    response = upload(auth_client, "flowers.zip", image_zip(names))
    assert response.status_code == 201, response.data
    data = response.data
    assert data["kind"] == "images"
    assert data["name"] == "flowers"
    assert data["info"]["class_names"] == ["roses", "tulips"]
    assert data["info"]["class_counts"] == [2, 2]
    assert data["info"]["subdir"] == "flowers"
    images = training_dir / "datasets" / str(data["id"]) / "images"
    assert (images / "flowers" / "roses" / "a.png").is_file()
    assert not (images / "flowers" / "readme.txt").exists()
    assert not (images / "__MACOSX").exists()
    assert data["size_bytes"] == sum(p.stat().st_size for p in images.rglob("*") if p.is_file())
    assert entries(training_dir / "datasets") == [str(data["id"])]


def test_small_zip_upload_in_memory(auth_client, training_dir):
    response = upload(auth_client, "pets.zip", image_zip(["cats/1.png", "cats/2.png"]))
    assert response.status_code == 400
    assert "one folder per class" in response.data["detail"]
    response = upload(
        auth_client, "pets.zip", image_zip(["cats/1.png", "cats/2.png", "dogs/1.png", "dogs/2.png"])
    )
    assert response.status_code == 201, response.data
    directory = training_dir / "datasets" / str(response.data["id"])
    assert entries(directory) == ["images"]  # the ZIP itself is not kept


def test_zip_slip_is_rejected(auth_client, training_dir):
    response = upload(auth_client, "evil.zip", image_zip(["cats/1.png", "../../evil.png"]))
    assert response.status_code == 400
    assert "unsafe path" in response.data["detail"]
    assert not list(training_dir.parent.rglob("evil.png"))
    assert Dataset.objects.count() == 0
    assert entries(training_dir / "datasets") == []


def test_abandoned_upload_folders_are_removed(auth_client, training_dir):
    abandoned = training_dir / "datasets" / ".upload-old"
    abandoned.mkdir(parents=True)
    os.utime(abandoned, (time.time() - 7 * 3600,) * 2)
    recent = training_dir / "datasets" / ".upload-recent"
    recent.mkdir()
    assert upload(auth_client, "iris.csv", iris_csv()).status_code == 201
    assert not abandoned.exists()
    assert recent.exists()  # may belong to an upload that is still being processed


def test_zip_without_class_folders(auth_client, training_dir):
    response = upload(auth_client, "flat.zip", image_zip(["1.png", "2.png", "3.png", "4.png"]))
    assert response.status_code == 400
    assert "one folder per class" in response.data["detail"]
    response = upload(auth_client, "broken.zip", b"PK this is not a zip")
    assert response.status_code == 400
    assert response.data["detail"] == "This is not a valid ZIP file."
    assert entries(training_dir / "datasets") == []


def test_dataset_delete_removes_files(
    auth_client, training_dir, django_capture_on_commit_callbacks
):
    dataset_id = upload(auth_client, "iris.csv", iris_csv()).data["id"]
    directory = training_dir / "datasets" / str(dataset_id)
    assert directory.is_dir()
    with django_capture_on_commit_callbacks(execute=True):
        assert auth_client.delete(f"/api/training/datasets/{dataset_id}/").status_code == 204
    assert not directory.exists()
    assert auth_client.get(f"/api/training/datasets/{dataset_id}/").status_code == 404


def test_dataset_cannot_be_deleted_while_training(auth_client, training_dir, settings):
    use_runner(settings, "until_stop")
    dataset_id = upload(auth_client, "iris.csv", iris_csv()).data["id"]
    response = start(auth_client, dataset={"kind": "upload", "id": dataset_id})
    assert response.status_code == 201, response.data
    run_id = response.data["id"]
    assert response.data["dataset"] == {"kind": "upload", "id": dataset_id, "name": "iris"}
    assert response.data["task"] == "classification"
    job = read_job(training_dir / "runs" / run_id)
    assert job["dataset"]["kind"] == "csv"
    assert job["dataset"]["path"] == str(training_dir / "datasets" / str(dataset_id) / "data.csv")
    assert job["dataset"]["info"]["num_features"] == 4

    response = auth_client.delete(f"/api/training/datasets/{dataset_id}/")
    assert response.status_code == 409
    assert "still running" in response.data["detail"]

    auth_client.post(f"/api/training/runs/{run_id}/stop/")
    assert wait_for(auth_client, run_id)["status"] == "stopped"
    assert auth_client.delete(f"/api/training/datasets/{dataset_id}/").status_code == 204
    run = TrainingRun.objects.get(pk=run_id)
    assert run.upload is None
    assert run.dataset["name"] == "iris"  # the run keeps its description of the data


# --------------------------------------------------------------------------- #
# Compatibility check
# --------------------------------------------------------------------------- #


def check(client, graph, dataset):
    return client.post("/api/training/check/", {"graph": graph, "dataset": dataset}, format="json")


def test_check_compatible(auth_client, training_dir):
    data = check(auth_client, DEMO_GRAPH, RANDOM).data
    assert data["ok"] is True
    assert data["plan"]["task"] == "demo"

    dataset_id = upload(auth_client, "iris.csv", iris_csv()).data["id"]
    response = check(auth_client, DEMO_GRAPH, {"kind": "upload", "id": dataset_id})
    assert response.status_code == 200
    assert response.data["ok"] is True
    assert response.data["errors"] == []
    assert response.data["plan"]["task"] == "classification"
    assert response.data["plan"]["class_names"] == ["setosa", "versicolor", "virginica"]
    # Ids may also arrive as strings (e.g. from a <select>).
    assert check(auth_client, DEMO_GRAPH, {"kind": "upload", "id": str(dataset_id)}).data["ok"]


def test_check_incompatible(auth_client, training_dir):
    dataset_id = upload(auth_client, "iris.csv", iris_csv()).data["id"]
    graph = chain(("Input", {"shape": [5]}), ("Linear", {"out_features": 3}))
    data = check(auth_client, graph, {"kind": "upload", "id": dataset_id}).data
    assert data["ok"] is False
    assert data["plan"] is None
    assert "Set the Input shape to 4" in data["errors"][0]

    data = check(auth_client, DEMO_GRAPH, {"kind": "builtin", "id": "mnist"}).data
    assert data["ok"] is False
    assert "Input" in data["errors"][0]

    broken = chain(("Input", {"shape": [3, 8, 8]}), ("Linear", {"in_features": 5}))
    data = check(auth_client, broken, RANDOM).data
    assert data["ok"] is False
    assert "Fix the layers marked in red" in data["errors"][0]


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({"graph": DEMO_GRAPH}, "Choose a dataset."),
        ({"graph": DEMO_GRAPH, "dataset": {"kind": "nope"}}, "Unknown dataset type"),
        ({"graph": DEMO_GRAPH, "dataset": {"kind": "builtin", "id": "imagenet"}}, "Unknown"),
        ({"graph": DEMO_GRAPH, "dataset": {"kind": "upload", "id": 999}}, "does not exist"),
        ({"graph": DEMO_GRAPH, "dataset": {"kind": "upload", "id": True}}, "uploaded datasets"),
        ({"graph": "x", "dataset": RANDOM}, "The graph must be a JSON object."),
    ],
)
def test_check_bad_requests(auth_client, training_dir, body, message):
    response = auth_client.post("/api/training/check/", body, format="json")
    assert response.status_code == 400
    assert message in response.data["detail"]


# --------------------------------------------------------------------------- #
# Ownership
# --------------------------------------------------------------------------- #


def test_users_only_see_their_own_data(auth_client, training_dir, other_user):
    other = APIClient()
    other.force_authenticate(other_user)
    dataset_id = upload(other, "iris.csv", iris_csv()).data["id"]
    run_id = start(other).data["id"]
    wait_for(other, run_id)

    assert auth_client.get("/api/training/datasets/").data == []
    assert auth_client.get("/api/training/runs/").data == []
    assert auth_client.get(f"/api/training/datasets/{dataset_id}/").status_code == 404
    assert auth_client.delete(f"/api/training/datasets/{dataset_id}/").status_code == 404
    assert auth_client.get(f"/api/training/runs/{run_id}/").status_code == 404
    assert auth_client.post(f"/api/training/runs/{run_id}/stop/").status_code == 404
    assert auth_client.get(f"/api/training/runs/{run_id}/weights/").status_code == 404
    assert auth_client.delete(f"/api/training/runs/{run_id}/").status_code == 404
    assert auth_client.get("/api/training/runs/not-a-uuid/").status_code == 404
    response = check(auth_client, DEMO_GRAPH, {"kind": "upload", "id": dataset_id})
    assert response.status_code == 400
    response = start(auth_client, dataset={"kind": "upload", "id": dataset_id})
    assert response.status_code == 400
    assert Dataset.objects.filter(pk=dataset_id).exists()


def test_deleting_a_user_removes_their_files(
    auth_client, training_dir, user, django_capture_on_commit_callbacks
):
    dataset_id = upload(auth_client, "iris.csv", iris_csv()).data["id"]
    run_id = start(auth_client).data["id"]
    wait_for(auth_client, run_id)
    assert (training_dir / "runs" / run_id).is_dir()
    with django_capture_on_commit_callbacks(execute=True):
        user.delete()
    assert not (training_dir / "datasets" / str(dataset_id)).exists()
    assert not (training_dir / "runs" / run_id).exists()


# --------------------------------------------------------------------------- #
# Training runs
# --------------------------------------------------------------------------- #


def test_run_lifecycle(auth_client, training_dir):
    response = start(auth_client)
    assert response.status_code == 201, response.data
    data = response.data
    run_id = data["id"]
    assert data["name"] == "Small net"
    assert data["status"] in ("starting", "preparing", "running", "completed")
    assert data["dataset"] == {"kind": "random", "id": None, "name": "Demo data"}
    assert data["config"]["epochs"] == 1 and data["config"]["device"] == "cpu"
    assert data["config"]["test_split"] == 0.0
    assert data["tuning"] is None
    assert data["trials_done"] == 0 and data["best_trial"] is None
    assert data["epochs"] == 1
    assert data["task"] == "demo"
    assert data["graph"] == DEMO_GRAPH
    assert data["plan"]["task"] == "demo"
    assert data["total_params"] == 4 * 8 + 8 + 8 * 3 + 3
    assert data["stop_requested"] is False
    assert data["has_weights"] is False

    job = read_job(training_dir / "runs" / run_id)
    assert job["name"] == "Small net"
    assert job["graph"] == DEMO_GRAPH
    assert job["dataset"]["kind"] == "random"
    assert job["dataset"]["path"] is None
    assert job["dataset"]["info"]["name"] == "Demo data"
    assert job["config"]["max_samples"] == 64
    assert job["tuning"] is None
    assert job["data_root"] == str(training_dir / "builtin")

    data = wait_for(auth_client, run_id)
    assert data["status"] == "completed"
    assert data["message"] == "Finished after 1 epoch."
    assert data["epochs_done"] == 1
    assert data["metrics"] == {
        "train_loss": 0.5,
        "val_loss": 0.4,
        "val_accuracy": 0.9,
        "val_rmse": None,
        "test_loss": None,
        "test_accuracy": None,
        "test_rmse": None,
    }
    assert data["best"] == {"epoch": 1, "val_loss": 0.4, "val_accuracy": 0.9}
    assert data["has_weights"] is True
    assert data["total_params"] == 99  # as reported by the runner
    assert data["finished_at"] is not None
    assert data["state"]["status"] == "completed"
    assert len(data["state"]["history"]) == 1

    listing = auth_client.get("/api/training/runs/").data
    assert [r["id"] for r in listing] == [run_id]
    assert listing[0]["status"] == "completed"
    assert "graph" not in listing[0] and "state" not in listing[0]

    response = auth_client.get(f"/api/training/runs/{run_id}/weights/")
    assert response.status_code == 200
    assert response["Content-Type"] == "application/octet-stream"
    assert response["Content-Disposition"] == 'attachment; filename="small-net.pt"'
    assert b"".join(response.streaming_content) == b"fake weights"
    response.close()
    response = auth_client.get(
        f"/api/training/runs/{run_id}/weights/", HTTP_ACCEPT="application/octet-stream"
    )
    assert response.status_code == 200
    response.close()


def test_hyperparameter_search_run(auth_client, training_dir, settings):
    use_runner(settings, "search")
    tuning = {
        "trials": 3,
        "space": {"learning_rate": {"min": "1e-4", "max": 10}, "optimizer": ["SGD", "adam"]},
    }
    config = {**QUICK, "test_split": 0.2}
    response = auth_client.post(
        "/api/training/runs/",
        {
            "name": "Search",
            "graph": DEMO_GRAPH,
            "dataset": RANDOM,
            "config": config,
            "tuning": tuning,
        },
        format="json",
    )
    assert response.status_code == 201, response.data
    run_id = response.data["id"]
    normalised = {
        "method": "random",
        "trials": 3,
        "space": {"learning_rate": {"min": 1e-4, "max": 10.0}, "optimizer": ["sgd", "adam"]},
    }
    assert response.data["tuning"] == normalised
    assert response.data["config"]["test_split"] == 0.2
    job = read_job(training_dir / "runs" / run_id)
    assert job["tuning"] == normalised
    assert job["config"]["test_split"] == 0.2

    data = wait_for(auth_client, run_id)
    assert data["status"] == "completed", data["message"]
    assert data["trials_done"] == 3
    assert data["best_trial"] == {
        "trial": 2,
        "params": {"learning_rate": 0.001},
        "val_loss": 0.3,
        "val_accuracy": 0.95,
        "val_rmse": None,
    }
    assert data["metrics"]["val_accuracy"] == 0.95
    assert data["metrics"]["test_loss"] == 0.35
    assert data["metrics"]["test_accuracy"] == 0.92
    assert data["metrics"]["test_rmse"] is None
    assert data["state"]["test_size"] == 13
    assert [t["trial"] for t in data["state"]["trials"]] == [1, 2, 3]
    listing = auth_client.get("/api/training/runs/").data
    assert listing[0]["tuning"] == normalised and listing[0]["best_trial"]["trial"] == 2


def test_runs_are_listed_newest_first(auth_client, training_dir, settings):
    settings.TRAINING_MAX_RUNS = 5
    first = start(auth_client, name="first").data["id"]
    second = start(auth_client, name="second").data["id"]
    wait_for(auth_client, first)
    wait_for(auth_client, second)
    assert [r["id"] for r in auth_client.get("/api/training/runs/").data] == [second, first]


def test_stop_run(auth_client, training_dir, settings):
    use_runner(settings, "until_stop")
    run_id = start(auth_client).data["id"]
    wait_for_state(training_dir, run_id)
    response = auth_client.post(f"/api/training/runs/{run_id}/stop/")
    assert response.status_code == 200
    assert response.data["stop_requested"] is True
    assert (training_dir / "runs" / run_id / "STOP").exists()
    data = wait_for(auth_client, run_id)
    assert data["status"] == "stopped"
    assert data["message"] == "Stopped after 0 epochs."
    assert data["has_weights"] is True
    # Stopping a finished run changes nothing.
    response = auth_client.post(f"/api/training/runs/{run_id}/stop/")
    assert response.status_code == 200
    assert response.data["status"] == "stopped"


def test_crashed_runner_fails_the_run(auth_client, training_dir, settings):
    use_runner(settings, "crash")
    run_id = start(auth_client).data["id"]
    data = wait_for(auth_client, run_id)
    assert data["status"] == "failed"
    assert data["message"].startswith("The training process stopped unexpectedly.")
    assert "RuntimeError: something broke" in data["message"]
    assert data["finished_at"] is not None
    # state.json still says "running"; the API's verdict is shown instead.
    assert data["state"]["status"] == "failed"
    assert data["state"]["message"] == data["message"]
    assert data["state"]["progress"] is None
    response = auth_client.get(f"/api/training/runs/{run_id}/weights/")
    assert response.status_code == 404
    assert response.data["detail"] == "No weights available for this run."


def test_stale_heartbeat_fails_the_run(auth_client, training_dir, settings, monkeypatch):
    use_runner(settings, "stale")
    monkeypatch.setattr(training_service, "STALE_GRACE_SECONDS", 0.2)
    run_id = start(auth_client).data["id"]
    process = training_service._PROCESSES[run_id]
    deadline = time.monotonic() + 10
    while (state := read_state(training_dir / "runs" / run_id)) is None or state[
        "updated_at"
    ] > time.time() - 300:
        assert time.monotonic() < deadline
        time.sleep(0.02)
    # A single stale sighting is not enough (e.g. right after the laptop woke up) ...
    assert auth_client.get(f"/api/training/runs/{run_id}/").data["status"] == "running"
    time.sleep(0.3)
    # ... but a heartbeat that stays stale is.
    data = auth_client.get(f"/api/training/runs/{run_id}/").data
    assert data["status"] == "failed"
    assert data["message"].startswith("The training process stopped unexpectedly.")
    process.wait(timeout=5)  # the hanging process was killed


def test_runner_that_never_reports_fails(auth_client, training_dir, settings):
    use_runner(settings, "silent")
    run_id = start(auth_client).data["id"]
    process = training_service._PROCESSES[run_id]
    assert auth_client.get(f"/api/training/runs/{run_id}/").data["status"] == "starting"
    long_ago = timezone.now() - timedelta(minutes=2)
    TrainingRun.objects.filter(pk=run_id).update(created_at=long_ago, started_at=long_ago)
    data = auth_client.get(f"/api/training/runs/{run_id}/").data
    assert data["status"] == "failed"
    assert data["message"].startswith("The training process did not start.")
    process.wait(timeout=5)


def test_runner_ignoring_stop_is_killed(auth_client, training_dir, settings):
    use_runner(settings, "hang")
    run_id = start(auth_client).data["id"]
    process = training_service._PROCESSES[run_id]
    wait_for_state(training_dir, run_id)
    auth_client.post(f"/api/training/runs/{run_id}/stop/")
    assert auth_client.get(f"/api/training/runs/{run_id}/").data["status"] == "running"
    TrainingRun.objects.filter(pk=run_id).update(
        stop_requested_at=timezone.now() - timedelta(seconds=31)
    )
    data = auth_client.get(f"/api/training/runs/{run_id}/").data
    assert data["status"] == "stopped"
    assert data["has_weights"] is False
    assert process.wait(timeout=5) == -signal.SIGKILL


def test_concurrent_runs_are_limited(auth_client, training_dir, settings, other_user):
    use_runner(settings, "until_stop")
    first = start(auth_client)
    assert first.status_code == 201
    response = start(auth_client)
    assert response.status_code == 409
    assert response.data["detail"] == (
        "Another training is still running. Stop it or wait until it finishes."
    )
    # The limit counts the runs of all users.
    other = APIClient()
    other.force_authenticate(other_user)
    assert start(other).status_code == 409

    settings.TRAINING_MAX_RUNS = 2
    second = start(other)
    assert second.status_code == 201
    for client, response in ((auth_client, first), (other, second)):
        client.post(f"/api/training/runs/{response.data['id']}/stop/")
        wait_for(client, response.data["id"])
    settings.TRAINING_MAX_RUNS = 1
    assert start(auth_client).status_code == 201  # finished runs free their slot


def test_delete_active_run(auth_client, training_dir, settings, django_capture_on_commit_callbacks):
    use_runner(settings, "hang")
    run_id = start(auth_client).data["id"]
    process = training_service._PROCESSES[run_id]
    wait_for_state(training_dir, run_id)
    with django_capture_on_commit_callbacks(execute=True):
        assert auth_client.delete(f"/api/training/runs/{run_id}/").status_code == 204
    assert process.wait(timeout=5) == -signal.SIGKILL
    assert not (training_dir / "runs" / run_id).exists()
    assert auth_client.get(f"/api/training/runs/{run_id}/").status_code == 404
    assert start(auth_client).status_code == 201  # the slot is free again


def test_delete_finished_run(auth_client, training_dir, django_capture_on_commit_callbacks):
    run_id = start(auth_client).data["id"]
    wait_for(auth_client, run_id)
    with django_capture_on_commit_callbacks(execute=True):
        assert auth_client.delete(f"/api/training/runs/{run_id}/").status_code == 204
    assert entries(training_dir / "runs") == []
    assert TrainingRun.objects.count() == 0


def test_run_rejects_incompatible_networks(auth_client, training_dir):
    dataset_id = upload(auth_client, "iris.csv", iris_csv()).data["id"]
    graph = chain(("Input", {"shape": [5]}), ("Linear", {"out_features": 3}))
    response = start(auth_client, graph=graph, dataset={"kind": "upload", "id": dataset_id})
    assert response.status_code == 400
    assert response.data["detail"] == response.data["errors"][0]
    assert "Set the Input shape to 4" in response.data["detail"]
    assert response.data["warnings"] == []
    assert TrainingRun.objects.count() == 0
    assert entries(training_dir / "runs") == []


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"config": {"epochs": 0}}, "Epochs must be between 1 and 1,000."),
        ({"config": {"optimizer": "rmsprop"}}, "Optimizer must be one of"),
        ({"config": "fast"}, "The training settings must be an object."),
        ({"config": {"test_split": 0.7}}, "Test split must be at least 0 and at most 0.5."),
        (
            {"config": {"val_split": 0.5, "test_split": 0.5}},
            "Validation and test split together can be at most 80%.",
        ),
        ({"tuning": "yes"}, "The hyperparameter search settings must be an object."),
        ({"tuning": {"space": {}}}, "choose at least one setting to tune."),
        ({"tuning": {"trials": 500}}, "the number of trials must be at least 1 and at most 100"),
        (
            {"tuning": {"method": "grid", "space": {"learning_rate": {"min": 0.1, "max": 1}}}},
            "Grid search needs a list of values",
        ),
        ({"graph": {"nodes": "x"}}, "'nodes' and 'edges' must be lists."),
        ({"dataset": None}, "Choose a dataset."),
        ({"name": "x" * 121}, "at most 120 characters"),
        ({"name": 5}, "The name must be text."),
    ],
)
def test_run_rejects_bad_input(auth_client, training_dir, changes, message):
    body = {"name": "net", "graph": DEMO_GRAPH, "dataset": RANDOM, "config": QUICK, **changes}
    response = auth_client.post("/api/training/runs/", body, format="json")
    assert response.status_code == 400
    assert message in response.data["detail"]
    assert TrainingRun.objects.count() == 0


def test_run_name_defaults(auth_client, training_dir):
    response = auth_client.post(
        "/api/training/runs/", {"graph": DEMO_GRAPH, "dataset": RANDOM}, format="json"
    )
    assert response.status_code == 201
    assert response.data["name"] == "Training run"
    assert response.data["config"]["epochs"] == 10  # defaults
    wait_for(auth_client, response.data["id"])


def test_runner_that_cannot_be_started(auth_client, training_dir, settings, tmp_path):
    settings.TRAINING_RUNNER_COMMAND = [str(tmp_path / "does-not-exist")]
    response = start(auth_client)
    assert response.status_code == 201
    assert response.data["status"] == "failed"
    assert response.data["message"].startswith("The training process could not be started.")
    assert str(tmp_path) not in response.data["message"]
    assert start(auth_client).status_code == 201  # failed runs do not block new ones


# --------------------------------------------------------------------------- #
# End to end with the real runner and PyTorch
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not torch_installed(), reason="PyTorch is not installed")
def test_real_training_end_to_end(auth_client, training_dir, settings):
    if importlib.util.find_spec("brickdnn.training.runner") is None:
        pytest.skip("brickdnn.training.runner is not available")
    settings.TRAINING_RUNNER_COMMAND = None
    response = start(auth_client, name="Demo MLP")
    assert response.status_code == 201, response.data
    run_id = response.data["id"]
    data = wait_for(auth_client, run_id, timeout=90, interval=0.5)
    log = (training_dir / "runs" / run_id / "output.log").read_text(errors="replace")
    assert data["status"] == "completed", f"{data['message']}\n{log[-3000:]}"
    assert data["epochs_done"] == 1
    assert data["has_weights"] is True
    assert data["metrics"]["train_loss"] is not None
    assert data["state"]["device"] == "cpu"
    response = auth_client.get(f"/api/training/runs/{run_id}/weights/")
    assert response.status_code == 200
    assert response["Content-Disposition"] == 'attachment; filename="demo-mlp.pt"'
    assert len(b"".join(response.streaming_content)) > 100
    response.close()


@pytest.mark.skipif(not torch_installed(), reason="PyTorch is not installed")
def test_real_search_with_a_test_set_end_to_end(auth_client, training_dir, settings):
    if importlib.util.find_spec("brickdnn.training.runner") is None:
        pytest.skip("brickdnn.training.runner is not available")
    settings.TRAINING_RUNNER_COMMAND = None
    response = auth_client.post(
        "/api/training/runs/",
        {
            "name": "Search",
            "graph": DEMO_GRAPH,
            "dataset": RANDOM,
            "config": {**QUICK, "test_split": 0.2},
            "tuning": {"method": "grid", "space": {"learning_rate": [0.01, 0.001]}},
        },
        format="json",
    )
    assert response.status_code == 201, response.data
    run_id = response.data["id"]
    data = wait_for(auth_client, run_id, timeout=90, interval=0.5)
    log = (training_dir / "runs" / run_id / "output.log").read_text(errors="replace")
    assert data["status"] == "completed", f"{data['message']}\n{log[-3000:]}"
    assert data["trials_done"] == 2
    assert data["best_trial"]["trial"] in (1, 2)
    assert data["metrics"]["val_loss"] == data["best_trial"]["val_loss"]
    assert data["metrics"]["test_loss"] is not None
    assert data["state"]["test_size"] > 0 and data["has_weights"] is True
    assert data["message"].startswith("Finished 2 trials. Best: trial ")


@pytest.mark.skipif(not torch_installed(), reason="PyTorch is not installed")
def test_real_training_on_csv_can_be_stopped(auth_client, training_dir, settings):
    if importlib.util.find_spec("brickdnn.training.runner") is None:
        pytest.skip("brickdnn.training.runner is not available")
    settings.TRAINING_RUNNER_COMMAND = None
    dataset_id = upload(auth_client, "iris.csv", iris_csv(rows=60)).data["id"]
    config = {"epochs": 1000, "batch_size": 4, "device": "cpu"}
    response = start(auth_client, dataset={"kind": "upload", "id": dataset_id}, config=config)
    assert response.status_code == 201, response.data
    run_id = response.data["id"]
    data = wait_for(auth_client, run_id, statuses=("running", *TERMINAL_STATUSES), timeout=90)
    assert data["status"] == "running", data["message"]
    assert auth_client.post(f"/api/training/runs/{run_id}/stop/").data["stop_requested"]
    data = wait_for(auth_client, run_id, timeout=60, interval=0.25)
    log = (training_dir / "runs" / run_id / "output.log").read_text(errors="replace")
    assert data["status"] == "stopped", f"{data['message']}\n{log[-3000:]}"
    assert data["task"] == "classification"


def test_non_finite_numbers_in_the_state_become_null(auth_client, training_dir, settings):
    use_runner(settings, "hang")
    run_id = start(auth_client).data["id"]
    wait_for_state(training_dir, run_id)
    directory = training_dir / "runs" / run_id
    state = read_state(directory)
    state["history"] = [
        {"epoch": 1, "train_loss": float("inf"), "val_loss": float("nan"), "seconds": 1.0}
    ]
    state["status"] = "failed"
    (directory / "state.json").write_text(json.dumps(state))  # allows NaN/Infinity
    response = auth_client.get(f"/api/training/runs/{run_id}/")
    assert response.status_code == 200
    assert response.data["metrics"]["train_loss"] is None
    assert response.data["state"]["history"][0]["val_loss"] is None
    assert auth_client.get("/api/training/runs/").status_code == 200


def test_kill_ignores_a_reused_pid(training_dir):
    """A pid from an old run that now belongs to another process is never killed."""
    other = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True
    )
    try:
        run = TrainingRun(pk=uuid.uuid4(), pid=other.pid, status="running")
        training_service._kill(run, wait=0.2)
        assert other.poll() is None  # still alive: its command line is not this run's
    finally:
        other.kill()
        other.wait(timeout=5)
