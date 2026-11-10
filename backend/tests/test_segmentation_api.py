"""Segmentation in the web API and the MCP server: uploads (web, upload link), datasets,
checks, real trainings with the built-in Shapes and an uploaded dataset, the preview images
and who may see them, predictions, storage, the training queue and compact MCP answers."""

# ruff: noqa: F811 (fixtures are imported from other test modules)

from __future__ import annotations

import base64
import io
import json
import time

import pytest
from django.contrib.auth import get_user_model
from rest_framework.test import APIClient

pytest.importorskip("PIL")
np = pytest.importorskip("numpy")
from PIL import Image  # noqa: E402

from brick_dnn_core import budgets, training_service  # noqa: E402
from brick_dnn_core.models import Dataset, Project, ProjectMember, TrainingRun  # noqa: E402
from brick_dnn_core.training_worker import Worker  # noqa: E402
from brickdnn import EXAMPLES  # noqa: E402
from brickdnn.training import torch_installed  # noqa: E402
from brickdnn.training.shapes import shapes_sample  # noqa: E402

from .segmentation_helpers import (  # noqa: E402
    SIZE,
    dataset_files,
    flat_net,
    png,
    tiny_unet,
    zip_bytes,
)
from .test_budgets import limits  # noqa: E402, F401 (a fixture)
from .test_mcp import call, mcp_client  # noqa: E402
from .test_training_api import (  # noqa: E402, F401
    DEMO_GRAPH,
    training_dir,
    upload,
    wait_for,
)

pytestmark = pytest.mark.django_db

needs_torch = pytest.mark.skipif(not torch_installed(), reason="PyTorch is not installed")
SHAPES = {"kind": "builtin", "id": "shapes"}
NET = tiny_unet(classes=4, size=32, width=16)
QUICK = {
    "epochs": 3,
    "batch_size": 8,
    "learning_rate": 0.01,
    "max_samples": 120,
    "seed": 3,
    "device": "cpu",
}
User = get_user_model()


@pytest.fixture
def user():
    return User.objects.create_user("ada", password="correct-horse-battery")


@pytest.fixture
def web(user):
    client = APIClient()
    client.force_authenticate(user)
    return client


@pytest.fixture
def mcp(user):
    return mcp_client(user)


@pytest.fixture
def real_runner(training_dir, settings):
    """The real training process (PyTorch on the CPU) instead of the fake runner."""
    settings.TRAINING_RUNNER_COMMAND = None
    settings.TRAINING_MAX_RUNS = 2
    return training_dir


def client_of(user) -> APIClient:
    client = APIClient()
    client.force_authenticate(user)
    return client


def answer(client, tool, **arguments) -> tuple[dict, int]:
    result, text = call(client, tool, **arguments)
    assert "isError" not in result, text
    return result.get("structuredContent", text), len(text.encode())


def error_of(client, tool, **arguments) -> str:
    result, text = call(client, tool, **arguments)
    assert result.get("isError") is True, text
    return text


def start(client, graph=NET, dataset=SHAPES, config=QUICK, name="Shapes net", **extra):
    return client.post(
        "/api/training/runs/",
        {"name": name, "graph": graph, "dataset": dataset, "config": config, **extra},
        format="json",
    )


def train(client, training_dir, **kwargs) -> dict:
    response = start(client, **kwargs)
    assert response.status_code == 201, response.data
    data = wait_for(client, response.data["id"], timeout=120, interval=0.3)
    log = (training_dir / "runs" / data["id"] / "output.log").read_text(errors="replace")
    assert data["status"] == "completed", f"{data['message']}\n{log[-3000:]}"
    return data


def shape_file(index: int) -> str:
    return base64.b64encode(png(shapes_sample(index)[0])).decode()


# --------------------------------------------------------------------------- #
# Uploads and datasets
# --------------------------------------------------------------------------- #


def test_capabilities_list_shapes_and_the_limits(web, training_dir, monkeypatch):
    monkeypatch.setattr(training_service, "torchvision_installed", lambda: False)
    monkeypatch.setattr(training_service, "run_probe", lambda: {"torchvision": False})
    data = web.get("/api/training/").data
    shapes = data["builtin_datasets"][-1]
    assert shapes["id"] == "shapes" and shapes["task"] == "segmentation"
    assert shapes["available"] is True  # generated: no torchvision needed
    assert (shapes["train_size"], shapes["test_size"], shapes["download_mb"]) == (400, 100, 0)
    assert shapes["class_names"] == ["background", "circle", "square", "triangle"]
    assert data["builtin_datasets"][0]["available"] is False
    assert data["builtin_datasets"][0]["task"] == "classification"
    assert data["segmentation"] == {
        "max_classes": 32,
        "max_images": 20_000,
        "max_input_side": 512,
        "batch_memory_mb": 900,
        "losses": ["cross_entropy", "cross_entropy_dice"],
    }
    # Without torchvision, Shapes still passes the check; MNIST does not.
    check = web.post("/api/training/check/", {"graph": NET, "dataset": SHAPES}, format="json")
    assert check.data["ok"] is True
    mnist = EXAMPLES["cnn_mnist"]["graph"]
    check = web.post(
        "/api/training/check/",
        {"graph": mnist, "dataset": {"kind": "builtin", "id": "mnist"}},
        format="json",
    )
    assert "torchvision is not installed" in check.data["errors"][0]


def test_upload_a_segmentation_zip(web, user, training_dir):
    files = dataset_files(12, layout="split")
    files["classes.txt"] = b"nothing\nred\nblue\n"
    del files["val/masks/002.png"]
    response = upload(web, "squares.zip", zip_bytes(files), name="Squares")
    assert response.status_code == 201, response.data
    data = response.data
    assert (data["kind"], data["name"], data["mine"]) == ("segmentation", "Squares", True)
    info = data["info"]
    assert info["kind"] == "segmentation" and info["task"] == "segmentation"
    assert info["num_samples"] == 11 and info["num_classes"] == 3
    assert info["class_names"] == ["nothing", "red", "blue"]
    assert info["sample_shape"] == [3, SIZE, SIZE]
    assert info["predefined_split"] == [6, 2, 3]
    assert info["segmentation"]["mask_format"] == "indices"
    assert info["segmentation"]["unmatched_images"] == 1
    assert len(info["segmentation"]["class_pixel_share"]) == 3
    dataset = Dataset.objects.get(pk=data["id"])
    root = training_dir / "datasets" / str(dataset.pk) / "images"
    assert (root / "classes.txt").is_file() and (root / "train" / "masks" / "000.png").is_file()
    assert dataset.size_bytes == sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    assert budgets.storage_used(user)["datasets"] == dataset.size_bytes
    listed = web.get("/api/training/datasets/").data
    assert [d["kind"] for d in listed] == ["segmentation"]
    assert web.get(f"/api/training/datasets/{dataset.pk}/").data["info"] == info


def test_the_folders_decide_what_a_zip_is_unless_kind_says_so(web, training_dir):
    classes = {f"{label}/{i}.png": png(shapes_sample(i)[0]) for label in "ab" for i in range(3)}
    assert upload(web, "classes.zip", zip_bytes(classes)).data["kind"] == "images"
    response = upload(web, "classes.zip", zip_bytes(classes), kind="segmentation")
    assert response.status_code == 400
    assert response.data["detail"].startswith("No folders with images and masks were found.")

    files = dataset_files(6)
    assert upload(web, "seg.zip", zip_bytes(files)).data["kind"] == "segmentation"
    assert upload(web, "seg.zip", zip_bytes(files), kind="segmentation").status_code == 201
    # Read as class folders, images/ and masks/ are two classes.
    forced = upload(web, "seg.zip", zip_bytes(files), kind="images")
    assert forced.data["kind"] == "images"
    assert forced.data["info"]["class_names"] == ["images", "masks"]

    response = upload(web, "seg.zip", zip_bytes(files), kind="video")
    assert (
        response.status_code == 400
        and "'kind' says what a .zip file holds" in response.data["detail"]
    )
    response = upload(web, "data.csv", b"a,b\n1,2\n3,4\n5,6\n7,8\n", kind="segmentation")
    assert response.status_code == 400


@pytest.mark.parametrize(
    ("masks", "message"),
    [
        ("jpeg", "Save the masks as PNG"),
        ("missing", "The dataset has images but no masks."),
    ],
)
def test_refused_uploads_leave_nothing_behind(web, training_dir, masks, message):
    files = dataset_files(6, masks="jpeg" if masks == "jpeg" else "indices")
    if masks == "missing":
        files = {name: content for name, content in files.items() if name.startswith("images/")}
        files["masks/readme.txt"] = b"soon"
    response = upload(web, "bad.zip", zip_bytes(files), kind="segmentation")
    assert response.status_code == 400
    assert message in response.data["detail"]
    assert Dataset.objects.count() == 0
    assert list((training_dir / "datasets").iterdir()) == []


def test_zip_slip_and_the_storage_quota_also_hold_for_segmentation(web, user, training_dir, limits):
    files = {**dataset_files(6), "../../masks/evil.png": b"x"}
    response = upload(web, "slip.zip", zip_bytes(files))
    assert response.status_code == 400 and "unsafe path" in response.data["detail"]
    assert not (training_dir.parent / "masks").exists()

    limits.BUDGET_STORAGE_MB = 0.002  # 2 KB
    response = upload(web, "seg.zip", zip_bytes(dataset_files(8)))
    assert response.status_code == 403
    assert response.data["code"] == "storage_quota_exceeded"
    assert Dataset.objects.count() == 0 and list((training_dir / "datasets").iterdir()) == []


def test_upload_link_for_a_segmentation_zip(mcp, user, training_dir):
    assert "'kind' says what a .zip file holds" in error_of(
        mcp, "create_upload_link", purpose="training_dataset", filename="x.zip", kind="masks"
    )
    assert "'kind' says" in error_of(
        mcp, "create_upload_link", purpose="training_dataset", filename="x.csv", kind="segmentation"
    )
    link, _ = answer(
        mcp,
        "create_upload_link",
        purpose="training_dataset",
        filename="cells.zip",
        name="Cells",
        kind="segmentation",
    )
    from urllib.parse import urlsplit

    parts = urlsplit(link["url"])
    path = f"{parts.path}?{parts.query}"
    content = zip_bytes(dataset_files(8, masks="binary", classes=2))
    response = APIClient().put(path, content, content_type="application/octet-stream")
    assert response.status_code == 201, response.content
    body = response.json()
    dataset = Dataset.objects.get()
    assert body["reference"] == {"kind": "upload", "id": dataset.pk}
    assert dataset.kind == "segmentation" and dataset.owner == user and dataset.name == "Cells"
    summary = body["dataset"]
    assert summary["kind"] == "segmentation" and summary["task"] == "segmentation"
    assert summary["rows"] == 8 and summary["classes"] == 2
    assert summary["image_shape"] == [3, SIZE, SIZE]
    assert summary["mask_format"] == "binary"
    assert set(summary["class_pixel_share"]) == {"background", "foreground"}

    # A link that asks for segmentation says so when the ZIP holds class folders.
    link, _ = answer(
        mcp, "create_upload_link", purpose="training_dataset", filename="b.zip", kind="segmentation"
    )
    parts = urlsplit(link["url"])
    classes = {f"{label}/{i}.png": png(shapes_sample(i)[0]) for label in "ab" for i in range(3)}
    response = APIClient().put(
        f"{parts.path}?{parts.query}", zip_bytes(classes), content_type="application/octet-stream"
    )
    assert response.status_code == 400
    assert "The file cannot be used: No folders with images and masks" in response.json()["detail"]


def test_datasets_over_mcp_are_compact(mcp, web, user, training_dir):
    files = dataset_files(12, classes=3)
    del files["masks/011.png"]
    dataset_id = upload(web, "squares.zip", zip_bytes(files), name="Squares").data["id"]
    options, size = answer(mcp, "get_training_options")
    assert size < 5000
    assert options["segmentation"]["max_classes"] == 32
    assert options["config_defaults"]["loss"] == "cross_entropy"
    shapes = options["builtin_datasets"][-1]
    assert (shapes["id"], shapes["task"], shapes["available"]) == ("shapes", "segmentation", True)
    assert options["your_datasets"] == [
        {
            "id": dataset_id,
            "name": "Squares",
            "kind": "segmentation",
            "owner": "ada",
            "mine": True,
            "rows": 11,
            "image_shape": [3, SIZE, SIZE],
            "task": "segmentation",
            "classes": 3,
            "prepared": False,
            "size_bytes": Dataset.objects.get().size_bytes,
            "reference": {"kind": "upload", "id": dataset_id},
        }
    ]
    detail, size = answer(mcp, "get_dataset", id=dataset_id)
    assert size < 1500
    assert detail["mask_format"] == "indices" and "ignore_index" not in detail
    assert list(detail["class_pixel_share"]) == ["background", "class 1", "class 2"]
    assert detail["image_sizes"] == {"most_common": [SIZE, SIZE]}
    assert detail["unmatched_images"] == 1 and detail["unmatched_examples"] == ["images/011.png"]
    assert detail["class_names"] == ["background", "class 1", "class 2"]
    assert "columns_note" not in detail


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #


def test_check_training_explains_and_fixes(mcp, web, training_dir):
    unet = EXAMPLES["unet"]["graph"]
    result, size = answer(mcp, "check_training", graph=unet, dataset=SHAPES)
    assert result["ok"] is True and result["fixes"] == [] and size < 4000
    assert result["plan"]["task"] == "segmentation"
    assert result["estimate"]["details"]["train_samples"] == 320

    # The default batch size does not fit: the answer says which one does.
    result, _ = answer(mcp, "check_training", graph=unet, dataset=SHAPES, config={"batch_size": 64})
    assert result["ok"] is False and result["plan"] is None and result["estimate"] is None
    assert "Set the batch size to" in result["errors"][0]

    one_channel = json.loads(json.dumps(unet))
    one_channel["nodes"][-1]["params"]["out_channels"] = 1
    assert one_channel["nodes"][-1]["id"] == "mask"
    result, _ = answer(mcp, "check_training", graph=one_channel, dataset=SHAPES)
    assert result["ok"] is False
    assert result["fixes"][0]["summary"] == "Output channels of the last layer 1 → 4"

    response = web.post(
        "/api/training/check/",
        {"graph": EXAMPLES["cnn_mnist"]["graph"], "dataset": SHAPES},
        format="json",
    )
    assert response.data["ok"] is False
    assert "Shapes is a segmentation dataset: every pixel has a class" in response.data["errors"][0]
    response = web.post(
        "/api/training/check/",
        {"graph": EXAMPLES["svm_classifier"]["graph"], "dataset": SHAPES},
        format="json",
    )
    assert "is a classical model" in response.data["errors"][0]

    # Starting what does not fit is refused the same way.
    response = start(web, graph=unet, config={"batch_size": 64, "epochs": 1})
    assert response.status_code == 400
    assert "A batch of 64 images of 64×64 pixels" in response.data["detail"]
    assert TrainingRun.objects.count() == 0


# --------------------------------------------------------------------------- #
# Trainings, previews and predictions
# --------------------------------------------------------------------------- #


@needs_torch
def test_train_on_shapes_end_to_end(web, user, real_runner):
    data = train(web, real_runner, config={**QUICK, "loss": "cross_entropy_dice"})
    assert data["task"] == "segmentation" and data["has_weights"] is True
    assert data["config"]["loss"] == "cross_entropy_dice"
    metrics = data["metrics"]
    assert set(metrics) >= {
        "train_loss",
        "val_loss",
        "val_iou",
        "val_dice",
        "val_pixel_accuracy",
        "test_loss",
        "test_iou",
        "test_dice",
        "test_pixel_accuracy",
    }
    assert metrics["val_accuracy"] is None and metrics["test_accuracy"] is None
    assert metrics["val_iou"] > 0.4 and metrics["test_iou"] > 0.4  # all background: 0.22
    assert metrics["val_pixel_accuracy"] > 0.9
    state = data["state"]
    kept = next(e for e in state["history"] if e["epoch"] == state["weights_epoch"])
    assert metrics["val_iou"] == kept["val_iou"]  # the metrics of the kept weights
    assert data["best"]["val_iou"] is not None
    assert state["segmentation"]["previews"]["count"] == 8
    assert len(state["segmentation"]["validation"]["class_iou"]) == 4
    assert "mean IoU" in data["message"]
    listed = web.get("/api/training/runs/").data[0]
    assert listed["task"] == "segmentation" and listed["metrics"]["test_iou"] == metrics["test_iou"]

    # Storage: the run counts with its weights, state and previews.
    run = TrainingRun.objects.get(pk=data["id"])
    directory = real_runner / "runs" / data["id"]
    previews = sum(p.stat().st_size for p in (directory / "previews").iterdir())
    assert 0 < previews < 200_000
    assert run.size_bytes == sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())
    assert run.size_bytes > (directory / "weights.pt").stat().st_size + previews
    assert budgets.storage_used(user)["training_runs"] == run.size_bytes

    # The previews: an index with every image as a data URL, and the files themselves.
    response = web.get(f"/api/training/runs/{data['id']}/previews/")
    assert response.status_code == 200
    index = response.data
    assert index["epoch"] == state["weights_epoch"] and index["split"] == "validation"
    assert index["class_names"] == ["background", "circle", "square", "triangle"]
    assert len(index["colors"]) == 4 and len(index["samples"]) == 8
    assert len(json.dumps(index)) < 300_000
    sample = index["samples"][0]
    assert 0 <= sample["iou"] <= 1 and sample["classes"][0] == 0
    for part in ("image", "truth", "prediction", "overlay"):
        assert sample[part]["file"] == f"0_{part}.png"
        encoded = sample[part]["url"].removeprefix("data:image/png;base64,")
        picture = Image.open(io.BytesIO(base64.b64decode(encoded)))
        assert picture.size == (32, 32)
        response = web.get(f"/api/training/runs/{data['id']}/previews/0_{part}.png")
        assert response.status_code == 200 and response["Content-Type"] == "image/png"
        assert b"".join(response.streaming_content) == base64.b64decode(encoded)
    bare = web.get(f"/api/training/runs/{data['id']}/previews/?embed=0").data
    assert bare["samples"][0]["image"] == {"file": "0_image.png"}
    for name in ("9_image.png", "0_weights.png", "0_image.png.bak"):
        response = web.get(f"/api/training/runs/{data['id']}/previews/{name}")
        assert response.status_code == 404

    # Deleting the run removes them.
    assert web.delete(f"/api/training/runs/{data['id']}/").status_code == 204


@needs_torch
def test_previews_follow_the_roles_of_a_shared_project(web, user, real_runner):
    project = Project.objects.create(owner=user, name="Shared", graph=NET)
    viewer = User.objects.create_user("vera", password="correct-horse-battery")
    outsider = User.objects.create_user("otto", password="correct-horse-battery")
    ProjectMember.objects.create(project=project, user=viewer, role="viewer")
    data = train(web, real_runner, project=project.pk)
    urls = [
        f"/api/training/runs/{data['id']}/previews/",
        f"/api/training/runs/{data['id']}/previews/0_overlay.png",
    ]
    for url in urls:
        assert client_of(viewer).get(url).status_code == 200  # members see the run
        assert client_of(outsider).get(url).status_code == 404  # others do not know it exists
        assert APIClient().get(url).status_code == 401
    # A run outside the project stays private.
    private = train(web, real_runner, name="Private")
    assert client_of(viewer).get(f"/api/training/runs/{private['id']}/previews/").status_code == 404
    # Members who leave lose access.
    ProjectMember.objects.filter(user=viewer).delete()
    assert client_of(viewer).get(urls[0]).status_code == 404


def test_runs_without_previews_answer_404(web, training_dir):
    response = web.post(
        "/api/training/runs/",
        {
            "name": "Demo",
            "graph": DEMO_GRAPH,
            "dataset": {"kind": "random"},
            "config": {"epochs": 1},
        },
        format="json",
    )
    run = wait_for(web, response.data["id"])
    assert "val_iou" not in run["metrics"]  # only segmentation runs have the keys
    response = web.get(f"/api/training/runs/{run['id']}/previews/")
    assert response.status_code == 404
    assert response.data["detail"] == "This run has no preview images."
    # A file that only looks like a preview is not served.
    directory = training_dir / "runs" / run["id"] / "previews"
    directory.mkdir()
    (directory / "index.json").write_text(
        json.dumps({"samples": [{"index": 0, "image": "../weights.pt", "truth": "0_truth.png"}]})
    )
    (directory / "0_truth.png").write_bytes(b"x" * (2 * 1024 * 1024))  # far too large
    assert web.get(f"/api/training/runs/{run['id']}/previews/").data["samples"] == []
    assert web.get(f"/api/training/runs/{run['id']}/previews/0_truth.png").status_code == 404


@needs_torch
def test_predict_a_segmentation(web, mcp, real_runner):
    run = train(web, real_runner)
    url = f"/api/training/runs/{run['id']}/predict/"
    response = web.post(url, {"image": shape_file(470)}, format="json")
    assert response.status_code == 200, response.data
    data = response.data
    assert data["task"] == "segmentation"
    assert data["class_names"] == ["background", "circle", "square", "triangle"]
    assert len(data["colors"]) == 4 and "mask_png" in data["mask_note"]
    (prediction,) = data["predictions"]
    assert (prediction["width"], prediction["height"]) == (32, 32)
    assert prediction["classes"][0]["class"] == "background"
    assert sum(c["share"] for c in prediction["classes"]) == pytest.approx(1.0, abs=1e-3)
    mask = Image.open(io.BytesIO(base64.b64decode(prediction["mask_png"].split(",")[1])))
    assert mask.size == (32, 32) and mask.mode == "P"

    # The same image as an array of pixels gives the same class map.
    picture = np.asarray(Image.fromarray(shapes_sample(470)[0]).resize((32, 32), Image.BILINEAR))
    response = web.post(
        url,
        {"inputs": [picture.transpose(2, 0, 1).tolist()], "input_scale": "raw", "mask": "rle"},
        format="json",
    )
    assert response.status_code == 200, response.data
    runs = response.data["predictions"][0]["mask_rle"]
    assert "mask_png" not in response.data["predictions"][0]
    rebuilt = np.concatenate(
        [np.full(n, v) for v, n in zip(runs[::2], runs[1::2], strict=True)]
    ).reshape(32, 32)
    assert (rebuilt == np.asarray(mask)).all()

    # Over MCP: two images, small answers.
    result, size = answer(
        mcp, "predict", run_id=run["id"], image=[shape_file(471), shape_file(472)]
    )
    assert len(result["predictions"]) == 2 and size < 6000
    assert all(p["mask_png"].startswith("data:image/png;base64,") for p in result["predictions"])
    result, size = answer(mcp, "predict", run_id=run["id"], image=shape_file(471), mask="none")
    assert size < 1200 and set(result["predictions"][0]) == {"width", "height", "classes"}

    # Errors say what the network takes.
    response = web.post(url, {"image": "nonsense!"}, format="json")
    assert response.status_code == 400
    assert response.data["detail"].startswith("The image is not base64 text.")
    assert "pass image files (PNG, JPEG …) as base64 text in 'image'" in response.data["inputs"]
    response = web.post(url, {"image": shape_file(1), "inputs": [[1]]}, format="json")
    assert response.data["detail"] == "Give either 'inputs' or 'image', not both."
    response = web.post(url, {"image": shape_file(1), "mask": "svg"}, format="json")
    assert response.data["detail"] == "mask must be one of: png, rle, none."
    not_an_image = base64.b64encode(b"plain text").decode()
    response = web.post(url, {"image": not_an_image}, format="json")
    assert response.status_code == 500
    assert "cannot be read as an image file" in response.data["detail"]
    assert "Give the samples in 'inputs', or an image file" in error_of(
        mcp, "predict", run_id=run["id"]
    )

    # The weights download explains the same for a script.
    info, _ = answer(mcp, "get_weights_download", id=run["id"])
    assert info["task"] == "segmentation"
    assert "classes = output.argmax(dim=1)[0].to(torch.uint8)" in info["how_to_use"]
    assert info["class_names"] == ["background", "circle", "square", "triangle"]
    assert "one of 4 classes for each of the 32×32 pixels" in info["inputs"]


@needs_torch
def test_image_files_also_work_for_classifiers(web, real_runner):
    files = {f"{label}/{i}.png": png(shapes_sample(i)[0]) for label in ("a", "b") for i in range(6)}
    dataset_id = upload(web, "classes.zip", zip_bytes(files)).data["id"]
    graph = {
        "nodes": [
            {"id": "in", "type": "Input", "params": {"shape": [3, 16, 16]}},
            {"id": "flat", "type": "Flatten", "params": {}},
            {"id": "out", "type": "Linear", "params": {"out_features": 2}},
        ],
        "edges": [{"source": "in", "target": "flat"}, {"source": "flat", "target": "out"}],
    }
    run = train(
        web,
        real_runner,
        graph=graph,
        dataset={"kind": "upload", "id": dataset_id},
        config={"epochs": 1, "batch_size": 4, "device": "cpu"},
    )
    response = web.post(
        f"/api/training/runs/{run['id']}/predict/", {"image": [shape_file(0)]}, format="json"
    )
    assert response.status_code == 200, response.data
    assert response.data["task"] == "classification"
    assert response.data["predictions"][0]["class"] in ("a", "b")
    assert "mask_note" not in response.data


@needs_torch
def test_upload_train_and_search_over_mcp(mcp, web, real_runner):
    files = dataset_files(24, masks="binary", classes=2)
    dataset_id = upload(web, "squares.zip", zip_bytes(files), name="Squares").data["id"]
    reference = {"kind": "upload", "id": dataset_id}
    graph = flat_net(classes=1)
    check, _ = answer(mcp, "check_training", graph=graph, dataset=reference)
    assert check["ok"] is True
    assert check["plan"]["targets"] == [{"kind": "mask", "num_classes": 2, "binary": True}]
    started, _ = answer(
        mcp,
        "start_training",
        name="Squares search",
        graph=graph,
        dataset=reference,
        config={"epochs": 3, "batch_size": 4, "seed": 1, "device": "cpu", "test_split": 0.2},
        tuning={"method": "grid", "space": {"learning_rate": [0.03, 0.0001]}},
    )
    assert started["task"] == "segmentation" and started["tuning"]["trials"] == 2
    wait_for(web, started["id"], timeout=120, interval=0.3)
    detail, size = answer(mcp, "get_training_run", id=started["id"])
    assert detail["status"] == "completed", detail
    assert size < 3500  # a search with a test set: the largest summary
    assert "class_iou" not in detail["test"]  # per class: once, by name, in "segmentation"
    assert detail["task"] == "segmentation" and detail["class_names"] == [
        "background",
        "foreground",
    ]
    assert detail["metrics"]["val_iou"] > 0.6 and detail["metrics"]["test_iou"] > 0.5
    assert detail["tuning"]["best_trial"]["params"] == {"learning_rate": 0.03}
    assert detail["tuning"]["best_trial"]["val_iou"] == detail["metrics"]["val_iou"]
    seg = detail["segmentation"]
    assert seg["binary"] is True and seg["loss"] == "cross_entropy"
    assert (
        set(seg["validation_class_iou"])
        == set(seg["test_class_iou"])
        == {
            "background",
            "foreground",
        }
    )
    assert seg["previews"].startswith("5 validation images with true and predicted masks")
    assert "/previews/" in seg["previews"]
    with_trials, _ = answer(
        mcp, "get_training_run", id=started["id"], include=["trials", "history"]
    )
    assert [t["status"] for t in with_trials["trials"]] == ["completed", "completed"]
    assert all("val_iou" in t for t in with_trials["trials"])
    assert all("val_iou" in e for e in with_trials["history"])
    listed, size = answer(mcp, "list_training_runs")
    assert size < 1000
    assert listed["runs"][0]["task"] == "segmentation"
    assert listed["runs"][0]["metrics"]["test_iou"] == detail["metrics"]["test_iou"]
    # The previews are those of the best trial, binary masks.
    index = web.get(f"/api/training/runs/{started['id']}/previews/").data
    assert index["class_names"] == ["background", "foreground"] and len(index["samples"]) == 5


def test_activity_log_names_the_iou(user):
    from brick_dnn_core.agent_activity import describe_tool

    result = {
        "id": "8d1f3f0e-6f43-4c0a-9d3b-1d2d6a4f0a11",
        "name": "U-Net",
        "status": "completed",
        "metrics": {"val_loss": 0.2, "val_iou": 0.71234, "test_iou": 0.69876, "test_loss": 0.3},
    }
    described = describe_tool(user, "get_training_run", {"id": result["id"]}, result)
    assert described.result == ": completed, test mean IoU 69.9 %"
    del result["metrics"]["test_iou"]
    described = describe_tool(user, "get_training_run", {"id": result["id"]}, result)
    assert described.result == ": completed, test loss 0.3"
    result["metrics"] = {"val_loss": 0.2, "val_iou": 0.71234}
    described = describe_tool(user, "get_training_run", {"id": result["id"]}, result)
    assert described.result == ": completed, validation mean IoU 71.2 %"
    described = describe_tool(
        user, "predict", {"run_id": result["id"], "image": ["AAAA", "BBBB"]}, None
    )
    assert described.phrase.startswith("predict 2 images with")
    described = describe_tool(user, "predict", {"run_id": result["id"], "image": "AAAA"}, None)
    assert described.phrase.startswith("predict 1 image with")


# --------------------------------------------------------------------------- #
# The training queue
# --------------------------------------------------------------------------- #


@needs_torch
def test_a_worker_trains_a_segmentation_from_the_queue(web, user, real_runner, limits, settings):
    limits.TRAINING_WORKER = True
    limits.WORKER_NAME = ""
    limits.WORKER_SLOTS = 0
    limits.WORKER_TIMEOUT = 60
    limits.WORKER_MAX_RESTARTS = 3
    limits.WORKER_DRAIN_SECONDS = 0
    limits.QUEUE_MAX_WAITING = None
    dataset_id = upload(web, "squares.zip", zip_bytes(dataset_files(16))).data["id"]
    response = start(
        web,
        graph=tiny_unet(classes=3, size=SIZE),
        dataset={"kind": "upload", "id": dataset_id},
        config={"epochs": 2, "batch_size": 4, "seed": 0, "device": "cpu"},
    )
    assert response.status_code == 201, response.data
    run_id = response.data["id"]
    assert response.data["status"] == "queued" and response.data["state"] is None
    assert web.get(f"/api/training/runs/{run_id}/previews/").status_code == 404

    worker = Worker(name="worker-1", slots=1)
    worker.start()
    deadline = time.monotonic() + 120
    while True:
        worker.step()
        if (
            run_id not in worker.jobs
            and not worker.ending
            and TrainingRun.objects.get(pk=run_id).status
            not in ("queued", "starting", "preparing", "running")
        ):
            break
        assert time.monotonic() < deadline, "the worker did not finish the training"
        time.sleep(0.1)
    data = web.get(f"/api/training/runs/{run_id}/").data
    log = (real_runner / "runs" / run_id / "output.log").read_text(errors="replace")
    assert data["status"] == "completed", f"{data['message']}\n{log[-3000:]}"
    assert data["task"] == "segmentation" and data["metrics"]["val_iou"] is not None
    assert data["state"]["train_size"] == 13 and data["state"]["val_size"] == 3
    previews = web.get(f"/api/training/runs/{run_id}/previews/").data
    assert len(previews["samples"]) == 3
    assert previews["class_names"] == ["background", "class 1", "class 2"]
    run = TrainingRun.objects.get(pk=run_id)
    assert run.size_bytes and run.cpu_seconds > 0
