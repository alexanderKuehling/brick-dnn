"""Training a segmentation with PyTorch: loss and metrics against hand-made references, the
data sources, real short trainings on the CPU (multi-class, binary, with ignored pixels),
the preview images, the search, the runner, predictions and the command line."""

from __future__ import annotations

import base64
import io
import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")

import torch.nn.functional as F  # noqa: E402

from brickdnn.__main__ import main as brickdnn_main  # noqa: E402
from brickdnn.training import (  # noqa: E402
    TrainingConfig,
    builtin_info,
    check_compatibility,
    inference,
    inspect_segmentation_folder,
    runner,
    segdata,
)
from brickdnn.training import predict as predict_module  # noqa: E402
from brickdnn.training import state as st  # noqa: E402
from brickdnn.training.engine import Reporter, TrainingJob, train  # noqa: E402
from brickdnn.training.previews import PREVIEW_SAMPLES, PREVIEW_SIDE, save_previews  # noqa: E402
from brickdnn.training.segmentation import IGNORE_INDEX, class_colors, rgb  # noqa: E402
from brickdnn.training.segmetrics import (  # noqa: E402
    SegmentationLoss,
    SegmentationMetrics,
    confusion,
    confusion_metrics,
    predicted_classes,
)
from brickdnn.training.tuning import TuningConfig  # noqa: E402

from .segmentation_helpers import (  # noqa: E402
    SIZE,
    dataset_files,
    encode_mask,
    flat_net,
    png,
    sample,
    tiny_unet,
    write_files,
    zip_bytes,
)

SHAPES = builtin_info("shapes")
# Shapes at half its size and a small U-Net: learns most of it in about a second.
QUICK = {
    "epochs": 4,
    "batch_size": 8,
    "learning_rate": 0.01,
    "max_samples": 160,
    "seed": 3,
    "device": "cpu",
}
# What a network scores on Shapes that calls every pixel background (86 % of them are).
CHANCE_IOU = 0.22


class Recorder(Reporter):
    def __init__(self, stop_after_epochs: int | None = None):
        self.info: dict = {}
        self.history: list[dict] = []
        self.best = None
        self.logs: list[str] = []
        self.test = None
        self.kept = None
        self.segmentation = None
        self.trials: list[dict] = []
        self.search = None
        self.finished = None
        self.events: list[str] = []
        self.stop_after_epochs = stop_after_epochs

    def log(self, line):
        self.logs.append(line)

    def setup(self, info):
        self.info.update(info)

    def epoch_end(self, entry, best):
        self.history.append(entry)
        self.best = best

    def trial_end(self, trial, best_trial):
        self.trials.append(trial)

    def search_end(self, result):
        self.search = result

    def weights_kept(self, result):
        self.kept = result
        self.events.append("weights_kept")

    def test_end(self, result):
        self.test = result
        self.events.append("test_end")

    def segmentation_end(self, result):
        self.segmentation = result
        self.events.append("segmentation_end")

    def finish(self, status, message, weights):
        self.finished = (status, message, weights)
        self.events.append("finish")

    def should_stop(self):
        return self.stop_after_epochs is not None and len(self.history) >= self.stop_after_epochs


def shapes_job(**config) -> TrainingJob:
    return TrainingJob(
        graph=tiny_unet(classes=4, size=32, width=16),
        dataset=SHAPES,
        config=TrainingConfig.from_dict({**QUICK, **config}),
    )


def folder_job(path: Path, graph: dict, **config) -> TrainingJob:
    settings = {"epochs": 6, "batch_size": 4, "learning_rate": 0.02, "seed": 1, "device": "cpu"}
    return TrainingJob(
        graph=graph,
        dataset=inspect_segmentation_folder(path),
        config=TrainingConfig.from_dict({**settings, **config}),
        path=str(path),
    )


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def test_metrics_of_a_hand_made_example():
    truth = torch.tensor([[0, 0, 1, 1], [2, 2, IGNORE_INDEX, 1]])
    predicted = torch.tensor([[0, 1, 1, 1], [2, 0, 2, 1]])
    matrix = confusion(truth, predicted, 3)
    # Rows: the true class, columns: the predicted one. The ignored pixel is left out.
    assert matrix.tolist() == [[1, 1, 0], [0, 3, 0], [1, 0, 1]]
    metrics = confusion_metrics(matrix)
    # Class 0: 1 hit of 2 true and 2 predicted pixels: IoU 1 / (2 + 2 - 1), Dice 2 / (2 + 2).
    assert metrics["class_iou"] == pytest.approx([1 / 3, 3 / 4, 1 / 2])
    assert metrics["class_dice"] == pytest.approx([2 / 4, 6 / 7, 2 / 3])
    assert metrics["iou"] == pytest.approx((1 / 3 + 3 / 4 + 1 / 2) / 3)
    assert metrics["dice"] == pytest.approx((2 / 4 + 6 / 7 + 2 / 3) / 3)
    assert metrics["pixel_accuracy"] == pytest.approx(5 / 7)


def test_classes_that_do_not_occur_are_left_out_of_the_means():
    truth = torch.tensor([[0, 0, 1, 1]])
    predicted = torch.tensor([[0, 1, 1, 1]])
    metrics = confusion_metrics(confusion(truth, predicted, 4))
    assert metrics["class_iou"] == pytest.approx([0.5, 2 / 3, None, None])
    assert metrics["iou"] == pytest.approx((0.5 + 2 / 3) / 2)
    # A class that is only predicted (never true) counts, with an IoU of 0.
    wrong = confusion_metrics(confusion(truth, torch.tensor([[0, 3, 1, 1]]), 4))
    assert wrong["class_iou"] == pytest.approx([0.5, 1.0, None, 0.0])
    assert wrong["iou"] == pytest.approx(0.5)

    empty = confusion_metrics(torch.zeros((3, 3), dtype=torch.long))
    assert empty == {
        "iou": None,
        "dice": None,
        "pixel_accuracy": None,
        "class_iou": [None, None, None],
        "class_dice": [None, None, None],
    }


def reference_metrics(truth: np.ndarray, predicted: np.ndarray, classes: int) -> dict:
    """IoU and Dice per class straight from their definitions (sets of pixels)."""
    valid = truth != IGNORE_INDEX
    ious, dices = [], []
    for label in range(classes):
        true_pixels = (truth == label) & valid
        predicted_pixels = (predicted == label) & valid
        union = (true_pixels | predicted_pixels).sum()
        if union == 0:
            continue
        overlap = (true_pixels & predicted_pixels).sum()
        ious.append(overlap / union)
        dices.append(2 * overlap / (true_pixels.sum() + predicted_pixels.sum()))
    return {
        "iou": float(np.mean(ious)),
        "dice": float(np.mean(dices)),
        "pixel_accuracy": float(((truth == predicted) & valid).sum() / valid.sum()),
    }


@pytest.mark.parametrize("classes", [2, 5])
def test_metrics_added_up_over_batches_match_a_reference(classes):
    generator = torch.Generator().manual_seed(classes)
    scores = torch.randn((12, classes, 9, 7), generator=generator)
    truth = torch.randint(0, classes, (12, 9, 7), generator=generator)
    truth[torch.rand((12, 9, 7), generator=generator) < 0.1] = IGNORE_INDEX
    metrics = SegmentationMetrics(classes, binary=False)
    for start in range(0, 12, 5):  # batches of 5, 5 and 2
        metrics.add(scores[start : start + 5], truth[start : start + 5])
    result = metrics.result()
    expected = reference_metrics(truth.numpy(), scores.argmax(dim=1).numpy(), classes)
    for key, value in expected.items():
        assert result[key] == pytest.approx(value), key


def test_binary_predictions_threshold_the_logit_at_zero():
    logits = torch.tensor([[[[-2.0, 0.5], [3.0, -0.1]]]])
    assert predicted_classes(logits, binary=True).tolist() == [[[0, 1], [1, 0]]]
    truth = torch.tensor([[[0, 1], [0, IGNORE_INDEX]]])
    metrics = SegmentationMetrics(2, binary=True)
    metrics.add(logits, truth)
    result = metrics.result()
    assert metrics.matrix.tolist() == [[1, 1], [0, 1]]
    assert result["class_iou"] == pytest.approx([0.5, 0.5])
    assert result["pixel_accuracy"] == pytest.approx(2 / 3)


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #


def test_cross_entropy_is_the_mean_over_the_pixels_that_count():
    generator = torch.Generator().manual_seed(0)
    output = torch.randn((2, 3, 4, 4), generator=generator)
    target = torch.randint(0, 3, (2, 4, 4), generator=generator)
    target[0, 0] = IGNORE_INDEX
    loss = SegmentationLoss(3, binary=False)(output, target)
    log_probabilities = output.log_softmax(dim=1)
    by_hand = [
        -log_probabilities[n, target[n, y, x], y, x]
        for n in range(2)
        for y in range(4)
        for x in range(4)
        if target[n, y, x] != IGNORE_INDEX
    ]
    assert len(by_hand) == 28
    assert float(loss) == pytest.approx(float(sum(by_hand) / len(by_hand)), rel=1e-5)


def test_dice_term_of_a_hand_made_example():
    # Two pixels, two classes; the probabilities are chosen through the logits.
    probabilities = torch.tensor([[0.8, 0.2], [0.4, 0.6]])  # pixel × class
    output = probabilities.log().T.reshape(1, 2, 1, 2)
    target = torch.tensor([[[0, 1]]])
    plain = SegmentationLoss(2, binary=False)(output, target)
    with_dice = SegmentationLoss(2, binary=False, dice=True)(output, target)
    cross_entropy = -(math.log(0.8) + math.log(0.6)) / 2
    assert float(plain) == pytest.approx(cross_entropy, rel=1e-5)
    # Class 0: overlap 0.8, predicted 0.8 + 0.4, true 1. Class 1: overlap 0.6, predicted 0.8.
    dice_0 = (2 * 0.8 + 1) / (1.2 + 1 + 1)
    dice_1 = (2 * 0.6 + 1) / (0.8 + 1 + 1)
    assert float(with_dice) == pytest.approx(cross_entropy + 1 - (dice_0 + dice_1) / 2, rel=1e-5)


def test_binary_loss_and_its_dice_term():
    logits = torch.tensor([[[[2.0, -1.0, 0.5]]]])
    target = torch.tensor([[[1, 0, IGNORE_INDEX]]])
    loss = SegmentationLoss(2, binary=True)(logits, target)
    expected = F.binary_cross_entropy_with_logits(
        torch.tensor([2.0, -1.0]), torch.tensor([1.0, 0.0])
    )
    assert float(loss) == pytest.approx(float(expected), rel=1e-6)
    p1, p2 = torch.sigmoid(torch.tensor([2.0, -1.0])).tolist()
    dice = (2 * p1 + 1) / (p1 + p2 + 1 + 1)
    with_dice = SegmentationLoss(2, binary=True, dice=True)(logits, target)
    assert float(with_dice) == pytest.approx(float(expected) + 1 - dice, rel=1e-5)


def test_perfect_predictions_and_batches_of_ignored_pixels():
    target = torch.tensor([[[0, 1], [2, 1]]])
    output = F.one_hot(target, 3).permute(0, 3, 1, 2).float() * 50
    assert float(SegmentationLoss(3, False, dice=True)(output, target)) == pytest.approx(
        0, abs=1e-4
    )
    ignored = torch.full((1, 2, 2), IGNORE_INDEX)
    scores = torch.randn((1, 3, 2, 2), requires_grad=True)
    for dice in (False, True):
        loss = SegmentationLoss(3, False, dice=dice)(scores, ignored)
        assert float(loss.detach()) == 0.0
        loss.backward()  # nothing to learn, but no error and no NaN
    logit = torch.randn((1, 1, 2, 2), requires_grad=True)
    assert float(SegmentationLoss(2, True)(logit, ignored).detach()) == 0.0


def test_dice_gradient_pulls_towards_the_mask():
    target = torch.tensor([[[0, 1]]])
    output = torch.zeros((1, 2, 1, 2), requires_grad=True)
    SegmentationLoss(2, False, dice=True)(output, target).backward()
    # Raising the score of the true class lowers the loss.
    assert output.grad[0, 0, 0, 0] < 0 and output.grad[0, 1, 0, 1] < 0


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


def prepare(job: TrainingJob, seed: int = 0):
    plan = check_compatibility(job.graph, job.dataset).plan
    assert plan is not None
    generator = torch.Generator().manual_seed(seed)
    return segdata.segmentation_data(plan, job.dataset, job.path, job.config, Recorder(), generator)


def test_shapes_data_split_and_batches():
    job = TrainingJob(graph=tiny_unet(4, 64), dataset=SHAPES, config=TrainingConfig())
    data = prepare(job)
    assert (len(data.train), len(data.val), len(data.test)) == (320, 80, 100)
    inputs, targets = data.train.batch(torch.arange(5))
    assert inputs[0].shape == (5, 3, 64, 64) and inputs[0].dtype == torch.float32
    assert targets[0].shape == (5, 64, 64) and targets[0].dtype == torch.long
    assert set(targets[0].unique().tolist()) <= {0, 1, 2, 3}
    assert abs(float(inputs[0].mean())) < 1.0  # normalised with the statistics of Shapes
    pre = data.preprocessing
    assert pre["kind"] == "image" and pre["task"] == "segmentation" and pre["dataset"] == "shapes"
    assert pre["mask"] == {"resize": "nearest", "binary": False, "ignore_index": None}
    assert pre["class_names"] == ["background", "square", "circle", "triangle"]
    # The test images are the last hundred, whatever the seed; the rest is split by it.
    other = prepare(job, seed=1)
    assert torch.equal(data.test.indices, torch.arange(400, 500))
    assert not torch.equal(data.val.indices, other.val.indices)
    assert sorted(torch.cat([data.train.indices, data.val.indices]).tolist()) == list(range(400))


def test_shapes_resized_and_capped():
    job = TrainingJob(
        graph=tiny_unet(4, 32, channels=1),
        dataset=SHAPES,
        config=TrainingConfig.from_dict({"max_samples": 40}),
    )
    data = prepare(job)
    assert (len(data.train), len(data.val), len(data.test)) == (40, 10, 10)
    inputs, targets = data.val.batch(torch.arange(3))
    assert inputs[0].shape == (3, 1, 32, 32) and targets[0].shape == (3, 32, 32)
    # Nearest neighbour: only classes, no values in between.
    assert set(targets[0].unique().tolist()) <= {0, 1, 2, 3}


def test_masks_are_resized_with_nearest_neighbour():
    masks = torch.tensor([[[0, 3], [1, 2]]], dtype=torch.uint8)
    resized = segdata.resize_masks(masks, 4, 4)
    assert resized.dtype == torch.uint8
    assert resized[0].tolist() == [[0, 0, 3, 3], [0, 0, 3, 3], [1, 1, 2, 2], [1, 1, 2, 2]]
    assert segdata.resize_masks(masks, 2, 2) is masks


def test_folder_data_in_memory_and_from_disk_are_the_same(tmp_path, monkeypatch):
    write_files(tmp_path, dataset_files(12))
    job = folder_job(tmp_path, tiny_unet(3, SIZE), test_split=0.2)
    data = prepare(job)
    assert isinstance(data.train, segdata.MaskTensorSource)
    assert (len(data.train), len(data.val), len(data.test)) == (8, 2, 2)
    monkeypatch.setattr(segdata, "PRELOAD_BYTES", 0)
    reporter = Recorder()
    plan = check_compatibility(job.graph, job.dataset).plan
    disk = segdata.segmentation_data(
        plan, job.dataset, job.path, job.config, reporter, torch.Generator().manual_seed(0)
    )
    assert isinstance(disk.train.source, segdata.MaskFileSource)
    assert any("read from disk during training" in line for line in reporter.logs)
    for part in ("train", "val", "test"):
        indices = torch.arange(len(getattr(data, part)))
        a_inputs, a_targets = getattr(data, part).batch(indices)
        b_inputs, b_targets = getattr(disk, part).batch(indices)
        assert torch.equal(a_inputs[0], b_inputs[0]) and torch.equal(a_targets[0], b_targets[0])
    # The masks are the class maps the files were made from.
    order = torch.cat([disk.train.indices, disk.val.indices, disk.test.indices]).tolist()
    _, targets = data.train.batch(torch.arange(1))
    assert (targets[0][0].numpy() == sample(order[0])[1]).all()


def test_folder_split_from_folders_and_other_sizes(tmp_path):
    files = dataset_files(12, layout="split")
    picture, mask = sample(0, size=40)
    files["train/images/000.png"] = png(picture)  # larger than the others
    files["train/masks/000.png"] = encode_mask(mask)
    write_files(tmp_path, files)
    job = folder_job(tmp_path, tiny_unet(3, SIZE), val_split=0.5, test_split=0.3)
    data = prepare(job)
    assert (len(data.train), len(data.val), len(data.test)) == (6, 3, 3)  # the folders decide
    assert data.preprocessing["split"] == {"source": "folders", "counts": [6, 3, 3]}
    inputs, targets = data.train.batch(torch.arange(6))
    assert inputs[0].shape == (6, 3, SIZE, SIZE) and targets[0].shape == (6, SIZE, SIZE)
    assert set(targets[0].unique().tolist()) <= {0, 1, 2}


def test_a_folder_that_changed_is_refused(tmp_path):
    write_files(tmp_path, dataset_files(8))
    job = folder_job(tmp_path, tiny_unet(3, SIZE))
    (tmp_path / "masks" / "003.png").unlink()
    reporter = Recorder()
    result = train(job, reporter)
    assert result.status == "failed"
    assert result.message == (
        "The dataset has changed since it was checked. Upload or select it again."
    )


# --------------------------------------------------------------------------- #
# Real trainings
# --------------------------------------------------------------------------- #


def test_training_on_shapes_beats_chance_clearly(tmp_path):
    reporter = Recorder()
    weights, previews = tmp_path / "weights.pt", tmp_path / "previews"
    result = train(shapes_job(), reporter, weights, previews)
    assert result.status == "completed", result.message
    assert reporter.info["task"] == "segmentation"
    assert reporter.info["class_names"] == ["background", "square", "circle", "triangle"]
    assert (reporter.info["train_size"], reporter.info["val_size"]) == (160, 40)
    assert reporter.info["test_size"] == 40

    history = result.history
    assert len(history) == 4
    for entry in history:
        assert entry["train_accuracy"] is None and entry["val_accuracy"] is None
        assert 0 <= entry["val_iou"] <= entry["val_dice"] <= 1
        assert 0 <= entry["val_pixel_accuracy"] <= 1
    kept = next(e for e in history if e["epoch"] == result.weights_epoch)
    assert kept["val_loss"] == min(e["val_loss"] for e in history)  # the lowest validation loss
    assert kept["val_iou"] > CHANCE_IOU + 0.25, history
    assert kept["val_iou"] > history[0]["val_iou"]  # it rises
    assert result.best == {
        "epoch": kept["epoch"],
        "val_loss": kept["val_loss"],
        "val_accuracy": None,
        "val_iou": kept["val_iou"],
    }
    test = result.test
    assert test["accuracy"] is None and test["rmse"] is None and test["r2"] is None
    assert test["iou"] > CHANCE_IOU + 0.2 and test["pixel_accuracy"] > 0.9
    assert len(test["class_iou"]) == len(test["class_dice"]) == 4
    assert test["class_iou"][0] > 0.9  # the background
    assert f"alidation mean IoU {kept['val_iou'] * 100:.1f}%" in result.message
    assert f"test mean IoU {test['iou'] * 100:.1f}%" in result.message
    assert reporter.events[-3:] == ["test_end", "segmentation_end", "finish"]

    details = reporter.segmentation
    assert details["loss"] == "cross_entropy" and details["binary"] is False
    assert details["class_names"] == ["background", "square", "circle", "triangle"]
    assert details["colors"] == class_colors(4)
    assert details["validation"]["epoch"] == result.weights_epoch
    assert details["validation"]["iou"] == kept["val_iou"]
    assert len(details["validation"]["class_iou"]) == 4
    assert details["test"]["iou"] == test["iou"]
    assert details["test"]["class_iou"] == test["class_iou"]
    assert details["previews"] == {
        "count": PREVIEW_SAMPLES,
        "epoch": result.weights_epoch,
        "split": "validation",
    }
    assert weights.is_file()
    saved = json.loads((tmp_path / "weights.preprocessing.json").read_text())
    assert saved["task"] == "segmentation" and saved["kind"] == "image"

    # The same seed gives the same run.
    again = train(shapes_job(), Recorder())
    assert [e["val_iou"] for e in again.history] == [e["val_iou"] for e in history]


def test_previews_are_written_bounded_and_true(tmp_path):
    reporter = Recorder()
    previews = tmp_path / "previews"
    (previews / "old").mkdir(parents=True)  # left over from an earlier attempt
    result = train(shapes_job(), reporter, tmp_path / "weights.pt", previews)
    assert result.status == "completed"
    index = json.loads((previews / "index.json").read_text())
    assert index["epoch"] == result.weights_epoch and index["split"] == "validation"
    assert (index["width"], index["height"], index["input_size"]) == (32, 32, [32, 32])
    assert index["class_names"] == ["background", "square", "circle", "triangle"]
    assert index["colors"] == class_colors(4)
    assert len(index["samples"]) == PREVIEW_SAMPLES == 8
    names = sorted(p.name for p in previews.iterdir())
    assert len(names) == 1 + 4 * PREVIEW_SAMPLES and "old" not in names
    assert sum(p.stat().st_size for p in previews.iterdir()) < 200_000

    # The files show what the index says: the masks are class maps in the class colours.
    job = shapes_job()
    data = prepare(job, seed=QUICK["seed"])
    _, targets = data.val.batch(torch.arange(PREVIEW_SAMPLES))
    for number, entry in enumerate(index["samples"]):
        assert entry["index"] == number
        picture = Image.open(previews / entry["image"])
        truth = Image.open(previews / entry["truth"])
        prediction = Image.open(previews / entry["prediction"])
        overlay = Image.open(previews / entry["overlay"])
        assert picture.size == truth.size == prediction.size == overlay.size == (32, 32)
        assert picture.mode == "RGB" and overlay.mode == "RGB"
        assert truth.mode == "P" and prediction.mode == "P"
        true_map, predicted_map = np.asarray(truth), np.asarray(prediction)
        assert (true_map == targets[0][number].numpy()).all()
        assert truth.getpalette()[:6] == [*rgb(class_colors(4)[0]), *rgb(class_colors(4)[1])]
        expected = reference_metrics(true_map, predicted_map, 4)
        assert entry["iou"] == pytest.approx(expected["iou"])
        assert entry["pixel_accuracy"] == pytest.approx(expected["pixel_accuracy"])
        assert entry["classes"] == sorted(set(true_map.ravel()) | set(predicted_map.ravel()))
        # The overlay is the image where the background is predicted, tinted elsewhere.
        same = np.asarray(overlay) == np.asarray(picture)
        assert same[predicted_map == 0].all()
        if (predicted_map > 0).any():
            assert not same[predicted_map > 0].all()


def test_large_previews_are_scaled_down(tmp_path):
    images = torch.randint(0, 255, (10, 1, 300, 400), dtype=torch.uint8)
    truth = torch.randint(0, 2, (10, 300, 400), dtype=torch.uint8)
    truth[:, :10] = IGNORE_INDEX
    index = save_previews(tmp_path / "p", images, truth, 1 - truth.clamp(max=1), ["a", "b"], 3)
    assert len(index["samples"]) == PREVIEW_SAMPLES
    assert (index["width"], index["height"]) == (PREVIEW_SIDE, 144)
    assert index["input_size"] == [300, 400]
    assert Image.open(tmp_path / "p" / "0_image.png").size == (PREVIEW_SIDE, 144)
    assert Image.open(
        tmp_path / "p" / "9_image.png" if False else tmp_path / "p" / "7_truth.png"
    ).size == (
        PREVIEW_SIDE,
        144,
    )
    # Noise is the worst case for PNG: the folder still stays small.
    assert sum(p.stat().st_size for p in (tmp_path / "p").iterdir()) < 2 * 1024**2
    assert index["samples"][0]["iou"] == 0.0  # everything predicted wrongly, ignored pixels aside


def test_binary_training_with_one_output_channel(tmp_path):
    write_files(tmp_path / "data", dataset_files(24, masks="binary", classes=2))
    job = folder_job(tmp_path / "data", flat_net(classes=1), loss="cross_entropy_dice")
    reporter = Recorder()
    result = train(job, reporter, tmp_path / "weights.pt", tmp_path / "previews")
    assert result.status == "completed", result.message
    assert reporter.info["num_classes"] == 2
    assert reporter.info["class_names"] == ["background", "foreground"]
    assert reporter.info["preprocessing"]["mask"]["binary"] is True
    kept = next(e for e in result.history if e["epoch"] == result.weights_epoch)
    assert kept["val_iou"] > 0.8, result.history  # red squares on grey: easy
    details = reporter.segmentation
    assert details["binary"] is True and details["loss"] == "cross_entropy_dice"
    assert len(details["validation"]["class_iou"]) == 2
    assert details["test"] is None  # no test split
    assert details["previews"]["count"] == 5  # the validation images: 20 % of 24
    index = json.loads((tmp_path / "previews" / "index.json").read_text())
    assert set(
        np.unique(np.asarray(Image.open(tmp_path / "previews" / index["samples"][0]["prediction"])))
    ) <= {0, 1}


@pytest.mark.parametrize(
    ("test_split", "split", "count"), [(0.25, "test", 6), (0.0, "training", 8)]
)
def test_training_without_validation_images(tmp_path, test_split, split, count):
    write_files(tmp_path / "data", dataset_files(24, masks="binary", classes=2))
    job = folder_job(
        tmp_path / "data", flat_net(classes=1), epochs=2, val_split=0, test_split=test_split
    )
    reporter = Recorder()
    result = train(job, reporter, tmp_path / "weights.pt", tmp_path / "previews")
    assert result.status == "completed", result.message
    assert reporter.info["val_size"] == 0
    assert all(e["val_loss"] is None and e["val_iou"] is None for e in result.history)
    assert result.weights_epoch == 2 and reporter.best is None
    details = reporter.segmentation
    assert details["validation"] is None
    assert (details["test"] is not None) == (test_split > 0)
    # The previews come from the test images, or without them from the training images.
    assert details["previews"]["split"] == split and details["previews"]["count"] == count
    index = json.loads((tmp_path / "previews" / "index.json").read_text())
    assert index["split"] == split


def test_multi_class_training_ignores_marked_pixels(tmp_path):
    files = dataset_files(24)
    for index in range(24):
        mask = sample(index)[1].copy()
        mask[:, :2] = IGNORE_INDEX  # an unlabelled border
        files[f"masks/{index:03d}.png"] = encode_mask(mask)
    write_files(tmp_path / "data", files)
    job = folder_job(tmp_path / "data", tiny_unet(3, SIZE), test_split=0.2)
    assert job.dataset.segmentation["ignore_index"] == IGNORE_INDEX
    reporter = Recorder()
    result = train(job, reporter, tmp_path / "weights.pt", tmp_path / "previews")
    assert result.status == "completed", result.message
    assert all(math.isfinite(e["val_loss"]) for e in result.history)
    assert reporter.info["preprocessing"]["mask"]["ignore_index"] == IGNORE_INDEX
    assert reporter.segmentation["ignore_index"] == IGNORE_INDEX
    kept = next(e for e in result.history if e["epoch"] == result.weights_epoch)
    assert kept["val_iou"] > 0.6, result.history
    assert result.test["iou"] > 0.5
    # The border is grey in the true masks of the previews and does not count for the IoU.
    index = json.loads((tmp_path / "previews" / "index.json").read_text())
    truth = np.asarray(Image.open(tmp_path / "previews" / index["samples"][0]["truth"]))
    assert (truth[:, :2] == IGNORE_INDEX).all()
    prediction = np.asarray(Image.open(tmp_path / "previews" / index["samples"][0]["prediction"]))
    expected = reference_metrics(truth, prediction, 3)
    assert index["samples"][0]["iou"] == pytest.approx(expected["iou"])


def test_stopped_run_keeps_the_best_epoch_and_writes_previews(tmp_path):
    reporter = Recorder(stop_after_epochs=2)
    result = train(shapes_job(epochs=20), reporter, tmp_path / "w.pt", tmp_path / "previews")
    assert result.status == "stopped" and result.weights
    # The stop is noticed a moment later: an epoch takes less than that here.
    assert 2 <= len(result.history) < 20
    assert result.weights_epoch in [e["epoch"] for e in result.history]
    assert reporter.segmentation["validation"]["epoch"] == result.weights_epoch
    assert reporter.segmentation["test"] is None  # the test set is only measured at the end
    assert reporter.segmentation["previews"]["count"] == PREVIEW_SAMPLES
    assert (tmp_path / "previews" / "index.json").is_file()


def test_early_stopping_and_no_previews_without_a_folder():
    reporter = Recorder()
    result = train(shapes_job(epochs=30, patience=1, learning_rate=0.05), reporter)
    assert result.status == "completed"
    assert result.stopped_early and len(result.history) < 30
    assert reporter.segmentation["previews"] is None  # nowhere to write them
    assert reporter.segmentation["validation"]["epoch"] == reporter.kept["weights_epoch"]


def test_batch_that_does_not_fit_fails_with_the_reason(monkeypatch):
    monkeypatch.setenv("BRICKDNN_SEGMENTATION_MEMORY_MB", "16")
    result = train(shapes_job(batch_size=64), Recorder())
    assert result.status == "failed"
    assert result.message.startswith("A batch of 64 images of 32×32 pixels needs about")
    assert "Set the batch size to" in result.message


# --------------------------------------------------------------------------- #
# Hyperparameter search
# --------------------------------------------------------------------------- #


def test_search_ranks_trials_and_previews_the_best(tmp_path, monkeypatch):
    job = shapes_job(epochs=2)
    job.tuning = TuningConfig.from_dict(
        {"method": "grid", "space": {"learning_rate": [0.01, 0.0001], "batch_size": [8, 64]}}
    )
    # Batches of 64 do not fit into this "server": those trials fail on their own.
    monkeypatch.setenv("BRICKDNN_SEGMENTATION_MEMORY_MB", "30")
    reporter = Recorder()
    result = train(job, reporter, tmp_path / "weights.pt", tmp_path / "previews")
    assert result.status == "completed", result.message
    assert len(result.trials) == 4
    failed = [t for t in result.trials if t["status"] == "failed"]
    done = [t for t in result.trials if t["status"] == "completed"]
    assert len(failed) == 2 and all(t["params"]["batch_size"] == 64 for t in failed)
    assert all("A batch of 64 images" in t["message"] for t in failed)
    for trial in done:
        assert trial["val_accuracy"] is None
        assert 0 <= trial["val_iou"] <= trial["val_dice"] <= 1 and trial["val_pixel_accuracy"] > 0
    best = next(t for t in done if t["trial"] == result.best_trial)
    assert best["val_loss"] == min(t["val_loss"] for t in done)
    assert best["params"]["learning_rate"] == 0.01  # the other one barely learns in 2 epochs
    assert reporter.segmentation["validation"]["iou"] == best["val_iou"]
    assert reporter.segmentation["previews"]["epoch"] == best["weights_epoch"]
    assert "(2 failed)" in result.message and "validation mean IoU" in result.message
    assert (tmp_path / "previews" / "index.json").is_file()


# --------------------------------------------------------------------------- #
# The runner (what the web app starts)
# --------------------------------------------------------------------------- #


def test_runner_reports_a_segmentation_in_the_state(tmp_path):
    run_dir = tmp_path / "run"
    st.write_job(
        run_dir,
        {
            "name": "Shapes",
            "graph": tiny_unet(classes=4, size=32, width=16),
            "dataset": {"kind": "builtin", "path": None, "info": SHAPES.to_dict()},
            "config": {**QUICK, "epochs": 2},
            "tuning": None,
            "data_root": str(tmp_path / "builtin"),
        },
    )
    state = runner.run(run_dir)
    assert state["status"] == "completed", state["message"]
    assert state["task"] == "segmentation" and state["num_classes"] == 4
    assert [e["epoch"] for e in state["history"]] == [1, 2]
    assert set(state["history"][0]) >= {"val_iou", "val_dice", "val_pixel_accuracy"}
    assert state["best"]["val_iou"] is not None
    assert state["test"]["iou"] is not None and len(state["test"]["class_iou"]) == 4
    assert state["segmentation"]["previews"]["count"] == PREVIEW_SAMPLES
    assert state["segmentation"]["validation"]["epoch"] == state["weights_epoch"]
    assert state["preprocessing"]["task"] == "segmentation"
    assert (run_dir / st.WEIGHTS_FILE).is_file()
    assert (run_dir / st.PREVIEW_DIR / st.PREVIEW_INDEX_FILE).is_file()
    assert st.read_state(run_dir)["segmentation"] == state["segmentation"]
    assert st.initial_state()["segmentation"] is None


# --------------------------------------------------------------------------- #
# Predictions
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory):
    """A run directory with a tiny U-Net trained on Shapes, its plan and preprocessing."""
    run_dir = tmp_path_factory.mktemp("run")
    graph = tiny_unet(classes=4, size=32, width=16)
    st.write_job(
        run_dir,
        {
            "name": "Shapes",
            "graph": graph,
            "dataset": {"kind": "builtin", "path": None, "info": SHAPES.to_dict()},
            "config": QUICK,
            "tuning": None,
        },
    )
    state = runner.run(run_dir)
    assert state["status"] == "completed", state["message"]
    plan = check_compatibility(graph, SHAPES).plan.to_dict()
    return run_dir, plan, state["preprocessing"]


def shape_image(index: int) -> tuple[str, np.ndarray]:
    """A Shapes image as a base64 PNG file, and its mask."""
    from brickdnn.training.shapes import shapes_sample

    picture, mask = shapes_sample(index)
    return base64.b64encode(png(picture)).decode(), mask


def test_predict_image_files_as_class_maps(trained_run):
    run_dir, plan, preprocessing = trained_run
    files = [shape_image(450)[0], shape_image(451)[0]]
    prepared = inference.prepare_image_files(plan, preprocessing, files)
    assert (prepared.kind, prepared.count, prepared.tensors) == ("image", 2, [])
    request = {
        "images": prepared.files,
        "image": prepared.image,
        "segmentation": {"binary": False, "num_classes": 4, "mask": "png"},
    }
    result = predict_module.predict(run_dir, request)
    assert set(result) == {"masks"} and len(result["masks"]) == 2
    predictions = inference.read_masks(plan, preprocessing, result["masks"], 2)
    for index, prediction in zip((450, 451), predictions, strict=True):
        assert (prediction["width"], prediction["height"]) == (32, 32)
        assert prediction["mask_png"].startswith("data:image/png;base64,")
        decoded = Image.open(io.BytesIO(base64.b64decode(prediction["mask_png"].split(",")[1])))
        assert decoded.mode == "P" and decoded.size == (32, 32)
        classes = np.asarray(decoded)
        # The shares are those of the class map, largest first, and add up to one.
        shares = {c["class"]: c["share"] for c in prediction["classes"]}
        assert list(shares.values()) == sorted(shares.values(), reverse=True)
        assert sum(shares.values()) == pytest.approx(1.0, abs=1e-3)
        assert shares["background"] == pytest.approx(float((classes == 0).mean()), abs=1e-4)
        assert [c["index"] for c in prediction["classes"]] == [
            int(i)
            for i in np.argsort(-np.bincount(classes.ravel(), minlength=4), kind="stable")
            if (classes == i).any()
        ]
        # The network learned something: most pixels match the true mask (resized).
        truth = np.asarray(Image.fromarray(shape_image(index)[1]).resize((32, 32), Image.NEAREST))
        assert (classes == truth).mean() > 0.85
    assert len(json.dumps(predictions)) < 4000  # small enough for an assistant's context


def test_predict_run_lengths_and_no_mask(trained_run):
    run_dir, plan, preprocessing = trained_run
    files = [shape_image(452)[0]]
    request = {
        "images": files,
        "image": inference.image_request(preprocessing),
        "segmentation": {"binary": False, "num_classes": 4, "mask": "png"},
    }
    as_png = predict_module.predict(run_dir, request)["masks"][0]
    classes = np.asarray(Image.open(io.BytesIO(base64.b64decode(as_png["png"]))))
    request["segmentation"]["mask"] = "rle"
    as_rle = predict_module.predict(run_dir, request)["masks"][0]
    assert "png" not in as_rle and as_rle["counts"] == as_png["counts"]
    runs = as_rle["rle"]
    rebuilt = np.concatenate(
        [np.full(length, value) for value, length in zip(runs[::2], runs[1::2], strict=True)]
    )
    assert (rebuilt.reshape(32, 32) == classes).all()
    assert all(a != b for a, b in zip(runs[::2], runs[2::2], strict=False))  # runs are maximal
    request["segmentation"]["mask"] = "none"
    bare = predict_module.predict(run_dir, request)["masks"][0]
    assert set(bare) == {"width", "height", "counts"}
    prediction = inference.read_masks(plan, preprocessing, [bare], 1)[0]
    assert set(prediction) == {"width", "height", "classes"}


def test_predict_arrays_give_the_same_as_image_files(trained_run):
    run_dir, plan, preprocessing = trained_run
    from brickdnn.training.shapes import shapes_sample

    picture = np.asarray(Image.fromarray(shapes_sample(453)[0]).resize((32, 32), Image.BILINEAR))
    nested = picture.transpose(2, 0, 1).tolist()
    prepared = inference.prepare_inputs(plan, preprocessing, [nested], "raw")
    options = {"binary": False, "num_classes": 4, "mask": "rle"}
    arrays = predict_module.predict(run_dir, {"inputs": prepared.tensors, "segmentation": options})
    file = base64.b64encode(png(shapes_sample(453)[0])).decode()
    images = predict_module.predict(
        run_dir,
        {
            "images": [file],
            "image": inference.image_request(preprocessing),
            "segmentation": options,
        },
    )
    assert arrays["masks"][0]["rle"] == images["masks"][0]["rle"]


def test_run_lengths():
    classes = torch.tensor([[0, 0, 1], [1, 1, 2]], dtype=torch.uint8)
    assert predict_module.run_lengths(classes) == [0, 2, 1, 3, 2, 1]
    assert predict_module.run_lengths(torch.zeros((2, 2), dtype=torch.uint8)) == [0, 4]
    assert predict_module.run_lengths(torch.zeros((0, 2), dtype=torch.uint8)) == []


def test_predict_refuses_what_is_not_an_image(trained_run):
    run_dir, plan, preprocessing = trained_run
    image = inference.image_request(preprocessing)
    options = {"binary": False, "num_classes": 4, "mask": "png"}

    def error(files):
        with pytest.raises(predict_module.PredictionError) as raised:
            predict_module.predict(
                run_dir, {"images": files, "image": image, "segmentation": options}
            )
        return str(raised.value)

    not_an_image = base64.b64encode(b"just some text").decode()
    assert error([not_an_image]) == (
        "The image cannot be read as an image file (PNG, JPEG, BMP, GIF or WebP)."
    )
    assert error([shape_image(1)[0], "$$$"]) == "Image 2 is not base64 text."
    huge = base64.b64encode(png(np.zeros((5000, 10), dtype=np.uint8))).decode()
    assert "has 10×5000 pixels; at most 4096×4096 are read" in error([huge])


def test_image_files_are_checked_before_a_process_starts(trained_run):
    _, plan, preprocessing = trained_run
    good = shape_image(2)[0]
    data_url = "data:image/png;base64," + good
    assert inference.prepare_image_files(plan, preprocessing, data_url).files == [good]
    wrapped = "\n".join(good[i : i + 60] for i in range(0, len(good), 60))
    assert inference.prepare_image_files(plan, preprocessing, [wrapped]).files == [good]

    def error(image, **kwargs):
        with pytest.raises(inference.InvalidInputs) as raised:
            inference.prepare_image_files(plan, preprocessing, image, **kwargs)
        return str(raised.value)

    assert error(5).startswith("'image' must be an image file (PNG, JPEG …) as base64 text")
    assert error([]).startswith("'image' must be an image file")
    assert error("not base64 !!") == (
        "The image is not base64 text. Encode the bytes of the image file with base64."
    )
    assert error([good, "data:image/png,raw"]) == "Image 2: a data URL must be base64 encoded."
    assert error([good] * 9) == "9 images are too many: predict at most 8 at a time."
    assert "The images are too large: at most 900 kB of base64 text" in error("A" * 900_004)
    assert error(good, top_k=0) == "top_k must be a whole number from 1 to 10."
    tabular = {"kind": "tabular", "columns": ["a"], "mean": [0.0], "std": [1.0]}
    tabular_plan = {
        "task": "regression",
        "inputs": [{"shape": [1], "dtype": "float"}],
        "targets": [{"kind": "value", "shape": [1]}],
    }
    tabular.update(target_mean=0.0, target_std=1.0)
    with pytest.raises(inference.InvalidInputs, match="'image' is for networks trained on images"):
        inference.prepare_image_files(tabular_plan, tabular, good)
    assert inference.mask_format(None) == "png" and inference.mask_format("rle") == "rle"
    with pytest.raises(inference.InvalidInputs, match="mask must be one of: png, rle, none"):
        inference.mask_format("jpeg")


def test_inputs_and_how_to_use_describe_a_segmentation(trained_run, tmp_path):
    run_dir, plan, preprocessing = trained_run
    assert inference.is_segmentation(plan) and inference.kind_of(plan, preprocessing) == "image"
    assert inference.class_names(plan, preprocessing) == [
        "background",
        "square",
        "circle",
        "triangle",
    ]
    text = inference.describe_inputs(plan, preprocessing)
    assert "One RGB image of 32×32 pixels per sample" in text
    assert "pass image files (PNG, JPEG …) as base64 text in 'image'" in text
    assert text.endswith("The network answers with one of 4 classes for each of the 32×32 pixels.")

    # The snippet runs as it is, next to the model code, the weights and an image.
    from brickdnn import generate_code

    job = st.read_job(run_dir)
    (tmp_path / "shapes.py").write_text(generate_code(job["graph"], {"include_main": False}))
    (tmp_path / "shapes.pt").write_bytes((run_dir / st.WEIGHTS_FILE).read_bytes())
    (tmp_path / "example.png").write_bytes(base64.b64decode(shape_image(460)[0]))
    snippet = inference.usage_snippet(plan, preprocessing, "shapes", "shapes.pt")
    assert "classes = output.argmax(dim=1)[0].to(torch.uint8)" in snippet
    import os
    import subprocess
    import sys

    done = subprocess.run(
        [sys.executable, "-c", snippet],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines()[0].startswith("background: ")
    mask = np.asarray(Image.open(tmp_path / "mask.png"))
    assert mask.shape == (32, 32) and set(np.unique(mask)) <= {0, 1, 2, 3}

    binary_plan = {**plan, "targets": [{"kind": "mask", "num_classes": 2, "binary": True}]}
    binary = inference.usage_snippet(binary_plan, preprocessing, "m", "m.pt")
    assert "classes = (output[0, 0] > 0).to(torch.uint8)" in binary


def test_read_masks_checks_what_the_process_reports(trained_run):
    _, plan, preprocessing = trained_run
    mask = {"width": 2, "height": 1, "counts": [1, 0, 1, 0], "png": "AAAA"}
    assert inference.read_masks(plan, preprocessing, [mask], 1) == [
        {
            "width": 2,
            "height": 1,
            "classes": [
                {"class": "background", "index": 0, "share": 0.5},
                {"class": "circle", "index": 2, "share": 0.5},
            ],
            "mask_png": "data:image/png;base64,AAAA",
        }
    ]
    with pytest.raises(ValueError, match="unexpected number of samples"):
        inference.read_masks(plan, preprocessing, [mask], 2)
    with pytest.raises(ValueError, match="unexpected number of classes"):
        inference.read_masks(plan, preprocessing, [{**mask, "counts": [1, 1]}], 1)


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #


def test_cli_trains_on_shapes_and_prints_the_metrics(tmp_path, capsys):
    graph = tmp_path / "net.json"
    graph.write_text(json.dumps(tiny_unet(classes=4, size=32, width=16)))
    code = brickdnn_main(
        [
            "train",
            str(graph),
            "--data",
            "shapes",
            "--epochs",
            "2",
            "--batch-size",
            "8",
            "--lr",
            "0.01",
            "--max-samples",
            "80",
            "--seed",
            "3",
            "--device",
            "cpu",
            "--loss",
            "cross_entropy_dice",
            "--save",
            str(tmp_path / "net.pt"),
            "--previews",
            str(tmp_path / "out"),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0, out
    assert "Segmentation into 4 classes at 32×32 pixels (Shapes" in out
    assert "80 training, 20 validation and 20 test samples" in out
    lines = [line for line in out.splitlines() if line.startswith("Epoch")]
    assert len(lines) == 2
    assert "val IoU" in lines[0] and "val Dice" in lines[0] and "val pixel acc" in lines[0]
    assert "val acc " not in lines[0]
    test_line = next(line for line in out.splitlines() if line.startswith("Test:"))
    assert "mean IoU" in test_line and "Dice" in test_line and "pixel accuracy" in test_line
    assert "IoU per class (of the kept weights):" in out
    table = out[out.index("IoU per class") :].splitlines()
    assert table[1].split() == ["class", "val", "test"]
    assert [row.split()[0] for row in table[2:6]] == ["background", "square", "circle", "triangle"]
    assert f"Saved 8 preview images (image, mask, prediction, overlay) to {tmp_path / 'out'}" in out
    assert (tmp_path / "net.pt").is_file() and (tmp_path / "out" / "index.json").is_file()


def test_cli_reads_segmentation_folders_and_zips(tmp_path, capsys):
    files = dataset_files(16, masks="binary", classes=2)
    files["classes.txt"] = b"nothing\nsquare\n"
    write_files(tmp_path / "data", files)
    (tmp_path / "data.zip").write_bytes(zip_bytes(files))
    graph = tmp_path / "net.json"
    graph.write_text(json.dumps(flat_net(classes=1)))
    common = ["--epochs", "1", "--batch-size", "4", "--device", "cpu", "--seed", "0"]
    for source in ("data", "data.zip"):
        save = tmp_path / f"{source}.pt"
        code = brickdnn_main(
            ["train", str(graph), "--data", str(tmp_path / source), *common, "--save", str(save)]
        )
        out = capsys.readouterr().out
        assert code == 0, out
        assert "Segmentation into 2 classes at 16×16 pixels" in out
        assert "16 images with masks" in out
        assert "how likely a pixel is 'square'" in out
        assert "val IoU" in out and "Saved" in out and "preview images" not in out
        assert save.is_file()

    # A classifier on the same data gets the reason, as an error.
    classifier = tmp_path / "classifier.json"
    from .helpers import chain

    classifier.write_text(
        json.dumps(
            chain(
                ("Input", {"shape": [3, 16, 16]}), ("Flatten", {}), ("Linear", {"out_features": 2})
            )
        )
    )
    code = brickdnn_main(["train", str(classifier), "--data", str(tmp_path / "data"), *common])
    captured = capsys.readouterr()
    assert code == 1
    assert "is a segmentation dataset: every pixel has a class" in captured.err


def test_cli_search_shows_the_iou_of_every_trial(tmp_path, capsys):
    graph = tmp_path / "net.json"
    graph.write_text(json.dumps(tiny_unet(classes=4, size=32, width=8)))
    code = brickdnn_main(
        [
            "train",
            str(graph),
            "--data",
            "shapes",
            "--epochs",
            "1",
            "--max-samples",
            "40",
            "--device",
            "cpu",
            "--seed",
            "1",
            "--batch-size",
            "8",
            "--search",
            "grid",
            "--tune-lr",
            "0.01,0.001",
            "--save",
            str(tmp_path / "net.pt"),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0, out
    header = next(line for line in out.splitlines() if line.strip().startswith("trial"))
    assert header.split()[-2:] == ["val", "IoU"]
    assert out.count("val IoU") >= 4  # two epochs, two trial summaries, the table


# --------------------------------------------------------------------------- #
# The exported script
# --------------------------------------------------------------------------- #


def test_exported_script_of_the_unet_example_runs(tmp_path):
    """The export is the model and a check with a random batch; its training loop is only
    offered for tabular models, and the docstring does not promise one."""
    import subprocess
    import sys

    from brickdnn import EXAMPLES, generate_code

    code = generate_code(EXAMPLES["unet"]["graph"], {"include_training": True})
    assert "Input: (batch, 3, 64, 64)" in code and "Output: (batch, 4, 64, 64)" in code
    assert "--data" not in code and "def train" not in code
    assert "python this_file.py" in code
    script = tmp_path / "unet.py"
    script.write_text(code)
    done = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=300
    )
    assert done.returncode == 0, done.stderr
    assert "Output shape: (2, 4, 64, 64)" in done.stdout
