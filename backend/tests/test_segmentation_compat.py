"""Segmentation: which networks fit a dataset with a class for every pixel, the errors that
explain what does not fit, and the suggested fixes (no PyTorch needed)."""

from __future__ import annotations

import copy

import pytest

from brickdnn import EXAMPLES, analyze
from brickdnn.training import (
    DatasetInfo,
    TrainingConfig,
    TrainingPlan,
    builtin_info,
    check_compatibility,
    compat,
    inspect_csv,
    random_info,
    suggest_fixes,
)
from brickdnn.training.config import ConfigError, segmentation_memory_bytes
from brickdnn.training.estimate import estimate_training

from .helpers import chain
from .segmentation_helpers import flat_net, tiny_unet

SHAPES = builtin_info("shapes")
UNET = EXAMPLES["unet"]["graph"]


def folder(classes: int = 3, shape=(3, 100, 160), **details) -> DatasetInfo:
    names = (
        ["background", "foreground"]
        if classes == 2
        else ["background"] + [f"class {i}" for i in range(1, classes)]
    )
    return DatasetInfo(
        kind="segmentation",
        name="Scans",
        task="segmentation",
        num_samples=40,
        sample_shape=list(shape),
        class_names=names,
        segmentation={"mask_format": "indices", "ignore_index": None, **details},
    )


def with_params(graph, **changes):
    graph = copy.deepcopy(graph)
    for node in graph["nodes"]:
        if node["id"] in changes:
            node["params"].update(changes[node["id"]])
    return graph


# --------------------------------------------------------------------------- #
# The U-Net example and Shapes
# --------------------------------------------------------------------------- #


def test_unet_example_trains_on_shapes_as_it_is():
    result = check_compatibility(UNET, SHAPES)
    assert result.ok, result.errors
    assert result.fixes == [] and result.warnings == []
    plan = result.plan
    assert plan.task == "segmentation"
    assert plan.summary == (
        "Segmentation into 4 classes at 64×64 pixels (Shapes, 400 training and 100 test images)."
    )
    assert plan.targets == [{"kind": "mask", "num_classes": 4, "binary": False}]
    assert plan.image == {"channels": 3, "height": 64, "width": 64, "layout": "chw"}
    assert plan.class_names == ["background", "circle", "square", "triangle"]
    assert plan.num_classes == 4
    assert plan.segmentation["binary"] is False and plan.segmentation["ignore_index"] is None
    assert any("generated on the server" in note for note in plan.notes)
    assert any("mean IoU and Dice" in note for note in plan.notes)
    # A plan survives the trip through JSON (it is stored with the run).
    assert TrainingPlan.from_dict(plan.to_dict()) == plan
    # Plans of older runs have no "segmentation" key.
    old = {k: v for k, v in plan.to_dict().items() if k != "segmentation"}
    assert TrainingPlan.from_dict(old).segmentation is None


def test_other_sizes_and_channels_are_converted():
    graph = with_params(UNET, input={"shape": [1, 128, 128]})
    result = check_compatibility(graph, SHAPES)
    assert result.ok
    assert result.warnings == [
        "Shapes images (64×64) are resized to 128×128; masks are resized with nearest neighbour."
    ]
    assert "Shapes images are converted to grayscale." in result.plan.notes


def test_fixes_set_the_output_channels_and_the_input():
    graph = with_params(UNET, input={"shape": [3, 128, 128]}, mask={"out_channels": 1})
    result = check_compatibility(graph, SHAPES)
    assert not result.ok
    assert result.errors == [
        "The network outputs 1 value per pixel, which fits two classes, but Shapes has 4 "
        "classes. The last layer should output 4 channels (e.g. Conv2d with 4 output channels "
        "and kernel size 1)."
    ]
    # The Input works (images are resized), so only the last layer changes.
    assert result.fixes == [
        {
            "node": "mask",
            "type": "Conv2d",
            "params": {"out_channels": 4},
            "previous": {"out_channels": 1},
            "summary": "Output channels of the last layer 1 → 4",
        }
    ]

    flat = with_params(UNET, input={"shape": [784]}, mask={"out_channels": 10})
    assert not analyze(flat).valid  # a flat Input cannot feed a convolution
    assert check_compatibility(flat, SHAPES).fixes == []

    wrong = with_params(UNET, input={"shape": [4, 64, 64]})  # images have 1 or 3 channels
    fixes = check_compatibility(wrong, SHAPES).fixes
    assert [fix["summary"] for fix in fixes] == ["Input shape 4×64×64 → 3×64×64"]
    fixed = with_params(wrong, **{fix["node"]: fix["params"] for fix in fixes})
    assert check_compatibility(fixed, SHAPES).ok


def test_surplus_channels_are_a_warning_with_a_fix():
    graph = with_params(UNET, mask={"out_channels": 6})
    result = check_compatibility(graph, SHAPES)
    assert result.ok
    assert result.warnings == [
        "The network outputs 6 channels, but Shapes has only 4 classes. The extra channels "
        "are never the right answer."
    ]
    assert result.plan.class_names[4:] == ["class 4", "class 5"]
    assert result.plan.targets[0]["num_classes"] == 6
    assert [fix["params"] for fix in result.fixes] == [{"out_channels": 4}]


# --------------------------------------------------------------------------- #
# Binary and multi-class
# --------------------------------------------------------------------------- #


def test_one_channel_is_binary_for_two_classes():
    result = check_compatibility(flat_net(classes=1), folder(classes=2))
    assert result.ok and result.fixes == []
    assert result.plan.targets == [{"kind": "mask", "num_classes": 2, "binary": True}]
    assert result.plan.segmentation["binary"] is True
    assert result.plan.num_classes == 2
    assert result.plan.class_names == ["background", "foreground"]
    assert any("how likely a pixel is 'foreground'" in note for note in result.plan.notes)

    two = check_compatibility(flat_net(classes=2), folder(classes=2))
    assert two.ok and two.fixes == []
    assert two.plan.targets == [{"kind": "mask", "num_classes": 2, "binary": False}]

    # Too few channels for the classes: one per class is suggested, also from one channel.
    for channels in (1, 2):
        result = check_compatibility(flat_net(classes=channels), folder(classes=3))
        assert not result.ok
        assert [fix["params"] for fix in result.fixes] == [{"out_channels": 3}]
    assert "outputs 2 channels, but Scans has 3 classes" in result.errors[0]


def test_folder_notes_and_warnings():
    info = folder(
        classes=3,
        ignore_index=255,
        unmatched_images=2,
        unmatched_masks=0,
        unmatched_examples=["images/a.png"],
        mask_size_mismatches=1,
        class_pixel_share=[0.985, 0.01, 0.005],
    )
    info.predefined_split = [30, 6, 4]
    result = check_compatibility(tiny_unet(classes=3, size=32), info)
    assert result.ok
    assert result.plan.summary == (
        "Segmentation into 3 classes at 32×32 pixels (Scans, 40 images with masks)."
    )
    assert result.plan.segmentation["ignore_index"] == 255
    notes = " ".join(result.plan.notes)
    assert "Images are converted to RGB and resized to 32×32" in notes
    assert "Pixels with the mask value 255 are ignored" in notes
    warnings = " ".join(result.warnings)
    assert "Uses the split of the folders (30 training, 6 validation and 4 test images)" in warnings
    assert "Not used: 2 images without a mask (e.g. images/a.png)." in warnings
    assert "1 mask has another size than the image" in warnings
    assert "98% of the pixels are 'background'" in warnings
    assert "Cross-entropy + Dice" in warnings
    # With the Dice loss chosen, the hint is not needed.
    config = TrainingConfig.from_dict({"loss": "cross_entropy_dice", "batch_size": 8})
    assert "Dice" not in " ".join(check_compatibility(tiny_unet(3, 32), info, config).warnings)


def test_final_activation_is_a_warning():
    graph = flat_net(classes=3)
    graph["nodes"].append({"id": "act", "type": "Activation", "params": {"function": "sigmoid"}})
    graph["edges"].append({"source": "n3", "target": "act"})
    result = check_compatibility(graph, folder(classes=3))
    assert result.ok
    assert any("ends with a sigmoid activation" in warning for warning in result.warnings)
    # The last convolution is found behind the activation.
    wide = check_compatibility(graph, folder(classes=5))
    assert [fix["node"] for fix in wide.fixes] == ["n3"]


# --------------------------------------------------------------------------- #
# What does not fit
# --------------------------------------------------------------------------- #


def test_output_smaller_than_the_input():
    valid = chain(
        ("Input", {"shape": [3, 64, 64]}),
        ("Conv2d", {"out_channels": 8, "padding": "valid"}),
        ("Conv2d", {"out_channels": 4, "kernel_size": [1, 1]}),
    )
    result = check_compatibility(valid, SHAPES)
    assert not result.ok and result.fixes == []
    message = result.errors[0]
    assert "outputs 62×62 values for an image of 64×64 pixels" in message
    assert "convolutions with 'valid' padding cut off the border (e.g. 'n1'" in message

    pooled = chain(
        ("Input", {"shape": [3, 64, 64]}),
        ("Conv2d", {"out_channels": 8}),
        ("MaxPool2d", {}),
        ("Conv2d", {"out_channels": 4, "kernel_size": [1, 1]}),
    )
    message = check_compatibility(pooled, SHAPES).errors[0]
    assert "outputs 32×32 values" in message
    assert "every pooling layer needs an upsampling layer" in message
    assert "divisible by 2 (1 pooling layer)" in message


def test_uneven_pooling_gets_an_input_size_that_works():
    graph = tiny_unet(classes=4, size=30)
    graph["nodes"][0]["params"]["shape"] = [3, 33, 33]  # 33 → 16 → 32: the skip does not fit
    assert not analyze(graph).valid
    result = check_compatibility(graph, SHAPES)
    assert result.errors == ["The network has problems. Fix the layers marked in red first."]
    assert suggest_fixes(graph, SHAPES) == []  # only valid networks are adapted

    no_skip = chain(
        ("Input", {"shape": [3, 33, 33]}),
        ("Conv2d", {"out_channels": 8}),
        ("MaxPool2d", {}),
        ("ConvTranspose2d", {"out_channels": 4}),
    )
    result = check_compatibility(no_skip, SHAPES)
    assert "outputs 32×32 values for an image of 33×33 pixels" in result.errors[0]
    assert [fix["summary"] for fix in result.fixes] == ["Input shape 3×33×33 → 3×64×64"]


def test_input_shapes_that_cannot_be_images():
    flat = chain(("Input", {"shape": [12]}), ("Linear", {"out_features": 4}))
    result = check_compatibility(flat, SHAPES)
    assert len(result.errors) == 2
    assert result.errors[0] == (
        "Shapes is a segmentation dataset: the Input takes whole images. Set the Input shape "
        "to channels, height, width, e.g. 3, 64, 64."
    )
    assert "every pixel has a class" in result.errors[1]

    channels = with_params(tiny_unet(4, 32), input={"shape": [5, 32, 32]})
    assert "The Input has 5 channels" in check_compatibility(channels, SHAPES).errors[0]

    large = with_params(tiny_unet(4, 32), input={"shape": [3, 1024, 1024]})
    result = check_compatibility(large, SHAPES)
    assert result.errors == [
        "Segmentation trains on images of at most 512×512 pixels, but the Input is 1024×1024. "
        "Set a smaller Input size: the images and masks are resized to it."
    ]
    assert [fix["summary"] for fix in result.fixes] == ["Input shape 3×1024×1024 → 3×64×64"]


def test_folder_input_sizes_follow_the_images():
    graph = with_params(tiny_unet(3, 32), input={"shape": [5, 32, 32]})
    fixes = check_compatibility(graph, folder(shape=(1, 100, 160))).fixes
    assert [fix["params"] for fix in fixes] == [{"shape": [1, 32, 32]}]  # keeps the size

    huge = with_params(tiny_unet(3, 32), input={"shape": [3, 600, 600]})
    fixes = check_compatibility(huge, folder(shape=(3, 480, 640))).fixes
    assert [fix["params"] for fix in fixes] == [{"shape": [3, 96, 128]}]  # 640 → 128, by 32
    small = check_compatibility(huge, folder(shape=(3, 20, 20))).fixes
    assert [fix["params"] for fix in small] == [{"shape": [3, 32, 32]}]


def test_classifier_on_a_segmentation_dataset():
    result = check_compatibility(EXAMPLES["cnn_mnist"]["graph"], SHAPES)
    assert not result.ok and result.fixes == []
    assert result.errors == [
        "Shapes is a segmentation dataset: every pixel has a class, so the network must "
        "output one score per class for every pixel, e.g. shape (4, height, width). This "
        "network outputs shape (10) (one answer per image). Use a network that ends with a "
        "convolution and keeps the image size, e.g. the U-Net example."
    ]


def test_segmentation_network_on_other_data(tmp_path):
    result = check_compatibility(UNET, builtin_info("cifar10"))
    assert not result.ok and result.fixes == []
    assert result.errors[0].endswith(
        "A network with one output per pixel is trained on a segmentation dataset (images "
        "with masks), e.g. the built-in Shapes."
    )
    path = tmp_path / "data.csv"
    path.write_text("a,b,y\n" + "".join(f"{i},{i * 2},{i % 2}\n" for i in range(8)))
    result = check_compatibility(UNET, inspect_csv(path))
    assert any("segmentation dataset" in error for error in result.errors)
    # Demo data still fits every network: one value per pixel is a regression there.
    demo = check_compatibility(UNET, random_info())
    assert demo.ok and demo.plan.task == "demo"


@pytest.mark.parametrize("example", ["gp_regression", "svm_classifier"])
def test_classical_models_are_refused(example):
    for info in (SHAPES, folder()):
        result = check_compatibility(EXAMPLES[example]["graph"], info)
        assert not result.ok and result.fixes == []
        assert "is a classical model: it trains on tabular data" in result.errors[0]
        assert f"not on images like {info.name}" in result.errors[0]


# --------------------------------------------------------------------------- #
# Memory, settings and the estimate
# --------------------------------------------------------------------------- #


def test_batches_that_do_not_fit_into_memory(monkeypatch):
    analysis = analyze(UNET)
    fixed, sample = compat.segmentation_memory(analysis)
    assert fixed == 16 * analysis.total_params
    assert 12 * 1024**2 < sample < 24 * 1024**2  # measured: about 14 to 18 MB per image

    assert check_compatibility(UNET, SHAPES, TrainingConfig.from_dict({"batch_size": 32})).ok
    result = check_compatibility(UNET, SHAPES, TrainingConfig.from_dict({"batch_size": 64}))
    assert not result.ok and result.fixes == []
    fits = (segmentation_memory_bytes() - fixed) // sample
    assert result.errors == [
        "A batch of 64 images of 64×64 pixels needs about 1.0 GB of memory in this network, "
        f"but a training may use 900 MB on this server. Set the batch size to {fits} or less, "
        "or make the Input smaller."
    ]
    # Without settings (e.g. when a network is adapted), the batch size is not checked.
    assert check_compatibility(UNET, SHAPES).ok

    monkeypatch.setenv("BRICKDNN_SEGMENTATION_MEMORY_MB", "4000")
    assert check_compatibility(UNET, SHAPES, TrainingConfig.from_dict({"batch_size": 64})).ok
    monkeypatch.setenv("BRICKDNN_SEGMENTATION_MEMORY_MB", "40")
    result = check_compatibility(UNET, SHAPES, TrainingConfig.from_dict({"batch_size": 1}))
    assert result.errors[0].endswith("Make the Input smaller or use fewer channels in the layers.")
    monkeypatch.setenv("BRICKDNN_SEGMENTATION_MEMORY_MB", "nonsense")
    assert segmentation_memory_bytes() == 900 * 1024**2


def test_loss_setting():
    assert TrainingConfig().loss == "cross_entropy"
    assert TrainingConfig.from_dict({"loss": "Cross_Entropy_Dice"}).loss == "cross_entropy_dice"
    assert TrainingConfig.from_dict({"loss": ""}).loss == "cross_entropy"
    with pytest.raises(ConfigError, match="Loss must be one of: cross_entropy, cross_entropy_dice"):
        TrainingConfig.from_dict({"loss": "focal"})


def test_estimate_counts_the_images_of_a_segmentation():
    analysis = analyze(UNET)
    config = TrainingConfig.from_dict({"batch_size": 16, "epochs": 5})
    plan = check_compatibility(analysis, SHAPES, config).plan.to_dict()
    estimate = estimate_training(analysis, SHAPES, config, plan)
    assert estimate.details["train_samples"] == 320
    assert estimate.details["val_samples"] == 80
    assert estimate.details["test_samples"] == 100
    assert estimate.details["steps_per_epoch"] == 20
    assert estimate.details["forward_flops_per_sample"] > 1e9
    assert 60 < estimate.cpu_seconds < 1200
    twice = estimate_training(analysis, SHAPES, TrainingConfig.from_dict({"epochs": 10}), plan)
    assert twice.cpu_seconds > 1.8 * estimate.cpu_seconds - 10

    info = folder()
    info.predefined_split = [30, 6, 4]
    small = analyze(tiny_unet(3, 32))
    plan = check_compatibility(small, info).plan.to_dict()
    estimate = estimate_training(small, info, TrainingConfig(), plan, trials=3)
    assert (estimate.details["train_samples"], estimate.details["val_samples"]) == (30, 6)
    assert estimate.details["test_samples"] == 4 and estimate.details["trials"] == 3
    info.predefined_split = None
    estimate = estimate_training(small, info, TrainingConfig(), plan)
    assert estimate.details["train_samples"] + estimate.details["val_samples"] == 40
