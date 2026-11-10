"""The example "U-Net classifier (Fashion-MNIST)": a U-Net with a classification head."""

from __future__ import annotations

from brickdnn import EXAMPLES, analyze
from brickdnn.training import builtin_info, check_compatibility


def test_it_follows_the_unet_example():
    ids = list(EXAMPLES)
    assert ids.index("unet_classifier") == ids.index("unet") + 1
    orders = [example["order"] for example in EXAMPLES.values()]
    assert orders == sorted(orders) and len(set(orders)) == len(orders)
    assert EXAMPLES["unet_classifier"]["name"] == "U-Net classifier (Fashion-MNIST)"


def test_shapes_and_size():
    graph = EXAMPLES["unet_classifier"]["graph"]
    analysis = analyze(graph)
    assert analysis.valid, analysis.errors
    shape = {nid: tuple(info.shape) for nid, info in analysis.nodes.items()}
    assert shape["input"] == (1, 32, 32)
    # Three halvings: 32 → 16 → 8 → 4, half the filters of the U-Net example.
    assert shape["pool3"] == (64, 4, 4)
    assert shape["bottleneck_relu2"] == (128, 4, 4)
    assert shape["skip1"] == (32, 32, 32)
    assert shape["dec1_relu2"] == (16, 32, 32)
    # Instead of the 1×1 mask convolution: pooling, flatten and 10 class scores.
    assert "mask" not in shape
    assert (shape["pool"], shape["flatten"], shape["classifier"]) == ((16, 1, 1), (16,), (10,))
    assert analysis.outputs == ["classifier"]
    unet = analyze(EXAMPLES["unet"]["graph"]).total_params
    assert analysis.total_params < unet / 3


def test_trains_on_fashion_mnist_with_resized_images():
    result = check_compatibility(
        EXAMPLES["unet_classifier"]["graph"], builtin_info("fashion_mnist")
    )
    assert result.ok, result.errors
    assert result.warnings == ["Fashion-MNIST images (28×28) are resized to 32×32."]
    assert result.plan.task == "classification"
    assert result.plan.class_names is not None and len(result.plan.class_names) == 10
