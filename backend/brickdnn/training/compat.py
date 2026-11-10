"""Check whether a network can be trained on a dataset (no PyTorch needed).

:func:`check_compatibility` is used twice: by the API before a run starts (so the UI can
show what is wrong and how to fix it) and by the training engine, which follows the
resulting :class:`TrainingPlan` when it prepares the data.

Given a graph document (not an :class:`Analysis`), the result also lists ``fixes``: the
changes to the Input layer and the last layer that make the network fit the data, which the
UI can apply with one click (see :func:`suggest_fixes`).
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field
from typing import Any

from ..graph import Analysis, analyze
from ..layers import ClassicalModel, fmt_shape
from .config import TrainingConfig, classical_max_rows, segmentation_memory_bytes
from .datasets import BUILTIN_DATASETS, MAX_CLASSES, DatasetInfo
from .reporting import TrainingError
from .tabular import split_sizes

MAX_IMAGE_SIDE = 1024
# A flat Input takes images of an image folder only if they are at least this many pixels wide
# (otherwise e.g. 4 features would count as 2×2 grayscale images).
MIN_FLAT_SIDE = 8
# Input shape suggested for image folders when nothing better is known (images are resized).
DEFAULT_IMAGE_SHAPE = [3, 64, 64]
# Layers between the last Linear layer and the output that keep the number of features.
_PASS_THROUGH = ("Activation", "Dropout")
# Segmentation: the largest Input (height and width) a network is trained on. Every layer
# keeps its output for all pixels of a batch, so memory and CPU time grow with the area.
MAX_SEGMENTATION_SIDE = 512
# Input sizes suggested for segmentation folders, largest first (images are resized).
SEGMENTATION_SIDES = (128, 64)
# Layers that decide the number of output channels of a segmentation network.
_OUTPUT_CONVS = ("Conv2d", "ConvTranspose2d")
# Memory of a training step, in bytes per value that the layers output: 4 for the value
# (float32), and about as much again for gradients and the work space of the convolutions
# (measured with a U-Net on a CPU: 1.4 to 1.7 times the outputs of all layers).
_BYTES_PER_VALUE = 4 * 1.6
# Weights, their gradients and the two moments of Adam.
_BYTES_PER_PARAM = 4 * 4
SEGMENTATION_HINT = (
    "A network with one output per pixel is trained on a segmentation dataset (images with "
    "masks), e.g. the built-in Shapes."
)


@dataclass
class TrainingPlan:
    """How the data is fed to the network and which loss is used.

    ``inputs``  one entry per Input layer: ``{"shape": [...], "dtype": "float"|"long",
                "vocab": int|None}`` (``vocab`` = smallest Embedding vocabulary fed by it)
    ``targets`` one entry per output: ``{"kind": "class", "num_classes": K}`` (cross-entropy),
                ``{"kind": "value", "shape": [...]}`` (mean squared error) or, for
                segmentation, ``{"kind": "mask", "num_classes": K, "binary": bool}`` (a class
                for every pixel; ``binary``: the network outputs one channel, the score of
                class 1)
    ``image``   for image data: ``{"channels": C, "height": H, "width": W,
                "layout": "chw"|"hw"|"flat"}``. Images are converted to C channels and
                resized to H×W; ``hw`` drops the channel axis, ``flat`` flattens C·H·W.
    ``classical`` for classical models (scikit-learn), else None: ``{"node", "type",
                "estimator", "label", "task", "kernel", "scale", "max_rows"}`` (``kernel``:
                the starting kernel of a Gaussian process; ``max_rows``: the most training
                rows it is fitted on, see config.classical_max_rows)
    ``segmentation`` for segmentation, else None: ``{"binary": bool, "ignore_index": 255 |
                None, "sample_bytes": int, "fixed_bytes": int}`` (the estimated memory of a
                training step is ``fixed_bytes + batch size × sample_bytes``)
    """

    task: str  # classification | regression | segmentation | demo
    summary: str
    inputs: list[dict[str, Any]]
    targets: list[dict[str, Any]]
    image: dict[str, Any] | None = None
    num_classes: int | None = None
    class_names: list[str] | None = None
    notes: list[str] = field(default_factory=list)
    classical: dict[str, Any] | None = None
    segmentation: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingPlan:
        return cls(**data)


@dataclass
class Compatibility:
    """``fixes``: suggested parameter changes, see :func:`suggest_fixes`."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    plan: TrainingPlan | None = None
    fixes: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and self.plan is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "plan": self.plan.to_dict() if self.plan and self.ok else None,
            "fixes": copy.deepcopy(self.fixes),
        }


def _x(shape: list[int] | tuple[int, ...]) -> str:
    return "×".join(str(s) for s in shape)


def _count(n: int | None, word: str) -> str:
    if n is None:
        return word + "s"
    return f"{n:,} {word}" + ("" if n == 1 else "s")


def check_compatibility(
    graph: Any, dataset: DatasetInfo, config: TrainingConfig | None = None
) -> Compatibility:
    """Check ``graph`` (a graph document or an :class:`Analysis`) against ``dataset``.

    For a graph document, ``fixes`` are suggested as well (an analysis cannot be changed).
    ``config`` (optional) adds the checks that depend on the training settings, e.g. the
    row limit of classical models against Max. samples.
    """
    if isinstance(graph, Analysis):
        return _check(graph, dataset, config)
    analysis = analyze(graph)
    result = _check(analysis, dataset, config)
    result.fixes = suggest_fixes(graph, dataset, analysis, result)
    return result


def _check(
    analysis: Analysis, dataset: DatasetInfo, config: TrainingConfig | None = None
) -> Compatibility:
    result = Compatibility()
    if not analysis.valid:
        result.errors.append("The network has problems. Fix the layers marked in red first.")
        return result

    inputs = _input_specs(analysis)
    if analysis.family == "classical":
        _check_classical(result, analysis, dataset, inputs, config)
        if result.errors:
            result.plan = None
        return result
    if dataset.kind == "random":
        result.plan = _demo_plan(analysis, inputs)
        return result
    if dataset.kind not in ("builtin", "csv", "images", "segmentation"):
        result.errors.append(f"Unknown dataset kind '{dataset.kind}'.")
        return result

    name = dataset.name
    if len(analysis.inputs) != 1:
        result.errors.append(
            f"Training on {name} needs a network with exactly one Input layer. "
            "Networks with several inputs can only be trained on demo data."
        )
    if len(analysis.outputs) != 1:
        result.errors.append(
            f"Training on {name} needs a network with exactly one output. "
            "Networks with several outputs can only be trained on demo data."
        )
    if result.errors:
        return result
    if inputs[0]["dtype"] == "long":
        result.errors.append(
            "This network expects integer token ids (Input data type 'long'). "
            "Text models can only be trained on demo data so far."
        )
        return result

    in_shape = list(analysis.nodes[analysis.inputs[0]].shape or ())
    out_shape = list(analysis.nodes[analysis.outputs[0]].shape or ())
    if is_segmentation(dataset):
        _check_segmentation(result, analysis, dataset, inputs, in_shape, out_shape, config)
    elif dataset.kind == "csv":
        _check_csv(result, dataset, inputs, in_shape, out_shape)
    else:
        _check_images(result, dataset, inputs, in_shape, out_shape)
    if result.errors:
        result.plan = None
    return result


def is_segmentation(dataset: DatasetInfo) -> bool:
    """True for data with a class for every pixel: segmentation folders and the built-in
    Shapes."""
    return dataset.kind == "segmentation" or dataset.task == "segmentation"


def _input_specs(analysis: Analysis) -> list[dict[str, Any]]:
    specs = []
    for nid in analysis.inputs:
        info = analysis.nodes[nid]
        vocab = None
        if info.dtype == "long":
            sizes = [
                analysis.nodes[t].layer.p["num_embeddings"]  # type: ignore[union-attr]
                for t in info.outputs
                if analysis.nodes[t].type == "Embedding"
            ]
            vocab = min(sizes) if sizes else 2
        specs.append({"shape": list(info.shape or ()), "dtype": info.dtype, "vocab": vocab})
    return specs


def _demo_plan(analysis: Analysis, inputs: list[dict[str, Any]]) -> TrainingPlan:
    targets: list[dict[str, Any]] = []
    for nid in analysis.outputs:
        shape = list(analysis.nodes[nid].shape or ())
        if len(shape) == 1 and shape[0] > 1:
            targets.append({"kind": "class", "num_classes": shape[0]})
        else:
            targets.append({"kind": "value", "shape": shape})
    kinds = {t["kind"] for t in targets}
    what = (
        "classification"
        if kinds == {"class"}
        else "regression"
        if kinds == {"value"}
        else "classification and regression"
    )
    return TrainingPlan(
        task="demo",
        summary=f"Demo data: random inputs and a hidden rule to learn ({what}).",
        inputs=inputs,
        targets=targets,
        num_classes=targets[0].get("num_classes") if len(targets) == 1 else None,
        notes=[
            "Demo data checks whether the network can learn at all. "
            "The numbers say nothing about real-world accuracy."
        ],
    )


def _check_classes(
    result: Compatibility, out_shape: list[int], num_classes: int, what: str
) -> int | None:
    """The output must be (K,) with K >= num_classes. Returns K or None."""
    hint = f"The last layer should output {num_classes} values (e.g. Linear with {num_classes} outputs)."
    if len(out_shape) != 1:
        per_pixel = f" {SEGMENTATION_HINT}" if len(out_shape) == 3 else ""
        result.errors.append(
            f"The network outputs shape {fmt_shape(tuple(out_shape))}, but {what} needs one "
            f"score per class. {hint}{per_pixel}"
        )
        return None
    k = out_shape[0]
    if k < num_classes:
        result.errors.append(
            f"The network outputs {_count(k, 'value')}, but {what} has {num_classes} classes. {hint}"
        )
        return None
    if k > num_classes:
        result.warnings.append(
            f"The network outputs {k} values, but {what} has only {num_classes} classes. "
            f"The extra outputs are never the right answer."
        )
    return k


def _shown(names: list[str], limit: int = 6) -> str:
    return ", ".join(names[:limit]) + (" …" if len(names) > limit else "")


def _predefined_split_note(counts: list[int]) -> str:
    train, val, test = [*counts, 0, 0, 0][:3]
    if test:
        parts = f"{train:,} training, {val:,} validation and {test:,} test rows"
    else:
        parts = f"{train:,} training and {val:,} validation rows, no test set"
    return (
        f"Uses the split of the data preparation ({parts}); the validation and test split "
        "settings are ignored."
    )


def _check_csv(
    result: Compatibility,
    info: DatasetInfo,
    inputs: list[dict[str, Any]],
    in_shape: list[int],
    out_shape: list[int],
) -> None:
    n = info.num_features or 0
    if in_shape != [n]:
        cols = ", ".join(info.columns[:-1][:8]) + (" …" if n > 8 else "")
        result.errors.append(
            f"The Input expects shape {fmt_shape(tuple(in_shape))}, but the CSV file has "
            f"{_count(n, 'feature column')} ({cols}). Set the Input shape to {n}."
        )
    if len(out_shape) != 1:
        per_pixel = f" {SEGMENTATION_HINT}" if len(out_shape) == 3 else ""
        result.errors.append(
            f"The network outputs shape {fmt_shape(tuple(out_shape))}. For CSV data it must "
            f"output 1 value (regression) or one value per class (classification).{per_pixel}"
        )
    if result.errors:
        return

    k = out_shape[0]
    target = info.columns[-1] if info.columns else "target"
    rows = _count(info.num_samples, "row")
    # Made in the Data tab: class targets are written as indices, `class_names` are the labels.
    prepared = info.predefined_split is not None
    if prepared:
        result.warnings.append(_predefined_split_note(info.predefined_split or []))
    if k == 1:
        if info.target_kind == "text":
            classes = info.num_classes or 0
            result.errors.append(
                f"The target column '{target}' contains text labels ({classes} classes), so this "
                f"is a classification task: the last layer needs {classes} outputs, one per class."
            )
            return
        if prepared and info.target_kind == "integer" and info.class_names:
            result.warnings.append(
                f"The target column '{target}' holds {len(info.class_names)} classes "
                f"({_shown(info.class_names)}). With 1 output the network is trained as a "
                f"regression. For classification, give the last layer {len(info.class_names)} "
                "outputs instead."
            )
        elif info.target_kind == "integer" and info.class_names and len(info.class_names) <= 10:
            result.warnings.append(
                f"The target column '{target}' only contains the whole numbers "
                f"{', '.join(info.class_names)}. With 1 output the network is trained as a "
                f"regression. For classification, give the last layer {len(info.class_names)} "
                "outputs instead."
            )
        result.plan = TrainingPlan(
            task="regression",
            summary=f"Regression: predict '{target}' from {_count(n, 'feature')} ({rows}).",
            inputs=inputs,
            targets=[{"kind": "value", "shape": [1]}],
            notes=["Features are standardised (mean 0, standard deviation 1)."],
        )
        return

    if info.target_kind == "number":
        result.errors.append(
            f"The target column '{target}' contains decimal numbers, which is a regression "
            f"task. Give the last layer 1 output, or use class indices 0..{k - 1} as targets."
        )
        return
    if info.target_kind == "integer":
        low, high = info.target_range or [0, 0]
        if low < 0 or high > k - 1:
            result.errors.append(
                f"The target column '{target}' holds values from {low:g} to {high:g}, but with "
                f"{k} outputs the class indices must be between 0 and {k - 1}. "
                f"Set the last layer to {math.floor(high) + 1} outputs or relabel the classes."
            )
            return
        class_names = [str(i) for i in range(k)]
        notes = [f"The target column holds the class index 0..{k - 1}."]
        if prepared and info.class_names and len(info.class_names) <= k:
            class_names = list(info.class_names)
            class_names += [f"class {i}" for i in range(len(class_names), k)]  # unused outputs
            notes = [f"Classes in index order: {_shown(info.class_names)}."]
    else:
        class_names = list(info.class_names or [])
        if _check_classes(result, out_shape, len(class_names), f"the column '{target}'") is None:
            return
        shown = ", ".join(class_names[:6]) + (" …" if len(class_names) > 6 else "")
        notes = [f"Text labels are mapped to class indices in alphabetical order: {shown}."]
        class_names += [f"class {i}" for i in range(len(class_names), k)]  # unused outputs
    result.plan = TrainingPlan(
        task="classification",
        summary=f"Classification into {k} classes from {_count(n, 'feature')} ({rows}).",
        inputs=inputs,
        targets=[{"kind": "class", "num_classes": k}],
        num_classes=k,
        class_names=class_names,
        notes=["Features are standardised (mean 0, standard deviation 1).", *notes],
    )


def _square_image(values: int) -> tuple[int, int] | None:
    """``(channels, side)`` if ``values`` is the size of a flattened square RGB image
    (3·side²) or grayscale image (side²) of at least MIN_FLAT_SIDE pixels, else None.
    At most one of the two fits: 3·a² = b² has no solution in whole numbers."""
    for channels in (3, 1):
        side = math.isqrt(values // channels)
        if channels * side * side == values and MIN_FLAT_SIDE <= side <= MAX_IMAGE_SIDE:
            return channels, side
    return None


def _check_images(
    result: Compatibility,
    info: DatasetInfo,
    inputs: list[dict[str, Any]],
    in_shape: list[int],
    out_shape: list[int],
) -> None:
    builtin = BUILTIN_DATASETS.get(info.builtin or "") if info.kind == "builtin" else None
    if info.kind == "builtin" and builtin is None:
        result.errors.append(f"Unknown built-in dataset '{info.builtin}'.")
        return
    name = info.name
    notes: list[str] = []
    image: dict[str, Any] | None = None
    rank = len(in_shape)

    if rank == 3:
        c, h, w = in_shape
        if c not in (1, 3):
            result.errors.append(
                f"The Input has {c} channels, but images have 1 (grayscale) or 3 (RGB). "
                "Set the Input shape to 1, height, width or 3, height, width."
            )
        else:
            image = {"channels": c, "height": h, "width": w, "layout": "chw"}
    elif rank == 2:
        h, w = in_shape
        image = {"channels": 1, "height": h, "width": w, "layout": "hw"}
        notes.append(f"Each image is fed as {h} rows of {w} grayscale pixels.")
    elif rank == 1 and builtin is not None:
        native = math.prod(builtin.shape)
        if in_shape[0] != native:
            result.errors.append(
                f"The Input expects {in_shape[0]:,} values, but {name} images have {native:,} "
                f"({_x(builtin.shape)}). Set the Input shape to {native} or to "
                f"{', '.join(str(s) for s in builtin.shape)}."
            )
        else:
            c, h, w = builtin.shape
            image = {"channels": c, "height": h, "width": w, "layout": "flat"}
            notes.append(f"Each image is flattened to {native:,} values.")
    elif rank == 1 and (square := _square_image(in_shape[0])) is not None:
        # A flat Input (e.g. followed by a Linear layer): square RGB or grayscale images.
        c, side = square
        image = {"channels": c, "height": side, "width": side, "layout": "flat"}
        notes.append(f"Each image is flattened to {in_shape[0]:,} values ({_x([c, side, side])}).")
    elif rank == 1:
        result.errors.append(
            "For image data, set the Input shape to channels, height, width, "
            "e.g. 3, 64, 64 for RGB images or 1, 28, 28 for grayscale."
        )
    else:
        result.errors.append(
            f"Images cannot be fed into an Input of shape {fmt_shape(tuple(in_shape))}. "
            "Use channels, height, width, e.g. 3, 32, 32."
        )

    if image is not None and max(image["height"], image["width"]) > MAX_IMAGE_SIDE:
        result.errors.append(f"Images can be at most {MAX_IMAGE_SIDE}×{MAX_IMAGE_SIDE} pixels.")
        image = None

    num_classes = info.num_classes or 0
    k = _check_classes(result, out_shape, num_classes, name)
    if result.errors or image is None or k is None:
        return

    size = f"{image['height']}×{image['width']}"
    if builtin is not None:
        nc, nh, nw = builtin.shape
        if (nh, nw) != (image["height"], image["width"]):
            result.warnings.append(f"{name} images ({nh}×{nw}) are resized to {size}.")
        if nc == 3 and image["channels"] == 1:
            notes.append(f"{name} images are converted to grayscale.")
        elif nc == 1 and image["channels"] == 3:
            notes.append("The grayscale images are copied into 3 channels.")
        sizes = f"{builtin.train_size:,} training and {builtin.test_size:,} test images"
        notes.append(
            f"Validation data is taken from the {builtin.train_size:,} training images "
            f"(validation split). The {builtin.test_size:,} official test images are the test "
            "set: the kept weights are measured on them once, after training."
        )
        notes.append(f"{name} is downloaded on first use (about {builtin.download_mb} MB).")
    else:
        mode = "RGB" if image["channels"] == 3 else "grayscale"
        notes.append(f"Images are converted to {mode} and resized to {size}.")
        sizes = _count(info.num_samples, "image")

    class_names = list(info.class_names or [])
    if k > len(class_names):
        class_names += [f"class {i}" for i in range(len(class_names), k)]
    result.plan = TrainingPlan(
        task="classification",
        summary=f"Classification into {num_classes} classes ({name}, {sizes}).",
        inputs=inputs,
        targets=[{"kind": "class", "num_classes": k}],
        image=image,
        num_classes=k,
        class_names=class_names,
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #


def _memory(size: float) -> str:
    return f"{size / 1024**3:.1f} GB" if size >= 1024**3 else f"{size / 1024**2:.0f} MB"


def segmentation_memory(analysis: Analysis) -> tuple[int, int]:
    """``(fixed_bytes, sample_bytes)``: the estimated memory of a training step is
    ``fixed_bytes + batch size × sample_bytes`` (rough, for CPU training)."""
    values = sum(math.prod(info.shape) for info in analysis.nodes.values() if info.shape)
    return int(_BYTES_PER_PARAM * analysis.total_params), int(_BYTES_PER_VALUE * values)


def segmentation_memory_error(
    segmentation: dict[str, Any], image: dict[str, Any], batch_size: int
) -> str | None:
    """Why a batch of ``batch_size`` images does not fit into memory (None if it does)."""
    fixed = int(segmentation.get("fixed_bytes") or 0)
    sample = int(segmentation.get("sample_bytes") or 0)
    limit = segmentation_memory_bytes()
    needed = fixed + batch_size * sample
    if needed <= limit or sample <= 0:
        return None
    fits = (limit - fixed) // sample
    size = f"{image['height']}×{image['width']}"
    text = (
        f"A batch of {batch_size} images of {size} pixels needs about {_memory(needed)} of "
        f"memory in this network, but a training may use {_memory(limit)} on this server. "
    )
    if fits >= 1:
        return text + f"Set the batch size to {fits} or less, or make the Input smaller."
    return text + "Make the Input smaller or use fewer channels in the layers."


def _shrink_hint(analysis: Analysis) -> str:
    """What usually makes the output of a segmentation network smaller than its input."""
    valid = [
        info.id
        for info in analysis.nodes.values()
        if info.type == "Conv2d"
        and info.layer is not None
        and info.layer.p.get("padding") == "valid"
        and max(info.layer.p.get("kernel_size") or (1,)) > 1
    ]
    pools = sum(1 for info in analysis.nodes.values() if "Pool" in info.type)
    hints = []
    if valid:
        hints.append(
            f"convolutions with 'valid' padding cut off the border (e.g. '{valid[0]}': use "
            "'same' padding)"
        )
    if pools:
        hints.append(
            "every pooling layer needs an upsampling layer (ConvTranspose2d) that undoes it, "
            f"and the Input height and width must be divisible by {2**pools} "
            f"({pools} pooling layer{'s' if pools != 1 else ''})"
        )
    if not hints:
        hints.append("use 'same' padding and stride 1, or upsample again")
    return "Check: " + "; ".join(hints) + "."


def _check_segmentation(
    result: Compatibility,
    analysis: Analysis,
    info: DatasetInfo,
    inputs: list[dict[str, Any]],
    in_shape: list[int],
    out_shape: list[int],
    config: TrainingConfig | None,
) -> None:
    builtin = BUILTIN_DATASETS.get(info.builtin or "") if info.kind == "builtin" else None
    if info.kind == "builtin" and builtin is None:
        result.errors.append(f"Unknown built-in dataset '{info.builtin}'.")
        return
    name = info.name
    class_names = list(info.class_names or [])
    n = len(class_names)
    native = list(info.sample_shape or [3, 64, 64])
    example = ", ".join(str(v) for v in (native if builtin is not None else [native[0], 64, 64]))
    image: dict[str, Any] | None = None
    if len(in_shape) != 3:
        result.errors.append(
            f"{name} is a segmentation dataset: the Input takes whole images. Set the Input "
            f"shape to channels, height, width, e.g. {example}."
        )
    elif in_shape[0] not in (1, 3):
        result.errors.append(
            f"The Input has {in_shape[0]} channels, but images have 1 (grayscale) or 3 (RGB). "
            "Set the Input shape to 1, height, width or 3, height, width."
        )
    elif max(in_shape[1:]) > MAX_SEGMENTATION_SIDE:
        limit = MAX_SEGMENTATION_SIDE
        result.errors.append(
            f"Segmentation trains on images of at most {limit}×{limit} pixels, but the Input "
            f"is {in_shape[1]}×{in_shape[2]}. Set a smaller Input size: the images and masks "
            "are resized to it."
        )
    else:
        c, h, w = in_shape
        image = {"channels": c, "height": h, "width": w, "layout": "chw"}

    channels_hint = (
        f"The last layer should output {n} channels (e.g. Conv2d with {n} output channels "
        "and kernel size 1)."
    )
    k: int | None = None
    if len(out_shape) != 3:
        per_image = " (one answer per image)" if len(out_shape) == 1 else ""
        result.errors.append(
            f"{name} is a segmentation dataset: every pixel has a class, so the network must "
            f"output one score per class for every pixel, e.g. shape ({n}, height, width). "
            f"This network outputs shape {fmt_shape(tuple(out_shape))}{per_image}. Use a "
            "network that ends with a convolution and keeps the image size, e.g. the U-Net "
            "example."
        )
    else:
        k = out_shape[0]
        if image is not None and out_shape[1:] != in_shape[1:]:
            result.errors.append(
                f"The network outputs {out_shape[1]}×{out_shape[2]} values for an image of "
                f"{in_shape[1]}×{in_shape[2]} pixels, but segmentation needs one output per "
                f"pixel: the output must be as high and wide as the Input. "
                f"{_shrink_hint(analysis)}"
            )
        if k == 1 and n != 2:
            result.errors.append(
                f"The network outputs 1 value per pixel, which fits two classes, but {name} "
                f"has {n} classes. {channels_hint}"
            )
        elif 1 < k < n:
            result.errors.append(
                f"The network outputs {k} channels, but {name} has {n} classes. {channels_hint}"
            )
        elif k > n:
            result.warnings.append(
                f"The network outputs {k} channels, but {name} has only {n} classes. The extra "
                "channels are never the right answer."
            )
    if result.errors or image is None or k is None:
        return

    last = analysis.nodes[analysis.outputs[0]]
    if (
        last.type == "Activation"
        and last.layer is not None
        and last.layer.p.get("function") in ("sigmoid", "softmax", "tanh")
    ):
        result.warnings.append(
            f"The network ends with a {last.layer.p['function']} activation, but the loss "
            "expects raw scores (logits) and applies the sigmoid or softmax itself. Remove "
            "the last activation."
        )

    details = info.segmentation or {}
    binary = k == 1
    size = f"{image['height']}×{image['width']}"
    mode = "RGB" if image["channels"] == 3 else "grayscale"
    notes = []
    if builtin is not None:
        nc, nh, nw = builtin.shape
        if (nh, nw) != (image["height"], image["width"]):
            result.warnings.append(
                f"{name} images ({nh}×{nw}) are resized to {size}; masks are resized with "
                "nearest neighbour."
            )
        if nc == 3 and image["channels"] == 1:
            notes.append(f"{name} images are converted to grayscale.")
        notes.append(
            f"{name} is generated on the server, nothing is downloaded. Validation data is "
            f"taken from its {builtin.train_size:,} training images (validation split); the "
            f"{builtin.test_size:,} test images are the test set: the kept weights are "
            "measured on them once, after training."
        )
        sizes = f"{builtin.train_size:,} training and {builtin.test_size:,} test images"
    else:
        notes.append(
            f"Images are converted to {mode} and resized to {size}; masks are resized with "
            "nearest neighbour."
        )
        sizes = _count(info.num_samples, "image") + " with masks"
        if info.predefined_split is not None:
            train, val, test = [*info.predefined_split, 0, 0, 0][:3]
            parts = (
                f"{train:,} training, {val:,} validation and {test:,} test images"
                if test
                else f"{train:,} training and {val:,} validation images, no test set"
            )
            result.warnings.append(
                f"Uses the split of the folders ({parts}); the validation and test split "
                "settings are ignored."
            )
        from .segmentation import describe

        result.warnings.extend(describe(info))
    if binary:
        notes.append(
            f"One output channel: its score says how likely a pixel is '{class_names[1]}' "
            "(binary cross-entropy on logits)."
        )
    ignore = details.get("ignore_index")
    if ignore is not None:
        notes.append(
            f"Pixels with the mask value {ignore} are ignored: they count neither for the "
            "loss nor for the metrics."
        )
    notes.append(
        "After every epoch the validation images are measured: mean IoU and Dice (the mean "
        "over the classes that occur) and pixel accuracy. The kept epoch is the one with the "
        "lowest validation loss."
    )
    shares = [s for s in details.get("class_pixel_share") or [] if isinstance(s, (int, float))]
    loss = config.loss if config is not None else "cross_entropy"
    if shares and loss == "cross_entropy" and max(shares) > 0.97:
        result.warnings.append(
            f"{max(shares):.0%} of the pixels are '{class_names[shares.index(max(shares))]}'. "
            "The loss 'Cross-entropy + Dice' helps the other classes to be learned."
        )

    classes = n if binary else k
    class_names += [f"class {i}" for i in range(len(class_names), classes)]
    fixed, sample = segmentation_memory(analysis)
    segmentation = {
        "binary": binary,
        "ignore_index": ignore,
        "fixed_bytes": fixed,
        "sample_bytes": sample,
    }
    if config is not None:
        error = segmentation_memory_error(segmentation, image, config.batch_size)
        if error is not None:
            result.errors.append(error)
            return
    result.plan = TrainingPlan(
        task="segmentation",
        summary=f"Segmentation into {n} classes at {size} pixels ({name}, {sizes}).",
        inputs=inputs,
        targets=[{"kind": "mask", "num_classes": classes, "binary": binary}],
        image=image,
        num_classes=classes,
        class_names=class_names,
        notes=notes,
        segmentation=segmentation,
    )


# --------------------------------------------------------------------------- #
# Suggested fixes
# --------------------------------------------------------------------------- #


def suggest_fixes(
    graph: Any,
    dataset: DatasetInfo,
    analysis: Analysis | None = None,
    result: Compatibility | None = None,
) -> list[dict[str, Any]]:
    """Changes to the Input layer and the last layer that make ``graph`` fit ``dataset``.

    Returns ``[{"node", "type", "params", "previous", "summary"}, ...]``: ``params`` are the
    parameters to set on the layer ``node`` (``previous`` their current values), e.g.
    ``{"node": "in", "type": "Input", "params": {"shape": [4]}, "previous": {"shape": [784]},
    "summary": "Input shape 784 → 4"}``. Only two layers are ever changed:

    - the Input layer: ``shape`` (and ``dtype`` "float") as the data is fed. Shapes that work
      are kept; otherwise the dataset's own shape, flat (``[C·H·W]``) when the Input already
      is flat, e.g. because a Linear layer follows it;
    - the last Linear layer (also behind activations or dropout): ``out_features`` = one per
      class for classification, 1 for regression. Surplus class outputs are corrected too,
      also when the network fits otherwise (then only a warning explains them);
    - for segmentation datasets instead the last Conv2d or ConvTranspose2d layer:
      ``out_channels`` = one per class (one channel is kept for two classes).

    Every suggestion is checked: applied together, the changes make the network fit the data.
    When there is no such change (e.g. a CNN on CSV data, or a Linear layer with a fixed number
    of input features after the Input), nothing is suggested and the errors explain the rest.
    ``analysis`` and ``result`` (of ``graph``) save work when they are known already.
    """
    if dataset.kind not in ("builtin", "csv", "images", "segmentation"):
        return []
    if isinstance(graph, dict) and isinstance(graph.get("graph"), dict):
        graph = graph["graph"]  # a saved project file
    if not isinstance(graph, dict) or not isinstance(graph.get("nodes"), list):
        return []
    analysis = analysis or analyze(graph)
    if not analysis.valid or len(analysis.inputs) != 1 or len(analysis.outputs) != 1:
        return []
    classical = analysis.family == "classical"
    if is_segmentation(dataset) and not classical:
        return _segmentation_fixes(graph, dataset, analysis, result)
    if result is not None and result.ok and (classical or _outputs_fit(analysis, dataset)):
        return []  # nothing to improve

    input_id = analysis.inputs[0]
    current = list(analysis.nodes[input_id].shape or ())
    dtype = analysis.nodes[input_id].dtype
    for shape in _input_candidates(dataset, current):
        changes: dict[str, dict[str, Any]] = {}
        params: dict[str, Any] = {}
        if shape != current:
            params["shape"] = shape
        if dtype != "float":
            params["dtype"] = "float"
        if params:
            changes[input_id] = params
        trial = analyze(_with_params(graph, changes))
        if not trial.valid or len(trial.outputs) != 1:
            continue
        if classical:  # a classical model's output follows the data: only the Input matters
            if _check(trial, dataset).ok:
                return [_describe_fix(analysis, nid, params) for nid, params in changes.items()]
            continue
        out_shape = list(trial.nodes[trial.outputs[0]].shape or ())
        if len(out_shape) != 1:
            continue
        wanted = _wanted_outputs(dataset, out_shape[0])
        if wanted != out_shape[0]:
            last = _last_linear(trial)
            if last is None:
                continue
            changes[last] = {"out_features": wanted}
            trial = analyze(_with_params(graph, changes))
        if not trial.valid or not _check(trial, dataset).ok:
            continue
        return [_describe_fix(analysis, nid, params) for nid, params in changes.items()]
    return []


def _segmentation_fixes(
    graph: dict[str, Any],
    dataset: DatasetInfo,
    analysis: Analysis,
    result: Compatibility | None,
) -> list[dict[str, Any]]:
    """Fixes for a segmentation dataset: the Input shape (channels, height, width; a size
    the network keeps from Input to output) and the output channels of the last
    convolution (one per class; one channel is kept for two classes)."""
    classes = dataset.num_classes or 0
    input_id = analysis.inputs[0]
    current = list(analysis.nodes[input_id].shape or ())
    dtype = analysis.nodes[input_id].dtype

    def wanted(channels: int) -> int:
        return 1 if channels == 1 and classes == 2 else classes

    out_shape = list(analysis.nodes[analysis.outputs[0]].shape or ())
    fits = len(out_shape) == 3 and wanted(out_shape[0]) == out_shape[0]
    if fits and ((result is not None and result.ok) or _check(analysis, dataset).ok):
        return []  # nothing to improve (a batch that is too large is not for the layers)
    for shape in _segmentation_candidates(dataset, current):
        changes: dict[str, dict[str, Any]] = {}
        params: dict[str, Any] = {}
        if shape != current:
            params["shape"] = shape
        if dtype != "float":
            params["dtype"] = "float"
        if params:
            changes[input_id] = params
        trial = analyze(_with_params(graph, changes))
        if not trial.valid or len(trial.outputs) != 1:
            continue
        out_shape = list(trial.nodes[trial.outputs[0]].shape or ())
        if len(out_shape) != 3:
            continue
        if wanted(out_shape[0]) != out_shape[0]:
            last = _last_layer(trial, _OUTPUT_CONVS)
            if last is None:
                continue
            changes[last] = {"out_channels": wanted(out_shape[0])}
            trial = analyze(_with_params(graph, changes))
        if not trial.valid or not _check(trial, dataset).ok:
            continue
        return [_describe_fix(analysis, nid, params) for nid, params in changes.items()]
    return []


def _segmentation_candidates(info: DatasetInfo, current: list[int]) -> list[list[int]]:
    """Input shapes to try for a segmentation dataset, best first: the current one, the
    current size with the channels of the images, then the size of the images (folders:
    scaled down to SEGMENTATION_SIDES and to multiples of 32, which pooling layers divide)."""
    native = list(info.sample_shape or [3, 64, 64])
    if len(native) != 3 or native[0] not in (1, 3):
        native = [3, 64, 64]
    channels = native[0]
    candidates = [current]
    if len(current) == 3 and max(current[1:]) <= MAX_SEGMENTATION_SIDE:
        keep = current[0] if current[0] in (1, 3) else channels
        candidates += [[keep, *current[1:]], [channels, *current[1:]]]
    sizes = []
    if info.kind == "builtin":
        sizes.append(native[1:])
    longest = max(native[1:])
    for side in SEGMENTATION_SIDES:
        scale = min(1.0, side / longest)
        sizes.append([max(32, int(s * scale) // 32 * 32) for s in native[1:]])
    sizes += [[side, side] for side in SEGMENTATION_SIDES]
    candidates += [[channels, *size] for size in sizes]
    unique: list[list[int]] = []
    for shape in candidates:
        if shape and shape not in unique and all(s > 0 for s in shape):
            unique.append(shape)
    return unique


def _outputs_fit(analysis: Analysis, dataset: DatasetInfo) -> bool:
    shape = list(analysis.nodes[analysis.outputs[0]].shape or ())
    return len(shape) == 1 and _wanted_outputs(dataset, shape[0]) == shape[0]


def _wanted_outputs(info: DatasetInfo, current: int) -> int:
    """How many values the network should output for ``info`` (it outputs ``current`` now)."""
    if info.kind in ("builtin", "images"):
        return info.num_classes or current
    if info.predefined_split is not None:
        # Made in the Data tab: class names are only stored for classification.
        return len(info.class_names) if info.class_names else 1
    if info.target_kind == "text":
        return info.num_classes or current
    if info.target_kind == "integer" and current > 1:
        # Trained as classification by class index (with 1 output: as a regression).
        low, high = info.target_range or [0, 0]
        classes = math.floor(high) + 1
        return classes if low >= 0 and classes <= MAX_CLASSES else 1
    return 1


def _input_candidates(info: DatasetInfo, current: list[int]) -> list[list[int]]:
    """Input shapes to try, best first; the current one comes first (kept if it works)."""
    candidates = [current]
    if info.kind == "csv":
        candidates.append([info.num_features or 0])
    elif info.kind == "builtin":
        builtin = BUILTIN_DATASETS.get(info.builtin or "")
        if builtin is None:
            return candidates
        native = list(builtin.shape)
        flat = [math.prod(native)]
        if len(current) == 3 and max(current[1:]) <= MAX_IMAGE_SIDE:
            candidates.append([native[0], *current[1:]])  # keep the size, fix the channels
        if len(current) == 2:
            candidates.append(native[1:])
        candidates += [flat, native] if len(current) == 1 else [native, flat]
    else:
        # Image folders: the images (of any size) are resized to the Input.
        flat = [math.prod(DEFAULT_IMAGE_SHAPE)]
        if len(current) in (2, 3):
            size = current[-2:]
            if max(size) > MAX_IMAGE_SIDE:
                size = [max(1, s * MAX_IMAGE_SIDE // max(size)) for s in size]
            if len(current) == 2:
                candidates.append(size)
            else:
                channels = current[0] if current[0] in (1, 3) else 3
                candidates += [[channels, *size], [1, *size]]
        candidates += (
            [flat, DEFAULT_IMAGE_SHAPE] if len(current) == 1 else [DEFAULT_IMAGE_SHAPE, flat]
        )
    unique: list[list[int]] = []
    for shape in candidates:
        if shape and shape not in unique and all(s > 0 for s in shape):
            unique.append(shape)
    return unique


def _last_linear(analysis: Analysis) -> str | None:
    """The Linear layer that decides how many values the (single) output has."""
    return _last_layer(analysis, ("Linear",))


def _last_layer(analysis: Analysis, types: tuple[str, ...]) -> str | None:
    """The layer of one of ``types`` that decides the size of the (single) output: the
    output itself, or the layer before the activations and dropout that end the network."""
    nid = analysis.outputs[0]
    while analysis.nodes[nid].type not in types:
        info = analysis.nodes[nid]
        if info.type not in _PASS_THROUGH or len(info.inputs) != 1:
            return None
        nid = info.inputs[0]
        if len(analysis.nodes[nid].outputs) != 1:
            return None  # another branch uses it as well
    return nid


def _with_params(graph: dict[str, Any], changes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """A copy of the graph document with ``changes`` merged into the layers' params."""
    nodes = []
    for raw in graph["nodes"]:
        node_id = str(raw.get("id")) if isinstance(raw, dict) else None
        if node_id is not None and node_id in changes:
            params = raw.get("params")
            merged = {**(params if isinstance(params, dict) else {}), **changes[node_id]}
            raw = {**raw, "params": merged}
        nodes.append(raw)
    return {**graph, "nodes": nodes}


def _describe_fix(analysis: Analysis, nid: str, params: dict[str, Any]) -> dict[str, Any]:
    info = analysis.nodes[nid]
    assert info.layer is not None
    previous = {
        k: list(v) if isinstance(v, tuple) else v for k, v in info.layer.p.items() if k in params
    }
    if info.type == "Input":
        parts = []
        if "shape" in params:
            parts.append(f"Input shape {_x(previous['shape'])} → {_x(params['shape'])}")
        if "dtype" in params:
            parts.append(f"Input data type {previous['dtype']} → {params['dtype']}")
        summary = ", ".join(parts)
    elif "out_channels" in params:
        summary = (
            f"Output channels of the last layer {previous['out_channels']} → "
            f"{params['out_channels']}"
        )
    else:
        summary = f"Outputs of the last layer {previous['out_features']} → {params['out_features']}"
    return {
        "node": nid,
        "type": info.type,
        "params": params,
        "previous": previous,
        "summary": summary,
    }


# --------------------------------------------------------------------------- #
# Classical models (scikit-learn)
# --------------------------------------------------------------------------- #

# Demo data of classical classifiers has this many classes.
CLASSICAL_DEMO_CLASSES = 3
# How the row limits are described: (models, variable, cost)
LIMITS = {
    "gaussian_process": ("Gaussian processes", "BRICKDNN_GP_MAX_ROWS", "n³"),
    "gaussian_process_classifier": (
        "Gaussian process classifiers",
        "BRICKDNN_GPC_MAX_ROWS",
        "n³ for every class",
    ),
    "svm": ("Support vector machines", "BRICKDNN_SVM_MAX_ROWS", "n² or more"),
}
CLASSICAL_NOTE = (
    "A classical model is fitted once, without epochs: the settings for epochs, batch size, "
    "learning rate, optimizer, weight decay, device, early stopping and the kept weights are "
    "not used."
)


def _check_classical(
    result: Compatibility,
    analysis: Analysis,
    info: DatasetInfo,
    inputs: list[dict[str, Any]],
    config: TrainingConfig | None,
) -> None:
    node = analysis.model
    assert node is not None and isinstance(node.layer, ClassicalModel)
    layer = node.layer
    max_rows = classical_max_rows(layer.limit)
    models, variable, cost = LIMITS[layer.limit]
    classical: dict[str, Any] = {
        "node": node.id,
        "type": node.type,
        "estimator": layer.estimator,
        "label": layer.label,
        "task": layer.task,
        "kernel": node.kernel.code() if node.kernel is not None else None,
        "scale": layer.scale,
        "max_rows": max_rows,
    }
    max_samples = config.max_samples if config is not None else None
    if max_samples is not None and max_samples > max_rows:
        result.errors.append(
            f"Max. samples is {max_samples:,}, but {models.lower()} are fitted on at most "
            f"{max_rows:,} training rows on this server ({variable}). Lower it, or leave it "
            f"empty to fit on {max_rows:,} random training rows."
        )
    if info.kind == "random":
        result.plan = _classical_demo_plan(layer, inputs, classical)
        return
    if info.kind in ("builtin", "images", "segmentation"):
        result.errors.append(
            f"{layer.label} is a classical model: it trains on tabular data (a CSV file or a "
            f"dataset from the Data tab) or on demo data, not on images like {info.name}. "
            "Build a neural network (e.g. a CNN) for images."
        )
        return
    if info.kind != "csv":
        result.errors.append(f"Unknown dataset kind '{info.kind}'.")
        return

    n = info.num_features or 0
    in_shape = inputs[0]["shape"]
    if in_shape != [n]:
        cols = ", ".join(info.columns[:-1][:8]) + (" …" if n > 8 else "")
        result.errors.append(
            f"The Input expects shape {fmt_shape(tuple(in_shape))}, but the CSV file has "
            f"{_count(n, 'feature column')} ({cols}). Set the Input shape to {n}."
        )
        return
    target = info.columns[-1] if info.columns else "target"
    rows = _count(info.num_samples, "row")
    prepared = info.predefined_split is not None
    if prepared:
        result.warnings.append(_predefined_split_note(info.predefined_split or []))
    notes = [
        "Features are standardised (mean 0, standard deviation 1) with the statistics of the "
        "training rows; the scaler is saved as part of the model.",
        CLASSICAL_NOTE,
    ]

    if layer.task == "regression":
        if info.target_kind == "text":
            result.errors.append(
                f"The target column '{target}' contains text labels ({info.num_classes or 0} "
                f"classes), so this is a classification task: use GaussianProcessClassifier or "
                f"SVC instead of {layer.label}."
            )
            return
        if info.target_kind == "integer" and info.class_names and len(info.class_names) <= 10:
            result.warnings.append(
                f"The target column '{target}' only holds {len(info.class_names)} different "
                f"values or classes ({_shown(info.class_names)}). {layer.label} predicts them "
                "as numbers; for classification use GaussianProcessClassifier or SVC instead."
            )
        plan = TrainingPlan(
            task="regression",
            summary=f"Regression: predict '{target}' from {_count(n, 'feature')} ({rows}) "
            f"with {layer.estimator}.",
            inputs=inputs,
            targets=[{"kind": "value", "shape": [1]}],
            notes=notes,
            classical=classical,
        )
    else:
        if info.target_kind == "number":
            result.errors.append(
                f"The target column '{target}' contains decimal numbers, which is a regression "
                f"task: use GaussianProcessRegressor or SVR instead of {layer.label}."
            )
            return
        if not info.class_names:
            result.errors.append(
                f"This dataset was prepared for regression: its target '{target}' has no "
                "classes. Prepare it for classification in the Data tab, or use "
                "GaussianProcessRegressor or SVR."
                if prepared
                else f"The target column '{target}' has more than {MAX_CLASSES:,} different "
                "values: it cannot be used for classification."
            )
            return
        class_names = list(info.class_names)
        if len(class_names) < 2:
            result.errors.append(
                f"The target column '{target}' holds only one class ('{class_names[0]}'). "
                "Classification needs at least two."
            )
            return
        # How the target column names the classes: text labels, whole numbers, or (prepared
        # in the Data tab) class indices 0 … K-1 with the labels in class_names.
        if info.target_kind == "text":
            classical["class_targets"] = "labels"
            note = f"Text labels are classes in alphabetical order: {_shown(class_names)}."
        elif prepared:
            classical["class_targets"] = "indices"
            note = f"Classes in index order: {_shown(class_names)}."
        else:
            classical["class_targets"] = "values"
            note = f"The classes are the values of the target column: {_shown(class_names)}."
        k = len(class_names)
        plan = TrainingPlan(
            task="classification",
            summary=f"Classification into {k} classes from {_count(n, 'feature')} ({rows}) "
            f"with {layer.estimator}.",
            inputs=inputs,
            targets=[{"kind": "class", "num_classes": k}],
            num_classes=k,
            class_names=class_names,
            notes=[*notes, note],
            classical=classical,
        )

    n_train = _training_rows(info, config)
    if n_train is not None and n_train > max_rows:
        result.warnings.append(
            f"{models} are fitted on at most {max_rows:,} training rows on this server "
            f"({variable}; the time to fit grows with {cost}): {max_rows:,} random rows of the "
            f"{n_train:,} training rows are used, and validation and test data shrink in "
            "proportion."
        )
    result.plan = plan


def _training_rows(info: DatasetInfo, config: TrainingConfig | None) -> int | None:
    """Training rows of a CSV dataset with these settings (the defaults without)."""
    config = config or TrainingConfig()
    if info.predefined_split:
        n_train = int(info.predefined_split[0])
    else:
        try:
            n_train = split_sizes(info.num_samples or 0, config.val_split, config.test_split)[0]
        except TrainingError:
            return None
    if config.max_samples is not None:
        n_train = min(n_train, config.max_samples)
    return n_train


def _classical_demo_plan(
    layer: ClassicalModel, inputs: list[dict[str, Any]], classical: dict[str, Any]
) -> TrainingPlan:
    notes = [
        "Demo data checks whether the model can learn at all: random features and a hidden "
        "rule. The numbers say nothing about real-world accuracy.",
        CLASSICAL_NOTE,
    ]
    if layer.task == "regression":
        return TrainingPlan(
            task="demo",
            summary="Demo data: random features and a hidden rule to learn (regression).",
            inputs=inputs,
            targets=[{"kind": "value", "shape": [1]}],
            notes=notes,
            classical=classical,
        )
    k = CLASSICAL_DEMO_CLASSES
    return TrainingPlan(
        task="demo",
        summary=f"Demo data: random features and a hidden rule to learn ({k} classes).",
        inputs=inputs,
        targets=[{"kind": "class", "num_classes": k}],
        num_classes=k,
        class_names=[f"class {i}" for i in range(k)],
        notes=notes,
        classical={**classical, "class_targets": "indices"},
    )
