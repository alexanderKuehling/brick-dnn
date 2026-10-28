"""Check whether a network can be trained on a dataset (no PyTorch needed).

:func:`check_compatibility` is used twice: by the API before a run starts (so the UI can
show what is wrong and how to fix it) and by the training engine, which follows the
resulting :class:`TrainingPlan` when it prepares the data.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from ..graph import Analysis, analyze
from ..layers import fmt_shape
from .datasets import BUILTIN_DATASETS, DatasetInfo

MAX_IMAGE_SIDE = 1024


@dataclass
class TrainingPlan:
    """How the data is fed to the network and which loss is used.

    ``inputs``  one entry per Input layer: ``{"shape": [...], "dtype": "float"|"long",
                "vocab": int|None}`` (``vocab`` = smallest Embedding vocabulary fed by it)
    ``targets`` one entry per output: ``{"kind": "class", "num_classes": K}`` (cross-entropy)
                or ``{"kind": "value", "shape": [...]}`` (mean squared error)
    ``image``   for image data: ``{"channels": C, "height": H, "width": W,
                "layout": "chw"|"hw"|"flat"}``. Images are converted to C channels and
                resized to H×W; ``hw`` drops the channel axis, ``flat`` flattens C·H·W.
    """

    task: str  # classification | regression | demo
    summary: str
    inputs: list[dict[str, Any]]
    targets: list[dict[str, Any]]
    image: dict[str, Any] | None = None
    num_classes: int | None = None
    class_names: list[str] | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingPlan:
        return cls(**data)


@dataclass
class Compatibility:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    plan: TrainingPlan | None = None

    @property
    def ok(self) -> bool:
        return not self.errors and self.plan is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "plan": self.plan.to_dict() if self.plan and self.ok else None,
        }


def _x(shape: list[int] | tuple[int, ...]) -> str:
    return "×".join(str(s) for s in shape)


def _count(n: int | None, word: str) -> str:
    if n is None:
        return word + "s"
    return f"{n:,} {word}" + ("" if n == 1 else "s")


def check_compatibility(graph: Any, dataset: DatasetInfo) -> Compatibility:
    """Check ``graph`` (a graph document or an :class:`Analysis`) against ``dataset``."""
    analysis = graph if isinstance(graph, Analysis) else analyze(graph)
    result = Compatibility()
    if not analysis.valid:
        result.errors.append("The network has problems. Fix the layers marked in red first.")
        return result

    inputs = _input_specs(analysis)
    if dataset.kind == "random":
        result.plan = _demo_plan(analysis, inputs)
        return result
    if dataset.kind not in ("builtin", "csv", "images"):
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
    if dataset.kind == "csv":
        _check_csv(result, dataset, inputs, in_shape, out_shape)
    else:
        _check_images(result, dataset, inputs, in_shape, out_shape)
    if result.errors:
        result.plan = None
    return result


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
        result.errors.append(
            f"The network outputs shape {fmt_shape(tuple(out_shape))}, but {what} needs one "
            f"score per class. {hint}"
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
        result.errors.append(
            f"The network outputs shape {fmt_shape(tuple(out_shape))}. For CSV data it must "
            "output 1 value (regression) or one value per class (classification)."
        )
    if result.errors:
        return

    k = out_shape[0]
    target = info.columns[-1] if info.columns else "target"
    rows = _count(info.num_samples, "row")
    if k == 1:
        if info.target_kind == "text":
            classes = info.num_classes or 0
            result.errors.append(
                f"The target column '{target}' contains text labels ({classes} classes), so this "
                f"is a classification task: the last layer needs {classes} outputs, one per class."
            )
            return
        if info.target_kind == "integer" and info.class_names and len(info.class_names) <= 10:
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
        sizes = f"{builtin.train_size:,} training and {builtin.val_size:,} validation images"
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
