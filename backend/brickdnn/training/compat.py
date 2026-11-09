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
from ..layers import fmt_shape
from .datasets import BUILTIN_DATASETS, MAX_CLASSES, DatasetInfo

MAX_IMAGE_SIDE = 1024
# A flat Input takes images of an image folder only if they are at least this many pixels wide
# (otherwise e.g. 4 features would count as 2×2 grayscale images).
MIN_FLAT_SIDE = 8
# Input shape suggested for image folders when nothing better is known (images are resized).
DEFAULT_IMAGE_SHAPE = [3, 64, 64]
# Layers between the last Linear layer and the output that keep the number of features.
_PASS_THROUGH = ("Activation", "Dropout")


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


def check_compatibility(graph: Any, dataset: DatasetInfo) -> Compatibility:
    """Check ``graph`` (a graph document or an :class:`Analysis`) against ``dataset``.

    For a graph document, ``fixes`` are suggested as well (an analysis cannot be changed).
    """
    if isinstance(graph, Analysis):
        return _check(graph, dataset)
    analysis = analyze(graph)
    result = _check(analysis, dataset)
    result.fixes = suggest_fixes(graph, dataset, analysis, result)
    return result


def _check(analysis: Analysis, dataset: DatasetInfo) -> Compatibility:
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
        result.errors.append(
            f"The network outputs shape {fmt_shape(tuple(out_shape))}. For CSV data it must "
            "output 1 value (regression) or one value per class (classification)."
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
      also when the network fits otherwise (then only a warning explains them).

    Every suggestion is checked: applied together, the changes make the network fit the data.
    When there is no such change (e.g. a CNN on CSV data, or a Linear layer with a fixed number
    of input features after the Input), nothing is suggested and the errors explain the rest.
    ``analysis`` and ``result`` (of ``graph``) save work when they are known already.
    """
    if dataset.kind not in ("builtin", "csv", "images"):
        return []
    if isinstance(graph, dict) and isinstance(graph.get("graph"), dict):
        graph = graph["graph"]  # a saved project file
    if not isinstance(graph, dict) or not isinstance(graph.get("nodes"), list):
        return []
    analysis = analysis or analyze(graph)
    if not analysis.valid or len(analysis.inputs) != 1 or len(analysis.outputs) != 1:
        return []
    if result is not None and result.ok and _outputs_fit(analysis, dataset):
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
    nid = analysis.outputs[0]
    while analysis.nodes[nid].type != "Linear":
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
    else:
        summary = f"Outputs of the last layer {previous['out_features']} → {params['out_features']}"
    return {
        "node": nid,
        "type": info.type,
        "params": params,
        "previous": previous,
        "summary": summary,
    }
