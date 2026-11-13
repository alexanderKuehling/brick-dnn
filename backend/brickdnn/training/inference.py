"""Using a trained network (no PyTorch needed): what it takes as input, how to prepare new
inputs the way the training data was prepared, and how to read its outputs. The network
itself runs in a separate process (:mod:`.predict`), like a training.

Everything here works on two documents of a run: the training plan (``TrainingRun.plan``,
see compat.py) and the ``preprocessing`` that the training wrote into ``state.json``
(see data.py)::

    {"kind": "tabular", "columns": [...], "target": "price", "mean": [...], "std": [...],
     "target_mean": 3.2, "target_std": 1.1, "class_names": null, ...}
    {"kind": "image", "channels": 1, "height": 28, "width": 28, "layout": "chw",
     "mean": [0.1307], "std": [0.3081], "class_names": ["0", ...], ...}
    {"kind": "demo"}            # random demo data: inputs are used as they are

A segmentation (``plan["task"]`` "segmentation") takes images like any image network and
answers with a class for every pixel instead of one for the image (see
:func:`read_masks`). Image networks also take image files (PNG, JPEG …) as base64 text
instead of arrays of numbers (:func:`prepare_image_files`): the prediction process decodes
them and prepares them like the training images.

Classical models (scikit-learn, ``plan["classical"]`` is set) predict in the units of the
target column: their tabular preprocessing has no ``target_mean``/``target_std``, and
classifiers report probabilities instead of raw scores (see :func:`read_outputs`).
"""

from __future__ import annotations

import keyword
import math
import re
from dataclasses import dataclass
from typing import Any

from .datasets import parse_number

KINDS = ("tabular", "image", "demo")
# Limits of one prediction request.
MAX_SAMPLES = 64
MAX_VALUES = 250_000  # input numbers of all samples together
MAX_TOP_K = 10
DEFAULT_TOP_K = 3
# Binary classification: the class whose probability is compared with the run's decision
# threshold (the second one, as in the evaluation's threshold curve).
BINARY_POSITIVE = 1
# How input values are given: "raw" as in the CSV file or image file (pixels 0-255), "unit"
# for pixels scaled to 0-1, "normalized" for values that are already prepared.
INPUT_SCALES = ("raw", "unit", "normalized")
# Image files of one request (base64): how many, and their size together as base64 text
# (JSON requests of the REST API are limited to 1 MB).
MAX_IMAGE_FILES = 8
MAX_IMAGE_TEXT = 900_000
# What a segmentation answers with besides the pixel share of every class: the class map as
# a small PNG file (one colour per class), as run-lengths, or not at all.
MASK_FORMATS = ("png", "rle", "none")


class UnknownPreprocessing(ValueError):
    """The run cannot be used for predictions: how its inputs were prepared is unknown."""


class InvalidInputs(ValueError):
    """The inputs for a prediction do not fit the network. The message is for the user."""


# --------------------------------------------------------------------------- #
# What the network takes
# --------------------------------------------------------------------------- #


def _numbers(value: Any, length: int | None = None) -> bool:
    return (
        isinstance(value, list)
        and (length is None or len(value) == length)
        and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
            for v in value
        )
    )


def is_classical(plan: Any) -> bool:
    """True for a run of a classical model (scikit-learn), see compat.TrainingPlan; also for
    a hybrid network, which predicts with one (see :func:`is_hybrid`)."""
    return isinstance(plan, dict) and isinstance(plan.get("classical"), dict)


def is_hybrid(plan: Any) -> bool:
    """True for a run of a hybrid network: neural layers compute the features of a
    classical model. It takes inputs like a neural network and answers like a classical
    model (probabilities, values in the units of the target)."""
    return isinstance(plan, dict) and isinstance(plan.get("hybrid"), dict)


def is_segmentation(plan: Any) -> bool:
    """True for a run that predicts a class for every pixel, see compat.TrainingPlan."""
    return isinstance(plan, dict) and plan.get("task") == "segmentation"


def is_binary(plan: Any) -> bool:
    """True for a binary classification: one class output with two classes (a neural
    network with two outputs, or a classical classifier of two classes). Its predictions
    use the run's decision threshold on the probability of the second class."""
    if not isinstance(plan, dict) or plan.get("task") != "classification":
        return False
    targets = plan.get("targets")
    if not isinstance(targets, list) or len(targets) != 1 or not isinstance(targets[0], dict):
        return False
    target = targets[0]
    return target.get("kind") == "class" and target.get("num_classes") == 2


def input_specs(plan: Any) -> list[dict[str, Any]]:
    inputs = plan.get("inputs") if isinstance(plan, dict) else None
    return [i for i in inputs if isinstance(i, dict)] if isinstance(inputs, list) else []


def kind_of(plan: Any, preprocessing: Any) -> str:
    """``tabular``, ``image`` or ``demo``, after checking that both documents are complete.

    Raises :class:`UnknownPreprocessing`.
    """
    unknown = UnknownPreprocessing(
        "How the inputs of this run were prepared is not known (the training did not get "
        "far enough to record it), so it cannot be used for predictions."
    )
    if not isinstance(preprocessing, dict) or preprocessing.get("kind") not in KINDS:
        raise unknown
    kind = preprocessing["kind"]
    inputs = input_specs(plan)
    targets = plan.get("targets") if isinstance(plan, dict) else None
    if not inputs or not isinstance(targets, list) or not targets:
        raise unknown
    if kind != "demo" and (len(inputs) != 1 or len(targets) != 1):
        raise unknown
    for spec in inputs:
        shape = spec.get("shape")
        if not isinstance(shape, list) or not all(isinstance(s, int) and s > 0 for s in shape):
            raise unknown
    if kind == "tabular":
        columns = preprocessing.get("columns")
        n = len(columns) if isinstance(columns, list) else -1
        if inputs[0]["shape"] != [n] or not (
            _numbers(preprocessing.get("mean"), n) and _positive(preprocessing.get("std"), n)
        ):
            raise unknown
        if (
            targets[0].get("kind") == "value"
            and not is_classical(plan)
            and not _numbers([preprocessing.get("target_mean"), preprocessing.get("target_std")])
        ):
            raise unknown
    elif kind == "image":
        c, h, w = (preprocessing.get(key) for key in ("channels", "height", "width"))
        if not (
            c in (1, 3)
            and isinstance(h, int)
            and isinstance(w, int)
            and _numbers(preprocessing.get("mean"), c)
            and _positive(preprocessing.get("std"), c)
        ):
            raise unknown
        layouts = {"chw": [c, h, w], "hw": [h, w], "flat": [c * h * w]}
        if layouts.get(preprocessing.get("layout")) != inputs[0]["shape"]:
            raise unknown
    return kind


def _positive(value: Any, length: int) -> bool:
    return _numbers(value, length) and all(v > 0 for v in value)


def class_names(plan: Any, preprocessing: Any, index: int = 0) -> list[str] | None:
    """Names of the classes of output ``index`` (``None`` if it is no class output)."""
    targets = plan.get("targets") if isinstance(plan, dict) else None
    target = targets[index] if isinstance(targets, list) and index < len(targets) else None
    if not isinstance(target, dict) or target.get("kind") not in ("class", "mask"):
        return None
    k = int(target.get("num_classes") or 0)
    names = None
    if isinstance(preprocessing, dict) and isinstance(preprocessing.get("class_names"), list):
        names = preprocessing["class_names"]
    if not names and isinstance(plan.get("class_names"), list):
        names = plan["class_names"]
    names = [str(n) for n in (names or [])][:k]
    return names + [f"class {i}" for i in range(len(names), k)]


def _shape_text(shape: list[int]) -> str:
    return "×".join(str(s) for s in shape)


def describe_inputs(plan: Any, preprocessing: Any) -> str:
    """One or two sentences on what a sample looks like."""
    kind = kind_of(plan, preprocessing)
    if kind == "tabular":
        columns = preprocessing["columns"]
        return (
            f"One row of {len(columns)} numbers per sample, the feature columns of the CSV file "
            f"in this order: {', '.join(columns)}. They are standardised with the mean and "
            "standard deviation of the training data (x - mean) / std."
        )
    if is_classical(plan) and not is_hybrid(plan):
        n = input_specs(plan)[0]["shape"][0]
        return (
            f"Trained on random demo data, so its predictions mean nothing. One row of {n} "
            "numbers per sample."
        )
    if kind == "image":
        c, h, w = preprocessing["channels"], preprocessing["height"], preprocessing["width"]
        mode = "RGB" if c == 3 else "grayscale"
        network = {
            "chw": "",
            "hw": f" The network takes it as {h}×{w} (height, width).",
            "flat": f" The network takes it flattened to {c * h * w} values.",
        }[preprocessing["layout"]]
        also = f" or {h}×{w}" if c == 1 else ""
        output = (
            f" The network answers with one of {len(class_names(plan, preprocessing) or [])} "
            f"classes for each of the {h}×{w} pixels."
            if is_segmentation(plan)
            else ""
        )
        return (
            f"One {mode} image of {w}×{h} pixels per sample, as a {c}×{h}×{w} array (channels, "
            f"height, width){also}.{network} Pixels are scaled to 0–1 (pixel / 255), then "
            f"normalised per channel: (x - mean) / std with mean {_list(preprocessing['mean'])} "
            f"and std {_list(preprocessing['std'])}. Or pass image files (PNG, JPEG …) as "
            f"base64 text in 'image': they are converted to {mode} and resized to {w}×{h}."
            f"{output}"
        )
    shapes = "; ".join(
        f"{_shape_text(s['shape'])} {'token ids' if s.get('dtype') == 'long' else 'numbers'}"
        for s in input_specs(plan)
    )
    return (
        "Trained on random demo data, so its outputs mean nothing. Per sample one array per "
        f"Input layer ({shapes}), used as it is (no preprocessing)."
    )


# --------------------------------------------------------------------------- #
# Code to use the weights
# --------------------------------------------------------------------------- #


def module_name(name: str) -> str:
    """A Python module name for the generated script, e.g. ``iris_classifier``."""
    ident = re.sub(r"\W+", "_", name.strip().lower()).strip("_")[:60]
    if ident[:1].isdigit():
        ident = f"model_{ident}"
    if not ident.isascii() or not ident.isidentifier() or keyword.iskeyword(ident):
        return "model"
    return ident


def _list(values: list[float], digits: int = 6) -> str:
    return "[" + ", ".join(format(float(v), f".{digits}g") for v in values) + "]"


def usage_snippet(
    plan: Any,
    preprocessing: Any,
    module: str,
    weights_file: str,
    class_name: str = "NeuralNetwork",
) -> str:
    """Python code that loads the weights and predicts one sample like the training saw it."""
    kind = kind_of(plan, preprocessing)
    if is_hybrid(plan):
        return _hybrid_snippet(plan, preprocessing, module, weights_file)
    if is_classical(plan):
        return _classical_snippet(plan, preprocessing, weights_file)
    lines = [
        "import torch",
        f"from {module} import {class_name}  # model_code, saved as {module}.py",
        "",
        f"model = {class_name}()",
        f'model.load_state_dict(torch.load("{weights_file}", map_location="cpu", '
        "weights_only=True))",
        "model.eval()",
        "",
    ]
    names = class_names(plan, preprocessing)
    if kind == "tabular":
        columns = preprocessing["columns"]
        example = _list(preprocessing["mean"], 4)  # the average row: a valid example
        lines += [
            f"# Features in this order: {', '.join(columns)}",
            f"MEAN = torch.tensor({_list(preprocessing['mean'])})",
            f"STD = torch.tensor({_list(preprocessing['std'])})",
            f"rows = [{example}]  # one list of {len(columns)} values per sample",
            "x = (torch.tensor(rows, dtype=torch.float32) - MEAN) / STD",
            "with torch.no_grad():",
            "    output = model(x)",
        ]
        if names is None:
            mean, std = preprocessing["target_mean"], preprocessing["target_std"]
            target = preprocessing.get("target") or "target"
            lines.append(
                f"print((output[:, 0] * {float(std):.6g} + {float(mean):.6g}).tolist())"
                f"  # predicted {target}"
            )
    elif kind == "image":
        c, h, w = preprocessing["channels"], preprocessing["height"], preprocessing["width"]
        layout = preprocessing["layout"]
        lines += [
            "from PIL import Image",
            "",
            f"MEAN = torch.tensor({_list(preprocessing['mean'])}).view(-1, 1, 1)",
            f"STD = torch.tensor({_list(preprocessing['std'])}).view(-1, 1, 1)",
            f'image = Image.open("example.png").convert("{"RGB" if c == 3 else "L"}")',
            f"image = image.resize(({w}, {h}), Image.BILINEAR)",
            "pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)",
            f"x = pixels.view({h}, {w}, {c}).permute(2, 0, 1).float() / 255",
            f"x = ((x - MEAN) / STD).unsqueeze(0)  # a batch of one image: 1×{c}×{h}×{w}",
        ]
        if layout == "hw":
            lines.append(f"x = x[:, 0]  # the network takes {h}×{w}")
        elif layout == "flat":
            lines.append(f"x = x.flatten(1)  # the network takes {c * h * w} values")
        lines += ["with torch.no_grad():", "    output = model(x)"]
    else:
        lines.append("# Trained on random demo data: the outputs mean nothing.")
        variables = []
        for i, spec in enumerate(input_specs(plan), start=1):
            dims = ", ".join(str(d) for d in (1, *spec["shape"]))
            name = f"x{i}"
            variables.append(name)
            if spec.get("dtype") == "long":
                vocab = int(spec.get("vocab") or 2)
                lines.append(f"{name} = torch.randint(0, {vocab}, ({dims},))  # token ids")
            else:
                lines.append(f"{name} = torch.randn({dims})")
        lines += ["with torch.no_grad():", f"    output = model({', '.join(variables)})"]
    if kind == "demo":
        lines.append("print(output)")
    elif is_segmentation(plan):
        binary = bool(plan["targets"][0].get("binary"))
        lines += [
            f"CLASS_NAMES = {names!r}",
            "# One class per pixel: the class map has the size of the resized image.",
            (
                "classes = (output[0, 0] > 0).to(torch.uint8)  # 1 where the score is positive"
                if binary
                else "classes = output.argmax(dim=1)[0].to(torch.uint8)"
            ),
            "for index, name in enumerate(CLASS_NAMES):",
            '    print(f"{name}: {(classes == index).float().mean().item():.1%} of the pixels")',
            'Image.fromarray(classes.numpy()).save("mask.png")  # pixel value = class index',
        ]
    elif names is not None:
        lines += [
            f"CLASS_NAMES = {names!r}",
            "probabilities = output.softmax(dim=1)",
            "print([CLASS_NAMES[i] for i in probabilities.argmax(dim=1).tolist()])",
        ]
    return "\n".join(lines) + "\n"


def _classical_snippet(plan: dict[str, Any], preprocessing: dict[str, Any], path: str) -> str:
    """Code that loads the fitted scikit-learn pipeline (joblib) and predicts one row."""
    classical = plan["classical"]
    estimator = classical.get("estimator") or "the model"
    if preprocessing.get("kind") == "tabular":
        columns = preprocessing["columns"]
        example = _list(preprocessing["mean"], 4)  # the average row: a valid example
        about = f"# Features in this order: {', '.join(columns)}"
        rows = f"rows = [{example}]  # one list of {len(columns)} values per sample, as in the CSV"
    else:
        n = input_specs(plan)[0]["shape"][0]
        about = "# Trained on random demo data: the predictions mean nothing."
        rows = f"rows = [{_list([0.0] * n)}]  # one list of {n} values per sample"
    lines = [
        "import joblib",
        "",
        f'bundle = joblib.load("{path}")  # a pickle: only load files you trust',
        f'model = bundle["model"]  # scikit-learn Pipeline: StandardScaler, then {estimator}',
        "",
        about,
        rows,
    ]
    target = preprocessing.get("target") or "target"
    if classical.get("voting") == "hard":  # predicts classes, but no probabilities
        lines += [
            'names = dict(zip(model.classes_, bundle["class_names"]))',
            "print([names[label] for label in model.predict(rows)])",
        ]
    elif class_names(plan, preprocessing) is not None:
        lines += [
            "probabilities = model.predict_proba(rows)  # one column per bundle['class_names']",
            'print([bundle["class_names"][i] for i in probabilities.argmax(axis=1)])',
        ]
    elif classical.get("type") == "GaussianProcessRegressor":
        lines += [
            "mean, std = model.predict(rows, return_std=True)",
            f"print(mean.tolist(), std.tolist())  # predicted {target} and its uncertainty",
        ]
    else:
        lines.append(f"print(model.predict(rows).tolist())  # predicted {target}")
    return "\n".join(lines) + "\n"


def _hybrid_snippet(
    plan: dict[str, Any], preprocessing: dict[str, Any], module: str, archive: str
) -> str:
    """Code that loads both parts of a hybrid network (from its ZIP file) and predicts one
    row: the neural layers compute the features, the classical model predicts."""
    stem = archive.removesuffix(".zip")
    classical = plan["classical"]
    lines = [
        "import joblib",
        "import torch",
        f"from {module} import FeatureExtractor  # {module}.py from the ZIP file",
        "",
        "extractor = FeatureExtractor()",
        f'checkpoint = torch.load("{stem}.features.pt", map_location="cpu", weights_only=True)',
        'extractor.load_state_dict(checkpoint["state_dict"])',
        "extractor.eval()",
        f'bundle = joblib.load("{stem}.joblib")  # a pickle: only load files you trust',
        'model = bundle["model"]  # fitted on the learned features',
        "",
    ]
    if preprocessing.get("kind") == "tabular":
        columns = preprocessing["columns"]
        example = _list(preprocessing["mean"], 4)  # the average row: a valid example
        lines += [
            f"# Input columns in this order: {', '.join(columns)}",
            f"MEAN = torch.tensor({_list(preprocessing['mean'])})",
            f"STD = torch.tensor({_list(preprocessing['std'])})",
            f"rows = [{example}]  # one list of {len(columns)} values per sample",
            "x = (torch.tensor(rows, dtype=torch.float32) - MEAN) / STD",
        ]
    else:
        dims = ", ".join(str(d) for d in (1, *input_specs(plan)[0]["shape"]))
        lines.append(f"x = torch.randn({dims})  # prepared like the training inputs")
    lines += [
        "with torch.no_grad():",
        "    features = extractor(x).double().numpy()  # the learned features",
    ]
    target = preprocessing.get("target") or "target"
    if classical.get("voting") == "hard":
        lines += [
            'names = dict(zip(model.classes_, bundle["class_names"]))',
            "print([names[label] for label in model.predict(features)])",
        ]
    elif class_names(plan, preprocessing) is not None:
        lines += [
            "probabilities = model.predict_proba(features)  # one column per class_names",
            'print([bundle["class_names"][i] for i in probabilities.argmax(axis=1)])',
        ]
    else:
        lines.append(f"print(model.predict(features).tolist())  # predicted {target}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Inputs of a prediction
# --------------------------------------------------------------------------- #


@dataclass
class PreparedInputs:
    """Inputs as the network gets them: one tensor per Input layer, ``{"dtype": "float" |
    "long", "shape": [N, ...], "values": [all numbers, row-major]}``."""

    kind: str
    count: int
    tensors: list[dict[str, Any]]
    top_k: int
    # Image files as base64 text instead of ``tensors``: the prediction process decodes
    # them and prepares them like the training images (``image``: how, see image_request).
    files: list[str] | None = None
    image: dict[str, Any] | None = None


def _dims(value: Any) -> list[int]:
    """The shape of a nested list, following its first elements."""
    dims = []
    while isinstance(value, list):
        dims.append(len(value))
        if not value:
            break
        value = value[0]
    return dims


def _shape_of(value: Any) -> str:
    dims = _dims(value)
    if dims:
        return _shape_text(dims)
    return "a single value" if isinstance(value, (int, float)) else type(value).__name__


def _number(value: Any, where: str, decimal_comma: bool = False) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(value):
            return float(value)
    elif isinstance(value, str):
        parsed = parse_number(value, decimal_comma)
        if parsed is not None:
            return parsed
    raise InvalidInputs(f"{where}: {repr(value)[:30]} is not a number.")


def _flatten(sample: Any, shape: list[int], label: str, wanted: str) -> list[float]:
    """The numbers of a nested list of ``shape``, row-major."""
    values: list[float] = []

    def walk(value: Any, depth: int) -> None:
        if depth == len(shape):
            values.append(_number(value, label))
            return
        if not isinstance(value, list) or len(value) != shape[depth]:
            raise InvalidInputs(
                f"{label} has the shape {_shape_of(sample)}, but the network takes {wanted}."
            )
        for item in value:
            walk(item, depth + 1)

    walk(sample, 0)
    return values


def _sample_list(inputs: Any, shapes: list[list[int]], wanted: str) -> list[Any]:
    """The list of samples; a single sample of one of ``shapes`` may be given unwrapped."""
    if not isinstance(inputs, list) or not inputs:
        raise InvalidInputs(
            f"'inputs' must be a list of samples (at most {MAX_SAMPLES}), each {wanted}."
        )
    if _dims(inputs) in shapes:
        inputs = [inputs]
    if len(inputs) > MAX_SAMPLES:
        raise InvalidInputs(
            f"{len(inputs)} samples are too many: predict at most {MAX_SAMPLES} at a time."
        )
    return inputs


def _check_size(count: int, shape: list[int]) -> None:
    if count * math.prod(shape) > MAX_VALUES:
        raise InvalidInputs(
            f"The inputs are too large: at most {MAX_VALUES:,} numbers per request "
            f"({count} samples of {math.prod(shape):,} numbers). Send fewer samples at a time."
        )


def _label(number: int, count: int) -> str:
    return f"Sample {number}" if count > 1 else "The sample"


def _tabular_inputs(preprocessing: dict[str, Any], inputs: Any, scale: str) -> list[float]:
    columns = [str(c) for c in preprocessing["columns"]]
    n = len(columns)
    wanted = f"an object {{column: value}} or a list of {n} numbers ({', '.join(columns)})"
    if scale == "unit":
        raise InvalidInputs("input_scale 'unit' is for image models; use 'raw' or 'normalized'.")
    if isinstance(inputs, dict) or (
        isinstance(inputs, list) and inputs and not isinstance(inputs[0], (list, dict))
    ):
        inputs = [inputs]  # a single row
    rows = _sample_list(inputs, [], wanted)
    _check_size(len(rows), [n])
    index = {name: i for i, name in enumerate(columns)}
    target = preprocessing.get("target")
    decimal_comma = bool(preprocessing.get("decimal_comma"))
    mean, std = preprocessing["mean"], preprocessing["std"]
    values: list[float] = []
    for number, row in enumerate(rows, start=1):
        label = _label(number, len(rows))
        if isinstance(row, dict):
            if len(index) != n:
                raise InvalidInputs(
                    "The CSV file has columns with the same name: give each sample as a list "
                    f"of {n} numbers in column order."
                )
            unknown = [str(k) for k in row if k not in index and k != target]
            if unknown:
                raise InvalidInputs(
                    f"{label}: unknown column '{unknown[0][:60]}'. The network takes: "
                    f"{', '.join(columns)}."
                )
            missing = [c for c in columns if c not in row]
            if missing:
                more = f" (and {len(missing) - 1} more)" if len(missing) > 1 else ""
                raise InvalidInputs(
                    f"{label}: the column '{missing[0]}' is missing{more}. The network takes: "
                    f"{', '.join(columns)}."
                )
            cells = [row[c] for c in columns]
        elif isinstance(row, list):
            if len(row) != n:
                raise InvalidInputs(
                    f"{label} has {len(row)} values, but the network takes {n}: "
                    f"{', '.join(columns)}."
                )
            cells = row
        else:
            raise InvalidInputs(f"{label} must be {wanted}.")
        for i, cell in enumerate(cells):
            x = _number(cell, f"{label}, '{columns[i]}'", decimal_comma)
            values.append((x - mean[i]) / std[i] if scale == "raw" else x)
    return values


def _image_inputs(
    plan: Any, preprocessing: dict[str, Any], inputs: Any, scale: str | None
) -> tuple[int, list[float]]:
    c, h, w = preprocessing["channels"], preprocessing["height"], preprocessing["width"]
    shape = input_specs(plan)[0]["shape"]
    mean, std = preprocessing["mean"], preprocessing["std"]
    if scale is None:
        raise InvalidInputs(
            "Say how the pixel values are given with input_scale: 'raw' (0–255, as in an "
            "image file), 'unit' (0–1) or 'normalized' (scaled to 0–1, then normalised per "
            f"channel with mean {mean} and std {std})."
        )
    # An image as channels × height × width (grayscale: also height × width) or in the shape
    # of the Input layer: all of them flatten to the same order (data.apply_layout).
    shapes = [[c, h, w]] + ([[h, w]] if c == 1 else [])
    if shape not in shapes:
        shapes.append(shape)
    wanted = f"{'an RGB' if c == 3 else 'a grayscale'} image as a nested list of " + " or ".join(
        _shape_text(s) for s in shapes
    )
    samples = _sample_list(inputs, shapes, wanted)
    _check_size(len(samples), shape)
    values: list[float] = []
    plane = h * w
    for number, sample in enumerate(samples, start=1):
        label = _label(number, len(samples))
        dims = _dims(sample)
        own = next((s for s in shapes if s == dims), shapes[0])
        flat = _flatten(sample, own, label, wanted)
        if scale == "normalized":
            values += flat
            continue
        top = 255.0 if scale == "raw" else 1.0
        for i, x in enumerate(flat):
            if not 0 <= x <= top:
                hint = " Use input_scale 'raw' for values up to 255." if scale == "unit" else ""
                raise InvalidInputs(
                    f"{label}: {x:g} is not a pixel value between 0 and {top:g}.{hint}"
                )
            channel = i // plane
            values.append((x / top - mean[channel]) / std[channel])
    return len(samples), values


def _demo_inputs(plan: Any, inputs: Any) -> tuple[int, list[float], str]:
    specs = input_specs(plan)
    if len(specs) != 1:
        raise InvalidInputs(
            f"This network has {len(specs)} Input layers; predictions support one Input."
        )
    spec = specs[0]
    shape, dtype = spec["shape"], spec.get("dtype") or "float"
    wanted = _shape_text(shape) + (" token ids" if dtype == "long" else " numbers")
    samples = _sample_list(inputs, [shape], f"a nested list of {wanted}")
    _check_size(len(samples), shape)
    values: list[float] = []
    vocab = int(spec.get("vocab") or 2)
    for number, sample in enumerate(samples, start=1):
        label = _label(number, len(samples))
        flat = _flatten(sample, shape, label, wanted)
        if dtype == "long" and not all(x.is_integer() and 0 <= x < vocab for x in flat):
            raise InvalidInputs(f"{label}: token ids must be whole numbers from 0 to {vocab - 1}.")
        values += flat
    return len(samples), values, dtype


def prepare_inputs(
    plan: Any,
    preprocessing: Any,
    inputs: Any,
    input_scale: Any = None,
    top_k: Any = None,
) -> PreparedInputs:
    """Check ``inputs`` against the network and prepare them like the training data.

    Raises :class:`UnknownPreprocessing` (the run cannot be used) or :class:`InvalidInputs`.
    """
    kind = kind_of(plan, preprocessing)
    if input_scale is not None and input_scale not in INPUT_SCALES:
        raise InvalidInputs(f"input_scale must be one of: {', '.join(INPUT_SCALES)}.")
    if top_k is None:
        top_k = DEFAULT_TOP_K
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= MAX_TOP_K:
        raise InvalidInputs(f"top_k must be a whole number from 1 to {MAX_TOP_K}.")
    shape = input_specs(plan)[0]["shape"]
    dtype = "float"
    if kind == "tabular":
        values = _tabular_inputs(preprocessing, inputs, input_scale or "raw")
        count = len(values) // shape[0]
    elif kind == "image":
        count, values = _image_inputs(plan, preprocessing, inputs, input_scale)
    else:
        count, values, dtype = _demo_inputs(plan, inputs)
    tensor = {"dtype": dtype, "shape": [count, *shape], "values": values}
    return PreparedInputs(kind, count, [tensor], top_k)


def image_request(preprocessing: dict[str, Any]) -> dict[str, Any]:
    """How the prediction process prepares image files: what the training wrote down."""
    keys = ("channels", "height", "width", "layout", "mean", "std")
    return {key: preprocessing[key] for key in keys}


def prepare_image_files(
    plan: Any, preprocessing: Any, image: Any, top_k: Any = None
) -> PreparedInputs:
    """Check image files given as base64 text (one, or a list of at most MAX_IMAGE_FILES;
    a ``data:image/png;base64,`` prefix is fine). They are decoded in the prediction
    process. Raises :class:`UnknownPreprocessing` or :class:`InvalidInputs`."""
    kind = kind_of(plan, preprocessing)
    if kind != "image":
        raise InvalidInputs(
            "'image' is for networks trained on images; this one takes numbers in 'inputs'."
        )
    if top_k is None:
        top_k = DEFAULT_TOP_K
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= MAX_TOP_K:
        raise InvalidInputs(f"top_k must be a whole number from 1 to {MAX_TOP_K}.")
    files = [image] if isinstance(image, str) else image
    wanted = (
        "'image' must be an image file (PNG, JPEG …) as base64 text, or a list of at most "
        f"{MAX_IMAGE_FILES} of them."
    )
    if not isinstance(files, list) or not files or not all(isinstance(f, str) for f in files):
        raise InvalidInputs(wanted)
    if len(files) > MAX_IMAGE_FILES:
        raise InvalidInputs(
            f"{len(files)} images are too many: predict at most {MAX_IMAGE_FILES} at a time."
        )
    cleaned = []
    for number, text in enumerate(files, start=1):
        label = _label(number, len(files)).replace("sample", "image").replace("Sample", "Image")
        text = text.strip()
        if text.startswith("data:"):
            head, _, text = text.partition(",")
            if not head.endswith(";base64"):
                raise InvalidInputs(f"{label}: a data URL must be base64 encoded.")
        text = "".join(text.split())
        if not re.fullmatch(r"[A-Za-z0-9+/]+={0,2}", text) or len(text) % 4 == 1:
            raise InvalidInputs(
                f"{label} is not base64 text. Encode the bytes of the image file with base64."
            )
        cleaned.append(text)
    if sum(len(text) for text in cleaned) > MAX_IMAGE_TEXT:
        raise InvalidInputs(
            f"The images are too large: at most {MAX_IMAGE_TEXT // 1000} kB of base64 text "
            f"per request (about {MAX_IMAGE_TEXT * 3 // 4000} kB of image files). Scale them "
            "down first: the network only sees "
            f"{preprocessing['width']}×{preprocessing['height']} pixels anyway."
        )
    return PreparedInputs(
        kind, len(cleaned), [], top_k, files=cleaned, image=image_request(preprocessing)
    )


def mask_format(value: Any) -> str:
    """The checked ``mask`` argument of a segmentation prediction (default "png")."""
    if value is None:
        return MASK_FORMATS[0]
    if value not in MASK_FORMATS:
        raise InvalidInputs(f"mask must be one of: {', '.join(MASK_FORMATS)}.")
    return value


# --------------------------------------------------------------------------- #
# Outputs of a prediction
# --------------------------------------------------------------------------- #


def read_masks(plan: Any, preprocessing: Any, masks: Any, count: int) -> list[dict[str, Any]]:
    """Turn what the prediction process reports for a segmentation (per sample ``{"width",
    "height", "counts": [pixels per class], "png": base64 | None, "rle": [...] | None}``)
    into one prediction per sample::

        {"width": 64, "height": 64,
         "classes": [{"class": "background", "index": 0, "share": 0.91}, ...],  # largest first
         "mask_png": "data:image/png;base64,…",    # the class map, one colour per class
         "mask_rle": [0, 130, 2, 5, ...]}          # or run-lengths: class, pixels, class, …

    ``classes`` lists the classes that occur. Raises ``ValueError`` for masks that do not fit
    the training plan."""
    names = class_names(plan, preprocessing) or []
    if not isinstance(masks, list) or len(masks) != count:
        raise ValueError("The network returned an unexpected number of samples.")
    predictions = []
    for mask in masks:
        counts = mask.get("counts") if isinstance(mask, dict) else None
        if not _numbers(counts, len(names)) or not sum(counts):
            raise ValueError("The network returned an unexpected number of classes.")
        total = sum(counts)
        classes = [
            {"class": names[i], "index": i, "share": round(counts[i] / total, 4)}
            for i in sorted(range(len(names)), key=lambda i: -counts[i])
            if counts[i] > 0
        ]
        prediction: dict[str, Any] = {
            "width": mask.get("width"),
            "height": mask.get("height"),
            "classes": classes,
        }
        if isinstance(mask.get("png"), str):
            prediction["mask_png"] = "data:image/png;base64," + mask["png"]
        if isinstance(mask.get("rle"), list):
            prediction["mask_rle"] = mask["rle"]
        predictions.append(prediction)
    return predictions


def _finite(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, list):
        return [_finite(v) for v in value]
    return value


def _flat_numbers(value: Any) -> list[float | None]:
    if isinstance(value, list):
        return [x for item in value for x in _flat_numbers(item)]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [float(value) if math.isfinite(value) else None]
    if value is None:  # NaN or infinity, written as null
        return [None]
    raise ValueError("The network returned something that is not a number.")


def _classes(
    scores: list[float | None],
    names: list[str],
    top_k: int,
    probabilities: bool = False,
    threshold: float | None = None,
) -> dict[str, Any]:
    """The top classes; ``scores`` are raw scores (softmax) or already ``probabilities``.
    With a ``threshold`` (two classes) the predicted class is the second one when its
    probability is at least the threshold (``positive_probability``), else the first; the
    top classes stay in the order of their probabilities."""
    if any(s is None for s in scores):
        empty: dict[str, Any] = {"class": None, "index": None, "probability": None, "top": []}
        if threshold is not None:
            empty["positive_probability"] = None
        return empty
    numbers = [float(s) for s in scores if s is not None]
    if probabilities:
        total = sum(numbers) or 1.0
        probs = [max(p, 0.0) / total for p in numbers]
    else:
        peak = max(numbers)
        exps = [math.exp(s - peak) for s in numbers]
        total = sum(exps)
        probs = [e / total for e in exps]
    order = sorted(range(len(numbers)), key=lambda i: -probs[i])
    top = [
        {"class": names[i], "index": i, "probability": round(probs[i], 4)} for i in order[:top_k]
    ]
    if threshold is None or len(probs) != 2:
        return {**top[0], "top": top}
    positive = probs[BINARY_POSITIVE]
    chosen = BINARY_POSITIVE if positive >= threshold else 1 - BINARY_POSITIVE
    return {
        "class": names[chosen],
        "index": chosen,
        "probability": round(probs[chosen], 4),
        # Unrounded enough to compare with the threshold (0.29996 is not 0.3).
        "positive_probability": float(f"{positive:.6g}"),
        "top": top,
    }


def classes_from_probabilities(
    probabilities: list[float | None],
    names: list[str],
    top_k: int = DEFAULT_TOP_K,
    threshold: float | None = None,
) -> dict[str, Any]:
    """One prediction from class probabilities (e.g. an ensemble's combined ones), with
    the top classes and, for two classes, the decision ``threshold`` (see :func:`_classes`)."""
    return _classes(probabilities, names, top_k, probabilities=True, threshold=threshold)


def read_outputs(
    plan: Any,
    preprocessing: Any,
    outputs: Any,
    count: int,
    top_k: int = DEFAULT_TOP_K,
    probabilities: bool = False,
    std: Any = None,
    threshold: float | None = None,
) -> list[dict[str, Any]]:
    """Turn the raw outputs of the network (one nested list per output, ``count`` samples
    each) into one prediction per sample: the classes with their probabilities (softmax,
    as the training's cross-entropy loss reads them; classical classifiers already give
    ``probabilities``) or the predicted value (with ``std``, a Gaussian process's
    predictive standard deviation per sample, as "std"). ``threshold``: the decision
    threshold of a binary classification (see :func:`is_binary` and :func:`_classes`),
    ignored for other runs. Raises ``ValueError`` for outputs that do not fit the training
    plan."""
    kind = kind_of(plan, preprocessing)
    targets = plan["targets"]
    if not is_binary(plan):
        threshold = None
    if not isinstance(outputs, list) or len(outputs) != len(targets):
        raise ValueError("The network returned an unexpected number of outputs.")
    per_output = []
    for index, output in enumerate(outputs):
        if not isinstance(output, list) or len(output) != count:
            raise ValueError("The network returned an unexpected number of samples.")
        names = class_names(plan, preprocessing, index)
        results = []
        for sample in output:
            if names is not None:
                scores = _flat_numbers(sample)
                if len(scores) != len(names):
                    raise ValueError("The network returned an unexpected number of classes.")
                results.append(_classes(scores, names, top_k, probabilities, threshold))
            elif kind == "tabular":
                scores = _flat_numbers(sample)
                if len(scores) != 1:
                    raise ValueError("The network returned an unexpected number of values.")
                value = scores[0]
                if value is not None and preprocessing.get("target_std") is not None:
                    value = value * preprocessing["target_std"] + preprocessing["target_mean"]
                results.append({"value": value})
            else:
                value = _finite(sample)
                if isinstance(value, list) and len(value) == 1:
                    value = value[0]
                results.append({"value": value})
        per_output.append(results)
    if std is not None:
        spreads = _flat_numbers(std)
        if len(per_output) != 1 or len(spreads) != count:
            raise ValueError("The model returned an unexpected number of uncertainties.")
        for result, spread in zip(per_output[0], spreads, strict=True):
            result["std"] = spread
    if len(per_output) == 1:
        return per_output[0]
    return [{"outputs": [results[i] for results in per_output]} for i in range(count)]
