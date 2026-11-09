"""Using a trained network (no PyTorch needed): what it takes as input and how to prepare
new inputs the way the training data was prepared.

Everything here works on two documents of a run: the training plan (``TrainingRun.plan``,
see compat.py) and the ``preprocessing`` that the training wrote into ``state.json``
(see data.py)::

    {"kind": "tabular", "columns": [...], "target": "price", "mean": [...], "std": [...],
     "target_mean": 3.2, "target_std": 1.1, "class_names": null, ...}
    {"kind": "image", "channels": 1, "height": 28, "width": 28, "layout": "chw",
     "mean": [0.1307], "std": [0.3081], "class_names": ["0", ...], ...}
    {"kind": "demo"}            # random demo data: inputs are used as they are
"""

from __future__ import annotations

import keyword
import math
import re
from typing import Any

KINDS = ("tabular", "image", "demo")


class UnknownPreprocessing(ValueError):
    """The run cannot be used for predictions: how its inputs were prepared is unknown."""


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
            _numbers(preprocessing.get("mean"), n) and _numbers(preprocessing.get("std"), n)
        ):
            raise unknown
        if targets[0].get("kind") == "value" and not (
            _numbers([preprocessing.get("target_mean"), preprocessing.get("target_std")])
        ):
            raise unknown
    elif kind == "image":
        channels = preprocessing.get("channels")
        if not (
            channels in (1, 3)
            and isinstance(preprocessing.get("height"), int)
            and isinstance(preprocessing.get("width"), int)
            and preprocessing.get("layout") in ("chw", "hw", "flat")
            and _numbers(preprocessing.get("mean"), channels)
            and _numbers(preprocessing.get("std"), channels)
        ):
            raise unknown
    return kind


def class_names(plan: Any, preprocessing: Any, index: int = 0) -> list[str] | None:
    """Names of the classes of output ``index`` (``None`` if it is no class output)."""
    targets = plan.get("targets") if isinstance(plan, dict) else None
    target = targets[index] if isinstance(targets, list) and index < len(targets) else None
    if not isinstance(target, dict) or target.get("kind") != "class":
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
    if kind == "image":
        c, h, w = preprocessing["channels"], preprocessing["height"], preprocessing["width"]
        mode = "RGB" if c == 3 else "grayscale"
        layout = {
            "chw": f"a {c}×{h}×{w} array (channels, height, width)",
            "hw": f"a {h}×{w} array (height, width)",
            "flat": f"{c * h * w} values (channels × height × width, flattened)",
        }[preprocessing["layout"]]
        return (
            f"One {mode} image of {w}×{h} pixels per sample, as {layout}. Pixels are scaled to "
            "0–1 (pixel / 255), then normalised per channel: (x - mean) / std."
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
    ident = re.sub(r"\W+", "_", name.strip().lower()).strip("_")
    if not ident or not ident.isascii() or not ident.isidentifier() or keyword.iskeyword(ident):
        return "model"
    return f"model_{ident}" if ident[0].isdigit() else ident[:60]


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
    elif names is not None:
        lines += [
            f"CLASS_NAMES = {names!r}",
            "probabilities = output.softmax(dim=1)",
            "print([CLASS_NAMES[i] for i in probabilities.argmax(dim=1).tolist()])",
        ]
    return "\n".join(lines) + "\n"
