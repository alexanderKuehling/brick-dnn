"""Explanations of a trained PyTorch network: saliency maps, Grad-CAM, occlusion,
integrated gradients, the output of every brick and the learned weights.

The network is the generated ``NeuralNetwork`` class (as for training and predictions),
but generated so that every brick keeps its own variable and its forward pass records the
output of every brick (see :func:`build_traced_model`). Its attribute names, and with them
the keys of the state dict, are those of the normal class, so the trained weights load.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ...codegen import GeneratorOptions, _Generator
from ...graph import Analysis, analyze
from . import render

# SmoothGrad: gradients of this many noisy copies of the image are averaged; the noise is
# this share of the value range of the input.
SMOOTHGRAD_SAMPLES = 16
SMOOTHGRAD_NOISE = 0.15
# Integrated gradients: steps along the straight line from the baseline to the input.
IG_STEPS = 32
# Occlusion: about this many patches per side; images are run in batches of this size.
OCCLUSION_PATCHES = 7
OCCLUSION_BATCH = 64
# Activations: feature maps shown per brick, the largest tile side, values of a vector.
MAX_CHANNELS = 16
MAX_TILE = 48
MAX_VECTOR = 256
MAX_MATRIX = (64, 128)
# Learned filters of the first convolution shown.
MAX_FILTERS = 64
HISTOGRAM_BINS = 24


# --------------------------------------------------------------------------- #
# The network, with the output of every brick
# --------------------------------------------------------------------------- #


class _TracingGenerator(_Generator):
    """The generated class, but every brick writes its output into ``self._brick_outputs``
    (generated names never start with an underscore)."""

    def _is_chain(self) -> bool:
        return False  # a variable per brick, so that no output is overwritten

    def _model_class(self) -> str:
        head = super()._model_class().split("\n\n    def forward(", 1)[0]
        args = ", ".join(f"{self.var[nid]}: torch.Tensor" for nid in self.a.inputs)
        lines = [head, "", f"    def forward(self, {args}):", "        self._brick_outputs = {}"]
        for nid in self.a.order:
            info = self._node(nid)
            if info.type != "Input":
                assert info.layer is not None
                inputs = [self.var[src] for src in info.inputs]
                for line in info.layer.forward_lines(self.attr.get(nid, ""), inputs, self.var[nid]):
                    lines.append(f"        {line}")
            lines.append(f"        self._brick_outputs[{nid!r}] = {self.var[nid]}")
        lines.append("        return " + ", ".join(self.var[nid] for nid in self.a.outputs))
        return "\n".join(lines)


def build_traced_model(graph: Any) -> tuple[torch.nn.Module, Analysis]:
    analysis = analyze(graph)
    generator = _TracingGenerator(analysis, GeneratorOptions(include_main=False))
    code = "import torch\nimport torch.nn as nn\n\n\n" + generator._model_class() + "\n"
    namespace: dict[str, Any] = {"__name__": "brickdnn_explain"}
    exec(compile(code, "<brickdnn explain>", "exec"), namespace)
    return namespace["NeuralNetwork"](), analysis


def node_label(analysis: Analysis, nid: str) -> str:
    info = analysis.nodes[nid]
    return info.name or info.type


# --------------------------------------------------------------------------- #
# Forward passes
# --------------------------------------------------------------------------- #


def forward(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    output = model(x)
    if isinstance(output, (tuple, list)):
        output = output[0]
    return output


def outputs_numpy(model: torch.nn.Module, x: np.ndarray, batch: int = 512) -> np.ndarray:
    parts = []
    with torch.no_grad():
        for start in range(0, len(x), batch):
            chunk = torch.as_tensor(x[start : start + batch], dtype=torch.float32)
            parts.append(forward(model, chunk).reshape(len(chunk), -1).double().numpy())
    return np.concatenate(parts, axis=0) if parts else np.zeros((0, 1))


# --------------------------------------------------------------------------- #
# Images: saliency, Grad-CAM, occlusion
# --------------------------------------------------------------------------- #


def saliency(
    model: torch.nn.Module, image: torch.Tensor, to_net: Any, target: int, seed: int = 0
) -> np.ndarray:
    """SmoothGrad: |d score / d pixel|, the maximum over the colour channels, averaged over
    noisy copies of the image (Smilkov et al., 2017). ``image`` is 1×C×H×W (normalised)."""
    generator = torch.Generator().manual_seed(seed)
    spread = float(image.max() - image.min()) or 1.0
    noise = torch.randn((SMOOTHGRAD_SAMPLES, *image.shape[1:]), generator=generator)
    batch = (image + noise * SMOOTHGRAD_NOISE * spread).requires_grad_(True)
    scores = forward(model, to_net(batch))[:, target].sum()
    (gradient,) = torch.autograd.grad(scores, batch)
    return gradient.abs().amax(dim=1).mean(dim=0).numpy()


def gradcam_layers(analysis: Analysis) -> list[str]:
    """Bricks whose output is a stack of feature maps (channels, height, width) larger than
    one pixel: where Grad-CAM can look. Convolutions first (the usual choice)."""
    found = []
    for nid in analysis.order:
        info = analysis.nodes[nid]
        shape = info.shape
        if info.type == "Input" or shape is None or len(shape) != 3:
            continue
        if shape[1] * shape[2] > 1:
            found.append(nid)
    return found


def default_gradcam_layer(analysis: Analysis, layers: list[str]) -> str | None:
    convolutions = [nid for nid in layers if analysis.nodes[nid].type == "Conv2d"]
    if convolutions:
        return convolutions[-1]
    return layers[-1] if layers else None


def gradcam(
    model: torch.nn.Module, image: torch.Tensor, to_net: Any, target: int, layer: str
) -> tuple[np.ndarray, np.ndarray]:
    """Grad-CAM (Selvaraju et al., 2017): the feature maps of ``layer`` weighted by the mean
    gradient of the score over each map, summed and clipped at zero. Returns the map at the
    layer's resolution and upsampled to the image (both unnormalised)."""
    # The weights do not need gradients: the input does, so that the maps get them.
    output = forward(model, to_net(image.clone().requires_grad_(True)))
    maps = model._brick_outputs[layer]
    (gradient,) = torch.autograd.grad(output[0, target], maps)
    weights = gradient[0].mean(dim=(1, 2))
    cam = torch.relu((weights[:, None, None] * maps[0]).sum(dim=0)).detach()
    height, width = image.shape[-2:]
    upsampled = F.interpolate(cam[None, None], size=(height, width), mode="bilinear")
    return cam.numpy(), upsampled[0, 0].numpy()


def occlusion(
    model: torch.nn.Module,
    image: torch.Tensor,
    to_net: Any,
    target: int,
    deadline: float | None = None,
) -> dict[str, Any] | None:
    """How much the probability of ``target`` drops when a square patch of the image is
    set to the dataset mean (0 after normalisation), for patches on a grid (Zeiler & Fergus,
    2014). None if the deadline comes first."""
    _, _, height, width = image.shape
    side = min(height, width)
    patch = max(1, math.ceil(side / OCCLUSION_PATCHES))
    stride = max(1, patch // 2)
    tops = list(range(0, max(1, height - patch + 1), stride))
    lefts = list(range(0, max(1, width - patch + 1), stride))
    if tops[-1] + patch < height:
        tops.append(height - patch)
    if lefts[-1] + patch < width:
        lefts.append(width - patch)
    with torch.no_grad():
        reference = torch.softmax(forward(model, to_net(image)), dim=1)[0, target].item()
        positions = [(t, left) for t in tops for left in lefts]
        drops = np.zeros(len(positions))
        for start in range(0, len(positions), OCCLUSION_BATCH):
            if deadline is not None and time.monotonic() > deadline:
                return None
            chunk = positions[start : start + OCCLUSION_BATCH]
            batch = image.repeat(len(chunk), 1, 1, 1)
            for i, (top, left) in enumerate(chunk):
                batch[i, :, top : top + patch, left : left + patch] = 0.0
            probs = torch.softmax(forward(model, to_net(batch)), dim=1)[:, target]
            drops[start : start + len(chunk)] = reference - probs.numpy()
    # Every pixel gets the mean drop of the patches that covered it.
    total = np.zeros((height, width))
    count = np.zeros((height, width))
    for (top, left), drop in zip(positions, drops, strict=True):
        total[top : top + patch, left : left + patch] += drop
        count[top : top + patch, left : left + patch] += 1
    pixel_map = total / np.maximum(count, 1)
    coarse = drops.reshape(len(tops), len(lefts))
    return {"map": pixel_map, "coarse": coarse, "patch": patch, "stride": stride}


# --------------------------------------------------------------------------- #
# Tabular: integrated gradients
# --------------------------------------------------------------------------- #


def integrated_gradients(
    score: Any, x: torch.Tensor, baseline: torch.Tensor, steps: int = IG_STEPS
) -> np.ndarray:
    """Integrated gradients (Sundararajan et al., 2017) of ``score(rows) -> one number per
    row`` from ``baseline`` to ``x`` (trapezoidal rule). They add up to about
    score(x) - score(baseline)."""
    alphas = torch.linspace(0, 1, steps + 1)[:, None]
    path = (baseline + alphas * (x - baseline)).requires_grad_(True)
    (gradient,) = torch.autograd.grad(score(path).sum(), path)
    average = (gradient[:-1] + gradient[1:]).mean(dim=0) / 2
    return ((x - baseline) * average).detach().numpy().ravel()


# --------------------------------------------------------------------------- #
# Inside the network: activations and weights
# --------------------------------------------------------------------------- #


def _stats(values: np.ndarray) -> dict[str, Any]:
    flat = values.ravel()
    return {
        "mean": render.sig(flat.mean()),
        "std": render.sig(flat.std()),
        "min": render.sig(flat.min()),
        "max": render.sig(flat.max()),
        "zeros": render.sig((flat == 0).mean(), 3),
    }


def activation_view(values: np.ndarray, image_like: bool) -> dict[str, Any]:
    """How to show the output of a brick for one sample: feature maps as a grid of small
    images, a sequence or matrix as a heatmap, a vector as its values."""
    if values.ndim == 3:
        channels = values.shape[0]
        shown = values[:MAX_CHANNELS]
        maps = [render.shrink(render.stretch(m), MAX_TILE) for m in shown]
        columns = min(len(maps), 8 if len(maps) > 4 else len(maps))
        return {
            "kind": "maps",
            "png": render.data_url(render.tiles(maps, columns, pad=1, background=40)),
            "channels": channels,
            "shown": len(maps),
            "columns": columns,
            "tile": list(maps[0].shape[:2]),
            "channel_means": render.sig_list(shown.mean(axis=(1, 2)), 3),
        }
    if values.ndim == 2:
        rows, cols = values.shape
        shown = values[: MAX_MATRIX[0], : MAX_MATRIX[1]]
        heat = render.heat_rgba(render.normalise(shown - shown.min(), 100), alpha=False)
        return {
            "kind": "image" if image_like else "matrix",
            "png": render.data_url(heat),
            "rows": rows,
            "cols": cols,
            "shown": list(shown.shape),
        }
    flat = values.ravel()
    if flat.size == 1:
        return {"kind": "scalar", "value": render.sig(flat[0])}
    return {
        "kind": "vector",
        "values": render.sig_list(flat[:MAX_VECTOR], 3),
        "total": int(flat.size),
    }


def activations(
    model: torch.nn.Module,
    analysis: Analysis,
    inputs: torch.Tensor,
    only: str | None = None,
) -> list[dict[str, Any]]:
    with torch.no_grad():
        forward(model, inputs)
    nodes = []
    for nid in analysis.order:
        if only is not None and nid != only:
            continue
        value = model._brick_outputs.get(nid)
        if not isinstance(value, torch.Tensor):
            continue
        array = value[0].detach().double().numpy()
        info = analysis.nodes[nid]
        image_like = info.type == "Input" and array.ndim == 2
        nodes.append(
            {
                "id": nid,
                "type": info.type,
                "name": node_label(analysis, nid),
                "shape": list(array.shape),
                "stats": _stats(array),
                "histogram": render.histogram(array, HISTOGRAM_BINS),
                "view": activation_view(array, image_like),
            }
        )
    return nodes


def first_conv_filters(model: torch.nn.Module, analysis: Analysis, attrs: dict[str, str]) -> Any:
    """The learned filters of the first Conv2d brick as a grid of small pictures: red for
    positive and blue for negative weights (one input channel), or as colours (three)."""
    nid = next((n for n in analysis.order if analysis.nodes[n].type == "Conv2d"), None)
    if nid is None or nid not in attrs:
        return None
    weight = getattr(model, attrs[nid]).weight.detach().double().numpy()
    out_channels, in_channels, kh, kw = weight.shape
    shown = weight[:MAX_FILTERS]
    if in_channels == 3:
        pictures = []
        for w in shown:
            low, high = w.min(), w.max()
            unit = (w - low) / (high - low) if high - low > 1e-12 else np.full_like(w, 0.5)
            pictures.append(render.unit_to_uint8(unit))
        rgb = True
    else:
        scale = float(np.abs(shown[:, 0]).max()) or 1.0
        pictures = [render.diverging_rgb(w[0] / scale) for w in shown]
        rgb = False
    columns = min(len(pictures), 8)
    return {
        "node": nid,
        "name": node_label(analysis, nid),
        "shape": [out_channels, in_channels, kh, kw],
        "count": len(pictures),
        "columns": columns,
        "rgb": rgb,
        "png": render.data_url(render.tiles(pictures, columns, pad=1, background=128)),
        "note": None
        if in_channels in (1, 3)
        else f"Only the first of the {in_channels} input channels of each filter is shown.",
    }


def weight_histograms(
    model: torch.nn.Module, analysis: Analysis, attrs: dict[str, str]
) -> list[dict[str, Any]]:
    layers = []
    for nid in analysis.order:
        if nid not in attrs:
            continue
        module = getattr(model, attrs[nid])
        params = []
        for name, parameter in module.named_parameters():
            values = parameter.detach().double().numpy()
            params.append(
                {
                    "name": name,
                    "shape": list(values.shape),
                    "histogram": render.histogram(values, HISTOGRAM_BINS),
                    "norm": render.sig(np.sqrt((values**2).sum())),
                }
            )
        if params:
            layers.append(
                {
                    "id": nid,
                    "type": analysis.nodes[nid].type,
                    "name": node_label(analysis, nid),
                    "params": params,
                }
            )
    return layers


def module_attributes(analysis: Analysis) -> dict[str, str]:
    """node id -> attribute of the generated class (bricks with weights or a module)."""
    return dict(_Generator(analysis, GeneratorOptions(include_main=False)).attr)
