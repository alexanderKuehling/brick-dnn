"""Run a trained network on a few inputs: ``python -m brickdnn.training.predict RUN_DIR``.

The web server never imports PyTorch, so predictions run in their own short-lived process,
like a training. It builds the network from ``RUN_DIR/job.json`` (the same code as the
export), loads ``RUN_DIR/weights.pt`` and reads the inputs as JSON from stdin, already
prepared like the training data (see :func:`.inference.prepare_inputs`)::

    {"inputs": [{"dtype": "float", "shape": [2, 4], "values": [8 numbers, row-major]}]}

It prints one line of JSON: the raw outputs of the network, one nested list per output
(NaN and infinity become null), or an error for the user::

    {"outputs": [[[0.1, 2.3, -1.0], [1.2, -0.3, 0.4]]], "cpu_seconds": 1.9}
    {"error": "The trained weights do not fit the network of this run: ...", "cpu_seconds": 1.2}

``cpu_seconds`` is the CPU time the process used (for the server's usage records).

Image networks also take image files instead of prepared tensors: ``"images"`` (base64
text of PNG, JPEG … files) with ``"image"`` (how the training prepared its images:
``{"channels", "height", "width", "layout", "mean", "std"}``); they are decoded, converted,
resized and normalised here. With ``"segmentation": {"binary": bool, "num_classes": K,
"mask": "png" | "rle" | "none"}`` the answer is not the raw scores (K numbers per pixel) but
one class map per sample: the pixels per class, and the map as a PNG file (base64, one
colour per class) or as run-lengths (class, pixels, class, pixels … row by row)::

    {"masks": [{"width": 64, "height": 64, "counts": [3900, 196], "png": "iVBOR…"}]}

Classical models (scikit-learn) load ``RUN_DIR/model.joblib`` instead and need NumPy and
scikit-learn, not PyTorch. Their request also says how the inputs were prepared
(``"kind": "tabular"``: standardised like the training data, so the pipeline's scaler is
skipped; otherwise they are used as they are). Classifiers answer with class
probabilities (one column per class of the training plan), a Gaussian process regressor
adds its predictive standard deviation::

    {"outputs": [[[0.9, 0.1, 0.0], [0.2, 0.7, 0.1]]], "probabilities": true}
    {"outputs": [[[3.2], [2.9]]], "std": [0.4, 0.5]}
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import io
import json
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Any

from .job import graph_family
from .reporting import short_error
from .runner import lower_priority
from .state import CLASSICAL_WEIGHTS_FILE, WEIGHTS_FILE, cpu_seconds, finite, read_job

# Outputs larger than this (numbers of all samples together) are not returned.
MAX_OUTPUT_VALUES = 250_000
# CPU threads: a prediction is small and must not take the CPU from the web server.
THREADS = 2
# Image files larger than this (pixels per side) are refused before they are decoded.
MAX_IMAGE_SIDE = 4096


class PredictionError(Exception):
    """The message is shown to the user."""


def _tensor(spec: Any, torch: Any) -> Any:
    if not isinstance(spec, dict):
        raise PredictionError("The inputs are invalid.")
    shape, values = spec.get("shape"), spec.get("values")
    if not isinstance(shape, list) or not isinstance(values, list):
        raise PredictionError("The inputs are invalid.")
    if math.prod(shape) != len(values):
        raise PredictionError("The inputs do not have the shape they claim.")
    dtype = torch.long if spec.get("dtype") == "long" else torch.float32
    return torch.tensor(values, dtype=dtype).view(*shape)


def predict(run_dir: str | Path, request: Any) -> dict[str, Any]:
    """The outputs of the trained network of ``run_dir`` for ``request["inputs"]``.

    Raises :class:`PredictionError`.
    """
    run_dir = Path(run_dir)
    job = read_job(run_dir)
    if job is None or not isinstance(job.get("graph"), dict):
        raise PredictionError("The network of this run is missing.")
    if graph_family(job["graph"]) == "classical":
        return _predict_classical(run_dir, request)
    try:
        import torch

        from .engine import build_model
    except ImportError as exc:
        raise PredictionError(
            "PyTorch is not installed on the server. Install it with: pip install torch"
        ) from exc
    torch.set_num_threads(max(1, min(THREADS, os.cpu_count() or 1)))

    weights = run_dir / WEIGHTS_FILE
    if not weights.is_file():
        raise PredictionError("This run has no trained weights.")
    try:
        model = build_model(job["graph"])
    except Exception as exc:
        raise PredictionError(f"The network cannot be built: {short_error(exc)}") from exc
    try:
        state = torch.load(weights, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    except Exception as exc:
        raise PredictionError(
            f"The trained weights do not fit the network of this run: {short_error(exc)}"
        ) from exc
    model.eval()

    if isinstance(request, dict) and request.get("images") is not None:
        inputs = [_image_tensor(request["images"], request.get("image"), torch)]
    else:
        specs = request.get("inputs") if isinstance(request, dict) else None
        if not isinstance(specs, list) or not specs:
            raise PredictionError("No inputs were given.")
        inputs = [_tensor(spec, torch) for spec in specs]
    try:
        with torch.no_grad():
            output = model(*inputs)
    except Exception as exc:
        raise PredictionError(
            f"The network could not process the inputs: {short_error(exc)}"
        ) from exc
    outputs = output if isinstance(output, (tuple, list)) else (output,)
    if isinstance(request.get("segmentation"), dict):
        return {"masks": _masks(outputs[0], request["segmentation"], torch)}
    if sum(o.numel() for o in outputs) > MAX_OUTPUT_VALUES:
        raise PredictionError(
            f"The outputs are too large to return (more than {MAX_OUTPUT_VALUES:,} numbers). "
            "Predict fewer samples at a time."
        )
    return {"outputs": [finite(o.detach().float().cpu().tolist()) for o in outputs]}


def _image_tensor(files: Any, image: Any, torch: Any) -> Any:
    """Image files (base64) as one input tensor, prepared like the training images: converted
    to the planned channels, resized (bilinear), scaled to 0-1 and normalised."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise PredictionError(
            "Reading image files needs Pillow on the server: pip install pillow"
        ) from exc
    if not isinstance(files, list) or not files or not isinstance(image, dict):
        raise PredictionError("The inputs are invalid.")
    channels, height, width = (int(image[key]) for key in ("channels", "height", "width"))
    mean = torch.tensor(image["mean"], dtype=torch.float32).view(-1, 1, 1)
    std = torch.tensor(image["std"], dtype=torch.float32).view(-1, 1, 1)
    samples = []
    for number, text in enumerate(files, start=1):
        label = f"Image {number}" if len(files) > 1 else "The image"
        try:
            data = base64.b64decode(text, validate=True)
        except (binascii.Error, ValueError, TypeError):
            raise PredictionError(f"{label} is not base64 text.") from None
        try:
            with Image.open(io.BytesIO(data)) as picture:
                if max(picture.size) > MAX_IMAGE_SIDE:
                    raise PredictionError(
                        f"{label} has {picture.size[0]}×{picture.size[1]} pixels; at most "
                        f"{MAX_IMAGE_SIDE}×{MAX_IMAGE_SIDE} are read. Scale it down first."
                    )
                picture = picture.convert("L" if channels == 1 else "RGB")
                if picture.size != (width, height):
                    picture = picture.resize((width, height), Image.BILINEAR)
                pixels = bytearray(picture.tobytes())
        except PredictionError:
            raise
        except Exception:
            raise PredictionError(
                f"{label} cannot be read as an image file (PNG, JPEG, BMP, GIF or WebP)."
            ) from None
        x = torch.frombuffer(pixels, dtype=torch.uint8).view(height, width, channels)
        samples.append((x.permute(2, 0, 1).float() / 255.0 - mean) / std)
    batch = torch.stack(samples)
    layout = image.get("layout")
    if layout == "hw":
        return batch[:, 0]
    return batch.flatten(1) if layout == "flat" else batch


def run_lengths(classes: Any) -> list[int]:
    """A class map (2-D tensor) as run-lengths, row by row: ``[class, pixels, class,
    pixels, …]``."""
    flat = classes.reshape(-1)
    if flat.numel() == 0:
        return []
    change = (flat[1:] != flat[:-1]).nonzero().flatten() + 1
    starts = [0, *change.tolist(), flat.numel()]
    values = flat[starts[:-1]].tolist()
    runs: list[int] = []
    for value, start, stop in zip(values, starts[:-1], starts[1:], strict=True):
        runs += [int(value), stop - start]
    return runs


def _mask_png(classes: Any, num_classes: int) -> str | None:
    """A class map as a PNG file (base64): the pixel values are the class indices, the
    palette shows every class in its colour. None without Pillow."""
    try:
        from PIL import Image
    except ImportError:
        return None
    from .segmentation import class_colors, rgb

    height, width = classes.shape
    picture = Image.frombytes("P", (width, height), classes.contiguous().numpy().tobytes())
    picture.putpalette([c for colour in class_colors(num_classes) for c in rgb(colour)])
    buffer = io.BytesIO()
    picture.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _masks(output: Any, options: dict[str, Any], torch: Any) -> list[dict[str, Any]]:
    """The predicted class map of every sample, see the module docstring."""
    from .segmetrics import predicted_classes

    if output.dim() != 4:
        raise PredictionError("The network does not output one score per pixel.")
    binary = bool(options.get("binary"))
    num_classes = int(options.get("num_classes") or (2 if binary else output.shape[1]))
    classes = predicted_classes(output.detach(), binary).to(torch.uint8).cpu()
    kind = options.get("mask")
    masks = []
    for sample in classes:
        mask: dict[str, Any] = {
            "width": int(sample.shape[1]),
            "height": int(sample.shape[0]),
            "counts": torch.bincount(sample.reshape(-1).long(), minlength=num_classes).tolist()[
                :num_classes
            ],
        }
        if kind == "png":
            mask["png"] = _mask_png(sample, num_classes)
        if kind == "rle" or (kind == "png" and mask["png"] is None):
            mask["rle"] = run_lengths(sample)
        masks.append(mask)
    return masks


def _predict_classical(run_dir: Path, request: Any) -> dict[str, Any]:
    """Predict with the fitted scikit-learn pipeline of a classical model."""
    try:
        import joblib
        import numpy as np
    except ImportError as exc:
        raise PredictionError(
            "scikit-learn is not installed on the server. Install it with: pip install scikit-learn"
        ) from exc
    path = run_dir / CLASSICAL_WEIGHTS_FILE
    if not path.is_file():
        raise PredictionError("This run has no fitted model.")
    try:
        bundle = joblib.load(path)  # written by the training of this run
        pipeline = bundle["model"]
    except Exception as exc:
        raise PredictionError(f"The fitted model cannot be loaded: {short_error(exc)}") from exc

    specs = request.get("inputs") if isinstance(request, dict) else None
    if not isinstance(specs, list) or len(specs) != 1 or not isinstance(specs[0], dict):
        raise PredictionError("A classical model takes exactly one input.")
    shape, values = specs[0].get("shape"), specs[0].get("values")
    if not isinstance(shape, list) or not isinstance(values, list) or len(shape) != 2:
        raise PredictionError("The inputs are invalid.")
    if math.prod(shape) != len(values):
        raise PredictionError("The inputs do not have the shape they claim.")
    x = np.asarray(values, dtype=np.float64).reshape(shape)
    # Standardised inputs skip the pipeline's StandardScaler, which would do the same.
    model = pipeline[1:] if request.get("kind") == "tabular" else pipeline
    try:
        with _thread_limit():
            if bundle.get("task") == "classification":
                proba = model.predict_proba(x)
                indices = bundle.get("class_indices") or list(range(proba.shape[1]))
                full = np.zeros((len(x), int(bundle.get("num_classes") or proba.shape[1])))
                full[:, indices] = proba
                return {"outputs": [finite(full.tolist())], "probabilities": True}
            if type(pipeline[-1]).__name__ == "GaussianProcessRegressor":
                mean, std = model.predict(x, return_std=True)
                return {
                    "outputs": [finite(np.asarray(mean).reshape(-1, 1).tolist())],
                    "std": finite(np.asarray(std).reshape(-1).tolist()),
                }
            predicted = model.predict(x)
    except Exception as exc:
        raise PredictionError(
            f"The model could not process the inputs: {short_error(exc)}"
        ) from exc
    return {"outputs": [finite(np.asarray(predicted).reshape(-1, 1).tolist())]}


@contextlib.contextmanager
def _thread_limit():
    """At most THREADS threads for NumPy's linear algebra (like for PyTorch)."""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:  # pragma: no cover - comes with scikit-learn
        yield
        return
    with threadpool_limits(limits=max(1, min(THREADS, os.cpu_count() or 1))):
        yield


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m brickdnn.training.predict RUN_DIR < request.json", file=sys.stderr)
        return 2
    lower_priority()
    try:
        result = predict(args[0], json.load(sys.stdin))
    except PredictionError as exc:
        result = {"error": str(exc)}
    except Exception as exc:
        traceback.print_exc()
        result = {"error": f"The prediction failed: {short_error(exc)}"}
    # The CPU time of this process (all threads), for the server's usage records.
    result["cpu_seconds"] = round(cpu_seconds(), 3)
    print(json.dumps(result, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
