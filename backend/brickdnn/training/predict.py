"""Run a trained network on a few inputs: ``python -m brickdnn.training.predict RUN_DIR``.

The web server never imports PyTorch, so predictions run in their own short-lived process,
like a training. It builds the network from ``RUN_DIR/job.json`` (the same code as the
export), loads ``RUN_DIR/weights.pt`` and reads the inputs as JSON from stdin, already
prepared like the training data (see :func:`.inference.prepare_inputs`)::

    {"inputs": [{"dtype": "float", "shape": [2, 4], "values": [8 numbers, row-major]}]}

It prints one line of JSON: the raw outputs of the network, one nested list per output
(NaN and infinity become null), or an error for the user::

    {"outputs": [[[0.1, 2.3, -1.0], [1.2, -0.3, 0.4]]]}
    {"error": "The trained weights do not fit the network of this run: ..."}

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

import contextlib
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
from .state import CLASSICAL_WEIGHTS_FILE, WEIGHTS_FILE, finite, read_job

# Outputs larger than this (numbers of all samples together) are not returned.
MAX_OUTPUT_VALUES = 250_000
# CPU threads: a prediction is small and must not take the CPU from the web server.
THREADS = 2


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
    if sum(o.numel() for o in outputs) > MAX_OUTPUT_VALUES:
        raise PredictionError(
            f"The outputs are too large to return (more than {MAX_OUTPUT_VALUES:,} numbers). "
            "Predict fewer samples at a time."
        )
    return {"outputs": [finite(o.detach().float().cpu().tolist()) for o in outputs]}


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
    print(json.dumps(result, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
