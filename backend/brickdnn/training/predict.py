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
"""

from __future__ import annotations

import json
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Any

from .reporting import short_error
from .runner import lower_priority
from .state import WEIGHTS_FILE, finite, read_job

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
    try:
        import torch

        from .engine import build_model
    except ImportError as exc:
        raise PredictionError(
            "PyTorch is not installed on the server. Install it with: pip install torch"
        ) from exc
    torch.set_num_threads(max(1, min(THREADS, os.cpu_count() or 1)))

    job = read_job(run_dir)
    if job is None or not isinstance(job.get("graph"), dict):
        raise PredictionError("The network of this run is missing.")
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
