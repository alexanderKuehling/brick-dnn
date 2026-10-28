"""Report what this machine can train on: ``python -m brickdnn.training.probe``.

Prints one JSON object, e.g.::

    {"torch_version": "2.2.2", "torchvision": true, "torchvision_version": "0.17.2",
     "devices": [{"id": "cpu", "label": "CPU"}, {"id": "mps", "label": "Apple GPU (MPS)"}],
     "default_device": "mps"}

Exits with 1 (and the error on stderr) if PyTorch cannot be imported.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from typing import Any


def _torchvision_version() -> str | None:
    """The torchvision version, or None if it is missing or cannot be imported (e.g. a
    build that does not match the installed PyTorch); the reason goes to stderr."""
    if importlib.util.find_spec("torchvision") is None:
        return None
    try:
        import torchvision
    except Exception as exc:
        print(f"torchvision cannot be imported: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None
    return str(torchvision.__version__)


def probe() -> dict[str, Any]:
    import torch

    from .devices import available_devices, default_device

    vision = _torchvision_version()
    return {
        "torch_version": str(torch.__version__),
        "torchvision": vision is not None,
        "torchvision_version": vision,
        "devices": available_devices(),
        "default_device": default_device(),
    }


def main() -> int:
    try:
        result = probe()
    except Exception as exc:
        print(f"PyTorch is not usable: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
