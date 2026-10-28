"""The public code generation endpoints as plain functions: JSON-like data in and out.

The Django API (``brick_dnn_core.views``) and the online demo, which runs this package in
the browser with Pyodide, both use these functions, so they always answer the same way.
"""

from __future__ import annotations

import re
from typing import Any

from .codegen import GeneratorOptions, generate_code
from .examples import EXAMPLES
from .graph import GraphValidationError, analyze
from .layers import CATEGORIES, layer_specs


def graph_from(data: Any) -> Any:
    """Requests send ``{"graph": {...}}``; a bare graph is accepted as well."""
    if isinstance(data, dict) and "graph" in data:
        return data["graph"]
    return data


def layers() -> dict[str, Any]:
    """All layer types with their parameters: the UI builds its palette and forms from this."""
    return {"categories": list(CATEGORIES), "layers": layer_specs()}


def examples() -> list[dict[str, Any]]:
    return list(EXAMPLES.values())


def analyze_graph(data: Any) -> dict[str, Any]:
    """Validate a graph and return the inferred shape and parameter count of every layer."""
    return analyze(graph_from(data)).to_dict()


def safe_filename(raw: Any) -> str:
    stem = str(raw or "model").strip().removesuffix(".py")
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")[:60] or "model"
    return stem + ".py"


def generate(data: Any) -> tuple[int, dict[str, Any]]:
    """Generate a PyTorch script. Returns an HTTP status code and the response body.

    Body: ``{"graph": {...}, "options": {"class_name": "...", "include_main": true,
    "include_training": true, "filename": "model"}}``.
    """
    graph = graph_from(data)
    raw_options = data.get("options") if isinstance(data, dict) else None
    if raw_options is not None and not isinstance(raw_options, dict):
        return 400, {"detail": "'options' must be an object."}
    try:
        options = GeneratorOptions.from_dict(raw_options)
        code = generate_code(graph, options)
    except GraphValidationError as exc:
        return 400, {"detail": str(exc), "analysis": exc.analysis.to_dict()}
    except ValueError as exc:
        return 400, {"detail": str(exc)}
    return 200, {"code": code, "filename": safe_filename((raw_options or {}).get("filename"))}
