from __future__ import annotations

from typing import Any


def chain(*layers: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    """Build a sequential graph: chain(("Input", {...}), ("Linear", {...}), ...)."""
    nodes = [
        {"id": f"n{i}", "type": layer_type, "params": params}
        for i, (layer_type, params) in enumerate(layers)
    ]
    edges = [{"source": f"n{i}", "target": f"n{i + 1}"} for i in range(len(layers) - 1)]
    return {"nodes": nodes, "edges": edges}


def build_model(code: str, class_name: str = "NeuralNetwork"):
    namespace: dict[str, Any] = {"__name__": "generated_model"}
    exec(compile(code, "<generated>", "exec"), namespace)
    return namespace[class_name]()
