"""Graph parsing, validation and shape inference.

A graph is plain JSON::

    {
      "nodes": [
        {"id": "in", "type": "Input", "params": {"shape": [784]}},
        {"id": "fc", "type": "Linear", "params": {"out_features": 10}}
      ],
      "edges": [{"source": "in", "target": "fc"}]
    }

Extra keys (``position``, ``data`` …) are ignored, so the UI can store its own
layout information in the same document. A node may carry an optional
``name`` that becomes the attribute name in the generated code.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any

from .layers import Layer, LayerError, Shape, get_layer_class

MAX_NODES = 500
MAX_EDGES = 2000


@dataclass
class Issue:
    message: str
    node: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"node": self.node, "message": self.message}


@dataclass
class NodeInfo:
    id: str
    type: str
    name: str | None
    index: int
    layer: Layer | None = None
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    shape: Shape | None = None
    dtype: str | None = None

    @property
    def num_params(self) -> int | None:
        if self.layer is None or self.shape is None:
            return None
        return self.layer.num_params()

    def to_dict(self) -> dict[str, Any]:
        params = None
        if self.layer is not None:
            params = {k: list(v) if isinstance(v, tuple) else v for k, v in self.layer.p.items()}
        return {
            "type": self.type,
            "output_shape": list(self.shape) if self.shape is not None else None,
            "dtype": self.dtype,
            "num_params": self.num_params,
            "params": params,
        }


@dataclass
class Analysis:
    nodes: dict[str, NodeInfo] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    errors: list[Issue] = field(default_factory=list)
    warnings: list[Issue] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors

    @property
    def inputs(self) -> list[str]:
        return [nid for nid in self.order if self.nodes[nid].type == "Input"]

    @property
    def outputs(self) -> list[str]:
        return [nid for nid in self.order if not self.nodes[nid].outputs]

    @property
    def total_params(self) -> int | None:
        counts = [info.num_params for info in self.nodes.values()]
        if not counts or any(c is None for c in counts):
            return None
        return sum(counts)

    def error(self, message: str, node: str | None = None) -> None:
        self.errors.append(Issue(message, node))

    def warn(self, message: str, node: str | None = None) -> None:
        self.warnings.append(Issue(message, node))

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
            "nodes": {nid: info.to_dict() for nid, info in self.nodes.items()},
            "inputs": self.inputs if self.valid else [],
            "outputs": self.outputs if self.valid else [],
            "total_params": self.total_params if self.valid else None,
        }


class GraphValidationError(ValueError):
    """Raised by the code generator when the graph has errors."""

    def __init__(self, analysis: Analysis):
        self.analysis = analysis
        first = analysis.errors[0].message if analysis.errors else "invalid graph"
        super().__init__(first)


def analyze(graph: Any) -> Analysis:
    """Validate ``graph`` and infer the output shape of every node."""
    result = Analysis()
    if isinstance(graph, dict) and isinstance(graph.get("graph"), dict):
        graph = graph["graph"]  # accept a saved project file as well
    if not isinstance(graph, dict):
        result.error("The graph must be a JSON object with 'nodes' and 'edges'.")
        return result

    raw_nodes = graph.get("nodes", [])
    raw_edges = graph.get("edges", [])
    if not isinstance(raw_nodes, list) or not isinstance(raw_edges, list):
        result.error("'nodes' and 'edges' must be lists.")
        return result
    if len(raw_nodes) > MAX_NODES or len(raw_edges) > MAX_EDGES:
        result.error(f"Graphs are limited to {MAX_NODES} layers and {MAX_EDGES} connections.")
        return result
    if not raw_nodes:
        result.error("The canvas is empty. Add an Input layer to get started.")
        return result

    _parse_nodes(raw_nodes, result)
    _parse_edges(raw_edges, result)
    if result.errors:
        return result

    _topological_sort(result)
    _infer_shapes(result)
    _check_structure(result)
    return result


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def _parse_nodes(raw_nodes: list[Any], result: Analysis) -> None:
    for index, raw in enumerate(raw_nodes):
        if not isinstance(raw, dict) or "id" not in raw:
            result.error(f"Node #{index + 1} is malformed (missing 'id').")
            continue
        node_id = str(raw["id"])
        if node_id in result.nodes:
            result.error(f"Duplicate node id '{node_id}'.", node_id)
            continue
        layer_type = raw.get("type")
        name = raw.get("name")
        info = NodeInfo(
            id=node_id,
            type=str(layer_type),
            name=str(name) if isinstance(name, str) and name.strip() else None,
            index=index,
        )
        result.nodes[node_id] = info
        try:
            info.layer = get_layer_class(str(layer_type))(raw.get("params"))
        except LayerError as exc:
            result.error(str(exc), node_id)


def _parse_edges(raw_edges: list[Any], result: Analysis) -> None:
    seen: set[tuple[str, str]] = set()
    for index, raw in enumerate(raw_edges):
        if not isinstance(raw, dict) or "source" not in raw or "target" not in raw:
            result.error(f"Connection #{index + 1} is malformed (needs 'source' and 'target').")
            continue
        source, target = str(raw["source"]), str(raw["target"])
        if source not in result.nodes or target not in result.nodes:
            result.error(f"Connection #{index + 1} refers to a layer that does not exist.")
            continue
        if source == target:
            result.error("A layer cannot be connected to itself.", source)
            continue
        if (source, target) in seen:
            continue
        seen.add((source, target))
        result.nodes[source].outputs.append(target)
        result.nodes[target].inputs.append(source)


# --------------------------------------------------------------------------- #
# Analysis passes
# --------------------------------------------------------------------------- #


def _topological_sort(result: Analysis) -> None:
    """Kahn's algorithm; ties are broken by the order of the nodes in the document."""
    in_degree = {nid: len(info.inputs) for nid, info in result.nodes.items()}
    ready = [(info.index, nid) for nid, info in result.nodes.items() if in_degree[nid] == 0]
    heapq.heapify(ready)
    while ready:
        _, nid = heapq.heappop(ready)
        result.order.append(nid)
        for target in result.nodes[nid].outputs:
            in_degree[target] -= 1
            if in_degree[target] == 0:
                heapq.heappush(ready, (result.nodes[target].index, target))

    if len(result.order) != len(result.nodes):
        for nid in sorted(result.nodes, key=lambda n: result.nodes[n].index):
            if in_degree[nid] > 0:
                result.error("This layer is part of a cycle. Networks must be acyclic.", nid)
                result.order.append(nid)


def _infer_shapes(result: Analysis) -> None:
    for nid in result.order:
        info = result.nodes[nid]
        layer = info.layer
        if layer is None:
            continue

        n_inputs = len(info.inputs)
        if n_inputs < layer.min_inputs:
            if layer.min_inputs == 1:
                result.error(f"{layer.label} has no incoming connection.", nid)
            else:
                result.error(
                    f"{layer.label} needs at least {layer.min_inputs} incoming connections.", nid
                )
            continue
        if layer.max_inputs is not None and n_inputs > layer.max_inputs:
            if layer.max_inputs == 0:
                result.error("Input layers cannot have incoming connections.", nid)
            else:
                result.error(
                    f"{layer.label} accepts only one incoming connection. "
                    "Use Add or Concat to merge branches.",
                    nid,
                )
            continue

        sources = [result.nodes[s] for s in info.inputs]
        if any(src.shape is None for src in sources):
            continue  # an upstream layer already reported an error

        bad = [src for src in sources if src.dtype not in layer.input_dtypes]
        if bad:
            if "long" in layer.input_dtypes:
                result.error(
                    f"{layer.label} expects integer token ids. "
                    "Set the data type of the Input layer to 'long'.",
                    nid,
                )
            else:
                result.error(
                    f"{layer.label} expects floating point input, but receives integer "
                    "token ids. Add an Embedding layer first.",
                    nid,
                )
            continue

        try:
            info.shape = layer.infer([src.shape for src in sources])  # type: ignore[misc]
        except LayerError as exc:
            result.error(str(exc), nid)
            continue
        info.dtype = layer.output_dtype


def _check_structure(result: Analysis) -> None:
    if not any(info.type == "Input" for info in result.nodes.values()):
        result.error("Add an Input layer to define the shape of your data.")

    for nid in result.order:
        info = result.nodes[nid]
        if info.type == "Input" and not info.outputs:
            result.error("This Input is not connected to any layer.", nid)

    if result.errors:
        return

    for nid in result.outputs:
        layer = result.nodes[nid].layer
        if layer is not None and layer.type == "Activation" and layer.p["function"] == "softmax":
            result.warn(
                "The network ends with Softmax. If you train with nn.CrossEntropyLoss, "
                "remove it: the loss already applies log-softmax to raw logits.",
                nid,
            )
        if layer is not None and layer.type == "Dropout":
            result.warn("The network ends with Dropout, which is unusual.", nid)
