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

A connection carries what its source puts out: a tensor, or a kernel (the kernel bricks of
Gaussian processes). A *classical* network (scikit-learn) consists of an Input layer, one
classical model (Gaussian process, SVM or gradient boosting) that takes the features straight
from it and, for a Gaussian process, optionally kernel bricks that feed its kernel. Classical
models and neural layers cannot be mixed (yet).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any

from .layers import (
    SVM_KERNELS,
    ClassicalModel,
    Kernel,
    KernelExpr,
    Layer,
    LayerError,
    Shape,
    fmt_shape,
    get_layer_class,
)

KERNEL_TARGETS = (
    "Kernel bricks only connect to a Gaussian process (GaussianProcessRegressor or "
    "GaussianProcessClassifier) or to KernelSum and KernelProduct."
)
NOT_MIXED = "Classical models and neural layers can't be mixed yet"

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
    # Kernel bricks: the kernel built up to here; Gaussian processes: the kernel they fit.
    kernel: KernelExpr | None = None
    # True once the node was inferred without errors (its shape or kernel is known).
    resolved: bool = False

    @property
    def output_kind(self) -> str | None:
        return self.layer.output_kind if self.layer is not None else None

    @property
    def num_params(self) -> int | None:
        if self.layer is None or not self.resolved:
            return None
        return self.layer.num_params()

    def to_dict(self) -> dict[str, Any]:
        params = None
        if self.layer is not None:
            params = {
                k: list(v) if isinstance(v, tuple) else v
                for k, v in self.layer.p.items()
                if self.layer.param_visible(k)
            }
        return {
            "type": self.type,
            "output_shape": list(self.shape) if self.shape is not None else None,
            "dtype": self.dtype,
            "num_params": self.num_params,
            "params": params,
            "output_kind": self.output_kind,
            "kernel": self.kernel.code() if self.kernel is not None else None,
            "task": self.layer.task if isinstance(self.layer, ClassicalModel) else None,
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

    def _ordered(self) -> list[str]:
        """Topological order; document order if parsing failed before the graph was sorted."""
        if self.order:
            return self.order
        return sorted(self.nodes, key=lambda nid: self.nodes[nid].index)

    @property
    def inputs(self) -> list[str]:
        return [nid for nid in self._ordered() if self.nodes[nid].type == "Input"]

    @property
    def outputs(self) -> list[str]:
        return [nid for nid in self._ordered() if not self.nodes[nid].outputs]

    @property
    def total_params(self) -> int | None:
        counts = [info.num_params for info in self.nodes.values()]
        if not counts or any(c is None for c in counts):
            return None
        return sum(counts)  # type: ignore[arg-type]

    @property
    def family(self) -> str:
        """``classical`` if the graph contains kernels or a classical model, else ``neural``."""
        classical = any(
            info.layer is not None and info.layer.family == "classical"
            for info in self.nodes.values()
        )
        return "classical" if classical else "neural"

    @property
    def model(self) -> NodeInfo | None:
        """The classical model of the graph (the first one if there are several)."""
        for nid in self._ordered():
            if isinstance(self.nodes[nid].layer, ClassicalModel):
                return self.nodes[nid]
        return None

    def classical_dict(self) -> dict[str, Any] | None:
        model = self.model
        if model is None:
            return None
        layer = model.layer
        assert isinstance(layer, ClassicalModel)
        return {
            "node": model.id,
            "type": model.type,
            "estimator": layer.estimator,
            "task": layer.task,
            "kernel": model.kernel.code() if model.kernel is not None else None,
        }

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
            # Also for invalid graphs: whatever could be resolved.
            "inputs": self.inputs,
            "outputs": self.outputs,
            "total_params": self.total_params if self.valid else None,
            "family": self.family,
            # Classical networks: {"node", "type", "estimator", "task", "kernel"}
            "classical": self.classical_dict(),
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
    _check_classical(result)
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
        sources = [result.nodes[s] for s in info.inputs]
        if layer.family == "classical" or any(s.output_kind == "kernel" for s in sources):
            _infer_classical(result, info, layer, sources)
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

        if any(src.shape is None for src in sources):
            continue  # an upstream layer already reported an error (or is a classifier)

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
        info.resolved = True


def _infer_classical(
    result: Analysis, info: NodeInfo, layer: Layer, sources: list[NodeInfo]
) -> None:
    """Kernel bricks, classical models, and any brick that is fed a kernel."""
    nid = info.id
    kernels = [s for s in sources if s.output_kind == "kernel"]
    tensors = [s for s in sources if s.output_kind == "tensor"]

    if isinstance(layer, Kernel):
        if layer.max_inputs == 0:
            if sources:
                result.error(
                    f"{layer.label} has no inputs: a kernel is not fed with data. Connect it "
                    "to a Gaussian process (or to KernelSum or KernelProduct) instead.",
                    nid,
                )
                return
        elif tensors:
            other = tensors[0].layer.label if tensors[0].layer else tensors[0].type
            result.error(
                f"{layer.label} combines kernels: connect kernel bricks such as RBF or "
                f"WhiteKernel to it, not {other}.",
                nid,
            )
            return
        elif len(kernels) < layer.min_inputs:
            result.error(f"{layer.label} needs at least {layer.min_inputs} kernels.", nid)
            return
        if any(not s.resolved for s in kernels):
            return  # an upstream kernel already reported an error
        info.kernel = layer.kernel([s.kernel for s in kernels])  # type: ignore[misc]
        info.resolved = True
        return

    if not isinstance(layer, ClassicalModel):  # an Input or neural layer fed a kernel
        if layer.max_inputs == 0:
            result.error("Input layers cannot have incoming connections.", nid)
        else:
            result.error(f"{layer.label} cannot take a kernel. {KERNEL_TARGETS}", nid)
        return

    if kernels and not layer.takes_kernel:
        if layer.scale == "svm":
            text = (
                f"{layer.label} chooses its kernel with its own Kernel setting "
                f"({', '.join(SVM_KERNELS)})."
            )
        else:
            text = f"{layer.label} builds decision trees and takes no kernel."
        result.error(f"{text} {KERNEL_TARGETS}", nid)
        return
    if len(kernels) > 1:
        result.error(
            "A Gaussian process takes one kernel. Combine several with KernelSum or KernelProduct.",
            nid,
        )
        return
    if not tensors:
        result.error(f"{layer.label} has no features: connect the Input layer to it.", nid)
        return
    if len(tensors) > 1:
        result.error(f"{layer.label} takes its features from one Input layer.", nid)
        return
    source = tensors[0]
    if source.type != "Input":
        result.error(
            f"{layer.label} must take its features straight from the Input layer. "
            f"{NOT_MIXED}: remove the layers between the Input and {layer.label}.",
            nid,
        )
        return
    if not source.resolved or any(not k.resolved for k in kernels):
        return
    assert source.shape is not None
    if len(source.shape) != 1:
        result.error(
            f"{layer.label} needs a flat feature vector: set the Input shape to the number of "
            f"features, e.g. 8 (it is {fmt_shape(source.shape)}).",
            nid,
        )
        return
    if source.dtype != "float":
        result.error(
            f"{layer.label} needs numeric features: set the data type of the Input layer to "
            "'float'.",
            nid,
        )
        return
    info.shape = layer.setup(kernels[0].kernel if kernels else None)
    info.kernel = layer.kernel
    info.dtype = "float" if info.shape is not None else None
    info.resolved = True


def _check_classical(result: Analysis) -> None:
    """Rules for the network as a whole: every kernel is used exactly once, one classical
    model at most, nothing after it and no neural layers besides it."""
    models: list[NodeInfo] = []
    neural: list[NodeInfo] = []
    for nid in result.order:
        info = result.nodes[nid]
        if isinstance(info.layer, Kernel):
            if not info.outputs:
                result.error("This kernel is not connected to a Gaussian process.", nid)
            elif len(info.outputs) > 1:
                result.error(
                    "A kernel brick can only be connected once. Add another kernel brick for "
                    "each further use.",
                    nid,
                )
        elif isinstance(info.layer, ClassicalModel):
            models.append(info)
        elif info.layer is not None and info.layer.family == "neural":
            neural.append(info)

    for extra in models[1:]:
        result.error(
            "Only one classical model per network. Remove this one or build it in a network "
            "of its own.",
            extra.id,
        )
    if not models:
        return
    model = models[0]
    label = model.layer.label  # type: ignore[union-attr]
    if model.outputs:
        result.error(
            f"Nothing can follow {label}: a classical model ends the network, its output is "
            f"the prediction. {NOT_MIXED}.",
            model.id,
        )
    elif neural and not any(issue.node == model.id for issue in result.errors):
        names = sorted({n.layer.label for n in neural if n.layer is not None})
        shown = ", ".join(names[:3]) + (" …" if len(names) > 3 else "")
        result.error(
            f"{NOT_MIXED}, but this network also contains {shown}. Remove the neural layers: "
            f"{label} takes its features straight from the Input layer.",
            model.id,
        )


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
