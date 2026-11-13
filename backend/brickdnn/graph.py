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

A connection carries what its source puts out: a tensor, a kernel (the kernel bricks of
Gaussian processes) or an estimator (a classical model connected to an ensemble). A
*classical* network (scikit-learn) consists of an Input layer, one classical model (Gaussian
process, SVM or gradient boosting) that takes the features straight from it and, for a
Gaussian process, optionally kernel bricks that feed its kernel. Or of several such models,
all fed by the same Input, that are connected to one ensemble (voting or stacking), which
then ends the network.

A *hybrid* network puts neural layers between the Input and the classical model: they turn
the input into the features that the model is fitted on. There is exactly one *boundary*:
the last neural layer, whose flat output (shape ``[n]``) feeds the model (or every model of
the ensemble). Every neural layer leads to it, and nothing neural follows a classical model.
"""

from __future__ import annotations

import heapq
import keyword
import re
from dataclasses import dataclass, field
from typing import Any

from .layers import (
    ENSEMBLE_ARGUMENTS,
    SVM_KERNELS,
    ClassicalModel,
    Ensemble,
    Kernel,
    KernelExpr,
    Layer,
    LayerError,
    Member,
    Shape,
    fmt_shape,
    get_layer_class,
)

KERNEL_TARGETS = (
    "Kernel bricks only connect to a Gaussian process (GaussianProcessRegressor or "
    "GaussianProcessClassifier) or to KernelSum and KernelProduct."
)
FEATURES_ROLE = (
    "In a network with a classical model, the neural layers turn the input into its features"
)
ENSEMBLE_TYPES = "VotingClassifier, VotingRegressor, StackingClassifier or StackingRegressor"
# Names that the models of an ensemble do not get (variables of the generated make_model()).
RESERVED_NAMES = frozenset({"kernel", "model", "make_model", "pipeline"}) | ENSEMBLE_ARGUMENTS

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
        result = {
            "type": self.type,
            "output_shape": list(self.shape) if self.shape is not None else None,
            "dtype": self.dtype,
            "num_params": self.num_params,
            "params": params,
            "output_kind": self.output_kind,
            "kernel": self.kernel.code() if self.kernel is not None else None,
            "task": self.layer.task if isinstance(self.layer, ClassicalModel) else None,
        }
        if isinstance(self.layer, Ensemble):
            # The models it combines, in the order of their connections (that of the weights).
            result["members"] = [_member_dict(m) for m in self.layer.members]
        return result


def _member_dict(member: Member) -> dict[str, Any]:
    return {
        "node": member.node,
        "name": member.name,
        "type": member.layer.type,
        "estimator": member.layer.estimator,
        "label": member.layer.label,
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
        """``hybrid`` if the graph has a classical model and neural layers (they learn its
        features), ``classical`` if it contains kernels or a classical model, else
        ``neural``."""
        layers = [info.layer for info in self.nodes.values() if info.layer is not None]
        if not any(layer.family == "classical" for layer in layers):
            return "neural"
        model = any(isinstance(layer, ClassicalModel) for layer in layers)
        neural = any(layer.family == "neural" for layer in layers)
        return "hybrid" if model and neural else "classical"

    @property
    def boundary(self) -> NodeInfo | None:
        """Hybrid networks: the last neural layer, whose output is the feature vector of the
        classical model (of every model of an ensemble). None otherwise."""
        model = self.model
        if model is None or self.family != "hybrid":
            return None
        layer = model.layer
        nodes = [self.nodes[m.node] for m in layer.members] if isinstance(layer, Ensemble) else []
        if isinstance(layer, Ensemble) and not nodes:  # not resolved: its direct models
            nodes = [self.nodes[s] for s in model.inputs]
        for info in nodes or [model]:
            for source in info.inputs:
                found = self.nodes[source]
                if found.output_kind == "tensor" and found.type != "Input":
                    return found
        return None

    def hybrid_dict(self) -> dict[str, Any] | None:
        """``{"boundary", "features", "neural_params"}`` of a hybrid network, else None:
        the last neural layer, the length of the feature vector it puts out and the
        trainable parameters of the neural layers."""
        boundary = self.boundary
        if boundary is None:
            return None
        counts = [
            info.num_params
            for info in self.nodes.values()
            if info.layer is not None and info.layer.family != "classical"
        ]
        shape = boundary.shape
        return {
            "boundary": boundary.id,
            "features": shape[0] if shape is not None and len(shape) == 1 else None,
            "neural_params": None if any(c is None for c in counts) else sum(counts),  # type: ignore[arg-type]
        }

    @property
    def model(self) -> NodeInfo | None:
        """The classical model of the graph: the ensemble if there is one, otherwise the
        model (the first one of each if there are several)."""
        models = [
            self.nodes[nid]
            for nid in self._ordered()
            if isinstance(self.nodes[nid].layer, ClassicalModel)
        ]
        ensembles = [info for info in models if isinstance(info.layer, Ensemble)]
        return (ensembles or models or [None])[0]

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
            # Ensembles: the models they combine ({"node", "name", "type", "estimator",
            # "label"}), else None.
            "members": (
                [_member_dict(m) for m in layer.members] if isinstance(layer, Ensemble) else None
            ),
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
            # Classical and hybrid networks: {"node", "type", "estimator", "task", "kernel",
            # "members"}
            "classical": self.classical_dict(),
            # Hybrid networks: {"boundary", "features", "neural_params"}
            "hybrid": self.hybrid_dict(),
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
        if layer.family == "classical" or any(s.output_kind != "tensor" for s in sources):
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
    """Kernel bricks, classical models, ensembles, and any brick that is fed a kernel or an
    estimator."""
    nid = info.id
    kernels = [s for s in sources if s.output_kind == "kernel"]
    tensors = [s for s in sources if s.output_kind == "tensor"]
    estimators = [s for s in sources if s.output_kind == "estimator"]

    if isinstance(layer, Kernel):
        if layer.max_inputs == 0:
            if sources:
                result.error(
                    f"{layer.label} has no inputs: a kernel is not fed with data. Connect it "
                    "to a Gaussian process (or to KernelSum or KernelProduct) instead.",
                    nid,
                )
                return
        elif tensors or estimators:
            first = (tensors or estimators)[0]
            other = first.layer.label if first.layer else first.type
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

    if not isinstance(layer, ClassicalModel):  # an Input or neural layer
        if layer.max_inputs == 0:
            result.error("Input layers cannot have incoming connections.", nid)
        elif kernels:
            result.error(f"{layer.label} cannot take a kernel. {KERNEL_TARGETS}", nid)
        # Fed a classical model: the model says that nothing can follow it.
        return

    if isinstance(layer, Ensemble):
        _infer_ensemble(result, info, layer, sources)
        return
    if estimators:
        other = estimators[0].layer.label if estimators[0].layer else estimators[0].type
        result.error(
            f"{layer.label} takes its features from the Input layer (or from neural layers), "
            f"not the predictions of {other}. To combine models, connect each of them to an "
            f"ensemble ({ENSEMBLE_TYPES}).",
            nid,
        )
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
        result.error(
            f"{layer.label} has no features: connect the Input layer (or the last of the "
            "neural layers that learn its features) to it.",
            nid,
        )
        return
    if len(tensors) > 1:
        result.error(
            f"{layer.label} takes its features from one connection: the Input layer, or the "
            "last of the neural layers that learn them. Merge branches with Concat first.",
            nid,
        )
        return
    source = tensors[0]
    if not source.resolved or any(not k.resolved for k in kernels):
        return
    assert source.shape is not None
    if len(source.shape) != 1 and source.type == "Input":
        result.error(
            f"{layer.label} needs a flat feature vector: set the Input shape to the number of "
            f"features, e.g. 8 (it is {fmt_shape(source.shape)}), or put neural layers that "
            "end in a flat vector (Flatten, Linear) in between.",
            nid,
        )
        return
    if len(source.shape) != 1:
        other = source.layer.label if source.layer is not None else source.type
        result.error(
            f"{layer.label} needs a flat feature vector, but {other} outputs "
            f"{fmt_shape(source.shape)}. Add Flatten (or a Linear layer) before "
            f"{layer.label}: its output, shape [n], is the feature vector.",
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


def _infer_ensemble(
    result: Analysis, info: NodeInfo, layer: Ensemble, sources: list[NodeInfo]
) -> None:
    """An ensemble: two or more models of its task, all fed by the same Input."""
    nid = info.id
    others = [s for s in sources if s.output_kind != "estimator"]
    if others:
        first = others[0]
        what = "kernel" if first.output_kind == "kernel" else "data"
        name = first.layer.label if first.layer else first.type
        result.error(
            f"{layer.label} combines classical models and takes no {what} ({name}): connect "
            "model bricks such as SVC or gradient boosting to it. The models take the "
            "features from the Input layer themselves.",
            nid,
        )
        return
    nested = [s for s in sources if isinstance(s.layer, Ensemble)]
    if nested:
        inner = nested[0].layer.label  # type: ignore[union-attr]
        result.error(
            f"Ensembles can't be nested yet: connect the models of {inner} to {layer.label} "
            "directly.",
            nid,
        )
        return
    if len(sources) < 2:
        result.error(
            f"{layer.label} combines at least 2 models, but "
            f"{'only one is' if sources else 'none is'} connected.",
            nid,
        )
        return
    if any(not s.resolved for s in sources):
        return  # a model already reported an error
    models: list[ClassicalModel] = [s.layer for s in sources]  # type: ignore[misc]
    classification = layer.task == "classification"
    wrong = [m for m in models if m.task != layer.task]
    if wrong:
        kinds = (
            ("classifiers", "regression model")
            if classification
            else ("regression models", "classifier")
        )
        result.error(
            f"{layer.label} combines {kinds[0]}, but {wrong[0].label} is a {kinds[1]}. All "
            "models of an ensemble predict the same kind of target.",
            nid,
        )
        return
    features = {
        next(i for i in s.inputs if result.nodes[i].output_kind == "tensor") for s in sources
    }
    if len(features) > 1:
        result.error(
            "The models of an ensemble take their features from the same Input layer (or the "
            "same last neural layer): scikit-learn hands every model the same rows.",
            nid,
        )
        return
    if classification and layer.probabilities:
        without = [m for m in models if not m.probabilities]
        if without:
            result.error(
                f"Soft voting averages class probabilities, but {without[0].label} predicts "
                "none. Choose hard voting, or another model.",
                nid,
            )
            return
    names = _member_names(sources)
    layer.members = [
        Member(s.id, name, model) for s, name, model in zip(sources, names, models, strict=True)
    ]
    try:
        layer.check_members()
    except LayerError as exc:
        result.error(str(exc), nid)
        return
    info.shape = layer.setup(None)
    info.dtype = "float" if info.shape is not None else None
    info.resolved = True


def _member_names(sources: list[NodeInfo]) -> list[str]:
    """Names of the models of an ensemble (scikit-learn wants unique names, and they are
    variables of the generated code): the name of the brick or its id as a lower-case
    identifier, e.g. "Boosting 1" -> "boosting_1"."""
    names: list[str] = []
    for source in sources:
        text = re.sub(r"[^0-9a-zA-Z]+", "_", source.name or source.id).strip("_").lower()
        prefix = source.layer.name_prefix if source.layer is not None else "model"
        if not text or text[0].isdigit():
            text = f"{prefix}_{text}" if text else prefix
        if keyword.iskeyword(text) or text in RESERVED_NAMES:
            text = f"{text}_{prefix}"
        name, number = text, 2
        while name in names:
            name, number = f"{text}_{number}", number + 1
        names.append(name)
    return names


def _check_classical(result: Analysis) -> None:
    """Rules for the network as a whole: every kernel is used exactly once, one classical
    model at most (or one ensemble and the models it combines), nothing after it, and
    neural layers only before it (see :func:`_check_hybrid`)."""
    models: list[NodeInfo] = []
    ensembles: list[NodeInfo] = []
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
        elif isinstance(info.layer, Ensemble):
            ensembles.append(info)
        elif isinstance(info.layer, ClassicalModel):
            models.append(info)
        elif info.layer is not None and info.layer.family == "neural":
            neural.append(info)

    if ensembles:
        _check_members(result, ensembles[0], models)
        for extra in ensembles[1:]:
            result.error(
                "Only one ensemble per network. Remove this one or build it in a network of "
                "its own.",
                extra.id,
            )
        model = ensembles[0]
    else:
        for extra in models[1:]:
            result.error(
                "Only one classical model per network. Remove this one, build it in a network "
                f"of its own, or combine the models with an ensemble ({ENSEMBLE_TYPES}).",
                extra.id,
            )
        if not models:
            return
        model = models[0]
    label = model.layer.label  # type: ignore[union-attr]
    if model.outputs:
        result.error(
            f"Nothing can follow {label}: a classical model ends the network, its output is "
            "the prediction. Neural layers go before it, where they learn its features.",
            model.id,
        )
    elif neural:
        _check_hybrid(result, model, models if ensembles else [model], neural)


def _check_hybrid(
    result: Analysis, model: NodeInfo, fitted: list[NodeInfo], neural: list[NodeInfo]
) -> None:
    """Neural layers and a classical model: there is exactly one boundary (the layer that
    feeds the model, or every model of the ensemble), every neural layer leads to it, and
    the network has one Input."""
    label = model.layer.label  # type: ignore[union-attr]
    sources = {
        source
        for info in fitted
        for source in info.inputs
        if result.nodes[source].output_kind == "tensor"
    }
    if len(sources) != 1:
        return  # the model or the ensemble already reported it
    boundary = result.nodes[next(iter(sources))]
    if len(result.inputs) > 1:
        result.error(
            f"{FEATURES_ROLE}, from one Input layer: remove the other Input layers.", model.id
        )
        return
    feeds: set[str] = set()
    stack = [boundary.id]
    while stack:
        nid = stack.pop()
        if nid not in feeds:
            feeds.add(nid)
            stack += result.nodes[nid].inputs
    for info in neural:
        if info.id in feeds:
            continue
        name = info.layer.label if info.layer is not None else info.type
        if boundary.type == "Input":
            text = (
                f"{name} does not lead to {label}, which takes its features straight from the "
                f"Input layer. {FEATURES_ROLE}: connect the last of them to {label} instead "
                "of the Input, or remove this layer."
            )
        else:
            kind = boundary.layer.label if boundary.layer is not None else boundary.type
            last = f"{kind} '{boundary.name or boundary.id}'"
            text = (
                f"{name} does not lead to {label}. {FEATURES_ROLE}, and {last} passes them "
                "on: every neural layer must lead to it (its output is the only boundary), so "
                "connect this layer into that path or remove it."
            )
        result.error(text, info.id)


def _check_members(result: Analysis, ensemble: NodeInfo, models: list[NodeInfo]) -> None:
    """Every model of a network with an ensemble feeds that ensemble, and nothing else."""
    label = ensemble.layer.label  # type: ignore[union-attr]
    for info in models:
        name = info.layer.label  # type: ignore[union-attr]
        if not info.outputs:
            result.error(
                f"{name} is not connected to {label}. Connect it to the ensemble, or remove "
                "it: the ensemble ends the network.",
                info.id,
            )
        elif any(target != ensemble.id for target in info.outputs):
            result.error(
                f"{name} can only be connected to {label}: what it passes on is the model "
                "itself, which the ensemble combines. Nothing else can follow a classical "
                "model.",
                info.id,
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
