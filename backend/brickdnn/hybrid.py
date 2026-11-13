"""Hybrid networks: neural layers that learn the features of a classical model.

A hybrid graph has an Input, neural layers and a classical model (or an ensemble of them).
The last neural layer, the *boundary*, puts out a flat feature vector that the classical
model is fitted on (see :mod:`brickdnn.graph` for the rules). It is split into two graphs
that the rest of Brick DNN already knows (:func:`split_graph`):

- the **feature graph**: the Input and the neural layers, with the boundary as its only
  output. Its ``NeuralNetwork`` is the feature extractor.
- the **classical graph**: an Input of shape ``[features]`` (with the boundary's id, so the
  edges into the models stay as they are), the classical model(s) and the kernel bricks.

The training (training/hybrid_engine.py) first trains the feature graph with a temporary
linear head on the target (:func:`head_graph`), then fits the classical model on the
features it puts out. The generated script (:func:`generate_hybrid`) does the same with
PyTorch and scikit-learn. Like the rest of :mod:`brickdnn`, this module only writes text.

The files of a trained hybrid network (also those that the script writes)::

    features.pt     torch.save({"state_dict": <FeatureExtractor weights>,
                                "preprocessing": {"columns", "target", "mean", "std", ...}})
    model.joblib    {"format": "brickdnn-sklearn-1", "model": <Pipeline on the features>,
                     "task", "columns": ["feature_1", ...], "target", "class_names",
                     "hybrid": {"features": 32, "input_columns": [...]}}
"""

from __future__ import annotations

import copy
import textwrap
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .classical import (
    _NUMBERS,
    _READ_CSV,
    LINE_LENGTH,
    MODEL_FORMAT,
    _ClassicalGenerator,
)
from .graph import Analysis, GraphValidationError, analyze
from .layers import ClassicalModel
from .links import BANNER

if TYPE_CHECKING:
    from .codegen import GeneratorOptions

# The temporary head of the feature training (a node of the head graph).
HEAD_NODE = "__brickdnn_head__"
FEATURE_CLASS = "FeatureExtractor"


@dataclass
class HybridParts:
    """A hybrid graph split at its boundary (see the module docstring)."""

    feature_graph: dict[str, Any]
    classical_graph: dict[str, Any]
    boundary: str  # the id of the last neural layer
    features: int  # the length of the feature vector
    input_shape: list[int]
    input_dtype: str


def _document(graph: Any) -> dict[str, Any]:
    if isinstance(graph, dict) and isinstance(graph.get("graph"), dict):
        graph = graph["graph"]  # a saved project file
    assert isinstance(graph, dict)
    return graph


def split_graph(graph: Any, analysis: Analysis | None = None) -> HybridParts:
    """Split a valid hybrid graph (a document). Raises :class:`GraphValidationError` for an
    invalid graph and ``ValueError`` for one that is not hybrid."""
    document = _document(graph)
    analysis = analysis or analyze(document)
    if not analysis.valid:
        raise GraphValidationError(analysis)
    boundary = analysis.boundary
    if boundary is None or boundary.shape is None:
        raise ValueError("The network has no neural layers before a classical model.")
    neural = {
        nid
        for nid, info in analysis.nodes.items()
        if info.layer is not None and info.layer.family != "classical"
    }
    nodes = [n for n in document.get("nodes") or [] if isinstance(n, dict)]
    edges = [e for e in document.get("edges") or [] if isinstance(e, dict)]

    def key(value: Any) -> str:
        return str(value)

    feature_graph = {
        "version": document.get("version", 1),
        "nodes": [copy.deepcopy(n) for n in nodes if key(n.get("id")) in neural],
        "edges": [
            copy.deepcopy(e)
            for e in edges
            if key(e.get("source")) in neural and key(e.get("target")) in neural
        ],
    }
    features = int(boundary.shape[0])
    rest = [copy.deepcopy(n) for n in nodes if key(n.get("id")) not in neural]
    classical_graph = {
        "version": document.get("version", 1),
        "nodes": [
            {"id": boundary.id, "type": "Input", "params": {"shape": [features]}},
            *rest,
        ],
        "edges": [
            copy.deepcopy(e)
            for e in edges
            if key(e.get("target")) not in neural
            and (key(e.get("source")) not in neural or key(e.get("source")) == boundary.id)
        ],
    }
    source = analysis.nodes[analysis.inputs[0]]
    return HybridParts(
        feature_graph,
        classical_graph,
        boundary.id,
        features,
        list(source.shape or ()),
        source.dtype or "float",
    )


def head_graph(parts: HybridParts, outputs: int) -> dict[str, Any]:
    """The feature graph with the temporary head: a Linear layer with ``outputs`` outputs
    (one per class, or 1 for a regression) after the boundary. It is what the feature
    training checks against the data; the head is dropped afterwards."""
    graph = copy.deepcopy(parts.feature_graph)
    graph["nodes"].append(
        {"id": HEAD_NODE, "type": "Linear", "params": {"out_features": outputs}, "name": "head"}
    )
    graph["edges"].append({"source": parts.boundary, "target": HEAD_NODE})
    return graph


def model_layer(analysis: Analysis) -> ClassicalModel:
    model = analysis.model
    assert model is not None and isinstance(model.layer, ClassicalModel)
    return model.layer


# --------------------------------------------------------------------------- #
# The script
# --------------------------------------------------------------------------- #


def generate_hybrid(analysis: Analysis, options: GeneratorOptions) -> str:
    return _HybridGenerator(analysis, options).render()


class _HybridGenerator:
    def __init__(self, analysis: Analysis, options: GeneratorOptions):
        from .codegen import GeneratorOptions, _Generator

        self.a = analysis
        self.opt = options
        graph = document_of(analysis)
        self.parts = split_graph(graph, analysis)
        self.class_name = (
            FEATURE_CLASS if options.class_name == "NeuralNetwork" else options.class_name
        )
        feature_options = GeneratorOptions(self.class_name, include_main=False)
        self.neural = _Generator(analyze(self.parts.feature_graph), feature_options)
        self.classical = _ClassicalGenerator(
            analyze(self.parts.classical_graph), GeneratorOptions(include_main=False)
        )
        self.layer = model_layer(analysis)
        self.classification = self.layer.task == "classification"
        # CSV training and predictions: one flat float Input (images: the demo only).
        self.tabular = len(self.parts.input_shape) == 1 and self.parts.input_dtype == "float"

    def render(self) -> str:
        main = self.opt.include_main
        training = main and self.opt.include_training and self.tabular
        header = BANNER + "\n" + self._docstring(main, training) + "\n\n" + self._imports(main)
        if main:
            header += "\n\n" + self._constants()
        parts = [header, self.neural._model_class(), self.classical._make_model()]
        if main:
            parts.append(_EXTRACT)
            if self.tabular:
                parts += [_READ_CSV, _NUMBERS]
            if training:
                parts += [
                    self.classical._load_csv(),
                    _SPLIT,
                    self._learn_features(),
                    _FIT.replace("[[REPORT]]", self.classical._report()),
                    self._save_model(),
                ]
            if self.tabular:
                parts.append(
                    _PREDICT.replace("[[OUTPUT]]", self.classical._output()).replace(
                        "[[CLASS]]", self.class_name
                    )
                )
            parts += [self._demo(), self._main(training)]
        return "\n\n\n".join(parts) + "\n"

    # -- header ------------------------------------------------------------ #

    def _docstring(self, main: bool, training: bool) -> str:
        layer = self.layer
        parts = self.parts
        what = "the class of each row" if self.classification else "one number per row"
        source = (
            f"{parts.input_shape[0]} numeric feature{'s' if parts.input_shape[0] != 1 else ''}"
            if self.tabular
            else "inputs of shape " + "×".join(str(s) for s in parts.input_shape)
        )
        about = (
            f"Neural layers ({self.class_name}) turn {source} into {parts.features} learned "
            f"features; {layer.estimator} is fitted on them and predicts {what}. The training "
            "has two phases: first the neural layers learn with a temporary linear head on "
            "the target (with --frozen they keep their random initial weights), then the "
            "head is dropped and the classical model is fitted on the features of the "
            "training rows."
        )
        lines = [
            f'"""{layer.label} on learned features: generated with Brick DNN.',
            "",
            *textwrap.wrap(about, width=LINE_LENGTH - 8),
        ]
        if main:
            lines += [
                "",
                "Print the models and fit them to random data:",
                "",
                "    python this_file.py",
            ]
        if training:
            lines += [
                "",
                "Train on a CSV file (every column but the last is a feature, the last column is",
                "the target); this writes features.pt (the neural layers) and model.joblib (the",
                "classical model). Then predict new rows with both:",
                "",
                "    python this_file.py --csv data.csv --epochs 20",
                "    python this_file.py --predict new_data.csv",
            ]
        elif main and self.tabular:
            lines += [
                "",
                "Predict new rows with the files of a model trained in Brick DNN:",
                "",
                "    python this_file.py --predict new_data.csv --weights features.pt "
                "--save model.joblib",
            ]
        lines.append('"""')
        return "\n".join(lines)

    def _imports(self, main: bool) -> str:
        lines = []
        if main:
            lines += ["import argparse"]
            if self.tabular:
                lines += ["import csv"]
            if self.opt.include_training and self.tabular:
                lines += ["import time"]  # the fit reports its time
            lines += ["", "import joblib", "import numpy as np"]
        lines += ["import torch", "import torch.nn as nn"]
        lines += self.classical._imports(False, False).split("\n")
        if main and self.opt.include_training and self.tabular:
            lines.append("from torch.utils.data import DataLoader, TensorDataset")
        return "\n".join(lines)

    def _constants(self) -> str:
        shape = self.parts.input_shape
        lines = []
        if self.tabular:
            lines.append(f"NUM_FEATURES = {shape[0]}  # input columns of the CSV file")
        else:
            lines.append(f"INPUT_SHAPE = {tuple(shape)!r}  # one sample, without the batch")
        lines.append(f"NUM_LEARNED = {self.parts.features}  # features the neural layers learn")
        return "\n".join(lines)

    # -- training ---------------------------------------------------------- #

    def _learn_features(self) -> str:
        head = _CLASSIFICATION_HEAD if self.classification else _REGRESSION_HEAD
        return _LEARN_FEATURES.replace("[[HEAD]]", head).replace("[[CLASS]]", self.class_name)

    def _save_model(self) -> str:
        if self.classification:
            task = (
                '        "task": "classification",\n'
                "        # one name per column of model.predict_proba()\n"
                '        "class_names": [str(name) for name in model.classes_],'
            )
        else:
            task = '        "task": "regression",\n        "class_names": None,'
        return (
            _SAVE_MODEL.replace("[[FORMAT]]", MODEL_FORMAT)
            .replace("[[TASK]]", task)
            .replace("[[CLASS]]", self.class_name)
        )

    def _demo(self) -> str:
        if self.classification:
            target = '    target = np.where(learned[:, 0] + learned[:, -1] > 0, "yes", "no")'
        else:
            target = "    target = learned[:, 0] + np.sin(2 * learned[:, -1])"
        shape = "(40, NUM_FEATURES)" if self.tabular else "(40, *INPUT_SHAPE)"
        return (
            _DEMO.replace("[[TARGET]]", target)
            .replace("[[SHAPE]]", shape)
            .replace("[[CLASS]]", self.class_name)
        )

    def _main(self, training: bool) -> str:
        name = self.layer.label
        lines = [
            "def main() -> None:",
            "    parser = argparse.ArgumentParser(",
            f'        description="{name} on learned features, generated with Brick DNN"',
            "    )",
        ]
        if training:
            lines.append(
                '    parser.add_argument("--csv", help="train on this CSV file (last column = '
                'target)")'
            )
        if self.tabular:
            lines.append(
                '    parser.add_argument("--predict", metavar="CSV", help="predict the rows of '
                'this file")'
            )
        if training:
            lines += [
                '    parser.add_argument("--epochs", type=int, default=10)',
                '    parser.add_argument("--batch-size", type=int, default=32)',
                '    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")',
                "    parser.add_argument(",
                '        "--frozen",',
                '        action="store_true",',
                '        help="keep the random initial weights of the neural layers (no training)",',
                "    )",
                '    parser.add_argument("--seed", type=int, default=0, help="for a repeatable run")',
            ]
        if self.tabular:
            lines += [
                "    parser.add_argument(",
                '        "--weights", default="features.pt", help="the neural layers (--csv writes '
                'it)"',
                "    )",
                "    parser.add_argument(",
                '        "--save", default="model.joblib", help="the classical model (--csv writes '
                'it)"',
                "    )",
            ]
        lines.append("    args = parser.parse_args()")
        lines.append(
            "    torch.manual_seed(args.seed)  # the same initial weights for the same seed"
            if training
            else "    torch.manual_seed(0)  # the same initial weights every time"
        )
        lines.append("")
        if self.tabular:
            lines += [
                "    if args.predict:",
                "        predict(args.weights, args.save, args.predict)",
                "        return",
            ]
        lines += [
            f"    extractor = {self.class_name}()",
            "    model = make_model()",
            "    print(extractor)",
            "    print(model)",
        ]
        if training:
            lines += [
                "    if args.csv:",
                "        header, features, target = load_csv(args.csv)",
                "        train_rows, val_rows = split_rows(len(features), args.seed)",
                "        mean, std = standardise(features, train_rows)",
                "        features = (features - mean) / std",
                "        if args.frozen:",
                '            print("Frozen random features: the neural layers are not trained.")',
                "        else:",
                "            learn_features(",
                *[
                    f"                {arg},"
                    for arg in (
                        "extractor",
                        "features",
                        "target",
                        "train_rows",
                        "val_rows",
                        "args.epochs",
                        "args.batch_size",
                        "args.lr",
                        "args.seed",
                    )
                ],
                "            )",
                "        learned = extract(extractor, features)",
                "        fit(model, learned, target, train_rows, val_rows)",
                "        save_model(extractor, model, args.weights, args.save, header, mean, std)",
                "        return",
            ]
        lines += ["    demo(extractor, model)", "", "", 'if __name__ == "__main__":', "    main()"]
        return "\n".join(lines)


def document_of(analysis: Analysis) -> dict[str, Any]:
    """A graph document from an analysis (ids, types, names, parameters and edges in the
    order of the edges into each node), for :func:`split_graph`."""
    nodes = []
    for nid in analysis.order:
        info = analysis.nodes[nid]
        assert info.layer is not None
        node: dict[str, Any] = {
            "id": nid,
            "type": info.type,
            "params": {k: list(v) if isinstance(v, tuple) else v for k, v in info.layer.p.items()},
        }
        if info.name:
            node["name"] = info.name
        nodes.append((info.index, node))
    edges = [
        {"source": source, "target": nid}
        for nid in analysis.order
        for source in analysis.nodes[nid].inputs
    ]
    return {
        "version": 1,
        "nodes": [n for _, n in sorted(nodes, key=lambda x: x[0])],
        "edges": edges,
    }


_EXTRACT = '''\
# --------------------------------------------------------------------------- #
# The two phases: learning the features, then fitting the classical model
# --------------------------------------------------------------------------- #


def extract(extractor: nn.Module, inputs: np.ndarray) -> np.ndarray:
    """The learned features of every row (or sample): the output of the neural layers,
    computed in batches without gradients."""
    extractor.eval()
    x = torch.as_tensor(inputs, dtype=torch.float32)
    with torch.no_grad():
        parts = [extractor(x[i : i + 1024]) for i in range(0, len(x), 1024)]
    if not parts:
        return np.empty((0, NUM_LEARNED))
    return torch.cat(parts).double().numpy()'''

_SPLIT = '''\
def split_rows(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A random 80 % of the rows for training, the rest for validation (fixed by ``seed``).
    Both phases use the same split."""
    if n < 4:
        raise SystemExit("The CSV file needs at least 4 rows (training + validation).")
    order = np.random.default_rng(seed).permutation(n)
    n_val = max(1, n // 5)
    return order[n_val:], order[:n_val]


def standardise(features: np.ndarray, train_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The mean and standard deviation of the training rows. New rows are standardised with
    them as well, before the neural layers."""
    mean = features[train_rows].mean(axis=0)
    std = features[train_rows].std(axis=0)
    std[std < 1e-8] = 1.0  # constant columns are left as they are
    return mean, std'''

_CLASSIFICATION_HEAD = """\
    # Classification: one output per class and the cross-entropy loss.
    classes, labels = np.unique(target, return_inverse=True)
    y = torch.as_tensor(labels, dtype=torch.long)
    head = nn.Linear(NUM_LEARNED, len(classes))
    loss_fn = nn.CrossEntropyLoss()"""

_REGRESSION_HEAD = """\
    # Regression: one output and the mean squared error of the standardised target.
    spread = float(target[train_rows].std()) or 1.0
    scaled = (target - target[train_rows].mean()) / spread
    y = torch.as_tensor(scaled, dtype=torch.float32).unsqueeze(1)
    head = nn.Linear(NUM_LEARNED, 1)
    loss_fn = nn.MSELoss()"""

_LEARN_FEATURES = '''\
def learn_features(
    extractor: [[CLASS]],
    features: np.ndarray,
    target: np.ndarray,
    train_rows: np.ndarray,
    val_rows: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> None:
    """Phase 1: train the neural layers with a temporary linear head on the target, then
    drop the head. ``features`` are standardised."""
[[HEAD]]
    x = torch.as_tensor(features, dtype=torch.float32)
    train, val = torch.as_tensor(train_rows), torch.as_tensor(val_rows)
    network = nn.Sequential(extractor, head)
    loader = DataLoader(
        TensorDataset(x[train], y[train]),
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(train) > batch_size,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    print(f"Learning the features on {len(train)} rows, {len(val)} for validation")
    for epoch in range(1, epochs + 1):
        network.train()
        total, seen = 0.0, 0
        for inputs, targets in loader:
            optimizer.zero_grad()
            loss = loss_fn(network(inputs), targets)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(inputs)
            seen += len(inputs)
        network.eval()
        with torch.no_grad():
            val_loss = loss_fn(network(x[val]), y[val]).item()
        print(
            f"Epoch {epoch:3d}/{epochs}  train loss {total / max(seen, 1):.4f}"
            f"  val loss {val_loss:.4f}"
        )'''

_FIT = '''\
def fit(
    model: Pipeline,
    features: np.ndarray,
    target: np.ndarray,
    train_rows: np.ndarray,
    val_rows: np.ndarray,
) -> None:
    """Phase 2: fit the classical model on the learned features of the training rows and
    validate it on the other rows. Its StandardScaler learns from the training rows only."""
    print(f"Fitting on the {features.shape[1]} learned features of {len(train_rows)} rows")
    started = time.perf_counter()
    model.fit(features[train_rows], target[train_rows])
    print(f"Fitted in {time.perf_counter() - started:.1f} s")
[[REPORT]]'''

_SAVE_MODEL = '''\
def save_model(
    extractor: [[CLASS]],
    model: Pipeline,
    weights_path: str,
    model_path: str,
    header: list[str],
    mean: np.ndarray,
    std: np.ndarray,
) -> None:
    """Save the neural layers with the preprocessing of the input columns (PyTorch) and the
    fitted classical model (joblib, a pickle: only load files you trust)."""
    preprocessing = {
        "columns": header[:-1],  # the input columns by name (empty: no header line)
        "target": header[-1] if header else "target",
        "mean": mean.tolist(),  # every input column is standardised: (value - mean) / std
        "std": std.tolist(),
    }
    torch.save({"state_dict": extractor.state_dict(), "preprocessing": preprocessing}, weights_path)
    bundle = {
        "format": "[[FORMAT]]",
        "model": model,
        "columns": [f"feature_{i + 1}" for i in range(NUM_LEARNED)],  # the learned features
        "target": preprocessing["target"],
[[TASK]]
        "hybrid": {"features": NUM_LEARNED, "input_columns": preprocessing["columns"]},
    }
    joblib.dump(bundle, model_path)
    print(f"Saved the neural layers to {weights_path} and the classical model to {model_path}")'''

_PREDICT = '''\
def predict(weights_path: str, model_path: str, csv_path: str) -> None:
    """Print a prediction for every row of a CSV file: standardise the input columns like
    the training rows, compute the learned features, then let the classical model predict.

    If both the saved preprocessing and the file have a header line, the input columns are
    found by their names; otherwise they have to come first. Other columns are ignored.
    """
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    extractor = [[CLASS]]()
    extractor.load_state_dict(checkpoint["state_dict"])
    preprocessing = checkpoint["preprocessing"]
    bundle = joblib.load(model_path)  # a pickle: only load files you trust
    model, columns = bundle["model"], preprocessing["columns"]
    header, rows = read_csv(csv_path)
    if columns and header:
        missing = [name for name in columns if name not in header]
        if missing:
            raise SystemExit(f"{csv_path} lacks the columns {', '.join(missing)}.")
        rows = [[row[header.index(name)] for name in columns] for row in rows]
    elif all(len(row) >= NUM_FEATURES for row in rows):
        rows = [row[:NUM_FEATURES] for row in rows]
    else:
        raise SystemExit(f"Expected {NUM_FEATURES} input columns.")
    inputs = numbers(rows, csv_path)
    mean, std = np.asarray(preprocessing["mean"]), np.asarray(preprocessing["std"])
    features = extract(extractor, (inputs - mean) / std)
[[OUTPUT]]'''

_DEMO = '''\
def demo(extractor: [[CLASS]], model: Pipeline) -> None:
    """Compute the features of 40 random samples with the untrained neural layers, fit the
    model on them and predict two (shows that both parts work together)."""
    rng = np.random.default_rng(0)
    learned = extract(extractor, rng.normal(size=[[SHAPE]]))
[[TARGET]]
    model.fit(learned, target)
    print("Predictions for 2 random samples:", model.predict(learned[:2]))'''
