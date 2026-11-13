"""Turn an analysed graph into a self-contained, runnable PyTorch script."""

from __future__ import annotations

import builtins
import keyword
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .classical import generate_classical
from .graph import Analysis, GraphValidationError, NodeInfo, analyze
from .layers import fmt_shape
from .links import BANNER, PROJECT_URL, SITE_URL  # noqa: F401 (re-exported)

# Names that must not be used for layers, inputs or the class in the generated module:
# Python keywords and builtins, public attributes of nn.Module (a layer called "compile"
# would be shadowed by the method) and every module-level or local name the script uses.
NN_MODULE_ATTRIBUTES = {
    "T_destination",
    "add_module",
    "apply",
    "bfloat16",
    "buffers",
    "call_super_init",
    "children",
    "compile",
    "cpu",
    "cuda",
    "double",
    "dump_patches",
    "eval",
    "extra_repr",
    "float",
    "forward",
    "get_buffer",
    "get_extra_state",
    "get_parameter",
    "get_submodule",
    "half",
    "ipu",
    "load_state_dict",
    "modules",
    "mtia",
    "named_buffers",
    "named_children",
    "named_modules",
    "named_parameters",
    "parameters",
    "register_backward_hook",
    "register_buffer",
    "register_forward_hook",
    "register_forward_pre_hook",
    "register_full_backward_hook",
    "register_full_backward_pre_hook",
    "register_load_state_dict_post_hook",
    "register_load_state_dict_pre_hook",
    "register_module",
    "register_parameter",
    "register_state_dict_post_hook",
    "register_state_dict_pre_hook",
    "requires_grad_",
    "set_extra_state",
    "set_submodule",
    "share_memory",
    "smart_apply",
    "state_dict",
    "to",
    "to_empty",
    "train",
    "training",
    "type",
    "xpu",
    "zero_grad",
}
SCRIPT_NAMES = {
    "self",
    "torch",
    "nn",
    "argparse",
    "csv",
    "DataLoader",
    "TensorDataset",
    "random_split",
    "count_parameters",
    "read_csv",
    "load_csv",
    "train",
    "save_model",
    "predict",
    "main",
    "NUM_FEATURES",
    "NUM_CLASSES",
    "x",
    "model",
    "out",
    "hidden",
    "args",
    "parser",
    "tensor",
    "i",
}
RESERVED = set(keyword.kwlist) | set(dir(builtins)) | NN_MODULE_ATTRIBUTES | SCRIPT_NAMES


@dataclass
class GeneratorOptions:
    class_name: str = "NeuralNetwork"
    include_main: bool = True
    include_training: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> GeneratorOptions:
        data = data or {}
        name = str(data.get("class_name") or cls.class_name).strip()
        if not name.isidentifier() or keyword.iskeyword(name):
            raise ValueError(f"'{name}' is not a valid Python class name.")
        if name in RESERVED:
            raise ValueError(f"'{name}' is already used in the generated code; pick another name.")
        return cls(
            class_name=name,
            include_main=bool(data.get("include_main", True)),
            include_training=bool(data.get("include_training", True)),
        )


def generate_code(graph: Any, options: GeneratorOptions | dict[str, Any] | None = None) -> str:
    """Generate PyTorch code for ``graph``, or a scikit-learn script for a classical
    network (a Gaussian process, SVM or gradient boosting, see :mod:`brickdnn.classical`).

    Raises :class:`GraphValidationError` if the graph contains errors.
    """
    if not isinstance(options, GeneratorOptions):
        options = GeneratorOptions.from_dict(options)
    analysis = analyze(graph)
    if not analysis.valid:
        raise GraphValidationError(analysis)
    if analysis.family == "classical":
        return generate_classical(analysis, options)
    return _Generator(analysis, options).render()


def _identifier(text: str) -> str | None:
    ident = re.sub(r"\W+", "_", text.strip().lower()).strip("_")
    if not ident or not ident.isascii():
        return None
    if ident[0].isdigit():
        ident = f"layer_{ident}"
    return ident


class _Generator:
    def __init__(self, analysis: Analysis, options: GeneratorOptions):
        self.a = analysis
        self.opt = options
        self.var: dict[str, str] = {}
        self.attr: dict[str, str] = {}
        self._assign_names()

    # -- naming ----------------------------------------------------------- #

    def _assign_names(self) -> None:
        used: set[str] = set(RESERVED) | {self.opt.class_name}
        counters: Counter[str] = Counter()

        def unique(base: str) -> str:
            while True:
                counters[base] += 1
                candidate = f"{base}_{counters[base]}"
                if candidate not in used:
                    used.add(candidate)
                    return candidate

        def claim(custom: str | None, base: str) -> str:
            if custom and custom not in used:
                used.add(custom)
                return custom
            return unique(base)

        inputs = self.a.inputs
        for nid in self.a.order:
            info = self.a.nodes[nid]
            assert info.layer is not None
            custom = _identifier(info.name) if info.name else None
            if info.type == "Input":
                if len(inputs) == 1:
                    self.var[nid] = "x"
                else:
                    self.var[nid] = claim(custom, "x")
                continue
            name = claim(custom, info.layer.name_hint())
            if info.layer.has_module:
                self.attr[nid] = name
            self.var[nid] = name

        if self._is_chain():
            for nid in self.a.order:
                self.var[nid] = "x"

    def _is_chain(self) -> bool:
        if len(self.a.inputs) != 1 or len(self.a.outputs) != 1:
            return False
        return all(len(n.inputs) <= 1 and len(n.outputs) <= 1 for n in self.a.nodes.values())

    # -- rendering -------------------------------------------------------- #

    def render(self) -> str:
        training = self.opt.include_main and self.opt.include_training and self._can_train()
        parts = [self._model_class()]
        if self.opt.include_main:
            parts.append(_COUNT_PARAMETERS)
            if training:
                parts.append(self._training_code())
            parts.append(self._main(training))
        header = BANNER + "\n" + self._docstring(training) + "\n\n" + self._imports(training)
        # PEP 8: two blank lines between top-level definitions.
        return "\n\n\n".join([header, *parts]) + "\n"

    def _node(self, nid: str) -> NodeInfo:
        return self.a.nodes[nid]

    def _shape_text(self, nid: str) -> str:
        shape = self._node(nid).shape
        assert shape is not None
        return fmt_shape(("batch", *shape))  # type: ignore[arg-type]

    def _docstring(self, training: bool) -> str:
        name = self.opt.class_name
        lines = [f'"""{name}: generated with Brick DNN.', ""]
        for label, ids in (("Input", self.a.inputs), ("Output", self.a.outputs)):
            for nid in ids:
                suffix = f" {self.var[nid]}" if len(ids) > 1 else ""
                dtype = "  (integer token ids)" if self._node(nid).dtype == "long" else ""
                lines.append(f"{label}{suffix}: {self._shape_text(nid)}{dtype}")
        lines.append(f"Trainable parameters: {self.a.total_params:,}")
        if self.opt.include_main:
            lines += [
                "",
                "Print the model and run it on a random batch:",
                "",
                "    python this_file.py",
            ]
        if training:
            lines += [
                "",
                "Train it on a CSV file (every column but the last is a feature,",
                "the last column is the target), then predict new rows with the saved model:",
                "",
                "    python this_file.py --csv data.csv --epochs 20",
                "    python this_file.py --predict new_data.csv",
            ]
        lines.append('"""')
        return "\n".join(lines)

    def _imports(self, training: bool) -> str:
        lines = []
        if training:
            lines += ["import argparse", "import csv", ""]
        lines += ["import torch", "import torch.nn as nn"]
        if training:
            lines.append("from torch.utils.data import DataLoader, TensorDataset")
        return "\n".join(lines)

    def _model_class(self) -> str:
        lines = [
            f"class {self.opt.class_name}(nn.Module):",
            "    def __init__(self) -> None:",
            "        super().__init__()",
        ]
        for nid in self.a.order:
            if nid in self.attr:
                code = self._node(nid).layer.init_code()  # type: ignore[union-attr]
                first, *rest = code.split("\n")
                lines.append(f"        self.{self.attr[nid]} = {first}")
                lines += [f"        {line}" for line in rest]

        args = ", ".join(f"{self.var[nid]}: torch.Tensor" for nid in self.a.inputs)
        outputs = self.a.outputs
        returns = (
            "torch.Tensor"
            if len(outputs) == 1
            else "tuple[" + ", ".join(["torch.Tensor"] * len(outputs)) + "]"
        )
        lines += ["", f"    def forward(self, {args}) -> {returns}:"]
        for nid in self.a.order:
            info = self._node(nid)
            if info.type == "Input":
                continue
            assert info.layer is not None
            inputs = [self.var[src] for src in info.inputs]
            for line in info.layer.forward_lines(self.attr.get(nid, ""), inputs, self.var[nid]):
                lines.append(f"        {line}")
        lines.append("        return " + ", ".join(self.var[nid] for nid in outputs))
        return "\n".join(lines)

    # -- __main__ ---------------------------------------------------------- #

    def _example_input(self, nid: str) -> str:
        info = self._node(nid)
        assert info.shape is not None
        dims = ", ".join(str(d) for d in (2, *info.shape))
        if info.dtype == "long":
            vocab = [
                self._node(t).layer.p["num_embeddings"]  # type: ignore[union-attr]
                for t in info.outputs
                if self._node(t).type == "Embedding"
            ]
            high = min(vocab) if vocab else 2
            return f"torch.randint(0, {high}, ({dims},))"
        return f"torch.randn({dims})"

    def _sanity_check(self, indent: str) -> list[str]:
        lines = [
            f"{indent}# Check the model with a random batch of 2 samples.",
            f"{indent}model.eval()",
        ]
        for nid in self.a.inputs:
            lines.append(f"{indent}{self.var[nid]} = {self._example_input(nid)}")
        call = ", ".join(self.var[nid] for nid in self.a.inputs)
        lines += [f"{indent}with torch.no_grad():", f"{indent}    out = model({call})"]
        if len(self.a.outputs) == 1:
            expected = fmt_shape((2, *self._node(self.a.outputs[0]).shape))  # type: ignore[arg-type]
            lines.append(f'{indent}print("Output shape:", tuple(out.shape))  # expected {expected}')
        else:
            lines += [
                f"{indent}for i, tensor in enumerate(out, start=1):",
                f'{indent}    print(f"Output {{i}} shape:", tuple(tensor.shape))',
            ]
        return lines

    def _main(self, training: bool) -> str:
        name = self.opt.class_name
        lines = ["def main() -> None:"]
        if training:
            lines += [
                f'    parser = argparse.ArgumentParser(description="{name} generated with Brick DNN")',
                '    parser.add_argument("--csv", help="train on this CSV file (last column = target)")',
                '    parser.add_argument("--predict", metavar="CSV", help="predict the rows of this file")',
                '    parser.add_argument("--epochs", type=int, default=10)',
                '    parser.add_argument("--batch-size", type=int, default=32)',
                '    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")',
                '    parser.add_argument("--seed", type=int, default=0, help="for a repeatable run")',
                "    parser.add_argument(",
                '        "--save", default="model.pt", help="the trained model (--csv writes it, '
                '--predict reads it)"',
                "    )",
                "    args = parser.parse_args()",
                "    torch.manual_seed(args.seed)  # the same initial weights for the same seed",
                "",
            ]
        lines.append(f"    model = {name}()")
        if training:
            lines += [
                "    if args.predict:",
                "        predict(model, args.save, args.predict)",
                "        return",
            ]
        lines += [
            "    print(model)",
            '    print(f"Trainable parameters: {count_parameters(model):,}")',
            "",
        ]
        if training:
            lines += [
                "    if args.csv:",
                "        header, features, target = load_csv(args.csv)",
                "        mean, std = train(",
                "            model, features, target, args.epochs, args.batch_size, args.lr, args.seed",
                "        )",
                "        save_model(model, args.save, header, mean, std)",
                "        return",
                "",
            ]
        lines += self._sanity_check("    ")
        lines += ["", "", 'if __name__ == "__main__":', "    main()"]
        return "\n".join(lines)

    # -- training --------------------------------------------------------- #

    def _can_train(self) -> bool:
        """CSV training is offered for tabular models: one flat float input, one flat output."""
        if len(self.a.inputs) != 1 or len(self.a.outputs) != 1:
            return False
        source = self._node(self.a.inputs[0])
        target = self._node(self.a.outputs[0])
        return (
            source.dtype == "float"
            and source.shape is not None
            and len(source.shape) == 1
            and target.shape is not None
            and len(target.shape) == 1
        )

    def _training_code(self) -> str:
        n_features = self._node(self.a.inputs[0]).shape[0]  # type: ignore[index]
        n_outputs = self._node(self.a.outputs[0]).shape[0]  # type: ignore[index]
        classification = n_outputs > 1
        if classification:
            header = (
                f"NUM_FEATURES = {n_features}\n"
                f"NUM_CLASSES = {n_outputs}  # classification: the target column holds the class "
                f"index 0..{n_outputs - 1}"
            )
            targets = (
                "    if target.min() < 0 or target.max() >= NUM_CLASSES:\n"
                '        raise SystemExit(f"Targets must be class indices 0..{NUM_CLASSES - 1}.")\n'
                "    return header, features, target.long()"
            )
            task = '        "task": "classification",\n        "num_classes": NUM_CLASSES,'
            loss = "nn.CrossEntropyLoss()"
            metric_init = "        val_loss, correct = 0.0, 0\n"
            metric_update = (
                "                correct += (outputs.argmax(dim=1) == targets).sum().item()\n"
            )
            metric_print = "  val acc {correct / len(val_set):.1%}"
            predictions = "outputs.argmax(dim=1).tolist()"
        else:
            header = (
                f"NUM_FEATURES = {n_features}\n"
                "# regression: the model has a single output, the target column is a number"
            )
            targets = "    return header, features, target.unsqueeze(1)"
            task = '        "task": "regression",'
            loss = "nn.MSELoss()"
            metric_init = "        val_loss = 0.0\n"
            metric_update = ""
            metric_print = ""
            predictions = "outputs[:, 0].tolist()"

        return (
            _TRAINING_TEMPLATE.replace("[[HEADER]]", header)
            .replace("[[TARGETS]]", targets)
            .replace("[[TASK]]", task)
            .replace("[[LOSS]]", loss)
            .replace("[[METRIC_INIT]]", metric_init)
            .replace("[[METRIC_UPDATE]]", metric_update)
            .replace("[[METRIC_PRINT]]", metric_print)
            .replace("[[PREDICTIONS]]", predictions)
        )


_COUNT_PARAMETERS = """\
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)"""


_TRAINING_TEMPLATE = '''\
# --------------------------------------------------------------------------- #
# Training on a CSV file, and predictions for new rows
# --------------------------------------------------------------------------- #

[[HEADER]]


def read_csv(path: str) -> tuple[list[str], torch.Tensor]:
    """Read a CSV file of numbers: the column names of its header line (an empty list if it
    has none) and the values."""
    with open(path, newline="") as f:
        rows = [row for row in csv.reader(f) if row]
    header: list[str] = []
    if rows:
        try:
            [float(value) for value in rows[0]]
        except ValueError:
            header, rows = [name.strip() for name in rows[0]], rows[1:]
    if not rows:
        raise SystemExit(f"{path} contains no data rows.")
    values = [[float(value) for value in row] for row in rows]
    return header, torch.tensor(values, dtype=torch.float32)


def load_csv(path: str) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    """Training data: every column but the last is a feature, the last one is the target."""
    header, data = read_csv(path)
    features, target = data[:, :-1], data[:, -1]
    if features.shape[1] != NUM_FEATURES:
        raise SystemExit(f"Expected {NUM_FEATURES} feature columns, found {features.shape[1]}.")
[[TARGETS]]


def train(
    model: nn.Module,
    features: torch.Tensor,
    target: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Train on a random 80 % of the rows and validate on the rest (the split depends only
    on ``seed``).

    Returns the mean and standard deviation that standardise the features. They come from
    the training rows only, and new data has to be standardised with them as well.
    """
    if len(features) < 4:
        raise SystemExit("The CSV file needs at least 4 rows (training + validation).")
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(features), generator=generator)
    n_val = max(1, len(features) // 5)
    val_rows, train_rows = order[:n_val], order[n_val:]

    mean = features[train_rows].mean(dim=0)
    std = features[train_rows].std(dim=0)
    std[std < 1e-8] = 1.0  # constant columns are left as they are
    features = (features - mean) / std

    train_set = TensorDataset(features[train_rows], target[train_rows])
    val_set = TensorDataset(features[val_rows], target[val_rows])
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(train_set) > batch_size,
        generator=generator,
    )
    val_loader = DataLoader(val_set, batch_size=batch_size)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Training on {device}: {len(train_set)} rows, {len(val_set)} for validation")
    model.to(device)

    loss_fn = [[LOSS]]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, seen = 0.0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(inputs), targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(inputs)
            seen += len(inputs)

        model.eval()
[[METRIC_INIT]]        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += loss_fn(outputs, targets).item() * len(inputs)
[[METRIC_UPDATE]]        print(
            f"Epoch {epoch:3d}/{epochs}  train loss {train_loss / max(seen, 1):.4f}"
            f"  val loss {val_loss / len(val_set):.4f}[[METRIC_PRINT]]"
        )
    model.cpu()
    return mean, std


def save_model(
    model: nn.Module, path: str, header: list[str], mean: torch.Tensor, std: torch.Tensor
) -> None:
    """Save the weights together with what predict() needs to prepare new rows."""
    preprocessing = {
        "columns": header[:-1],  # the feature columns by name (empty: no header line)
        "target": header[-1] if header else "target",
        "mean": mean.tolist(),  # every feature is standardised: (value - mean) / std
        "std": std.tolist(),
[[TASK]]
    }
    torch.save({"state_dict": model.state_dict(), "preprocessing": preprocessing}, path)
    print(f"Saved the weights and the preprocessing to {path}")


def predict(model: nn.Module, path: str, csv_path: str) -> None:
    """Print a prediction for every row of a CSV file, prepared like the training data.

    If both files have a header line, the feature columns are found by their names;
    otherwise they have to come first. Other columns (such as the target) are ignored.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    preprocessing = checkpoint["preprocessing"]
    columns = preprocessing["columns"]
    header, data = read_csv(csv_path)
    if columns and header:
        missing = [name for name in columns if name not in header]
        if missing:
            raise SystemExit(f"{csv_path} lacks the columns {', '.join(missing)}.")
        features = data[:, [header.index(name) for name in columns]]
    elif data.shape[1] >= NUM_FEATURES:
        features = data[:, :NUM_FEATURES]
    else:
        raise SystemExit(f"Expected {NUM_FEATURES} feature columns, found {data.shape[1]}.")
    mean = torch.tensor(preprocessing["mean"])
    std = torch.tensor(preprocessing["std"])
    with torch.no_grad():
        outputs = model((features - mean) / std)
    print(preprocessing["target"])
    for value in [[PREDICTIONS]]:
        print(value)'''
