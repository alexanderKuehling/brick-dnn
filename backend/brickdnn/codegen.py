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
    "LOSS",
    "CLASS_WEIGHTS",
    "FOCAL_GAMMA",
    "LABEL_SMOOTHING",
    "HUBER_DELTA",
    "class_weights",
    "focal_loss",
    "make_loss",
    # Logging to Brick DNN (brickdnn/tracking.py, copied into the script).
    "BrickDNNRun",
    "BRICKDNN_URL",
    "BRICKDNN_PROJECT",
    "json",
    "math",
    "os",
    "sys",
    "time",
    "urllib",
    "run",
    "summary",
    "squared",
    "test_rows",
    "scores",
    "model_settings",
    "x",
    "model",
    "out",
    "hidden",
    "args",
    "parser",
    "tensor",
    "i",
    # Training on a built-in text dataset (the generator of textsets.py is embedded).
    "random",
    "re",
    "DATASET",
    "TEXT_DATASETS",
    "text_dataset",
    "text_sample",
    "text_vocabulary",
    "text_examples",
    "tokenize",
    "encode",
    "decode",
    "load_dataset",
    "evaluate",
    "predict_text",
}
RESERVED = set(keyword.kwlist) | set(dir(builtins)) | NN_MODULE_ATTRIBUTES | SCRIPT_NAMES


# The settings of the loss that the training code of a script takes (see
# brickdnn.training.losses), with the names of the training settings.
LOSS_OPTIONS = ("loss", "class_weights", "focal_gamma", "label_smoothing", "huber_delta")
# The server a script logs to: no quotes, spaces or anything else that would break the code.
TRACKING_URL = re.compile(r"^https?://[A-Za-z0-9.\-]+(:[0-9]{1,5})?(/[A-Za-z0-9._~\-/]*)?$")


@dataclass
class GeneratorOptions:
    class_name: str = "NeuralNetwork"
    include_main: bool = True
    include_training: bool = True
    # The loss of the training code (like the training settings of Brick DNN): "auto" is
    # the default of the task (cross-entropy, or the mean squared error for one output).
    loss: str = "auto"
    class_weights: str | list[float] | dict[str, float] | None = None
    focal_gamma: float = 2.0
    label_smoothing: float = 0.0
    huber_delta: float = 1.0
    # Networks that read token ids: the built-in text dataset (textsets.py) whose generator
    # the training code embeds. None: the one the network fits (length, vocabulary, classes).
    dataset: str | None = None
    # The training of the script (the CSV loop, the training on a built-in text dataset, the
    # fit of a classical model, both phases of a hybrid network) logs its curves, settings and
    # final numbers to Brick DNN, but only while the environment variable BRICKDNN_TOKEN is
    # set (brickdnn/tracking.py): without it the script runs offline with the same results.
    # ``tracking_url`` is the default server, ``tracking_project`` the saved network the runs
    # belong to (both overridable with BRICKDNN_URL and BRICKDNN_PROJECT). The token is never
    # written into the script.
    log_to_brickdnn: bool = True
    tracking_url: str = SITE_URL
    tracking_project: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> GeneratorOptions:
        data = data or {}
        name = str(data.get("class_name") or cls.class_name).strip()
        if not name.isidentifier() or keyword.iskeyword(name):
            raise ValueError(f"'{name}' is not a valid Python class name.")
        if name in RESERVED:
            raise ValueError(f"'{name}' is already used in the generated code; pick another name.")
        from .training.config import TrainingConfig  # ConfigError is a ValueError

        loss = TrainingConfig.from_dict({key: data.get(key) for key in LOSS_OPTIONS})
        dataset = data.get("dataset")
        if dataset is not None:
            from .training.textsets import TEXT_DATASETS

            if dataset not in TEXT_DATASETS:
                raise ValueError(
                    f"'{dataset}' is not a built-in text dataset. Possible: "
                    f"{', '.join(TEXT_DATASETS)}."
                )
        url = data.get("tracking_url") or SITE_URL
        if not isinstance(url, str) or len(url) > 200 or not TRACKING_URL.match(url):
            raise ValueError(
                "'tracking_url' must be an http(s) address such as https://brickdnn.com."
            )
        project = data.get("tracking_project")
        if project is not None and (
            isinstance(project, bool) or not isinstance(project, int) or project < 1
        ):
            raise ValueError("'tracking_project' must be the id of a saved network.")
        return cls(
            class_name=name,
            include_main=bool(data.get("include_main", True)),
            include_training=bool(data.get("include_training", True)),
            **{key: getattr(loss, key) for key in LOSS_OPTIONS},
            dataset=dataset,
            log_to_brickdnn=bool(data.get("log_to_brickdnn", True)),
            tracking_url=url.rstrip("/"),
            tracking_project=project,
        )


def generate_code(graph: Any, options: GeneratorOptions | dict[str, Any] | None = None) -> str:
    """Generate PyTorch code for ``graph``, a scikit-learn script for a classical network
    (a Gaussian process, SVM or gradient boosting, see :mod:`brickdnn.classical`), or both
    for a hybrid network (neural layers that learn the features of a classical model, see
    :mod:`brickdnn.hybrid`).

    Raises :class:`GraphValidationError` if the graph contains errors.
    """
    if not isinstance(options, GeneratorOptions):
        options = GeneratorOptions.from_dict(options)
    analysis = analyze(graph)
    if not analysis.valid:
        raise GraphValidationError(analysis)
    if analysis.family == "classical":
        return generate_classical(analysis, options)
    if analysis.family == "hybrid":
        from .hybrid import generate_hybrid

        return generate_hybrid(analysis, options)
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
        wanted = self.opt.include_main and self.opt.include_training
        self.text = self._text_dataset() if wanted else None
        training = wanted and (self.text is not None or self._can_train())
        # The CSV loop and the training on a text dataset log to Brick DNN.
        self.logging = training and self.opt.log_to_brickdnn
        parts = [self._model_class()]
        if self.opt.include_main:
            parts.append(_COUNT_PARAMETERS)
            if self.text is not None:
                parts += [_text_generator(), self._text_training_code()]
            elif training:
                parts.append(self._training_code())
            if self.logging:
                parts.append(tracking_code(self.opt))
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
        if training and self.text is not None:
            from .training.datasets import BUILTIN_DATASETS

            name = BUILTIN_DATASETS[self.text].name
            lines += [
                "",
                f"Train it on {name}, a built-in text dataset of Brick DNN (generated below:",
                "the same texts as in Brick DNN), then classify new texts with the saved model:",
                "",
                "    python this_file.py --train --epochs 10",
                '    python this_file.py --predict "your text"',
            ]
        elif training:
            lines += [
                "",
                "Train it on a CSV file (every column but the last is a feature,",
                "the last column is the target), then predict new rows with the saved model:",
                "",
                "    python this_file.py --csv data.csv --epochs 20",
                "    python this_file.py --predict new_data.csv",
                "",
                "--test-split 0.2 holds out a fifth of the rows and measures the model on them",
                "once at the end.",
            ]
        if self.logging and self.text is not None:
            lines += tracking_docstring(
                self.opt,
                "python this_file.py --train --epochs 10",
                "It sends the loss and the accuracy of every epoch, the settings, the name of the "
                "dataset and the test accuracy",
            )
        elif self.logging:
            lines += tracking_docstring(
                self.opt,
                "python this_file.py --csv data.csv --epochs 20",
                "It sends the loss and the metrics of every epoch, the settings, the name of the "
                "CSV file and the test results",
            )
        lines.append('"""')
        return "\n".join(lines)

    def _imports(self, training: bool) -> str:
        lines = []
        if training and self.text is not None:
            lines += stdlib_imports(["argparse", "random", "re"], self.logging)
        elif training:
            lines += stdlib_imports(["argparse", "csv"], self.logging)
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
        if training and self.text is not None:
            return self._text_main()
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
                '        "--test-split", type=float, default=0.0, help="share of the rows held out '
                'as a test set"',
                "    )",
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
        if training and self.logging:
            task = "classification" if self._classification() else "regression"
            config = [
                '"epochs": args.epochs',
                '"batch_size": args.batch_size',
                '"learning_rate": args.lr',
                '"optimizer": "adam"',
                '"seed": args.seed',
                f'"loss": "{self._loss_name(task)}"',
                '"val_split": 0.2',
                '"test_split": args.test_split',
            ]
            lines += [
                "    if args.csv:",
                "        header, features, target = load_csv(args.csv)",
                *tracking_start(
                    name, config, task, "os.path.basename(args.csv)", "args.epochs", "        "
                ),
                "            mean, std = train(",
                "                model,",
                "                features,",
                "                target,",
                "                args.epochs,",
                "                args.batch_size,",
                "                args.lr,",
                "                args.seed,",
                "                args.test_split,",
                "                run,",
                "            )",
                "        save_model(model, args.save, header, mean, std)",
                "        return",
                "",
            ]
        elif training:
            lines += [
                "    if args.csv:",
                "        header, features, target = load_csv(args.csv)",
                "        mean, std = train(",
                "            model,",
                "            features,",
                "            target,",
                "            args.epochs,",
                "            args.batch_size,",
                "            args.lr,",
                "            args.seed,",
                "            args.test_split,",
                "        )",
                "        save_model(model, args.save, header, mean, std)",
                "        return",
                "",
            ]
        lines += self._sanity_check("    ")
        lines += ["", "", 'if __name__ == "__main__":', "    main()"]
        return "\n".join(lines)

    def _text_main(self) -> str:
        name = self.opt.class_name
        lines = [
            "def main() -> None:",
            f'    parser = argparse.ArgumentParser(description="{name} generated with Brick DNN")',
            "    parser.add_argument(",
            '        "--train", action="store_true", help=f"train on the built-in dataset {DATASET}"',
            "    )",
            '    parser.add_argument("--predict", metavar="TEXT", help="classify this text")',
            '    parser.add_argument("--epochs", type=int, default=10)',
            '    parser.add_argument("--batch-size", type=int, default=64)',
            '    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")',
            '    parser.add_argument("--seed", type=int, default=0, help="for a repeatable run")',
            "    parser.add_argument(",
            '        "--save",',
            '        default="model.pt",',
            '        help="the trained model (--train writes it, --predict reads it)",',
            "    )",
            "    args = parser.parse_args()",
            "    torch.manual_seed(args.seed)  # the same initial weights for the same seed",
            "",
            f"    model = {name}()",
            "    if args.predict:",
            "        predict_text(model, args.save, args.predict)",
            "        return",
            "    print(model)",
            '    print(f"Trainable parameters: {count_parameters(model):,}")',
            "",
            *self._text_train_call(),
            *self._sanity_check("    "),
            "",
            "",
            'if __name__ == "__main__":',
            "    main()",
        ]
        return "\n".join(lines)

    def _text_train_call(self) -> list[str]:
        if not self.logging:
            return [
                "    if args.train:",
                "        train(model, args.epochs, args.batch_size, args.lr, args.seed)",
                "        save_model(model, args.save)",
                "        return",
                "",
            ]
        from .training.datasets import BUILTIN_DATASETS

        config = [
            '"epochs": args.epochs',
            '"batch_size": args.batch_size',
            '"learning_rate": args.lr',
            '"optimizer": "adam"',
            '"seed": args.seed',
            f'"loss": "{self._loss_name("classification")}"',
            '"val_split": 0.2',
        ]
        dataset = f'"{BUILTIN_DATASETS[self.text].name}"'  # type: ignore[index]
        return [
            "    if args.train:",
            *tracking_start(
                self.opt.class_name, config, "classification", dataset, "args.epochs", "        "
            ),
            "            train(model, args.epochs, args.batch_size, args.lr, args.seed, run)",
            "        save_model(model, args.save)",
            "        return",
            "",
        ]

    # -- training --------------------------------------------------------- #

    def _text_dataset(self) -> str | None:
        """The built-in text dataset the training code embeds, or None. Only for a network
        with one Input of token ids (rank 1) that only feeds Embedding layers, and one
        output of K values. ``options.dataset`` must fit it; without it, the first dataset
        of its length whose words its Embeddings know, with as many classes if possible."""
        from .training.textsets import TEXT_DATASETS

        if len(self.a.inputs) != 1 or len(self.a.outputs) != 1:
            return None
        source = self._node(self.a.inputs[0])
        target = self._node(self.a.outputs[0])
        if source.dtype != "long" or source.shape is None or len(source.shape) != 1:
            return None
        if target.shape is None or len(target.shape) != 1 or target.shape[0] < 2:
            return None
        sizes = [
            self._node(t).layer.p["num_embeddings"]  # type: ignore[union-attr]
            for t in source.outputs
            if self._node(t).type == "Embedding"
        ]
        if not sizes or len(sizes) != len(source.outputs):
            return None
        outputs = target.shape[0]

        def fits(dataset: str) -> bool:
            _, vocabulary, length, classes, *_ = TEXT_DATASETS[dataset]
            return (
                source.shape == (length,)
                and outputs >= len(classes)
                and min(sizes) >= len(vocabulary)
            )

        wanted = self.opt.dataset
        if wanted is not None:
            if not fits(wanted):
                _, vocabulary, length, classes, *_ = TEXT_DATASETS[wanted]
                raise ValueError(
                    f"The network does not fit the built-in dataset '{wanted}': it needs an "
                    f"Input of {length} token ids, an Embedding with at least "
                    f"{len(vocabulary)} entries and {len(classes)} outputs."
                )
            return wanted
        fitting = [d for d in TEXT_DATASETS if fits(d)]
        same = [d for d in fitting if outputs == len(TEXT_DATASETS[d][3])]
        return (same or fitting or [None])[0]

    def _text_training_code(self) -> str:
        n_outputs = self._node(self.a.outputs[0]).shape[0]  # type: ignore[index]
        header = (
            f'DATASET = "{self.text}"  # the built-in text dataset (see TEXT_DATASETS above)\n'
            f"NUM_CLASSES = {n_outputs}"
        )
        loss = "nn.CrossEntropyLoss()"
        code = self._classification_loss(n_outputs)
        if code is not None:
            header += "\n" + code
            loss = "make_loss(train_y, device)"
        if self.logging:
            signature = (
                "def train(\n"
                "    model: nn.Module,\n"
                "    epochs: int,\n"
                "    batch_size: int,\n"
                "    lr: float,\n"
                "    seed: int,\n"
                "    run: BrickDNNRun | None = None,\n"
                ") -> None:"
            )
            log_init = "    summary: dict[str, float] = {}\n"
            log_epoch = (
                "        summary = {\n"
                '            "train_loss": train_loss / max(seen, 1),\n'
                '            "val_loss": val_loss,\n'
                '            "val_accuracy": val_accuracy,\n'
                "        }\n"
                "        if run is not None:\n"
                "            run.log(epoch=epoch, **summary)\n"
            )
            test_loss = "test_loss"
            log_end = (
                "    if run is not None:\n"
                "        summary.update(test_loss=test_loss, test_accuracy=test_accuracy)\n"
                "        run.finish(summary=summary)\n"
            )
        else:
            signature = (
                "def train(model: nn.Module, epochs: int, batch_size: int, lr: float, seed: int) "
                "-> None:"
            )
            log_init = log_epoch = log_end = ""
            test_loss = "_"
        return (
            _TEXT_TRAINING_TEMPLATE.replace("[[HEADER]]", header)
            .replace("[[LOSS]]", loss)
            .replace("[[SIGNATURE]]", signature)
            .replace("[[LOG_INIT]]", log_init)
            .replace("[[LOG_EPOCH]]", log_epoch)
            .replace("[[TEST_LOSS]]", test_loss)
            .replace("[[LOG_END]]", log_end)
        )

    # -- training (CSV) ----------------------------------------------------- #

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

    def _classification(self) -> bool:
        return self._node(self.a.outputs[0]).shape[0] > 1  # type: ignore[index]

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
            code = self._classification_loss(n_outputs)
            if code is not None:
                header += "\n" + code
                loss = "make_loss(target[train_rows], device)"
            metric_init = "        val_loss, correct = 0.0, 0\n"
            metric_update = (
                "                correct += (outputs.argmax(dim=1) == targets).sum().item()\n"
            )
            metric_print = "  val acc {correct / len(val_set):.1%}"
            predictions = "outputs.argmax(dim=1).tolist()"
            val_metric = '"val_accuracy": correct / len(val_set)'
            test_metric = (
                '            summary["test_accuracy"] = '
                "(outputs.argmax(dim=1) == targets).float().mean().item()\n"
            )
        else:
            header = (
                f"NUM_FEATURES = {n_features}\n"
                "# regression: the model has a single output, the target column is a number"
            )
            targets = "    return header, features, target.unsqueeze(1)"
            task = '        "task": "regression",'
            loss = "nn.MSELoss()"
            code = self._regression_loss()
            if code is not None:
                header += "\n" + code
                loss = "make_loss(target[train_rows])"
            metric_init = "        val_loss, squared = 0.0, 0.0\n"
            metric_update = "                squared += ((outputs - targets) ** 2).sum().item()\n"
            metric_print = ""
            predictions = "outputs[:, 0].tolist()"
            val_metric = '"val_rmse": (squared / len(val_set)) ** 0.5'
            test_metric = (
                '            summary["test_rmse"] = '
                "((outputs - targets) ** 2).mean().sqrt().item()\n"
            )

        if self.logging:
            run_param = "\n    run: BrickDNNRun | None = None,"
            log_epoch = "        if run is not None:\n            run.log(epoch=epoch, **summary)\n"
            log_end = "    if run is not None:\n        run.finish(summary=summary)\n"
        else:
            run_param = log_epoch = log_end = ""
        return (
            _TRAINING_TEMPLATE.replace("[[RUN_PARAM]]", run_param)
            .replace("[[LOG_EPOCH]]", log_epoch)
            .replace("[[LOG_END]]", log_end)
            .replace("[[VAL_METRIC]]", val_metric)
            .replace("[[TEST_METRIC]]", test_metric)
            .replace("[[HEADER]]", header)
            .replace("[[TARGETS]]", targets)
            .replace("[[TASK]]", task)
            .replace("[[LOSS]]", loss)
            .replace("[[METRIC_INIT]]", metric_init)
            .replace("[[METRIC_UPDATE]]", metric_update)
            .replace("[[METRIC_PRINT]]", metric_print)
            .replace("[[PREDICTIONS]]", predictions)
        )

    def _loss_name(self, task: str) -> str:
        from .training.losses import TASK_LOSSES, loss_for

        name = self.opt.loss
        if name != "auto" and name not in TASK_LOSSES[task]:
            if task == "regression" and name == "cross_entropy":
                return "mse"  # the old default of every task
            losses = ", ".join(["auto", *TASK_LOSSES[task]])
            output = "one output" if task == "regression" else "one output per class"
            raise ValueError(
                f"The loss '{name}' does not fit the training code of this network "
                f"({task}: {output}). Possible: {losses}."
            )
        return loss_for(name, task)

    def _classification_loss(self, classes: int) -> str | None:
        """The loss settings and functions of a classification script, or None for the
        plain cross-entropy."""
        name = self._loss_name("classification")
        smoothing = self.opt.label_smoothing if name != "focal" else 0.0
        if name == "cross_entropy" and smoothing == 0.0:
            return None
        weights: Any = self.opt.class_weights if name != "cross_entropy" else None
        if weights is None and name == "weighted_cross_entropy":
            weights = "balanced"
        if isinstance(weights, dict):
            vector = [1.0] * classes
            for key, value in weights.items():
                if not (key.isdigit() and int(key) < classes):
                    raise ValueError(
                        f"Class weights: '{key}' is not a class of the script, whose classes "
                        f"are the indices 0 to {classes - 1}."
                    )
                vector[int(key)] = value
            weights = vector
        if isinstance(weights, list) and len(weights) != classes:
            raise ValueError(
                f"Class weights: give one weight per class ({classes}), not {len(weights)}."
            )
        lines = [
            "# The loss: cross_entropy, weighted_cross_entropy or focal (as in Brick DNN).",
            f'LOSS = "{name}"',
            '# "balanced" (from the class counts of the training rows), a weight per class, or',
            "# None. The focal loss uses them as alpha.",
            f"CLASS_WEIGHTS = {_literal(weights)}",
            f"FOCAL_GAMMA = {float(self.opt.focal_gamma)!r}  # focal: 0 is the cross-entropy",
            f"LABEL_SMOOTHING = {float(smoothing)!r}  # the cross-entropies",
        ]
        return "\n".join(lines) + "\n\n\n" + _CLASSIFICATION_LOSS

    def _regression_loss(self) -> str | None:
        """The loss settings of a regression script, or None for the mean squared error."""
        name = self._loss_name("regression")
        if name == "mse":
            return None
        lines = [
            "# The loss: mse, mae or huber (as in Brick DNN).",
            f'LOSS = "{name}"',
            "# huber: where the loss turns from squared to linear, in standard deviations of",
            "# the target (Brick DNN standardises the target; this script scales delta).",
            f"HUBER_DELTA = {float(self.opt.huber_delta)!r}",
        ]
        return "\n".join(lines) + "\n\n\n" + _REGRESSION_LOSS


# --------------------------------------------------------------------------- #
# Logging to Brick DNN: shared by every script that trains (also classical.py, hybrid.py)
# --------------------------------------------------------------------------- #

# The standard library modules of the embedded client (brickdnn/tracking.py).
TRACKING_MODULES = ("json", "math", "os", "sys", "time", "urllib.error", "urllib.request")


def stdlib_imports(modules: list[str], logging: bool) -> list[str]:
    """The ``import`` lines of a script's standard library modules (with those of the
    embedded client when it logs, which also needs postponed annotations), sorted, and a
    blank line after them."""
    names = sorted(set(modules) | (set(TRACKING_MODULES) if logging else set()))
    future = ["from __future__ import annotations", ""] if logging else []
    return [*future, *(f"import {name}" for name in names), ""]


def tracking_code(options: GeneratorOptions) -> str:
    """The client of brickdnn/tracking.py (one copy per script), with the server and the
    saved network the runs belong to."""
    import inspect

    from .tracking import BrickDNNRun

    project = str(options.tracking_project) if options.tracking_project else "None"
    return (
        _TRACKING_HEADER.replace("[[URL]]", options.tracking_url).replace("[[PROJECT]]", project)
        + "\n\n\n"
        + inspect.getsource(BrickDNNRun).rstrip()
    )


def tracking_docstring(
    options: GeneratorOptions,
    command: str,
    sends: str,
    intro: str = "Log the training to Brick DNN, to see its curves",
) -> list[str]:
    """The paragraph of a script's docstring that says how it logs to Brick DNN:
    ``command`` trains, ``sends`` says what it sends (without the full stop)."""
    import textwrap

    text = (
        f"{intro} next to the runs trained there (the runs page): create an API token on the "
        "page API tokens (one that may only log runs is enough) and set it in the "
        "environment before you train:"
    )
    rest = (
        f"{sends} to BRICKDNN_URL (default {options.tracking_url}). Without BRICKDNN_TOKEN "
        "nothing is sent and the script runs offline with the same results; a network error "
        "never stops the training. The token is never stored in this file."
    )
    return [
        "",
        *textwrap.wrap(text, width=88),
        "",
        "    export BRICKDNN_TOKEN=<your token>",
        f"    {command}",
        "",
        *textwrap.wrap(rest, width=88, break_long_words=False, break_on_hyphens=False),
    ]


def tracking_start(
    name: str, config: list[str], task: str, dataset: str, epochs: str | None, indent: str
) -> list[str]:
    """``with BrickDNNRun.start(...) as run:`` of a script's main(), at ``indent``: the
    run's name, the entries of its config (code: ``'"seed": args.seed'``), its task, the
    code of the dataset's name and of the number of epochs (None: no epochs)."""
    return [
        f"{indent}# Logged to Brick DNN only with the environment variable BRICKDNN_TOKEN.",
        f"{indent}with BrickDNNRun.start(",
        f'{indent}    name="{name}",',
        f"{indent}    config={{",
        *[f"{indent}        {entry}," for entry in config],
        f"{indent}    }},",
        f"{indent}    project=BRICKDNN_PROJECT,",
        f"{indent}    dataset={dataset},",
        f'{indent}    task="{task}",',
        *([f"{indent}    epochs={epochs},"] if epochs is not None else []),
        f"{indent}    url=BRICKDNN_URL,",
        f"{indent}) as run:",
    ]


def _text_generator() -> str:
    """The generator of the built-in text datasets (textsets.py below its imports: only the
    standard library), so that the script makes the same texts as Brick DNN."""
    from pathlib import Path

    source = (Path(__file__).parent / "training" / "textsets.py").read_text(encoding="utf-8")
    body = source[source.index("TEXT_PAD = ") :].strip()
    return (
        "# --------------------------------------------------------------------------- #\n"
        "# The built-in text datasets of Brick DNN (brickdnn/training/textsets.py)\n"
        "# --------------------------------------------------------------------------- #\n"
        "\n" + body
    )


def _literal(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, str):
        return f'"{value}"'
    return "[" + ", ".join(f"{float(v):g}" for v in value) + "]"


_CLASSIFICATION_LOSS = '''\
def class_weights(target: torch.Tensor) -> torch.Tensor | None:
    """The weight of every class. "balanced": n / (k * count) from the class counts of the
    training rows (k: the classes that occur; a class without rows gets 1)."""
    if CLASS_WEIGHTS is None:
        return None
    if CLASS_WEIGHTS != "balanced":
        return torch.tensor(CLASS_WEIGHTS, dtype=torch.float32)
    counts = torch.bincount(target, minlength=NUM_CLASSES).float()
    weights = len(target) / (int((counts > 0).sum()) * counts.clamp(min=1))
    weights[counts == 0] = 1.0
    return weights


def focal_loss(
    outputs: torch.Tensor, targets: torch.Tensor, gamma: float, weight: torch.Tensor | None
) -> torch.Tensor:
    """The focal loss: the mean of -alpha * (1 - p)^gamma * log(p), p the probability of the
    true class, alpha its class weight (1 without weights)."""
    log_p = torch.log_softmax(outputs, dim=1).gather(1, targets.unsqueeze(1)).squeeze(1)
    loss = -((1 - log_p.exp()).clamp(min=0) ** gamma) * log_p
    if weight is not None:
        loss = loss * weight[targets]
    return loss.mean()


def make_loss(target: torch.Tensor, device: str):
    """The loss function, with the class weights of the training targets."""
    weight = class_weights(target)
    if weight is not None:
        weight = weight.to(device)
        print("Class weights:", [round(w, 4) for w in weight.tolist()])
    if LOSS == "focal":
        return lambda outputs, targets: focal_loss(outputs, targets, FOCAL_GAMMA, weight)
    if LOSS == "cross_entropy":
        weight = None
    return nn.CrossEntropyLoss(weight=weight, label_smoothing=LABEL_SMOOTHING)'''


_REGRESSION_LOSS = '''\
def make_loss(target: torch.Tensor):
    """The loss function. The target is not standardised here, so Huber's delta is scaled
    by its standard deviation on the training rows."""
    if LOSS == "mae":
        return nn.L1Loss()
    if LOSS == "huber":
        return nn.HuberLoss(delta=HUBER_DELTA * max(float(target.std()), 1e-8))
    return nn.MSELoss()'''


_COUNT_PARAMETERS = """\
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)"""


_TEXT_TRAINING_TEMPLATE = '''\
# --------------------------------------------------------------------------- #
# Training on the built-in text dataset, and predictions for new texts
# --------------------------------------------------------------------------- #

[[HEADER]]


def load_dataset() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The token ids and classes of the training texts and of the test texts."""
    *_, train_size, test_size = TEXT_DATASETS[DATASET]
    ids, labels = text_dataset(DATASET, 0, train_size + test_size)
    x, y = torch.tensor(ids), torch.tensor(labels)
    return x[:train_size], y[:train_size], x[train_size:], y[train_size:]


def evaluate(model: nn.Module, loader: DataLoader, loss_fn, device: str) -> tuple[float, float]:
    """The mean loss and the accuracy on the texts of ``loader``."""
    model.eval()
    total, correct, seen = 0.0, 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            total += loss_fn(outputs, targets).item() * len(inputs)
            correct += (outputs.argmax(dim=1) == targets).sum().item()
            seen += len(inputs)
    return total / max(seen, 1), correct / max(seen, 1)


[[SIGNATURE]]
    """Train on 80 % of the training texts (the split depends only on ``seed``), validate
    on the rest after every epoch and measure the test texts once at the end."""
    x, y, test_x, test_y = load_dataset()
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(x), generator=generator)
    n_val = len(x) // 5
    val_rows, train_rows = order[:n_val], order[n_val:]
    train_y = y[train_rows]
    train_loader = DataLoader(
        TensorDataset(x[train_rows], train_y),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(TensorDataset(x[val_rows], y[val_rows]), batch_size=256)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=256)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(
        f"Training on {device}: {len(train_rows)} texts of {DATASET}, {n_val} for validation, "
        f"{len(test_x)} for the test"
    )
    model.to(device)
    loss_fn = [[LOSS]]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

[[LOG_INIT]]    for epoch in range(1, epochs + 1):
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
        val_loss, val_accuracy = evaluate(model, val_loader, loss_fn, device)
        print(
            f"Epoch {epoch:3d}/{epochs}  train loss {train_loss / max(seen, 1):.4f}"
            f"  val loss {val_loss:.4f}  val acc {val_accuracy:.1%}"
        )
[[LOG_EPOCH]]    [[TEST_LOSS]], test_accuracy = evaluate(model, test_loader, loss_fn, device)
    print(f"Test accuracy: {test_accuracy:.1%}")
[[LOG_END]]    model.cpu()


def save_model(model: nn.Module, path: str) -> None:
    """Save the weights together with what predict_text() needs to read new texts."""
    _, vocabulary, length, class_names, *_ = TEXT_DATASETS[DATASET]
    preprocessing = {
        "dataset": DATASET,
        "vocabulary": list(vocabulary),  # token id i is vocabulary[i]
        "length": length,  # texts are cut or padded to this many tokens
        "class_names": list(class_names),
    }
    torch.save({"state_dict": model.state_dict(), "preprocessing": preprocessing}, path)
    print(f"Saved the weights and the vocabulary to {path}")


def predict_text(model: nn.Module, path: str, text: str) -> None:
    """Print the class of a text and the probability of every class."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    preprocessing = checkpoint["preprocessing"]
    vocabulary = tuple(preprocessing["vocabulary"])
    ids = encode(tokenize(text), vocabulary, preprocessing["length"])
    print("Tokens:", " ".join(decode(ids, vocabulary)))
    with torch.no_grad():
        probabilities = model(torch.tensor([ids])).softmax(dim=1)[0]
    names = preprocessing["class_names"]
    for index in probabilities.argsort(descending=True).tolist():
        name = names[index] if index < len(names) else f"class {index}"
        print(f"{name}: {probabilities[index].item():.1%}")'''


_TRACKING_HEADER = """\
# --------------------------------------------------------------------------- #
# Logging to Brick DNN, only with the environment variable BRICKDNN_TOKEN
# --------------------------------------------------------------------------- #

# The server and the saved network the runs belong to (BRICKDNN_URL and BRICKDNN_PROJECT
# override them). The token is never stored here: it is read from BRICKDNN_TOKEN.
BRICKDNN_URL = "[[URL]]"
BRICKDNN_PROJECT = [[PROJECT]]"""


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
    test_split: float = 0.0,[[RUN_PARAM]]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Train on a random 80 % of the rows and validate on the rest (the split depends only
    on ``seed``). With ``test_split``, that share of the rows is held out as a test set and
    measured once at the end.

    Returns the mean and standard deviation that standardise the features. They come from
    the training rows only, and new data has to be standardised with them as well.
    """
    if len(features) < 4:
        raise SystemExit("The CSV file needs at least 4 rows (training + validation).")
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(features), generator=generator)
    n_val = max(1, len(features) // 5)
    n_test = int(len(features) * test_split)
    if not 0 <= test_split < 1 or len(features) - n_val - n_test < 2:
        raise SystemExit("The test split leaves too few rows for training.")
    val_rows, test_rows = order[:n_val], order[n_val : n_val + n_test]
    train_rows = order[n_val + n_test :]

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
    if n_test:
        print(f"{n_test} rows are held out for the test")
    model.to(device)

    loss_fn = [[LOSS]]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    summary: dict[str, float] = {}
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
        summary = {
            "train_loss": train_loss / max(seen, 1),
            "val_loss": val_loss / len(val_set),
            [[VAL_METRIC]],
        }
[[LOG_EPOCH]]
    if n_test:
        model.eval()
        with torch.no_grad():
            inputs, targets = features[test_rows].to(device), target[test_rows].to(device)
            outputs = model(inputs)
            summary["test_loss"] = loss_fn(outputs, targets).item()
[[TEST_METRIC]]        tested = [f"{key[5:]} {value:.4f}" for key, value in summary.items() if "test" in key]
        print("Test:", ", ".join(tested))
[[LOG_END]]    model.cpu()
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
