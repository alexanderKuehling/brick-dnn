"""Turn an analysed classical network into a runnable scikit-learn script.

A classical network is an Input layer, one classical model (Gaussian process, support vector
machine or gradient boosting) and, for a Gaussian process, optionally kernel bricks; or several
such models combined by an ensemble (voting or stacking). The script defines
``make_model()``, a ``Pipeline`` that standardises the features and then fits the model,
and (with ``include_main``) a command line: fit it to a CSV file, save it with joblib and
predict new rows. Like the rest of :mod:`brickdnn`, this module only writes text: it needs
neither scikit-learn nor NumPy.

The saved file (``--save model.joblib``) is a dictionary; the training on the server
writes the same format, so the script can also predict with a model trained there::

    {"format": "brickdnn-sklearn-1", "model": <fitted Pipeline>, "task": "regression",
     "columns": ["rooms", ...], "target": "price", "class_names": None}

``class_names`` (classification) has one name per column of ``model.predict_proba``.
"""

from __future__ import annotations

import textwrap
from collections import Counter
from typing import TYPE_CHECKING

from .layers import ClassicalModel, Ensemble, KernelExpr, Member
from .links import BANNER

if TYPE_CHECKING:
    from .codegen import GeneratorOptions
    from .graph import Analysis

MODEL_FORMAT = "brickdnn-sklearn-1"
LINE_LENGTH = 100
# Names used inside make_model(); kernel variables must not take them.
MAKE_MODEL_NAMES = {"kernel", "model", "svc", "Pipeline", "StandardScaler"}


def model_of(analysis: Analysis) -> ClassicalModel:
    model = analysis.model
    assert model is not None and isinstance(model.layer, ClassicalModel)
    return model.layer


def num_features(analysis: Analysis) -> int:
    """The length of the feature vector: the shape of the Input that feeds the model."""
    model = analysis.model
    assert model is not None
    if isinstance(model.layer, Ensemble):  # its models share the Input
        model = analysis.nodes[model.layer.members[0].node]
    source = next(s for s in model.inputs if analysis.nodes[s].type == "Input")
    shape = analysis.nodes[source].shape
    assert shape is not None
    return shape[0]


def members(layer: ClassicalModel) -> list[Member]:
    """The models of an ensemble (none for a single model)."""
    return layer.members if isinstance(layer, Ensemble) else []


def hard_voting(layer: ClassicalModel) -> bool:
    """A VotingClassifier with hard voting: it predicts classes, but no probabilities."""
    return layer.type == "VotingClassifier" and layer.p["voting"] == "hard"


def boosted(layer: ClassicalModel) -> bool:
    """Gradient boosting (HistGradientBoostingRegressor or ...Classifier)."""
    return layer.scale == "gradient_boosting"


def calibrated(layer: ClassicalModel) -> bool:
    """SVC gets its class probabilities from Platt scaling (CalibratedClassifierCV)."""
    return layer.type == "SVC"


# --------------------------------------------------------------------------- #
# Code helpers
# --------------------------------------------------------------------------- #


def _call(target: str, func: str, args: list[str], indent: str = "    ") -> list[str]:
    """``target = func(args)`` on one line if it fits, otherwise one argument per line."""
    line = f"{indent}{target} = {func}({', '.join(args)})"
    if len(line) <= LINE_LENGTH:
        return [line]
    return [f"{indent}{target} = {func}(", *[f"{indent}    {arg}," for arg in args], f"{indent})"]


def _import(module: str, names: list[str]) -> list[str]:
    # isort order: CONSTANTS, Classes, functions
    names = sorted(set(names), key=lambda n: (not n.isupper(), not n[:1].isupper(), n))
    line = f"from {module} import {', '.join(names)}"
    if len(line) <= LINE_LENGTH:
        return [line]
    return [f"from {module} import (", *[f"    {name}," for name in names], ")"]


def _variable_names(
    leaves: list[KernelExpr], prefix: str = "", taken: set[str] | None = None
) -> list[str]:
    counts = Counter(leaf.hint for leaf in leaves)
    seen: Counter[str] = Counter()
    names = []
    for leaf in leaves:
        base = leaf.hint if leaf.hint not in MAKE_MODEL_NAMES else f"{leaf.hint}_kernel"
        seen[leaf.hint] += 1
        name = prefix + (base if counts[leaf.hint] == 1 else f"{base}_{seen[leaf.hint]}")
        while taken is not None and name in taken:
            name += "_k"
        names.append(name)
    return names


def kernel_lines(
    kernel: KernelExpr, target: str = "kernel", prefix: str = "", taken: set[str] | None = None
) -> list[str]:
    """``kernel = ...`` inside make_model(); long kernels get a variable per brick.
    ``target`` names the kernel, ``prefix`` starts the names of those variables, which avoid
    the names in ``taken`` (the models of an ensemble each have a kernel of their own)."""
    inline = f"    {target} = {kernel.code()}"
    if len(inline) <= LINE_LENGTH:
        return [inline]
    leaves = kernel.leaves()
    if len(leaves) == 1:
        return _call(target, kernel.name, list(kernel.args))
    names = _variable_names(leaves, prefix, taken)
    if taken is not None:
        taken.update(names)
    lines: list[str] = []
    for name, leaf in zip(names, leaves, strict=True):
        lines += _call(name, leaf.name, list(leaf.args))
    combined = f"    {target} = {kernel.code(names)}"
    if len(combined) <= LINE_LENGTH:
        return [*lines, combined]
    parts = [f"        {part}" for part in kernel.split(names)]
    return [*lines, f"    {target} = (", *parts, "    )"]


def kernel_text(kernel: KernelExpr, indent: str = "    ") -> list[str]:
    """The kernel expression wrapped for a docstring."""
    lines = []
    for part in kernel.split():
        lines += textwrap.wrap(
            part,
            width=LINE_LENGTH - 4,
            initial_indent=indent,
            subsequent_indent=indent + "    ",
            break_long_words=False,
            break_on_hyphens=False,
        )
    return lines


# --------------------------------------------------------------------------- #
# The script
# --------------------------------------------------------------------------- #


def generate_classical(analysis: Analysis, options: GeneratorOptions) -> str:
    return _ClassicalGenerator(analysis, options).render()


class _ClassicalGenerator:
    def __init__(self, analysis: Analysis, options: GeneratorOptions):
        self.a = analysis
        self.opt = options
        self.layer = model_of(analysis)
        self.members = members(self.layer)
        self.n_features = num_features(analysis)
        self.classification = self.layer.task == "classification"
        self.gp = self.layer.takes_kernel
        # The models of the network: the members of an ensemble, or the model itself.
        self.models = [m.layer for m in self.members] or [self.layer]

    def render(self) -> str:
        main = self.opt.include_main
        training = main and self.opt.include_training
        header = (
            BANNER + "\n" + self._docstring(main, training) + "\n\n" + self._imports(main, training)
        )
        if main:
            header += f"\n\nNUM_FEATURES = {self.n_features}"
        parts = [header, self._make_model()]
        if main:
            parts += [_READ_CSV, _NUMBERS]
            if training:
                parts += [self._load_csv(), self._fit(), self._save_model()]
            parts += [self._predict(), self._demo(), self._main(training)]
        return "\n\n\n".join(parts) + "\n"

    # -- header ------------------------------------------------------------ #

    def _docstring(self, main: bool, training: bool) -> str:
        layer = self.layer
        what = "the class of each row" if self.classification else "one number per row"
        features = f"{self.n_features} numeric feature{'s' if self.n_features != 1 else ''}"
        if self.members:
            about = (
                f"A scikit-learn ensemble for {features}: they are standardised, then "
                f"{layer.estimator} combines {len(self.members)} models ("
                + ", ".join(f"{m.name}: {m.layer.estimator}" for m in self.members)
                + f") and predicts {what}. "
                + self._combination()
            )
        else:
            about = (
                f"A scikit-learn model for {features}: they are standardised, then "
                f"{layer.estimator} predicts {what}."
            )
        if any(calibrated(model) for model in self.models):
            about += (
                " Platt scaling (5-fold cross-validation) turns the SVM's scores into class "
                "probabilities."
            )
        if boosted(layer):
            about += (
                " The trees do not need standardised features (their splits stay the same); "
                "the scaler keeps the saved model in the format of the other classical models."
            )
        lines = [
            f'"""{layer.label}: generated with Brick DNN.',
            "",
            *textwrap.wrap(about, width=LINE_LENGTH - 8),
        ]
        if layer.kernel is not None:
            lines += ["", "Kernel (the starting point; fitting optimises its hyperparameters):"]
            lines += kernel_text(layer.kernel)
        for member in self.members:
            if member.layer.kernel is not None:
                lines += [
                    "",
                    f"Kernel of {member.name} (the starting point; fitting optimises it):",
                ]
                lines += kernel_text(member.layer.kernel)
        if main:
            lines += [
                "",
                "Print the model and fit it to random data:",
                "",
                "    python this_file.py",
            ]
        if training:
            lines += [
                "",
                "Fit it to a CSV file (every column but the last is a feature, the last column is",
                "the target), then predict new rows with the saved model:",
                "",
                "    python this_file.py --csv data.csv",
                "    python this_file.py --predict new_data.csv",
            ]
        elif main:
            lines += [
                "",
                "Predict new rows with a fitted model (e.g. trained in Brick DNN):",
                "",
                "    python this_file.py --predict new_data.csv --save model.joblib",
            ]
        lines.append('"""')
        return "\n".join(lines)

    def _combination(self) -> str:
        """How an ensemble combines its models, in words."""
        layer = self.layer
        if layer.type == "VotingClassifier":
            how = (
                "Soft voting averages their class probabilities"
                if layer.p["voting"] == "soft"
                else "Hard voting takes the class most of them predict"
            )
        elif layer.type == "VotingRegressor":
            how = "It averages their predictions"
        else:
            final = layer.final_estimator  # type: ignore[attr-defined]
            how = (
                f"{final} learns from their predictions for rows they were not fitted on "
                f"({layer.p['cv']}-fold cross-validation) how to combine them"
            )
            if layer.p["passthrough"]:
                how += ", together with the features"
        if layer.p.get("weights"):
            how += ", weighted " + ":".join(str(w) for w in layer.p["weights"])
        return how + "."

    def _imports(self, main: bool, training: bool) -> str:
        lines = []
        if main:
            lines += ["import argparse", "import csv"]
            lines += ["import time", ""] if training else [""]
            lines += ["import joblib", "import numpy as np"]
        layer = self.layer
        modules: dict[str, list[str]] = {layer.module: [layer.estimator]}
        for model in self.models:
            modules.setdefault(model.module, []).append(model.estimator)
            if calibrated(model):
                modules["sklearn.calibration"] = ["CalibratedClassifierCV"]
            if model.kernel is not None:
                kernels = modules.setdefault("sklearn.gaussian_process.kernels", [])
                kernels += model.kernel.classes()
        if isinstance(layer, Ensemble) and layer.combines == "stacking":
            modules[layer.final_module] = [layer.final_estimator]  # type: ignore[attr-defined]
        modules["sklearn.pipeline"] = ["Pipeline"]
        modules["sklearn.preprocessing"] = ["StandardScaler"]
        for module in sorted(modules):
            lines += _import(module, modules[module])
        return "\n".join(lines)

    # -- the model --------------------------------------------------------- #

    def _make_model(self) -> str:
        layer = self.layer
        if self.members:
            return self._make_ensemble()
        lines = [
            "def make_model() -> Pipeline:",
            '    """Standardise the features (with the mean and standard deviation of the data',
            f'    it is fitted to), then fit {layer.estimator}."""',
        ]
        if layer.kernel is not None:
            lines += kernel_lines(layer.kernel)
        if calibrated(layer):
            lines += _call("svc", layer.estimator, layer.estimator_args())
            lines += [
                "    # Platt scaling: 5-fold cross-validation turns the scores into probabilities.",
                "    model = CalibratedClassifierCV(svc, ensemble=False)",
            ]
        else:
            lines += _call("model", layer.estimator, layer.estimator_args())
        lines.append('    return Pipeline([("scale", StandardScaler()), ("model", model)])')
        return "\n".join(lines)

    def _make_ensemble(self) -> str:
        """make_model() of an ensemble: a variable per model (named like the model in the
        ensemble), then the ensemble."""
        layer = self.layer
        lines = [
            "def make_model() -> Pipeline:",
            '    """Standardise the features (with the mean and standard deviation of the data',
            f"    it is fitted to), then fit {layer.estimator} with its "
            f'{len(self.members)} models."""',
        ]
        taken = {member.name for member in self.members} | MAKE_MODEL_NAMES

        def free(base: str) -> str:
            name = base
            while name in taken:
                name += "_"
            taken.add(name)
            return name

        for member in self.members:
            model, name = member.layer, member.name
            args = model.estimator_args()
            if model.kernel is not None:
                kernel = free(f"{name}_kernel")
                lines += kernel_lines(model.kernel, kernel, f"{name}_", taken)
                args = [f"kernel={kernel}" if arg == "kernel=kernel" else arg for arg in args]
            if calibrated(model):
                svm = free(f"{name}_svm")
                lines += _call(svm, model.estimator, args)
                lines += [
                    "    # Platt scaling: 5-fold cross-validation turns the scores into "
                    "probabilities.",
                    f"    {name} = CalibratedClassifierCV({svm}, ensemble=False)",
                ]
            else:
                lines += _call(name, model.estimator, args)
        pairs = [f'("{m.name}", {m.name})' for m in self.members]
        estimators = f"        estimators=[{', '.join(pairs)}],"
        lines.append(f"    model = {layer.estimator}(")
        if len(estimators) <= LINE_LENGTH:
            lines.append(estimators)
        else:
            lines += ["        estimators=[", *[f"            {p}," for p in pairs], "        ],"]
        lines += [f"        {arg}," for arg in layer.estimator_args()]
        lines += ["    )"]
        lines.append('    return Pipeline([("scale", StandardScaler()), ("model", model)])')
        return "\n".join(lines)

    # -- CSV training ------------------------------------------------------ #

    def _load_csv(self) -> str:
        if self.classification:
            target = "    target = np.array([row[-1].strip() for row in rows])  # the class labels"
        else:
            target = "    target = numbers([row[-1:] for row in rows], path)[:, 0]"
        return _LOAD_CSV.replace("[[TARGET]]", target)

    def _fit(self) -> str:
        return _FIT.replace("[[REPORT]]", self._report())

    def _report(self) -> str:
        """What the fitted ``model`` did on ``features[val_rows]`` and what it learned."""
        report = _CLASSIFICATION_REPORT if self.classification else _REGRESSION_REPORT
        if self.members:
            report += _ENSEMBLE_REPORT
            if isinstance(self.layer, Ensemble) and self.layer.combines == "stacking":
                report += _STACKING_REPORT
        elif self.gp:
            report += _GP_REPORT
        elif calibrated(self.layer):
            report += _SVC_REPORT
        elif boosted(self.layer):
            report += _BOOSTING_REPORT
        else:
            report += _SVR_REPORT
        return report

    def _save_model(self) -> str:
        if self.classification:
            task = (
                '        "task": "classification",\n'
                "        # one name per column of model.predict_proba()\n"
                '        "class_names": [str(name) for name in model.classes_],'
            )
        else:
            task = '        "task": "regression",\n        "class_names": None,'
        return _SAVE_MODEL.replace("[[FORMAT]]", MODEL_FORMAT).replace("[[TASK]]", task)

    def _predict(self) -> str:
        return _PREDICT.replace("[[OUTPUT]]", self._output())

    def _output(self) -> str:
        """Print the predictions of the fitted ``model`` for ``features``."""
        if hard_voting(self.layer):
            return _PREDICT_LABELS
        if self.classification:
            return _PREDICT_CLASSES
        if self.gp:
            return _PREDICT_WITH_STD
        return _PREDICT_VALUES

    def _demo(self) -> str:
        if self.classification:
            target = '    target = np.where(features[:, 0] + features[:, -1] > 0, "yes", "no")'
        else:
            target = "    target = features[:, 0] + np.sin(2 * features[:, -1])"
        return _DEMO.replace("[[TARGET]]", target)

    def _main(self, training: bool) -> str:
        name = self.layer.label
        description = f'description="{name}, generated with Brick DNN"'
        parser = f"    parser = argparse.ArgumentParser({description})"
        lines = ["def main() -> None:"]
        if len(parser) <= LINE_LENGTH:
            lines.append(parser)
        else:
            lines += ["    parser = argparse.ArgumentParser(", f"        {description}", "    )"]
        if training:
            lines.append(
                '    parser.add_argument("--csv", help="fit the model to this CSV file '
                '(last column = target)")'
            )
        lines.append(
            '    parser.add_argument("--predict", metavar="CSV", help="predict the rows of this file")'
        )
        if training:
            lines += [
                '    parser.add_argument("--seed", type=int, default=0, help="for a repeatable split")',
                "    parser.add_argument(",
                '        "--save",',
                '        default="model.joblib",',
                '        help="the fitted model (--csv writes it, --predict reads it)",',
                "    )",
            ]
        else:
            lines += [
                "    parser.add_argument(",
                '        "--save", default="model.joblib", help="the fitted model that --predict uses"',
                "    )",
            ]
        lines += [
            "    args = parser.parse_args()",
            "",
            "    if args.predict:",
            "        predict(args.save, args.predict)",
            "        return",
            "    model = make_model()",
            "    print(model)",
        ]
        if training:
            lines += [
                "    if args.csv:",
                "        header, features, target = load_csv(args.csv)",
                "        fit(model, features, target, args.seed)",
                "        save_model(model, args.save, header)",
                "        return",
            ]
        lines += ["    demo(model)", "", "", 'if __name__ == "__main__":', "    main()"]
        return "\n".join(lines)


_READ_CSV = '''\
# --------------------------------------------------------------------------- #
# CSV files: fitting, saving and predictions for new rows
# --------------------------------------------------------------------------- #


def read_csv(path: str) -> tuple[list[str], list[list[str]]]:
    """Read a CSV file: the column names of its header line (an empty list if it has none)
    and the rows (as text)."""
    with open(path, newline="") as f:
        rows = [row for row in csv.reader(f) if row]
    header: list[str] = []
    if rows:
        cells = rows[0][:-1] or rows[0]  # the last column (the target) may hold text
        try:
            [float(value) for value in cells]
        except ValueError:
            header, rows = [name.strip() for name in rows[0]], rows[1:]
    if not rows:
        raise SystemExit(f"{path} contains no data rows.")
    return header, rows'''

_NUMBERS = """\
def numbers(rows: list[list[str]], path: str) -> np.ndarray:
    try:
        return np.array([[float(value) for value in row] for row in rows])
    except ValueError as exc:
        raise SystemExit(f"{path}: {exc}") from None"""

_LOAD_CSV = '''\
def load_csv(path: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Training data: every column but the last is a feature, the last one is the target."""
    header, rows = read_csv(path)
    if any(len(row) != NUM_FEATURES + 1 for row in rows):
        raise SystemExit(f"Expected {NUM_FEATURES} feature columns and the target in every row.")
    features = numbers([row[:-1] for row in rows], path)
[[TARGET]]
    return header, features, target'''

_FIT = '''\
def fit(model: Pipeline, features: np.ndarray, target: np.ndarray, seed: int) -> None:
    """Fit on a random 80 % of the rows and validate on the rest (the split depends only on
    ``seed``). The StandardScaler of the pipeline learns from the training rows only."""
    if len(features) < 4:
        raise SystemExit("The CSV file needs at least 4 rows (training + validation).")
    order = np.random.default_rng(seed).permutation(len(features))
    n_val = max(1, len(features) // 5)
    val_rows, train_rows = order[:n_val], order[n_val:]
    print(f"Fitting on {len(train_rows)} rows, {len(val_rows)} for validation")
    started = time.perf_counter()
    model.fit(features[train_rows], target[train_rows])
    print(f"Fitted in {time.perf_counter() - started:.1f} s")
[[REPORT]]'''

_REGRESSION_REPORT = """\
    errors = model.predict(features[val_rows]) - target[val_rows]
    rmse = float(np.sqrt(np.mean(errors**2)))
    spread = float(np.sum((target[val_rows] - target[val_rows].mean()) ** 2))
    r2 = 1 - float(np.sum(errors**2)) / spread if spread > 0 else float("nan")
    print(f"Validation RMSE {rmse:.4g}, R² {r2:.3f}")"""

_CLASSIFICATION_REPORT = """\
    accuracy = float(np.mean(model.predict(features[val_rows]) == target[val_rows]))
    print(f"Validation accuracy {accuracy:.1%}")"""

_GP_REPORT = """
    gp = model.named_steps["model"]
    print(f"Learned kernel: {gp.kernel_}")
    print(f"Log marginal likelihood: {gp.log_marginal_likelihood_value_:.4g}")"""

_SVR_REPORT = """
    print(f"Support vectors: {len(model.named_steps['model'].support_)}")"""

_SVC_REPORT = """
    svc = model.named_steps["model"].calibrated_classifiers_[0].estimator
    print(f"Support vectors: {int(svc.n_support_.sum())}")"""

_ENSEMBLE_REPORT = """
    ensemble = model.named_steps["model"]
    print(f"Models: {', '.join(ensemble.named_estimators_)}")"""

_STACKING_REPORT = """
    print(f"Final estimator: {ensemble.final_estimator_}")"""

_BOOSTING_REPORT = """
    boosting = model.named_steps["model"]
    # Early stopping may end before max_iter.
    print(f"Boosting iterations: {boosting.n_iter_} of at most {boosting.max_iter}")"""

_SAVE_MODEL = '''\
def save_model(model: Pipeline, path: str, header: list[str]) -> None:
    """Save the fitted pipeline with what predict() needs to read new rows. joblib files
    are pickles: only load files you trust."""
    bundle = {
        "format": "[[FORMAT]]",
        "model": model,
        "columns": header[:-1],  # the feature columns by name (empty: no header line)
        "target": header[-1] if header else "target",
[[TASK]]
    }
    joblib.dump(bundle, path)
    print(f"Saved the fitted model to {path}")'''

_PREDICT = '''\
def predict(path: str, csv_path: str) -> None:
    """Print a prediction for every row of a CSV file.

    If both the saved model and the file have a header line, the feature columns are found
    by their names; otherwise they have to come first. Other columns (such as the target)
    are ignored. The pipeline standardises the features like the training data.
    """
    bundle = joblib.load(path)  # a pickle: only load files you trust
    model, columns = bundle["model"], bundle["columns"]
    header, rows = read_csv(csv_path)
    if columns and header:
        missing = [name for name in columns if name not in header]
        if missing:
            raise SystemExit(f"{csv_path} lacks the columns {', '.join(missing)}.")
        rows = [[row[header.index(name)] for name in columns] for row in rows]
    elif all(len(row) >= NUM_FEATURES for row in rows):
        rows = [row[:NUM_FEATURES] for row in rows]
    else:
        raise SystemExit(f"Expected {NUM_FEATURES} feature columns.")
    features = numbers(rows, csv_path)
[[OUTPUT]]'''

_PREDICT_VALUES = """\
    print(bundle["target"])
    for value in model.predict(features):
        print(f"{value:.6g}")"""

_PREDICT_WITH_STD = """\
    # The Gaussian process also says how uncertain it is: one standard deviation.
    mean, std = model.predict(features, return_std=True)
    print(f"{bundle['target']},std")
    for value, spread in zip(mean, std):
        print(f"{value:.6g},{spread:.6g}")"""

_PREDICT_CLASSES = """\
    probabilities = model.predict_proba(features)
    names = bundle["class_names"]
    print(f"{bundle['target']},probability")
    for row in probabilities:
        best = int(np.argmax(row))
        print(f"{names[best]},{row[best]:.4f}")"""

_PREDICT_LABELS = """\
    # Hard voting predicts a class, but no probabilities.
    names = dict(zip(model.classes_, bundle["class_names"]))
    print(bundle["target"])
    for label in model.predict(features):
        print(names[label])"""

_DEMO = '''\
def demo(model: Pipeline) -> None:
    """Fit the model to 40 random rows and predict two of them (shows that it works)."""
    rng = np.random.default_rng(0)
    features = rng.normal(size=(40, NUM_FEATURES))
[[TARGET]]
    model.fit(features, target)
    print("Predictions for 2 random rows:", model.predict(features[:2]))'''
