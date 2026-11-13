"""Classical models (scikit-learn): kernel and model bricks, the analysis rules and the
generated script."""

from __future__ import annotations

import ast
import shutil
import subprocess
import sys

import pytest

from brickdnn import EXAMPLES, REGISTRY, analyze, generate_code, layer_specs
from brickdnn.__main__ import main as brickdnn_main
from brickdnn.classical import kernel_lines
from brickdnn.layers import CATEGORIES, KernelExpr, combine

from .helpers import CLASSICAL_EXAMPLES

try:
    import sklearn
except ImportError:  # pragma: no cover - scikit-learn is optional for the package
    sklearn = None

needs_sklearn = pytest.mark.skipif(sklearn is None, reason="scikit-learn is not installed")

KERNELS = [
    "RBF",
    "Matern",
    "RationalQuadratic",
    "ExpSineSquared",
    "DotProduct",
    "WhiteKernel",
    "ConstantKernel",
]
COMBINATIONS = ["KernelSum", "KernelProduct"]
MODELS = [
    "GaussianProcessRegressor",
    "GaussianProcessClassifier",
    "SVR",
    "SVC",
    "HistGradientBoostingRegressor",
    "HistGradientBoostingClassifier",
]
BOOSTING = ["HistGradientBoostingRegressor", "HistGradientBoostingClassifier"]
# Tested in test_classical_ensembles.py.
ENSEMBLES = ["VotingClassifier", "VotingRegressor", "StackingClassifier", "StackingRegressor"]


def graph(nodes, edges):
    return {
        "nodes": [{"id": i, "type": t, "params": p} for i, t, p in nodes],
        "edges": [{"source": s, "target": t} for s, t in edges],
    }


def model_graph(model="SVR", params=None, features=4, kernel=None):
    """Input -> model, optionally with one kernel brick ``(type, params)``."""
    nodes = [("in", "Input", {"shape": [features]}), ("m", model, params or {})]
    edges = [("in", "m")]
    if kernel is not None:
        nodes.append(("k", kernel[0], kernel[1]))
        edges.append(("k", "m"))
    return graph(nodes, edges)


def errors(g):
    return [(e.node, e.message) for e in analyze(g).errors]


# --------------------------------------------------------------------------- #
# Catalogue
# --------------------------------------------------------------------------- #


def test_every_classical_brick_is_covered():
    classical = {t for t, cls in REGISTRY.items() if cls.family == "classical"}
    assert classical == set(KERNELS + COMBINATIONS + MODELS + ENSEMBLES)


def test_catalogue_describes_kinds_and_slots():
    specs = {spec["type"]: spec for spec in layer_specs()}
    assert {"Kernels", "Classical models", "Ensembles"} <= set(CATEGORIES)
    for spec in specs.values():
        assert spec["family"] in ("neural", "classical", "shared")
        assert spec["output_kind"] in ("tensor", "kernel", "estimator")
        assert isinstance(spec["input_slots"], list)
    assert specs["Input"]["family"] == "shared" and specs["Input"]["input_slots"] == []
    assert specs["Linear"]["input_slots"] == [
        {"name": "input", "label": "Input", "kind": "tensor", "min": 1, "max": 1}
    ]
    for kernel in KERNELS:
        spec = specs[kernel]
        assert spec["category"] == "Kernels" and spec["output_kind"] == "kernel"
        assert spec["min_inputs"] == spec["max_inputs"] == 0 and spec["input_slots"] == []
    for combination in COMBINATIONS:
        assert specs[combination]["input_slots"] == [
            {"name": "kernels", "label": "Kernels", "kind": "kernel", "min": 2, "max": None}
        ]
    for model in MODELS:
        spec = specs[model]
        # What a model passes on is itself (an estimator for an ensemble).
        assert spec["category"] == "Classical models" and spec["output_kind"] == "estimator"
        kinds = [slot["kind"] for slot in spec["input_slots"]]
        assert kinds == (["tensor", "kernel"] if model.startswith("Gaussian") else ["tensor"])
        assert spec["max_inputs"] == len(kinds)


# --------------------------------------------------------------------------- #
# Kernel expressions
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "kernel,params,code,count",
    [
        ("RBF", {}, "RBF(length_scale=1.0)", 1),
        (
            "RBF",
            {"hyperparameters": "fixed"},
            'RBF(length_scale=1.0, length_scale_bounds="fixed")',
            0,
        ),
        (
            "RBF",
            {"hyperparameters": "custom bounds", "length_scale_min": 0.1, "length_scale_max": 10},
            "RBF(length_scale=1.0, length_scale_bounds=(0.1, 10.0))",
            1,
        ),
        ("Matern", {"nu": "2.5", "length_scale": 2}, "Matern(length_scale=2.0, nu=2.5)", 1),
        ("RationalQuadratic", {}, "RationalQuadratic(length_scale=1.0, alpha=1.0)", 2),
        (
            "ExpSineSquared",
            {"periodicity": 3},
            "ExpSineSquared(length_scale=1.0, periodicity=3.0)",
            2,
        ),
        ("DotProduct", {}, "DotProduct(sigma_0=1.0)", 1),
        ("WhiteKernel", {"noise_level": 0.01}, "WhiteKernel(noise_level=0.01)", 1),
        ("ConstantKernel", {}, "ConstantKernel(constant_value=1.0)", 1),
    ],
)
def test_kernel_bricks(kernel, params, code, count):
    result = analyze(model_graph("GaussianProcessRegressor", kernel=(kernel, params)))
    assert result.valid, result.errors
    node = result.to_dict()["nodes"]["k"]
    assert node["output_kind"] == "kernel" and node["output_shape"] is None
    assert node["kernel"] == code and node["num_params"] == count
    model = result.to_dict()["nodes"]["m"]
    assert model["kernel"] == code and model["num_params"] == 0  # counted at the kernel
    assert result.total_params == count


def test_kernel_combinations_respect_precedence():
    g = graph(
        [
            ("in", "Input", {"shape": [2]}),
            ("a", "RBF", {}),
            ("b", "WhiteKernel", {}),
            ("sum", "KernelSum", {}),
            ("c", "ConstantKernel", {"hyperparameters": "fixed"}),
            ("prod", "KernelProduct", {}),
            ("gp", "GaussianProcessRegressor", {}),
        ],
        [
            ("in", "gp"),
            ("a", "sum"),
            ("b", "sum"),
            ("c", "prod"),
            ("sum", "prod"),
            ("prod", "gp"),
        ],
    )
    result = analyze(g)
    assert result.valid, result.errors
    data = result.to_dict()
    expected = (
        'ConstantKernel(constant_value=1.0, constant_value_bounds="fixed") * '
        "(RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1))"
    )
    assert data["nodes"]["prod"]["kernel"] == expected
    assert data["classical"] == {
        "node": "gp",
        "type": "GaussianProcessRegressor",
        "estimator": "GaussianProcessRegressor",
        "task": "regression",
        "kernel": expected,
        "members": None,
    }
    assert data["total_params"] == 2  # RBF and WhiteKernel; the constant is fixed
    assert data["family"] == "classical"


def test_default_kernels_and_parameter_counts():
    gpr = analyze(model_graph("GaussianProcessRegressor")).to_dict()
    assert gpr["nodes"]["m"]["kernel"] == (
        "ConstantKernel(constant_value=1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)"
    )
    assert gpr["nodes"]["m"]["num_params"] == 3 and gpr["total_params"] == 3
    assert gpr["nodes"]["m"]["output_shape"] == [1] and gpr["nodes"]["m"]["task"] == "regression"
    gpc = analyze(model_graph("GaussianProcessClassifier")).to_dict()
    assert (
        gpc["nodes"]["m"]["kernel"] == "ConstantKernel(constant_value=1.0) * RBF(length_scale=1.0)"
    )
    assert gpc["nodes"]["m"]["output_shape"] is None  # the classes come from the data
    assert gpc["nodes"]["m"]["task"] == "classification"
    svc = analyze(model_graph("SVC")).to_dict()
    assert svc["valid"] and svc["nodes"]["m"]["num_params"] is None
    assert svc["total_params"] is None  # support vectors are only known after fitting
    assert svc["nodes"]["m"]["kernel"] is None
    for model in BOOSTING:
        boosting = analyze(model_graph(model)).to_dict()
        assert boosting["valid"] and boosting["family"] == "classical"
        assert boosting["nodes"]["m"]["num_params"] is None  # the trees come from the fit
        assert boosting["total_params"] is None and boosting["nodes"]["m"]["kernel"] is None
        assert boosting["classical"]["estimator"] == model
        params = boosting["nodes"]["m"]["params"]
        assert params["max_depth"] is None and params["early_stopping"] == "auto"
        assert params["learning_rate"] == 0.1 and params["max_iter"] == 200
    assert analyze(model_graph(BOOSTING[0])).to_dict()["nodes"]["m"]["output_shape"] == [1]
    neural = analyze(EXAMPLES["mlp_mnist"]["graph"]).to_dict()
    assert neural["family"] == "neural" and neural["classical"] is None
    assert {n["output_kind"] for n in neural["nodes"].values()} == {"tensor"}


def test_kernel_lines_break_long_kernels():
    rbf = KernelExpr(name="RBF", args=("length_scale=1.0",), hint="rbf")
    short = combine(
        "+", [rbf, KernelExpr(name="WhiteKernel", args=("noise_level=0.1",), hint="white")]
    )
    assert kernel_lines(short) == [
        "    kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)"
    ]
    long_args = tuple(f"argument_{i}=1.0" for i in range(8))
    leaves = [KernelExpr(name="RBF", args=long_args, hint="rbf") for _ in range(3)]
    expr = combine("*", [combine("+", leaves[:2]), leaves[2]])
    lines = kernel_lines(expr)
    assert all(len(line) <= 100 for line in lines)
    assert lines[0] == "    rbf_1 = RBF("
    assert "    kernel = (rbf_1 + rbf_2) * rbf_3" in lines


# --------------------------------------------------------------------------- #
# Rules
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "g,node,message",
    [
        (model_graph("SVR", kernel=("RBF", {})), "m", "chooses its kernel with its own Kernel"),
        (
            graph(
                [("in", "Input", {"shape": [4]}), ("k", "RBF", {}), ("fc", "Linear", {})],
                [("in", "fc"), ("k", "fc")],
            ),
            "fc",
            "Linear cannot take a kernel",
        ),
        (
            graph(
                [("in", "Input", {"shape": [1, 8, 8]}), ("c", "Conv2d", {}), ("m", "SVC", {})],
                [("in", "c"), ("c", "m")],
            ),
            "m",
            "but Conv2d outputs (32, 8, 8). Add Flatten (or a Linear layer) before",
        ),
        (
            graph(
                [("in", "Input", {"shape": [4]}), ("m", "SVR", {}), ("fc", "Linear", {})],
                [("in", "m"), ("m", "fc")],
            ),
            "m",
            "Nothing can follow",
        ),
        (
            graph(
                [("in", "Input", {"shape": [4]}), ("m", "SVC", {}), ("fc", "Linear", {})],
                [("in", "m"), ("in", "fc")],
            ),
            "fc",
            "Linear does not lead to Support vector classification, which takes its features "
            "straight from the Input layer",
        ),
        (
            graph(
                [("in", "Input", {"shape": [4]}), ("a", "SVC", {}), ("b", "SVR", {})],
                [("in", "a"), ("in", "b")],
            ),
            "b",
            "Only one classical model",
        ),
        (
            graph([("in", "Input", {"shape": [1, 8, 8]}), ("m", "SVC", {})], [("in", "m")]),
            "m",
            "needs a flat feature vector",
        ),
        (
            graph(
                [("in", "Input", {"shape": [8], "dtype": "long"}), ("m", "SVC", {})],
                [("in", "m")],
            ),
            "m",
            "needs numeric features",
        ),
        (
            graph(
                [("in", "Input", {"shape": [4]}), ("k", "RBF", {}), ("m", "SVR", {})],
                [("in", "m")],
            ),
            "k",
            "not connected to a Gaussian process",
        ),
        (
            graph(
                [
                    ("in", "Input", {"shape": [4]}),
                    ("k", "RBF", {}),
                    ("a", "KernelSum", {}),
                    ("b", "KernelSum", {}),
                    ("w", "WhiteKernel", {}),
                    ("gp", "GaussianProcessRegressor", {}),
                ],
                [("in", "gp"), ("k", "a"), ("k", "b"), ("w", "a"), ("a", "gp")],
            ),
            "k",
            "can only be connected once",
        ),
        (
            graph(
                [
                    ("in", "Input", {"shape": [4]}),
                    ("k", "RBF", {}),
                    ("s", "KernelSum", {}),
                    ("gp", "GaussianProcessRegressor", {}),
                ],
                [("in", "gp"), ("k", "s"), ("s", "gp")],
            ),
            "s",
            "needs at least 2 kernels",
        ),
        (
            graph(
                [
                    ("in", "Input", {"shape": [4]}),
                    ("k", "RBF", {}),
                    ("s", "KernelSum", {}),
                    ("gp", "GaussianProcessRegressor", {}),
                ],
                [("in", "gp"), ("in", "s"), ("k", "s"), ("s", "gp")],
            ),
            "s",
            "combines kernels",
        ),
        (
            graph(
                [("in", "Input", {"shape": [4]}), ("k", "RBF", {}), ("k2", "RBF", {})],
                [("in", "k"), ("k", "k2")],
            ),
            "k",
            "has no inputs",
        ),
        (
            graph(
                [
                    ("in", "Input", {"shape": [4]}),
                    ("a", "RBF", {}),
                    ("b", "RBF", {}),
                    ("gp", "GaussianProcessClassifier", {}),
                ],
                [("in", "gp"), ("a", "gp"), ("b", "gp")],
            ),
            "gp",
            "takes one kernel",
        ),
        (
            graph([("k", "RBF", {}), ("gp", "GaussianProcessRegressor", {})], [("k", "gp")]),
            "gp",
            "has no features",
        ),
        (model_graph(kernel=None, params={"C": 0}), "m", "C must be greater than 0"),
        (
            model_graph("HistGradientBoostingClassifier", kernel=("RBF", {})),
            "m",
            "builds decision trees and takes no kernel",
        ),
        (
            model_graph("HistGradientBoostingRegressor", {"learning_rate": 0}),
            "m",
            "Learning rate must be greater than 0",
        ),
        (
            model_graph("HistGradientBoostingClassifier", {"max_iter": 0}),
            "m",
            "Max. iterations must be at least 1",
        ),
        (
            model_graph("HistGradientBoostingClassifier", {"max_leaf_nodes": 1}),
            "m",
            "Max. leaves per tree must be at least 2",
        ),
        (
            model_graph("HistGradientBoostingRegressor", {"max_depth": 0}),
            "m",
            "Max. depth must be at least 1",
        ),
        (
            model_graph("HistGradientBoostingRegressor", {"min_samples_leaf": 0}),
            "m",
            "Min. rows per leaf must be at least 1",
        ),
        (
            model_graph("HistGradientBoostingRegressor", {"l2_regularization": -1}),
            "m",
            "L2 regularisation must be at least 0",
        ),
        (
            model_graph("HistGradientBoostingClassifier", {"early_stopping": "yes"}),
            "m",
            "Early stopping must be one of: auto, on, off",
        ),
        (
            model_graph("HistGradientBoostingClassifier", {"class_weight": "heavy"}),
            "m",
            "Class weights must be one of: none, balanced",
        ),
        (
            model_graph("HistGradientBoostingClassifier", {"random_state": -1}),
            "m",
            "Random state must be at least 0",
        ),
        (
            model_graph("SVC", params={"gamma": "value", "gamma_value": -1}),
            "m",
            "Gamma value must be greater than 0",
        ),
        (
            model_graph("GaussianProcessRegressor", kernel=("RBF", {"length_scale": 1e6})),
            "k",
            "outside the bounds 1e-05 … 1e+05",
        ),
        (
            model_graph(
                "GaussianProcessRegressor",
                kernel=(
                    "RBF",
                    {
                        "hyperparameters": "custom bounds",
                        "length_scale_min": 2,
                        "length_scale_max": 3,
                    },
                ),
            ),
            "k",
            "outside its bounds 2 … 3",
        ),
        (
            model_graph(
                "GaussianProcessRegressor",
                kernel=(
                    "RBF",
                    {
                        "hyperparameters": "custom bounds",
                        "length_scale_min": 3,
                        "length_scale_max": 2,
                    },
                ),
            ),
            "k",
            "lower one below the upper one",
        ),
        (
            model_graph("GaussianProcessRegressor", kernel=("WhiteKernel", {"noise_level": 0})),
            "k",
            "Noise level must be greater than 0",
        ),
    ],
)
def test_invalid_classical_graphs(g, node, message):
    found = errors(g)
    assert any(n == node and message in m for n, m in found), found


def test_fixed_hyperparameters_may_lie_outside_the_default_bounds():
    g = model_graph(
        "GaussianProcessRegressor",
        kernel=("RBF", {"length_scale": 1e7, "hyperparameters": "fixed"}),
    )
    assert analyze(g).valid


def test_hidden_bounds_are_not_listed():
    data = analyze(model_graph("GaussianProcessRegressor", kernel=("RBF", {}))).to_dict()
    assert set(data["nodes"]["k"]["params"]) == {"length_scale", "hyperparameters"}
    data = analyze(
        model_graph(
            "GaussianProcessRegressor", kernel=("RBF", {"hyperparameters": "custom bounds"})
        )
    ).to_dict()
    assert "length_scale_min" in data["nodes"]["k"]["params"]


# --------------------------------------------------------------------------- #
# Generated scikit-learn script
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("options", [{}, {"include_main": False}, {"include_training": False}])
def test_script_is_valid_python(model, options):
    code = generate_code(model_graph(model), options)
    ast.parse(code)
    assert "def make_model() -> Pipeline:" in code
    assert code.startswith("# Built with Brick DNN: https://brickdnn.com")
    assert "import torch" not in code
    assert ("def main()" in code) == options.get("include_main", True)
    assert ("def fit(" in code) == (options == {})
    assert ("def predict(" in code) == options.get("include_main", True)


def test_script_builds_the_estimator():
    code = generate_code(model_graph("SVC", {"kernel": "poly", "class_weight": "balanced"}))
    assert (
        '    svc = SVC(kernel="poly", C=1.0, gamma="scale", degree=3, coef0=0.0, '
        'class_weight="balanced")'
    ) in code
    assert "    model = CalibratedClassifierCV(svc, ensemble=False)" in code
    code = generate_code(model_graph("SVR", {"kernel": "linear", "epsilon": 0.5}))
    assert '    model = SVR(kernel="linear", C=1.0, epsilon=0.5)' in code
    code = generate_code(model_graph("SVR", {"gamma": "value", "gamma_value": 0.25}))
    assert "gamma=0.25" in code
    code = generate_code(model_graph("GaussianProcessClassifier", {"n_restarts_optimizer": 3}))
    assert "n_restarts_optimizer=3, random_state=0" in code
    assert "from sklearn.gaussian_process.kernels import RBF, ConstantKernel" in code


def test_script_builds_gradient_boosting():
    code = generate_code(
        model_graph(
            "HistGradientBoostingClassifier",
            {"class_weight": "balanced", "max_depth": 6, "early_stopping": "on"},
        )
    )
    assert "from sklearn.ensemble import HistGradientBoostingClassifier" in code
    assert (
        "    model = HistGradientBoostingClassifier(\n"
        "        learning_rate=0.1,\n"
        "        max_iter=200,\n"
        "        max_leaf_nodes=31,\n"
        "        max_depth=6,\n"
        "        min_samples_leaf=20,\n"
        "        l2_regularization=0.0,\n"
        "        early_stopping=True,\n"
        '        class_weight="balanced",\n'
        "        random_state=0,\n"
        "    )\n"
    ) in code
    assert "CalibratedClassifierCV" not in code and "kernel" not in code
    assert "Boosting iterations:" in code
    code = generate_code(
        model_graph("HistGradientBoostingRegressor", {"early_stopping": "off", "random_state": 7})
    )
    assert "max_depth=None" in code and "early_stopping=False" in code
    assert "random_state=7" in code and "class_weight" not in code
    assert 'early_stopping="auto"' in generate_code(model_graph("HistGradientBoostingRegressor"))


@pytest.mark.skipif(shutil.which("ruff") is None, reason="ruff is not installed")
@pytest.mark.parametrize("example_id", CLASSICAL_EXAMPLES)
@pytest.mark.parametrize("options", [{}, {"include_main": False}, {"include_training": False}])
def test_script_is_lint_clean(example_id, options, tmp_path):
    target = tmp_path / "model.py"
    target.write_text(generate_code(EXAMPLES[example_id]["graph"], options))
    ruff = shutil.which("ruff")
    result = subprocess.run(
        [ruff, "check", "--isolated", "--select", "E,F,W,I", "--line-length", "100", str(target)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout
    result = subprocess.run(
        [ruff, "format", "--isolated", "--line-length", "100", "--check", str(target)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout


def run_script(code, tmp_path, *args):
    script = tmp_path / "model.py"
    script.write_text(code)
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=300,
    )


def write_regression_csv(path, n=60, header=True):
    import numpy as np

    rng = np.random.default_rng(3)
    x = rng.normal(size=(n, 4))
    y = 2 * x[:, 0] - x[:, 1] + 0.05 * rng.normal(size=n)
    lines = ["a,b,c,d,price"] if header else []
    lines += [",".join(f"{v:.5f}" for v in [*row, t]) for row, t in zip(x, y, strict=True)]
    path.write_text("\n".join(lines) + "\n")
    return x, y


def write_classification_csv(path, n=80):
    import numpy as np

    rng = np.random.default_rng(4)
    x = rng.normal(size=(n, 4))
    labels = np.where(x[:, 0] + x[:, 2] > 0, "spam", "ham")
    lines = ["a,b,c,d,label"]
    lines += [
        ",".join([*(f"{v:.5f}" for v in row), lab]) for row, lab in zip(x, labels, strict=True)
    ]
    path.write_text("\n".join(lines) + "\n")
    return x, labels


@needs_sklearn
@pytest.mark.parametrize("model", MODELS)
def test_script_demo_runs(model, tmp_path):
    result = run_script(generate_code(model_graph(model)), tmp_path)
    assert result.returncode == 0, result.stderr
    assert "Pipeline(" in result.stdout
    assert "Predictions for 2 random rows:" in result.stdout


@needs_sklearn
@pytest.mark.parametrize(
    "model", ["GaussianProcessRegressor", "SVR", "HistGradientBoostingRegressor"]
)
def test_script_fits_and_predicts_regression(model, tmp_path):
    import joblib

    write_regression_csv(tmp_path / "data.csv")
    # 48 training rows: smaller leaves than the default 20 rows.
    params = {"min_samples_leaf": 5} if model in BOOSTING else None
    code = generate_code(model_graph(model, params))
    result = run_script(code, tmp_path, "--csv", "data.csv", "--seed", "1")
    assert result.returncode == 0, result.stderr
    assert "Fitting on 48 rows, 12 for validation" in result.stdout
    assert "Validation RMSE" in result.stdout and "R²" in result.stdout
    if model == "SVR":
        assert "Support vectors:" in result.stdout
    elif model in BOOSTING:
        assert "Boosting iterations: 200 of at most 200" in result.stdout
    else:
        assert "Learned kernel:" in result.stdout and "Log marginal likelihood:" in result.stdout
    bundle = joblib.load(tmp_path / "model.joblib")
    assert bundle["format"] == "brickdnn-sklearn-1"
    assert bundle["task"] == "regression" and bundle["class_names"] is None
    assert bundle["columns"] == ["a", "b", "c", "d"] and bundle["target"] == "price"
    # The scaler learned from the training rows only (80 % of them, seeded).
    assert bundle["model"].named_steps["scale"].n_samples_seen_ == 48

    # New rows: columns by name, in another order, the target is ignored.
    (tmp_path / "new.csv").write_text("d,price,c,b,a\n0,99,0,0,1\n0,99,0,0,-1\n")
    result = run_script(code, tmp_path, "--predict", "new.csv")
    assert result.returncode == 0, result.stderr
    lines = result.stdout.split()
    header, values = lines[0], lines[1:]
    if model != "GaussianProcessRegressor":
        assert header == "price"
        first, second = float(values[0]), float(values[1])
    else:
        assert header == "price,std"  # the GP's uncertainty
        first, second = (float(v.split(",")[0]) for v in values)
        assert all(float(v.split(",")[1]) > 0 for v in values)
    assert first > 1 > -1 > second  # about 2 and -2

    (tmp_path / "missing.csv").write_text("a,b\n1,2\n")
    result = run_script(code, tmp_path, "--predict", "missing.csv")
    assert result.returncode != 0 and "lacks the columns c, d" in result.stderr


@needs_sklearn
@pytest.mark.parametrize(
    "model", ["GaussianProcessClassifier", "SVC", "HistGradientBoostingClassifier"]
)
def test_script_fits_and_predicts_classes(model, tmp_path):
    import joblib

    write_classification_csv(tmp_path / "data.csv")
    code = generate_code(model_graph(model))
    result = run_script(code, tmp_path, "--csv", "data.csv", "--save", "clf.joblib")
    assert result.returncode == 0, result.stderr
    assert "Validation accuracy" in result.stdout
    bundle = joblib.load(tmp_path / "clf.joblib")
    assert bundle["task"] == "classification" and bundle["class_names"] == ["ham", "spam"]

    # Without a header line the features come first.
    (tmp_path / "new.csv").write_text("2,0,2,0\n-2,0,-2,0\n")
    result = run_script(code, tmp_path, "--predict", "new.csv", "--save", "clf.joblib")
    assert result.returncode == 0, result.stderr
    lines = result.stdout.split()
    assert lines[0] == "label,probability"
    assert lines[1].startswith("spam,") and lines[2].startswith("ham,")
    assert all(0.5 < float(line.split(",")[1]) <= 1 for line in lines[1:])


@needs_sklearn
def test_script_without_training_predicts_with_a_saved_model(tmp_path):
    write_regression_csv(tmp_path / "data.csv")
    g = model_graph("SVR")
    assert run_script(generate_code(g), tmp_path, "--csv", "data.csv").returncode == 0
    code = generate_code(g, {"include_training": False})
    assert "--csv" not in code
    result = run_script(code, tmp_path, "--predict", "data.csv")
    assert result.returncode == 0, result.stderr
    assert len(result.stdout.split()) == 61  # the header and one value per row


@needs_sklearn
def test_script_rejects_bad_csv_files(tmp_path):
    code = generate_code(model_graph("SVR"))
    (tmp_path / "tiny.csv").write_text("1,2,3,4,5\n2,3,4,5,6\n")
    result = run_script(code, tmp_path, "--csv", "tiny.csv")
    assert result.returncode != 0 and "at least 4 rows" in result.stderr
    (tmp_path / "wide.csv").write_text("1,2,3,4,5,6\n" * 5)
    result = run_script(code, tmp_path, "--csv", "wide.csv")
    assert result.returncode != 0 and "Expected 4 feature columns" in result.stderr
    (tmp_path / "text.csv").write_text("a,b,c,d,y\n1,2,x,4,5\n" * 3)
    result = run_script(code, tmp_path, "--csv", "text.csv")
    assert result.returncode != 0 and "could not convert" in result.stderr


@needs_sklearn
@pytest.mark.parametrize("example_id", CLASSICAL_EXAMPLES)
def test_make_model_matches_the_bricks(example_id):
    namespace: dict = {"__name__": "generated"}
    code = generate_code(EXAMPLES[example_id]["graph"], {"include_main": False})
    exec(compile(code, "<generated>", "exec"), namespace)
    pipeline = namespace["make_model"]()
    assert [name for name, _ in pipeline.steps] == ["scale", "model"]
    model = analyze(EXAMPLES[example_id]["graph"]).model
    estimator = pipeline.named_steps["model"]
    if model.kernel is not None:
        # The kernel brick tree and the scikit-learn kernel have the same hyperparameters.
        assert len(estimator.kernel.hyperparameters) == model.kernel.count()
        assert (
            sum(1 for h in estimator.kernel.hyperparameters if not h.fixed)
            == analyze(EXAMPLES[example_id]["graph"]).total_params
        )


def test_check_on_the_command_line(capsys):
    assert brickdnn_main(["--example", "gp_regression", "--check"]) == 0
    out = capsys.readouterr().out
    assert "KernelSum" in out and "-> kernel" in out
    assert "kernel: ConstantKernel(constant_value=1.0) * RBF(length_scale=1.0)" in out
    assert "fitted: 3 kernel hyperparameters" in out
    assert brickdnn_main(["--example", "svm_classifier", "--check"]) == 0
    out = capsys.readouterr().out
    assert "-> class" in out and "fitted: support vectors" in out
    assert brickdnn_main(["--example", "gradient_boosting_classifier", "--check"]) == 0
    out = capsys.readouterr().out
    assert "-> class" in out and "fitted: decision trees" in out
