"""Generated code must be valid Python and, if PyTorch is installed, must actually run."""

from __future__ import annotations

import ast
import shutil
import subprocess
import sys

import pytest

from brickdnn import EXAMPLES, GraphValidationError, analyze, generate_code
from brickdnn.codegen import GeneratorOptions

from .helpers import build_model, chain
from .test_layers import CASES, single_layer_graph

try:
    import torch
except ImportError:  # pragma: no cover - PyTorch is optional
    torch = None

needs_torch = pytest.mark.skipif(torch is None, reason="PyTorch is not installed")


def example_input(analysis, nid, batch=3):
    info = analysis.nodes[nid]
    if info.dtype == "long":
        return torch.randint(0, 2, (batch, *info.shape))
    return torch.randn(batch, *info.shape)


def check_against_torch(graph, **options):
    """Run the generated model and compare shapes and parameter counts with the analysis."""
    analysis = analyze(graph)
    code = generate_code(graph, options)
    model = build_model(code, options.get("class_name", "NeuralNetwork"))
    model.eval()
    inputs = [example_input(analysis, nid) for nid in analysis.inputs]
    with torch.no_grad():
        out = model(*inputs)
    outputs = out if isinstance(out, tuple) else (out,)
    assert len(outputs) == len(analysis.outputs)
    for tensor, nid in zip(outputs, analysis.outputs, strict=True):
        assert tuple(tensor.shape) == (3, *analysis.nodes[nid].shape)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params == analysis.total_params
    return code


@pytest.mark.parametrize("example_id", sorted(EXAMPLES))
def test_example_code_is_valid_python(example_id):
    code = generate_code(EXAMPLES[example_id]["graph"])
    ast.parse(code)


@needs_torch
@pytest.mark.parametrize("example_id", sorted(EXAMPLES))
def test_example_runs_in_pytorch(example_id):
    check_against_torch(EXAMPLES[example_id]["graph"])


@needs_torch
@pytest.mark.parametrize("shape,dtype,layer_type,params,expected", CASES)
def test_every_layer_runs_in_pytorch(shape, dtype, layer_type, params, expected):
    check_against_torch(single_layer_graph(shape, dtype, layer_type, params))


def test_sequential_graph_uses_x():
    code = generate_code(chain(("Input", {"shape": [4]}), ("Linear", {}), ("Activation", {})))
    assert "x = self.linear_1(x)" in code
    assert "x = self.relu_1(x)" in code
    assert "return x" in code


def test_custom_names_and_class_name():
    payload = chain(("Input", {"shape": [4]}), ("Linear", {"out_features": 2}))
    payload["nodes"][1]["name"] = "Classifier Head"
    code = generate_code(payload, {"class_name": "MyNet"})
    assert "class MyNet(nn.Module):" in code
    assert "self.classifier_head = nn.Linear(4, 2)" in code


def test_reserved_and_duplicate_names_are_made_unique():
    payload = chain(("Input", {"shape": [4]}), ("Linear", {}), ("Linear", {}))
    payload["nodes"][1]["name"] = "forward"
    payload["nodes"][2]["name"] = "forward"
    code = generate_code(payload)
    assert "self.forward =" not in code
    ast.parse(code)


def test_invalid_class_name():
    with pytest.raises(ValueError):
        GeneratorOptions.from_dict({"class_name": "not valid"})


def test_invalid_graph_raises():
    with pytest.raises(GraphValidationError) as excinfo:
        generate_code({"nodes": [], "edges": []})
    assert not excinfo.value.analysis.valid


def test_without_main():
    code = generate_code(EXAMPLES["mlp_mnist"]["graph"], {"include_main": False})
    assert "__main__" not in code
    assert "argparse" not in code


def test_training_code_only_for_tabular_models():
    assert "def train(" in generate_code(EXAMPLES["mlp_mnist"]["graph"])
    assert "def train(" not in generate_code(EXAMPLES["cnn_mnist"]["graph"])
    assert "def train(" not in generate_code(
        EXAMPLES["mlp_mnist"]["graph"], {"include_training": False}
    )


@needs_torch
def test_multiple_outputs():
    payload = {
        "nodes": [
            {"id": "in", "type": "Input", "params": {"shape": [8]}},
            {"id": "a", "type": "Linear", "params": {"out_features": 3}},
            {"id": "b", "type": "Linear", "params": {"out_features": 1}},
        ],
        "edges": [{"source": "in", "target": "a"}, {"source": "in", "target": "b"}],
    }
    code = check_against_torch(payload)
    assert "return linear_1, linear_2" in code


@pytest.mark.skipif(shutil.which("ruff") is None, reason="ruff is not installed")
@pytest.mark.parametrize("example_id", sorted(EXAMPLES))
def test_generated_code_is_lint_clean(example_id, tmp_path):
    ruff = shutil.which("ruff")
    target = tmp_path / "model.py"
    target.write_text(generate_code(EXAMPLES[example_id]["graph"]))
    result = subprocess.run(
        [ruff, "check", "--isolated", "--select", "E,F,W", "--line-length", "100", str(target)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout


# --------------------------------------------------------------------------- #
# The generated script end to end (python model.py [--csv ...])
# --------------------------------------------------------------------------- #


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


@needs_torch
def test_script_sanity_check(tmp_path):
    result = run_script(generate_code(EXAMPLES["multi_input"]["graph"]), tmp_path)
    assert result.returncode == 0, result.stderr
    assert "Output shape: (2, 1)" in result.stdout


@needs_torch
def test_script_trains_classifier_on_csv(tmp_path):
    payload = chain(
        ("Input", {"shape": [3]}),
        ("Linear", {"out_features": 16}),
        ("Activation", {}),
        ("BatchNorm1d", {}),
        ("Linear", {"out_features": 2}),
    )
    rows = ["a,b,c,label"] + [f"{i % 7},{i % 5},{i % 3},{i % 2}" for i in range(100)]
    (tmp_path / "data.csv").write_text("\n".join(rows) + "\n")
    result = run_script(generate_code(payload), tmp_path, "--csv", "data.csv", "--epochs", "2")
    assert result.returncode == 0, result.stderr
    assert "val acc" in result.stdout
    assert (tmp_path / "model.pt").exists()


@needs_torch
def test_script_trains_regression_on_csv(tmp_path):
    rows = [",".join(str((i * j) % 11) for j in range(9)) for i in range(64)]  # no header
    (tmp_path / "data.csv").write_text("\n".join(rows) + "\n")
    code = generate_code(EXAMPLES["tabular_regression"]["graph"])
    result = run_script(code, tmp_path, "--csv", "data.csv", "--epochs", "1", "--save", "w.pt")
    assert result.returncode == 0, result.stderr
    assert "val loss" in result.stdout
    assert (tmp_path / "w.pt").exists()
