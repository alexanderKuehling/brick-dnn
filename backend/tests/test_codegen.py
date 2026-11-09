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
    assert "Training on cpu: 80 rows, 20 for validation" in result.stdout
    assert "val acc" in result.stdout
    checkpoint = torch.load(tmp_path / "model.pt", weights_only=True)
    pre = checkpoint["preprocessing"]
    assert pre["columns"] == ["a", "b", "c"] and pre["target"] == "label"
    assert pre["task"] == "classification" and pre["num_classes"] == 2
    build_model(generate_code(payload)).load_state_dict(checkpoint["state_dict"])


@needs_torch
def test_script_trains_regression_on_csv(tmp_path):
    rows = [",".join(str((i * j) % 11) for j in range(9)) for i in range(64)]  # no header
    (tmp_path / "data.csv").write_text("\n".join(rows) + "\n")
    code = generate_code(EXAMPLES["tabular_regression"]["graph"])
    result = run_script(code, tmp_path, "--csv", "data.csv", "--epochs", "1", "--save", "w.pt")
    assert result.returncode == 0, result.stderr
    assert "val loss" in result.stdout
    assert "Saved the weights and the preprocessing to w.pt" in result.stdout
    pre = torch.load(tmp_path / "w.pt", weights_only=True)["preprocessing"]
    assert pre["columns"] == [] and pre["target"] == "target" and pre["task"] == "regression"
    assert len(pre["mean"]) == len(pre["std"]) == 8
    assert pre["std"][0] == 1.0  # the first column is constant (0) and stays as it is


def regression_csv(tmp_path, n=50):
    """A CSV file with a header and features whose statistics differ between subsets."""
    gen = torch.Generator().manual_seed(3)
    values = torch.randn(n, 3, generator=gen) * torch.tensor([1.0, 10.0, 100.0])
    values[:, 0] += torch.arange(n) / 5  # a trend: every subset has another mean
    target = (values @ torch.tensor([1.0, 0.1, 0.01])).tolist()
    lines = ["a,b,c,price"] + [
        ",".join(f"{v:.5f}" for v in row) + f",{t:.5f}" for row, t in zip(values.tolist(), target)
    ]
    (tmp_path / "data.csv").write_text("\n".join(lines) + "\n")
    return values


@needs_torch
def test_script_split_is_seeded_and_standardises_with_the_training_rows(tmp_path):
    values = regression_csv(tmp_path)
    code = generate_code(chain(("Input", {"shape": [3]}), ("Linear", {"out_features": 1})))

    def train(*args):
        result = run_script(code, tmp_path, "--csv", "data.csv", "--epochs", "3", *args)
        assert result.returncode == 0, result.stderr
        return [line for line in result.stdout.splitlines() if line.startswith("Epoch")]

    first = train("--save", "one.pt")
    assert train("--save", "two.pt") == first  # same seed (0), same split and training
    assert train("--save", "three.pt", "--seed", "7") != first
    one = torch.load(tmp_path / "one.pt", weights_only=True)
    two = torch.load(tmp_path / "two.pt", weights_only=True)
    assert all(torch.equal(one["state_dict"][k], two["state_dict"][k]) for k in one["state_dict"])

    # Mean and standard deviation come from the 40 training rows of the seeded split.
    order = torch.randperm(50, generator=torch.Generator().manual_seed(0))
    train_rows = order[10:]
    pre = one["preprocessing"]
    assert pre["columns"] == ["a", "b", "c"] and pre["target"] == "price"
    assert pre["mean"] == pytest.approx(values[train_rows].mean(dim=0).tolist(), rel=1e-4)
    assert pre["std"] == pytest.approx(values[train_rows].std(dim=0).tolist(), rel=1e-4)
    assert pre["mean"][0] != pytest.approx(float(values[:, 0].mean()), rel=1e-4)


@needs_torch
def test_script_predicts_new_rows_like_the_training_data(tmp_path):
    payload = chain(
        ("Input", {"shape": [3]}),
        ("Linear", {"out_features": 8}),
        ("Activation", {}),
        ("BatchNorm1d", {}),
        ("Linear", {"out_features": 3}),
    )
    code = generate_code(payload)
    rows = ["a,b,c,label"] + [f"{i % 7},{(i * 3) % 5},{i % 11},{i % 3}" for i in range(60)]
    (tmp_path / "data.csv").write_text("\n".join(rows) + "\n")
    result = run_script(code, tmp_path, "--csv", "data.csv", "--epochs", "3")
    assert result.returncode == 0, result.stderr

    new = torch.tensor([[1.0, 2.0, 3.0], [6.0, 0.0, 10.0], [3.0, 4.0, 0.0], [0.0, 1.0, 7.0]])
    checkpoint = torch.load(tmp_path / "model.pt", weights_only=True)
    model = build_model(code)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    pre = checkpoint["preprocessing"]
    with torch.no_grad():
        x = (new - torch.tensor(pre["mean"])) / torch.tensor(pre["std"])
        expected = ["label", *(str(v) for v in model(x).argmax(dim=1).tolist())]

    # Columns found by name (in another order, with the target and an extra column) …
    lines = ["id,c,label,b,a"] + [
        f"{i},{c:g},0,{b:g},{a:g}" for i, (a, b, c) in enumerate(new.tolist())
    ]
    (tmp_path / "by_name.csv").write_text("\n".join(lines) + "\n")
    result = run_script(code, tmp_path, "--predict", "by_name.csv")
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == expected
    # … or the first columns of a file without a header line.
    lines = [",".join(f"{v:g}" for v in row) for row in new.tolist()]
    (tmp_path / "plain.csv").write_text("\n".join(lines) + "\n")
    result = run_script(code, tmp_path, "--predict", "plain.csv")
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == expected

    (tmp_path / "missing.csv").write_text("a,b,label\n1,2,0\n")
    result = run_script(code, tmp_path, "--predict", "missing.csv")
    assert result.returncode != 0 and "missing.csv lacks the columns c." in result.stderr


# --------------------------------------------------------------------------- #
# Regression tests from the pre-release review
# --------------------------------------------------------------------------- #


@needs_torch
@pytest.mark.parametrize("layer_type", ["RNN", "LSTM", "GRU"])
def test_bidirectional_rnn_uses_final_states_of_both_directions(layer_type):
    graph = chain(
        ("Input", {"shape": [6, 3]}),
        (layer_type, {"hidden_size": 4, "bidirectional": True, "num_layers": 2}),
    )
    model = build_model(generate_code(graph))
    model.eval()
    x = torch.randn(2, 6, 3)
    rnn = getattr(model, f"{layer_type.lower()}_1")
    with torch.no_grad():
        out = model(x)
        sequence, state = rnn(x)
    hidden = state[0] if layer_type == "LSTM" else state
    # forward direction after the last step, backward direction after the first step
    assert torch.allclose(out[:, :4], sequence[:, -1, :4])
    assert torch.allclose(out[:, 4:], sequence[:, 0, 4:])
    assert torch.allclose(out, torch.cat([hidden[-2], hidden[-1]], dim=-1))


@needs_torch
@pytest.mark.parametrize("name", ["compile", "named_modules", "hidden", "tuple", "main"])
def test_reserved_layer_names_are_renamed(name):
    payload = chain(("Input", {"shape": [4]}), ("Linear", {"out_features": 2}))
    payload["nodes"][1]["name"] = name
    check_against_torch(payload)


@needs_torch
def test_reserved_input_names_in_multi_input_scripts(tmp_path):
    payload = {
        "nodes": [
            {"id": "a", "type": "Input", "params": {"shape": [3]}, "name": "count_parameters"},
            {"id": "b", "type": "Input", "params": {"shape": [3]}, "name": "enumerate"},
            {"id": "add", "type": "Add"},
            {"id": "o1", "type": "Linear", "params": {"out_features": 2}},
            {"id": "o2", "type": "Linear", "params": {"out_features": 1}},
        ],
        "edges": [
            {"source": "a", "target": "add"},
            {"source": "b", "target": "add"},
            {"source": "add", "target": "o1"},
            {"source": "add", "target": "o2"},
        ],
    }
    result = run_script(generate_code(payload), tmp_path)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("class_name", ["main", "nn", "torch", "train", "x", "count_parameters"])
def test_reserved_class_names_are_rejected(class_name):
    with pytest.raises(ValueError, match="already used"):
        GeneratorOptions.from_dict({"class_name": class_name})


def test_huge_integers_are_rejected():
    result = analyze(chain(("Input", {"shape": [10**4000, 2]}), ("Flatten", {})))
    assert "too large" in result.errors[0].message


@needs_torch
def test_script_rejects_tiny_csv(tmp_path):
    (tmp_path / "data.csv").write_text("1,2,3,4,5,6,7,8,1\n2,3,4,5,6,7,8,9,0\n")
    code = generate_code(EXAMPLES["tabular_regression"]["graph"])
    result = run_script(code, tmp_path, "--csv", "data.csv")
    assert result.returncode != 0
    assert "at least 4 rows" in result.stderr
