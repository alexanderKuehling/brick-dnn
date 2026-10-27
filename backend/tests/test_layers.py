"""Shape inference and parameter validation for every layer type."""

from __future__ import annotations

import pytest

from brickdnn import REGISTRY, analyze, layer_specs
from brickdnn.layers import CATEGORIES, PARAM_KINDS

from .helpers import chain

# (input shape, input dtype, layer type, params, expected output shape)
CASES = [
    ((784,), "float", "Linear", {"out_features": 10}, (10,)),
    ((5, 16), "float", "Linear", {"out_features": 4, "bias": False}, (5, 4)),
    ((3, 4, 5), "float", "Flatten", {}, (60,)),
    ((784,), "float", "Reshape", {"shape": [1, 28, 28]}, (1, 28, 28)),
    ((784,), "float", "Reshape", {"shape": [-1, 28]}, (28, 28)),
    ((10,), "float", "Dropout", {"p": 0.1}, (10,)),
    ((10,), "float", "Activation", {"function": "gelu"}, (10,)),
    ((10,), "float", "Activation", {"function": "log_softmax"}, (10,)),
    (
        (3, 100),
        "float",
        "Conv1d",
        {"out_channels": 8, "kernel_size": [5], "padding": "valid"},
        (8, 96),
    ),
    ((3, 32, 32), "float", "Conv2d", {"out_channels": 16}, (16, 32, 32)),
    (
        (3, 32, 32),
        "float",
        "Conv2d",
        {
            "out_channels": 16,
            "kernel_size": [3, 3],
            "stride": [2, 2],
            "padding": "custom",
            "padding_size": [1, 1],
        },
        (16, 16, 16),
    ),
    (
        (3, 28, 28),
        "float",
        "Conv2d",
        {"out_channels": 4, "kernel_size": [5, 3], "padding": "valid"},
        (4, 24, 26),
    ),
    ((8, 100), "float", "MaxPool1d", {"kernel_size": [4]}, (8, 25)),
    ((8, 28, 28), "float", "MaxPool2d", {}, (8, 14, 14)),
    ((8, 28, 28), "float", "AvgPool2d", {"kernel_size": [3, 3], "stride": [1, 1]}, (8, 26, 26)),
    ((8, 13, 7), "float", "AdaptiveAvgPool2d", {"output_size": [2, 3]}, (8, 2, 3)),
    ((32,), "float", "BatchNorm1d", {}, (32,)),
    ((4, 50), "float", "BatchNorm1d", {}, (4, 50)),
    ((16, 8, 8), "float", "BatchNorm2d", {}, (16, 8, 8)),
    ((10, 64), "float", "LayerNorm", {}, (10, 64)),
    ((20,), "long", "Embedding", {"num_embeddings": 100, "embedding_dim": 8}, (20, 8)),
    ((20, 8), "float", "PositionalEmbedding", {"max_len": 50}, (20, 8)),
    ((20, 8), "float", "RNN", {"hidden_size": 16}, (16,)),
    ((20, 8), "float", "LSTM", {"hidden_size": 16, "num_layers": 2, "bidirectional": True}, (32,)),
    ((20, 8), "float", "GRU", {"hidden_size": 16, "return_sequences": True}, (20, 16)),
    (
        (20, 32),
        "float",
        "TransformerEncoder",
        {"nhead": 4, "dim_feedforward": 64, "num_layers": 1},
        (20, 32),
    ),
    ((20, 32), "float", "SequencePool", {"mode": "mean"}, (32,)),
    ((20, 32), "float", "SequencePool", {"mode": "last"}, (32,)),
]


def single_layer_graph(shape, dtype, layer_type, params):
    return chain(("Input", {"shape": list(shape), "dtype": dtype}), (layer_type, params))


@pytest.mark.parametrize("shape,dtype,layer_type,params,expected", CASES)
def test_shape_inference(shape, dtype, layer_type, params, expected):
    result = analyze(single_layer_graph(shape, dtype, layer_type, params))
    assert result.valid, result.errors
    assert result.nodes["n1"].shape == expected


def test_every_registered_layer_has_a_case():
    covered = {case[2] for case in CASES} | {"Input", "Add", "Concat"}
    assert covered == set(REGISTRY)


def test_auto_params_are_filled_in():
    result = analyze(chain(("Input", {"shape": [3, 8, 8]}), ("Conv2d", {"out_channels": 5})))
    assert result.nodes["n1"].layer.p["in_channels"] == 3
    assert result.nodes["n1"].to_dict()["params"]["kernel_size"] == [3, 3]


def test_explicit_param_must_match_input():
    result = analyze(chain(("Input", {"shape": [100]}), ("Linear", {"in_features": 50})))
    assert not result.valid
    assert "Input features is set to 50" in result.errors[0].message
    assert result.errors[0].node == "n1"


@pytest.mark.parametrize(
    "layer_type,params,message",
    [
        ("Linear", {"out_features": 0}, "at least 1"),
        ("Linear", {"out_features": "abc"}, "must be an integer"),
        ("Linear", {"bias": "maybe"}, "true or false"),
        ("Dropout", {"p": 1.5}, "at most 1"),
        ("Activation", {"function": "magic"}, "must be one of"),
        ("Conv2d", {"kernel_size": [3, 3, 3]}, "exactly 2 values"),
    ],
)
def test_invalid_params(layer_type, params, message):
    shape = [3, 8, 8] if layer_type == "Conv2d" else [10]
    result = analyze(chain(("Input", {"shape": shape}), (layer_type, params)))
    assert not result.valid
    assert message in result.errors[0].message


@pytest.mark.parametrize(
    "shape,layer_type,params,message",
    [
        ([10], "Conv2d", {}, "expects a 3-D"),
        ([3, 4, 4], "Conv2d", {"kernel_size": [7, 7], "padding": "valid"}, "larger than"),
        ([3, 8, 8], "Conv2d", {"stride": [2, 2]}, "'same' padding only works with stride 1"),
        ([3, 2, 2], "MaxPool2d", {"kernel_size": [3, 3]}, "larger than"),
        ([10, 30], "TransformerEncoder", {"nhead": 4}, "divisible"),
        ([600, 8], "PositionalEmbedding", {"max_len": 512}, "exceeds"),
        ([10], "Reshape", {"shape": [3, 3]}, "Cannot reshape"),
        ([10], "Reshape", {"shape": [-1, -1]}, "Only one entry"),
        ([10], "SequencePool", {}, "expects a (sequence, features)"),
        ([10], "Embedding", {}, "expects integer token ids"),
    ],
)
def test_invalid_inputs(shape, layer_type, params, message):
    result = analyze(chain(("Input", {"shape": shape}), (layer_type, params)))
    assert not result.valid
    assert message in result.errors[0].message


def test_float_layer_rejects_token_ids():
    result = analyze(chain(("Input", {"shape": [10], "dtype": "long"}), ("Linear", {})))
    assert "Add an Embedding layer first" in result.errors[0].message


def test_layer_specs_are_well_formed():
    specs = layer_specs()
    assert {s["type"] for s in specs} == set(REGISTRY)
    for spec in specs:
        assert spec["category"] in CATEGORIES
        assert spec["description"]
        for param in spec["params"]:
            assert param["kind"] in PARAM_KINDS
            if param["kind"] == "select":
                assert param["default"] in param["options"]
            if param["show_if"]:
                assert param["show_if"]["param"] in {p["name"] for p in spec["params"]}
