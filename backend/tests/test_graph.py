"""Graph-level validation: structure, cycles, merges, multiple inputs/outputs."""

from __future__ import annotations

import pytest

from brickdnn import EXAMPLES, analyze

from .helpers import chain


def graph(nodes, edges):
    return {
        "nodes": [{"id": i, "type": t, "params": p} for i, t, p in nodes],
        "edges": [{"source": s, "target": t} for s, t in edges],
    }


@pytest.mark.parametrize("example_id", sorted(EXAMPLES))
def test_examples_are_valid(example_id):
    result = analyze(EXAMPLES[example_id]["graph"])
    assert result.valid, result.errors
    assert result.total_params > 0


@pytest.mark.parametrize(
    "payload,message",
    [
        (None, "JSON object"),
        ({"nodes": "x", "edges": []}, "must be lists"),
        ({"nodes": [], "edges": []}, "canvas is empty"),
        ({"nodes": [{"type": "Linear"}], "edges": []}, "missing 'id'"),
        ({"nodes": [{"id": "a", "type": "Nope"}], "edges": []}, "Unknown layer type"),
        (
            {"nodes": [{"id": 1, "type": "Input"}, {"id": 1, "type": "Input"}], "edges": []},
            "Duplicate node id",
        ),
        (
            {"nodes": [{"id": "a", "type": "Input"}], "edges": [{"source": "a", "target": "b"}]},
            "does not exist",
        ),
        ({"nodes": [{"id": "a", "type": "Linear"}], "edges": []}, "no incoming connection"),
        ({"nodes": [{"id": "a", "type": "Input"}], "edges": []}, "not connected"),
    ],
)
def test_structural_errors(payload, message):
    result = analyze(payload)
    assert not result.valid
    assert any(message in e.message for e in result.errors), result.errors


def test_missing_input_layer():
    result = analyze(graph([("a", "Linear", {}), ("b", "Linear", {})], [("a", "b")]))
    assert any("Add an Input layer" in e.message for e in result.errors)


def test_cycle_is_reported():
    result = analyze(
        graph(
            [
                ("in", "Input", {"shape": [4]}),
                ("a", "Linear", {}),
                ("b", "Linear", {}),
                ("add", "Add", {}),
            ],
            [("in", "add"), ("add", "a"), ("a", "b"), ("b", "add")],
        )
    )
    assert not result.valid
    assert {e.node for e in result.errors if "cycle" in e.message} == {"add", "a", "b"}


def test_self_loop_is_rejected():
    result = analyze(graph([("in", "Input", {})], [("in", "in")]))
    assert "connected to itself" in result.errors[0].message


def test_single_input_layer_rejects_two_inputs():
    result = analyze(
        graph(
            [("a", "Input", {"shape": [4]}), ("b", "Input", {"shape": [4]}), ("fc", "Linear", {})],
            [("a", "fc"), ("b", "fc")],
        )
    )
    assert "Use Add or Concat" in result.errors[0].message


def test_add_requires_equal_shapes():
    result = analyze(
        graph(
            [
                ("in", "Input", {"shape": [4]}),
                ("a", "Linear", {"out_features": 3}),
                ("add", "Add", {}),
            ],
            [("in", "a"), ("in", "add"), ("a", "add")],
        )
    )
    assert "identical shape" in result.errors[0].message


def test_concat_shapes():
    result = analyze(
        graph(
            [
                ("a", "Input", {"shape": [2, 5]}),
                ("b", "Input", {"shape": [3, 5]}),
                ("cat", "Concat", {"dim": 0}),
            ],
            [("a", "cat"), ("b", "cat")],
        )
    )
    assert result.valid
    assert result.nodes["cat"].shape == (5, 5)

    result = analyze(
        graph(
            [
                ("a", "Input", {"shape": [2, 5]}),
                ("b", "Input", {"shape": [3, 5]}),
                ("cat", "Concat", {"dim": -1}),
            ],
            [("a", "cat"), ("b", "cat")],
        )
    )
    assert "must match in every dimension" in result.errors[0].message


def test_errors_do_not_cascade():
    result = analyze(
        chain(("Input", {"shape": [3, 8, 8]}), ("Linear", {"in_features": 1}), ("Flatten", {}))
    )
    assert [e.node for e in result.errors] == ["n1"]
    assert result.nodes["n2"].shape is None


def test_softmax_output_warning():
    result = analyze(chain(("Input", {"shape": [4]}), ("Activation", {"function": "softmax"})))
    assert result.valid
    assert "CrossEntropyLoss" in result.warnings[0].message


def test_duplicate_edges_are_ignored():
    payload = chain(("Input", {"shape": [4]}), ("Linear", {}))
    payload["edges"].append(dict(payload["edges"][0]))
    assert analyze(payload).valid


def test_saved_project_document_is_accepted():
    payload = {"name": "x", "graph": chain(("Input", {"shape": [4]}), ("Linear", {}))}
    assert analyze(payload).valid


def test_to_dict_is_json_friendly():
    data = analyze(EXAMPLES["cnn_mnist"]["graph"]).to_dict()
    assert data["valid"] is True
    assert data["inputs"] == ["input"]
    assert data["outputs"] == ["fc2"]
    assert data["nodes"]["flatten"]["output_shape"] == [3136]
    assert data["nodes"]["fc1"]["params"]["in_features"] == 3136
    assert data["total_params"] == 421_642
