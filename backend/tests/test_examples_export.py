"""The data of the website's example pages (brickdnn/examples/export.py)."""

from __future__ import annotations

import json

import pytest

from brickdnn import EXAMPLES, analyze, generate_code
from brickdnn.examples import export


def test_committed_file_matches_the_examples_and_the_generator():
    if not export.DEFAULT_PATH.exists():  # pragma: no cover - the backend without the frontend
        pytest.skip("frontend/landing/examples/examples.json is not in this checkout")
    committed = export.DEFAULT_PATH.read_text(encoding="utf-8")
    assert committed == export.render(), (
        "frontend/landing/examples/examples.json is out of date: "
        "run `python -m brickdnn.examples.export` in backend/"
    )


def test_every_example_with_shapes_parameters_and_code():
    data = export.site_data()
    assert [example["id"] for example in data["examples"]] == list(EXAMPLES)
    for example in data["examples"]:
        graph = EXAMPLES[example["id"]]["graph"]
        analysis = analyze(graph)
        assert example["code"] == generate_code(graph)
        assert example["total_params"] == analysis.total_params
        assert [node["id"] for node in example["nodes"]] == [n["id"] for n in graph["nodes"]]
        assert len(example["edges"]) == len(graph["edges"])
        for node in example["nodes"]:
            info = analysis.nodes[node["id"]]
            assert node["output_shape"] == (list(info.shape) if info.shape else None)
            assert node["num_params"] == info.num_params
    unet = next(e for e in data["examples"] if e["id"] == "unet")
    assert unet["framework"] == "pytorch"
    assert unet["total_params"] == 1_925_700
    gp = next(e for e in data["examples"] if e["id"] == "gp_regression")
    assert gp["framework"] == "scikit-learn"
    assert {edge["kind"] for edge in gp["edges"]} == {"tensor", "kernel"}
    assert "GaussianProcessRegressor" in gp["code"]


def test_main_writes_and_checks(tmp_path, capsys):
    out = tmp_path / "examples.json"
    assert export.main(["--output", str(out), "--check"]) == 1
    assert "out of date" in capsys.readouterr().err
    assert export.main(["--output", str(out)]) == 0
    assert export.main(["--output", str(out), "--check"]) == 0
    assert json.loads(out.read_text())["examples"][0]["id"] == next(iter(EXAMPLES))
    capsys.readouterr()
    assert export.main(["--stdout"]) == 0
    assert capsys.readouterr().out == export.render()
