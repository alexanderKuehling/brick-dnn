from __future__ import annotations

import json

from brickdnn.__main__ import main

from .helpers import chain


def test_cli_generates_code(tmp_path, capsys):
    graph_file = tmp_path / "graph.json"
    graph_file.write_text(json.dumps(chain(("Input", {"shape": [4]}), ("Linear", {}))))
    assert main([str(graph_file), "--class-name", "Tiny"]) == 0
    assert "class Tiny(nn.Module)" in capsys.readouterr().out


def test_cli_writes_file(tmp_path):
    out = tmp_path / "model.py"
    assert main(["--example", "cnn_mnist", "-o", str(out)]) == 0
    assert "nn.Conv2d" in out.read_text()


def test_cli_check_reports_errors(tmp_path, capsys):
    graph_file = tmp_path / "graph.json"
    graph_file.write_text(json.dumps({"nodes": [{"id": "a", "type": "Linear"}], "edges": []}))
    assert main([str(graph_file), "--check"]) == 1
    assert "no incoming connection" in capsys.readouterr().err


def test_cli_lists_layers(capsys):
    assert main(["--list-layers"]) == 0
    assert "TransformerEncoder" in capsys.readouterr().out
