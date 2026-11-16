"""The examples as data for the example pages of the website (frontend/landing/examples/).

The pages at brickdnn.com/examples/ show every example with its layers, their shapes and
parameter counts, a diagram and the generated code. The frontend is built without the backend
(its Docker image has no Python), so this data is made here and committed as
``frontend/landing/examples/examples.json``; tests/test_examples_export.py and the frontend's
landing.test.ts check that the committed file is what this module makes from the examples and
the code generator now. After changing an example, a layer or the generator, run::

    python -m brickdnn.examples.export          # writes the file
    python -m brickdnn.examples.export --check  # exit code 1 if the file is out of date
    python -m brickdnn.examples.export --stdout # print it instead
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ..codegen import GeneratorOptions, generate_code
from ..graph import analyze
from ..layers import REGISTRY
from . import EXAMPLES

# frontend/landing/examples/examples.json of the repository this file is in.
DEFAULT_PATH = Path(__file__).resolve().parents[3] / "frontend/landing/examples/examples.json"

FRAMEWORKS = {"neural": "pytorch", "classical": "scikit-learn", "hybrid": "pytorch+scikit-learn"}


def _example(example: dict[str, Any]) -> dict[str, Any]:
    graph = example["graph"]
    analysis = analyze(graph)
    if not analysis.valid:
        raise ValueError(f"The example {example['id']} has errors: {analysis.errors}")
    result = analysis.to_dict()
    nodes = []
    for raw in graph["nodes"]:
        info = result["nodes"][raw["id"]]
        nodes.append(
            {
                "id": raw["id"],
                "type": raw["type"],
                "category": REGISTRY[raw["type"]].category,
                # The parameters the example sets (the others are the layer's defaults).
                "params": raw.get("params", {}),
                "position": raw["position"],
                "output_shape": info["output_shape"],
                "output_kind": info["output_kind"],
                "num_params": info["num_params"],
                "task": info["task"],
                "kernel": info["kernel"],
            }
        )
    kinds = {node["id"]: node["output_kind"] for node in nodes}
    edges = [
        {"source": edge["source"], "target": edge["target"], "kind": kinds[edge["source"]]}
        for edge in graph["edges"]
    ]
    classical = result["classical"]
    hybrid = result["hybrid"]
    return {
        "id": example["id"],
        "name": example["name"],
        "description": example["description"],
        "family": result["family"],
        "framework": FRAMEWORKS[result["family"]],
        "task": classical["task"] if classical else None,
        "estimator": classical["estimator"] if classical else None,
        "total_params": result["total_params"],
        "neural_params": hybrid["neural_params"] if hybrid else None,
        "inputs": result["inputs"],
        "outputs": result["outputs"],
        "nodes": nodes,
        "edges": edges,
        # What the editor's code dialog shows and downloads with its default options.
        "code": generate_code(graph, GeneratorOptions()),
    }


def site_data() -> dict[str, Any]:
    """The data of the example pages: every example in the order of the Examples menu."""
    return {
        "note": "Made by `python -m brickdnn.examples.export` (backend/brickdnn/examples/"
        "export.py) from the examples and the code generator. Do not edit by hand.",
        "examples": [_example(example) for example in EXAMPLES.values()],
    }


def render() -> str:
    return json.dumps(site_data(), indent=1, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m brickdnn.examples.export",
        description="Write the data of the website's example pages.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_PATH, help="the JSON file")
    parser.add_argument("--check", action="store_true", help="only check that it is up to date")
    parser.add_argument("--stdout", action="store_true", help="print the JSON instead")
    args = parser.parse_args(argv)
    text = render()
    if args.stdout:
        sys.stdout.write(text)
        return 0
    if args.check:
        current = args.output.read_text(encoding="utf-8") if args.output.exists() else ""
        if current != text:
            print(
                f"{args.output} is out of date: run python -m brickdnn.examples.export",
                file=sys.stderr,
            )
            return 1
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
