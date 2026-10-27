"""Command line interface.

python -m brickdnn graph.json                 # print the PyTorch code
python -m brickdnn graph.json -o model.py     # write it to a file
python -m brickdnn graph.json --check         # only validate and show shapes
python -m brickdnn --example cnn_mnist        # start from a bundled example
python -m brickdnn --list-layers
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import EXAMPLES, GeneratorOptions, GraphValidationError, analyze, generate_code, get_example
from .layers import REGISTRY, fmt_shape


def _print_analysis(graph: dict) -> int:
    result = analyze(graph)
    for nid in result.order:
        info = result.nodes[nid]
        shape = fmt_shape(info.shape) if info.shape is not None else "?"
        params = f"{info.num_params:,}" if info.num_params is not None else "-"
        print(f"  {info.type:<20} {nid:<16} -> {shape:<20} params: {params}")
    for issue in result.warnings:
        print(f"warning: {issue.message}" + (f" [{issue.node}]" if issue.node else ""))
    for issue in result.errors:
        print(
            f"error: {issue.message}" + (f" [{issue.node}]" if issue.node else ""), file=sys.stderr
        )
    if result.valid:
        print(f"OK - {result.total_params:,} trainable parameters")
        return 0
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="brickdnn", description="Generate PyTorch code from a Brick DNN graph (JSON)."
    )
    parser.add_argument("graph", nargs="?", help="graph JSON file ('-' for stdin)")
    parser.add_argument("-o", "--output", help="write the code to this file instead of stdout")
    parser.add_argument("--class-name", default="NeuralNetwork", help="name of the nn.Module class")
    parser.add_argument("--no-main", action="store_true", help="omit the __main__ section")
    parser.add_argument("--no-training", action="store_true", help="omit the CSV training loop")
    parser.add_argument("--check", action="store_true", help="validate only, print shapes")
    parser.add_argument("--example", choices=sorted(EXAMPLES), help="use a bundled example graph")
    parser.add_argument("--list-layers", action="store_true", help="list all layer types")
    args = parser.parse_args(argv)

    if args.list_layers:
        for cls in REGISTRY.values():
            print(f"{cls.category:<14} {cls.type:<20} {cls.description}")
        return 0

    if args.example:
        graph = get_example(args.example)["graph"]
    elif args.graph:
        text = sys.stdin.read() if args.graph == "-" else Path(args.graph).read_text()
        graph = json.loads(text)
    else:
        parser.error("pass a graph file or --example NAME")

    if args.check:
        return _print_analysis(graph)

    options = GeneratorOptions(
        class_name=args.class_name,
        include_main=not args.no_main,
        include_training=not args.no_training,
    )
    try:
        code = generate_code(graph, options)
    except GraphValidationError:
        return _print_analysis(graph)

    if args.output:
        Path(args.output).write_text(code)
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
