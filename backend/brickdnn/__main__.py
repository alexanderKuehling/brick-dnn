"""Command line interface.

python -m brickdnn graph.json                 # print the PyTorch code
python -m brickdnn graph.json -o model.py     # write it to a file
python -m brickdnn graph.json --check         # only validate and show shapes
python -m brickdnn --example cnn_mnist        # start from a bundled example
python -m brickdnn --list-layers
python -m brickdnn train graph.json --data mnist   # train it (PyTorch or scikit-learn, see train --help)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import EXAMPLES, GeneratorOptions, GraphValidationError, analyze, generate_code, get_example
from .layers import REGISTRY, Ensemble, fmt_shape


def _output(info) -> str:
    if info.output_kind == "kernel" and info.resolved:
        return "kernel"
    if info.output_kind == "estimator" and info.outputs and info.resolved:
        return "estimator"  # a model of an ensemble
    if info.shape is not None:
        return fmt_shape(info.shape)
    if info.resolved:  # a classifier: one class per sample
        return "class"
    return "?"


def _print_analysis(graph: dict) -> int:
    result = analyze(graph)
    for nid in result.order:
        info = result.nodes[nid]
        params = f"{info.num_params:,}" if info.num_params is not None else "-"
        print(f"  {info.type:<24} {nid:<16} -> {_output(info):<20} params: {params}")
        if info.kernel is not None:
            print(f"      kernel: {info.kernel.code()}")
    for issue in result.warnings:
        print(f"warning: {issue.message}" + (f" [{issue.node}]" if issue.node else ""))
    for issue in result.errors:
        print(
            f"error: {issue.message}" + (f" [{issue.node}]" if issue.node else ""), file=sys.stderr
        )
    if not result.valid:
        return 1
    if result.family == "classical":
        total = result.total_params
        layer = result.model.layer if result.model else None
        if isinstance(layer, Ensemble):
            print(
                f"OK - a {layer.combines} ensemble of {len(layer.members)} classical models "
                "(scikit-learn)"
            )
            return 0
        if total is not None:
            fitted = f"{total:,} kernel hyperparameters"
        elif result.model and getattr(result.model.layer, "scale", None) == "gradient_boosting":
            fitted = "decision trees"
        else:
            fitted = "support vectors"
        print(f"OK - a classical model (scikit-learn), fitted: {fitted}")
    else:
        print(f"OK - {result.total_params:,} trainable parameters")
    return 0


def main(argv: list[str] | None = None) -> int:
    args_list = sys.argv[1:] if argv is None else list(argv)
    if args_list[:1] == ["train"]:
        from .training.cli import main as train_main

        return train_main(args_list[1:])

    parser = argparse.ArgumentParser(
        prog="brickdnn",
        description="Generate PyTorch code (scikit-learn code for classical models) from a "
        "Brick DNN graph (JSON).",
        epilog="To train a network with PyTorch, run: brickdnn train --help",
    )
    parser.add_argument("graph", nargs="?", help="graph JSON file ('-' for stdin)")
    parser.add_argument("-o", "--output", help="write the code to this file instead of stdout")
    parser.add_argument("--class-name", default="NeuralNetwork", help="name of the nn.Module class")
    parser.add_argument("--no-main", action="store_true", help="omit the __main__ section")
    parser.add_argument("--no-training", action="store_true", help="omit the CSV training loop")
    parser.add_argument("--check", action="store_true", help="validate only, print shapes")
    parser.add_argument("--example", choices=sorted(EXAMPLES), help="use a bundled example graph")
    parser.add_argument("--list-layers", action="store_true", help="list all layer types")
    args = parser.parse_args(args_list)

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

    try:
        options = GeneratorOptions.from_dict(
            {
                "class_name": args.class_name,
                "include_main": not args.no_main,
                "include_training": not args.no_training,
            }
        )
    except ValueError as exc:
        parser.error(str(exc))
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
