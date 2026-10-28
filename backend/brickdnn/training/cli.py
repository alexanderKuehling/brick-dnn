"""Train a network from the command line (needs PyTorch).

brickdnn train --example cnn_mnist --data mnist --epochs 5
brickdnn train graph.json --data random                # demo data with a hidden rule
brickdnn train graph.json --data houses.csv --save houses.pt
brickdnn train graph.json --data photos.zip            # or a folder with one folder per class

Press Ctrl+C to stop after the current batch; the weights are saved anyway.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, TextIO

from .. import EXAMPLES, analyze, get_example
from . import torch_installed
from .compat import check_compatibility
from .config import DEVICES, OPTIMIZERS, ConfigError, TrainingConfig
from .datasets import (
    BUILTIN_DATASETS,
    DEFAULT_DATA_ROOT,
    DatasetError,
    DatasetInfo,
    builtin_info,
    extract_image_zip,
    inspect_csv,
    inspect_image_folder,
    random_info,
)
from .reporting import Reporter

CSV_SUFFIXES = (".csv", ".tsv", ".txt")
INSTALL_HINT = 'Training needs PyTorch. Install it with: pip install "brickdnn[train]"'


class ConsoleReporter(Reporter):
    """Prints the engine's progress, similar to the exported training script."""

    def __init__(self, epochs: int, save: Path, out: TextIO | None = None):
        self.epochs = epochs
        self.save = save
        self.out = out or sys.stdout
        self.live = self.out.isatty()
        self.stop = False
        self._line = False
        self._last_message = ""

    def _print(self, text: str, err: bool = False) -> None:
        self._clear()
        print(text, file=sys.stderr if err else self.out, flush=True)

    def _show(self, text: str) -> None:
        """An updating status line (only on a terminal)."""
        if self.live:
            self.out.write("\r\033[K" + text)
            self.out.flush()
            self._line = True

    def _clear(self) -> None:
        if self._line:
            self.out.write("\r\033[K")
            self._line = False

    def status(self, status: str, message: str) -> None:
        if status != "preparing" or message == self._last_message:
            return
        self._last_message = message
        if message.startswith("Loading images"):
            self._show(message)
        else:
            self._print(message)

    def log(self, line: str) -> None:
        self._print(line)

    def setup(self, info: dict[str, Any]) -> None:
        if "train_size" not in info:
            self._print(f"Now training on {info.get('device_label')}.")
            return
        self._print(
            f"Training on {info['device_label']}: {info['train_size']:,} training and "
            f"{info['val_size']:,} validation samples, {info['steps_per_epoch']:,} steps per "
            f"epoch, {info['total_params']:,} parameters."
        )

    def progress(self, progress: dict[str, Any]) -> None:
        self._show(
            f"Epoch {progress['epoch']:3d}/{self.epochs}  step {progress['step']}/"
            f"{progress['steps']}  loss {progress['loss']:.4f}  "
            f"{progress['samples_per_second']:,.0f} samples/s"
        )

    def epoch_end(self, entry: dict[str, Any], best: dict[str, Any]) -> None:
        line = (
            f"Epoch {entry['epoch']:3d}/{self.epochs}  train loss {entry['train_loss']:.4f}"
            f"  val loss {entry['val_loss']:.4f}"
        )
        if entry.get("val_accuracy") is not None:
            line += f"  val acc {entry['val_accuracy']:.1%}"
        if entry.get("val_rmse") is not None:
            line += f"  val rmse {entry['val_rmse']:.4g}"
        self._print(line + f"  ({entry['seconds']:.1f}s)")

    def finish(self, status: str, message: str, weights: bool) -> None:
        if status == "failed":
            self._print(f"error: {message}", err=True)
            return
        self._print(message)
        if weights:
            self._print(f"Saved weights to {self.save}")

    def should_stop(self) -> bool:
        return self.stop


def build_parser() -> argparse.ArgumentParser:
    builtin = ", ".join(BUILTIN_DATASETS)
    parser = argparse.ArgumentParser(
        prog="brickdnn train",
        description="Train a Brick DNN network with PyTorch and save its weights.",
        epilog="Ctrl+C stops after the current batch and still saves the weights.",
    )
    parser.add_argument("graph", nargs="?", help="graph JSON file ('-' for stdin)")
    parser.add_argument("--example", choices=sorted(EXAMPLES), help="use a bundled example graph")
    parser.add_argument(
        "--data",
        default="random",
        help=f"random (demo data, default), {builtin}, a .csv file, a .zip file or a folder "
        "with one sub-folder of images per class",
    )
    parser.add_argument("--epochs", type=int, help="number of epochs (default 10)")
    parser.add_argument("--batch-size", type=int, help="samples per batch (default 64)")
    parser.add_argument("--lr", type=float, help="learning rate (default 0.001)")
    parser.add_argument("--optimizer", choices=OPTIMIZERS, help="default adam")
    parser.add_argument("--weight-decay", type=float, help="default 0")
    parser.add_argument(
        "--val-split", type=float, help="share of the data used for validation (default 0.2)"
    )
    parser.add_argument("--device", choices=DEVICES, help="default auto")
    parser.add_argument("--max-samples", type=int, help="train on at most this many samples")
    parser.add_argument("--seed", type=int, help="random seed for a repeatable run")
    parser.add_argument("--save", help="where to store the weights (default <graph name>.pt)")
    parser.add_argument(
        "--data-root",
        default=str(DEFAULT_DATA_ROOT),
        help="folder for downloaded datasets (default %(default)s)",
    )
    return parser


def resolve_data(value: str, workdir: Path) -> tuple[DatasetInfo, str | None]:
    """Turn ``--data`` into dataset metadata and the path the engine reads."""
    key = value.strip().lower().replace("-", "_")
    if key in ("random", "demo"):
        return random_info(), None
    if key in BUILTIN_DATASETS:
        return builtin_info(key), None
    path = Path(value).expanduser()
    if path.is_dir():
        return inspect_image_folder(path), str(path)
    if path.is_file() and path.suffix.lower() == ".zip":
        dest = workdir / "images"
        count = extract_image_zip(path, dest)
        print(f"Extracted {count:,} images from {path.name}.")
        return inspect_image_folder(dest, name=path.stem), str(dest)
    if path.is_file() and path.suffix.lower() in CSV_SUFFIXES:
        return inspect_csv(path), str(path)
    if path.is_file():
        raise DatasetError(f"'{value}' is not a .csv or .zip file.")
    names = ", ".join(["random", *BUILTIN_DATASETS])
    raise DatasetError(f"'{value}' is neither a dataset ({names}) nor a file or folder.")


def _load_graph(args: argparse.Namespace, parser: argparse.ArgumentParser):
    if args.example:
        return get_example(args.example)["graph"], args.example
    if not args.graph:
        parser.error("pass a graph file or --example NAME")
    try:
        text = sys.stdin.read() if args.graph == "-" else Path(args.graph).read_text()
        graph = json.loads(text)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"error: cannot read the graph: {exc}") from exc
    return graph, "model" if args.graph == "-" else Path(args.graph).stem


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    graph, stem = _load_graph(args, parser)

    if not torch_installed():
        print(f"error: {INSTALL_HINT}", file=sys.stderr)
        return 1
    analysis = analyze(graph)
    if not analysis.valid:
        for issue in analysis.errors:
            node = f" [{issue.node}]" if issue.node else ""
            print(f"error: {issue.message}{node}", file=sys.stderr)
        return 1
    try:
        config = TrainingConfig.from_dict(
            {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "optimizer": args.optimizer,
                "weight_decay": args.weight_decay,
                "val_split": args.val_split,
                "device": args.device,
                "max_samples": args.max_samples,
                "seed": args.seed,
            }
        )
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    save = Path(args.save or f"{stem}.pt")
    with tempfile.TemporaryDirectory(prefix="brickdnn-") as tmp:
        try:
            info, path = resolve_data(args.data, Path(tmp))
        except DatasetError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        compat = check_compatibility(analysis, info)
        for warning in compat.warnings:
            print(f"warning: {warning}")
        if not compat.ok or compat.plan is None:
            for error in compat.errors:
                print(f"error: {error}", file=sys.stderr)
            return 1
        print(compat.plan.summary)
        for note in compat.plan.notes:
            print(f"note: {note}")

        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        from . import engine

        job = engine.TrainingJob(
            graph=graph,
            dataset=info,
            config=config,
            path=path,
            data_root=args.data_root,
            name=stem,
        )
        reporter = ConsoleReporter(config.epochs, save)
        restore = _catch_ctrl_c(reporter)
        try:
            result = engine.train(job, reporter, save)
        except KeyboardInterrupt:
            print("\nAborted, no weights saved.", file=sys.stderr)
            return 130
        finally:
            restore()
    return 1 if result.status == "failed" else 0


def _catch_ctrl_c(reporter: ConsoleReporter) -> Callable[[], None]:
    """First Ctrl+C: stop gracefully. Second Ctrl+C: abort. Returns a restore function."""

    def handler(signum: int, frame: Any) -> None:
        if reporter.stop:
            raise KeyboardInterrupt
        reporter.stop = True
        reporter._print("Stopping after the current batch … (press Ctrl+C again to abort)")

    try:
        previous = signal.signal(signal.SIGINT, handler)
    except ValueError:  # not in the main thread
        return lambda: None
    return lambda: signal.signal(signal.SIGINT, signal.SIG_DFL if previous is None else previous)
