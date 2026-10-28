"""Train a network from the command line (needs PyTorch).

brickdnn train --example cnn_mnist --data mnist --epochs 5
brickdnn train graph.json --data random                # demo data with a hidden rule
brickdnn train graph.json --data houses.csv --save houses.pt
brickdnn train graph.json --data photos.zip            # or a folder with one folder per class
brickdnn train graph.json --data houses.csv --test-split 0.2       # hold out a test set
brickdnn train graph.json --data mnist --tune --trials 20          # hyperparameter search
brickdnn train graph.json --data mnist --tune-lr 1e-4:1e-2 --tune-batch-size 32,64

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
from .tuning import LABELS, METHODS, TUNABLE, TuningConfig, describe_params, format_value

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
        self._trial: dict[str, Any] | None = None
        self._trials: list[dict[str, Any]] = []

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
        if info.get("test_size"):
            samples = (
                f"{info['train_size']:,} training, {info['val_size']:,} validation and "
                f"{info['test_size']:,} test samples"
            )
        else:
            samples = f"{info['train_size']:,} training and {info['val_size']:,} validation samples"
        if info.get("tuning"):
            tuning = TuningConfig.from_dict(info["tuning"])
            self._print(
                f"Training on {info['device_label']}: {samples}, "
                f"{info['total_params']:,} parameters."
            )
            self._print(
                f"Hyperparameter search: {tuning.describe()} of {self.epochs} "
                f"epoch{'s' if self.epochs != 1 else ''} each."
            )
            return
        self._print(
            f"Training on {info['device_label']}: {samples}, {info['steps_per_epoch']:,} "
            f"steps per epoch, {info['total_params']:,} parameters."
        )

    def progress(self, progress: dict[str, Any]) -> None:
        trial = f"Trial {self._trial['trial']}/{self._trial['trials']}  " if self._trial else ""
        self._show(
            f"{trial}Epoch {progress['epoch']:3d}/{self.epochs}  step {progress['step']}/"
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

    def trial_start(self, trial: dict[str, Any]) -> None:
        self._trial = trial
        self._print(
            f"\nTrial {trial['trial']}/{trial['trials']}: {describe_params(trial['params'])}"
        )

    def trial_end(self, trial: dict[str, Any], best_trial: int | None) -> None:
        self._trials.append(trial)
        name = f"Trial {trial['trial']}/{self._trial['trials'] if self._trial else '?'}"
        if trial["status"] == "failed":
            self._print(f"{name} failed: {trial['message']}")
        elif trial["status"] == "completed":
            best = "  (best so far)" if best_trial == trial["trial"] else ""
            self._print(f"{name}: {_metrics(trial, 'val')}{best}")

    def search_end(self, result: dict[str, Any]) -> None:
        self._trial = None
        if not self._trials:
            return
        names = [name for name in TUNABLE if name in self._trials[0]["params"]]
        columns = ["trial", *[LABELS[name] for name in names], "val loss"]
        accuracy = any(t.get("val_accuracy") is not None for t in self._trials)
        rmse = any(t.get("val_rmse") is not None for t in self._trials)
        columns += ["val acc"] if accuracy else ["val rmse"] if rmse else []
        ranked = sorted(
            self._trials,
            key=lambda t: (t["val_loss"] is None, t["val_loss"] or 0.0, t["trial"]),
        )
        rows = []
        for trial in ranked:
            mark = "*" if trial["trial"] == result["best_trial"] else " "
            cells = [f"{mark}{trial['trial']}"]
            cells += [format_value(name, trial["params"][name]) for name in names]
            if trial["status"] != "completed":
                cells.append(trial["status"])
            else:
                cells.append(f"{trial['val_loss']:.4f}")
                if accuracy:
                    cells.append(f"{trial['val_accuracy']:.1%}")
                elif rmse:
                    cells.append(f"{trial['val_rmse']:.4g}")
            rows.append(cells)
        widths = [
            max([len(columns[i])] + [len(row[i]) for row in rows if i < len(row)])
            for i in range(len(columns))
        ]
        self._print("\nTrials, best first:")
        self._print("  ".join(c.rjust(w) for c, w in zip(columns, widths, strict=True)))
        for row in rows:
            self._print("  ".join(c.rjust(w) for c, w in zip(row, widths, strict=False)))
        self._print(f"Best: trial {result['best_trial']} ({describe_params(result['params'])})")

    def test_end(self, result: dict[str, Any]) -> None:
        line = f"Test: loss {result['loss']:.4f}"
        if result.get("accuracy") is not None:
            line += f", accuracy {result['accuracy']:.1%}"
        if result.get("rmse") is not None:
            line += f", rmse {result['rmse']:.4g}"
        self._print(line)

    def finish(self, status: str, message: str, weights: bool) -> None:
        if status == "failed":
            self._print(f"error: {message}", err=True)
            return
        self._print(message)
        if weights:
            self._print(f"Saved weights to {self.save}")

    def should_stop(self) -> bool:
        return self.stop


def _metrics(trial: dict[str, Any], prefix: str) -> str:
    text = f"{prefix} loss {trial[f'{prefix}_loss']:.4f}"
    if trial.get(f"{prefix}_accuracy") is not None:
        text += f", {prefix} acc {trial[f'{prefix}_accuracy']:.1%}"
    if trial.get(f"{prefix}_rmse") is not None:
        text += f", {prefix} rmse {trial[f'{prefix}_rmse']:.4g}"
    return text


def parse_values(text: str, name: str) -> dict[str, float] | list[Any]:
    """``"1e-4:1e-2"`` -> a range, ``"32,64,128"`` -> a list (validated by TuningConfig)."""
    text = text.strip()
    if ":" in text:
        low, _, high = text.partition(":")
        return {"min": _to_number(low), "max": _to_number(high)}
    items = [item.strip() for item in text.split(",") if item.strip()]
    if name == "optimizer":
        return items
    return [_to_number(item) for item in items]


def _to_number(text: str) -> float | str:
    try:
        return float(text)
    except ValueError:
        return text.strip()  # TuningConfig reports it


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
    parser.add_argument(
        "--test-split",
        type=float,
        help="share of the data held out as a test set and only used after training "
        "(default 0: none; built-in datasets use their official test images)",
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
    search = parser.add_argument_group(
        "hyperparameter search",
        "Train several times with different settings on the same data and keep the trial "
        "with the lowest validation loss. Any of these options starts a search. VALUES is "
        "a list (32,64,128) or a range MIN:MAX for the learning rate and weight decay, "
        "sampled log-uniformly. Without --tune-* options, the learning rate (1e-4:1e-2) "
        "and the batch size (32,64,128) are tuned.",
    )
    search.add_argument("--tune", action="store_true", help="run a hyperparameter search")
    search.add_argument("--search", choices=METHODS, help="random (default) or grid")
    search.add_argument("--trials", type=int, help="trials of a random search (default 10)")
    search.add_argument("--tune-lr", metavar="VALUES", help="learning rates to try")
    search.add_argument("--tune-batch-size", metavar="VALUES", help="batch sizes to try")
    search.add_argument("--tune-optimizer", metavar="VALUES", help="optimizers to try")
    search.add_argument("--tune-weight-decay", metavar="VALUES", help="weight decays to try")
    return parser


TUNE_OPTIONS = {
    "learning_rate": "tune_lr",
    "batch_size": "tune_batch_size",
    "optimizer": "tune_optimizer",
    "weight_decay": "tune_weight_decay",
}


def tuning_from_args(args: argparse.Namespace) -> TuningConfig | None:
    """The search settings from the command line, or None for a normal training."""
    given = {
        name: getattr(args, option)
        for name, option in TUNE_OPTIONS.items()
        if getattr(args, option) is not None
    }
    if not (given or args.tune or args.search or args.trials is not None):
        return None
    space = {name: parse_values(text, name) for name, text in given.items()} or None
    return TuningConfig.from_dict({"method": args.search, "trials": args.trials, "space": space})


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
                "test_split": args.test_split,
                "device": args.device,
                "max_samples": args.max_samples,
                "seed": args.seed,
            }
        )
        tuning = tuning_from_args(args)
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
            tuning=tuning,
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
