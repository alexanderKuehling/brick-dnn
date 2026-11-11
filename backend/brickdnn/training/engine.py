"""Train a Brick DNN network with PyTorch.

The model is built from exactly the code that the export produces, so the saved
``weights.pt`` loads into the user's exported script::

    model = NeuralNetwork()
    model.load_state_dict(torch.load("weights.pt"))

How the inputs were prepared (normalisation, image size, CSV columns, class names) is saved
next to it, in ``weights.preprocessing.json`` (see :func:`.state.preprocessing_path`).

Use :func:`train` with a :class:`TrainingJob` and a :class:`Reporter`; the web runner
(:mod:`.runner`) and the command line (:mod:`.cli`) only differ in their reporter.
A job with ``tuning`` settings runs a hyperparameter search (see :mod:`.tuning`).
"""

from __future__ import annotations

import gc
import math
import os
import random
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ..codegen import generate_code
from .compat import TrainingPlan, check_compatibility, segmentation_memory_error
from .config import TrainingConfig
from .data import PreparedData, Source, iterate_batches, prepare_data
from .datasets import DEFAULT_DATA_ROOT
from .devices import available_devices, default_device, device_label, select_device
from .job import TrainingJob, TrainingResult
from .reporting import (
    DivergedError,
    MemoryLimitError,
    Reporter,
    StopTraining,
    TrainingError,
    short_error,
)
from .segmentation import class_colors
from .segmetrics import SegmentationLoss, SegmentationMetrics, predicted_classes
from .state import finite, preprocessing_path, write_json_atomic
from .tuning import TuningConfig, describe_params, trial_config

# Progress callbacks per second (at most).
PROGRESS_INTERVAL = 0.5
# How often the stop flag is checked (seconds).
STOP_CHECK_INTERVAL = 0.2
# Segmentation: what an evaluation adds to loss, accuracy, rmse and r2 (the means over the
# classes and the values per class), and the keys of a trial's validation results.
SEGMENTATION_KEYS = ("iou", "dice", "pixel_accuracy")
CLASS_KEYS = ("class_iou", "class_dice")
TRIAL_SEGMENTATION_KEYS = ("val_iou", "val_dice", "val_pixel_accuracy")

__all__ = [
    "DEFAULT_DATA_ROOT",
    "Reporter",
    "StopTraining",
    "TrainingError",
    "TrainingJob",
    "TrainingResult",
    "available_devices",
    "build_model",
    "default_device",
    "device_label",
    "select_device",
    "train",
]


def build_model(graph: Any) -> nn.Module:
    """Instantiate the generated ``NeuralNetwork`` class for ``graph``.

    The generated code only contains validated numbers and identifiers, so running it
    is safe. Raises :class:`~brickdnn.GraphValidationError` for invalid graphs.
    """
    code = generate_code(graph, {"include_main": False})
    namespace: dict[str, Any] = {"__name__": "brickdnn_model"}
    exec(compile(code, "<brickdnn model>", "exec"), namespace)
    return namespace["NeuralNetwork"]()


def train(
    job: TrainingJob,
    reporter: Reporter,
    weights_path: str | Path | None = None,
    previews_dir: str | Path | None = None,
) -> TrainingResult:
    """Run one training job to the end. Never raises (except ``KeyboardInterrupt``).

    Weights are saved to ``weights_path`` when the run completes or is stopped after at
    least one optimizer step: those of the epoch with the lowest validation loss (or of the
    last epoch without ``restore_best``); a search saves those of the best trial. The test
    set, if any, is measured with the same weights. The outcome is reported with
    ``reporter.finish`` and returned.

    A segmentation also writes preview images of a few validation samples into
    ``previews_dir`` (see :mod:`.previews`) once the weights are saved.

    Classical models (scikit-learn) are handed to :mod:`.classical_engine`, which saves
    the fitted model with joblib instead.
    """
    if job.family == "classical":
        from . import classical_engine

        return classical_engine.train(job, reporter, weights_path)
    if job.tuning is not None:
        return _Search(job, reporter, weights_path, previews_dir).run()
    return _Run(job, reporter, weights_path, previews_dir).run()


def _is_out_of_memory(exc: BaseException) -> bool:
    if isinstance(exc, MemoryError):
        return True
    oom = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom is not None and isinstance(exc, oom):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _is_batch_norm_error(exc: BaseException) -> bool:
    return isinstance(exc, ValueError) and "Expected more than 1 value per channel" in str(exc)


def _free_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        empty = getattr(getattr(torch, "mps", None), "empty_cache", None)
        if empty is not None:
            empty()


def _capitalized(text: str) -> str:
    return text[:1].upper() + text[1:]


def _metrics_text(
    name: str,
    loss: float,
    accuracy: float | None,
    rmse: float | None,
    r2: float | None,
    iou: float | None = None,
    pixel_accuracy: float | None = None,
) -> str:
    """``"validation accuracy 97.1%"``, ``"test loss 0.123, RMSE 4.5, R² 0.912"`` or, for a
    segmentation, ``"validation mean IoU 81.2%, pixel accuracy 96.4%"``."""
    if iou is not None:
        text = f"{name} mean IoU {iou * 100:.1f}%"
        if pixel_accuracy is not None:
            text += f", pixel accuracy {pixel_accuracy * 100:.1f}%"
        return text
    if accuracy is not None:
        return f"{name} accuracy {accuracy * 100:.1f}%"
    text = f"{name} loss {loss:.4g}"
    if rmse is not None:
        text += f", RMSE {rmse:.4g}"
    if r2 is not None:
        text += f", R² {r2:.3f}"
    return text


class _R2:
    """R² (coefficient of determination) of the value outputs, summed up batch by batch.

    Per output column it keeps the residual sum of squares and the sum of squares of the
    targets around their mean in this split (merged batch by batch, Chan et al.), so
    ``1 - SS_res / SS_tot`` measures against the split's own mean, not the training
    variance. Targets and predictions are standardised alike (shifted and scaled by the
    same numbers), which leaves R² unchanged: it is the R² in the units of the data.
    """

    def __init__(self) -> None:
        self.count = 0
        self.mean = self.spread = self.residual = torch.zeros(0, dtype=torch.float64)
        self.low = self.high = torch.zeros(0, dtype=torch.float64)

    def add(self, outputs: list[torch.Tensor], targets: list[torch.Tensor]) -> None:
        """One batch: the predictions and targets of every value output."""
        n = len(targets[0])
        if n == 0:
            return
        y = torch.cat([t.detach().cpu().double().reshape(n, -1) for t in targets], dim=1)
        pred = torch.cat([o.detach().cpu().double().reshape(n, -1) for o in outputs], dim=1)
        mean = y.mean(dim=0)
        spread = ((y - mean) ** 2).sum(dim=0)
        residual = ((y - pred) ** 2).sum(dim=0)
        low, high = y.min(dim=0).values, y.max(dim=0).values
        if self.count == 0:
            self.mean, self.spread, self.residual = mean, spread, residual
            self.low, self.high = low, high
        else:
            total = self.count + n
            delta = mean - self.mean
            self.spread = self.spread + spread + delta**2 * (self.count * n / total)
            self.mean = self.mean + delta * (n / total)
            self.residual = self.residual + residual
            self.low, self.high = torch.minimum(self.low, low), torch.maximum(self.high, high)
        self.count += n

    def value(self) -> float | None:
        """The mean R² over the output columns (a uniform average, like scikit-learn's
        default). None without samples, if the targets of a column are all equal in this
        split (R² is undefined then), and if it is not a finite number."""
        if self.count == 0 or not bool((self.high > self.low).all()):
            return None
        r2 = float((1 - self.residual / self.spread).mean())
        return r2 if math.isfinite(r2) else None


class _Run:
    def __init__(
        self,
        job: TrainingJob,
        reporter: Reporter,
        weights_path: str | Path | None,
        previews_dir: str | Path | None = None,
    ):
        self.job = job
        self.config = job.config
        self.reporter = reporter
        self.weights_path = Path(weights_path) if weights_path is not None else None
        self.previews_dir = Path(previews_dir) if previews_dir is not None else None
        self.segmentation = False
        # Segmentation: per-class results on the validation data, by epoch (of this trial)
        self.class_results: dict[int, dict[str, Any]] = {}
        self.device = torch.device("cpu")
        self.device_label = device_label("cpu")
        self.model: nn.Module | None = None
        self.steps = 0  # optimizer steps so far
        self.history: list[dict[str, Any]] = []
        self.best: dict[str, Any] | None = None
        self.test_result: dict[str, Any] | None = None
        self.trials: list[dict[str, Any]] = []
        self.best_trial: dict[str, Any] | None = None
        self.seed = 0
        self._last_stop_check = 0.0
        # Copy of the weights after the best epoch so far (restore_best), on the CPU
        self.best_weights: dict[str, torch.Tensor] | None = None
        self.weights_epoch: int | None = None  # the epoch whose weights are kept
        self.stopped_early = False
        # The final evaluation of the test set, collected while it is tested
        self.test_collector: _Collector | None = None

    # -- outcome ---------------------------------------------------------- #

    def run(self) -> TrainingResult:
        try:
            try:
                self._execute()
            except StopTraining:
                self._on_stopped()
                self._report_kept()
                weights = self._save_weights()
                self._report_segmentation(weights)
                return self._end("stopped", self._stopped_message(), weights)
            weights = self._save_weights()
            self._report_segmentation(weights)
            return self._end("completed", self._summary(), weights)
        except TrainingError as exc:
            return self._end("failed", str(exc), False)
        except Exception as exc:
            traceback.print_exc()
            return self._end("failed", self._explain(exc), False)

    def _execute(self) -> None:
        self._prepare()
        self._fit()
        self._keep_best()
        self._report_kept()
        self._test()
        self._report_evaluation()

    def _on_stopped(self) -> None:
        """Called when the user stopped the run, before the weights are saved: keeps the
        weights of the best epoch so far (restore_best), not those of the unfinished one."""
        if self.weights_epoch is None and self.history:
            self._keep_best(finished=False)

    def _stopped_message(self) -> str:
        if self.steps == 0:
            return "Stopped before the training started."
        text = f"Stopped after {len(self.history)} of {self.config.epochs} epochs."
        if self.weights_epoch is not None:
            text += f" Kept the weights of epoch {self.weights_epoch}"
            text += " (lowest validation loss)." if self.config.restore_best else "."
        return text

    def _explain(self, exc: Exception) -> str:
        if _is_out_of_memory(exc):
            return (
                f"Out of memory on {self.device_label}. "
                "Try a smaller batch size or a smaller network."
            )
        if _is_batch_norm_error(exc):
            return (
                "Batch normalization needs at least 2 samples per batch. Increase the batch size."
            )
        return f"Training failed: {short_error(exc)}"

    def _end(self, status: str, message: str, weights: bool) -> TrainingResult:
        self.reporter.finish(status, message, weights)
        best_trial = self.best_trial["trial"] if self.best_trial else None
        return TrainingResult(
            status,
            message,
            weights,
            self.history,
            self.best,
            self.test_result,
            self.trials,
            best_trial,
            self.weights_epoch if weights else None,
            self.stopped_early,
        )

    def _kept_entry(self) -> dict[str, Any] | None:
        """The history entry of the kept weights (the last epoch if not chosen yet)."""
        for entry in self.history:
            if entry["epoch"] == self.weights_epoch:
                return entry
        return self.history[-1] if self.history else None

    def _result_text(self) -> str:
        """Validation (and test) result of the kept weights."""
        kept = self._kept_entry()
        assert kept is not None
        text = _metrics_text(
            "validation",
            kept["val_loss"],
            kept["val_accuracy"],
            kept.get("val_rmse"),
            kept.get("val_r2"),
            kept.get("val_iou"),
            kept.get("val_pixel_accuracy"),
        )
        if self.test_result is not None:
            test = self.test_result
            text += "; " + _metrics_text(
                "test",
                test["loss"],
                test["accuracy"],
                test["rmse"],
                test["r2"],
                test.get("iou"),
                test.get("pixel_accuracy"),
            )
        return text

    def _summary(self) -> str:
        result = self._result_text()
        last = self.history[-1]["epoch"]
        kept = self.weights_epoch
        if self.stopped_early:
            text = (
                f"Stopped early after epoch {last}: no improvement for "
                f"{self.config.patience} epochs"
            )
            if kept is not None and kept != last:
                text += f"; kept the weights of epoch {kept}"
            return f"{text}. {_capitalized(result)}."
        if kept is not None and kept != last:
            return (
                f"Finished {last} epochs; kept the weights of epoch {kept} (lowest validation "
                f"loss). {_capitalized(result)}."
            )
        return f"Finished: {result}"

    def _report_kept(self) -> None:
        self.reporter.weights_kept(
            {"weights_epoch": self.weights_epoch, "stopped_early": self.stopped_early}
        )

    def _save_weights(self) -> bool:
        if self.model is None or self.steps == 0:
            return False
        return self._write_weights(self._weights_copy())

    def _write_weights(self, state: dict[str, torch.Tensor]) -> bool:
        """Save ``state`` and, next to it, how the inputs were prepared: the statistics and
        settings that new data needs to be prepared the same way (not for demo data)."""
        if self.weights_path is None:
            return False
        tmp = self.weights_path.with_name(f".{self.weights_path.name}.tmp")
        torch.save(state, tmp)
        os.replace(tmp, self.weights_path)
        preprocessing = self.data.preprocessing
        if preprocessing.get("kind") != "demo":
            write_json_atomic(preprocessing_path(self.weights_path), finite(preprocessing))
        return True

    def _check_stop(self) -> None:
        now = time.monotonic()
        if now - self._last_stop_check >= STOP_CHECK_INTERVAL:
            self._last_stop_check = now
            if self.reporter.should_stop():
                raise StopTraining

    # -- preparation ------------------------------------------------------- #

    def _prepare(self) -> None:
        self._prepare_data()
        self._check_memory(self.config)
        self._build(self.config)
        self.reporter.setup(self._setup_info())
        self._to_device(check=True)
        self.optimizer = self._optimizer()

    def _prepare_data(self) -> None:
        """Check the network, choose seed and device, load the data and pick the losses."""
        job, config, reporter = self.job, self.config, self.reporter
        reporter.status("preparing", "Preparing the data …")
        compat = check_compatibility(job.graph, job.dataset)
        if not compat.ok or compat.plan is None:
            raise TrainingError(
                compat.errors[0] if compat.errors else "The network cannot be trained."
            )
        self.plan: TrainingPlan = compat.plan

        seed = config.seed
        if seed is None:
            seed = random.SystemRandom().randrange(2**31)
            reporter.log(f"Random seed: {seed}")
        self.seed = seed
        self.device = select_device(config.device)
        self.device_label = device_label(self.device)

        generator = torch.Generator().manual_seed(seed)
        data_root = job.data_root or (
            str(DEFAULT_DATA_ROOT) if job.dataset.kind == "builtin" else None
        )
        self.data: PreparedData = prepare_data(
            self.plan, job.dataset, job.path, config, data_root, reporter, generator
        )
        self._check_stop()

        targets = self.plan.targets
        self.losses = [self._loss_function(t) for t in targets]
        self.accuracy = len(targets) == 1 and targets[0]["kind"] == "class"
        self.regression = self.plan.task == "regression"
        self.segmentation = self.plan.task == "segmentation"

    def _loss_function(self, target: dict[str, Any]) -> nn.Module:
        if target["kind"] == "mask":
            return SegmentationLoss(
                int(target["num_classes"]),
                bool(target.get("binary")),
                dice=self.job.config.loss == "cross_entropy_dice",
            )
        return nn.CrossEntropyLoss() if target["kind"] == "class" else nn.MSELoss()

    def _check_memory(self, config: TrainingConfig) -> None:
        """Segmentation: refuse a batch that would not fit into memory (the process would be
        killed instead of failing with a message)."""
        if self.segmentation and self.plan.image is not None:
            error = segmentation_memory_error(
                self.plan.segmentation or {}, self.plan.image, config.batch_size
            )
            if error is not None:
                raise MemoryLimitError(error)

    def _build(self, config: TrainingConfig) -> None:
        """A fresh model and shuffling order for ``config``.

        The seed fixes the initial weights and the order of the batches, so every trial of
        a search starts from the same point and only its settings make the difference.
        """
        self.config = config
        self.history = []
        self.best = None
        self.best_weights = None
        self.weights_epoch = None
        self.stopped_early = False
        self.steps = 0
        self.class_results = {}
        self.shuffle = torch.Generator().manual_seed(self.seed + 1)
        torch.manual_seed(self.seed)
        self.model = build_model(self.job.graph)
        n_train = len(self.data.train)
        # A last batch of a single sample breaks BatchNorm ("Expected more than 1 value per
        # channel"), so it is skipped; every other sample is used in every epoch.
        self.drop_last = n_train > config.batch_size and n_train % config.batch_size == 1
        self.steps_per_epoch = math.ceil(n_train / config.batch_size) - int(self.drop_last)

    def _setup_info(self) -> dict[str, Any]:
        assert self.model is not None
        return {
            "device": self.device.type,
            "device_label": self.device_label,
            "task": self.plan.task,
            "num_classes": self.plan.num_classes,
            "class_names": self.plan.class_names,
            "train_size": len(self.data.train),
            "val_size": len(self.data.val),
            "test_size": len(self.data.test) if self.data.test is not None else None,
            "epochs": self.config.epochs,
            "steps_per_epoch": self.steps_per_epoch,
            "total_params": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "preprocessing": self.data.preprocessing,
        }

    def _to_device(self, check: bool) -> None:
        """Move the model to the device; ``check`` tests the Apple GPU first (see _try_mps)."""
        assert self.model is not None
        self.model.to(self.device)
        if check and self.device.type == "mps":
            self._try_mps()

    def _try_mps(self) -> None:
        """Check one batch on the Apple GPU against the CPU; fall back to the CPU if needed.

        PyTorch's MPS backend lacks some operations, and on some Macs it computes wrong
        gradients without any error. So the gradients of a small batch are computed on
        both devices (in eval mode, so that dropout is off) and compared. The model state
        is restored afterwards.
        """
        assert self.model is not None
        model = self.model
        saved = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        inputs, targets = next(iter(iterate_batches(self.data.train, self.config.batch_size)))
        inputs, targets = [x[:16] for x in inputs], [y[:16] for y in targets]
        mps = self.device
        reason = None
        try:
            model.eval()
            reference = self._gradients(torch.device("cpu"), inputs, targets)
            on_mps = self._gradients(mps, inputs, targets)
            for name, expected in reference.items():
                error = (on_mps[name] - expected).norm() / (expected.norm() + 1e-6)
                if not math.isfinite(float(error)) or error > 1e-2:
                    reason = (
                        "The Apple GPU computed wrong gradients for this network (a known "
                        "problem of PyTorch on some Macs); training on the CPU instead."
                    )
                    break
        except (NotImplementedError, RuntimeError) as exc:
            if _is_out_of_memory(exc) or (
                isinstance(exc, RuntimeError) and "mps" not in str(exc).lower()
            ):
                raise
            traceback.print_exc()
            reason = (
                "The Apple GPU does not support every operation of this network; "
                "training on the CPU instead."
            )
        finally:
            self.device = mps
            model.to(mps)
            model.load_state_dict(saved)
            model.zero_grad(set_to_none=True)
            model.train()
        if reason is not None:
            self.reporter.log(reason)
            self.device = torch.device("cpu")
            self.device_label = device_label(self.device)
            model.to(self.device)
            self.reporter.setup({"device": "cpu", "device_label": self.device_label})

    def _gradients(
        self, device: torch.device, inputs: list[torch.Tensor], targets: list[torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Gradients of the loss of one batch on ``device`` (on the CPU, by parameter name)."""
        assert self.model is not None
        self.device = device
        self.model.to(device)
        self.model.zero_grad(set_to_none=True)
        self._loss(self._forward(inputs), targets).backward()
        return {
            name: (p.grad.detach().cpu() if p.grad is not None else torch.zeros(p.shape))
            for name, p in self.model.named_parameters()
        }

    def _optimizer(self) -> torch.optim.Optimizer:
        assert self.model is not None
        params = self.model.parameters()
        lr, wd = self.config.learning_rate, self.config.weight_decay
        if self.config.optimizer == "sgd":
            return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
        if self.config.optimizer == "adamw":
            return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)

    # -- training ---------------------------------------------------------- #

    def _forward(self, inputs: list[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        assert self.model is not None
        out = self.model(*[x.to(self.device) for x in inputs])
        return out if isinstance(out, tuple) else (out,)

    def _loss(self, outputs: tuple[torch.Tensor, ...], targets: list[torch.Tensor]):
        return sum(
            fn(out, y.to(self.device))
            for fn, out, y in zip(self.losses, outputs, targets, strict=True)
        )

    def _correct(self, outputs: tuple[torch.Tensor, ...], targets: list[torch.Tensor]) -> int:
        return int((outputs[0].argmax(dim=1).cpu() == targets[0]).sum())

    def _not_finite(self, value: float, epoch: int) -> TrainingError:
        what = "NaN" if math.isnan(value) else "infinite"
        return DivergedError(f"The loss became {what} in epoch {epoch}. Try a lower learning rate.")

    def _epoch_message(self, epoch: int) -> str:
        return f"Epoch {epoch} of {self.config.epochs}"

    def _weights_copy(self) -> dict[str, torch.Tensor]:
        assert self.model is not None
        return {k: v.detach().to("cpu", copy=True) for k, v in self.model.state_dict().items()}

    def _fit(self) -> None:
        """Train for the planned epochs, or fewer if early stopping (``patience``) ends the
        training. With ``restore_best``, the weights of the best epoch are copied aside."""
        epochs, patience = self.config.epochs, self.config.patience
        for epoch in range(1, epochs + 1):
            self.reporter.status("running", self._epoch_message(epoch))
            started = time.perf_counter()
            train_loss, train_acc = self._train_epoch(epoch)
            validated = self._validate(epoch)
            val_loss, val_acc = validated["loss"], validated["accuracy"]
            entry: dict[str, Any] = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
                "seconds": round(time.perf_counter() - started, 3),
            }
            if self.regression:
                entry["val_rmse"] = validated["rmse"]
                entry["val_r2"] = validated["r2"]
            if self.segmentation:
                entry["val_iou"] = validated["iou"]
                entry["val_dice"] = validated["dice"]
                entry["val_pixel_accuracy"] = validated["pixel_accuracy"]
                self.class_results[epoch] = {
                    key: validated[key] for key in (*SEGMENTATION_KEYS, *CLASS_KEYS)
                }
            self.history.append(entry)
            if self.best is None or val_loss < self.best["val_loss"]:
                self.best = {"epoch": epoch, "val_loss": val_loss, "val_accuracy": val_acc}
                if self.segmentation:
                    self.best["val_iou"] = validated["iou"]
                if self.config.restore_best:
                    self.best_weights = self._weights_copy()
            self.reporter.epoch_end(dict(entry), dict(self.best))
            if patience is not None and epoch < epochs and epoch - self.best["epoch"] >= patience:
                self.stopped_early = True
                self.reporter.log(
                    f"Stopping early after epoch {epoch}: the validation loss has not improved "
                    f"for {patience} epochs (best: epoch {self.best['epoch']})."
                )
                break

    def _keep_best(self, finished: bool = True) -> None:
        """Choose the weights that are tested and saved and put them into the model: those
        of the best epoch with ``restore_best``, otherwise those of the last epoch (or, for
        an unfinished run, the current ones). Sets ``weights_epoch``."""
        if self.config.restore_best and self.best is not None and self.best_weights is not None:
            assert self.model is not None
            self.model.load_state_dict(self.best_weights)
            self.weights_epoch = self.best["epoch"]
        elif finished and self.history:
            self.weights_epoch = self.history[-1]["epoch"]
        self.best_weights = None

    def _train_epoch(self, epoch: int) -> tuple[float, float | None]:
        assert self.model is not None
        self.model.train()
        loss_sum, seen, correct = 0.0, 0, 0
        started = last_report = time.perf_counter()
        batches = iterate_batches(
            self.data.train,
            self.config.batch_size,
            shuffle=True,
            drop_last=self.drop_last,
            generator=self.shuffle,
        )
        for step, (inputs, targets) in enumerate(batches, start=1):
            self._check_stop()
            outputs = self._forward(inputs)
            loss = self._loss(outputs, targets)
            value = float(loss.item())
            if not math.isfinite(value):
                raise self._not_finite(value, epoch)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            self.steps += 1

            n = len(targets[0])
            loss_sum += value * n
            seen += n
            if self.accuracy:
                correct += self._correct(outputs, targets)
            now = time.perf_counter()
            if now - last_report >= PROGRESS_INTERVAL:
                last_report = now
                self.reporter.progress(
                    {
                        "epoch": epoch,
                        "step": step,
                        "steps": self.steps_per_epoch,
                        "loss": loss_sum / seen,
                        "samples_per_second": round(seen / max(now - started, 1e-9), 1),
                    }
                )
        seen = max(seen, 1)
        return loss_sum / seen, (correct / seen if self.accuracy else None)

    def _validate(self, epoch: int) -> dict[str, Any]:
        result = self._evaluate(self.data.val)
        if not math.isfinite(result["loss"]):
            raise self._not_finite(result["loss"], epoch)
        return result

    def _evaluate(
        self,
        source: Source,
        collect: Callable[[tuple[torch.Tensor, ...], list[torch.Tensor]], None] | None = None,
    ) -> dict[str, Any]:
        """``{"loss", "accuracy", "rmse", "r2"}`` on ``source``: the accuracy for a single
        classification output, RMSE (in the units of the data) and R² for a regression,
        otherwise None. A segmentation adds ``"iou"`` and ``"dice"`` (means over the classes
        that occur), ``"pixel_accuracy"`` and per class ``"class_iou"`` and ``"class_dice"``.
        ``collect`` gets the outputs and targets of every batch (the final evaluation)."""
        assert self.model is not None
        self.model.eval()
        loss_sum, seen, correct = 0.0, 0, 0
        r2 = _R2() if self.regression else None
        masks = None
        if self.segmentation:
            target = self.plan.targets[0]
            masks = SegmentationMetrics(int(target["num_classes"]), bool(target.get("binary")))
        values = [i for i, t in enumerate(self.plan.targets) if t["kind"] == "value"]
        with torch.no_grad():
            for inputs, targets in iterate_batches(source, self.config.batch_size):
                self._check_stop()
                outputs = self._forward(inputs)
                n = len(targets[0])
                loss_sum += float(self._loss(outputs, targets).item()) * n
                seen += n
                if self.accuracy:
                    correct += self._correct(outputs, targets)
                if r2 is not None:
                    r2.add([outputs[i] for i in values], [targets[i] for i in values])
                if masks is not None:
                    masks.add(outputs[0], targets[0])
                if collect is not None:
                    collect(outputs, targets)
        seen = max(seen, 1)
        loss = loss_sum / seen
        rmse = None
        if self.regression and self.data.target_std is not None and math.isfinite(loss):
            rmse = math.sqrt(loss) * self.data.target_std
        result = {
            "loss": loss,
            "accuracy": correct / seen if self.accuracy else None,
            "rmse": rmse,
            "r2": r2.value() if r2 is not None else None,
        }
        if masks is not None:
            result.update(masks.result())
        return result

    def _test(self) -> None:
        """Measure the kept weights once on the test set (if there is one)."""
        if self.data.test is None:
            return
        self.reporter.status("running", "Testing the final model …")
        self.test_collector = self._collector()
        collect = self.test_collector.add if self.test_collector is not None else None
        self.test_result = self._evaluate(self.data.test, collect)
        self.reporter.test_end(dict(self.test_result))

    # -- final evaluation -------------------------------------------------- #

    def _evaluation_task(self) -> str | None:
        """ "classification" or "regression" if the run gets a final evaluation: networks
        with a single class output or a single value (not segmentations)."""
        targets = self.plan.targets
        if self.segmentation or len(targets) != 1:
            return None
        target = targets[0]
        if target["kind"] == "class":
            return "classification"
        if target["kind"] == "value" and math.prod(target.get("shape") or [1]) == 1:
            return "regression"
        return None

    def _collector(self) -> _Collector | None:
        task = self._evaluation_task()
        if task is None:
            return None
        try:
            return _Collector(task, self.plan.targets[0], self.data)
        except ImportError:  # NumPy is missing
            return None

    def _report_evaluation(self) -> None:
        """Evaluate the kept weights on the validation data (the test data was collected
        while it was tested) and report it (state.json key ``evaluation``). A failure is
        logged and does not fail the run."""
        collector = self._collector()
        if collector is None:
            return
        try:
            self.reporter.status("running", "Evaluating the final model …")
            validated = self._evaluate(self.data.val, collector.add)
            tested = self.test_collector if self.test_result is not None else None
            for part in (collector, tested):
                if part is not None and part.error is not None:
                    raise part.error
            from .evaluation import class_names_for, evaluation

            kept = self._kept_entry() or {}
            target = self.plan.targets[0]
            names = None
            if collector.task == "classification":
                names = class_names_for(self.plan.class_names, int(target["num_classes"]))
            result = evaluation(
                collector.task,
                collector.result(validated["loss"]),
                tested.result(self.test_result["loss"])
                if tested is not None and self.test_result is not None
                else None,
                {"loss": kept.get("train_loss"), "accuracy": kept.get("train_accuracy")},
                class_names=names,
                epoch=self.weights_epoch,
                trial=self.best_trial["trial"] if self.best_trial else None,
            )
            self.reporter.evaluation_end(result)
        except StopTraining:
            raise
        except Exception as exc:
            traceback.print_exc()
            self.reporter.log(f"The final evaluation failed: {short_error(exc)}")

    # -- segmentation ------------------------------------------------------ #

    def _kept_model(self) -> nn.Module | None:
        """The model with the weights that were saved."""
        return self.model

    def _report_segmentation(self, weights: bool) -> None:
        """Segmentation, once the weights are saved: write the preview images and report the
        per-class results of the kept weights. Never fails the run."""
        if not self.segmentation or not (weights or self.history):
            return  # nothing was trained
        try:
            plan, target = self.plan, self.plan.targets[0]
            names = list(plan.class_names or [])[: int(target["num_classes"])]
            kept = self.class_results.get(self.weights_epoch or 0)
            test = self.test_result
            result: dict[str, Any] = {
                "loss": self.job.config.loss,
                "binary": bool(target.get("binary")),
                "ignore_index": (plan.segmentation or {}).get("ignore_index"),
                "class_names": names,
                "colors": class_colors(len(names)),
                "validation": {"epoch": self.weights_epoch, **kept} if kept else None,
                "test": (
                    {key: test.get(key) for key in (*SEGMENTATION_KEYS, *CLASS_KEYS)}
                    if test is not None
                    else None
                ),
                "previews": self._write_previews(names),
            }
            self.reporter.segmentation_end(result)
        except Exception as exc:
            traceback.print_exc()
            self.reporter.log(f"The preview images could not be written: {short_error(exc)}")

    def _write_previews(self, names: list[str]) -> dict[str, Any] | None:
        """Predict the first validation samples with the kept weights and save them as
        images (see previews.py). Returns ``{"count", "epoch", "split"}``, or None."""
        model = self._kept_model()
        if self.previews_dir is None or model is None or len(self.data.val) == 0:
            return None
        from .previews import PREVIEW_SAMPLES, save_previews

        count = min(PREVIEW_SAMPLES, len(self.data.val))
        inputs, targets = self.data.val.batch(torch.arange(count))
        model.eval()
        with torch.no_grad():
            out = model(*[x.to(self.device) for x in inputs])
        output = out[0] if isinstance(out, tuple) else out
        binary = bool(self.plan.targets[0].get("binary"))
        predicted = predicted_classes(output, binary).cpu()
        preprocessing = self.data.preprocessing
        mean = torch.tensor(preprocessing["mean"], dtype=torch.float32).view(1, -1, 1, 1)
        std = torch.tensor(preprocessing["std"], dtype=torch.float32).view(1, -1, 1, 1)
        pixels = ((inputs[0].cpu() * std + mean) * 255.0).round().clamp(0, 255).to(torch.uint8)
        index = save_previews(
            self.previews_dir, pixels, targets[0], predicted, names, self.weights_epoch
        )
        if index is None:
            self.reporter.log("Preview images need Pillow: pip install pillow")
            return None
        return {"count": len(index["samples"]), "epoch": index["epoch"], "split": index["split"]}


class _Search(_Run):
    """Hyperparameter search: train one model per trial on the same data, keep the best.

    The data is prepared once. Every trial starts from the same initial weights and batch
    order; a trial whose loss explodes or that does not fit into memory fails on its own
    and the search goes on. Each trial keeps the weights of its best epoch (restore_best,
    otherwise of its last one) and is ranked by their validation loss; the best trial is
    kept in memory, tested and saved at the end.
    """

    def __init__(
        self,
        job: TrainingJob,
        reporter: Reporter,
        weights_path: str | Path | None,
        previews_dir: str | Path | None = None,
    ):
        super().__init__(job, reporter, weights_path, previews_dir)
        assert job.tuning is not None
        self.tuning: TuningConfig = job.tuning
        self.base_config = job.config
        self.total = 0
        self.trial = 0
        self.best_state: dict[str, torch.Tensor] | None = None
        self.best_history: list[dict[str, Any]] = []
        self.best_epoch: dict[str, Any] | None = None
        self.best_class_results: dict[int, dict[str, Any]] = {}
        self.device_checked = False  # the Apple GPU check (_try_mps) is done once

    def _execute(self) -> None:
        self._prepare_data()
        trials = self.tuning.sample(self.seed)
        self.total = len(trials)
        for index, params in enumerate(trials, start=1):
            self._check_stop()
            self._run_trial(index, params)
        if self.best_trial is None:
            first = next(t for t in self.trials if t["status"] == "failed")
            raise TrainingError(
                f"All {self.total} trials failed. Trial {first['trial']}: {first['message']}"
            )
        self._use_best()
        self._report_kept()
        self._test()
        self._report_evaluation()

    def _epoch_message(self, epoch: int) -> str:
        return f"Trial {self.trial} of {self.total}: epoch {epoch} of {self.config.epochs}"

    def _run_trial(self, index: int, params: dict[str, Any]) -> None:
        self.trial = index
        self._build(trial_config(self.base_config, params))
        if index == 1:
            info = self._setup_info()
            info.update(tuning=self.tuning.to_dict(), trials_total=self.total)
            self.reporter.setup(info)
        self.reporter.trial_start(
            {
                "trial": index,
                "trials": self.total,
                "params": dict(params),
                "steps_per_epoch": self.steps_per_epoch,
            }
        )
        entry: dict[str, Any] = {
            "trial": index,
            "params": dict(params),
            "status": "running",
            "message": "",
            "epochs_done": 0,
            "train_loss": None,
            "val_loss": None,
            "val_accuracy": None,
            "val_rmse": None,
            "val_r2": None,
            "weights_epoch": None,
            "stopped_early": False,
            "seconds": None,
        }
        if self.segmentation:
            entry.update(dict.fromkeys(TRIAL_SEGMENTATION_KEYS))
        self.trials.append(entry)
        started = time.perf_counter()
        failure: str | None = None
        try:
            self._check_memory(self.config)
            self._to_device(check=not self.device_checked)
            self.device_checked = True
            self.optimizer = self._optimizer()
            self._fit()
            self._keep_best()
        except StopTraining:
            self._trial_done(entry, "stopped", "Stopped.", started)
            raise
        except Exception as exc:
            if not (
                isinstance(exc, (DivergedError, MemoryLimitError))
                or _is_out_of_memory(exc)
                or _is_batch_norm_error(exc)
            ):
                raise
            failure = str(exc) if isinstance(exc, TrainingError) else self._explain(exc)
        if failure is not None:
            self.reporter.log(f"Trial {index} failed: {failure}")
            self.model = None
            self.optimizer = None  # type: ignore[assignment]
            _free_memory(self.device)
            self._trial_done(entry, "failed", failure, started)
            return

        best = self.best_trial
        if best is None or entry_score(self.history, self.weights_epoch) < best["val_loss"]:
            self.best_state = self._weights_copy()
            self.best_history = list(self.history)
            self.best_epoch = dict(self.best) if self.best else None
            self.best_class_results = dict(self.class_results)
            self._trial_done(entry, "completed", "", started, new_best=True)
        else:
            self._trial_done(entry, "completed", "", started)

    def _trial_done(
        self,
        entry: dict[str, Any],
        status: str,
        message: str,
        started: float,
        new_best: bool = False,
    ) -> None:
        kept = self._kept_entry() or {}  # completed: the kept epoch, otherwise the last
        entry.update(
            status=status,
            message=message,
            epochs_done=len(self.history),
            train_loss=kept.get("train_loss"),
            val_loss=kept.get("val_loss"),
            val_accuracy=kept.get("val_accuracy"),
            val_rmse=kept.get("val_rmse"),
            val_r2=kept.get("val_r2"),
            weights_epoch=self.weights_epoch,
            stopped_early=self.stopped_early,
            seconds=round(time.perf_counter() - started, 3),
        )
        if self.segmentation:
            entry.update({key: kept.get(key) for key in TRIAL_SEGMENTATION_KEYS})
        if new_best:
            self.best_trial = entry
        best = self.best_trial["trial"] if self.best_trial else None
        self.reporter.trial_end(dict(entry), best)

    def _use_best(self) -> None:
        """Make the best trial the result: its model, history and settings."""
        best = self.best_trial
        assert best is not None and self.best_state is not None
        self._build(trial_config(self.base_config, best["params"]))
        assert self.model is not None
        self.model.load_state_dict(self.best_state)
        self.model.to(self.device)
        self.history = list(self.best_history)
        self.best = self.best_epoch
        self.class_results = dict(self.best_class_results)
        self.weights_epoch = best["weights_epoch"]
        self.stopped_early = best["stopped_early"]
        self.trial = 0
        self.reporter.search_end(
            {
                "best_trial": best["trial"],
                "params": dict(best["params"]),
                "history": list(self.best_history),
                "best": dict(self.best_epoch) if self.best_epoch else None,
            }
        )

    def _on_stopped(self) -> None:
        """Keep the best finished trial, or the best epoch so far of the first trial."""
        if self.best_trial is None or self.best_state is None:
            super()._on_stopped()
        else:
            best = self.best_trial
            self.history = list(self.best_history)
            self.best = self.best_epoch
            self.class_results = dict(self.best_class_results)
            self.weights_epoch = best["weights_epoch"]
            self.stopped_early = best["stopped_early"]
            self.reporter.search_end(
                {
                    "best_trial": best["trial"],
                    "params": dict(best["params"]),
                    "history": list(self.best_history),
                    "best": dict(self.best_epoch) if self.best_epoch else None,
                }
            )

    def _finished_trials(self) -> int:
        return sum(1 for t in self.trials if t["status"] in ("completed", "failed"))

    def _stopped_message(self) -> str:
        done = self._finished_trials()
        if done == 0:
            if self.steps == 0:
                return "Stopped before the training started."
            return f"Stopped during trial {self.trial} of {self.total}."
        message = f"Stopped after {done} of {self.total} trials."
        if self.best_trial is not None:
            best = self.best_trial
            message += f" Best so far: trial {best['trial']} ({describe_params(best['params'])})."
        return message

    def _summary(self) -> str:
        best = self.best_trial
        assert best is not None
        failed = sum(1 for t in self.trials if t["status"] == "failed")
        head = f"Finished {self.total} trial{'s' if self.total != 1 else ''}"
        if failed:
            head += f" ({failed} failed)"
        kept = ""
        if self.stopped_early:
            kept = f"stopped early, weights of epoch {self.weights_epoch}: "
        elif self.history and self.weights_epoch != self.history[-1]["epoch"]:
            kept = f"weights of epoch {self.weights_epoch}: "
        return (
            f"{head}. Best: trial {best['trial']} ({describe_params(best['params'])}), "
            f"{kept}{self._result_text()}"
        )

    def _save_weights(self) -> bool:
        """The weights of the best trial, or of the first trial if it was stopped early."""
        if self.best_state is not None:
            return self._write_weights(self.best_state)
        return super()._save_weights()

    def _kept_model(self) -> nn.Module | None:
        """A model with the weights of the best trial (the current model may be that of a
        later trial when the search was stopped)."""
        if self.best_state is None:
            return super()._kept_model()
        model = build_model(self.job.graph)
        model.load_state_dict(self.best_state)
        return model.to(self.device)


class _Collector:
    """Collects the outputs of an evaluation pass for the final evaluation (see
    evaluation.py): class probabilities, or values in the units of the target. An error only
    switches it off; it is raised when the result is asked for."""

    def __init__(self, task: str, target: dict[str, Any], data: PreparedData):
        from .evaluation import ClassificationCounter, RegressionCollector

        self.task = task
        self.error: Exception | None = None
        self.sink: Any
        if task == "classification":
            self.sink = ClassificationCounter(int(target["num_classes"]))
        else:
            self.sink = RegressionCollector()
        # The targets were standardised (CSV data): back to the units of the column.
        std = data.target_std
        mean = data.preprocessing.get("target_mean") if std is not None else None
        self.scale = float(std) if std is not None else 1.0
        self.shift = float(mean) if isinstance(mean, (int, float)) else 0.0

    def add(self, outputs: tuple[torch.Tensor, ...], targets: list[torch.Tensor]) -> None:
        if self.error is not None:
            return
        try:
            output, target = outputs[0].detach(), targets[0].detach()
            if self.task == "classification":
                proba = torch.softmax(output.double(), dim=1).cpu().numpy()
                self.sink.add(proba, target.cpu().long().numpy())
            else:
                n = len(target)
                predicted = output.double().reshape(n).cpu().numpy() * self.scale + self.shift
                actual = target.double().reshape(n).cpu().numpy() * self.scale + self.shift
                self.sink.add(predicted, actual)
        except Exception as exc:
            self.error = exc

    def result(self, loss: float | None) -> dict[str, Any]:
        return self.sink.result(loss)


def entry_score(history: list[dict[str, Any]], weights_epoch: int | None = None) -> float:
    """What a trial is ranked by: the validation loss of the kept weights (those of
    ``weights_epoch``, by default the last epoch)."""
    for entry in history:
        if entry["epoch"] == weights_epoch:
            return float(entry["val_loss"])
    return float(history[-1]["val_loss"]) if history else math.inf
