"""Train a Brick DNN network with PyTorch.

The model is built from exactly the code that the export produces, so the saved
``weights.pt`` loads into the user's exported script::

    model = NeuralNetwork()
    model.load_state_dict(torch.load("weights.pt"))

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ..codegen import generate_code
from .compat import TrainingPlan, check_compatibility
from .config import ConfigError, TrainingConfig
from .data import PreparedData, Source, iterate_batches, prepare_data
from .datasets import DEFAULT_DATA_ROOT, DatasetInfo, builtin_info, random_info
from .devices import available_devices, default_device, device_label, select_device
from .reporting import DivergedError, Reporter, StopTraining, TrainingError, short_error
from .tuning import TuningConfig, describe_params, trial_config

# Progress callbacks per second (at most).
PROGRESS_INTERVAL = 0.5
# How often the stop flag is checked (seconds).
STOP_CHECK_INTERVAL = 0.2

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


@dataclass
class TrainingJob:
    """Everything needed for one run (the content of ``job.json``)."""

    graph: dict[str, Any]
    dataset: DatasetInfo
    config: TrainingConfig = field(default_factory=TrainingConfig)
    path: str | None = None  # CSV file or image folder
    data_root: str | None = None  # where torchvision datasets are downloaded
    name: str = ""
    tuning: TuningConfig | None = None  # set for a hyperparameter search

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingJob:
        """Read a job (see the contract / state.py). Raises :class:`TrainingError`."""
        if not isinstance(data, dict):
            raise TrainingError("The training job is invalid.")
        graph = data.get("graph")
        if not isinstance(graph, dict):
            raise TrainingError("The training job contains no network.")
        dataset = data.get("dataset") or {}
        if not isinstance(dataset, dict):
            raise TrainingError("The training job describes no valid dataset.")
        info = dataset.get("info")
        try:
            if isinstance(info, dict):
                dataset_info = DatasetInfo.from_dict(info)
            elif dataset.get("kind") == "builtin":
                dataset_info = builtin_info(str(dataset.get("id")))
            elif dataset.get("kind", "random") == "random":
                dataset_info = random_info()
            else:
                raise TrainingError("The training job describes no valid dataset.")
        except (TypeError, ValueError) as exc:
            raise TrainingError("The training job describes no valid dataset.") from exc
        try:
            config = TrainingConfig.from_dict(data.get("config"))
            tuning = data.get("tuning")
            tuning = TuningConfig.from_dict(tuning) if tuning is not None else None
        except ConfigError as exc:
            raise TrainingError(str(exc)) from exc
        path = dataset.get("path")
        data_root = data.get("data_root")
        return cls(
            graph=graph,
            dataset=dataset_info,
            config=config,
            path=str(path) if path else None,
            data_root=str(data_root) if data_root else None,
            name=str(data.get("name") or ""),
            tuning=tuning,
        )


@dataclass
class TrainingResult:
    status: str  # completed | stopped | failed
    message: str
    weights: bool = False
    history: list[dict[str, Any]] = field(default_factory=list)
    best: dict[str, Any] | None = None
    # {"loss", "accuracy", "rmse"} of the final model on the test set
    test: dict[str, Any] | None = None
    # Hyperparameter search: every trial (see state.py) and the number of the best one
    trials: list[dict[str, Any]] = field(default_factory=list)
    best_trial: int | None = None


def train(
    job: TrainingJob, reporter: Reporter, weights_path: str | Path | None = None
) -> TrainingResult:
    """Run one training job to the end. Never raises (except ``KeyboardInterrupt``).

    Weights are saved to ``weights_path`` when the run completes or is stopped after at
    least one optimizer step (a search saves the best trial). The outcome is reported with
    ``reporter.finish`` and returned.
    """
    if job.tuning is not None:
        return _Search(job, reporter, weights_path).run()
    return _Run(job, reporter, weights_path).run()


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


def _metrics_text(name: str, loss: float, accuracy: float | None, rmse: float | None) -> str:
    """``"validation accuracy 97.1%"`` or ``"test loss 0.123, RMSE 4.5"``."""
    if accuracy is not None:
        return f"{name} accuracy {accuracy * 100:.1f}%"
    text = f"{name} loss {loss:.4g}"
    if rmse is not None:
        text += f", RMSE {rmse:.4g}"
    return text


class _Run:
    def __init__(self, job: TrainingJob, reporter: Reporter, weights_path: str | Path | None):
        self.job = job
        self.config = job.config
        self.reporter = reporter
        self.weights_path = Path(weights_path) if weights_path is not None else None
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

    # -- outcome ---------------------------------------------------------- #

    def run(self) -> TrainingResult:
        try:
            try:
                self._execute()
            except StopTraining:
                self._on_stopped()
                weights = self._save_weights()
                return self._end("stopped", self._stopped_message(), weights)
            return self._end("completed", self._summary(), self._save_weights())
        except TrainingError as exc:
            return self._end("failed", str(exc), False)
        except Exception as exc:
            traceback.print_exc()
            return self._end("failed", self._explain(exc), False)

    def _execute(self) -> None:
        self._prepare()
        self._fit()
        self._test()

    def _on_stopped(self) -> None:
        """Called when the user stopped the run, before the weights are saved."""

    def _stopped_message(self) -> str:
        if self.steps == 0:
            return "Stopped before the training started."
        return f"Stopped after {len(self.history)} of {self.config.epochs} epochs."

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
        )

    def _result_text(self) -> str:
        """Validation (and test) result of the final model."""
        last = self.history[-1]
        text = _metrics_text(
            "validation", last["val_loss"], last["val_accuracy"], last.get("val_rmse")
        )
        if self.test_result is not None:
            test = self.test_result
            text += "; " + _metrics_text("test", test["loss"], test["accuracy"], test["rmse"])
        return text

    def _summary(self) -> str:
        return f"Finished: {self._result_text()}"

    def _save_weights(self) -> bool:
        if self.weights_path is None or self.model is None or self.steps == 0:
            return False
        state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
        tmp = self.weights_path.with_name(f".{self.weights_path.name}.tmp")
        torch.save(state, tmp)
        os.replace(tmp, self.weights_path)
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
        self.losses = [
            nn.CrossEntropyLoss() if t["kind"] == "class" else nn.MSELoss() for t in targets
        ]
        self.accuracy = len(targets) == 1 and targets[0]["kind"] == "class"
        self.regression = self.plan.task == "regression"

    def _build(self, config: TrainingConfig) -> None:
        """A fresh model and shuffling order for ``config``.

        The seed fixes the initial weights and the order of the batches, so every trial of
        a search starts from the same point and only its settings make the difference.
        """
        self.config = config
        self.history = []
        self.best = None
        self.steps = 0
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

    def _fit(self) -> None:
        epochs = self.config.epochs
        for epoch in range(1, epochs + 1):
            self.reporter.status("running", self._epoch_message(epoch))
            started = time.perf_counter()
            train_loss, train_acc = self._train_epoch(epoch)
            val_loss, val_acc, val_rmse = self._validate(epoch)
            entry: dict[str, Any] = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
                "seconds": round(time.perf_counter() - started, 3),
            }
            if self.regression:
                entry["val_rmse"] = val_rmse
            self.history.append(entry)
            if self.best is None or val_loss < self.best["val_loss"]:
                self.best = {"epoch": epoch, "val_loss": val_loss, "val_accuracy": val_acc}
            self.reporter.epoch_end(dict(entry), dict(self.best))

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

    def _validate(self, epoch: int) -> tuple[float, float | None, float | None]:
        val_loss, accuracy, rmse = self._evaluate(self.data.val)
        if not math.isfinite(val_loss):
            raise self._not_finite(val_loss, epoch)
        return val_loss, accuracy, rmse

    def _evaluate(self, source: Source) -> tuple[float, float | None, float | None]:
        """Loss, accuracy (single classification output) and RMSE (CSV regression)."""
        assert self.model is not None
        self.model.eval()
        loss_sum, seen, correct = 0.0, 0, 0
        with torch.no_grad():
            for inputs, targets in iterate_batches(source, self.config.batch_size):
                self._check_stop()
                outputs = self._forward(inputs)
                n = len(targets[0])
                loss_sum += float(self._loss(outputs, targets).item()) * n
                seen += n
                if self.accuracy:
                    correct += self._correct(outputs, targets)
        seen = max(seen, 1)
        loss = loss_sum / seen
        rmse = None
        if self.regression and self.data.target_std is not None and math.isfinite(loss):
            rmse = math.sqrt(loss) * self.data.target_std
        return loss, (correct / seen if self.accuracy else None), rmse

    def _test(self) -> None:
        """Measure the final model once on the test set (if there is one)."""
        if self.data.test is None:
            return
        self.reporter.status("running", "Testing the final model …")
        loss, accuracy, rmse = self._evaluate(self.data.test)
        self.test_result = {"loss": loss, "accuracy": accuracy, "rmse": rmse}
        self.reporter.test_end(dict(self.test_result))


class _Search(_Run):
    """Hyperparameter search: train one model per trial on the same data, keep the best.

    The data is prepared once. Every trial starts from the same initial weights and batch
    order; a trial whose loss explodes or that does not fit into memory fails on its own
    and the search goes on. The best trial (lowest validation loss after its last epoch)
    is kept in memory, tested and saved at the end.
    """

    def __init__(self, job: TrainingJob, reporter: Reporter, weights_path: str | Path | None):
        super().__init__(job, reporter, weights_path)
        assert job.tuning is not None
        self.tuning: TuningConfig = job.tuning
        self.base_config = job.config
        self.total = 0
        self.trial = 0
        self.best_state: dict[str, torch.Tensor] | None = None
        self.best_history: list[dict[str, Any]] = []
        self.best_epoch: dict[str, Any] | None = None
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
        self._test()

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
            "seconds": None,
        }
        self.trials.append(entry)
        started = time.perf_counter()
        failure: str | None = None
        try:
            self._to_device(check=not self.device_checked)
            self.device_checked = True
            self.optimizer = self._optimizer()
            self._fit()
        except StopTraining:
            self._trial_done(entry, "stopped", "Stopped.", started)
            raise
        except Exception as exc:
            if not (
                isinstance(exc, DivergedError)
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
        if best is None or entry_score(self.history) < best["val_loss"]:
            assert self.model is not None
            self.best_state = {
                k: v.detach().to("cpu", copy=True) for k, v in self.model.state_dict().items()
            }
            self.best_history = list(self.history)
            self.best_epoch = dict(self.best) if self.best else None
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
        last = self.history[-1] if self.history else {}
        entry.update(
            status=status,
            message=message,
            epochs_done=len(self.history),
            train_loss=last.get("train_loss"),
            val_loss=last.get("val_loss"),
            val_accuracy=last.get("val_accuracy"),
            val_rmse=last.get("val_rmse"),
            seconds=round(time.perf_counter() - started, 3),
        )
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
        if self.best_trial is not None and self.best_state is not None:
            best = self.best_trial
            self.history = list(self.best_history)
            self.best = self.best_epoch
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
        return (
            f"{head}. Best: trial {best['trial']} ({describe_params(best['params'])}), "
            f"{self._result_text()}"
        )

    def _save_weights(self) -> bool:
        """The weights of the best trial, or of the first trial if it was stopped early."""
        state = self.best_state
        if state is None and self.model is not None and self.steps > 0:
            state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
        if self.weights_path is None or state is None:
            return False
        tmp = self.weights_path.with_name(f".{self.weights_path.name}.tmp")
        torch.save(state, tmp)
        os.replace(tmp, self.weights_path)
        return True


def entry_score(history: list[dict[str, Any]]) -> float:
    """What a trial is ranked by: the validation loss after its last epoch."""
    return float(history[-1]["val_loss"]) if history else math.inf
