"""Train a hybrid network: neural layers learn the features, a classical model is fitted on
them (see :mod:`brickdnn.hybrid` for how the graph is split). Needs PyTorch and
scikit-learn.

Two phases, in one run:

1. **Learning the features** (:class:`_FeatureRun`, the PyTorch engine): the neural layers
   get a temporary linear head (one output per class, or one value for a regression; the
   task is the classical model's) and are trained like a network of their own, with the
   epochs, optimizer, early stopping and kept weights of the settings. Their curves are the
   run's ``history``. Then the head is dropped. With ``feature_training`` "frozen" nothing
   is trained: the random initial weights (fixed by the seed) make the features.
2. **Fitting the classical model** (:class:`_ClassicalFit`, the scikit-learn engine): the
   neural layers compute the features of the training, validation and test rows (the split
   of phase 1, so no validation row was learned from), the classical model (or ensemble) is
   fitted on the features of the training rows, at most its row limit of them, and measured
   and evaluated like any classical model, also with a search over its parameters.

The run directory gets ``weights.pt`` (``{"state_dict": <weights of the neural layers>,
"preprocessing": {...}}``, the format of the generated script's ``--weights``) with
``weights.preprocessing.json`` next to it, and ``model.joblib`` (the fitted pipeline on the
features, the classical format with ``"hybrid": {"features", "input_columns"}``).
``state.json`` keeps the curves of phase 1 in ``history`` and the results of phase 2 in
``classical`` (with ``validation``: the fitted model on the validation rows), ``test`` and
``evaluation``. ``preprocessing`` is that of the neural layers, without the statistics of
the target: the classical model predicts in the units of the target column.

A search (see tuning.py) varies the parameters of the classical model, the training
settings of the neural layers, or both. Without neural settings phase 1 runs once and phase
2 is the search of the classical model (classical_search.py, also with cross-validation).
With them (:class:`HybridSearch`) every trial runs both phases: the neural layers learn the
features with the trial's settings (the data is prepared once, every trial starts from the
same initial weights), the classical model is fitted with the trial's parameters on the
features of the training rows, and the trial is ranked by that model on the validation rows.
The best trial's neural layers and fitted model are kept, tested and evaluated.

Stopping during phase 1 keeps the features learned so far (the best epoch so far with
``restore_best``, as a network does) and fits the classical model on them, if the estimate
of that fit (with the feature pass) is at most :data:`STOP_FIT_CPU_SECONDS`: the server
kills a runner that has not ended soon after a stop request, and a run stopped for its
budget should not use much more. That fit cannot be stopped; the model is measured on the
training and validation rows and saved, but not tested or evaluated (like a stopped
network). Without a finished epoch, or when the fit would take longer, nothing is saved. A
search of the neural settings keeps its best trial the same way (its model is already
fitted), or, during its first trial, what that trial has learned. A stop during phase 2
ends the run like a classical run (a search of the classical model keeps its best trial).
"""

from __future__ import annotations

import dataclasses
import os
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ..graph import analyze
from ..hybrid import HybridParts, head_graph, split_graph
from ..layers import ClassicalModel
from .classical_engine import _ClassicalRun, _Data, models_of
from .classical_search import ClassicalSearch, _TrialEvaluation
from .classical_tuning import (
    NEURAL_SEARCH_FROZEN,
    NEURAL_SEARCH_NO_VALIDATION,
    trial_graph,
)
from .compat import TrainingPlan, check_compatibility
from .data import iterate_batches
from .engine import _free_memory, _is_batch_norm_error, _is_out_of_memory, _Run, build_model
from .estimate import hybrid_fit_seconds
from .job import TrainingJob, TrainingResult
from .objectives import DEFAULT_METRIC, better, label, metric_error
from .objectives import format_value as format_metric
from .reporting import Reporter, StopTraining, TrainingError, short_error
from .state import WEIGHTS_FILE, finite, preprocessing_path, write_json_atomic
from .tuning import TuningConfig, describe_params, split_params, trial_config

# Rows per batch when the features are computed.
EXTRACT_BATCH = 256
# After a stop during phase 1 the classical model is only fitted if this is estimated to
# take at most this much CPU time (the feature pass included).
STOP_FIT_CPU_SECONDS = 10.0
STOPPED_FEATURES = (
    "Stopped before the neural layers had learned the features: nothing was saved (the "
    "classical model was not fitted yet)."
)


def train(
    job: TrainingJob,
    reporter: Reporter,
    weights_path: str | Path | None = None,
    *,
    hard_stop: bool = False,
    features_path: str | Path | None = None,
) -> TrainingResult:
    """Train the hybrid network of ``job``: the classical model is saved to
    ``weights_path`` (joblib), the neural layers to ``features_path`` (default:
    ``weights.pt`` next to it). Never raises (except ``KeyboardInterrupt``); the outcome is
    reported with ``reporter.finish`` and returned. ``hard_stop``: see
    classical_engine.train."""
    if job.tuning is not None and job.tuning.searches_neural:
        return HybridSearch(job, reporter, weights_path, hard_stop, features_path).run()
    return HybridRun(job, reporter, weights_path, hard_stop, features_path).run()


class _HeadModel(nn.Module):
    """The neural layers with the temporary head of phase 1."""

    def __init__(self, features: nn.Module, head: nn.Module):
        super().__init__()
        self.features = features
        self.head = head

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(*inputs))


class _FeatureRun(_Run):
    """Phase 1: the PyTorch engine on the neural layers with their head (``job.graph`` is
    the head graph, which the data is checked and prepared for)."""

    def __init__(self, job: TrainingJob, reporter: Reporter, parts: HybridParts, outputs: int):
        super().__init__(job, reporter, None)
        self.parts = parts
        self.outputs = outputs
        self.trial_label = ""  # a search: "Trial 2 of 10: "

    def _build(self, config: Any) -> None:
        super()._build(config)
        torch.manual_seed(self.seed)  # the seed fixes the initial weights (frozen features too)
        self.model = _HeadModel(
            build_model(self.parts.feature_graph), nn.Linear(self.parts.features, self.outputs)
        )

    def _setup_info(self) -> dict[str, Any]:
        info = super()._setup_info()
        assert isinstance(self.model, _HeadModel)
        # The parameters of the neural layers (the head is dropped).
        info["total_params"] = sum(
            p.numel() for p in self.model.features.parameters() if p.requires_grad
        )
        return info

    def _epoch_message(self, epoch: int) -> str:
        if self.trial_label:
            return f"{self.trial_label}epoch {epoch} of {self.config.epochs}"
        return super()._epoch_message(epoch)

    def learn(self, frozen: bool) -> None:
        self._prepare()
        if frozen:
            self.reporter.setup({"epochs": 0})
            self.reporter.log(
                "Frozen features: the neural layers keep their random initial weights."
            )
            return
        self._fit()
        self._keep_best()
        self._report_kept()

    @property
    def extractor(self) -> nn.Module:
        assert isinstance(self.model, _HeadModel)
        return self.model.features


class HybridRun:
    def __init__(
        self,
        job: TrainingJob,
        reporter: Reporter,
        weights_path: str | Path | None,
        hard_stop: bool,
        features_path: str | Path | None = None,
    ):
        self.job = job
        self.reporter = reporter
        self.weights_path = Path(weights_path) if weights_path is not None else None
        if features_path is None and self.weights_path is not None:
            features_path = self.weights_path.with_name(WEIGHTS_FILE)
        self.features_path = Path(features_path) if features_path is not None else None
        self.hard_stop = hard_stop
        self.history: list[dict[str, Any]] = []
        self.best: dict[str, Any] | None = None
        self.weights_epoch: int | None = None
        self.stopped_early = False
        self.steps_per_epoch: int | None = None
        self.feature_run: _FeatureRun | None = None
        self.finished = False
        self.seed = 0
        # A search of the neural settings (HybridSearch): its trials.
        self.trials: list[dict[str, Any]] = []
        self.best_trial: dict[str, Any] | None = None
        self.subsampled_logged = False

    # -- outcome ---------------------------------------------------------- #

    def run(self) -> TrainingResult:
        try:
            self._learn_features()
        except StopTraining:
            return self._stopped_learning()
        except TrainingError as exc:
            return self._end("failed", str(exc))
        except Exception as exc:
            traceback.print_exc()
            return self._end("failed", self._explain(exc))
        return self._fit_model()

    def _explain(self, exc: Exception) -> str:
        run = self.feature_run
        return run._explain(exc) if run else f"Training failed: {short_error(exc)}"

    def _end(self, status: str, message: str) -> TrainingResult:
        if not self.finished:
            self.finished = True
            self.reporter.finish(status, message, False)
        return self._with_trials(TrainingResult(status, message, False, self.history, self.best))

    def _with_trials(self, result: TrainingResult) -> TrainingResult:
        """A search of the neural settings adds its trials to the result."""
        if not self.trials:
            return result
        best = self.best_trial["trial"] if self.best_trial else None
        return dataclasses.replace(result, trials=list(self.trials), best_trial=best)

    # -- phase 1 ----------------------------------------------------------- #

    def _setup(self) -> None:
        """Check the network against the data and split it: sets the plan, the parts, the
        classical model and the run of phase 1 (not started yet)."""
        job, reporter = self.job, self.reporter
        reporter.status("preparing", "Preparing the data …")
        analysis = analyze(job.graph)
        compat = check_compatibility(analysis, job.dataset, job.config)
        plan = compat.plan
        if not compat.ok or plan is None or plan.hybrid is None or plan.classical is None:
            raise TrainingError(
                compat.errors[0] if compat.errors else "The network cannot be trained."
            )
        self.plan: TrainingPlan = plan
        self.analysis = analysis
        self.parts = split_graph(job.graph, analysis)
        model = analysis.model
        assert model is not None and isinstance(model.layer, ClassicalModel)
        self.layer: ClassicalModel = model.layer
        self.classification = self.layer.task == "classification"
        outputs = int(plan.hybrid["head_outputs"])
        head_job = dataclasses.replace(job, graph=head_graph(self.parts, outputs), tuning=None)
        self.feature_run = _FeatureRun(head_job, reporter, self.parts, outputs)

    def _learn_features(self) -> None:
        self._setup()
        run = self.feature_run
        assert run is not None
        frozen = self.job.config.feature_training == "frozen"
        if not frozen:
            self.reporter.log(
                f"Phase 1 of 2: the neural layers learn {self.parts.features} features with a "
                "temporary linear head on the target."
            )
        run.learn(frozen)
        self._take_phase_one()
        self.reporter.status("running", "Computing the learned features …")
        self.data = self._features()
        self._save_features()

    def _take_phase_one(self) -> None:
        """What phase 1 learned becomes the run's: its seed, curves and kept epoch."""
        run = self.feature_run
        assert run is not None
        self.seed = run.seed
        self.history = list(run.history)
        self.best = dict(run.best) if run.best else None
        self.weights_epoch = run.weights_epoch
        self.stopped_early = run.stopped_early
        self.steps_per_epoch = run.steps_per_epoch

    def _features(self, stoppable: bool = True) -> _Data:
        """The features (and targets) of the training, validation and test rows, as the
        classical engine's data; the training rows are subsampled to the model's row
        limit. ``stoppable``: check the stop request between the batches."""
        run = self.feature_run
        assert run is not None
        extractor = run.extractor
        extractor.eval()
        preprocessing = run.data.preprocessing
        t_mean, t_std = preprocessing.get("target_mean"), preprocessing.get("target_std")
        parts: list[tuple[np.ndarray, np.ndarray]] = []
        sources = [run.data.train, run.data.val]
        if run.data.test is not None:
            sources.append(run.data.test)
        for source in sources:
            xs, ys = [], []
            with torch.no_grad():
                for inputs, targets in iterate_batches(source, EXTRACT_BATCH):
                    if stoppable:
                        run._check_stop()
                    out = extractor(*[x.to(run.device) for x in inputs])
                    xs.append(out.detach().cpu().double().reshape(len(out), -1).numpy())
                    ys.append(targets[0].detach().cpu().numpy())
            width = self.parts.features
            x = np.concatenate(xs) if xs else np.empty((0, width))
            y = np.concatenate(ys) if ys else np.empty(0)
            if self.classification:
                y = y.astype(np.int64).reshape(-1)
            else:
                y = y.astype(np.float64).reshape(-1)
                if t_std is not None and t_mean is not None:
                    y = y * float(t_std) + float(t_mean)  # back to the units of the target
            parts.append((x, y))
        sizes = [len(x) for x, _ in parts]
        x = np.concatenate([p[0] for p in parts])
        y = np.concatenate([p[1] for p in parts])
        starts = np.cumsum([0, *sizes])
        train = np.arange(starts[0], starts[1])
        val = np.arange(starts[1], starts[2])
        test = np.arange(starts[2], starts[3]) if len(parts) > 2 else None
        if not np.isfinite(x).all():
            raise TrainingError(
                "The neural layers put out features that are not finite numbers (NaN or "
                "infinity). Try a lower learning rate."
            )
        limit = int(self.plan.classical["max_rows"])  # type: ignore[index]
        subsampled_from = None
        if len(train) > limit:
            subsampled_from = len(train)
            pick = np.random.default_rng([self.seed, 3]).choice(len(train), limit, replace=False)
            train = np.sort(train[pick])
            if not self.subsampled_logged:
                self.subsampled_logged = True
                self.reporter.log(
                    f"{self.layer.label} is fitted on at most {limit:,} rows on this server: "
                    f"on the features of {limit:,} random rows of the {subsampled_from:,} "
                    "training rows (the neural layers learned from all of them)."
                )
        target = str(preprocessing.get("target") or "target")
        columns = [*(f"feature_{i + 1}" for i in range(self.parts.features)), target]
        delimiter = str(preprocessing.get("delimiter") or ",")
        return _Data(x, y, train, val, test, columns, delimiter, subsampled_from, None)

    def preprocessing(self) -> dict[str, Any]:
        """How the inputs are prepared (that of the neural layers), with what the run adds:
        the classical model predicts in the units of the target."""
        run = self.feature_run
        assert run is not None
        preprocessing = dict(run.data.preprocessing)
        if preprocessing.get("kind") == "demo":
            return preprocessing
        preprocessing.update(
            target_mean=None,
            target_std=None,
            estimator=self.layer.estimator,
            hybrid={"boundary": self.parts.boundary, "features": self.parts.features},
        )
        return preprocessing

    def _save_features(self) -> None:
        """``weights.pt``: the weights of the neural layers with the preprocessing (the
        format of the generated script's ``--weights``)."""
        if self.features_path is None:
            return
        run = self.feature_run
        assert run is not None
        preprocessing = finite(self.preprocessing())
        state = {k: v.detach().to("cpu", copy=True) for k, v in run.extractor.state_dict().items()}
        tmp = self.features_path.with_name(f".{self.features_path.name}.tmp")
        torch.save({"state_dict": state, "preprocessing": preprocessing}, tmp)
        os.replace(tmp, self.features_path)
        if preprocessing.get("kind") != "demo":
            write_json_atomic(preprocessing_path(self.features_path), preprocessing)

    # -- a stop during phase 1 -------------------------------------------- #

    def _stopped_learning(self) -> TrainingResult:
        """Stopped while the neural layers learned (or computed) the features: keep what
        they learned so far and fit the classical model on it (see the module)."""
        run = self.feature_run
        frozen = self.job.config.feature_training == "frozen"
        if run is None or run.model is None or not (run.history or frozen):
            return self._end("stopped", STOPPED_FEATURES)
        run._on_stopped()  # the best epoch so far (restore_best)
        run._report_kept()
        self._take_phase_one()
        if frozen:
            text = "Stopped while the frozen random features were computed."
        else:
            text = run._stopped_message()
        if self.job.tuning is not None:
            text += " The search of the classical model had not started."
        return self._fit_after_stop(text)

    def _fit_after_stop(
        self,
        text: str,
        params: dict[str, Any] | None = None,
        fitted: tuple[Any, float] | None = None,
    ) -> TrainingResult:
        """Save the neural layers with the classical model fitted on their features (with
        the brick's parameters, or ``params``) after a stop. ``fitted``: the model is
        already fitted on them (with the seconds of that fit). ``text``: what was stopped,
        the start of the final message."""
        estimator = self.layer.estimator
        if fitted is None:
            seconds = hybrid_fit_seconds(
                self.analysis, self.job.dataset, self.job.config, self.plan.to_dict()
            )
            if seconds is None or seconds > STOP_FIT_CPU_SECONDS:
                about = f" (about {seconds:.0f} s of CPU time)" if seconds is not None else ""
                return self._end(
                    "stopped",
                    f"{text} Fitting {estimator} on the features learned so far would take "
                    f"too long after a stop{about}: nothing was saved.",
                )
            self.reporter.status(
                "running", f"Stopped: fitting {estimator} on the features learned so far …"
            )
            try:
                self.data = self._features(stoppable=False)
            except Exception as exc:
                traceback.print_exc()
                return self._end(
                    "stopped",
                    f"{text} The features could not be computed ({short_error(exc)}): nothing "
                    "was saved.",
                )
        fit = _StoppedFit(self.job, _AfterStop(self.reporter, self), self.weights_path, False)
        fit.hybrid = self
        fit.params = params
        fit.fitted = fitted
        fit.text = text
        result = fit.run()
        self.finished = True
        return self._with_trials(
            dataclasses.replace(
                result,
                history=self.history,
                best=self.best,
                weights_epoch=self.weights_epoch if result.weights else None,
                stopped_early=self.stopped_early,
            )
        )

    # -- phase 2 ----------------------------------------------------------- #

    def _fit_model(self) -> TrainingResult:
        reporter = _PhaseTwo(self.reporter, self)
        self.reporter.log(
            f"Phase 2 of 2: fitting {self.layer.estimator} on the {self.parts.features} "
            "learned features."
        )
        fit: _ClassicalRun
        if self.best_trial is not None:
            fit = _SearchedFit(self.job, reporter, self.weights_path, self.hard_stop)
        elif self.job.tuning is not None:
            fit = _ClassicalSearchFit(self.job, reporter, self.weights_path, self.hard_stop)
        else:
            fit = _ClassicalFit(self.job, reporter, self.weights_path, self.hard_stop)
        fit.hybrid = self  # type: ignore[attr-defined]
        result = fit.run()
        self.finished = True
        return self._with_trials(
            dataclasses.replace(
                result,
                message=reporter.message or result.message,
                history=self.history,
                best=self.best,
                weights_epoch=self.weights_epoch,
                stopped_early=self.stopped_early,
            )
        )

    def features_text(self) -> str:
        """What phase 1 did, for the final message."""
        k = self.parts.features
        if self.job.config.feature_training == "frozen":
            return f"{k} frozen random features"
        epochs = len(self.history)
        text = f"{k} features learned in {epochs} epoch{'s' if epochs != 1 else ''}"
        if self.weights_epoch is not None and self.weights_epoch != epochs:
            text += f" (kept epoch {self.weights_epoch})"
        return text

    def completed_message(self, message: str) -> str:
        """The final message of a completed run from that of phase 2."""
        rest = message.removeprefix("Finished: ")
        return (
            f"Finished: {self.features_text()}; {self.layer.estimator} on them: "
            f"{rest[:1].lower()}{rest[1:]}"
        )


class HybridSearch(HybridRun):
    """A search over the training settings of the neural layers (and the parameters of the
    classical model): every trial learns the features and fits the model on them, ranked by
    that model on the validation rows (see the module)."""

    def __init__(
        self,
        job: TrainingJob,
        reporter: Reporter,
        weights_path: str | Path | None,
        hard_stop: bool,
        features_path: str | Path | None = None,
    ):
        super().__init__(job, reporter, weights_path, hard_stop, features_path)
        assert job.tuning is not None
        self.tuning: TuningConfig = job.tuning
        self.total = 0
        self.device_checked = False
        # The best trial so far: the weights of its neural layers (on the CPU), what phase 1
        # learned, its features and its fitted model.
        self.best_state: dict[str, torch.Tensor] | None = None
        self.best_phase: dict[str, Any] = {}
        self.best_data: _Data | None = None
        self.best_pipeline: Any = None
        self.best_fit_seconds = 0.0
        self.fitter: _TrialFitter | None = None

    def run(self) -> TrainingResult:
        try:
            self._search()
        except StopTraining:
            result = self._stopped_search()
            if self.fitter is not None and self.fitter.abandoned:
                os._exit(0)  # the left-behind fit cannot be interrupted
            return result
        except TrainingError as exc:
            return self._end("failed", str(exc))
        except Exception as exc:
            traceback.print_exc()
            return self._end("failed", self._explain(exc))
        self._use_best()
        self._save_features()
        return self._fit_model()

    # -- the trials ---------------------------------------------------------- #

    def _search(self) -> None:
        self._setup()
        run = self.feature_run
        assert run is not None
        if self.job.config.feature_training == "frozen":
            raise TrainingError(NEURAL_SEARCH_FROZEN)
        error = metric_error(self.tuning.metric, self.plan)
        if error is not None:
            raise TrainingError(error)
        run._prepare_data()
        if not run.data.has_val:
            raise TrainingError(NEURAL_SEARCH_NO_VALIDATION)
        self.seed = run.seed
        params = self.tuning.sample(self.seed)
        self.total = len(params)
        self.fitter = _TrialFitter(self.job, self.reporter, None, self.hard_stop)
        self.fitter.hybrid = self
        self.fitter._check()
        metric = "the loss" if self.tuning.metric == DEFAULT_METRIC else label(self.tuning.metric)
        self.reporter.log(
            f"Hyperparameter search: {self.tuning.describe()}, every trial learns "
            f"{self.parts.features} features and fits {self.layer.estimator} on them, ranked "
            f"by {metric} on the {len(run.data.val):,} validation rows."
        )
        for index, values in enumerate(params, start=1):
            if self.reporter.should_stop():
                raise StopTraining
            self._run_trial(index, values)
        if self.best_trial is None:
            first = next(t for t in self.trials if t["status"] == "failed")
            raise TrainingError(
                f"All {self.total} trials failed. Trial {first['trial']}: {first['message']}"
            )

    def _run_trial(self, index: int, params: dict[str, Any]) -> None:
        run, fitter, reporter = self.feature_run, self.fitter, self.reporter
        assert run is not None and fitter is not None
        neural, classical = split_params(params)
        run.trial_label = f"Trial {index} of {self.total}: "
        run._build(trial_config(self.job.config, neural))
        if index == 1:
            info = run._setup_info()
            info.update(tuning=self.tuning.to_dict(), trials_total=self.total)
            reporter.setup(info)
        reporter.trial_start(
            {
                "trial": index,
                "trials": self.total,
                "params": dict(params),
                "steps_per_epoch": run.steps_per_epoch,
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
            "score": None,
            "folds": None,
            "weights_epoch": None,
            "stopped_early": False,
            "seconds": None,
        }
        self.trials.append(entry)
        started = time.perf_counter()
        failure: str | None = None
        try:
            failure = self._learn_trial()
            if failure is None:
                data = self.data = self._features()
                fitter.data = data
                if not self.classification:
                    spread = float(np.var(data.y[data.train]))
                    fitter.target_variance = spread if spread > 1e-12 else 1.0
                failure, pipeline, fit_seconds, metrics, score = self._fit_trial(index, classical)
        except StopTraining:
            entry["epochs_done"] = len(run.history)
            self._trial_done(entry, "stopped", "Stopped.", started)
            raise
        except TrainingError as exc:  # e.g. features that are not finite
            failure = str(exc)
        entry["epochs_done"] = len(run.history)
        if failure is not None:
            reporter.log(f"Trial {index} failed: {failure}")
            run.model = None
            run.optimizer = None  # type: ignore[assignment]
            _free_memory(run.device)
            self._trial_done(entry, "failed", failure, started)
            return
        entry.update(
            val_loss=metrics["loss"],
            val_accuracy=metrics["accuracy"],
            val_rmse=metrics["rmse"],
            val_r2=metrics["r2"],
            score=score,
            weights_epoch=run.weights_epoch,
            stopped_early=run.stopped_early,
        )
        best = self.best_trial
        new_best = best is None or better(score, best["score"], self.tuning.metric)
        if new_best:
            self.best_state = {
                k: v.detach().to("cpu", copy=True) for k, v in run.extractor.state_dict().items()
            }
            self.best_phase = {
                "config": run.config,
                "history": list(run.history),
                "best": dict(run.best) if run.best else None,
                "weights_epoch": run.weights_epoch,
                "stopped_early": run.stopped_early,
                "steps_per_epoch": run.steps_per_epoch,
            }
            self.best_data = self.data
            self.best_pipeline = pipeline
            self.best_fit_seconds = fit_seconds
        self._trial_done(entry, "completed", "", started, new_best)

    def _learn_trial(self) -> str | None:
        """Phase 1 of a trial; returns why it failed (None: it did not)."""
        run = self.feature_run
        assert run is not None
        try:
            run._check_memory(run.config)
            run._to_device(check=not self.device_checked)
            self.device_checked = True
            run.optimizer = run._optimizer()
            run._fit()
            run._keep_best()
        except StopTraining:
            raise
        except Exception as exc:
            if isinstance(exc, TrainingError):
                return str(exc)
            if _is_out_of_memory(exc) or _is_batch_norm_error(exc):
                return run._explain(exc)
            raise
        return None

    def _fit_trial(self, index: int, params: dict[str, Any]) -> tuple[Any, ...]:
        """Phase 2 of a trial: ``(failure, pipeline, fit seconds, validation metrics,
        score)``; a failure (None: none) leaves the rest None."""
        fitter = self.fitter
        assert fitter is not None
        data = fitter.data
        self.reporter.status(
            "running", f"Trial {index} of {self.total}: fitting {self.layer.estimator} …"
        )
        fitter.best_trial = self.best_trial  # what a hard stop keeps (classical_search.py)
        try:
            pipeline = fitter._trial_pipeline(params)
            started = time.perf_counter()
            fitter._fit_with(pipeline, data.x[data.train], data.y[data.train])
            seconds = time.perf_counter() - started
            fitter.pipeline = pipeline
            predictions = fitter._predictions(data.val)
            metrics = fitter._metrics(data.val, predictions)
            score = fitter._score(data.val, predictions, metrics)
        except StopTraining:
            raise
        except TrainingError as exc:
            return str(exc), None, 0.0, None, None
        except MemoryError:
            return f"Out of memory while fitting {len(data.train):,} rows.", None, 0.0, None, None
        except Exception as exc:
            traceback.print_exc()
            return f"Fitting failed: {short_error(exc)}", None, 0.0, None, None
        return None, pipeline, seconds, metrics, score

    def _trial_done(
        self,
        entry: dict[str, Any],
        status: str,
        message: str,
        started: float,
        new_best: bool = False,
    ) -> None:
        entry.update(
            status=status, message=message, seconds=round(time.perf_counter() - started, 3)
        )
        if new_best:
            self.best_trial = entry
        best = self.best_trial["trial"] if self.best_trial else None
        self.reporter.trial_end(dict(entry), best)

    # -- the outcome -------------------------------------------------------- #

    def _use_best(self) -> None:
        """Make the best trial the run's: the weights of its neural layers, what phase 1
        learned and its features."""
        run, best = self.feature_run, self.best_trial
        assert run is not None and best is not None and self.best_state is not None
        phase = self.best_phase
        run.trial_label = ""
        run._build(phase["config"])
        run.extractor.load_state_dict(self.best_state)
        run.model.to(run.device)  # type: ignore[union-attr]
        self.history = list(phase["history"])
        self.best = dict(phase["best"]) if phase["best"] else None
        self.weights_epoch = phase["weights_epoch"]
        self.stopped_early = phase["stopped_early"]
        self.steps_per_epoch = phase["steps_per_epoch"]
        assert self.best_data is not None
        self.data = self.best_data
        self.reporter.search_end(
            {
                "best_trial": best["trial"],
                "params": dict(best["params"]),
                "history": list(self.history),
                "best": dict(self.best) if self.best else None,
            }
        )
        self.reporter.weights_kept(
            {"weights_epoch": self.weights_epoch, "stopped_early": self.stopped_early}
        )

    def _best_text(self) -> str:
        """``"Best: trial 3 (neural layers: learning rate 0.003; C 3.2), validation accuracy
        0.9"``."""
        best = self.best_trial
        assert best is not None
        metric = self.tuning.metric
        name = "loss" if metric == DEFAULT_METRIC else label(metric)
        what = describe_params(best["params"], self.tuning.model)
        return (
            f"Best: trial {best['trial']} ({what}), validation {name} "
            f"{format_metric(metric, best['score'])}"
        )

    def completed_message(self, message: str) -> str:
        failed = sum(1 for t in self.trials if t["status"] == "failed")
        head = f"Finished {self.total} trial{'s' if self.total != 1 else ''}"
        if failed:
            head += f" ({failed} failed)"
        rest = super().completed_message(message).removeprefix("Finished: ")
        return f"{head}. {self._best_text()}. {rest[:1].upper()}{rest[1:]}"

    def _stopped_search(self) -> TrainingResult:
        """Keep the best trial so far (its model is fitted already), or during the first
        trial what it has learned (see the module)."""
        done = sum(1 for t in self.trials if t["status"] in ("completed", "failed"))
        if self.best_trial is not None:
            self._use_best()
            text = f"Stopped after {done} of {self.total} trials. {self._best_text()}."
            _, classical = split_params(self.best_trial["params"])
            return self._fit_after_stop(
                text, classical, (self.best_pipeline, self.best_fit_seconds)
            )
        run = self.feature_run
        if not self.trials or run is None or run.model is None or not run.history:
            return self._end("stopped", STOPPED_FEATURES)
        run.trial_label = ""
        run._on_stopped()
        run._report_kept()
        self._take_phase_one()
        trial = self.trials[-1]
        _, classical = split_params(trial["params"])
        what = describe_params(trial["params"], self.tuning.model)
        text = (
            f"Stopped during trial {trial['trial']} of {self.total} ({what}): "
            f"{run._stopped_message()}"
        )
        return self._fit_after_stop(text, classical)


class _PhaseTwo(Reporter):
    """The reporter of phase 2: passes everything on, except what would replace the curves
    of phase 1 (the classical fit's single history entry and kept epoch), and adds what
    phase 1 did to the final message."""

    def __init__(self, reporter: Reporter, run: HybridRun):
        self._reporter = reporter
        self._run = run
        self._classical: dict[str, Any] | None = None
        self.message: str | None = None  # the final message

    def __getattr__(self, name: str) -> Any:
        return getattr(self._reporter, name)

    def status(self, status: str, message: str) -> None:
        self._reporter.status(status, message)

    def log(self, line: str) -> None:
        self._reporter.log(line)

    def progress(self, progress: dict[str, Any]) -> None:
        self._reporter.progress(progress)

    def setup(self, info: dict[str, Any]) -> None:
        kept = {k: v for k, v in info.items() if k in ("classical", "tuning", "trials_total")}
        if "classical" in kept:
            self._classical = {**kept["classical"], "hybrid": self._hybrid()}
            kept["classical"] = self._classical
        if "preprocessing" in info:
            kept["preprocessing"] = self._run.preprocessing()
        if kept:
            self._reporter.setup(kept)

    def _hybrid(self) -> dict[str, Any]:
        run = self._run
        return {
            "boundary": run.parts.boundary,
            "features": run.parts.features,
            "feature_training": run.job.config.feature_training,
            "epochs": len(run.history),
        }

    def _restore(self) -> None:
        run = self._run
        self._reporter.setup(
            {"history": run.history, "best": run.best, "steps_per_epoch": run.steps_per_epoch}
        )

    def epoch_end(self, entry: dict[str, Any], best: dict[str, Any] | None) -> None:
        """The fitted model on the training and validation rows: ``classical.validation``."""
        validation = {
            "loss": entry.get("val_loss"),
            "accuracy": entry.get("val_accuracy"),
            "rmse": entry.get("val_rmse"),
            "r2": entry.get("val_r2"),
        }
        self._classical = {
            **(self._classical or {}),
            "validation": validation if entry.get("val_loss") is not None else None,
        }
        self._reporter.setup({"classical": self._classical})

    def trial_start(self, trial: dict[str, Any]) -> None:
        self._reporter.trial_start(trial)
        self._restore()

    def trial_end(self, trial: dict[str, Any], best_trial: int | None) -> None:
        self._reporter.trial_end(trial, best_trial)

    def search_end(self, result: dict[str, Any]) -> None:
        run = self._run
        self._reporter.search_end({**result, "history": run.history, "best": run.best})
        if result.get("history"):
            self.epoch_end(result["history"][0], None)

    def weights_kept(self, result: dict[str, Any]) -> None:
        pass  # phase 1's kept epoch stays

    def test_end(self, result: dict[str, Any]) -> None:
        self._reporter.test_end(result)

    def evaluation_end(self, result: dict[str, Any]) -> None:
        self._reporter.evaluation_end({**result, "epoch": self._run.weights_epoch})

    def finish(self, status: str, message: str, weights: bool) -> None:
        run = self._run
        if status == "completed":
            message = run.completed_message(message)
        elif status == "stopped" and not weights:
            message = f"{message} The neural layers had learned {run.features_text()}."
        run.finished = True
        self.message = message
        self._reporter.finish(status, message, weights)

    def should_stop(self) -> bool:
        return self._reporter.should_stop()


class _AfterStop(_PhaseTwo):
    """The reporter of the fit after a stop: that fit cannot be stopped (the stop request
    is still there), and its final message is passed on as it is."""

    def should_stop(self) -> bool:
        return False

    def finish(self, status: str, message: str, weights: bool) -> None:
        self._run.finished = True
        self.message = message
        self._reporter.finish(status, message, weights)


class _HybridFit:
    """Phase 2 as a classical run: its plan, model and data come from phase 1."""

    hybrid: HybridRun

    @property
    def model_graph(self) -> dict[str, Any]:
        return self.hybrid.parts.classical_graph

    def _check(self) -> Any:
        run = self.hybrid
        self.plan = run.plan
        self.classical = run.plan.classical
        self.layer = run.layer
        self.models = models_of(run.layer)
        self.classification = run.classification
        self.num_classes = run.plan.num_classes or 0
        self.seed = run.seed
        return run.analysis

    def _load(self) -> _Data:
        return self.hybrid.data

    def _preprocessing(self) -> dict[str, Any]:
        return self.hybrid.preprocessing()

    def _save(self) -> bool:
        """``model.joblib``: the fitted pipeline on the features (the neural layers were
        saved after phase 1)."""
        weights_path: Path | None = self.weights_path  # type: ignore[attr-defined]
        if weights_path is None:
            return False
        import joblib

        from ..classical import MODEL_FORMAT

        run = self.hybrid
        data = run.data
        pipeline = self.pipeline  # type: ignore[attr-defined]
        preprocessing = run.preprocessing()
        bundle: dict[str, Any] = {
            "format": MODEL_FORMAT,
            "model": pipeline,
            "task": run.layer.task,
            "columns": data.columns[:-1],
            "target": data.columns[-1],
            "class_names": None,
            "hybrid": {
                "features": run.parts.features,
                "input_columns": preprocessing.get("columns"),
                "boundary": run.parts.boundary,
            },
        }
        if run.classification:
            indices = [int(c) for c in pipeline.classes_]
            names = run.plan.class_names or []
            bundle["class_names"] = [names[i] for i in indices]
            bundle["class_indices"] = indices
            bundle["num_classes"] = self.num_classes
        tmp = weights_path.with_name(f".{weights_path.name}.tmp")
        joblib.dump(bundle, tmp)
        os.replace(tmp, weights_path)
        return True


class _ClassicalFit(_HybridFit, _ClassicalRun):
    pass


class _ClassicalSearchFit(_HybridFit, ClassicalSearch):
    pass


class _TrialFitter(_HybridFit, ClassicalSearch):
    """The classical part of the trials of a :class:`HybridSearch`: fits, predicts and
    scores like a classical search (its ``data`` are the features of the running trial)."""


def _brick_graph(run: HybridRun, params: dict[str, Any] | None) -> dict[str, Any]:
    """The classical part with the values of a trial on the model brick."""
    graph = run.parts.classical_graph
    if not params:
        return graph
    return trial_graph(graph, run.plan.classical["node"], params)  # type: ignore[index]


class _SearchedFit(_HybridFit, _ClassicalRun):
    """Phase 2 after a search of the neural settings: the best trial's model, fitted on its
    features already, is measured, tested, evaluated and saved."""

    hybrid: HybridSearch

    @property
    def model_graph(self) -> dict[str, Any]:
        assert self.hybrid.best_trial is not None
        return _brick_graph(self.hybrid, split_params(self.hybrid.best_trial["params"])[1])

    def _fit_model(self, x: np.ndarray, y: np.ndarray) -> float:
        self.pipeline = self.hybrid.best_pipeline
        return self.hybrid.best_fit_seconds

    def _report_evaluation(self, *args: Any) -> None:
        reporter = self.reporter
        number = self.hybrid.best_trial["trial"] if self.hybrid.best_trial else None
        self.reporter = _TrialEvaluation(reporter, number)  # type: ignore[assignment]
        try:
            super()._report_evaluation(*args)
        finally:
            self.reporter = reporter


class _StoppedFit(_HybridFit, _ClassicalRun):
    """The classical model on the features learned before a stop: fitted (unless
    ``fitted``), measured on the training and validation rows and saved with the neural
    layers, without a test or an evaluation. The run ends as stopped either way."""

    params: dict[str, Any] | None = None
    fitted: tuple[Any, float] | None = None
    text = ""

    @property
    def model_graph(self) -> dict[str, Any]:
        return _brick_graph(self.hybrid, self.params)

    def _load(self) -> _Data:
        return dataclasses.replace(self.hybrid.data, test=None)  # not tested

    def _fit_model(self, x: np.ndarray, y: np.ndarray) -> float:
        if self.fitted is not None:
            self.pipeline, seconds = self.fitted
            return seconds
        return super()._fit_model(x, y)

    def _report_evaluation(self, *args: Any) -> None:
        pass  # stopped runs get no final evaluation

    def _save(self) -> bool:
        self.hybrid._save_features()
        return super()._save()

    def _end(self, status: str, message: str) -> TrainingResult:
        weights = self.saved and status == "completed"
        estimator = self.hybrid.layer.estimator
        if status == "completed":
            rest = message.removeprefix("Finished: ")
            text = (
                f"{self.text} Kept the {self.hybrid.parts.features} features learned so far "
                f"with {estimator} fitted on them (not tested): {rest[:1].lower()}{rest[1:]}"
            )
        else:
            text = f"{self.text} {estimator} could not be fitted on the features: {message}"
            text += "" if text.endswith(".") else "."
        if not weights:
            text += " Nothing was saved."
        self.reporter.finish("stopped", text, weights)
        return TrainingResult(
            "stopped", text, weights, self.history, self.best, weights_epoch=1 if weights else None
        )


def load_extractor(run_dir: Path, graph: dict[str, Any]) -> tuple[nn.Module, dict[str, Any]]:
    """The trained neural layers of a hybrid run (``weights.pt``) and their preprocessing.
    Raises ``FileNotFoundError`` and the errors of building or loading them."""
    path = Path(run_dir) / WEIGHTS_FILE
    if not path.is_file():
        raise FileNotFoundError(path)
    parts = split_graph(graph)
    extractor = build_model(parts.feature_graph)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    extractor.load_state_dict(checkpoint["state_dict"])
    extractor.eval()
    return extractor, dict(checkpoint.get("preprocessing") or {})


def features_of(extractor: nn.Module, inputs: list[torch.Tensor]) -> np.ndarray:
    """The learned features of a batch (float64, one row per sample)."""
    with torch.no_grad():
        parts = []
        n = len(inputs[0])
        for start in range(0, n, EXTRACT_BATCH):
            out = extractor(*[x[start : start + EXTRACT_BATCH] for x in inputs])
            parts.append(out.detach().cpu().double().reshape(len(out), -1).numpy())
    return np.concatenate(parts) if parts else np.empty((0, 0))
