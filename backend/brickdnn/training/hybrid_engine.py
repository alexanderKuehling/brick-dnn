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

Stopping during phase 1 ends the run without saving anything. A stop during phase 2 ends it
like a classical run (a search keeps its best trial).
"""

from __future__ import annotations

import dataclasses
import os
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
from .classical_search import ClassicalSearch
from .compat import TrainingPlan, check_compatibility
from .data import iterate_batches
from .engine import _Run, build_model
from .job import TrainingJob, TrainingResult
from .reporting import Reporter, StopTraining, TrainingError
from .state import WEIGHTS_FILE, finite, preprocessing_path, write_json_atomic

# Rows per batch when the features are computed.
EXTRACT_BATCH = 256
STOPPED_FEATURES = (
    "Stopped while the neural layers learned the features: nothing was saved (the classical "
    "model was not fitted yet)."
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

    # -- outcome ---------------------------------------------------------- #

    def run(self) -> TrainingResult:
        try:
            self._learn_features()
        except StopTraining:
            return self._end("stopped", STOPPED_FEATURES)
        except TrainingError as exc:
            return self._end("failed", str(exc))
        except Exception as exc:
            traceback.print_exc()
            run = self.feature_run
            return self._end("failed", run._explain(exc) if run else f"Training failed: {exc}")
        return self._fit_model()

    def _end(self, status: str, message: str) -> TrainingResult:
        if not self.finished:
            self.finished = True
            self.reporter.finish(status, message, False)
        return TrainingResult(status, message, False, self.history, self.best)

    # -- phase 1 ----------------------------------------------------------- #

    def _learn_features(self) -> None:
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
        frozen = job.config.feature_training == "frozen"
        head_job = dataclasses.replace(
            job, graph=head_graph(self.parts, int(plan.hybrid["head_outputs"])), tuning=None
        )
        run = self.feature_run = _FeatureRun(
            head_job, reporter, self.parts, int(plan.hybrid["head_outputs"])
        )
        if not frozen:
            reporter.log(
                f"Phase 1 of 2: the neural layers learn {self.parts.features} features with a "
                "temporary linear head on the target."
            )
        run.learn(frozen)
        self.seed = run.seed
        self.history = list(run.history)
        self.best = dict(run.best) if run.best else None
        self.weights_epoch = run.weights_epoch
        self.stopped_early = run.stopped_early
        self.steps_per_epoch = run.steps_per_epoch
        reporter.status("running", "Computing the learned features …")
        self.data = self._features()
        self._save_features()

    def _features(self) -> _Data:
        """The features (and targets) of the training, validation and test rows, as the
        classical engine's data; the training rows are subsampled to the model's row
        limit."""
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
            self.reporter.log(
                f"{self.layer.label} is fitted on at most {limit:,} rows on this server: on "
                f"the features of {limit:,} random rows of the {subsampled_from:,} training "
                "rows (the neural layers learned from all of them)."
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

    # -- phase 2 ----------------------------------------------------------- #

    def _fit_model(self) -> TrainingResult:
        reporter = _PhaseTwo(self.reporter, self)
        self.reporter.log(
            f"Phase 2 of 2: fitting {self.layer.estimator} on the {self.parts.features} "
            "learned features."
        )
        if self.job.tuning is not None:
            fit: _ClassicalRun = _ClassicalSearchFit(
                self.job, reporter, self.weights_path, self.hard_stop
            )
        else:
            fit = _ClassicalFit(self.job, reporter, self.weights_path, self.hard_stop)
        fit.hybrid = self  # type: ignore[attr-defined]
        result = fit.run()
        self.finished = True
        return dataclasses.replace(
            result,
            message=reporter.message or result.message,
            history=self.history,
            best=self.best,
            weights_epoch=self.weights_epoch,
            stopped_early=self.stopped_early,
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
            rest = message.removeprefix("Finished: ")
            message = (
                f"Finished: {run.features_text()}; {run.layer.estimator} on them: "
                f"{rest[:1].lower()}{rest[1:]}"
            )
        elif status == "stopped" and not weights:
            message = f"{message} The neural layers had learned {run.features_text()}."
        run.finished = True
        self.message = message
        self._reporter.finish(status, message, weights)

    def should_stop(self) -> bool:
        return self._reporter.should_stop()


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
