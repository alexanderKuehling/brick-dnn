"""Hyperparameter search for a classical model: fit the model with other values of its
brick parameters (*trials*), keep the best one and measure it like a normal run. Needs
scikit-learn (see classical_tuning.py for the search space).

The data is loaded and split once, as for a normal fit. Every trial builds the model from
the graph with the trial's values on the model brick and is compared by the objective
metric (see objectives.py):

- on the **validation rows**: the trial's model is fitted on the training rows and
  predicts the validation rows; the best trial's model is kept as it is (it already is the
  model fitted on all training rows);
- by **k-fold cross-validation** on the training rows (``cv_folds``, and always without
  validation rows): the training rows are split into k folds (stratified by class for a
  classifier), the model is fitted k times, each time without one fold, and the metric is
  computed from the predictions of every training row by the model that did not see it.
  The best trial is fitted once more on all training rows; the validation rows, if there
  are some, are then not used by the search at all.

The kept model is measured on the training, validation and test rows, evaluated and saved
like the model of a normal run. A trial that fails (for example an SVC whose calibration
needs more rows of a class than a fold has) is marked as failed and the search goes on.

Stopping keeps the best trial so far: its model is saved (with cross-validation it is
fitted on all training rows first; that last fit cannot be stopped), but not tested or
evaluated, like a stopped neural network. In the runner (``hard_stop``) a fit that is
running when the stop comes is left behind and the process ends once the best trial is
saved; before the first trial has finished, the process ends at once and nothing is saved.
"""

from __future__ import annotations

import os
import threading
import time
import traceback
import warnings
from typing import Any

import numpy as np

from ..graph import analyze
from .classical_engine import (
    MAX_WARNINGS,
    STOP_CHECK_INTERVAL,
    STOPPED,
    _ClassicalRun,
    _seconds,
    build_pipeline,
)
from .classical_tuning import DEFAULT_FOLDS, MIN_ROWS_PER_FOLD, trial_graph
from .job import TrainingResult
from .objectives import DEFAULT_METRIC, better, label, metric_error, split_value
from .objectives import format_value as format_metric
from .reporting import StopTraining, TrainingError, short_error
from .tuning import TuningConfig, describe_params


class _TrialEvaluation:
    """Passes everything to the reporter and adds the best trial to the final evaluation."""

    def __init__(self, reporter: Any, trial: int | None):
        self._reporter = reporter
        self._trial = trial

    def __getattr__(self, name: str) -> Any:
        return getattr(self._reporter, name)

    def evaluation_end(self, result: dict[str, Any]) -> None:
        self._reporter.evaluation_end({**result, "trial": self._trial})


class ClassicalSearch(_ClassicalRun):
    """A classical run whose fit is a search: the trials run in :meth:`_fit_model`, and the
    rest (measuring, evaluating, saving) is the normal run with the best trial's model."""

    def __init__(self, job: Any, reporter: Any, weights_path: Any, hard_stop: bool):
        super().__init__(job, reporter, weights_path, hard_stop)
        assert job.tuning is not None
        self.tuning: TuningConfig = job.tuning
        self.trials: list[dict[str, Any]] = []
        self.best_trial: dict[str, Any] | None = None
        # Without cross-validation: the fitted model of the best trial so far.
        self.best_pipeline: Any = None
        self.total = 0
        self.trial = 0
        self.folds: int | None = None
        self.fold_index: np.ndarray | None = None
        self.abandoned = False  # a fit was left running when the search was stopped
        self.warned: set[str] = set()

    # -- the search --------------------------------------------------------- #

    def _fit_model(self, x: np.ndarray, y: np.ndarray) -> float:
        """Run the trials and make the best one the model of the run; returns the seconds
        of its fit on the training rows."""
        tuning = self.tuning
        if tuning.model != self.layer.type:
            what = tuning.model or "a neural network (training settings)"
            raise TrainingError(
                f"The search settings are for {what}, but the model is {self.layer.type}: "
                "a classical model searches over the parameters of its brick."
            )
        error = metric_error(tuning.metric, self.plan)
        if error is not None:
            raise TrainingError(error)
        data = self.data
        self.folds = tuning.cv_folds or (None if len(data.val) else DEFAULT_FOLDS)
        if self.folds is not None:
            self.fold_index = self._fold_index(self.folds)
        params = tuning.sample(self.seed)
        self.total = len(params)
        info = {**tuning.to_dict(), "cv_folds": self.folds}
        self.reporter.setup({"tuning": info, "trials_total": self.total})
        self.reporter.log(self._plan_text())
        for index, values in enumerate(params, start=1):
            self._check_stop()
            self._run_trial(index, values)
        if self.best_trial is None:
            first = next(t for t in self.trials if t["status"] == "failed")
            raise TrainingError(
                f"All {self.total} trials failed. Trial {first['trial']}: {first['message']}"
            )
        return self._use_best(x, y)

    def _plan_text(self) -> str:
        """``"Hyperparameter search: random search, 20 trials, ranked by ROC AUC on 5-fold
        cross-validation of the 800 training rows."``"""
        text = f"Hyperparameter search: {self.tuning.describe()}"
        metric = "the loss" if self.tuning.metric == DEFAULT_METRIC else label(self.tuning.metric)
        if self.folds is None:
            where = f"the {len(self.data.val):,} validation rows"
        else:
            where = (
                f"{self.folds}-fold cross-validation of the {len(self.data.train):,} training rows"
            )
            if self.tuning.cv_folds is None:
                where += " (there are no validation rows)"
        return f"{text}, ranked by {metric} on {where}."

    def _fold_index(self, folds: int) -> np.ndarray:
        """The fold of every training row: stratified by class for a classifier (every class
        is spread over the folds), otherwise at random; repeatable for the seed."""
        data = self.data
        n = len(data.train)
        if n < folds * MIN_ROWS_PER_FOLD:
            raise TrainingError(
                f"{folds}-fold cross-validation needs at least {folds * MIN_ROWS_PER_FOLD} "
                f"training rows, there are {n}."
            )
        rng = self._rng(3)
        index = np.empty(n, dtype=np.int64)
        if self.classification:
            y = data.y[data.train]
            start = 0
            for value in np.unique(y):
                rows = np.flatnonzero(y == value)
                rows = rows[rng.permutation(len(rows))]
                index[rows] = (np.arange(len(rows)) + start) % folds
                start += len(rows)
        else:
            index[rng.permutation(n)] = np.arange(n) % folds
        return index

    def _run_trial(self, index: int, params: dict[str, Any]) -> None:
        self.trial = index
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
            "folds": self.folds,
            "weights_epoch": None,
            "stopped_early": False,
            "seconds": None,
        }
        self.reporter.trial_start(
            {"trial": index, "trials": self.total, "params": dict(params), "steps_per_epoch": 1}
        )
        self.trials.append(entry)
        started = time.perf_counter()
        failure: str | None = None
        pipeline = None
        try:
            if self.folds is None:
                self.reporter.status("running", f"Trial {index} of {self.total} …")
                pipeline = self._trial_pipeline(params)
                self._fit_with(pipeline, self.data.x[self.data.train], self.data.y[self.data.train])
                self.pipeline = pipeline
                rows = self.data.val
                predictions = self._predictions(rows)
            else:
                rows = self.data.train
                predictions = self._cross_validate(index, params)
            metrics = self._metrics(rows, predictions)
            score = self._score(rows, predictions, metrics)
        except StopTraining:
            self._trial_done(entry, "stopped", "Stopped.", started)
            raise
        except TrainingError as exc:
            failure = str(exc)
        except MemoryError:
            failure = f"Out of memory while fitting {len(self.data.train):,} rows."
        except Exception as exc:
            traceback.print_exc()
            failure = f"Fitting failed: {short_error(exc)}"
        if failure is not None:
            self.reporter.log(f"Trial {index} failed: {failure}")
            self._trial_done(entry, "failed", failure, started)
            return
        entry.update(
            epochs_done=1,
            val_loss=metrics["loss"],
            val_accuracy=metrics["accuracy"],
            val_rmse=metrics["rmse"],
            val_r2=metrics["r2"],
            score=score,
            weights_epoch=1,
        )
        best = self.best_trial
        new_best = best is None or better(score, best["score"], self.tuning.metric)
        if new_best and self.folds is None:
            self.best_pipeline = pipeline
        self._trial_done(entry, "completed", "", started, new_best)

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

    def _trial_pipeline(self, params: dict[str, Any]) -> Any:
        """The unfitted model with the values of a trial on the model brick."""
        graph = trial_graph(self.job.graph, self.classical["node"], params)
        analysis = analyze(graph)
        if not analysis.valid:
            raise TrainingError(
                f"The model does not accept these values: {analysis.errors[0].message}"
            )
        return build_pipeline(graph)

    def _cross_validate(self, index: int, params: dict[str, Any]) -> np.ndarray:
        """The predictions of every training row by the model of the fold that did not see
        it."""
        data = self.data
        assert self.folds is not None and self.fold_index is not None
        train = data.train
        found: np.ndarray | None = None
        for fold in range(self.folds):
            self._check_stop()
            self.reporter.status(
                "running", f"Trial {index} of {self.total}: fold {fold + 1} of {self.folds} …"
            )
            held = self.fold_index == fold
            pipeline = self._trial_pipeline(params)
            self._fit_with(pipeline, data.x[train[~held]], data.y[train[~held]])
            self.pipeline = pipeline
            predictions = self._predictions(train[held])
            if found is None:
                found = np.zeros((len(train), *predictions.shape[1:]), dtype=np.float64)
            found[held] = predictions
        assert found is not None
        return found

    def _score(
        self, rows: np.ndarray, predictions: np.ndarray, metrics: dict[str, Any]
    ) -> float | None:
        """The objective metric of a trial."""
        metric = self.tuning.metric
        if metric == DEFAULT_METRIC:
            return metrics["loss"]
        from .evaluation import ClassificationCounter, RegressionCollector

        sink: Any = (
            ClassificationCounter(self.num_classes)
            if self.classification
            else RegressionCollector()
        )
        sink.add(predictions, self.data.y[rows])
        return split_value(metric, sink.result(metrics["loss"]))

    def _fit_with(
        self, pipeline: Any, x: np.ndarray, y: np.ndarray, stoppable: bool = True
    ) -> None:
        """Fit ``pipeline`` in a thread and check the stop request meanwhile (``stoppable``).
        In the runner a stop before any trial has finished ends the process (nothing to
        keep); later the fit is left behind and the best trial is kept."""
        outcome: dict[str, Any] = {}

        def work() -> None:
            try:
                with warnings.catch_warnings(record=True) as caught, self._threads():
                    warnings.simplefilter("always")
                    pipeline.fit(x, y)
                outcome["warnings"] = caught
            except BaseException as exc:  # handed to the main thread
                outcome["error"] = exc

        thread = threading.Thread(target=work, name="brickdnn-fit", daemon=True)
        thread.start()
        while thread.is_alive():
            thread.join(STOP_CHECK_INTERVAL)
            if stoppable and thread.is_alive() and self.reporter.should_stop():
                if self.hard_stop and self.best_trial is None:
                    # Nothing to keep, and scikit-learn cannot be interrupted.
                    self.reporter.finish("stopped", STOPPED, False)
                    os._exit(0)
                if self.hard_stop:
                    self.abandoned = True
                else:
                    thread.join()
                raise StopTraining
        if "error" in outcome:
            raise outcome["error"]
        for warning in outcome.get("warnings", []):
            text = f"{warning.category.__name__}: {str(warning.message).strip()}"
            if text not in self.warned and len(self.warned) < MAX_WARNINGS:
                self.warned.add(text)
                self.reporter.log(text)

    def _use_best(self, x: np.ndarray, y: np.ndarray) -> float:
        """Make the best trial the model of the run: without cross-validation its fitted
        model, otherwise the model fitted once more on all training rows (this fit cannot be
        stopped). Returns the seconds of that fit."""
        best = self.best_trial
        assert best is not None
        what = describe_params(best["params"], self.tuning.model)
        if self.best_pipeline is not None:
            self.pipeline = self.best_pipeline
            seconds = float(best["seconds"] or 0.0)
        else:
            self.reporter.status(
                "running",
                f"Fitting the best trial ({best['trial']}: {what}) on all {len(x):,} training "
                "rows …",
            )
            self.pipeline = self._trial_pipeline(best["params"])
            started = time.perf_counter()
            self._fit_with(self.pipeline, x, y, stoppable=False)
            seconds = time.perf_counter() - started
            self.reporter.log(
                f"Fitted the best trial on all {len(x):,} training rows in {_seconds(seconds)}."
            )
        self.trial = 0
        self.reporter.search_end(
            {
                "best_trial": best["trial"],
                "params": dict(best["params"]),
                "history": [],
                "best": None,
            }
        )
        return seconds

    # -- the outcome -------------------------------------------------------- #

    def _report_evaluation(self, *args: Any) -> None:
        reporter = self.reporter
        number = self.best_trial["trial"] if self.best_trial else None
        self.reporter = _TrialEvaluation(reporter, number)  # type: ignore[assignment]
        try:
            super()._report_evaluation(*args)
        finally:
            self.reporter = reporter

    def _best_text(self) -> str:
        """``"Best: trial 7 (C 3.16, gamma 0.0123), cross-validated ROC AUC 0.931"``."""
        best = self.best_trial
        assert best is not None
        metric = self.tuning.metric
        name = "loss" if metric == DEFAULT_METRIC else label(metric)
        where = "cross-validated" if self.folds is not None else "validation"
        score = format_metric(metric, best["score"])
        what = describe_params(best["params"], self.tuning.model)
        return f"Best: trial {best['trial']} ({what}), {where} {name} {score}"

    def _summary(self) -> str:
        failed = sum(1 for t in self.trials if t["status"] == "failed")
        head = f"Finished {self.total} trial{'s' if self.total != 1 else ''}"
        if failed:
            head += f" ({failed} failed)"
        result = super()._summary().removeprefix("Finished: ")
        return f"{head}. {self._best_text()}. {result[:1].upper()}{result[1:]}"

    def _stopped_message(self) -> str:
        done = sum(1 for t in self.trials if t["status"] in ("completed", "failed"))
        if self.best_trial is None:
            return STOPPED
        return (
            f"Stopped after {done} of {self.total} trials. {self._best_text()}; its model was "
            "kept (not tested)."
        )

    def _end(self, status: str, message: str) -> TrainingResult:
        if status == "stopped":
            return self._end_stopped()
        weights = self.saved and status == "completed"
        self.reporter.finish(status, message, weights)
        return self._result(status, message, weights)

    def _result(self, status: str, message: str, weights: bool) -> TrainingResult:
        return TrainingResult(
            status,
            message,
            weights,
            self.history,
            self.best,
            self.test_result,
            self.trials,
            self.best_trial["trial"] if self.best_trial else None,
            weights_epoch=1 if weights else None,
        )

    def _end_stopped(self) -> TrainingResult:
        """Keep the best trial so far: fit it on the training rows if needed, measure it on
        the training and validation rows and save it (no test, no evaluation)."""
        weights = False
        if self.best_trial is not None:
            try:
                self._keep_best()
                weights = self.saved = self._save()
            except Exception as exc:
                traceback.print_exc()
                self.reporter.log(f"The best trial could not be kept: {short_error(exc)}")
        message = self._stopped_message() if weights else STOPPED
        self.reporter.finish("stopped", message, weights)
        result = self._result("stopped", message, weights)
        if self.abandoned:
            os._exit(0)  # the left-behind fit cannot be interrupted
        return result

    def _keep_best(self) -> None:
        best = self.best_trial
        assert best is not None
        data = self.data
        x, y = data.x[data.train], data.y[data.train]
        if self.best_pipeline is not None:
            self.pipeline = self.best_pipeline
            fit_seconds = float(best["seconds"] or 0.0)
        else:
            self.reporter.status(
                "running",
                f"Stopped: fitting the best trial ({best['trial']}) on all {len(x):,} training "
                "rows …",
            )
            self.pipeline = self._trial_pipeline(best["params"])
            started = time.perf_counter()
            self._fit_with(self.pipeline, x, y, stoppable=False)
            fit_seconds = time.perf_counter() - started
        train_predictions = self._predictions(data.train)
        trained = self._metrics(data.train, train_predictions)
        validated = self._metrics(data.val) if len(data.val) else {}
        entry: dict[str, Any] = {
            "epoch": 1,
            "train_loss": trained["loss"],
            "val_loss": validated.get("loss"),
            "train_accuracy": trained["accuracy"],
            "val_accuracy": validated.get("accuracy"),
            "seconds": round(fit_seconds, 3),
        }
        if not self.classification:
            entry["val_rmse"] = validated.get("rmse")
            entry["val_r2"] = validated.get("r2")
        self.history = [entry]
        self.best = (
            {"epoch": 1, "val_loss": entry["val_loss"], "val_accuracy": entry["val_accuracy"]}
            if validated
            else None
        )
        self.details.update(fit_seconds=round(fit_seconds, 3), train=trained, **self._learned())
        self.reporter.setup(
            {"classical": dict(self.details), "preprocessing": self._preprocessing()}
        )
        self.reporter.search_end(
            {
                "best_trial": best["trial"],
                "params": dict(best["params"]),
                "history": [dict(entry)],
                "best": dict(self.best) if self.best else None,
            }
        )
        self.reporter.weights_kept({"weights_epoch": 1, "stopped_early": False})
