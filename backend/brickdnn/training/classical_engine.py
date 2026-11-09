"""Fit a classical model (scikit-learn) as a training run: the same job, reporter and run
directory as the PyTorch engine, but a single fit instead of epochs. Needs scikit-learn
(and with it NumPy and joblib), not PyTorch.

The model is the ``make_model()`` pipeline of the exported script (see
:mod:`brickdnn.classical`), built by executing that code, and it is saved with joblib in
the format of the script's ``--save``, so the downloaded file works with the script::

    {"format": "brickdnn-sklearn-1", "model": <fitted Pipeline>, "task": "classification",
     "columns": [...], "target": "species", "class_names": [...],    # as the script writes
     "class_indices": [0, 1, 2], "num_classes": 3}                  # server only

The data is read and split by the same rules as for neural networks (seeded split,
``val_split``/``test_split`` or the split of a prepared dataset, ``max_samples``); the
features are standardised by the pipeline with the statistics of the training rows. Larger
training sets than the row limit of the model (see :func:`.config.classical_max_rows`) are
subsampled like with ``max_samples``.

scikit-learn cannot be interrupted while it fits. The fit runs in a thread and the stop
request is checked meanwhile: the runner (``hard_stop``) then ends its process at once,
other callers wait for the fit and discard it.
"""

from __future__ import annotations

import math
import os
import random
import threading
import time
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from ..classical import MODEL_FORMAT
from ..codegen import generate_code
from ..graph import analyze
from ..layers import ClassicalModel
from .compat import TrainingPlan, check_compatibility
from .config import TrainingConfig
from .datasets import DatasetInfo
from .job import TrainingJob, TrainingResult
from .reporting import Reporter, StopTraining, TrainingError, short_error
from .state import finite, preprocessing_path, write_json_atomic
from .tabular import CHANGED, csv_columns, csv_rows, held_out_size, read_split, split_sizes

# Demo data: this many random rows.
DEMO_ROWS = 1000
# Rows per prediction call: a Gaussian process builds a (training rows × chunk) matrix.
PREDICT_CHUNK = 2048
# How often the stop request is checked while fitting (seconds).
STOP_CHECK_INTERVAL = 0.2
# Warnings of the fit (e.g. hyperparameters at a bound) shown in the run log.
MAX_WARNINGS = 5
# SVC calibrates its probabilities with this many folds (CalibratedClassifierCV).
CALIBRATION_FOLDS = 5

NO_TUNING = (
    "Hyperparameter search is not available for classical models yet. Switch it off: a "
    "Gaussian process optimises its kernel hyperparameters while it is fitted."
)
STOPPED = "Stopped before the model was fitted: nothing was saved."
WHAT = {"gaussian_process": "Gaussian process", "svm": "support vector machine"}


def _seconds(seconds: float) -> str:
    return f"{seconds:.1f} s" if seconds >= 0.1 else "less than 0.1 s"


def build_pipeline(graph: Any) -> Any:
    """The ``make_model()`` pipeline of the exported script for ``graph`` (unfitted).

    The generated code only contains validated numbers and identifiers, so running it is
    safe. Raises :class:`~brickdnn.GraphValidationError` for invalid graphs.
    """
    code = generate_code(graph, {"include_main": False})
    namespace: dict[str, Any] = {"__name__": "brickdnn_model"}
    exec(compile(code, "<brickdnn model>", "exec"), namespace)
    return namespace["make_model"]()


def final_estimator(pipeline: Any) -> Any:
    """The fitted model of the pipeline (SVC: the SVM inside the calibration)."""
    model = pipeline.named_steps["model"]
    calibrated = getattr(model, "calibrated_classifiers_", None)
    return calibrated[0].estimator if calibrated else model


def train(
    job: TrainingJob,
    reporter: Reporter,
    weights_path: str | Path | None = None,
    *,
    hard_stop: bool = False,
) -> TrainingResult:
    """Fit the classical model of ``job`` and save it to ``weights_path`` (joblib). Never
    raises (except ``KeyboardInterrupt``); the outcome is reported with ``reporter.finish``
    and returned. ``hard_stop``: end the process when a stop is requested during the fit
    (for the runner, which has a process of its own)."""
    return _ClassicalRun(job, reporter, weights_path, hard_stop).run()


@dataclass
class _Data:
    x: np.ndarray  # features (float64), all rows
    y: np.ndarray  # targets: values (float64) or class indices (int64)
    train: np.ndarray  # row indices
    val: np.ndarray
    test: np.ndarray | None
    columns: list[str]  # feature columns + target column
    delimiter: str
    subsampled_from: int | None  # training rows before the row limit applied
    split: dict[str, Any] | None  # the split of a prepared dataset, for the preprocessing


class _ClassicalRun:
    def __init__(
        self,
        job: TrainingJob,
        reporter: Reporter,
        weights_path: str | Path | None,
        hard_stop: bool,
    ):
        self.job = job
        self.config: TrainingConfig = job.config
        self.reporter = reporter
        self.weights_path = Path(weights_path) if weights_path is not None else None
        self.hard_stop = hard_stop
        self.history: list[dict[str, Any]] = []
        self.best: dict[str, Any] | None = None
        self.test_result: dict[str, Any] | None = None
        self.saved = False
        self.pipeline: Any = None
        self.rows = 0
        self._last_stop_check = 0.0

    # -- outcome ---------------------------------------------------------- #

    def run(self) -> TrainingResult:
        try:
            self._execute()
        except StopTraining:
            return self._end("stopped", STOPPED)
        except TrainingError as exc:
            return self._end("failed", str(exc))
        except Exception as exc:
            traceback.print_exc()
            return self._end("failed", self._explain(exc))
        return self._end("completed", self._summary())

    def _end(self, status: str, message: str) -> TrainingResult:
        weights = self.saved and status == "completed"
        self.reporter.finish(status, message, weights)
        return TrainingResult(
            status,
            message,
            weights,
            self.history,
            self.best,
            self.test_result,
            weights_epoch=1 if weights else None,
        )

    def _explain(self, exc: Exception) -> str:
        if isinstance(exc, MemoryError):
            return (
                f"Out of memory while fitting {self.rows:,} rows. Set Max. samples to fit on "
                "fewer rows."
            )
        text = f"Fitting failed: {short_error(exc)}"
        layer = getattr(self, "layer", None)
        if isinstance(exc, np.linalg.LinAlgError) and layer and layer.scale == "gaussian_process":
            text += " Try a larger alpha, or add a WhiteKernel to the kernel."
        return text

    def _check_stop(self, force: bool = True) -> None:
        now = time.monotonic()
        if force or now - self._last_stop_check >= STOP_CHECK_INTERVAL:
            self._last_stop_check = now
            if self.reporter.should_stop():
                raise StopTraining

    # -- the run ----------------------------------------------------------- #

    def _execute(self) -> None:
        job, reporter = self.job, self.reporter
        reporter.status("preparing", "Preparing the data …")
        if job.tuning is not None:
            raise TrainingError(NO_TUNING)
        analysis = analyze(job.graph)
        compat = check_compatibility(analysis, job.dataset, self.config)
        if not compat.ok or compat.plan is None or compat.plan.classical is None:
            raise TrainingError(
                compat.errors[0] if compat.errors else "The model cannot be trained."
            )
        self.plan: TrainingPlan = compat.plan
        self.classical: dict[str, Any] = compat.plan.classical
        model = analysis.model
        assert model is not None and isinstance(model.layer, ClassicalModel)
        self.layer: ClassicalModel = model.layer
        self.classification = self.layer.task == "classification"
        self.num_classes = self.plan.num_classes or 0

        seed = self.config.seed
        if seed is None:
            seed = random.SystemRandom().randrange(2**31)
            reporter.log(f"Random seed: {seed}")
        self.seed = seed
        self._check_stop()
        data = self.data = self._load()
        self.rows = len(data.train)
        if self.classification:
            self._check_classes(data.y[data.train])
        else:
            spread = float(np.var(data.y[data.train]))
            self.target_variance = spread if spread > 1e-12 else 1.0

        self.pipeline = build_pipeline(job.graph)
        self.details: dict[str, Any] = {
            "estimator": self.layer.estimator,
            "label": self.layer.label,
            "max_rows": self.classical["max_rows"],
            "rows": len(data.train),
            "subsampled_from": data.subsampled_from,
            "fit_seconds": None,
            "initial_kernel": self.classical.get("kernel"),
            "kernel": None,
            "log_marginal_likelihood": None,
            "support_vectors": None,
            "train": None,
        }
        reporter.setup(
            {
                "device": "cpu",
                "device_label": "CPU (scikit-learn)",
                "family": "classical",
                "task": self.plan.task,
                "num_classes": self.plan.num_classes,
                "class_names": self.plan.class_names,
                "train_size": len(data.train),
                "val_size": len(data.val),
                "test_size": len(data.test) if data.test is not None else None,
                "epochs": 1,
                "steps_per_epoch": 1,
                "total_params": analysis.total_params,
                "classical": dict(self.details),
            }
        )
        self._check_stop()

        what = WHAT[self.layer.scale]
        reporter.status("running", f"Fitting the {what} on {len(data.train):,} rows …")
        started = time.perf_counter()
        self._fit(data.x[data.train], data.y[data.train])
        fit_seconds = time.perf_counter() - started
        reporter.log(f"Fitted the {what} on {len(data.train):,} rows in {_seconds(fit_seconds)}.")

        reporter.status("running", "Measuring the fitted model …")
        trained = self._metrics(data.train)
        validated = self._metrics(data.val)
        entry: dict[str, Any] = {
            "epoch": 1,
            "train_loss": trained["loss"],
            "val_loss": validated["loss"],
            "train_accuracy": trained["accuracy"],
            "val_accuracy": validated["accuracy"],
            "seconds": round(time.perf_counter() - started, 3),
        }
        if not self.classification:
            entry["val_rmse"] = validated["rmse"]
            entry["val_r2"] = validated["r2"]
        self.history = [entry]
        self.best = {
            "epoch": 1,
            "val_loss": entry["val_loss"],
            "val_accuracy": entry["val_accuracy"],
        }
        self.details.update(fit_seconds=round(fit_seconds, 3), train=trained, **self._learned())
        reporter.setup({"classical": dict(self.details), "preprocessing": self._preprocessing()})
        reporter.epoch_end(dict(entry), dict(self.best))
        reporter.weights_kept({"weights_epoch": 1, "stopped_early": False})
        if data.test is not None:
            reporter.status("running", "Testing the fitted model …")
            self.test_result = self._metrics(data.test)
            reporter.test_end(dict(self.test_result))
        self.saved = self._save()

    def _summary(self) -> str:
        text = f"Finished: {self._metrics_text('validation', self.history[0], 'val_')}"
        if self.test_result is not None:
            text += f"; {self._metrics_text('test', self.test_result)}"
        seconds = self.details["fit_seconds"]
        return f"{text}. Fitted on {self.rows:,} rows in {_seconds(seconds)}."

    def _metrics_text(self, name: str, metrics: dict[str, Any], prefix: str = "") -> str:
        accuracy = metrics.get(f"{prefix}accuracy")
        if accuracy is not None:
            return f"{name} accuracy {accuracy * 100:.1f}%"
        text = f"{name} RMSE {metrics[f'{prefix}rmse']:.4g}"
        r2 = metrics.get(f"{prefix}r2")
        return text + (f", R² {r2:.3f}" if r2 is not None else "")

    # -- data -------------------------------------------------------------- #

    def _load(self) -> _Data:
        info = self.job.dataset
        n_features = self.plan.inputs[0]["shape"][0]
        stored: np.ndarray | None = None  # the split of a prepared dataset
        if info.kind == "random":
            x, y = self._demo(n_features)
            columns = [*(f"x{i + 1}" for i in range(n_features)), "y"]
            delimiter = ","
        else:
            if self.job.path is None:
                raise TrainingError("The dataset has no file. Upload it again.")
            path = Path(self.job.path)
            if not path.is_file():
                raise TrainingError("The CSV file was not found. Upload it again.")
            self.reporter.status("preparing", "Reading the CSV file …")
            x, y, columns, delimiter = self._read_csv(path, info)
            if info.predefined_split is not None:
                raw = read_split(path, len(x), info.predefined_split)
                stored = np.frombuffer(raw, dtype=np.uint8)

        def split(max_samples: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
            if stored is not None:
                return self._predefined_split(stored, max_samples)
            return self._random_split(len(x), max_samples)

        train, val, test = split(self.config.max_samples)
        # The row limit of the model works like Max. samples (compat refuses a larger one).
        limit = int(self.classical["max_rows"])
        subsampled_from = None
        if len(train) > limit:
            subsampled_from = len(train)
            train, val, test = split(limit)
            self.reporter.log(
                f"{self.layer.label} is fitted on at most {limit:,} rows on this server: using "
                f"{len(train):,} random rows of the {subsampled_from:,} training rows "
                "(validation and test data shrink in proportion)."
            )
        counts = None
        if stored is not None:
            sizes = [len(train), len(val), len(test) if test is not None else 0]
            counts = {"source": "dataset", "counts": sizes}
        return _Data(x, y, train, val, test, columns, delimiter, subsampled_from, counts)

    def _rng(self, purpose: int) -> np.random.Generator:
        """A generator that depends only on the seed and what it is for, so a smaller
        ``max_samples`` takes the first rows of the same split."""
        return np.random.default_rng([self.seed, purpose])

    def _random_split(
        self, n: int, max_samples: int | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Like :func:`.data.split_indices` (with NumPy's random numbers)."""
        config = self.config
        _, n_val, n_test = split_sizes(n, config.val_split, config.test_split)
        order = self._rng(1).permutation(n)
        test = order[:n_test] if n_test else None
        val = order[n_test : n_test + n_val]
        train = order[n_test + n_val :]
        if max_samples is not None:
            share = 1 - config.val_split - config.test_split
            train = train[:max_samples]
            val = val[: held_out_size(max_samples, config.val_split, share)]
            if test is not None:
                test = test[: held_out_size(max_samples, config.test_split, share)]
        return train, val, test

    def _predefined_split(
        self, split: np.ndarray, max_samples: int | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Like :func:`.data.predefined_split_indices`."""
        parts = [np.flatnonzero(split == label) for label in range(3)]
        n_train = len(parts[0])
        if max_samples is not None and max_samples < n_train:
            sizes = [max_samples]
            sizes += [min(len(p), held_out_size(max_samples, len(p), n_train)) for p in parts[1:]]
            pick = self._rng(2)
            parts = [
                np.sort(p[pick.permutation(len(p))[:size]])
                for p, size in zip(parts, sizes, strict=True)
            ]
        train, val, test = parts
        return train, val, test if len(test) else None

    def _read_csv(
        self, path: Path, info: DatasetInfo
    ) -> tuple[np.ndarray, np.ndarray, list[str], str]:
        n = info.num_samples or 0
        x = np.empty((n, info.num_features or 0), dtype=np.float64)
        y = np.empty(n, dtype=np.int64 if self.classification else np.float64)
        names = list(self.plan.class_names or [])
        index = {name: i for i, name in enumerate(names)}
        mode = self.classical.get("class_targets")
        filled = 0
        with csv_rows(path, info) as (delimiter, _, rows):
            for line, values, cell, number in rows:
                if not self.classification:
                    if number is None:
                        raise TrainingError(
                            f"Line {line}: the target '{cell[:30]}' is not a number."
                        )
                    y[filled] = number
                elif mode == "labels":
                    if cell not in index:
                        raise TrainingError(f"Line {line}: unknown class '{cell[:30]}'.")
                    y[filled] = index[cell]
                elif number is None or not number.is_integer():
                    raise TrainingError(
                        f"Line {line}: the target '{cell[:30]}' is not a whole number."
                    )
                elif mode == "indices":
                    if not 0 <= number < len(names):
                        raise TrainingError(
                            f"Line {line}: the target {cell[:30]} is not a class index between "
                            f"0 and {len(names) - 1}."
                        )
                    y[filled] = int(number)
                else:  # the classes are the values of the column
                    key = str(int(number))
                    if key not in index:
                        raise TrainingError(f"Line {line}: unknown class '{cell[:30]}'.")
                    y[filled] = index[key]
                x[filled] = values
                filled += 1
                if filled % 4096 == 0:
                    self._check_stop(force=False)
        if filled != n:
            raise TrainingError(CHANGED)
        return x, y, csv_columns(info), delimiter

    def _demo(self, n_features: int) -> tuple[np.ndarray, np.ndarray]:
        """Random features and a hidden smooth rule (regression) or class boundaries."""
        rng = self._rng(0)
        x = rng.normal(size=(DEMO_ROWS, n_features))
        direction = rng.normal(size=n_features) / math.sqrt(n_features)
        z = x @ direction
        if not self.classification:
            return x, np.sin(2 * z) + 0.5 * z + 0.1 * rng.normal(size=DEMO_ROWS)
        weights = rng.normal(size=(n_features, self.num_classes))
        scores = x @ weights + np.sin(2 * z)[:, None] * rng.normal(size=self.num_classes)
        return x, np.argmax(scores, axis=1).astype(np.int64)

    def _check_classes(self, y: np.ndarray) -> None:
        names = self.plan.class_names or []
        counts = np.bincount(y, minlength=len(names))
        present = counts[counts > 0]
        if len(present) < 2:
            only = names[int(np.argmax(counts))] if len(names) else "?"
            raise TrainingError(
                f"The training rows only hold one class ('{only}'). Classification needs at "
                "least two."
            )
        if self.layer.type == "SVC" and (present < CALIBRATION_FOLDS).all():
            raise TrainingError(
                f"SVC needs at least {CALIBRATION_FOLDS} training rows of some class to turn "
                f"its scores into probabilities ({CALIBRATION_FOLDS}-fold cross-validation)."
            )

    # -- fitting ------------------------------------------------------------- #

    def _fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fit the pipeline in a thread and check the stop request meanwhile."""
        outcome: dict[str, Any] = {}

        def work() -> None:
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    self.pipeline.fit(x, y)
                outcome["warnings"] = caught
            except BaseException as exc:  # handed to the main thread
                outcome["error"] = exc

        thread = threading.Thread(target=work, name="brickdnn-fit", daemon=True)
        thread.start()
        while thread.is_alive():
            thread.join(STOP_CHECK_INTERVAL)
            if thread.is_alive() and self.reporter.should_stop():
                if self.hard_stop:
                    # scikit-learn cannot be interrupted: end the runner's process.
                    self.reporter.finish("stopped", STOPPED, False)
                    os._exit(0)
                thread.join()
                raise StopTraining
        if "error" in outcome:
            raise outcome["error"]
        seen: set[str] = set()
        for warning in outcome.get("warnings", []):
            text = f"{warning.category.__name__}: {str(warning.message).strip()}"
            if text not in seen and len(seen) < MAX_WARNINGS:
                seen.add(text)
                self.reporter.log(text)

    def _learned(self) -> dict[str, Any]:
        """What the fit found: the kernel of a Gaussian process, the support vectors of an
        SVM."""
        model = final_estimator(self.pipeline)
        if self.layer.scale == "gaussian_process":
            likelihood = getattr(model, "log_marginal_likelihood_value_", None)
            return {
                "kernel": self._fitted_kernel(model),
                "log_marginal_likelihood": float(likelihood) if likelihood is not None else None,
            }
        support = getattr(model, "support_", None)
        return {"support_vectors": int(len(support)) if support is not None else None}

    def _fitted_kernel(self, model: Any) -> str:
        """The fitted kernel; with more than two classes a Gaussian process classifier fits
        one per class (one against the rest): ``"setosa: …; versicolor: …"``."""
        per_class = getattr(getattr(model, "base_estimator_", None), "estimators_", None)
        if not per_class:
            return str(model.kernel_)
        names = self.plan.class_names or []
        classes = [int(c) for c in model.classes_]
        return "; ".join(
            f"{names[c] if c < len(names) else c}: {estimator.kernel_}"
            for c, estimator in zip(classes, per_class, strict=False)
        )

    # -- measuring ----------------------------------------------------------- #

    def _chunks(self, x: np.ndarray, method: str) -> np.ndarray:
        call = getattr(self.pipeline, method)
        parts = [call(x[i : i + PREDICT_CHUNK]) for i in range(0, len(x), PREDICT_CHUNK)]
        return np.concatenate(parts) if parts else np.empty(0)

    def probabilities(self, x: np.ndarray) -> np.ndarray:
        """One column per class of the plan (classes missing from the training rows: 0)."""
        proba = self._chunks(x, "predict_proba")
        full = np.zeros((len(x), self.num_classes))
        full[:, np.asarray(self.pipeline.classes_, dtype=np.int64)] = proba
        return full

    def _metrics(self, rows: np.ndarray) -> dict[str, Any]:
        """``{"loss", "accuracy", "rmse", "r2"}`` on ``rows``. The loss is the cross-entropy
        (classification) or the mean squared error of the standardised target, like the
        loss of a neural network."""
        x, y = self.data.x[rows], self.data.y[rows]
        if self.classification:
            proba = self.probabilities(x)
            true = proba[np.arange(len(y)), y]
            loss = float(-np.mean(np.log(np.clip(true, 1e-15, 1.0))))
            accuracy = float(np.mean(np.argmax(proba, axis=1) == y))
            return {"loss": loss, "accuracy": accuracy, "rmse": None, "r2": None}
        errors = self._chunks(x, "predict") - y
        squares = float(np.sum(errors**2))
        mse = squares / max(len(y), 1)
        spread = float(np.sum((y - y.mean()) ** 2))
        return {
            "loss": mse / self.target_variance,
            "accuracy": None,
            "rmse": math.sqrt(mse),
            "r2": 1 - squares / spread if spread > 0 else None,
        }

    # -- saving -------------------------------------------------------------- #

    def _preprocessing(self) -> dict[str, Any]:
        if self.job.dataset.kind == "random":
            return {"kind": "demo"}
        data = self.data
        scaler = self.pipeline.named_steps["scale"]
        preprocessing: dict[str, Any] = {
            "kind": "tabular",
            "columns": data.columns[:-1],
            "target": data.columns[-1],
            "delimiter": data.delimiter,
            "decimal_comma": data.delimiter != ",",
            "mean": [float(v) for v in scaler.mean_],
            "std": [float(v) for v in scaler.scale_],
            # Classical models predict in the units of the target column.
            "target_mean": None,
            "target_std": None,
            "class_names": list(self.plan.class_names or []) if self.classification else None,
            "estimator": self.layer.estimator,
        }
        if data.split is not None:
            preprocessing["split"] = data.split
        return preprocessing

    def _save(self) -> bool:
        if self.weights_path is None:
            return False
        data = self.data
        bundle: dict[str, Any] = {
            "format": MODEL_FORMAT,
            "model": self.pipeline,
            "task": self.layer.task,
            "columns": data.columns[:-1],
            "target": data.columns[-1],
            "class_names": None,
        }
        if self.classification:
            indices = [int(c) for c in self.pipeline.classes_]
            names = self.plan.class_names or []
            bundle["class_names"] = [names[i] for i in indices]
            bundle["class_indices"] = indices
            bundle["num_classes"] = self.num_classes
        tmp = self.weights_path.with_name(f".{self.weights_path.name}.tmp")
        joblib.dump(bundle, tmp)
        os.replace(tmp, self.weights_path)
        preprocessing = self._preprocessing()
        if preprocessing["kind"] != "demo":
            write_json_atomic(preprocessing_path(self.weights_path), finite(preprocessing))
        return True
