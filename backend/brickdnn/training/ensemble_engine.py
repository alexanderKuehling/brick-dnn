"""The "training" of an ensemble (see ensemble.py), run by the runner like any training.

job.json of an ensemble names its members instead of a network::

    {"name": "Fraud ensemble",
     "dataset": {"kind": "csv", "path": ".../data.csv", "info": {...}},
     "ensemble": {"method": "average", "weights": [0.5, 0.5], "task": "classification",
                  "num_classes": 2, "class_names": ["no", "yes"], "class_targets": "indices",
                  "members": [{"id": "...", "name": "MLP", "dir": ".../runs/<id>",
                               "family": "neural"}, ...]},
     "limits": {...}}

It reads the dataset's validation and test rows (without either: its training rows) with the
stored split, lets every member predict them (a neural network with PyTorch, a classical model
with its scikit-learn pipeline), keeps the predictions in ``ensemble.npz``, combines and
measures them. Neural members need PyTorch, which is imported only for them.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from . import ensemble
from .datasets import DatasetInfo
from .reporting import Reporter, TrainingError, short_error
from .state import CLASSICAL_WEIGHTS_FILE, ENSEMBLE_FILE, WEIGHTS_FILE, read_job, read_state
from .tabular import CHANGED, csv_rows, read_split

# Rows a neural member predicts at a time.
BATCH_ROWS = 8192
SPLIT_LABELS = {"val": 1, "test": 2, "training": 0}


def run(job: dict[str, Any], reporter: Reporter, run_dir: str | Path) -> None:
    """Predict, combine, measure and save. Never raises; the outcome goes to ``reporter``."""
    started = time.monotonic()
    try:
        spec = job.get("ensemble") if isinstance(job, dict) else None
        if not isinstance(spec, dict):
            raise TrainingError("The ensemble job is invalid.")
        dataset = job.get("dataset") if isinstance(job.get("dataset"), dict) else {}
        info = DatasetInfo.from_dict(dataset.get("info") or {})
        path = Path(dataset.get("path") or "")
        task = spec["task"]
        members = spec["members"]
        reporter.status("preparing", "Reading the CSV file …")
        x, y, split = _read_table(path, info, spec)
        rows = {key: np.flatnonzero(split == SPLIT_LABELS[key]) for key in ("val", "test")}
        rows = {key: found for key, found in rows.items() if len(found)}
        if not rows:  # measured on the training rows, like a trained run without held-out rows
            rows = {"training": np.flatnonzero(split == 0)}
        wanted = np.concatenate(list(rows.values()))
        reporter.setup(
            {
                "family": "ensemble",
                "task": task,
                "num_classes": spec.get("num_classes"),
                "class_names": spec.get("class_names"),
                "train_size": int(np.count_nonzero(split == 0)),
                "val_size": int(len(rows.get("val", []))),
                "test_size": int(len(rows["test"])) if "test" in rows else None,
                "total_params": None,
            }
        )
        outputs: dict[str, list[np.ndarray]] = {key: [] for key in rows}
        preprocessing = None
        for number, member in enumerate(members, start=1):
            if reporter.should_stop():
                reporter.finish("stopped", "Stopped before all members had predicted.", False)
                return
            reporter.status(
                "running", f"Member {number} of {len(members)} predicts: {member['name']} …"
            )
            directory = Path(member["dir"])
            found = (read_state(directory) or {}).get("preprocessing")
            if not isinstance(found, dict) or found.get("kind") != "tabular":
                raise TrainingError(
                    f"How the inputs of '{member['name']}' were prepared is not known, so it "
                    "cannot predict."
                )
            preprocessing = preprocessing or found
            predicted = _predict(member, directory, found, x[wanted], task)
            start = 0
            for key, chosen in rows.items():
                outputs[key].append(predicted[start : start + len(chosen)])
                start += len(chosen)
            reporter.log(f"{member['name']} predicted {len(wanted):,} rows.")
        targets = {key: y[chosen] for key, chosen in rows.items()}
        reporter.status("running", "Combining and measuring …")
        result = ensemble.ensemble_result(
            task,
            spec.get("method"),
            outputs,
            targets,
            weights=spec.get("weights"),
            num_classes=spec.get("num_classes"),
            class_names=spec.get("class_names"),
            member_names=[m["name"] for m in members],
        )
        ensemble.save_outputs(Path(run_dir) / ENSEMBLE_FILE, outputs, targets, rows)
        result["history"][0]["seconds"] = round(time.monotonic() - started, 3)
        reporter.setup(
            {
                "preprocessing": preprocessing,
                "history": result["history"],
                "best": result["best"],
                "test": result["test"],
                "weights_epoch": 1,
                "ensemble": result["ensemble"],
            }
        )
        reporter.evaluation_end(result["evaluation"])
        reporter.finish("completed", ensemble.summary(task, result), True)
    except ensemble.EnsembleError as exc:
        reporter.finish("failed", str(exc), False)
    except TrainingError as exc:
        reporter.finish("failed", str(exc), False)
    except Exception as exc:  # reported to the user, with the details in the log
        import traceback

        traceback.print_exc()
        reporter.finish("failed", f"The ensemble failed: {short_error(exc)}", False)


def _read_table(
    path: Path, info: DatasetInfo, spec: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Features (float64, in the units of the CSV file), targets (class indices or values)
    and the stored split of every row."""
    if not path.is_file():
        raise TrainingError("The CSV file of the dataset was not found.")
    if info.predefined_split is None:
        raise TrainingError("The dataset has no stored split.")
    n = info.num_samples or 0
    classification = spec["task"] == "classification"
    names = [str(name) for name in spec.get("class_names") or []]
    index = {name: i for i, name in enumerate(names)}
    mode = spec.get("class_targets") or "indices"
    x = np.empty((n, info.num_features or 0), dtype=np.float64)
    y = np.empty(n, dtype=np.int64 if classification else np.float64)
    filled = 0
    with csv_rows(path, info) as (_, _, rows):
        for line, values, cell, number in rows:
            if not classification:
                if number is None:
                    raise TrainingError(f"Line {line}: the target '{cell[:30]}' is not a number.")
                y[filled] = number
            elif mode == "labels":
                if cell not in index:
                    raise TrainingError(f"Line {line}: unknown class '{cell[:30]}'.")
                y[filled] = index[cell]
            elif number is None or not number.is_integer():
                raise TrainingError(f"Line {line}: the target '{cell[:30]}' is not a whole number.")
            elif mode == "values":
                key = str(int(number))
                if key not in index:
                    raise TrainingError(f"Line {line}: unknown class '{cell[:30]}'.")
                y[filled] = index[key]
            else:
                if not 0 <= number < len(names):
                    raise TrainingError(
                        f"Line {line}: the target {cell[:30]} is not a class index between 0 "
                        f"and {len(names) - 1}."
                    )
                y[filled] = int(number)
            x[filled] = values
            filled += 1
    if filled != n:
        raise TrainingError(CHANGED)
    split = np.frombuffer(read_split(path, n, info.predefined_split), dtype=np.uint8)
    return x, y, split


def _predict(
    member: dict[str, Any],
    directory: Path,
    preprocessing: dict[str, Any],
    x: np.ndarray,
    task: str,
) -> np.ndarray:
    """A member's predictions of the rows ``x`` (CSV units): probabilities or values in the
    units of the target."""
    columns = preprocessing.get("columns")
    if not isinstance(columns, list) or len(columns) != x.shape[1]:
        raise TrainingError(f"'{member['name']}' takes other columns than this dataset.")
    if member.get("family") == "classical":
        return _predict_classical(member, directory, x, task)
    return _predict_neural(member, directory, preprocessing, x, task)


def _predict_classical(
    member: dict[str, Any], directory: Path, x: np.ndarray, task: str
) -> np.ndarray:
    import joblib

    path = directory / CLASSICAL_WEIGHTS_FILE
    if not path.is_file():
        raise TrainingError(f"'{member['name']}' has no fitted model.")
    bundle = joblib.load(path)  # written by the training of that run
    pipeline = bundle["model"]  # standardises the raw features itself
    if task == "classification":
        proba = pipeline.predict_proba(x)
        indices = bundle.get("class_indices") or list(range(proba.shape[1]))
        full = np.zeros((len(x), int(bundle.get("num_classes") or proba.shape[1])))
        full[:, indices] = proba
        return full
    return np.asarray(pipeline.predict(x), dtype=np.float64).reshape(-1)


def _predict_neural(
    member: dict[str, Any],
    directory: Path,
    preprocessing: dict[str, Any],
    x: np.ndarray,
    task: str,
) -> np.ndarray:
    try:
        import torch

        from .engine import build_model
    except ImportError as exc:
        raise TrainingError(
            f"PyTorch is not installed on the server, which '{member['name']}' needs."
        ) from exc
    job = read_job(directory)
    weights = directory / WEIGHTS_FILE
    if job is None or not isinstance(job.get("graph"), dict) or not weights.is_file():
        raise TrainingError(f"The network or the weights of '{member['name']}' are missing.")
    model = build_model(job["graph"])
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model.eval()
    mean = np.asarray(preprocessing["mean"], dtype=np.float64)
    std = np.asarray(preprocessing["std"], dtype=np.float64)
    # Standardised like the training (data.csv_data: float32).
    prepared = ((x - mean) / std).astype(np.float32)
    parts = []
    with torch.no_grad():
        for start in range(0, len(prepared), BATCH_ROWS):
            batch = torch.from_numpy(prepared[start : start + BATCH_ROWS])
            output = model(batch)
            output = output[0] if isinstance(output, (tuple, list)) else output
            output = output.double()
            if task == "classification":
                parts.append(torch.softmax(output, dim=1).numpy())
            else:
                parts.append(output.reshape(len(batch)).numpy())
    if not parts:
        return np.empty((0, 0)) if task == "classification" else np.empty(0)
    predicted = np.concatenate(parts)
    if task == "regression":
        scale = preprocessing.get("target_std")
        shift = preprocessing.get("target_mean")
        if isinstance(scale, (int, float)) and isinstance(shift, (int, float)):
            predicted = predicted * float(scale) + float(shift)
    return predicted
