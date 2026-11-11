"""Predictions from raw rows, and what the prediction form of the web app needs.

A run trained on a dataset exported from a data preparation (Data tab) can predict rows in
the columns of the imported source table: the export stores the preparation with the values
it learned (:mod:`brickdnn.data.frozen`, ``<dataset>/prepare/``), and :func:`prepare_raw`
turns raw rows into the prepared feature columns the network takes. A run keeps its own copy
when its dataset is deleted (``<run>/prepare/``).

:func:`prediction_form` describes the inputs of a run for the form in the run view: the
feature columns (or the raw columns) with example rows, or the image size.
"""

from __future__ import annotations

import csv
import logging
import os
import shutil
from pathlib import Path
from typing import Any

from django.db.models.signals import pre_delete
from django.dispatch import receiver

from brickdnn.training import inference

from . import budgets
from . import training_service as training
from .models import Dataset, TrainingRun

logger = logging.getLogger(__name__)

INPUT_FORMATS = ("prepared", "raw")
# Rows the web app sends from a pasted or uploaded CSV file (in requests of MAX_SAMPLES).
MAX_FORM_ROWS = 500
SAMPLE_ROWS = 5
NOT_FROM_PREPARATION = (
    "input_format 'raw' is for runs trained on a dataset exported from a data preparation; "
    "this run takes the feature columns of its dataset (input_format 'prepared')."
)
OLD_EXPORT = (
    "This run's dataset was exported before raw rows were supported: export the "
    "preparation again and train on the new dataset to predict from raw rows."
)


def _frozen_name() -> str:
    from brickdnn.data.frozen import FROZEN_DIR

    return FROZEN_DIR


def frozen_dir(run: TrainingRun) -> Path | None:
    """The folder of the frozen preparation of ``run``'s dataset (its own copy first)."""
    try:
        name = _frozen_name()
    except ImportError:  # pandas is not installed
        return None
    own = training.run_dir(run.pk) / name
    if own.is_dir():
        return own
    if run.upload_id is not None:
        shared = training.dataset_dir(run.upload_id) / name
        if shared.is_dir():
            return shared
    return None


def from_preparation(run: TrainingRun) -> bool:
    upload = run.upload
    return upload is not None and upload.prepared_from_id is not None


def recipe_of(run: TrainingRun) -> tuple[dict[str, Any], Path] | None:
    from brickdnn.data.frozen import read_recipe

    directory = frozen_dir(run)
    if directory is None:
        return None
    recipe = read_recipe(directory)
    return (recipe, directory) if recipe is not None else None


def check_format(value: Any) -> str:
    if value is None:
        return "prepared"
    if value not in INPUT_FORMATS:
        raise training.TrainingInputError(
            f"input_format must be one of: {', '.join(INPUT_FORMATS)}."
        )
    return value


class RawRowsError(training.TrainingInputError):
    """Raw rows that cannot be prepared; ``errors`` lists the problems per row and column."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__(errors[0] if errors else "The rows are invalid.")


def prepare_raw(run: TrainingRun, inputs: Any) -> tuple[list[dict[str, float]], list[str]]:
    """Raw rows of ``run``'s source table as prepared rows ``{feature: number}`` plus
    warnings. Raises :class:`training.TrainingInputError` (:class:`RawRowsError`)."""
    try:
        from brickdnn.data.frozen import RawInputError, prepare_rows
    except ImportError:
        raise training.TrainingInputError(
            "Raw rows need pandas on the server, which is not installed; send the prepared "
            "feature columns instead."
        ) from None
    found = recipe_of(run)
    if found is None:
        raise training.TrainingInputError(
            OLD_EXPORT if from_preparation(run) else NOT_FROM_PREPARATION
        )
    recipe, directory = found
    try:
        prepared = prepare_rows(recipe, inputs, directory=directory, max_rows=inference.MAX_SAMPLES)
    except RawInputError as exc:
        raise RawRowsError(exc.messages) from None
    return prepared.rows, prepared.warnings


def raw_hint(run: TrainingRun) -> str | None:
    """A sentence for describe_inputs: how to send raw rows (None without a frozen prep)."""
    found = recipe_of(run)
    if found is None:
        return None
    recipe = found[0]
    names = [c["name"] for c in recipe.get("columns") or [] if c.get("used", True)]
    source = (recipe.get("source") or {}).get("name") or "the source table"
    shown = ", ".join(names[:30]) + (f" and {len(names) - 30} more" if len(names) > 30 else "")
    return (
        f"Or raw rows with input_format 'raw': the columns of '{source}' as imported, before "
        f"the data preparation ({shown}); the server prepares them with the values the "
        "preparation learned from the training rows. Dates as 2024-05-31."
    )


# --------------------------------------------------------------------------- #
# A run keeps the frozen preparation when its dataset is deleted
# --------------------------------------------------------------------------- #


def _copy_tree(source: Path, target: Path) -> None:
    def link(src: str, dst: str) -> None:
        try:
            os.link(src, dst)  # the same files: no extra disk space
        except OSError:
            shutil.copy2(src, dst)

    shutil.copytree(source, target, copy_function=link, dirs_exist_ok=True)


@receiver(pre_delete, sender=Dataset, dispatch_uid="brickdnn_keep_frozen_preparation")
def _keep_frozen_preparation(sender, instance: Dataset, **kwargs) -> None:
    if instance.prepared_from_id is None:
        return
    try:
        name = _frozen_name()
    except ImportError:
        return
    source = training.dataset_dir(instance.pk) / name
    if not source.is_dir():
        return
    for run in TrainingRun.objects.filter(upload=instance, has_weights=True).only("pk"):
        target = training.run_dir(run.pk) / name
        if target.exists() or not target.parent.is_dir():
            continue
        try:
            _copy_tree(source, target)
        except OSError:
            logger.exception("Could not keep the preparation of dataset %s", instance.pk)


# --------------------------------------------------------------------------- #
# The prediction form of the web app
# --------------------------------------------------------------------------- #


def _csv_samples(run: TrainingRun, preprocessing: dict[str, Any]) -> list[dict[str, Any]]:
    """A few rows of the dataset (its validation rows if the split is known)."""
    upload = run.upload
    if upload is None or upload.kind != "csv":
        return []
    directory = training.dataset_dir(upload.pk)
    path = directory / "data.csv"
    if not path.is_file():
        return []
    wanted: set[int] | None = None
    split = directory / "split.bin"
    if split.is_file():
        try:
            marks = split.read_bytes()
        except OSError:
            marks = b""
        wanted = {i for i, mark in enumerate(marks) if mark == 1}
        wanted = set(sorted(wanted)[:SAMPLE_ROWS]) or None
    columns = [str(c) for c in preprocessing.get("columns") or []]
    target = preprocessing.get("target")
    names = inference.class_names(run.plan, preprocessing)
    delimiter = preprocessing.get("delimiter") or ","
    samples: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, delimiter=delimiter)
            header = [h.strip() for h in next(reader, [])]
            positions = {name: header.index(name) for name in columns if name in header}
            if len(positions) != len(columns):
                return []
            target_at = header.index(target) if target in header else None
            for number, row in enumerate(reader):
                if wanted is not None and number not in wanted:
                    if number > max(wanted):
                        break
                    continue
                values = {name: row[i].strip() for name, i in positions.items() if i < len(row)}
                sample: dict[str, Any] = {"values": values}
                if target_at is not None and target_at < len(row):
                    actual = row[target_at].strip()
                    if names is not None and actual.isdigit() and int(actual) < len(names):
                        actual = names[int(actual)]
                    sample["target"] = actual
                samples.append(sample)
                if len(samples) >= SAMPLE_ROWS:
                    break
    except (OSError, UnicodeDecodeError, csv.Error):
        return []
    return samples


def _budget_note(actor: budgets.Actor) -> str | None:
    try:
        budgets.check_compute(actor)
    except budgets.BudgetError as exc:
        return exc.message
    return None


def prediction_form(run: TrainingRun, actor: budgets.Actor) -> dict[str, Any]:
    """What the prediction panel of the run view shows (``PredictionForm`` in
    frontend/src/api/types.ts)."""
    form: dict[str, Any] = {
        "available": False,
        "reason": None,
        "kind": None,
        "task": run.plan.get("task") if isinstance(run.plan, dict) else None,
        "class_names": None,
        "max_rows": inference.MAX_SAMPLES,
        "max_form_rows": MAX_FORM_ROWS,
        "max_top_k": inference.MAX_TOP_K,
        "budget_note": None,
    }
    family = training.run_family(run)
    reason = training.unavailable_reason(family)
    if reason is None and training.weights_path(run) is None:
        reason = training.no_weights_reason(run)
    preprocessing = (training.state_of(run) or {}).get("preprocessing")
    try:
        kind = inference.kind_of(run.plan, preprocessing)
    except inference.UnknownPreprocessing as exc:
        reason = reason or str(exc)
        kind = None
    form["kind"] = kind
    if reason is not None or kind is None:
        form["reason"] = reason
        return form
    form["available"] = True
    form["class_names"] = inference.class_names(run.plan, preprocessing)
    form["budget_note"] = _budget_note(actor)
    if kind == "tabular":
        columns = [str(c) for c in preprocessing["columns"]]
        form["target"] = preprocessing.get("target")
        form["prepared"] = {
            "columns": [
                {"name": name, "mean": mean, "std": std}
                for name, mean, std in zip(
                    columns, preprocessing["mean"], preprocessing["std"], strict=True
                )
            ],
            "samples": _csv_samples(run, preprocessing),
        }
        form["gaussian_process"] = (
            isinstance(run.plan.get("classical"), dict)
            and run.plan["classical"].get("type") == "GaussianProcessRegressor"
        )
        found = recipe_of(run)
        if found is not None:
            from brickdnn.data.frozen import describe

            form["raw"] = describe(found[0])
        elif from_preparation(run):
            form["raw_note"] = OLD_EXPORT
    elif kind == "image":
        c, h, w = (preprocessing[key] for key in ("channels", "height", "width"))
        form["image"] = {"channels": c, "height": h, "width": w}
    return form
