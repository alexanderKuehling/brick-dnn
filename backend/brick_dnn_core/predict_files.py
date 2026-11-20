"""Predict every row of an uploaded CSV file ("Predict a file" in the run view).

Typing or pasting rows (the ``predict`` action) is for a quick try. A whole file, e.g. the
test file of a competition, is uploaded instead: the server stores it in a work folder,
lets a process of its own predict it in chunks (``python -m brickdnn.training.predict_file``,
which loads the model once and prepares the rows like a single prediction) and keeps the
result file for the download. Every run trained on a table can: neural networks, classical
models, hybrid networks and ensembles of runs.

Both files are temporary. The uploaded file is deleted as soon as the prediction has ended,
the result when it is downloaded, and at the latest TRAINING_PREDICT_FILE_KEEP_MINUTES
after it was made (checked with every new file and by ``manage.py purge_old_data``). They
live in ``TRAINING_DIR/predict-files/<id>/`` and do not count as the account's storage.

Limits: the file may have TRAINING_PREDICT_FILE_MAX_MB (and never more than a dataset
upload, TRAINING_MAX_UPLOAD_MB) and TRAINING_PREDICT_FILE_MAX_ROWS rows, and the prediction
may take TRAINING_PREDICT_FILE_TIMEOUT seconds. The CPU time counts as a prediction against
the budgets of who asks, like a single prediction (budgets.record_prediction), and nothing
is predicted when a budget is used up.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import time
import uuid
from datetime import datetime, timedelta
from datetime import timezone as dt_timezone
from pathlib import Path
from typing import Any

from django.conf import settings
from django.utils.text import slugify

from brickdnn.training import inference

from . import budgets, raw_predictions
from . import training_service as training
from .budgets import Actor
from .models import TrainingRun

logger = logging.getLogger(__name__)

FOLDER = "predict-files"
INPUT_NAME = "input.csv"
RESULT_NAME = "result.csv"
JOB_NAME = "job.json"
META_NAME = "meta.json"
RESULT_ID = re.compile(r"[0-9a-f]{32}")
SUFFIXES = (".csv", ".tsv", ".txt")
NOT_A_TABLE = (
    "Only runs trained on a table (a CSV file or a dataset from the Data tab) predict a file."
)
WRONG_FILE = "Choose a CSV file (.csv, .tsv or .txt)."
GONE = (
    "This result is no longer there: it is deleted when it is downloaded and "
    "{minutes} minutes after it was made. Predict the file again."
)


class FileTooLarge(training.TrainingInputError):
    """The uploaded file is larger than the limit (HTTP 413)."""


class ResultGone(Exception):
    """The result file was downloaded already, has expired or never existed."""


# --------------------------------------------------------------------------- #
# Limits
# --------------------------------------------------------------------------- #


def max_megabytes() -> float:
    return min(float(settings.TRAINING_PREDICT_FILE_MAX_MB), float(settings.TRAINING_MAX_UPLOAD_MB))


def max_bytes() -> int:
    return int(max_megabytes() * 1024 * 1024)


def keep_seconds() -> int:
    return max(1, int(settings.TRAINING_PREDICT_FILE_KEEP_MINUTES)) * 60


def limits() -> dict[str, Any]:
    """For the prediction form: what a file may have and how long its result is kept."""
    megabytes = max_megabytes()
    return {
        "max_mb": int(megabytes) if megabytes.is_integer() else megabytes,
        "max_rows": int(settings.TRAINING_PREDICT_FILE_MAX_ROWS),
        "keep_minutes": keep_seconds() // 60,
    }


def too_large_message() -> str:
    return f"The file is too large. A file to predict may have {max_megabytes():g} MB."


# --------------------------------------------------------------------------- #
# The temporary files
# --------------------------------------------------------------------------- #


def files_dir() -> Path:
    return training.training_dir() / FOLDER


def remove_expired(now: float | None = None) -> int:
    """Delete the results that were not downloaded in time, and work folders that a
    prediction left behind (e.g. a server restart). Returns how many folders went."""
    root = files_dir()
    if not root.is_dir():
        return 0
    now = time.time() if now is None else now
    # A prediction that still runs is younger than its time limit.
    limit = max(keep_seconds(), int(settings.TRAINING_PREDICT_FILE_TIMEOUT) + 60)
    removed = 0
    for path in root.iterdir():
        with contextlib.suppress(OSError):
            age = now - path.stat().st_mtime
            if age > (limit if path.name.startswith(".work-") else keep_seconds()):
                shutil.rmtree(path, ignore_errors=True)
                removed += 1
    return removed


def _command() -> list[str]:
    return training._command(
        settings.TRAINING_PREDICT_FILE_COMMAND, "brickdnn.training.predict_file"
    )


# --------------------------------------------------------------------------- #
# What the prediction process needs to know about the run
# --------------------------------------------------------------------------- #


def _model(run: TrainingRun, preprocessing: Any) -> dict[str, Any]:
    return {
        "run_dir": str(training.run_dir(run.pk)),
        "name": run.name,
        "preprocessing": preprocessing,
    }


def describe(run: TrainingRun) -> dict[str, Any]:
    """The part of the job that describes ``run``: its model (or the members of an
    ensemble), classes, decision threshold and data preparation. Raises
    :class:`training.TrainingUnavailable`, :class:`training.PredictionUnavailable` or
    :class:`training.TrainingInputError` like training.prepare_prediction."""
    plan = run.plan if isinstance(run.plan, dict) else {}
    ensemble = None
    raw_source = run
    if run.kind == "ensemble":
        from . import ensembles

        reason = training.unavailable_reason("ensemble")
        if reason:
            raise training.TrainingUnavailable(reason)
        rows = ensembles.members_of(run)
        for row in rows:
            if row.run is not None:
                reason = training.unavailable_reason(training.run_family(row.run))
                if reason:
                    raise training.TrainingUnavailable(reason)
        reason = ensembles.prediction_reason(run)
        if reason:
            raise training.PredictionUnavailable(reason)
        members = [row.run for row in rows]
        models = []
        for member in members:
            preprocessing = (training.state_of(member) or {}).get("preprocessing")
            _check_table(member, preprocessing, f"'{member.name}': ")
            models.append(_model(member, preprocessing))
        state = training.state_of(run) or {}
        own = state.get("preprocessing") or models[0]["preprocessing"]
        ensemble = {
            "method": ensembles.method_of(run),
            "weights": [row.weight for row in rows],
            "meta": (state.get("ensemble") or {}).get("meta"),
        }
        raw_source = ensembles._raw_source(run, members)
    else:
        reason = training.unavailable_reason(training.run_family(run))
        if reason:
            raise training.TrainingUnavailable(reason)
        if training.weights_path(run) is None:
            raise training.PredictionUnavailable(training.no_weights_reason(run))
        own = (training.state_of(run) or {}).get("preprocessing")
        _check_table(run, own)
        models = [_model(run, own)]
    if plan.get("task") not in ("classification", "regression"):
        raise training.TrainingInputError(NOT_A_TABLE)
    frozen = raw_predictions.frozen_dir(raw_source)
    if frozen is not None and raw_predictions.recipe_of(raw_source) is None:
        frozen = None
    return {
        "task": plan.get("task"),
        "class_names": inference.class_names(plan, own),
        "threshold": training.decision_threshold(run),
        "models": models,
        "ensemble": ensemble,
        "raw": str(frozen) if frozen is not None else None,
    }


def _check_table(run: TrainingRun, preprocessing: Any, prefix: str = "") -> None:
    try:
        kind = inference.kind_of(run.plan, preprocessing)
    except inference.UnknownPreprocessing as exc:
        raise training.PredictionUnavailable(f"{prefix}{exc}") from None
    if kind != "tabular":
        raise training.TrainingInputError(NOT_A_TABLE)


def _options(data: Any) -> tuple[dict[str, Any], str]:
    """The options of the result from the form fields of the upload (``columns`` is JSON)
    and the input format. Their values are checked by the prediction process."""
    get = data.get if hasattr(data, "get") else (lambda key, default=None: default)
    columns = get("columns")
    if isinstance(columns, str) and columns.strip() not in ("", "all"):
        try:
            columns = json.loads(columns)
        except ValueError:
            raise training.TrainingInputError(
                '\'columns\' must be "all" or a JSON list of column names, e.g. ["id"].'
            ) from None
    elif isinstance(columns, str):
        columns = "all"
    options = {
        "columns": "all" if columns is None else columns,
        "prediction_column": get("prediction_column"),
        "output": get("output") or None,
        "class": get("class") or None,
    }
    return options, get("input_format") or "auto"


# --------------------------------------------------------------------------- #
# Predicting a file
# --------------------------------------------------------------------------- #


def result_filename(run: TrainingRun, filename: str) -> str:
    """``test.csv`` predicted by "Sales net" -> ``test-predictions.csv``."""
    stem = slugify(Path(filename).stem) or slugify(run.name) or "file"
    return f"{stem[:80]}-predictions.csv"


def predict(run: TrainingRun, upload: Any, data: Any, actor: Actor) -> dict[str, Any]:
    """Predict every row of the uploaded file with ``run`` and keep the result for the
    download. Returns the summary of the prediction process (see
    brickdnn.training.predict_file) with ``id``, ``filename`` and ``expires_at``.

    Raises :class:`training.TrainingUnavailable`, :class:`training.PredictionUnavailable`,
    :class:`training.TrainingInputError` (:class:`FileTooLarge`), a
    :class:`budgets.BudgetError`, :class:`training.TrainingBusy` or
    :class:`training.PredictionFailed`.
    """
    job = describe(run)
    options, input_format = _options(data)
    filename = Path(getattr(upload, "name", "") or "").name
    if Path(filename).suffix.lower() not in SUFFIXES:
        raise training.TrainingInputError(WRONG_FILE)
    if int(getattr(upload, "size", 0) or 0) > max_bytes():
        raise FileTooLarge(too_large_message())
    budgets.check_compute(actor)
    remove_expired()
    root = files_dir()
    root.mkdir(parents=True, exist_ok=True)
    result_id = uuid.uuid4().hex
    work = root / f".work-{result_id}"
    work.mkdir()
    try:
        with (work / INPUT_NAME).open("wb") as out:
            for chunk in upload.chunks():
                out.write(chunk)
        job.update(
            {
                "input": str(work / INPUT_NAME),
                "output": str(work / RESULT_NAME),
                "input_format": input_format,
                "options": options,
                "max_rows": int(settings.TRAINING_PREDICT_FILE_MAX_ROWS),
            }
        )
        (work / JOB_NAME).write_text(json.dumps(job), encoding="utf-8")
        with training.prediction_slot():
            summary = _process(run, work / JOB_NAME, actor)
        # Only the result stays, and only until it is downloaded or expires.
        (work / INPUT_NAME).unlink()
        (work / JOB_NAME).unlink()
        expires = datetime.now(dt_timezone.utc) + timedelta(seconds=keep_seconds())
        summary.pop("cpu_seconds", None)
        summary.update(
            {
                "id": result_id,
                "run": {"id": str(run.pk), "name": run.name},
                "filename": result_filename(run, filename),
                "expires_at": expires.isoformat(timespec="seconds"),
                "keep_minutes": keep_seconds() // 60,
            }
        )
        meta = {"user": actor.user.pk, "run": str(run.pk), "filename": summary["filename"]}
        (work / META_NAME).write_text(json.dumps(meta), encoding="utf-8")
        os.replace(work, root / result_id)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    return summary


def _process(run: TrainingRun, job_file: Path, actor: Actor) -> dict[str, Any]:
    """Start the prediction process, count its CPU time as a prediction of ``run`` and
    return its answer. Raises :class:`training.TrainingInputError` for a file or options
    that do not fit, else :class:`training.PredictionFailed`."""
    timeout = int(settings.TRAINING_PREDICT_FILE_TIMEOUT)
    command = [*_command(), str(job_file)]
    started = time.monotonic()
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            cwd=job_file.parent,
            env=training._subprocess_env(),
        )
    except subprocess.TimeoutExpired:
        # Killed before it could report: count its wall time on all its threads.
        threads = max(1, min(training.PREDICT_THREADS, os.cpu_count() or 1))
        budgets.record_prediction(actor, run, (time.monotonic() - started) * threads)
        raise training.PredictionFailed(
            f"Predicting this file took longer than {timeout} seconds and was stopped. "
            "Split it into several files."
        ) from None
    except OSError:
        logger.exception("Could not start the prediction process %s", command)
        raise training.PredictionFailed(
            "The prediction process could not be started. See the server log for details."
        ) from None
    answer = None
    for line in reversed(result.stdout.strip().splitlines()):
        if line.strip().startswith("{"):
            with contextlib.suppress(ValueError):
                answer = json.loads(line)
            break
    spent = answer.get("cpu_seconds") if isinstance(answer, dict) else None
    if isinstance(spent, bool) or not isinstance(spent, (int, float)) or spent < 0:
        spent = time.monotonic() - started  # it did not report: at least its wall time
    budgets.record_prediction(actor, run, float(spent))
    if isinstance(answer, dict) and isinstance(answer.get("error"), str):
        message = answer["error"][:2000]
        if answer.get("kind") == "file":  # the file or the options do not fit
            raise training.TrainingInputError(message)
        raise training.PredictionFailed(message)
    if result.returncode == 0 and isinstance(answer, dict) and "rows" in answer:
        return answer
    if result.returncode in (-signal.SIGKILL, 137) or "MemoryError" in result.stderr:
        raise training.PredictionFailed(
            "Predicting this file needs more memory than the server has. Split it into "
            "several files."
        )
    lines = [line.strip() for line in result.stderr.splitlines() if line.strip()]
    detail = lines[-1][:300] if lines else f"exit code {result.returncode}"
    logger.warning("Predicting a file with run %s failed: %s", run.pk, result.stderr[-2000:])
    raise training.PredictionFailed(f"The prediction failed: {detail}")


# --------------------------------------------------------------------------- #
# The result file
# --------------------------------------------------------------------------- #


def _folder(run: TrainingRun, result_id: str, user) -> tuple[Path, dict[str, Any]]:
    """The folder of a result that ``user`` made with ``run``. Raises :class:`ResultGone`."""
    gone = ResultGone(GONE.format(minutes=keep_seconds() // 60))
    if not RESULT_ID.fullmatch(str(result_id)):
        raise gone
    folder = files_dir() / str(result_id)
    try:
        meta = json.loads((folder / META_NAME).read_text(encoding="utf-8"))
        age = time.time() - folder.stat().st_mtime
    except (OSError, ValueError):
        raise gone from None
    if age > keep_seconds():
        shutil.rmtree(folder, ignore_errors=True)
        raise gone
    # Only who predicted the file gets its result (it holds the rows they uploaded).
    if meta.get("user") != user.pk or meta.get("run") != str(run.pk):
        raise gone
    return folder, meta


def open_result(run: TrainingRun, result_id: str, user) -> tuple[Any, str, int]:
    """The result file, open for reading, its download name and its size. The file is
    deleted here already: the open file is the last way to read it, so it is gone once the
    download has ended. Raises :class:`ResultGone`."""
    folder, meta = _folder(run, result_id, user)
    try:
        handle = (folder / RESULT_NAME).open("rb")
    except OSError:
        raise ResultGone(GONE.format(minutes=keep_seconds() // 60)) from None
    size = os.fstat(handle.fileno()).st_size
    shutil.rmtree(folder, ignore_errors=True)
    return handle, str(meta.get("filename") or "predictions.csv"), size


def discard(run: TrainingRun, result_id: str, user) -> None:
    """Delete a result that is not wanted any more (nothing happens if it is gone)."""
    with contextlib.suppress(ResultGone):
        folder, _ = _folder(run, result_id, user)
        shutil.rmtree(folder, ignore_errors=True)
