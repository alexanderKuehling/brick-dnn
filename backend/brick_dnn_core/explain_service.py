"""Explanations of trained models (see ``brickdnn.training.explain`` and docs/explain.md).

Like a prediction, an explanation runs the trained model in a short-lived process
(``python -m brickdnn.training.predict RUN_DIR`` with ``{"explain": {...}}``), with the same
time limit, the same slots (at most TRAINING_MAX_PREDICTIONS at a time) and its CPU time
counted as a prediction of whoever asks. The web server never imports PyTorch.

Answers are cached in the run directory (``explain/cache/<key>.json``, at most
CACHE_FILES files and CACHE_BYTES bytes per run, the oldest go first): the same question
about the same run is answered from the file without a process and without using compute.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from brickdnn.training import inference

from . import budgets
from . import training_service as training
from .budgets import Actor
from .models import TrainingRun

logger = logging.getLogger(__name__)

# What each method needs: a sample, the run's data, a neural network, which kinds of runs.
METHODS: dict[str, dict[str, Any]] = {
    "samples": {
        "label": "Validation samples",
        "sample": False,
        "kinds": ("image", "tabular", "demo"),
    },
    "image": {"label": "Saliency, Grad-CAM, occlusion", "sample": True, "kinds": ("image",)},
    "tabular": {"label": "Feature attributions", "sample": True, "kinds": ("tabular", "demo")},
    "permutation": {
        "label": "Permutation importance",
        "sample": False,
        "kinds": ("tabular", "demo"),
    },
    "boundary": {"label": "Decision boundary", "sample": False, "kinds": ("tabular", "demo")},
    "activations": {
        "label": "Activations",
        "sample": True,
        "kinds": ("image", "tabular", "demo"),
    },
    "weights": {
        "label": "Filters and weights",
        "sample": False,
        "kinds": ("image", "tabular", "demo"),
    },
}
NEURAL_ONLY = ("image", "activations", "weights")
# Classical models trained on demo data have reproducible demo rows; networks do not.
DATA_METHODS = ("samples", "tabular", "permutation", "boundary")
CACHE_DIR = Path("explain") / "cache"
CACHE_FILES = 64
CACHE_BYTES = 8 * 1024**2
# Seconds an explanation may take (like a prediction: it blocks a web server worker).
TIMEOUT_SECONDS = training.PREDICT_TIMEOUT_SECONDS


@dataclass
class Explanation:
    """A checked request, ready for :func:`run_explanation`."""

    run: TrainingRun
    request: dict[str, Any]
    key: str


def run_kind(run: TrainingRun) -> str | None:
    """``image``, ``tabular`` or ``demo`` (how the run's inputs were prepared), None if
    unknown."""
    preprocessing = (training.state_of(run) or {}).get("preprocessing")
    try:
        return inference.kind_of(run.plan, preprocessing)
    except inference.UnknownPreprocessing:
        return None


def available(run: TrainingRun) -> dict[str, Any]:
    """Which methods work for ``run`` and why the others do not (cheap: no process)."""
    family = training.run_family(run)
    kind = run_kind(run)
    reason = training.unavailable_reason(family)
    if reason is None and training.weights_path(run) is None:
        reason = training.no_weights_reason(run)
    plan = run.plan if isinstance(run.plan, dict) else {}
    targets = plan.get("targets") or []
    inputs = plan.get("inputs") or []
    features = (
        int(inputs[0]["shape"][0])
        if len(inputs) == 1 and len(inputs[0].get("shape") or []) == 1
        else None
    )
    methods = []
    for name, spec in METHODS.items():
        why = _why_not(name, spec, family, kind, targets, features)
        methods.append(
            {"id": name, "label": spec["label"], "available": why is None, "reason": why}
        )
    return {
        "family": family,
        "kind": kind,
        "unavailable": reason,
        "methods": methods,
        "demo": kind == "demo",
    }


def _why_not(
    name: str,
    spec: dict[str, Any],
    family: str,
    kind: str | None,
    targets: list[Any],
    features: int | None,
) -> str | None:
    if family == "classical" and name in NEURAL_ONLY:
        return "A classical model (scikit-learn) has no layers to look into."
    if family == "hybrid" and (name == "image" or (name == "samples" and kind == "image")):
        return (
            "Not for hybrid networks yet: their prediction comes from the classical model "
            "after the neural layers, which the network's gradients and scores do not reach."
        )
    if kind is None:
        if name == "weights":
            return None
        return "How the inputs of this run were prepared is not known."
    if kind not in spec["kinds"]:
        return {
            "image": "For image models only.",
            "tabular": "For models trained on a CSV file only.",
            "boundary": "For models trained on a CSV file only.",
            "permutation": "For models trained on a CSV file only.",
        }.get(name, "Not for this kind of run.")
    if kind == "demo" and family != "classical" and name in DATA_METHODS:
        return "Trained on random demo data, which is not kept: train on a dataset to explain it."
    one = len(targets) == 1 and targets[0].get("kind") in ("class", "value")
    if name in ("image", "tabular", "permutation", "boundary") and not one:
        return "For models with one output only."
    if name == "image" and targets and targets[0].get("kind") != "class":
        return "For classification networks only."
    if name == "boundary" and features != 2:
        return "For models with exactly 2 input features."
    return None


def prepare(run: TrainingRun, body: Any) -> Explanation:
    """Check a request ``{"method", "sample"?: n, "input"?: ..., "input_scale"?, "target"?,
    "layer"?, "node"?}``. Raises :class:`training.TrainingUnavailable`,
    :class:`training.PredictionUnavailable` or :class:`training.TrainingInputError`."""
    body = body if isinstance(body, dict) else {}
    family = training.run_family(run)
    reason = training.unavailable_reason(family)
    if reason:
        raise training.TrainingUnavailable(reason)
    if training.weights_path(run) is None:
        raise training.PredictionUnavailable(training.no_weights_reason(run))
    method = body.get("method")
    if method not in METHODS:
        raise training.TrainingInputError(f"'method' must be one of: {', '.join(METHODS)}.")
    info = available(run)
    entry = next(m for m in info["methods"] if m["id"] == method)
    if not entry["available"]:
        raise training.PredictionUnavailable(entry["reason"])
    preprocessing = (training.state_of(run) or {}).get("preprocessing")
    request: dict[str, Any] = {
        "method": method,
        "plan": run.plan,
        "preprocessing": preprocessing if isinstance(preprocessing, dict) else None,
    }
    if METHODS[method]["sample"]:
        request["sample"] = _sample(run, preprocessing, body, method, info["kind"])
    for name in ("target", "seed"):
        value = body.get(name)
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 10**6:
                raise training.TrainingInputError(f"'{name}' must be a whole number.")
            request[name] = value
    for name in ("layer", "node"):
        value = body.get(name)
        if value is not None:
            if not isinstance(value, str) or not value or len(value) > 200:
                raise training.TrainingInputError(f"'{name}' must be the id of a brick.")
            request[name] = value
    key = hashlib.sha256(
        json.dumps(
            {k: v for k, v in request.items() if k not in ("plan", "preprocessing")},
            sort_keys=True,
        ).encode()
    ).hexdigest()[:32]
    return Explanation(run, request, key)


def _sample(
    run: TrainingRun, preprocessing: Any, body: dict[str, Any], method: str, kind: str | None
) -> dict[str, Any] | None:
    index, raw = body.get("sample"), body.get("input")
    if raw is not None:
        try:
            prepared = inference.prepare_inputs(
                run.plan, preprocessing, [raw], body.get("input_scale")
            )
        except inference.UnknownPreprocessing as exc:
            raise training.PredictionUnavailable(str(exc)) from None
        except inference.InvalidInputs as exc:
            raise training.TrainingInputError(str(exc)) from None
        return {"tensor": prepared.tensors[0]}
    if index is None:
        if method == "activations" and kind == "demo":
            return None  # a random input
        raise training.TrainingInputError(
            "Pick a sample: 'sample' (the number of a validation sample, see the method "
            "'samples') or 'input' (one sample like for a prediction)."
        )
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < 10_000:
        raise training.TrainingInputError("'sample' must be the number of a validation sample.")
    return {"index": index}


def run_explanation(explanation: Explanation, actor: Actor) -> dict[str, Any]:
    """The answer, from the cache or from a new process (whose CPU time counts against
    ``actor``'s budgets). Raises :class:`budgets.BudgetExceeded`,
    :class:`training.TrainingBusy` or :class:`training.PredictionFailed`."""
    run = explanation.run
    cached = _cached(run, explanation.key)
    if cached is not None:
        return {**cached, "cached": True}
    budgets.check_compute(actor)
    with training._prediction_slot():
        answer = _process(run, {"explain": explanation.request}, actor)
    answer.pop("cpu_seconds", None)
    answer["run"] = {"id": str(run.pk), "name": run.name}
    _store(run, explanation.key, answer)
    return {**answer, "cached": False}


def needs_process(explanation: Explanation) -> bool:
    """False if the answer is cached (it then counts against no rate limit)."""
    return _cache_path(explanation.run, explanation.key).is_file() is False


# --------------------------------------------------------------------------- #
# The process
# --------------------------------------------------------------------------- #


def _process(run: TrainingRun, request: dict[str, Any], actor: Actor) -> dict[str, Any]:
    directory = training.run_dir(run.pk)
    command = [*training.predict_command(), str(directory)]
    started = time.monotonic()
    try:
        result = subprocess.run(
            command,
            input=json.dumps(request),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=TIMEOUT_SECONDS,
            cwd=directory,
            env=training._subprocess_env(),
        )
    except subprocess.TimeoutExpired:
        threads = max(1, min(training.PREDICT_THREADS, os.cpu_count() or 1))
        budgets.record_prediction(actor, run, (time.monotonic() - started) * threads)
        raise training.PredictionFailed(
            f"The explanation took longer than {TIMEOUT_SECONDS} seconds and was stopped."
        ) from None
    except OSError:
        logger.exception("Could not start the explanation process %s", command)
        raise training.PredictionFailed(
            "The explanation process could not be started. See the server log for details."
        ) from None
    answer = None
    for line in reversed(result.stdout.strip().splitlines()):
        if line.strip().startswith("{"):
            with contextlib.suppress(ValueError):
                answer = json.loads(line)
            break
    spent = answer.get("cpu_seconds") if isinstance(answer, dict) else None
    if isinstance(spent, bool) or not isinstance(spent, (int, float)) or spent < 0:
        spent = time.monotonic() - started
    budgets.record_prediction(actor, run, float(spent))
    if isinstance(answer, dict) and isinstance(answer.get("error"), str):
        raise training.PredictionFailed(answer["error"][:500])
    if result.returncode == 0 and isinstance(answer, dict) and "method" in answer:
        return answer
    lines = [line.strip() for line in result.stderr.splitlines() if line.strip()]
    detail = lines[-1][:300] if lines else f"exit code {result.returncode}"
    logger.warning("Explanation for run %s failed: %s", run.pk, result.stderr[-2000:])
    raise training.PredictionFailed(f"The explanation failed: {detail}")


# --------------------------------------------------------------------------- #
# The cache
# --------------------------------------------------------------------------- #


def _cache_path(run: TrainingRun, key: str) -> Path:
    return training.run_dir(run.pk) / CACHE_DIR / f"{key}.json"


def _cached(run: TrainingRun, key: str) -> dict[str, Any] | None:
    path = _cache_path(run, key)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    with contextlib.suppress(OSError):
        os.utime(path)  # recently used: kept longer
    return data


def _store(run: TrainingRun, key: str, answer: dict[str, Any]) -> None:
    path = _cache_path(run, key)
    text = json.dumps(answer, separators=(",", ":"))
    if len(text) > CACHE_BYTES // 4:
        return  # too large to keep
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
        _trim(path.parent)
    except OSError:
        logger.warning("Could not cache an explanation of run %s", run.pk, exc_info=True)


def _trim(directory: Path) -> None:
    """Remove the least recently used answers beyond CACHE_FILES / CACHE_BYTES."""
    files = []
    for item in directory.glob("*.json"):
        with contextlib.suppress(OSError):
            stat = item.stat()
            files.append((stat.st_mtime, stat.st_size, item))
    files.sort(reverse=True)
    total = 0
    for index, (_, size, item) in enumerate(files):
        total += size
        if index >= CACHE_FILES or total > CACHE_BYTES:
            with contextlib.suppress(OSError):
                item.unlink()
