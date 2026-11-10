"""Compact answers of the MCP server for datasets, training runs, layers and examples.

Tool results go into the context of an AI assistant, so the defaults are summaries: no
column lists of every dataset, no per-epoch history, no graphs. Arguments ask for more
(``include``, paging), and every capped list says how long it is. The REST API and the web
app keep their full answers.
"""

from __future__ import annotations

from typing import Any

from .models import Dataset, TrainingRun

DEFAULT_COLUMNS = 50
MAX_COLUMNS = 500
MAX_CLASS_NAMES = 50
MAX_HISTORY = 50
DEFAULT_LOG_LINES = 20
MAX_LOG_LINES = 200
FAILED_LOG_LINES = 10
MAX_TRIALS = 50
MAX_TEXT = 500
RUN_INCLUDES = ("history", "graph", "plan", "log", "trials", "preprocessing")


class CompactError(ValueError):
    """Invalid arguments; the message says how to fix them."""


def _short(value: Any) -> Any:
    if isinstance(value, str) and len(value) > MAX_TEXT:
        return value[: MAX_TEXT - 1] + "…"
    return value


def _no_none(data: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in data.items() if v is not None}


def capped_names(names: list[Any] | None, limit: int = MAX_CLASS_NAMES) -> list[Any] | None:
    if names is None:
        return None
    names = list(names)
    if len(names) <= limit:
        return names
    return [*names[:limit], f"… and {len(names) - limit:,} more"]


def _whole(value: Any, name: str, default: int, low: int, high: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise CompactError(f"'{name}' must be a whole number from {low} to {high}.")
    return value


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #


def _task(info: dict[str, Any], dataset: Dataset) -> str | None:
    if dataset.kind == "images":
        return "classification"
    if dataset.prepared_from_id is not None or info.get("predefined_split"):
        return "classification" if info.get("class_names") else "regression"
    kind = info.get("target_kind")
    if kind == "text":
        return "classification"
    if kind == "number":
        return "regression"
    if kind == "integer":
        return "classification or regression (chosen by the outputs of the network)"
    return None


def dataset_summary(dataset: Dataset, user) -> dict[str, Any]:
    """What it takes to choose a dataset: no column lists."""
    info = dataset.info if isinstance(dataset.info, dict) else {}
    columns = info.get("columns") or []
    classes = info.get("class_names")
    summary = {
        "id": dataset.pk,
        "name": dataset.name,
        "kind": dataset.kind,
        "owner": dataset.owner.get_username(),
        "mine": dataset.owner_id == user.pk,
        "rows": info.get("num_samples"),
        "features": info.get("num_features"),
        "image_shape": info.get("sample_shape") if dataset.kind == "images" else None,
        "target": str(columns[-1]) if dataset.kind == "csv" and columns else None,
        "task": _task(info, dataset),
        "classes": len(classes) if classes else None,
        "prepared": dataset.prepared_from_id is not None,
        "size_bytes": dataset.size_bytes,
        "reference": {"kind": "upload", "id": dataset.pk},
    }
    return _no_none(summary)


def dataset_detail(dataset: Dataset, user, limit: Any = None, offset: Any = None) -> dict:
    """One dataset with a page of its columns and its class names (capped)."""
    from .shared_data import shared_in

    limit = _whole(limit, "columns_limit", DEFAULT_COLUMNS, 0, MAX_COLUMNS)
    info = dataset.info if isinstance(dataset.info, dict) else {}
    columns = [str(c) for c in info.get("columns") or []]
    offset = _whole(offset, "columns_offset", 0, 0, max(len(columns), 0))
    detail = dataset_summary(dataset, user)
    page = columns[offset : offset + limit]
    detail.update(
        created_at=dataset.created_at.isoformat(),
        shared_in=shared_in(user, dataset),
        prepared_from=(
            {"id": dataset.prepared_from_id, "name": dataset.prepared_from.name}
            if dataset.prepared_from_id is not None
            else None
        ),
        n_columns=len(columns),
        columns=page,
        columns_offset=offset,
    )
    if dataset.kind == "csv" and columns:
        detail["columns_note"] = "The features, then the target as the last column."
    if offset + len(page) < len(columns):
        detail["more_columns"] = (
            f"Columns {offset + 1}–{offset + len(page)} of {len(columns):,}; pass "
            f"columns_offset={offset + len(page)} for the next ones."
        )
    detail["class_names"] = capped_names(info.get("class_names"))
    for key in ("target_kind", "target_range", "predefined_split", "delimiter", "has_header"):
        detail[key] = info.get(key)
    if dataset.kind == "images":
        detail["class_counts"] = capped_names(info.get("class_counts"))
    return _no_none(detail)


# --------------------------------------------------------------------------- #
# Training runs
# --------------------------------------------------------------------------- #


def _family(run: TrainingRun) -> str:
    plan = run.plan if isinstance(run.plan, dict) else {}
    return "classical" if isinstance(plan.get("classical"), dict) else "neural"


def _dataset_name(run: TrainingRun) -> Any:
    data = run.dataset if isinstance(run.dataset, dict) else {}
    return _no_none({"kind": data.get("kind"), "id": data.get("id"), "name": data.get("name")})


def _epochs(run: TrainingRun) -> int | None:
    if _family(run) == "classical":
        return 1
    return run.config.get("epochs") if isinstance(run.config, dict) else None


def _metrics(values: Any) -> dict[str, Any] | None:
    if not isinstance(values, dict):
        return None
    kept = {k: v for k, v in values.items() if v is not None}
    return kept or None


def _usage(run: TrainingRun) -> dict[str, Any] | None:
    """The CPU minutes a run used, the most it may use, the estimate before it started and
    why it stopped (user, budget, run_limit): see budgets.py."""

    def minutes(seconds: float | None) -> float | None:
        return round(seconds / 60, 2) if seconds is not None else None

    usage = _no_none(
        {
            "cpu_minutes": minutes(run.cpu_seconds or 0.0),
            "limit_cpu_minutes": minutes(run.cpu_limit_seconds),
            "estimated_cpu_minutes": minutes(run.estimated_cpu_seconds),
            "stop_reason": run.stop_reason or None,
        }
    )
    return usage or None


def _started_via(run: TrainingRun) -> str | None:
    """The name of the app or API token that started the run (None: the web app)."""
    if run.connection_id is not None:
        return run.connection.client.name if run.connection else None
    if run.api_token_id is not None:
        return run.api_token.name if run.api_token else None
    return None


def queue_info(run: TrainingRun, context: dict[str, Any], full: bool = False) -> dict | None:
    """Where a queued run waits: its position among the waiting trainings and a rough
    estimate of the minutes until it starts. ``context``: shared by the runs of one answer,
    so the queue is looked at once."""
    if run.status != "queued":
        return None
    from . import training_queue

    info = training_queue.run_info(run, training_queue.snapshot_in(context))
    if info is None:
        return None
    seconds = info["estimated_wait_seconds"]
    brief = {
        "position": info["position"],
        "waiting": info["waiting"],
        "estimated_wait_minutes": round(seconds / 60, 1) if seconds is not None else None,
        "worker_online": None if info["worker_online"] else False,
    }
    if full:
        brief["estimated_start_at"] = info["estimated_start_at"]
        if info["blocked_by"] == "account":
            brief["note"] = (
                "Waits until a running training of this account ends (the plan's trainings "
                "at a time)."
            )
        elif not info["worker_online"]:
            brief["note"] = training_queue.NO_WORKER_MESSAGE
        elif seconds is not None:
            brief["note"] = "The wait is a rough estimate."
        brief["priority_hint"] = info["hint"]
    return _no_none(brief)


def run_summary(run: TrainingRun, context: dict[str, Any] | None = None) -> dict[str, Any]:
    """One line of list_training_runs."""
    plan = run.plan if isinstance(run.plan, dict) else {}
    summary = {
        "id": str(run.pk),
        "name": run.name,
        "status": run.status,
        "queue": queue_info(run, context if context is not None else {}),
        "family": _family(run),
        "task": plan.get("task"),
        "dataset": _dataset_name(run).get("name"),
        "epochs": f"{run.epochs_done}/{_epochs(run)}" if _epochs(run) else run.epochs_done,
        "metrics": _metrics(run.metrics),
        "trials": run.trials_done or None,
        "has_weights": run.has_weights,
        "started_by": run.owner.get_username(),
        "project": run.project_id,
        "created_at": run.created_at.isoformat(),
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
    }
    return _no_none(summary)


def run_includes(value: Any) -> set[str]:
    if value is None:
        return set()
    if not isinstance(value, list) or any(v not in RUN_INCLUDES for v in value):
        raise CompactError(f"'include' must be a list of: {', '.join(RUN_INCLUDES)}.")
    return set(value)


def _history(history: list[Any], keep: set[Any]) -> tuple[list[dict[str, Any]], str | None]:
    """At most MAX_HISTORY evenly spaced epochs, plus the best and the last."""
    entries = [e for e in history if isinstance(e, dict)]
    if len(entries) <= MAX_HISTORY:
        return entries, None
    step = (len(entries) - 1) / (MAX_HISTORY - 1)
    chosen = {round(i * step) for i in range(MAX_HISTORY)}
    chosen |= {i for i, e in enumerate(entries) if e.get("epoch") in keep}
    chosen.add(len(entries) - 1)
    shown = [entries[i] for i in sorted(chosen)]
    return shown, (
        f"{len(shown)} of {len(entries)} epochs, evenly spaced, with the best and the last."
    )


def run_detail(
    run: TrainingRun,
    state: dict[str, Any] | None,
    includes: set[str],
    log_lines: Any = None,
) -> dict[str, Any]:
    """A training run as a summary, with the parts in ``includes`` in full (capped)."""
    log_lines = _whole(log_lines, "log_lines", DEFAULT_LOG_LINES, 1, MAX_LOG_LINES)
    state = state or {}
    plan = run.plan if isinstance(run.plan, dict) else {}
    config = run.config if isinstance(run.config, dict) else {}
    started, finished = state.get("started_at"), state.get("finished_at")
    detail: dict[str, Any] = {
        "id": str(run.pk),
        "name": run.name,
        "status": run.status,
        "message": run.message,
        "family": _family(run),
        "task": plan.get("task"),
        "plan": _short(plan.get("summary")),
        "dataset": _dataset_name(run),
        "config": {k: v for k, v in config.items() if v is not None},
        "epochs": {"done": run.epochs_done, "total": _epochs(run)},
        "progress": state.get("progress") if run.status in ("running", "preparing") else None,
        "best": run.best or state.get("best"),
        "metrics": _metrics(run.metrics),
        "test": _metrics(state.get("test")),
        "weights_epoch": state.get("weights_epoch"),
        "stopped_early": state.get("stopped_early") or None,
        "rows": _no_none(
            {
                "train": state.get("train_size"),
                "validation": state.get("val_size"),
                "test": state.get("test_size"),
            }
        )
        or None,
        "class_names": capped_names(state.get("class_names") or plan.get("class_names")),
        "total_params": run.total_params,
        "device": state.get("device_label") or state.get("device"),
        "usage": _usage(run),
        "queue": queue_info(run, {}, full=True),
        "started_via": _started_via(run),
        "has_weights": run.has_weights,
        "started_by": run.owner.get_username(),
        "project": run.project_id,
        "created_at": run.created_at.isoformat(),
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        "seconds": round(finished - started, 1)
        if isinstance(started, (int, float)) and isinstance(finished, (int, float))
        else None,
    }
    if isinstance(state.get("classical"), dict):
        detail["classical"] = {k: _short(v) for k, v in state["classical"].items()}
    if run.tuning or state.get("trials"):
        detail["tuning"] = _no_none(
            {
                "trials_total": state.get("trials_total"),
                "trials_done": run.trials_done,
                "best_trial": run.best_trial,
            }
        )
    notes = [n for n in plan.get("notes") or [] if isinstance(n, str)]
    if notes:
        detail["notes"] = capped_names([_short(n) for n in notes], 5)
    log = [str(line) for line in state.get("log") or []]
    if "log" in includes:
        detail["log"] = log[-log_lines:]
        detail["log_note"] = f"The last {len(detail['log'])} of {len(log)} lines kept."
    elif run.status == "failed" and log:
        detail["log_tail"] = log[-FAILED_LOG_LINES:]
    if "history" in includes:
        best = run.best.get("epoch") if isinstance(run.best, dict) else None
        detail["history"], note = _history(
            state.get("history") or [], {best, state.get("weights_epoch")}
        )
        if note:
            detail["history_note"] = note
    if "trials" in includes:
        trials = [t for t in state.get("trials") or [] if isinstance(t, dict)]
        detail["trials"] = trials[:MAX_TRIALS]
        if len(trials) > MAX_TRIALS:
            detail["trials_note"] = f"The first {MAX_TRIALS} of {len(trials)} trials."
    if "graph" in includes:
        detail["graph"] = run.graph
    if "plan" in includes:
        detail["plan_details"] = {
            **plan,
            "class_names": capped_names(plan.get("class_names")),
        }
    if "preprocessing" in includes and isinstance(state.get("preprocessing"), dict):
        preprocessing = dict(state["preprocessing"])
        for key in ("columns", "mean", "std", "class_names"):
            if isinstance(preprocessing.get(key), list):
                preprocessing[key] = capped_names(preprocessing[key], MAX_COLUMNS)
        detail["preprocessing"] = preprocessing
    missing = [part for part in RUN_INCLUDES if part not in includes]
    if missing:
        detail["more"] = f"include: {missing} adds these parts (history is downsampled)."
    return _no_none(detail)


# --------------------------------------------------------------------------- #
# Layers and examples
# --------------------------------------------------------------------------- #


def _param_line(param: dict[str, Any]) -> str:
    kind = param.get("kind")
    options = [o.get("value") if isinstance(o, dict) else o for o in param.get("options") or []]
    text = "one of " + " | ".join(map(str, options)) if options else str(kind)
    if param.get("length"):
        text += f" (list of {param['length']})"
    low, high = param.get("min"), param.get("max")
    if low is not None and high is not None:
        text += f" from {low:g} to {high:g}"
    elif low is not None:
        text += f" ≥ {low:g}"
    if param.get("auto"):
        text += "; null = inferred"
    default = param.get("default")
    if default not in (None, ""):
        text += (
            f"; default {default!r}"
            if not isinstance(default, bool)
            else f"; default {str(default).lower()}"
        )
    if param.get("show_if"):
        text += f"; only if {param['show_if']}"
    return text


def layer_catalog(catalog: dict[str, Any], types: Any = None) -> dict[str, Any]:
    """All layer types with one line per parameter, or the full specs of ``types``."""
    layers = catalog.get("layers") or []
    if types is not None:
        if not isinstance(types, list) or not all(isinstance(t, str) for t in types):
            raise CompactError("'types' must be a list of layer types, e.g. ['Conv2d'].")
        known = {layer["type"]: layer for layer in layers}
        unknown = [t for t in types if t not in known]
        if unknown:
            raise CompactError(f"Unknown layer type '{unknown[0]}'. The types: {', '.join(known)}.")
        return {"layers": [known[t] for t in types]}
    compact = []
    for layer in layers:
        entry = {
            "type": layer.get("type"),
            "category": layer.get("category"),
            "family": layer.get("family"),
            "description": layer.get("description"),
            "params": {p["name"]: _param_line(p) for p in layer.get("params") or []},
        }
        if layer.get("output_kind") != "tensor":
            entry["output_kind"] = layer.get("output_kind")
        slots = layer.get("input_slots") or []
        if len(slots) > 1 or any(s.get("kind") != "tensor" for s in slots):
            entry["input_slots"] = [
                f"{s.get('name')} ({s.get('kind')}, {s.get('min')}–{s.get('max') or 'any'})"
                for s in slots
            ]
        elif layer.get("max_inputs") != 1:
            entry["inputs"] = f"{layer.get('min_inputs')}–{layer.get('max_inputs') or 'any'}"
        compact.append(entry)
    return {
        "categories": catalog.get("categories"),
        "layers": compact,
        "details": "types: ['Conv2d', …] gives the full specs with help texts.",
    }


def examples(found: list[dict[str, Any]], example_id: Any = None) -> Any:
    """Names and descriptions of the examples, or one example with its graph."""
    if example_id is None:
        return {
            "examples": [
                {"id": e.get("id"), "name": e.get("name"), "description": e.get("description")}
                for e in found
            ],
            "details": "id gives the example with its graph.",
        }
    for example in found:
        if example.get("id") == example_id:
            return example
    known = ", ".join(str(e.get("id")) for e in found)
    raise CompactError(f"Unknown example '{example_id}'. The examples: {known}.")


# --------------------------------------------------------------------------- #
# Predictions
# --------------------------------------------------------------------------- #

MAX_WARNINGS = 10


def _compact_prediction(item: Any) -> Any:
    """A prediction without the class indices (the class names say the same)."""
    if not isinstance(item, dict):
        return item
    if isinstance(item.get("outputs"), list):
        return {**item, "outputs": [_compact_prediction(o) for o in item["outputs"]]}
    if "top" not in item:
        return item
    top = [
        {"class": t.get("class"), "probability": t.get("probability")}
        for t in item.get("top") or []
        if isinstance(t, dict)
    ]
    rest = {k: v for k, v in item.items() if k not in ("index", "top")}
    return {**rest, "top": top} if len(top) > 1 else rest


def prediction_answer(answer: dict[str, Any]) -> dict[str, Any]:
    """The answer of ``predict``: without class indices, at most MAX_WARNINGS warnings."""
    result = dict(answer)
    result["predictions"] = [_compact_prediction(p) for p in answer.get("predictions") or []]
    warnings = answer.get("warnings")
    if isinstance(warnings, list) and len(warnings) > MAX_WARNINGS:
        more = len(warnings) - MAX_WARNINGS
        result["warnings"] = [*warnings[:MAX_WARNINGS], f"… and {more:,} more"]
    return result
