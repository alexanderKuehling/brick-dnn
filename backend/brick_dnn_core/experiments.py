"""The experiment tracker: runs side by side.

- **The runs table** (:func:`table`): every run the user can see (their own and those of
  their shared networks) as one row each, with the main settings and the headline numbers
  of the run, filtered, sorted and paged on the server. The numbers of a finished run come
  from its state once and are kept in ``TrainingRun.summary`` (:func:`summary_of`).
- **Curves** (:func:`curves`): the training and validation curves of up to
  :data:`MAX_COMPARE` runs, for one chart.
- **Tags, a note and a star** per run (:func:`annotate`): the owner of a run and the
  editors (and owner) of its network change them.
- **Shared comparisons** (:class:`RunComparison`): a read-only link like the view link of a
  network (a random token, deleted to revoke it) that shows the chosen runs' table, curves
  and scatter plot, without who started them, their networks or (unless chosen) their tags
  and notes. Only runs the user started, or runs of networks they own, can be shared.
"""

from __future__ import annotations

import math
import re
import secrets
import unicodedata
import uuid
from collections import Counter
from datetime import datetime
from typing import Any

from django.core.exceptions import ValidationError
from django.utils import timezone
from django.utils.translation import gettext

from brickdnn.training import ACTIVE_STATUSES, TERMINAL_STATUSES

from .models import Project, ProjectMember, RunComparison, TrainingRun, UserProfile

# Runs in one comparison (the overlay chart and a shared link).
MAX_COMPARE = 10
# Rows of one page of the table, by default and at most.
PAGE_SIZE = 50
MAX_PAGE_SIZE = 200
# The newest runs the table looks at (filters and sorting work on these).
MAX_TABLE_RUNS = 5000
# Points of one curve sent to the browser; longer curves are thinned out evenly.
MAX_CURVE_POINTS = 2000
# Tags and notes.
MAX_TAGS = 20
MAX_TAG_LENGTH = 40
MAX_NOTE_LENGTH = 2000
# Columns of the table a user can remember, and shared comparisons per user.
MAX_COLUMNS = 60
MAX_COLUMN_LENGTH = 80
MAX_COMPARISONS = 100
COLUMN_NAME = re.compile(r"^[A-Za-z0-9_.:-]{1,80}$")

# Settings of ``config`` shown as columns (all numbers or short words).
SETTING_KEYS = (
    "optimizer",
    "learning_rate",
    "batch_size",
    "epochs",
    "weight_decay",
    "loss",
    "val_split",
    "test_split",
    "patience",
    "seed",
    "max_samples",
    "label_smoothing",
    "focal_gamma",
    "huber_delta",
    "feature_training",
)
# The numbers of state.evaluation per split (val_*, test_*): classification and regression.
EVALUATION_KEYS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "log_loss",
    "mae",
    "mse",
    "rmse",
    "r2",
    "mape",
    "median_absolute_error",
    "max_error",
)
# What one point of a curve may carry (state.json history; a run logged elsewhere adds
# its own metrics).
CURVE_KEYS = (
    "train_loss",
    "val_loss",
    "train_accuracy",
    "val_accuracy",
    "val_rmse",
    "val_r2",
    "val_iou",
    "val_dice",
    "val_pixel_accuracy",
)
SORT_FIELDS = ("name", "status", "created_at", "duration", "dataset", "model", "kind")


class ExperimentInputError(ValueError):
    """A request the user can fix (HTTP 400)."""


# --------------------------------------------------------------------------- #
# Numbers of a run
# --------------------------------------------------------------------------- #


def _number(value: Any) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) else None


def _numbers(values: Any) -> dict[str, float | int]:
    if not isinstance(values, dict):
        return {}
    found = {}
    for key, value in values.items():
        number = _number(value)
        if number is not None and isinstance(key, str):
            found[key] = number
    return found


def summary_of(run: TrainingRun, state: dict[str, Any] | None) -> dict[str, Any]:
    """The headline numbers of a run as one flat dict: the metrics of its kept epoch
    (``val_loss``, ``test_accuracy`` …), the final evaluation of the kept weights per split
    (``val_macro_f1``, ``test_pr_auc``, ``test_r2`` …), a segmentation's IoU, Dice and
    pixel accuracy, and ``best_epoch``. Keys without a value are left out."""
    found: dict[str, Any] = dict(_numbers(run.metrics))
    state = state if isinstance(state, dict) else {}
    evaluation = state.get("evaluation")
    if isinstance(evaluation, dict):
        for split in ("val", "test"):
            part = evaluation.get(split)
            if not isinstance(part, dict):
                continue
            for key in EVALUATION_KEYS:
                number = _number(part.get(key))
                if number is not None:
                    found.setdefault(f"{split}_{key}", number)
            curve = part.get("threshold_curve")
            if isinstance(curve, dict):
                for key in ("roc_auc", "pr_auc"):
                    number = _number(curve.get(key))
                    if number is not None:
                        found[f"{split}_{key}"] = number
    segmentation = state.get("segmentation")
    if isinstance(segmentation, dict):
        for split, prefix in (("validation", "val"), ("test", "test")):
            part = segmentation.get(split)
            if isinstance(part, dict):
                for key in ("iou", "dice", "pixel_accuracy"):
                    number = _number(part.get(key))
                    if number is not None:
                        found.setdefault(f"{prefix}_{key}", number)
    kept = state.get("weights_epoch")
    if isinstance(kept, int) and not isinstance(kept, bool):
        found["best_epoch"] = kept
    if run.kind == "external":
        external = run.plan.get("external") if isinstance(run.plan, dict) else None
        if isinstance(external, dict):
            found.update(_numbers(external.get("summary")))  # what the script reported
    return found


def ensure_summaries(runs: list[TrainingRun]) -> None:
    """Fill ``summary`` of finished runs that have none yet (once per run: their state is
    read from disk) and of active runs (from their metrics, not stored)."""
    from .training_service import state_of

    for run in runs:
        if run.status in TERMINAL_STATUSES and run.summary is None:
            try:
                state = state_of(run)
            except (OSError, ValueError):
                state = None
            run.summary = summary_of(run, state)
            TrainingRun.objects.filter(pk=run.pk).update(summary=run.summary)
        elif run.status in ACTIVE_STATUSES:
            run.summary = _numbers(run.metrics)


# --------------------------------------------------------------------------- #
# Rows of the table
# --------------------------------------------------------------------------- #


def editable_project_ids(user) -> set[int]:
    """Networks whose runs ``user`` may annotate: the ones they own or edit."""
    if user is None or not user.is_authenticated:
        return set()
    owned = Project.objects.filter(owner=user).values_list("pk", flat=True)
    edited = ProjectMember.objects.filter(user=user, role="editor").values_list(
        "project_id", flat=True
    )
    return set(owned) | set(edited)


def can_annotate(user, run: TrainingRun, editable: set[int] | None = None) -> bool:
    """Tags, note and star: who started the run, and the owner and editors of its network."""
    if user is None or not user.is_authenticated:
        return False
    if run.owner_id == user.pk:
        return True
    if run.project_id is None:
        return False
    if editable is None:
        editable = editable_project_ids(user)
    return run.project_id in editable


def _duration(run: TrainingRun, now: datetime) -> float | None:
    start = run.started_at or run.created_at
    if start is None:
        return None
    if run.status in TERMINAL_STATUSES:
        if run.finished_at is None:
            return None
        end = run.finished_at
    elif run.status == "queued":
        return None
    else:
        end = now
    return max(0.0, round((end - start).total_seconds(), 1))


def model_label(run: TrainingRun) -> str:
    """'Neural network', the classical model ('HistGradientBoostingClassifier'), a hybrid
    ('SVC on learned features') or 'Ensemble'."""
    if run.kind == "ensemble":
        return "Ensemble"
    if run.kind == "external" and not (isinstance(run.graph, dict) and run.graph.get("nodes")):
        return "Script"
    from .ensembles import _model_label

    return _model_label(run) or "Neural network"


def dataset_key(run: TrainingRun) -> str:
    data = run.dataset if isinstance(run.dataset, dict) else {}
    kind = str(data.get("kind") or "")
    ident = data.get("id")
    if ident is None:
        return f"{kind}:{data.get('name') or ''}"[:200]
    return f"{kind}:{ident}"[:200]


def _settings(run: TrainingRun) -> dict[str, Any]:
    config = run.config if isinstance(run.config, dict) else {}
    found = {}
    for key in SETTING_KEYS:
        value = config.get(key)
        if isinstance(value, str):
            found[key] = value[:60]
        elif _number(value) is not None:
            found[key] = value
    if run.kind == "ensemble":
        method = config.get("method")
        if isinstance(method, str):
            found["method"] = method
    if isinstance(run.tuning, dict):
        found["search"] = str(run.tuning.get("method") or "random")
        trials = _number(run.tuning.get("trials"))
        if trials is not None:
            found["trials"] = trials
        metric = run.tuning.get("metric")
        if isinstance(metric, str):
            found["search_metric"] = metric
    return found


def _family(run: TrainingRun) -> str:
    from .training_service import run_family

    return run_family(run)


def _loss(run: TrainingRun) -> str | None:
    """The loss a run trained with (``auto`` became the default of the task)."""
    config = run.config if isinstance(run.config, dict) else {}
    loss = config.get("loss")
    task = run.plan.get("task") if isinstance(run.plan, dict) else None
    if loss in (None, "auto", "cross_entropy") and task == "regression":
        return "mse"  # runs of older versions say cross_entropy for every task
    if loss in (None, "auto"):
        return "cross_entropy" if task in ("classification", "segmentation") else None
    return loss if isinstance(loss, str) else None


def _external(run: TrainingRun) -> dict[str, Any] | None:
    """A run logged from elsewhere: its client, points and last report (null otherwise)."""
    from .tracking import describe

    return describe(run)


def run_row(
    run: TrainingRun,
    user,
    editable: set[int] | None = None,
    now: datetime | None = None,
    public: bool = False,
    include_notes: bool = True,
) -> dict[str, Any]:
    """One row of the runs table. ``public``: for a shared comparison, without who started
    the run and its network (and without tags and notes unless ``include_notes``)."""
    now = now or timezone.now()
    plan = run.plan if isinstance(run.plan, dict) else {}
    data = run.dataset if isinstance(run.dataset, dict) else {}
    family = _family(run)
    config = run.config if isinstance(run.config, dict) else {}
    epochs = 1 if family in ("classical", "ensemble") else config.get("epochs")
    row: dict[str, Any] = {
        "id": str(run.pk),
        "name": run.name,
        "status": run.status,
        "kind": run.kind,
        "family": family,
        "task": plan.get("task"),
        "model": model_label(run),
        "dataset": {"key": dataset_key(run), "kind": data.get("kind"), "name": data.get("name")},
        "created_at": run.created_at,
        "started_at": run.started_at,
        "finished_at": run.finished_at,
        "duration_seconds": _duration(run, now),
        "epochs": epochs if _number(epochs) is not None else None,
        "epochs_done": run.epochs_done,
        "trials_done": run.trials_done,
        "search": bool(run.tuning),
        "total_params": run.total_params,
        "loss": _loss(run),
        "settings": _settings(run),
        "metrics": dict(run.summary) if isinstance(run.summary, dict) else {},
        "starred": run.starred,
        "external": _external(run),
    }
    if public:
        if include_notes:
            row["tags"] = list(run.tags or [])
            row["note"] = run.note
        else:
            row["starred"] = False  # stars are annotations too
        return row
    row.update(
        {
            "tags": list(run.tags or []),
            "note": run.note,
            "project": run.project_id,
            "project_name": run.project.name if run.project_id and run.project else None,
            "started_by": run.owner.get_username(),
            "mine": user is not None and run.owner_id == user.pk,
            "can_annotate": can_annotate(user, run, editable),
        }
    )
    return row


# --------------------------------------------------------------------------- #
# The table: filters, sorting, pages
# --------------------------------------------------------------------------- #


def _int_param(params, name: str, default: int, low: int, high: int) -> int:
    raw = params.get(name)
    if raw in (None, ""):
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise ExperimentInputError(
            gettext("'%(name)s' must be a whole number.") % {"name": name}
        ) from None
    return min(high, max(low, value))


def _sort_value(row: dict[str, Any], field: str) -> Any:
    if field.startswith("metric:"):
        return _number(row["metrics"].get(field[7:]))
    if field.startswith("setting:"):
        return row["settings"].get(field[8:])
    if field == "duration":
        return row["duration_seconds"]
    if field == "dataset":
        return (row["dataset"].get("name") or "").lower()
    if field == "name":
        return row["name"].lower()
    if field == "model":
        return row["model"].lower()
    return row.get(field)


def _sorted(rows: list[dict[str, Any]], field: str, descending: bool) -> list[dict[str, Any]]:
    """Sorted by ``field``; rows without a value always come last. Numbers before words."""

    def key(row: dict[str, Any]) -> tuple:
        value = _sort_value(row, field)
        if isinstance(value, datetime):
            value = value.timestamp()
        if isinstance(value, bool):
            value = int(value)
        if isinstance(value, (int, float)):
            return (0, value, "")
        return (1, 0, str(value))

    present = [r for r in rows if _sort_value(r, field) not in (None, "")]
    missing = [r for r in rows if _sort_value(r, field) in (None, "")]
    return sorted(present, key=key, reverse=descending) + missing


def _facets(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tags: Counter[str] = Counter()
    datasets: dict[str, str] = {}
    projects: dict[int, str] = {}
    families: Counter[str] = Counter()
    for row in rows:
        tags.update(row.get("tags") or [])
        datasets.setdefault(row["dataset"]["key"], row["dataset"].get("name") or "")
        if row.get("project") is not None and row.get("project_name"):
            projects.setdefault(row["project"], row["project_name"])
        families["external" if row["kind"] == "external" else row["family"]] += 1
    return {
        "tags": [{"name": t, "count": n} for t, n in sorted(tags.items())],
        "datasets": [
            {"key": k, "name": v} for k, v in sorted(datasets.items(), key=lambda i: i[1])
        ],
        "projects": [{"id": k, "name": v} for k, v in sorted(projects.items(), key=lambda i: i[1])],
        "families": dict(families),
    }


def table(user, params) -> dict[str, Any]:
    """``GET /api/training/runs/table/``: the runs ``user`` can see, filtered, sorted and
    paged. Query parameters: ``project``, ``dataset`` (a key of ``facets.datasets``),
    ``status``, ``family`` (neural, classical, hybrid, ensemble, or external: logged from
    elsewhere), ``tag``,
    ``starred=true``, ``mine=true``, ``q`` (in the name, note or tags), ``ids`` (comma
    separated), ``sort`` (name, status, created_at, duration, dataset, model, kind,
    ``setting:<key>``, ``metric:<key>``), ``order`` (asc, desc), ``page``, ``page_size``."""
    from . import training_service as training
    from .shared_data import visible_runs

    runs_qs = visible_runs(user).select_related("owner", "project").order_by("-created_at")
    project = params.get("project")
    if project not in (None, ""):
        if not str(project).isdigit():
            raise ExperimentInputError(gettext("'project' must be the id of a network."))
        runs_qs = runs_qs.filter(project_id=int(project))
    status = params.get("status")
    if status not in (None, ""):
        runs_qs = runs_qs.filter(status=status)
    if params.get("mine") == "true":
        runs_qs = runs_qs.filter(owner=user)
    if params.get("starred") == "true":
        runs_qs = runs_qs.filter(starred=True)
    ids = params.get("ids")
    if ids not in (None, ""):
        runs_qs = runs_qs.filter(pk__in=_run_ids(str(ids).split(","), limit=MAX_PAGE_SIZE))
    runs = list(runs_qs[:MAX_TABLE_RUNS])
    training.sync_runs([run for run in runs if run.status in ACTIVE_STATUSES])
    ensure_summaries(runs)
    editable = editable_project_ids(user)
    now = timezone.now()
    rows = [run_row(run, user, editable, now) for run in runs]
    facets = _facets(rows)

    dataset = params.get("dataset")
    if dataset not in (None, ""):
        rows = [r for r in rows if r["dataset"]["key"] == dataset]
    family = params.get("family")
    if family not in (None, ""):
        if family == "external":
            rows = [r for r in rows if r["kind"] == "external"]
        else:
            rows = [r for r in rows if r["family"] == family and r["kind"] != "external"]
    tag = params.get("tag")
    if tag not in (None, ""):
        rows = [r for r in rows if tag in (r.get("tags") or [])]
    query = (params.get("q") or "").strip().lower()
    if query:
        rows = [
            r
            for r in rows
            if query in r["name"].lower()
            or query in (r.get("note") or "").lower()
            or any(query in t.lower() for t in r.get("tags") or [])
            or query in (r["dataset"].get("name") or "").lower()
        ]

    sort = params.get("sort") or "created_at"
    if sort not in SORT_FIELDS and not re.match(r"^(metric|setting):[A-Za-z0-9_./-]{1,64}$", sort):
        raise ExperimentInputError(
            gettext("Unknown sort column '%(sort)s'.") % {"sort": str(sort)[:80]}
        )
    order = params.get("order") or ("asc" if sort in ("name", "dataset", "model") else "desc")
    if order not in ("asc", "desc"):
        raise ExperimentInputError(gettext("'order' must be asc or desc."))
    rows = _sorted(rows, sort, order == "desc")

    page_size = _int_param(params, "page_size", PAGE_SIZE, 1, MAX_PAGE_SIZE)
    pages = max(1, math.ceil(len(rows) / page_size))
    page = _int_param(params, "page", 1, 1, pages)
    start = (page - 1) * page_size
    return {
        "count": len(rows),
        "page": page,
        "pages": pages,
        "page_size": page_size,
        "sort": sort,
        "order": order,
        "results": rows[start : start + page_size],
        "facets": facets,
        "truncated": len(runs) >= MAX_TABLE_RUNS,
        "limits": {"compare": MAX_COMPARE, "tags": MAX_TAGS, "note": MAX_NOTE_LENGTH},
    }


# --------------------------------------------------------------------------- #
# Curves of several runs
# --------------------------------------------------------------------------- #


def _run_ids(raw: Any, limit: int = MAX_COMPARE) -> list[str]:
    if not isinstance(raw, list) or any(not isinstance(i, str) for i in raw):
        raise ExperimentInputError(gettext("Give the runs as a list of run ids."))
    ids: list[str] = []
    for value in raw:
        value = value.strip()
        if not value:
            continue
        try:
            ident = str(uuid.UUID(value))
        except ValueError:
            raise ExperimentInputError(
                gettext("No such training run: '%(id)s'.") % {"id": value[:40]}
            ) from None
        if ident not in ids:
            ids.append(ident)
    if len(ids) > limit:
        raise ExperimentInputError(
            gettext("Compare at most %(count)d runs at a time.") % {"count": limit}
        )
    return ids


def _thin(points: list[dict[str, Any]], limit: int = MAX_CURVE_POINTS) -> list[dict[str, Any]]:
    """At most ``limit`` evenly spaced points, always with the first and the last."""
    if len(points) <= limit:
        return points
    step = (len(points) - 1) / (limit - 1)
    return [points[round(i * step)] for i in range(limit)]


def curve_points(run: TrainingRun, state: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The points of a run's curves: ``{"x", "epoch", "time", <metrics>}`` per finished
    epoch (``time``: seconds since the training started, from the epochs' durations)."""
    state = state if isinstance(state, dict) else {}
    history = state.get("history")
    if not isinstance(history, list):
        return []
    points = []
    elapsed = 0.0
    for index, entry in enumerate(history):
        if not isinstance(entry, dict):
            continue
        epoch = entry.get("epoch")
        epoch = epoch if isinstance(epoch, int) and not isinstance(epoch, bool) else index + 1
        seconds = _number(entry.get("seconds"))
        logged = _number(entry.get("time"))  # a run logged elsewhere: since its start
        if logged is not None:
            elapsed, seconds = logged, logged
        elif seconds is not None:
            elapsed += seconds
        point: dict[str, Any] = {
            "x": epoch,
            "epoch": epoch,
            "time": round(elapsed, 3) if seconds is not None else None,
        }
        for key in CURVE_KEYS:
            number = _number(entry.get(key))
            if number is not None:
                point[key] = number
        points.append(point)
    return _thin(points)


def curve_of(run: TrainingRun) -> dict[str, Any]:
    from .training_service import state_of

    state = state_of(run) if run.status != "queued" else None
    loss = state.get("loss") if isinstance(state, dict) else None
    external = state.get("external") if isinstance(state, dict) else None
    return {
        "id": str(run.pk),
        "name": run.name,
        "status": run.status,
        "kind": run.kind,
        "family": _family(run),
        "task": run.plan.get("task") if isinstance(run.plan, dict) else None,
        "loss": _loss(run),
        "loss_label": loss.get("label") if isinstance(loss, dict) else None,
        "x_unit": external.get("x_unit", "epoch") if isinstance(external, dict) else "epoch",
        "points": curve_points(run, state),
    }


def curves(user, raw_ids: Any) -> dict[str, Any]:
    """``GET /api/training/runs/curves/?ids=<id>,<id>``: the curves of up to MAX_COMPARE
    runs the user can see, in the order asked for; unknown ids are left out."""
    from . import training_service as training
    from .shared_data import visible_runs

    ids = _run_ids(raw_ids)
    found = {str(r.pk): r for r in visible_runs(user).filter(pk__in=ids)}
    runs = [training.sync_run(found[i]) for i in ids if i in found]
    return {"runs": [curve_of(run) for run in runs], "missing": [i for i in ids if i not in found]}


# --------------------------------------------------------------------------- #
# Tags, note, star
# --------------------------------------------------------------------------- #


def _clean_text(value: str) -> str:
    """Without control characters (newlines and tabs are kept)."""
    return "".join(
        c for c in value if c in "\n\t" or not unicodedata.category(c).startswith("C")
    ).strip()


def clean_tags(value: Any) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(t, str) for t in value):
        raise ExperimentInputError(gettext("'tags' must be a list of words."))
    tags: list[str] = []
    for raw in value:
        tag = " ".join(_clean_text(raw).split())
        if not tag:
            continue
        if len(tag) > MAX_TAG_LENGTH:
            raise ExperimentInputError(
                gettext("A tag has at most %(count)d characters.") % {"count": MAX_TAG_LENGTH}
            )
        if tag not in tags:
            tags.append(tag)
    if len(tags) > MAX_TAGS:
        raise ExperimentInputError(
            gettext("A run has at most %(count)d tags.") % {"count": MAX_TAGS}
        )
    return tags


def annotate(user, run: TrainingRun, data: Any) -> TrainingRun:
    """``PATCH /api/training/runs/<id>/annotations/`` ``{tags?, note?, starred?}``. The
    caller checks :func:`can_annotate`."""
    if not isinstance(data, dict) or not ({"tags", "note", "starred"} & set(data)):
        raise ExperimentInputError(gettext("Send tags, note or starred."))
    fields = []
    if "tags" in data:
        run.tags = clean_tags(data["tags"]) or None
        fields.append("tags")
    if "note" in data:
        note = data["note"]
        if note is None:
            note = ""
        if not isinstance(note, str):
            raise ExperimentInputError(gettext("'note' must be text."))
        note = _clean_text(note)
        if len(note) > MAX_NOTE_LENGTH:
            raise ExperimentInputError(
                gettext("A note has at most %(count)d characters.") % {"count": MAX_NOTE_LENGTH}
            )
        run.note = note
        fields.append("note")
    if "starred" in data:
        if not isinstance(data["starred"], bool):
            raise ExperimentInputError(gettext("'starred' must be true or false."))
        run.starred = data["starred"]
        fields.append("starred")
    TrainingRun.objects.filter(pk=run.pk).update(**{f: getattr(run, f) for f in fields})
    return run


def annotations(run: TrainingRun) -> dict[str, Any]:
    return {
        "id": str(run.pk),
        "tags": list(run.tags or []),
        "note": run.note,
        "starred": run.starred,
    }


# --------------------------------------------------------------------------- #
# Remembered columns
# --------------------------------------------------------------------------- #


def get_preferences(user) -> dict[str, Any]:
    profile = UserProfile.objects.filter(user=user).first()
    prefs = profile.preferences if profile and isinstance(profile.preferences, dict) else {}
    columns = prefs.get("runs_columns")
    return {"columns": columns if isinstance(columns, list) else None}


def set_preferences(user, data: Any) -> dict[str, Any]:
    """``PUT /api/training/runs/preferences/`` ``{"columns": [...] | null}``: the columns of
    the runs table, kept on the account (null: the defaults again)."""
    if not isinstance(data, dict) or "columns" not in data:
        raise ExperimentInputError(gettext("Send the columns as a list."))
    columns = data["columns"]
    if columns is not None:
        if (
            not isinstance(columns, list)
            or len(columns) > MAX_COLUMNS
            or any(not isinstance(c, str) or not COLUMN_NAME.match(c) for c in columns)
        ):
            raise ExperimentInputError(
                gettext("The columns must be a list of at most %(count)d column names.")
                % {"count": MAX_COLUMNS}
            )
        columns = list(dict.fromkeys(columns))
    profile, _ = UserProfile.objects.get_or_create(user=user)
    prefs = dict(profile.preferences) if isinstance(profile.preferences, dict) else {}
    if columns is None:
        prefs.pop("runs_columns", None)
    else:
        prefs["runs_columns"] = columns
    profile.preferences = prefs or None
    profile.save(update_fields=["preferences"])
    return {"columns": columns}


# --------------------------------------------------------------------------- #
# Shared comparisons
# --------------------------------------------------------------------------- #


def can_share(user, run: TrainingRun) -> bool:
    """Runs a user may put into a public comparison: their own, and those of networks they
    own (whose runs they may delete anyway). Runs of other people's networks are not."""
    if run.owner_id == user.pk:
        return True
    return (
        run.project_id is not None and run.project is not None and run.project.owner_id == user.pk
    )


def _clean_columns(value: Any) -> list[str]:
    if value is None:
        return []
    if (
        not isinstance(value, list)
        or len(value) > MAX_COLUMNS
        or any(not isinstance(c, str) or not COLUMN_NAME.match(c) for c in value)
    ):
        raise ExperimentInputError(
            gettext("The columns must be a list of at most %(count)d column names.")
            % {"count": MAX_COLUMNS}
        )
    return list(dict.fromkeys(value))


OPTION_VALUES = {
    "metric": re.compile(r"^[A-Za-z0-9_./-]{1,64}$"),
    "x_axis": re.compile(r"^(epoch|time)$"),
    "scatter_x": re.compile(r"^(metric|setting):[A-Za-z0-9_./-]{1,64}$"),
    "scatter_y": re.compile(r"^(metric|setting):[A-Za-z0-9_./-]{1,64}$"),
}


def _clean_options(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ExperimentInputError(gettext("'options' must be an object."))
    options: dict[str, Any] = {}
    for key, pattern in OPTION_VALUES.items():
        item = value.get(key)
        if item is None:
            continue
        if not isinstance(item, str) or not pattern.match(item):
            raise ExperimentInputError(
                gettext("The option '%(name)s' is not valid.") % {"name": key}
            )
        options[key] = item
    if "smoothing" in value:
        smoothing = _number(value["smoothing"])
        if smoothing is None or not 0 <= smoothing < 1:
            raise ExperimentInputError(gettext("'smoothing' must be a number from 0 to 0.99."))
        options["smoothing"] = smoothing
    return options


def comparison_data(comparison: RunComparison) -> dict[str, Any]:
    return {
        "id": comparison.pk,
        "name": comparison.name,
        "token": comparison.token,
        "runs": list(comparison.run_ids),
        "columns": list(comparison.columns),
        "options": dict(comparison.options),
        "include_notes": comparison.include_notes,
        "created_at": comparison.created_at,
    }


def create_comparison(user, data: Any) -> RunComparison:
    """``POST /api/training/comparisons/`` ``{name?, runs, columns?, options?,
    include_notes?}``: a read-only link to these runs."""
    from .shared_data import visible_runs

    if not isinstance(data, dict):
        raise ExperimentInputError(gettext("Send the runs to share."))
    ids = _run_ids(data.get("runs"))
    if not ids:
        raise ExperimentInputError(gettext("Choose at least one run to share."))
    found = {str(r.pk): r for r in visible_runs(user).select_related("project").filter(pk__in=ids)}
    for ident in ids:
        run = found.get(ident)
        if run is None:
            raise ExperimentInputError(gettext("No such training run: '%(id)s'.") % {"id": ident})
        if not can_share(user, run):
            raise ExperimentInputError(
                gettext(
                    "'%(name)s' was started by someone else in a network you do not own: "
                    "share only your own runs and runs of your networks."
                )
                % {"name": run.name}
            )
    name = data.get("name")
    if name is not None and not isinstance(name, str):
        raise ExperimentInputError(gettext("'name' must be text."))
    name = " ".join(_clean_text(name or "").split())[:120] or gettext("Shared comparison")
    include_notes = data.get("include_notes", False)
    if not isinstance(include_notes, bool):
        raise ExperimentInputError(gettext("'include_notes' must be true or false."))
    if RunComparison.objects.filter(owner=user).count() >= MAX_COMPARISONS:
        raise ExperimentInputError(
            gettext("You can have at most %(count)d shared comparisons. Delete one first.")
            % {"count": MAX_COMPARISONS}
        )
    return RunComparison.objects.create(
        owner=user,
        name=name,
        token=secrets.token_urlsafe(18),
        run_ids=ids,
        columns=_clean_columns(data.get("columns")),
        options=_clean_options(data.get("options")),
        include_notes=include_notes,
    )


def shared_comparison(token: str) -> dict[str, Any] | None:
    """``GET /api/shared/runs/<token>/``: what a shared comparison shows (None: no such
    link). The runs are checked again: a run that was deleted, or that its owner may no
    longer share, drops out."""
    from . import training_service as training

    if not isinstance(token, str) or not 8 <= len(token) <= 64:
        return None
    comparison = RunComparison.objects.select_related("owner").filter(token=token).first()
    if comparison is None:
        return None
    owner = comparison.owner
    try:
        found = {
            str(r.pk): r
            for r in TrainingRun.objects.select_related("project").filter(pk__in=comparison.run_ids)
        }
    except ValidationError:
        found = {}
    runs = [
        training.sync_run(found[i])
        for i in comparison.run_ids
        if i in found and can_share(owner, found[i])
    ]
    ensure_summaries(runs)
    now = timezone.now()
    return {
        "name": comparison.name,
        "created_at": comparison.created_at,
        "columns": list(comparison.columns),
        "options": dict(comparison.options),
        "include_notes": comparison.include_notes,
        "runs": [
            run_row(run, None, now=now, public=True, include_notes=comparison.include_notes)
            for run in runs
        ],
        "curves": [curve_of(run) for run in runs],
    }


def export_comparisons(user) -> list[dict[str, Any]]:
    """The user's shared comparisons for the data export."""
    return [comparison_data(c) for c in RunComparison.objects.filter(owner=user).order_by("pk")]
