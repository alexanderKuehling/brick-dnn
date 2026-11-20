"""Runs that ran elsewhere: a script logs its curves into Brick DNN (``kind`` "external").

A training on the user's own machine (the exported PyTorch script, or any script with
``brickdnn.tracking``) creates a run with a personal API token, sends its points per epoch
or step in batches, its settings and final numbers, and ends it. The run then appears in
the history, the runs table and comparisons next to the runs trained on the server, with
the badge *ran elsewhere*. Nothing is computed here: such runs use no CPU time of the
server and are not in the training queue, but they count against limits of their own
(:func:`limits`: runs per account, runs reporting at a time, points per run, the size of a
request), the request rate (``BRICKDNN_TRACKING_RATE``) and the storage quota (their points
are a file in the run directory).

The points live in ``TRAINING_DIR/runs/<id>/points.jsonl`` (one JSON object per line,
appended under a lock of the run's row); :func:`state_of` turns them into the ``history`` of
a state, so the run view draws them like the epochs of any run. ``plan["external"]`` keeps
the client, the number of points, the time of the last report and the reported summary.

Endpoints (personal API tokens only, also tokens with the scope ``runs:log``):

- ``POST /api/tracking/runs/`` ``{name?, project?, dataset?, task?, config?, epochs?,
  client?}`` creates a run (``running``);
- ``POST /api/tracking/runs/<id>/log/`` ``{points: [{epoch?, step?, time?, <metric>: n}]}``;
- ``PATCH /api/tracking/runs/<id>/`` ``{name?, config?, summary?, message?}``;
- ``POST /api/tracking/runs/<id>/finish/`` ``{status, summary?, message?}``;
- ``GET /api/tracking/runs/<id>/`` where it stands.
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timedelta
from typing import Any

from django.conf import settings
from django.db import transaction
from django.utils import timezone
from django.utils.translation import gettext
from rest_framework import permissions, status
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.throttling import ScopedRateThrottle
from rest_framework.views import APIView

from .api_tokens import uses_api_token
from .models import ApiToken, TrainingRun

KIND = "external"
POINTS_FILE = "points.jsonl"
# One request: points, bytes of its body; metrics per point; summary values; config bytes.
MAX_BATCH = 1000
MAX_BODY_BYTES = 512 * 1024
MAX_METRICS = 32
MAX_SUMMARY = 64
MAX_CONFIG_BYTES = 16 * 1024
MAX_CONFIG_DEPTH = 4
MAX_TEXT = 500
MAX_STEP = 10**9
# A run that has not reported for this long is given up (it ends as failed).
STALE_AFTER = timedelta(hours=24)
# Points of a run shown in its state (the run view); the file keeps all of them.
STATE_POINTS = 2000
TASKS = ("classification", "regression", "segmentation")
ENDINGS = ("completed", "failed", "stopped")
METRIC_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_./-]{0,63}$")
RESERVED = {"epoch", "step", "time", "x"}
# Metrics of the last point that become the run's metrics; the final summary overrides them.
RUN_METRICS = (
    "train_loss",
    "train_accuracy",
    "val_loss",
    "val_accuracy",
    "val_rmse",
    "val_r2",
    "val_iou",
    "val_dice",
    "val_pixel_accuracy",
)
TEST_METRICS = ("test_loss", "test_accuracy", "test_rmse", "test_r2", "test_iou", "test_dice")


def limits() -> dict[str, int]:
    """What one account and one run may log (settings BRICKDNN_TRACKING_*)."""
    return {
        "max_runs": settings.TRACKING_MAX_RUNS,
        "max_active_runs": settings.TRACKING_MAX_ACTIVE,
        "max_points": settings.TRACKING_MAX_POINTS,
        "max_batch": MAX_BATCH,
        "max_body_bytes": MAX_BODY_BYTES,
        "max_metrics": MAX_METRICS,
    }


class TrackingError(Exception):
    """A refused request: its HTTP status and a message (and a code for limits)."""

    def __init__(self, message: str, status_code: int = 400, code: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.code = code

    def response(self) -> Response:
        body: dict[str, Any] = {"detail": str(self)}
        if self.code:
            body["code"] = self.code
        return Response(body, status=self.status_code)


# --------------------------------------------------------------------------- #
# Checking what a script sends
# --------------------------------------------------------------------------- #


def _number(value: Any) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) and abs(value) < 1e15 else None


def _text(value: Any, field: str, limit: int = 120) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TrackingError(gettext("'%(field)s' must be text.") % {"field": field})
    text = " ".join("".join(c for c in value if c.isprintable()).split())
    return text[:limit] or None


def _metric_name(key: Any, where: str) -> str:
    if not isinstance(key, str) or not METRIC_NAME.match(key) or key in RESERVED:
        raise TrackingError(
            gettext(
                "%(where)s: '%(name)s' is not a metric name (letters, digits and _ . / -, "
                "starting with a letter, at most 64 characters)."
            )
            % {"where": where, "name": str(key)[:64]}
        )
    return key


def _clean_value(value: Any, depth: int) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        number = _number(value)
        if number is None:
            raise TrackingError(gettext("The config holds a number that is not finite."))
        return number
    if isinstance(value, str):
        return value[:MAX_TEXT]
    if depth >= MAX_CONFIG_DEPTH:
        raise TrackingError(
            gettext("The config is nested too deeply (at most %(depth)d levels).")
            % {"depth": MAX_CONFIG_DEPTH}
        )
    if isinstance(value, list):
        return [_clean_value(item, depth + 1) for item in value[:100]]
    if isinstance(value, dict):
        return {
            str(key)[:64]: _clean_value(item, depth + 1) for key, item in list(value.items())[:100]
        }
    raise TrackingError(gettext("The config may only hold numbers, text, lists and objects."))


def clean_config(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TrackingError(gettext("'config' must be an object."))
    config = _clean_value(value, 1)
    if len(json.dumps(config)) > MAX_CONFIG_BYTES:
        raise TrackingError(
            gettext("The config is too large (at most %(kb)d KB).")
            % {"kb": MAX_CONFIG_BYTES // 1024},
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            "too_large",
        )
    return config


def clean_summary(value: Any) -> dict[str, float | int]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TrackingError(gettext("'summary' must be an object of numbers."))
    if len(value) > MAX_SUMMARY:
        raise TrackingError(
            gettext("A summary has at most %(count)d numbers.") % {"count": MAX_SUMMARY}
        )
    found = {}
    for key, item in value.items():
        name = _metric_name(key, gettext("Summary"))
        if item is None:
            continue
        number = _number(item)
        if number is None:
            raise TrackingError(
                gettext("Summary: '%(name)s' must be a finite number.") % {"name": name}
            )
        found[name] = number
    return found


def clean_points(value: Any) -> list[dict[str, Any]]:
    """The points of a log request: 1 to MAX_BATCH objects with ``epoch`` or ``step`` (whole
    numbers from 0), optionally ``time`` (seconds since the start) and up to MAX_METRICS
    metrics (finite numbers; null leaves a gap)."""
    if not isinstance(value, list) or not value:
        raise TrackingError(gettext("'points' must be a list of points."))
    if len(value) > MAX_BATCH:
        raise TrackingError(
            gettext("Send at most %(count)d points at a time.") % {"count": MAX_BATCH},
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            "too_large",
        )
    points = []
    for index, raw in enumerate(value, start=1):
        where = gettext("Point %(index)d") % {"index": index}
        if not isinstance(raw, dict):
            raise TrackingError(gettext("%(where)s is not an object.") % {"where": where})
        point: dict[str, Any] = {}
        for key in ("epoch", "step"):
            if raw.get(key) is not None:
                item = raw[key]
                if isinstance(item, bool) or not isinstance(item, int) or not 0 <= item <= MAX_STEP:
                    raise TrackingError(
                        gettext("%(where)s: '%(key)s' must be a whole number from 0.")
                        % {"where": where, "key": key}
                    )
                point[key] = item
        if not point:
            raise TrackingError(gettext("%(where)s needs an epoch or a step.") % {"where": where})
        if raw.get("time") is not None:
            seconds = _number(raw["time"])
            if seconds is None or seconds < 0:
                raise TrackingError(
                    gettext("%(where)s: 'time' must be the seconds since the start.")
                    % {"where": where}
                )
            point["time"] = round(seconds, 3)
        metrics = [key for key in raw if key not in RESERVED]
        if len(metrics) > MAX_METRICS:
            raise TrackingError(
                gettext("%(where)s has more than %(count)d metrics.")
                % {"where": where, "count": MAX_METRICS}
            )
        for key in metrics:
            name = _metric_name(key, where)
            item = raw[key]
            if item is None:
                continue
            number = _number(item)
            if number is None:
                raise TrackingError(
                    gettext("%(where)s: '%(name)s' must be a finite number.")
                    % {"where": where, "name": name}
                )
            point[name] = number
        points.append(point)
    return points


# --------------------------------------------------------------------------- #
# Runs
# --------------------------------------------------------------------------- #


def _external(run: TrainingRun) -> dict[str, Any]:
    plan = run.plan if isinstance(run.plan, dict) else {}
    found = plan.get("external")
    return found if isinstance(found, dict) else {}


def points_path(run: TrainingRun):
    from .training_service import run_dir

    return run_dir(run.pk) / POINTS_FILE


def read_points(run: TrainingRun) -> list[dict[str, Any]]:
    try:
        lines = points_path(run).read_text().splitlines()
    except OSError:
        return []
    points = []
    for line in lines:
        try:
            point = json.loads(line)
        except ValueError:
            continue
        if isinstance(point, dict):
            points.append(point)
    return points


def _refuse_finished(run: TrainingRun) -> None:
    if run.status != "running":
        raise TrackingError(
            gettext("This run has ended (%(status)s): it takes no more reports.")
            % {"status": run.status},
            status.HTTP_409_CONFLICT,
            "finished",
        )


def create_run(user, token: ApiToken | None, data: Any) -> TrainingRun:
    from . import budgets
    from .shared_data import editable_projects
    from .training_service import run_dir

    if not isinstance(data, dict):
        raise TrackingError(gettext("Send the run as a JSON object."))
    runs = TrainingRun.objects.filter(owner=user, kind=KIND)
    if runs.count() >= settings.TRACKING_MAX_RUNS:
        raise TrackingError(
            gettext(
                "Your account has %(count)d runs logged from elsewhere, the most it may keep. "
                "Delete some on the runs page."
            )
            % {"count": settings.TRACKING_MAX_RUNS},
            status.HTTP_409_CONFLICT,
            "runs_limit",
        )
    if runs.filter(status="running").count() >= settings.TRACKING_MAX_ACTIVE:
        raise TrackingError(
            gettext(
                "%(count)d runs of your account are logging at the moment, the most at a time. "
                "Finish one first."
            )
            % {"count": settings.TRACKING_MAX_ACTIVE},
            status.HTTP_409_CONFLICT,
            "active_limit",
        )
    try:
        budgets.check_storage(user)
    except budgets.StorageQuotaExceeded as exc:
        raise TrackingError(str(exc), status.HTTP_403_FORBIDDEN, "storage_quota_exceeded") from None
    project = None
    if data.get("project") is not None:
        ident = data["project"]
        if isinstance(ident, bool) or not isinstance(ident, int):
            raise TrackingError(gettext("'project' must be the id of a network."))
        project = editable_projects(user).filter(pk=ident).first()
        if project is None:
            raise TrackingError(
                gettext("Network %(id)d is not one you own or edit.") % {"id": ident},
                status.HTTP_400_BAD_REQUEST,
                "project",
            )
    task = data.get("task")
    if task is not None and task not in TASKS:
        raise TrackingError(gettext("'task' must be classification, regression or segmentation."))
    epochs = data.get("epochs")
    if epochs is not None and (
        isinstance(epochs, bool) or not isinstance(epochs, int) or not 0 < epochs <= MAX_STEP
    ):
        raise TrackingError(gettext("'epochs' must be a whole number from 1."))
    config = clean_config(data.get("config"))
    if epochs is not None:
        config.setdefault("epochs", epochs)
    name = _text(data.get("name"), "name") or gettext("Run from a script")
    dataset = _text(data.get("dataset"), "dataset") or gettext("Elsewhere")
    client = _text(data.get("client"), "client", 60)
    now = timezone.now()
    run = TrainingRun.objects.create(
        owner=user,
        kind=KIND,
        name=name,
        graph=project.graph if project is not None else {"version": 1, "nodes": [], "edges": []},
        dataset={"kind": KIND, "id": None, "name": dataset},
        project=project,
        config=config,
        plan={
            # The shape of a plan of the server (brickdnn.training.compat), with nothing in it.
            "task": task,
            "summary": "",
            "inputs": [],
            "targets": [],
            "image": None,
            "num_classes": None,
            "class_names": None,
            "notes": [],
            "external": {
                "client": client,
                "points": 0,
                "reported_at": now.isoformat(),
                "summary": {},
            },
        },
        status="running",
        message=gettext("Logging from a script elsewhere."),
        started_at=now,
        api_token=token if token is not None and token.connection_id is None else None,
        size_bytes=0,
    )
    directory = run_dir(run.pk)
    directory.mkdir(parents=True, exist_ok=True)
    points_path(run).touch()
    return run


def _last_metrics(run: TrainingRun, points: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = dict(run.metrics) if isinstance(run.metrics, dict) else {}
    for point in points:
        for key in RUN_METRICS:
            if key in point:
                metrics[key] = point[key]
    return metrics


def append_points(run: TrainingRun, data: Any, size: int) -> TrainingRun:
    """Add the points of a log request (``size``: its bytes, for the storage quota)."""
    from . import budgets

    if not isinstance(data, dict):
        raise TrackingError(gettext('Send the points as {"points": [...]}.'))
    points = clean_points(data.get("points"))
    try:
        budgets.check_storage(run.owner, adding=size)
    except budgets.StorageQuotaExceeded as exc:
        raise TrackingError(str(exc), status.HTTP_403_FORBIDDEN, "storage_quota_exceeded") from None
    with transaction.atomic():
        # One request after the other per run: the count and the file stay in step.
        run = TrainingRun.objects.select_for_update().get(pk=run.pk)
        _refuse_finished(run)
        external = _external(run)
        count = int(external.get("points") or 0)
        if count + len(points) > settings.TRACKING_MAX_POINTS:
            raise TrackingError(
                gettext(
                    "A run logged from elsewhere keeps at most %(count)d points; this one has "
                    "%(has)d. Log less often (e.g. once per epoch)."
                )
                % {"count": settings.TRACKING_MAX_POINTS, "has": count},
                status.HTTP_409_CONFLICT,
                "points_limit",
            )
        path = points_path(run)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as out:
            out.write("".join(json.dumps(p, separators=(",", ":")) + "\n" for p in points))
        steps = [p.get("epoch", p.get("step", 0)) for p in points]
        external.update(points=count + len(points), reported_at=timezone.now().isoformat())
        run.plan = {**run.plan, "external": external}
        run.epochs_done = max([run.epochs_done, *steps])
        run.metrics = _last_metrics(run, points)
        run.size_bytes = path.stat().st_size
        run.save(update_fields=["plan", "epochs_done", "metrics", "size_bytes"])
    return run


def _apply_summary(run: TrainingRun, summary: dict[str, Any]) -> None:
    external = _external(run)
    external["summary"] = {**(external.get("summary") or {}), **summary}
    run.plan = {**run.plan, "external": external}
    metrics = dict(run.metrics) if isinstance(run.metrics, dict) else {}
    metrics.update({key: summary[key] for key in (*RUN_METRICS, *TEST_METRICS) if key in summary})
    run.metrics = metrics


def update_run(run: TrainingRun, data: Any) -> TrainingRun:
    if not isinstance(data, dict):
        raise TrackingError(gettext("Send the changes as a JSON object."))
    with transaction.atomic():
        run = TrainingRun.objects.select_for_update().get(pk=run.pk)
        _refuse_finished(run)
        if "name" in data:
            run.name = _text(data["name"], "name") or run.name
        if "config" in data:
            run.config = clean_config({**(run.config or {}), **clean_config(data["config"])})
        if "summary" in data:
            _apply_summary(run, clean_summary(data["summary"]))
        if "message" in data:
            run.message = _text(data["message"], "message", MAX_TEXT) or run.message
        external = _external(run)
        external["reported_at"] = timezone.now().isoformat()
        run.plan = {**run.plan, "external": external}
        run.save(update_fields=["name", "config", "plan", "metrics", "message"])
    return run


def finish_run(run: TrainingRun, data: Any) -> TrainingRun:
    if not isinstance(data, dict):
        raise TrackingError(gettext("Send the end of the run as a JSON object."))
    ending = data.get("status", "completed")
    if ending not in ENDINGS:
        raise TrackingError(gettext("'status' must be completed, failed or stopped."))
    summary = clean_summary(data.get("summary"))
    message = _text(data.get("message"), "message", MAX_TEXT)
    with transaction.atomic():
        run = TrainingRun.objects.select_for_update().get(pk=run.pk)
        _refuse_finished(run)
        _apply_summary(run, summary)
        run.status = ending
        run.message = (
            message
            or {
                "completed": gettext("Finished elsewhere."),
                "failed": gettext("The script reported a failure."),
                "stopped": gettext("Stopped elsewhere."),
            }[ending]
        )
        run.finished_at = timezone.now()
        run.summary = None  # the runs table reads it again (experiments.summary_of)
        run.save(update_fields=["plan", "metrics", "status", "message", "finished_at", "summary"])
    return run


def sync(run: TrainingRun) -> TrainingRun:
    """A run that stopped reporting a day ago ends as failed (the script may have died)."""
    if run.status != "running":
        return run
    reported = _external(run).get("reported_at")
    try:
        last = datetime.fromisoformat(reported) if isinstance(reported, str) else run.created_at
    except ValueError:
        last = run.created_at
    if timezone.now() - last < STALE_AFTER:
        return run
    updated = TrainingRun.objects.filter(pk=run.pk, status="running").update(
        status="failed",
        message=gettext("The script stopped reporting: nothing arrived for 24 hours."),
        finished_at=timezone.now(),
        summary=None,
    )
    if updated:
        run.refresh_from_db()
    return run


def stop(run: TrainingRun) -> TrainingRun:
    """Stopped in the app: further reports of the script are refused (it gives up then)."""
    if run.status == "running":
        TrainingRun.objects.filter(pk=run.pk, status="running").update(
            status="stopped",
            stop_reason="user",
            message=gettext("Stopped in Brick DNN; the script's further reports are refused."),
            finished_at=timezone.now(),
            summary=None,
        )
        run.refresh_from_db()
    return run


def state_of(run: TrainingRun) -> dict[str, Any]:
    """A state like state.json of a run on the server (brickdnn.training.state), built from
    the logged points: ``history`` (the points, at most STATE_POINTS), ``test`` (the reported
    test numbers) and ``external``."""
    from .experiments import _thin

    external = _external(run)
    points = read_points(run)
    by_step = bool(points) and all("epoch" not in p for p in points)
    history = [
        {
            **{k: v for k, v in p.items() if k not in ("epoch", "step")},
            "epoch": p.get("epoch", p.get("step")),
        }
        for p in points
    ]
    summary = external.get("summary") if isinstance(external.get("summary"), dict) else {}
    test = {key[5:]: value for key, value in summary.items() if key.startswith("test_")}
    config = run.config if isinstance(run.config, dict) else {}
    loss = config.get("loss")
    started = run.started_at or run.created_at
    return {
        "status": run.status,
        "message": run.message,
        "family": "neural",
        "task": run.plan.get("task") if isinstance(run.plan, dict) else None,
        "epochs": config.get("epochs") if isinstance(config.get("epochs"), int) else None,
        "started_at": started.timestamp() if started else None,
        "finished_at": run.finished_at.timestamp() if run.finished_at else None,
        "history": _thin(history, STATE_POINTS),
        "progress": None,
        "best": None,
        "test": test or None,
        "loss": {"name": loss, "label": loss} if isinstance(loss, str) else None,
        "weights": False,
        "log": [],
        "external": {
            "client": external.get("client"),
            "points": external.get("points", len(points)),
            "reported_at": external.get("reported_at"),
            "x_unit": "step" if by_step else "epoch",
            "summary": summary,
        },
    }


def describe(run: TrainingRun) -> dict[str, Any] | None:
    """``external`` of a run in the API: where it stands (null for runs on the server)."""
    if run.kind != KIND:
        return None
    external = _external(run)
    return {
        "client": external.get("client"),
        "points": external.get("points", 0),
        "reported_at": external.get("reported_at"),
    }


def run_data(run: TrainingRun) -> dict[str, Any]:
    """The answer of the tracking endpoints."""
    return {
        "id": str(run.pk),
        "name": run.name,
        "status": run.status,
        "project": run.project_id,
        "points": _external(run).get("points", 0),
        "url": f"/runs?ids={run.pk}",
        "limits": limits(),
    }


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #


class PersonalApiToken(permissions.BasePermission):
    """Logged in with a personal API token (not a website login, not an app's token)."""

    message = "Log runs with a personal API token (Authorization: Bearer bdnn_…)."

    def has_permission(self, request, view) -> bool:
        return (
            request.user is not None
            and request.user.is_authenticated
            and uses_api_token(request)
            and request.auth.connection_id is None
        )


class TrackingView(APIView):
    permission_classes = [PersonalApiToken]
    throttle_scope = "tracking"

    def get_throttles(self):
        return [*super().get_throttles(), ScopedRateThrottle()]

    def initial(self, request, *args, **kwargs):
        super().initial(request, *args, **kwargs)
        length = request.META.get("CONTENT_LENGTH") or "0"
        if length.isdigit() and int(length) > MAX_BODY_BYTES:
            raise _TooLarge()

    def handle_exception(self, exc):
        if isinstance(exc, _TooLarge):
            return TrackingError(
                gettext("A request may hold at most %(kb)d KB: send fewer points at a time.")
                % {"kb": MAX_BODY_BYTES // 1024},
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                "too_large",
            ).response()
        if isinstance(exc, TrackingError):
            return exc.response()
        return super().handle_exception(exc)

    def run_of(self, request: Request, pk) -> TrainingRun:
        run = TrainingRun.objects.filter(pk=pk, owner=request.user, kind=KIND).first()
        if run is None:
            raise TrackingError(
                gettext("No such run logged from elsewhere."), status.HTTP_404_NOT_FOUND
            )
        return sync(run)


class _TooLarge(Exception):
    pass


class TrackingRunsView(TrackingView):
    """``POST /api/tracking/runs/``: a new run (``201``)."""

    def post(self, request: Request) -> Response:
        run = create_run(request.user, request.auth, request.data)
        return Response(run_data(run), status=status.HTTP_201_CREATED)


class TrackingRunView(TrackingView):
    """``GET`` where a run stands, ``PATCH {name?, config?, summary?, message?}``."""

    def get(self, request: Request, pk) -> Response:
        return Response(run_data(self.run_of(request, pk)))

    def patch(self, request: Request, pk) -> Response:
        return Response(run_data(update_run(self.run_of(request, pk), request.data)))


class TrackingLogView(TrackingView):
    """``POST {points: [...]}``: more points of the curves."""

    def post(self, request: Request, pk) -> Response:
        run = self.run_of(request, pk)
        size = int(request.META.get("CONTENT_LENGTH") or 0)
        return Response(run_data(append_points(run, request.data, size)))


class TrackingFinishView(TrackingView):
    """``POST {status, summary?, message?}``: the end of the run."""

    def post(self, request: Request, pk) -> Response:
        return Response(run_data(finish_run(self.run_of(request, pk), request.data)))
