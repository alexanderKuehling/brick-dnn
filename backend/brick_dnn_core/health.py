"""``GET /api/health/``: is the site working? Cheap enough for a check every few seconds.

Checks
------
- ``database``: a ``SELECT 1``. Without the database nothing works: the answer is ``503``.
- ``worker``: with the training queue (``BRICKDNN_TRAINING_WORKER``), a training worker sent a
  heartbeat within ``BRICKDNN_WORKER_TIMEOUT`` seconds.
- ``disk``: the data directory has at least ``BRICKDNN_HEALTH_MIN_FREE_MB`` free (0: no check).

The answer is ``{"status": "ok" | "degraded" | "error", "version": …, "checks": {name:
"ok" | "failed"}}`` (``"error"`` only when the database fails). Everyone sees the names of the
checks and whether they passed, nothing else: numbers and error messages (``detail``) only go to
staff accounts and to requests with the header ``X-Health-Token: <BRICKDNN_HEALTH_TOKEN>``.
"""

from __future__ import annotations

import hmac
import shutil
from typing import Any

from django.conf import settings
from django.db import connection
from django.utils import timezone

import brickdnn

OK = "ok"
FAILED = "failed"
TOKEN_HEADER = "HTTP_X_HEALTH_TOKEN"


def _database() -> tuple[bool, dict[str, Any]]:
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            cursor.fetchone()
    except Exception as exc:  # any driver error: the database is not usable
        return False, {"error": f"{type(exc).__name__}: {exc}"[:300]}
    return True, {"vendor": connection.vendor}


def _worker() -> tuple[bool, dict[str, Any]] | None:
    if not (settings.TRAINING_ENABLED and settings.TRAINING_WORKER):
        return None  # the web server trains itself (or not at all): no worker to check
    from .models import TrainingWorker

    latest = (
        TrainingWorker.objects.filter(stopped_at__isnull=True)
        .order_by("-heartbeat_at")
        .values_list("heartbeat_at", flat=True)
        .first()
    )
    if latest is None:
        return False, {"error": "No training worker is running."}
    age = (timezone.now() - latest).total_seconds()
    return age <= settings.WORKER_TIMEOUT, {
        "heartbeat_seconds_ago": round(age, 1),
        "timeout_seconds": settings.WORKER_TIMEOUT,
    }


def _disk() -> tuple[bool, dict[str, Any]] | None:
    minimum = settings.HEALTH_MIN_FREE_MB
    if minimum <= 0:
        return None
    try:
        usage = shutil.disk_usage(settings.DATA_DIR)
    except OSError as exc:
        return False, {"error": str(exc)[:300]}
    free_mb = usage.free / 1024**2
    return free_mb >= minimum, {
        "free_mb": round(free_mb),
        "total_mb": round(usage.total / 1024**2),
        "min_free_mb": minimum,
    }


CHECKS = {"database": _database, "worker": _worker, "disk": _disk}
# Checks that read the database: skipped while it fails.
NEEDS_DATABASE = {"worker"}


def may_see_details(request) -> bool:
    """Staff accounts, and requests with the secret header (for monitoring)."""
    token = settings.HEALTH_TOKEN
    given = request.META.get(TOKEN_HEADER, "")
    if token and given and hmac.compare_digest(given.encode(), token.encode()):
        return True
    user = getattr(request, "user", None)
    return bool(user is not None and user.is_authenticated and user.is_staff)


def report(details: bool = False) -> tuple[int, dict[str, Any]]:
    """The HTTP status and the body of ``/api/health/``."""
    checks: dict[str, str] = {}
    detail: dict[str, Any] = {}
    for name, check in CHECKS.items():
        if name in NEEDS_DATABASE and checks.get("database") != OK:
            continue  # it would only repeat the database's error
        try:
            result = check()
        except Exception as exc:  # a broken check must not break the health endpoint
            result = (False, {"error": f"{type(exc).__name__}: {exc}"[:300]})
        if result is None:
            continue  # not used on this server
        passed, info = result
        checks[name] = OK if passed else FAILED
        detail[name] = info
    if checks["database"] != OK:
        state, http_status = "error", 503
    elif FAILED in checks.values():
        state, http_status = "degraded", 200
    else:
        state, http_status = "ok", 200
    body: dict[str, Any] = {"status": state, "version": brickdnn.__version__, "checks": checks}
    if details:
        body["detail"] = detail
    return http_status, body
