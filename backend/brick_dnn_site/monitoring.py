"""Error monitoring with Sentry (docs/monitoring.md): off unless ``BRICKDNN_SENTRY_DSN`` is set.

The web backend, the realtime server, the training worker and the daily clean-up all load the
Django settings, which call :func:`init_sentry`; so do management commands run by hand. The
processes that train a network (``python -m brickdnn.training…``) do not load the settings and
report nothing themselves: their failures reach Sentry through the worker that started them.

Privacy: no personal data is sent. No user (id, name, e-mail), no IP address, no cookies and
no request bodies (form data, JSON, uploads); the SDK's scrubber removes headers and values
such as tokens and passwords from what is left. Health checks are not traced.
"""

from __future__ import annotations

import os
from typing import Any

# Requests that are not worth a trace: the health check runs every few seconds.
UNTRACED_PATHS = ("/api/health/",)
# Loggers whose messages are not sent (see init_sentry).
IGNORED_LOGGERS = ("django.channels.server", "daphne.server", "daphne.http_protocol")


def _rate(value: str | None, default: float) -> float:
    try:
        rate = float(value) if value not in (None, "") else default
    except ValueError:
        return default
    return min(max(rate, 0.0), 1.0)


def options(environ=os.environ, debug: bool = False) -> dict[str, Any] | None:
    """The arguments of ``sentry_sdk.init``, or None when monitoring is off."""
    dsn = (environ.get("BRICKDNN_SENTRY_DSN") or "").strip()
    if not dsn:
        return None
    default_rate = 1.0 if debug else 0.1
    traces_rate = _rate(environ.get("BRICKDNN_SENTRY_TRACES_SAMPLE_RATE"), default_rate)

    def traces_sampler(context: dict[str, Any]) -> float:
        scope = context.get("asgi_scope") or {}
        environ_ = context.get("wsgi_environ") or {}
        path = scope.get("path") or environ_.get("PATH_INFO") or ""
        if path.startswith(UNTRACED_PATHS):
            return 0.0
        parent = context.get("parent_sampled")
        if parent is not None:  # follow the browser's decision for the same request
            return 1.0 if parent else 0.0
        return traces_rate

    return {
        "dsn": dsn,
        "environment": environ.get("BRICKDNN_SENTRY_ENVIRONMENT")
        or ("development" if debug else "production"),
        "release": environ.get("BRICKDNN_SENTRY_RELEASE") or None,
        "send_default_pii": False,
        "max_request_body_size": "never",
        "traces_sampler": traces_sampler,
        "enable_logs": True,
    }


def init_sentry(environ=os.environ, debug: bool = False) -> bool:
    """Start the Sentry SDK if a DSN is set; True if it was started."""
    found = options(environ, debug)
    if found is None:
        return False
    import sentry_sdk
    from sentry_sdk.integrations.logging import ignore_logger

    sentry_sdk.init(**found)
    # The access logs of Daphne and of `runserver` log every 500 as an error: the exception
    # itself is reported already, so these would only add a second issue per error.
    for name in IGNORED_LOGGERS:
        ignore_logger(name)
    return True
