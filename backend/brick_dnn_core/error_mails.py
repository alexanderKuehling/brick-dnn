"""Server errors by e-mail to the operator (``BRICKDNN_ADMIN_EMAIL``, Django's ``ADMINS``).

Django logs every unhandled error of a request (HTTP 500) to the logger ``django.request``;
this handler mails it, but at most once an hour per kind of error (the exception's class and
where it was raised), so that a broken page does not flood the inbox. Nothing is sent while
e-mail is not set up (see mail.py). The e-mail is Django's plain-text report: the traceback and
the request, with passwords, tokens and keys masked.

This module is loaded while logging is configured, before the apps: it imports no models.
"""

from __future__ import annotations

import hashlib
import logging
import traceback

from django.conf import settings
from django.utils.log import AdminEmailHandler

logger = logging.getLogger(__name__)

# One e-mail per kind of error in this time.
THROTTLE_SECONDS = 3600
CACHE_PREFIX = "brickdnn:error-mail:"


def error_kind(record: logging.LogRecord) -> str:
    """What makes errors "the same": the exception class and the innermost frame of its
    traceback, or the message template for errors without an exception."""
    if record.exc_info and record.exc_info[0] is not None:
        exc_type, _, tb = record.exc_info
        where = ""
        frames = traceback.extract_tb(tb) if tb is not None else []
        if frames:
            last = frames[-1]
            where = f"{last.filename}:{last.lineno}"
        return f"{exc_type.__module__}.{exc_type.__qualname__}@{where}"
    return f"{record.name}:{record.msg}"


class ThrottledAdminEmailHandler(AdminEmailHandler):
    def emit(self, record: logging.LogRecord) -> None:
        if not settings.ADMINS or not getattr(settings, "EMAIL_ENABLED", False):
            return
        try:
            from django.core.cache import cache

            key = CACHE_PREFIX + hashlib.sha256(error_kind(record).encode()).hexdigest()[:32]
            if not cache.add(key, 1, THROTTLE_SECONDS):
                return  # the same error was mailed within the hour
        except Exception:  # the cache is broken: rather no mail than an error in the error path
            return
        super().emit(record)
