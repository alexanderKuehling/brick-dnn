"""Sending e-mail: with any SMTP provider, and only if the operator set one up.

``BRICKDNN_EMAIL_HOST`` and friends (see settings.py and docs/email.md) configure Django's SMTP
backend. Without them, :func:`enabled` is false: nothing is sent, and every feature that needs
e-mail says so instead (password reset, verifying addresses; warnings before inactive accounts
are deleted are skipped, and those accounts are deleted as before). With ``DEBUG``, mails that
would have been sent are logged instead, links included, for trying things out.

All e-mails share one layout (``templates/brick_dnn_core/email/``), plain text and simple
HTML, in English like the app. Requests do not wait for the SMTP server: messages are handed
to it in a background thread (``BRICKDNN_EMAIL_BACKGROUND``), so a slow provider delays no
answer and "forgot password" answers as fast for unknown addresses. Failures are logged, never
shown to the user as an error. Commands that need to know whether a mail went out (the
deletion warnings of ``purge_old_data``, ``send_test_email``) send right away instead.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from django.conf import settings
from django.core.mail import EmailMultiAlternatives, get_connection
from django.template.loader import render_to_string

logger = logging.getLogger(__name__)


def enabled() -> bool:
    """Whether this server sends e-mail."""
    return bool(getattr(settings, "EMAIL_ENABLED", False))


def site_url(request=None) -> str:
    """The public address for links in e-mails, e.g. ``https://brickdnn.com``:
    BRICKDNN_PUBLIC_URL, else taken from the request (empty without either)."""
    if settings.PUBLIC_URL:
        return settings.PUBLIC_URL
    if request is not None:
        try:
            return request.build_absolute_uri("/").rstrip("/")
        except Exception:  # a disallowed host
            return ""
    return ""


def site_domain(base: str) -> str:
    return urlsplit(base).netloc or settings.SITE_NAME


def link(path: str, request=None) -> str:
    """An absolute link to a page of the web app, e.g. ``link("/account")``."""
    return f"{site_url(request)}{path}"


@dataclass
class Message:
    """One e-mail in the shared layout."""

    to: str
    subject: str
    # Paragraphs before the button; plain text (escaped in the HTML version).
    paragraphs: list[str]
    # The button (label, URL); in the text version a line with the URL.
    action: tuple[str, str] | None = None
    # Paragraphs after the button, e.g. "If this was not you, ...".
    after: list[str] = field(default_factory=list)
    greeting: str = "Hello,"
    # Why the recipient gets this e-mail (small print at the end).
    reason: str = ""
    base_url: str = ""

    def build(self) -> EmailMultiAlternatives:
        base = self.base_url or site_url()
        context = {
            "site_name": settings.SITE_NAME,
            "site_url": base,
            "site_domain": site_domain(base),
            "greeting": self.greeting,
            "paragraphs": self.paragraphs,
            "action": self.action,
            "after": self.after,
            "reason": self.reason,
        }
        text = render_to_string("brick_dnn_core/email/base.txt", context)
        html = render_to_string("brick_dnn_core/email/base.html", context)
        message = EmailMultiAlternatives(
            subject=f"{self.subject} – {settings.SITE_NAME}",
            body=text,
            from_email=settings.DEFAULT_FROM_EMAIL,
            to=[self.to],
            reply_to=list(settings.EMAIL_REPLY_TO) or None,
        )
        message.attach_alternative(html, "text/html")
        return message


def _deliver(message: EmailMultiAlternatives, what: str) -> bool:
    try:
        sent = message.send(fail_silently=False)
    except Exception as exc:  # SMTP errors, timeouts, DNS ...
        # Not the recipient: the log should not collect addresses.
        logger.warning("Could not send the e-mail %r: %s", what, exc)
        return False
    return bool(sent)


def _synchronous_backend() -> bool:
    # Django's test backend collects mails in memory: tests read them right away.
    return settings.EMAIL_BACKEND.endswith("locmem.EmailBackend")


def send(message: Message, background: bool | None = None) -> bool:
    """Send ``message``. Returns whether it was handed to the SMTP server (in the background:
    whether it was queued). Never raises."""
    if not enabled():
        if settings.DEBUG:
            logger.info(
                "E-mail is not set up; not sending %r to %s:\n%s",
                message.subject,
                message.to,
                message.build().body,
            )
        return False
    if not message.to:
        return False
    try:
        built = message.build()
    except Exception:
        logger.exception("Could not build the e-mail %r", message.subject)
        return False
    if background is None:
        background = settings.EMAIL_IN_BACKGROUND and not _synchronous_backend()
    if not background:
        return _deliver(built, message.subject)
    thread = threading.Thread(
        target=_deliver, args=(built, message.subject), name="brickdnn-mail", daemon=False
    )
    thread.start()
    return True


def check_connection() -> None:
    """Open and close a connection to the SMTP server; raises what goes wrong."""
    connection = get_connection(fail_silently=False)
    connection.open()
    connection.close()
