"""Helpers for the tests of e-mail, password reset and two-factor authentication."""

from __future__ import annotations

import re

import pytest
from django.core import mail as django_mail
from rest_framework.test import APIClient

from brick_dnn_core import two_factor
from brick_dnn_core.models import TwoFactorAuth

SITE = "https://brickdnn.example"


@pytest.fixture
def email_on(settings):
    """E-mail set up (Django's in-memory backend in tests): returns the outbox."""
    settings.EMAIL_ENABLED = True
    settings.EMAIL_BACKEND = "django.core.mail.backends.locmem.EmailBackend"
    settings.PUBLIC_URL = SITE
    settings.DEFAULT_FROM_EMAIL = "Brick DNN <no-reply@brickdnn.example>"
    django_mail.outbox = []
    return django_mail.outbox


def links(message) -> list[str]:
    """The links of the site in a mail's text."""
    return re.findall(re.escape(SITE) + r"/[^\s]+", message.body)


def query_value(url: str, name: str) -> str:
    match = re.search(rf"[?&]{name}=([^&\s]+)", url)
    assert match, url
    return match.group(1)


def html_of(message) -> str:
    return next(content for content, kind in message.alternatives if kind == "text/html")


# --------------------------------------------------------------------------- #
# Two-factor authentication
# --------------------------------------------------------------------------- #

NOW = 1_800_000_000.0  # a fixed moment for the authenticator "app"


@pytest.fixture
def frozen_clock(monkeypatch):
    """The server's clock for codes, which tests move with ``frozen_clock.move(seconds)``."""

    class Clock:
        now = NOW

        def __call__(self) -> float:
            return self.now

        def move(self, seconds: float) -> None:
            self.now += seconds

    clock = Clock()
    monkeypatch.setattr(two_factor, "clock", clock)
    return clock


def secret_of(user) -> str:
    record = TwoFactorAuth.objects.get(user=user)
    return two_factor.decrypt(record.secret)


def app_code(secret: str, moment: float, steps: int = 0) -> str:
    """What an authenticator app shows at ``moment`` (``steps`` time steps later)."""
    return two_factor.totp(two_factor.secret_bytes(secret), moment + steps * two_factor.PERIOD)


def bearer(access: str) -> APIClient:
    client = APIClient()
    client.credentials(HTTP_AUTHORIZATION=f"Bearer {access}")
    return client


def turn_on_two_factor(client: APIClient, password: str, clock) -> tuple[str, list[str], dict]:
    """Set it up through the API: ``(secret, recovery codes, new tokens)``."""
    response = client.post("/api/auth/two-factor/setup/", {"password": password}, format="json")
    assert response.status_code == 200, response.data
    secret = response.data["secret"]
    response = client.post(
        "/api/auth/two-factor/enable/", {"code": app_code(secret, clock.now)}, format="json"
    )
    assert response.status_code == 200, response.data
    clock.move(two_factor.PERIOD)  # the next code is a new one
    tokens = {"access": response.data["access"], "refresh": response.data["refresh"]}
    return secret, response.data["recovery_codes"], tokens
