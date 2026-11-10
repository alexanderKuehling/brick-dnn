"""E-mail: sending, notices, server errors for the operator, ``send_test_email``."""

from __future__ import annotations

import io
import logging
import sys

import pytest
from django.contrib.auth import get_user_model
from django.core import mail as django_mail
from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import Client
from django.urls import path

from brick_dnn_core import error_mails, mail, notices
from brick_dnn_core.models import UserProfile

from .mail_helpers import (
    SITE,
    app_code,
    bearer,
    html_of,
    turn_on_two_factor,
)
from .test_account import PASSWORD, ada, login  # noqa: F401 (fixture)

pytestmark = pytest.mark.django_db

User = get_user_model()


def verified(user, address: str | None = None) -> None:
    from django.utils import timezone

    UserProfile.objects.update_or_create(
        user=user,
        defaults={"email_verified_at": timezone.now(), "email_verified_for": address or user.email},
    )


# --------------------------------------------------------------------------- #
# Sending
# --------------------------------------------------------------------------- #


def test_without_configuration_nothing_is_sent(settings, caplog):
    assert settings.EMAIL_ENABLED is False  # the default: no BRICKDNN_EMAIL_HOST
    django_mail.outbox = []
    message = mail.Message(to="ada@example.com", subject="Hi", paragraphs=["Text"])
    assert mail.send(message) is False
    assert django_mail.outbox == []
    # With DEBUG the mail is logged instead, to try things out.
    settings.DEBUG = True
    with caplog.at_level(logging.INFO, logger="brick_dnn_core.mail"):
        mail.send(message)
    assert "not sending 'Hi'" in caplog.text and "Text" in caplog.text


def test_messages_have_text_and_html_in_one_layout(email_on, settings):
    settings.SITE_NAME = "Brick DNN"
    settings.EMAIL_REPLY_TO = ["help@brickdnn.example"]
    message = mail.Message(
        to="ada@example.com",
        subject="Test <subject>",
        greeting="Hello ada,",
        paragraphs=["First <b>paragraph</b>.", "Second."],
        action=("Open it", f"{SITE}/account"),
        after=["After."],
        reason="Why you get this.",
    )
    assert mail.send(message) is True
    sent = email_on[0]
    assert sent.subject == "Test <subject> – Brick DNN"
    assert sent.to == ["ada@example.com"]
    assert sent.from_email == "Brick DNN <no-reply@brickdnn.example>"
    assert sent.reply_to == ["help@brickdnn.example"]
    assert "Hello ada," in sent.body and "First <b>paragraph</b>." in sent.body
    assert f"Open it:\n{SITE}/account" in sent.body and "Why you get this." in sent.body
    html = html_of(sent)
    assert "First &lt;b&gt;paragraph&lt;/b&gt;." in html  # escaped
    assert f'href="{SITE}/account"' in html and "brickdnn.example" in html


def test_failures_are_logged_not_raised(email_on, monkeypatch, caplog):
    def broken(self, fail_silently=False):
        raise OSError("connection refused")

    monkeypatch.setattr("django.core.mail.EmailMultiAlternatives.send", broken)
    message = mail.Message(to="ada@example.com", subject="Hi", paragraphs=["Text"])
    with caplog.at_level(logging.WARNING, logger="brick_dnn_core.mail"):
        assert mail.send(message) is False
    assert "Could not send the e-mail 'Hi': connection refused" in caplog.text
    assert "ada@example.com" not in caplog.text


def test_requests_hand_mails_to_a_background_thread(email_on, settings, monkeypatch):
    settings.EMAIL_BACKEND = "django.core.mail.backends.dummy.EmailBackend"
    settings.EMAIL_IN_BACKGROUND = True
    started = []

    class FakeThread:
        def __init__(self, target, args, name, daemon):
            started.append((target, args, daemon))

        def start(self):
            target, args, _ = started[-1]
            target(*args)

    monkeypatch.setattr(mail.threading, "Thread", FakeThread)
    message = mail.Message(to="ada@example.com", subject="Hi", paragraphs=["Text"])
    assert mail.send(message) is True
    assert len(started) == 1 and started[0][2] is False  # finishes before the process exits
    # Commands that need the result send right away.
    assert mail.send(message, background=False) is True
    assert len(started) == 1


def test_settings_from_the_environment(monkeypatch):
    import importlib

    from brick_dnn_site import settings as site_settings

    for name in list(__import__("os").environ):
        if name.startswith("BRICKDNN_EMAIL"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("BRICKDNN_EMAIL_HOST", "smtp.example.com")
    monkeypatch.setenv("BRICKDNN_EMAIL_PORT", "465")
    monkeypatch.setenv("BRICKDNN_EMAIL_USER", "apikey")
    monkeypatch.setenv("BRICKDNN_EMAIL_PASSWORD", "s3cret")
    monkeypatch.setenv("BRICKDNN_EMAIL_FROM", "Brick DNN <no-reply@brickdnn.com>")
    monkeypatch.setenv("BRICKDNN_ADMIN_EMAIL", "owner@example.com")
    try:
        loaded = importlib.reload(site_settings)
        assert loaded.EMAIL_ENABLED is True
        assert loaded.EMAIL_USE_SSL is True and loaded.EMAIL_USE_TLS is False
        assert loaded.EMAIL_HOST_PASSWORD == "s3cret"
        assert loaded.DEFAULT_FROM_EMAIL == loaded.SERVER_EMAIL
        assert loaded.ADMINS == [("Operator", "owner@example.com")]
        monkeypatch.setenv("BRICKDNN_EMAIL_PORT", "587")
        # Docker Compose passes unset variables as empty strings: STARTTLS stays on.
        monkeypatch.setenv("BRICKDNN_EMAIL_USE_TLS", "")
        monkeypatch.setenv("BRICKDNN_EMAIL_USE_SSL", "")
        loaded = importlib.reload(site_settings)
        assert loaded.EMAIL_USE_TLS is True and loaded.EMAIL_USE_SSL is False
        monkeypatch.setenv("BRICKDNN_EMAIL_USE_TLS", "false")
        assert importlib.reload(site_settings).EMAIL_USE_TLS is False
        monkeypatch.delenv("BRICKDNN_EMAIL_HOST")
        assert importlib.reload(site_settings).EMAIL_ENABLED is False
        monkeypatch.setenv("BRICKDNN_EMAIL_BACKEND", "file")
        assert importlib.reload(site_settings).EMAIL_ENABLED is True
    finally:
        for name in list(__import__("os").environ):
            if name.startswith("BRICKDNN_EMAIL") or name == "BRICKDNN_ADMIN_EMAIL":
                monkeypatch.delenv(name)
        importlib.reload(site_settings)


def test_the_password_is_never_read_from_the_site_file(tmp_path):
    from brick_dnn_site.settings import load_site_settings

    site = tmp_path / "settings.env"
    site.write_text("BRICKDNN_EMAIL_HOST=smtp.example.com\nBRICKDNN_EMAIL_PASSWORD=leaked\n")
    environ = {"BRICKDNN_SITE_ENV": str(site)}
    load_site_settings(environ)
    assert environ["BRICKDNN_EMAIL_HOST"] == "smtp.example.com"
    assert "BRICKDNN_EMAIL_PASSWORD" not in environ


# --------------------------------------------------------------------------- #
# send_test_email
# --------------------------------------------------------------------------- #


def test_send_test_email(email_on, monkeypatch):
    monkeypatch.setattr(mail, "check_connection", lambda: None)
    out = io.StringIO()
    call_command("send_test_email", "owner@example.com", stdout=out)
    assert "Sent a test e-mail to owner@example.com" in out.getvalue()
    assert email_on[0].to == ["owner@example.com"]
    assert "sending e-mail works" in email_on[0].body


def test_send_test_email_explains_what_is_wrong(settings, email_on, monkeypatch):
    with pytest.raises(CommandError, match="Not an e-mail address"):
        call_command("send_test_email", "nope", stdout=io.StringIO())

    def refuse():
        raise ConnectionRefusedError("refused")

    monkeypatch.setattr(mail, "check_connection", refuse)
    with pytest.raises(CommandError, match="Cannot connect"):
        call_command("send_test_email", "owner@example.com", stdout=io.StringIO())
    settings.EMAIL_ENABLED = False
    with pytest.raises(CommandError, match="not set up"):
        call_command("send_test_email", "owner@example.com", stdout=io.StringIO())


# --------------------------------------------------------------------------- #
# Security notices
# --------------------------------------------------------------------------- #


def test_password_change_notice(ada, email_on):  # noqa: F811
    verified(ada)
    client = login("ada")[0]
    response = client.post(
        "/api/auth/account/password/",
        {"current_password": PASSWORD, "new_password": "a-new-long-password"},
        format="json",
    )
    assert response.status_code == 200
    assert [m.subject for m in email_on] == ["Your password was changed – Brick DNN"]
    assert email_on[0].to == ["ada@example.com"]
    assert f"{SITE}/account" in email_on[0].body


def test_notices_only_go_to_trusted_addresses(ada, email_on):  # noqa: F811
    # A new account whose address was never confirmed gets no security notices.
    client = login("ada")[0]
    client.post(
        "/api/auth/account/password/",
        {"current_password": PASSWORD, "new_password": "a-new-long-password"},
        format="json",
    )
    assert email_on == []
    # An account from before addresses were verified does.
    UserProfile.objects.filter(user=ada).update(email_legacy=True)
    assert notices.password_changed(ada) is True
    assert email_on[0].to == ["ada@example.com"]


def test_email_change_notice_goes_to_the_old_address(ada, email_on):  # noqa: F811
    verified(ada)
    client = login("ada")[0]
    response = client.patch("/api/auth/account/", {"email": "new@example.com"}, format="json")
    assert response.status_code == 200
    assert response.data["email_verified"] is False
    by_address = {m.to[0]: m for m in email_on}
    assert set(by_address) == {"ada@example.com", "new@example.com"}
    assert "changed from ada@example.com to new@example.com" in by_address["ada@example.com"].body
    assert by_address["new@example.com"].subject.startswith("Confirm your e-mail address")


def test_two_factor_notices(ada, email_on, frozen_clock):  # noqa: F811
    verified(ada)
    secret, _, tokens = turn_on_two_factor(login("ada")[0], PASSWORD, frozen_clock)
    assert email_on[-1].subject.startswith("Two-factor authentication turned on")
    bearer(tokens["access"]).post(
        "/api/auth/two-factor/disable/",
        {"password": PASSWORD, "code": app_code(secret, frozen_clock.now)},
        format="json",
    )
    assert email_on[-1].subject.startswith("Two-factor authentication turned off")
    assert len(email_on) == 2


def test_api_token_and_app_notices(ada, email_on, settings):  # noqa: F811
    from .test_oauth import sign_in

    verified(ada)
    client = login("ada")[0]
    response = client.post("/api/auth/api-tokens/", {"name": "my script"}, format="json")
    assert response.status_code == 201
    assert email_on[-1].subject.startswith("A new API token was created")
    assert "“my script”" in email_on[-1].body
    assert response.data["token"] not in email_on[-1].body

    settings.PUBLIC_URL = ""  # the OAuth issuer of the test server
    sign_in(client)
    assert email_on[-1].subject.startswith("An app was connected to your account")
    assert "“Claude”" in email_on[-1].body
    assert len(email_on) == 2


# --------------------------------------------------------------------------- #
# Server errors for the operator
# --------------------------------------------------------------------------- #


def broken_view(request):
    raise RuntimeError("something broke")


def other_broken_view(request):
    raise KeyError("missing")


urlpatterns = [
    path("broken/", broken_view),
    path("other/", other_broken_view),
]


@pytest.fixture
def operator(settings, email_on):
    settings.ADMINS = [("Operator", "owner@example.com")]
    settings.ROOT_URLCONF = sys.modules[__name__]
    settings.DEBUG = False
    return email_on


def get_500(url: str):
    client = Client(raise_request_exception=False)
    response = client.get(url)
    assert response.status_code == 500


def test_server_errors_are_mailed_at_most_once_an_hour_per_kind(operator):
    for _ in range(3):
        get_500("/broken/")
    assert len(operator) == 1
    sent = operator[0]
    assert sent.to == ["owner@example.com"]
    assert "RuntimeError" in sent.body and "something broke" in sent.body
    get_500("/other/")  # another kind of error is mailed too
    assert len(operator) == 2 and "KeyError" in operator[1].body


def test_no_error_mails_without_an_address_or_e_mail(settings, email_on):
    settings.ROOT_URLCONF = sys.modules[__name__]
    settings.ADMINS = []
    get_500("/broken/")
    settings.ADMINS = [("Operator", "owner@example.com")]
    settings.EMAIL_ENABLED = False
    get_500("/broken/")
    assert email_on == []


def test_error_kinds():
    def record_of(exc_type):
        try:
            raise exc_type("x")
        except exc_type:
            return logging.LogRecord(
                "django.request", logging.ERROR, __file__, 1, "Error", (), sys.exc_info()
            )

    first, second = record_of(ValueError), record_of(ValueError)
    assert error_mails.error_kind(first) == error_mails.error_kind(second)
    assert error_mails.error_kind(first) != error_mails.error_kind(record_of(TypeError))
    plain = logging.LogRecord("django.request", logging.ERROR, __file__, 1, "Oops %s", ("a",), None)
    assert error_mails.error_kind(plain) == "django.request:Oops %s"
