"""Verifying e-mail addresses and resetting forgotten passwords (recovery.py)."""

from __future__ import annotations

import pytest
from django.contrib.auth import get_user_model
from django.utils import timezone
from rest_framework.test import APIClient
from rest_framework.throttling import ScopedRateThrottle

from brick_dnn_core import recovery
from brick_dnn_core.models import UserProfile

from .mail_helpers import (
    bearer,
    links,
    query_value,
    turn_on_two_factor,
)
from .test_account import PASSWORD, ada, login, refresh  # noqa: F401 (fixture)

pytestmark = pytest.mark.django_db

User = get_user_model()
NEW_PASSWORD = "a-brand-new-password"


def sign_up(username: str = "grace", email: str = "grace@example.com"):
    return APIClient().post(
        "/api/auth/register/",
        {"username": username, "email": email, "password": PASSWORD},
        format="json",
    )


def ask_for_reset(login_name: str, client: APIClient | None = None):
    return (client or APIClient()).post(
        "/api/auth/password-reset/", {"login": login_name}, format="json"
    )


def link_parts(message) -> tuple[str, str]:
    url = next(link for link in links(message) if "/reset-password?" in link)
    return query_value(url, "uid"), query_value(url, "token")


def reset(uid: str, token: str, password: str = NEW_PASSWORD):
    return APIClient().post(
        "/api/auth/password-reset/confirm/",
        {"uid": uid, "token": token, "new_password": password},
        format="json",
    )


def verify_token(message) -> str:
    url = next(link for link in links(message) if "/verify-email?" in link)
    return query_value(url, "token")


@pytest.fixture
def legacy(ada):  # noqa: F811
    """Ada signed up before addresses were verified (what the migration marks)."""
    UserProfile.objects.update_or_create(user=ada, defaults={"email_legacy": True})
    return ada


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #


def test_sign_up_sends_a_verification_link(email_on):
    response = sign_up()
    assert response.status_code == 201
    assert len(email_on) == 1
    message = email_on[0]
    assert message.to == ["grace@example.com"]
    assert message.subject == "Confirm your e-mail address – Brick DNN"
    token = verify_token(message)
    client = bearer(response.data["access"])
    assert client.get("/api/auth/account/").data["email_verified"] is False

    confirmed = client.post("/api/auth/account/email/verify/", {"token": token}, format="json")
    assert confirmed.status_code == 200 and confirmed.data["email_verified"] is True
    account = client.get("/api/auth/account/").data
    assert account["email_verified"] is True and account["email_verified_at"]
    assert account["password_reset_available"] is True
    # Opening the link again does no harm.
    again = client.post("/api/auth/account/email/verify/", {"token": token}, format="json")
    assert again.status_code == 200


def test_sign_up_without_e_mail_sends_nothing(settings):
    from django.core import mail

    mail.outbox = []
    assert sign_up().status_code == 201
    assert mail.outbox == []


def test_only_the_own_account_can_confirm_the_address(email_on):
    sign_up()
    token = verify_token(email_on[0])
    # Nobody logged in: refused. Someone else logged in: refused.
    assert (
        APIClient()
        .post("/api/auth/account/email/verify/", {"token": token}, format="json")
        .status_code
        == 401
    )
    User.objects.create_user("eve", email="eve@example.com", password=PASSWORD)
    other = login("eve")[0].post("/api/auth/account/email/verify/", {"token": token}, format="json")
    assert other.status_code == 400 and other.data["code"] == "other_account"
    assert not recovery.is_verified(User.objects.get(username="grace"))


def test_verification_links_expire_and_follow_the_address(email_on, monkeypatch):
    client = bearer(sign_up().data["access"])
    old_token = verify_token(email_on[0])
    bad = client.post("/api/auth/account/email/verify/", {"token": "nonsense"}, format="json")
    assert bad.status_code == 400 and bad.data["code"] == "invalid_link"
    # A link for the previous address does not confirm the new one.
    client.patch("/api/auth/account/", {"email": "grace@new.example"}, format="json")
    stale = client.post("/api/auth/account/email/verify/", {"token": old_token}, format="json")
    assert stale.status_code == 400 and stale.data["code"] == "other_address"
    new_token = verify_token(email_on[-1])
    assert email_on[-1].to == ["grace@new.example"]
    monkeypatch.setattr(recovery, "VERIFY_DAYS", -1)
    expired = client.post("/api/auth/account/email/verify/", {"token": new_token}, format="json")
    assert expired.status_code == 400 and expired.data["code"] == "invalid_link"


def test_changing_the_address_makes_it_unverified(ada, email_on):  # noqa: F811
    recovery.mark_verified(ada)
    UserProfile.objects.filter(user=ada).update(email_legacy=True)
    client = login("ada")[0]
    assert client.get("/api/auth/account/").data["email_verified"] is True
    response = client.patch("/api/auth/account/", {"email": "ada@new.example"}, format="json")
    assert response.data["email_verified"] is False
    assert response.data["password_reset_available"] is False
    profile = UserProfile.objects.get(user=ada)
    assert profile.email_legacy is False and profile.email_verified_at is None
    # Changing the address elsewhere (e.g. in the Django admin) also leaves it unverified.
    recovery.mark_verified(User.objects.get(pk=ada.pk))
    User.objects.filter(pk=ada.pk).update(email="ada@third.example")
    assert recovery.is_verified(User.objects.get(pk=ada.pk)) is False


def test_resending_the_link_is_rate_limited(ada, email_on, monkeypatch):  # noqa: F811
    monkeypatch.setitem(ScopedRateThrottle.THROTTLE_RATES, "email_verify", "2/hour")
    client = login("ada")[0]
    statuses = [client.post("/api/auth/account/email/resend/").status_code for _ in range(3)]
    assert statuses == [200, 200, 429]
    assert len(email_on) == 2 and email_on[0].to == ["ada@example.com"]


def test_resending_without_e_mail_or_address(ada, settings):  # noqa: F811
    client = login("ada")[0]
    response = client.post("/api/auth/account/email/resend/")
    assert response.status_code == 503 and "sends no e-mail" in response.data["detail"]
    settings.EMAIL_ENABLED = True
    User.objects.filter(pk=ada.pk).update(email="")
    assert client.post("/api/auth/account/email/resend/").status_code == 400


def test_the_account_page_shows_the_state(ada, settings):  # noqa: F811
    data = login("ada")[0].get("/api/auth/account/").data
    assert data["email_verified"] is False and data["email_verified_at"] is None
    assert data["email_sending"] is False and data["password_reset_available"] is False
    assert data["two_factor"] == {"enabled": False, "enabled_at": None, "recovery_codes_left": 0}


# --------------------------------------------------------------------------- #
# Password reset
# --------------------------------------------------------------------------- #


def test_reset_is_unavailable_without_e_mail():
    assert APIClient().get("/api/auth/password-reset/").data == {"available": False}
    response = ask_for_reset("ada")
    assert response.status_code == 503
    assert "not available on this server" in response.data["detail"]


def test_reset_end_to_end(legacy, email_on):
    _, old_tokens = login("ada")
    assert APIClient().get("/api/auth/password-reset/").data == {"available": True}
    response = ask_for_reset("ada")
    assert response.status_code == 200 and "valid for one hour" in response.data["detail"]
    assert len(email_on) == 1 and email_on[0].to == ["ada@example.com"]
    uid, token = link_parts(email_on[0])

    check = APIClient().get(f"/api/auth/password-reset/confirm/?uid={uid}&token={token}")
    assert check.data == {"valid": True, "username": "ada"}
    weak = reset(uid, token, "123")
    assert weak.status_code == 400 and "new_password" in weak.data  # as at sign-up
    done = reset(uid, token)
    assert done.status_code == 200 and done.data["username"] == "ada"

    # The new password works, the old one and every earlier login do not.
    login("ada", NEW_PASSWORD)
    assert (
        APIClient()
        .post("/api/auth/token/", {"username": "ada", "password": PASSWORD}, format="json")
        .status_code
        == 401
    )
    assert refresh(old_tokens["refresh"]).status_code == 401
    # The link worked once; it proved the address, too.
    assert reset(uid, token, "yet-another-password").status_code == 400
    assert (
        APIClient().get(f"/api/auth/password-reset/confirm/?uid={uid}&token={token}").data["valid"]
        is False
    )
    assert recovery.is_verified(User.objects.get(username="ada"))


def test_reset_by_e_mail_address(legacy, email_on):
    assert ask_for_reset("ADA@example.com").status_code == 200
    assert email_on[0].to == ["ada@example.com"]


def test_no_account_enumeration(legacy, email_on):
    User.objects.create_user("nomail", email="", password=PASSWORD)
    User.objects.create_user("fresh", email="fresh@example.com", password=PASSWORD)  # unverified
    answers = {
        name: ask_for_reset(name).data
        for name in ("ada", "nobody", "nobody@example.com", "nomail", "fresh")
    }
    assert len({str(answer) for answer in answers.values()}) == 1
    # Only Ada's (old, unchanged) address gets a link.
    assert [m.to for m in email_on] == [["ada@example.com"]]


def test_new_accounts_need_a_verified_address(email_on):
    sign_up()
    email_on.clear()
    ask_for_reset("grace")
    assert email_on == []
    recovery.mark_verified(User.objects.get(username="grace"))
    ask_for_reset("grace")
    assert email_on[0].to == ["grace@example.com"]


def test_links_expire_and_end_with_a_password_change(legacy, email_on, settings):
    ask_for_reset("ada")
    uid, token = link_parts(email_on[0])
    user = User.objects.get(username="ada")
    user.set_password("changed-meanwhile-123")
    user.save()
    assert reset(uid, token).status_code == 400

    ask_for_reset("ada")
    uid, token = link_parts(email_on[-1])
    settings.PASSWORD_RESET_TIMEOUT = -1  # expired
    response = reset(uid, token)
    assert response.status_code == 400 and "invalid or has expired" in response.data["detail"]
    assert reset("garbage", "garbage").status_code == 400


def test_a_changed_address_voids_the_link(legacy, email_on):
    ask_for_reset("ada")
    uid, token = link_parts(email_on[0])
    User.objects.filter(username="ada").update(email="other@example.com")
    assert reset(uid, token).status_code == 400


def test_reset_links_per_account_are_limited(legacy, email_on):
    for _ in range(recovery.RESET_MAILS_PER_HOUR + 2):
        assert ask_for_reset("ada").status_code == 200
    assert len(email_on) == recovery.RESET_MAILS_PER_HOUR


def test_reset_requests_are_rate_limited_per_address(legacy, email_on, monkeypatch):
    monkeypatch.setitem(ScopedRateThrottle.THROTTLE_RATES, "password_reset", "2/hour")
    statuses = [ask_for_reset(f"user{i}").status_code for i in range(3)]
    assert statuses == [200, 200, 429]
    # Asking whether reset is available does not count.
    assert APIClient().get("/api/auth/password-reset/").status_code == 200


def test_two_factor_stays_on_after_a_reset(legacy, email_on, frozen_clock):
    secret, _, _ = turn_on_two_factor(login("ada")[0], PASSWORD, frozen_clock)
    email_on.clear()
    ask_for_reset("ada")
    uid, token = link_parts(email_on[0])
    assert reset(uid, token).status_code == 200
    response = APIClient().post(
        "/api/auth/token/", {"username": "ada", "password": NEW_PASSWORD}, format="json"
    )
    assert response.data["two_factor_required"] is True


def test_legacy_flag_from_the_migration(ada):  # noqa: F811
    """What 0016 does for accounts that exist: trusted until the address changes."""
    from importlib import import_module

    from django.apps import apps

    old = User.objects.create_user("old", email="old@example.com", password=PASSWORD)
    User.objects.filter(pk=old.pk).update(last_login=timezone.now() - timezone.timedelta(days=9))
    UserProfile.objects.filter(user=old).delete()
    migration = import_module("brick_dnn_core.migrations.0016_accounts_email_2fa")
    migration.mark_existing_addresses(apps, None)
    old_profile = UserProfile.objects.get(user=old)
    assert old_profile.email_legacy is True
    # The new profile keeps the account's activity (for the deletion of inactive accounts).
    assert old_profile.last_active_at == User.objects.get(pk=old.pk).last_login
    assert recovery.trusted_email(User.objects.get(pk=old.pk)) == "old@example.com"
