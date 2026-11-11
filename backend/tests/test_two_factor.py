"""Two-factor authentication: TOTP, setup, the two-step login, the Django admin, OAuth."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile

import pytest
from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.test import Client
from rest_framework.test import APIClient

from brick_dnn_core import two_factor
from brick_dnn_core.models import TwoFactorAuth

from .mail_helpers import (
    app_code,
    bearer,
    secret_of,
    turn_on_two_factor,
)
from .test_account import PASSWORD, ada, login  # noqa: F401 (fixture)

pytestmark = pytest.mark.django_db

User = get_user_model()


def password_login(username: str = "ada", password: str = PASSWORD):
    return APIClient().post(
        "/api/auth/token/", {"username": username, "password": password}, format="json"
    )


def second_step(challenge: str, code: str):
    return APIClient().post(
        "/api/auth/token/two-factor/", {"challenge": challenge, "code": code}, format="json"
    )


@pytest.fixture
def ada_2fa(ada, frozen_clock):  # noqa: F811
    """Ada with two-factor authentication: ``(secret, recovery codes, tokens)``."""
    client = login("ada")[0]
    return turn_on_two_factor(client, PASSWORD, frozen_clock)


# --------------------------------------------------------------------------- #
# TOTP itself
# --------------------------------------------------------------------------- #

RFC_KEYS = {
    hashlib.sha1: b"12345678901234567890",
    hashlib.sha256: b"12345678901234567890123456789012",
    hashlib.sha512: b"1234567890123456789012345678901234567890123456789012345678901234",
}
# RFC 6238, appendix B: time, then the 8-digit codes for SHA-1, SHA-256 and SHA-512.
RFC_VECTORS = [
    (59, "94287082", "46119246", "90693936"),
    (1111111109, "07081804", "68084774", "25091201"),
    (1111111111, "14050471", "67062674", "99943326"),
    (1234567890, "89005924", "91819424", "93441116"),
    (2000000000, "69279037", "90698825", "38618901"),
    (20000000000, "65353130", "77737706", "47863826"),
]


@pytest.mark.parametrize("moment, sha1, sha256, sha512", RFC_VECTORS)
def test_totp_matches_the_rfc_test_vectors(moment, sha1, sha256, sha512):
    for digest, expected in (
        (hashlib.sha1, sha1),
        (hashlib.sha256, sha256),
        (hashlib.sha512, sha512),
    ):
        assert two_factor.totp(RFC_KEYS[digest], moment, digits=8, digest=digest) == expected


def test_hotp_matches_rfc_4226():
    expected = ["755224", "287082", "359152", "969429", "338314"]
    assert [two_factor.hotp(b"12345678901234567890", i) for i in range(5)] == expected


def test_codes_of_the_neighbouring_time_steps_are_accepted():
    key = two_factor.secret_bytes(two_factor.new_secret())
    now = 1_700_000_015.0
    step = two_factor.time_step(now)
    for offset in (-1, 0, 1):
        code = two_factor.totp(key, now + offset * 30)
        assert two_factor.matching_step(key, code, now) == step + offset
    assert two_factor.matching_step(key, two_factor.totp(key, now - 60), now) is None
    assert two_factor.matching_step(key, two_factor.totp(key, now + 60), now) is None


def test_the_secret_is_stored_encrypted(settings):
    secret = two_factor.new_secret()
    assert len(secret) == 32 and two_factor.secret_bytes(secret)
    stored = two_factor.encrypt(secret)
    assert secret not in stored and stored.startswith("f1:")
    assert two_factor.decrypt(stored) == secret
    # A new secret key with the old one as a fallback still decrypts it; without, nothing.
    old = settings.SECRET_KEY
    settings.SECRET_KEY = "another-secret-key-" + "x" * 40
    assert two_factor.decrypt(stored) is None
    settings.SECRET_KEY_FALLBACKS = [old]
    assert two_factor.decrypt(stored) == secret


def test_otpauth_uri(settings):
    settings.SITE_NAME = "Brick DNN"
    uri = two_factor.otpauth_uri("JBSWY3DPEHPK3PXP", "ada lovelace")
    assert uri.startswith("otpauth://totp/Brick%20DNN%3Aada%20lovelace?")
    assert "secret=JBSWY3DPEHPK3PXP" in uri and "issuer=Brick+DNN" in uri
    assert "digits=6" in uri and "period=30" in uri


# --------------------------------------------------------------------------- #
# Setting it up and turning it off
# --------------------------------------------------------------------------- #


def test_setup_shows_the_secret_once_and_needs_a_code(ada, frozen_clock):  # noqa: F811
    client = login("ada")[0]
    assert client.get("/api/auth/two-factor/").data == {
        "enabled": False,
        "enabled_at": None,
        "recovery_codes_left": 0,
    }
    assert (
        client.post("/api/auth/two-factor/setup/", {"password": "wrong"}, format="json").status_code
        == 400
    )
    response = client.post("/api/auth/two-factor/setup/", {"password": PASSWORD}, format="json")
    assert response.status_code == 200
    assert response["Cache-Control"] == "no-store"
    secret = response.data["secret"]
    assert response.data["otpauth_uri"].startswith("otpauth://totp/")
    # Not on until confirmed: logging in needs no code yet.
    assert "access" in password_login().data
    record = TwoFactorAuth.objects.get(user=ada)
    assert record.confirmed_at is None and secret not in record.secret

    wrong = client.post("/api/auth/two-factor/enable/", {"code": "000000"}, format="json")
    assert wrong.status_code == 400 and "code" in wrong.data
    response = client.post(
        "/api/auth/two-factor/enable/",
        {"code": app_code(secret, frozen_clock.now)},
        format="json",
    )
    assert response.status_code == 200, response.data
    codes = response.data["recovery_codes"]
    assert len(codes) == 10 and len(set(codes)) == 10
    assert response.data["state"]["enabled"] is True
    assert response.data["state"]["recovery_codes_left"] == 10
    # Stored as hashes only, and the secret is never answered again.
    record.refresh_from_db()
    assert not any(code in json.dumps(record.recovery_codes) for code in codes)
    state = bearer(response.data["access"]).get("/api/auth/two-factor/").data
    assert state["enabled"] is True and "secret" not in state
    again = bearer(response.data["access"]).post(
        "/api/auth/two-factor/setup/", {"password": PASSWORD}, format="json"
    )
    assert again.status_code == 400


def test_turning_it_on_ends_other_logins(ada, frozen_clock):  # noqa: F811
    client, other = login("ada")
    _, _, tokens = turn_on_two_factor(client, PASSWORD, frozen_clock)
    refreshed = APIClient().post(
        "/api/auth/token/refresh/", {"refresh": other["refresh"]}, format="json"
    )
    assert refreshed.status_code == 401
    assert bearer(tokens["access"]).get("/api/auth/account/").status_code == 200


def test_a_new_setup_replaces_one_that_was_not_confirmed(ada):  # noqa: F811
    client = login("ada")[0]
    first = client.post("/api/auth/two-factor/setup/", {"password": PASSWORD}, format="json")
    second = client.post("/api/auth/two-factor/setup/", {"password": PASSWORD}, format="json")
    assert first.data["secret"] != second.data["secret"]
    assert TwoFactorAuth.objects.filter(user=ada).count() == 1
    assert secret_of(ada) == second.data["secret"]


def test_turning_it_off_needs_password_and_code(ada, ada_2fa, frozen_clock):  # noqa: F811
    secret, codes, tokens = ada_2fa
    client = bearer(tokens["access"])
    code = app_code(secret, frozen_clock.now)
    bad_password = client.post(
        "/api/auth/two-factor/disable/", {"password": "nope", "code": code}, format="json"
    )
    assert bad_password.status_code == 400 and "password" in bad_password.data
    bad_code = client.post(
        "/api/auth/two-factor/disable/", {"password": PASSWORD, "code": "123456"}, format="json"
    )
    assert bad_code.status_code == 400 and "code" in bad_code.data
    assert two_factor.is_enabled(ada)
    response = client.post(
        "/api/auth/two-factor/disable/", {"password": PASSWORD, "code": code}, format="json"
    )
    assert response.status_code == 200 and response.data["enabled"] is False
    assert not TwoFactorAuth.objects.filter(user=ada).exists()
    assert "access" in password_login().data


def test_recovery_codes_can_be_replaced(ada, ada_2fa, frozen_clock):  # noqa: F811
    secret, codes, tokens = ada_2fa
    client = bearer(tokens["access"])
    response = client.post(
        "/api/auth/two-factor/recovery-codes/",
        {"password": PASSWORD, "code": codes[0]},
        format="json",
    )
    assert response.status_code == 200
    new_codes = response.data["recovery_codes"]
    assert len(new_codes) == 10 and not set(new_codes) & set(codes)
    challenge = password_login().data["challenge"]
    assert second_step(challenge, codes[1]).status_code == 400
    assert second_step(challenge, new_codes[0]).status_code == 200


def test_api_tokens_cannot_manage_it(ada):  # noqa: F811
    from brick_dnn_core.api_tokens import create_token

    _, key = create_token(ada, "script")
    client = bearer(key)
    assert client.get("/api/auth/two-factor/").status_code == 403
    response = client.post("/api/auth/two-factor/setup/", {"password": PASSWORD}, format="json")
    assert response.status_code == 403


# --------------------------------------------------------------------------- #
# Logging in
# --------------------------------------------------------------------------- #


def test_login_needs_the_code_in_a_second_step(ada, ada_2fa, frozen_clock):  # noqa: F811
    from rest_framework_simplejwt.token_blacklist.models import OutstandingToken

    secret, _, _ = ada_2fa
    issued = OutstandingToken.objects.count()
    response = password_login()
    assert response.status_code == 200
    assert response.data["two_factor_required"] is True
    assert "access" not in response.data and "refresh" not in response.data
    assert OutstandingToken.objects.count() == issued  # no tokens yet
    # A wrong password does not get that far.
    assert password_login(password="wrong-password").status_code == 401

    challenge = response.data["challenge"]
    wrong = second_step(challenge, "000000")
    assert wrong.status_code == 400 and "access" not in wrong.data
    done = second_step(challenge, app_code(secret, frozen_clock.now))
    assert done.status_code == 200, done.data
    assert bearer(done.data["access"]).get("/api/auth/me/").data["username"] == "ada"
    assert (
        APIClient()
        .post("/api/auth/token/refresh/", {"refresh": done.data["refresh"]}, format="json")
        .status_code
        == 200
    )


def test_a_code_works_only_once(ada, ada_2fa, frozen_clock):  # noqa: F811
    secret, _, _ = ada_2fa
    code = app_code(secret, frozen_clock.now)
    assert second_step(password_login().data["challenge"], code).status_code == 200
    replay = second_step(password_login().data["challenge"], code)
    assert replay.status_code == 400
    # Nor an older one, once a newer one was used.
    frozen_clock.move(30)
    newer = app_code(secret, frozen_clock.now)
    older = app_code(secret, frozen_clock.now, steps=-1)
    assert second_step(password_login().data["challenge"], newer).status_code == 200
    assert second_step(password_login().data["challenge"], older).status_code == 400


def test_one_time_step_of_clock_drift_is_tolerated(ada, ada_2fa, frozen_clock):  # noqa: F811
    secret, _, _ = ada_2fa
    frozen_clock.move(60)  # the code used for the setup is older than the window
    late = app_code(secret, frozen_clock.now, steps=-1)
    early = app_code(secret, frozen_clock.now, steps=1)
    far = app_code(secret, frozen_clock.now, steps=3)
    assert second_step(password_login().data["challenge"], far).status_code == 400
    assert second_step(password_login().data["challenge"], late).status_code == 200
    assert second_step(password_login().data["challenge"], early).status_code == 200


def test_recovery_codes_work_once(ada, ada_2fa):  # noqa: F811
    _, codes, _ = ada_2fa
    challenge = password_login().data["challenge"]
    # Any spelling: upper case, without the dashes.
    response = second_step(challenge, codes[0].upper().replace("-", ""))
    assert response.status_code == 200 and response.data["recovery_codes_left"] == 9
    assert second_step(password_login().data["challenge"], codes[0]).status_code == 400
    assert two_factor.state(ada)["recovery_codes_left"] == 9


def test_wrong_codes_lock_the_account_for_a_while(ada, ada_2fa, frozen_clock):  # noqa: F811
    secret, _, _ = ada_2fa
    challenge = password_login().data["challenge"]
    for _ in range(two_factor.MAX_FAILURES):
        assert second_step(challenge, "000000").status_code == 400
    locked = second_step(challenge, app_code(secret, frozen_clock.now))
    assert locked.status_code == 429
    assert "quarter of an hour" in str(locked.data["detail"])


def test_the_second_step_is_rate_limited_per_address(ada, ada_2fa, settings, monkeypatch):  # noqa: F811
    from rest_framework.throttling import ScopedRateThrottle

    monkeypatch.setitem(ScopedRateThrottle.THROTTLE_RATES, "two_factor", "2/hour")
    challenge = password_login().data["challenge"]
    assert second_step(challenge, "000000").status_code == 400
    assert second_step(challenge, "000000").status_code == 400
    assert second_step(challenge, "000000").status_code == 429


def test_challenges_expire_and_end_with_a_password_change(ada, ada_2fa, frozen_clock, monkeypatch):  # noqa: F811
    secret, _, _ = ada_2fa
    assert second_step("forged", "123456").data["code"] == "challenge_expired"
    challenge = password_login().data["challenge"]
    ada.set_password("a-completely-new-password")
    ada.save()
    response = second_step(challenge, app_code(secret, frozen_clock.now))
    assert response.status_code == 400 and response.data["code"] == "challenge_expired"

    challenge = password_login(password="a-completely-new-password").data["challenge"]
    monkeypatch.setattr(two_factor, "CHALLENGE_MAX_AGE", -1)
    response = second_step(challenge, app_code(secret, frozen_clock.now))
    assert response.data["code"] == "challenge_expired"


def test_login_and_the_secret_are_not_logged(ada, frozen_clock, caplog):  # noqa: F811
    client = login("ada")[0]
    with caplog.at_level("DEBUG"):
        secret, codes, _ = turn_on_two_factor(client, PASSWORD, frozen_clock)
        challenge = password_login().data["challenge"]
        second_step(challenge, codes[0])
    assert secret not in caplog.text
    assert not any(code in caplog.text for code in codes)


# --------------------------------------------------------------------------- #
# The OAuth consent page and API tokens
# --------------------------------------------------------------------------- #


def test_oauth_consent_needs_the_two_step_login(ada, ada_2fa, frozen_clock):  # noqa: F811
    from .test_oauth import authorize_query, pkce, register

    secret, _, tokens = ada_2fa
    app = register()
    query = authorize_query(app["client_id"], pkce()[1])
    # Without a complete login there is no consent page; the password alone gives no token.
    assert APIClient().get(f"/api/oauth/authorize/?{query}").status_code == 401
    assert "access" not in password_login().data
    done = second_step(password_login().data["challenge"], app_code(secret, frozen_clock.now))
    browser = bearer(done.data["access"])
    page = browser.get(f"/api/oauth/authorize/?{query}")
    assert page.status_code == 200 and page.data["client"]["name"] == "Claude"
    decision = browser.post(
        "/api/oauth/authorize/", {"query": query, "approve": True}, format="json"
    )
    assert decision.status_code == 200 and "code=" in decision.data["redirect_to"]


def test_api_tokens_keep_working_with_two_factor(ada):  # noqa: F811
    from brick_dnn_core.api_tokens import create_token

    _, key = create_token(ada, "script")
    TwoFactorAuth.objects.create(
        user=ada,
        secret=two_factor.encrypt(two_factor.new_secret()),
        confirmed_at="2026-01-01T00:00Z",
    )
    assert bearer(key).get("/api/projects/").status_code == 200


# --------------------------------------------------------------------------- #
# The Django admin
# --------------------------------------------------------------------------- #


@pytest.fixture
def boss(frozen_clock):
    user = User.objects.create_user("boss", email="boss@example.com", password=PASSWORD)
    user.is_staff = user.is_superuser = True
    user.save()
    return user


def admin_login(username: str, password: str = PASSWORD, code: str = "") -> tuple[Client, object]:
    client = Client()
    response = client.post(
        "/admin/login/?next=/admin/",
        {"username": username, "password": password, "code": code, "next": "/admin/"},
    )
    return client, response


def test_the_admin_login_has_a_code_field(boss):
    response = Client().get("/admin/login/")
    assert response.status_code == 200
    assert b'name="code"' in response.content and b"one-time-code" in response.content


def test_admin_without_two_factor_logs_in_as_before(boss):
    client, response = admin_login("boss")
    assert response.status_code == 302 and response["Location"] == "/admin/"
    assert client.get("/admin/").status_code == 200


def test_admin_with_two_factor_needs_the_code(boss, frozen_clock):
    secret, codes, _ = turn_on_two_factor(login("boss")[0], PASSWORD, frozen_clock)
    client, response = admin_login("boss")
    assert response.status_code == 200
    assert b"Enter the code from your authenticator app" in response.content
    assert client.get("/admin/").status_code == 302  # still logged out

    _, response = admin_login("boss", code="000000")
    assert b"wrong or was used already" in response.content
    client, response = admin_login("boss", code=app_code(secret, frozen_clock.now))
    assert response.status_code == 302
    assert client.get("/admin/").status_code == 200
    # The same code does not work twice; a recovery code does, once.
    _, response = admin_login("boss", code=app_code(secret, frozen_clock.now))
    assert response.status_code == 200
    client, response = admin_login("boss", code=codes[0])
    assert response.status_code == 302
    _, response = admin_login("boss", code=codes[0])
    assert response.status_code == 200


def test_admin_code_failures_count_towards_the_lock(boss, frozen_clock):
    secret, _, _ = turn_on_two_factor(login("boss")[0], PASSWORD, frozen_clock)
    for _ in range(two_factor.MAX_FAILURES):
        admin_login("boss", code="000000")
    _, response = admin_login("boss", code=app_code(secret, frozen_clock.now))
    assert b"Too many wrong codes" in response.content


def test_staff_can_be_required_to_use_two_factor(boss, settings, frozen_clock):
    client, _ = admin_login("boss")  # logged in before the setting was turned on
    settings.REQUIRE_2FA_FOR_STAFF = True
    assert client.get("/admin/").status_code == 302  # the open session loses access
    _, response = admin_login("boss")
    assert response.status_code == 200
    assert b"The admin needs two-factor authentication" in response.content
    assert b"/account" in response.content
    # The web app still works for this account.
    assert login("boss")[0].get("/api/auth/account/").status_code == 200

    secret, _, _ = turn_on_two_factor(login("boss")[0], PASSWORD, frozen_clock)
    client, response = admin_login("boss", code=app_code(secret, frozen_clock.now))
    assert response.status_code == 302
    assert client.get("/admin/").status_code == 200


def test_the_admin_never_shows_the_secret(boss, frozen_clock):
    secret, codes, _ = turn_on_two_factor(login("boss")[0], PASSWORD, frozen_clock)
    client, _ = admin_login("boss", code=codes[0])
    record = TwoFactorAuth.objects.get(user=boss)
    listing = client.get("/admin/brick_dnn_core/twofactorauth/")
    page = client.get(f"/admin/brick_dnn_core/twofactorauth/{record.pk}/change/")
    assert listing.status_code == 200 and page.status_code == 200
    for response in (listing, page):
        assert secret.encode() not in response.content
        assert record.secret.encode() not in response.content


def test_reset_two_factor_command(ada, ada_2fa, email_on):  # noqa: F811
    out = io.StringIO()
    call_command("reset_two_factor", "ada", stdout=out)
    assert "is off for ada" in out.getvalue()
    assert not two_factor.is_enabled(ada)
    assert "access" in password_login().data


# --------------------------------------------------------------------------- #
# Export and deletion
# --------------------------------------------------------------------------- #


def test_export_says_whether_it_is_on_but_not_the_secret(ada, ada_2fa, settings, tmp_path):  # noqa: F811
    settings.TRAINING_DIR = tmp_path / "training"
    settings.DATA_PREP_DIR = tmp_path / "prepare"
    secret, codes, tokens = ada_2fa
    response = bearer(tokens["access"]).get("/api/auth/account/export/")
    archive = zipfile.ZipFile(io.BytesIO(b"".join(response.streaming_content)))
    account = json.loads(archive.read("account.json"))
    assert account["two_factor"]["enabled"] is True
    assert account["two_factor"]["recovery_codes_left"] == 10
    everything = b"".join(archive.read(name) for name in archive.namelist())
    assert secret.encode() not in everything
    assert TwoFactorAuth.objects.get(user=ada).secret.encode() not in everything


def test_deleting_the_account_removes_it(ada, ada_2fa, settings, tmp_path):  # noqa: F811
    settings.TRAINING_DIR = tmp_path / "training"
    settings.DATA_PREP_DIR = tmp_path / "prepare"
    _, _, tokens = ada_2fa
    response = bearer(tokens["access"]).post(
        "/api/auth/account/delete/", {"password": PASSWORD, "username": "ada"}, format="json"
    )
    assert response.status_code == 204
    assert not TwoFactorAuth.objects.exists()
