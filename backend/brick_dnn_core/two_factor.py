"""Two-factor authentication with an authenticator app (TOTP, RFC 6238) and recovery codes.

Optional for every account, set up on the account page: the server makes a secret, the app
shows it as a QR code (the ``otpauth://`` URI), the user types a code to confirm and gets ten
recovery codes, once. From then on:

- **Logging in** (``/api/auth/token/``) with the right password answers
  ``{"two_factor_required": true, "challenge": …}`` instead of tokens. The challenge is signed,
  valid for five minutes and ends with a password change; ``/api/auth/token/two-factor/``
  takes it with a 6-digit code or a recovery code and only then issues the JWT pair. The OAuth
  consent page uses this website login, so connecting an app needs the code too; API tokens
  and connected apps keep working (they were created after such a login).
- **The Django admin** asks for the code on its login page (admin_site.py). With
  ``BRICKDNN_REQUIRE_2FA_FOR_STAFF`` staff accounts need two-factor authentication for it.

Codes are checked with one time step (30 s) of tolerance either way, and each code works only
once (the last used time step is stored). After ``MAX_FAILURES`` wrong codes an account takes
no codes for ``FAILURE_WINDOW`` seconds. The secret is stored encrypted (Fernet, with a key
derived from ``DJANGO_SECRET_KEY``; ``SECRET_KEY_FALLBACKS`` keep working while it is rotated)
and never shown again after the setup; recovery codes are stored as SHA-256 hashes (80 random
bits each) and work once. Neither is ever logged. Turning it off needs the password and a code.
``manage.py reset_two_factor <username>`` turns it off for an account that lost both.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import struct
import time
from typing import Any
from urllib.parse import quote, urlencode

from cryptography.fernet import Fernet, InvalidToken
from django.conf import settings
from django.contrib.auth import get_user_model
from django.contrib.auth.models import update_last_login
from django.core import signing
from django.core.cache import cache
from django.db import transaction
from django.utils import timezone
from django.utils.crypto import salted_hmac
from rest_framework import serializers, status
from rest_framework.exceptions import Throttled
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.throttling import ScopedRateThrottle
from rest_framework.views import APIView

from . import notices
from .accounts import AccountApiView, log_out_everywhere, tokens_for
from .models import TwoFactorAuth
from .retention import record_activity

User = get_user_model()

DIGITS = 6
PERIOD = 30
# Codes of one time step before and after the current one are accepted (clocks drift).
WINDOW = 1
RECOVERY_CODES = 10
CHALLENGE_SALT = "brickdnn.two-factor-login"
CHALLENGE_MAX_AGE = 300  # seconds
MAX_FAILURES = 5
FAILURE_WINDOW = 15 * 60  # seconds
ENCRYPTED_PREFIX = "f1:"


# --------------------------------------------------------------------------- #
# TOTP (RFC 6238 on top of HOTP, RFC 4226)
# --------------------------------------------------------------------------- #


def hotp(key: bytes, counter: int, digits: int = DIGITS, digest: Any = hashlib.sha1) -> str:
    mac = hmac.new(key, struct.pack(">Q", counter), digest).digest()
    offset = mac[-1] & 0x0F
    number = struct.unpack(">I", mac[offset : offset + 4])[0] & 0x7FFFFFFF
    return str(number % 10**digits).zfill(digits)


# The clock codes are checked against (tests replace it).
clock = time.time


def time_step(moment: float | None = None) -> int:
    return int((clock() if moment is None else moment) // PERIOD)


def totp(key: bytes, moment: float | None = None, digits: int = DIGITS, digest=hashlib.sha1):
    return hotp(key, time_step(moment), digits, digest)


def matching_step(key: bytes, code: str, moment: float | None = None) -> int | None:
    """The time step ``code`` belongs to (within WINDOW of now), or None."""
    current = time_step(moment)
    for step in range(current - WINDOW, current + WINDOW + 1):
        if hmac.compare_digest(hotp(key, step), code):
            return step
    return None


def new_secret() -> str:
    """A random secret of 160 bits in Base32, as authenticator apps expect it."""
    return base64.b32encode(secrets.token_bytes(20)).decode().rstrip("=")


def secret_bytes(secret: str) -> bytes:
    padded = secret.upper() + "=" * (-len(secret) % 8)
    return base64.b32decode(padded)


def otpauth_uri(secret: str, username: str) -> str:
    """The URI authenticator apps read from the QR code."""
    issuer = settings.SITE_NAME
    label = quote(f"{issuer}:{username}", safe="")
    query = urlencode(
        {
            "secret": secret,
            "issuer": issuer,
            "algorithm": "SHA1",
            "digits": DIGITS,
            "period": PERIOD,
        }
    )
    return f"otpauth://totp/{label}?{query}"


# --------------------------------------------------------------------------- #
# Storing the secret and the recovery codes
# --------------------------------------------------------------------------- #


def _fernet(key: str) -> Fernet:
    digest = hashlib.sha256(b"brickdnn.two-factor-secret:" + key.encode()).digest()
    return Fernet(base64.urlsafe_b64encode(digest))


def encrypt(secret: str) -> str:
    return ENCRYPTED_PREFIX + _fernet(settings.SECRET_KEY).encrypt(secret.encode()).decode()


def decrypt(stored: str) -> str | None:
    """The secret, or None if no secret key of this server can decrypt it."""
    if not stored.startswith(ENCRYPTED_PREFIX):
        return None
    token = stored[len(ENCRYPTED_PREFIX) :].encode()
    for key in [settings.SECRET_KEY, *getattr(settings, "SECRET_KEY_FALLBACKS", [])]:
        try:
            return _fernet(key).decrypt(token).decode()
        except InvalidToken:
            continue
    return None


def _normalize(code: str) -> str:
    return "".join(ch for ch in str(code).lower() if ch.isalnum())


def hash_recovery_code(code: str) -> str:
    return hashlib.sha256(_normalize(code).encode()).hexdigest()


def new_recovery_codes() -> list[str]:
    """Ten codes like ``k3vq-7xmp-2hcd-9wtn`` (16 Base32 characters, 80 bits)."""
    codes = []
    for _ in range(RECOVERY_CODES):
        raw = base64.b32encode(secrets.token_bytes(10)).decode().lower()
        codes.append("-".join(raw[i : i + 4] for i in range(0, 16, 4)))
    return codes


# --------------------------------------------------------------------------- #
# State of an account, checking codes
# --------------------------------------------------------------------------- #


def confirmed(user) -> TwoFactorAuth | None:
    if user is None or not getattr(user, "pk", None):
        return None
    return TwoFactorAuth.objects.filter(user=user, confirmed_at__isnull=False).first()


def is_enabled(user) -> bool:
    return confirmed(user) is not None


def state(user) -> dict[str, Any]:
    """``TwoFactorState`` (frontend/src/api/types.ts)."""
    record = confirmed(user)
    return {
        "enabled": record is not None,
        "enabled_at": record.confirmed_at if record else None,
        "recovery_codes_left": len(record.recovery_codes) if record else 0,
    }


class TooManyAttempts(Exception):
    """The account took MAX_FAILURES wrong codes within FAILURE_WINDOW."""

    message = "Too many wrong codes. Wait a quarter of an hour and try again."


def _failures_key(user) -> str:
    return f"brickdnn:2fa-failures:{user.pk}"


def _check_attempts(user) -> None:
    if (cache.get(_failures_key(user)) or 0) >= MAX_FAILURES:
        raise TooManyAttempts()


def _failed(user) -> None:
    key = _failures_key(user)
    if not cache.add(key, 1, FAILURE_WINDOW):
        try:
            cache.incr(key)
        except ValueError:  # expired in between
            cache.add(key, 1, FAILURE_WINDOW)


def verify(user, code: str, allow_recovery: bool = True, pending: bool = False) -> str | None:
    """Check ``code`` for ``user``: ``"totp"`` or ``"recovery"`` if it is right (and use it
    up), else None. ``pending``: check against a secret that is being set up. Raises
    TooManyAttempts while the account is locked."""
    _check_attempts(user)
    code = _normalize(code)
    used = None
    with transaction.atomic():
        records = TwoFactorAuth.objects.select_for_update().filter(user=user)
        record = records.filter(confirmed_at__isnull=pending).first()
        if record is not None and len(code) == DIGITS and code.isdigit():
            secret = decrypt(record.secret)
            step = matching_step(secret_bytes(secret), code) if secret else None
            if step is not None and step > record.last_used_step:
                record.last_used_step = step
                record.save(update_fields=["last_used_step"])
                used = "totp"
        elif record is not None and allow_recovery and not pending and len(code) == 16:
            hashed = hash_recovery_code(code)
            if hashed in record.recovery_codes:
                record.recovery_codes = [h for h in record.recovery_codes if h != hashed]
                record.save(update_fields=["recovery_codes"])
                used = "recovery"
    if used is None:
        _failed(user)
    return used


def start_setup(user) -> tuple[str, str]:
    """A new secret for ``user`` (replacing one that was not confirmed): ``(secret, uri)``."""
    secret = new_secret()
    TwoFactorAuth.objects.filter(user=user, confirmed_at__isnull=True).delete()
    TwoFactorAuth.objects.create(user=user, secret=encrypt(secret))
    return secret, otpauth_uri(secret, user.get_username())


def enable(user, code: str) -> list[str] | None:
    """Switch it on if ``code`` matches the secret being set up: the recovery codes, or None."""
    if verify(user, code, pending=True) is None:
        return None
    codes = new_recovery_codes()
    TwoFactorAuth.objects.filter(user=user, confirmed_at__isnull=True).update(
        confirmed_at=timezone.now(), recovery_codes=[hash_recovery_code(c) for c in codes]
    )
    return codes


def regenerate_recovery_codes(user) -> list[str]:
    codes = new_recovery_codes()
    TwoFactorAuth.objects.filter(user=user, confirmed_at__isnull=False).update(
        recovery_codes=[hash_recovery_code(c) for c in codes]
    )
    return codes


def disable(user) -> bool:
    deleted, _ = TwoFactorAuth.objects.filter(user=user).delete()
    return deleted > 0


# --------------------------------------------------------------------------- #
# Logging in in two steps
# --------------------------------------------------------------------------- #


def _password_fingerprint(user) -> str:
    # A challenge ends when the password changes.
    return salted_hmac(CHALLENGE_SALT, user.password).hexdigest()[:16]


def make_challenge(user) -> str:
    return signing.dumps({"u": user.pk, "p": _password_fingerprint(user)}, salt=CHALLENGE_SALT)


def read_challenge(challenge: Any):
    """The active account a challenge was made for, or None (unknown, expired, changed)."""
    if not isinstance(challenge, str) or len(challenge) > 500:
        return None
    try:
        data = signing.loads(challenge, salt=CHALLENGE_SALT, max_age=CHALLENGE_MAX_AGE)
    except signing.BadSignature:
        return None
    user = User.objects.filter(pk=data.get("u"), is_active=True).first()
    if user is None or not hmac.compare_digest(_password_fingerprint(user), str(data.get("p"))):
        return None
    return user


def challenge_response(user) -> dict[str, Any]:
    return {
        "two_factor_required": True,
        "challenge": make_challenge(user),
        "expires_in": CHALLENGE_MAX_AGE,
    }


def issue_login(user) -> dict[str, str]:
    """The JWT pair of a completed website login."""
    tokens = tokens_for(user)
    update_last_login(None, user)
    record_activity(user.pk)
    return tokens


class TwoFactorLoginSerializer(serializers.Serializer):
    challenge = serializers.CharField()
    code = serializers.CharField(max_length=40)


class TwoFactorLoginView(APIView):
    """``POST {"challenge", "code"}``: the second step of a login with two-factor
    authentication. ``code``: 6 digits from the app, or a recovery code. Answers
    ``{"access", "refresh"}``."""

    throttle_classes = [ScopedRateThrottle]
    throttle_scope = "two_factor"

    def post(self, request: Request) -> Response:
        serializer = TwoFactorLoginSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = read_challenge(serializer.validated_data["challenge"])
        if user is None or not is_enabled(user):
            return Response(
                {"detail": "The login has expired. Log in again.", "code": "challenge_expired"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        try:
            used = verify(user, serializer.validated_data["code"])
        except TooManyAttempts as exc:
            raise Throttled(detail=exc.message) from None
        if used is None:
            return Response(
                {"code": ["The code is wrong or was used already."]},
                status=status.HTTP_400_BAD_REQUEST,
            )
        response = issue_login(user)
        if used == "recovery":
            response["recovery_codes_left"] = state(user)["recovery_codes_left"]
        return Response(response)


# --------------------------------------------------------------------------- #
# Setting it up on the account page
# --------------------------------------------------------------------------- #


def _check_password(user, value: str) -> str:
    if not user.check_password(value):
        raise serializers.ValidationError("The password is wrong.")
    return value


class PasswordSerializer(serializers.Serializer):
    password = serializers.CharField(write_only=True)

    def validate_password(self, value: str) -> str:
        return _check_password(self.context["user"], value)


class CodeSerializer(serializers.Serializer):
    code = serializers.CharField(max_length=40)


class PasswordAndCodeSerializer(PasswordSerializer, CodeSerializer):
    pass


def _code_or_error(user, code: str, **kwargs) -> str:
    try:
        used = verify(user, code, **kwargs)
    except TooManyAttempts as exc:
        raise Throttled(detail=exc.message) from None
    if used is None:
        raise serializers.ValidationError({"code": ["The code is wrong or was used already."]})
    return used


class TwoFactorView(AccountApiView):
    """``GET``: whether two-factor authentication is on (``TwoFactorState``)."""

    unthrottled_methods = ("GET", "HEAD", "OPTIONS")

    def get(self, request: Request) -> Response:
        return Response(state(request.user))


class TwoFactorSetupView(AccountApiView):
    """``POST {"password"}``: start the setup: ``{"secret", "otpauth_uri"}`` (shown once)."""

    def post(self, request: Request) -> Response:
        user = request.user
        PasswordSerializer(data=request.data, context={"user": user}).is_valid(raise_exception=True)
        if is_enabled(user):
            return Response(
                {"detail": "Two-factor authentication is on already."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        secret, uri = start_setup(user)
        response = Response({"secret": secret, "otpauth_uri": uri})
        response["Cache-Control"] = "no-store"
        return response


class TwoFactorEnableView(AccountApiView):
    """``POST {"code"}``: confirm the setup with a code from the app. Answers the recovery
    codes (shown once) and new tokens for this session: every other login ends."""

    def post(self, request: Request) -> Response:
        user = request.user
        serializer = CodeSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        if is_enabled(user):
            return Response(
                {"detail": "Two-factor authentication is on already."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if not TwoFactorAuth.objects.filter(user=user, confirmed_at__isnull=True).exists():
            return Response(
                {"detail": "Start the setup first."}, status=status.HTTP_400_BAD_REQUEST
            )
        try:
            codes = enable(user, serializer.validated_data["code"])
        except TooManyAttempts as exc:
            raise Throttled(detail=exc.message) from None
        if codes is None:
            raise serializers.ValidationError({"code": ["The code is wrong. Check the app."]})
        log_out_everywhere(user)
        notices.two_factor_changed(user, enabled=True, request=request)
        response = Response(
            {"recovery_codes": codes, "state": state(user), **issue_login(user)},
            status=status.HTTP_200_OK,
        )
        response["Cache-Control"] = "no-store"
        return response


class TwoFactorDisableView(AccountApiView):
    """``POST {"password", "code"}``: turn it off (a recovery code works too)."""

    def post(self, request: Request) -> Response:
        user = request.user
        serializer = PasswordAndCodeSerializer(data=request.data, context={"user": user})
        serializer.is_valid(raise_exception=True)
        if not is_enabled(user):
            return Response(state(user))
        _code_or_error(user, serializer.validated_data["code"])
        disable(user)
        notices.two_factor_changed(user, enabled=False, request=request)
        return Response(state(user))


class RecoveryCodesView(AccountApiView):
    """``POST {"password", "code"}``: new recovery codes (the old ones stop working)."""

    def post(self, request: Request) -> Response:
        user = request.user
        serializer = PasswordAndCodeSerializer(data=request.data, context={"user": user})
        serializer.is_valid(raise_exception=True)
        if not is_enabled(user):
            return Response(
                {"detail": "Two-factor authentication is off."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        _code_or_error(user, serializer.validated_data["code"])
        response = Response(
            {"recovery_codes": regenerate_recovery_codes(user), "state": state(user)}
        )
        response["Cache-Control"] = "no-store"
        return response
