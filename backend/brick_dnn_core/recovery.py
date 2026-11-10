"""Verified e-mail addresses and password reset by e-mail (only with e-mail set up, mail.py).

**Verifying an address.** New and changed addresses get a link (valid ``VERIFY_DAYS`` days).
Opening it confirms the address only for the logged-in account it was sent to: the web app
asks to log in first. So a stranger who gets the link because of a typo cannot confirm the
address for someone else's account. The profile stores when and which address was confirmed
(``email_verified_at``, ``email_verified_for``). Unverified addresses still work for logging
in, invitations and everything else.

**Which address may receive password reset links** (:func:`trusted_email`): a verified one,
or the unchanged address of an account that existed before addresses were verified
(``email_legacy``, set by the migration). A link to an address someone typed and never
confirmed could hand the account to whoever owns that address, so new accounts must confirm
theirs first; old accounts could not do that before, and would otherwise have no way back in.
Opening a reset link proves the address, so it counts as verified afterwards. Security notices
(notices.py) use the same rule.

**Password reset.** ``POST /api/auth/password-reset/ {"login"}`` (username or e-mail) always
gives the same answer, whether the account exists or not, and sends at most
``RESET_MAILS_PER_HOUR`` links per account; the endpoint is also limited per IP address
(``password_reset`` rate). The link carries Django's reset token: valid for an hour
(``PASSWORD_RESET_TIMEOUT``), and void once the password changed, the account logged in or the
address changed, so it works once. The new password is checked like at sign-up, and every login
of the account ends (refresh tokens are blacklisted). Two-factor authentication stays on.
"""

from __future__ import annotations

import logging
from typing import Any

from django.contrib.auth import get_user_model
from django.contrib.auth.password_validation import validate_password
from django.contrib.auth.tokens import default_token_generator
from django.core import signing
from django.core.cache import cache
from django.core.exceptions import ValidationError as DjangoValidationError
from django.utils import timezone
from django.utils.encoding import force_bytes, force_str
from django.utils.http import urlsafe_base64_decode, urlsafe_base64_encode
from rest_framework import serializers, status
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.throttling import ScopedRateThrottle
from rest_framework.views import APIView

from . import mail
from .accounts import AccountApiView, log_out_everywhere
from .models import UserProfile

logger = logging.getLogger(__name__)
User = get_user_model()

VERIFY_DAYS = 7
VERIFY_SALT = "brickdnn.verify-email"
RESET_MAILS_PER_HOUR = 3

UNAVAILABLE = (
    "Password reset is not available on this server: it sends no e-mail. Ask its operator for help."
)
RESET_SENT = (
    "If an account with this username or e-mail address exists and its e-mail address may "
    "receive reset links, we sent a link to it. It is valid for one hour."
)
INVALID_LINK = "This link is invalid or has expired. Ask for a new one."


# --------------------------------------------------------------------------- #
# The state of an address
# --------------------------------------------------------------------------- #


def _profile(user) -> UserProfile | None:
    try:
        return user.profile
    except UserProfile.DoesNotExist:
        return None


def is_verified(user, profile: UserProfile | None = None) -> bool:
    profile = profile or _profile(user)
    return bool(
        user.email
        and profile is not None
        and profile.email_verified_at is not None
        and profile.email_verified_for.lower() == user.email.lower()
    )


def trusted_email(user) -> str | None:
    """The address that may receive reset links and security notices, or None."""
    if not user.email:
        return None
    profile = _profile(user)
    if is_verified(user, profile) or (profile is not None and profile.email_legacy):
        return user.email
    return None


def email_state(user) -> dict[str, Any]:
    """The e-mail fields of ``Account`` (frontend/src/api/types.ts)."""
    profile = _profile(user)
    verified = is_verified(user, profile)
    return {
        "email_verified": verified,
        "email_verified_at": profile.email_verified_at if verified and profile else None,
        "email_sending": mail.enabled(),
        "password_reset_available": mail.enabled() and trusted_email(user) is not None,
    }


def mark_verified(user) -> None:
    profile, _ = UserProfile.objects.update_or_create(
        user=user,
        defaults={"email_verified_at": timezone.now(), "email_verified_for": user.email},
    )
    user.profile = profile  # what user.profile gives from now on


# --------------------------------------------------------------------------- #
# Verification links
# --------------------------------------------------------------------------- #


def verification_token(user) -> str:
    return signing.dumps({"u": user.pk, "e": user.email.lower()}, salt=VERIFY_SALT)


def send_verification(user, request=None) -> bool:
    from . import notices

    if not user.email or not mail.enabled():
        return False
    url = mail.link(f"/verify-email?token={verification_token(user)}", request)
    return notices.verify_email(user, url, request)


def after_signup(user, request=None) -> None:
    send_verification(user, request)


def email_changed(user, old_address: str, old_trusted: bool, request=None) -> None:
    """After ``user.email`` was changed from ``old_address`` on the account page
    (``old_trusted``: whether that address was trusted_email)."""
    from . import notices

    if old_address.lower() == user.email.lower():
        return
    # A new address: unverified, and no reset links until it is confirmed.
    UserProfile.objects.update_or_create(
        user=user,
        defaults={"email_legacy": False, "email_verified_at": None, "email_verified_for": ""},
    )
    user.refresh_from_db()  # drops the cached profile
    send_verification(user, request)
    if old_address and old_trusted:
        notices.email_changed(user, old_address, request)


class TokenSerializer(serializers.Serializer):
    token = serializers.CharField(max_length=500)


class VerifyEmailView(AccountApiView):
    """``POST {"token"}``: confirm the address with the link from the e-mail. Only the
    account it was sent to can do it (logged in)."""

    def post(self, request: Request) -> Response:
        serializer = TokenSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        try:
            data = signing.loads(
                serializer.validated_data["token"],
                salt=VERIFY_SALT,
                max_age=VERIFY_DAYS * 24 * 3600,
            )
        except signing.BadSignature:
            return _error("This link is invalid or has expired. Send a new one from your account.")
        user = request.user
        if data.get("u") != user.pk:
            return _error(
                "This link is for another account. Log in with the account it was sent to.",
                code="other_account",
            )
        if not user.email or data.get("e") != user.email.lower():
            return _error(
                "This link is for an e-mail address the account does not use any more.",
                code="other_address",
            )
        if not is_verified(user):
            mark_verified(user)
        return Response({"email": user.email, **email_state(user)})


class ResendVerificationView(AccountApiView):
    """``POST``: send the verification link again (``email_verify`` rate per account)."""

    throttle_scope = "email_verify"

    def post(self, request: Request) -> Response:
        user = request.user
        if not mail.enabled():
            return Response(
                {"detail": "This server sends no e-mail, so addresses cannot be confirmed."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        if not user.email:
            return _error("Add an e-mail address first.")
        if is_verified(user):
            return Response({"detail": "Your e-mail address is confirmed already."})
        send_verification(user, request)
        return Response({"detail": f"We sent a link to {user.email}."})


def _error(detail: str, code: str = "invalid_link") -> Response:
    return Response({"detail": detail, "code": code}, status=status.HTTP_400_BAD_REQUEST)


# --------------------------------------------------------------------------- #
# Password reset
# --------------------------------------------------------------------------- #


def find_account(login: str):
    """The active account with this username, or else with this e-mail address."""
    login = login.strip()
    if not login:
        return None
    users = User.objects.filter(is_active=True)
    user = users.filter(username__iexact=login).first()
    if user is None and "@" in login:
        matches = list(users.filter(email__iexact=login)[:2])
        user = matches[0] if len(matches) == 1 else None
    return user


def _reset_allowed(user) -> bool:
    """At most RESET_MAILS_PER_HOUR links per account and hour."""
    key = f"brickdnn:password-reset:{user.pk}"
    if cache.add(key, 1, 3600):
        return True
    try:
        return cache.incr(key) <= RESET_MAILS_PER_HOUR
    except ValueError:
        return cache.add(key, 1, 3600)


def reset_link(user, request=None) -> str:
    uid = urlsafe_base64_encode(force_bytes(user.pk))
    token = default_token_generator.make_token(user)
    return mail.link(f"/reset-password?uid={uid}&token={token}", request)


def request_reset(login: str, request=None) -> bool:
    """Send a reset link if the account may get one. Returns whether one was sent."""
    from . import notices

    user = find_account(login)
    if user is None or trusted_email(user) is None or not _reset_allowed(user):
        return False
    return notices.password_reset(user, reset_link(user, request), request)


def user_from_link(uid: Any, token: Any):
    """The account a reset link is for, if the link is still valid."""
    if not isinstance(uid, str) or not isinstance(token, str) or len(token) > 100:
        return None
    try:
        pk = force_str(urlsafe_base64_decode(uid))
        user = User.objects.filter(pk=int(pk), is_active=True).first()
    except (ValueError, TypeError, OverflowError):
        return None
    if user is None or not default_token_generator.check_token(user, token):
        return None
    return user


class ResetRequestSerializer(serializers.Serializer):
    login = serializers.CharField(max_length=254)


class PasswordResetView(APIView):
    """``GET``: ``{"available"}``, whether this server can send reset links. ``POST
    {"login"}``: send a link to the account with this username or e-mail address. The
    answer is the same whether such an account exists or not."""

    throttle_scope = "password_reset"

    def get_throttles(self):
        throttles = super().get_throttles()
        if self.request.method == "POST":
            throttles.append(ScopedRateThrottle())
        return throttles

    def get(self, request: Request) -> Response:
        return Response({"available": mail.enabled()})

    def post(self, request: Request) -> Response:
        if not mail.enabled():
            return Response({"detail": UNAVAILABLE}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        serializer = ResetRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        request_reset(serializer.validated_data["login"], request)
        return Response({"detail": RESET_SENT})


class ResetConfirmSerializer(serializers.Serializer):
    uid = serializers.CharField(max_length=40)
    token = serializers.CharField(max_length=100)
    new_password = serializers.CharField(write_only=True)


class PasswordResetConfirmView(APIView):
    """``GET ?uid=&token=``: ``{"valid", "username"}``. ``POST {"uid", "token",
    "new_password"}``: set the new password; all logins of the account end."""

    throttle_classes = [ScopedRateThrottle]
    throttle_scope = "auth"

    def get(self, request: Request) -> Response:
        user = user_from_link(request.query_params.get("uid"), request.query_params.get("token"))
        return Response(
            {"valid": user is not None, "username": user.get_username() if user else None}
        )

    def post(self, request: Request) -> Response:
        serializer = ResetConfirmSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        user = user_from_link(data["uid"], data["token"])
        if user is None:
            return _error(INVALID_LINK)
        try:
            validate_password(data["new_password"], user=user)
        except DjangoValidationError as exc:
            return Response({"new_password": list(exc.messages)}, status=400)
        user.set_password(data["new_password"])
        user.save(update_fields=["password"])
        log_out_everywhere(user)
        if not is_verified(user):
            mark_verified(user)  # the link reached this address
        logger.info("Account #%s reset its password with an e-mail link.", user.pk)
        return Response(
            {
                "detail": "Your password was changed. Log in with the new one.",
                "username": user.get_username(),
            }
        )
