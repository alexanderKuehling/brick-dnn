"""Personal API tokens: create them, and log requests in with them.

A token looks like ``bdnn_<43 random characters>`` and is sent as
``Authorization: Bearer <token>``, like the JWT access tokens of the web app. Only its
SHA-256 hash is stored: the token is random enough that a plain hash is safe, and a lookup
by hash is fast.
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import timedelta

from django.utils import timezone
from rest_framework import authentication, exceptions, permissions
from rest_framework.request import Request

from .models import ApiToken

PREFIX = "bdnn_"
# Shown in the list of tokens, so users can tell them apart.
VISIBLE_CHARS = 8
MAX_TOKENS_PER_USER = 20
EXPIRY_DAYS = (7, 30, 90, 365)
# last_used_at is written at most this often, not on every request.
LAST_USED_INTERVAL = timedelta(minutes=1)
# POST endpoints that change nothing: read-only tokens may use them.
# The MCP server itself only offers read-only tokens the tools that change nothing.
READ_ONLY_POSTS = {"analyze", "generate", "training-check", "mcp", "mcp-no-slash"}


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def create_token(
    user, name: str, read_only: bool = False, expires_in_days: int | None = None
) -> tuple[ApiToken, str]:
    """Create a token and return it with its secret, which is not stored anywhere."""
    key = PREFIX + secrets.token_urlsafe(32)
    expires_at = timezone.now() + timedelta(days=expires_in_days) if expires_in_days else None
    token = ApiToken.objects.create(
        owner=user,
        name=name,
        prefix=key[: len(PREFIX) + VISIBLE_CHARS],
        key_hash=hash_key(key),
        read_only=read_only,
        expires_at=expires_at,
    )
    return token, key


class ApiTokenAuthentication(authentication.BaseAuthentication):
    """``Authorization: Bearer bdnn_…``. Other bearer tokens are left to the JWT login."""

    keyword = "Bearer"

    def authenticate(self, request: Request):
        header = authentication.get_authorization_header(request).split()
        if len(header) != 2 or header[0].decode(errors="replace") != self.keyword:
            return None
        key = header[1].decode(errors="replace")
        if not key.startswith(PREFIX):
            return None

        token = ApiToken.objects.select_related("owner").filter(key_hash=hash_key(key)).first()
        if token is None:
            raise exceptions.AuthenticationFailed("Invalid API token.")
        now = timezone.now()
        if token.expires_at is not None and token.expires_at <= now:
            raise exceptions.AuthenticationFailed("This API token has expired.")
        if not token.owner.is_active:
            raise exceptions.AuthenticationFailed("This account is disabled.")
        if token.read_only and request.method not in permissions.SAFE_METHODS:
            match = request._request.resolver_match
            if match is None or match.url_name not in READ_ONLY_POSTS:
                raise exceptions.PermissionDenied("This API token is read-only.")

        if token.last_used_at is None or now - token.last_used_at > LAST_USED_INTERVAL:
            ApiToken.objects.filter(pk=token.pk).update(last_used_at=now)
            token.last_used_at = now
        return token.owner, token

    def authenticate_header(self, request: Request) -> str:
        return f'{self.keyword} realm="api"'


def uses_api_token(request: Request) -> bool:
    return isinstance(request.auth, ApiToken)


class IsLoggedInUser(permissions.IsAuthenticated):
    """Logged in with username and password (JWT), not with an API token.

    For managing API tokens: a leaked token must not be able to create more tokens.
    """

    message = "Manage API tokens after logging in on the website; API tokens cannot do this."

    def has_permission(self, request, view) -> bool:
        return super().has_permission(request, view) and not uses_api_token(request)
