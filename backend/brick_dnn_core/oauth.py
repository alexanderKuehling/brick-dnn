"""OAuth 2.1 sign-in for the MCP server, as the MCP authorization spec describes it.

Apps such as the Claude connectors register themselves (dynamic client registration,
RFC 7591), find the endpoints through the metadata documents (RFC 9728, RFC 8414) and send
the user to ``/oauth/authorize``: a page of the web app, where the logged-in user allows or
denies access. The app then exchanges the code (with PKCE) for an access token, which is an
:class:`ApiToken` tied to an :class:`OAuthConnection` and only valid for the MCP server, and a
refresh token. Users see and disconnect their connected apps on the API tokens page.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import secrets
from datetime import timedelta
from typing import Any
from urllib.parse import parse_qsl, unquote_plus, urlencode, urlsplit, urlunsplit

from django.conf import settings
from django.db import transaction
from django.utils import timezone
from rest_framework import mixins, serializers, status, viewsets
from rest_framework.parsers import FormParser, JSONParser
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.throttling import ScopedRateThrottle
from rest_framework.views import APIView

from . import notices
from .api_tokens import (
    PREFIX,
    VISIBLE_CHARS,
    ApiTokenAuthentication,
    IsLoggedInUser,
    hash_key,
)
from .budgets import token_usage
from .models import ApiToken, OAuthClient, OAuthCode, OAuthConnection
from .serializers import BudgetSerializer, UsageMixin

SCOPES = ("read", "write")  # write includes read
ACCESS_TOKEN_LIFETIME = timedelta(hours=1)
REFRESH_TOKEN_LIFETIME = timedelta(days=60)  # from the last refresh
CODE_LIFETIME = timedelta(minutes=5)
REFRESH_PREFIX = "bdnr_"
MAX_REDIRECT_URIS = 10
# Registered apps that were never used are removed after this long.
UNUSED_CLIENT_LIFETIME = timedelta(days=30)
MCP_PATH = "/api/mcp"


# --------------------------------------------------------------------------- #
# URLs and metadata
# --------------------------------------------------------------------------- #


def site_url(request: Request) -> str:
    """The public address of this site, e.g. ``https://brickdnn.com``: the OAuth issuer."""
    configured = getattr(settings, "PUBLIC_URL", "")
    return configured or f"{request.scheme}://{request.get_host()}"


def resource_metadata_url(request: Request) -> str:
    """Where the metadata of the MCP server lives, for the path the client used."""
    path = request.path if request.path.startswith(MCP_PATH) else MCP_PATH
    return f"{site_url(request)}/.well-known/oauth-protected-resource{path}"


class McpAuthentication(ApiTokenAuthentication):
    """API tokens, and a 401 that tells MCP clients where to sign in (RFC 9728)."""

    def authenticate_header(self, request: Request) -> str:
        return (
            f'Bearer resource_metadata="{resource_metadata_url(request)}", '
            f'scope="{" ".join(SCOPES)}"'
        )


class ProtectedResourceMetadata(APIView):
    """``/.well-known/oauth-protected-resource[/api/mcp[/]]``: who issues tokens for MCP."""

    authentication_classes: list = []
    permission_classes: list = []

    def get(self, request: Request, path: str = "") -> Response:
        resource_path = f"/{path}" if path else MCP_PATH
        return Response(
            {
                "resource": site_url(request) + resource_path,
                "authorization_servers": [site_url(request)],
                "scopes_supported": list(SCOPES),
                "bearer_methods_supported": ["header"],
                "resource_name": "Brick DNN",
                "resource_documentation": "https://github.com/alexanderKuehling/brick-dnn/blob/main/docs/mcp.md",
            }
        )


class AuthorizationServerMetadata(APIView):
    """``/.well-known/oauth-authorization-server`` (RFC 8414)."""

    authentication_classes: list = []
    permission_classes: list = []

    def get(self, request: Request) -> Response:
        base = site_url(request)
        auth_methods = ["none", "client_secret_post", "client_secret_basic"]
        return Response(
            {
                "issuer": base,
                # A page of the web app: the user has to be logged in there.
                "authorization_endpoint": f"{base}/oauth/authorize",
                "token_endpoint": f"{base}/api/oauth/token/",
                "registration_endpoint": f"{base}/api/oauth/register/",
                "revocation_endpoint": f"{base}/api/oauth/revoke/",
                "response_types_supported": ["code"],
                "grant_types_supported": ["authorization_code", "refresh_token"],
                "code_challenge_methods_supported": ["S256"],
                "token_endpoint_auth_methods_supported": auth_methods,
                "revocation_endpoint_auth_methods_supported": auth_methods,
                "scopes_supported": list(SCOPES),
                "authorization_response_iss_parameter_supported": True,
            }
        )


# --------------------------------------------------------------------------- #
# Dynamic client registration (RFC 7591)
# --------------------------------------------------------------------------- #


def valid_redirect_uri(uri: Any) -> bool:
    """HTTPS, or HTTP on this computer (apps like Claude Code listen on localhost)."""
    if not isinstance(uri, str) or len(uri) > 2000:
        return False
    try:
        parts = urlsplit(uri)
    except ValueError:
        return False
    if parts.fragment or not parts.netloc:
        return False
    if parts.scheme == "https":
        return True
    return parts.scheme == "http" and parts.hostname in ("localhost", "127.0.0.1", "::1")


class RegisterClientView(APIView):
    authentication_classes: list = []
    permission_classes: list = []
    throttle_classes = [ScopedRateThrottle]
    throttle_scope = "oauth"

    def post(self, request: Request) -> Response:
        data = request.data if isinstance(request.data, dict) else {}
        redirect_uris = data.get("redirect_uris")
        if (
            not isinstance(redirect_uris, list)
            or not 0 < len(redirect_uris) <= MAX_REDIRECT_URIS
            or not all(valid_redirect_uri(uri) for uri in redirect_uris)
        ):
            return _registration_error(
                "invalid_redirect_uri",
                "redirect_uris must list 1 to 10 HTTPS URLs (or http://localhost URLs).",
            )
        method = data.get("token_endpoint_auth_method", "client_secret_basic")
        if method not in ("none", "client_secret_post", "client_secret_basic"):
            return _registration_error(
                "invalid_client_metadata", f"Unsupported token_endpoint_auth_method: {method}"
            )
        grant_types = data.get("grant_types", ["authorization_code"])
        response_types = data.get("response_types", ["code"])
        if not isinstance(grant_types, list) or not set(grant_types) <= {
            "authorization_code",
            "refresh_token",
        }:
            return _registration_error("invalid_client_metadata", "Unsupported grant_types.")
        if not isinstance(response_types, list) or not set(response_types) <= {"code"}:
            return _registration_error("invalid_client_metadata", "Unsupported response_types.")
        name = data.get("client_name")
        name = name.strip()[:200] if isinstance(name, str) and name.strip() else "Unnamed app"
        uri = data.get("client_uri")
        if not (isinstance(uri, str) and uri.startswith("https://") and len(uri) <= 500):
            uri = ""

        _remove_unused_clients()
        secret = None if method == "none" else secrets.token_urlsafe(32)
        client = OAuthClient.objects.create(
            client_id=secrets.token_urlsafe(24),
            secret_hash=hash_key(secret) if secret else "",
            name=name,
            uri=uri,
            redirect_uris=redirect_uris,
        )
        body = {
            "client_id": client.client_id,
            "client_id_issued_at": int(client.created_at.timestamp()),
            "client_name": client.name,
            "redirect_uris": client.redirect_uris,
            "grant_types": grant_types,
            "response_types": response_types,
            "token_endpoint_auth_method": method,
        }
        if client.uri:
            body["client_uri"] = client.uri
        if secret:
            body["client_secret"] = secret
            body["client_secret_expires_at"] = 0
        return _no_store(Response(body, status=status.HTTP_201_CREATED))


def _registration_error(error: str, description: str) -> Response:
    return Response(
        {"error": error, "error_description": description}, status=status.HTTP_400_BAD_REQUEST
    )


def _remove_unused_clients() -> None:
    cutoff = timezone.now() - UNUSED_CLIENT_LIFETIME
    OAuthClient.objects.filter(created_at__lt=cutoff, connections__isnull=True).delete()


# --------------------------------------------------------------------------- #
# Authorization (the consent page of the web app calls this)
# --------------------------------------------------------------------------- #


class AuthorizationError(Exception):
    def __init__(self, error: str, description: str, redirect_to: str | None = None):
        super().__init__(description)
        self.error = error
        self.description = description
        self.redirect_to = redirect_to

    def response(self) -> Response:
        body = {"error": self.error, "error_description": self.description}
        if self.redirect_to:
            body["redirect_to"] = self.redirect_to
        return Response(body, status=status.HTTP_400_BAD_REQUEST)


def _with_params(uri: str, params: dict[str, str]) -> str:
    parts = urlsplit(uri)
    query = parse_qsl(parts.query, keep_blank_values=True) + list(params.items())
    return urlunsplit(parts._replace(query=urlencode(query)))


def _valid_resource(request: Request, resource: str) -> bool:
    """RFC 8707: tokens may only be requested for this site's MCP server."""
    base = site_url(request)
    return resource.rstrip("/") in (base, base + MCP_PATH)


class AuthorizationRequest:
    """A parsed and checked ``/oauth/authorize`` query string."""

    def __init__(self, request: Request, query: str):
        params = dict(parse_qsl(query or "", keep_blank_values=True))
        self.state = params.get("state")
        self.issuer = site_url(request)

        client = OAuthClient.objects.filter(client_id=params.get("client_id", "")).first()
        if client is None:
            raise AuthorizationError("invalid_request", "Unknown app (client_id).")
        redirect_uri = params.get("redirect_uri")
        if redirect_uri is None and len(client.redirect_uris) == 1:
            redirect_uri = client.redirect_uris[0]
        if redirect_uri not in client.redirect_uris:
            raise AuthorizationError(
                "invalid_request", "The redirect_uri is not registered for this app."
            )
        self.client = client
        self.redirect_uri = redirect_uri

        # From here on, errors go back to the app.
        if params.get("response_type") != "code":
            self.fail("unsupported_response_type", "Only response_type=code is supported.")
        self.code_challenge = params.get("code_challenge", "")
        if (
            params.get("code_challenge_method") != "S256"
            or not 43 <= len(self.code_challenge) <= 128
        ):
            self.fail("invalid_request", "PKCE with code_challenge_method=S256 is required.")
        scopes = params.get("scope", "").split()
        if not set(scopes) <= set(SCOPES):
            self.fail("invalid_scope", f"Supported scopes: {' '.join(SCOPES)}.")
        # No scope asked for: everything (the user can still allow read-only access).
        self.write_requested = not scopes or "write" in scopes
        resource = params.get("resource")
        if resource and not _valid_resource(request, resource):
            self.fail("invalid_target", "Tokens can only be issued for this site's MCP server.")

    def fail(self, error: str, description: str) -> None:
        raise AuthorizationError(
            error, description, self.redirect(error=error, error_description=description)
        )

    def redirect(self, **params: str) -> str:
        if self.state is not None:
            params["state"] = self.state
        params["iss"] = self.issuer
        return _with_params(self.redirect_uri, params)


class AuthorizeView(APIView):
    """``GET``: check a request for the consent page. ``POST``: the user's decision."""

    permission_classes = [IsLoggedInUser]

    def get(self, request: Request) -> Response:
        try:
            auth = AuthorizationRequest(request, request.META.get("QUERY_STRING", ""))
        except AuthorizationError as exc:
            return exc.response()
        return Response(
            {
                "client": {"name": auth.client.name, "uri": auth.client.uri or None},
                "redirect_host": urlsplit(auth.redirect_uri).netloc,
                "write_requested": auth.write_requested,
            }
        )

    def post(self, request: Request) -> Response:
        data = request.data if isinstance(request.data, dict) else {}
        query = data.get("query")
        try:
            auth = AuthorizationRequest(request, query if isinstance(query, str) else "")
        except AuthorizationError as exc:
            return exc.response()
        if data.get("approve") is not True:
            return Response({"redirect_to": auth.redirect(error="access_denied")})

        code = secrets.token_urlsafe(32)
        OAuthCode.objects.create(
            code_hash=hash_key(code),
            client=auth.client,
            owner=request.user,
            redirect_uri=auth.redirect_uri,
            code_challenge=auth.code_challenge,
            read_only=not (auth.write_requested and data.get("allow_write") is True),
            expires_at=timezone.now() + CODE_LIFETIME,
        )
        return Response({"redirect_to": auth.redirect(code=code)})


# --------------------------------------------------------------------------- #
# Tokens (RFC 6749) and revocation (RFC 7009)
# --------------------------------------------------------------------------- #


class TokenError(Exception):
    def __init__(self, error: str, description: str, status_code: int = 400):
        super().__init__(description)
        self.error = error
        self.description = description
        self.status_code = status_code


def _no_store(response: Response) -> Response:
    response["Cache-Control"] = "no-store"
    response["Pragma"] = "no-cache"
    return response


def _authenticate_client(request: Request, data: dict[str, Any]) -> OAuthClient:
    """client_secret_basic, client_secret_post or (public clients) just the client_id."""
    client_id = data.get("client_id")
    secret = data.get("client_secret")
    header = request.headers.get("Authorization", "")
    if header.startswith("Basic "):
        try:
            decoded = base64.b64decode(header[6:], validate=True).decode()
            client_id, secret = (unquote_plus(part) for part in decoded.split(":", 1))
        except (binascii.Error, UnicodeDecodeError, ValueError):
            raise TokenError("invalid_client", "Malformed client credentials.", 401) from None
    client = OAuthClient.objects.filter(client_id=client_id or "").first()
    if client is None:
        raise TokenError("invalid_client", "Unknown client.", 401)
    secret_ok = isinstance(secret, str) and hmac.compare_digest(
        client.secret_hash, hash_key(secret)
    )
    if client.secret_hash and not secret_ok:
        raise TokenError("invalid_client", "Wrong client secret.", 401)
    return client


def _pkce_ok(verifier: Any, challenge: str) -> bool:
    if not isinstance(verifier, str) or not 43 <= len(verifier) <= 128:
        return False
    digest = hashlib.sha256(verifier.encode()).digest()
    expected = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return hmac.compare_digest(expected, challenge)


def _issue_tokens(connection: OAuthConnection) -> dict[str, Any]:
    """A new access token and a new refresh token (the old refresh token stops working)."""
    now = timezone.now()
    connection.access_tokens.filter(expires_at__lte=now).delete()
    refresh = REFRESH_PREFIX + secrets.token_urlsafe(32)
    connection.refresh_hash = hash_key(refresh)
    connection.refresh_expires_at = now + REFRESH_TOKEN_LIFETIME
    connection.save(update_fields=["refresh_hash", "refresh_expires_at"])
    access = PREFIX + secrets.token_urlsafe(32)
    ApiToken.objects.create(
        owner=connection.owner,
        name=connection.client.name[:100],
        prefix=access[: len(PREFIX) + VISIBLE_CHARS],
        key_hash=hash_key(access),
        read_only=connection.read_only,
        expires_at=now + ACCESS_TOKEN_LIFETIME,
        connection=connection,
    )
    return {
        "access_token": access,
        "token_type": "Bearer",
        "expires_in": int(ACCESS_TOKEN_LIFETIME.total_seconds()),
        "refresh_token": refresh,
        "scope": "read" if connection.read_only else "read write",
    }


def _exchange_code(request: Request, client: OAuthClient, data: dict[str, Any]) -> dict:
    code = data.get("code")
    record = (
        OAuthCode.objects.select_for_update()
        .select_related("owner")
        .filter(code_hash=hash_key(code) if isinstance(code, str) else "")
        .first()
    )
    if record is None:
        raise TokenError("invalid_grant", "Unknown or already used authorization code.")
    record.delete()  # one use only, also when the checks below fail
    if record.client_id != client.pk:
        raise TokenError("invalid_grant", "The code was issued to another app.")
    if record.expires_at <= timezone.now():
        raise TokenError("invalid_grant", "The authorization code has expired.")
    if data.get("redirect_uri", record.redirect_uri) != record.redirect_uri:
        raise TokenError("invalid_grant", "redirect_uri does not match the authorization.")
    if not _pkce_ok(data.get("code_verifier"), record.code_challenge):
        raise TokenError("invalid_grant", "Wrong or missing code_verifier (PKCE).")
    resource = data.get("resource")
    if resource and not _valid_resource(request, resource):
        raise TokenError("invalid_target", "Tokens can only be issued for the MCP server.")
    if not record.owner.is_active:
        raise TokenError("invalid_grant", "This account is disabled.")
    connection = OAuthConnection(
        owner=record.owner,
        client=client,
        read_only=record.read_only,
        refresh_hash=hash_key(secrets.token_urlsafe(32)),  # replaced right away
        refresh_expires_at=timezone.now(),
    )
    connection.save()
    # What the app may do without asking: the account's default (approvals.py).
    from .approvals import adopt_default

    adopt_default(record.owner, connection=connection)
    notices.app_connected(record.owner, client.name, request)
    return _issue_tokens(connection)


def _refresh(request: Request, client: OAuthClient, data: dict[str, Any]) -> dict:
    refresh = data.get("refresh_token")
    connection = (
        OAuthConnection.objects.select_for_update()
        .select_related("owner", "client")
        .filter(refresh_hash=hash_key(refresh) if isinstance(refresh, str) else "", client=client)
        .first()
    )
    if connection is None or connection.refresh_expires_at <= timezone.now():
        raise TokenError("invalid_grant", "Unknown, expired or already used refresh token.")
    if not connection.owner.is_active:
        raise TokenError("invalid_grant", "This account is disabled.")
    scopes = str(data.get("scope", "")).split()
    if not set(scopes) <= set(SCOPES) or (connection.read_only and "write" in scopes):
        raise TokenError("invalid_scope", "The app was not allowed this scope.")
    return _issue_tokens(connection)


class TokenView(APIView):
    authentication_classes: list = []
    permission_classes: list = []
    parser_classes = [FormParser, JSONParser]
    throttle_classes = [ScopedRateThrottle]
    throttle_scope = "oauth"

    def post(self, request: Request) -> Response:
        data = request.data if hasattr(request.data, "get") else {}
        error = None
        # Errors are caught inside the transaction, so that a used code stays deleted.
        with transaction.atomic():
            try:
                client = _authenticate_client(request, data)
                grant_type = data.get("grant_type")
                if grant_type == "authorization_code":
                    body = _exchange_code(request, client, data)
                elif grant_type == "refresh_token":
                    body = _refresh(request, client, data)
                else:
                    raise TokenError("unsupported_grant_type", f"Unsupported: {grant_type}")
            except TokenError as exc:
                error = exc
        if error is not None:
            exc = error
            response = Response(
                {"error": exc.error, "error_description": exc.description},
                status=exc.status_code,
            )
            if exc.status_code == 401:
                response["WWW-Authenticate"] = 'Basic realm="oauth"'
            return _no_store(response)
        return _no_store(Response(body))


class RevokeView(APIView):
    """Apps can sign out: revoking the refresh token disconnects the app."""

    authentication_classes: list = []
    permission_classes: list = []
    parser_classes = [FormParser, JSONParser]
    throttle_classes = [ScopedRateThrottle]
    throttle_scope = "oauth"

    def post(self, request: Request) -> Response:
        data = request.data if hasattr(request.data, "get") else {}
        try:
            client = _authenticate_client(request, data)
        except TokenError as exc:
            return Response({"error": exc.error, "error_description": exc.description}, 401)
        token = data.get("token")
        if isinstance(token, str):
            key_hash = hash_key(token)
            OAuthConnection.objects.filter(refresh_hash=key_hash, client=client).delete()
            ApiToken.objects.filter(key_hash=key_hash, connection__client=client).delete()
        return Response(status=status.HTTP_200_OK)


# --------------------------------------------------------------------------- #
# Connected apps of the logged-in user
# --------------------------------------------------------------------------- #


class ConnectionSerializer(UsageMixin, serializers.ModelSerializer):
    usage_key = "app"
    client_name = serializers.CharField(source="client.name", read_only=True)
    client_uri = serializers.SerializerMethodField()

    class Meta:
        model = OAuthConnection
        fields = [
            "id",
            "client_name",
            "client_uri",
            "read_only",
            "created_at",
            "last_used_at",
            "budget_cpu_minutes",
            "used_cpu_minutes",
        ]
        read_only_fields = fields

    def get_client_uri(self, obj: OAuthConnection) -> str | None:
        return obj.client.uri or None


class ConnectionViewSet(mixins.ListModelMixin, mixins.DestroyModelMixin, viewsets.GenericViewSet):
    """Apps the user connected with OAuth. Deleting one revokes its tokens at once. ``PATCH
    {"budget_cpu_minutes"}`` sets the app's monthly compute budget (null: none)."""

    serializer_class = ConnectionSerializer
    permission_classes = [IsLoggedInUser]

    def get_queryset(self):
        return OAuthConnection.objects.filter(owner=self.request.user).select_related("client")

    def get_serializer_context(self):
        return {**super().get_serializer_context(), "usage": token_usage(self.request.user)}

    def partial_update(self, request: Request, pk=None) -> Response:
        connection = self.get_object()
        serializer = BudgetSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        connection.budget_cpu_minutes = serializer.validated_data["budget_cpu_minutes"]
        connection.save(update_fields=["budget_cpu_minutes"])
        return Response(self.get_serializer(connection).data)
