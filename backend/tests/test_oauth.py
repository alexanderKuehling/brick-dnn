"""OAuth sign-in for the MCP server: metadata, app registration, consent, tokens."""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
from datetime import timedelta
from urllib.parse import parse_qs, urlencode, urlsplit

import pytest
from django.contrib.auth import get_user_model
from django.utils import timezone
from rest_framework.test import APIClient

from brick_dnn_core.api_tokens import create_token
from brick_dnn_core.models import ApiToken, OAuthClient, OAuthCode, OAuthConnection

pytestmark = pytest.mark.django_db

REDIRECT = "https://claude.ai/api/mcp/auth_callback"
BASE = "http://testserver"


@pytest.fixture
def user():
    return get_user_model().objects.create_user("ada", password="correct-horse-battery")


@pytest.fixture
def browser(user):
    """The web app, logged in (JWT): it shows the consent page."""
    client = APIClient()
    response = client.post(
        "/api/auth/token/", {"username": "ada", "password": "correct-horse-battery"}, format="json"
    )
    client.credentials(HTTP_AUTHORIZATION=f"Bearer {response.data['access']}")
    return client


def register(**metadata) -> dict:
    body = {"client_name": "Claude", "redirect_uris": [REDIRECT], **metadata}
    if body.get("token_endpoint_auth_method") is None:
        body["token_endpoint_auth_method"] = "none"
    response = APIClient().post("/api/oauth/register/", body, format="json")
    assert response.status_code == 201, response.data
    return response.data


def pkce() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(48)
    digest = hashlib.sha256(verifier.encode()).digest()
    return verifier, base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def authorize_query(client_id: str, challenge: str, **extra) -> str:
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": REDIRECT,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": "xyz",
        "resource": f"{BASE}/api/mcp/",
        **extra,
    }
    return urlencode({k: v for k, v in params.items() if v is not None})


def approve(browser, query: str, allow_write: bool = True, approve: bool = True) -> dict:
    response = browser.post(
        "/api/oauth/authorize/",
        {"query": query, "approve": approve, "allow_write": allow_write},
        format="json",
    )
    assert response.status_code == 200, response.data
    redirect = urlsplit(response.data["redirect_to"])
    assert f"{redirect.scheme}://{redirect.netloc}{redirect.path}" == REDIRECT
    return {k: v[0] for k, v in parse_qs(redirect.query).items()}


def form_post(path: str, data: dict, **headers):
    """Form-encoded, like real OAuth clients."""
    return APIClient().post(
        path, urlencode(data), content_type="application/x-www-form-urlencoded", **headers
    )


def token_request(**data):
    return form_post("/api/oauth/token/", data)


def sign_in(browser, allow_write: bool = True, **registration) -> tuple[dict, dict]:
    """The whole flow; returns the client registration and the token response."""
    client = register(**registration)
    verifier, challenge = pkce()
    params = approve(browser, authorize_query(client["client_id"], challenge), allow_write)
    assert params["state"] == "xyz" and params["iss"] == BASE
    data = {
        "grant_type": "authorization_code",
        "code": params["code"],
        "redirect_uri": REDIRECT,
        "code_verifier": verifier,
        "client_id": client["client_id"],
        "resource": f"{BASE}/api/mcp/",
    }
    if "client_secret" in client:
        data["client_secret"] = client["client_secret"]
    response = token_request(**data)
    assert response.status_code == 200, response.data
    assert response["Cache-Control"] == "no-store"
    return client, response.data


def mcp(access_token: str, method: str = "tools/list", params=None):
    client = APIClient()
    client.credentials(HTTP_AUTHORIZATION=f"Bearer {access_token}")
    message = {"jsonrpc": "2.0", "id": 1, "method": method}
    if params is not None:
        message["params"] = params
    return client.post("/api/mcp/", json.dumps(message), content_type="application/json")


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


def test_mcp_401_points_to_the_resource_metadata():
    response = APIClient().post("/api/mcp/", "{}", content_type="application/json")
    assert response.status_code == 401
    header = response["WWW-Authenticate"]
    assert header.startswith("Bearer ")
    assert f'resource_metadata="{BASE}/.well-known/oauth-protected-resource/api/mcp/"' in header

    response = APIClient().post("/api/mcp", "{}", content_type="application/json")
    assert '/.well-known/oauth-protected-resource/api/mcp"' in response["WWW-Authenticate"]


@pytest.mark.parametrize(
    ("path", "resource"),
    [
        ("/.well-known/oauth-protected-resource/api/mcp/", f"{BASE}/api/mcp/"),
        ("/.well-known/oauth-protected-resource/api/mcp", f"{BASE}/api/mcp"),
        ("/.well-known/oauth-protected-resource", f"{BASE}/api/mcp"),
    ],
)
def test_protected_resource_metadata(path, resource):
    data = APIClient().get(path).json()
    assert data["resource"] == resource
    assert data["authorization_servers"] == [BASE]
    assert data["scopes_supported"] == ["read", "write"]


def test_authorization_server_metadata(settings):
    data = APIClient().get("/.well-known/oauth-authorization-server").json()
    assert data["issuer"] == BASE
    assert data["authorization_endpoint"] == f"{BASE}/oauth/authorize"
    assert data["token_endpoint"] == f"{BASE}/api/oauth/token/"
    assert data["registration_endpoint"] == f"{BASE}/api/oauth/register/"
    assert data["code_challenge_methods_supported"] == ["S256"]
    assert "none" in data["token_endpoint_auth_methods_supported"]

    settings.PUBLIC_URL = "https://brickdnn.example"
    data = APIClient().get("/.well-known/oauth-authorization-server").json()
    assert data["issuer"] == "https://brickdnn.example"


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #


def test_register_public_and_confidential_clients():
    public = register()
    assert public["token_endpoint_auth_method"] == "none"
    assert "client_secret" not in public
    confidential = register(token_endpoint_auth_method="client_secret_post")
    assert confidential["client_secret"] and confidential["client_secret_expires_at"] == 0
    stored = OAuthClient.objects.get(client_id=confidential["client_id"])
    assert stored.secret_hash and confidential["client_secret"] not in stored.secret_hash


@pytest.mark.parametrize(
    "metadata",
    [
        {"redirect_uris": []},
        {"redirect_uris": ["http://evil.example/callback"]},
        {"redirect_uris": ["https://ok.example/cb#fragment"]},
        {"redirect_uris": "https://ok.example/cb"},
        {"redirect_uris": [REDIRECT], "token_endpoint_auth_method": "private_key_jwt"},
        {"redirect_uris": [REDIRECT], "grant_types": ["client_credentials"]},
    ],
)
def test_register_rejects_bad_metadata(metadata):
    response = APIClient().post("/api/oauth/register/", metadata, format="json")
    assert response.status_code == 400
    assert response.data["error"] in ("invalid_redirect_uri", "invalid_client_metadata")


def test_localhost_redirects_are_allowed():
    register(redirect_uris=["http://localhost:33418/callback", "http://127.0.0.1:5000/cb"])


def test_unused_clients_are_removed_eventually(user):
    old = register()
    used = register()
    OAuthClient.objects.update(created_at=timezone.now() - timedelta(days=31))
    OAuthConnection.objects.create(
        owner=user,
        client=OAuthClient.objects.get(client_id=used["client_id"]),
        refresh_hash="x" * 64,
        refresh_expires_at=timezone.now(),
    )
    register()
    ids = set(OAuthClient.objects.values_list("client_id", flat=True))
    assert old["client_id"] not in ids and used["client_id"] in ids


# --------------------------------------------------------------------------- #
# Consent
# --------------------------------------------------------------------------- #


def test_consent_page_details(browser):
    client = register(client_uri="https://claude.ai")
    _, challenge = pkce()
    response = browser.get(
        f"/api/oauth/authorize/?{authorize_query(client['client_id'], challenge)}"
    )
    assert response.status_code == 200
    assert response.data == {
        "client": {"name": "Claude", "uri": "https://claude.ai"},
        "redirect_host": "claude.ai",
        "write_requested": True,
    }
    query = authorize_query(client["client_id"], challenge, scope="read")
    assert browser.get(f"/api/oauth/authorize/?{query}").data["write_requested"] is False


def test_consent_needs_a_website_login(user):
    client = register()
    _, challenge = pkce()
    query = authorize_query(client["client_id"], challenge)
    assert APIClient().get(f"/api/oauth/authorize/?{query}").status_code == 401
    _, key = create_token(user, "script")
    token_client = APIClient()
    token_client.credentials(HTTP_AUTHORIZATION=f"Bearer {key}")
    assert token_client.get(f"/api/oauth/authorize/?{query}").status_code == 403


def test_bad_client_or_redirect_is_shown_not_redirected(browser):
    client = register()
    _, challenge = pkce()
    for query in [
        authorize_query("unknown", challenge),
        authorize_query(client["client_id"], challenge, redirect_uri="https://evil.example/cb"),
    ]:
        response = browser.get(f"/api/oauth/authorize/?{query}")
        assert response.status_code == 400
        assert "redirect_to" not in response.data


@pytest.mark.parametrize(
    ("extra", "error"),
    [
        ({"response_type": "token"}, "unsupported_response_type"),
        ({"code_challenge_method": "plain"}, "invalid_request"),
        ({"code_challenge": None}, "invalid_request"),
        ({"scope": "admin"}, "invalid_scope"),
        ({"resource": "https://other.example/mcp"}, "invalid_target"),
    ],
)
def test_other_errors_go_back_to_the_app(browser, extra, error):
    client = register()
    _, challenge = pkce()
    query = authorize_query(client["client_id"], challenge, **extra)
    response = browser.get(f"/api/oauth/authorize/?{query}")
    assert response.status_code == 400
    params = parse_qs(urlsplit(response.data["redirect_to"]).query)
    assert params["error"] == [error] and params["state"] == ["xyz"]


def test_deny(browser):
    client = register()
    _, challenge = pkce()
    params = approve(browser, authorize_query(client["client_id"], challenge), approve=False)
    assert params["error"] == "access_denied" and params["state"] == "xyz"
    assert not OAuthCode.objects.exists()


# --------------------------------------------------------------------------- #
# Tokens
# --------------------------------------------------------------------------- #


def test_full_flow_with_mcp_refresh_and_disconnect(browser, user):
    client, tokens = sign_in(browser)
    assert tokens["token_type"] == "Bearer" and tokens["expires_in"] == 3600
    assert tokens["scope"] == "read write"

    response = mcp(tokens["access_token"])
    assert response.status_code == 200
    names = {tool["name"] for tool in response.json()["result"]["tools"]}
    assert "save_project" in names
    result = mcp(
        tokens["access_token"],
        "tools/call",
        {
            "name": "save_project",
            "arguments": {"name": "Via OAuth", "graph": {"nodes": [], "edges": []}},
        },
    ).json()["result"]
    assert "isError" not in result

    # The access token is only for the MCP server, and not listed as a personal token.
    other = APIClient()
    other.credentials(HTTP_AUTHORIZATION=f"Bearer {tokens['access_token']}")
    assert other.get("/api/projects/").status_code == 401
    assert browser.get("/api/auth/api-tokens/").data == []

    connections = browser.get("/api/auth/connections/").data
    assert len(connections) == 1
    assert connections[0]["client_name"] == "Claude" and connections[0]["read_only"] is False
    assert connections[0]["last_used_at"] is not None

    # Refreshing rotates the refresh token.
    response = token_request(
        grant_type="refresh_token",
        refresh_token=tokens["refresh_token"],
        client_id=client["client_id"],
    )
    assert response.status_code == 200
    new = response.data
    assert new["refresh_token"] != tokens["refresh_token"]
    assert mcp(new["access_token"]).status_code == 200
    response = token_request(
        grant_type="refresh_token",
        refresh_token=tokens["refresh_token"],
        client_id=client["client_id"],
    )
    assert response.status_code == 400 and response.data["error"] == "invalid_grant"

    # Disconnecting on the website revokes everything at once.
    assert browser.delete(f"/api/auth/connections/{connections[0]['id']}/").status_code == 204
    assert mcp(new["access_token"]).status_code == 401
    response = token_request(
        grant_type="refresh_token",
        refresh_token=new["refresh_token"],
        client_id=client["client_id"],
    )
    assert response.status_code == 400
    assert not ApiToken.objects.exists()


def test_read_only_permission(browser):
    _, tokens = sign_in(browser, allow_write=False)
    assert tokens["scope"] == "read"
    names = {tool["name"] for tool in mcp(tokens["access_token"]).json()["result"]["tools"]}
    assert "analyze_network" in names and "save_project" not in names


def test_confidential_client_needs_its_secret(browser):
    client = register(token_endpoint_auth_method="client_secret_basic")
    verifier, challenge = pkce()
    params = approve(browser, authorize_query(client["client_id"], challenge))
    data = {
        "grant_type": "authorization_code",
        "code": params["code"],
        "redirect_uri": REDIRECT,
        "code_verifier": verifier,
    }
    response = form_post("/api/oauth/token/", {**data, "client_id": client["client_id"]})
    assert response.status_code == 401 and response.data["error"] == "invalid_client"

    # The failed attempt used up the code; a new one with HTTP Basic works.
    params = approve(browser, authorize_query(client["client_id"], challenge))
    basic = base64.b64encode(f"{client['client_id']}:{client['client_secret']}".encode()).decode()
    response = form_post(
        "/api/oauth/token/", {**data, "code": params["code"]}, HTTP_AUTHORIZATION=f"Basic {basic}"
    )
    assert response.status_code == 200, response.data


def test_code_checks(browser):
    client = register()
    verifier, challenge = pkce()

    def exchange(code, **changes):
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": REDIRECT,
            "code_verifier": verifier,
            "client_id": client["client_id"],
            **changes,
        }
        return token_request(**data)

    code = approve(browser, authorize_query(client["client_id"], challenge))["code"]
    response = exchange(code, code_verifier=pkce()[0])
    assert response.status_code == 400 and "PKCE" in response.data["error_description"]
    assert exchange(code).status_code == 400  # used up by the failed attempt

    code = approve(browser, authorize_query(client["client_id"], challenge))["code"]
    assert exchange(code, redirect_uri="https://claude.ai/other").status_code == 400

    code = approve(browser, authorize_query(client["client_id"], challenge))["code"]
    other = register()
    assert exchange(code, client_id=other["client_id"]).status_code == 400

    code = approve(browser, authorize_query(client["client_id"], challenge))["code"]
    OAuthCode.objects.update(expires_at=timezone.now() - timedelta(seconds=1))
    response = exchange(code)
    assert response.status_code == 400 and "expired" in response.data["error_description"]

    code = approve(browser, authorize_query(client["client_id"], challenge))["code"]
    assert exchange(code).status_code == 200
    assert exchange(code).status_code == 400  # only once

    assert token_request(grant_type="password", client_id=client["client_id"]).data["error"] == (
        "unsupported_grant_type"
    )
    assert token_request(grant_type="refresh_token", client_id="nope").status_code == 401


def test_expired_access_token(browser):
    _, tokens = sign_in(browser)
    ApiToken.objects.update(expires_at=timezone.now() - timedelta(seconds=1))
    assert mcp(tokens["access_token"]).status_code == 401


def test_app_can_revoke_its_tokens(browser):
    client, tokens = sign_in(browser)
    response = form_post(
        "/api/oauth/revoke/", {"token": tokens["refresh_token"], "client_id": client["client_id"]}
    )
    assert response.status_code == 200
    assert not OAuthConnection.objects.exists()
    assert mcp(tokens["access_token"]).status_code == 401


def test_connections_are_private_and_need_a_website_login(browser, user):
    sign_in(browser)
    other = get_user_model().objects.create_user("mallory", password="whatever-password-1")
    theirs = OAuthConnection.objects.create(
        owner=other,
        client=OAuthClient.objects.get(),
        refresh_hash="y" * 64,
        refresh_expires_at=timezone.now(),
    )
    assert len(browser.get("/api/auth/connections/").data) == 1
    assert browser.delete(f"/api/auth/connections/{theirs.pk}/").status_code == 404
    assert APIClient().get("/api/auth/connections/").status_code == 401
