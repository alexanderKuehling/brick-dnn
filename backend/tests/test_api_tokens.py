"""Personal API tokens: managing them and logging in with them."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from django.contrib.auth import get_user_model
from django.utils import timezone
from rest_framework.test import APIClient

from brick_dnn_core.api_tokens import MAX_TOKENS_PER_USER, create_token, hash_key
from brick_dnn_core.models import ApiToken, Project

from .helpers import chain

pytestmark = pytest.mark.django_db

GRAPH = chain(("Input", {"shape": [4]}), ("Linear", {"out_features": 2}))


@pytest.fixture
def user():
    return get_user_model().objects.create_user("ada", password="correct-horse-battery")


@pytest.fixture
def auth_client(user):
    """Logged in on the website (JWT)."""
    client = APIClient()
    response = client.post(
        "/api/auth/token/", {"username": "ada", "password": "correct-horse-battery"}, format="json"
    )
    client.credentials(HTTP_AUTHORIZATION=f"Bearer {response.data['access']}")
    return client


def token_client(key: str) -> APIClient:
    client = APIClient()
    client.credentials(HTTP_AUTHORIZATION=f"Bearer {key}")
    return client


def test_create_list_and_revoke(auth_client):
    response = auth_client.post(
        "/api/auth/api-tokens/",
        {"name": "Claude", "read_only": True, "expires_in_days": 30},
        format="json",
    )
    assert response.status_code == 201
    key = response.data["token"]
    assert key.startswith("bdnn_") and len(key) == 5 + 43
    assert response.data["prefix"] == key[:13]
    assert response.data["read_only"] is True
    assert response.data["last_used_at"] is None
    expires = datetime.fromisoformat(response.data["expires_at"])
    assert timedelta(days=29) < expires - timezone.now() <= timedelta(days=30)

    # Only a hash is stored, and the list never shows the token again.
    stored = ApiToken.objects.get()
    assert stored.key_hash == hash_key(key) and key not in stored.key_hash
    listing = auth_client.get("/api/auth/api-tokens/")
    assert [t["name"] for t in listing.data] == ["Claude"]
    assert "token" not in listing.data[0] and "key_hash" not in listing.data[0]

    assert token_client(key).get("/api/projects/").status_code == 200
    assert auth_client.delete(f"/api/auth/api-tokens/{stored.pk}/").status_code == 204
    response = token_client(key).get("/api/projects/")
    assert response.status_code == 401
    assert response["WWW-Authenticate"].startswith("Bearer")


def test_defaults_and_validation(auth_client):
    response = auth_client.post("/api/auth/api-tokens/", {"name": "Script"}, format="json")
    assert response.status_code == 201
    assert response.data["read_only"] is False and response.data["expires_at"] is None
    assert auth_client.post("/api/auth/api-tokens/", {}, format="json").status_code == 400
    response = auth_client.post(
        "/api/auth/api-tokens/", {"name": "x", "expires_in_days": 5}, format="json"
    )
    assert response.status_code == 400 and "expires_in_days" in response.data
    response = auth_client.post("/api/auth/api-tokens/", {"name": "x" * 101}, format="json")
    assert response.status_code == 400 and "name" in response.data


def test_limit_per_user(auth_client, user):
    for i in range(MAX_TOKENS_PER_USER):
        create_token(user, f"t{i}")
    response = auth_client.post("/api/auth/api-tokens/", {"name": "one more"}, format="json")
    assert response.status_code == 400
    assert str(MAX_TOKENS_PER_USER) in response.data["detail"]


def test_tokens_cannot_manage_tokens(user):
    _, key = create_token(user, "leaked")
    client = token_client(key)
    assert client.get("/api/auth/api-tokens/").status_code == 403
    assert client.post("/api/auth/api-tokens/", {"name": "more"}, format="json").status_code == 403
    assert ApiToken.objects.count() == 1


def test_management_needs_login():
    client = APIClient()
    assert client.get("/api/auth/api-tokens/").status_code == 401
    assert client.post("/api/auth/api-tokens/", {"name": "x"}, format="json").status_code == 401


def test_users_only_see_and_revoke_their_own(auth_client):
    other = get_user_model().objects.create_user("mallory", password="whatever-password-1")
    token, _ = create_token(other, "theirs")
    assert auth_client.get("/api/auth/api-tokens/").data == []
    assert auth_client.delete(f"/api/auth/api-tokens/{token.pk}/").status_code == 404
    assert ApiToken.objects.filter(pk=token.pk).exists()


def test_token_acts_as_its_owner(user):
    Project.objects.create(owner=user, name="Mine", graph=GRAPH)
    _, key = create_token(user, "script")
    client = token_client(key)
    assert [p["name"] for p in client.get("/api/projects/").data] == ["Mine"]
    response = client.post("/api/projects/", {"name": "New", "graph": GRAPH}, format="json")
    assert response.status_code == 201
    assert Project.objects.get(name="New").owner == user
    assert client.get("/api/auth/me/").data["username"] == "ada"


def test_read_only_token(user):
    Project.objects.create(owner=user, name="Mine", graph=GRAPH)
    _, key = create_token(user, "reader", read_only=True)
    client = token_client(key)
    assert client.get("/api/projects/").status_code == 200
    response = client.post("/api/projects/", {"name": "New", "graph": GRAPH}, format="json")
    assert response.status_code == 403
    assert "read-only" in response.data["detail"]
    assert client.delete(f"/api/projects/{Project.objects.get().pk}/").status_code == 403
    # Analysing and generating code change nothing: allowed.
    assert client.post("/api/analyze/", {"graph": GRAPH}, format="json").status_code == 200
    assert client.post("/api/generate/", {"graph": GRAPH}, format="json").status_code == 200
    assert Project.objects.count() == 1


def test_invalid_expired_and_disabled_tokens(user):
    assert token_client("bdnn_" + "x" * 43).get("/api/projects/").status_code == 401
    token, key = create_token(user, "old", expires_in_days=7)
    token.expires_at = timezone.now() - timedelta(seconds=1)
    token.save()
    response = token_client(key).get("/api/projects/")
    assert response.status_code == 401 and "expired" in response.data["detail"]

    _, key = create_token(user, "new")
    user.is_active = False
    user.save()
    assert token_client(key).get("/api/projects/").status_code == 401


def test_other_bearer_tokens_are_still_jwt(auth_client):
    """The web app's login keeps working next to API tokens."""
    assert auth_client.get("/api/auth/me/").status_code == 200
    assert token_client("not-a-jwt").get("/api/auth/me/").status_code == 401


def test_last_used_is_recorded(user):
    token, key = create_token(user, "script")
    token_client(key).get("/api/projects/")
    token.refresh_from_db()
    assert token.last_used_at is not None
    first = token.last_used_at
    token_client(key).get("/api/projects/")  # within a minute: not written again
    token.refresh_from_db()
    assert token.last_used_at == first
