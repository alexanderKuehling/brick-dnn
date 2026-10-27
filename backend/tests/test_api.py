"""HTTP API: code generation, analysis, accounts and saved projects."""

from __future__ import annotations

import pytest
from django.contrib.auth import get_user_model
from rest_framework.test import APIClient

from brickdnn import EXAMPLES

from .helpers import chain

pytestmark = pytest.mark.django_db

MLP = EXAMPLES["mlp_mnist"]["graph"]


@pytest.fixture
def client():
    return APIClient()


@pytest.fixture
def user():
    return get_user_model().objects.create_user("ada", password="correct-horse-battery")


@pytest.fixture
def auth_client(client, user):
    response = client.post(
        "/api/auth/token/", {"username": "ada", "password": "correct-horse-battery"}, format="json"
    )
    client.credentials(HTTP_AUTHORIZATION=f"Bearer {response.data['access']}")
    return client


def test_health(client):
    response = client.get("/api/health/")
    assert response.status_code == 200
    assert response.data["status"] == "ok"


def test_api_root(client):
    response = client.get("/api/")
    assert response.status_code == 200
    assert "generate" in response.data["endpoints"]


def test_layers(client):
    response = client.get("/api/layers/")
    assert response.status_code == 200
    types = {layer["type"] for layer in response.data["layers"]}
    assert {"Input", "Linear", "Conv2d", "LSTM", "TransformerEncoder", "Add"} <= types
    assert response.data["categories"][0] == "Input"


def test_examples(client):
    response = client.get("/api/examples/")
    assert response.status_code == 200
    assert [e["id"] for e in response.data][:2] == ["mlp_mnist", "cnn_mnist"]


def test_analyze_valid(client):
    response = client.post("/api/analyze/", {"graph": MLP}, format="json")
    assert response.status_code == 200
    assert response.data["valid"] is True
    assert response.data["nodes"]["fc3"]["output_shape"] == [10]


def test_analyze_accepts_bare_graph(client):
    response = client.post("/api/analyze/", MLP, format="json")
    assert response.data["valid"] is True


def test_analyze_invalid(client):
    graph = chain(("Input", {"shape": [3, 8, 8]}), ("Linear", {"in_features": 5}))
    response = client.post("/api/analyze/", {"graph": graph}, format="json")
    assert response.status_code == 200
    assert response.data["valid"] is False
    assert response.data["errors"][0]["node"] == "n1"


def test_generate_json(client):
    response = client.post(
        "/api/generate/",
        {"graph": MLP, "options": {"class_name": "Mnist", "filename": "my net"}},
        format="json",
    )
    assert response.status_code == 200
    assert "class Mnist(nn.Module):" in response.data["code"]
    assert response.data["filename"] == "my_net.py"


def test_generate_download(client):
    response = client.post("/api/generate/?download=1", {"graph": MLP}, format="json")
    assert response.status_code == 200
    assert response["Content-Disposition"] == 'attachment; filename="model.py"'
    assert b"nn.Linear(784, 256)" in response.content


def test_generate_invalid_graph(client):
    response = client.post("/api/generate/", {"graph": {"nodes": [], "edges": []}}, format="json")
    assert response.status_code == 400
    assert response.data["analysis"]["valid"] is False


def test_generate_invalid_options(client):
    response = client.post(
        "/api/generate/", {"graph": MLP, "options": {"class_name": "1abc"}}, format="json"
    )
    assert response.status_code == 400
    response = client.post("/api/generate/", {"graph": MLP, "options": "x"}, format="json")
    assert response.status_code == 400


def test_register_login_me_logout(client):
    response = client.post(
        "/api/auth/register/",
        {"username": "grace", "password": "a-long-enough-password"},
        format="json",
    )
    assert response.status_code == 201
    assert response.data["user"]["username"] == "grace"
    tokens = response.data

    client.credentials(HTTP_AUTHORIZATION=f"Bearer {tokens['access']}")
    assert client.get("/api/auth/me/").data["username"] == "grace"

    response = client.post(
        "/api/auth/token/refresh/", {"refresh": tokens["refresh"]}, format="json"
    )
    assert response.status_code == 200
    new_refresh = response.data["refresh"]

    # The old refresh token was rotated and blacklisted.
    response = client.post(
        "/api/auth/token/refresh/", {"refresh": tokens["refresh"]}, format="json"
    )
    assert response.status_code == 401

    assert (
        client.post("/api/auth/logout/", {"refresh": new_refresh}, format="json").status_code == 200
    )
    response = client.post("/api/auth/token/refresh/", {"refresh": new_refresh}, format="json")
    assert response.status_code == 401


def test_register_rejects_weak_password(client):
    response = client.post(
        "/api/auth/register/", {"username": "bob", "password": "123"}, format="json"
    )
    assert response.status_code == 400
    assert "password" in str(response.data).lower()


def test_register_rejects_duplicate_username(client, user):
    response = client.post(
        "/api/auth/register/",
        {"username": "ada", "password": "another-long-password"},
        format="json",
    )
    assert response.status_code == 400


def test_registration_can_be_disabled(client, settings):
    settings.ALLOW_REGISTRATION = False
    response = client.post(
        "/api/auth/register/",
        {"username": "eve", "password": "a-long-enough-password"},
        format="json",
    )
    assert response.status_code == 403


def test_login_wrong_password(client, user):
    response = client.post(
        "/api/auth/token/", {"username": "ada", "password": "wrong"}, format="json"
    )
    assert response.status_code == 401


def test_projects_require_login(client):
    assert client.get("/api/projects/").status_code == 401
    assert client.get("/api/auth/me/").status_code == 401


def test_project_crud(auth_client):
    response = auth_client.post(
        "/api/projects/", {"name": "MNIST MLP", "graph": MLP}, format="json"
    )
    assert response.status_code == 201
    project_id = response.data["id"]
    assert response.data["num_layers"] == len(MLP["nodes"])

    listing = auth_client.get("/api/projects/")
    assert [p["name"] for p in listing.data] == ["MNIST MLP"]

    response = auth_client.patch(f"/api/projects/{project_id}/", {"name": "Renamed"}, format="json")
    assert response.data["name"] == "Renamed"

    assert auth_client.delete(f"/api/projects/{project_id}/").status_code == 204
    assert auth_client.get("/api/projects/").data == []


def test_project_graph_validation(auth_client):
    response = auth_client.post(
        "/api/projects/", {"name": "Broken", "graph": {"nodes": "x"}}, format="json"
    )
    assert response.status_code == 400


def test_users_only_see_their_own_projects(auth_client):
    other = get_user_model().objects.create_user("mallory", password="whatever-password-1")
    other_project = other.projects.create(name="Secret", graph=MLP)

    assert auth_client.get("/api/projects/").data == []
    assert auth_client.get(f"/api/projects/{other_project.id}/").status_code == 404
    assert auth_client.delete(f"/api/projects/{other_project.id}/").status_code == 404


def test_filename_keeps_py_suffix(client):
    response = client.post(
        "/api/generate/", {"graph": MLP, "options": {"filename": "mnist.py"}}, format="json"
    )
    assert response.data["filename"] == "mnist.py"


def test_deeply_nested_json_is_a_client_error(client):
    response = client.post("/api/analyze/", data="[" * 100_000, content_type="application/json")
    assert response.status_code == 400


def test_huge_numbers_do_not_crash_the_api(client):
    graph = chain(("Input", {"shape": [10**4000, 10**4000]}), ("Flatten", {}))
    response = client.post("/api/analyze/", {"graph": graph}, format="json")
    assert response.status_code == 200
    assert response.data["valid"] is False


def test_login_rate_limit_ignores_spoofed_forwarded_for(client, user):
    statuses = [
        client.post(
            "/api/auth/token/",
            {"username": "ada", "password": "wrong"},
            format="json",
            HTTP_X_FORWARDED_FOR=f"10.0.0.{i}",
        ).status_code
        for i in range(25)
    ]
    assert 429 in statuses
