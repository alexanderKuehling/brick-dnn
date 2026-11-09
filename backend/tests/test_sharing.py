"""Sharing projects: members and roles, view links, conflict-safe saving."""

from __future__ import annotations

import pytest
from django.contrib.auth import get_user_model
from rest_framework.test import APIClient

from brick_dnn_core.api_tokens import create_token
from brick_dnn_core.models import Project, ProjectMember

from .helpers import chain

pytestmark = pytest.mark.django_db

GRAPH = chain(("Input", {"shape": [4]}), ("Linear", {"out_features": 2}))
OTHER_GRAPH = chain(("Input", {"shape": [8]}), ("Linear", {"out_features": 3}))


def make_user(name: str, email: str = ""):
    return get_user_model().objects.create_user(name, email=email, password="correct-horse-1")


def client_for(user) -> APIClient:
    client = APIClient()
    client.force_authenticate(user)
    return client


@pytest.fixture
def ada():
    return make_user("ada", "ada@example.com")


@pytest.fixture
def bob():
    return make_user("bob", "bob@example.com")


@pytest.fixture
def cy():
    return make_user("cy")


@pytest.fixture
def project(ada):
    return Project.objects.create(owner=ada, name="Net", graph=GRAPH, updated_by=ada)


def invite(client, project, user, role="editor"):
    return client.post(
        f"/api/projects/{project.pk}/members/", {"user": user, "role": role}, format="json"
    )


def test_owner_view_of_a_project(ada, project):
    data = client_for(ada).get(f"/api/projects/{project.pk}/").data
    assert data["role"] == "owner"
    assert data["owner"] == {"id": ada.pk, "username": "ada"}
    assert data["version"] == 1 and data["updated_by"] == "ada"
    assert data["shared"] is False and data["link_access"] == "none"


def test_invite_and_roles(ada, bob, cy, project):
    owner = client_for(ada)
    response = invite(owner, project, "bob", "editor")
    assert response.status_code == 201
    assert response.data["user"] == {"id": bob.pk, "username": "bob"}
    assert invite(owner, project, "CY", "viewer").status_code == 201  # case-insensitive

    members = owner.get(f"/api/projects/{project.pk}/members/").data
    assert members["owner"]["username"] == "ada"
    assert [(m["user"]["username"], m["role"]) for m in members["members"]] == [
        ("bob", "editor"),
        ("cy", "viewer"),
    ]
    # Members see it in their list, with their role.
    shared = client_for(bob).get("/api/projects/").data
    assert [(p["name"], p["role"], p["owner"]["username"]) for p in shared] == [
        ("Net", "editor", "ada")
    ]
    assert shared[0]["share_token"] is None
    assert client_for(bob).get("/api/projects/?scope=mine").data == []
    assert len(client_for(bob).get("/api/projects/?scope=shared").data) == 1
    assert owner.get("/api/projects/").data[0]["shared"] is True


def test_invite_by_email_and_errors(ada, bob, project):
    owner = client_for(ada)
    assert invite(owner, project, "BOB@example.com").status_code == 201
    response = invite(owner, project, "bob")
    assert response.status_code == 400 and "already has access" in response.data["user"][0]
    response = invite(owner, project, "nobody")
    assert response.status_code == 400 and "no account" in response.data["user"][0]
    assert invite(owner, project, "ada").status_code == 400
    assert invite(owner, project, "bob", "admin").status_code == 400

    make_user("twin1", "same@example.com")
    make_user("twin2", "same@example.com")
    response = invite(owner, project, "same@example.com")
    assert response.status_code == 400 and "username" in response.data["user"][0]


def test_only_the_owner_manages_members(ada, bob, cy, project):
    ProjectMember.objects.create(project=project, user=bob, role="editor")
    editor = client_for(bob)
    assert invite(editor, project, "cy").status_code == 403
    member = ProjectMember.objects.get(user=bob)
    url = f"/api/projects/{project.pk}/members/{member.pk}/"
    assert editor.patch(url, {"role": "viewer"}, format="json").status_code == 403
    assert client_for(ada).patch(url, {"role": "viewer"}, format="json").data["role"] == "viewer"

    # Members can leave; others cannot remove them.
    cy_member = ProjectMember.objects.create(project=project, user=cy, role="viewer")
    assert editor.delete(f"/api/projects/{project.pk}/members/{cy_member.pk}/").status_code == 403
    assert (
        client_for(cy).delete(f"/api/projects/{project.pk}/members/{cy_member.pk}/").status_code
        == 204
    )
    assert client_for(cy).get(f"/api/projects/{project.pk}/").status_code == 404
    assert client_for(ada).delete(url).status_code == 204
    assert not ProjectMember.objects.exists()


def test_viewers_look_editors_edit_owners_delete(ada, bob, cy, project):
    ProjectMember.objects.create(project=project, user=bob, role="editor")
    ProjectMember.objects.create(project=project, user=cy, role="viewer")
    url = f"/api/projects/{project.pk}/"

    viewer = client_for(cy)
    assert viewer.get(url).data["role"] == "viewer"
    assert viewer.patch(url, {"name": "Mine"}, format="json").status_code == 403
    assert viewer.delete(url).status_code == 403

    editor = client_for(bob)
    response = editor.patch(url, {"name": "Better", "graph": OTHER_GRAPH}, format="json")
    assert response.status_code == 200
    assert response.data["version"] == 2 and response.data["updated_by"] == "bob"
    assert editor.delete(url).status_code == 403
    assert client_for(ada).get(url).data["name"] == "Better"

    stranger = client_for(make_user("eve"))
    assert stranger.get(url).status_code == 404
    assert stranger.patch(url, {"name": "x"}, format="json").status_code == 404

    assert client_for(ada).delete(url).status_code == 204
    assert not ProjectMember.objects.exists()


def test_saving_an_old_version_is_a_conflict(ada, bob, project):
    ProjectMember.objects.create(project=project, user=bob, role="editor")
    url = f"/api/projects/{project.pk}/"
    assert (
        client_for(bob).patch(url, {"name": "Bob's", "version": 1}, format="json").data["version"]
        == 2
    )

    response = client_for(ada).patch(url, {"graph": OTHER_GRAPH, "version": 1}, format="json")
    assert response.status_code == 409
    assert "bob changed this network" in response.data["detail"]
    assert response.data["project"]["version"] == 2
    assert response.data["project"]["name"] == "Bob's"
    project.refresh_from_db()
    assert project.graph == GRAPH  # nothing was overwritten

    # Saving without a version overwrites ("keep mine").
    response = client_for(ada).patch(url, {"graph": OTHER_GRAPH}, format="json")
    assert response.status_code == 200 and response.data["version"] == 3
    assert client_for(ada).patch(url, {"version": "2"}, format="json").status_code == 400


def test_duplicate(ada, bob, project):
    ProjectMember.objects.create(project=project, user=bob, role="viewer")
    response = client_for(bob).post(f"/api/projects/{project.pk}/duplicate/")
    assert response.status_code == 201
    assert response.data["name"] == "Net (copy)" and response.data["role"] == "owner"
    copy = Project.objects.get(pk=response.data["id"])
    assert copy.owner == bob and copy.graph == GRAPH and copy.version == 1


def test_view_link(ada, bob, project):
    owner = client_for(ada)
    link = f"/api/projects/{project.pk}/link/"
    assert client_for(bob).post(link, {"enabled": True}, format="json").status_code == 404

    response = owner.post(link, {"enabled": True}, format="json")
    token = response.data["share_token"]
    assert response.data["link_access"] == "view" and len(token) >= 20
    assert owner.get(f"/api/projects/{project.pk}/").data["share_token"] == token

    public = APIClient().get(f"/api/shared/{token}/")
    assert public.status_code == 200
    assert public.data["name"] == "Net" and public.data["owner"] == "ada"
    assert public.data["graph"] == GRAPH

    # A copy needs an account.
    assert APIClient().post(f"/api/shared/{token}/duplicate/").status_code == 401
    response = client_for(bob).post(f"/api/shared/{token}/duplicate/")
    assert response.status_code == 201 and Project.objects.get(pk=response.data["id"]).owner == bob

    # A new link replaces the old one; turning it off kills it.
    new = owner.post(link, {"enabled": True, "regenerate": True}, format="json").data["share_token"]
    assert new != token
    assert APIClient().get(f"/api/shared/{token}/").status_code == 404
    assert owner.post(link, {"enabled": False}, format="json").data == {
        "link_access": "none",
        "share_token": None,
    }
    assert APIClient().get(f"/api/shared/{new}/").status_code == 404


def test_read_only_api_tokens_cannot_share(ada, project):
    _, key = create_token(ada, "reader", read_only=True)
    client = APIClient()
    client.credentials(HTTP_AUTHORIZATION=f"Bearer {key}")
    assert client.get(f"/api/projects/{project.pk}/members/").status_code == 200
    assert invite(client, project, "bob").status_code == 403
    assert client.post(f"/api/projects/{project.pk}/link/", {"enabled": True}).status_code == 403


def test_mcp_sees_shared_projects_and_respects_roles(ada, bob, project):
    import json

    ProjectMember.objects.create(project=project, user=bob, role="viewer")
    _, key = create_token(bob, "assistant")
    client = APIClient()
    client.credentials(HTTP_AUTHORIZATION=f"Bearer {key}")

    def call(tool, **arguments):
        message = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": tool, "arguments": arguments},
        }
        response = client.post("/api/mcp/", json.dumps(message), content_type="application/json")
        return response.json()["result"]

    listing = json.loads(call("list_projects")["content"][0]["text"])
    assert [(p["name"], p["role"]) for p in listing] == [("Net", "viewer")]
    assert call("get_project", id=project.pk)["structuredContent"]["graph"] == GRAPH
    result = call("save_project", id=project.pk, name="Hijacked")
    assert result["isError"] is True and "view" in result["content"][0]["text"]

    ProjectMember.objects.filter(user=bob).update(role="editor")
    assert "isError" not in call("save_project", id=project.pk, name="Bob's", version=1)
    result = call("save_project", id=project.pk, name="Stale", version=1)
    assert result["isError"] is True and "changed" in result["content"][0]["text"]
    project.refresh_from_db()
    assert project.name == "Bob's" and project.version == 2 and project.updated_by == bob
