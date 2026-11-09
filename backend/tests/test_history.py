"""Version history and comments of shared projects."""

from __future__ import annotations

from datetime import timedelta

import pytest
from django.contrib.auth import get_user_model
from django.utils import timezone
from rest_framework.test import APIClient

from brick_dnn_core.history import MAX_AUTO_VERSIONS, record
from brick_dnn_core.live import apply_ops
from brick_dnn_core.models import Comment, Project, ProjectMember, ProjectVersion

from .helpers import chain

pytestmark = pytest.mark.django_db

GRAPH = chain(("Input", {"shape": [4]}), ("Linear", {"out_features": 2}))
BIGGER = chain(("Input", {"shape": [4]}), ("Linear", {"out_features": 8}), ("ReLU", {}))


def make_user(name):
    return get_user_model().objects.create_user(name, password="correct-horse-1")


def client_for(user) -> APIClient:
    client = APIClient()
    client.force_authenticate(user)
    return client


@pytest.fixture
def team():
    ada, bob, cy = make_user("ada"), make_user("bob"), make_user("cy")
    response = client_for(ada).post(
        "/api/projects/", {"name": "Net", "graph": GRAPH}, format="json"
    )
    project = Project.objects.get(pk=response.data["id"])
    ProjectMember.objects.create(project=project, user=bob, role="editor")
    ProjectMember.objects.create(project=project, user=cy, role="viewer")
    return project, ada, bob, cy


def versions(client, project):
    return client.get(f"/api/projects/{project.pk}/versions/").data


def age_history(project, minutes=11):
    ProjectVersion.objects.filter(project=project).update(
        updated_at=timezone.now() - timedelta(minutes=minutes)
    )


def test_changes_are_recorded_and_merged_per_person(team):
    project, ada, bob, cy = team
    url = f"/api/projects/{project.pk}/"
    entries = versions(client_for(cy), project)  # viewers see the history
    assert [(e["version"], e["created_by"], e["kind"]) for e in entries] == [(1, "ada", "auto")]

    # More changes by the same person shortly after: still one entry, now at version 3.
    client_for(ada).patch(url, {"name": "Net 2"}, format="json")
    client_for(ada).patch(url, {"graph": BIGGER}, format="json")
    entries = versions(client_for(ada), project)
    assert [(e["version"], e["name"], e["num_layers"]) for e in entries] == [(3, "Net 2", 3)]

    # Someone else (here: live ops) starts a new entry.
    apply_ops(project.pk, bob, [{"op": "set_meta", "name": "Bob's"}])
    assert [(e["version"], e["created_by"]) for e in versions(client_for(ada), project)] == [
        (4, "bob"),
        (3, "ada"),
    ]
    # After a break, the same person gets a new entry too.
    age_history(project)
    apply_ops(project.pk, bob, [{"op": "set_meta", "name": "Bob's again"}])
    assert len(versions(client_for(ada), project)) == 3


def test_name_view_and_restore(team):
    project, ada, bob, cy = team
    base = f"/api/projects/{project.pk}/versions/"
    named = client_for(bob).post(base, {"label": "Baseline"}, format="json")
    assert named.status_code == 201 and named.data["kind"] == "named"
    assert client_for(cy).post(base, {"label": "x"}, format="json").status_code == 403
    assert client_for(bob).post(base, {"label": " "}, format="json").status_code == 400

    age_history(project)
    client_for(ada).patch(f"/api/projects/{project.pk}/", {"graph": BIGGER}, format="json")
    detail = client_for(cy).get(f"{base}{named.data['id']}/").data
    assert detail["graph"] == GRAPH and detail["label"] == "Baseline"

    assert client_for(cy).post(f"{base}{named.data['id']}/restore/").status_code == 403
    response = client_for(bob).post(f"{base}{named.data['id']}/restore/")
    assert response.status_code == 200 and response.data["version"] == 3
    project.refresh_from_db()
    assert project.graph == GRAPH and project.updated_by.username == "bob"
    latest = versions(client_for(ada), project)[0]
    assert latest["kind"] == "restore" and latest["label"] == "Restored Baseline"

    # Naming an automatic entry keeps it for good.
    auto = next(e for e in versions(client_for(ada), project) if e["kind"] == "auto")
    response = client_for(ada).patch(f"{base}{auto['id']}/", {"label": "Big one"}, format="json")
    assert response.data["kind"] == "named" and response.data["label"] == "Big one"
    assert client_for(ada).get(f"{base}999999/").status_code == 404


def test_old_automatic_entries_are_pruned(team):
    project, ada, bob, _ = team
    users = [ada, bob]
    for i in range(MAX_AUTO_VERSIONS + 5):
        project.version += 1
        record(project, users[i % 2])  # alternating people: a new entry every time
    record(project, ada, "named", "Keep me")
    assert project.history.filter(kind="auto").count() == MAX_AUTO_VERSIONS
    assert project.history.filter(kind="named").count() == 1


def test_history_is_private(team):
    project, *_ = team
    stranger = client_for(make_user("eve"))
    assert stranger.get(f"/api/projects/{project.pk}/versions/").status_code == 404
    assert stranger.get(f"/api/projects/{project.pk}/comments/").status_code == 404


def test_comments(team):
    project, ada, bob, cy = team
    base = f"/api/projects/{project.pk}/comments/"
    first = client_for(cy).post(base, {"text": "Why 2 outputs?", "node_id": "n1"}, format="json")
    assert first.status_code == 201  # viewers can comment
    assert first.data["author"]["username"] == "cy" and first.data["node_id"] == "n1"
    reply = client_for(ada).post(
        base, {"text": "Two classes.", "parent": first.data["id"]}, format="json"
    )
    assert reply.data["parent"] == first.data["id"] and reply.data["node_id"] == "n1"
    client_for(bob).post(base, {"text": "Looks good overall"}, format="json")

    assert len(client_for(bob).get(base).data) == 3
    assert [c["text"] for c in client_for(bob).get(f"{base}?node=n1").data] == [
        "Why 2 outputs?",
        "Two classes.",
    ]
    assert client_for(ada).post(base, {"text": ""}, format="json").status_code == 400
    assert (
        client_for(ada)
        .post(base, {"text": "x", "parent": reply.data["id"]}, format="json")
        .status_code
        == 400
    )

    url = f"{base}{first.data['id']}/"
    assert client_for(bob).patch(url, {"text": "edited"}, format="json").status_code == 403
    assert client_for(cy).patch(url, {"text": "Why two outputs?"}, format="json").data["text"] == (
        "Why two outputs?"
    )
    resolved = client_for(bob).patch(url, {"resolved": True}, format="json").data
    assert resolved["resolved"] is True and resolved["resolved_by"] == "bob"

    other = client_for(bob).post(base, {"text": "mine"}, format="json").data
    assert client_for(cy).delete(f"{base}{other['id']}/").status_code == 403
    assert client_for(ada).delete(url).status_code == 204  # the owner deletes any; replies go too
    assert not Comment.objects.filter(pk=reply.data["id"]).exists()
