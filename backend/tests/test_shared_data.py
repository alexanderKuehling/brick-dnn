"""Training runs and data that members of a shared project use together."""

from __future__ import annotations

import pytest
from django.contrib.auth import get_user_model
from rest_framework.test import APIClient

from brick_dnn_core.models import DataPrep, Dataset, Project, ProjectMember, TrainingRun

from .test_data_api import churn_csv, create_prep, data_dirs, export, import_csv  # noqa: F401
from .test_training_api import DEMO_GRAPH, QUICK, RANDOM, training_dir  # noqa: F401

pytestmark = pytest.mark.django_db


def client_for(user) -> APIClient:
    client = APIClient()
    client.force_authenticate(user)
    return client


@pytest.fixture
def team():
    users = {
        name: get_user_model().objects.create_user(name, password="correct-horse-1")
        for name in ("ada", "bob", "cy", "eve")
    }
    project = Project.objects.create(owner=users["ada"], name="Shared", graph=DEMO_GRAPH)
    ProjectMember.objects.create(project=project, user=users["bob"], role="editor")
    ProjectMember.objects.create(project=project, user=users["cy"], role="viewer")
    return project, users


def start(client, project_id=None):
    body = {"name": "Run", "graph": DEMO_GRAPH, "dataset": RANDOM, "config": QUICK}
    if project_id is not None:
        body["project"] = project_id
    return client.post("/api/training/runs/", body, format="json")


# --------------------------------------------------------------------------- #
# Training runs
# --------------------------------------------------------------------------- #


def test_runs_of_a_project(team, training_dir, settings):  # noqa: F811
    settings.TRAINING_MAX_RUNS = 5
    project, users = team
    response = start(client_for(users["bob"]), project.pk)
    assert response.status_code == 201, response.data
    assert response.data["project"] == project.pk and response.data["started_by"] == "bob"
    run_id = response.data["id"]

    # Everyone in the project sees it, also filtered by project; strangers don't.
    for name in ("ada", "bob", "cy"):
        runs = client_for(users[name]).get(f"/api/training/runs/?project={project.pk}").data
        assert [r["id"] for r in runs] == [run_id]
    assert client_for(users["eve"]).get(f"/api/training/runs/{run_id}/").status_code == 404

    # Viewers can't train in the project, nor can strangers.
    assert start(client_for(users["cy"]), project.pk).status_code == 400
    assert start(client_for(users["eve"]), project.pk).status_code == 400
    assert TrainingRun.objects.count() == 1

    # Stop: who started it and the project's editors; not viewers.
    assert client_for(users["cy"]).post(f"/api/training/runs/{run_id}/stop/").status_code == 403
    assert client_for(users["ada"]).post(f"/api/training/runs/{run_id}/stop/").status_code == 200
    # Delete: who started it and the project's owner.
    assert client_for(users["cy"]).delete(f"/api/training/runs/{run_id}/").status_code == 403
    assert client_for(users["ada"]).delete(f"/api/training/runs/{run_id}/").status_code == 204


def test_private_runs_stay_private(team, training_dir):  # noqa: F811
    _, users = team
    run_id = start(client_for(users["bob"])).data["id"]
    assert client_for(users["ada"]).get(f"/api/training/runs/{run_id}/").status_code == 404
    assert client_for(users["ada"]).get("/api/training/runs/").data == []


# --------------------------------------------------------------------------- #
# Datasets and data preparations
# --------------------------------------------------------------------------- #


@pytest.fixture
def bobs_data(team):
    """Bob (editor) imports a CSV, prepares it and exports a training dataset."""
    _, users = team
    bob = client_for(users["bob"])
    source = import_csv(bob, "churn.csv", churn_csv()).data
    prep = create_prep(bob, source["id"], settings={"target": "churn"}).data
    dataset = export(bob, prep["id"]).data
    return source, prep, dataset


def share(client, project, kind, item_id):
    return client.post(
        f"/api/projects/{project.pk}/data/", {"kind": kind, "id": item_id}, format="json"
    )


def test_share_data_in_a_project(team, bobs_data):
    project, users = team
    source, prep, dataset = bobs_data
    bob, cy, ada = (client_for(users[n]) for n in ("bob", "cy", "ada"))

    # Only your own items, and only in projects you own or edit.
    assert share(ada, project, "dataset", dataset["id"]).status_code == 400
    other = Project.objects.create(owner=users["eve"], name="Elsewhere")
    ProjectMember.objects.create(project=other, user=users["bob"], role="viewer")
    assert share(bob, other, "dataset", dataset["id"]).status_code == 403

    assert share(bob, project, "dataset", dataset["id"]).status_code == 201
    response = share(bob, project, "prep", prep["id"])
    assert [d["name"] for d in response.data["datasets"]] == [dataset["name"]]
    assert [(p["name"], p["owner"]) for p in response.data["preps"]] == [("Churn", "bob")]
    assert cy.get(f"/api/projects/{project.pk}/data/").data == response.data

    # Members see the shared dataset next to their own, marked with the owner.
    listing = cy.get("/api/training/datasets/").data
    assert [(d["name"], d["owner"], d["mine"]) for d in listing] == [
        (dataset["name"], "bob", False)
    ]
    assert listing[0]["shared_in"] == [{"id": project.pk, "name": "Shared"}]
    assert cy.delete(f"/api/training/datasets/{dataset['id']}/").status_code == 404

    # ... and can check a network against it (as for training).
    check = cy.post(
        "/api/training/check/",
        {"graph": DEMO_GRAPH, "dataset": {"kind": "upload", "id": dataset["id"]}},
        format="json",
    )
    assert check.status_code == 200, check.data
    eve = client_for(users["eve"])
    assert (
        eve.post(
            "/api/training/check/",
            {"graph": DEMO_GRAPH, "dataset": {"kind": "upload", "id": dataset["id"]}},
            format="json",
        ).status_code
        == 400
    )

    # Shared preparations: viewers look, editors change, only the owner deletes.
    url = f"/api/data/preps/{prep['id']}/"
    seen = cy.get(url).data
    assert seen["role"] == "viewer" and seen["owner"] == "bob"
    assert cy.get(f"/api/data/sources/{source['id']}/").status_code == 200
    assert cy.patch(url, {"name": "Mine"}, format="json").status_code == 403
    assert ada.get(url).data["role"] == "editor"
    assert ada.patch(url, {"name": "Churn v2"}, format="json").status_code == 200
    assert ada.delete(url).status_code == 404
    assert eve.get(url).status_code == 404

    # An editor's export is shared in the same project.
    exported = export(ada, prep["id"]).data
    assert Dataset.objects.get(pk=exported["id"]).projects.filter(pk=project.pk).exists()

    # Removing: the item's owner or the project's owner.
    assert cy.delete(f"/api/projects/{project.pk}/data/prep/{prep['id']}/").status_code == 403
    assert ada.delete(f"/api/projects/{project.pk}/data/prep/{prep['id']}/").status_code == 204
    assert cy.get(url).status_code == 404
    assert DataPrep.objects.filter(pk=prep["id"]).exists()
