"""The experiment tracker: the runs table, curves of several runs, tags/notes/stars, the
remembered columns and comparisons shared with a link (brick_dnn_core/experiments.py)."""

from __future__ import annotations

import io
import json
import zipfile
from datetime import timedelta

import pytest
from django.contrib.auth import get_user_model
from django.utils import timezone
from rest_framework.test import APIClient

from brick_dnn_core import training_service as training
from brick_dnn_core.accounts import export_account
from brick_dnn_core.api_tokens import create_token
from brick_dnn_core.models import Project, ProjectMember, RunComparison, TrainingRun
from brickdnn.training.state import write_state

from .helpers import chain

pytestmark = pytest.mark.django_db

GRAPH = chain(("Input", {"shape": [4]}), ("Linear", {"out_features": 3}))


def make_user(name: str):
    return get_user_model().objects.create_user(name, email=f"{name}@example.com", password="x")


def client_for(user) -> APIClient:
    client = APIClient()
    client.force_authenticate(user)
    return client


@pytest.fixture(autouse=True)
def training_dir(settings, tmp_path):
    settings.TRAINING_DIR = tmp_path / "training"
    return settings.TRAINING_DIR


@pytest.fixture
def ada():
    return make_user("ada")


@pytest.fixture
def bob():
    return make_user("bob")


@pytest.fixture
def cy():
    return make_user("cy")


@pytest.fixture
def net(ada, bob, cy):
    """Ada's network: Bob edits it, Cy may only look."""
    project = Project.objects.create(owner=ada, name="Iris net", graph=GRAPH, updated_by=ada)
    ProjectMember.objects.create(project=project, user=bob, role="editor")
    ProjectMember.objects.create(project=project, user=cy, role="viewer")
    return project


def save_state(run: TrainingRun, state: dict) -> None:
    directory = training.run_dir(run.pk)
    directory.mkdir(parents=True, exist_ok=True)
    write_state(directory, state)


def history(epochs: int, start: float = 1.0, accuracy: float = 0.5) -> list[dict]:
    return [
        {
            "epoch": e,
            "train_loss": start / e,
            "val_loss": start / e + 0.1,
            "train_accuracy": accuracy + 0.01 * e,
            "val_accuracy": accuracy + 0.02 * e,
            "seconds": 2.0,
        }
        for e in range(1, epochs + 1)
    ]


def make_run(
    owner,
    name: str,
    project=None,
    lr: float = 0.001,
    batch: int = 64,
    optimizer: str = "adam",
    epochs: int = 3,
    loss: str = "cross_entropy",
    status: str = "completed",
    dataset: dict | None = None,
    f1: float = 0.8,
    **fields,
) -> TrainingRun:
    hist = history(epochs, start=lr * 1000)
    last = hist[-1]
    run = TrainingRun.objects.create(
        owner=owner,
        name=name,
        graph=GRAPH,
        project=project,
        dataset=dataset or {"kind": "upload", "id": 7, "name": "iris.csv"},
        config={
            "epochs": epochs,
            "batch_size": batch,
            "learning_rate": lr,
            "optimizer": optimizer,
            "loss": loss,
        },
        plan={"task": "classification"},
        status=status,
        epochs_done=epochs,
        metrics={
            "train_loss": last["train_loss"],
            "val_loss": last["val_loss"],
            "val_accuracy": last["val_accuracy"],
            "test_accuracy": 0.9,
            "test_loss": 0.3,
        },
        started_at=timezone.now() - timedelta(minutes=5),
        finished_at=timezone.now() if status in ("completed", "stopped", "failed") else None,
        **fields,
    )
    save_state(
        run,
        {
            "status": status,
            "history": hist,
            "weights_epoch": epochs,
            "loss": {"name": loss, "label": loss.replace("_", " ")},
            "evaluation": {
                "task": "classification",
                "val": {
                    "accuracy": 0.88,
                    "balanced_accuracy": 0.86,
                    "macro_f1": f1,
                    "threshold_curve": {"roc_auc": 0.93, "pr_auc": 0.71},
                },
                "test": {"accuracy": 0.9, "macro_f1": f1 - 0.05},
            },
        },
    )
    return run


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #


def test_table_lists_settings_and_headline_numbers(ada):
    run = make_run(ada, "MLP", lr=0.01, batch=32)
    data = client_for(ada).get("/api/training/runs/table/").json()
    assert data["count"] == 1 and data["page"] == 1 and data["pages"] == 1
    row = data["results"][0]
    assert row["id"] == str(run.pk) and row["name"] == "MLP"
    assert row["settings"] == {
        "optimizer": "adam",
        "learning_rate": 0.01,
        "batch_size": 32,
        "epochs": 3,
        "loss": "cross_entropy",
    }
    metrics = row["metrics"]
    assert metrics["val_macro_f1"] == 0.8 and metrics["val_pr_auc"] == 0.71
    assert metrics["test_macro_f1"] == pytest.approx(0.75) and metrics["best_epoch"] == 3
    assert metrics["test_accuracy"] == 0.9
    assert row["model"] == "Neural network" and row["family"] == "neural"
    assert row["duration_seconds"] == pytest.approx(300, abs=5)
    assert row["dataset"] == {"key": "upload:7", "kind": "upload", "name": "iris.csv"}
    assert row["mine"] is True and row["can_annotate"] is True
    # Read from state.json once, then kept on the run.
    assert TrainingRun.objects.get(pk=run.pk).summary["val_pr_auc"] == 0.71


def test_table_sorts_by_metric_and_setting_with_missing_values_last(ada):
    make_run(ada, "a", lr=0.1, f1=0.6)
    make_run(ada, "b", lr=0.001, f1=0.9)
    make_run(ada, "c", lr=0.01, f1=0.7)
    plain = TrainingRun.objects.create(owner=ada, name="d", status="failed", config={})
    client = client_for(ada)

    def names(**params):
        rows = client.get("/api/training/runs/table/", params).json()["results"]
        return [r["name"] for r in rows]

    assert names(sort="metric:val_macro_f1") == ["b", "c", "a", "d"]
    assert names(sort="metric:val_macro_f1", order="asc") == ["a", "c", "b", "d"]
    assert names(sort="setting:learning_rate", order="asc") == ["b", "c", "a", "d"]
    assert names(sort="name") == ["a", "b", "c", "d"]
    assert plain.pk  # a failed run without anything still has a row
    bad = client.get("/api/training/runs/table/", {"sort": "drop table"})
    assert bad.status_code == 400
    assert client.get("/api/training/runs/table/", {"order": "up"}).status_code == 400


def test_table_filters_and_pages(ada):
    other = {"kind": "builtin", "id": "mnist", "name": "MNIST"}
    for i in range(5):
        make_run(ada, f"iris {i}")
    mnist = make_run(ada, "mnist", dataset=other, status="failed")
    TrainingRun.objects.filter(pk=mnist.pk).update(starred=True, tags=["cnn"])
    client = client_for(ada)
    data = client.get("/api/training/runs/table/", {"page_size": 2}).json()
    assert data["count"] == 6 and data["pages"] == 3 and len(data["results"]) == 2
    last = client.get("/api/training/runs/table/", {"page_size": 2, "page": 99}).json()
    assert last["page"] == 3 and len(last["results"]) == 2
    datasets = {d["key"]: d["name"] for d in data["facets"]["datasets"]}
    assert datasets == {"upload:7": "iris.csv", "builtin:mnist": "MNIST"}
    assert data["facets"]["tags"] == [{"name": "cnn", "count": 1}]

    def count(**params):
        return client.get("/api/training/runs/table/", params).json()["count"]

    assert count(dataset="builtin:mnist") == 1
    assert count(status="failed") == 1
    assert count(starred="true") == 1
    assert count(tag="cnn") == 1
    assert count(q="iris 3") == 1
    assert count(q="CNN") == 1  # tags are searched too
    assert count(family="neural") == 6
    assert count(family="ensemble") == 0
    assert client.get("/api/training/runs/table/", {"page": "x"}).status_code == 400


def test_table_shows_own_and_shared_runs_only(ada, bob, cy, net):
    mine = make_run(ada, "ada alone")
    shared = make_run(bob, "bob in the net", project=net)
    hidden = make_run(cy, "cy elsewhere")

    def ids(user, **params):
        rows = client_for(user).get("/api/training/runs/table/", params).json()["results"]
        return {r["id"]: r for r in rows}

    assert set(ids(ada)) == {str(mine.pk), str(shared.pk)}
    assert set(ids(cy)) == {str(shared.pk), str(hidden.pk)}
    assert set(ids(bob, project=net.pk)) == {str(shared.pk)}
    assert set(ids(ada, mine="true")) == {str(mine.pk)}
    # Who may annotate: the starter, the owner and editors of the network, not viewers.
    assert ids(ada)[str(shared.pk)]["can_annotate"] is True
    assert ids(cy)[str(shared.pk)]["can_annotate"] is False
    assert ids(ada)[str(shared.pk)]["started_by"] == "bob"
    assert ids(ada)[str(shared.pk)]["project_name"] == "Iris net"
    assert APIClient().get("/api/training/runs/table/").status_code == 401


def test_table_of_classical_and_ensemble_runs(ada):
    classical = TrainingRun.objects.create(
        owner=ada,
        name="boosting",
        status="completed",
        plan={"task": "classification", "classical": {"type": "SVC", "label": "SVC"}},
        config={},
        metrics={"val_accuracy": 0.8},
    )
    ensemble = TrainingRun.objects.create(
        owner=ada,
        name="ens",
        kind="ensemble",
        status="completed",
        plan={"task": "classification"},
        config={"method": "stacking"},
    )
    search = make_run(ada, "search", tuning={"method": "grid", "trials": 4, "metric": "macro_f1"})
    rows = {
        r["name"]: r for r in client_for(ada).get("/api/training/runs/table/").json()["results"]
    }
    assert rows["boosting"]["model"] == "SVC" and rows["boosting"]["family"] == "classical"
    assert rows["boosting"]["epochs"] == 1
    assert rows["ens"]["family"] == "ensemble" and rows["ens"]["settings"]["method"] == "stacking"
    assert rows["search"]["search"] is True
    assert rows["search"]["settings"]["search"] == "grid"
    assert rows["search"]["settings"]["trials"] == 4
    assert {classical.pk, ensemble.pk, search.pk}


# --------------------------------------------------------------------------- #
# Curves
# --------------------------------------------------------------------------- #


def test_curves_of_several_runs(ada, cy):
    a = make_run(ada, "a", epochs=3)
    b = make_run(ada, "b", epochs=5, loss="focal")
    hidden = make_run(cy, "hidden")
    client = client_for(ada)
    data = client.get("/api/training/runs/curves/", {"ids": f"{b.pk},{a.pk},{hidden.pk}"}).json()
    assert [r["name"] for r in data["runs"]] == ["b", "a"]
    assert data["missing"] == [str(hidden.pk)]
    first = data["runs"][0]
    assert first["loss"] == "focal" and first["loss_label"] == "focal"
    assert len(first["points"]) == 5
    assert first["points"][0]["epoch"] == 1 and first["points"][0]["time"] == 2.0
    assert first["points"][-1]["time"] == 10.0  # the epochs' durations add up
    assert first["points"][0]["val_accuracy"] == pytest.approx(0.52)
    many = ",".join(str(make_run(ada, f"r{i}").pk) for i in range(11))
    assert client.get("/api/training/runs/curves/", {"ids": many}).status_code == 400
    bad = client.get("/api/training/runs/curves/", {"ids": "not-a-run"})
    assert bad.status_code == 400


def test_long_curves_are_thinned_out(ada):
    run = make_run(ada, "long", epochs=2)
    save_state(run, {"status": "completed", "history": history(3000)})
    points = (
        client_for(ada).get("/api/training/runs/curves/", {"ids": str(run.pk)}).json()["runs"][0]
    )
    assert len(points["points"]) == 2000
    assert points["points"][0]["epoch"] == 1 and points["points"][-1]["epoch"] == 3000


# --------------------------------------------------------------------------- #
# Tags, note, star
# --------------------------------------------------------------------------- #


def test_owner_and_editors_annotate_viewers_and_strangers_do_not(ada, bob, cy, net):
    run = make_run(bob, "shared", project=net)
    url = f"/api/training/runs/{run.pk}/annotations/"
    body = {"tags": ["baseline", " baseline ", "lr sweep"], "note": "Good.\x07", "starred": True}
    answer = client_for(bob).patch(url, body, format="json")
    assert answer.status_code == 200
    assert answer.json() == {
        "id": str(run.pk),
        "tags": ["baseline", "lr sweep"],
        "note": "Good.",
        "starred": True,
    }
    # The network's owner and editors may, viewers and strangers may not.
    assert client_for(ada).patch(url, {"starred": False}, format="json").status_code == 200
    assert client_for(cy).patch(url, {"note": "x"}, format="json").status_code == 403
    stranger = make_user("dan")
    assert client_for(stranger).patch(url, {"note": "x"}, format="json").status_code == 404
    run.refresh_from_db()
    assert run.tags == ["baseline", "lr sweep"] and run.note == "Good." and not run.starred
    detail = client_for(cy).get(f"/api/training/runs/{run.pk}/").json()
    assert detail["tags"] == ["baseline", "lr sweep"] and detail["note"] == "Good."


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"tags": "one"},
        {"tags": ["x" * 41]},
        {"tags": [f"t{i}" for i in range(21)]},
        {"note": "x" * 2001},
        {"note": 5},
        {"starred": "yes"},
    ],
)
def test_annotations_are_checked(ada, body):
    run = make_run(ada, "run")
    answer = client_for(ada).patch(f"/api/training/runs/{run.pk}/annotations/", body, format="json")
    assert answer.status_code == 400


def test_read_only_tokens_cannot_annotate(ada):
    run = make_run(ada, "run")
    _, key = create_token(ada, "ro", read_only=True)
    client = APIClient()
    client.credentials(HTTP_AUTHORIZATION=f"Bearer {key}")
    answer = client.patch(
        f"/api/training/runs/{run.pk}/annotations/", {"starred": True}, format="json"
    )
    assert answer.status_code == 403
    assert client.get("/api/training/runs/table/").status_code == 200


# --------------------------------------------------------------------------- #
# Remembered columns
# --------------------------------------------------------------------------- #


def test_columns_are_remembered_per_user(ada, bob):
    client = client_for(ada)
    assert client.get("/api/training/runs/preferences/").json() == {"columns": None}
    columns = ["status", "setting:learning_rate", "metric:val_pr_auc", "status"]
    answer = client.put("/api/training/runs/preferences/", {"columns": columns}, format="json")
    assert answer.json() == {"columns": ["status", "setting:learning_rate", "metric:val_pr_auc"]}
    assert client.get("/api/training/runs/preferences/").json()["columns"][1] == (
        "setting:learning_rate"
    )
    assert client_for(bob).get("/api/training/runs/preferences/").json() == {"columns": None}
    for bad in (["<script>"], "status", [1], ["c"] * 61):
        assert (
            client.put(
                "/api/training/runs/preferences/", {"columns": bad}, format="json"
            ).status_code
            == 400
        )
    reset = client.put("/api/training/runs/preferences/", {"columns": None}, format="json")
    assert reset.json() == {"columns": None}


# --------------------------------------------------------------------------- #
# Shared comparisons
# --------------------------------------------------------------------------- #


def test_share_a_comparison_read_only_and_revoke_it(ada, bob, net):
    a = make_run(ada, "mine")
    b = make_run(bob, "bob's run in ada's net", project=net)
    TrainingRun.objects.filter(pk=a.pk).update(
        note="private thoughts", tags=["secret"], starred=True
    )
    client = client_for(ada)
    created = client.post(
        "/api/training/comparisons/",
        {
            "name": "LR sweep",
            "runs": [str(a.pk), str(b.pk)],
            "columns": ["status", "metric:val_macro_f1"],
            "options": {"metric": "val_accuracy", "x_axis": "time", "smoothing": 0.5},
        },
        format="json",
    )
    assert created.status_code == 201, created.json()
    token = created.json()["token"]
    assert created.json()["options"] == {
        "metric": "val_accuracy",
        "x_axis": "time",
        "smoothing": 0.5,
    }

    shared = APIClient().get(f"/api/shared/runs/{token}/")
    assert shared.status_code == 200
    data = shared.json()
    assert data["name"] == "LR sweep" and data["columns"] == ["status", "metric:val_macro_f1"]
    assert [r["name"] for r in data["runs"]] == ["mine", "bob's run in ada's net"]
    text = json.dumps(data)
    # Who started the runs, their networks and (not chosen) the notes and tags stay private.
    for private in ("started_by", "bob@", "ada@", "Iris net", "private thoughts", "secret"):
        assert private not in text
    assert "project" not in data["runs"][0] and "tags" not in data["runs"][0]
    assert data["runs"][0]["starred"] is False
    assert len(data["curves"]) == 2 and len(data["curves"][0]["points"]) == 3

    # Listed for its owner only, and revoked by deleting it.
    assert len(client.get("/api/training/comparisons/").json()) == 1
    assert client_for(bob).get("/api/training/comparisons/").json() == []
    pk = created.json()["id"]
    assert client_for(bob).delete(f"/api/training/comparisons/{pk}/").status_code == 404
    assert client.delete(f"/api/training/comparisons/{pk}/").status_code == 204
    assert APIClient().get(f"/api/shared/runs/{token}/").status_code == 404


def test_shared_notes_only_when_chosen(ada):
    run = make_run(ada, "run")
    TrainingRun.objects.filter(pk=run.pk).update(note="worth sharing", tags=["best"])
    created = client_for(ada).post(
        "/api/training/comparisons/",
        {"runs": [str(run.pk)], "include_notes": True},
        format="json",
    )
    row = APIClient().get(f"/api/shared/runs/{created.json()['token']}/").json()["runs"][0]
    assert row["note"] == "worth sharing" and row["tags"] == ["best"]
    assert created.json()["name"] == "Shared comparison"


def test_only_own_runs_and_runs_of_own_networks_can_be_shared(ada, bob, cy, net):
    bobs = make_run(bob, "bob's", project=net)
    cys = make_run(cy, "cy's", project=net)
    # Bob edits Ada's network but cannot share Cy's run in it; Ada owns it and can.
    refused = client_for(bob).post(
        "/api/training/comparisons/", {"runs": [str(bobs.pk), str(cys.pk)]}, format="json"
    )
    assert refused.status_code == 400 and "cy's" in refused.json()["detail"]
    ok = client_for(ada).post(
        "/api/training/comparisons/", {"runs": [str(bobs.pk), str(cys.pk)]}, format="json"
    )
    assert ok.status_code == 201
    stranger = make_user("dan")
    hidden = client_for(stranger).post(
        "/api/training/comparisons/", {"runs": [str(bobs.pk)]}, format="json"
    )
    assert hidden.status_code == 400
    for bad in ({"runs": []}, {"runs": "x"}, {"runs": [str(bobs.pk)], "options": {"x_axis": "z"}}):
        answer = client_for(ada).post("/api/training/comparisons/", bad, format="json")
        assert answer.status_code == 400


def test_a_deleted_run_drops_out_of_a_shared_comparison(ada):
    a, b = make_run(ada, "a"), make_run(ada, "b")
    created = client_for(ada).post(
        "/api/training/comparisons/", {"runs": [str(a.pk), str(b.pk)]}, format="json"
    )
    b.delete()
    data = APIClient().get(f"/api/shared/runs/{created.json()['token']}/").json()
    assert [r["name"] for r in data["runs"]] == ["a"]
    assert APIClient().get("/api/shared/runs/short/").status_code == 404


def test_the_data_export_has_tags_notes_and_share_links(ada):
    run = make_run(ada, "run")
    TrainingRun.objects.filter(pk=run.pk).update(note="n", tags=["t"], starred=True)
    RunComparison.objects.create(owner=ada, name="cmp", token="tok123456", run_ids=[str(run.pk)])
    out = io.BytesIO()
    export_account(ada, out)
    with zipfile.ZipFile(out) as archive:
        runs = json.loads(archive.read("training_runs.json"))
        comparisons = json.loads(archive.read("run_comparisons.json"))
        readme = archive.read("README.txt").decode()
    assert runs[0]["tags"] == ["t"] and runs[0]["note"] == "n" and runs[0]["starred"] is True
    assert comparisons[0]["token"] == "tok123456" and comparisons[0]["runs"] == [str(run.pk)]
    assert "run_comparisons.json" in readme


def test_deleting_the_account_deletes_its_comparisons(ada):
    RunComparison.objects.create(owner=ada, name="cmp", token="tok123456", run_ids=[])
    ada.delete()
    assert not RunComparison.objects.exists()


def test_a_token_whose_rule_covers_publishing_cannot_share_runs(ada):
    from brick_dnn_core.approvals import adopt_default

    run = make_run(ada, "run")
    token, key = create_token(ada, "agent")
    adopt_default(ada, token=token)  # the default asks before publishing
    client = APIClient()
    client.credentials(HTTP_AUTHORIZATION=f"Bearer {key}")
    answer = client.post("/api/training/comparisons/", {"runs": [str(run.pk)]}, format="json")
    assert answer.status_code == 403
    assert "approval" in answer.json()["detail"]
    assert not RunComparison.objects.exists()
