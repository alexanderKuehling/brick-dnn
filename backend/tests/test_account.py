"""The own account: e-mail address, password, data export and deleting the account."""

from __future__ import annotations

import io
import json
import os
import signal
import sys
import zipfile
from datetime import timedelta

import pytest
from django.contrib.auth import get_user_model
from django.utils import timezone
from rest_framework.test import APIClient
from rest_framework.throttling import ScopedRateThrottle

from brick_dnn_core import accounts, live, training_service
from brick_dnn_core.api_tokens import create_token
from brick_dnn_core.models import (
    ApiToken,
    Comment,
    DataPrep,
    Dataset,
    DataSource,
    OAuthClient,
    OAuthConnection,
    Project,
    ProjectMember,
    ProjectVersion,
    TrainingRun,
    UserProfile,
)

from .test_data_api import churn_csv, create_prep, import_csv
from .test_training_api import (
    DEMO_GRAPH,
    FAKE_RUNNER,
    image_zip,
    iris_csv,
    start,
    upload,
    use_runner,
    wait_for,
    wait_for_state,
)

pytestmark = pytest.mark.django_db

User = get_user_model()
PASSWORD = "correct-horse-battery"
OTHER_PASSWORD = "whatever-password-1"


# --------------------------------------------------------------------------- #
# Fixtures and helpers
# --------------------------------------------------------------------------- #


@pytest.fixture
def dirs(settings, tmp_path, monkeypatch):
    """Fresh folders for files, training with a fake runner, data preparation switched on."""
    settings.TRAINING_DIR = tmp_path / "training"
    settings.DATA_PREP_DIR = tmp_path / "prepare"
    settings.LEGAL_DIR = tmp_path / "legal"
    settings.TRAINING_ENABLED = True
    settings.DATA_PREP_ENABLED = True
    settings.TRAINING_MAX_RUNS = 5
    settings.EXPORT_MAX_FILES_MB = 200
    monkeypatch.setattr(training_service, "torch_installed", lambda: True)
    script = tmp_path / "fake_runner.py"
    script.write_text(FAKE_RUNNER)
    settings.TRAINING_RUNNER_COMMAND = [sys.executable, str(script), "complete"]
    yield tmp_path
    for process in list(training_service._PROCESSES.values()):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                process.kill()
            process.wait(timeout=5)
    training_service._PROCESSES.clear()


@pytest.fixture
def ada():
    return User.objects.create_user("ada", email="ada@example.com", password=PASSWORD)


@pytest.fixture
def mallory():
    return User.objects.create_user("mallory", email="mallory@example.org", password=OTHER_PASSWORD)


def login(username: str, password: str = PASSWORD) -> tuple[APIClient, dict]:
    client = APIClient()
    response = client.post(
        "/api/auth/token/", {"username": username, "password": password}, format="json"
    )
    assert response.status_code == 200, response.data
    client.credentials(HTTP_AUTHORIZATION=f"Bearer {response.data['access']}")
    return client, response.data


@pytest.fixture
def ada_client(ada):
    return login("ada")[0]


@pytest.fixture
def mallory_client(mallory):
    return login("mallory", OTHER_PASSWORD)[0]


def refresh(token: str):
    return APIClient().post("/api/auth/token/refresh/", {"refresh": token}, format="json")


def project(client, name: str, graph=DEMO_GRAPH) -> dict:
    response = client.post("/api/projects/", {"name": name, "graph": graph}, format="json")
    assert response.status_code == 201, response.data
    return response.data


def comment(client, project_id: int, text: str, parent: int | None = None) -> dict:
    body = {"text": text, **({"parent": parent} if parent else {})}
    response = client.post(f"/api/projects/{project_id}/comments/", body, format="json")
    assert response.status_code == 201, response.data
    return response.data


def invite(client, project_id: int, username: str, role: str = "editor") -> None:
    response = client.post(
        f"/api/projects/{project_id}/members/", {"user": username, "role": role}, format="json"
    )
    assert response.status_code == 201, response.data


def connect_app(user, name: str = "Claude") -> OAuthConnection:
    client = OAuthClient.objects.create(client_id=f"client-{name}", name=name)
    return OAuthConnection.objects.create(
        owner=user,
        client=client,
        refresh_hash=f"{name:x<64}"[:64],
        refresh_expires_at=timezone.now() + timedelta(days=30),
    )


def export(client) -> zipfile.ZipFile:
    response = client.get("/api/auth/account/export/")
    assert response.status_code == 200, getattr(response, "data", None)
    assert response["Content-Type"] == "application/zip"
    assert "attachment" in response["Content-Disposition"]
    return zipfile.ZipFile(io.BytesIO(b"".join(response.streaming_content)))


def read_json(archive: zipfile.ZipFile, name: str):
    return json.loads(archive.read(name))


@pytest.fixture
def ada_data(dirs, ada, ada_client, mallory, mallory_client):
    """Ada's things, and Mallory's: they share a project each way and comment."""
    own = project(ada_client, "Ada's net")
    ada_client.patch(
        f"/api/projects/{own['id']}/",
        {"graph": {**DEMO_GRAPH, "nodes": DEMO_GRAPH["nodes"][:2]}},
        format="json",
    )
    invite(ada_client, own["id"], "mallory", "viewer")
    theirs = project(mallory_client, "Shared net")
    invite(mallory_client, theirs["id"], "ada", "editor")

    thread = comment(ada_client, theirs["id"], "Ada starts a thread")
    reply = comment(mallory_client, theirs["id"], "A reply to Ada", parent=thread["id"])
    kept = comment(mallory_client, theirs["id"], "Mallory's own thread")
    own_comment = comment(ada_client, own["id"], "Ada on her own net")
    foreign = comment(mallory_client, own["id"], "Mallory secretly comments")

    source = import_csv(ada_client, "churn.csv", churn_csv()).data
    prep = create_prep(ada_client, source["id"], settings={"target": "churn"}).data
    csv_dataset = upload(ada_client, "iris.csv", iris_csv(), name="Iris").data
    pictures = ["cat/1.png", "cat/2.png", "dog/1.png", "dog/2.png"]
    images = upload(ada_client, "pets.zip", image_zip(pictures)).data
    run = start(ada_client, name="Ada's run").data
    wait_for(ada_client, run["id"])
    _, key = create_token(ada, "script")
    connect_app(ada)

    their_dataset = upload(mallory_client, "secret.csv", iris_csv(), name="Mallory data").data
    their_run = start(mallory_client, name="Mallory's run in Ada's net").data
    wait_for(mallory_client, their_run["id"])
    TrainingRun.objects.filter(pk=their_run["id"]).update(project_id=own["id"])
    return {
        "own": own,
        "theirs": theirs,
        "thread": thread,
        "reply": reply,
        "kept": kept,
        "own_comment": own_comment,
        "foreign": foreign,
        "source": source,
        "prep": prep,
        "csv_dataset": csv_dataset,
        "images": images,
        "run": run,
        "key": key,
        "their_dataset": their_dataset,
        "their_run": their_run,
    }


# --------------------------------------------------------------------------- #
# Who may
# --------------------------------------------------------------------------- #

ENDPOINTS = [
    ("get", "/api/auth/account/"),
    ("patch", "/api/auth/account/"),
    ("post", "/api/auth/account/password/"),
    ("get", "/api/auth/account/export/"),
    ("post", "/api/auth/account/delete/"),
]


@pytest.mark.parametrize(("method", "url"), ENDPOINTS)
def test_account_needs_a_website_login(ada, method, url):
    assert getattr(APIClient(), method)(url, {}, format="json").status_code == 401
    _, key = create_token(ada, "script")
    client = APIClient()
    client.credentials(HTTP_AUTHORIZATION=f"Bearer {key}")
    response = getattr(client, method)(url, {}, format="json")
    assert response.status_code == 403
    assert "API tokens cannot" in response.data["detail"]
    assert User.objects.filter(pk=ada.pk).exists()


def test_account_changes_are_rate_limited(ada, ada_client, monkeypatch):
    monkeypatch.setitem(ScopedRateThrottle.THROTTLE_RATES, "account", "2/hour")
    statuses = [
        ada_client.patch("/api/auth/account/", {"email": f"a{i}@example.com"}, format="json")
        for i in range(3)
    ]
    assert [r.status_code for r in statuses] == [200, 200, 429]
    assert ada_client.get("/api/auth/account/").status_code == 200  # reading does not count
    wrong = {"current_password": "guess", "new_password": "another-long-pass"}
    url = "/api/auth/account/password/"
    assert ada_client.post(url, wrong, format="json").status_code == 429
    assert ada_client.post("/api/auth/account/delete/", {}, format="json").status_code == 429


# --------------------------------------------------------------------------- #
# E-mail address and password
# --------------------------------------------------------------------------- #


def test_account_info(dirs, ada, ada_client):
    project(ada_client, "Net")
    import_csv(ada_client, "churn.csv", churn_csv())
    data = ada_client.get("/api/auth/account/").data
    assert data["username"] == "ada"
    assert data["email"] == "ada@example.com"
    assert data["date_joined"] and data["last_login"]
    assert data["terms_accepted_at"] is None
    assert data["stored"]["projects"] == 1
    assert data["stored"]["data_sources"] == 1
    assert data["stored"]["upload_bytes"] == len(churn_csv())


def test_change_email(ada, ada_client, mallory):
    response = ada_client.patch("/api/auth/account/", {"email": " Ada@Example.NET "}, format="json")
    assert response.status_code == 200, response.data
    assert response.data["email"] == "Ada@example.net"
    ada.refresh_from_db()
    assert ada.email == "Ada@example.net"
    assert ada_client.get("/api/auth/me/").data["email"] == "Ada@example.net"

    for email, message in [
        ("MALLORY@example.org", "An account with this e-mail address exists already."),
        ("not an address", "Enter a valid email address."),
        ("", "Enter your e-mail address."),
    ]:
        response = ada_client.patch("/api/auth/account/", {"email": email}, format="json")
        assert response.status_code == 400
        assert response.data["email"] == [message]
    # The own address again is fine.
    response = ada_client.patch("/api/auth/account/", {"email": "ada@EXAMPLE.net"}, format="json")
    assert response.status_code == 200


def test_accounts_without_email_can_add_one():
    User.objects.create_user("old", password=PASSWORD)
    client = login("old")[0]
    assert client.get("/api/auth/account/").data["email"] == ""
    response = client.patch("/api/auth/account/", {"email": "old@example.com"}, format="json")
    assert response.status_code == 200
    assert User.objects.get(username="old").email == "old@example.com"


def test_change_password(ada):
    client, tokens = login("ada")
    other_session = login("ada")[1]

    wrong = {"current_password": "nope", "new_password": "a-new-long-password"}
    response = client.post("/api/auth/account/password/", wrong, format="json")
    assert response.status_code == 400
    assert response.data["current_password"] == ["The current password is wrong."]

    weak = {"current_password": PASSWORD, "new_password": "123"}
    response = client.post("/api/auth/account/password/", weak, format="json")
    assert response.status_code == 400
    assert any("too short" in message for message in response.data["new_password"])

    good = {"current_password": PASSWORD, "new_password": "a-new-long-password"}
    response = client.post("/api/auth/account/password/", good, format="json")
    assert response.status_code == 200
    new_tokens = response.data
    assert set(new_tokens) == {"access", "refresh"}

    # The old password is gone, every older login ended, the new tokens work.
    old = {"username": "ada", "password": PASSWORD}
    bad = APIClient().post("/api/auth/token/", old, format="json")
    assert bad.status_code == 401
    login("ada", "a-new-long-password")
    assert refresh(tokens["refresh"]).status_code == 401
    assert refresh(other_session["refresh"]).status_code == 401
    assert refresh(new_tokens["refresh"]).status_code == 200


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #


def test_export_holds_the_own_data(ada, ada_client, ada_data):
    archive = export(ada_client)
    names = set(archive.namelist())
    for name in [
        "README.txt",
        "account.json",
        "projects.json",
        "memberships.json",
        "comments.json",
        "api_tokens.json",
        "connected_apps.json",
        "data_sources.json",
        "data_preparations.json",
        "datasets.json",
        "training_runs.json",
        "files.json",
    ]:
        assert name in names

    account = read_json(archive, "account.json")
    assert account["username"] == "ada"
    assert account["email"] == "ada@example.com"

    projects = read_json(archive, "projects.json")
    assert [p["name"] for p in projects] == ["Ada's net"]
    assert len(projects[0]["graph"]["nodes"]) == 2
    assert projects[0]["number_of_members"] == 1
    assert len(projects[0]["history"]) >= 1
    assert all(entry["saved_by_you"] for entry in projects[0]["history"])

    memberships = read_json(archive, "memberships.json")
    assert memberships == [
        {
            "project": {"id": ada_data["theirs"]["id"], "name": "Shared net"},
            "role": "editor",
            "since": memberships[0]["since"],
        }
    ]
    texts = sorted(c["text"] for c in read_json(archive, "comments.json"))
    assert texts == ["Ada on her own net", "Ada starts a thread"]

    tokens = read_json(archive, "api_tokens.json")
    assert [t["name"] for t in tokens] == ["script"]
    assert set(tokens[0]) == {
        "name",
        "prefix",
        "read_only",
        "created_at",
        "last_used_at",
        "expires_at",
    }
    assert [c["app"] for c in read_json(archive, "connected_apps.json")] == ["Claude"]

    assert [s["name"] for s in read_json(archive, "data_sources.json")] == ["churn"]
    preps = read_json(archive, "data_preparations.json")
    assert preps[0]["data_source"] == ada_data["source"]["id"]
    assert preps[0]["settings"]["target"] == "churn"
    assert sorted(d["name"] for d in read_json(archive, "datasets.json")) == ["Iris", "pets"]
    runs = read_json(archive, "training_runs.json")
    assert [r["name"] for r in runs] == ["Ada's run"]
    assert runs[0]["status"] == "completed"
    assert runs[0]["graph"] == DEMO_GRAPH

    # The uploaded files themselves.
    files = {entry["name"]: entry for entry in read_json(archive, "files.json")}
    assert all(entry["included"] for entry in files.values())
    source_id = ada_data["source"]["id"]
    assert archive.read(f"files/data_sources/{source_id}/churn.csv") == churn_csv()
    assert archive.read(files["Iris"]["path"]) == iris_csv()
    image_folder = files["pets"]["path"]
    assert image_folder.endswith("/images/")
    assert {name for name in names if name.startswith(image_folder)} == {
        f"{image_folder}cat/1.png",
        f"{image_folder}cat/2.png",
        f"{image_folder}dog/1.png",
        f"{image_folder}dog/2.png",
    }


def test_export_holds_nothing_of_others(ada, ada_client, ada_data):
    archive = export(ada_client)
    everything = b"".join(archive.read(name) for name in archive.namelist())
    for secret in [
        b"mallory",
        b"Mallory",
        b"A reply to Ada",
        b"Mallory's own thread",
        b"Mallory secretly comments",
        b"Mallory data",
        ada_data["key"].encode(),
    ]:
        assert secret not in everything, secret
    key_hash = ApiToken.objects.get(owner=ada).key_hash
    assert key_hash.encode() not in everything


def test_export_lists_files_beyond_the_limit(ada, ada_client, ada_data, settings):
    # Room for the Iris CSV file only; the larger churn file comes first and does not fit.
    assert len(churn_csv()) > len(iris_csv())
    settings.EXPORT_MAX_FILES_MB = (len(iris_csv()) + 1) / 1024 / 1024
    archive = export(ada_client)
    files = {entry["name"]: entry for entry in read_json(archive, "files.json")}
    churn = files["churn"]
    assert churn["included"] is False
    assert churn["note"].startswith("Not included: the export holds at most")
    assert "path" not in churn
    assert files["Iris"]["included"] is True
    assert files["pets"]["included"] is False
    included = [name for name in archive.namelist() if name.startswith("files/")]
    assert included == [files["Iris"]["path"]]


def test_export_reports_missing_files(ada, ada_client, ada_data, dirs):
    source_id = ada_data["source"]["id"]
    (dirs / "prepare" / "sources" / str(source_id) / "original.csv").unlink()
    archive = export(ada_client)
    churn = next(e for e in read_json(archive, "files.json") if e["kind"] == "data_source")
    assert churn["included"] is False
    assert churn["note"] == "The file is missing on the server."


# --------------------------------------------------------------------------- #
# Deleting the account
# --------------------------------------------------------------------------- #


def test_deleting_needs_the_password_and_the_username(ada, ada_client):
    url = "/api/auth/account/delete/"
    response = ada_client.post(url, {"password": "wrong", "username": "ada"}, format="json")
    assert response.status_code == 400
    assert response.data["password"] == ["The password is wrong."]
    response = ada_client.post(url, {"password": PASSWORD, "username": "Ada"}, format="json")
    assert response.status_code == 400
    assert response.data["username"] == ["Type your username exactly to confirm."]
    response = ada_client.post(url, {}, format="json")
    assert response.status_code == 400
    assert User.objects.filter(pk=ada.pk).exists()


def test_deleting_the_account_removes_everything(
    ada, ada_data, dirs, mallory, mallory_client, settings, monkeypatch,
    django_capture_on_commit_callbacks,
):  # fmt: skip
    client, tokens = login("ada")
    # A training that is still running is stopped first.
    use_runner(settings, "hang")
    active = start(client, name="Still running").data
    process = training_service._PROCESSES[active["id"]]
    wait_for_state(dirs / "training", active["id"])

    notified = []
    monkeypatch.setattr(live, "notify_deleted", lambda pid: notified.append(("deleted", pid)))
    monkeypatch.setattr(
        live, "notify_access", lambda pid, uid, role: notified.append(("access", pid, uid, role))
    )
    monkeypatch.setattr(
        live, "notify_comment", lambda pid, action, c: notified.append((action, pid, c["id"]))
    )
    folders = [
        dirs / "prepare" / "sources" / str(ada_data["source"]["id"]),
        dirs / "training" / "datasets" / str(ada_data["csv_dataset"]["id"]),
        dirs / "training" / "datasets" / str(ada_data["images"]["id"]),
        dirs / "training" / "runs" / ada_data["run"]["id"],
        dirs / "training" / "runs" / active["id"],
    ]
    assert all(folder.is_dir() for folder in folders)
    ada_id, own_id, theirs_id = ada.pk, ada_data["own"]["id"], ada_data["theirs"]["id"]

    with django_capture_on_commit_callbacks(execute=True):
        response = client.post(
            "/api/auth/account/delete/", {"password": PASSWORD, "username": "ada"}, format="json"
        )
    assert response.status_code == 204

    # The account and everything it owned, with the files.
    assert not User.objects.filter(pk=ada_id).exists()
    assert not UserProfile.objects.filter(user_id=ada_id).exists()
    assert not Project.objects.filter(pk=own_id).exists()
    assert not ProjectVersion.objects.filter(project_id=own_id).exists()
    for model in (DataSource, DataPrep, Dataset, TrainingRun, ApiToken, OAuthConnection):
        assert not model.objects.filter(owner_id=ada_id).exists(), model
    assert not ProjectMember.objects.filter(user_id=ada_id).exists()
    assert not Comment.objects.filter(author_id=ada_id).exists()
    assert not any(folder.exists() for folder in folders)
    assert process.wait(timeout=5) == -signal.SIGKILL

    # Mallory keeps her project, her own thread, her dataset and her run (without project);
    # her reply to Ada's thread went with the thread.
    assert Project.objects.filter(pk=theirs_id).exists()
    assert list(Comment.objects.values_list("text", flat=True)) == ["Mallory's own thread"]
    assert Dataset.objects.filter(pk=ada_data["their_dataset"]["id"]).exists()
    their_run = TrainingRun.objects.get(pk=ada_data["their_run"]["id"])
    assert their_run.project_id is None
    assert (dirs / "training" / "runs" / str(their_run.pk)).is_dir()
    assert mallory_client.get(f"/api/projects/{theirs_id}/").status_code == 200
    assert mallory_client.get(f"/api/projects/{own_id}/").status_code == 404

    # Live editors were told.
    assert ("deleted", own_id) in notified
    assert ("access", theirs_id, ada_id, None) in notified
    assert ("deleted", theirs_id, ada_data["thread"]["id"]) in notified
    assert ("deleted", own_id, ada_data["own_comment"]["id"]) not in notified

    # Every login ended, the API token too.
    assert refresh(tokens["refresh"]).status_code == 401
    assert client.get("/api/auth/me/").status_code == 401
    api = APIClient()
    api.credentials(HTTP_AUTHORIZATION=f"Bearer {ada_data['key']}")
    assert api.get("/api/projects/").status_code == 401


def test_refreshing_a_login_of_an_account_deleted_elsewhere(ada):
    _, tokens = login("ada")
    ada.delete()  # e.g. in the Django admin, without blacklisting the tokens
    response = refresh(tokens["refresh"])
    assert response.status_code == 401


def test_delete_account_function_keeps_running_trainings_if_asked(
    ada, dirs, settings, django_capture_on_commit_callbacks
):
    """The purge command (another container) never kills processes it cannot see."""
    client = login("ada")[0]
    use_runner(settings, "wait")
    run = start(client).data
    wait_for_state(dirs / "training", run["id"])
    called = []
    original = training_service.terminate_run
    training_service.terminate_run = lambda r: called.append(r.pk)
    try:
        with django_capture_on_commit_callbacks(execute=True):
            accounts.delete_account(ada, stop_trainings=False)
    finally:
        training_service.terminate_run = original
    assert called == []
    assert not User.objects.filter(username="ada").exists()
