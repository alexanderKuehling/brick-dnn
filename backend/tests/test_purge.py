"""When data was last used, and deleting old data with ``manage.py purge_old_data``."""

from __future__ import annotations

import io
import re
from datetime import timedelta

import pytest
from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.utils import timezone
from rest_framework.test import APIClient
from rest_framework_simplejwt.token_blacklist.models import OutstandingToken

from brick_dnn_core import retention
from brick_dnn_core.api_tokens import create_token
from brick_dnn_core.models import (
    DataPrep,
    Dataset,
    DataSource,
    OAuthClient,
    OAuthCode,
    Project,
    TrainingRun,
    UserProfile,
)

from .test_account import PASSWORD, dirs, login, project  # noqa: F401 (fixture)
from .test_data_api import churn_csv, create_prep, import_csv, preview
from .test_training_api import iris_csv, start, upload, use_runner, wait_for, wait_for_state

pytestmark = pytest.mark.django_db

User = get_user_model()
LONG_AGO = timedelta(days=400)


def ago(delta: timedelta):
    return timezone.now() - delta


def age(model, pk, delta: timedelta = LONG_AGO) -> None:
    """Pretend ``model`` #pk was last used ``delta`` ago."""
    model.objects.filter(pk=pk).update(last_used_at=ago(delta))


def make_inactive(user, delta: timedelta = LONG_AGO) -> None:
    UserProfile.objects.update_or_create(user=user, defaults={"last_active_at": ago(delta)})
    User.objects.filter(pk=user.pk).update(last_login=ago(delta), date_joined=ago(delta))


def purge_command(*args: str) -> str:
    out = io.StringIO()
    call_command("purge_old_data", *args, stdout=out)
    return out.getvalue()


@pytest.fixture
def ada():
    return User.objects.create_user("ada", email="ada@example.com", password=PASSWORD)


@pytest.fixture
def ada_client(ada):
    return login("ada")[0]


@pytest.fixture
def uploads(dirs, ada_client):  # noqa: F811
    """A data source with a preparation, a dataset and a finished run of Ada's."""
    source = import_csv(ada_client, "churn.csv", churn_csv()).data
    prep = create_prep(ada_client, source["id"], settings={"target": "churn"}).data
    dataset = upload(ada_client, "iris.csv", iris_csv()).data
    run = start(ada_client).data
    wait_for(ada_client, run["id"])
    TrainingRun.objects.filter(pk=run["id"]).update(finished_at=ago(LONG_AGO))
    DataPrep.objects.filter(pk=prep["id"]).update(updated_at=ago(LONG_AGO))
    return {"source": source, "prep": prep, "dataset": dataset, "run": run}


# --------------------------------------------------------------------------- #
# Last used
# --------------------------------------------------------------------------- #


def test_opening_and_using_uploads_counts_as_use(uploads, ada_client):
    source_id, prep_id = uploads["source"]["id"], uploads["prep"]["id"]
    dataset_id, run_id = uploads["dataset"]["id"], uploads["run"]["id"]
    recent = timezone.now() - timedelta(minutes=1)

    age(DataSource, source_id)
    ada_client.get("/api/data/sources/")  # listing is not using
    assert DataSource.objects.get(pk=source_id).last_used_at < recent
    ada_client.get(f"/api/data/sources/{source_id}/")
    assert DataSource.objects.get(pk=source_id).last_used_at > recent

    age(DataSource, source_id)
    ada_client.get(f"/api/data/preps/{prep_id}/")  # opening a preparation keeps its data
    assert DataSource.objects.get(pk=source_id).last_used_at > recent
    age(DataSource, source_id)
    assert preview(ada_client, prep_id).status_code == 200
    assert DataSource.objects.get(pk=source_id).last_used_at > recent

    age(Dataset, dataset_id)
    ada_client.get("/api/training/datasets/")
    assert Dataset.objects.get(pk=dataset_id).last_used_at < recent
    ada_client.get(f"/api/training/datasets/{dataset_id}/")
    assert Dataset.objects.get(pk=dataset_id).last_used_at > recent
    age(Dataset, dataset_id)
    check = {"graph": uploads["run"]["graph"], "dataset": {"kind": "upload", "id": dataset_id}}
    ada_client.post("/api/training/check/", check, format="json")
    assert Dataset.objects.get(pk=dataset_id).last_used_at > recent

    age(TrainingRun, run_id)
    ada_client.get("/api/training/runs/")
    assert TrainingRun.objects.get(pk=run_id).last_used_at < recent
    ada_client.get(f"/api/training/runs/{run_id}/")
    assert TrainingRun.objects.get(pk=run_id).last_used_at > recent


def test_last_used_is_written_at_most_once_an_hour(uploads, ada_client, django_assert_num_queries):
    dataset = Dataset.objects.get(pk=uploads["dataset"]["id"])
    first = dataset.last_used_at
    retention.touch(dataset)  # created a moment ago: nothing to write
    assert Dataset.objects.get(pk=dataset.pk).last_used_at == first
    dataset.last_used_at = ago(timedelta(hours=2))
    with django_assert_num_queries(1):
        retention.touch(dataset)
    assert Dataset.objects.get(pk=dataset.pk).last_used_at > first


def test_logins_refreshes_and_api_tokens_count_as_activity(ada):
    make_inactive(ada)
    client, tokens = login("ada")
    profile = UserProfile.objects.get(user=ada)
    assert profile.last_active_at > ago(timedelta(minutes=1))

    make_inactive(ada)
    response = APIClient().post(
        "/api/auth/token/refresh/", {"refresh": tokens["refresh"]}, format="json"
    )
    assert response.status_code == 200
    assert UserProfile.objects.get(user=ada).last_active_at > ago(timedelta(minutes=1))

    make_inactive(ada)
    _, key = create_token(ada, "script")
    api = APIClient()
    api.credentials(HTTP_AUTHORIZATION=f"Bearer {key}")
    assert api.get("/api/projects/").status_code == 200
    assert UserProfile.objects.get(user=ada).last_active_at > ago(timedelta(minutes=1))


def test_accounts_without_a_profile_get_one_when_used(ada):
    UserProfile.objects.filter(user=ada).delete()
    login("ada")
    assert UserProfile.objects.filter(user=ada).exists()


# --------------------------------------------------------------------------- #
# Purging uploads and runs
# --------------------------------------------------------------------------- #


def test_purge_is_off_by_default(uploads, ada, settings):
    settings.DELETE_UNUSED_UPLOADS_DAYS = 0
    settings.DELETE_INACTIVE_ACCOUNTS_DAYS = 0
    age(DataSource, uploads["source"]["id"])
    age(Dataset, uploads["dataset"]["id"])
    age(TrainingRun, uploads["run"]["id"])
    make_inactive(ada)
    output = purge_command()
    assert "never (off)" in output
    assert "Deleted 0 accounts, 0 data sources, 0 preparations, 0 datasets, 0 training" in output
    assert DataSource.objects.count() == Dataset.objects.count() == TrainingRun.objects.count() == 1
    assert User.objects.filter(pk=ada.pk).exists()


def test_purge_unused_uploads(uploads, dirs, settings, django_capture_on_commit_callbacks):  # noqa: F811
    settings.DELETE_UNUSED_UPLOADS_DAYS = 90
    source_id, prep_id = uploads["source"]["id"], uploads["prep"]["id"]
    dataset_id, run_id = uploads["dataset"]["id"], uploads["run"]["id"]
    folders = [
        dirs / "prepare" / "sources" / str(source_id),
        dirs / "training" / "datasets" / str(dataset_id),
        dirs / "training" / "runs" / run_id,
    ]
    # Used 89 days ago: kept.
    age(DataSource, source_id, timedelta(days=89))
    age(Dataset, dataset_id, timedelta(days=89))
    age(TrainingRun, run_id, timedelta(days=89))
    assert "Deleted 0 accounts, 0 data sources" in purge_command()

    age(DataSource, source_id, timedelta(days=91))
    age(Dataset, dataset_id, timedelta(days=91))
    age(TrainingRun, run_id, timedelta(days=91))
    output = purge_command("--dry-run")
    assert f"Would delete data source #{source_id} of user #" in output
    assert f"with its preparation #{prep_id}" in output
    assert f"Would delete dataset #{dataset_id}" in output
    assert f"Would delete training run {run_id}" in output
    # Users are named by id only (as a word: run ids are random hex and may contain "ada").
    assert not re.search(r"\bada\b", output)
    assert "Would delete 0 accounts, 1 data sources, 1 preparations, 1 datasets, 1" in output
    assert DataSource.objects.count() == DataPrep.objects.count() == 1
    assert all(folder.is_dir() for folder in folders)

    with django_capture_on_commit_callbacks(execute=True):
        output = purge_command()
    assert "Deleted 0 accounts, 1 data sources, 1 preparations, 1 datasets, 1 training" in output
    for model in (DataSource, DataPrep, Dataset, TrainingRun):
        assert not model.objects.exists(), model
    assert not any(folder.exists() for folder in folders)
    assert User.objects.filter(username="ada").exists()


def test_purge_options_override_the_settings(uploads, settings):
    settings.DELETE_UNUSED_UPLOADS_DAYS = 0
    age(Dataset, uploads["dataset"]["id"], timedelta(days=40))
    output = purge_command("--dry-run", "--unused-uploads-days", "30")
    assert "after 30 days without use" in output
    assert "1 datasets" in output
    settings.DELETE_UNUSED_UPLOADS_DAYS = 30
    assert "0 datasets" in purge_command("--dry-run", "--unused-uploads-days", "0")


def test_purge_keeps_what_is_in_use(uploads, ada_client, dirs, settings):  # noqa: F811
    cutoff = ago(timedelta(days=30))
    source_id, prep_id = uploads["source"]["id"], uploads["prep"]["id"]
    # A preparation changed recently keeps its old source.
    age(DataSource, source_id)
    assert list(retention.unused_sources(cutoff)) != []
    DataPrep.objects.filter(pk=prep_id).update(updated_at=timezone.now())
    assert list(retention.unused_sources(cutoff)) == []

    # A dataset an active training uses is kept, and so is the active run.
    use_runner(settings, "wait")
    dataset_id = uploads["dataset"]["id"]
    run = start(ada_client, dataset={"kind": "upload", "id": dataset_id}).data
    assert run["status"] in ("starting", "preparing", "running"), run
    wait_for_state(dirs / "training", run["id"])
    age(Dataset, dataset_id)
    age(TrainingRun, run["id"])
    assert list(retention.unused_datasets(cutoff)) == []
    assert [r.pk for r in retention.unused_runs(cutoff)] == []  # the finished one: too recent

    # A run that finished recently is kept even if nobody opened it.
    finished = TrainingRun.objects.get(pk=uploads["run"]["id"])
    age(TrainingRun, finished.pk)
    TrainingRun.objects.filter(pk=finished.pk).update(finished_at=timezone.now())
    assert list(retention.unused_runs(cutoff)) == []


def test_the_purge_does_not_look_at_training_processes(uploads, ada_client, monkeypatch):
    """It runs in another container: training processes would look dead there."""
    from brick_dnn_core import training_service

    def forbidden(*args, **kwargs):
        raise AssertionError("purge_old_data must not sync runs")

    monkeypatch.setattr(training_service, "sync_run", forbidden)
    monkeypatch.setattr(training_service, "sync_runs", forbidden)
    retention.purge(30, 30, dry_run=True)


# --------------------------------------------------------------------------- #
# Purging inactive accounts
# --------------------------------------------------------------------------- #


def test_purge_inactive_accounts(uploads, ada, dirs, settings, django_capture_on_commit_callbacks):  # noqa: F811
    settings.DELETE_INACTIVE_ACCOUNTS_DAYS = 365
    active = User.objects.create_user("bob", password=PASSWORD)
    admin = User.objects.create_user("root", password=PASSWORD, is_staff=True)
    make_inactive(admin)
    never_logged_in = User.objects.create_user("cy", password=PASSWORD)
    User.objects.filter(pk=never_logged_in.pk).update(date_joined=ago(LONG_AGO))
    project(login("bob")[0], "Bob's net")

    make_inactive(ada, timedelta(days=300))
    output = purge_command("--dry-run")
    assert f"account #{ada.pk}" not in output
    # Signed up long ago and never logged in (no profile): counts from the sign-up.
    assert f"Would delete account #{never_logged_in.pk}: last active" in output

    make_inactive(ada)
    output = purge_command("--dry-run")
    assert f"Would delete account #{ada.pk}: last active" in output
    assert f"account #{admin.pk}" not in output and f"account #{active.pk}" not in output
    assert User.objects.filter(pk=ada.pk).exists()

    with django_capture_on_commit_callbacks(execute=True):
        output = purge_command()
    assert "Deleted 2 accounts, 0 data sources" in output
    assert set(User.objects.values_list("username", flat=True)) == {"bob", "root"}
    assert not DataSource.objects.exists() and not TrainingRun.objects.exists()
    assert not (dirs / "training" / "runs" / uploads["run"]["id"]).exists()
    assert Project.objects.filter(owner=active).exists()


def test_recent_api_token_use_keeps_an_account(ada):
    make_inactive(ada)
    token, _ = create_token(ada, "script")
    cutoff = ago(timedelta(days=30))
    assert list(retention.inactive_users(cutoff)) == [ada]
    token.last_used_at = timezone.now()
    token.save()
    assert list(retention.inactive_users(cutoff)) == []


def test_accounts_with_an_active_training_are_kept(ada, ada_client, dirs, settings):  # noqa: F811
    use_runner(settings, "wait")
    run = start(ada_client).data
    wait_for_state(dirs / "training", run["id"])
    make_inactive(ada)
    assert list(retention.inactive_users(ago(timedelta(days=30)))) == []


def test_expired_logins_are_removed(ada, settings):
    settings.DELETE_INACTIVE_ACCOUNTS_DAYS = 0
    login("ada")
    login("ada")
    OutstandingToken.objects.filter(pk=OutstandingToken.objects.first().pk).update(
        expires_at=ago(timedelta(days=1))
    )
    client = OAuthClient.objects.create(client_id="c", name="App")
    OAuthCode.objects.create(
        code_hash="h" * 64,
        client=client,
        owner=ada,
        redirect_uri="https://example.com/cb",
        code_challenge="x" * 43,
        expires_at=ago(timedelta(minutes=1)),
    )
    assert "Would delete 0 accounts" in purge_command("--dry-run")
    assert OutstandingToken.objects.count() == 2
    assert "and 2 expired login tokens" in purge_command()
    assert OutstandingToken.objects.count() == 1
    assert not OAuthCode.objects.exists()


def test_bad_options():
    from django.core.management.base import CommandError

    with pytest.raises(CommandError):
        call_command("purge_old_data", "--unused-uploads-days", "-3")
