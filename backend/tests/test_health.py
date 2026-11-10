"""``GET /api/health/``: database, training worker and disk space (brick_dnn_core.health)."""

from __future__ import annotations

import shutil
from collections import namedtuple
from datetime import timedelta

import pytest
from django.contrib.auth import get_user_model
from django.db import OperationalError
from django.utils import timezone
from rest_framework.test import APIClient

from brick_dnn_core import health
from brick_dnn_core.models import TrainingWorker

pytestmark = pytest.mark.django_db

Usage = namedtuple("Usage", "total used free")
GB = 1024**3


@pytest.fixture
def client():
    return APIClient()


@pytest.fixture
def plenty_of_disk(monkeypatch, settings):
    settings.HEALTH_MIN_FREE_MB = 1024
    monkeypatch.setattr(shutil, "disk_usage", lambda path: Usage(40 * GB, 10 * GB, 30 * GB))


def test_all_fine_without_details_for_the_public(client, settings, plenty_of_disk):
    settings.TRAINING_WORKER = False
    response = client.get("/api/health/")
    assert response.status_code == 200
    assert response.data["status"] == "ok"
    assert response.data["checks"] == {"database": "ok", "disk": "ok"}
    assert "detail" not in response.data
    assert response["Cache-Control"] == "no-store"


def test_a_missing_or_silent_worker_degrades_the_site(client, settings, plenty_of_disk):
    settings.TRAINING_ENABLED = True
    settings.TRAINING_WORKER = True
    settings.WORKER_TIMEOUT = 60
    response = client.get("/api/health/")
    assert response.status_code == 200
    assert response.data["status"] == "degraded"
    assert response.data["checks"]["worker"] == "failed"

    worker = TrainingWorker.objects.create(name="w1", heartbeat_at=timezone.now())
    assert client.get("/api/health/").data["status"] == "ok"

    worker.heartbeat_at = timezone.now() - timedelta(seconds=120)
    worker.save()
    assert client.get("/api/health/").data["checks"]["worker"] == "failed"

    # A worker that stopped in an orderly way does not count either.
    worker.heartbeat_at = timezone.now()
    worker.stopped_at = timezone.now()
    worker.save()
    assert client.get("/api/health/").data["checks"]["worker"] == "failed"


def test_the_worker_is_not_checked_when_the_web_server_trains_itself(client, settings):
    settings.TRAINING_WORKER = False
    assert "worker" not in client.get("/api/health/").data["checks"]


def test_little_free_disk_space_degrades_the_site(client, settings, monkeypatch):
    settings.TRAINING_WORKER = False
    settings.HEALTH_MIN_FREE_MB = 2048
    monkeypatch.setattr(shutil, "disk_usage", lambda path: Usage(40 * GB, 39 * GB, GB))
    response = client.get("/api/health/")
    assert response.data["status"] == "degraded"
    assert response.data["checks"]["disk"] == "failed"
    settings.HEALTH_MIN_FREE_MB = 0  # off
    assert "disk" not in client.get("/api/health/").data["checks"]


def test_without_the_database_the_answer_is_503(client, settings, monkeypatch, plenty_of_disk):
    settings.TRAINING_WORKER = True
    settings.TRAINING_ENABLED = True

    def broken():
        return False, {"error": "OperationalError: connection refused"}

    monkeypatch.setitem(health.CHECKS, "database", broken)
    response = client.get("/api/health/")
    assert response.status_code == 503
    assert response.data["status"] == "error"
    # The worker check needs the database: it is left out rather than failing twice.
    assert response.data["checks"] == {"database": "failed", "disk": "ok"}
    assert "connection refused" not in str(response.content)


def test_a_check_that_raises_counts_as_failed(client, settings, monkeypatch, plenty_of_disk):
    settings.TRAINING_WORKER = False

    def raises():
        raise OperationalError("boom")

    monkeypatch.setitem(health.CHECKS, "disk", raises)
    response = client.get("/api/health/")
    assert response.data["checks"]["disk"] == "failed"
    assert "boom" not in str(response.content)


def test_details_for_the_secret_header(client, settings, plenty_of_disk):
    settings.TRAINING_WORKER = False
    settings.HEALTH_TOKEN = "s3cret-token"
    assert "detail" not in client.get("/api/health/", HTTP_X_HEALTH_TOKEN="wrong").data
    response = client.get("/api/health/", HTTP_X_HEALTH_TOKEN="s3cret-token")
    assert response.data["detail"]["disk"]["free_mb"] == 30 * 1024
    assert response.data["detail"]["database"]["vendor"]
    # Without a token configured, no header unlocks the details.
    settings.HEALTH_TOKEN = ""
    assert "detail" not in client.get("/api/health/", HTTP_X_HEALTH_TOKEN="").data


def test_details_for_staff_accounts_only(client, settings, plenty_of_disk):
    settings.TRAINING_WORKER = False
    user = get_user_model().objects.create_user("ada", password="x")
    client.force_authenticate(user)
    assert "detail" not in client.get("/api/health/").data
    user.is_staff = True
    user.save()
    client.force_authenticate(user)
    assert "disk" in client.get("/api/health/").data["detail"]
