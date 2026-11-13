"""The training worker waits for the database migrations before it takes trainings: on a
deploy the web server migrates while it starts, and the worker must not query new columns
before they exist."""

from __future__ import annotations

from io import StringIO

import pytest

from brick_dnn_core.management.commands import training_worker as command


@pytest.fixture
def worker_runs(monkeypatch):
    started = []

    class FakeWorker:
        def __init__(self, **kwargs):
            pass

        def run(self):
            started.append(True)

    monkeypatch.setattr(command, "Worker", FakeWorker)
    monkeypatch.setattr(command.queue, "enabled", lambda: True)
    monkeypatch.setattr(command, "touch_alive_file", lambda: None)
    monkeypatch.setattr(command, "MIGRATION_POLL", 0)
    return started


@pytest.mark.django_db
def test_the_test_database_has_every_migration():
    assert command.pending_migrations() == []


def test_starts_at_once_when_the_database_is_up_to_date(worker_runs, monkeypatch):
    monkeypatch.setattr(command, "pending_migrations", lambda: [])
    out = StringIO()
    command.Command(stdout=out).handle(once=False, slots=None, name=None, poll=2, idle_exit=None)
    assert worker_runs == [True]
    assert "Waiting" not in out.getvalue()


def test_waits_until_the_web_server_has_migrated(worker_runs, monkeypatch):
    answers = iter([["0020"], ["0020"], []])
    monkeypatch.setattr(command, "pending_migrations", lambda: next(answers))
    out = StringIO()
    command.Command(stdout=out).handle(once=False, slots=None, name=None, poll=2, idle_exit=None)
    assert worker_runs == [True]
    assert "Waiting for the web server to migrate the database" in out.getvalue()
    assert "The database is up to date, starting." in out.getvalue()


def test_ends_when_the_migrations_never_come(worker_runs, monkeypatch):
    monkeypatch.setattr(command, "pending_migrations", lambda: ["0020"])
    monkeypatch.setattr(command, "MIGRATION_TIMEOUT", 0)
    out = StringIO()
    cmd = command.Command(stdout=out)
    assert cmd._wait_for_migrations(timeout=0) is False
    assert "run `manage.py migrate`" in out.getvalue()
    assert worker_runs == []
