"""``manage.py backup`` and ``manage.py restore`` (brick_dnn_core.backups), on SQLite and, with
DATABASE_URL, on PostgreSQL."""

from __future__ import annotations

import io
import json
import tarfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.core.management.base import CommandError
from django.db import connection
from rest_framework.test import APIClient

from brick_dnn_core import backups
from brick_dnn_core.models import Project

User = get_user_model()


@pytest.fixture
def data(settings, tmp_path):
    """A data directory with files of every kind, and a backup folder inside it (as on the
    server: /data/backups)."""
    root = tmp_path / "data"
    (root / "training" / "run-1").mkdir(parents=True)
    (root / "training" / "run-1" / "weights.pt").write_bytes(b"w" * 2048)
    (root / "prepare").mkdir()
    (root / "prepare" / "table.parquet").write_bytes(b"p" * 100)
    (root / "secret_key").write_text("key")
    (root / "cache").mkdir()
    (root / "cache" / "throttle").write_text("not backed up")
    (root / "db.sqlite3").write_text("not backed up: the database part has it")
    settings.DATA_DIR = root
    settings.BACKUP_DIR = str(root / "backups")
    settings.BACKUP_FILES = "archive"
    settings.BACKUP_MAX_FILES_MB = 100
    settings.HEALTH_MIN_FREE_MB = 0
    return root


def run(*args) -> str:
    out = io.StringIO()
    call_command(*args, stdout=out)
    return out.getvalue()


def names() -> list[str]:
    return sorted(User.objects.values_list("username", flat=True))


def test_off_without_a_backup_folder(settings):
    settings.BACKUP_DIR = ""
    assert "Backups are off" in run("backup")


def fake(when: datetime, tmp_path: Path) -> backups.Backup:
    return backups.Backup(tmp_path / (backups.PREFIX + when.strftime(backups.STAMP)), when)


def test_rotation_keeps_daily_and_weekly_backups(tmp_path):
    start = datetime(2026, 1, 1, 3, 0, tzinfo=UTC)
    every_day = [fake(start + timedelta(days=day), tmp_path) for day in range(60)]
    # Two backups on the last day (a deploy restarts the cleanup service): the newer is kept.
    extra = fake(start + timedelta(days=59, hours=5), tmp_path)
    all_backups = [*every_day, extra]
    deleted = {b.path for b in backups.to_delete(all_backups, daily=7, weekly=4)}
    kept = sorted((b for b in all_backups if b.path not in deleted), key=lambda b: b.created)
    days = [b.created.date() for b in kept]
    assert extra in kept and every_day[-1] not in kept
    # The last seven days, plus the newest of the four weeks: 7 + up to 3 older ones.
    assert days[-7:] == [(start + timedelta(days=d)).date() for d in range(53, 60)]
    assert 8 <= len(kept) <= 10
    weeks = {b.created.isocalendar()[:2] for b in kept}
    assert len(weeks) == 4
    # Never everything: the newest stays even with nothing to keep.
    assert extra not in backups.to_delete(all_backups, daily=0, weekly=0)


@pytest.mark.django_db(transaction=True)
def test_backup_and_restore_of_database_and_files(data, settings):
    ada = User.objects.create_user("ada", password="correct-horse")
    Project.objects.create(owner=ada, name="mlp")
    output = run("backup")
    assert "Backup brickdnn-" in output

    [backup] = backups.list_backups(Path(settings.BACKUP_DIR))
    manifest = backup.manifest()
    expected = "sqlite" if connection.vendor == "sqlite" else ("pgdump", "json")
    assert manifest["database"]["format"] in expected
    assert manifest["files"]["mode"] == "archive"
    assert manifest["migrations"]["brick_dnn_core"]
    with tarfile.open(backup.path / backups.ARCHIVE_FILE) as archive:
        members = sorted(archive.getnames())
    # No cache, no SQLite file, no backups of backups.
    assert members == ["prepare/table.parquet", "secret_key", "training/run-1/weights.pt"]
    assert oct(backup.path.stat().st_mode)[-3:] == "700"

    # Things go wrong after the backup.
    Project.objects.all().delete()
    User.objects.create_user("eve")
    (data / "training" / "run-1" / "weights.pt").unlink()

    output = run("restore", "latest", "--no-input")
    assert "Restored the database and the files" in output
    assert names() == ["ada"]
    assert User.objects.get(username="ada").check_password("correct-horse")
    assert Project.objects.get().name == "mlp"
    assert (data / "training" / "run-1" / "weights.pt").read_bytes() == b"w" * 2048
    # New rows get new ids after a restore (sequences on PostgreSQL).
    Project.objects.create(owner=User.objects.get(username="ada"), name="cnn")


@pytest.mark.django_db(transaction=True)
def test_json_backups_restore_on_either_database(data, settings):
    ada = User.objects.create_user("ada", password="pw")
    Project.objects.create(owner=ada, name="gp", graph={"version": 1, "nodes": [], "edges": []})
    run("backup", "--database-format", "json", "--files", "none")
    [backup] = backups.list_backups(Path(settings.BACKUP_DIR))
    assert backup.manifest()["database"]["format"] == "json"
    assert not (backup.path / backups.ARCHIVE_FILE).exists()

    User.objects.all().delete()
    User.objects.create_user("eve")
    run("restore", backup.name, "--no-input", "--database-only")
    assert names() == ["ada"]
    assert Project.objects.get().graph == {"version": 1, "nodes": [], "edges": []}
    User.objects.create_user("bob")  # the id sequence continues after the loaded rows
    assert names() == ["ada", "bob"]


@pytest.mark.django_db(transaction=True)
def test_native_postgresql_backup(data, settings):
    if connection.vendor != "postgresql":
        pytest.skip("PostgreSQL only")
    try:
        backups.database_format("native")
    except backups.BackupError as exc:
        pytest.skip(str(exc))
    User.objects.create_user("ada")
    run("backup", "--database-format", "native", "--files", "none")
    [backup] = backups.list_backups(Path(settings.BACKUP_DIR))
    assert (backup.path / backups.PGDUMP_FILE).is_file()
    User.objects.all().delete()
    run("restore", "latest", "--no-input")
    assert names() == ["ada"]
    User.objects.create_user("bob")


@pytest.mark.django_db
def test_large_files_are_listed_instead_of_archived(data, settings):
    settings.BACKUP_MAX_FILES_MB = 1
    (data / "training" / "run-1" / "big.bin").write_bytes(b"x" * (2 * 1024**2))
    output = run("backup", "--database-format", "json")
    assert "writing a list of them instead" in output
    [backup] = backups.list_backups(Path(settings.BACKUP_DIR))
    assert backup.manifest()["files"]["mode"] == "list"
    listing = (backup.path / backups.LIST_FILE).read_text()
    assert "training/run-1/big.bin\t2097152\t" in listing
    assert "cache/" not in listing
    # Restoring files from a list only says where to get them.
    assert "has no copy of the files" in run("restore", "latest", "--no-input", "--files-only")


@pytest.mark.django_db
def test_restore_asks_first(data, monkeypatch):
    run("backup", "--database-format", "json")
    monkeypatch.setattr("builtins.input", lambda prompt: "no")
    with pytest.raises(CommandError, match="Nothing restored"):
        run("restore", "latest")
    with pytest.raises(CommandError, match="no backup 'brickdnn-19990101T000000Z'"):
        run("restore", "brickdnn-19990101T000000Z", "--no-input")


@pytest.mark.django_db
def test_an_unfinished_backup_does_not_count_and_is_cleaned_up(data, settings):
    directory = Path(settings.BACKUP_DIR)
    directory.mkdir()
    partial = directory / ".partial-brickdnn-20260101T000000Z"
    partial.mkdir()
    assert backups.list_backups(directory) == []
    import os

    os.utime(partial, (0, 0))
    run("backup", "--database-format", "json")
    assert not partial.exists()
    assert len(backups.list_backups(directory)) == 1


@pytest.mark.django_db
def test_health_reports_failed_and_stale_backups(data, settings, monkeypatch):
    settings.TRAINING_WORKER = False
    settings.BACKUP_MAX_AGE_HOURS = 50
    client = APIClient()
    # Not run yet: not part of the check.
    assert "backup" not in client.get("/api/health/").data["checks"]

    run("backup", "--database-format", "json")
    assert client.get("/api/health/").data["checks"]["backup"] == "ok"

    def broken(*args, **kwargs):
        raise backups.BackupError("disk full")

    monkeypatch.setattr(backups, "dump_database", broken)
    with pytest.raises(CommandError, match="disk full"):
        run("backup", "--database-format", "json")
    response = client.get("/api/health/")
    assert response.data["status"] == "degraded"
    assert response.data["checks"]["backup"] == "failed"
    status = json.loads((Path(settings.BACKUP_DIR) / backups.STATUS_FILE).read_text())
    assert status == {"ok": False, "error": "BackupError: disk full", "time": status["time"]}
    monkeypatch.undo()

    # A good backup that is too old.
    old = datetime.now(UTC) - timedelta(hours=60)
    backups.create(Path(settings.BACKUP_DIR), database="json", now=old)
    for backup in backups.list_backups(Path(settings.BACKUP_DIR))[:-1]:
        import shutil

        shutil.rmtree(backup.path)
    assert client.get("/api/health/").data["checks"]["backup"] == "failed"


def test_restore_refuses_a_backup_of_another_database(tmp_path, settings):
    folder = tmp_path / "brickdnn-20260101T000000Z"
    folder.mkdir()
    (folder / "database.pgdump").write_bytes(b"PGDMP")
    (folder / "manifest.json").write_text(
        json.dumps({"database": {"format": "pgdump", "file": "database.pgdump"}})
    )
    backup = backups.find(str(folder))
    if connection.vendor == "postgresql":
        pytest.skip("the other way round needs SQLite")
    with pytest.raises(backups.BackupError, match="needs a PostgreSQL database"):
        backups.restore_database(backup)
