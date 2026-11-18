"""``manage.py move_database`` (brick_dnn_core.database_move): which tables, how a copy is
verified, and, with DATABASE_URL on PostgreSQL and pg_dump of the server's version, a whole
move from the test database into a second one."""

from __future__ import annotations

import io
from urllib.parse import quote

import pytest
from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.core.management.base import CommandError
from django.db import connection

from brick_dnn_core import backups, database_move
from brick_dnn_core.database_move import (
    Check,
    MoveError,
    SequenceState,
    TableStats,
    check_sequences,
    check_unchanged,
    compare_tables,
)
from brick_dnn_core.models import Project

User = get_user_model()

SUPABASE = "postgresql://postgres.abc:secret@aws-0-eu-west-2.pooler.supabase.com:5432/postgres"
LOCAL = "postgresql://brickdnn:hunter2@db:5432/brickdnn"


def run(*args, **options) -> str:
    out = io.StringIO()
    call_command("move_database", *args, stdout=out, **options)
    return out.getvalue()


# --------------------------------------------------------------------------- #
# Which databases
# --------------------------------------------------------------------------- #


def test_urls_come_from_the_command_line_or_the_environment():
    assert database_move.resolve_url(LOCAL) == LOCAL
    assert database_move.resolve_url("env:TARGET", {"TARGET": f" {LOCAL} "}) == LOCAL
    with pytest.raises(MoveError, match="TARGET is not set"):
        database_move.resolve_url("env:TARGET", {})
    with pytest.raises(MoveError, match="No database URL"):
        database_move.resolve_url("")
    # SQLite (an empty DATABASE_URL) or anything else is not a PostgreSQL database.
    with pytest.raises(MoveError, match="Not a PostgreSQL database URL"):
        database_move.resolve_url("sqlite:///data/db.sqlite3")


def test_the_same_database_is_recognised_whoever_logs_in():
    assert database_move.same_database(LOCAL, "postgresql://other:pw@DB/brickdnn")
    assert database_move.same_database(LOCAL, "postgresql://brickdnn@db:5432/brickdnn?sslmode=off")
    assert not database_move.same_database(LOCAL, "postgresql://brickdnn:pw@db:5432/postgres")
    assert not database_move.same_database(LOCAL, SUPABASE)
    with pytest.raises(MoveError, match="same database"):
        database_move.move(LOCAL, "postgresql://other:pw@db/brickdnn")


def test_descriptions_leave_out_the_password():
    assert database_move.describe_url(SUPABASE) == (
        "postgres.abc@aws-0-eu-west-2.pooler.supabase.com:5432/postgres"
    )
    assert "hunter2" not in database_move.describe_url(LOCAL)


def test_tables_that_are_not_djangos_are_left_out():
    moved = ["auth_user", "django_migrations"]
    existing = ["auth_user", "django_migrations", "supabase_extra", "notes"]
    assert database_move.leftover_tables(moved, existing) == ["notes", "supabase_extra"]
    assert database_move.leftover_tables(moved, moved) == []


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #


def stats(**tables) -> dict[str, TableStats]:
    return {name: TableStats(rows, checksum) for name, (rows, checksum) in tables.items()}


def by_name(checks: list[Check]) -> dict[str, Check]:
    return {check.name: check for check in checks}


def test_identical_tables_pass():
    source = stats(auth_user=(3, "123"), brick_dnn_core_project=(10, "456"))
    checks = by_name(compare_tables(source, dict(source)))
    assert all(check.ok for check in checks.values())
    assert checks["row counts"].detail == "identical: 13 rows in 2 tables"


def test_a_different_row_count_or_content_fails():
    source = stats(auth_user=(3, "123"), project=(10, "456"), run=(2, "7"))
    target = stats(auth_user=(3, "123"), project=(9, "455"), run=(2, "8"))
    checks = by_name(compare_tables(source, target))
    assert checks["tables"].ok
    assert not checks["row counts"].ok
    assert "project (10 → 9)" in checks["row counts"].detail
    # A table whose rows already differ in number is not reported twice.
    assert not checks["checksums"].ok
    assert checks["checksums"].detail == "different content in run"


def test_missing_and_extra_tables_fail():
    source = stats(auth_user=(1, "1"), project=(0, "0"))
    target = stats(auth_user=(1, "1"), stray=(0, "0"))
    check = by_name(compare_tables(source, target))["tables"]
    assert not check.ok
    assert check.detail == "missing on the target: project; only on the target: stray"


def seq(next_value: int, highest: int | None, source_next: int | None = None) -> SequenceState:
    return SequenceState("project", "id", "project_id_seq", next_value, highest, source_next)


def test_sequences_must_be_ahead_of_the_highest_id_and_not_behind_the_source():
    assert check_sequences([seq(11, 10), seq(1, None), seq(16, 10, source_next=16)]).ok
    behind_ids = check_sequences([seq(10, 10)])
    assert not behind_ids.ok and "next 10, highest id 10" in behind_ids.detail
    # Ids 11 to 15 were used on the source by rows deleted since: not handed out again.
    behind_source = check_sequences([seq(11, 10, source_next=16)])
    assert not behind_source.ok and "on the source 16" in behind_source.detail


def test_a_source_that_changed_meanwhile_fails():
    before = stats(auth_user=(3, "1"), project=(5, "2"))
    assert check_unchanged(before, dict(before)).ok
    changed = check_unchanged(before, stats(auth_user=(3, "1"), project=(5, "3")))
    assert not changed.ok and "writes were not stopped" in changed.detail


def test_the_report_says_pass_or_fail():
    source = stats(auth_user=(3, "1"), project=(5, "2"))
    text, passed = database_move.render(source, dict(source), compare_tables(source, source))
    assert passed and text.endswith("RESULT: PASS")
    assert "auth_user" in text and "ok" in text
    target = stats(auth_user=(3, "1"))
    text, passed = database_move.render(source, target, compare_tables(source, target))
    assert not passed
    assert "FAIL  tables: missing on the target: project" in text
    assert text.endswith("RESULT: FAIL")


# --------------------------------------------------------------------------- #
# The command
# --------------------------------------------------------------------------- #


def test_the_command_needs_two_postgresql_databases(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    # Default source: the app's DATABASE_URL (none here: SQLite).
    with pytest.raises(CommandError, match="DATABASE_URL is not set"):
        run("--to", LOCAL)
    with pytest.raises(CommandError, match="Not a PostgreSQL database URL"):
        run("--from", SUPABASE, "--to", "mysql://db/brickdnn")
    with pytest.raises(CommandError, match="--keep only goes with --dry-run"):
        run("--from", SUPABASE, "--to", LOCAL, "--keep")
    with pytest.raises(CommandError, match="same database"):
        run("--from", LOCAL, "--to", "postgresql://x:y@db/brickdnn")


# --------------------------------------------------------------------------- #
# A whole move, on PostgreSQL
# --------------------------------------------------------------------------- #


def url_of(settings_dict: dict, name: str) -> str:
    user = quote(settings_dict["USER"] or "", safe="")
    password = quote(settings_dict["PASSWORD"] or "", safe="")
    login = f"{user}:{password}@" if password else (f"{user}@" if user else "")
    host = settings_dict["HOST"] or "localhost"
    port = f":{settings_dict['PORT']}" if settings_dict["PORT"] else ""
    return f"postgresql://{login}{host}{port}/{quote(name, safe='')}"


@pytest.fixture
def move_urls(django_db_blocker):
    """The test database as source and a new, empty database as target."""
    if connection.vendor != "postgresql":
        pytest.skip("PostgreSQL only")
    with django_db_blocker.unblock():
        server = backups._server_major()
        for tool in ("pg_dump", "pg_restore"):
            major = backups.tool_major(tool)
            if major is None or major < server:
                pytest.skip(f"needs {tool} {server} or newer")
        name = connection.settings_dict["NAME"] + "_move"
        with connection.cursor() as cursor:
            cursor.execute(f'DROP DATABASE IF EXISTS "{name}"')
            cursor.execute(f'CREATE DATABASE "{name}"')
    yield (
        url_of(connection.settings_dict, connection.settings_dict["NAME"]),
        url_of(connection.settings_dict, name),
    )
    with django_db_blocker.unblock(), connection.cursor() as cursor:
        cursor.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


def target_tables(url: str) -> list[str]:
    with database_move.open_database("brickdnn_test_target", url) as target:
        return database_move.schema_objects(target)


def seed() -> None:
    ada = User.objects.create_user("ada", password="correct-horse", email="ada@example.org")
    for name in ("mlp", "cnn", "unet"):
        Project.objects.create(owner=ada, name=name, graph={"version": 1, "nodes": []})
    # The newest row goes: the copy must not hand out its id again.
    Project.objects.get(name="unet").delete()


@pytest.mark.django_db(transaction=True)
def test_a_rehearsal_and_a_move(move_urls):
    source, target = move_urls
    seed()
    output = run("--from", source, "--to", target, "--dry-run")
    assert "RESULT: PASS" in output
    assert "PASS  migrations" in output
    assert "source unchanged" not in output  # only checked when moving for real
    assert "Dry run: the copy was dropped again" in output
    assert target_tables(target) == []

    output = run("--from", source, "--to", target)
    assert "PASS  source unchanged" in output and "RESULT: PASS" in output
    newest = Project.objects.order_by("-id").first().id + 1  # the id of the deleted one
    with (
        database_move.open_database("brickdnn_test_target", target) as copy,
        copy.cursor() as cursor,
    ):
        cursor.execute("SELECT username, password FROM auth_user")
        [(username, password)] = cursor.fetchall()
        cursor.execute("SELECT name FROM brick_dnn_core_project ORDER BY id")
        assert [row[0] for row in cursor.fetchall()] == ["mlp", "cnn"]
        cursor.execute("SELECT nextval(pg_get_serial_sequence('brick_dnn_core_project', 'id'))")
        assert cursor.fetchone()[0] == newest + 1
    assert username == "ada" and password == User.objects.get().password

    # Never over a database that is in use, unless asked.
    with pytest.raises(CommandError, match="The target is not empty"):
        run("--from", source, "--to", target)
    assert "RESULT: PASS" in run("--from", source, "--to", target, "--replace", "--no-input")


@pytest.mark.django_db(transaction=True)
def test_writes_during_a_move_fail_it_but_not_a_rehearsal(move_urls, monkeypatch):
    source, target = move_urls
    seed()
    original = database_move.restore

    def restore_while_someone_writes(db, path):
        original(db, path)
        # On the source, after its snapshot.
        User.objects.create_user(f"eve{User.objects.count()}")

    monkeypatch.setattr(database_move, "restore", restore_while_someone_writes)
    # A rehearsal compares the snapshot that was dumped: later writes do not matter.
    assert "RESULT: PASS" in run("--from", source, "--to", target, "--dry-run")
    with pytest.raises(CommandError, match="did not pass"):
        run("--from", source, "--to", target)


@pytest.mark.django_db(transaction=True)
def test_a_copy_that_differs_fails(move_urls, monkeypatch):
    source, target = move_urls
    seed()
    original = database_move.restore

    def lossy_restore(db, path):
        original(db, path)
        with (
            database_move.open_database("brickdnn_test_lossy", target) as copy,
            copy.cursor() as cursor,
        ):
            cursor.execute("UPDATE brick_dnn_core_project SET name = 'x' WHERE name = 'cnn'")

    monkeypatch.setattr(database_move, "restore", lossy_restore)
    out = io.StringIO()
    with pytest.raises(CommandError, match="did not pass"):
        call_command("move_database", "--from", source, "--to", target, stdout=out)
    report = out.getvalue()
    assert "FAIL  checksums: different content in brick_dnn_core_project" in report
    assert "RESULT: FAIL" in report and "NOT usable" in report
