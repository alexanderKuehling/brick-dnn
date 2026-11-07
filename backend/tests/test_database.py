"""Database settings (SQLite or PostgreSQL via DATABASE_URL) and the Supabase table protection.

The PostgreSQL tests run when the suite itself runs on PostgreSQL:
    DATABASE_URL=postgresql://postgres@127.0.0.1:5432/brickdnn pytest
"""

from io import StringIO

import pytest
from django.conf import settings
from django.contrib.auth.models import User
from django.core.exceptions import ImproperlyConfigured
from django.core.management import call_command
from django.db import DatabaseError, connection, transaction

from brick_dnn_core.supabase import API_ROLES, api_roles, protect_after_migrate, protect_tables
from brick_dnn_site.database import database_settings, postgres_settings

ON_POSTGRES = settings.DATABASES["default"]["ENGINE"] == "django.db.backends.postgresql"
postgres_only = pytest.mark.skipif(not ON_POSTGRES, reason="needs DATABASE_URL (PostgreSQL)")

SESSION_POOLER = (
    "postgresql://postgres.abcdefghijklmnop:s3cr%23t%2Fpw@aws-0-eu-central-1.pooler.supabase.com"
    ":5432/postgres"
)


# --- DATABASE_URL --------------------------------------------------------------------------


def test_no_url_means_sqlite(tmp_path):
    db = database_settings("  ", sqlite_path=tmp_path / "db.sqlite3")
    assert db == {"ENGINE": "django.db.backends.sqlite3", "NAME": tmp_path / "db.sqlite3"}


def test_postgres_url():
    db = database_settings(
        "postgres://brick:pa%40ss%3Aword@db.example.org:5433/brick_dnn?connect_timeout=10",
        sqlite_path="unused",
        conn_max_age=0,
    )
    assert db["ENGINE"] == "django.db.backends.postgresql"
    assert (db["USER"], db["PASSWORD"]) == ("brick", "pa@ss:word")
    assert (db["HOST"], db["PORT"], db["NAME"]) == ("db.example.org", "5433", "brick_dnn")
    # Options go to libpq; other hosts keep libpq's own SSL default.
    assert db["OPTIONS"] == {"connect_timeout": "10"}
    assert db["CONN_MAX_AGE"] == 0
    assert db["CONN_HEALTH_CHECKS"] is True
    assert db["DISABLE_SERVER_SIDE_CURSORS"] is False


def test_postgres_url_without_password_and_port():
    db = postgres_settings("postgresql://postgres@[::1]/brickdnn")
    assert (db["USER"], db["PASSWORD"], db["HOST"], db["PORT"]) == ("postgres", "", "::1", "")


def test_postgres_url_for_a_local_unix_socket():
    assert postgres_settings("postgresql:///brickdnn")["HOST"] == ""
    db = postgres_settings("postgresql://brick@/brickdnn?host=/var/run/postgresql")
    assert (db["USER"], db["HOST"], db["OPTIONS"]) == ("brick", "/var/run/postgresql", {})


def test_supabase_session_pooler():
    db = postgres_settings(SESSION_POOLER)
    assert db["USER"] == "postgres.abcdefghijklmnop"
    assert db["PASSWORD"] == "s3cr#t/pw"
    assert db["HOST"] == "aws-0-eu-central-1.pooler.supabase.com"
    assert db["NAME"] == "postgres"
    # Encrypted connections only; the session pooler keeps server-side cursors.
    assert db["OPTIONS"] == {"sslmode": "require"}
    assert db["DISABLE_SERVER_SIDE_CURSORS"] is False


def test_supabase_transaction_pooler_and_explicit_ssl_mode():
    url = SESSION_POOLER.replace(":5432/", ":6543/") + "?sslmode=verify-full"
    db = postgres_settings(url)
    assert db["OPTIONS"] == {"sslmode": "verify-full"}
    assert db["DISABLE_SERVER_SIDE_CURSORS"] is True


def test_direct_supabase_connection():
    db = postgres_settings("postgresql://postgres:pw@db.abcdefghijklmnop.supabase.co:5432/postgres")
    assert db["OPTIONS"] == {"sslmode": "require"}
    assert db["DISABLE_SERVER_SIDE_CURSORS"] is False


def test_pgbouncer_flag_marks_any_transaction_pooler():
    db = postgres_settings("postgresql://u:p@pgbouncer.local:6432/app?pgbouncer=true")
    assert db["DISABLE_SERVER_SIDE_CURSORS"] is True
    assert "pgbouncer" not in db["OPTIONS"]


@pytest.mark.parametrize(
    ("url", "message"),
    [
        ("mysql://u:p@host/db", "must start with postgresql://"),
        ("postgresql://u:p@host:5432", "must end with the database name"),
        # Passwords with unencoded / # ? break the URL apart.
        ("postgresql://u:pa/ss@host:5432/db", "invalid port"),
        ("postgresql://u:12/34@host:5432/db", "must end with the database name"),
        ("postgresql://u:pa#ss@host:5432/db", "invalid port"),
    ],
)
def test_invalid_urls(url, message):
    with pytest.raises(ImproperlyConfigured, match=message) as error:
        postgres_settings(url)
    assert "percent-encoded" in str(error.value)


# --- Supabase: hide the tables from the Data API --------------------------------------------


@pytest.mark.django_db
def test_other_databases_are_left_alone():
    if ON_POSTGRES and api_roles():
        pytest.skip("this PostgreSQL server has Supabase's API roles")
    assert protect_tables() == ([], [])
    out = StringIO()
    protect_after_migrate(None, using="default", verbosity=1, stdout=out)
    assert out.getvalue() == ""


def _create_api_roles() -> None:
    """Supabase's API roles, with its default privileges on two tables (rolled back after
    the test, like everything else in it)."""
    with connection.cursor() as cursor:
        for role in API_ROLES:
            cursor.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", [role])
            if cursor.fetchone() is None:
                cursor.execute(f'CREATE ROLE "{role}" NOLOGIN')
        # If the roles existed already, migrate has protected the test database: undo that.
        for table in ("auth_user", "brick_dnn_core_project"):
            cursor.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")
        cursor.execute(
            "GRANT ALL ON TABLE auth_user, brick_dnn_core_project TO anon, authenticated"
        )


def _select_as(role: str, table: str) -> int:
    with transaction.atomic(), connection.cursor() as cursor:
        cursor.execute(f'SET LOCAL ROLE "{role}"')
        cursor.execute(f"SELECT count(*) FROM {table}")
        count = cursor.fetchone()[0]
        cursor.execute("RESET ROLE")
        return count


@postgres_only
@pytest.mark.django_db
def test_supabase_tables_are_hidden_from_api_roles():
    User.objects.create_user("owner", password="a long password 123")
    _create_api_roles()
    assert _select_as("anon", "auth_user") == 1  # the leak this protects against

    protected, failed = protect_tables()

    assert failed == []
    assert {
        "auth_user",
        "brick_dnn_core_project",
        "brick_dnn_core_trainingrun",
        "brick_dnn_core_datasource",
        "django_migrations",
        "token_blacklist_outstandingtoken",
    } <= set(protected)
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT relname FROM pg_class WHERE relname = ANY(%s) AND NOT relrowsecurity",
            [protected],
        )
        assert cursor.fetchall() == []
        cursor.execute(
            "SELECT has_table_privilege('anon', 'auth_user', 'SELECT'),"
            " has_table_privilege('authenticated', 'brick_dnn_core_project', 'INSERT')"
        )
        assert cursor.fetchone() == (False, False)
    for role in API_ROLES:
        with pytest.raises(DatabaseError, match="permission denied"):
            _select_as(role, "auth_user")
    # Even if privileges are granted again, RLS without a policy shows no rows.
    with connection.cursor() as cursor:
        cursor.execute("GRANT SELECT ON TABLE auth_user TO anon")
    assert _select_as("anon", "auth_user") == 0
    # Django owns the tables: RLS does not affect it.
    User.objects.create_user("second", password="a long password 123")
    assert User.objects.count() == 2


@postgres_only
@pytest.mark.django_db
def test_migrate_protects_the_tables():
    _create_api_roles()
    out = StringIO()
    call_command("migrate", verbosity=1, stdout=out)
    assert "Supabase:" in out.getvalue()
    assert "hidden from the Data API" in out.getvalue()
    with connection.cursor() as cursor:
        cursor.execute("SELECT has_table_privilege('anon', 'auth_user', 'SELECT')")
        assert cursor.fetchone() == (False,)
