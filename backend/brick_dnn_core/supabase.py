"""Keep Supabase's automatic REST API away from Brick DNN's tables.

Supabase publishes every table in the ``public`` schema through its Data API, and by default
grants the API roles (``anon``, ``authenticated``) all privileges on new tables. Without row
level security (RLS), anyone with the project's anon key could read or change the tables Django
creates, including the user table with its password hashes.

Brick DNN talks to its database only through Django, which connects as the owner of these
tables; owners are not restricted by RLS. So after every ``migrate`` all Django tables get RLS
without any policy, and the API roles lose their privileges on them. Databases without these
roles (plain PostgreSQL) are left alone.
"""

from __future__ import annotations

import sys
from typing import Any

from django.db import DatabaseError, connections, transaction
from django.db.migrations.recorder import MigrationRecorder

API_ROLES = ("anon", "authenticated")


def api_roles(using: str = "default") -> list[str]:
    """The Supabase API roles in this database (empty unless it is a Supabase database)."""
    connection = connections[using]
    if connection.vendor != "postgresql":
        return []
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT rolname FROM pg_roles WHERE rolname = ANY(%s) ORDER BY rolname",
            [list(API_ROLES)],
        )
        return [row[0] for row in cursor.fetchall()]


def django_tables(using: str = "default") -> list[str]:
    """All existing tables of Django models, plus Django's migration table."""
    connection = connections[using]
    tables = set(connection.introspection.django_table_names(only_existing=True))
    existing = set(connection.introspection.table_names(include_views=False))
    tables.add(MigrationRecorder.Migration._meta.db_table)
    return sorted(tables & existing)


def protect_tables(using: str = "default") -> tuple[list[str], list[str]]:
    """Switch on RLS for all Django tables and revoke the API roles' privileges on them.

    Returns the protected tables and the tables that could not be changed (e.g. because they
    belong to another database role)."""
    roles = api_roles(using)
    if not roles:
        return [], []
    connection = connections[using]
    quote = connection.ops.quote_name
    grantees = ", ".join(quote(role) for role in roles)
    protected: list[str] = []
    failed: list[str] = []
    for table in django_tables(using):
        try:
            with transaction.atomic(using=using), connection.cursor() as cursor:
                cursor.execute(f"ALTER TABLE {quote(table)} ENABLE ROW LEVEL SECURITY")
                cursor.execute(f"REVOKE ALL ON TABLE {quote(table)} FROM {grantees}")
        except DatabaseError:
            failed.append(table)
        else:
            protected.append(table)
    return protected, failed


def protect_after_migrate(
    sender: Any, *, using: str = "default", verbosity: int = 1, **kwargs: Any
) -> None:
    """``post_migrate`` handler: runs once after all migrations of a ``migrate`` command."""
    protected, failed = protect_tables(using)
    stdout = kwargs.get("stdout") or sys.stdout
    if protected and verbosity >= 1:
        stdout.write(
            f"Supabase: {len(protected)} tables are hidden from the Data API "
            "(row level security on, no access for anon/authenticated).\n"
        )
    if failed:
        # Not fatal (the app works), but the tables may be readable through the Data API.
        sys.stderr.write(
            "WARNING: Supabase: could not switch on row level security for "
            f"{', '.join(failed)}. These tables may be readable through Supabase's Data API "
            "with the anon key. Run migrate as the owner of the tables, or switch on RLS for "
            "them in the Supabase dashboard (see docs/database.md).\n"
        )
