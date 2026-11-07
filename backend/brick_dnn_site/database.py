"""The database connection: a SQLite file by default, PostgreSQL (e.g. Supabase) via a URL.

``DATABASE_URL`` has the usual form ``postgresql://USER:PASSWORD@HOST:PORT/NAME?OPTIONS``.
Query parameters are passed on to the PostgreSQL client (libpq), e.g. ``sslmode=require`` or
``connect_timeout=10``. See docs/database.md.
"""

from __future__ import annotations

from os import PathLike
from typing import Any
from urllib.parse import parse_qsl, unquote, urlsplit

from django.core.exceptions import ImproperlyConfigured

POSTGRES_SCHEMES = ("postgres", "postgresql")
SUPABASE_DOMAINS = (".supabase.com", ".supabase.co")
# Supabase's transaction pooler. It may run every transaction on another server connection.
SUPABASE_TRANSACTION_PORT = 6543

URL_HELP = (
    "Expected postgresql://USER:PASSWORD@HOST:PORT/NAME. Characters like # / ? @ : in the "
    "password must be percent-encoded (# → %23, / → %2F, ? → %3F, @ → %40, : → %3A)."
)


def is_supabase(host: str) -> bool:
    return host.endswith(SUPABASE_DOMAINS)


def database_settings(
    url: str, *, sqlite_path: str | PathLike[str], conn_max_age: int = 60
) -> dict[str, Any]:
    """The ``DATABASES["default"]`` entry: PostgreSQL if ``url`` is set, else SQLite."""
    url = url.strip()
    if not url:
        return {"ENGINE": "django.db.backends.sqlite3", "NAME": sqlite_path}
    return postgres_settings(url, conn_max_age=conn_max_age)


def postgres_settings(url: str, *, conn_max_age: int = 60) -> dict[str, Any]:
    """Django settings for a ``postgresql://`` URL. Raises :class:`ImproperlyConfigured`."""
    parts = urlsplit(url)
    if parts.scheme.lower() not in POSTGRES_SCHEMES:
        raise ImproperlyConfigured(
            "DATABASE_URL must start with postgresql:// (leave it empty to use SQLite). " + URL_HELP
        )
    try:
        port = parts.port
    except ValueError as exc:
        raise ImproperlyConfigured(f"DATABASE_URL has an invalid port. {URL_HELP}") from exc
    name = unquote(parts.path.lstrip("/"))
    if not name or "/" in name:
        raise ImproperlyConfigured(
            f"DATABASE_URL must end with the database name (on Supabase: /postgres). {URL_HELP}"
        )

    options: dict[str, Any] = dict(parse_qsl(parts.query, keep_blank_values=True))
    # No host (postgresql:///name): a local server through its Unix socket, like libpq does.
    host = parts.hostname or options.pop("host", "")
    # Prisma's name for a transaction pooler (PgBouncer, Supavisor); not a libpq option.
    pooled = options.pop("pgbouncer", "").strip().lower() in ("1", "true", "yes", "on")
    if is_supabase(host):
        # Never fall back to an unencrypted connection (libpq's default would) over the internet.
        options.setdefault("sslmode", "require")
        pooled = pooled or port == SUPABASE_TRANSACTION_PORT

    return {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": name,
        "USER": unquote(parts.username or ""),
        "PASSWORD": unquote(parts.password or ""),
        "HOST": host,
        "PORT": str(port or ""),
        "OPTIONS": options,
        # Keep connections open between requests (seconds), but check them before reuse.
        "CONN_MAX_AGE": conn_max_age,
        "CONN_HEALTH_CHECKS": True,
        # A transaction pooler does not keep server-side cursors between transactions.
        "DISABLE_SERVER_SIDE_CURSORS": pooled,
    }
