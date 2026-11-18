"""Copy the app's PostgreSQL database into another one and verify the copy
(``manage.py move_database``).

Written for moving brickdnn.com from Supabase to the PostgreSQL container on its own server
(deploy/server/README.md, "Moving the database onto the server"), but it works between any two
PostgreSQL databases:

1. Checks: both are PostgreSQL and not the same database, ``pg_dump`` is not older than the
   source and the target not older than the source, and the target schema is empty
   (``--replace`` drops what is in it, after asking).
2. Opens a read-only ``REPEATABLE READ`` transaction on the source and exports its snapshot:
   ``pg_dump`` dumps exactly that moment, and the counts and checksums of the source are read in
   the same transaction, so a rehearsal while the site keeps running compares like with like.
3. Dumps only Django's tables (``supabase.django_tables()``) of the source's schema, without
   owners, privileges, tablespaces or security labels: nothing of Supabase's own roles,
   schemas or grants.
4. Restores them into the target in one transaction, moves every sequence to at least the
   highest id (never below its value on the source), switches row level security off (on
   Supabase as the target: ``supabase.protect_tables()``) and runs ``ANALYZE``.
5. Verifies: the same tables, the same number of rows and the same checksum of all rows per
   table, sequences ahead of their highest id, ``migrate --check``; for a real move also that
   the source did not change meanwhile (i.e. writes were stopped).
6. ``--dry-run`` drops the copy again afterwards (``--keep`` leaves it for a look).

The source is only read: every query on it runs in a read-only transaction, and pg_dump only
reads.
"""

from __future__ import annotations

import contextlib
import io
import os
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from django.core.exceptions import ImproperlyConfigured
from django.core.management import call_command
from django.db import DatabaseError, connections, transaction
from django.db.utils import load_backend

from brick_dnn_site.database import postgres_settings

from . import backups, supabase

SOURCE = "brickdnn_move_source"
TARGET = "brickdnn_move_target"
DEFAULT_PORT = "5432"

# Row and value output that is the same on both servers, whatever their configuration: the
# checksums compare the text of every row.
STABLE_OUTPUT = (
    "SET LOCAL TIME ZONE 'UTC'",
    "SET LOCAL DateStyle TO 'ISO, YMD'",
    "SET LOCAL IntervalStyle TO 'postgres'",
    "SET LOCAL extra_float_digits TO 1",
    "SET LOCAL bytea_output TO 'hex'",
)

# Sequences that belong to a column of a table in the current schema: identity columns
# (deptype i, Django's AutoField since 4.1) and serial columns (deptype a, older tables).
SEQUENCES_SQL = """
SELECT c.relname, a.attname, s.relname
FROM pg_depend d
JOIN pg_class s ON s.oid = d.objid AND s.relkind = 'S'
JOIN pg_class c ON c.oid = d.refobjid
JOIN pg_namespace n ON n.oid = c.relnamespace
JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum = d.refobjsubid
WHERE d.classid = 'pg_class'::regclass AND d.refclassid = 'pg_class'::regclass
  AND d.deptype IN ('a', 'i') AND n.nspname = current_schema() AND c.relname = ANY(%s)
ORDER BY 1, 2
"""


class MoveError(Exception):
    """Something is in the way of the move; the message says what. Nothing was changed on the
    source."""


Log = Callable[[str], None]


def _quiet(message: str) -> None:
    pass


# --------------------------------------------------------------------------- #
# Which databases
# --------------------------------------------------------------------------- #


def resolve_url(spec: str, environ: Mapping[str, str] = os.environ) -> str:
    """A ``postgresql://`` URL, given as such or as ``env:NAME`` (the URL is in that environment
    variable: keeps the password off command lines and out of the shell history)."""
    spec = (spec or "").strip()
    if spec.startswith("env:"):
        name = spec[len("env:") :]
        value = (environ.get(name) or "").strip()
        if not value:
            raise MoveError(f"The environment variable {name} is not set or empty.")
        spec = value
    if not spec:
        raise MoveError("No database URL given.")
    try:
        postgres_settings(spec)
    except ImproperlyConfigured as exc:
        raise MoveError(f"Not a PostgreSQL database URL: {exc}") from exc
    return spec


def _identity(url: str) -> tuple[str, str, str]:
    db = postgres_settings(url)
    return ((db["HOST"] or "localhost").lower(), str(db["PORT"] or DEFAULT_PORT), db["NAME"])


def same_database(a: str, b: str) -> bool:
    """Whether two URLs name the same database (host, port and name; users do not matter)."""
    return _identity(a) == _identity(b)


def describe_url(url: str) -> str:
    """``user@host:port/name``, without the password."""
    db = postgres_settings(url)
    host, port, name = _identity(url)
    return f"{db['USER'] or '?'}@{host}:{port}/{name}"


@contextlib.contextmanager
def open_database(alias: str, url: str) -> Iterator[Any]:
    """A Django connection to ``url`` under ``alias``, for as long as the block runs."""
    # Not in settings.DATABASES: a connection of its own, which `using=alias` and
    # `--database alias` find while it exists.
    config = postgres_settings(url, conn_max_age=0)
    config = connections.configure_settings({"default": {}, alias: config})[alias]
    connections[alias] = load_backend(config["ENGINE"]).DatabaseWrapper(config, alias)
    try:
        yield connections[alias]
    finally:
        with contextlib.suppress(Exception):
            connections[alias].close()
        with contextlib.suppress(Exception):
            del connections[alias]


# --------------------------------------------------------------------------- #
# What is compared
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TableStats:
    rows: int
    # Sum of the first 60 bits of the MD5 of every row's text: any changed, missing or extra
    # row changes it (order does not matter).
    checksum: str


@dataclass(frozen=True)
class SequenceState:
    table: str
    column: str
    sequence: str
    # What nextval() would return next, and the highest value in the column (None: no rows).
    next_value: int
    highest: int | None
    # next_value on the source (None: not known).
    source_next: int | None = None


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def leftover_tables(moved: list[str], existing: list[str]) -> list[str]:
    """Tables of the source's schema that are not moved (not Django's, e.g. Supabase's own)."""
    return sorted(set(existing) - set(moved))


def _names(items: list[str], limit: int = 8) -> str:
    shown = ", ".join(items[:limit])
    return shown + (f" and {len(items) - limit} more" if len(items) > limit else "")


def compare_tables(source: dict[str, TableStats], target: dict[str, TableStats]) -> list[Check]:
    """The same tables, rows and checksums on both sides."""
    missing = sorted(set(source) - set(target))
    extra = sorted(set(target) - set(source))
    if missing or extra:
        detail = "; ".join(
            part
            for part in (
                f"missing on the target: {_names(missing)}" if missing else "",
                f"only on the target: {_names(extra)}" if extra else "",
            )
            if part
        )
    else:
        detail = f"{len(source)} tables on both sides"
    checks = [Check("tables", not missing and not extra, detail)]
    common = sorted(set(source) & set(target))
    counts = [t for t in common if source[t].rows != target[t].rows]
    total = sum(source[t].rows for t in common)
    checks.append(
        Check(
            "row counts",
            not counts,
            f"different in {_names([f'{t} ({source[t].rows} → {target[t].rows})' for t in counts])}"
            if counts
            else f"identical: {total} rows in {len(common)} tables",
        )
    )
    sums = [t for t in common if t not in counts and source[t].checksum != target[t].checksum]
    checks.append(
        Check(
            "checksums",
            not sums,
            f"different content in {_names(sums)}"
            if sums
            else f"identical for all {len(common) - len(counts)} tables with the same row count",
        )
    )
    return checks


def check_sequences(states: list[SequenceState]) -> Check:
    """Every sequence hands out ids above the highest one in its column, and none is behind its
    value on the source (which would hand out ids again that rows deleted there had)."""
    problems = []
    for state in states:
        name = f"{state.sequence} ({state.table}.{state.column})"
        if state.highest is not None and state.next_value <= state.highest:
            problems.append(f"{name}: next {state.next_value}, highest id {state.highest}")
        elif state.source_next is not None and state.next_value < state.source_next:
            problems.append(f"{name}: next {state.next_value}, on the source {state.source_next}")
    if problems:
        return Check("sequences", False, "; ".join(problems))
    return Check(
        "sequences",
        True,
        f"{len(states)} sequences ahead of their highest id and not behind the source",
    )


def check_unchanged(before: dict[str, TableStats], after: dict[str, TableStats]) -> Check:
    """The source still holds what was copied: nobody wrote to it during the move."""
    changed = [table for table in sorted(before) if after.get(table) != before[table]]
    if changed:
        return Check(
            "source unchanged",
            False,
            f"the source changed during the move in {_names(changed)}: writes were not stopped "
            "(stop the app and the workers, then run the move again with --replace)",
        )
    return Check("source unchanged", True, "no writes to the source during the move")


def render(source: dict[str, TableStats], target: dict[str, TableStats], checks: list[Check]):
    """The report: a line per table, the checks and PASS or FAIL. Returns (text, passed)."""
    width = max([len("Table"), *(len(t) for t in source)])
    lines = [f"{'Table':<{width}}  {'Source rows':>11}  {'Target rows':>11}  Checksum"]
    for table in sorted(source):
        theirs = target.get(table)
        if theirs is None:
            lines.append(f"{table:<{width}}  {source[table].rows:>11}  {'-':>11}  missing")
            continue
        same = theirs == source[table]
        lines.append(
            f"{table:<{width}}  {source[table].rows:>11}  {theirs.rows:>11}  "
            + ("ok" if same else "DIFFERENT")
        )
    passed = all(check.ok for check in checks)
    lines.append("")
    lines += [f"{'PASS' if c.ok else 'FAIL'}  {c.name}: {c.detail}" for c in checks]
    lines.append("")
    lines.append(f"RESULT: {'PASS' if passed else 'FAIL'}")
    return "\n".join(lines), passed


# --------------------------------------------------------------------------- #
# On the databases
# --------------------------------------------------------------------------- #


def server_major(conn: Any) -> int:
    conn.ensure_connection()
    return conn.pg_version // 10000


def server_version(conn: Any) -> str:
    version = conn.pg_version
    return f"{version // 10000}.{version % 100}"


def current_schema(conn: Any) -> str:
    with conn.cursor() as cursor:
        cursor.execute("SELECT current_schema()")
        return cursor.fetchone()[0]


def schema_objects(conn: Any) -> list[str]:
    """Tables, views and sequences in the current schema."""
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT c.relname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname = current_schema() AND c.relkind IN ('r', 'p', 'v', 'm', 'S', 'f') "
            "ORDER BY 1"
        )
        return [row[0] for row in cursor.fetchall()]


def drop_all(conn: Any) -> int:
    """Drops every table, view and sequence of the current schema (the target's)."""
    quote = conn.ops.quote_name
    with transaction.atomic(using=conn.alias), conn.cursor() as cursor:
        cursor.execute(
            "SELECT c.relname, c.relkind FROM pg_class c "
            "JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname = current_schema() AND c.relkind IN ('r', 'p', 'v', 'm', 'S', 'f')"
        )
        objects = cursor.fetchall()
        kinds = {
            "v": "VIEW",
            "m": "MATERIALIZED VIEW",
            "r": "TABLE",
            "p": "TABLE",
            "f": "FOREIGN TABLE",
            "S": "SEQUENCE",
        }
        # Views first, sequences last (tables take the sequences they own along).
        for kind in ("v", "m", "f", "r", "p", "S"):
            for name, relkind in objects:
                if relkind == kind:
                    cursor.execute(f"DROP {kinds[kind]} IF EXISTS {quote(name)} CASCADE")
    return len(objects)


def table_stats(conn: Any, tables: list[str]) -> dict[str, TableStats]:
    """Rows and checksum of every table. Runs in the caller's transaction, if there is one."""
    quote = conn.ops.quote_name
    stats = {}
    with transaction.atomic(using=conn.alias), conn.cursor() as cursor:
        for statement in STABLE_OUTPUT:
            cursor.execute(statement)
        for table in tables:
            cursor.execute(
                "SELECT count(*), coalesce(sum(('x' || substr(md5(ROW(r.*)::text), 1, 15))"
                f"::bit(60)::bigint), 0)::text FROM {quote(table)} AS r"
            )
            rows, checksum = cursor.fetchone()
            stats[table] = TableStats(int(rows), str(checksum))
    return stats


def sequence_states(
    conn: Any, tables: list[str], source: dict[str, int] | None = None
) -> list[SequenceState]:
    """The sequences of these tables; ``source``: their next values on the source, by name."""
    quote = conn.ops.quote_name
    states = []
    with conn.cursor() as cursor:
        cursor.execute(SEQUENCES_SQL, [list(tables)])
        for table, column, sequence in cursor.fetchall():
            cursor.execute(f"SELECT last_value, is_called FROM {quote(sequence)}")
            last, called = cursor.fetchone()
            cursor.execute(f"SELECT max({quote(column)}) FROM {quote(table)}")
            highest = cursor.fetchone()[0]
            states.append(
                SequenceState(
                    table,
                    column,
                    sequence,
                    next_value=int(last) + 1 if called else int(last),
                    highest=None if highest is None else int(highest),
                    source_next=(source or {}).get(sequence),
                )
            )
    return states


def advance_sequences(conn: Any, states: list[SequenceState]) -> None:
    """Moves each sequence past the highest id, never back: pg_restore already set the values
    of the source."""
    quote = conn.ops.quote_name
    with transaction.atomic(using=conn.alias), conn.cursor() as cursor:
        for state in states:
            last = max(state.highest or 0, state.next_value - 1)
            if last >= 1:
                cursor.execute(
                    "SELECT setval(%s::regclass, %s, true)", [quote(state.sequence), last]
                )
            else:
                cursor.execute("SELECT setval(%s::regclass, 1, false)", [quote(state.sequence)])


def secure_tables(conn: Any, tables: list[str]) -> str:
    """Row level security as fits the target. The dump carries Supabase's (on, no policies):
    on Supabase it stays and the API roles lose their privileges, as after every migrate;
    elsewhere there is no Data API and it goes, like on any database Django created."""
    if supabase.api_roles(conn.alias):
        protected, failed = supabase.protect_tables(conn.alias)
        if failed:
            raise MoveError(f"Could not protect {', '.join(failed)} from Supabase's Data API.")
        return f"Supabase: {len(protected)} tables hidden from the Data API"
    quote = conn.ops.quote_name
    with transaction.atomic(using=conn.alias), conn.cursor() as cursor:
        cursor.execute(
            "SELECT c.relname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname = current_schema() AND c.relrowsecurity AND c.relname = ANY(%s)",
            [list(tables)],
        )
        secured = [row[0] for row in cursor.fetchall()]
        for table in secured:
            cursor.execute(f"ALTER TABLE {quote(table)} DISABLE ROW LEVEL SECURITY")
    return f"row level security switched off: {len(secured)} of {len(tables)} tables (no Data API here)"


def check_migrations(alias: str) -> Check:
    """``migrate --check`` on the copy: the code finds every migration applied."""
    out = io.StringIO()
    try:
        call_command("migrate", "--check", database=alias, verbosity=0, stdout=out, stderr=out)
    except SystemExit as exc:
        if exc.code:
            return Check("migrations", False, "migrate --check: there are unapplied migrations")
    return Check("migrations", True, "migrate --check: nothing to apply")


def _run(command: list[str], env: dict[str, str], what: str) -> None:
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    if result.returncode:
        raise MoveError(f"{what} failed: {result.stderr.strip()[-1500:]}")


def dump(db: dict[str, Any], schema: str, tables: list[str], path: Path, snapshot: str | None):
    conninfo, env = backups.pg_conninfo(db)
    command = [
        "pg_dump",
        "--format=custom",
        "--no-owner",
        "--no-privileges",
        "--no-tablespaces",
        "--no-security-labels",
        # Every table named must exist.
        "--strict-names",
        f"--file={path}",
        f"--dbname={conninfo}",
    ]
    if snapshot:
        command.append(f"--snapshot={snapshot}")
    # Quoted names: taken literally, not as patterns.
    quoted_schema = '"' + schema.replace('"', '""') + '"'
    command += [f'--table={quoted_schema}."{t.replace(chr(34), chr(34) * 2)}"' for t in tables]
    _run(command, env, "pg_dump")


def restore(db: dict[str, Any], path: Path) -> None:
    conninfo, env = backups.pg_conninfo(db)
    command = [
        "pg_restore",
        "--no-owner",
        "--no-privileges",
        "--no-tablespaces",
        "--no-security-labels",
        "--single-transaction",
        "--exit-on-error",
        f"--dbname={conninfo}",
        str(path),
    ]
    _run(command, env, "pg_restore")


# --------------------------------------------------------------------------- #
# The move
# --------------------------------------------------------------------------- #


@dataclass
class Outcome:
    passed: bool
    report: str


class _NoSnapshot(Exception):
    """The source cannot export its snapshot (e.g. behind some poolers): start over without."""


def _dump_source(
    source: Any, schema: str, path: Path, log: Log, snapshot: bool = True
) -> tuple[list[str], dict[str, TableStats], dict[str, int]]:
    """Dumps the Django tables of the source and reads their rows, checksums and sequences,
    all in one read-only transaction that pg_dump shares (its exported snapshot)."""
    started = time.monotonic()
    with transaction.atomic(using=SOURCE):
        _read_only(source)
        name = None
        if snapshot:
            try:
                with source.cursor() as cursor:
                    cursor.execute("SELECT pg_export_snapshot()")
                    name = cursor.fetchone()[0]
            except DatabaseError as exc:
                raise _NoSnapshot() from exc
        tables = supabase.django_tables(SOURCE)
        if not tables:
            raise MoveError("The source has no tables of Brick DNN.")
        left_out = leftover_tables(tables, source.introspection.table_names(include_views=False))
        log(
            f"{len(tables)} tables of Django in the schema {schema!r}; left out: "
            + (_names(left_out) if left_out else "none")
        )
        dump(source.settings_dict, schema, tables, path, name)
        log(f"Dumped {path.stat().st_size / 1024**2:.1f} MB in {time.monotonic() - started:.1f} s.")
        stats = table_stats(source, tables)
        sequences = {s.sequence: s.next_value for s in sequence_states(source, tables)}
    return tables, stats, sequences


def _read_only(conn: Any) -> None:
    """Makes the caller's new transaction a read-only snapshot (the first statement in it)."""
    with conn.cursor() as cursor:
        cursor.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")


def _verify(
    target: Any,
    source: Any,
    tables: list[str],
    source_stats: dict[str, TableStats],
    source_sequences: dict[str, int],
    dry_run: bool,
) -> tuple[str, bool]:
    """Finishes the copy (row level security, sequences, statistics) and compares it."""
    quote = target.ops.quote_name
    log_line = secure_tables(target, tables)
    advance_sequences(target, sequence_states(target, tables))
    with target.cursor() as cursor:
        cursor.execute("ANALYZE " + ", ".join(quote(table) for table in tables))
    target_stats = table_stats(target, tables)
    checks = compare_tables(source_stats, target_stats)
    checks.append(check_sequences(sequence_states(target, tables, source_sequences)))
    checks.append(check_migrations(TARGET))
    if not dry_run:
        with transaction.atomic(using=SOURCE):
            _read_only(source)
            checks.append(check_unchanged(source_stats, table_stats(source, tables)))
    report, passed = render(source_stats, target_stats, checks)
    return f"{log_line}.\n\n{report}", passed


def move(
    source_url: str,
    target_url: str,
    *,
    dry_run: bool = False,
    replace: bool = False,
    keep: bool = False,
    dump_file: Path | None = None,
    confirm: Callable[[str], bool] = lambda question: True,
    log: Log = _quiet,
) -> Outcome:
    """Copies and verifies, see the module's docstring. Raises :class:`MoveError` when it cannot
    start or a step fails; a copy that does not match is a report with ``passed=False``."""
    if same_database(source_url, target_url):
        raise MoveError("The source and the target are the same database.")
    tools = {tool: backups.tool_major(tool) for tool in ("pg_dump", "pg_restore")}
    if None in tools.values():
        raise MoveError(
            "pg_dump and pg_restore are needed (the backend's Docker image has them; elsewhere "
            "install the PostgreSQL client of the source's version or newer)."
        )
    with open_database(SOURCE, source_url) as source, open_database(TARGET, target_url) as target:
        try:
            source_major, target_major = server_major(source), server_major(target)
        except DatabaseError as exc:
            raise MoveError(f"Cannot connect: {exc}") from exc
        if tools["pg_dump"] < source_major or tools["pg_restore"] < source_major:
            raise MoveError(
                f"pg_dump {tools['pg_dump']} / pg_restore {tools['pg_restore']} cannot copy "
                f"PostgreSQL {source_major}: install the client {source_major} or newer."
            )
        if target_major < source_major:
            raise MoveError(
                f"The target runs PostgreSQL {target_major}, older than the source "
                f"({source_major}): use PostgreSQL {source_major} or newer."
            )
        schema = current_schema(source)
        if current_schema(target) != schema:
            raise MoveError(
                f"The tables are in the schema {schema!r} on the source, but the target uses "
                f"{current_schema(target)!r}."
            )
        log(f"From  {describe_url(source_url)} (PostgreSQL {server_version(source)}), read only")
        log(f"To    {describe_url(target_url)} (PostgreSQL {server_version(target)})")

        existing = schema_objects(target)
        if existing:
            if not replace:
                raise MoveError(
                    f"The target is not empty: {len(existing)} tables, views or sequences in "
                    f"the schema {schema!r} ({_names(existing, 5)}). Use an empty database, or "
                    "--replace to drop them first."
                )
            if not confirm(
                f"--replace drops {len(existing)} tables, views and sequences in the schema "
                f"{schema!r} of {describe_url(target_url)}."
            ):
                raise MoveError("Nothing changed.")
            log(f"Dropped {drop_all(target)} tables, views and sequences in the target.")

        with contextlib.ExitStack() as stack:
            if dump_file is None:
                folder = Path(stack.enter_context(tempfile.TemporaryDirectory()))
                path = folder / "move.pgdump"
            else:
                path = dump_file
            try:
                tables, source_stats, source_sequences = _dump_source(source, schema, path, log)
            except _NoSnapshot as exc:
                log(
                    f"No shared snapshot ({exc.__cause__}): pg_dump takes its own, so rows "
                    "written during the dump can show up as differences."
                )
                tables, source_stats, source_sequences = _dump_source(
                    source, schema, path, log, snapshot=False
                )
            started = time.monotonic()
            restore(target.settings_dict, path)
            log(f"Restored in {time.monotonic() - started:.1f} s.")

        target.close()  # pg_restore changed the tables behind this connection's back
        try:
            report, passed = _verify(
                target, source, tables, source_stats, source_sequences, dry_run
            )
        finally:
            if dry_run and not keep:
                drop_all(target)
        if dry_run and not keep:
            report += "\nDry run: the copy was dropped again; the app still uses the source."
        elif dry_run:
            report += (
                "\nDry run: the copy stays in the target (--keep); the app still uses the source."
            )
        elif passed:
            report += (
                "\nThe target holds a verified copy. Point DATABASE_URL to it and start the app."
            )
        else:
            report += (
                "\nThe copy is NOT usable. Keep DATABASE_URL on the source; fix the cause and run "
                "the move again with --replace."
            )
        return Outcome(passed, report)
