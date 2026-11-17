"""Backups of the database and the data directory (``manage.py backup`` / ``manage.py restore``).

A backup is a folder ``brickdnn-YYYYMMDDTHHMMSSZ`` in ``BRICKDNN_BACKUP_DIR`` with

- the database, as a consistent snapshot:
  - SQLite: a copy made with SQLite's online backup (``database.sqlite3``);
  - PostgreSQL: ``pg_dump`` in its custom format (``database.pgdump``) if ``pg_dump`` is
    installed and not older than the server (the Docker image has pg_dump 17); otherwise
    Django's ``dumpdata`` inside one read-only transaction (``database.json.gz``), which also
    loads into another database or another PostgreSQL version;
- the files of the data directory (uploads, datasets, trained weights, the generated secret
  key) as ``files.tar.gz``, without the cache, the SQLite file and the backups themselves. If
  they are larger than ``BRICKDNN_BACKUP_MAX_FILES_MB`` or would not fit on the disk, only a list
  (``files.txt``: path, size, time) is written and the files must come from another backup
  (Hetzner's). ``BRICKDNN_BACKUP_FILES=list`` or ``none`` chooses that up front;
- ``manifest.json``: when, which version, what is in it.

Hourly backups (``manage.py backup --hourly``, with ``BRICKDNN_BACKUP_KEEP_HOURLY`` above 0) hold
the database only, in folders ``brickdnn-YYYYMMDDTHHMMSSZ-hourly`` (so that offsite copies can
keep them for a shorter time than the daily ones).

Old backups are deleted: the newest of each of the last ``BRICKDNN_BACKUP_KEEP_HOURLY`` hours (of
either kind), and the newest daily backup of each of the last ``BRICKDNN_BACKUP_KEEP_DAILY`` days
and of each of the last ``BRICKDNN_BACKUP_KEEP_WEEKLY`` weeks are kept. A folder is only renamed
to its final name when it is complete. The backups hold password hashes and the secret key:
the folder is readable by its owner only. See deploy/server/README.md, "Backups".
"""

from __future__ import annotations

import contextlib
import gzip
import json
import os
import re
import shutil
import sqlite3
import subprocess
import tarfile
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from django.conf import settings
from django.core.management import call_command
from django.db import connection, transaction

import brickdnn

PREFIX = "brickdnn-"
NAME = re.compile(r"^brickdnn-(\d{8}T\d{6}Z)(-hourly)?$")
STAMP = "%Y%m%dT%H%M%SZ"
DAILY = "daily"
HOURLY = "hourly"
KINDS = (DAILY, HOURLY)
STATUS_FILE = ".last-run.json"
HOURLY_STATUS_FILE = ".last-run-hourly.json"
# With hourly backups on, /api/health/ reports a failed backup when the newest backup of either
# kind is older than this.
HOURLY_MAX_AGE_HOURS = 3
LOCK_FILE = ".lock"
MANIFEST = "manifest.json"
SQLITE_FILE = "database.sqlite3"
PGDUMP_FILE = "database.pgdump"
JSON_FILE = "database.json.gz"
ARCHIVE_FILE = "files.tar.gz"
LIST_FILE = "files.txt"
FILE_MODES = ("archive", "list", "none")
DATABASE_FORMATS = ("auto", "native", "json")
# Top-level entries of the data directory that are not backed up: the cache can be rebuilt,
# the SQLite database is in the database part of the backup.
EXCLUDED = {"cache", "db.sqlite3", "db.sqlite3-journal", "db.sqlite3-wal", "db.sqlite3-shm"}
# Not in a JSON dump: rebuilt by ``migrate`` (content types, permissions) or not worth keeping
# (sessions of the Django admin).
JSON_EXCLUDE = ["contenttypes", "auth.permission", "sessions"]
# Connection options that libpq (pg_dump, pg_restore) understands.
LIBPQ_OPTIONS = {
    "sslmode",
    "sslrootcert",
    "sslcert",
    "sslkey",
    "sslcrl",
    "connect_timeout",
    "options",
    "target_session_attrs",
    "application_name",
    "channel_binding",
    "gssencmode",
    "keepalives",
}


class BackupError(Exception):
    """Something is in the way of a backup or restore; the message says what."""


Log = Callable[[str], None]


def _quiet(message: str) -> None:
    pass


# --------------------------------------------------------------------------- #
# The folder and its backups
# --------------------------------------------------------------------------- #


def backup_dir() -> Path | None:
    value = settings.BACKUP_DIR
    return Path(value) if value else None


@dataclass(frozen=True)
class Backup:
    path: Path
    created: datetime

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def kind(self) -> str:
        """``hourly`` (the database only) or ``daily`` (the database and the files)."""
        return HOURLY if self.name.endswith("-" + HOURLY) else DAILY

    def manifest(self) -> dict[str, Any]:
        try:
            return json.loads((self.path / MANIFEST).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def size(self) -> int:
        return sum(f.stat().st_size for f in self.path.iterdir() if f.is_file())


def list_backups(directory: Path) -> list[Backup]:
    """The complete backups in ``directory``, newest first."""
    found = []
    with contextlib.suppress(OSError):
        for entry in directory.iterdir():
            match = NAME.match(entry.name)
            if match and entry.is_dir() and (entry / MANIFEST).is_file():
                created = datetime.strptime(match.group(1), STAMP).replace(tzinfo=UTC)
                found.append(Backup(entry, created))
    return sorted(found, key=lambda backup: backup.created, reverse=True)


def to_delete(backups: list[Backup], daily: int, weekly: int, hourly: int = 0) -> list[Backup]:
    """What rotation removes: all but the newest backup of each of the last ``hourly`` hours,
    and the newest daily backup of each of the last ``daily`` days and of each of the last
    ``weekly`` ISO weeks (an hourly one where a day or week has no daily backup). The newest
    backup is always kept."""
    newest_first = sorted(backups, key=lambda b: b.created, reverse=True)
    keep: set[Path] = set()

    def newest_of(key: Callable[[Backup], Any], count: int, prefer_daily: bool) -> None:
        groups: dict[Any, list[Backup]] = {}
        for backup in newest_first:
            groups.setdefault(key(backup), []).append(backup)
        for group in list(groups.values())[: max(count, 0)]:
            chosen = group[0]
            if prefer_daily:
                chosen = next((b for b in group if b.kind == DAILY), chosen)
            keep.add(chosen.path)

    newest_of(lambda b: b.created.replace(minute=0, second=0, microsecond=0), hourly, False)
    newest_of(lambda b: b.created.date(), daily, True)
    newest_of(lambda b: b.created.isocalendar()[:2], weekly, True)
    if newest_first:
        keep.add(newest_first[0].path)
    return [backup for backup in backups if backup.path not in keep]


def find(reference: str, directory: Path | None = None) -> Backup:
    """A backup by path, folder name or ``latest``."""
    path = Path(reference)
    if path.is_dir() and (path / MANIFEST).is_file():
        match = NAME.match(path.name)
        created = (
            datetime.strptime(match.group(1), STAMP).replace(tzinfo=UTC)
            if match
            else datetime.fromtimestamp(path.stat().st_mtime, UTC)
        )
        return Backup(path, created)
    if directory is None:
        raise BackupError(
            f"{reference} is not a backup folder, and BRICKDNN_BACKUP_DIR is not set."
        )
    backups = list_backups(directory)
    if reference == "latest":
        if not backups:
            raise BackupError(f"There is no backup in {directory}.")
        return backups[0]
    for backup in backups:
        if backup.name == reference:
            return backup
    raise BackupError(f"There is no backup {reference!r} in {directory}.")


@contextlib.contextmanager
def _locked(directory: Path) -> Iterator[None]:
    """One backup at a time (the cleanup service and someone by hand)."""
    try:
        import fcntl
    except ImportError:  # pragma: no cover - Windows
        yield
        return
    with open(directory / LOCK_FILE, "w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise BackupError("Another backup is running.") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _status_file(kind: str) -> str:
    return HOURLY_STATUS_FILE if kind == HOURLY else STATUS_FILE


def _write_status(directory: Path, kind: str = DAILY, **status: Any) -> None:
    status["time"] = datetime.now(UTC).isoformat()
    with contextlib.suppress(OSError):
        (directory / _status_file(kind)).write_text(json.dumps(status), encoding="utf-8")


def _read_status(directory: Path, kind: str) -> dict[str, Any] | None:
    try:
        return json.loads((directory / _status_file(kind)).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _age_hours(backup: Backup | None) -> float | None:
    if backup is None:
        return None
    return (datetime.now(UTC) - backup.created).total_seconds() / 3600


def health() -> tuple[bool, dict[str, Any]] | None:
    """For /api/health/: the last backup worked and the newest daily one is recent; with hourly
    backups on, also the last hourly run worked and the newest backup is at most
    ``HOURLY_MAX_AGE_HOURS`` old. None: backups are off or have not run yet."""
    directory = backup_dir()
    if directory is None:
        return None
    status = _read_status(directory, DAILY)
    if status is None:
        return None
    backups = list_backups(directory)
    newest_daily = next((b for b in backups if b.kind == DAILY), None)
    age_hours = _age_hours(newest_daily)
    passed = bool(status.get("ok")) and age_hours is not None
    passed = passed and age_hours <= settings.BACKUP_MAX_AGE_HOURS
    info: dict[str, Any] = {
        "last_run": status.get("time"),
        "newest": newest_daily.name if newest_daily else None,
        "newest_age_hours": round(age_hours, 1) if age_hours is not None else None,
        "max_age_hours": settings.BACKUP_MAX_AGE_HOURS,
        "count": len(backups),
    }
    if status.get("error"):
        info["error"] = str(status["error"])[:300]
    hourly = _read_status(directory, HOURLY) if settings.BACKUP_KEEP_HOURLY else None
    if hourly is not None:
        newest = backups[0] if backups else None
        newest_age = _age_hours(newest)
        hourly_ok = bool(hourly.get("ok")) and newest_age is not None
        hourly_ok = hourly_ok and newest_age <= HOURLY_MAX_AGE_HOURS
        info["hourly"] = {
            "last_run": hourly.get("time"),
            "newest": newest.name if newest else None,
            "newest_age_hours": round(newest_age, 1) if newest_age is not None else None,
            "max_age_hours": HOURLY_MAX_AGE_HOURS,
        }
        if hourly.get("error"):
            info["hourly"]["error"] = str(hourly["error"])[:300]
        passed = passed and hourly_ok
    return passed, info


# --------------------------------------------------------------------------- #
# The database
# --------------------------------------------------------------------------- #


def pg_conninfo(db: dict[str, Any] | None = None) -> tuple[str, dict[str, str]]:
    """A libpq connection string for pg_dump / pg_restore, and the environment with the password
    (not on the command line, where other processes could see it). ``db``: the Django settings
    of a database (default: the app's)."""
    db = connection.settings_dict if db is None else db
    parts = {
        "host": db.get("HOST") or "",
        "port": str(db.get("PORT") or ""),
        "dbname": db.get("NAME") or "",
        "user": db.get("USER") or "",
    }
    for key, value in (db.get("OPTIONS") or {}).items():
        if key in LIBPQ_OPTIONS:
            parts[key] = str(value)

    def quote(value: str) -> str:
        return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"

    conninfo = " ".join(f"{key}={quote(value)}" for key, value in parts.items() if value)
    env = {**os.environ}
    if db.get("PASSWORD"):
        env["PGPASSWORD"] = db["PASSWORD"]
    return conninfo, env


def tool_major(tool: str) -> int | None:
    """The major version of an installed PostgreSQL program (pg_dump, pg_restore), or None."""
    path = shutil.which(tool)
    if not path:
        return None
    try:
        output = subprocess.run(
            [path, "--version"], capture_output=True, text=True, timeout=30, check=True
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"(\d+)(?:\.\d+)?", output)
    return int(match.group(1)) if match else None


def _server_major() -> int:
    connection.ensure_connection()
    return connection.pg_version // 10000


def database_format(requested: str = "auto") -> str:
    """``sqlite``, ``pgdump`` or ``json`` for this database."""
    if requested == "json":
        return "json"
    if connection.vendor == "sqlite":
        return "sqlite"
    if connection.vendor != "postgresql":
        if requested == "native":
            raise BackupError(f"No native backup for {connection.vendor}; use json.")
        return "json"
    tool = tool_major("pg_dump")
    server = _server_major()
    if tool is not None and tool >= server:
        return "pgdump"
    if requested == "native":
        raise BackupError(
            "pg_dump is not installed"
            if tool is None
            else f"pg_dump {tool} cannot dump PostgreSQL {server}; install pg_dump {server}"
        )
    return "json"


def _schema() -> str:
    with connection.cursor() as cursor:
        cursor.execute("SELECT current_schema()")
        return cursor.fetchone()[0]


def dump_database(target: Path, fmt: str, log: Log = _quiet) -> str:
    """Writes the database into the folder ``target``; returns the file name."""
    if fmt == "sqlite":
        connection.ensure_connection()
        destination = sqlite3.connect(target / SQLITE_FILE)
        try:
            connection.connection.backup(destination)
        finally:
            destination.close()
        return SQLITE_FILE
    if fmt == "pgdump":
        conninfo, env = pg_conninfo()
        command = [
            "pg_dump",
            "--format=custom",
            "--no-owner",
            f"--schema={_schema()}",
            f"--file={target / PGDUMP_FILE}",
            f"--dbname={conninfo}",
        ]
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        if result.returncode:
            raise BackupError(f"pg_dump failed: {result.stderr.strip()[-500:]}")
        return PGDUMP_FILE
    # json: one transaction, so that all tables show the same moment (inside a transaction
    # someone else opened, e.g. a test's, that one decides).
    outer = connection.in_atomic_block
    with transaction.atomic():
        if connection.vendor == "postgresql" and not outer:
            with connection.cursor() as cursor:
                cursor.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        with gzip.open(target / JSON_FILE, "wt", encoding="utf-8") as out:
            call_command(
                "dumpdata",
                natural_foreign=True,
                exclude=JSON_EXCLUDE,
                stdout=out,
                verbosity=0,
            )
    return JSON_FILE


def _empty_tables() -> None:
    """Deletes all rows of the app's tables except content types and permissions (``migrate``
    made them; a JSON dump refers to them by name)."""
    from django.contrib.auth.models import Permission
    from django.contrib.contenttypes.models import ContentType
    from django.core.management.color import no_style

    kept = {ContentType._meta.db_table, Permission._meta.db_table}
    tables = [
        table
        for table in connection.introspection.django_table_names(
            only_existing=True, include_views=False
        )
        if table not in kept
    ]
    statements = connection.ops.sql_flush(no_style(), tables, reset_sequences=True)
    connection.ops.execute_sql_flush(statements)


def restore_database(backup: Backup, log: Log = _quiet) -> None:
    manifest = backup.manifest()
    fmt = (manifest.get("database") or {}).get("format")
    path = backup.path / ((manifest.get("database") or {}).get("file") or "")
    if not path.is_file():
        raise BackupError(f"The backup {backup.name} has no database file.")
    if fmt == "sqlite":
        if connection.vendor != "sqlite":
            raise BackupError(
                "This backup is a SQLite file, but the database is "
                f"{connection.vendor}. Restore it with DATABASE_URL unset, then move the data "
                "with dumpdata/loaddata (docs/database.md)."
            )
        connection.ensure_connection()
        source = sqlite3.connect(path)
        try:
            source.backup(connection.connection)
        finally:
            source.close()
        call_command("migrate", interactive=False, verbosity=0)
    elif fmt == "pgdump":
        if connection.vendor != "postgresql":
            raise BackupError("This backup was made with pg_dump; it needs a PostgreSQL database.")
        if not shutil.which("pg_restore"):
            raise BackupError("pg_restore is not installed.")
        conninfo, env = pg_conninfo()
        connection.close()
        command = [
            "pg_restore",
            "--clean",
            "--if-exists",
            "--no-owner",
            "--single-transaction",
            f"--dbname={conninfo}",
            str(path),
        ]
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        if result.returncode:
            raise BackupError(f"pg_restore failed: {result.stderr.strip()[-500:]}")
        # Newer migrations of this code, and on Supabase the protection of the tables.
        call_command("migrate", interactive=False, verbosity=0)
    elif fmt == "json":
        call_command("migrate", interactive=False, verbosity=0)
        with transaction.atomic():
            _empty_tables()
            call_command("loaddata", str(path), verbosity=0)
    else:
        raise BackupError(f"Unknown database format {fmt!r} in {backup.name}.")
    log(f"Database restored from {path.name} ({fmt}).")


# --------------------------------------------------------------------------- #
# The files
# --------------------------------------------------------------------------- #


def _data_entries(data_dir: Path, skip: set[Path]) -> list[Path]:
    entries = []
    with contextlib.suppress(OSError):
        for entry in sorted(data_dir.iterdir()):
            if entry.name in EXCLUDED or entry.name.startswith("."):
                continue
            if any(entry == path or path.is_relative_to(entry) for path in skip):
                if entry in skip:
                    continue
                # The backup folder is somewhere inside: go one level deeper.
                entries += _data_entries(entry, skip)
                continue
            entries.append(entry)
    return entries


def _walk(entries: list[Path]) -> Iterator[Path]:
    for entry in entries:
        if entry.is_symlink() or entry.is_file():
            yield entry
        elif entry.is_dir():
            for root, dirs, files in os.walk(entry):
                dirs.sort()
                for name in sorted(files):
                    yield Path(root) / name


def write_files(
    target: Path, mode: str, data_dir: Path, skip: set[Path], log: Log = _quiet
) -> dict[str, Any]:
    """The data directory as archive or list; returns its part of the manifest."""
    if mode == "none":
        return {"mode": "none"}
    entries = _data_entries(data_dir, skip)
    files = []
    total = 0
    for path in _walk(entries):
        try:
            stat = path.lstat()
        except OSError:
            continue  # deleted meanwhile
        files.append((path, stat))
        total += stat.st_size
    info: dict[str, Any] = {
        "folders": [entry.name for entry in entries],
        "count": len(files),
        "size_bytes": total,
    }
    if mode == "archive":
        limit = settings.BACKUP_MAX_FILES_MB * 1024**2
        free = shutil.disk_usage(target).free - settings.HEALTH_MIN_FREE_MB * 1024**2
        if limit and total > limit:
            log(
                f"The files are {total / 1024**2:.0f} MB, more than BRICKDNN_BACKUP_MAX_FILES_MB "
                f"({settings.BACKUP_MAX_FILES_MB} MB): writing a list of them instead."
            )
            mode = "list"
        elif total > free:
            log(f"The files ({total / 1024**2:.0f} MB) do not fit on the disk: writing a list.")
            mode = "list"
    if mode == "archive":
        with tarfile.open(target / ARCHIVE_FILE, "w:gz", compresslevel=6) as archive:
            for path, _stat in files:
                with contextlib.suppress(FileNotFoundError):
                    archive.add(path, arcname=str(path.relative_to(data_dir)), recursive=False)
        info.update(mode="archive", file=ARCHIVE_FILE)
    else:
        with open(target / LIST_FILE, "w", encoding="utf-8") as out:
            out.write("# path\tbytes\tmodified (UTC)\n")
            for path, stat in files:
                modified = datetime.fromtimestamp(stat.st_mtime, UTC).strftime("%Y-%m-%d %H:%M:%S")
                out.write(f"{path.relative_to(data_dir)}\t{stat.st_size}\t{modified}\n")
        info.update(mode="list", file=LIST_FILE)
    return info


def restore_files(backup: Backup, data_dir: Path, log: Log = _quiet) -> None:
    files = backup.manifest().get("files") or {}
    if files.get("mode") != "archive":
        log(
            f"The backup {backup.name} has no copy of the files ({files.get('mode', 'none')}); "
            "restore them from another backup, e.g. Hetzner's."
        )
        return
    with tarfile.open(backup.path / ARCHIVE_FILE, "r:gz") as archive:
        # "data": no absolute paths, nothing outside data_dir, no device files.
        archive.extractall(data_dir, filter="data")
    log(f"{files.get('count', '?')} files restored into {data_dir}.")


# --------------------------------------------------------------------------- #
# Backup and restore
# --------------------------------------------------------------------------- #


def create(
    directory: Path,
    *,
    files: str | None = None,
    database: str = "auto",
    keep_daily: int | None = None,
    keep_weekly: int | None = None,
    keep_hourly: int | None = None,
    kind: str = DAILY,
    log: Log = _quiet,
    now: datetime | None = None,
) -> Backup:
    """Makes a backup in ``directory`` and deletes old ones. Raises :class:`BackupError`.
    ``kind="hourly"``: the database only, in a folder whose name ends in ``-hourly``."""
    if kind not in KINDS:
        raise BackupError(f"Unknown kind of backup {kind!r}.")
    files = "none" if kind == HOURLY else (files or settings.BACKUP_FILES)
    if files not in FILE_MODES:
        raise BackupError(f"BRICKDNN_BACKUP_FILES must be one of {', '.join(FILE_MODES)}.")
    keep_daily = settings.BACKUP_KEEP_DAILY if keep_daily is None else keep_daily
    keep_weekly = settings.BACKUP_KEEP_WEEKLY if keep_weekly is None else keep_weekly
    keep_hourly = settings.BACKUP_KEEP_HOURLY if keep_hourly is None else keep_hourly
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    with contextlib.suppress(OSError):
        directory.chmod(0o700)
    with _locked(directory):
        try:
            backup = _create(directory, files, database, now or datetime.now(UTC), log, kind)
        except Exception as exc:
            _write_status(directory, kind, ok=False, error=f"{type(exc).__name__}: {exc}")
            raise
        for old in to_delete(list_backups(directory), keep_daily, keep_weekly, keep_hourly):
            shutil.rmtree(old.path, ignore_errors=True)
            log(f"Deleted the old backup {old.name}.")
        _write_status(directory, kind, ok=True, backup=backup.name)
    return backup


def _create(
    directory: Path, files: str, database: str, now: datetime, log: Log, kind: str = DAILY
) -> Backup:
    # Left-overs of backups that were interrupted.
    for entry in directory.glob(".partial-*"):
        if time.time() - entry.stat().st_mtime > 3600:
            shutil.rmtree(entry, ignore_errors=True)
    now = now.replace(microsecond=0)
    endings = ("", "-" + HOURLY)
    # Two backups within a second (of either kind: the times stay unique).
    while any((directory / (PREFIX + now.strftime(STAMP) + end)).exists() for end in endings):
        now += timedelta(seconds=1)
    name = PREFIX + now.strftime(STAMP) + ("-" + HOURLY if kind == HOURLY else "")
    final = directory / name
    partial = directory / f".partial-{name}"
    partial.mkdir(mode=0o700)
    started = time.monotonic()
    try:
        fmt = database_format(database)
        db_file = dump_database(partial, fmt, log)
        log(f"Database: {db_file} ({(partial / db_file).stat().st_size / 1024**2:.1f} MB).")
        data_dir = Path(settings.DATA_DIR).resolve()
        files_info = write_files(partial, files, data_dir, {directory.resolve()}, log)
        if files_info["mode"] != "none":
            log(
                f"Files: {files_info['count']} ({files_info['size_bytes'] / 1024**2:.1f} MB) "
                f"as {files_info['file']}."
            )
        manifest = {
            "created_at": now.isoformat(),
            "kind": kind,
            "brickdnn_version": brickdnn.__version__,
            "database": {"vendor": connection.vendor, "format": fmt, "file": db_file},
            "migrations": _migrations(),
            "files": files_info,
            "seconds": round(time.monotonic() - started, 1),
        }
        (partial / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        partial.rename(final)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return Backup(final, now)


def _migrations() -> dict[str, str]:
    """The newest applied migration of each app (to compare with the code when restoring)."""
    from django.db.migrations.recorder import MigrationRecorder

    newest: dict[str, str] = {}
    with contextlib.suppress(Exception):
        for app, name in MigrationRecorder(connection).applied_migrations():
            newest[app] = max(newest.get(app, ""), name)
    return dict(sorted(newest.items()))


def restore(
    backup: Backup, *, database: bool = True, files: bool = True, log: Log = _quiet
) -> None:
    if database:
        restore_database(backup, log)
    if files:
        restore_files(backup, Path(settings.DATA_DIR), log)


def describe(backup: Backup) -> str:
    manifest = backup.manifest()
    db = manifest.get("database") or {}
    files = manifest.get("files") or {}
    return (
        f"{backup.name}: {backup.kind}, {backup.size() / 1024**2:.1f} MB, "
        f"database {db.get('vendor')} "
        f"({db.get('format')}), files {files.get('mode')}"
        + (f" ({files.get('count')})" if files.get("mode") != "none" else "")
        + f", Brick DNN {manifest.get('brickdnn_version', '?')}"
    )
