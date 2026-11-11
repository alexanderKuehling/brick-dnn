"""``python manage.py backup [--list]``: a backup of the database and the data directory.

Off unless ``BRICKDNN_BACKUP_DIR`` (or ``--dir``) names a folder. Docker Compose's cleanup
service runs it once a day. See brick_dnn_core/backups.py and deploy/server/README.md.
"""

from __future__ import annotations

from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from brick_dnn_core import backups


class Command(BaseCommand):
    help = (
        "Back up the database and the files of the data directory into BRICKDNN_BACKUP_DIR "
        "and delete old backups (BRICKDNN_BACKUP_KEEP_DAILY / _WEEKLY)."
    )

    def add_arguments(self, parser) -> None:
        parser.add_argument("--dir", help="Instead of BRICKDNN_BACKUP_DIR.")
        parser.add_argument(
            "--files",
            choices=backups.FILE_MODES,
            help="Instead of BRICKDNN_BACKUP_FILES: a tarball, a list or nothing of the files.",
        )
        parser.add_argument(
            "--database-format",
            choices=backups.DATABASE_FORMATS,
            default="auto",
            help="auto: SQLite's own backup, pg_dump if installed, else dumpdata; json: always "
            "dumpdata (loads into SQLite and PostgreSQL alike); native: fail without pg_dump.",
        )
        parser.add_argument("--keep-daily", type=int, metavar="N")
        parser.add_argument("--keep-weekly", type=int, metavar="N")
        parser.add_argument("--list", action="store_true", help="Only list the backups.")

    def handle(self, *args, **options) -> None:
        directory = Path(options["dir"]) if options["dir"] else backups.backup_dir()
        if directory is None:
            self.stdout.write("Backups are off: BRICKDNN_BACKUP_DIR is not set.")
            return
        if options["list"]:
            found = backups.list_backups(directory)
            for backup in found:
                self.stdout.write(backups.describe(backup))
            self.stdout.write(f"{len(found)} backups in {directory}.")
            return
        try:
            backup = backups.create(
                directory,
                files=options["files"],
                database=options["database_format"],
                keep_daily=options["keep_daily"],
                keep_weekly=options["keep_weekly"],
                log=self.stdout.write,
            )
        except backups.BackupError as exc:
            raise CommandError(f"Backup failed: {exc}") from exc
        self.stdout.write(self.style.SUCCESS(f"Backup {backups.describe(backup)}"))
