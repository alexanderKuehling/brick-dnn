"""``python manage.py restore BACKUP``: put a backup of ``manage.py backup`` back.

Replaces the whole database and writes the files of the backup into the data directory (files
that were added later stay). Stop the site first: see deploy/server/README.md, "Backups".
"""

from __future__ import annotations

from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from brick_dnn_core import backups


class Command(BaseCommand):
    help = (
        "Restore a backup made by `manage.py backup` (a folder, its name in "
        "BRICKDNN_BACKUP_DIR, or 'latest'). Replaces the database."
    )

    def add_arguments(self, parser) -> None:
        parser.add_argument("backup", help="Folder of the backup, its name, or 'latest'.")
        parser.add_argument("--dir", help="Instead of BRICKDNN_BACKUP_DIR.")
        part = parser.add_mutually_exclusive_group()
        part.add_argument("--database-only", action="store_true", help="Leave the files alone.")
        part.add_argument("--files-only", action="store_true", help="Leave the database alone.")
        parser.add_argument(
            "--no-input", action="store_false", dest="interactive", help="Do not ask."
        )

    def handle(self, *args, **options) -> None:
        directory = Path(options["dir"]) if options["dir"] else backups.backup_dir()
        try:
            backup = backups.find(options["backup"], directory)
        except backups.BackupError as exc:
            raise CommandError(str(exc)) from exc
        database = not options["files_only"]
        files = not options["database_only"]
        self.stdout.write(f"Restoring {backups.describe(backup)}")
        parts = " and ".join(
            part for part, chosen in (("the database", database), ("the files", files)) if chosen
        )
        if options["interactive"]:
            answer = input(
                f"This replaces {parts} with the backup. Stop the site first. Type 'yes' to go on: "
            )
            if answer.strip().lower() != "yes":
                raise CommandError("Nothing restored.")
        try:
            backups.restore(backup, database=database, files=files, log=self.stdout.write)
        except backups.BackupError as exc:
            raise CommandError(f"Restore failed: {exc}") from exc
        self.stdout.write(self.style.SUCCESS(f"Restored {parts} from {backup.name}."))
