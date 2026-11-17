"""``python manage.py move_database --to URL [--from URL] [--dry-run]``: copy the database.

Copies Django's tables from one PostgreSQL database into another, empty one and verifies the
copy (rows, checksums, sequences, migrations); prints PASS or FAIL. The source is only read.
See brick_dnn_core/database_move.py and deploy/server/README.md, "Moving the database onto the
server".
"""

from __future__ import annotations

from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from brick_dnn_core import database_move


class Command(BaseCommand):
    help = (
        "Copy Django's tables from one PostgreSQL database (default: DATABASE_URL) into another, "
        "empty one, and verify the copy. The source is only read."
    )

    def add_arguments(self, parser) -> None:
        url = (
            "a postgresql:// URL, or env:NAME for the URL in the environment variable NAME "
            "(keeps the password off the command line)"
        )
        parser.add_argument(
            "--from",
            dest="source",
            default="env:DATABASE_URL",
            help=f"The database to copy: {url}. Default: env:DATABASE_URL (the app's).",
        )
        parser.add_argument("--to", dest="target", required=True, help=f"The new database: {url}.")
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="A rehearsal: copy and verify, then drop the copy again. Safe while the site "
            "runs; the counts compare the same moment of the source.",
        )
        parser.add_argument(
            "--keep",
            action="store_true",
            help="With --dry-run: leave the copy in the target to look at.",
        )
        parser.add_argument(
            "--replace",
            action="store_true",
            help="Drop all tables, views and sequences in the target's schema first (asks).",
        )
        parser.add_argument(
            "--dump-file",
            type=Path,
            help="Keep the dump (pg_dump's custom format) in this file. It holds password hashes.",
        )
        parser.add_argument(
            "--no-input", action="store_false", dest="interactive", help="Do not ask."
        )

    def handle(self, *args, **options) -> None:
        if options["keep"] and not options["dry_run"]:
            raise CommandError("--keep only goes with --dry-run.")
        try:
            source = database_move.resolve_url(options["source"])
            target = database_move.resolve_url(options["target"])
        except database_move.MoveError as exc:
            raise CommandError(str(exc)) from exc

        def confirm(question: str) -> bool:
            if not options["interactive"]:
                return True
            return input(f"{question} Type 'yes' to go on: ").strip().lower() == "yes"

        self.stdout.write("Dry run (rehearsal)." if options["dry_run"] else "Moving the database.")
        try:
            outcome = database_move.move(
                source,
                target,
                dry_run=options["dry_run"],
                replace=options["replace"],
                keep=options["keep"],
                dump_file=options["dump_file"],
                confirm=confirm,
                log=self.stdout.write,
            )
        except database_move.MoveError as exc:
            raise CommandError(f"Move failed: {exc}") from exc
        self.stdout.write("")
        self.stdout.write(outcome.report)
        if not outcome.passed:
            raise CommandError("The copy did not pass the checks (FAIL above).")
