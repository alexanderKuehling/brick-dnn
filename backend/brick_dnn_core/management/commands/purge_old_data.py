"""``python manage.py purge_old_data [--dry-run]``: delete data nobody has used for a while.

What counts as old comes from the settings ``BRICKDNN_DELETE_UNUSED_UPLOADS_DAYS`` and
``BRICKDNN_DELETE_INACTIVE_ACCOUNTS_DAYS`` (both off by default), ``BRICKDNN_AGENT_LOG_DAYS``
(the activity log of agents, 90 days by default) or the options below. Expired logins are
always cleaned up. With e-mail set up, inactive accounts are warned before they are deleted
(``BRICKDNN_INACTIVE_ACCOUNT_WARNING_DAYS``, see retention.py). Docker Compose runs this once a
day (the ``cleanup`` service); see docs/privacy.md.
"""

from __future__ import annotations

from django.conf import settings
from django.core.management import call_command
from django.core.management.base import BaseCommand, CommandError

from brick_dnn_core.retention import purge


def _days(value: str) -> int:
    try:
        days = int(value)
    except ValueError:
        raise CommandError(f"Not a number of days: {value!r}") from None
    if days < 0:
        raise CommandError("The number of days cannot be negative.")
    return days


class Command(BaseCommand):
    help = (
        "Delete data sources, datasets and finished training runs nobody has used for "
        "BRICKDNN_DELETE_UNUSED_UPLOADS_DAYS days and accounts nobody has used for "
        "BRICKDNN_DELETE_INACTIVE_ACCOUNTS_DAYS days (0: keep them), entries of the agent "
        "activity log older than BRICKDNN_AGENT_LOG_DAYS days, and expired logins."
    )

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--dry-run", action="store_true", help="Only list what would be deleted."
        )
        parser.add_argument(
            "--unused-uploads-days",
            type=_days,
            default=None,
            metavar="DAYS",
            help="Instead of BRICKDNN_DELETE_UNUSED_UPLOADS_DAYS (0: keep them).",
        )
        parser.add_argument(
            "--inactive-accounts-days",
            type=_days,
            default=None,
            metavar="DAYS",
            help="Instead of BRICKDNN_DELETE_INACTIVE_ACCOUNTS_DAYS (0: keep them).",
        )
        parser.add_argument(
            "--agent-log-days",
            type=_days,
            default=None,
            metavar="DAYS",
            help="Instead of BRICKDNN_AGENT_LOG_DAYS (0: delete the whole agent activity log).",
        )

    def handle(self, *args, **options) -> None:
        uploads = options["unused_uploads_days"]
        accounts = options["inactive_accounts_days"]
        if uploads is None:
            uploads = max(0, settings.DELETE_UNUSED_UPLOADS_DAYS)
        if accounts is None:
            accounts = max(0, settings.DELETE_INACTIVE_ACCOUNTS_DAYS)
        dry_run = options["dry_run"]

        def kept(days: int) -> str:
            return f"after {days} days without use" if days else "never (off)"

        self.stdout.write(
            f"purge_old_data{' (dry run)' if dry_run else ''}: deleting uploads and runs "
            f"{kept(uploads)}, accounts {kept(accounts)}."
        )
        agent_log = options["agent_log_days"]
        if agent_log is None:
            agent_log = settings.AGENT_LOG_DAYS
        self.stdout.write(
            f"Agent activity log: kept for {agent_log} days."
            if agent_log
            else "Agent activity log: off, deleting all of it."
        )
        result = purge(
            uploads, accounts, dry_run=dry_run, log=self.stdout.write, agent_log_days=agent_log
        )
        if not dry_run:
            call_command("clearsessions")  # expired logins of the Django admin
        if result.warned or result.not_warned_yet:
            self.stdout.write(
                f"{'Would warn' if dry_run else 'Warned'} {result.warned} inactive accounts by "
                f"e-mail; kept {result.not_warned_yet} that were not warned in time yet."
            )
        done = "Would delete" if dry_run else "Deleted"
        self.stdout.write(
            f"{done} {result.accounts} accounts, {result.data_sources} data sources, "
            f"{result.data_preps} preparations, {result.datasets} datasets, "
            f"{result.training_runs} training runs, {result.agent_actions} entries of the agent "
            f"activity log and {result.tokens} expired login tokens."
        )
