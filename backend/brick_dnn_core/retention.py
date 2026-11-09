"""How long data is kept: "last used" times, and deleting old data automatically.

Uploads (data sources, datasets) and training runs remember when they were last used, and
accounts when they were last logged in to or used with an API token. ``manage.py
purge_old_data`` deletes what nobody has used for ``BRICKDNN_DELETE_UNUSED_UPLOADS_DAYS``
(uploads, finished runs) or ``BRICKDNN_DELETE_INACTIVE_ACCOUNTS_DAYS`` days (accounts with
everything they own). Both are off by default. Entries of the activity log of AI agents are
deleted after ``BRICKDNN_AGENT_LOG_DAYS`` days (default 90, see agent_activity.py). See
docs/privacy.md.

The purge never asks training processes for their state: Docker Compose runs it in a
container of its own, which cannot see them (they would look dead). It skips everything an
active run uses instead.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from django.conf import settings
from django.contrib.auth import get_user_model
from django.db.models import QuerySet
from django.db.models.functions import Coalesce
from django.utils import timezone

from brickdnn.training import ACTIVE_STATUSES, TERMINAL_STATUSES

from .models import Dataset, DataSource, TrainingRun, UserProfile

# "Last used" times are written at most this often, not on every request.
TOUCH_INTERVAL = timedelta(hours=1)


def touch(item: DataSource | Dataset | TrainingRun) -> None:
    """Remember that ``item`` was used now (at most once per TOUCH_INTERVAL)."""
    now = timezone.now()
    last = getattr(item, "last_used_at", None)
    if item.pk is None or (last is not None and now - last < TOUCH_INTERVAL):
        return
    type(item).objects.filter(pk=item.pk).update(last_used_at=now)
    item.last_used_at = now


def record_activity(user_id: Any) -> None:
    """Remember that the account was used now: a log-in, a token refresh or an API token."""
    if user_id is None:
        return
    now = timezone.now()
    recent = UserProfile.objects.filter(
        user_id=user_id, last_active_at__gte=now - TOUCH_INTERVAL
    ).exists()
    if not recent and get_user_model().objects.filter(pk=user_id).exists():
        UserProfile.objects.update_or_create(user_id=user_id, defaults={"last_active_at": now})


# --------------------------------------------------------------------------- #
# What is old
# --------------------------------------------------------------------------- #


def unused_sources(cutoff: datetime) -> QuerySet[DataSource]:
    """Imported files not used since ``cutoff``. Their preparations go with them; one that
    was changed since then keeps its source."""
    return (
        DataSource.objects.filter(last_used_at__lt=cutoff)
        .exclude(preps__updated_at__gte=cutoff)
        .select_related("owner")
        .order_by("pk")
    )


def unused_datasets(cutoff: datetime) -> QuerySet[Dataset]:
    """Training datasets not used since ``cutoff`` and not used by an active training."""
    return (
        Dataset.objects.filter(last_used_at__lt=cutoff)
        .exclude(runs__status__in=ACTIVE_STATUSES)
        .select_related("owner")
        .order_by("pk")
    )


def unused_runs(cutoff: datetime) -> QuerySet[TrainingRun]:
    """Finished training runs not looked at since ``cutoff`` (nor finished after it)."""
    return (
        TrainingRun.objects.filter(status__in=TERMINAL_STATUSES, last_used_at__lt=cutoff)
        .exclude(finished_at__gte=cutoff)
        .select_related("owner")
        .order_by("created_at")
    )


def inactive_users(cutoff: datetime) -> QuerySet:
    """Accounts nobody has logged in to, refreshed a login of or used an API token or a
    connected app of since ``cutoff``. Staff accounts and accounts with an active training
    are never included."""
    User = get_user_model()
    return (
        User.objects.filter(is_staff=False, is_superuser=False)
        .annotate(last_seen=Coalesce("profile__last_active_at", "last_login", "date_joined"))
        .filter(last_seen__lt=cutoff)
        .exclude(api_tokens__last_used_at__gte=cutoff)
        .exclude(oauth_connections__last_used_at__gte=cutoff)
        .exclude(training_runs__status__in=ACTIVE_STATUSES)
        .order_by("pk")
    )


# --------------------------------------------------------------------------- #
# Deleting it
# --------------------------------------------------------------------------- #


@dataclass
class PurgeResult:
    accounts: int = 0
    data_sources: int = 0
    data_preps: int = 0
    datasets: int = 0
    training_runs: int = 0
    tokens: int = 0
    agent_actions: int = 0
    lines: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return (
            self.accounts + self.data_sources + self.data_preps + self.datasets + self.training_runs
        )


def _day(moment: datetime | None) -> str:
    return moment.date().isoformat() if moment else "never"


def purge(
    unused_uploads_days: int,
    inactive_accounts_days: int,
    dry_run: bool = False,
    log: Callable[[str], None] | None = None,
    now: datetime | None = None,
    agent_log_days: int | None = None,
) -> PurgeResult:
    """Delete old data (0 days: that kind is kept). Logs a line per deleted object, naming
    users by id only. With ``dry_run``, nothing is deleted. ``agent_log_days``: how long the
    activity log of agents is kept (default BRICKDNN_AGENT_LOG_DAYS; 0: the log is off, all
    of it is deleted)."""
    from . import agent_activity
    from .accounts import delete_account

    now = now or timezone.now()
    result = PurgeResult()
    verb = "Would delete" if dry_run else "Deleting"

    def say(line: str) -> None:
        result.lines.append(line)
        if log is not None:
            log(line)

    if inactive_accounts_days > 0:
        cutoff = now - timedelta(days=inactive_accounts_days)
        for user in inactive_users(cutoff):
            say(f"{verb} account #{user.pk}: last active {_day(user.last_seen)}")
            result.accounts += 1
            if not dry_run:
                delete_account(user, stop_trainings=False)

    if unused_uploads_days > 0:
        cutoff = now - timedelta(days=unused_uploads_days)
        for source in unused_sources(cutoff):
            preps = list(source.preps.order_by("pk"))
            names = "".join(f", with its preparation #{prep.pk}" for prep in preps)
            say(
                f"{verb} data source #{source.pk} of user #{source.owner_id}{names}: "
                f"last used {_day(source.last_used_at)}"
            )
            result.data_sources += 1
            result.data_preps += len(preps)
            if not dry_run:
                for prep in preps:
                    prep.delete()
                source.delete()  # the files are removed by a post_delete signal
        for dataset in unused_datasets(cutoff):
            say(
                f"{verb} dataset #{dataset.pk} of user #{dataset.owner_id}: "
                f"last used {_day(dataset.last_used_at)}"
            )
            result.datasets += 1
            if not dry_run:
                dataset.delete()
        for run in unused_runs(cutoff):
            say(
                f"{verb} training run {run.pk} of user #{run.owner_id}: "
                f"last used {_day(run.last_used_at)}"
            )
            result.training_runs += 1
            if not dry_run:
                run.delete()

    if agent_log_days is None:
        agent_log_days = settings.AGENT_LOG_DAYS
    result.agent_actions = agent_activity.purge(now, agent_log_days, dry_run)
    if result.agent_actions:
        kept = f"older than {agent_log_days} days" if agent_log_days > 0 else "(the log is off)"
        say(f"{verb} {result.agent_actions} entries of the agent activity log {kept}")

    # Housekeeping that needs no setting: logins that have expired anyway.
    result.tokens = _expired_tokens(now, dry_run)
    return result


def _expired_tokens(now: datetime, dry_run: bool) -> int:
    """Expired refresh tokens (and their blacklist entries) and unused OAuth codes."""
    from rest_framework_simplejwt.token_blacklist.models import OutstandingToken

    from .models import OAuthCode

    tokens = OutstandingToken.objects.filter(expires_at__lte=now)
    codes = OAuthCode.objects.filter(expires_at__lte=now)
    count = tokens.count() + codes.count()
    if not dry_run:
        tokens.delete()
        codes.delete()
    return count
