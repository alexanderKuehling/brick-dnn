"""Warning inactive accounts by e-mail before ``purge_old_data`` deletes them."""

from __future__ import annotations

from datetime import timedelta

import pytest
from django.contrib.auth import get_user_model
from django.utils import timezone

from brick_dnn_core import retention
from brick_dnn_core.models import UserProfile

from .mail_helpers import SITE
from .test_account import PASSWORD, login
from .test_purge import make_inactive, purge_command

pytestmark = pytest.mark.django_db

User = get_user_model()
DAYS = 730


@pytest.fixture
def accounts(settings):
    settings.DELETE_INACTIVE_ACCOUNTS_DAYS = DAYS
    settings.INACTIVE_ACCOUNT_WARNING_DAYS = [14, 3]
    return {
        name: User.objects.create_user(name, email=f"{name}@example.com", password=PASSWORD)
        for name in ("ada", "bob")
    }


def purge_at(now):
    return retention.purge(0, DAYS, now=now)


def exists(name: str) -> bool:
    return User.objects.filter(username=name).exists()


def test_warnings_before_the_deletion(accounts, email_on):  # noqa: F811
    ada = accounts["ada"]
    make_inactive(ada, timedelta(days=DAYS - 20))
    now = timezone.now()
    # 20 days before: nothing yet.
    assert purge_at(now).warned == 0 and email_on == []
    # 14 days before: the first warning, with a link to log in.
    result = purge_at(now + timedelta(days=6))
    assert result.warned == 1 and len(email_on) == 1
    message = email_on[0]
    assert message.to == ["ada@example.com"]
    assert message.subject == "Your account will be deleted – Brick DNN"
    assert f"{SITE}/login" in message.body and "730 days" in message.body
    # The next runs do not repeat it ...
    assert purge_at(now + timedelta(days=7)).warned == 0
    # ... until 3 days before: the second warning.
    assert purge_at(now + timedelta(days=17)).warned == 1 and len(email_on) == 2
    assert purge_at(now + timedelta(days=18)).warned == 0
    assert exists("ada")
    # When it is due, it is deleted; Bob, who is active, stays.
    result = purge_at(now + timedelta(days=20, hours=1))
    assert result.accounts == 1 and not exists("ada") and exists("bob")


def test_overdue_accounts_are_warned_first(accounts, email_on):  # noqa: F811
    """E-mail switched on while accounts are overdue: nobody is deleted without notice."""
    make_inactive(accounts["ada"], timedelta(days=DAYS + 100))
    now = timezone.now()
    result = purge_at(now)
    assert result.warned == 1 and result.accounts == 0 and result.not_warned_yet == 1
    assert exists("ada")
    expected = (now + timedelta(days=3)).strftime("%d %B %Y")
    assert expected in email_on[0].body
    assert purge_at(now + timedelta(days=2)).accounts == 0
    assert purge_at(now + timedelta(days=3, hours=1)).accounts == 1
    assert not exists("ada")


def test_logging_in_keeps_the_account_and_starts_over(accounts, email_on):  # noqa: F811
    ada = accounts["ada"]
    make_inactive(ada, timedelta(days=DAYS - 10))
    now = timezone.now()
    assert purge_at(now).warned == 1
    login("ada")  # the account is used again
    assert purge_at(now + timedelta(days=11)).accounts == 0
    assert exists("ada")
    # Inactive again for the whole period after that login: a warning of its own first.
    later = now + timedelta(days=DAYS + 1)
    assert purge_at(later).accounts == 0  # Bob, inactive as long, is warned now too
    assert [m.to[0] for m in email_on].count("ada@example.com") == 2


def test_failed_warnings_keep_the_account(accounts, email_on, monkeypatch):  # noqa: F811
    from brick_dnn_core import notices

    make_inactive(accounts["ada"], timedelta(days=DAYS + 1))
    monkeypatch.setattr(notices, "deletion_warning", lambda *args: False)
    result = purge_at(timezone.now())
    assert result.warned == 0 and result.accounts == 0
    assert any("Could not warn account" in line for line in result.lines)
    assert purge_at(timezone.now() + timedelta(days=10)).accounts == 0
    assert exists("ada")


def test_without_e_mail_accounts_are_deleted_as_before(accounts, settings):
    settings.EMAIL_ENABLED = False
    make_inactive(accounts["ada"], timedelta(days=DAYS + 1))
    result = purge_at(timezone.now())
    assert result.warned == 0 and result.accounts == 1 and not exists("ada")


def test_accounts_without_an_address_are_deleted_as_before(accounts, email_on):  # noqa: F811
    ada = accounts["ada"]
    User.objects.filter(pk=ada.pk).update(email="")
    make_inactive(ada, timedelta(days=DAYS + 1))
    result = purge_at(timezone.now())
    assert result.accounts == 1 and email_on == []


def test_unverified_addresses_are_warned_too(accounts, email_on):  # noqa: F811
    # Bob's address was never confirmed: a warning reveals nothing and may save the account.
    make_inactive(accounts["bob"], timedelta(days=DAYS + 1))
    assert purge_at(timezone.now()).warned == 1
    assert email_on[0].to == ["bob@example.com"]


def test_accounts_without_a_profile_keep_their_activity(accounts, email_on):  # noqa: F811
    ada = accounts["ada"]
    make_inactive(ada, timedelta(days=DAYS + 1))
    UserProfile.objects.filter(user=ada).delete()
    assert purge_at(timezone.now()).warned == 1
    profile = UserProfile.objects.get(user=ada)
    assert profile.deletion_warned_at is not None
    assert profile.last_active_at < timezone.now() - timedelta(days=DAYS)


def test_dry_run_and_command_output(accounts, email_on):  # noqa: F811
    make_inactive(accounts["ada"], timedelta(days=DAYS + 1))
    out = purge_command("--dry-run")
    assert "Would warn account #" in out and email_on == []
    assert "Would warn 1 inactive accounts" in out
    out = purge_command()
    assert "Warned account #" in out and len(email_on) == 1
    assert "kept 1 that were not warned in time yet" in out
    assert exists("ada")


def test_warning_days_fit_the_period(settings, email_on):  # noqa: F811
    settings.INACTIVE_ACCOUNT_WARNING_DAYS = [30, 14, 3]
    assert retention.warning_days(20) == [14, 3]
    settings.EMAIL_ENABLED = False
    assert retention.warning_days(20) == []
