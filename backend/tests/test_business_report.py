"""The business report (business_report.py): its numbers from a small history of accounts,
trainings, subscriptions and links, the command and the admin page.

The plans are brickdnn.com's; ``NOW`` is the moment the report is made. Subscriptions are
written the way billing.handle_event leaves them: a row of the event log per notification,
and the state of each customer.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from datetime import timezone as dt_timezone
from io import StringIO
from pathlib import Path

import pytest
from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.test import Client

from brick_dnn_core import billing, budgets, business_report
from brick_dnn_core.models import (
    BillingCustomer,
    BillingEvent,
    DataSource,
    Project,
    RunComparison,
    TrainingRun,
    TrainingWorker,
    UsageRecord,
)

from .fake_paddle import FakePaddle

pytestmark = pytest.mark.django_db

User = get_user_model()
SITE_ENV = (
    Path(__file__).resolve().parents[2] / "deploy" / "sites" / "brickdnn.com" / "settings.env"
)
NOW = datetime(2026, 11, 18, 12, 0, tzinfo=dt_timezone.utc)
GRAPH = {"version": 1, "nodes": [], "edges": []}


def at(month: int, day: int, hour: int = 10) -> datetime:
    return datetime(2026, month, day, hour, tzinfo=dt_timezone.utc)


def site_setting(name: str) -> str:
    for line in SITE_ENV.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{name}="):
            return line.split("=", 1)[1].strip()
    raise AssertionError(f"{name} is not in {SITE_ENV}")


@pytest.fixture
def site(settings, tmp_path):
    """brickdnn.com's plans, no paid plans yet, trainings by the web server itself."""
    settings.PLANS = json.loads(site_setting("BRICKDNN_PLANS"))
    settings.DEFAULT_PLAN = "free"
    settings.PRICE_CURRENCY = "EUR"
    settings.STAFF_UNLIMITED = True
    settings.TRAINING_ENABLED = True
    settings.TRAINING_WORKER = False
    settings.TRAINING_MAX_RUNS = 1
    settings.TRAINING_DIR = tmp_path / "training"
    settings.DATA_DIR = tmp_path
    return settings


@pytest.fixture
def selling(site, monkeypatch):
    site.PADDLE_ENV = "live"
    site.PADDLE_CLIENT_TOKEN = "live_0123456789abcdef"
    site.PADDLE_API_KEY = "pdl_live_apikey_secret"
    site.PADDLE_WEBHOOK_SECRETS = ["pdl_ntfset_secret"]
    site.PADDLE_PRICES = "starter=pri_starter,plus=pri_plus,pro=pri_pro"
    site.PADDLE_PRICES_YEARLY = "pro=pri_pro_year"
    site.PLANS = {**site.PLANS, "pro": {**site.PLANS["pro"], "price_per_year": 450}}
    site.PADDLE_OVERAGE_PRODUCT = ""
    monkeypatch.setattr(billing, "client", lambda cfg=None: FakePaddle())
    return site


def account(name: str, joined: datetime, **extra):
    user = User.objects.create_user(name, f"{name}@example.com", "correct-horse-battery", **extra)
    User.objects.filter(pk=user.pk).update(date_joined=joined)
    return user


def trained(user, moment: datetime, cpu_seconds: float = 600, kind: str = "training"):
    """A training started at ``moment``: the run and its row in the ledger."""
    run = TrainingRun.objects.create(owner=user, name="run", graph=GRAPH, kind=kind)
    TrainingRun.objects.filter(pk=run.pk).update(created_at=moment)
    if kind != "external":
        record = UsageRecord.objects.create(
            user=user, kind="training", quantity=cpu_seconds, period=budgets.period_of(moment)
        )
        UsageRecord.objects.filter(pk=record.pk).update(created_at=moment)
    return run


def subscription_event(user, moment: datetime, status: str, plan: str, price: str | None = None):
    """A notification about the account's subscription, applied; the customer follows."""
    count = BillingEvent.objects.count()
    price = price or f"pri_{plan}"
    BillingEvent.objects.create(
        event_id=f"evt_{count}",
        event_type="subscription.canceled" if status == "canceled" else "subscription.updated",
        occurred_at=moment,
        user=user,
        subscription_id=f"sub_{user.pk}",
        outcome="applied",
        status=status,
        plan=plan,
        price_id=price,
    )
    budgets.set_plan(user, plan if status == "active" else "")
    BillingCustomer.objects.update_or_create(
        user=user,
        defaults={
            "environment": "live",
            "customer_id": f"ctm_{user.pk}",
            "subscription_id": f"sub_{user.pk}",
            "status": status,
            "plan": plan,
            "price_id": price,
        },
    )


@pytest.fixture
def history(selling):
    """September to November 2026:

    - ada signs up on 3 September, trains the next day, buys Plus on 10 September and moves
      to Pro (paid per year) on 5 November;
    - ben signs up on 20 September, trains for the first time a month later (not activated),
      buys Starter on 2 October and ends it on 12 November;
    - cleo signs up on 14 October and trains on the 16th: activated, never pays;
    - dora signs up on 16 November, three days ago: still open;
    - the owner (staff) trains a lot, logs runs from a script and shares links.
    """
    owner = account("owner", at(8, 1), is_staff=True)
    ada = account("ada", at(9, 3))
    ben = account("ben", at(9, 20))
    cleo = account("cleo", at(10, 14))
    account("dora", at(11, 16))

    trained(ada, at(9, 4), 1200)
    trained(ada, at(11, 10), 3600)
    trained(ben, at(10, 21), 900)
    trained(cleo, at(10, 16), 1800)
    trained(owner, at(10, 2), 7200)
    trained(owner, at(11, 3), 0, kind="external")
    trained(owner, at(11, 4), 0, kind="external")

    subscription_event(ada, at(9, 10), "active", "plus")
    subscription_event(ada, at(11, 5), "active", "pro", "pri_pro_year")
    subscription_event(ben, at(10, 2), "active", "starter")
    subscription_event(ben, at(11, 12), "canceled", "starter")

    project = Project.objects.create(
        owner=owner, name="net", link_access="view", share_token="t1", link_shared_at=at(10, 5)
    )
    Project.objects.create(owner=owner, name="old", link_access="view", share_token="t2")
    comparison = RunComparison.objects.create(owner=owner, name="runs", token="c1")
    RunComparison.objects.filter(pk=comparison.pk).update(created_at=at(11, 6))
    DataSource.objects.create(
        owner=ada, name="data", filename="d.csv", size_bytes=3 * 1024**3, n_rows=1, n_columns=1
    )
    return {"ada": ada, "ben": ben, "cleo": cleo, "owner": owner, "project": project}


def months(data) -> dict[str, dict]:
    return {row["label"]: row for row in data["periods"]}


# --------------------------------------------------------------------------- #
# The numbers
# --------------------------------------------------------------------------- #


def test_signups_activation_and_trained_accounts(history):
    data = business_report.report("month", 3, NOW)
    assert [row["label"] for row in data["periods"]] == ["2026-09", "2026-10", "2026-11"]
    september, october, november = data["periods"]
    # Staff accounts are not counted as sign-ups.
    assert (september["signups"], october["signups"], november["signups"]) == (2, 1, 1)
    # ada trained the next day, ben only a month later.
    assert (september["activated"], september["activation_rate"]) == (1, 0.5)
    assert (october["activated"], october["activation_rate"]) == (1, 1.0)
    assert (november["activated"], november["activation_rate"]) == (0, 0.0)
    assert september["activation_open"] is False and november["activation_open"] is True
    assert (september["trained_accounts"], october["trained_accounts"]) == (1, 2)
    assert november["trained_accounts"] == 1
    assert (september["complete"], november["complete"]) == (True, False)


def test_activation_survives_a_deleted_run(history):
    TrainingRun.objects.filter(owner=history["ada"]).delete()
    data = business_report.report("month", 3, NOW)
    assert months(data)["2026-09"]["activated"] == 1  # the ledger still knows


def test_subscriptions_paying_accounts_and_mrr(history):
    data = business_report.report("month", 3, NOW)
    september, october, november = data["periods"]
    assert (september["new_subscriptions"], september["ended_subscriptions"]) == (1, 0)
    assert (october["new_subscriptions"], november["new_subscriptions"]) == (1, 0)
    assert november["ended_subscriptions"] == 1
    # At the end of each month: Plus; Plus and Starter; Pro per year (450 / 12).
    assert (september["paying_accounts"], september["mrr"]) == (1, 14.0)
    assert (october["paying_accounts"], october["mrr"]) == (2, 19.0)
    assert (november["paying_accounts"], november["mrr"]) == (1, 37.5)
    assert october["paying_by_plan"] == [
        {"plan": "starter", "name": "Starter", "accounts": 1, "yearly": 0, "mrr": 5.0},
        {"plan": "plus", "name": "Plus", "accounts": 1, "yearly": 0, "mrr": 14.0},
    ]
    # Both accounts of September have paid since, cleo (October) has not.
    assert (september["converted"], september["conversion_rate"]) == (2, 1.0)
    assert (october["converted"], october["conversion_rate"]) == (0, 0.0)

    now = data["now"]
    assert now["accounts"] == 4
    assert now["paying"]["accounts"] == 1 and now["paying"]["mrr"] == 37.5
    assert now["paying"]["by_plan"] == [
        {"plan": "pro", "name": "Pro", "accounts": 1, "yearly": 1, "mrr": 37.5}
    ]
    assert now["conversion_rate"] == 0.25
    assert (now["paying"]["ending"], now["paying"]["past_due"]) == (0, 0)


def test_subscriptions_that_end_soon_or_failed_to_pay_are_named(history):
    BillingCustomer.objects.filter(user=history["ada"]).update(cancel_at=NOW + timedelta(days=9))
    BillingCustomer.objects.filter(user=history["ben"]).update(status="past_due")
    paying = business_report.report("month", 1, NOW)["now"]["paying"]
    assert (paying["accounts"], paying["ending"], paying["past_due"]) == (1, 1, 1)


def test_sandbox_subscriptions_from_before_going_live_do_not_count(history):
    subscription_event(history["cleo"], at(10, 20), "active", "plus", "pri_sandbox_plus")
    data = business_report.report("month", 3, NOW)
    assert months(data)["2026-10"]["paying_accounts"] == 2
    assert months(data)["2026-10"]["new_subscriptions"] == 1


def test_compute_against_capacity(history, site, monkeypatch):
    monkeypatch.setattr(budgets, "fallback_threads", lambda: 4)
    data = business_report.report("month", 3, NOW)
    september, october, november = data["periods"]
    # ada 20 min; ben 15, cleo 30 and the owner 120 min; ada 60 min.
    assert (september["cpu_hours_used"], october["cpu_hours_used"]) == (0.33, 2.75)
    assert november["cpu_hours_used"] == 1.0
    # One training at a time with 4 threads, around the clock (November: up to now).
    assert september["cpu_hours_capacity"] == 30 * 24 * 4
    assert november["cpu_hours_capacity"] == (17 * 24 + 12) * 4
    assert october["utilisation"] == round(2.75 / (31 * 24 * 4), 4)
    assert data["now"]["capacity_source"] == "trainings at a time × CPU threads of this server"


def test_capacity_with_training_workers(history, site):
    site.TRAINING_WORKER = True
    site.WORKER_TIMEOUT = 60
    # Two slots with two threads each for the first ten days of October, then gone.
    TrainingWorker.objects.create(
        name="w1",
        slots=2,
        threads=2,
        started_at=at(10, 1, 0),
        heartbeat_at=at(10, 11, 0),
        stopped_at=at(10, 11, 0),
    )
    data = business_report.report("month", 3, NOW)
    assert months(data)["2026-10"]["cpu_hours_capacity"] == 10 * 24 * 4
    assert months(data)["2026-11"]["cpu_hours_capacity"] == 0
    assert months(data)["2026-11"]["utilisation"] is None
    assert "training workers" in data["now"]["capacity_source"]
    # Before the rows of workers are kept (90 days), the capacity is not known.
    old = business_report.report("month", 8, NOW)
    assert months(old)["2026-04"]["cpu_hours_capacity"] is None


def test_storage_logged_runs_and_shared_links(history, monkeypatch):
    class Disk:
        total, used, free = 80 * 1024**3, 30 * 1024**3, 50 * 1024**3

    monkeypatch.setattr(business_report.shutil, "disk_usage", lambda path: Disk)
    data = business_report.report("month", 3, NOW)
    assert data["now"]["storage"] == {
        "used_bytes": 3 * 1024**3,
        "disk_total_bytes": 80 * 1024**3,
        "disk_used_bytes": 30 * 1024**3,
        "disk_free_bytes": 50 * 1024**3,
    }
    assert [row["external_runs"] for row in data["periods"]] == [0, 0, 2]
    # A network link in October, a comparison in November; the undated link only counts now.
    assert [row["shared_links"] for row in data["periods"]] == [0, 1, 1]
    assert data["now"]["shared_links"] == {"networks": 2, "comparisons": 1}


def test_accounts_at_their_allowance(history):
    assert business_report.report("month", 1, NOW)["now"]["at_allowance"] == 0
    UsageRecord.objects.create(
        user=history["cleo"], kind="training", quantity=1800, period=budgets.period_of(NOW)
    )
    assert business_report.report("month", 1, NOW)["now"]["at_allowance"] == 1


def test_days_and_weeks(history):
    days = business_report.report("day", 3, NOW)
    assert [row["label"] for row in days["periods"]] == ["2026-11-16", "2026-11-17", "2026-11-18"]
    assert [row["signups"] for row in days["periods"]] == [1, 0, 0]
    weeks = business_report.report("week", 2, NOW)
    # Weeks start on Monday: 9 and 16 November 2026.
    assert [row["label"] for row in weeks["periods"]] == ["2026-W46", "2026-W47"]
    assert [row["start"][:10] for row in weeks["periods"]] == ["2026-11-09", "2026-11-16"]
    assert [row["ended_subscriptions"] for row in weeks["periods"]] == [1, 0]
    assert len(business_report.report("day", None, NOW)["periods"]) == 14
    assert len(business_report.report("month", 999, NOW)["periods"]) == 60
    with pytest.raises(ValueError, match="day, week, month"):
        business_report.report("year")


def test_without_paid_plans_there_are_no_subscription_numbers(site):
    ada = account("ada", at(11, 2))
    trained(ada, at(11, 3))
    data = business_report.report("month", 1, NOW)
    row = data["periods"][0]
    assert data["billing"] is False and data["now"]["paying"] is None
    assert (row["signups"], row["activated"], row["trained_accounts"]) == (1, 1, 1)
    for key in ("new_subscriptions", "ended_subscriptions", "paying_accounts", "mrr", "converted"):
        assert row[key] is None
    text = business_report.text(data)
    assert "MRR" not in text.split("Paid plans are off")[0]
    assert "Paid plans are off on this server: no subscription numbers." in text


# --------------------------------------------------------------------------- #
# The command
# --------------------------------------------------------------------------- #


def test_the_command_prints_text_and_json(history):
    out = StringIO()
    call_command("business_report", "--period", "month", "--count", "2", stdout=out)
    text = out.getvalue()
    assert text.startswith("Brick DNN business report, ")
    assert "Accounts:" in text and "Estimated MRR:" in text and "37.50 EUR" in text
    assert "Paying accounts:" in text and "Pro 1 (1 yearly)" in text
    heading = next(line for line in text.splitlines() if line.startswith("Period"))
    for column in (
        "Sign-ups",
        "Activated",
        "Trained",
        "New subs.",
        "Ended subs.",
        "Paying",
        "MRR (EUR)",
        "Converted",
        "CPU h used",
        "CPU h capacity",
        "Runs logged",
        "Links shared",
    ):
        assert column in heading
    assert "(running)" in text and "- MRR is an estimate" in text
    assert not any(line != line.rstrip() for line in text.splitlines())

    out = StringIO()
    call_command("business_report", "--json", "--period", "week", stdout=out)
    data = json.loads(out.getvalue())
    assert data["period"] == "week" and len(data["periods"]) == 8
    assert data["currency"] == "EUR" and data["billing"] is True
    assert set(data["now"]) >= {"accounts", "paying", "storage", "shared_links", "at_allowance"}


def test_the_report_names_no_account(history):
    """Aggregates only: no user name, address or id is in what the command prints."""
    out = StringIO()
    call_command("business_report", "--json", stdout=out)
    text = out.getvalue()
    for user in User.objects.all():
        assert user.username not in text and user.email not in text


# --------------------------------------------------------------------------- #
# The admin page
# --------------------------------------------------------------------------- #

URL = "/admin/business-report/"


def test_the_admin_page_is_for_staff_only(history):
    anonymous = Client()
    response = anonymous.get(URL)
    assert response.status_code == 302 and response["Location"].startswith("/admin/login/")

    user = Client()
    user.force_login(history["ada"])
    response = user.get(URL)
    assert response.status_code == 302 and response["Location"].startswith("/admin/login/")

    staff = Client()
    staff.force_login(history["owner"])
    response = staff.get(URL)
    assert response.status_code == 200
    page = response.content.decode()
    assert "Business report" in page and "Estimated MRR" in page and "37.50 EUR" in page
    assert "Sign-ups" in page and "By month" in page and 'href="?period=week"' in page
    assert "ada" not in page.replace("metadata", "")  # no account is named
    # Other periods, and nonsense falls back to months.
    assert "2026-W" in staff.get(URL, {"period": "week"}).content.decode()
    assert staff.get(URL, {"period": "nonsense"}).status_code == 200
    # Linked from the start page of the admin.
    assert 'href="/admin/business-report/"' in staff.get("/admin/").content.decode()


def test_the_admin_page_needs_two_factor_authentication_where_the_admin_does(history, settings):
    settings.REQUIRE_2FA_FOR_STAFF = True
    staff = Client()
    staff.force_login(history["owner"])
    response = staff.get(URL)
    assert response.status_code == 302 and response["Location"].startswith("/admin/login/")
