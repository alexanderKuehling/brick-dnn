"""Plans, unlimited accounts and overage (see budgets.py).

The plans are brickdnn.com's, read from deploy/sites/brickdnn.com/settings.env, so the tests
also check that file. The controlled fake runner and the helpers are those of test_budgets.
"""

# ruff: noqa: F811 (fixtures are imported from test_budgets and test_training_api)

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pytest
from django.contrib.auth import get_user_model
from django.core.exceptions import ImproperlyConfigured
from django.core.management import CommandError, call_command
from rest_framework.test import APIClient

from brick_dnn_core import budgets
from brick_dnn_core.admin import UserProfileAdmin
from brick_dnn_core.models import TrainingRun, UsageRecord, UserProfile
from brickdnn.training.state import read_job

from .test_budgets import (  # noqa: F401 (fixtures)
    LONG,
    QUICK,
    ada,
    controlled,
    ledger,
    limits,
    run_dir,
    spend,
    start,
    started,
    token_client,
    use_cpu,
    wait_for,
    web,
)
from .test_training_api import training_dir, wait_for_state  # noqa: F401

pytestmark = pytest.mark.django_db

User = get_user_model()
SITE_ENV = (
    Path(__file__).resolve().parents[2] / "deploy" / "sites" / "brickdnn.com" / "settings.env"
)


def site_setting(name: str) -> str:
    for line in SITE_ENV.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{name}="):
            return line.split("=", 1)[1].strip()
    raise AssertionError(f"{name} is not in {SITE_ENV}")


@pytest.fixture
def site(limits):
    """The plans of brickdnn.com."""
    limits.PLANS = json.loads(site_setting("BRICKDNN_PLANS"))
    limits.DEFAULT_PLAN = site_setting("BRICKDNN_DEFAULT_PLAN")
    limits.TRAINING_MAX_RUNS = 5
    return limits


def test_the_plans_of_brickdnn_com(site):
    plans = budgets.plans()
    assert list(plans) == ["free", "starter", "plus", "pro"]
    free, starter, plus, pro = plans.values()
    assert (free.price, free.cpu_hours_per_month, free.storage_gb) == (0, 0.5, 1)
    assert (free.cpu_minutes_per_run, free.concurrent_trainings, free.priority) == (15, 1, 0)
    assert free.overage_price_per_cpu_hour is None
    assert (starter.price, starter.cpu_hours_per_month, starter.storage_gb) == (5, 10, 5)
    assert (starter.cpu_minutes_per_run, starter.priority) == (60, 1)
    assert (plus.price, plus.cpu_hours_per_month, plus.storage_gb, plus.cpu_minutes_per_run) == (
        14,
        30,
        20,
        180,
    )
    assert (pro.price, pro.cpu_hours_per_month, pro.storage_gb, pro.cpu_minutes_per_run) == (
        45,
        120,
        100,
        480,
    )
    assert pro.concurrent_trainings == 2
    assert {p.overage_price_per_cpu_hour for p in (starter, plus, pro)} == {0.4}
    assert free.priority < starter.priority < plus.priority < pro.priority
    assert budgets.default_plan().key == "free"


def test_bad_plans_are_reported(limits):
    limits.PLANS = {"free": {"cpu_hours_per_month": "lots"}}
    with pytest.raises(ImproperlyConfigured, match="cpu_hours_per_month of plan 'free'"):
        budgets.plans()
    limits.PLANS = {"free": {"cpu_hours": 1}}
    with pytest.raises(ImproperlyConfigured, match="unknown setting 'cpu_hours'"):
        budgets.plans()
    limits.PLANS = ["free"]
    with pytest.raises(ImproperlyConfigured, match="object of plans"):
        budgets.plans()


def test_self_hosted_servers_have_no_limits(web, ada, limits, controlled):
    plan = budgets.plan_of(ada)
    assert (plan.key, plan.unlimited) == ("default", True)
    assert budgets.limits_for(ada) == budgets.Limits(
        None, None, None, None, plan=plan, concurrent_trainings=None
    )
    spend(ada, 10**6)
    run_id = started(web)
    assert read_job(run_dir(controlled, run_id))["limits"] is None
    data = web.get("/api/account/usage/").data
    assert data["plan"]["key"] == "default" and data["plans"] == []
    assert data["compute"]["limit"] is None and data["storage"]["limit_bytes"] is None
    assert data["overage"]["available"] is False


def test_plan_limits_apply(web, ada, site, controlled):
    limits = budgets.limits_for(ada)
    assert limits.plan.key == "free" and not limits.unlimited
    assert (limits.monthly_seconds, limits.run_seconds) == (1800, 900)
    assert limits.storage_bytes == 1024**3 and limits.concurrent_trainings == 1

    first = started(web)
    job = read_job(run_dir(controlled, first))["limits"]
    assert job["cpu_seconds"] == 900 and job["reason"] == "run_limit"
    assert "the most one training may use with the Free plan" in job["message"]
    # One training at a time.
    response = start(web)
    assert response.status_code == 409
    assert response.data["detail"].startswith("Your plan (Free) allows 1 training at a time")

    data = web.get("/api/account/usage/").data
    assert data["plan"]["name"] == "Free" and data["plan"]["price"] == 0
    assert data["plan"]["currency"] == "EUR"
    assert data["compute"]["allowance"] == 30 and data["compute"]["run_limit"] == 15
    assert [p["key"] for p in data["plans"]] == ["free", "starter", "plus", "pro"]


def test_a_plan_change_takes_effect_at_once(web, ada, site, controlled):
    spend(ada, 1800)
    response = start(web)
    assert response.status_code == 403
    assert "allowance of this account (Free plan) is used up" in response.data["detail"]
    assert "Higher plans include more" in response.data["detail"]

    assert budgets.set_plan(ada, "starter").name == "Starter"
    run_id = started(web)
    assert read_job(run_dir(controlled, run_id))["limits"]["cpu_seconds"] == 3600
    wait_for_state(controlled, run_id)
    # Back to Free while it runs: its limit per training applies on the next look.
    budgets.set_plan(ada, "free")
    data = use_cpu(web, controlled, run_id, 901.0)
    assert data["stop_requested"] is True
    data = wait_for(web, run_id)
    assert data["usage"]["stop_reason"] == "run_limit"
    with pytest.raises(KeyError):
        budgets.set_plan(ada, "gold")


def test_unlimited_accounts_are_measured_but_never_limited(ada, site, controlled):
    budgets.set_unlimited(ada, True)
    spend(ada, 100 * 3600)
    web = APIClient()
    web.force_authenticate(ada)
    first = started(web, config=LONG)  # far beyond any plan
    assert read_job(run_dir(controlled, first))["limits"] is None
    second = started(web)  # more than one at a time
    wait_for_state(controlled, first)
    data = use_cpu(web, controlled, first, 5000.0)
    assert data["stop_requested"] is False
    assert ledger(ada) == 100 * 3600 + 5000
    usage = web.get("/api/account/usage/").data
    assert usage["unlimited"] is True and usage["compute"]["used"] > 6000
    assert usage["compute"]["limit"] is None
    web.post(f"/api/training/runs/{second}/stop/")

    # The budgets the user set themselves still apply: for an agent ...
    agent, token = token_client(ada, "agent")
    token.budget_cpu_minutes = 1
    token.save()
    spend(ada, 61, api_token=token, source="token", via="agent")
    response = start(agent)
    assert response.status_code == 403
    assert "API token “agent” is used up" in response.data["detail"]
    # ... and for the account.
    UserProfile.objects.filter(user=ada).update(budget_cpu_minutes=60)
    response = start(web)
    assert response.status_code == 403
    assert "budget set for this account is used up" in response.data["detail"]


def test_staff_accounts_are_unlimited_by_default(site):
    staff = User.objects.create_user("root", password="x", is_staff=True)
    boss = User.objects.create_superuser("boss", password="x")
    member = User.objects.create_user("member", password="x")
    assert budgets.is_unlimited(staff) and budgets.is_unlimited(boss)
    assert not budgets.is_unlimited(member)
    budgets.set_unlimited(staff, False)  # back to the limits of the plan
    assert not budgets.is_unlimited(staff)
    assert budgets.limits_for(staff).monthly_seconds == 1800
    budgets.set_unlimited(staff, None)
    assert budgets.is_unlimited(staff)
    site.STAFF_UNLIMITED = False
    assert not budgets.is_unlimited(boss)
    budgets.set_unlimited(member, True)
    assert budgets.is_unlimited(member)


def test_set_plan_command(ada, site):
    out = StringIO()
    call_command("set_plan", "--list", stdout=out)
    lines = out.getvalue().splitlines()
    assert lines[0].startswith("free: Free, 0 per month; 0.5 CPU hours/month, 1 GB")
    assert lines[0].endswith("(default)") and "overage 0.4 per CPU hour" in lines[1]

    out = StringIO()
    call_command("set_plan", "ada", "--plan", "plus", stdout=out)
    assert "ada: plan Plus (plus)" in out.getvalue()
    assert "compute per month:    30 CPU hours" in out.getvalue()
    assert budgets.plan_of(ada).key == "plus"

    out = StringIO()
    ada.email = "ada@example.com"
    ada.save()
    call_command("set_plan", "ada@example.com", "--unlimited", stdout=out)
    assert "unlimited: no limits of the plan apply" in out.getvalue()
    assert budgets.is_unlimited(ada)
    call_command("set_plan", "ada", "--limited", "--plan", "default", stdout=StringIO())
    assert not budgets.is_unlimited(ada) and budgets.plan_of(ada).key == "free"
    assert UserProfile.objects.get(user=ada).plan == ""

    with pytest.raises(CommandError, match="Unknown plan 'gold'"):
        call_command("set_plan", "ada", "--plan", "gold")
    with pytest.raises(CommandError, match="No account 'nobody'"):
        call_command("set_plan", "nobody")


def test_admin_actions(ada, site, rf):
    other = User.objects.create_user("bob", password="x")
    for user in (ada, other):
        UserProfile.objects.get_or_create(user=user)
    model_admin = UserProfileAdmin(UserProfile, None)
    model_admin.message_user = lambda *args, **kwargs: None
    queryset = UserProfile.objects.all()
    model_admin.make_unlimited(rf.get("/"), queryset)
    assert budgets.is_unlimited(ada) and budgets.is_unlimited(other)
    model_admin.back_to_plan(rf.get("/"), queryset.filter(user=other))
    assert not budgets.is_unlimited(other)
    profile = UserProfile.objects.get(user=ada)
    assert model_admin.is_unlimited(profile) is True
    assert model_admin.plan_name(profile) == "Free"


def test_overage(web, ada, site, controlled):
    budgets.set_plan(ada, "starter")
    spend(ada, 10 * 3600)
    assert start(web).status_code == 403  # the allowance is used up, overage is off

    # The Free plan has none.
    budgets.set_plan(ada, "free")
    response = web.patch("/api/account/usage/", {"overage_enabled": True}, format="json")
    assert response.status_code == 400
    budgets.set_plan(ada, "starter")
    # An agent cannot switch it on.
    agent, _ = token_client(ada)
    response = agent.patch("/api/account/usage/", {"overage_enabled": True}, format="json")
    assert response.status_code == 403

    response = web.patch(
        "/api/account/usage/",
        {"overage_enabled": True, "overage_cap_cpu_hours": 0.5},
        format="json",
    )
    assert response.status_code == 200, response.data
    overage = response.data["overage"]
    assert overage["available"] and overage["enabled"] and overage["cap_cpu_hours"] == 0.5
    assert overage["price_per_cpu_hour"] == 0.4 and overage["currency"] == "EUR"
    assert response.data["compute"]["limit"] == 630  # 10 h + 0.5 h, in minutes

    run_id = started(web)
    assert read_job(run_dir(controlled, run_id))["limits"]["cpu_seconds"] == 1800
    wait_for_state(controlled, run_id)
    use_cpu(web, controlled, run_id, 100.0)
    record = UsageRecord.objects.get(run_id=run_id)
    assert (record.quantity, record.overage) == (100, 100)
    data = web.get("/api/account/usage/").data
    assert data["overage"]["used_cpu_minutes"] == pytest.approx(1.67, abs=0.01)
    assert data["overage"]["cost"] == pytest.approx(100 / 3600 * 0.4, abs=0.0001)
    # Its cap is a limit like the allowance.
    data = use_cpu(web, controlled, run_id, 1800.0)
    data = wait_for(web, run_id)
    assert data["usage"]["stop_reason"] == "budget"

    # Switched off again: the allowance applies.
    web.patch("/api/account/usage/", {"overage_enabled": False}, format="json")
    assert not budgets.limits_for(ada).overage
    response = start(web)
    assert response.status_code == 403


def test_overage_is_only_the_part_beyond_the_allowance(ada, site):
    budgets.set_plan(ada, "starter")
    UserProfile.objects.filter(user=ada).update(overage_enabled=True)
    spend(ada, 10 * 3600 - 30)
    assert budgets.overage_part(ada, 100) == 70
    assert budgets.overage_part(ada, 20) == 0
    UserProfile.objects.filter(user=ada).update(overage_enabled=False)
    assert budgets.overage_part(ada, 100) == 0
    assert TrainingRun.objects.count() == 0
