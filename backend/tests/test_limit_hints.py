"""Where a limit leads: every refusal names its limit with a code and, on a server that sells
plans, the next plan that lifts it (budgets.next_plan, budgets.limit_data).

The plans are brickdnn.com's (deploy/sites/brickdnn.com/settings.env): Free, Starter, Plus and
Pro. ``selling`` switches paid plans on, ``site`` alone is the same server without Paddle. The
controlled fake runner and the helpers are those of test_budgets.
"""

# ruff: noqa: F811 (fixtures are imported from other test modules)

from __future__ import annotations

import pytest
from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from rest_framework.test import APIClient

from brick_dnn_core import billing, budgets
from brick_dnn_core.api_tokens import create_token
from brick_dnn_core.models import BillingCustomer, DataSource, UserProfile

from .fake_paddle import FakePaddle
from .test_budgets import (  # noqa: F401 (fixtures)
    LONG,
    QUICK,
    ada,
    controlled,
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
from .test_mcp import call
from .test_plans import site  # noqa: F401 (fixture)
from .test_training_api import (  # noqa: F401 (fixtures)
    DEMO_GRAPH,
    RANDOM,
    iris_csv,
    training_dir,
    upload,
    wait_for_state,
)

pytestmark = pytest.mark.django_db

User = get_user_model()
GB = 1024**3
STARTER = {
    "key": "starter",
    "name": "Starter",
    "price": 5.0,
    "price_per_year": None,
    "currency": "EUR",
    "cpu_hours_per_month": 10.0,
    "storage_gb": 3.0,
    "cpu_minutes_per_run": 60.0,
    "concurrent_trainings": 1,
    "queued_trainings": None,
    "priority": 1,
    "overage_price_per_cpu_hour": 0.4,
    "features": [],
    "url": "https://brickdnn.com/pricing?plan=starter",
}


@pytest.fixture
def selling(site, monkeypatch):
    """brickdnn.com with paid plans switched on (a fake Paddle behind it)."""
    site.PUBLIC_URL = "https://brickdnn.com"
    site.PADDLE_ENV = "sandbox"
    site.PADDLE_CLIENT_TOKEN = "test_0123456789abcdef"
    site.PADDLE_API_KEY = "pdl_sdbx_apikey_secret"
    site.PADDLE_WEBHOOK_SECRETS = ["pdl_ntfset_test_secret"]
    site.PADDLE_PRICES = "starter=pri_starter,plus=pri_plus,pro=pri_pro"
    site.PADDLE_PRICES_YEARLY = ""
    site.PADDLE_OVERAGE_PRODUCT = "pro_overage"
    fake = FakePaddle()
    monkeypatch.setattr(billing, "client", lambda cfg=None: fake)
    return site


def subscribed(user, plan: str) -> None:
    """The account pays for ``plan`` (as Paddle's notification would leave it)."""
    budgets.set_plan(user, plan)
    BillingCustomer.objects.create(
        user=user,
        environment="sandbox",
        customer_id=f"ctm_{user.pk}",
        subscription_id=f"sub_{user.pk}",
        status="active",
        plan=plan,
        price_id=f"pri_{plan}",
        granted_plan=plan,
    )


# --------------------------------------------------------------------------- #
# The next plan
# --------------------------------------------------------------------------- #


def test_the_next_plan_is_the_cheapest_that_lifts_the_limit(selling, ada):
    assert budgets.next_plan(ada, "cpu_hours_per_month") == STARTER
    assert budgets.next_plan(ada, "storage_gb")["key"] == "starter"
    assert budgets.next_plan(ada, "cpu_minutes_per_run")["key"] == "starter"
    # Only Pro trains two at a time.
    assert budgets.next_plan(ada, "concurrent_trainings")["key"] == "pro"
    # Every plan lets three trainings wait: no plan lifts that.
    assert budgets.next_plan(ada, "queued_trainings") is None
    # What is needed decides: 20 CPU hours need Plus, 100 minutes per training too.
    assert budgets.next_plan(ada, "cpu_hours_per_month", {"cpu_hours_per_month": 20})["key"] == (
        "plus"
    )
    assert budgets.next_plan(ada, "cpu_minutes_per_run", {"cpu_minutes_per_run": 100})["key"] == (
        "plus"
    )
    assert budgets.next_plan(ada, "storage_gb", {"storage_gb": 80}) is None

    budgets.set_plan(ada, "plus")
    assert budgets.next_plan(ada, "cpu_hours_per_month")["key"] == "pro"
    budgets.set_plan(ada, "pro")
    assert budgets.next_plan(ada, "cpu_hours_per_month") is None
    # Never a cheaper plan, even if it allowed more of something.
    selling.PLANS = {**selling.PLANS, "starter": {**selling.PLANS["starter"], "storage_gb": 99}}
    assert budgets.next_plan(ada, "storage_gb") is None


def test_no_plan_is_suggested_without_paid_plans_or_limits(site, ada, selling):
    budgets.set_unlimited(ada, True)
    assert budgets.next_plan(ada, "cpu_hours_per_month") is None
    budgets.set_unlimited(ada, False)
    assert budgets.next_plan(ada, "cpu_hours_per_month") is not None
    # A limit the operator raised for the account counts: Starter's 3 GB are no more.
    UserProfile.objects.filter(user=ada).update(storage_mb=4 * 1024)
    assert budgets.next_plan(ada, "storage_gb")["key"] == "plus"
    selling.PADDLE_ENV = ""  # half configured: billing is off
    assert budgets.next_plan(ada, "cpu_hours_per_month") is None


def test_a_plan_that_is_not_on_sale_is_never_suggested(selling, ada):
    selling.PADDLE_PRICES = "plus=pri_plus,pro=pri_pro"
    assert budgets.next_plan(ada, "cpu_hours_per_month")["key"] == "plus"


def test_the_usage_overview_names_the_next_plans(selling, web, ada):
    data = web.get("/api/account/usage/").data
    assert data["next_plans"]["compute"] == STARTER
    assert data["next_plans"]["storage"]["key"] == "starter"
    selling.PADDLE_ENV = ""
    assert web.get("/api/account/usage/").data["next_plans"] == {"compute": None, "storage": None}


# --------------------------------------------------------------------------- #
# CPU time
# --------------------------------------------------------------------------- #


def test_a_used_up_allowance_names_the_next_plan(selling, web, ada, controlled):
    spend(ada, 1800)
    response = start(web)
    assert response.status_code == 403
    data = response.data
    assert data["code"] == "budget_exceeded"
    assert data["limit"] == {
        "kind": "cpu_time",
        "used": 30.0,
        "limit": 30.0,
        "unit": "cpu_minutes",
        "resets_at": data["resets_at"],
    }
    assert data["plan"] == {"key": "free", "name": "Free"}
    assert data["upgrade"] == STARTER
    assert data["overage"] is None  # the free plan has none
    assert data["upgrade_url"] == "https://brickdnn.com/pricing"
    assert "The Starter plan (5.00 EUR per month) includes 10 CPU hours per month" in data["detail"]
    # The check before the start says the same, without an error.
    body = {"graph": DEMO_GRAPH, "dataset": RANDOM, "config": QUICK}
    budget = web.post("/api/training/check/", body, format="json").data["budget"]
    assert (budget["allowed"], budget["code"]) == (False, "budget_exceeded")
    assert budget["limit"]["kind"] == "cpu_time" and budget["upgrade"] == STARTER


def test_the_same_limit_without_paid_plans(site, web, ada, controlled):
    spend(ada, 1800)
    data = start(web).data
    assert data["code"] == "budget_exceeded" and data["limit"]["kind"] == "cpu_time"
    assert data["plan"] == {"key": "free", "name": "Free"}
    assert data["upgrade"] is None and data["overage"] is None
    assert "upgrade_url" not in data
    assert "Higher plans include more (see your account page)." in data["detail"]
    assert "pricing" not in data["detail"]


def test_a_paid_plan_that_ran_out_offers_overage_and_the_next_plan(selling, web, ada, controlled):
    subscribed(ada, "starter")
    spend(ada, 10 * 3600)
    data = start(web).data
    assert data["limit"]["kind"] == "cpu_time" and data["plan"]["key"] == "starter"
    assert data["upgrade"]["key"] == "plus"
    assert data["overage"] == {"price_per_cpu_hour": 0.4, "currency": "EUR"}

    # The highest plan: no plan is left, compute beyond the allowance is the way on.
    BillingCustomer.objects.filter(user=ada).update(plan="pro", price_id="pri_pro")
    budgets.set_plan(ada, "pro")
    spend(ada, 110 * 3600)
    data = start(web).data
    assert data["upgrade"] is None and data["overage"]["price_per_cpu_hour"] == 0.4
    assert "allow compute beyond the allowance on your account page, at 0.40 EUR" in data["detail"]

    # Switched on with a cap that is used up too: the cap is the limit, raised by the user.
    UserProfile.objects.filter(user=ada).update(overage_enabled=True, overage_cap_cpu_hours=0)
    data = start(web).data
    assert data["limit"]["kind"] == "overage_cap"
    assert data["upgrade"] is None and "overage" not in data
    assert "raise the overage cap on your account page" in data["detail"]


def test_own_budgets_lead_to_the_account_page_not_to_a_plan(selling, ada, controlled):
    agent, token = token_client(ada, "agent")
    token.budget_cpu_minutes = 1
    token.save()
    spend(ada, 61, api_token=token, source="token", via="agent")
    data = start(agent).data
    assert data["code"] == "budget_exceeded" and data["limit"]["kind"] == "token_budget"
    assert data["upgrade"] is None and "upgrade_url" not in data

    UserProfile.objects.update_or_create(user=ada, defaults={"budget_cpu_minutes": 1})
    client = APIClient()
    client.force_authenticate(ada)
    data = start(client).data
    assert data["limit"]["kind"] == "own_budget" and data["upgrade"] is None


def test_what_running_trainings_reserve_suggests_no_plan(selling, web, ada, controlled):
    budgets.set_plan(ada, "pro")  # two trainings at a time
    spend(ada, 120 * 3600 - 600)
    started(web)
    data = start(web).data
    assert data["code"] == "budget_exceeded" and data["reserved"] is True
    assert data["upgrade"] is None and "upgrade_url" not in data
    assert "reserved by trainings that are still running" in data["detail"]


# --------------------------------------------------------------------------- #
# The estimate, before a training starts
# --------------------------------------------------------------------------- #


def estimate_minutes(web, config) -> float:
    body = {"graph": DEMO_GRAPH, "dataset": RANDOM, "config": config}
    return web.post("/api/training/check/", body, format="json").data["estimate"]["cpu_minutes"]


def test_an_estimate_far_above_the_run_limit_is_refused_with_the_plan(
    selling, web, ada, controlled
):
    # A plan whose trainings may use next to nothing, and Starter with room for this one.
    selling.PLANS = {
        **selling.PLANS,
        "free": {**selling.PLANS["free"], "cpu_minutes_per_run": 0.05},
    }
    needed = estimate_minutes(web, LONG)
    assert 0.05 * 3 < needed < 60
    response = start(web, config=LONG)
    assert response.status_code == 403
    data = response.data
    assert data["code"] == "estimate_too_high"
    assert data["limit"] == {
        "kind": "run_time",
        "used": None,
        "limit": 0.05,
        "unit": "cpu_minutes",
        "resets_at": None,
    }
    assert data["upgrade"] == STARTER
    assert "Or: The Starter plan (5.00 EUR per month) includes" in data["detail"]
    assert data["estimate"]["cpu_minutes"] == needed
    # Said up front by the check, with the same plan.
    body = {"graph": DEMO_GRAPH, "dataset": RANDOM, "config": LONG}
    budget = web.post("/api/training/check/", body, format="json").data["budget"]
    assert (budget["allowed"], budget["code"]) == (False, "estimate_too_high")
    assert budget["limit"]["kind"] == "run_time" and budget["upgrade"] == STARTER


def test_an_estimate_above_what_is_left_warns_up_front_with_the_plan(selling, web, ada, controlled):
    config = {**QUICK, "epochs": 600}
    needed = estimate_minutes(web, config)
    # A little less is left of the month than the training is expected to use.
    spend(ada, 1800 - needed * 60 * 0.6)
    body = {"graph": DEMO_GRAPH, "dataset": RANDOM, "config": config}
    budget = web.post("/api/training/check/", body, format="json").data["budget"]
    assert budget["allowed"] is True and budget["limit_reason"] == "budget"
    assert budget["warning_code"] == "estimate_above_limit"
    assert "probably be stopped before it finishes" in budget["warning"]
    assert budget["limit"]["kind"] == "cpu_time" and budget["upgrade"] == STARTER
    assert budget["warning"].endswith("https://brickdnn.com/pricing?plan=starter")
    # It still starts (the estimate is rough), with the same warning.
    response = start(web, config=config)
    assert response.status_code == 201
    assert response.data["budget_warning"] == budget["warning"]


def test_the_estimate_without_paid_plans_names_no_plan(site, web, ada, controlled):
    # A training that fits: the check says nothing about limits or plans.
    body = {"graph": DEMO_GRAPH, "dataset": RANDOM, "config": QUICK}
    budget = web.post("/api/training/check/", body, format="json").data["budget"]
    assert budget["allowed"] is True and "upgrade" not in budget and "warning_code" not in budget
    site.PLANS = {**site.PLANS, "free": {**site.PLANS["free"], "cpu_minutes_per_run": 0.05}}
    data = start(web, config=LONG).data
    assert data["code"] == "estimate_too_high" and data["limit"]["kind"] == "run_time"
    assert data["upgrade"] is None and data["detail"].endswith("or a smaller network.")


# --------------------------------------------------------------------------- #
# Trainings at a time, trainings waiting
# --------------------------------------------------------------------------- #


def test_trainings_at_a_time(selling, web, ada, controlled):
    started(web)
    response = start(web)
    assert response.status_code == 409
    data = response.data
    assert data["code"] == "concurrent_trainings_limit"
    assert data["limit"] == {
        "kind": "concurrent_trainings",
        "used": 1,
        "limit": 1,
        "unit": "trainings",
        "resets_at": None,
    }
    assert data["plan"] == {"key": "free", "name": "Free"}
    assert data["upgrade"]["key"] == "pro" and data["upgrade"]["concurrent_trainings"] == 2
    assert data["detail"].startswith("Your plan (Free) allows 1 training at a time.")
    assert "The Pro plan (45.00 EUR per month) includes" in data["detail"]


def test_trainings_at_a_time_without_paid_plans(site, web, ada, controlled):
    started(web)
    data = start(web).data
    assert data["code"] == "concurrent_trainings_limit" and data["upgrade"] is None
    assert data["detail"] == (
        "Your plan (Free) allows 1 training at a time. Wait until your running training "
        "finishes, or stop it."
    )


def test_a_busy_server_is_no_limit_of_the_plan(selling, web, ada, controlled):
    selling.TRAINING_MAX_RUNS = 1
    other = User.objects.create_user("bob", password="correct-horse-battery")
    client = APIClient()
    client.force_authenticate(other)
    started(client)
    response = start(web)
    assert response.status_code == 409
    assert response.data == {
        "detail": "Another training is still running. Stop it or wait until it finishes.",
        "code": "server_busy",
    }


def test_trainings_waiting_in_the_queue(selling, web, ada, controlled):
    selling.TRAINING_WORKER = True
    selling.WORKER_TIMEOUT = 60
    selling.QUEUE_MAX_WAITING = None
    budgets.set_plan(ada, "starter")  # room for what three waiting trainings reserve
    for _ in range(3):
        started(web)
    response = start(web)
    assert response.status_code == 409
    data = response.data
    assert data["code"] == "queued_trainings_limit"
    assert data["limit"]["kind"] == "queued_trainings"
    assert (data["limit"]["used"], data["limit"]["limit"]) == (3, 3)
    assert data["upgrade"] is None  # every plan lets three wait
    # A plan that lets more wait is named.
    selling.PLANS = {**selling.PLANS, "plus": {**selling.PLANS["plus"], "queued_trainings": 10}}
    assert start(web).data["upgrade"]["key"] == "plus"


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #


def test_a_full_storage_names_the_plan_with_room(selling, web, ada, training_dir):
    UserProfile.objects.update_or_create(user=ada, defaults={"storage_mb": 0.002})
    assert upload(web, "iris.csv", iris_csv(12)).status_code == 201
    response = upload(web, "big.csv", iris_csv(200))
    assert response.status_code == 403
    data = response.data
    assert data["code"] == "storage_quota_exceeded"
    assert data["limit"]["kind"] == "storage" and data["limit"]["unit"] == "bytes"
    assert data["limit"]["limit"] == data["storage"]["limit_bytes"]
    assert data["upgrade"] == STARTER
    assert "Or: The Starter plan" in data["detail"]


def test_a_full_storage_refuses_trainings_too(selling, web, ada, controlled):
    """Not only uploads: a training stores its weights (budgets.check_storage in start_run)."""
    DataSource.objects.create(
        owner=ada, name="big", filename="big.csv", size_bytes=GB // 2, n_rows=1, n_columns=1
    )
    response = start(web)
    assert response.status_code == 403
    assert response.data["code"] == "storage_quota_exceeded"
    assert response.data["detail"].startswith("The storage of this account is full: 512.0 MB of")
    assert response.data["upgrade"]["key"] == "starter"


def test_the_data_tab_answers_with_the_whole_error(selling, web, ada, settings, tmp_path):
    pytest.importorskip("pandas")
    settings.DATA_PREP_DIR = tmp_path / "prepare"
    UserProfile.objects.update_or_create(user=ada, defaults={"storage_mb": 0.0001})
    response = web.post(
        "/api/data/sources/",
        {"file": SimpleUploadedFile("data.csv", iris_csv(50))},
        format="multipart",
    )
    assert response.status_code == 403
    assert response.data["code"] == "storage_quota_exceeded"
    assert response.data["limit"]["kind"] == "storage"
    assert response.data["upgrade"]["key"] == "starter"


def test_a_file_above_the_upload_limit_is_a_limit_of_the_server(selling, web, ada, training_dir):
    selling.TRAINING_MAX_UPLOAD_MB = 0
    response = upload(web, "iris.csv", iris_csv(12))
    assert response.status_code == 413
    assert response.data["code"] == "upload_too_large"
    assert response.data["limit"] == {"kind": "upload_size", "limit": 0, "unit": "bytes"}
    assert "upgrade" not in response.data


# --------------------------------------------------------------------------- #
# Runs logged from elsewhere
# --------------------------------------------------------------------------- #


def tracker(user) -> APIClient:
    _, key = create_token(user, "laptop", scope="runs:log")
    client = APIClient()
    client.credentials(HTTP_AUTHORIZATION=f"Bearer {key}")
    return client


def test_the_limits_of_logged_runs_are_the_servers(selling, ada, training_dir):
    selling.TRACKING_MAX_ACTIVE = 1
    script = tracker(ada)
    assert script.post("/api/tracking/runs/", {"name": "one"}, format="json").status_code == 201
    response = script.post("/api/tracking/runs/", {"name": "two"}, format="json")
    assert response.status_code == 409
    assert response.data["code"] == "active_limit"
    assert response.data["limit"] == {
        "kind": "external_active_runs",
        "used": 1,
        "limit": 1,
        "unit": "runs",
        "resets_at": None,
    }
    assert response.data["upgrade"] is None  # the same for every plan
    selling.TRACKING_MAX_RUNS = 1
    response = script.post("/api/tracking/runs/", {"name": "two"}, format="json")
    assert response.data["code"] == "runs_limit"
    assert response.data["limit"]["kind"] == "external_runs"


def test_logged_runs_and_a_full_storage(selling, ada, training_dir):
    script = tracker(ada)
    DataSource.objects.create(
        owner=ada, name="big", filename="big.csv", size_bytes=GB // 2, n_rows=1, n_columns=1
    )
    response = script.post("/api/tracking/runs/", {"name": "one"}, format="json")
    assert response.status_code == 403
    assert response.data["code"] == "storage_quota_exceeded"
    assert response.data["limit"]["kind"] == "storage"
    assert response.data["upgrade"]["key"] == "starter"
    assert "The storage of this account is full" in response.data["detail"]


# --------------------------------------------------------------------------- #
# MCP: the same in the error of a tool
# --------------------------------------------------------------------------- #


def test_mcp_errors_carry_the_code_and_the_plan(selling, ada, controlled):
    agent, _ = token_client(ada)
    spend(ada, 1800)
    result, text = call(agent, "start_training", graph=DEMO_GRAPH, dataset=RANDOM, config=QUICK)
    assert result["isError"] is True
    assert "The Starter plan (5.00 EUR per month) includes 10 CPU hours per month" in text
    assert "https://brickdnn.com/pricing?plan=starter" in text and "get_usage" in text
    details = result["structuredContent"]
    assert details["code"] == "budget_exceeded"
    assert details["limit"]["kind"] == "cpu_time"
    assert details["upgrade"]["key"] == "starter" and details["upgrade"]["price"] == 5.0


def test_mcp_names_the_plan_for_trainings_at_a_time(selling, ada, controlled):
    agent, _ = token_client(ada)
    result, _ = call(agent, "start_training", graph=DEMO_GRAPH, dataset=RANDOM, config=QUICK)
    assert "isError" not in result
    result, text = call(agent, "start_training", graph=DEMO_GRAPH, dataset=RANDOM, config=QUICK)
    assert result["isError"] is True
    assert text.startswith("Your plan (Free) allows 1 training at a time.")
    assert result["structuredContent"]["code"] == "concurrent_trainings_limit"
    assert result["structuredContent"]["upgrade"]["key"] == "pro"


def test_mcp_errors_without_paid_plans(site, ada, controlled):
    agent, _ = token_client(ada)
    spend(ada, 1800)
    result, text = call(agent, "start_training", graph=DEMO_GRAPH, dataset=RANDOM, config=QUICK)
    assert result["isError"] is True and "pricing" not in text
    assert result["structuredContent"]["code"] == "budget_exceeded"
    assert result["structuredContent"]["upgrade"] is None
    # Other errors of a tool stay text only.
    result, _ = call(agent, "start_training", graph={"nodes": []}, dataset=RANDOM)
    assert result["isError"] is True and "structuredContent" not in result


# --------------------------------------------------------------------------- #
# A run that a limit stopped
# --------------------------------------------------------------------------- #


def test_a_run_stopped_at_the_limit_per_training_says_where_that_leads(
    selling, web, ada, controlled
):
    run_id = started(web)
    wait_for_state(controlled, run_id)
    use_cpu(web, controlled, run_id, 901.0)
    data = wait_for(web, run_id)
    assert data["usage"]["stop_reason"] == "run_limit"
    hint = data["limit_hint"]
    assert hint["limit"]["kind"] == "run_time" and hint["limit"]["limit"] == 15.0
    assert hint["plan"] == {"key": "free", "name": "Free"}
    assert hint["upgrade"] == STARTER
    # Only for who started it, and not in lists.
    other = User.objects.create_user("eve", password="correct-horse-battery", is_superuser=True)
    client = APIClient()
    client.force_authenticate(other)
    listed = web.get("/api/training/runs/").data
    assert "limit_hint" not in (listed["results"] if isinstance(listed, dict) else listed)[0]


def test_a_run_stopped_by_the_budget_names_the_plan_while_it_is_used_up(
    selling, web, ada, controlled
):
    spend(ada, 1500)
    run_id = started(web)
    wait_for_state(controlled, run_id)
    use_cpu(web, controlled, run_id, 300.0)
    data = wait_for(web, run_id)
    assert data["usage"]["stop_reason"] == "budget"
    assert data["limit_hint"]["limit"]["kind"] == "cpu_time"
    assert data["limit_hint"]["upgrade"] == STARTER
    # With room again (a higher plan, a new month) the run no longer asks for anything.
    budgets.set_plan(ada, "plus")
    assert web.get(f"/api/training/runs/{run_id}/").data["limit_hint"] is None


def test_runs_that_ended_otherwise_have_no_hint(selling, web, ada, controlled):
    run_id = started(web)
    wait_for_state(controlled, run_id)
    (run_dir(controlled, run_id) / "finish").touch()
    assert wait_for(web, run_id)["limit_hint"] is None


# --------------------------------------------------------------------------- #
# Features of plans: nothing is gated until a plan lists a feature
# --------------------------------------------------------------------------- #


def test_features_belong_to_everybody_until_a_plan_lists_them(selling, ada):
    assert budgets.has_feature(ada, "share_comparisons") is True
    budgets.require_feature(ada, "share_comparisons", "Sharing a comparison")
    assert budgets.plans()["free"].features == ()
    assert budgets.plans()["free"].to_dict()["features"] == []


def test_a_feature_of_higher_plans_leads_to_the_cheapest_plan_with_it(selling, ada):
    selling.PLANS = {
        key: {**plan, **({"features": ["share_comparisons"]} if key in ("plus", "pro") else {})}
        for key, plan in selling.PLANS.items()
    }
    assert budgets.has_feature(ada, "share_comparisons") is False
    assert budgets.has_feature(ada, "something_else") is True
    with pytest.raises(budgets.FeatureNotInPlan) as caught:
        budgets.require_feature(ada, "share_comparisons", "Sharing a comparison")
    body = caught.value.body()
    assert body["code"] == "feature_not_in_plan"
    assert body["limit"]["kind"] == "feature" and body["limit"]["feature"] == "share_comparisons"
    assert body["upgrade"]["key"] == "plus"
    assert body["detail"].startswith(
        "Sharing a comparison is not part of your plan (Free); it comes with Plus, Pro. "
        "The Plus plan (14.00 EUR per month) includes"
    )
    budgets.set_plan(ada, "plus")
    assert budgets.has_feature(ada, "share_comparisons") is True
    # Accounts without limits have every feature; without paid plans no plan is offered.
    budgets.set_plan(ada, "free")
    budgets.set_unlimited(ada, True)
    assert budgets.has_feature(ada, "share_comparisons") is True
    budgets.set_unlimited(ada, False)
    selling.PADDLE_ENV = ""
    with pytest.raises(budgets.FeatureNotInPlan) as caught:
        budgets.require_feature(ada, "share_comparisons", "Sharing a comparison")
    assert caught.value.data["upgrade"] is None


def test_features_must_be_a_list_of_names(selling):
    from django.core.exceptions import ImproperlyConfigured

    selling.PLANS = {**selling.PLANS, "plus": {**selling.PLANS["plus"], "features": "sharing"}}
    with pytest.raises(ImproperlyConfigured, match="features of plan 'plus' must be a list"):
        budgets.plans()
