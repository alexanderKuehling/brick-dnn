"""Paid plans with Paddle (billing.py, paddle.py), against a fake Paddle: no network.

The plans are brickdnn.com's (deploy/sites/brickdnn.com/settings.env).
"""

from __future__ import annotations

import io
import json
import time
import zipfile
from datetime import date, timedelta
from decimal import Decimal
from io import StringIO
from pathlib import Path

import pytest
from django.contrib.admin.sites import site as admin_site
from django.contrib.auth import get_user_model
from django.core import checks
from django.core.management import call_command
from django.db import transaction
from django.utils import timezone
from rest_framework.test import APIClient

from brick_dnn_core import billing, budgets, retention
from brick_dnn_core.api_tokens import create_token
from brick_dnn_core.billing_admin import user_link
from brick_dnn_core.models import (
    AgentAction,
    BillingCustomer,
    BillingEvent,
    OverageCharge,
    UsageRecord,
    UserProfile,
)
from brick_dnn_core.paddle import PaddleError, signature_header
from brick_dnn_site.settings import SITE_ENV_SECRETS, load_site_settings

from .fake_paddle import FakePaddle, iso

pytestmark = pytest.mark.django_db

User = get_user_model()
PASSWORD = "correct-horse-battery"
SECRET = "pdl_ntfset_test_secret"
WEBHOOK = "/api/billing/paddle/webhook/"
PRICES = {"starter": "pri_starter", "plus": "pri_plus", "pro": "pri_pro"}
SITE_ENV = (
    Path(__file__).resolve().parents[2] / "deploy" / "sites" / "brickdnn.com" / "settings.env"
)


def site_setting(name: str) -> str:
    for line in SITE_ENV.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{name}="):
            return line.split("=", 1)[1].strip()
    raise AssertionError(f"{name} is not in {SITE_ENV}")


# --------------------------------------------------------------------------- #
# Fixtures and helpers
# --------------------------------------------------------------------------- #


@pytest.fixture
def plans(settings):
    """The plans of brickdnn.com, billing still off."""
    settings.PLANS = json.loads(site_setting("BRICKDNN_PLANS"))
    settings.DEFAULT_PLAN = site_setting("BRICKDNN_DEFAULT_PLAN")
    settings.PRICE_CURRENCY = "EUR"
    settings.STAFF_UNLIMITED = True
    settings.PUBLIC_URL = ""
    return settings


@pytest.fixture
def paddle(plans, monkeypatch):
    """Billing switched on (sandbox), with a fake Paddle behind it."""
    plans.PADDLE_ENV = "sandbox"
    plans.PADDLE_CLIENT_TOKEN = "test_0123456789abcdef"
    plans.PADDLE_API_KEY = "pdl_sdbx_apikey_secret"
    plans.PADDLE_WEBHOOK_SECRETS = [SECRET]
    plans.PADDLE_PRICES = ",".join(f"{plan}={price}" for plan, price in PRICES.items())
    plans.PADDLE_OVERAGE_PRODUCT = "pro_overage"
    plans.PADDLE_GRACE_DAYS = 7
    plans.PADDLE_WEBHOOK_TOLERANCE = 300
    plans.PADDLE_OVERAGE_MIN_CHARGE = 0.5
    fake = FakePaddle()
    monkeypatch.setattr(billing, "client", lambda cfg=None: fake)
    return fake


def make_user(name: str, **extra):
    return User.objects.create_user(name, f"{name}@example.com", PASSWORD, **extra)


def login(user) -> APIClient:
    client = APIClient()
    response = client.post(
        "/api/auth/token/", {"username": user.username, "password": PASSWORD}, format="json"
    )
    assert response.status_code == 200, response.content
    client.credentials(HTTP_AUTHORIZATION=f"Bearer {response.json()['access']}")
    return client


@pytest.fixture
def ada():
    return make_user("ada")


@pytest.fixture
def web(ada):
    return login(ada)


def post_event(payload, secret: str = SECRET, timestamp: int | None = None, **headers):
    """Post a notification the way Paddle does: the raw body, signed."""
    body = json.dumps(payload).encode()
    headers.setdefault("HTTP_PADDLE_SIGNATURE", signature_header(body, secret, timestamp))
    return APIClient().post(WEBHOOK, data=body, content_type="application/json", **headers)


def notify(paddle: FakePaddle, event_type: str, data: dict, at=None):
    response = post_event(paddle.event(event_type, data, at))
    assert response.status_code == 200, response.content
    return response.json()


def plan_of(user) -> str:
    return budgets.plan_of(User.objects.get(pk=user.pk)).key


def subscribe(paddle: FakePaddle, web: APIClient, user, plan: str = "plus") -> str:
    """Checkout, payment at Paddle and its notification. Returns the subscription id."""
    answer = web.post("/api/billing/checkout/", {"plan": plan}, format="json")
    assert answer.status_code == 200, answer.content
    checkout = answer.json()
    entity = paddle.subscribe(
        checkout["customer_id"], checkout["price_id"], checkout["custom_data"]
    )
    assert notify(paddle, "subscription.created", entity)["outcome"] == "applied"
    return entity["id"]


def state(web: APIClient) -> dict:
    response = web.get("/api/billing/subscription/")
    assert response.status_code == 200, response.content
    return response.json()


def use_overage(user, hours: float, period: date | None = None) -> UsageRecord:
    """The ledger says: this much compute beyond the allowance in the month."""
    return UsageRecord.objects.create(
        user=user,
        kind="training",
        quantity=hours * 3600 + 100,
        overage=hours * 3600,
        period=period or last_month(),
    )


def last_month() -> date:
    first = budgets.period_of()
    return (first - timedelta(days=1)).replace(day=1)


# --------------------------------------------------------------------------- #
# Off unless configured
# --------------------------------------------------------------------------- #

ACCOUNT_ENDPOINTS = [
    ("get", "/api/billing/subscription/"),
    ("post", "/api/billing/checkout/"),
    ("post", "/api/billing/change-plan/"),
    ("post", "/api/billing/cancel/"),
    ("post", "/api/billing/reactivate/"),
    ("post", "/api/billing/portal/"),
    ("post", "/api/billing/refresh/"),
]


def test_billing_is_off_by_default(plans, web):
    assert not billing.enabled()
    assert billing.pricing_url() is None
    assert APIClient().get("/api/billing/").json() == {"enabled": False}
    for method, url in ACCOUNT_ENDPOINTS:
        response = getattr(web, method)(url, {"plan": "plus"}, format="json")
        assert response.status_code == 404, url
    assert post_event({"event_id": "evt_1"}).status_code == 404
    assert post_event({"event_id": "evt_1"}, HTTP_PADDLE_SIGNATURE="").status_code == 404
    assert billing.sync_all().checked == 0
    assert BillingEvent.objects.count() == 0


def test_billing_stays_off_while_the_settings_are_incomplete(plans):
    plans.PADDLE_ENV = "sandbox"
    plans.PADDLE_CLIENT_TOKEN = "test_token"
    plans.PADDLE_PRICES = "plus=pri_plus,gold=pri_gold,free=pri_free"
    cfg = billing.config()
    assert cfg.configured and not cfg.enabled
    text = " ".join(cfg.problems)
    assert "BRICKDNN_PADDLE_API_KEY is missing" in text
    assert "BRICKDNN_PADDLE_WEBHOOK_SECRET is missing" in text
    assert "there is no plan 'gold'" in text
    assert "'free' is the default plan" in text
    assert APIClient().get("/api/billing/").json() == {"enabled": False}
    warnings = [m for m in checks.run_checks() if m.id == "brickdnn.W001"]
    assert len(warnings) == 1 and "switched off" in warnings[0].msg

    plans.PADDLE_API_KEY = "key"
    plans.PADDLE_WEBHOOK_SECRETS = ["secret"]
    plans.PADDLE_PRICES = "plus=pri_plus"
    plans.PADDLE_ENV = "production"
    assert billing.config().problems == ("BRICKDNN_PADDLE_ENV must be sandbox or live.",)
    plans.PADDLE_ENV = "live"
    assert "not a live token" in billing.config().problems[0]
    plans.PADDLE_CLIENT_TOKEN = "live_token"
    assert billing.enabled()
    assert [m for m in checks.run_checks() if m.id == "brickdnn.W001"] == []


def test_prices_setting_formats(plans):
    assert billing._prices("starter=pri_a, plus=pri_b|pri_old ,") == (
        {"starter": ("pri_a",), "plus": ("pri_b", "pri_old")},
        [],
    )
    assert billing._prices('{"starter": "pri_a", "plus": ["pri_b", "pri_old"]}') == (
        {"starter": ("pri_a",), "plus": ("pri_b", "pri_old")},
        [],
    )
    assert billing._prices("{nonsense")[1] == ["BRICKDNN_PADDLE_PRICES is not valid JSON."]
    assert "is not plan=price" in billing._prices("pri_a")[1][0]


def test_one_price_cannot_belong_to_two_plans(paddle, plans):
    plans.PADDLE_PRICES = "starter=pri_x,plus=pri_x"
    assert "is the price of two plans" in " ".join(billing.config().problems)


def test_secrets_are_never_read_from_the_site_settings_file(tmp_path):
    assert {"BRICKDNN_PADDLE_API_KEY", "BRICKDNN_PADDLE_WEBHOOK_SECRET"} <= SITE_ENV_SECRETS
    file = tmp_path / "settings.env"
    file.write_text(
        "BRICKDNN_PADDLE_ENV=sandbox\n"
        "BRICKDNN_PADDLE_API_KEY=leaked\n"
        "BRICKDNN_PADDLE_WEBHOOK_SECRET=leaked\n"
    )
    environ = {"BRICKDNN_SITE_ENV": str(file)}
    load_site_settings(environ)
    assert environ == {"BRICKDNN_SITE_ENV": str(file), "BRICKDNN_PADDLE_ENV": "sandbox"}


def test_the_site_settings_of_brickdnn_com_hold_no_paddle_secret_and_leave_billing_off():
    lines = SITE_ENV.read_text(encoding="utf-8").splitlines()
    active = [line.split("=")[0] for line in lines if line and not line.startswith("#")]
    assert not SITE_ENV_SECRETS & set(active)
    # Until the owner has a Paddle account, nothing of Paddle is set (only placeholders).
    assert not [name for name in active if name.startswith("BRICKDNN_PADDLE_")]
    assert any(line.startswith("# BRICKDNN_PADDLE_ENV=") for line in lines)


def test_info_has_what_the_checkout_needs_and_no_secret(paddle):
    response = APIClient().get("/api/billing/")
    assert response.status_code == 200
    info = response.json()
    assert info["enabled"] is True
    assert (info["provider"], info["environment"]) == ("paddle", "sandbox")
    assert info["client_token"] == "test_0123456789abcdef"
    assert (info["currency"], info["prices_include_tax"]) == ("EUR", True)
    assert info["overage_billed"] is True
    assert [(p["key"], p["price"], p["purchasable"], p["default"]) for p in info["plans"]] == [
        ("free", 0, False, True),
        ("starter", 5, True, False),
        ("plus", 14, True, False),
        ("pro", 45, True, False),
    ]
    assert info["plans"][1]["overage_price_per_cpu_hour"] == 0.4
    text = response.content.decode()
    assert "pdl_sdbx_apikey_secret" not in text and SECRET not in text and "pri_" not in text


def test_plans_without_a_price_are_not_offered(paddle, plans):
    plans.PADDLE_PRICES = "plus=pri_plus"
    info = APIClient().get("/api/billing/").json()
    assert [p["key"] for p in info["plans"]] == ["free", "plus"]


# --------------------------------------------------------------------------- #
# Checkout
# --------------------------------------------------------------------------- #


def test_the_account_endpoints_need_a_website_login(paddle, ada):
    for method, url in ACCOUNT_ENDPOINTS:
        assert getattr(APIClient(), method)(url).status_code == 401, url
    _, key = create_token(ada, name="agent")
    agent = APIClient()
    agent.credentials(HTTP_AUTHORIZATION=f"Bearer {key}")
    for method, url in ACCOUNT_ENDPOINTS:
        assert getattr(agent, method)(url, {"plan": "plus"}, format="json").status_code == 403, url
    assert paddle.calls == []


def test_checkout_creates_the_customer_with_the_accounts_email(paddle, ada, web):
    assert state(web)["can_subscribe"] is True
    response = web.post("/api/billing/checkout/", {"plan": "plus"}, format="json")
    assert response.status_code == 200, response.content
    customer = BillingCustomer.objects.get(user=ada)
    assert response.json() == {
        "plan": "plus",
        "price_id": "pri_plus",
        "customer_id": customer.customer_id,
        "custom_data": {"user_id": str(ada.pk)},
    }
    assert paddle.called("create_customer") == [("ada@example.com", {"user_id": str(ada.pk)})]
    assert (customer.environment, customer.status, customer.subscription_id) == (
        "sandbox",
        "",
        None,
    )
    # Opening the checkout does not change the plan, and asking again makes no new customer.
    assert plan_of(ada) == "free"
    again = web.post("/api/billing/checkout/", {"plan": "pro"}, format="json").json()
    assert (again["customer_id"], again["price_id"]) == (customer.customer_id, "pri_pro")
    assert len(paddle.called("create_customer")) == 1


@pytest.mark.parametrize("plan", ["free", "gold", ""])
def test_checkout_only_sells_plans_with_a_price(paddle, web, plan):
    response = web.post("/api/billing/checkout/", {"plan": plan}, format="json")
    assert response.status_code in (400, 409)
    assert paddle.calls == []


def test_checkout_needs_an_email_address(paddle):
    user = User.objects.create_user("noemail", "", PASSWORD)
    response = login(user).post("/api/billing/checkout/", {"plan": "plus"}, format="json")
    assert response.status_code == 409
    assert "e-mail address" in response.json()["detail"]
    assert paddle.calls == []


def test_checkout_is_refused_while_subscribed(paddle, ada, web):
    subscribe(paddle, web, ada)
    response = web.post("/api/billing/checkout/", {"plan": "pro"}, format="json")
    assert response.status_code == 409
    assert "already have a subscription" in response.json()["detail"]


def test_checkout_does_not_take_over_a_customer_of_somebody_else(paddle, ada, web):
    # Paddle knows the address from an account that is gone: its invoices are not ada's.
    paddle.create_customer("ada@example.com", {"user_id": "999"})
    response = web.post("/api/billing/checkout/", {"plan": "plus"}, format="json")
    assert response.status_code == 409
    assert "already has a billing profile" in response.json()["detail"]
    assert not BillingCustomer.objects.exists()


def test_checkout_finds_the_customer_it_made_for_the_account_before(paddle, ada, web):
    mine = paddle.create_customer("ada@example.com", {"user_id": str(ada.pk)})
    response = web.post("/api/billing/checkout/", {"plan": "plus"}, format="json")
    assert response.status_code == 200
    assert response.json()["customer_id"] == mine["id"]


def test_checkout_says_when_paddle_does_not_answer(paddle, ada, web):
    paddle.down("create_customer")
    response = web.post("/api/billing/checkout/", {"plan": "plus"}, format="json")
    assert response.status_code == 503
    assert "did not answer" in response.json()["detail"]
    assert not BillingCustomer.objects.exists()
    paddle.up()
    paddle.refuses("create_customer", "forbidden", 403)
    response = web.post("/api/billing/checkout/", {"plan": "plus"}, format="json")
    assert response.status_code == 503
    assert "refused the request (forbidden)" in response.json()["detail"]


# --------------------------------------------------------------------------- #
# The webhook: signature, size, idempotency, order
# --------------------------------------------------------------------------- #


def test_webhook_refuses_what_is_not_signed_by_paddle(paddle, ada, web):
    checkout = web.post("/api/billing/checkout/", {"plan": "plus"}, format="json").json()
    entity = paddle.subscribe(checkout["customer_id"], "pri_plus", checkout["custom_data"])
    payload = paddle.event("subscription.created", entity)
    body = json.dumps(payload).encode()

    def post(**headers):
        return APIClient().post(WEBHOOK, data=body, content_type="application/json", **headers)

    assert post().status_code == 403  # no signature
    assert post(HTTP_PADDLE_SIGNATURE="ts=1;h1=00").status_code == 403
    assert post_event(payload, secret="guessed").status_code == 403
    assert post_event(payload, timestamp=int(time.time()) - 3600).status_code == 403
    # Signed for another body.
    other = signature_header(body + b" ", SECRET)
    assert post(HTTP_PADDLE_SIGNATURE=other).status_code == 403
    assert plan_of(ada) == "free"
    assert BillingEvent.objects.count() == 0
    # The same notification, signed: it counts.
    assert post_event(payload).status_code == 200
    assert plan_of(ada) == "plus"


def test_webhook_refuses_large_and_unreadable_bodies(paddle):
    big = b"x" * (300 * 1024)
    response = APIClient().post(
        WEBHOOK,
        data=big,
        content_type="application/json",
        HTTP_PADDLE_SIGNATURE=signature_header(big, SECRET),
    )
    assert response.status_code == 413
    for body in (b"not json", b"[1, 2]", b'{"event_id": "evt_1"}', b"\xff\xfe"):
        response = APIClient().post(
            WEBHOOK,
            data=body,
            content_type="application/json",
            HTTP_PADDLE_SIGNATURE=signature_header(body, SECRET),
        )
        assert response.status_code == 400, body
    assert BillingEvent.objects.count() == 0


def test_webhook_answers_without_the_trailing_slash_too(paddle):
    payload = paddle.event("customer.updated", {"id": "ctm_1"})
    body = json.dumps(payload).encode()
    response = APIClient().post(
        WEBHOOK.rstrip("/"),
        data=body,
        content_type="application/json",
        HTTP_PADDLE_SIGNATURE=signature_header(body, SECRET),
    )
    assert response.status_code == 200
    assert response.json() == {"received": True, "new": True, "outcome": "ignored"}


def test_webhook_is_rate_limited(paddle, settings):
    from rest_framework.throttling import ScopedRateThrottle

    rates = {**settings.REST_FRAMEWORK["DEFAULT_THROTTLE_RATES"], "billing_webhook": "2/minute"}
    ScopedRateThrottle.THROTTLE_RATES = rates
    try:
        codes = [
            post_event(paddle.event("customer.updated", {"id": "ctm_1"})).status_code
            for _ in range(3)
        ]
    finally:
        ScopedRateThrottle.THROTTLE_RATES = settings.REST_FRAMEWORK["DEFAULT_THROTTLE_RATES"]
    assert codes == [200, 200, 429]


def test_an_event_is_processed_once(paddle, ada, web):
    checkout = web.post("/api/billing/checkout/", {"plan": "plus"}, format="json").json()
    entity = paddle.subscribe(checkout["customer_id"], "pri_plus", checkout["custom_data"])
    payload = paddle.event("subscription.created", entity)
    first = post_event(payload)
    assert first.json() == {"received": True, "new": True, "outcome": "applied"}
    assert plan_of(ada) == "plus"
    # The operator moves the account by hand; a repeated delivery does not undo that.
    budgets.set_plan(ada, "pro")
    second = post_event(payload)
    assert second.status_code == 200
    assert second.json() == {"received": True, "new": False, "outcome": "applied"}
    assert plan_of(ada) == "pro"
    assert BillingEvent.objects.filter(event_id=payload["event_id"]).count() == 1


def test_events_that_arrive_late_do_not_overwrite_newer_ones(paddle, ada, web):
    subscription = subscribe(paddle, web, ada)
    active = paddle.entity(subscription)
    earlier = paddle.tick()
    ended = paddle.end(subscription)
    assert notify(paddle, "subscription.canceled", ended)["outcome"] == "applied"
    assert plan_of(ada) == "free"
    # An update from before the cancellation arrives afterwards.
    late = notify(paddle, "subscription.updated", active, at=earlier)
    assert late["outcome"] == "stale"
    assert plan_of(ada) == "free"
    assert BillingCustomer.objects.get(user=ada).status == "canceled"
    # The same moment as the newest known one is not older.
    customer = BillingCustomer.objects.get(user=ada)
    same = notify(paddle, "subscription.canceled", ended, at=customer.last_event_at)
    assert same["outcome"] == "applied"


def test_other_events_are_stored_and_ignored(paddle, ada):
    for kind in ("customer.created", "address.updated", "payout.paid", "adjustment.created"):
        assert (
            notify(paddle, kind, {"id": "x_1", "customer_id": "ctm_none"})["outcome"] == "ignored"
        )
    assert BillingEvent.objects.filter(outcome="ignored", user=None).count() == 4


# --------------------------------------------------------------------------- #
# Whose subscription it is
# --------------------------------------------------------------------------- #


def test_a_subscription_of_an_unknown_customer_upgrades_nobody(paddle, ada):
    # Somebody opens a checkout with ada's user id as custom data, as another customer.
    foreign = paddle.create_customer("mallory@example.com", {})
    entity = paddle.subscribe(foreign["id"], "pri_pro", {"user_id": str(ada.pk)})
    assert notify(paddle, "subscription.created", entity)["outcome"] == "unknown"
    assert plan_of(ada) == "free"
    assert not BillingCustomer.objects.exists()


def test_a_subscription_for_someone_elses_customer_cannot_upgrade_me(paddle, ada, web):
    eve = make_user("eve")
    eve_web = login(eve)
    mine = web.post("/api/billing/checkout/", {"plan": "plus"}, format="json").json()
    hers = eve_web.post("/api/billing/checkout/", {"plan": "pro"}, format="json").json()
    # Eve pays as her own customer, but her checkout names ada's account.
    entity = paddle.subscribe(hers["customer_id"], "pri_pro", mine["custom_data"])
    answer = notify(paddle, "subscription.created", entity)
    assert answer["outcome"] == "rejected"
    assert (plan_of(ada), plan_of(eve)) == ("free", "free")
    event = BillingEvent.objects.get(outcome="rejected")
    assert event.user == eve and "named account" in event.note
    # Without the wrong name it is simply eve's subscription.
    entity = paddle.subscribe(hers["customer_id"], "pri_pro", hers["custom_data"])
    assert notify(paddle, "subscription.created", entity)["outcome"] == "applied"
    assert (plan_of(ada), plan_of(eve)) == ("free", "pro")


def test_an_event_cannot_move_a_subscription_to_another_customer(paddle, ada, web):
    eve = make_user("eve")
    hers = login(eve).post("/api/billing/checkout/", {"plan": "pro"}, format="json").json()
    subscription = subscribe(paddle, web, ada, "starter")
    forged = paddle.entity(subscription)
    forged["customer_id"] = hers["customer_id"]
    forged["items"][0]["price"]["id"] = "pri_pro"
    assert notify(paddle, "subscription.updated", forged)["outcome"] == "rejected"
    assert (plan_of(ada), plan_of(eve)) == ("starter", "free")


def test_the_plan_comes_from_the_price_paddle_reports(paddle, ada, web):
    checkout = web.post("/api/billing/checkout/", {"plan": "starter"}, format="json").json()
    # The browser was told to sell Starter; what was bought decides.
    entity = paddle.subscribe(checkout["customer_id"], "pri_pro", checkout["custom_data"])
    notify(paddle, "subscription.created", entity)
    assert plan_of(ada) == "pro"


def test_a_price_that_is_not_configured_gives_no_plan(paddle, ada, web):
    checkout = web.post("/api/billing/checkout/", {"plan": "plus"}, format="json").json()
    entity = paddle.subscribe(checkout["customer_id"], "pri_unknown", checkout["custom_data"])
    assert notify(paddle, "subscription.created", entity)["outcome"] == "applied"
    assert plan_of(ada) == "free"
    assert BillingCustomer.objects.get(user=ada).plan == ""


def test_an_older_price_of_a_plan_still_counts(paddle, plans, ada, web):
    plans.PADDLE_PRICES = "starter=pri_starter,plus=pri_plus_2027|pri_plus,pro=pri_pro"
    checkout = web.post("/api/billing/checkout/", {"plan": "plus"}, format="json").json()
    assert checkout["price_id"] == "pri_plus_2027"
    entity = paddle.subscribe(checkout["customer_id"], "pri_plus", checkout["custom_data"])
    notify(paddle, "subscription.created", entity)
    assert plan_of(ada) == "plus"


def test_a_second_subscription_of_the_same_account_is_rejected(paddle, ada, web):
    first = subscribe(paddle, web, ada, "plus")
    customer = BillingCustomer.objects.get(user=ada)
    second = paddle.subscribe(customer.customer_id, "pri_pro", {"user_id": str(ada.pk)})
    assert notify(paddle, "subscription.created", second)["outcome"] == "rejected"
    assert plan_of(ada) == "plus"
    assert BillingCustomer.objects.get(user=ada).subscription_id == first
    # After the first one ended, a new one takes its place.
    notify(paddle, "subscription.canceled", paddle.end(first))
    assert plan_of(ada) == "free"
    third = paddle.subscribe(customer.customer_id, "pri_starter", {"user_id": str(ada.pk)})
    assert notify(paddle, "subscription.created", third)["outcome"] == "applied"
    assert plan_of(ada) == "starter"
    # A late event of the old, ended subscription changes nothing.
    assert notify(paddle, "subscription.canceled", paddle.entity(first))["outcome"] == "ignored"
    assert plan_of(ada) == "starter"


# --------------------------------------------------------------------------- #
# The life of a subscription
# --------------------------------------------------------------------------- #


def test_lifecycle_from_checkout_to_cancellation(paddle, ada, web):
    # Checkout and payment.
    subscription = subscribe(paddle, web, ada, "starter")
    assert plan_of(ada) == "starter"
    current = state(web)
    assert current["plan"]["key"] == "starter"
    assert current["subscription"]["status"] == "active"
    assert current["subscription"]["plan_name"] == "Starter"
    assert current["subscription"]["renews_at"] is not None
    assert current["subscription"]["ends_at"] is None
    assert current["can_subscribe"] is False
    notify(paddle, "subscription.activated", paddle.entity(subscription))
    notify(
        paddle,
        "transaction.completed",
        {"id": "txn_1", "subscription_id": subscription, "customer_id": current_customer(ada)},
    )
    assert plan_of(ada) == "starter"

    # An upgrade in the app: it holds at once, the notification confirms it.
    changed = web.post("/api/billing/change-plan/", {"plan": "pro"}, format="json")
    assert changed.status_code == 200, changed.content
    assert changed.json()["plan"]["key"] == "pro"
    assert paddle.called("change_price") == [
        (subscription, "pri_pro", "prorated_immediately", False)
    ]
    assert plan_of(ada) == "pro"
    notify(paddle, "subscription.updated", paddle.entity(subscription))
    assert plan_of(ada) == "pro"

    # A payment fails: the plan stays while Paddle retries.
    notify(
        paddle,
        "transaction.payment_failed",
        {"id": "txn_2", "subscription_id": subscription, "customer_id": current_customer(ada)},
    )
    notify(paddle, "subscription.past_due", paddle.set_status(subscription, "past_due"))
    assert plan_of(ada) == "pro"
    current = state(web)["subscription"]
    assert (current["status"], current["payment_failed"]) == ("past_due", True)
    assert current["grace_until"] is not None and current["can_change_plan"] is False
    refused = web.post("/api/billing/change-plan/", {"plan": "plus"}, format="json")
    assert refused.status_code == 409 and "payment" in refused.json()["detail"]

    # The retry goes through.
    notify(paddle, "subscription.updated", paddle.set_status(subscription, "active"))
    notify(
        paddle,
        "transaction.completed",
        {"id": "txn_3", "subscription_id": subscription, "customer_id": current_customer(ada)},
    )
    assert plan_of(ada) == "pro"
    current = state(web)["subscription"]
    assert (current["status"], current["payment_failed"], current["grace_until"]) == (
        "active",
        False,
        None,
    )

    # Cancelled to the end of the period: the plan stays until Paddle reports the end.
    cancelled = web.post("/api/billing/cancel/")
    assert cancelled.status_code == 200, cancelled.content
    assert paddle.called("cancel_subscription") == [(subscription, False)]
    current = cancelled.json()["subscription"]
    assert current["status"] == "active" and current["renews_at"] is None
    assert current["ends_at"] == current["current_period_end"]
    assert (current["can_cancel"], current["can_reactivate"]) == (False, True)
    notify(paddle, "subscription.updated", paddle.entity(subscription))
    assert plan_of(ada) == "pro"

    # The period is over.
    notify(paddle, "subscription.canceled", paddle.end(subscription))
    assert plan_of(ada) == "free"
    current = state(web)
    assert current["plan"]["key"] == "free"
    assert current["subscription"]["status"] == "canceled"
    assert current["subscription"]["ended_at"] is not None
    assert current["can_subscribe"] is True
    # Nothing of the account was deleted, and it can buy a plan again.
    assert User.objects.filter(pk=ada.pk).exists()
    assert web.post("/api/billing/checkout/", {"plan": "plus"}, format="json").status_code == 200


def current_customer(user) -> str:
    return BillingCustomer.objects.get(user=user).customer_id


def test_a_failed_payment_costs_the_plan_after_the_grace_period(paddle, ada, web):
    subscription = subscribe(paddle, web, ada, "plus")
    notify(paddle, "subscription.past_due", paddle.set_status(subscription, "past_due"))
    customer = BillingCustomer.objects.get(user=ada)
    assert customer.past_due_since is not None
    now = timezone.now()
    assert billing.entitled_plan(customer, now + timedelta(days=6)) == "plus"
    assert billing.entitled_plan(customer, now + timedelta(days=8)) == ""
    # A later past_due notification does not restart the grace period.
    since = customer.past_due_since
    notify(paddle, "subscription.updated", paddle.set_status(subscription, "past_due"))
    assert BillingCustomer.objects.get(user=ada).past_due_since == since

    # Eight days later: the account page and the daily run both end the plan.
    BillingCustomer.objects.filter(user=ada).update(past_due_since=now - timedelta(days=8))
    assert state(web)["plan"]["key"] == "free"
    assert plan_of(ada) == "free"
    assert state(web)["subscription"]["status"] == "past_due"
    # Paid after all: the plan is back.
    notify(paddle, "subscription.updated", paddle.set_status(subscription, "active"))
    assert plan_of(ada) == "plus"
    assert BillingCustomer.objects.get(user=ada).past_due_since is None


def test_cancelling_at_once_and_pausing_end_the_plan(paddle, ada, web):
    subscription = subscribe(paddle, web, ada, "plus")
    notify(paddle, "subscription.canceled", paddle.end(subscription))
    assert plan_of(ada) == "free"

    eve = make_user("eve")
    hers = subscribe(paddle, login(eve), eve, "pro")
    notify(paddle, "subscription.paused", paddle.set_status(hers, "paused"))
    assert plan_of(eve) == "free"
    notify(paddle, "subscription.resumed", paddle.set_status(hers, "active"))
    assert plan_of(eve) == "pro"


def test_trialing_gives_the_plan(paddle, ada, web):
    checkout = web.post("/api/billing/checkout/", {"plan": "plus"}, format="json").json()
    entity = paddle.subscribe(
        checkout["customer_id"], "pri_plus", checkout["custom_data"], status="trialing"
    )
    notify(paddle, "subscription.trialing", entity)
    assert plan_of(ada) == "plus"


def test_downgrade_is_credited_to_the_next_invoice(paddle, ada, web):
    subscription = subscribe(paddle, web, ada, "pro")
    preview = web.post(
        "/api/billing/change-plan/", {"plan": "starter", "preview": True}, format="json"
    )
    assert preview.status_code == 200, preview.content
    answer = preview.json()
    assert (answer["plan"]["key"], answer["upgrade"]) == ("starter", False)
    assert answer["now"] == {"action": "credit", "amount": 3.1, "currency": "EUR"}
    assert plan_of(ada) == "pro"  # a preview changes nothing
    assert paddle.called("change_price")[-1] == (
        subscription,
        "pri_starter",
        "prorated_next_billing_period",
        True,
    )
    done = web.post("/api/billing/change-plan/", {"plan": "starter"}, format="json")
    assert done.status_code == 200
    assert plan_of(ada) == "starter"
    assert paddle.called("change_price")[-1][2:] == ("prorated_next_billing_period", False)


def test_upgrade_preview_says_what_is_charged_now(paddle, ada, web):
    subscribe(paddle, web, ada, "starter")
    answer = web.post(
        "/api/billing/change-plan/", {"plan": "plus", "preview": True}, format="json"
    ).json()
    assert answer["upgrade"] is True
    assert answer["now"] == {"action": "charge", "amount": 4.5, "currency": "EUR"}
    assert answer["plan"]["price"] == 14


def test_upgrade_below_paddles_minimum_charge_is_billed_with_the_next_invoice(paddle, ada, web):
    # E.g. a credit from an earlier downgrade leaves less than 70 cents to charge now: Paddle
    # refuses an immediate charge, so the rest goes onto the next invoice.
    subscription = subscribe(paddle, web, ada, "plus")
    paddle.below_charge_limit = True
    preview = web.post("/api/billing/change-plan/", {"plan": "pro", "preview": True}, format="json")
    assert preview.status_code == 200, preview.content
    assert plan_of(ada) == "plus"
    done = web.post("/api/billing/change-plan/", {"plan": "pro"}, format="json")
    assert done.status_code == 200, done.content
    assert plan_of(ada) == "pro"
    assert [call[2:] for call in paddle.called("change_price")[-2:]] == [
        ("prorated_immediately", False),
        ("prorated_next_billing_period", False),
    ]
    assert paddle.called("change_price")[-1][:2] == (subscription, "pri_pro")


def test_other_paddle_refusals_of_a_plan_change_are_not_retried(paddle, ada, web):
    subscribe(paddle, web, ada, "plus")
    paddle.failing["change_price"] = PaddleError("Paddle answered HTTP 400 bad", 400, "bad")
    refused = web.post("/api/billing/change-plan/", {"plan": "pro"}, format="json")
    assert refused.status_code >= 400
    assert len(paddle.called("change_price")) == 1
    assert plan_of(ada) == "plus"


def test_change_plan_refusals(paddle, ada, web):
    assert web.post("/api/billing/change-plan/", {"plan": "pro"}, format="json").status_code == 409
    subscribe(paddle, web, ada, "plus")
    same = web.post("/api/billing/change-plan/", {"plan": "plus"}, format="json")
    assert same.status_code == 409 and "already have this plan" in same.json()["detail"]
    for plan in ("free", "gold"):
        assert (
            web.post("/api/billing/change-plan/", {"plan": plan}, format="json").status_code == 409
        )
    assert web.post("/api/billing/change-plan/", {}, format="json").status_code == 400
    # Paddle refuses (e.g. the card was declined): nothing changes here.
    paddle.refuses("change_price", "subscription_payment_declined")
    declined = web.post("/api/billing/change-plan/", {"plan": "pro"}, format="json")
    assert declined.status_code == 503
    assert "subscription_payment_declined" in declined.json()["detail"]
    assert plan_of(ada) == "plus"
    paddle.up()
    # While a cancellation is scheduled, the plan cannot be changed.
    assert web.post("/api/billing/cancel/").status_code == 200
    scheduled = web.post("/api/billing/change-plan/", {"plan": "pro"}, format="json")
    assert scheduled.status_code == 409 and "Continue it first" in scheduled.json()["detail"]


def test_cancel_and_reactivate(paddle, ada, web):
    assert web.post("/api/billing/cancel/").status_code == 409  # nothing to cancel
    assert web.post("/api/billing/reactivate/").status_code == 409
    subscription = subscribe(paddle, web, ada, "plus")
    assert web.post("/api/billing/reactivate/").status_code == 409  # not cancelled
    assert web.post("/api/billing/cancel/").status_code == 200
    assert web.post("/api/billing/cancel/").status_code == 409  # cancelled already
    assert plan_of(ada) == "plus"
    back = web.post("/api/billing/reactivate/")
    assert back.status_code == 200, back.content
    assert paddle.called("remove_scheduled_change") == [(subscription,)]
    current = back.json()["subscription"]
    assert current["ends_at"] is None and current["renews_at"] is not None
    assert (current["can_cancel"], current["can_reactivate"]) == (True, False)
    # Paddle is down: the answer says so and nothing changes.
    paddle.down("cancel_subscription")
    assert web.post("/api/billing/cancel/").status_code == 503
    assert BillingCustomer.objects.get(user=ada).cancel_at is None


def test_portal_link_for_the_own_customer_only(paddle, ada, web):
    assert web.post("/api/billing/portal/").status_code == 409  # no customer yet
    subscription = subscribe(paddle, web, ada)
    response = web.post("/api/billing/portal/")
    assert response.status_code == 200
    assert response.json()["url"].startswith("https://sandbox-customer-portal.paddle.com/")
    assert paddle.called("create_portal_session") == [(current_customer(ada), [subscription])]


def test_refresh_asks_paddle_when_the_notification_did_not_arrive(paddle, ada, web):
    checkout = web.post("/api/billing/checkout/", {"plan": "plus"}, format="json").json()
    paddle.subscribe(checkout["customer_id"], "pri_plus", checkout["custom_data"])
    assert plan_of(ada) == "free"  # paid, but the notification is still on its way
    response = web.post("/api/billing/refresh/")
    assert response.status_code == 200, response.content
    assert response.json()["plan"]["key"] == "plus"
    assert plan_of(ada) == "plus"
    # A subscription of the customer that names another account is not taken.
    eve = make_user("eve")
    eve_web = login(eve)
    hers = eve_web.post("/api/billing/checkout/", {"plan": "pro"}, format="json").json()
    paddle.subscribe(hers["customer_id"], "pri_pro", {"user_id": str(ada.pk)})
    assert eve_web.post("/api/billing/refresh/").json()["plan"]["key"] == "free"
    # Without a customer there is nothing to ask.
    bob_web = login(make_user("bob"))
    assert bob_web.post("/api/billing/refresh/").json()["subscription"] is None


def test_changing_the_email_address_tells_paddle(paddle, ada, web):
    subscribe(paddle, web, ada)
    response = web.patch("/api/auth/account/", {"email": "ada@new.example"}, format="json")
    assert response.status_code == 200
    assert paddle.called("update_customer") == [
        (current_customer(ada), {"email": "ada@new.example"})
    ]
    # If Paddle is down the address changes here all the same.
    paddle.down("update_customer")
    assert (
        web.patch("/api/auth/account/", {"email": "ada@x.example"}, format="json").status_code
        == 200
    )


# --------------------------------------------------------------------------- #
# Unlimited accounts, plans set by hand
# --------------------------------------------------------------------------- #


def test_unlimited_accounts_need_no_plan_and_are_never_touched(paddle, ada, web):
    budgets.set_unlimited(ada, True)
    current = state(web)
    assert (current["unlimited"], current["can_subscribe"]) == (True, False)
    refused = web.post("/api/billing/checkout/", {"plan": "plus"}, format="json")
    assert refused.status_code == 409 and "no limits" in refused.json()["detail"]
    assert paddle.calls == []

    # An account that subscribed and was marked unlimited later keeps what it has.
    eve = make_user("eve")
    subscription = subscribe(paddle, login(eve), eve, "plus")
    budgets.set_unlimited(eve, True)
    notify(paddle, "subscription.canceled", paddle.end(subscription))
    assert UserProfile.objects.get(user=eve).plan == "plus"
    assert budgets.limits_for(eve).unlimited is True
    billing.sync_all()
    assert UserProfile.objects.get(user=eve).plan == "plus"


def test_staff_accounts_are_unlimited_by_default(paddle):
    boss = make_user("boss", is_staff=True)
    assert state(login(boss))["unlimited"] is True


def test_a_plan_set_by_the_operator_stays_until_the_subscription_changes(paddle, ada, web):
    subscription = subscribe(paddle, web, ada, "starter")
    notify(paddle, "subscription.canceled", paddle.end(subscription))
    assert plan_of(ada) == "free"
    budgets.set_plan(ada, "pro")  # a present
    billing.sync_all()
    state(web)
    assert plan_of(ada) == "pro"
    # An account that never bought anything is none of billing's business either.
    bob = make_user("bob")
    budgets.set_plan(bob, "plus")
    login(bob).post("/api/billing/checkout/", {"plan": "starter"}, format="json")
    billing.sync_all()
    assert plan_of(bob) == "plus"


# --------------------------------------------------------------------------- #
# Once a day
# --------------------------------------------------------------------------- #


def test_daily_sync_catches_a_lost_notification(paddle, ada, web):
    subscription = subscribe(paddle, web, ada, "plus")
    paddle.end(subscription)  # the notification never arrives
    assert plan_of(ada) == "plus"
    lines: list[str] = []
    result = billing.sync_all(log=lines.append)
    assert (result.checked, result.plans_changed, result.errors) == (1, 1, 0)
    assert plan_of(ada) == "free"
    assert lines == [f"Account #{ada.pk}: plan plus -> free (canceled)"]
    # Ended subscriptions are not asked about again.
    paddle.calls.clear()
    assert billing.sync_all().plans_changed == 0
    assert paddle.calls == []


def test_daily_sync_goes_on_when_paddle_fails_for_one_account(paddle, ada, web):
    subscribe(paddle, web, ada, "plus")
    eve = make_user("eve")
    hers = subscribe(paddle, login(eve), eve, "pro")
    paddle.end(hers)
    paddle.down("get_subscription")
    result = billing.sync_all()
    assert (result.checked, result.errors, result.plans_changed) == (2, 2, 0)
    assert (plan_of(ada), plan_of(eve)) == ("plus", "pro")  # nobody loses a plan for an outage
    paddle.up()
    assert billing.sync_all().plans_changed == 1
    assert (plan_of(ada), plan_of(eve)) == ("plus", "free")


def test_billing_sync_command(paddle, ada, web, plans):
    subscribe(paddle, web, ada, "plus")
    out = StringIO()
    call_command("billing_sync", stdout=out)
    assert "checked 1 subscriptions, changed 0 plans, 0 overage charges, 0 errors" in out.getvalue()

    out = StringIO()
    call_command("billing_sync", "--check", stdout=out)
    text = out.getvalue()
    assert "Paid plans are ON (sandbox)." in text and "Everything fits." in text
    assert "plus: pri_plus = 1400 EUR" in text
    assert "pdl_sdbx_apikey_secret" not in text and SECRET not in text

    plans.PADDLE_PRICES = "starter=pri_starter,plus=pri_pro,pro=pri_gone"
    out = StringIO()
    call_command("billing_sync", "--check", stdout=out)
    text = out.getvalue()
    assert "plus: pri_pro: it costs 4500 (smallest unit), the plan says 1400." in text
    assert "pro: pri_gone: Paddle answered HTTP 404" in text
    assert "2 problem(s)" in text


def test_billing_sync_command_with_billing_off(plans):
    out = StringIO()
    call_command("billing_sync", stdout=out)
    assert "paid plans are off" in out.getvalue()
    out = StringIO()
    call_command("billing_sync", "--check", stdout=out)
    assert "Paid plans are off: no BRICKDNN_PADDLE_… setting is set." in out.getvalue()
    plans.PADDLE_ENV = "live"
    out = StringIO()
    call_command("billing_sync", "--check", stdout=out)
    assert "Paid plans are OFF until this is fixed" in out.getvalue()


def test_a_sandbox_subscription_ends_when_the_server_goes_live(paddle, plans, ada, web):
    subscribe(paddle, web, ada, "pro")
    plans.PADDLE_ENV = "live"
    plans.PADDLE_CLIENT_TOKEN = "live_0123456789abcdef"
    paddle.calls.clear()
    billing.sync_all()
    assert plan_of(ada) == "free"
    assert paddle.calls == []  # the live API does not know the sandbox subscription
    current = state(web)
    assert current["subscription"] is None and current["can_subscribe"] is True
    # A new checkout makes a live customer for the account.
    paddle.customers.clear()
    checkout = web.post("/api/billing/checkout/", {"plan": "plus"}, format="json")
    assert checkout.status_code == 200
    assert len(paddle.called("create_customer")) == 1
    customer = BillingCustomer.objects.get(user=ada)
    assert (customer.environment, customer.subscription_id, customer.status) == ("live", None, "")


# --------------------------------------------------------------------------- #
# Overage
# --------------------------------------------------------------------------- #


def test_overage_of_a_finished_month_is_billed_once(paddle, ada, web):
    subscription = subscribe(paddle, web, ada, "plus")
    use_overage(ada, 2.5)
    use_overage(ada, 0.75)
    lines: list[str] = []
    assert billing.sync_all(log=lines.append).charges == 1
    month = last_month()
    charge = OverageCharge.objects.get()
    assert (charge.status, charge.amount, charge.currency) == ("charged", Decimal("1.30"), "EUR")
    assert (charge.period, charge.subscription_id) == (month, subscription)
    assert charge.cpu_seconds == pytest.approx(3.25 * 3600)
    assert lines == [f"Account #{ada.pk}: overage of {month:%Y-%m}, 1.30 EUR: charged"]
    (billed,) = paddle.charges
    assert billed[0] == subscription
    assert billed[1]["product_id"] == "pro_overage"
    assert (billed[1]["amount_minor"], billed[1]["currency"]) == (130, "EUR")
    assert billed[1]["immediately"] is True
    assert billed[1]["description"] == (
        f"Brick DNN: 3.25 CPU hours beyond the Plus plan in {month:%B %Y} (0.40 EUR per CPU hour)"
    )
    assert billed[1]["custom_data"] == {
        "brickdnn_overage_charge": str(charge.pk),
        "month": f"{month:%Y-%m}",
    }
    # The next runs find nothing left to bill.
    assert billing.sync_all().charges == 0
    assert billing.sync_all().charges == 0
    assert len(paddle.charges) == 1 and OverageCharge.objects.count() == 1


def test_overage_is_billed_up_to_the_accounts_cap(paddle, ada, web):
    subscribe(paddle, web, ada, "plus")
    UserProfile.objects.filter(user=ada).update(overage_enabled=True, overage_cap_cpu_hours=5)
    use_overage(ada, 5.2)  # the last training ran a little over the cap
    billing.sync_all()
    assert OverageCharge.objects.get().amount == Decimal("2.00")
    assert paddle.charges[0][1]["amount_minor"] == 200


def test_the_cap_cannot_go_below_overage_that_is_not_billed_yet(paddle, ada, web):
    subscribe(paddle, web, ada, "plus")
    UserProfile.objects.filter(user=ada).update(overage_enabled=True, overage_cap_cpu_hours=5)
    use_overage(ada, 3)  # last month, the daily job has not billed it yet

    def set_cap(hours):
        return web.patch("/api/account/usage/", {"overage_cap_cpu_hours": hours}, format="json")

    response = set_cap(1)
    assert response.status_code == 400
    assert "3.00 CPU hours" in response.data["overage_cap_cpu_hours"][0]
    assert set_cap(3).status_code == 200  # down to what is used
    billing.sync_all()
    assert OverageCharge.objects.get().amount == Decimal("1.20")  # 3 h at 0.40
    # Billed: that month no longer holds the cap up.
    assert set_cap(1).status_code == 200


def test_the_running_month_is_billed_only_when_the_subscription_ends(paddle, ada, web):
    subscribe(paddle, web, ada, "plus")
    use_overage(ada, 3, period=budgets.period_of())
    assert billing.sync_all().charges == 0
    assert paddle.charges == []
    # Cancelled to the end of the period: no renewal will come, so it is billed now.
    assert web.post("/api/billing/cancel/").status_code == 200
    assert [charge[1]["amount_minor"] for charge in paddle.charges] == [120]
    assert OverageCharge.objects.get().status == "charged"
    # More cannot be used while the cancellation stands, and nothing is billed twice.
    assert billing.overage_allowed(ada) is False
    assert billing.sync_all().charges == 0
    # Reactivated, more overage, the month ends: only the rest is billed.
    assert web.post("/api/billing/reactivate/").status_code == 200
    assert billing.overage_allowed(ada) is True
    UsageRecord.objects.filter(user=ada).update(overage=5 * 3600)
    assert billing.sync_all().charges == 0
    customer = BillingCustomer.objects.get(user=ada)
    next_month = timezone.now() + timedelta(days=32)
    assert len(billing.charge_due_overage(customer, now=next_month)) == 1
    assert [charge[1]["amount_minor"] for charge in paddle.charges] == [120, 80]
    assert billing.charge_due_overage(customer, now=next_month) == []


def test_small_amounts_are_not_billed(paddle, ada, web):
    subscribe(paddle, web, ada, "plus")
    use_overage(ada, 1)  # 0.40 EUR, below BRICKDNN_PADDLE_OVERAGE_MIN_CHARGE
    assert billing.sync_all().charges == 0
    assert paddle.charges == [] and not OverageCharge.objects.exists()


def test_a_charge_without_an_answer_is_never_billed_again_by_itself(paddle, ada, web):
    subscribe(paddle, web, ada, "plus")
    use_overage(ada, 10)
    # Paddle does not answer (or the server crashes right after asking): the outcome is unknown.
    paddle.down("create_charge")
    assert billing.sync_all().charges == 1
    charge = OverageCharge.objects.get()
    assert charge.status == "pending" and "No answer from Paddle" in charge.detail
    paddle.up()
    paddle.calls.clear()
    for _ in range(3):
        assert billing.sync_all().charges == 0
    assert paddle.called("create_charge") == []
    assert OverageCharge.objects.count() == 1

    # The operator looked in Paddle: it was not billed. The next run bills it.
    model_admin = admin_site._registry[OverageCharge]
    model_admin.message_user = lambda *args, **kwargs: None
    model_admin.mark_not_charged(None, OverageCharge.objects.all())
    assert OverageCharge.objects.get().status == "failed"
    assert billing.sync_all().charges == 1
    assert [c.status for c in OverageCharge.objects.order_by("pk")] == ["failed", "charged"]
    assert len(paddle.called("create_charge")) == 1
    assert billing.sync_all().charges == 0


def test_a_charge_the_operator_confirmed_is_not_billed_again(paddle, ada, web):
    subscribe(paddle, web, ada, "plus")
    use_overage(ada, 10)
    paddle.down("create_charge")
    billing.sync_all()
    paddle.up()
    model_admin = admin_site._registry[OverageCharge]
    model_admin.message_user = lambda *args, **kwargs: None
    model_admin.mark_charged(None, OverageCharge.objects.all())
    assert OverageCharge.objects.get().status == "charged"
    assert billing.sync_all().charges == 0
    assert paddle.charges == []


def test_the_pending_row_exists_before_paddle_is_asked(paddle, ada, web):
    subscribe(paddle, web, ada, "plus")
    use_overage(ada, 10)
    seen: list[list[str]] = []
    original = paddle.create_charge

    def create_charge(subscription_id, **charge):
        seen.append(list(OverageCharge.objects.values_list("status", flat=True)))
        return original(subscription_id, **charge)

    paddle.create_charge = create_charge
    billing.sync_all()
    assert seen == [["pending"]]
    assert OverageCharge.objects.get().status == "charged"


def test_a_refused_charge_is_tried_again_a_few_times(paddle, ada, web):
    subscribe(paddle, web, ada, "plus")
    use_overage(ada, 10)
    paddle.refuses("create_charge", "subscription_payment_declined")
    for _ in range(billing.MAX_CHARGE_ATTEMPTS + 2):
        billing.sync_all()
    assert OverageCharge.objects.filter(status="failed").count() == billing.MAX_CHARGE_ATTEMPTS
    assert len(paddle.called("create_charge")) == billing.MAX_CHARGE_ATTEMPTS
    assert "subscription_payment_declined" in OverageCharge.objects.first().detail


def test_a_refused_charge_that_works_the_next_day(paddle, ada, web):
    subscribe(paddle, web, ada, "plus")
    use_overage(ada, 10)
    paddle.refuses("create_charge", "subscription_payment_declined")
    billing.sync_all()
    paddle.up()
    billing.sync_all()
    billing.sync_all()
    assert [c.status for c in OverageCharge.objects.order_by("pk")] == ["failed", "charged"]
    assert len(paddle.charges) == 1 and paddle.charges[0][1]["amount_minor"] == 400


def test_who_is_not_billed_for_overage(paddle, plans, ada, web):
    subscription = subscribe(paddle, web, ada, "plus")
    use_overage(ada, 10)
    # While a payment is open, nothing more is billed.
    notify(paddle, "subscription.past_due", paddle.set_status(subscription, "past_due"))
    assert billing.sync_all().charges == 0
    assert billing.overage_allowed(ada) is False
    notify(paddle, "subscription.updated", paddle.set_status(subscription, "active"))
    # Unlimited accounts are not billed.
    budgets.set_unlimited(ada, True)
    assert billing.sync_all().charges == 0
    budgets.set_unlimited(ada, None)
    # Without an overage product nothing is billed (and budgets do not allow overage).
    plans.PADDLE_OVERAGE_PRODUCT = ""
    assert billing.sync_all().charges == 0
    assert billing.overage_allowed(ada) is False
    assert APIClient().get("/api/billing/").json()["overage_billed"] is False
    plans.PADDLE_OVERAGE_PRODUCT = "pro_overage"
    assert paddle.charges == []
    assert billing.overage_allowed(ada) is True
    assert billing.sync_all().charges == 1
    # Accounts without a subscription cannot use overage that nobody could bill.
    assert billing.overage_allowed(make_user("bob")) is False


def test_overage_is_up_to_the_operator_while_billing_is_off(plans, ada):
    assert billing.overage_allowed(ada) is True


def test_overage_that_was_not_billed_when_the_subscription_ended_is_written_off(paddle, ada, web):
    subscription = subscribe(paddle, web, ada, "plus")
    use_overage(ada, 6, period=budgets.period_of())
    # Cancelled in Paddle's portal at once: the notification is all this server sees.
    notify(paddle, "subscription.canceled", paddle.end(subscription))
    charge = OverageCharge.objects.get()
    assert (charge.status, charge.amount) == ("skipped", Decimal("2.40"))
    assert "ended before" in charge.detail
    assert paddle.charges == []
    # It is not written off twice.
    notify(paddle, "subscription.canceled", paddle.entity(subscription))
    billing.sync_all()
    assert OverageCharge.objects.count() == 1


# --------------------------------------------------------------------------- #
# Together with budgets: where to get more, and overage only where it can be billed
# --------------------------------------------------------------------------- #


def use_up_the_free_allowance(user) -> None:
    UsageRecord.objects.create(
        user=user, kind="training", quantity=0.5 * 3600, period=budgets.period_of()
    )


def test_budget_messages_point_to_the_pricing_page(paddle, plans, ada, web):
    use_up_the_free_allowance(ada)
    with pytest.raises(budgets.BudgetExceeded) as caught:
        budgets.check_compute(budgets.Actor(ada))
    assert caught.value.message.endswith("Higher plans include more: /pricing")
    assert caught.value.data["upgrade_url"] == "/pricing"
    assert web.get("/api/account/usage/").json()["upgrade_url"] == "/pricing"
    # With the site's address, scripts and AI assistants get a link they can pass on.
    plans.PUBLIC_URL = "https://brickdnn.com"
    with pytest.raises(budgets.BudgetExceeded) as caught:
        budgets.check_compute(budgets.Actor(ada))
    assert "Higher plans include more: https://brickdnn.com/pricing" in caught.value.message
    assert caught.value.body()["upgrade_url"] == "https://brickdnn.com/pricing"
    # A full storage names the pricing page too.
    with pytest.raises(budgets.StorageQuotaExceeded) as full:
        budgets.check_storage(ada, adding=2 * 1024**3)
    assert "choose a plan with more storage: https://brickdnn.com/pricing" in full.value.message
    assert full.value.data["upgrade_url"] == "https://brickdnn.com/pricing"


def test_budget_messages_without_paid_plans_are_as_before(plans, ada, web):
    use_up_the_free_allowance(ada)
    with pytest.raises(budgets.BudgetExceeded) as caught:
        budgets.check_compute(budgets.Actor(ada))
    assert caught.value.message.endswith("Higher plans include more (see your account page).")
    assert "upgrade_url" not in caught.value.data
    assert web.get("/api/account/usage/").json()["upgrade_url"] is None
    with pytest.raises(budgets.StorageQuotaExceeded) as full:
        budgets.check_storage(ada, adding=2 * 1024**3)
    assert full.value.message.endswith("ask the operator of this server for more space.")
    assert "upgrade_url" not in full.value.data


def test_overage_needs_a_subscription_it_can_be_billed_to(paddle, ada, web):
    def switch_on():
        return web.patch("/api/account/usage/", {"overage_enabled": True}, format="json")

    # A paid plan the operator gave by hand: nothing to bill overage to.
    budgets.set_plan(ada, "plus")
    refused = switch_on()
    assert refused.status_code == 400
    assert "needs a subscription" in refused.json()["overage_enabled"][0]
    assert web.get("/api/account/usage/").json()["overage"]["available"] is False

    budgets.set_plan(ada, "")
    subscribe(paddle, web, ada, "plus")
    answer = switch_on()
    assert answer.status_code == 200, answer.content
    overage = answer.json()["overage"]
    assert (overage["available"], overage["enabled"]) == (True, True)
    assert budgets.limits_for(ada).overage is True
    # While the subscription is cancelled, no new overage builds up; it comes back with it.
    assert web.post("/api/billing/cancel/").status_code == 200
    assert budgets.limits_for(ada).overage is False
    assert web.post("/api/billing/reactivate/").status_code == 200
    assert budgets.limits_for(ada).overage is True


def test_overage_without_paid_plans_is_as_before(plans, ada, web):
    budgets.set_plan(ada, "plus")
    answer = web.patch("/api/account/usage/", {"overage_enabled": True}, format="json")
    assert answer.status_code == 200
    assert budgets.limits_for(ada).overage is True


def test_notifications_are_not_logged_as_agent_activity(paddle, ada, web):
    subscription = subscribe(paddle, web, ada, "plus")
    notify(paddle, "subscription.updated", paddle.entity(subscription))
    state(web)
    assert post_event({"event_id": "evt_x"}, secret="wrong").status_code == 403
    assert AgentAction.objects.count() == 0


# --------------------------------------------------------------------------- #
# Deleting the account, the export, old accounts, the admin
# --------------------------------------------------------------------------- #


def delete(web: APIClient, user):
    return web.post(
        "/api/auth/account/delete/",
        {"password": PASSWORD, "username": user.username},
        format="json",
    )


def test_deleting_the_account_cancels_the_subscription_at_once(paddle, ada, web):
    subscription = subscribe(paddle, web, ada, "plus")
    use_overage(ada, 5, period=budgets.period_of())
    assert delete(web, ada).status_code == 204
    assert paddle.called("cancel_subscription") == [(subscription, True)]
    assert paddle.subscriptions[subscription]["status"] == "canceled"
    # What was used beyond the allowance is billed before.
    assert [charge[1]["amount_minor"] for charge in paddle.charges] == [200]
    assert not User.objects.filter(pk=ada.pk).exists()
    assert not BillingCustomer.objects.exists()
    assert not BillingEvent.objects.exists()
    assert not OverageCharge.objects.exists()
    # Paddle's notification about the cancellation finds no account and changes nothing.
    late = notify(paddle, "subscription.canceled", paddle.entity(subscription))
    assert late["outcome"] == "unknown"


def test_the_account_is_not_deleted_if_the_subscription_cannot_be_cancelled(paddle, ada, web):
    subscription = subscribe(paddle, web, ada, "plus")
    paddle.down("cancel_subscription")
    response = delete(web, ada)
    assert response.status_code == 503
    assert "could not be cancelled" in response.json()["detail"]
    assert "was not deleted" in response.json()["detail"]
    assert User.objects.filter(pk=ada.pk).exists()
    assert paddle.subscriptions[subscription]["status"] == "active"
    assert plan_of(ada) == "plus"
    assert state(web)["subscription"]["status"] == "active"  # still logged in
    paddle.up()
    assert delete(web, ada).status_code == 204
    assert paddle.subscriptions[subscription]["status"] == "canceled"


def test_deleting_an_account_whose_subscription_is_gone_at_paddle(paddle, ada, web):
    subscribe(paddle, web, ada, "plus")
    paddle.refuses("cancel_subscription", "not_found", 404)
    assert delete(web, ada).status_code == 204
    assert not User.objects.filter(pk=ada.pk).exists()


def test_deleting_accounts_without_a_live_subscription_asks_paddle_nothing(paddle, ada, web):
    subscription = subscribe(paddle, web, ada, "plus")
    notify(paddle, "subscription.canceled", paddle.end(subscription))
    paddle.calls.clear()
    assert delete(web, ada).status_code == 204
    bob = make_user("bob")
    assert delete(login(bob), bob).status_code == 204
    assert paddle.calls == []


def test_deleting_a_user_another_way_cancels_too(paddle, ada, web):
    subscription = subscribe(paddle, web, ada, "plus")
    paddle.down("cancel_subscription")
    with pytest.raises(billing.BillingUnavailable), transaction.atomic():
        User.objects.get(pk=ada.pk).delete()  # e.g. in the Django admin
    assert User.objects.filter(pk=ada.pk).exists()
    paddle.up()
    User.objects.get(pk=ada.pk).delete()
    assert paddle.subscriptions[subscription]["status"] == "canceled"


def test_inactive_accounts_with_a_subscription_are_not_purged(paddle, ada, web):
    subscribe(paddle, web, ada, "plus")
    eve = make_user("eve")
    long_ago = timezone.now() - timedelta(days=1000)
    User.objects.all().update(date_joined=long_ago, last_login=long_ago)
    UserProfile.objects.all().update(last_active_at=long_ago)
    cutoff = timezone.now() - timedelta(days=730)
    assert list(retention.inactive_users(cutoff)) == [eve]
    result = retention.purge(0, 730)
    assert result.accounts == 1
    assert User.objects.filter(pk=ada.pk).exists()
    assert paddle.called("cancel_subscription") == []


def export(web: APIClient) -> zipfile.ZipFile:
    response = web.get("/api/auth/account/export/")
    assert response.status_code == 200
    return zipfile.ZipFile(io.BytesIO(b"".join(response.streaming_content)))


def test_export_holds_the_billing_records(paddle, ada, web):
    subscription = subscribe(paddle, web, ada, "plus")
    use_overage(ada, 10)
    billing.sync_all()
    archive = export(web)
    assert "billing.json" in archive.namelist()
    assert "billing.json" in archive.read("README.txt").decode()
    data = json.loads(archive.read("billing.json"))
    assert "Paddle" in data["payment_provider"]
    customer = data["customer"]
    assert customer["paddle_customer_id"] == current_customer(ada)
    assert customer["paddle_subscription_id"] == subscription
    assert (customer["status"], customer["plan"], customer["environment"]) == (
        "active",
        "plus",
        "sandbox",
    )
    assert [n["type"] for n in data["notifications"]] == ["subscription.created"]
    (charge,) = data["overage_charges"]
    assert (charge["cpu_hours"], charge["amount"], charge["status"]) == (10.0, "4.00", "charged")
    # Nothing of anybody else, and nothing of the server's secrets.
    text = archive.read("billing.json").decode()
    assert "pdl_sdbx_apikey_secret" not in text and SECRET not in text


def test_export_without_billing_records_is_as_before(paddle):
    bob = make_user("bob")
    archive = export(login(bob))
    assert "billing.json" not in archive.namelist()
    assert "billing" not in archive.read("README.txt").decode().lower()


def test_admin_is_read_only_and_links_to_the_user(paddle, ada, web, rf):
    subscribe(paddle, web, ada, "plus")
    use_overage(ada, 10)
    billing.sync_all()
    boss = User.objects.create_superuser("boss", "boss@example.com", PASSWORD)
    request = rf.get("/admin/")
    request.user = boss
    for model in (BillingCustomer, BillingEvent, OverageCharge):
        model_admin = admin_site._registry[model]
        assert model_admin.has_add_permission(request) is False
        assert model_admin.has_change_permission(request) is False
        assert model_admin.has_view_permission(request) is True
    link = user_link(ada)
    assert link == f'<a href="/admin/auth/user/{ada.pk}/change/">ada</a>'
    assert admin_site._registry[BillingCustomer].account(BillingCustomer.objects.get()) == link
    assert user_link(None) == "-"

    admin_client = APIClient()
    admin_client.force_login(boss)
    for url in (
        "/admin/brick_dnn_core/billingcustomer/",
        "/admin/brick_dnn_core/billingevent/",
        "/admin/brick_dnn_core/overagecharge/",
        f"/admin/brick_dnn_core/billingcustomer/{BillingCustomer.objects.get().pk}/change/",
        f"/admin/brick_dnn_core/billingevent/{BillingEvent.objects.first().pk}/change/",
    ):
        response = admin_client.get(url)
        assert response.status_code == 200, url
        assert f"/admin/auth/user/{ada.pk}/change/" in response.content.decode(), url


def test_state_of_an_account_without_a_subscription(paddle, ada, web):
    assert state(web) == {
        "plan": budgets.plan_of(ada).to_dict(),
        "unlimited": False,
        "subscription": None,
        "can_subscribe": True,
        "has_billing_profile": False,
        "email_missing": False,
    }
    assert iso(None) is None
