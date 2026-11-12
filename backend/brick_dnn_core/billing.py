"""Paid plans with Paddle Billing: a subscription at Paddle sets the account's plan.

Paddle is the merchant of record: it sells the plans, takes the payments, handles the tax and
sends the invoices (paddle.py has the API calls). Brick DNN keeps for each paying account its
customer and subscription at Paddle (:class:`BillingCustomer`) and gives it the plan the
subscription pays for (``budgets.set_plan``); the limits themselves are those of budgets.py.

**Off unless configured** (:func:`config`): the environment, the client-side token, the price
of at least one plan, the API key and the webhook secret. With billing off, :func:`info` says
so, every other endpoint answers 404 and the app shows no prices.

**Buying a plan.** The account page asks for a checkout (:func:`checkout`): the account gets
a customer at Paddle with its e-mail address, and the browser opens Paddle's checkout for
that customer and the price of the plan. Nothing the browser says is trusted afterwards:
Paddle reports the subscription with a signed notification (:func:`handle_event`), and it
only counts for the account whose customer it belongs to; the plan comes from its price.

**What the status means** (:func:`entitled_plan`): ``active`` and ``trialing`` give the plan;
``past_due`` (a payment failed, Paddle retries) keeps it for ``BRICKDNN_PADDLE_GRACE_DAYS``;
``paused``, ``canceled`` or no subscription mean the default (free) plan. A cancellation to
the end of the paid period keeps the plan until Paddle reports the end. Nothing is deleted
when a plan ends; the limits of the default plan apply.

Accounts marked **unlimited** (budgets.py) need no subscription and their plan is never
touched. A plan the operator sets by hand stays until the account's subscription changes.

**Notifications** can arrive twice, late or in the wrong order: each is processed once (its
id is kept, :class:`BillingEvent`), and one that is older than what is already known about
the subscription is ignored. Once a day ``manage.py billing_sync`` (the cleanup service)
asks Paddle for every subscription again, in case a notification was lost, ends grace
periods and bills overage.

**Overage** (compute beyond the allowance, which users of a paid plan can switch on): the
usage ledger marks it per calendar month (``UsageRecord.overage``). After the month, or when
the subscription is about to end, :func:`charge_overage` bills it to the subscription as a
one-time charge, at most up to the account's cap. The charge is saved as ``pending`` before
Paddle is asked and updated with the answer: if the answer is lost (a crash, a timeout), it
stays pending and is never billed again by itself.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from datetime import timezone as dt_timezone
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from django.conf import settings
from django.core import checks
from django.db import IntegrityError, transaction
from django.db.models import Sum
from django.db.models.signals import pre_delete
from django.dispatch import receiver
from django.utils import timezone
from django.utils.dateparse import parse_datetime
from rest_framework import status
from rest_framework.exceptions import APIException, NotFound

from . import budgets
from .models import BillingCustomer, BillingEvent, OverageCharge, UsageRecord
from .paddle import PaddleClient, PaddleError, PaddleUnavailable

logger = logging.getLogger(__name__)

ENVIRONMENTS = ("sandbox", "live")
# Statuses of a subscription that gives its plan, and of one that still exists at Paddle.
ENTITLED = ("active", "trialing")
# Paddle's error for an update whose immediate charge is below its minimum charge.
BELOW_CHARGE_LIMIT = "subscription_update_transaction_balance_less_than_charge_limit"
LIVE = (*ENTITLED, "past_due", "paused")
PRICING_PATH = "/pricing"
# Paddle's error when a customer with the e-mail address exists already.
CUSTOMER_EXISTS = "customer_already_exists"
# A refused overage charge is tried again on the following days, this often in total.
MAX_CHARGE_ATTEMPTS = 5
# Currencies without a smaller unit (Paddle's amounts are in the smallest unit).
ZERO_DECIMAL = ("JPY", "KRW", "VND", "CLP")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Config:
    """The Paddle settings of this server. ``problems`` says what is missing or wrong when
    some of it is set; billing is on only without problems."""

    environment: str
    client_token: str
    api_key: str
    webhook_secrets: tuple[str, ...]
    # Plan key -> its Paddle prices; the first one is sold, the others are older prices that
    # existing subscriptions may still have.
    prices: dict[str, tuple[str, ...]]
    overage_product: str
    configured: bool
    problems: tuple[str, ...]

    @property
    def enabled(self) -> bool:
        return self.configured and not self.problems

    def price_of(self, plan: str) -> str | None:
        found = self.prices.get(plan)
        return found[0] if found else None

    def plan_of(self, price_id: str) -> str | None:
        for plan, prices in self.prices.items():
            if price_id in prices:
                return plan
        return None


def _prices(raw: str) -> tuple[dict[str, tuple[str, ...]], list[str]]:
    """``BRICKDNN_PADDLE_PRICES``: ``starter=pri_a,plus=pri_b|pri_old`` or the same as JSON
    (``{"starter": "pri_a", "plus": ["pri_b", "pri_old"]}``)."""
    problems: list[str] = []
    pairs: list[tuple[str, Any]] = []
    if raw.startswith("{"):
        try:
            parsed = json.loads(raw)
        except ValueError:
            return {}, ["BRICKDNN_PADDLE_PRICES is not valid JSON."]
        pairs = list(parsed.items()) if isinstance(parsed, dict) else []
    else:
        for part in raw.split(","):
            key, found, value = part.partition("=")
            if found:
                pairs.append((key.strip(), value.split("|")))
            elif part.strip():
                problems.append(f"BRICKDNN_PADDLE_PRICES: {part.strip()!r} is not plan=price.")
    prices: dict[str, tuple[str, ...]] = {}
    for key, value in pairs:
        ids = tuple(
            str(item).strip()
            for item in (value if isinstance(value, list) else [value])
            if str(item).strip()
        )
        if key and ids:
            prices[str(key)] = ids
    return prices, problems


def config() -> Config:
    """The settings as they are now (tests change them), with what is wrong about them."""
    environment = settings.PADDLE_ENV
    token = settings.PADDLE_CLIENT_TOKEN
    api_key = settings.PADDLE_API_KEY
    secrets = tuple(secret for secret in settings.PADDLE_WEBHOOK_SECRETS if secret)
    prices, problems = _prices(settings.PADDLE_PRICES)
    configured = bool(environment or token or api_key or secrets or settings.PADDLE_PRICES)
    if configured:
        if environment not in ENVIRONMENTS:
            problems.append("BRICKDNN_PADDLE_ENV must be sandbox or live.")
        if not token:
            problems.append("BRICKDNN_PADDLE_CLIENT_TOKEN (the client-side token) is missing.")
        elif environment in ENVIRONMENTS and token.startswith(("test_", "live_")):
            expected = "test_" if environment == "sandbox" else "live_"
            if not token.startswith(expected):
                problems.append(
                    f"BRICKDNN_PADDLE_CLIENT_TOKEN is not a {environment} token ({expected}…)."
                )
        if not api_key:
            problems.append("BRICKDNN_PADDLE_API_KEY is missing (set it in the server's .env).")
        if not secrets:
            problems.append(
                "BRICKDNN_PADDLE_WEBHOOK_SECRET is missing (set it in the server's .env)."
            )
        if not prices:
            problems.append("BRICKDNN_PADDLE_PRICES names no price (plan=pri_…).")
        try:
            known = budgets.plans()
            default = budgets.default_plan().key
        except Exception as exc:  # BRICKDNN_PLANS is broken: that error is reported elsewhere
            known, default = {}, ""
            problems.append(f"The plans could not be read: {exc}")
        for key in prices:
            if known and key not in known:
                problems.append(f"BRICKDNN_PADDLE_PRICES: there is no plan {key!r}.")
            elif key == default:
                problems.append(
                    f"BRICKDNN_PADDLE_PRICES: {key!r} is the default plan, which is not sold."
                )
        seen: dict[str, str] = {}
        for key, ids in prices.items():
            for price_id in ids:
                if price_id in seen and seen[price_id] != key:
                    problems.append(
                        f"BRICKDNN_PADDLE_PRICES: {price_id} is the price of two plans."
                    )
                seen[price_id] = key
    return Config(
        environment=environment,
        client_token=token,
        api_key=api_key,
        webhook_secrets=secrets,
        prices=prices,
        overage_product=settings.PADDLE_OVERAGE_PRODUCT,
        configured=configured,
        problems=tuple(problems),
    )


def enabled() -> bool:
    return config().enabled


def require() -> Config:
    """The configuration, or HTTP 404 when billing is off."""
    found = config()
    if not found.enabled:
        raise NotFound("This server has no paid plans.")
    return found


def client(cfg: Config | None = None) -> PaddleClient:
    """The Paddle API of this server (tests replace this function with a fake)."""
    cfg = cfg or config()
    return PaddleClient(cfg.api_key, cfg.environment, base_url=settings.PADDLE_API_URL)


@checks.register()
def check_configuration(app_configs, **kwargs) -> list[checks.CheckMessage]:
    """``manage.py check`` and every start say when billing is half configured (and off)."""
    try:
        found = config()
    except Exception as exc:  # pragma: no cover - settings broken beyond billing
        return [checks.Warning(f"Billing settings could not be read: {exc}", id="brickdnn.W001")]
    if not found.configured or not found.problems:
        return []
    return [
        checks.Warning(
            "Paid plans (Paddle) are switched off, because the settings are incomplete: "
            + " ".join(found.problems),
            hint="See docs/billing.md, or remove all BRICKDNN_PADDLE_… settings.",
            id="brickdnn.W001",
        )
    ]


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class BillingRefused(APIException):
    """The request does not fit the state of the account's subscription (409)."""

    status_code = status.HTTP_409_CONFLICT
    default_code = "billing_refused"


class BillingUnavailable(APIException):
    """Paddle did not answer or refused (503): nothing was changed here."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    default_code = "billing_unavailable"
    default_detail = (
        "The payment provider (Paddle) did not answer. Nothing was changed; please try again "
        "in a moment."
    )


def _unavailable(exc: PaddleError, what: str) -> BillingUnavailable:
    logger.warning("Paddle: %s failed: %s", what, exc)
    if isinstance(exc, PaddleUnavailable):
        return BillingUnavailable()
    code = f" ({exc.code})" if exc.code else ""
    return BillingUnavailable(
        f"The payment provider (Paddle) refused the request{code}. Nothing was changed; "
        "please try again later or contact the operator of this site."
    )


# --------------------------------------------------------------------------- #
# What the frontend shows
# --------------------------------------------------------------------------- #


def pricing_url() -> str | None:
    """Where plans can be bought (None with billing off): the address of the pricing page,
    with the site's address if it is known (for messages to scripts and AI assistants)."""
    if not enabled():
        return None
    return f"{settings.PUBLIC_URL}{PRICING_PATH}"


def info() -> dict[str, Any]:
    """``BillingInfo`` (frontend/src/api/types.ts): whether this server sells plans, what the
    browser needs for Paddle's checkout, and the plans. Public."""
    cfg = config()
    if not cfg.enabled:
        return {"enabled": False}
    default = budgets.default_plan().key
    offered = [
        {**plan.to_dict(), "purchasable": key in cfg.prices, "default": key == default}
        for key, plan in budgets.plans().items()
        if key in cfg.prices or key == default
    ]
    return {
        "enabled": True,
        "provider": "paddle",
        "environment": cfg.environment,
        "client_token": cfg.client_token,
        "currency": settings.PRICE_CURRENCY,
        # Paddle shows and charges the prices as configured there; these are for display.
        "prices_include_tax": True,
        "overage_billed": bool(cfg.overage_product),
        "plans": offered,
    }


def _grace() -> timedelta:
    return timedelta(days=max(0, settings.PADDLE_GRACE_DAYS))


def _customer(user, cfg: Config) -> BillingCustomer | None:
    """The account's customer at Paddle in the environment in use, if it has one."""
    return BillingCustomer.objects.filter(user=user, environment=cfg.environment).first()


def state(user) -> dict[str, Any]:
    """``BillingState`` (frontend/src/api/types.ts): the account's plan and subscription."""
    cfg = require()
    customer = _customer(user, cfg)
    if customer is not None:
        sync_plan(customer)  # a grace period may have ended since the last look
    unlimited = budgets.is_unlimited(user)
    subscription = None
    if customer is not None and customer.status:
        past_due = customer.status == "past_due"
        plan = budgets.plans().get(customer.plan)
        subscription = {
            "status": customer.status,
            "plan": customer.plan or None,
            "plan_name": plan.name if plan else None,
            # Renews then, unless it ends (cancelled to the end of the period).
            "renews_at": customer.next_billed_at
            if customer.status in ENTITLED and customer.cancel_at is None
            else None,
            "ends_at": customer.cancel_at if customer.status in LIVE else None,
            "ended_at": customer.canceled_at,
            "current_period_end": customer.current_period_end,
            "payment_failed": past_due or customer.payment_failed_at is not None,
            # Until then the plan is kept while Paddle retries the payment.
            "grace_until": (customer.past_due_since + _grace())
            if past_due and customer.past_due_since
            else None,
            # What the account can do now.
            "can_change_plan": customer.status in ENTITLED and customer.cancel_at is None,
            "can_cancel": customer.status in LIVE and customer.cancel_at is None,
            "can_reactivate": customer.status in LIVE and customer.cancel_at is not None,
        }
    live = customer is not None and customer.status in LIVE
    return {
        "plan": budgets.plan_of(user).to_dict(),
        "unlimited": unlimited,
        "subscription": subscription,
        "can_subscribe": not live and not unlimited,
        "has_billing_profile": customer is not None,
        "email_missing": not user.email,
    }


# --------------------------------------------------------------------------- #
# The plan an account has for its subscription
# --------------------------------------------------------------------------- #


def entitled_plan(customer: BillingCustomer, now: datetime | None = None) -> str:
    """The plan the subscription gives the account now ("" for the default plan)."""
    now = now or timezone.now()
    if customer.environment != config().environment:
        return ""  # e.g. a test subscription from the sandbox after going live
    if customer.status in ENTITLED:
        return customer.plan
    if customer.status == "past_due" and customer.past_due_since is not None:
        return customer.plan if now < customer.past_due_since + _grace() else ""
    return ""


def sync_plan(customer: BillingCustomer, now: datetime | None = None) -> bool:
    """Give the account the plan of its subscription if that changed. Returns whether the
    account's plan was changed."""
    if not customer.subscription_id:
        return False  # never bought anything: its plan is none of billing's business
    target = entitled_plan(customer, now)
    if target and target not in budgets.plans():
        logger.error(
            "Billing: account #%s pays for the plan %r, which does not exist.",
            customer.user_id,
            target,
        )
        return False
    if target == customer.granted_plan:
        return False
    user = customer.user
    if budgets.is_unlimited(user):
        return False  # unlimited accounts are never touched (nor downgraded)
    wanted = target or budgets.default_plan().key
    changed = budgets.plan_of(user).key != wanted
    if changed:
        budgets.set_plan(user, target)
        logger.info("Billing: account #%s now has the plan %r.", user.pk, wanted)
    BillingCustomer.objects.filter(pk=customer.pk).update(granted_plan=target)
    customer.granted_plan = target
    return changed


def _time(value: Any) -> datetime | None:
    parsed = parse_datetime(value) if isinstance(value, str) else None
    if parsed is not None and timezone.is_naive(parsed):
        parsed = parsed.replace(tzinfo=dt_timezone.utc)
    return parsed


def _sold_item(data: dict[str, Any], cfg: Config) -> tuple[str | None, str]:
    """``(plan key, price id)`` of the subscription's items (None: no plan has that price)."""
    first = ""
    for item in data.get("items") or []:
        price_id = str(((item or {}).get("price") or {}).get("id") or "")
        first = first or price_id
        plan = cfg.plan_of(price_id) if price_id else None
        if plan is not None:
            return plan, price_id
    return None, first


def apply_subscription(
    customer: BillingCustomer, data: dict[str, Any], at: datetime, cfg: Config
) -> tuple[str, str]:
    """Take over what Paddle says about a subscription of ``customer`` (``data``: Paddle's
    subscription entity, said at ``at``). Returns the outcome for the event log."""
    subscription_id = str(data.get("id") or "")
    new_status = str(data.get("status") or "")
    if not subscription_id or not new_status:
        return "ignored", "No subscription in the data."
    other = bool(customer.subscription_id) and customer.subscription_id != subscription_id
    if other:
        if new_status not in LIVE:
            return "ignored", "An ended subscription, not the one of the account."
        if customer.status in LIVE:
            logger.error(
                "Billing: account #%s has the subscription %s and now a second one, %s. "
                "Cancel and refund one of them in Paddle.",
                customer.user_id,
                customer.subscription_id,
                subscription_id,
            )
            return "rejected", f"The account already has {customer.subscription_id}."
    elif customer.last_event_at is not None and at < customer.last_event_at:
        return "stale", f"Known since {customer.last_event_at.isoformat()}."

    plan, price_id = _sold_item(data, cfg)
    if plan is None:
        logger.error(
            "Billing: the subscription %s of account #%s has the price %s, which is not in "
            "BRICKDNN_PADDLE_PRICES.",
            subscription_id,
            customer.user_id,
            price_id or "(none)",
        )
        plan = "" if other else customer.plan
    period = data.get("current_billing_period") or {}
    change = data.get("scheduled_change") or {}
    customer.subscription_id = subscription_id
    customer.status = new_status
    customer.plan = plan
    customer.price_id = price_id
    customer.current_period_start = _time(period.get("starts_at"))
    customer.current_period_end = _time(period.get("ends_at"))
    customer.next_billed_at = _time(data.get("next_billed_at"))
    customer.cancel_at = (
        _time(change.get("effective_at")) if change.get("action") == "cancel" else None
    )
    customer.canceled_at = _time(data.get("canceled_at"))
    if new_status == "past_due":
        customer.past_due_since = customer.past_due_since or at
    else:
        customer.past_due_since = None
        if new_status in ENTITLED:
            customer.payment_failed_at = None
    customer.last_event_at = at
    customer.save()
    sync_plan(customer)
    if new_status not in LIVE:
        write_off_overage(customer)
    return "applied", f"{new_status}, plan {plan or '-'}"[:300]


# --------------------------------------------------------------------------- #
# Notifications from Paddle (webhooks)
# --------------------------------------------------------------------------- #


class InvalidEvent(ValueError):
    """A notification that is not what Paddle sends."""


def _named_user(data: dict[str, Any]) -> str | None:
    """The account the checkout named in its custom data (the browser sent it)."""
    custom = data.get("custom_data")
    value = custom.get("user_id") if isinstance(custom, dict) else None
    return str(value) if value not in (None, "") else None


def _customer_of(data: dict[str, Any], subscription_id: str) -> BillingCustomer | None:
    """The account an entity from Paddle belongs to: the one with its subscription, else the
    one with its customer. Locked until the transaction ends."""
    rows = BillingCustomer.objects.select_for_update().select_related("user")
    found = rows.filter(subscription_id=subscription_id).first() if subscription_id else None
    customer_id = str(data.get("customer_id") or "")
    if found is None and customer_id:
        found = rows.filter(customer_id=customer_id).first()
    return found


def _process(
    event_type: str, data: dict[str, Any], at: datetime, cfg: Config
) -> tuple[str, str, BillingCustomer | None, str]:
    """``(outcome, note, account, subscription id)`` of one event."""
    if event_type.startswith("subscription."):
        subscription_id = str(data.get("id") or "")
    elif event_type.startswith("transaction."):
        subscription_id = str(data.get("subscription_id") or "")
    else:
        return "ignored", "", None, ""
    customer = _customer_of(data, subscription_id)
    if customer is None:
        # Not bought through this site's checkout (or its account was deleted).
        logger.warning(
            "Billing: %s for the Paddle customer %s, which no account has.",
            event_type,
            data.get("customer_id"),
        )
        return "unknown", f"Customer {data.get('customer_id')}"[:300], None, subscription_id
    # The subscription only counts for the account that owns its customer at Paddle, which
    # this server created: neither the customer nor the account named by the checkout may
    # be another one.
    if str(data.get("customer_id") or "") != customer.customer_id:
        note = f"Belongs to the customer {data.get('customer_id')}, not {customer.customer_id}."
        logger.error("Billing: %s rejected for account #%s: %s", event_type, customer.user_id, note)
        return "rejected", note[:300], customer, subscription_id
    named = _named_user(data)
    if named is not None and named != str(customer.user_id):
        note = f"The checkout named account #{named}."
        logger.error("Billing: %s rejected for account #%s: %s", event_type, customer.user_id, note)
        return "rejected", note[:300], customer, subscription_id
    if customer.environment != cfg.environment:
        return "ignored", "From another Paddle environment.", customer, subscription_id

    if event_type.startswith("subscription."):
        outcome, note = apply_subscription(customer, data, at, cfg)
        return outcome, note, customer, subscription_id
    if subscription_id and subscription_id != customer.subscription_id:
        return "ignored", "", customer, subscription_id
    if event_type in ("transaction.payment_failed", "transaction.past_due"):
        if customer.payment_failed_at is None or at > customer.payment_failed_at:
            customer.payment_failed_at = at
            customer.save(update_fields=["payment_failed_at", "updated_at"])
        return "applied", "A payment failed.", customer, subscription_id
    if event_type in ("transaction.completed", "transaction.paid"):
        if customer.payment_failed_at is not None and at >= customer.payment_failed_at:
            customer.payment_failed_at = None
            customer.save(update_fields=["payment_failed_at", "updated_at"])
        return "applied", "Paid.", customer, subscription_id
    return "ignored", "", customer, subscription_id


def handle_event(payload: Any) -> tuple[BillingEvent, bool]:
    """Process a notification whose signature was verified. Returns the log entry and whether
    it is new (False: the event was processed before; nothing happens). Raises
    :class:`InvalidEvent`."""
    cfg = require()
    if not isinstance(payload, dict):
        raise InvalidEvent("The notification is not an object.")
    event_id = payload.get("event_id")
    event_type = payload.get("event_type")
    at = _time(payload.get("occurred_at"))
    data = payload.get("data")
    if not (isinstance(event_id, str) and event_id and isinstance(event_type, str) and at):
        raise InvalidEvent("The notification has no event_id, event_type or occurred_at.")
    if len(event_id) > 64 or len(event_type) > 60:
        raise InvalidEvent("The event id or type is too long.")
    if not isinstance(data, dict):
        data = {}
    with transaction.atomic():
        try:
            with transaction.atomic():
                event = BillingEvent.objects.create(
                    event_id=event_id, event_type=event_type, occurred_at=at, outcome="ignored"
                )
        except IntegrityError:
            return BillingEvent.objects.get(event_id=event_id), False
        outcome, note, customer, subscription_id = _process(event_type, data, at, cfg)
        event.outcome = outcome
        event.note = note
        event.user_id = customer.user_id if customer is not None else None
        event.subscription_id = subscription_id[:64]
        event.save()
    return event, True


# --------------------------------------------------------------------------- #
# Buying, changing and cancelling (the account page)
# --------------------------------------------------------------------------- #


def ensure_customer(user, cfg: Config) -> BillingCustomer:
    """The account's customer at Paddle, created with its e-mail address if needed."""
    existing = BillingCustomer.objects.filter(user=user).first()
    if existing is not None and existing.environment == cfg.environment:
        return existing
    if not user.email:
        raise BillingRefused("Add an e-mail address to your account first: Paddle needs one.")
    if existing is not None:
        sync_plan(existing)  # a plan from the other environment (a sandbox test) ends here
    mine = {"user_id": str(user.pk)}
    api = client(cfg)
    try:
        try:
            data = api.create_customer(user.email, mine)
        except PaddleError as exc:
            if exc.code != CUSTOMER_EXISTS and exc.status != 409:
                raise
            # Paddle knows the address already. Only a customer this server made for this
            # very account is taken over (e.g. its row was not saved last time): another one
            # may belong to somebody else, whose invoices this account must not get to see.
            data = next(
                (
                    found
                    for found in api.find_customers(user.email)
                    if _named_user(found) == mine["user_id"]
                ),
                None,
            )
            if data is None:
                logger.warning(
                    "Billing: Paddle already has a customer with the e-mail address of "
                    "account #%s that was not created for it.",
                    user.pk,
                )
                raise BillingRefused(
                    "The payment provider (Paddle) already has a billing profile for your "
                    "e-mail address, for example from an account that was deleted. Change "
                    "the e-mail address of your account, or contact the operator of this site."
                ) from None
    except PaddleError as exc:
        raise _unavailable(exc, "creating a customer") from None
    fields = {
        "environment": cfg.environment,
        "customer_id": str(data["id"]),
        # A row of the other environment (sandbox before going live) starts over.
        "subscription_id": None,
        "status": "",
        "plan": "",
        "price_id": "",
        "current_period_start": None,
        "current_period_end": None,
        "next_billed_at": None,
        "cancel_at": None,
        "canceled_at": None,
        "past_due_since": None,
        "payment_failed_at": None,
        "last_event_at": None,
    }
    try:
        with transaction.atomic():
            if existing is not None:
                for name, value in fields.items():
                    setattr(existing, name, value)
                existing.save()
                return existing
            return BillingCustomer.objects.create(user=user, **fields)
    except IntegrityError:
        raise BillingRefused(
            "This billing profile belongs to another account. Contact the operator of this site."
        ) from None


def checkout(user, plan_key: str) -> dict[str, Any]:
    """What the browser needs to open Paddle's checkout for ``plan_key``: the price, the
    account's customer and the custom data that names the account."""
    cfg = require()
    price_id = cfg.price_of(plan_key)
    if price_id is None or plan_key not in budgets.plans():
        raise BillingRefused("This plan cannot be bought.")
    if budgets.is_unlimited(user):
        raise BillingRefused("Your account has no limits; it does not need a plan.")
    current = _customer(user, cfg)
    if current is not None and current.status in LIVE:
        raise BillingRefused("You already have a subscription. Change its plan instead.")
    customer = ensure_customer(user, cfg)
    return {
        "plan": plan_key,
        "price_id": price_id,
        "customer_id": customer.customer_id,
        "custom_data": {"user_id": str(user.pk)},
    }


def _subscribed(user, cfg: Config) -> BillingCustomer:
    customer = _customer(user, cfg)
    if customer is None or not customer.subscription_id or customer.status not in LIVE:
        raise BillingRefused("You have no subscription.")
    return customer


def _apply_answer(customer: BillingCustomer, data: dict[str, Any], cfg: Config) -> None:
    """Take over the subscription Paddle returned for a change this server asked for."""
    with transaction.atomic():
        locked = (
            BillingCustomer.objects.select_for_update().select_related("user").get(pk=customer.pk)
        )
        apply_subscription(locked, data, _time(data.get("updated_at")) or timezone.now(), cfg)


def _money(amount: Any, currency: str) -> float | None:
    """Paddle's amount (a string in the smallest unit of the currency) as a number."""
    try:
        value = Decimal(str(amount))
    except (ArithmeticError, ValueError):
        return None
    return float(value if currency.upper() in ZERO_DECIMAL else value / 100)


def change_plan(user, plan_key: str, preview: bool) -> dict[str, Any]:
    """Move the subscription to another plan. An upgrade starts at once and Paddle charges
    the difference for the rest of the period; a downgrade starts at once too and the
    difference is credited to the next invoice. With ``preview``, nothing changes and the
    answer says what Paddle would charge."""
    cfg = require()
    customer = _subscribed(user, cfg)
    if customer.status not in ENTITLED:
        raise BillingRefused(
            "The plan cannot be changed while a payment is open. Update your payment method first."
        )
    if customer.cancel_at is not None:
        raise BillingRefused("Your subscription ends soon. Continue it first, then change it.")
    price_id = cfg.price_of(plan_key)
    plans = budgets.plans()
    if price_id is None or plan_key not in plans:
        raise BillingRefused("This plan cannot be bought.")
    if plan_key == customer.plan:
        raise BillingRefused("You already have this plan.")
    new, old = plans[plan_key], plans.get(customer.plan)
    upgrade = old is None or (new.price or 0) >= (old.price or 0)
    proration = "prorated_immediately" if upgrade else "prorated_next_billing_period"
    api = client(cfg)
    try:
        try:
            data = api.change_price(customer.subscription_id, price_id, proration, preview=preview)
        except PaddleError as exc:
            # Paddle charges nothing below its minimum (70 US cents), e.g. when a credit from
            # an earlier downgrade covers almost all of the difference. The upgrade still
            # starts at once; the small rest is added to the next invoice instead.
            if exc.code != BELOW_CHARGE_LIMIT or proration != "prorated_immediately":
                raise
            data = api.change_price(
                customer.subscription_id,
                price_id,
                "prorated_next_billing_period",
                preview=preview,
            )
    except PaddleError as exc:
        raise _unavailable(exc, "changing the plan") from None
    if not preview:
        _apply_answer(customer, data, cfg)
        return state(user)
    summary = data.get("update_summary") or {}
    result = summary.get("result") or {}
    currency = str(result.get("currency_code") or data.get("currency_code") or "")
    amount = _money(result.get("amount"), currency) if result else None
    return {
        "plan": new.to_dict(),
        "upgrade": upgrade,
        # What Paddle charges now ("charge") or credits to the next invoice ("credit") for
        # the rest of the period that is paid already; null if Paddle did not say.
        "now": {"action": result.get("action"), "amount": amount, "currency": currency}
        if amount is not None
        else None,
        "next_billed_at": _time(data.get("next_billed_at")),
    }


def cancel(user) -> dict[str, Any]:
    """End the subscription with the period that is paid for; the plan stays until then."""
    cfg = require()
    customer = _subscribed(user, cfg)
    if customer.cancel_at is not None:
        raise BillingRefused("Your subscription is cancelled already.")
    # A paused subscription has no period to end with: it is cancelled at once.
    immediately = customer.status == "paused"
    try:
        data = client(cfg).cancel_subscription(customer.subscription_id, immediately=immediately)
    except PaddleError as exc:
        raise _unavailable(exc, "cancelling") from None
    _apply_answer(customer, data, cfg)
    customer.refresh_from_db()
    # No renewal will come: bill the overage that is open now.
    charge_due_overage(customer, cfg, final=True)
    return state(user)


def reactivate(user) -> dict[str, Any]:
    """Take a cancellation back: the subscription renews as before."""
    cfg = require()
    customer = _subscribed(user, cfg)
    if customer.cancel_at is None:
        raise BillingRefused("Your subscription is not cancelled.")
    try:
        data = client(cfg).remove_scheduled_change(customer.subscription_id)
    except PaddleError as exc:
        raise _unavailable(exc, "continuing the subscription") from None
    _apply_answer(customer, data, cfg)
    return state(user)


def portal_url(user) -> str:
    """A link to Paddle's customer portal for this account: payment method and invoices."""
    cfg = require()
    customer = _customer(user, cfg)
    if customer is None:
        raise BillingRefused("You have no billing profile yet.")
    subscriptions = [customer.subscription_id] if customer.subscription_id else []
    try:
        return client(cfg).create_portal_session(customer.customer_id, subscriptions)
    except PaddleError as exc:
        raise _unavailable(exc, "opening the customer portal") from None


def _pull(customer: BillingCustomer, cfg: Config) -> None:
    """Ask Paddle what the account's subscription is like now and take it over."""
    api = client(cfg)
    found = []
    if customer.subscription_id and customer.status in LIVE:
        found = [api.get_subscription(customer.subscription_id)]
    if not any(item.get("status") in LIVE for item in found):
        # No subscription that goes on is known: look for one, e.g. after a checkout whose
        # notification has not arrived (yet).
        found += [
            item
            for item in api.list_subscriptions(customer.customer_id)
            if item.get("status") in LIVE
        ]
    for data in found:
        if str(data.get("customer_id") or "") != customer.customer_id:
            continue
        named = _named_user(data)
        if named is not None and named != str(customer.user_id):
            continue
        _apply_answer(customer, data, cfg)


def refresh(user) -> dict[str, Any]:
    """Ask Paddle for the account's subscription instead of waiting for its notification."""
    cfg = require()
    customer = _customer(user, cfg)
    if customer is not None:
        try:
            _pull(customer, cfg)
        except PaddleError as exc:
            raise _unavailable(exc, "reading the subscription") from None
    return state(user)


def email_changed(user) -> None:
    """Tell Paddle the account's new e-mail address (invoices go there). Best effort."""
    cfg = config()
    customer = _customer(user, cfg) if cfg.enabled else None
    if customer is None or not user.email:
        return
    try:
        client(cfg).update_customer(customer.customer_id, email=user.email)
    except PaddleError as exc:
        logger.warning("Billing: the e-mail address of account #%s at Paddle: %s", user.pk, exc)


# --------------------------------------------------------------------------- #
# Deleting an account
# --------------------------------------------------------------------------- #


def has_live_subscription(user) -> bool:
    return BillingCustomer.objects.filter(user=user, status__in=LIVE).exists()


def cancel_for_deletion(user) -> None:
    """Before an account is deleted: cancel its subscription at Paddle at once. Raises
    :class:`BillingUnavailable` if that fails, so that the account is not deleted: its owner
    would go on paying for a subscription nobody can manage here any more. Overage that is
    open is billed first, as far as Paddle lets it."""
    customer = BillingCustomer.objects.filter(user=user, status__in=LIVE).first()
    if customer is None or not customer.subscription_id:
        return
    cfg = config()
    if not cfg.enabled or customer.environment != cfg.environment:
        logger.warning(
            "Billing: account #%s is deleted with the subscription %s (%s), which this "
            "server cannot cancel. Cancel it in Paddle.",
            user.pk,
            customer.subscription_id,
            customer.environment,
        )
        return
    try:
        charge_due_overage(customer, cfg, final=True)
    except Exception:  # the deletion matters more than the last charge
        logger.exception("Billing: the last overage of account #%s was not billed.", user.pk)
    try:
        data = client(cfg).cancel_subscription(customer.subscription_id, immediately=True)
    except PaddleError as exc:
        if exc.status == 404 or "cancel" in exc.code:
            # Gone or cancelled already: there is nothing left to pay for.
            BillingCustomer.objects.filter(pk=customer.pk).update(status="canceled")
            return
        logger.warning("Billing: cancelling for the deletion of account #%s: %s", user.pk, exc)
        raise BillingUnavailable(
            "Your subscription could not be cancelled, because the payment provider (Paddle) "
            "did not answer. Your account was not deleted; please try again in a moment."
        ) from None
    _apply_answer(customer, data, cfg)


@receiver(pre_delete, sender=settings.AUTH_USER_MODEL, dispatch_uid="brickdnn_billing_cancel")
def _cancel_before_delete(sender, instance, **kwargs) -> None:
    """Also when an account is deleted another way (the Django admin): no account goes
    while its subscription goes on. accounts.delete_account has done this before already."""
    cancel_for_deletion(instance)


# --------------------------------------------------------------------------- #
# Overage: compute beyond the allowance, billed to the subscription
# --------------------------------------------------------------------------- #


def overage_allowed(user) -> bool:
    """Whether usage beyond the allowance can be billed for this account, so that budgets.py
    may allow it. With billing off this is up to the operator (True). With billing on it
    needs an overage product and a subscription that is paid and goes on."""
    cfg = config()
    if not cfg.enabled:
        return True
    if not cfg.overage_product:
        return False
    customer = _customer(user, cfg)
    return customer is not None and customer.status in ENTITLED and customer.cancel_at is None


def _handled_seconds(user, period: date) -> float:
    """Overage of the month that is billed, being billed or written off already."""
    rows = OverageCharge.objects.filter(
        user=user, period=period, status__in=("pending", "charged", "skipped")
    )
    return float(rows.aggregate(total=Sum("cpu_seconds"))["total"] or 0.0)


def overage_seconds(user, period: date) -> float:
    """What the ledger recorded as overage in the month, at most the account's cap."""
    recorded = float(
        UsageRecord.objects.filter(user=user, period=period).aggregate(total=Sum("overage"))[
            "total"
        ]
        or 0.0
    )
    profile = getattr(user, "profile", None)
    cap = profile.overage_cap_cpu_hours if profile is not None else None
    if cap is not None:
        recorded = min(recorded, float(cap) * 3600)
    return max(0.0, recorded)


def unbilled_overage_seconds(user) -> float:
    """The most overage the ledger recorded in a month that is not billed yet (the running
    month included): the account's cap must not go below it, or the bill would shrink."""
    most = 0.0
    for period in _open_periods(user):
        recorded = float(
            UsageRecord.objects.filter(user=user, period=period).aggregate(total=Sum("overage"))[
                "total"
            ]
            or 0.0
        )
        if recorded - _handled_seconds(user, period) > 0:
            most = max(most, recorded)
    return most


def _open_periods(user) -> list[date]:
    """Months in which the account used overage, oldest first."""
    return list(
        UsageRecord.objects.filter(user=user, overage__gt=0)
        .order_by("period")
        .values_list("period", flat=True)
        .distinct()
    )


def _overage_price(customer: BillingCustomer) -> Decimal | None:
    """The price of a CPU hour of overage for the account: that of its plan, else that of
    the plan its subscription pays for."""
    plans = budgets.plans()
    for plan in (budgets.plan_of(customer.user), plans.get(customer.plan)):
        if plan is not None and plan.overage_price_per_cpu_hour:
            return Decimal(str(plan.overage_price_per_cpu_hour))
    return None


def _price_text(price: Decimal) -> str:
    """``0.40`` (more digits only if the price has them)."""
    return f"{price:.2f}" if price == price.quantize(Decimal("0.01")) else f"{price.normalize():f}"


def _minor(amount: Decimal, currency: str) -> int:
    return int(amount if currency in ZERO_DECIMAL else amount * 100)


def charge_overage(
    customer: BillingCustomer, period: date, cfg: Config | None = None
) -> OverageCharge | None:
    """Bill the overage of the month ``period`` that is not billed yet to the account's
    subscription. Returns the charge (``charged``, ``failed`` or ``pending``: see the module
    docstring), or None if there is nothing to bill."""
    cfg = cfg or require()
    currency = settings.PRICE_CURRENCY
    with transaction.atomic():
        locked = (
            BillingCustomer.objects.select_for_update().select_related("user").get(pk=customer.pk)
        )
        user = locked.user
        if not cfg.overage_product or not locked.subscription_id or locked.status not in ENTITLED:
            return None
        if budgets.is_unlimited(user):
            return None
        charges = OverageCharge.objects.filter(user=user, period=period)
        if charges.filter(status="pending").exists():
            return None  # asked before and the answer is unknown: the operator decides
        if charges.filter(status="failed").count() >= MAX_CHARGE_ATTEMPTS:
            return None
        seconds = overage_seconds(user, period) - _handled_seconds(user, period)
        price = _overage_price(locked)
        if seconds <= 0 or price is None:
            return None
        hours = Decimal(str(seconds)) / 3600
        cents = Decimal("1") if currency in ZERO_DECIMAL else Decimal("0.01")
        amount = (hours * price).quantize(cents, rounding=ROUND_HALF_UP)
        if amount <= 0 or amount < Decimal(str(settings.PADDLE_OVERAGE_MIN_CHARGE)):
            return None  # too little to bill (it may grow while the month goes on)
        try:
            with transaction.atomic():
                charge = OverageCharge.objects.create(
                    user=user,
                    subscription_id=locked.subscription_id,
                    period=period,
                    cpu_seconds=seconds,
                    price_per_cpu_hour=price,
                    amount=amount,
                    currency=currency,
                    status="pending",
                )
        except IntegrityError:
            return None  # another process is billing this month right now
    # The row is committed: from here on a crash leaves it pending, and it is not billed again.
    plan = budgets.plans().get(locked.plan)
    name = (
        f"Brick DNN: {hours.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)} CPU hours beyond "
        f"the {plan.name + ' ' if plan else ''}plan in {period:%B %Y} "
        f"({_price_text(price)} {currency} per CPU hour)"
    )
    try:
        client(cfg).create_charge(
            locked.subscription_id,
            product_id=cfg.overage_product,
            description=name,
            amount_minor=_minor(amount, currency),
            currency=currency,
            immediately=True,
            custom_data={"brickdnn_overage_charge": str(charge.pk), "month": f"{period:%Y-%m}"},
        )
    except PaddleUnavailable as exc:
        charge.detail = f"No answer from Paddle: {exc}"[:300]
        charge.save(update_fields=["detail", "updated_at"])
        logger.error(
            "Billing: the overage charge #%s (account #%s, %s, %s %s) got no answer from "
            "Paddle. Check in Paddle whether it was billed and set its status in the admin.",
            charge.pk,
            user.pk,
            f"{period:%Y-%m}",
            amount,
            currency,
        )
        return charge
    except PaddleError as exc:
        charge.status = "failed"
        charge.detail = str(exc)[:300]
        charge.save(update_fields=["status", "detail", "updated_at"])
        logger.warning("Billing: the overage charge #%s was refused: %s", charge.pk, exc)
        return charge
    charge.status = "charged"
    charge.save(update_fields=["status", "updated_at"])
    logger.info(
        "Billing: billed %s %s of overage (%s) to account #%s.",
        amount,
        currency,
        f"{period:%Y-%m}",
        user.pk,
    )
    return charge


def charge_due_overage(
    customer: BillingCustomer,
    cfg: Config | None = None,
    final: bool = False,
    now: datetime | None = None,
) -> list[OverageCharge]:
    """Bill the overage of every month that is over. ``final`` (or a cancellation that is
    scheduled): also the running month, because no later chance will come."""
    cfg = cfg or require()
    if not cfg.overage_product or customer.status not in ENTITLED:
        return []
    current = budgets.period_of(now)
    done = []
    for period in _open_periods(customer.user):
        if period >= current and not (final or customer.cancel_at is not None):
            continue
        charge = charge_overage(customer, period, cfg)
        if charge is not None:
            done.append(charge)
    return done


def write_off_overage(customer: BillingCustomer) -> None:
    """The subscription ended: overage that was not billed cannot be billed any more. It is
    recorded as not charged, for the operator."""
    user = customer.user
    for period in _open_periods(user):
        if OverageCharge.objects.filter(user=user, period=period, status="pending").exists():
            continue
        seconds = overage_seconds(user, period) - _handled_seconds(user, period)
        price = _overage_price(customer)
        if seconds <= 0 or price is None:
            continue
        amount = (Decimal(str(seconds)) / 3600 * price).quantize(Decimal("0.01"), ROUND_HALF_UP)
        if amount <= 0:
            continue
        OverageCharge.objects.create(
            user=user,
            subscription_id=customer.subscription_id or "",
            period=period,
            cpu_seconds=seconds,
            price_per_cpu_hour=price,
            amount=amount,
            currency=settings.PRICE_CURRENCY,
            status="skipped",
            detail="The subscription ended before this was billed.",
        )
        logger.warning(
            "Billing: %s %s of overage of account #%s (%s) was not billed: its subscription ended.",
            amount,
            settings.PRICE_CURRENCY,
            user.pk,
            f"{period:%Y-%m}",
        )


# --------------------------------------------------------------------------- #
# Once a day (manage.py billing_sync)
# --------------------------------------------------------------------------- #


@dataclass
class SyncResult:
    checked: int = 0
    plans_changed: int = 0
    charges: int = 0
    errors: int = 0


def sync_all(log: Callable[[str], None] | None = None) -> SyncResult:
    """For every account with a subscription: ask Paddle how it is (a notification may have
    been lost), end grace periods, and bill the overage that is due."""
    cfg = config()
    result = SyncResult()
    if not cfg.enabled:
        return result
    say = log or (lambda line: None)
    for customer in BillingCustomer.objects.exclude(subscription_id=None).select_related("user"):
        result.checked += 1
        before = budgets.plan_of(customer.user).key
        try:
            if customer.environment == cfg.environment and customer.status in LIVE:
                _pull(customer, cfg)
                customer.refresh_from_db()
            sync_plan(customer)
            for charge in charge_due_overage(customer, cfg):
                result.charges += 1
                say(
                    f"Account #{customer.user_id}: overage of {charge.period:%Y-%m}, "
                    f"{charge.amount} {charge.currency}: {charge.status}"
                )
        except PaddleError as exc:
            result.errors += 1
            say(f"Account #{customer.user_id}: {exc}")
            continue
        after = budgets.plan_of(customer.user).key
        if after != before:
            result.plans_changed += 1
            say(f"Account #{customer.user_id}: plan {before} -> {after} ({customer.status})")
    return result


# --------------------------------------------------------------------------- #
# The data export of an account
# --------------------------------------------------------------------------- #


def export_records(user) -> dict[str, Any] | None:
    """billing.json of the account's data export (None: nothing is stored)."""
    customer = BillingCustomer.objects.filter(user=user).first()
    events = BillingEvent.objects.filter(user=user).order_by("occurred_at", "pk")
    charges = OverageCharge.objects.filter(user=user).order_by("created_at", "pk")
    if customer is None and not events.exists() and not charges.exists():
        return None
    return {
        "payment_provider": "Paddle (Paddle.com Market Ltd), the seller of the plans. It keeps "
        "your payment data, billing address and invoices; ask Paddle for those.",
        "customer": {
            "environment": customer.environment,
            "paddle_customer_id": customer.customer_id,
            "paddle_subscription_id": customer.subscription_id,
            "status": customer.status or None,
            "plan": customer.plan or None,
            "paddle_price_id": customer.price_id or None,
            "current_period_start": customer.current_period_start,
            "current_period_end": customer.current_period_end,
            "next_billed_at": customer.next_billed_at,
            "ends_at": customer.cancel_at,
            "ended_at": customer.canceled_at,
            "past_due_since": customer.past_due_since,
            "payment_failed_at": customer.payment_failed_at,
            "created_at": customer.created_at,
            "updated_at": customer.updated_at,
        }
        if customer is not None
        else None,
        "notifications": [
            {
                "type": event.event_type,
                "occurred_at": event.occurred_at,
                "received_at": event.received_at,
                "paddle_subscription_id": event.subscription_id or None,
                "outcome": event.outcome,
            }
            for event in events
        ],
        "overage_charges": [
            {
                "month": f"{charge.period:%Y-%m}",
                "cpu_hours": round(charge.cpu_seconds / 3600, 4),
                "price_per_cpu_hour": charge.price_per_cpu_hour,
                "amount": charge.amount,
                "currency": charge.currency,
                "status": charge.status,
                "created_at": charge.created_at,
            }
            for charge in charges
        ],
    }
