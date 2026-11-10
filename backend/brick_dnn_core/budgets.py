"""Usage and budgets: the compute and storage each account uses, and its limits.

**Compute** is CPU time. A training measures its own: the runner reports it in state.json
(``usage``), so that every server process that reads the state counts it, also for a run
that another process started. A prediction reports its CPU time when it ends. Both go into
the ledger, :class:`UsageRecord`: one row per run and calendar month (UTC), which grows while
the run is active, and one per prediction, each with the API token or connected app that
caused it. Read-only actions (analysing a network, generating code, listing) never count.

**Storage** is not a monthly amount but what an account stores now: its imported files
(Data tab), datasets and the files of its training runs, from the sizes in the database.

**Limits** come from the account's **plan** (``BRICKDNN_PLANS``, see :func:`plans`; without
it one plan from ``BRICKDNN_BUDGET_…``, unlimited by default). A plan has

- a monthly compute allowance (CPU hours);
- the most compute one training run may use;
- a storage quota;
- how many trainings may run at the same time, and a queue priority (not used yet);
- optionally a price per CPU hour beyond the allowance (*overage*), which users of the plan
  can switch on for their account, with a cap of their own.

:func:`set_plan` changes an account's plan (billing.py calls it), :func:`limits_for` gives
the limits that apply. Accounts can be marked **unlimited** (:func:`set_unlimited`; staff and
superusers are by default): they are still measured, but no limit of a plan applies. The
operator can override single limits per account in the Django admin (user profile).

Users can set a lower monthly budget for their own account and a monthly budget for each
of their API tokens and connected apps (the budgets of agents); these apply to unlimited
accounts too. Every budget is in CPU minutes; with ``BRICKDNN_PRICE_PER_CPU_HOUR``, usage is
also shown in money.

**Enforcement.** A training only starts if every budget that applies (the account's, and
that of the token or app starting it) has CPU time left, and the storage quota is not used
up. The run gets the CPU time that is left as its limit (at most the per-run limit), which
the runner enforces itself: it stops like after a stop request, keeping the weights trained
so far. Runs that are active at the same time reserve what their limits allow, so two runs
started at once cannot spend the same remaining time; starts are serialised by the start
lock of training_service. Every look at an active run also compares the counted usage with
the budgets (e.g. after one was lowered) and stops the run when one is used up.
Predictions need compute left, uploads and imports room in the storage quota.
"""

from __future__ import annotations

import calendar
import contextlib
import math
import os
from dataclasses import dataclass
from datetime import date, datetime
from datetime import timezone as dt_timezone
from pathlib import Path
from typing import Any

from django.conf import settings
from django.db import transaction
from django.db.models import F, Q, Sum
from django.utils import timezone
from rest_framework import status
from rest_framework.exceptions import APIException

from brickdnn.training import ACTIVE_STATUSES, TERMINAL_STATUSES
from brickdnn.training.estimate import Estimate

from .models import (
    ApiToken,
    Dataset,
    DataSource,
    OAuthConnection,
    TrainingRun,
    UsageRecord,
    UserProfile,
)

UNIT = "cpu_second"
# Budgets that limit the whole account (not only a token or an app).
ACCOUNT_KINDS = ("allowance", "overage", "own")
# Usage of an active run is written to the ledger once this much has come together (and
# always when the run ends), not on every look at the run.
RECORD_STEP_SECONDS = 2.0
# A new run needs at least this much CPU time (otherwise it would stop right away).
MIN_RUN_SECONDS = 5.0
# A run is refused if even the low end of its estimate exceeds what it may use.
ESTIMATE_NOTE = (
    "A rough estimate from the network, the number of samples and epochs; the real CPU time "
    "is often two or three times lower or higher. Only the CPU time a training really uses "
    "is counted."
)


# --------------------------------------------------------------------------- #
# Who causes the usage
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Actor:
    """The account that pays, and the personal API token or connected app acting for it
    (both None in the web app)."""

    user: Any
    token: ApiToken | None = None
    connection: OAuthConnection | None = None

    @classmethod
    def of(cls, request) -> Actor:
        auth = getattr(request, "auth", None)
        if isinstance(auth, ApiToken):
            if auth.connection_id is not None:
                connection = OAuthConnection.objects.select_related("client").get(
                    pk=auth.connection_id
                )
                return cls(request.user, connection=connection)
            return cls(request.user, token=auth)
        return cls(request.user)

    @classmethod
    def of_run(cls, run: TrainingRun) -> Actor:
        return cls(run.owner, run.api_token, run.connection)

    @property
    def source(self) -> str:
        if self.connection is not None:
            return "app"
        return "token" if self.token is not None else "web"

    @property
    def via(self) -> str:
        """The name of the token or app (empty in the web app)."""
        if self.connection is not None:
            return self.connection.client.name[:200]
        return self.token.name if self.token is not None else ""


# --------------------------------------------------------------------------- #
# Months
# --------------------------------------------------------------------------- #


def period_of(moment: datetime | None = None) -> date:
    """The first day of the calendar month (UTC) of ``moment`` (default: now)."""
    moment = (moment or timezone.now()).astimezone(dt_timezone.utc)
    return date(moment.year, moment.month, 1)


def period_end(period: date) -> datetime:
    """When the month starting on ``period`` ends: the moment budgets reset (UTC)."""
    days = calendar.monthrange(period.year, period.month)[1]
    last = period.replace(day=days)
    year, month = (last.year + 1, 1) if last.month == 12 else (last.year, last.month + 1)
    return datetime(year, month, 1, tzinfo=dt_timezone.utc)


def _day(moment: datetime) -> str:
    return f"{moment.day} {moment:%B %Y}"


# --------------------------------------------------------------------------- #
# Limits
# --------------------------------------------------------------------------- #


def _minutes_setting(value: float | None) -> float | None:
    """CPU minutes from a setting or override (0 or less: unlimited) in seconds."""
    if value is None or not math.isfinite(value) or value <= 0:
        return None
    return float(value) * 60


# The plan of a server without BRICKDNN_PLANS (its limits are BRICKDNN_BUDGET_…).
IMPLICIT_PLAN = "default"
PLAN_FIELDS = {
    "name": str,
    "price": float,
    "cpu_hours_per_month": float,
    "storage_gb": float,
    "cpu_minutes_per_run": float,
    "concurrent_trainings": int,
    "priority": int,
    "overage_price_per_cpu_hour": float,
}


@dataclass(frozen=True)
class Plan:
    """A plan (subscription tier) and its limits; None (or 0 in the settings): unlimited.

    ``price`` per month and ``overage_price_per_cpu_hour`` are in BRICKDNN_PRICE_CURRENCY.
    The price is for display (with paid plans the customer pays the price set in Paddle, see
    billing.py); overage is billed at its price. ``priority`` is for the job queue (higher
    goes first); nothing uses it yet."""

    key: str
    name: str
    price: float | None = None
    cpu_hours_per_month: float | None = None
    storage_gb: float | None = None
    cpu_minutes_per_run: float | None = None
    concurrent_trainings: int | None = None
    priority: int = 0
    overage_price_per_cpu_hour: float | None = None

    @property
    def unlimited(self) -> bool:
        return (
            not any((self.cpu_hours_per_month, self.storage_gb, self.cpu_minutes_per_run))
            and not self.concurrent_trainings
        )

    def to_dict(self) -> dict[str, Any]:
        """``Plan`` in frontend/src/api/types.ts."""
        currency = settings.PRICE_CURRENCY
        return {
            "key": self.key,
            "name": self.name,
            "price": self.price,
            "currency": currency if self.price is not None else None,
            "cpu_hours_per_month": self.cpu_hours_per_month,
            "storage_gb": self.storage_gb,
            "cpu_minutes_per_run": self.cpu_minutes_per_run,
            "concurrent_trainings": self.concurrent_trainings,
            "priority": self.priority,
            "overage_price_per_cpu_hour": self.overage_price_per_cpu_hour,
        }


def _plan(key: str, raw: Any) -> Plan:
    from django.core.exceptions import ImproperlyConfigured

    if not isinstance(raw, dict):
        raise ImproperlyConfigured(f"BRICKDNN_PLANS: plan {key!r} must be an object.")
    values: dict[str, Any] = {}
    for field, kind in PLAN_FIELDS.items():
        value = raw.get(field)
        if value is None or value == "":
            continue
        try:
            value = kind(value)
        except (TypeError, ValueError):
            raise ImproperlyConfigured(
                f"BRICKDNN_PLANS: {field} of plan {key!r} must be a {kind.__name__}."
            ) from None
        # 0 means unlimited for the limits (and "free" for prices, which is kept).
        if field not in ("name", "price", "priority") and value <= 0:
            continue
        values[field] = value
    unknown = set(raw) - set(PLAN_FIELDS)
    if unknown:
        raise ImproperlyConfigured(
            f"BRICKDNN_PLANS: unknown setting {sorted(unknown)[0]!r} in plan {key!r}."
        )
    values.setdefault("name", key.replace("_", " ").title())
    return Plan(key=key, **values)


def plans() -> dict[str, Plan]:
    """The plans of this server by key, in the order of BRICKDNN_PLANS (an object of plans by
    key). Without it: one plan with the limits of BRICKDNN_BUDGET_… (all unlimited unless
    they are set)."""
    raw = settings.PLANS
    if not raw:
        hours = settings.BUDGET_CPU_MINUTES_PER_MONTH / 60
        return {
            IMPLICIT_PLAN: _plan(
                IMPLICIT_PLAN,
                {
                    "name": "Default",
                    "cpu_hours_per_month": hours,
                    "storage_gb": settings.BUDGET_STORAGE_MB / 1024,
                    "cpu_minutes_per_run": settings.BUDGET_CPU_MINUTES_PER_RUN,
                },
            )
        }
    from django.core.exceptions import ImproperlyConfigured

    if not isinstance(raw, dict) or not raw:
        raise ImproperlyConfigured("BRICKDNN_PLANS must be an object of plans by key.")
    return {str(key): _plan(str(key), value) for key, value in raw.items()}


def default_plan() -> Plan:
    """BRICKDNN_DEFAULT_PLAN, else the first plan."""
    available = plans()
    return available.get(settings.DEFAULT_PLAN) or next(iter(available.values()))


def _profile(user) -> UserProfile | None:
    return UserProfile.objects.filter(user=user).first()


def plan_of(user, profile: UserProfile | None = None) -> Plan:
    """The account's plan (the default plan if it has none, or one that no longer exists)."""
    profile = profile if profile is not None else _profile(user)
    key = profile.plan if profile is not None else ""
    return plans().get(key) or default_plan()


def is_unlimited(user, profile: UserProfile | None = None) -> bool:
    """No limit of the plan applies: marked so, or (unless marked otherwise) a staff member
    or superuser (BRICKDNN_STAFF_UNLIMITED)."""
    profile = profile if profile is not None else _profile(user)
    if profile is not None and profile.unlimited is not None:
        return profile.unlimited
    return bool(settings.STAFF_UNLIMITED and (user.is_staff or user.is_superuser))


def set_plan(user, key: str) -> Plan:
    """Give the account the plan ``key`` (a key of :func:`plans`, or "" for the default
    plan). Takes effect at once, also for running trainings. Raises ``KeyError``."""
    if key and key not in plans():
        raise KeyError(key)
    UserProfile.objects.update_or_create(user=user, defaults={"plan": key})
    return plan_of(user)


def set_unlimited(user, unlimited: bool | None) -> None:
    """Mark the account as unlimited (True), limited by its plan (False) or the default for
    its kind of account (None: unlimited for staff and superusers)."""
    UserProfile.objects.update_or_create(user=user, defaults={"unlimited": unlimited})


@dataclass(frozen=True)
class Limits:
    """The limits of one account, in seconds and bytes (None: unlimited)."""

    monthly_seconds: float | None
    own_seconds: float | None
    run_seconds: float | None
    storage_bytes: int | None
    plan: Plan = Plan(IMPLICIT_PLAN, "Default")
    unlimited: bool = False
    concurrent_trainings: int | None = None
    # Overage switched on (only with a plan that has an overage price): usage beyond the
    # allowance is allowed up to the cap (None: no cap).
    overage: bool = False
    overage_cap_seconds: float | None = None


def limits_for(user) -> Limits:
    """The limits that apply to the account: those of its plan, unless the operator
    overrides one (Django admin), none of them for an unlimited account. The user's own
    monthly budget applies in any case."""
    profile = _profile(user)
    plan = plan_of(user, profile)
    unlimited = is_unlimited(user, profile)
    own = profile.budget_cpu_minutes if profile is not None else None
    own_seconds = float(own) * 60 if own is not None else None
    if unlimited:
        return Limits(None, own_seconds, None, None, plan=plan, unlimited=True)

    def setting(field: str, default: float | None) -> float:
        override = getattr(profile, field) if profile is not None else None
        value = override if override is not None else default
        return value or 0.0

    monthly = _minutes_setting(
        setting("cpu_minutes_per_month", (plan.cpu_hours_per_month or 0) * 60)
    )
    run = _minutes_setting(setting("cpu_minutes_per_run", plan.cpu_minutes_per_run))
    storage_mb = setting("storage_mb", (plan.storage_gb or 0) * 1024)
    storage = int(storage_mb * 1024 * 1024) if storage_mb > 0 else None
    overage = bool(
        profile is not None
        and profile.overage_enabled
        and plan.overage_price_per_cpu_hour is not None
        and monthly is not None
        and overage_billable(user)
    )
    cap = profile.overage_cap_cpu_hours if overage and profile is not None else None
    return Limits(
        monthly_seconds=monthly,
        own_seconds=own_seconds,
        run_seconds=run,
        storage_bytes=storage,
        plan=plan,
        concurrent_trainings=plan.concurrent_trainings,
        overage=overage,
        overage_cap_seconds=cap * 3600 if cap is not None else None,
    )


def overage_billable(user) -> bool:
    """Whether compute beyond the allowance could be billed to the account. Always on a
    server without paid plans (billing.py), where that is up to the operator; with them it
    needs a subscription that is paid and goes on."""
    from . import billing  # billing.py builds on this module

    return billing.overage_allowed(user)


def overage_used_seconds(user) -> float:
    """Overage the account's cap may not go below: what it used beyond the allowance this
    month and, with paid plans, in months that are not billed yet. A cap set lower would not
    undo that usage, only hide it from the limit (and from the bill)."""
    from . import billing  # billing.py builds on this module

    this_month = float(
        UsageRecord.objects.filter(user=user, period=period_of()).aggregate(n=Sum("overage"))["n"]
        or 0.0
    )
    if billing.config().enabled:
        return max(this_month, billing.unbilled_overage_seconds(user))
    return this_month


def pricing_url() -> str | None:
    """Where a higher plan can be bought, on a server that sells plans (billing.py)."""
    from . import billing

    return billing.pricing_url()


def concurrency_refusal(user, active: int) -> str | None:
    """Why the account cannot start another training while ``active`` of its trainings run
    (its plan allows fewer at a time), or None."""
    limits = limits_for(user)
    allowed = limits.concurrent_trainings
    if allowed is None or active < allowed:
        return None
    count = "1 training" if allowed == 1 else f"{allowed} trainings"
    return (
        f"Your plan ({limits.plan.name}) allows {count} at a time. Wait until your running "
        "training finishes, or stop it."
    )


def cpu_price() -> float | None:
    """Price of one CPU hour (None: no price configured)."""
    price = settings.PRICE_PER_CPU_HOUR
    return price if price and price > 0 else None


def storage_price() -> float | None:
    price = settings.PRICE_PER_GB_MONTH
    return price if price and price > 0 else None


def cost(seconds: float | None) -> float | None:
    """What ``seconds`` of CPU time cost (None without a price)."""
    price = cpu_price()
    if price is None or seconds is None:
        return None
    return round(seconds / 3600 * price, 4)


def _money(amount: float) -> str:
    return f"{amount:.2f} {settings.PRICE_CURRENCY}"


def minutes(seconds: float | None, digits: int = 2) -> float | None:
    return round(seconds / 60, digits) if seconds is not None else None


def _min_text(seconds: float, money: bool = False) -> str:
    """``"12.5 CPU minutes"`` (with ``money``, also what they cost, if there is a price)."""
    value = seconds / 60
    number = f"{value:.1f}" if value >= 10 else f"{value:.2f}".rstrip("0").rstrip(".")
    text = f"{number or '0'} CPU minute{'' if number == '1' else 's'}"
    amount = cost(seconds)
    # Amounts below a cent would only clutter the messages.
    if money and amount is not None and amount >= 0.005:
        return f"{text} ({_money(amount)})"
    return text


def _iso(moment: datetime | None) -> str | None:
    """``2026-12-01T00:00:00Z``, the same in the REST API and in MCP tool results."""
    if moment is None:
        return None
    return moment.astimezone(dt_timezone.utc).isoformat().replace("+00:00", "Z")


# --------------------------------------------------------------------------- #
# Usage so far
# --------------------------------------------------------------------------- #


def used_seconds(
    user,
    token: ApiToken | None = None,
    connection: OAuthConnection | None = None,
    period: date | None = None,
) -> float:
    """CPU seconds counted this month (``period``) for the account, or only those caused by
    ``token`` or ``connection``."""
    records = UsageRecord.objects.filter(user=user, period=period or period_of())
    if token is not None:
        records = records.filter(api_token=token)
    if connection is not None:
        records = records.filter(connection=connection)
    return float(records.aggregate(total=Sum("quantity"))["total"] or 0.0)


@dataclass
class Budget:
    """One limit on compute this month: the account's allowance (``allowance``), the user's
    own budget for the account (``own``), or that of an API token (``token``) or a connected
    app (``app``). Seconds."""

    kind: str
    limit: float
    used: float
    name: str = ""
    resets_at: datetime | None = None

    @property
    def remaining(self) -> float:
        return max(0.0, self.limit - self.used)

    @property
    def exhausted(self) -> bool:
        return self.used >= self.limit

    @property
    def label(self) -> str:
        return {
            "allowance": "the monthly compute allowance of this account"
            + (f" ({self.name} plan)" if self.name else ""),
            "overage": "the monthly allowance and the overage cap of this account",
            "own": "the monthly budget set for this account",
            "token": f"the monthly budget of the API token “{self.name}”",
            "app": f"the monthly budget of the connected app “{self.name}”",
        }[self.kind]

    @property
    def how_to_raise(self) -> str:
        pricing = pricing_url()
        higher = (
            f"Higher plans include more: {pricing}"
            if pricing
            else "Higher plans include more (see your account page)."
            if len(plans()) > 1
            else "Ask the operator of this server for a higher allowance."
        )
        return {
            "allowance": higher,
            "overage": "You can raise the overage cap on your account page.",
            "own": "You can raise it on your account page.",
            "token": "Its owner can raise it on the API tokens page.",
            "app": "Its owner can raise it on the API tokens page (connected apps).",
        }[self.kind]

    def exhausted_message(self) -> str:
        resets = f" It resets on {_day(self.resets_at)} (UTC)." if self.resets_at else ""
        return (
            f"{self.label[0].upper()}{self.label[1:]} is used up: {_min_text(self.used, True)} of "
            f"{_min_text(self.limit, True)} this month.{resets} {self.how_to_raise}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name or None,
            "limit_cpu_minutes": minutes(self.limit),
            "used_cpu_minutes": minutes(self.used),
            "remaining_cpu_minutes": minutes(self.remaining),
            "exhausted": self.exhausted,
        }


def compute_budgets(actor: Actor, period: date | None = None) -> list[Budget]:
    """The budgets on compute that apply to ``actor`` this month."""
    period = period or period_of()
    resets = period_end(period)
    limits = limits_for(actor.user)
    budgets: list[Budget] = []
    account = None
    allowance = ("allowance", limits.monthly_seconds)
    if limits.overage:
        # Beyond the allowance up to the user's cap (no cap: no limit but their own budget).
        cap = limits.overage_cap_seconds
        allowance = ("overage", limits.monthly_seconds + cap if cap is not None else None)  # type: ignore[operator]
    for kind, limit in (allowance, ("own", limits.own_seconds)):
        if limit is not None:
            if account is None:
                account = used_seconds(actor.user, period=period)
            named = kind == "allowance" and limits.plan.key != IMPLICIT_PLAN
            name = limits.plan.name if named else ""
            budgets.append(Budget(kind, limit, account, name, resets_at=resets))
    token, connection = actor.token, actor.connection
    if token is not None and token.budget_cpu_minutes is not None:
        used = used_seconds(actor.user, token=token, period=period)
        budgets.append(Budget("token", token.budget_cpu_minutes * 60, used, token.name, resets))
    if connection is not None and connection.budget_cpu_minutes is not None:
        used = used_seconds(actor.user, connection=connection, period=period)
        name = connection.client.name
        budgets.append(Budget("app", connection.budget_cpu_minutes * 60, used, name, resets))
    return budgets


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class BudgetError(APIException):
    """A budget or quota does not allow the request (403). The message says which one, how
    much is used and how to get more; ``data`` is the same as JSON."""

    status_code = status.HTTP_403_FORBIDDEN
    default_code = "budget_exceeded"

    def __init__(self, message: str, data: dict[str, Any] | None = None, code: str | None = None):
        super().__init__(message, code or self.default_code)
        self.message = message
        self.data = {"code": code or self.default_code, **(data or {})}

    def body(self) -> dict[str, Any]:
        return {"detail": self.message, **self.data}


class BudgetExceeded(BudgetError):
    """A compute budget is used up (or reserved by running trainings)."""


class EstimateTooHigh(BudgetError):
    """The training would clearly need more CPU time than it may use."""

    default_code = "estimate_too_high"


class StorageQuotaExceeded(BudgetError):
    default_code = "storage_quota_exceeded"


def _exceeded(budget: Budget) -> BudgetExceeded:
    data = {"budget": budget.to_dict(), "resets_at": _iso(budget.resets_at)}
    if budget.kind == "allowance" and pricing_url():
        data["upgrade_url"] = pricing_url()  # a higher plan can be bought there
    return BudgetExceeded(budget.exhausted_message(), data)


def check_compute(actor: Actor) -> list[Budget]:
    """Raise :class:`BudgetExceeded` if a budget of ``actor`` is used up."""
    budgets = compute_budgets(actor)
    for budget in budgets:
        if budget.exhausted:
            raise _exceeded(budget)
    return budgets


# --------------------------------------------------------------------------- #
# Starting a training
# --------------------------------------------------------------------------- #


@dataclass
class RunLimit:
    """The CPU time a new run may use (None: unlimited), why, and the message the run
    stops with when it is used up."""

    seconds: float | None
    reason: str | None = None  # "budget" | "run_limit"
    message: str = ""
    budget: Budget | None = None
    # For the per-run limit: "the most one training may use with the Free plan".
    per_run: str = ""

    def job(self) -> dict[str, Any] | None:
        """``limits`` for job.json (see brickdnn.training.state)."""
        if self.seconds is None:
            return None
        return {"cpu_seconds": self.seconds, "reason": self.reason, "message": self.message}


def _reserved(runs) -> float:
    """CPU time the active ``runs`` may still use within their limits."""
    return sum(
        max(0.0, run.cpu_limit_seconds - run.cpu_seconds)
        for run in runs
        if run.cpu_limit_seconds is not None
    )


def run_limit(actor: Actor, exclude: Any = None) -> RunLimit:
    """The limit for a new run of ``actor``. Raises :class:`BudgetExceeded` if a budget is
    used up or the rest is reserved by active runs (``exclude``: a run not to count).

    Call it under the start lock, after the active runs were synced."""
    budgets = check_compute(actor)
    active = TrainingRun.objects.filter(owner=actor.user, status__in=ACTIVE_STATUSES)
    if exclude is not None:
        active = active.exclude(pk=exclude)
    candidates: list[tuple[float, str, Budget | None]] = []
    cap = limits_for(actor.user).run_seconds
    if cap is not None:
        candidates.append((cap, "run_limit", None))
    for budget in budgets:
        runs = active
        if budget.kind == "token":
            runs = active.filter(api_token=actor.token)
        elif budget.kind == "app":
            runs = active.filter(connection=actor.connection)
        reserved = _reserved(runs)
        available = budget.remaining - reserved
        if available < MIN_RUN_SECONDS:
            if reserved <= 0 and budget.exhausted:
                raise _exceeded(budget)
            if reserved <= 0:
                label = budget.label[0].upper() + budget.label[1:]
                resets = (
                    f" It resets on {_day(budget.resets_at)} (UTC)." if budget.resets_at else ""
                )
                raise BudgetExceeded(
                    f"{label} is almost used up: {_min_text(budget.used, True)} of "
                    f"{_min_text(budget.limit, True)} this month, too little is left for a training."
                    f"{resets} {budget.how_to_raise}",
                    {"budget": budget.to_dict(), "resets_at": _iso(budget.resets_at)},
                )
            raise BudgetExceeded(
                f"What is left of {budget.label} ({_min_text(budget.remaining)}) is reserved by "
                "trainings that are still running. Wait until they finish, or stop one.",
                {"budget": budget.to_dict(), "resets_at": _iso(budget.resets_at)},
            )
        candidates.append((available, "budget", budget))
    if not candidates:
        return RunLimit(None)
    seconds, reason, budget = min(candidates, key=lambda item: item[0])
    if budget is None:
        message = f"Stopped: this run used {_min_text(seconds)}, {_per_run(actor.user)}."
    else:
        message = (
            f"Stopped: the compute budget ran out ({budget.label}; "
            f"{_min_text(budget.remaining)} were left when the run started)."
        )
    return RunLimit(seconds, reason, message, budget, _per_run(actor.user))


def _per_run(user) -> str:
    """ "the most one training may use with the Free plan" (or "on this server")."""
    plan = plan_of(user)
    where = "on this server" if plan.key == IMPLICIT_PLAN else f"with the {plan.name} plan"
    return f"the most one training may use {where}"


def estimate_verdict(estimate: Estimate | None, limit: RunLimit) -> tuple[str | None, str | None]:
    """``(refusal, warning)`` for a run with ``estimate`` that may use ``limit``."""
    if estimate is None or limit.seconds is None:
        return None, None
    what = (
        limit.per_run
        if limit.reason == "run_limit"
        else f"what is left of {limit.budget.label}"
        if limit.budget
        else "what is left"
    )
    estimated = (
        f"The estimated CPU time of this training is {_min_text(estimate.cpu_seconds)} "
        f"(roughly {_min_text(estimate.low)} to {_min_text(estimate.high)})"
    )
    if estimate.low > limit.seconds:
        return (
            f"{estimated}, far more than it can use: {_min_text(limit.seconds)}, {what}. "
            "Train fewer epochs, on fewer samples (Max. samples) or a smaller network.",
            None,
        )
    if estimate.cpu_seconds > limit.seconds:
        return None, (
            f"{estimated}, more than it can use ({_min_text(limit.seconds)}, {what}): it will "
            "probably be stopped before it finishes, keeping the weights trained so far."
        )
    return None, None


def estimate_data(actor: Actor, estimate: Estimate | None, exclude: Any = None) -> dict[str, Any]:
    """``{"estimate", "budget"}`` for the training check (web, REST and MCP): the estimated
    CPU time of a planned run and whether the budgets allow it."""
    data: dict[str, Any] = {"estimate": None, "budget": None}
    if estimate is not None:
        data["estimate"] = {
            "cpu_minutes": minutes(estimate.cpu_seconds),
            "low_cpu_minutes": minutes(estimate.low),
            "high_cpu_minutes": minutes(estimate.high),
            "cost": cost(estimate.cpu_seconds),
            "currency": settings.PRICE_CURRENCY if cpu_price() is not None else None,
            "basis": estimate.basis,
            "note": ESTIMATE_NOTE,
            "details": estimate.details,
        }
    period = period_of()
    remaining = [b.remaining for b in compute_budgets(actor, period)]
    budget: dict[str, Any] = {
        "allowed": True,
        "reason": None,
        "warning": None,
        "run_limit_cpu_minutes": None,
        "limit_reason": None,
        # What is left this month of the budgets that apply (null: none applies).
        "remaining_cpu_minutes": minutes(min(remaining)) if remaining else None,
        "resets_at": _iso(period_end(period)),
    }
    try:
        check_storage(actor.user)
        limit = run_limit(actor, exclude=exclude)
    except BudgetError as exc:
        budget.update(allowed=False, reason=exc.message, code=exc.data["code"])
    else:
        refusal, warning = estimate_verdict(estimate, limit)
        budget.update(
            allowed=refusal is None,
            reason=refusal,
            warning=warning,
            run_limit_cpu_minutes=minutes(limit.seconds),
            limit_reason=limit.reason,
        )
        if refusal is not None:
            budget["code"] = "estimate_too_high"
    data["budget"] = budget
    return data


# --------------------------------------------------------------------------- #
# Counting usage
# --------------------------------------------------------------------------- #


def fallback_threads() -> int:
    """CPU threads a training uses, for runs that did not report their CPU time."""
    with contextlib.suppress(ValueError):
        threads = int(os.environ.get("OMP_NUM_THREADS", ""))
        if threads > 0:
            return threads
    return os.cpu_count() or 1


def measured_cpu(run: TrainingRun, state: dict[str, Any] | None) -> float | None:
    """The CPU time ``run`` used so far: what the runner reported, or for a run that ended
    without a report, its wall time × the CPU threads of a training."""
    usage = state.get("usage") if isinstance(state, dict) else None
    value = usage.get("cpu_seconds") if isinstance(usage, dict) else None
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
        return float(value)
    if run.status in TERMINAL_STATUSES:
        start = run.started_at or run.created_at
        end = run.finished_at or timezone.now()
        if start is not None and end > start:
            return (end - start).total_seconds() * fallback_threads()
    return None


def overage_part(user, seconds: float) -> float:
    """How much of ``seconds`` that the account uses now is beyond its plan's monthly
    allowance, with overage switched on (0 otherwise). Call before counting them."""
    limits = limits_for(user)
    if not limits.overage or limits.monthly_seconds is None:
        return 0.0
    before = used_seconds(user)
    return max(0.0, min(seconds, before + seconds - limits.monthly_seconds))


def _record(
    actor: Actor, kind: str, seconds: float, run: TrainingRun | None, overage: float = 0.0
) -> UsageRecord:
    return UsageRecord.objects.create(
        overage=overage,
        user=actor.user,
        source=actor.source,
        api_token=actor.token,
        connection=actor.connection,
        via=actor.via,
        kind=kind,
        quantity=seconds,
        unit=UNIT,
        run=run,
        run_name=run.name if run is not None else "",
        period=period_of(),
    )


def record_run(run: TrainingRun, measured: float, final: bool = False) -> float:
    """Count the CPU time of ``run`` up to ``measured`` (its total so far) in the ledger and
    return what was added. Safe across server processes: what another process counted
    already is not counted again. Small amounts wait for the next call unless ``final``."""
    previous = run.cpu_seconds or 0.0
    delta = measured - previous
    if delta <= 0 or (delta < RECORD_STEP_SECONDS and not final):
        return 0.0
    with transaction.atomic():
        updated = TrainingRun.objects.filter(pk=run.pk, cpu_seconds=previous).update(
            cpu_seconds=measured
        )
        if not updated:  # counted by another process meanwhile
            current = TrainingRun.objects.filter(pk=run.pk).values_list("cpu_seconds", flat=True)
            run.cpu_seconds = next(iter(current), previous)
            return 0.0
        period = period_of()
        over = overage_part(run.owner, delta)
        rows = UsageRecord.objects.filter(run=run, kind="training", period=period).update(
            quantity=F("quantity") + delta, overage=F("overage") + over, updated_at=timezone.now()
        )
        if not rows:
            _record(Actor.of_run(run), "training", delta, run, over)
    run.cpu_seconds = measured
    return delta


def record_prediction(actor: Actor, run: TrainingRun, seconds: float) -> None:
    if seconds > 0:
        _record(actor, "prediction", seconds, run, overage_part(actor.user, seconds))


def over_budget(run: TrainingRun) -> tuple[str, str] | None:
    """``(reason, message)`` if the active ``run`` must stop: it used the most one run may
    use, or a budget it counts against is used up."""
    cap = limits_for(run.owner).run_seconds
    if cap is not None and run.cpu_seconds >= cap:
        return "run_limit", (
            f"Stopped: this run used {_min_text(run.cpu_seconds)}, {_per_run(run.owner)}."
        )
    for budget in compute_budgets(Actor.of_run(run)):
        if budget.exhausted:
            return "budget", f"Stopped: {budget.exhausted_message()}"
    return None


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #


def folder_size(path: Path) -> int:
    total = 0
    with contextlib.suppress(OSError):
        for item in path.rglob("*"):
            with contextlib.suppress(OSError):
                if item.is_file() and not item.is_symlink():
                    total += item.stat().st_size
    return total


def storage_used(user) -> dict[str, int]:
    """Bytes the account stores now: ``{"data_sources", "datasets", "training_runs",
    "total"}``. Runs that finished before sizes were recorded are measured once."""
    from .training_service import run_dir

    for run in TrainingRun.objects.filter(
        owner=user, size_bytes__isnull=True, status__in=TERMINAL_STATUSES
    ).only("pk"):
        TrainingRun.objects.filter(pk=run.pk).update(size_bytes=folder_size(run_dir(run.pk)))
    sources = DataSource.objects.filter(owner=user).aggregate(n=Sum("size_bytes"))["n"] or 0
    datasets = Dataset.objects.filter(owner=user).aggregate(n=Sum("size_bytes"))["n"] or 0
    runs = TrainingRun.objects.filter(owner=user).aggregate(n=Sum("size_bytes"))["n"] or 0
    return {
        "data_sources": int(sources),
        "datasets": int(datasets),
        "training_runs": int(runs),
        "total": int(sources + datasets + runs),
    }


def _size(size: float) -> str:
    for unit, factor in (("GB", 1024**3), ("MB", 1024**2), ("kB", 1024)):
        if size >= factor:
            return f"{size / factor:.1f} {unit}"
    return f"{int(size)} bytes"


def check_storage(user, adding: int = 0) -> None:
    """Raise :class:`StorageQuotaExceeded` if ``adding`` more bytes do not fit (with 0:
    if the quota is used up)."""
    limit = limits_for(user).storage_bytes
    if limit is None:
        return
    used = storage_used(user)["total"]
    if used + max(0, adding) <= limit and (adding > 0 or used < limit):
        return
    if adding > 0 and used < limit:
        message = (
            f"This upload ({_size(adding)}) does not fit into the storage of this account: "
            f"{_size(used)} of {_size(limit)} are used."
        )
    else:
        message = f"The storage of this account is full: {_size(used)} of {_size(limit)}."
    pricing = pricing_url()
    message += " Delete data sources, datasets or training runs you no longer need, or " + (
        f"choose a plan with more storage: {pricing}"
        if pricing
        else "ask the operator of this server for more space."
    )
    data: dict[str, Any] = {
        "storage": {"used_bytes": used, "limit_bytes": limit, "adding_bytes": adding}
    }
    if pricing:
        data["upgrade_url"] = pricing
    raise StorageQuotaExceeded(message, data)


# --------------------------------------------------------------------------- #
# Overview (account page, API, MCP)
# --------------------------------------------------------------------------- #


def _sources(user, period: date) -> list[dict[str, Any]]:
    """This month's usage by the web app, each API token and each connected app."""
    rows = (
        UsageRecord.objects.filter(user=user, period=period)
        .values("source", "api_token", "connection", "via")
        .annotate(total=Sum("quantity"))
        .order_by("-total")
    )
    tokens = {t.pk: t for t in ApiToken.objects.filter(owner=user, connection__isnull=True)}
    apps = {c.pk: c for c in OAuthConnection.objects.filter(owner=user).select_related("client")}
    result: list[dict[str, Any]] = []
    for row in rows:
        token = tokens.get(row["api_token"])
        app = apps.get(row["connection"])
        budget = None
        if token is not None:
            budget = token.budget_cpu_minutes
        elif app is not None:
            budget = app.budget_cpu_minutes
        name = row["via"] or ("Web app" if row["source"] == "web" else "")
        result.append(
            {
                "source": row["source"],
                "id": row["api_token"] or row["connection"],
                "name": name,
                "deleted": row["source"] != "web" and token is None and app is None,
                "used_cpu_minutes": minutes(row["total"]),
                "budget_cpu_minutes": budget,
                "cost": cost(row["total"]),
            }
        )
    return result


def usage_summary(actor: Actor) -> dict[str, Any]:
    """This month's usage, limits, what is left and when it resets (``Usage`` in
    frontend/src/api/types.ts). Sync the account's active runs first."""
    user = actor.user
    period = period_of()
    limits = limits_for(user)
    used = used_seconds(user, period=period)
    by_kind = {
        row["kind"]: minutes(row["total"])
        for row in UsageRecord.objects.filter(user=user, period=period)
        .values("kind")
        .annotate(total=Sum("quantity"))
    }
    account_limits = [
        b.limit for b in compute_budgets(Actor(user), period) if b.kind in ACCOUNT_KINDS
    ]
    limit = min(account_limits) if account_limits else None
    overage_used = float(
        UsageRecord.objects.filter(user=user, period=period).aggregate(n=Sum("overage"))["n"] or 0.0
    )
    plan = limits.plan
    # Other plans are only mentioned when the server sells some (plans with prices).
    offered = [p.to_dict() for p in plans().values() if p.price is not None]
    storage = storage_used(user)
    price, gb_price = cpu_price(), storage_price()
    caller = None
    if actor.source != "web":
        own = [b for b in compute_budgets(actor, period) if b.kind in ("token", "app")]
        caller = {
            "source": actor.source,
            "id": actor.token.pk if actor.token else actor.connection.pk,  # type: ignore[union-attr]
            "name": actor.via,
            "used_cpu_minutes": minutes(
                used_seconds(user, token=actor.token, connection=actor.connection, period=period)
            ),
            "budget_cpu_minutes": minutes(own[0].limit) if own else None,
            "remaining_cpu_minutes": minutes(own[0].remaining) if own else None,
        }
    remaining = [b.remaining for b in compute_budgets(actor, period)]
    active = TrainingRun.objects.filter(owner=user, status__in=ACTIVE_STATUSES)
    return {
        "period": {"start": period.isoformat(), "resets_at": _iso(period_end(period))},
        "plan": plan.to_dict(),
        # No limit of the plan applies (still measured); the own budgets do.
        "unlimited": limits.unlimited,
        "plans": offered if len(offered) > 1 else [],
        # Where plans can be bought, on a server that sells them (else null).
        "upgrade_url": pricing_url(),
        "overage": {
            "available": plan.overage_price_per_cpu_hour is not None
            and not limits.unlimited
            and overage_billable(user),
            "enabled": limits.overage,
            "price_per_cpu_hour": plan.overage_price_per_cpu_hour,
            "currency": settings.PRICE_CURRENCY,
            "cap_cpu_hours": (
                round(limits.overage_cap_seconds / 3600, 3)
                if limits.overage_cap_seconds is not None
                else None
            ),
            "used_cpu_minutes": minutes(overage_used),
            "cost": (
                round(overage_used / 3600 * plan.overage_price_per_cpu_hour, 4)
                if plan.overage_price_per_cpu_hour is not None
                else None
            ),
        },
        "compute": {
            "unit": "cpu_minutes",
            "used": minutes(used),
            "limit": minutes(limit),
            "remaining": minutes(max(0.0, limit - used)) if limit is not None else None,
            # What the caller (with its token or app budget) can still use.
            "available": minutes(min(remaining)) if remaining else None,
            "allowance": minutes(limits.monthly_seconds),
            "own_budget": minutes(limits.own_seconds),
            "run_limit": minutes(limits.run_seconds),
            "reserved_by_active_runs": minutes(_reserved(active)),
            "by_kind": {kind: by_kind.get(kind, 0.0) for kind in ("training", "prediction")},
            "by_source": _sources(user, period),
        },
        "storage": {
            "used_bytes": storage["total"],
            "limit_bytes": limits.storage_bytes,
            "remaining_bytes": (
                max(0, limits.storage_bytes - storage["total"])
                if limits.storage_bytes is not None
                else None
            ),
            "by_kind": {k: v for k, v in storage.items() if k != "total"},
        },
        "prices": {
            "currency": settings.PRICE_CURRENCY if (price or gb_price) else None,
            "per_cpu_hour": price,
            "per_gb_month": gb_price,
            "compute_cost": cost(used),
            "compute_limit_cost": cost(limit),
            "storage_cost_per_month": (
                round(storage["total"] / 1024**3 * gb_price, 4) if gb_price else None
            ),
        },
        "caller": caller,
    }


def token_usage(user, period: date | None = None) -> dict[tuple[str, int], float]:
    """This month's CPU minutes by ``("token", id)`` and ``("app", id)``."""
    rows = (
        UsageRecord.objects.filter(user=user, period=period or period_of())
        .filter(Q(api_token__isnull=False) | Q(connection__isnull=False))
        .values("api_token", "connection")
        .annotate(total=Sum("quantity"))
    )
    usage: dict[tuple[str, int], float] = {}
    for row in rows:
        key = ("token", row["api_token"]) if row["api_token"] else ("app", row["connection"])
        usage[key] = usage.get(key, 0.0) + (minutes(row["total"]) or 0.0)
    return usage


def export_records(user) -> dict[str, Any]:
    """usage.json of the account's data export: the ledger and the limits."""
    limits = limits_for(user)
    return {
        "plan": limits.plan.key,
        "unlimited": limits.unlimited,
        "overage_enabled": limits.overage,
        "overage_cap_cpu_hours": (
            limits.overage_cap_seconds / 3600 if limits.overage_cap_seconds is not None else None
        ),
        "limits": {
            "cpu_minutes_per_month": minutes(limits.monthly_seconds),
            "own_monthly_budget_cpu_minutes": minutes(limits.own_seconds),
            "cpu_minutes_per_run": minutes(limits.run_seconds),
            "storage_bytes": limits.storage_bytes,
        },
        "records": [
            {
                "month": record.period.isoformat()[:7],
                "kind": record.kind,
                "quantity": record.quantity,
                "overage": record.overage,
                "unit": record.unit,
                "source": record.source,
                "via": record.via or None,
                "training_run": str(record.run_id) if record.run_id else None,
                "run_name": record.run_name or None,
                "created_at": record.created_at,
                "updated_at": record.updated_at,
            }
            for record in UsageRecord.objects.filter(user=user).order_by("created_at", "pk")
        ],
    }
