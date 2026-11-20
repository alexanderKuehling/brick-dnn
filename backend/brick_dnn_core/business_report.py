"""Business numbers for the operator: does the service work as a business?

Aggregates from the database only. Nothing is tracked for this: no page views, no events, no
identifiers; every number is a count or a sum over rows the service keeps anyway (accounts,
training runs, the usage ledger, what Paddle reported, share links). ``manage.py
business_report`` prints :func:`report`, the Django admin shows it (business_admin.py), and
docs/operations.md says how to read it.

**Per period** (calendar days, weeks from Monday, or months; UTC):

- *Sign-ups*: accounts created in the period.
- *Activated*: of those, the ones that started their first training within
  ``ACTIVATION_DAYS`` days of signing up (a training run on this server, or a row of the usage
  ledger: the ledger outlives a deleted run).
- *Trained*: accounts that started a training in the period.
- *New* and *ended subscriptions*: subscriptions whose first paid status, or whose end, Paddle
  reported in the period.
- *Paying* and *MRR* at the end of the period: what the notifications up to then say.
- *Converted*: of the accounts created in the period, the ones that have paid since.
- *CPU hours* used against what the training workers could have delivered.
- *Runs logged* from elsewhere and *links shared*.

**Now**: accounts, paying accounts by plan, MRR, storage against the disk, links that are on.

Honest limits, which the report states itself (``notes``):

- Deleted accounts take their rows with them (and unused runs are deleted after a while), so
  counts of past periods shrink over time. Staff accounts are left out of the account
  numbers; their compute still counts as load on the server.
- MRR is an estimate: the plans' prices from the settings (a yearly price counts with a
  twelfth), VAT included and before Paddle's fees; overage is not in it.
- A training's CPU time is counted in the period its ledger row was created in (when it
  started counting), not spread over the days it ran.
- The history of subscriptions is read from Paddle's notifications as they were applied
  (``BillingEvent.status``); "now" is the state of each subscription as last reported.
"""

from __future__ import annotations

import shutil
from datetime import datetime, timedelta
from datetime import timezone as dt_timezone
from typing import Any

from django.conf import settings
from django.contrib.auth import get_user_model
from django.db.models import Min, Q, Sum
from django.utils import timezone

from . import billing, budgets
from . import training_queue as queue
from .models import (
    BillingCustomer,
    BillingEvent,
    Dataset,
    DataSource,
    Project,
    RunComparison,
    TrainingRun,
    TrainingWorker,
    UsageRecord,
)

# How many periods are shown unless asked otherwise.
PERIODS = {"day": 14, "week": 8, "month": 6}
MAX_PERIODS = 60
# An account is "activated" if it starts its first training within this many days.
ACTIVATION_DAYS = 7
EXTERNAL = "external"


# --------------------------------------------------------------------------- #
# Periods
# --------------------------------------------------------------------------- #


def _start(period: str, moment: datetime) -> datetime:
    day = moment.astimezone(dt_timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    if period == "week":
        return day - timedelta(days=day.weekday())
    if period == "month":
        return day.replace(day=1)
    return day


def _next(period: str, start: datetime) -> datetime:
    if period == "day":
        return start + timedelta(days=1)
    if period == "week":
        return start + timedelta(days=7)
    year, month = (start.year + 1, 1) if start.month == 12 else (start.year, start.month + 1)
    return start.replace(year=year, month=month)


def _previous(period: str, start: datetime) -> datetime:
    return _start(period, start - timedelta(days=1))


def _label(period: str, start: datetime) -> str:
    if period == "month":
        return f"{start:%Y-%m}"
    if period == "week":
        year, week, _ = start.isocalendar()
        return f"{year}-W{week:02d}"
    return f"{start:%Y-%m-%d}"


def bounds(period: str, count: int, now: datetime) -> list[tuple[datetime, datetime]]:
    """The last ``count`` periods up to the running one, oldest first: ``(start, end)``."""
    start = _start(period, now)
    found = []
    for _ in range(count):
        found.append((start, _next(period, start)))
        start = _previous(period, start)
    return found[::-1]


# --------------------------------------------------------------------------- #
# Accounts and trainings
# --------------------------------------------------------------------------- #


def _accounts():
    """The accounts of users: staff members and superusers are the operator's own."""
    return get_user_model().objects.filter(is_staff=False, is_superuser=False)


def _first_trainings(user_ids: list[int]) -> dict[int, datetime]:
    """When each of the accounts first started a training (those that ever did)."""
    first: dict[int, datetime] = {}
    runs = (
        TrainingRun.objects.filter(owner__in=user_ids)
        .exclude(kind=EXTERNAL)
        .values_list("owner")
        .annotate(first=Min("created_at"))
    )
    ledger = (
        UsageRecord.objects.filter(user__in=user_ids, kind="training")
        .values_list("user")
        .annotate(first=Min("created_at"))
    )
    for user_id, moment in [*runs, *ledger]:
        if user_id not in first or moment < first[user_id]:
            first[user_id] = moment
    return first


def _trained(start: datetime, end: datetime) -> int:
    """Accounts that started a training in the period."""
    window = {"created_at__gte": start, "created_at__lt": end}
    users = set(
        TrainingRun.objects.filter(owner__is_staff=False, owner__is_superuser=False, **window)
        .exclude(kind=EXTERNAL)
        .values_list("owner", flat=True)
    )
    users.update(
        UsageRecord.objects.filter(
            user__is_staff=False, user__is_superuser=False, kind="training", **window
        ).values_list("user", flat=True)
    )
    return len(users)


# --------------------------------------------------------------------------- #
# Subscriptions
# --------------------------------------------------------------------------- #


def _monthly(plan: budgets.Plan | None, interval: str | None) -> float:
    """What a subscription to ``plan`` brings per month (a yearly price with a twelfth)."""
    if plan is None:
        return 0.0
    if interval == "year":
        return (plan.price_per_year or 0.0) / 12
    return plan.price or 0.0


def _events(cfg: billing.Config) -> list[BillingEvent]:
    """The applied notifications about subscriptions to this server's prices, oldest first.
    (A sandbox subscription from before going live has another price: it is left out.)"""
    rows = (
        BillingEvent.objects.filter(outcome="applied")
        .exclude(status="")
        .exclude(user=None)
        .order_by("occurred_at", "pk")
    )
    return [row for row in rows if cfg.interval_of(row.price_id) is not None]


def _paying(events: list[BillingEvent], until: datetime, cfg: billing.Config) -> dict[str, Any]:
    """Paying accounts by plan and the MRR, as the notifications before ``until`` say."""
    latest: dict[int, BillingEvent] = {}
    for event in events:
        if event.occurred_at < until:
            latest[event.user_id] = event
    return _by_plan(
        [(e.plan, cfg.interval_of(e.price_id)) for e in latest.values() if _pays(e.status)]
    )


def _pays(status: str) -> bool:
    return status in billing.ENTITLED


def _by_plan(paying: list[tuple[str, str | None]]) -> dict[str, Any]:
    plans = budgets.plans()
    counts: dict[str, dict[str, Any]] = {}
    for key, interval in paying:
        row = counts.setdefault(
            key,
            {
                "plan": key,
                "name": plans[key].name if key in plans else key,
                "accounts": 0,
                "yearly": 0,
                "mrr": 0.0,
            },
        )
        row["accounts"] += 1
        row["yearly"] += interval == "year"
        row["mrr"] += _monthly(plans.get(key), interval)
    order = list(plans)
    rows = sorted(
        counts.values(), key=lambda r: order.index(r["plan"]) if r["plan"] in order else 99
    )
    for row in rows:
        row["mrr"] = round(row["mrr"], 2)
    return {
        "accounts": sum(row["accounts"] for row in rows),
        "mrr": round(sum(row["mrr"] for row in rows), 2),
        "by_plan": rows,
    }


def _firsts(events: list[BillingEvent], wanted) -> dict[str, BillingEvent]:
    """For every subscription the first event whose status ``wanted`` accepts."""
    first: dict[str, BillingEvent] = {}
    for event in events:
        if event.subscription_id and wanted(event.status):
            first.setdefault(event.subscription_id, event)
    return first


# --------------------------------------------------------------------------- #
# Compute and storage
# --------------------------------------------------------------------------- #


def _capacity(start: datetime, end: datetime, now: datetime) -> tuple[float | None, str]:
    """CPU hours the server could have trained with in the period (up to now) and where the
    number comes from. None when it is not known."""
    end = min(end, now)
    if end <= start:
        return 0.0, ""
    if not settings.TRAINING_ENABLED:
        return 0.0, "training is switched off"
    if not queue.enabled():
        threads = settings.TRAINING_MAX_RUNS * budgets.fallback_threads()
        hours = threads * (end - start).total_seconds() / 3600
        return hours, "trainings at a time × CPU threads of this server"
    cutoff = now - timedelta(seconds=settings.WORKER_TIMEOUT)
    workers = TrainingWorker.objects.filter(started_at__lt=end).filter(
        Q(stopped_at__isnull=True) | Q(stopped_at__gte=start)
    )
    seconds, found = 0.0, False
    for worker in workers:
        found = True
        stopped = worker.stopped_at
        if stopped is None:  # still there, or dead and not noticed yet
            stopped = now if worker.heartbeat_at >= cutoff else worker.heartbeat_at
        overlap = (min(stopped, end) - max(worker.started_at, start)).total_seconds()
        seconds += max(0.0, overlap) * worker.slots * worker.threads
    if not found and start < now - timedelta(days=queue.KEEP_DAYS):
        return None, ""  # the rows of the workers of that time are deleted
    return seconds / 3600, "slots × CPU threads of the training workers while they ran"


def _cpu_hours(start: datetime, end: datetime) -> tuple[float, float]:
    """CPU hours counted in the period, and the part of them beyond plan allowances."""
    totals = UsageRecord.objects.filter(created_at__gte=start, created_at__lt=end).aggregate(
        used=Sum("quantity"), overage=Sum("overage")
    )
    return (totals["used"] or 0.0) / 3600, (totals["overage"] or 0.0) / 3600


def _storage() -> dict[str, Any]:
    used = sum(
        model.objects.aggregate(total=Sum("size_bytes"))["total"] or 0
        for model in (DataSource, Dataset, TrainingRun)
    )
    try:
        disk = shutil.disk_usage(settings.DATA_DIR)
    except OSError:
        disk = None
    return {
        "used_bytes": int(used),
        "disk_total_bytes": disk.total if disk else None,
        "disk_used_bytes": disk.used if disk else None,
        "disk_free_bytes": disk.free if disk else None,
    }


def _at_allowance(now: datetime) -> int:
    """Accounts that used up their plan's compute for the running month."""
    period = budgets.period_of(now)
    used = dict(
        UsageRecord.objects.filter(period=period, user__is_staff=False, user__is_superuser=False)
        .values_list("user")
        .annotate(total=Sum("quantity"))
    )
    count = 0
    for user in get_user_model().objects.filter(pk__in=used):
        allowance = budgets.limits_for(user).monthly_seconds
        count += allowance is not None and used[user.pk] >= allowance
    return count


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #


def _share(part: float, whole: float | None) -> float | None:
    return round(part / whole, 4) if whole else None


def _row(
    period: str,
    start: datetime,
    end: datetime,
    now: datetime,
    events: list[BillingEvent],
    cfg: billing.Config,
) -> dict[str, Any]:
    window = {"gte": start, "lt": end}
    cohort = dict(
        _accounts()
        .filter(date_joined__gte=start, date_joined__lt=end)
        .values_list("pk", "date_joined")
    )
    first = _first_trainings(list(cohort))
    deadline = timedelta(days=ACTIVATION_DAYS)
    activated = sum(
        1 for user, joined in cohort.items() if user in first and first[user] - joined <= deadline
    )
    paid_ever = {event.user_id for event in events if _pays(event.status)}
    converted = len(paid_ever & set(cohort))
    new = [e for e in _firsts(events, _pays).values() if start <= e.occurred_at < end]
    ended = [
        e
        for e in _firsts(events, lambda status: status == "canceled").values()
        if start <= e.occurred_at < end
    ]
    used, overage = _cpu_hours(start, end)
    capacity, _ = _capacity(start, end, now)
    paying = _paying(events, min(end, now), cfg) if cfg.enabled else None
    return {
        "label": _label(period, start),
        "start": start.isoformat(),
        "end": end.isoformat(),
        # The running period: its numbers still grow.
        "complete": end <= now,
        "signups": len(cohort),
        "activated": activated,
        "activation_rate": _share(activated, len(cohort)),
        # Sign-ups of the last days can still become activated.
        "activation_open": now < end + deadline,
        "trained_accounts": _trained(start, end),
        "new_subscriptions": len(new) if cfg.enabled else None,
        "ended_subscriptions": len(ended) if cfg.enabled else None,
        "paying_accounts": paying["accounts"] if paying else None,
        "paying_by_plan": paying["by_plan"] if paying else None,
        "mrr": paying["mrr"] if paying else None,
        "converted": converted if cfg.enabled else None,
        "conversion_rate": _share(converted, len(cohort)) if cfg.enabled else None,
        "cpu_hours_used": round(used, 2),
        "cpu_hours_overage": round(overage, 2),
        "cpu_hours_capacity": round(capacity, 1) if capacity is not None else None,
        "utilisation": _share(used, capacity),
        "external_runs": TrainingRun.objects.filter(
            kind=EXTERNAL, created_at__gte=start, created_at__lt=end
        ).count(),
        "shared_links": (
            Project.objects.filter(
                link_access="view", **{f"link_shared_at__{k}": v for k, v in window.items()}
            ).count()
            + RunComparison.objects.filter(
                **{f"created_at__{k}": v for k, v in window.items()}
            ).count()
        ),
    }


def _now(cfg: billing.Config, now: datetime) -> dict[str, Any]:
    accounts = _accounts().count()
    paying = None
    if cfg.enabled:
        customers = BillingCustomer.objects.filter(environment=cfg.environment)
        live = customers.filter(status__in=billing.ENTITLED)
        paying = {
            **_by_plan([(c.plan, cfg.interval_of(c.price_id)) for c in live]),
            # Still paying, but their subscription ends with the period they paid for.
            "ending": live.exclude(cancel_at=None).count(),
            # A payment failed; Paddle retries and the plan is kept for the grace period.
            "past_due": customers.filter(status="past_due").count(),
        }
    _, source = _capacity(now - timedelta(hours=1), now, now)
    return {
        "accounts": accounts,
        "paying": paying,
        "conversion_rate": _share(paying["accounts"], accounts) if paying else None,
        "at_allowance": _at_allowance(now),
        "storage": _storage(),
        "shared_links": {
            "networks": Project.objects.filter(link_access="view").count(),
            "comparisons": RunComparison.objects.count(),
        },
        "capacity_source": source,
    }


def notes(cfg: billing.Config) -> list[str]:
    """What the reader has to know to read the numbers honestly."""
    lines = [
        "Counted from the rows the database holds now: accounts that were deleted are gone "
        "from every period, and so are runs that were deleted (unused ones after "
        "BRICKDNN_DELETE_UNUSED_UPLOADS_DAYS). Past periods can shrink.",
        "Staff accounts are not in the account numbers; their trainings count as CPU hours.",
        f"Activated: first training within {ACTIVATION_DAYS} days of signing up. For sign-ups "
        "of the last days this can still grow (marked with *).",
        "CPU hours are counted in the period a training started counting, and the capacity is "
        "what the training workers could have used while they ran.",
        "Shared links: comparisons of runs and network links switched on in the period (a "
        "network link from before this was recorded has no date).",
    ]
    if cfg.enabled:
        lines += [
            "MRR is an estimate: the plans' prices from BRICKDNN_PLANS (a yearly price with a "
            "twelfth), VAT included, before Paddle's fees, without overage.",
            "Paying accounts and MRR of a period are its end, read from Paddle's "
            "notifications; “now” is each subscription as last reported.",
            "Converted: of the accounts created in the period, how many have paid since.",
        ]
    else:
        lines.append("Paid plans are off on this server: no subscription numbers.")
    return lines


def report(period: str = "month", count: int | None = None, now: datetime | None = None) -> dict:
    """The numbers of the last ``count`` periods (``"day"``, ``"week"`` or ``"month"``) and
    of now, as JSON-ready data. Raises ``ValueError`` for another period."""
    if period not in PERIODS:
        raise ValueError(f"The period must be one of {', '.join(PERIODS)}.")
    now = now or timezone.now()
    count = max(1, min(count or PERIODS[period], MAX_PERIODS))
    cfg = billing.config()
    events = _events(cfg) if cfg.enabled else []
    return {
        "generated_at": now.astimezone(dt_timezone.utc).isoformat(),
        "period": period,
        "billing": cfg.enabled,
        "currency": settings.PRICE_CURRENCY,
        "activation_days": ACTIVATION_DAYS,
        "now": _now(cfg, now),
        "periods": [
            _row(period, start, end, now, events, cfg) for start, end in bounds(period, count, now)
        ],
        "notes": notes(cfg),
    }


# --------------------------------------------------------------------------- #
# As text (the command) and as a table (the admin)
# --------------------------------------------------------------------------- #


def _percent(share: float | None) -> str:
    return f"{share * 100:.0f} %" if share is not None else "-"


def _number(value: Any, digits: int = 0) -> str:
    if value is None:
        return "-"
    return f"{value:,.{digits}f}" if digits else f"{value:,}"


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}{'' if number == 1 else 's'}"


def _gigabytes(size: int | None) -> str:
    return f"{size / 1024**3:.1f} GB" if size is not None else "-"


def columns(data: dict[str, Any]) -> list[tuple[str, list[str]]]:
    """The periods as columns of text, ``(heading, one cell per period)``: what the command
    prints and the admin page shows."""
    rows = data["periods"]
    currency = data["currency"]
    found: list[tuple[str, list[str]]] = [
        ("Period", [row["label"] + ("" if row["complete"] else " (running)") for row in rows]),
        ("Sign-ups", [_number(row["signups"]) for row in rows]),
        (
            "Activated",
            [
                f"{row['activated']} ({_percent(row['activation_rate'])})"
                + ("*" if row["activation_open"] and row["signups"] else "")
                for row in rows
            ],
        ),
        ("Trained", [_number(row["trained_accounts"]) for row in rows]),
    ]
    if data["billing"]:
        found += [
            ("New subs.", [_number(row["new_subscriptions"]) for row in rows]),
            ("Ended subs.", [_number(row["ended_subscriptions"]) for row in rows]),
            ("Paying", [_number(row["paying_accounts"]) for row in rows]),
            (f"MRR ({currency})", [_number(row["mrr"], 2) for row in rows]),
            (
                "Converted",
                [f"{row['converted']} ({_percent(row['conversion_rate'])})" for row in rows],
            ),
        ]
    found += [
        ("CPU h used", [_number(row["cpu_hours_used"], 2) for row in rows]),
        ("CPU h capacity", [_number(row["cpu_hours_capacity"], 1) for row in rows]),
        ("Used", [_percent(row["utilisation"]) for row in rows]),
        ("Runs logged", [_number(row["external_runs"]) for row in rows]),
        ("Links shared", [_number(row["shared_links"]) for row in rows]),
    ]
    return found


def summary(data: dict[str, Any]) -> list[tuple[str, str]]:
    """The numbers of now as ``(label, text)``."""
    now = data["now"]
    storage = now["storage"]
    lines = [("Accounts", _number(now["accounts"]))]
    paying = now["paying"]
    if paying is not None:
        plans = ", ".join(
            f"{row['name']} {row['accounts']}"
            + (f" ({row['yearly']} yearly)" if row["yearly"] else "")
            for row in paying["by_plan"]
        )
        lines += [
            (
                "Paying accounts",
                f"{paying['accounts']} ({_percent(now['conversion_rate'])} of all accounts)"
                + (f": {plans}" if plans else ""),
            ),
            ("Estimated MRR", f"{paying['mrr']:,.2f} {data['currency']}"),
            (
                "Of those",
                f"{paying['ending']} cancelled to the end of their period, "
                f"{paying['past_due']} more with a failed payment (in the grace period)",
            ),
        ]
    disk = (
        f" of a {_gigabytes(storage['disk_total_bytes'])} disk "
        f"({_gigabytes(storage['disk_free_bytes'])} free)"
        if storage["disk_total_bytes"]
        else ""
    )
    lines += [
        ("At their monthly allowance", _number(now["at_allowance"])),
        ("Stored by accounts", f"{_gigabytes(storage['used_bytes'])}{disk}"),
        (
            "Links that are on",
            f"{_count(now['shared_links']['networks'], 'network')}, "
            f"{_count(now['shared_links']['comparisons'], 'comparison')}",
        ),
    ]
    if now["capacity_source"]:
        lines.append(("CPU capacity is", now["capacity_source"]))
    return lines


def text(data: dict[str, Any]) -> str:
    """The report as plain text for a terminal."""
    out = [f"Brick DNN business report, {data['generated_at'][:16].replace('T', ' ')} UTC", ""]
    width = max(len(label) for label, _ in summary(data))
    out += [f"{label + ':':<{width + 2}}{value}" for label, value in summary(data)]
    out += ["", f"By {data['period']}:"]
    table = columns(data)
    widths = [max(len(head), *(len(cell) for cell in cells)) for head, cells in table]
    lines = [[head for head, _ in table]]
    lines += [[cells[index] for _, cells in table] for index in range(len(data["periods"]))]
    for line in lines:
        out.append("  ".join(f"{cell:<{w}}" for cell, w in zip(line, widths, strict=True)).rstrip())
    out += ["", *(f"- {line}" for line in data["notes"])]
    return "\n".join(out)
