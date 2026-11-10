"""``manage.py set_plan``: show or change an account's plan and whether it is unlimited.

    manage.py set_plan --list                    the plans of this server
    manage.py set_plan ada                       ada's plan and the limits that apply
    manage.py set_plan ada --plan plus           give ada the plan "plus"
    manage.py set_plan ada --plan default        back to the default plan
    manage.py set_plan ada --unlimited           no limits (usage is still measured)
    manage.py set_plan ada --limited             the limits of the plan, also for staff
    manage.py set_plan ada --staff-default       unlimited only if staff or superuser

The same as in the Django admin (user profiles); billing will set plans later
(brick_dnn_core.budgets.set_plan).
"""

from __future__ import annotations

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand, CommandError

from brick_dnn_core import budgets


def _amount(value, unit: str) -> str:
    return f"{value:g} {unit}" if value is not None else "unlimited"


class Command(BaseCommand):
    help = "Show or change the plan of an account, or mark it as unlimited."

    def add_arguments(self, parser) -> None:
        parser.add_argument("username", nargs="?", help="The account (username or e-mail)")
        parser.add_argument("--plan", help="A plan key (see --list); 'default' for the default")
        limits = parser.add_mutually_exclusive_group()
        limits.add_argument("--unlimited", action="store_true", help="No limits of the plan")
        limits.add_argument("--limited", action="store_true", help="The limits of the plan")
        limits.add_argument(
            "--staff-default",
            action="store_true",
            help="Unlimited only for staff members and superusers (the default)",
        )
        parser.add_argument("--list", action="store_true", help="List the plans of this server")

    def handle(self, *args, **options) -> None:
        if options["list"]:
            default = budgets.default_plan().key
            for plan in budgets.plans().values():
                self.stdout.write(
                    self._plan_line(plan) + ("  (default)" if plan.key == default else "")
                )
            if not options["username"]:
                return
        if not options["username"]:
            raise CommandError("Name an account, or use --list.")
        user = self._user(options["username"])

        key = options["plan"]
        if key is not None:
            try:
                budgets.set_plan(user, "" if key == "default" else key)
            except KeyError:
                known = ", ".join(budgets.plans())
                raise CommandError(f"Unknown plan {key!r}. Plans: {known}.") from None
        if options["unlimited"]:
            budgets.set_unlimited(user, True)
        elif options["limited"]:
            budgets.set_unlimited(user, False)
        elif options["staff_default"]:
            budgets.set_unlimited(user, None)

        limits = budgets.limits_for(user)
        self.stdout.write(f"{user.get_username()}: plan {limits.plan.name} ({limits.plan.key})")
        if limits.unlimited:
            self.stdout.write("  unlimited: no limits of the plan apply (usage is measured)")
            return
        monthly = limits.monthly_seconds / 3600 if limits.monthly_seconds is not None else None
        run = limits.run_seconds / 60 if limits.run_seconds is not None else None
        storage = limits.storage_bytes / 1024**3 if limits.storage_bytes is not None else None
        self.stdout.write(f"  compute per month:    {_amount(monthly, 'CPU hours')}")
        self.stdout.write(f"  compute per training: {_amount(run, 'CPU minutes')}")
        self.stdout.write(f"  storage:              {_amount(storage, 'GB')}")
        self.stdout.write(
            f"  trainings at a time: {_amount(limits.concurrent_trainings, '')}".rstrip()
        )
        if limits.overage:
            cap = limits.overage_cap_seconds
            self.stdout.write(
                f"  overage: on, cap {_amount(cap / 3600 if cap is not None else None, 'CPU hours')}"
            )

    def _user(self, name: str):
        User = get_user_model()
        user = User.objects.filter(username=name).first()
        if user is None and "@" in name:
            matches = list(User.objects.filter(email__iexact=name)[:2])
            user = matches[0] if len(matches) == 1 else None
        if user is None:
            raise CommandError(f"No account {name!r}.")
        return user

    def _plan_line(self, plan: budgets.Plan) -> str:
        price = f"{plan.price:g} per month" if plan.price is not None else "no price"
        parts = [
            _amount(plan.cpu_hours_per_month, "CPU hours/month"),
            _amount(plan.storage_gb, "GB"),
            _amount(plan.cpu_minutes_per_run, "CPU min per training"),
            f"{plan.concurrent_trainings or 'any'} at a time",
            f"priority {plan.priority}",
        ]
        if plan.overage_price_per_cpu_hour is not None:
            parts.append(f"overage {plan.overage_price_per_cpu_hour:g} per CPU hour")
        return f"{plan.key}: {plan.name}, {price}; " + ", ".join(parts)
