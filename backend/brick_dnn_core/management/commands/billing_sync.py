"""``python manage.py billing_sync [--check]``: paid plans, once a day (see billing.py).

Without options: asks Paddle for every subscription again (in case a notification was
lost), ends grace periods after failed payments, and bills compute beyond the allowance of
months that are over. Does nothing while billing is off. Docker Compose runs it daily after
``purge_old_data`` (the ``cleanup`` service).

``--check`` only looks at the configuration: whether billing is on and, with a look at
Paddle, whether the API key works and every price fits its plan. See docs/billing.md.
"""

from __future__ import annotations

from decimal import Decimal

from django.conf import settings
from django.core.management.base import BaseCommand

from brick_dnn_core import billing, budgets
from brick_dnn_core.paddle import PaddleError


class Command(BaseCommand):
    help = (
        "Paid plans (Paddle): refresh all subscriptions, end grace periods and bill overage. "
        "With --check: verify the configuration instead."
    )

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--check",
            action="store_true",
            help="Only check the settings and, with Paddle, the API key, prices and product.",
        )

    def handle(self, *args, **options) -> None:
        cfg = billing.config()
        if options["check"]:
            self.check_configuration(cfg)
            return
        if not cfg.enabled:
            if cfg.configured:
                # Half configured (or the plans are not set for this service): say so, because
                # nothing is checked or billed.
                self.stdout.write(
                    "billing_sync: paid plans are OFF, the settings are incomplete: "
                    + " ".join(cfg.problems)
                )
            else:
                self.stdout.write("billing_sync: paid plans are off, nothing to do.")
            return
        result = billing.sync_all(log=self.stdout.write)
        self.stdout.write(
            f"billing_sync: checked {result.checked} subscriptions, changed {result.plans_changed} "
            f"plans, {result.charges} overage charges, {result.errors} errors."
        )

    # ------------------------------------------------------------------ #

    def check_configuration(self, cfg: billing.Config) -> None:
        write = self.stdout.write
        if not cfg.configured:
            write("Paid plans are off: no BRICKDNN_PADDLE_… setting is set.")
            return
        for problem in cfg.problems:
            write(self.style.ERROR(f"✗ {problem}"))
        if not cfg.enabled:
            write("Paid plans are OFF until this is fixed (docs/billing.md).")
            return
        write(f"Paid plans are ON ({cfg.environment}).")
        write(
            f"  Notifications: {len(cfg.webhook_secrets)} secret(s) for /api/billing/paddle/webhook/"
        )
        api = billing.client(cfg)
        plans = budgets.plans()
        currency = settings.PRICE_CURRENCY
        problems = 0
        for key, price_ids in cfg.prices.items():
            for price_id in price_ids:
                try:
                    price = api.get_price(price_id)
                except PaddleError as exc:
                    problems += 1
                    write(self.style.ERROR(f"✗ {key}: {price_id}: {exc}"))
                    continue
                for line in self._price_problems(plans[key], price, currency):
                    problems += 1
                    write(self.style.ERROR(f"✗ {key}: {price_id}: {line}"))
                unit = price.get("unit_price") or {}
                write(
                    f"  {key}: {price_id} = {unit.get('amount')} {unit.get('currency_code')} "
                    f"(smallest unit), tax {price.get('tax_mode')}, "
                    f"every {(price.get('billing_cycle') or {}).get('interval', '?')}"
                )
        if cfg.overage_product:
            try:
                product = api.get_product(cfg.overage_product)
                write(f"  Overage product: {cfg.overage_product} ({product.get('name')})")
                if product.get("status") != "active":
                    problems += 1
                    write(self.style.ERROR("✗ The overage product is not active."))
            except PaddleError as exc:
                problems += 1
                write(self.style.ERROR(f"✗ Overage product {cfg.overage_product}: {exc}"))
        else:
            write(
                "  No BRICKDNN_PADDLE_OVERAGE_PRODUCT: compute beyond the allowance is not offered."
            )
        write(
            self.style.SUCCESS("Everything fits.")
            if problems == 0
            else self.style.ERROR(f"{problems} problem(s), see above.")
        )

    @staticmethod
    def _price_problems(plan: budgets.Plan, price: dict, currency: str) -> list[str]:
        """What about a Paddle price does not fit the plan it is configured for."""
        found = []
        unit = price.get("unit_price") or {}
        if price.get("status") != "active":
            found.append("the price is not active.")
        cycle = price.get("billing_cycle") or {}
        if cycle.get("interval") != "month" or cycle.get("frequency") != 1:
            found.append("it is not billed monthly.")
        if unit.get("currency_code") != currency:
            found.append(
                f"its currency is {unit.get('currency_code')}, BRICKDNN_PRICE_CURRENCY is {currency}."
            )
        elif plan.price is not None:
            factor = 1 if currency in billing.ZERO_DECIMAL else 100
            expected = int(Decimal(str(plan.price)) * factor)
            if str(unit.get("amount")) != str(expected):
                found.append(
                    f"it costs {unit.get('amount')} (smallest unit), the plan says {expected}."
                )
        if price.get("tax_mode") == "external":
            found.append("tax is added on top (tax_mode external); the app shows prices incl. tax.")
        return found
