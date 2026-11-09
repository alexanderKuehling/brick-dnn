"""A Paddle that lives in memory, for the billing tests: no network.

It has the methods of ``brick_dnn_core.paddle.PaddleClient`` that billing.py calls, keeps
customers and subscriptions the way Paddle's API returns them, and builds the notifications
Paddle would send (``event``), which the tests sign and post themselves.
"""

from __future__ import annotations

import itertools
from datetime import datetime, timedelta
from typing import Any

from django.utils import timezone

from brick_dnn_core.paddle import PaddleError, PaddleUnavailable


def iso(moment: datetime | None) -> str | None:
    return moment.isoformat().replace("+00:00", "Z") if moment else None


class FakePaddle:
    def __init__(self) -> None:
        self.customers: dict[str, dict[str, Any]] = {}
        self.subscriptions: dict[str, dict[str, Any]] = {}
        # One-time charges that were billed: (subscription id, the request).
        self.charges: list[tuple[str, dict[str, Any]]] = []
        # Every API call: (method name, arguments).
        self.calls: list[tuple[str, tuple]] = []
        # Method name -> the exception its next calls raise (until removed).
        self.failing: dict[str, Exception] = {}
        self.portal_sessions = 0
        self._ids = itertools.count(1)
        # Paddle's clock: every change moves it on a second, so that later is later.
        self.now = timezone.now().replace(microsecond=0) - timedelta(minutes=5)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _id(self, prefix: str) -> str:
        return f"{prefix}_{next(self._ids):026d}"

    def tick(self) -> datetime:
        self.now += timedelta(seconds=1)
        return self.now

    def _call(self, name: str, *args: Any) -> None:
        self.calls.append((name, args))
        if name in self.failing:
            raise self.failing[name]

    def called(self, name: str) -> list[tuple]:
        return [args for method, args in self.calls if method == name]

    def down(self, *names: str) -> None:
        """Paddle does not answer these calls (the outcome is unknown)."""
        for name in names:
            self.failing[name] = PaddleUnavailable("Paddle did not answer: timed out")

    def refuses(self, name: str, code: str = "invalid_field", status: int = 400) -> None:
        self.failing[name] = PaddleError(f"Paddle answered HTTP {status} {code}", status, code)

    def up(self) -> None:
        self.failing.clear()

    # ------------------------------------------------------------------ #
    # The API (paddle.PaddleClient)
    # ------------------------------------------------------------------ #

    def create_customer(self, email: str, custom_data: dict[str, Any]) -> dict[str, Any]:
        self._call("create_customer", email, custom_data)
        if any(customer["email"] == email for customer in self.customers.values()):
            raise PaddleError(
                "Paddle answered HTTP 409 customer_already_exists", 409, "customer_already_exists"
            )
        customer = {
            "id": self._id("ctm"),
            "email": email,
            "status": "active",
            "custom_data": dict(custom_data),
        }
        self.customers[customer["id"]] = customer
        return dict(customer)

    def find_customers(self, email: str) -> list[dict[str, Any]]:
        self._call("find_customers", email)
        return [dict(c) for c in self.customers.values() if c["email"] == email]

    def update_customer(self, customer_id: str, **fields: Any) -> dict[str, Any]:
        self._call("update_customer", customer_id, fields)
        self.customers[customer_id].update(fields)
        return dict(self.customers[customer_id])

    def create_portal_session(self, customer_id: str, subscription_ids: list[str]) -> str:
        self._call("create_portal_session", customer_id, subscription_ids)
        self.portal_sessions += 1
        return f"https://sandbox-customer-portal.paddle.com/cpl_test?token={self.portal_sessions}"

    def get_subscription(self, subscription_id: str) -> dict[str, Any]:
        self._call("get_subscription", subscription_id)
        if subscription_id not in self.subscriptions:
            raise PaddleError("Paddle answered HTTP 404 not_found", 404, "not_found")
        return self.entity(subscription_id)

    def list_subscriptions(self, customer_id: str) -> list[dict[str, Any]]:
        self._call("list_subscriptions", customer_id)
        return [
            self.entity(key)
            for key, value in self.subscriptions.items()
            if value["customer_id"] == customer_id
        ]

    def change_price(
        self, subscription_id: str, price_id: str, proration: str, preview: bool = False
    ) -> dict[str, Any]:
        self._call("change_price", subscription_id, price_id, proration, preview)
        subscription = self.subscriptions[subscription_id]
        if preview:
            entity = self.entity(subscription_id)
            entity["items"] = [{"status": "active", "quantity": 1, "price": {"id": price_id}}]
            charge = proration == "prorated_immediately"
            amount = "450" if charge else "310"
            entity["update_summary"] = {
                "charge": {"amount": amount if charge else "0", "currency_code": "EUR"},
                "credit": {"amount": "0" if charge else amount, "currency_code": "EUR"},
                "result": {
                    "action": "charge" if charge else "credit",
                    "amount": amount,
                    "currency_code": "EUR",
                },
            }
            return entity
        subscription["price_id"] = price_id
        subscription["updated_at"] = self.tick()
        return self.entity(subscription_id)

    def cancel_subscription(self, subscription_id: str, immediately: bool) -> dict[str, Any]:
        self._call("cancel_subscription", subscription_id, immediately)
        subscription = self.subscriptions[subscription_id]
        if immediately:
            self.end(subscription_id)
        else:
            subscription["cancel_at"] = subscription["period_end"]
            subscription["updated_at"] = self.tick()
        return self.entity(subscription_id)

    def remove_scheduled_change(self, subscription_id: str) -> dict[str, Any]:
        self._call("remove_scheduled_change", subscription_id)
        subscription = self.subscriptions[subscription_id]
        subscription["cancel_at"] = None
        subscription["updated_at"] = self.tick()
        return self.entity(subscription_id)

    def create_charge(self, subscription_id: str, **charge: Any) -> dict[str, Any]:
        self._call("create_charge", subscription_id, charge)
        self.charges.append((subscription_id, charge))
        return self.entity(subscription_id)

    def get_price(self, price_id: str) -> dict[str, Any]:
        self._call("get_price", price_id)
        amount = {"pri_starter": "500", "pri_plus": "1400", "pri_pro": "4500"}.get(price_id)
        if amount is None:
            raise PaddleError("Paddle answered HTTP 404 not_found", 404, "not_found")
        return {
            "id": price_id,
            "status": "active",
            "tax_mode": "internal",
            "unit_price": {"amount": amount, "currency_code": "EUR"},
            "billing_cycle": {"interval": "month", "frequency": 1},
        }

    def get_product(self, product_id: str) -> dict[str, Any]:
        self._call("get_product", product_id)
        return {"id": product_id, "name": "Extra compute", "status": "active"}

    # ------------------------------------------------------------------ #
    # What happens at Paddle
    # ------------------------------------------------------------------ #

    def subscribe(
        self,
        customer_id: str,
        price_id: str,
        custom_data: dict[str, Any] | None = None,
        status: str = "active",
    ) -> dict[str, Any]:
        """A checkout was paid: a new subscription. Returns its entity."""
        start = self.tick()
        subscription_id = self._id("sub")
        self.subscriptions[subscription_id] = {
            "customer_id": customer_id,
            "price_id": price_id,
            "status": status,
            "custom_data": custom_data,
            "period_start": start,
            "period_end": start + timedelta(days=30),
            "cancel_at": None,
            "canceled_at": None,
            "created_at": start,
            "updated_at": start,
        }
        return self.entity(subscription_id)

    def set_status(self, subscription_id: str, status: str) -> dict[str, Any]:
        """E.g. ``past_due`` after a failed payment, ``active`` once it went through."""
        subscription = self.subscriptions[subscription_id]
        subscription["status"] = status
        subscription["updated_at"] = self.tick()
        return self.entity(subscription_id)

    def end(self, subscription_id: str) -> dict[str, Any]:
        """The subscription is cancelled for good."""
        subscription = self.subscriptions[subscription_id]
        now = self.tick()
        subscription.update(status="canceled", cancel_at=None, canceled_at=now, updated_at=now)
        return self.entity(subscription_id)

    def entity(self, subscription_id: str) -> dict[str, Any]:
        """The subscription as Paddle's API and notifications describe it."""
        subscription = self.subscriptions[subscription_id]
        ended = subscription["status"] == "canceled"
        cancel_at = subscription["cancel_at"]
        return {
            "id": subscription_id,
            "status": subscription["status"],
            "customer_id": subscription["customer_id"],
            "address_id": "add_01test",
            "business_id": None,
            "currency_code": "EUR",
            "created_at": iso(subscription["created_at"]),
            "updated_at": iso(subscription["updated_at"]),
            "started_at": iso(subscription["created_at"]),
            "first_billed_at": iso(subscription["created_at"]),
            "next_billed_at": None if ended or cancel_at else iso(subscription["period_end"]),
            "paused_at": None,
            "canceled_at": iso(subscription["canceled_at"]),
            "collection_mode": "automatic",
            "current_billing_period": None
            if ended
            else {
                "starts_at": iso(subscription["period_start"]),
                "ends_at": iso(subscription["period_end"]),
            },
            "billing_cycle": {"interval": "month", "frequency": 1},
            "scheduled_change": {
                "action": "cancel",
                "effective_at": iso(cancel_at),
                "resume_at": None,
            }
            if cancel_at
            else None,
            "items": [
                {
                    "status": "active",
                    "quantity": 1,
                    "recurring": True,
                    "price": {"id": subscription["price_id"], "product_id": "pro_01plan"},
                }
            ],
            "custom_data": subscription["custom_data"],
        }

    def event(
        self, event_type: str, data: dict[str, Any], at: datetime | None = None
    ) -> dict[str, Any]:
        """A notification as Paddle sends it."""
        return {
            "event_id": self._id("evt"),
            "event_type": event_type,
            "occurred_at": iso(at or self.tick()),
            "notification_id": self._id("ntf"),
            "data": data,
        }
