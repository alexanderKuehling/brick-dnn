"""Paddle Billing: the handful of API calls Brick DNN needs, and webhook signatures.

Paddle is the merchant of record: it sells the plans, collects the payments and the tax and
sends the invoices; Brick DNN only learns which subscription an account has (see billing.py,
docs/billing.md). This module knows nothing about accounts or plans.

**API.** JSON over HTTPS with ``Authorization: Bearer <API key>`` and ``Paddle-Version: 1``,
at ``https://api.paddle.com`` (live) or ``https://sandbox-api.paddle.com`` (sandbox). An answer
is ``{"data": …, "meta": …}``, an error ``{"error": {"type", "code", "detail", …}}``. Requests
that change something are never repeated here: whoever calls decides what a failure means
(see the overage charges in billing.py). Reading requests are tried once more after a short
wait if Paddle is busy (HTTP 429 or 5xx).

**Webhooks.** Paddle signs every notification with the secret of its notification
destination: the header ``Paddle-Signature: ts=<unix time>;h1=<hex>``, where ``h1`` is the
HMAC-SHA256 of ``<ts>:<raw body>``. The body must be the bytes as received.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from typing import Any
from urllib.parse import quote, urlencode

logger = logging.getLogger(__name__)

API_URLS = {
    "live": "https://api.paddle.com",
    "sandbox": "https://sandbox-api.paddle.com",
}
API_VERSION = "1"
USER_AGENT = "BrickDNN (+https://github.com/alexanderKuehling/brick-dnn)"
SIGNATURE_HEADER = "Paddle-Signature"
# Paddle answers within a few seconds; a user waits for some of these calls.
TIMEOUT_SECONDS = 15
RETRY_STATUSES = (429, 500, 502, 503, 504)


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class PaddleError(Exception):
    """Paddle refused a request: ``status`` is the HTTP status, ``code`` Paddle's error code
    (e.g. ``customer_already_exists``)."""

    def __init__(self, message: str, status: int = 0, code: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.code = code


class PaddleUnavailable(PaddleError):
    """No answer, or Paddle is busy or broken (network errors, HTTP 429 and 5xx): it is not
    known whether the request was carried out."""


class SignatureError(Exception):
    """A webhook whose signature is missing, malformed, wrong or too old."""


# --------------------------------------------------------------------------- #
# Webhook signatures
# --------------------------------------------------------------------------- #


def parse_signature(header: str) -> tuple[int, list[str]]:
    """``(timestamp, [h1 signatures])`` from a ``Paddle-Signature`` header. There can be
    several signatures while a secret is being rotated."""
    timestamp: int | None = None
    signatures: list[str] = []
    for part in header.split(";"):
        key, found, value = part.strip().partition("=")
        if not found:
            continue
        if key == "ts":
            try:
                timestamp = int(value)
            except ValueError:
                raise SignatureError("The timestamp of the signature is not a number.") from None
        elif key == "h1":
            signatures.append(value.strip().lower())
    if timestamp is None or not signatures:
        raise SignatureError("The signature header has no timestamp or no signature.")
    return timestamp, signatures


def sign(body: bytes, secret: str, timestamp: int) -> str:
    """The ``h1`` signature of ``body`` at ``timestamp``."""
    message = str(timestamp).encode() + b":" + body
    return hmac.new(secret.encode(), message, hashlib.sha256).hexdigest()


def signature_header(body: bytes, secret: str, timestamp: int | None = None) -> str:
    """A ``Paddle-Signature`` header for ``body`` (for tests and for trying things out)."""
    timestamp = int(time.time()) if timestamp is None else timestamp
    return f"ts={timestamp};h1={sign(body, secret, timestamp)}"


def verify_signature(
    body: bytes,
    header: str | None,
    secrets: Iterable[str],
    tolerance: int,
    now: float | None = None,
) -> int:
    """Check that ``body`` was signed with one of ``secrets`` at most ``tolerance`` seconds
    ago (0: any age); return the timestamp. Raises :class:`SignatureError`."""
    if not header:
        raise SignatureError(f"The {SIGNATURE_HEADER} header is missing.")
    timestamp, signatures = parse_signature(header)
    valid = False
    for secret in secrets:
        if not secret:
            continue
        expected = sign(body, secret, timestamp)
        for signature in signatures:
            # Constant time; all candidates are compared, whichever matches.
            valid = hmac.compare_digest(expected, signature) or valid
    if not valid:
        raise SignatureError("The signature does not match.")
    if tolerance > 0:
        age = (time.time() if now is None else now) - timestamp
        if abs(age) > tolerance:
            raise SignatureError(
                f"The signature is {age:.0f} seconds old (allowed: {tolerance} seconds)."
            )
    return timestamp


# --------------------------------------------------------------------------- #
# API client
# --------------------------------------------------------------------------- #


class PaddleClient:
    """The API calls of billing.py. Every method returns the ``data`` of Paddle's answer and
    raises :class:`PaddleError` (refused) or :class:`PaddleUnavailable` (outcome unknown)."""

    def __init__(self, api_key: str, environment: str, base_url: str = "") -> None:
        self.api_key = api_key
        self.base_url = (base_url or API_URLS[environment]).rstrip("/")

    # -- transport ----------------------------------------------------------- #

    def request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        query: dict[str, str] | None = None,
    ) -> Any:
        url = self.base_url + path + (f"?{urlencode(query)}" if query else "")
        attempts = 2 if method == "GET" else 1
        for attempt in range(attempts):
            try:
                return self._send(method, url, body)
            except PaddleUnavailable:
                if attempt + 1 == attempts:
                    raise
                time.sleep(1.0)
        raise AssertionError("unreachable")  # pragma: no cover

    def _send(self, method: str, url: str, body: dict[str, Any] | None) -> Any:
        request = urllib.request.Request(  # noqa: S310 (the URL is https://…paddle.com)
            url,
            data=json.dumps(body).encode() if body is not None else None,
            method=method,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Paddle-Version": API_VERSION,
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": USER_AGENT,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:  # noqa: S310
                payload = _json(response.read())
        except urllib.error.HTTPError as exc:
            error = _json(exc.read()).get("error") or {}
            code = str(error.get("code") or "")
            detail = str(error.get("detail") or exc.reason or "")
            message = f"Paddle answered HTTP {exc.code} {code}: {detail}".strip()
            if exc.code in RETRY_STATUSES:
                raise PaddleUnavailable(message, exc.code, code) from None
            raise PaddleError(message, exc.code, code) from None
        except (OSError, ValueError) as exc:  # no connection, timeout
            raise PaddleUnavailable(f"Paddle did not answer: {exc}") from None
        if "data" not in payload:
            raise PaddleUnavailable("Paddle's answer could not be read.")
        return payload["data"]

    # -- customers ----------------------------------------------------------- #

    def create_customer(self, email: str, custom_data: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", "/customers", {"email": email, "custom_data": custom_data})

    def find_customers(self, email: str) -> list[dict[str, Any]]:
        """Customers with this e-mail address, archived ones included."""
        query = {"email": email, "status": "active,archived"}
        return self.request("GET", "/customers", query=query) or []

    def update_customer(self, customer_id: str, **fields: Any) -> dict[str, Any]:
        return self.request("PATCH", f"/customers/{quote(customer_id)}", fields)

    def create_portal_session(self, customer_id: str, subscription_ids: list[str]) -> str:
        """A link to Paddle's customer portal that logs the customer in (valid for a short
        time): payment method, invoices, cancelling."""
        data = self.request(
            "POST",
            f"/customers/{quote(customer_id)}/portal-sessions",
            {"subscription_ids": subscription_ids} if subscription_ids else {},
        )
        return data["urls"]["general"]["overview"]

    # -- subscriptions ------------------------------------------------------- #

    def get_subscription(self, subscription_id: str) -> dict[str, Any]:
        return self.request("GET", f"/subscriptions/{quote(subscription_id)}")

    def list_subscriptions(self, customer_id: str) -> list[dict[str, Any]]:
        query = {"customer_id": customer_id, "per_page": "50"}
        return self.request("GET", "/subscriptions", query=query) or []

    def change_price(
        self, subscription_id: str, price_id: str, proration: str, preview: bool = False
    ) -> dict[str, Any]:
        """Replace what the subscription bills by ``price_id``. ``proration`` is Paddle's
        ``proration_billing_mode``. With ``preview``, nothing changes: the answer says what
        would be charged (``update_summary``, ``immediate_transaction``)."""
        path = f"/subscriptions/{quote(subscription_id)}" + ("/preview" if preview else "")
        body = {
            "items": [{"price_id": price_id, "quantity": 1}],
            "proration_billing_mode": proration,
        }
        return self.request("PATCH", path, body)

    def cancel_subscription(self, subscription_id: str, immediately: bool) -> dict[str, Any]:
        """Cancel now, or at the end of the period that is paid for (a scheduled change)."""
        effective = "immediately" if immediately else "next_billing_period"
        path = f"/subscriptions/{quote(subscription_id)}/cancel"
        return self.request("POST", path, {"effective_from": effective})

    def remove_scheduled_change(self, subscription_id: str) -> dict[str, Any]:
        """Take back a scheduled cancellation: the subscription renews as before."""
        path = f"/subscriptions/{quote(subscription_id)}"
        return self.request("PATCH", path, {"scheduled_change": None})

    def create_charge(
        self,
        subscription_id: str,
        *,
        product_id: str,
        description: str,
        amount_minor: int,
        currency: str,
        immediately: bool,
        custom_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Bill a one-time amount (in the smallest unit of ``currency``, tax included) to the
        subscription: with its next renewal, or ``immediately`` as a transaction of its own.
        The item is a price made for this charge, of the product ``product_id``; its name is
        what the invoice shows."""
        price: dict[str, Any] = {
            "product_id": product_id,
            "description": description[:500],
            "name": description[:150],
            "unit_price": {"amount": str(int(amount_minor)), "currency_code": currency},
            "tax_mode": "internal",
        }
        if custom_data:
            price["custom_data"] = custom_data
        body = {
            "effective_from": "immediately" if immediately else "next_billing_period",
            "items": [{"quantity": 1, "price": price}],
        }
        return self.request("POST", f"/subscriptions/{quote(subscription_id)}/charge", body)

    # -- catalog (for checking the configuration) ---------------------------- #

    def get_price(self, price_id: str) -> dict[str, Any]:
        return self.request("GET", f"/prices/{quote(price_id)}")

    def get_product(self, product_id: str) -> dict[str, Any]:
        return self.request("GET", f"/products/{quote(product_id)}")


def _json(raw: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(raw.decode("utf-8")) if raw else {}
    except (UnicodeDecodeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}
