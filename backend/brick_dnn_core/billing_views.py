"""The API of paid plans (billing.py): what the pricing page and the account page call, and
the endpoint Paddle sends its notifications to.

``GET /api/billing/`` is public and says whether this server sells plans at all. Everything
else answers 404 while billing is off. The account's endpoints need a website login (not an
API token: an agent must not buy or cancel anything) and changes are rate limited.
"""

from __future__ import annotations

import json
import logging

from django.conf import settings
from django.core.exceptions import RequestDataTooBig
from rest_framework import permissions, serializers, status
from rest_framework.decorators import api_view
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.throttling import ScopedRateThrottle
from rest_framework.views import APIView

from . import billing
from .accounts import AccountApiView
from .paddle import SIGNATURE_HEADER, SignatureError, verify_signature

logger = logging.getLogger(__name__)

# Paddle's notifications are a few kilobytes.
MAX_WEBHOOK_BYTES = 256 * 1024


@api_view(["GET"])
def billing_info(request: Request) -> Response:
    """Whether this server sells plans, and if so what the checkout needs and the plans
    (public): ``{"enabled": false}`` or billing.info()."""
    return Response(billing.info())


class BillingApiView(AccountApiView):
    """An endpoint of the own subscription: website login, 404 with billing off."""

    throttle_scope = "billing"

    def initial(self, request: Request, *args, **kwargs) -> None:
        billing.require()
        super().initial(request, *args, **kwargs)


class SubscriptionView(BillingApiView):
    """``GET``: the account's plan and subscription (billing.state)."""

    unthrottled_methods = ("GET", "HEAD", "OPTIONS")

    def get(self, request: Request) -> Response:
        return Response(billing.state(request.user))


class PlanSerializer(serializers.Serializer):
    plan = serializers.CharField(max_length=40)
    preview = serializers.BooleanField(required=False, default=False)


class CheckoutView(BillingApiView):
    """``POST {"plan"}``: what Paddle's checkout needs to sell the plan to this account:
    ``{"plan", "price_id", "customer_id", "custom_data"}``. Creates the account's customer at
    Paddle (with its e-mail address) the first time."""

    def post(self, request: Request) -> Response:
        serializer = PlanSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        return Response(billing.checkout(request.user, serializer.validated_data["plan"]))


class ChangePlanView(BillingApiView):
    """``POST {"plan", "preview"?}``: move the subscription to another plan (the new
    billing.state), or with ``preview`` say what Paddle would charge now."""

    def post(self, request: Request) -> Response:
        serializer = PlanSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        return Response(billing.change_plan(request.user, data["plan"], data["preview"]))


class CancelView(BillingApiView):
    """``POST``: cancel to the end of the period that is paid for."""

    def post(self, request: Request) -> Response:
        return Response(billing.cancel(request.user))


class ReactivateView(BillingApiView):
    """``POST``: take the cancellation back."""

    def post(self, request: Request) -> Response:
        return Response(billing.reactivate(request.user))


class PortalView(BillingApiView):
    """``POST``: ``{"url"}``, a link to Paddle's customer portal for this account (payment
    method, invoices). It logs the customer in and is valid for a short time."""

    def post(self, request: Request) -> Response:
        return Response({"url": billing.portal_url(request.user)})


class RefreshView(BillingApiView):
    """``POST``: ask Paddle for the subscription now (after a checkout, if its notification
    takes long) and return the new billing.state."""

    def post(self, request: Request) -> Response:
        return Response(billing.refresh(request.user))


class PaddleWebhookView(APIView):
    """``POST``: a notification from Paddle. The only proof is its signature
    (``Paddle-Signature``, see paddle.py), checked against the body as it was received.
    Answers 200 as soon as the event is stored and applied; an event that was seen before
    is answered the same way and changes nothing."""

    authentication_classes: list = []
    permission_classes = [permissions.AllowAny]
    parser_classes: list = []  # the body is read as it is, never parsed by the framework
    throttle_classes = [ScopedRateThrottle]
    throttle_scope = "billing_webhook"

    def post(self, request: Request) -> Response:
        cfg = billing.require()
        try:
            length = int(request.META.get("CONTENT_LENGTH") or 0)
        except ValueError:
            length = 0
        if length > MAX_WEBHOOK_BYTES:
            return _refuse(
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "The notification is too large."
            )
        try:
            body = request._request.body
        except RequestDataTooBig:
            return _refuse(
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "The notification is too large."
            )
        if len(body) > MAX_WEBHOOK_BYTES:
            return _refuse(
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "The notification is too large."
            )
        try:
            verify_signature(
                body,
                request.META.get("HTTP_" + SIGNATURE_HEADER.upper().replace("-", "_")),
                cfg.webhook_secrets,
                settings.PADDLE_WEBHOOK_TOLERANCE,
            )
        except SignatureError as exc:
            logger.warning("Billing: a notification was refused: %s", exc)
            return _refuse(status.HTTP_403_FORBIDDEN, "The signature is not valid.")
        try:
            event, new = billing.handle_event(json.loads(body.decode("utf-8")))
        except (UnicodeDecodeError, ValueError) as exc:  # not JSON, or not an event
            logger.warning("Billing: a signed notification could not be read: %s", exc)
            return _refuse(status.HTTP_400_BAD_REQUEST, "The notification could not be read.")
        return Response({"received": True, "new": new, "outcome": event.outcome})


def _refuse(code: int, detail: str) -> Response:
    return Response({"detail": detail}, status=code)
