"""Django admin for paid plans (billing.py): what Paddle reported, read-only.

Subscriptions are changed in Paddle (refunds, cancelling for a customer); the next
notification or ``manage.py billing_sync`` brings the change here. The one thing decided
here is what happens to an overage charge whose answer from Paddle got lost.
"""

from __future__ import annotations

from django.contrib import admin, messages
from django.contrib.auth import get_user_model
from django.urls import NoReverseMatch, reverse
from django.utils.html import format_html

from .models import BillingCustomer, BillingEvent, OverageCharge


def user_link(user) -> str:
    """The account, linked to its page in the admin."""
    if user is None:
        return "-"
    model = get_user_model()._meta
    try:
        url = reverse(f"admin:{model.app_label}_{model.model_name}_change", args=[user.pk])
    except NoReverseMatch:  # the user model is not in the admin
        return str(user)
    return format_html('<a href="{}">{}</a>', url, user)


class ReadOnlyAdmin(admin.ModelAdmin):
    """Written by the server only: nothing can be added or edited."""

    def has_add_permission(self, request) -> bool:
        return False

    def has_change_permission(self, request, obj=None) -> bool:
        return False

    def get_queryset(self, request):
        return super().get_queryset(request).select_related("user")

    @admin.display(description="Account", ordering="user__username")
    def account(self, obj) -> str:
        return user_link(obj.user)


@admin.register(BillingCustomer)
class BillingCustomerAdmin(ReadOnlyAdmin):
    """Accounts with a customer at Paddle and their subscription, as Paddle last reported
    it. Deleting a row does not cancel anything at Paddle."""

    list_display = (
        "account",
        "plan",
        "status",
        "environment",
        "next_billed_at",
        "cancel_at",
        "past_due_since",
        "updated_at",
    )
    list_filter = ("status", "plan", "environment")
    search_fields = ("user__username", "user__email", "customer_id", "subscription_id")
    fields = (
        "account",
        "environment",
        "customer_id",
        "subscription_id",
        "status",
        "plan",
        "granted_plan",
        "price_id",
        "current_period_start",
        "current_period_end",
        "next_billed_at",
        "cancel_at",
        "canceled_at",
        "past_due_since",
        "payment_failed_at",
        "last_event_at",
        "created_at",
        "updated_at",
    )

    def has_delete_permission(self, request, obj=None) -> bool:
        return False  # goes with the account; the subscription would stay at Paddle


@admin.register(BillingEvent)
class BillingEventAdmin(ReadOnlyAdmin):
    """Notifications received from Paddle and what was done with them."""

    list_display = ("occurred_at", "event_type", "account", "subscription_id", "outcome", "note")
    list_filter = ("outcome", "event_type")
    search_fields = ("event_id", "subscription_id", "user__username", "user__email")
    date_hierarchy = "occurred_at"
    fields = (
        "event_id",
        "event_type",
        "occurred_at",
        "received_at",
        "account",
        "subscription_id",
        "outcome",
        "note",
    )


@admin.register(OverageCharge)
class OverageChargeAdmin(ReadOnlyAdmin):
    """Compute beyond the allowance that was billed to subscriptions. A *pending* charge got
    no answer from Paddle: look in Paddle whether the subscription was charged (a transaction
    of that amount at that time), then set it with one of the actions."""

    list_display = (
        "period",
        "account",
        "amount",
        "currency",
        "cpu_hours",
        "status",
        "detail",
        "created_at",
    )
    list_filter = ("status", "period")
    search_fields = ("user__username", "user__email", "subscription_id")
    actions = ["mark_charged", "mark_not_charged"]
    fields = (
        "account",
        "subscription_id",
        "period",
        "cpu_seconds",
        "price_per_cpu_hour",
        "amount",
        "currency",
        "status",
        "detail",
        "created_at",
        "updated_at",
    )

    @admin.display(description="CPU hours")
    def cpu_hours(self, charge: OverageCharge) -> str:
        return f"{charge.cpu_seconds / 3600:.2f}"

    def has_delete_permission(self, request, obj=None) -> bool:
        return False

    def _resolve(self, request, queryset, new_status: str, detail: str) -> None:
        count = queryset.filter(status="pending").update(status=new_status, detail=detail)
        level = messages.SUCCESS if count else messages.WARNING
        self.message_user(request, f"{count} pending charges were set to “{new_status}”.", level)

    @admin.action(description="Pending: Paddle did charge it")
    def mark_charged(self, request, queryset):
        self._resolve(request, queryset, "charged", "Confirmed in Paddle by the operator.")

    @admin.action(description="Pending: Paddle did not charge it (bill it again)")
    def mark_not_charged(self, request, queryset):
        self._resolve(request, queryset, "failed", "Not charged according to the operator.")
