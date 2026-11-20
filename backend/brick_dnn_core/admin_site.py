"""The Django admin site: its login respects two-factor authentication (two_factor.py).

The login page has a third field for the code. An account with two-factor authentication
gets in only with a valid code from its app (or a recovery code); each code works once, and
wrong codes count against the same limit as in the web app. With
``BRICKDNN_REQUIRE_2FA_FOR_STAFF`` an account without two-factor authentication is not let in
at all (it can still use the web app) and is told to set it up on its account page first;
sessions of such accounts that are open already lose access to the admin too.

The site also has the **business report** (business_report.py) at ``/admin/business-report/``,
linked from the start page: for staff accounts only, like every page of the admin.
"""

from __future__ import annotations

from django import forms
from django.conf import settings
from django.contrib import admin
from django.contrib.admin.forms import AdminAuthenticationForm
from django.core.exceptions import ValidationError
from django.template.response import TemplateResponse
from django.urls import path

from . import mail, two_factor


class AdminLoginForm(AdminAuthenticationForm):
    code = forms.CharField(
        label="Two-factor code",
        required=False,
        max_length=40,
        strip=True,
        help_text="From your authenticator app, or a recovery code. Only if two-factor "
        "authentication is on for your account.",
        widget=forms.TextInput(
            attrs={"autocomplete": "one-time-code", "inputmode": "numeric", "spellcheck": "false"}
        ),
    )

    def confirm_login_allowed(self, user) -> None:
        super().confirm_login_allowed(user)  # active staff account
        if two_factor.is_enabled(user):
            code = self.cleaned_data.get("code") or ""
            if not code:
                raise ValidationError(
                    "Enter the code from your authenticator app (or a recovery code).",
                    code="two_factor_required",
                )
            try:
                used = two_factor.verify(user, code)
            except two_factor.TooManyAttempts as exc:
                raise ValidationError(exc.message, code="two_factor_locked") from None
            if used is None:
                raise ValidationError(
                    "The two-factor code is wrong or was used already.",
                    code="two_factor_wrong",
                )
        elif settings.REQUIRE_2FA_FOR_STAFF:
            raise ValidationError(
                "The admin needs two-factor authentication. Turn it on on your account page "
                f"({mail.link('/account', self.request)}) first, then log in here again.",
                code="two_factor_missing",
            )


class AdminSite(admin.AdminSite):
    login_form = AdminLoginForm
    login_template = "brick_dnn_core/admin_login.html"
    index_template = "brick_dnn_core/admin_index.html"

    def has_permission(self, request) -> bool:
        if not super().has_permission(request):
            return False
        return not settings.REQUIRE_2FA_FOR_STAFF or two_factor.is_enabled(request.user)

    def get_urls(self):
        # admin_view: only for who may use the admin (else the login page).
        report = path(
            "business-report/", self.admin_view(self.business_report), name="business-report"
        )
        return [report, *super().get_urls()]

    def business_report(self, request):
        """The numbers of ``manage.py business_report`` as a page: ``?period=`` day, week
        or month."""
        from . import business_report

        period = request.GET.get("period")
        if period not in business_report.PERIODS:
            period = "month"
        data = business_report.report(period)
        table = business_report.columns(data)
        context = {
            **self.each_context(request),
            "title": "Business report",
            "report": data,
            "periods": list(business_report.PERIODS),
            "summary": business_report.summary(data),
            "headings": [heading for heading, _ in table],
            "rows": [[cells[index] for _, cells in table] for index in range(len(data["periods"]))][
                ::-1
            ],  # the newest period first
        }
        return TemplateResponse(request, "brick_dnn_core/admin_business_report.html", context)
