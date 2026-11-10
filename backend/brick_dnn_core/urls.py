from django.urls import include, path, re_path
from rest_framework.routers import DefaultRouter
from rest_framework_simplejwt.views import TokenBlacklistView

from . import (
    accounts,
    agent_activity,
    billing_views,
    data_views,
    legal,
    mcp,
    oauth,
    projects,
    recovery,
    two_factor,
    upload_links,
    views,
)

router = DefaultRouter(trailing_slash=True)
router.include_root_view = False
router.register("projects", projects.ProjectViewSet, basename="project")
router.register("auth/api-tokens", views.ApiTokenViewSet, basename="api-token")
router.register("auth/connections", oauth.ConnectionViewSet, basename="oauth-connection")
router.register("training/datasets", views.DatasetViewSet, basename="training-dataset")
router.register("training/runs", views.TrainingRunViewSet, basename="training-run")
router.register("data/sources", data_views.DataSourceViewSet, basename="data-source")
router.register("data/preps", data_views.DataPrepViewSet, basename="data-prep")

urlpatterns = [
    path("", views.ApiRoot.as_view(), name="api-root"),
    path("health/", views.health, name="health"),
    path("layers/", views.layers, name="layers"),
    path("examples/", views.examples, name="examples"),
    path("analyze/", views.analyze_graph, name="analyze"),
    path("generate/", views.generate, name="generate"),
    path("auth/register/", views.RegisterView.as_view(), name="register"),
    path("auth/token/", views.LoginView.as_view(), name="token"),
    # The second step of a login with two-factor authentication (two_factor.py).
    path(
        "auth/token/two-factor/",
        two_factor.TwoFactorLoginView.as_view(),
        name="token-two-factor",
    ),
    path("auth/token/refresh/", accounts.RefreshView.as_view(), name="token-refresh"),
    # Forgotten passwords: a link by e-mail (recovery.py; only with e-mail set up).
    path("auth/password-reset/", recovery.PasswordResetView.as_view(), name="password-reset"),
    path(
        "auth/password-reset/check/",
        recovery.PasswordResetCheckView.as_view(),
        name="password-reset-check",
    ),
    path(
        "auth/password-reset/confirm/",
        recovery.PasswordResetConfirmView.as_view(),
        name="password-reset-confirm",
    ),
    path("auth/logout/", TokenBlacklistView.as_view(), name="logout"),
    path("auth/me/", views.me, name="me"),
    # The own account: e-mail, password, data export, deletion (website login only).
    path("auth/account/", accounts.AccountView.as_view(), name="account"),
    path("auth/account/password/", accounts.PasswordView.as_view(), name="account-password"),
    path("auth/account/export/", accounts.ExportView.as_view(), name="account-export"),
    path("auth/account/delete/", accounts.DeleteAccountView.as_view(), name="account-delete"),
    # Confirming the e-mail address with the link from the e-mail, sending the link again.
    path("auth/account/email/verify/", recovery.VerifyEmailView.as_view(), name="email-verify"),
    path(
        "auth/account/email/resend/",
        recovery.ResendVerificationView.as_view(),
        name="email-resend",
    ),
    # Two-factor authentication with an authenticator app (two_factor.py).
    path("auth/two-factor/", two_factor.TwoFactorView.as_view(), name="two-factor"),
    path(
        "auth/two-factor/setup/", two_factor.TwoFactorSetupView.as_view(), name="two-factor-setup"
    ),
    path(
        "auth/two-factor/enable/",
        two_factor.TwoFactorEnableView.as_view(),
        name="two-factor-enable",
    ),
    path(
        "auth/two-factor/disable/",
        two_factor.TwoFactorDisableView.as_view(),
        name="two-factor-disable",
    ),
    path(
        "auth/two-factor/recovery-codes/",
        two_factor.RecoveryCodesView.as_view(),
        name="two-factor-recovery-codes",
    ),
    # Usage and budgets this month (any login; changing the budget: website login only).
    path("account/usage/", accounts.UsageView.as_view(), name="account-usage"),
    # What AI agents did with the user's API tokens and connected apps (website login only).
    path("agent-activity/", agent_activity.AgentActivityView.as_view(), name="agent-activity"),
    path(
        "agent-activity/summary/",
        agent_activity.AgentActivitySummaryView.as_view(),
        name="agent-activity-summary",
    ),
    # Paid plans with Paddle (billing.py). Only the first one answers while billing is off.
    path("billing/", billing_views.billing_info, name="billing"),
    path("billing/subscription/", billing_views.SubscriptionView.as_view(), name="billing-state"),
    path("billing/checkout/", billing_views.CheckoutView.as_view(), name="billing-checkout"),
    path("billing/change-plan/", billing_views.ChangePlanView.as_view(), name="billing-change"),
    path("billing/cancel/", billing_views.CancelView.as_view(), name="billing-cancel"),
    path("billing/reactivate/", billing_views.ReactivateView.as_view(), name="billing-reactivate"),
    path("billing/portal/", billing_views.PortalView.as_view(), name="billing-portal"),
    path("billing/refresh/", billing_views.RefreshView.as_view(), name="billing-refresh"),
    # Paddle posts its notifications here, to the address as it was entered there (Django
    # cannot redirect a POST to add the slash).
    path(
        "billing/paddle/webhook/", billing_views.PaddleWebhookView.as_view(), name="paddle-webhook"
    ),
    path(
        "billing/paddle/webhook",
        billing_views.PaddleWebhookView.as_view(),
        name="paddle-webhook-no-slash",
    ),
    # Imprint, privacy policy, terms of use: Markdown files on the server (public).
    path("legal/", legal.legal_index, name="legal"),
    re_path(r"^legal/(?P<page>imprint|privacy|terms)/$", legal.legal_page, name="legal-page"),
    # Networks shared by link (see projects.py).
    path("shared/<str:token>/", projects.SharedProjectView.as_view(), name="shared-project"),
    path(
        "shared/<str:token>/duplicate/",
        projects.SharedProjectDuplicateView.as_view(),
        name="shared-project-duplicate",
    ),
    # MCP server for AI assistants (log in with an API token). Both spellings: MCP clients
    # POST to the URL as given, and Django cannot redirect a POST to add the slash.
    path("mcp/", mcp.McpView.as_view(), name="mcp"),
    path("mcp", mcp.McpView.as_view(), name="mcp-no-slash"),
    # Uploads with a signed one-time link (MCP tool create_upload_link), without logging in.
    path("uploads/", upload_links.UploadLinkView.as_view(), name="upload-link"),
    # OAuth sign-in for MCP clients (the consent page itself is /oauth/authorize in the UI).
    path("oauth/register/", oauth.RegisterClientView.as_view(), name="oauth-register"),
    path("oauth/authorize/", oauth.AuthorizeView.as_view(), name="oauth-authorize"),
    path("oauth/token/", oauth.TokenView.as_view(), name="oauth-token"),
    path("oauth/revoke/", oauth.RevokeView.as_view(), name="oauth-revoke"),
    path("training/", views.training_info, name="training"),
    # A preview image of a segmentation run: a file, so without the slash at the end.
    path(
        "training/runs/<uuid:pk>/previews/<str:name>",
        views.TrainingRunViewSet.as_view(
            {"get": "preview_file"},
            renderer_classes=[views.renderers.JSONRenderer, views.PngRenderer],
        ),
        name="training-run-preview-file",
    ),
    path("training/check/", views.training_check, name="training-check"),
    path("data/", data_views.data_status, name="data"),
    path("data/steps/", data_views.data_steps, name="data-steps"),
    path("", include(router.urls)),
]
