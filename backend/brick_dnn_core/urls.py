from django.urls import include, path
from rest_framework.routers import DefaultRouter
from rest_framework_simplejwt.views import TokenBlacklistView, TokenRefreshView

from . import data_views, mcp, oauth, views

router = DefaultRouter(trailing_slash=True)
router.include_root_view = False
router.register("projects", views.ProjectViewSet, basename="project")
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
    path("auth/token/refresh/", TokenRefreshView.as_view(), name="token-refresh"),
    path("auth/logout/", TokenBlacklistView.as_view(), name="logout"),
    path("auth/me/", views.me, name="me"),
    # MCP server for AI assistants (log in with an API token). Both spellings: MCP clients
    # POST to the URL as given, and Django cannot redirect a POST to add the slash.
    path("mcp/", mcp.McpView.as_view(), name="mcp"),
    path("mcp", mcp.McpView.as_view(), name="mcp-no-slash"),
    # OAuth sign-in for MCP clients (the consent page itself is /oauth/authorize in the UI).
    path("oauth/register/", oauth.RegisterClientView.as_view(), name="oauth-register"),
    path("oauth/authorize/", oauth.AuthorizeView.as_view(), name="oauth-authorize"),
    path("oauth/token/", oauth.TokenView.as_view(), name="oauth-token"),
    path("oauth/revoke/", oauth.RevokeView.as_view(), name="oauth-revoke"),
    path("training/", views.training_info, name="training"),
    path("training/check/", views.training_check, name="training-check"),
    path("data/", data_views.data_status, name="data"),
    path("data/steps/", data_views.data_steps, name="data-steps"),
    path("", include(router.urls)),
]
