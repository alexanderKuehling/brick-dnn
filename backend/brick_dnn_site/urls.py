from django.contrib import admin
from django.urls import include, path, re_path

from brick_dnn_core import oauth

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", include("brick_dnn_core.urls")),
    # OAuth metadata for MCP clients (RFC 9728 and RFC 8414).
    re_path(
        r"^\.well-known/oauth-protected-resource(?:/(?P<path>api/mcp/?))?$",
        oauth.ProtectedResourceMetadata.as_view(),
    ),
    path(
        ".well-known/oauth-authorization-server",
        oauth.AuthorizationServerMetadata.as_view(),
    ),
]
