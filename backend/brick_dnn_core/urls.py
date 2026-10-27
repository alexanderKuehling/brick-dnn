from django.urls import include, path
from rest_framework.routers import DefaultRouter
from rest_framework_simplejwt.views import TokenBlacklistView, TokenRefreshView

from . import views

router = DefaultRouter(trailing_slash=True)
router.include_root_view = False
router.register("projects", views.ProjectViewSet, basename="project")

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
    path("", include(router.urls)),
]
