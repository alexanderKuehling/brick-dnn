"""The Django admin with Brick DNN's admin site (admin_site.py), which asks for the
two-factor code at login. Listed in INSTALLED_APPS instead of ``django.contrib.admin``; kept
apart from admin_site.py because it is imported before the models are ready."""

from django.contrib.admin.apps import AdminConfig as DjangoAdminConfig


class AdminConfig(DjangoAdminConfig):
    default_site = "brick_dnn_core.admin_site.AdminSite"
