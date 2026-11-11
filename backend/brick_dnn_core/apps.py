from django.apps import AppConfig
from django.db.models.signals import post_migrate


class BrickDnnCoreConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "brick_dnn_core"
    verbose_name = "Brick DNN"

    def ready(self) -> None:
        # Connects the signal handlers that delete training and data files with their objects.
        # billing: its check of the Paddle settings. raw_predictions: runs keep the frozen
        # preparation of a deleted dataset.
        from . import billing, data_service, raw_predictions, training_service  # noqa: F401
        from .site_admins import promote_after_migrate
        from .supabase import protect_after_migrate

        # On Supabase, hide the tables from its Data API after every migrate (see supabase.py).
        post_migrate.connect(protect_after_migrate, sender=self, dispatch_uid="brickdnn-supabase")
        # The operator's administrators (BRICKDNN_ADMIN_USERS, see site_admins.py).
        post_migrate.connect(promote_after_migrate, sender=self, dispatch_uid="brickdnn-admins")
