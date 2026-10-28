from django.apps import AppConfig


class BrickDnnCoreConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "brick_dnn_core"
    verbose_name = "Brick DNN"

    def ready(self) -> None:
        # Connects the signal handlers that delete training files with their objects.
        from . import training_service  # noqa: F401
