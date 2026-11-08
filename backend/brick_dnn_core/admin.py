from django.contrib import admin

from .models import ApiToken, DataPrep, Dataset, DataSource, Project, TrainingRun


@admin.register(Project)
class ProjectAdmin(admin.ModelAdmin):
    list_display = ("name", "owner", "updated_at")
    list_filter = ("owner",)
    search_fields = ("name", "description", "owner__username")


@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = ("name", "kind", "owner", "size_bytes", "created_at")
    list_filter = ("kind",)
    search_fields = ("name", "owner__username")
    readonly_fields = ("kind", "size_bytes", "info", "prepared_from", "created_at")


@admin.register(TrainingRun)
class TrainingRunAdmin(admin.ModelAdmin):
    list_display = ("name", "owner", "status", "epochs_done", "created_at", "finished_at")
    list_filter = ("status",)
    search_fields = ("name", "owner__username")
    readonly_fields = (
        "id",
        "graph",
        "dataset",
        "upload",
        "config",
        "tuning",
        "plan",
        "status",
        "message",
        "total_params",
        "epochs_done",
        "metrics",
        "best",
        "trials_done",
        "best_trial",
        "has_weights",
        "stop_requested_at",
        "pid",
        "created_at",
        "started_at",
        "finished_at",
    )


@admin.register(DataSource)
class DataSourceAdmin(admin.ModelAdmin):
    list_display = ("name", "owner", "filename", "n_rows", "n_columns", "size_bytes", "created_at")
    search_fields = ("name", "filename", "owner__username")
    readonly_fields = (
        "filename",
        "size_bytes",
        "n_rows",
        "n_columns",
        "columns",
        "import_options",
        "report",
        "created_at",
    )


@admin.register(DataPrep)
class DataPrepAdmin(admin.ModelAdmin):
    list_display = ("name", "owner", "source", "updated_at")
    search_fields = ("name", "owner__username", "source__name")
    raw_id_fields = ("source",)
    readonly_fields = ("created_at", "updated_at")


@admin.register(ApiToken)
class ApiTokenAdmin(admin.ModelAdmin):
    """Admins can see and revoke (delete) tokens, but not create them: only the hash is stored."""

    list_display = ("name", "owner", "prefix", "read_only", "created_at", "last_used_at")
    list_filter = ("read_only",)
    search_fields = ("name", "prefix", "owner__username")
    exclude = ("key_hash",)
    readonly_fields = ("owner", "name", "prefix", "read_only", "created_at", "last_used_at")

    def has_add_permission(self, request) -> bool:
        return False
