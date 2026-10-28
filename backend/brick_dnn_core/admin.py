from django.contrib import admin

from .models import Dataset, Project, TrainingRun


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
    readonly_fields = ("kind", "size_bytes", "info", "created_at")


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
        "plan",
        "status",
        "message",
        "total_params",
        "epochs_done",
        "metrics",
        "best",
        "has_weights",
        "stop_requested_at",
        "pid",
        "created_at",
        "started_at",
        "finished_at",
    )
