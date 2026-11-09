from django.contrib import admin

from .models import (
    ApiToken,
    DataPrep,
    Dataset,
    DataSource,
    OAuthClient,
    OAuthConnection,
    Project,
    ProjectMember,
    TrainingRun,
    UsageRecord,
    UserProfile,
)


class ProjectMemberInline(admin.TabularInline):
    model = ProjectMember
    fk_name = "project"
    extra = 0
    raw_id_fields = ("user", "invited_by")


@admin.register(Project)
class ProjectAdmin(admin.ModelAdmin):
    list_display = ("name", "owner", "version", "link_access", "updated_at")
    list_filter = ("owner", "link_access")
    search_fields = ("name", "description", "owner__username")
    readonly_fields = ("version", "updated_by", "share_token")
    inlines = [ProjectMemberInline]


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
        "api_token",
        "connection",
        "cpu_seconds",
        "cpu_limit_seconds",
        "estimated_cpu_seconds",
        "stop_reason",
        "size_bytes",
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
    list_filter = ("read_only", ("connection", admin.EmptyFieldListFilter))
    search_fields = ("name", "prefix", "owner__username")
    exclude = ("key_hash",)
    readonly_fields = (
        "owner",
        "name",
        "prefix",
        "read_only",
        "created_at",
        "last_used_at",
        "connection",
    )

    def has_add_permission(self, request) -> bool:
        return False


@admin.register(OAuthClient)
class OAuthClientAdmin(admin.ModelAdmin):
    """Apps that registered for OAuth sign-in. Deleting one disconnects it everywhere."""

    list_display = ("name", "client_id", "uri", "created_at")
    search_fields = ("name", "client_id", "uri")
    exclude = ("secret_hash",)
    readonly_fields = ("client_id", "name", "uri", "redirect_uris", "created_at")

    def has_add_permission(self, request) -> bool:
        return False


@admin.register(OAuthConnection)
class OAuthConnectionAdmin(admin.ModelAdmin):
    list_display = ("client", "owner", "read_only", "created_at", "last_used_at")
    search_fields = ("client__name", "owner__username")
    exclude = ("refresh_hash",)
    readonly_fields = (
        "client",
        "owner",
        "read_only",
        "refresh_expires_at",
        "created_at",
        "last_used_at",
    )

    def has_add_permission(self, request) -> bool:
        return False


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    """When accounts were last used and which terms they accepted (see retention.py), and
    the limits of each account (see budgets.py): empty uses the server's settings."""

    list_display = (
        "user",
        "last_active_at",
        "terms_accepted_at",
        "cpu_minutes_per_month",
        "cpu_minutes_per_run",
        "storage_mb",
        "budget_cpu_minutes",
    )
    search_fields = ("user__username",)
    raw_id_fields = ("user",)
    fieldsets = (
        (None, {"fields": ("user", "last_active_at")}),
        ("Terms", {"fields": ("terms_accepted_at", "terms_version", "privacy_version")}),
        (
            "Limits (empty: the server's settings, 0: unlimited)",
            {"fields": ("cpu_minutes_per_month", "cpu_minutes_per_run", "storage_mb")},
        ),
        ("Set by the user", {"fields": ("budget_cpu_minutes",)}),
    )

    def get_readonly_fields(self, request, obj=None):
        fixed = ("last_active_at", "terms_accepted_at", "terms_version", "privacy_version")
        # A new profile (e.g. for an account created in the admin) needs its user.
        return fixed if obj is None else ("user", *fixed)


@admin.register(UsageRecord)
class UsageRecordAdmin(admin.ModelAdmin):
    """The ledger of compute usage (see budgets.py). Written by the server only."""

    list_display = ("period", "user", "kind", "quantity", "unit", "source", "via", "run_name")
    list_filter = ("period", "kind", "source")
    search_fields = ("user__username", "via", "run_name")
    date_hierarchy = "created_at"

    def has_add_permission(self, request) -> bool:
        return False

    def has_change_permission(self, request, obj=None) -> bool:
        return False
