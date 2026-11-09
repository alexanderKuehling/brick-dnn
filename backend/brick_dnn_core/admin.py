from django.contrib import admin

from .models import (
    AgentAction,
    ApiToken,
    DataPrep,
    Dataset,
    DataSource,
    OAuthClient,
    OAuthConnection,
    Project,
    ProjectMember,
    TrainingRun,
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
    """When accounts were last used and which terms they accepted (see retention.py)."""

    list_display = ("user", "last_active_at", "terms_accepted_at", "terms_version")
    search_fields = ("user__username",)
    readonly_fields = (
        "user",
        "last_active_at",
        "terms_accepted_at",
        "terms_version",
        "privacy_version",
    )

    def has_add_permission(self, request) -> bool:
        return False


@admin.register(AgentAction)
class AgentActionAdmin(admin.ModelAdmin):
    """The activity log of AI agents (see agent_activity.py): to look at, not to change."""

    list_display = ("created_at", "owner", "agent_name", "tool", "outcome", "summary")
    list_filter = ("outcome", "source", "agent_kind")
    search_fields = ("owner__username", "agent_name", "tool", "summary")
    raw_id_fields = ("owner",)

    def has_add_permission(self, request) -> bool:
        return False

    def has_change_permission(self, request, obj=None) -> bool:
        return False
