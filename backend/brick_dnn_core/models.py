import uuid

from django.conf import settings
from django.db import models


def empty_graph() -> dict:
    return {"version": 1, "nodes": [], "edges": []}


class Project(models.Model):
    """A saved network: the graph JSON exactly as the editor produced it.

    The owner can share it with other users (:class:`ProjectMember`) and by link.
    """

    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="projects"
    )
    name = models.CharField(max_length=120)
    description = models.TextField(blank=True, max_length=2000)
    graph = models.JSONField(default=empty_graph)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    # Goes up with every change: a save based on an older version is a conflict.
    version = models.PositiveIntegerField(default=1)
    updated_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="+",
    )
    # "view": anyone with the link /share/<share_token> can look at the network.
    link_access = models.CharField(
        max_length=10,
        choices=[("none", "No link"), ("view", "Anyone with the link can view")],
        default="none",
    )
    share_token = models.CharField(max_length=64, null=True, blank=True, unique=True)

    class Meta:
        ordering = ["-updated_at"]

    def __str__(self) -> str:
        return f"{self.name} ({self.owner})"


class ProjectMember(models.Model):
    """A user the owner shared a project with. The owner is not a member."""

    ROLES = [("editor", "Can edit"), ("viewer", "Can view")]

    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name="members")
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="project_memberships"
    )
    role = models.CharField(max_length=10, choices=ROLES)
    invited_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="+",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at", "id"]
        constraints = [
            models.UniqueConstraint(fields=["project", "user"], name="unique_project_member")
        ]

    def __str__(self) -> str:
        return f"{self.user} ({self.role}) in {self.project.name}"


def default_prep_settings() -> dict:
    """Target and split settings of a new preparation (see brickdnn.data.pipeline)."""
    return {
        "target": None,
        "task": "auto",
        "split": {"val": 0.15, "test": 0.15, "seed": 42, "stratify": True},
    }


class DataSource(models.Model):
    """An imported CSV file (Data tab). The files live in DATA_PREP_DIR/sources/<id>/:
    the uploaded file (``original.csv``) and the typed table (``table/``)."""

    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="data_sources"
    )
    name = models.CharField(max_length=120)
    filename = models.CharField(max_length=255, blank=True)
    size_bytes = models.BigIntegerField(default=0)
    n_rows = models.BigIntegerField(default=0)
    n_columns = models.PositiveIntegerField(default=0)
    # [{"name", "kind", "missing"}, ...] in table order
    columns = models.JSONField(default=list)
    # brickdnn.data.io.ImportOptions as requested ("auto" = detect) and ImportReport (detected)
    import_options = models.JSONField(default=dict)
    report = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at", "-id"]

    def __str__(self) -> str:
        return f"{self.name} ({self.owner})"


class DataPrep(models.Model):
    """A data preparation: a main source, a list of steps and the target/split settings."""

    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="data_preps"
    )
    name = models.CharField(max_length=120)
    # RESTRICT (not PROTECT): a source cannot be deleted while a preparation uses it, but
    # deleting the user deletes both.
    source = models.ForeignKey(DataSource, on_delete=models.RESTRICT, related_name="preps")
    # [{"id", "type", "enabled", "params"}, ...]
    steps = models.JSONField(default=list)
    # {"target", "task", "split": {"val", "test", "seed", "stratify"}}
    settings = models.JSONField(default=default_prep_settings)
    # Projects whose members may use it (editors of those projects may change it too).
    projects = models.ManyToManyField("Project", blank=True, related_name="data_preps")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at", "-id"]

    def __str__(self) -> str:
        return f"{self.name} ({self.owner})"


class Dataset(models.Model):
    """An uploaded training dataset. The files live in TRAINING_DIR/datasets/<id>/."""

    KINDS = [("csv", "CSV file"), ("images", "Image folders")]

    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="datasets"
    )
    name = models.CharField(max_length=120)
    kind = models.CharField(max_length=10, choices=KINDS)
    size_bytes = models.BigIntegerField(default=0)
    # DatasetInfo.to_dict(): columns, classes, number of samples, ...
    info = models.JSONField(default=dict)
    # Created from a data preparation (Data tab) instead of an upload.
    prepared_from = models.ForeignKey(
        DataPrep, on_delete=models.SET_NULL, null=True, blank=True, related_name="datasets"
    )
    # Projects whose members may train on it.
    projects = models.ManyToManyField("Project", blank=True, related_name="datasets")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at", "-id"]

    def __str__(self) -> str:
        return f"{self.name} ({self.owner})"


class TrainingRun(models.Model):
    """One training of a network. Files (job, state, weights, log) live in TRAINING_DIR/runs/<id>/.

    The training process writes its progress to ``state.json``; the API copies the
    interesting parts into this row whenever an active run is looked at.
    """

    STATUSES = [
        ("starting", "Starting"),
        ("preparing", "Preparing"),
        ("running", "Running"),
        ("completed", "Completed"),
        ("stopped", "Stopped"),
        ("failed", "Failed"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="training_runs"
    )
    name = models.CharField(max_length=120)
    graph = models.JSONField(default=empty_graph)
    # {"kind": "random"|"builtin"|"upload", "id": ..., "name": ...} as chosen by the user
    dataset = models.JSONField(default=dict)
    upload = models.ForeignKey(
        Dataset, on_delete=models.SET_NULL, null=True, blank=True, related_name="runs"
    )
    # Started from a shared project: its members see the run.
    project = models.ForeignKey(
        "Project", on_delete=models.SET_NULL, null=True, blank=True, related_name="runs"
    )
    config = models.JSONField(default=dict)
    # Hyperparameter search settings (TuningConfig.to_dict()); null for a normal training.
    tuning = models.JSONField(null=True, blank=True)
    plan = models.JSONField(default=dict)
    status = models.CharField(max_length=10, choices=STATUSES, default="starting")
    message = models.TextField(blank=True)
    total_params = models.BigIntegerField(null=True, blank=True)
    epochs_done = models.PositiveIntegerField(default=0)
    metrics = models.JSONField(null=True, blank=True)
    best = models.JSONField(null=True, blank=True)
    # Hyperparameter search: finished trials and {"trial", "params", "val_loss", ...} of
    # the best one so far.
    trials_done = models.PositiveIntegerField(default=0)
    best_trial = models.JSONField(null=True, blank=True)
    has_weights = models.BooleanField(default=False)
    stop_requested_at = models.DateTimeField(null=True, blank=True)
    pid = models.IntegerField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.name} ({self.owner}, {self.status})"


class ApiToken(models.Model):
    """A personal API token: lets scripts and AI assistants (MCP) act as its owner.

    Only a SHA-256 hash of the token is stored; the token itself is shown once, when it is
    created (see brick_dnn_core.api_tokens).
    """

    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="api_tokens"
    )
    name = models.CharField(max_length=100)
    # The start of the token ("bdnn_" and 8 characters), so users can recognise it.
    prefix = models.CharField(max_length=20)
    key_hash = models.CharField(max_length=64, unique=True)
    # Read-only tokens can look at everything but change nothing.
    read_only = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    last_used_at = models.DateTimeField(null=True, blank=True)
    expires_at = models.DateTimeField(null=True, blank=True)
    # Set for the short-lived access tokens of an app connected with OAuth (see oauth.py);
    # empty for the tokens users create themselves.
    connection = models.ForeignKey(
        "OAuthConnection",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="access_tokens",
    )

    class Meta:
        ordering = ["-created_at", "-id"]

    def __str__(self) -> str:
        return f"{self.name} ({self.owner}, {self.prefix}…)"


class OAuthClient(models.Model):
    """An app that registered itself for OAuth sign-in (dynamic client registration)."""

    client_id = models.CharField(max_length=64, unique=True)
    # Empty for public clients, which prove themselves with PKCE only.
    secret_hash = models.CharField(max_length=64, blank=True)
    name = models.CharField(max_length=200)
    uri = models.URLField(max_length=500, blank=True)
    redirect_uris = models.JSONField(default=list)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.name} ({self.client_id})"


class OAuthCode(models.Model):
    """An authorization code: a user allowed a client access; valid for a few minutes."""

    code_hash = models.CharField(max_length=64, unique=True)
    client = models.ForeignKey(OAuthClient, on_delete=models.CASCADE, related_name="codes")
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    redirect_uri = models.CharField(max_length=2000)
    code_challenge = models.CharField(max_length=128)
    read_only = models.BooleanField(default=False)
    expires_at = models.DateTimeField()


class OAuthConnection(models.Model):
    """A user's permission for an app (an OAuth grant), with its current refresh token.

    Deleting it disconnects the app: its access tokens go with it.
    """

    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="oauth_connections"
    )
    client = models.ForeignKey(OAuthClient, on_delete=models.CASCADE, related_name="connections")
    read_only = models.BooleanField(default=False)
    refresh_hash = models.CharField(max_length=64, unique=True)
    refresh_expires_at = models.DateTimeField()
    created_at = models.DateTimeField(auto_now_add=True)
    last_used_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at", "-id"]

    def __str__(self) -> str:
        return f"{self.client.name} ({self.owner})"


class ProjectVersion(models.Model):
    """A saved state of a project for its history.

    Changes by the same person within a few minutes update the latest automatic entry
    instead of adding one each time (see history.py). Named versions are kept for good.
    """

    KINDS = [("auto", "Automatic"), ("named", "Named"), ("restore", "Restored")]

    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name="history")
    version = models.PositiveIntegerField()
    kind = models.CharField(max_length=10, choices=KINDS, default="auto")
    label = models.CharField(max_length=120, blank=True)
    name = models.CharField(max_length=120)
    description = models.TextField(blank=True, max_length=2000)
    graph = models.JSONField()
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="+",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at", "-id"]

    def __str__(self) -> str:
        return f"{self.project.name} v{self.version} ({self.kind})"


class Comment(models.Model):
    """A comment on a project, optionally on one of its layers (``node_id``).

    Replies point to the first comment of the thread (one level).
    """

    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name="comments")
    author = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="comments"
    )
    parent = models.ForeignKey(
        "self", on_delete=models.CASCADE, null=True, blank=True, related_name="replies"
    )
    node_id = models.CharField(max_length=100, blank=True)
    text = models.TextField(max_length=5000)
    resolved = models.BooleanField(default=False)
    resolved_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="+",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["created_at", "id"]

    def __str__(self) -> str:
        return f"{self.author} on {self.project.name}: {self.text[:40]}"
