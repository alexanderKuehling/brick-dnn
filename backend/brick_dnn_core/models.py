import uuid

from django.conf import settings
from django.core.validators import MinValueValidator
from django.db import models
from django.utils import timezone


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
    """An imported data file (Data tab). The files live in DATA_PREP_DIR/sources/<id>/:
    the uploaded or downloaded file (``original.csv``, ``original.xlsx`` ...) and the typed
    table (``table/``)."""

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
    # Opened, or its table read (preview, export, join): for deleting unused uploads
    # (BRICKDNN_DELETE_UNUSED_UPLOADS_DAYS, see retention.py).
    last_used_at = models.DateTimeField(default=timezone.now)

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
    # Opened or trained on (see retention.py).
    last_used_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ["-created_at", "-id"]

    def __str__(self) -> str:
        return f"{self.name} ({self.owner})"


class TrainingRun(models.Model):
    """One training of a network. Files (job, state, weights, log) live in TRAINING_DIR/runs/<id>/.

    The training process writes its progress to ``state.json``; the API copies the
    interesting parts into this row whenever an active run is looked at.

    With a training worker (BRICKDNN_TRAINING_WORKER, see training_queue.py) a new run is
    ``queued`` until a worker takes it; ``worker`` is the one that trains it.
    """

    STATUSES = [
        ("queued", "Queued"),
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
    # Opened, predicted with or its weights downloaded (see retention.py).
    last_used_at = models.DateTimeField(default=timezone.now)
    # Started with a personal API token or by an app connected with OAuth (both empty: in the
    # web app). Its usage counts against their budgets too (see budgets.py).
    api_token = models.ForeignKey(
        "ApiToken",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="training_runs",
    )
    connection = models.ForeignKey(
        "OAuthConnection",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="training_runs",
    )
    # CPU time counted in the usage records so far (the runner measures it, see budgets.py),
    # the CPU time after which the runner stops by itself (null: no limit) and the estimate
    # made before it started.
    cpu_seconds = models.FloatField(default=0)
    cpu_limit_seconds = models.FloatField(null=True, blank=True)
    estimated_cpu_seconds = models.FloatField(null=True, blank=True)
    # Why a stopped run stopped: "user", "budget" (a monthly budget was used up) or
    # "run_limit" (the most one run may use); empty otherwise.
    stop_reason = models.CharField(max_length=20, blank=True)
    # Bytes of its files (weights, log) once it has finished, for the storage quota.
    size_bytes = models.BigIntegerField(null=True, blank=True)
    # The queue (training_queue.py): since when the run waits for a worker (set again when
    # it is put back into the queue), the worker that took it, and how often it was put back
    # after a worker restarted or died. All empty when the web server trains itself.
    queued_at = models.DateTimeField(null=True, blank=True)
    worker = models.ForeignKey(
        "TrainingWorker", on_delete=models.SET_NULL, null=True, blank=True, related_name="runs"
    )
    # (The default is also in the database, so that a version from before the queue, which
    # does not know the column, can still create runs after it was deployed again.)
    restarts = models.PositiveSmallIntegerField(default=0, db_default=0)

    class Meta:
        ordering = ["-created_at"]
        indexes = [models.Index(fields=["status", "created_at"], name="run_status_created")]

    def __str__(self) -> str:
        return f"{self.name} ({self.owner}, {self.status})"


class TrainingWorker(models.Model):
    """A training worker process (``manage.py training_worker``): one row per start. It
    takes queued runs, trains them and says every few seconds that it is alive
    (``heartbeat_at``); a worker that stopped saying so is considered dead and its runs are
    put back into the queue (see training_queue.py)."""

    # BRICKDNN_WORKER_NAME, by default the host name: a worker that starts again under the
    # same name takes over what its predecessor left behind.
    name = models.CharField(max_length=120)
    hostname = models.CharField(max_length=255, blank=True)
    pid = models.IntegerField(null=True, blank=True)
    # Trainings it runs at the same time, and the CPU threads of each.
    slots = models.PositiveSmallIntegerField(default=1)
    threads = models.PositiveSmallIntegerField(default=1)
    # The memory limit of its container in megabytes, if it has one (for the operator).
    memory_mb = models.PositiveIntegerField(null=True, blank=True)
    started_at = models.DateTimeField(default=timezone.now)
    heartbeat_at = models.DateTimeField(default=timezone.now)
    # Set when it ended in an orderly way, or when it was found dead.
    stopped_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-started_at", "-id"]
        indexes = [models.Index(fields=["stopped_at", "heartbeat_at"], name="worker_alive")]

    def __str__(self) -> str:
        return f"{self.name} (started {self.started_at:%Y-%m-%d %H:%M})"


class TrainingAttempt(models.Model):
    """One start of a run by a worker: how long the run waited for it and how long the
    worker was busy with it. The numbers of the queue overview (waiting times, utilisation)
    come from these rows, which outlive the run; they name no user. Kept for 90 days."""

    OUTCOMES = [
        ("completed", "Completed"),
        ("stopped", "Stopped"),
        ("failed", "Failed"),
        ("requeued", "Put back into the queue (its worker restarted)"),
        ("lost", "Put back into the queue (its worker died)"),
        ("deleted", "Deleted while it ran"),
    ]

    run = models.ForeignKey(
        TrainingRun, on_delete=models.SET_NULL, null=True, blank=True, related_name="attempts"
    )
    worker = models.ForeignKey(
        TrainingWorker, on_delete=models.SET_NULL, null=True, blank=True, related_name="attempts"
    )
    # The plan of the account and its priority in the queue when the run started.
    plan = models.CharField(max_length=64, blank=True)
    priority = models.IntegerField(default=0)
    queued_at = models.DateTimeField()
    started_at = models.DateTimeField()
    ended_at = models.DateTimeField(null=True, blank=True)
    outcome = models.CharField(max_length=10, choices=OUTCOMES, blank=True)

    class Meta:
        ordering = ["-started_at", "-id"]
        indexes = [models.Index(fields=["started_at"], name="attempt_started")]

    def __str__(self) -> str:
        return f"{self.run_id or 'deleted run'} on {self.worker_id}: {self.outcome or 'running'}"


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
    # Monthly compute budget of this token in CPU minutes, set by its owner (null: none,
    # only the account's limits apply). Not used for the access tokens of apps.
    budget_cpu_minutes = models.FloatField(null=True, blank=True)

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
    # Monthly compute budget of this app in CPU minutes, set by the user (null: none).
    budget_cpu_minutes = models.FloatField(null=True, blank=True)

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


class UserProfile(models.Model):
    """What Brick DNN keeps about an account besides Django's user: when it was last used
    (for deleting inactive accounts, see retention.py) and which terms it accepted."""

    user = models.OneToOneField(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="profile"
    )
    # Log-ins, token refreshes and API token use; updated at most about once an hour.
    last_active_at = models.DateTimeField(default=timezone.now)
    # The terms of use and the privacy policy shown at sign-up (see legal.py): a hash of
    # each file, empty if the page was not set up.
    terms_accepted_at = models.DateTimeField(null=True, blank=True)
    terms_version = models.CharField(max_length=64, blank=True)
    privacy_version = models.CharField(max_length=64, blank=True)
    # The plan the limits come from (a key of BRICKDNN_PLANS; empty: the default plan) and
    # whether the account has no limits at all (empty: only staff and superusers). The
    # operator sets the plan (Django admin, `manage.py set_plan`), or a paid subscription
    # does (billing.py).
    plan = models.CharField(
        max_length=40,
        blank=True,
        help_text="A plan of BRICKDNN_PLANS, e.g. free or plus. Empty: the default plan.",
    )
    unlimited = models.BooleanField(
        null=True,
        blank=True,
        help_text="Yes: no limits of the plan apply (usage is still measured). Empty: only staff "
        "members and superusers are unlimited.",
    )
    # Paid plans: usage beyond the monthly allowance is allowed (and billed later) when the
    # user switched it on, up to their own cap in CPU hours (empty: no cap).
    overage_enabled = models.BooleanField(default=False)
    overage_cap_cpu_hours = models.FloatField(
        null=True, blank=True, validators=[MinValueValidator(0)]
    )
    # Limits of this account set by the operator (see budgets.py): empty = those of its plan,
    # 0 = unlimited.
    cpu_minutes_per_month = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0)],
        help_text="Monthly compute allowance in CPU minutes. Empty: the server's setting "
        "(BRICKDNN_BUDGET_CPU_MINUTES_PER_MONTH); 0: unlimited.",
    )
    cpu_minutes_per_run = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0)],
        help_text="The most CPU minutes one training may use. Empty: the server's setting "
        "(BRICKDNN_BUDGET_CPU_MINUTES_PER_RUN); 0: unlimited.",
    )
    storage_mb = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0)],
        help_text="Storage quota in MB. Empty: the server's setting (BRICKDNN_BUDGET_STORAGE_MB); "
        "0: unlimited.",
    )
    # The monthly budget the user set for their own account (account page), in CPU minutes;
    # empty: none. It can only lower the allowance.
    budget_cpu_minutes = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0)],
        help_text="The user's own monthly budget in CPU minutes (set on the account page).",
    )
    # E-mail (recovery.py): when and which address was verified (a link sent to it was
    # opened by the logged-in account). It is the account's address only while they match.
    email_verified_at = models.DateTimeField(null=True, blank=True)
    email_verified_for = models.EmailField(max_length=254, blank=True)
    # The address was entered before addresses were verified and has not changed since:
    # password reset links may still go to it (see recovery.trusted_email).
    email_legacy = models.BooleanField(
        default=False,
        help_text="The e-mail address dates from before addresses were verified; password "
        "reset links may go to it while it is unverified.",
    )
    # The last warning that the inactive account will be deleted (retention.py).
    deletion_warned_at = models.DateTimeField(null=True, blank=True)

    def __str__(self) -> str:
        return f"Profile of {self.user}"


class TwoFactorAuth(models.Model):
    """Two-factor authentication of an account with an authenticator app (two_factor.py).

    The TOTP secret is stored encrypted with a key derived from the server's secret key and is
    never shown again after the setup. The recovery codes are stored as SHA-256 hashes; each
    works once. Until ``confirmed_at`` is set (the user typed a code), it is not switched on.
    """

    user = models.OneToOneField(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="two_factor"
    )
    secret = models.TextField()
    confirmed_at = models.DateTimeField(null=True, blank=True)
    # The time step (30 s) of the last accepted code: a code works only once.
    last_used_step = models.BigIntegerField(default=0)
    recovery_codes = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "two-factor authentication"
        verbose_name_plural = "two-factor authentication"

    def __str__(self) -> str:
        state = "on" if self.confirmed_at else "being set up"
        return f"Two-factor authentication of {self.user} ({state})"


class UsageRecord(models.Model):
    """Compute an account used: the ledger behind budgets (and billing; see budgets.py).

    One row per training run and calendar month, whose quantity grows while the run is
    active, and one per prediction. ``quantity`` is in ``unit`` (CPU seconds). The API token
    or connected app that caused it is kept for the budgets of agents; its name is copied
    (``via``), so that the row still says where it came from once the token is deleted.
    """

    KINDS = [("training", "Training"), ("prediction", "Prediction")]
    SOURCES = [("web", "Web app"), ("token", "API token"), ("app", "Connected app")]
    UNITS = [("cpu_second", "CPU seconds")]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="usage_records"
    )
    source = models.CharField(max_length=10, choices=SOURCES, default="web")
    api_token = models.ForeignKey(
        ApiToken, on_delete=models.SET_NULL, null=True, blank=True, related_name="usage_records"
    )
    connection = models.ForeignKey(
        OAuthConnection,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="usage_records",
    )
    # The name of the token or app at the time (empty for the web app).
    via = models.CharField(max_length=200, blank=True)
    kind = models.CharField(max_length=20, choices=KINDS)
    quantity = models.FloatField(default=0)
    # The part of ``quantity`` beyond the monthly allowance of the plan, used with overage
    # switched on (billed at the plan's overage price).
    overage = models.FloatField(default=0)
    unit = models.CharField(max_length=20, choices=UNITS, default="cpu_second")
    run = models.ForeignKey(
        TrainingRun, on_delete=models.SET_NULL, null=True, blank=True, related_name="usage_records"
    )
    run_name = models.CharField(max_length=120, blank=True)
    # The first day of the calendar month (UTC) the usage belongs to.
    period = models.DateField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at", "-id"]
        indexes = [
            models.Index(fields=["user", "period"], name="usage_user_period"),
            models.Index(fields=["period"], name="usage_period"),
        ]

    def __str__(self) -> str:
        return f"{self.kind} of {self.user}: {self.quantity:.1f} {self.unit} ({self.period})"


class AgentAction(models.Model):
    """One thing an AI agent or a script did in a user's account: an MCP tool call, or a REST
    request made with an API token. The activity log (see agent_activity.py) shows them on the
    page "Agent activity"; ordinary use of the website is not logged.

    Arguments are summarised and redacted before they are stored: no file contents, no
    prediction inputs, no tokens or passwords. Entries are kept for BRICKDNN_AGENT_LOG_DAYS
    days and deleted with the account. They outlive the token or app that made them: the
    token or app is named here instead of being a foreign key.
    """

    SOURCES = [("mcp", "MCP tool"), ("api", "REST API")]
    AGENT_KINDS = [("token", "API token"), ("app", "Connected app")]
    OUTCOMES = [("ok", "Done"), ("error", "Failed")]

    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="agent_actions"
    )
    # When the action finished (it started duration_ms before).
    created_at = models.DateTimeField(default=timezone.now)
    source = models.CharField(max_length=3, choices=SOURCES)
    # Who: a personal API token (ApiToken.pk) or an app connected with OAuth
    # (OAuthConnection.pk), by number only, with its name at the time.
    agent_kind = models.CharField(max_length=5, choices=AGENT_KINDS)
    agent_id = models.PositiveBigIntegerField()
    agent_name = models.CharField(max_length=200)
    # The start of the token used ("bdnn_" and 8 characters); apps get a new one every hour.
    token_prefix = models.CharField(max_length=20, blank=True)
    user_agent = models.CharField(max_length=200, blank=True)
    # What: the MCP tool, or the name of the REST route (e.g. "training-run-stop") and method.
    tool = models.CharField(max_length=64)
    method = models.CharField(max_length=7, blank=True)
    # Changes something (saves, uploads, starts, stops, deletes) rather than only reading.
    writes = models.BooleanField(default=False)
    outcome = models.CharField(max_length=10, choices=OUTCOMES, default="ok")
    # What went wrong (outcome "error"), as the agent was told.
    error = models.CharField(max_length=500, blank=True)
    # REST requests: the HTTP status of the answer.
    status_code = models.PositiveSmallIntegerField(null=True, blank=True)
    duration_ms = models.PositiveIntegerField(default=0)
    # A sentence for people, e.g. 'Started training “GP 1” on “houses.csv”'.
    summary = models.CharField(max_length=300)
    # The arguments, summarised: {"graph": "5 layers, 4 connections", "csv": "48 KB, …"}.
    arguments = models.JSONField(default=dict)
    # What it touched: [{"kind": "project" | "run" | "dataset" | "data_source" | "data_prep",
    # "id", "name"?}, ...]
    targets = models.JSONField(default=list)
    # The network it was about, if any (for filtering). Not enforced by the database, so that
    # the entry keeps the number when the network is deleted.
    project = models.ForeignKey(
        Project,
        on_delete=models.DO_NOTHING,
        db_constraint=False,
        null=True,
        blank=True,
        related_name="+",
    )

    class Meta:
        ordering = ["-id"]
        indexes = [
            models.Index(fields=["owner", "-id"], name="agent_action_owner_id"),
            models.Index(fields=["created_at"], name="agent_action_created"),
        ]

    def __str__(self) -> str:
        return f"{self.agent_name} ({self.owner}): {self.summary}"


class BillingCustomer(models.Model):
    """An account at Paddle (billing.py): its customer there and, once it bought a plan, its
    subscription as Paddle last reported it. There is at most one subscription per account.

    The plan the account really has is ``UserProfile.plan``; billing sets it from ``status``
    and ``plan`` here. Paddle keeps the payment data, addresses and invoices.
    """

    user = models.OneToOneField(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="billing"
    )
    # "sandbox" or "live": ids of one are unknown in the other.
    environment = models.CharField(max_length=10)
    customer_id = models.CharField(max_length=64, unique=True)
    subscription_id = models.CharField(max_length=64, unique=True, null=True, blank=True)
    # Paddle's status: active, trialing, past_due, paused, canceled; empty: no subscription.
    status = models.CharField(max_length=20, blank=True)
    # The plan the subscription pays for (a key of BRICKDNN_PLANS) and its Paddle price.
    plan = models.CharField(max_length=40, blank=True)
    price_id = models.CharField(max_length=64, blank=True)
    # The plan billing last gave the account for it (empty: the default plan). Billing only
    # changes the account's plan when this changes, so a plan the operator sets by hand
    # afterwards stays.
    granted_plan = models.CharField(max_length=40, blank=True)
    # The period that is paid for; it renews at its end unless a cancellation is scheduled.
    current_period_start = models.DateTimeField(null=True, blank=True)
    current_period_end = models.DateTimeField(null=True, blank=True)
    next_billed_at = models.DateTimeField(null=True, blank=True)
    # Cancelled to the end of the period: when the subscription ends.
    cancel_at = models.DateTimeField(null=True, blank=True)
    canceled_at = models.DateTimeField(null=True, blank=True)
    # A payment failed and Paddle is retrying: the plan is kept for a grace period.
    past_due_since = models.DateTimeField(null=True, blank=True)
    payment_failed_at = models.DateTimeField(null=True, blank=True)
    # When Paddle said so: the newest event (or API answer) that was applied. Older events
    # that arrive later are ignored.
    last_event_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at", "-id"]

    def __str__(self) -> str:
        return f"{self.user}: {self.plan or 'no plan'} ({self.status or 'no subscription'})"


class BillingEvent(models.Model):
    """A notification from Paddle that was received (billing.py): every event is processed
    once. Only what is needed to follow what happened is kept, not the event's content."""

    OUTCOMES = [
        ("applied", "Applied"),
        ("stale", "Older than what is known"),
        ("ignored", "Not needed"),
        ("unknown", "No account for it"),
        ("rejected", "Does not belong to the account"),
    ]

    event_id = models.CharField(max_length=64, unique=True)
    event_type = models.CharField(max_length=60)
    occurred_at = models.DateTimeField()
    received_at = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="billing_events",
    )
    subscription_id = models.CharField(max_length=64, blank=True)
    outcome = models.CharField(max_length=10, choices=OUTCOMES)
    note = models.CharField(max_length=300, blank=True)

    class Meta:
        ordering = ["-occurred_at", "-id"]

    def __str__(self) -> str:
        return f"{self.event_type} ({self.event_id}): {self.outcome}"


class OverageCharge(models.Model):
    """Compute beyond the allowance of a month, billed to the account's subscription
    (billing.py). The row is saved as ``pending`` before Paddle is asked and updated with the
    answer, so that a crash in between cannot charge twice: a pending row is never charged
    again by itself (the operator decides in the Django admin)."""

    STATUSES = [
        ("pending", "Pending: Paddle was asked, the answer is unknown"),
        ("charged", "Charged"),
        ("failed", "Refused by Paddle"),
        ("skipped", "Not charged"),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="overage_charges"
    )
    subscription_id = models.CharField(max_length=64, blank=True)
    # The first day of the calendar month (UTC) the usage belongs to, as in UsageRecord.
    period = models.DateField()
    # What this row bills: CPU seconds beyond the allowance, their price and the amount.
    cpu_seconds = models.FloatField()
    price_per_cpu_hour = models.DecimalField(max_digits=10, decimal_places=4)
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    currency = models.CharField(max_length=3)
    status = models.CharField(max_length=10, choices=STATUSES, default="pending")
    detail = models.CharField(max_length=300, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at", "-id"]
        constraints = [
            # At most one charge per account and month whose outcome is unknown.
            models.UniqueConstraint(
                fields=["user", "period"],
                condition=models.Q(status="pending"),
                name="one_pending_overage_charge",
            )
        ]

    def __str__(self) -> str:
        return f"{self.user} {self.period:%Y-%m}: {self.amount} {self.currency} ({self.status})"
