import uuid

from django.conf import settings
from django.db import models


def empty_graph() -> dict:
    return {"version": 1, "nodes": [], "edges": []}


class Project(models.Model):
    """A saved network: the graph JSON exactly as the editor produced it."""

    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="projects"
    )
    name = models.CharField(max_length=120)
    description = models.TextField(blank=True, max_length=2000)
    graph = models.JSONField(default=empty_graph)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]

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
    config = models.JSONField(default=dict)
    plan = models.JSONField(default=dict)
    status = models.CharField(max_length=10, choices=STATUSES, default="starting")
    message = models.TextField(blank=True)
    total_params = models.BigIntegerField(null=True, blank=True)
    epochs_done = models.PositiveIntegerField(default=0)
    metrics = models.JSONField(null=True, blank=True)
    best = models.JSONField(null=True, blank=True)
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
