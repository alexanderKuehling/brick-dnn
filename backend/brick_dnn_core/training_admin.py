"""Django admin for the training queue (training_queue.py).

*Training workers* is the operator's view of the queue: above the list of workers it shows
which trainings wait (with account, plan, priority and waiting time) and which run, and how
the queue did in the last seven days: how long trainings waited for a free slot and how busy
the workers were. ``manage.py training_queue`` prints the same in a terminal.
"""

from __future__ import annotations

from typing import Any

from django.contrib import admin
from django.utils import timezone

from . import training_queue as queue
from .models import TrainingAttempt, TrainingWorker


def _presentable(data: dict[str, Any]) -> dict[str, Any]:
    """The overview with its durations as text, for the template."""
    text = queue.duration
    stats = data["stats"]
    share = stats["utilisation"]
    return {
        **data,
        "running": [{**run, "since": text(run["running_seconds"])} for run in data["running"]],
        "waiting": [
            {
                **run,
                "since": text(run["waiting_seconds"]),
                "estimate": text(run["estimated_wait_seconds"]),
            }
            for run in data["waiting"]
        ],
        "stats": {
            **stats,
            "median": text(stats["median_wait_seconds"]),
            "p95": text(stats["p95_wait_seconds"]),
            "longest": text(stats["max_wait_seconds"]),
            "busy": text(stats["busy_slot_seconds"]),
            "available": text(stats["available_slot_seconds"]),
            "utilisation_percent": f"{share * 100:.0f} %" if share is not None else "-",
            "by_plan": [
                {
                    **plan,
                    "median": text(plan["median_wait_seconds"]),
                    "p95": text(plan["p95_wait_seconds"]),
                }
                for plan in stats["by_plan"]
            ],
        },
    }


@admin.register(TrainingWorker)
class TrainingWorkerAdmin(admin.ModelAdmin):
    """The workers (one row per start) with the queue overview on top. Written by the
    workers themselves: nothing can be added or changed."""

    change_list_template = "admin/brick_dnn_core/trainingworker/change_list.html"
    list_display = (
        "name",
        "state",
        "heartbeat_at",
        "slots",
        "threads",
        "memory_mb",
        "current_jobs",
        "started_at",
        "stopped_at",
    )
    search_fields = ("name", "hostname")

    def has_add_permission(self, request) -> bool:
        return False

    def has_change_permission(self, request, obj=None) -> bool:
        return False

    @admin.display(description="State")
    def state(self, obj: TrainingWorker) -> str:
        if queue.is_alive(obj):
            return "online"
        return "stopped" if obj.stopped_at is not None else "no heartbeat"

    @admin.display(description="Trains")
    def current_jobs(self, obj: TrainingWorker) -> str:
        runs = obj.runs.filter(status__in=queue.RUNNING_STATUSES)
        return ", ".join(run.name for run in runs) or "-"

    def changelist_view(self, request, extra_context=None):
        context = {
            **(extra_context or {}),
            "title": "Training queue and workers",
            "queue": _presentable(queue.overview(timezone.now())),
        }
        return super().changelist_view(request, context)


@admin.register(TrainingAttempt)
class TrainingAttemptAdmin(admin.ModelAdmin):
    """Every start of a run by a worker, for the statistics of the queue (read-only)."""

    list_display = ("started_at", "run", "worker", "plan", "priority", "waited", "outcome")
    list_filter = ("outcome", "plan")
    raw_id_fields = ("run", "worker")

    def has_add_permission(self, request) -> bool:
        return False

    def has_change_permission(self, request, obj=None) -> bool:
        return False

    @admin.display(description="Waited")
    def waited(self, obj: TrainingAttempt) -> str:
        return queue.duration((obj.started_at - obj.queued_at).total_seconds())
