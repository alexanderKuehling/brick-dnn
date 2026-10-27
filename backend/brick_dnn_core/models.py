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
