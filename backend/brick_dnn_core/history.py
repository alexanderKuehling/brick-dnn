"""Version history and comments of projects.

History: after every change, :func:`record` keeps a copy of the new state. Changes by the
same person within :data:`MERGE_WINDOW` update the latest automatic entry instead of adding
one each time, so live editing (many small changes) gives one entry per person and session.
Users can also name the current state; named versions are never removed automatically.
Restoring a version is a normal change, so it can be undone by restoring another one.

Comments: everyone with access to a project (viewers too) can comment on it or on one of
its layers and reply; authors edit and delete their comments, the owner can delete any,
and authors, editors and the owner resolve threads. Connected live editors get every
change of the comments and of the history right away.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from django.db import transaction
from django.utils import timezone
from rest_framework import serializers

from . import live
from .models import Comment, Project, ProjectVersion

MERGE_WINDOW = timedelta(minutes=10)
MAX_AUTO_VERSIONS = 200
MAX_COMMENTS = 2000


# --------------------------------------------------------------------------- #
# History
# --------------------------------------------------------------------------- #


def record(project: Project, user, kind: str = "auto", label: str = "") -> ProjectVersion:
    """Keep the current state of ``project`` in its history (call after a change)."""
    latest = project.history.order_by("-updated_at", "-id").first()
    now = timezone.now()
    if (
        kind == "auto"
        and latest is not None
        and latest.kind == "auto"
        and latest.created_by_id == getattr(user, "pk", None)
        and now - latest.updated_at < MERGE_WINDOW
    ):
        latest.version = project.version
        latest.name = project.name
        latest.description = project.description
        latest.graph = project.graph
        latest.save()
        entry = latest
    else:
        entry = ProjectVersion.objects.create(
            project=project,
            version=project.version,
            kind=kind,
            label=label,
            name=project.name,
            description=project.description,
            graph=project.graph,
            created_by=user if getattr(user, "pk", None) else None,
        )
        _prune(project)
    transaction.on_commit(lambda: live.notify_history(project.pk))
    return entry


def _prune(project: Project) -> None:
    autos = project.history.filter(kind="auto").order_by("-updated_at", "-id")
    old = list(autos.values_list("pk", flat=True)[MAX_AUTO_VERSIONS:])
    if old:
        ProjectVersion.objects.filter(pk__in=old).delete()


class VersionSerializer(serializers.ModelSerializer):
    created_by = serializers.SerializerMethodField()
    num_layers = serializers.SerializerMethodField()

    class Meta:
        model = ProjectVersion
        fields = [
            "id",
            "version",
            "kind",
            "label",
            "name",
            "num_layers",
            "created_by",
            "created_at",
            "updated_at",
        ]
        read_only_fields = fields

    def get_created_by(self, obj: ProjectVersion) -> str | None:
        return obj.created_by.get_username() if obj.created_by else None

    def get_num_layers(self, obj: ProjectVersion) -> int:
        nodes = obj.graph.get("nodes", []) if isinstance(obj.graph, dict) else []
        return len(nodes)


def version_data(entry: ProjectVersion, with_graph: bool = False) -> dict[str, Any]:
    data = VersionSerializer(entry).data
    if with_graph:
        data["description"] = entry.description
        data["graph"] = entry.graph
    return data


def clean_label(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise serializers.ValidationError({"label": ["Give the version a name."]})
    return value.strip()[:120]


# --------------------------------------------------------------------------- #
# Comments
# --------------------------------------------------------------------------- #


class CommentSerializer(serializers.ModelSerializer):
    author = serializers.SerializerMethodField()
    resolved_by = serializers.SerializerMethodField()

    class Meta:
        model = Comment
        fields = [
            "id",
            "parent",
            "node_id",
            "text",
            "author",
            "resolved",
            "resolved_by",
            "created_at",
            "updated_at",
        ]
        read_only_fields = fields

    def get_author(self, obj: Comment) -> dict[str, Any]:
        return {"id": obj.author_id, "username": obj.author.get_username()}

    def get_resolved_by(self, obj: Comment) -> str | None:
        return obj.resolved_by.get_username() if obj.resolved_by else None


def comment_data(comment: Comment) -> dict[str, Any]:
    return CommentSerializer(comment).data


def clean_text(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise serializers.ValidationError({"text": ["Write something."]})
    if len(value) > 5000:
        raise serializers.ValidationError({"text": ["At most 5000 characters."]})
    return value.strip()


def create_comment(project: Project, user, data: dict[str, Any]) -> Comment:
    text = clean_text(data.get("text"))
    parent = None
    if data.get("parent") is not None:
        parent = project.comments.filter(pk=data["parent"], parent__isnull=True).first()
        if parent is None:
            raise serializers.ValidationError({"parent": ["No such thread."]})
    node_id = parent.node_id if parent else data.get("node_id") or ""
    if not isinstance(node_id, str) or len(node_id) > 100:
        raise serializers.ValidationError({"node_id": ["Invalid layer id."]})
    if project.comments.count() >= MAX_COMMENTS:
        raise serializers.ValidationError(
            {"detail": f"A network can have at most {MAX_COMMENTS} comments."}
        )
    comment = Comment.objects.create(
        project=project, author=user, parent=parent, node_id=node_id, text=text
    )
    live.notify_comment(project.pk, "created", comment_data(comment))
    return comment


def update_comment(comment: Comment, user, role: str, data: dict[str, Any]) -> Comment:
    """Authors change the text; authors, editors and the owner (un)resolve threads."""
    if "text" in data:
        if comment.author_id != user.pk:
            raise PermissionError("Only the author can edit a comment.")
        comment.text = clean_text(data["text"])
    if "resolved" in data:
        if comment.author_id != user.pk and role not in ("owner", "editor"):
            raise PermissionError("Only the author, editors and the owner can resolve it.")
        comment.resolved = data["resolved"] is True
        comment.resolved_by = user if comment.resolved else None
    comment.save()
    live.notify_comment(comment.project_id, "updated", comment_data(comment))
    return comment


def delete_comment(comment: Comment, user, role: str) -> None:
    if comment.author_id != user.pk and role != "owner":
        raise PermissionError("Only the author and the owner can delete a comment.")
    project_id, comment_id = comment.project_id, comment.pk
    comment.delete()  # replies go with it
    live.notify_comment(project_id, "deleted", {"id": comment_id})
