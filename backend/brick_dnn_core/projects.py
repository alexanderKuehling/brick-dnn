"""Saved networks and sharing them: members with roles, view links, conflict-safe saving.

Roles: the ``owner`` (one per project), ``editor`` (may change the network) and ``viewer``
(may look). Only the owner manages members and the link and may delete the project; members
can leave. Every change raises ``Project.version``: a save that names an older version is a
conflict (409), so two people cannot overwrite each other unnoticed.
"""

from __future__ import annotations

import secrets
from typing import Any

from django.contrib.auth import get_user_model
from django.db import transaction
from django.db.models import Exists, OuterRef, Q, QuerySet, Subquery
from rest_framework import permissions, serializers, status, viewsets
from rest_framework.decorators import action
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.throttling import ScopedRateThrottle
from rest_framework.views import APIView

from . import history, live, shared_data
from .models import Comment, Project, ProjectMember
from .serializers import ProjectSerializer

User = get_user_model()

MAX_MEMBERS = 50
ROLES = ("owner", "editor", "viewer")
MEMBER_ROLES = ("editor", "viewer")


# --------------------------------------------------------------------------- #
# Access
# --------------------------------------------------------------------------- #


def projects_for(user) -> QuerySet[Project]:
    """Projects the user owns or was invited to, with ``member_role`` and ``is_shared``."""
    memberships = ProjectMember.objects.filter(user=user)
    return (
        Project.objects.filter(Q(owner=user) | Q(pk__in=memberships.values("project")))
        .select_related("owner", "updated_by")
        .annotate(
            member_role=Subquery(memberships.filter(project=OuterRef("pk")).values("role")[:1]),
            has_members=Exists(ProjectMember.objects.filter(project=OuterRef("pk"))),
        )
    )


def role_of(user, project: Project) -> str | None:
    if not user.is_authenticated:
        return None
    if project.owner_id == user.pk:
        return "owner"
    if hasattr(project, "member_role"):
        return project.member_role
    member = ProjectMember.objects.filter(project=project, user=user).first()
    return member.role if member else None


def can_edit(user, project: Project) -> bool:
    return role_of(user, project) in ("owner", "editor")


class VersionConflict(Exception):
    def __init__(self, project: Project):
        super().__init__("This network was changed by someone else.")
        self.project = project


def update_project(
    project: Project,
    user,
    data: dict[str, Any],
    expected_version: Any = None,
    history_kind: str = "auto",
    history_label: str = "",
) -> Project:
    """Change name, description or graph. Raises :class:`VersionConflict` and
    ``serializers.ValidationError``. The caller checks that ``user`` may edit."""
    with transaction.atomic():
        current = Project.objects.select_for_update().get(pk=project.pk)
        if expected_version is not None and expected_version != current.version:
            raise VersionConflict(current)
        serializer = ProjectSerializer(current, data=data, partial=True)
        serializer.is_valid(raise_exception=True)
        updated = serializer.save(version=current.version + 1, updated_by=user)
        history.record(updated, user, history_kind, history_label)
        # People editing it live get the new state (after the commit, so they can load it).
        transaction.on_commit(lambda: live.notify_replaced(updated, user))
    for attr in ("member_role", "has_members"):
        if hasattr(project, attr):
            setattr(updated, attr, getattr(project, attr))
    return updated


def delete_project(project: Project) -> None:
    """Delete a project and close it for everyone editing it live. The caller checks that
    the user is its owner. Training runs started in it are kept (for who started them)."""
    project_id = project.pk
    project.delete()
    live.notify_deleted(project_id)


def copy_project(project: Project, user) -> Project:
    name = project.name if project.name.endswith("(copy)") else f"{project.name} (copy)"
    copy = Project.objects.create(
        owner=user,
        name=name[:120],
        description=project.description,
        graph=project.graph,
        updated_by=user,
    )
    history.record(copy, user)
    return copy


class ProjectAccess(permissions.IsAuthenticated):
    """Everyone with a role may look; editors may change; only the owner may delete."""

    def has_object_permission(self, request, view, obj: Project) -> bool:
        role = role_of(request.user, obj)
        if view.action in ("destroy", "link"):
            return role == "owner"
        if view.action in ("update", "partial_update"):
            return role in ("owner", "editor")
        return role is not None


# --------------------------------------------------------------------------- #
# Serializers
# --------------------------------------------------------------------------- #


def user_ref(user) -> dict[str, Any]:
    return {"id": user.pk, "username": user.get_username()}


class MemberSerializer(serializers.ModelSerializer):
    user = serializers.SerializerMethodField()

    class Meta:
        model = ProjectMember
        fields = ["id", "user", "role", "created_at"]
        read_only_fields = fields

    def get_user(self, obj: ProjectMember) -> dict[str, Any]:
        return user_ref(obj.user)


def project_data(project: Project, user) -> dict[str, Any]:
    """A project as the API returns it, seen by ``user``."""
    role = role_of(user, project)
    data = ProjectSerializer(project).data
    data.update(
        {
            "owner": user_ref(project.owner),
            "role": role,
            "version": project.version,
            "updated_by": project.updated_by.get_username() if project.updated_by else None,
            "shared": bool(getattr(project, "has_members", False)) or project.link_access != "none",
            "link_access": project.link_access,
            "share_token": project.share_token if role == "owner" else None,
        }
    )
    return data


# --------------------------------------------------------------------------- #
# Views
# --------------------------------------------------------------------------- #


class ProjectViewSet(viewsets.ModelViewSet):
    """The user's own projects and the ones shared with them (``?scope=mine|shared``)."""

    serializer_class = ProjectSerializer
    permission_classes = [ProjectAccess]
    throttle_scope = "sharing"

    def get_queryset(self):
        projects = projects_for(self.request.user)
        scope = self.request.query_params.get("scope")
        if scope == "mine":
            projects = projects.filter(owner=self.request.user)
        elif scope == "shared":
            projects = projects.exclude(owner=self.request.user)
        return projects.order_by("-updated_at")

    def get_throttles(self):
        throttles = super().get_throttles()
        if self.action in ("members", "link") and self.request.method == "POST":
            throttles.append(ScopedRateThrottle())
        return throttles

    def list(self, request: Request) -> Response:
        user = request.user
        return Response([project_data(p, user) for p in self.get_queryset()])

    def retrieve(self, request: Request, pk=None) -> Response:
        return Response(project_data(self.get_object(), request.user))

    def create(self, request: Request) -> Response:
        serializer = ProjectSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        project = serializer.save(owner=request.user, updated_by=request.user)
        history.record(project, request.user)
        return Response(project_data(project, request.user), status=status.HTTP_201_CREATED)

    def update(self, request: Request, pk=None, partial: bool = False) -> Response:
        project = self.get_object()
        data = request.data if isinstance(request.data, dict) else {}
        version = data.get("version")
        if version is not None and (not isinstance(version, int) or isinstance(version, bool)):
            return Response({"version": ["Must be a number."]}, status=400)
        changes = {key: data[key] for key in ("name", "description", "graph") if key in data}
        try:
            project = update_project(project, request.user, changes, version)
        except VersionConflict as exc:
            current = projects_for(request.user).get(pk=exc.project.pk)
            who = current.updated_by.get_username() if current.updated_by else "Someone"
            return Response(
                {
                    "detail": f"{who} changed this network since you opened it.",
                    "project": project_data(current, request.user),
                },
                status=status.HTTP_409_CONFLICT,
            )
        return Response(project_data(project, request.user))

    def partial_update(self, request: Request, pk=None) -> Response:
        return self.update(request, pk, partial=True)

    def perform_destroy(self, instance: Project) -> None:
        delete_project(instance)

    @action(detail=True, methods=["post"])
    def duplicate(self, request: Request, pk=None) -> Response:
        copy = copy_project(self.get_object(), request.user)
        return Response(project_data(copy, request.user), status=status.HTTP_201_CREATED)

    @action(detail=True, methods=["get", "post"])
    def members(self, request: Request, pk=None) -> Response:
        project = self.get_object()
        if request.method == "GET":
            members = project.members.select_related("user")
            return Response(
                {
                    "owner": user_ref(project.owner),
                    "members": MemberSerializer(members, many=True).data,
                }
            )
        if role_of(request.user, project) != "owner":
            return Response(
                {"detail": "Only the owner can invite people."}, status=status.HTTP_403_FORBIDDEN
            )
        data = request.data if isinstance(request.data, dict) else {}
        role = data.get("role")
        if role not in MEMBER_ROLES:
            return Response({"role": ["Choose 'editor' or 'viewer'."]}, status=400)
        try:
            invitee = _find_user(data.get("user"))
        except serializers.ValidationError as exc:
            return Response({"user": exc.detail}, status=400)
        if invitee.pk == project.owner_id:
            return Response({"user": ["You own this network already."]}, status=400)
        if project.members.filter(user=invitee).exists():
            return Response({"user": [f"{invitee.get_username()} already has access."]}, status=400)
        if project.members.count() >= MAX_MEMBERS:
            return Response(
                {"detail": f"A network can be shared with at most {MAX_MEMBERS} people."},
                status=400,
            )
        member = ProjectMember.objects.create(
            project=project, user=invitee, role=role, invited_by=request.user
        )
        return Response(MemberSerializer(member).data, status=status.HTTP_201_CREATED)

    @action(detail=True, methods=["patch", "delete"], url_path=r"members/(?P<member_id>\d+)")
    def member(self, request: Request, pk=None, member_id=None) -> Response:
        project = self.get_object()
        member = project.members.select_related("user").filter(pk=member_id).first()
        if member is None:
            return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)
        is_owner = role_of(request.user, project) == "owner"
        if request.method == "DELETE":
            # The owner removes anyone; members can leave.
            if not is_owner and member.user_id != request.user.pk:
                return Response(
                    {"detail": "Only the owner can remove people."},
                    status=status.HTTP_403_FORBIDDEN,
                )
            member.delete()
            live.notify_access(project.pk, member.user_id, None)
            return Response(status=status.HTTP_204_NO_CONTENT)
        if not is_owner:
            return Response(
                {"detail": "Only the owner can change roles."}, status=status.HTTP_403_FORBIDDEN
            )
        role = request.data.get("role") if isinstance(request.data, dict) else None
        if role not in MEMBER_ROLES:
            return Response({"role": ["Choose 'editor' or 'viewer'."]}, status=400)
        member.role = role
        member.save(update_fields=["role"])
        live.notify_access(project.pk, member.user_id, role)
        return Response(MemberSerializer(member).data)

    @action(detail=True, methods=["post"])
    def link(self, request: Request, pk=None) -> Response:
        """Turn the view link on or off, or replace it (``regenerate``)."""
        project = self.get_object()
        data = request.data if isinstance(request.data, dict) else {}
        if data.get("enabled") is True:
            project.link_access = "view"
            if not project.share_token or data.get("regenerate") is True:
                project.share_token = secrets.token_urlsafe(18)
        else:
            # Turning it off kills the old link for good; turning it on makes a new one.
            project.link_access = "none"
            project.share_token = None
        project.save(update_fields=["link_access", "share_token"])
        return Response({"link_access": project.link_access, "share_token": project.share_token})

    # Data of the project ----------------------------------------------------

    @action(detail=True, methods=["get", "post"])
    def data(self, request: Request, pk=None) -> Response:
        """``GET``: datasets and data preparations shared in this project.
        ``POST {kind: "dataset" | "prep", id}``: add one of your own (owner and editors)."""
        project = self.get_object()
        if request.method == "GET":
            return Response(shared_data.project_data_items(project))
        if not can_edit(request.user, project):
            return Response({"detail": "View only."}, status=status.HTTP_403_FORBIDDEN)
        body = request.data if isinstance(request.data, dict) else {}
        item = shared_data.find_item(body.get("kind"), body.get("id"))
        if item is None or item.owner_id != request.user.pk:
            return Response(
                {"detail": "Choose one of your own datasets or data preparations."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        item.projects.add(project)
        return Response(shared_data.project_data_items(project), status=status.HTTP_201_CREATED)

    @action(
        detail=True,
        methods=["delete"],
        url_path=r"data/(?P<kind>dataset|prep)/(?P<item_id>\d+)",
    )
    def data_item(self, request: Request, pk=None, kind=None, item_id=None) -> Response:
        """Stop sharing an item in this project (its owner or the project's owner)."""
        project = self.get_object()
        item = shared_data.find_item(kind, int(item_id))
        if item is None or not item.projects.filter(pk=project.pk).exists():
            return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)
        if item.owner_id != request.user.pk and project.owner_id != request.user.pk:
            return Response(
                {"detail": "Only its owner or the project's owner can remove it."},
                status=status.HTTP_403_FORBIDDEN,
            )
        item.projects.remove(project)
        return Response(status=status.HTTP_204_NO_CONTENT)

    # History ----------------------------------------------------------------

    @action(detail=True, methods=["get", "post"])
    def versions(self, request: Request, pk=None) -> Response:
        """``GET``: the history, newest first. ``POST {label}``: name the current state."""
        project = self.get_object()
        if request.method == "GET":
            entries = project.history.select_related("created_by")
            return Response([history.version_data(entry) for entry in entries])
        if not can_edit(request.user, project):
            return Response({"detail": "View only."}, status=status.HTTP_403_FORBIDDEN)
        data = request.data if isinstance(request.data, dict) else {}
        label = history.clean_label(data.get("label"))
        entry = history.record(project, request.user, "named", label)
        return Response(history.version_data(entry), status=status.HTTP_201_CREATED)

    @action(detail=True, methods=["get", "patch"], url_path=r"versions/(?P<version_id>\d+)")
    def version_detail(self, request: Request, pk=None, version_id=None) -> Response:
        project = self.get_object()
        entry = project.history.select_related("created_by").filter(pk=version_id).first()
        if entry is None:
            return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)
        if request.method == "GET":
            return Response(history.version_data(entry, with_graph=True))
        if not can_edit(request.user, project):
            return Response({"detail": "View only."}, status=status.HTTP_403_FORBIDDEN)
        data = request.data if isinstance(request.data, dict) else {}
        entry.label = history.clean_label(data.get("label"))
        entry.kind = "named"  # named versions are kept
        entry.save(update_fields=["label", "kind"])
        live.notify_history(project.pk)
        return Response(history.version_data(entry))

    @action(detail=True, methods=["post"], url_path=r"versions/(?P<version_id>\d+)/restore")
    def restore(self, request: Request, pk=None, version_id=None) -> Response:
        """Make an old version the current state (itself a new entry in the history)."""
        project = self.get_object()
        if not can_edit(request.user, project):
            return Response({"detail": "View only."}, status=status.HTTP_403_FORBIDDEN)
        entry = project.history.filter(pk=version_id).first()
        if entry is None:
            return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)
        when = entry.label or f"version {entry.version}"
        project = update_project(
            project,
            request.user,
            {"name": entry.name, "description": entry.description, "graph": entry.graph},
            history_kind="restore",
            history_label=f"Restored {when}"[:120],
        )
        return Response(project_data(project, request.user))

    # Comments ---------------------------------------------------------------

    @action(detail=True, methods=["get", "post"])
    def comments(self, request: Request, pk=None) -> Response:
        """``GET``: all comments, oldest first (``?node=<id>`` for one layer).
        ``POST {text, node_id?, parent?}``: comment (every role, viewers too)."""
        project = self.get_object()
        if request.method == "GET":
            comments = project.comments.select_related("author", "resolved_by")
            node = request.query_params.get("node")
            if node is not None:
                comments = comments.filter(node_id=node)
            return Response([history.comment_data(c) for c in comments])
        data = request.data if isinstance(request.data, dict) else {}
        comment = history.create_comment(project, request.user, data)
        return Response(history.comment_data(comment), status=status.HTTP_201_CREATED)

    @action(detail=True, methods=["patch", "delete"], url_path=r"comments/(?P<comment_id>\d+)")
    def comment(self, request: Request, pk=None, comment_id=None) -> Response:
        project = self.get_object()
        comment: Comment | None = (
            project.comments.select_related("author", "resolved_by").filter(pk=comment_id).first()
        )
        if comment is None:
            return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)
        role = role_of(request.user, project)
        try:
            if request.method == "DELETE":
                history.delete_comment(comment, request.user, role)
                return Response(status=status.HTTP_204_NO_CONTENT)
            data = request.data if isinstance(request.data, dict) else {}
            comment = history.update_comment(comment, request.user, role, data)
        except PermissionError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_403_FORBIDDEN)
        return Response(history.comment_data(comment))


def _find_user(identifier: Any):
    if not isinstance(identifier, str) or not identifier.strip():
        raise serializers.ValidationError("Enter a username or an e-mail address.")
    identifier = identifier.strip()
    users = User.objects.filter(is_active=True)
    matches = list(users.filter(username__iexact=identifier)[:2])
    if not matches and "@" in identifier:
        matches = list(users.filter(email__iexact=identifier)[:2])
        if len(matches) > 1:
            raise serializers.ValidationError(
                "Several accounts use this e-mail address. Enter the username instead."
            )
    if not matches:
        raise serializers.ValidationError("There is no account with this username or e-mail.")
    return matches[0]


def _shared_project(token: str) -> Project | None:
    return (
        Project.objects.select_related("owner")
        .filter(share_token=token, link_access="view")
        .first()
    )


class SharedProjectView(APIView):
    """``GET /api/shared/<token>/``: a network shared by link, for everyone."""

    permission_classes = [permissions.AllowAny]

    def get(self, request: Request, token: str) -> Response:
        project = _shared_project(token)
        if project is None:
            return Response(
                {"detail": "This link does not exist or was turned off."},
                status=status.HTTP_404_NOT_FOUND,
            )
        return Response(
            {
                "name": project.name,
                "description": project.description,
                "graph": project.graph,
                "owner": project.owner.get_username(),
                "updated_at": project.updated_at,
            }
        )


class SharedProjectDuplicateView(APIView):
    """``POST /api/shared/<token>/duplicate/``: copy a linked network into your account."""

    permission_classes = [permissions.IsAuthenticated]

    def post(self, request: Request, token: str) -> Response:
        project = _shared_project(token)
        if project is None:
            return Response(
                {"detail": "This link does not exist or was turned off."},
                status=status.HTTP_404_NOT_FOUND,
            )
        copy = copy_project(project, request.user)
        return Response(project_data(copy, request.user), status=status.HTTP_201_CREATED)
