"""Live editing of a project by several people at once, over a WebSocket.

The server holds the truth: clients send small operations ("ops": add a layer, change a
parameter, move it, connect two layers, …), the server applies them one after the other to
the stored graph (``Project.graph``), raises ``Project.version`` and sends them to everyone
in the project's room, the sender included (as confirmation). Two people changing different
things never conflict; changing the same field, the later change wins; a change to a layer
someone just deleted is dropped.

Who is online, what they have selected and where they drag a layer right now ("presence")
is only passed on, never stored.

Protocol (JSON messages over ``/api/ws/projects/<id>/``), client → server:

- ``{"type": "auth", "token": "<JWT access token>"}`` first; the server answers ``hello``
  with the project (graph, name, description, version), the user's role and ``client_id``.
- ``{"type": "ops", "id": "<client message id>", "ops": [op, …]}`` (owner and editors)
- ``{"type": "presence", "state": {…}}`` (e.g. selected layers, a layer being dragged)
- ``{"type": "ping"}``

Server → client: ``hello``, ``ops`` (applied ops with ``version``, ``by`` and the sender's
``client_id`` and ``id``), ``rejected`` (none of the ops could be applied), ``replaced`` (the
whole project changed, e.g. saved through the REST API or MCP, or a version was restored),
``presence``, ``joined``, ``left``, ``access`` (the user's role changed or was removed),
``comment`` (``action``: created, updated or deleted, and the ``comment``), ``history`` (the
version history changed), ``deleted``, ``error``, ``pong``.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import math
import time
import uuid
from typing import Any

from asgiref.sync import async_to_sync
from channels.db import database_sync_to_async
from channels.generic.websocket import AsyncJsonWebsocketConsumer
from channels.layers import get_channel_layer
from django.contrib.auth import get_user_model
from django.db import transaction
from rest_framework_simplejwt.exceptions import TokenError
from rest_framework_simplejwt.tokens import AccessToken

from brickdnn.graph import MAX_EDGES, MAX_NODES

from .models import Project

logger = logging.getLogger(__name__)
User = get_user_model()

AUTH_TIMEOUT = 10  # seconds to send the auth message
MAX_OPS_PER_MESSAGE = 500
MAX_ID_LENGTH = 100
# Per connection: at most this many messages per second on average (bursts allowed).
RATE_PER_SECOND = 30
RATE_BURST = 120

# Close codes (4000-4999 are free for applications).
CLOSE_UNAUTHORIZED = 4401
CLOSE_FORBIDDEN = 4403
CLOSE_NOT_FOUND = 4404
CLOSE_TOO_FAST = 4429


def group_name(project_id: int) -> str:
    return f"project-{project_id}"


# --------------------------------------------------------------------------- #
# Operations on a graph document
# --------------------------------------------------------------------------- #


class OpError(ValueError):
    """An op that cannot be applied (malformed, or about something that is gone)."""


def _check_id(value: Any, what: str) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        value = str(value)
    if not isinstance(value, str) or not value or len(value) > MAX_ID_LENGTH:
        raise OpError(f"Invalid {what}.")
    return value


def _check_position(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        raise OpError("Invalid position.")
    x, y = value.get("x"), value.get("y")
    for number in (x, y):
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            raise OpError("Invalid position.")
        if not math.isfinite(number) or abs(number) > 1e7:
            raise OpError("Invalid position.")
    return {"x": x, "y": y}


def _clean_node(node: Any) -> dict[str, Any]:
    if not isinstance(node, dict):
        raise OpError("A node must be an object.")
    clean: dict[str, Any] = {
        "id": _check_id(node.get("id"), "node id"),
        "type": _check_id(node.get("type"), "layer type"),
        "params": {} if node.get("params") is None else node["params"],
    }
    if not isinstance(clean["params"], dict):
        raise OpError("'params' must be an object.")
    if node.get("name") is not None:
        if not isinstance(node["name"], str) or len(node["name"]) > MAX_ID_LENGTH:
            raise OpError("Invalid name.")
        clean["name"] = node["name"]
    if node.get("position") is not None:
        clean["position"] = _check_position(node["position"])
    return clean


def _find_node(graph: dict[str, Any], node_id: str) -> dict[str, Any] | None:
    return next((n for n in graph["nodes"] if str(n.get("id")) == node_id), None)


def apply_op(graph: dict[str, Any], op: Any) -> dict[str, Any]:
    """Apply one op to ``graph`` (in place) and return it normalised. Raises OpError."""
    if not isinstance(op, dict):
        raise OpError("An op must be an object.")
    kind = op.get("op")
    nodes, edges = graph["nodes"], graph["edges"]

    if kind == "add_node":
        node = _clean_node(op.get("node"))
        if _find_node(graph, node["id"]) is not None:
            raise OpError("A layer with this id exists already.")
        if len(nodes) >= MAX_NODES:
            raise OpError(f"Networks are limited to {MAX_NODES} layers.")
        nodes.append(node)
        return {"op": kind, "node": node}

    if kind == "update_node":
        node_id = _check_id(op.get("id"), "node id")
        node = _find_node(graph, node_id)
        if node is None:
            raise OpError("This layer was deleted.")
        changes = op.get("set")
        if not isinstance(changes, dict) or not changes:
            raise OpError("'set' must be a non-empty object.")
        applied: dict[str, Any] = {}
        if "params" in changes:
            # Parameter by parameter: two people changing different ones both win.
            params = changes["params"]
            if not isinstance(params, dict):
                raise OpError("'params' must be an object.")
            current = node.setdefault("params", {})
            for key, value in params.items():
                if value is None:
                    current.pop(key, None)
                else:
                    current[key] = value
            applied["params"] = params
        if "position" in changes:
            node["position"] = applied["position"] = _check_position(changes["position"])
        if "name" in changes:
            name = changes["name"]
            if name is not None and (not isinstance(name, str) or len(name) > MAX_ID_LENGTH):
                raise OpError("Invalid name.")
            if name:
                node["name"] = name
            else:
                node.pop("name", None)
            applied["name"] = name or None
        if "type" in changes:
            node["type"] = applied["type"] = _check_id(changes["type"], "layer type")
        if not applied:
            raise OpError("Nothing to change.")
        return {"op": kind, "id": node_id, "set": applied}

    if kind == "remove_node":
        node_id = _check_id(op.get("id"), "node id")
        if _find_node(graph, node_id) is None:
            raise OpError("This layer was deleted already.")
        graph["nodes"] = [n for n in nodes if str(n.get("id")) != node_id]
        graph["edges"] = [
            e for e in edges if str(e.get("source")) != node_id and str(e.get("target")) != node_id
        ]
        return {"op": kind, "id": node_id}

    if kind in ("add_edge", "remove_edge"):
        source = _check_id(op.get("source"), "source")
        target = _check_id(op.get("target"), "target")
        exists = any(
            str(e.get("source")) == source and str(e.get("target")) == target for e in edges
        )
        if kind == "remove_edge":
            if not exists:
                raise OpError("This connection was removed already.")
            graph["edges"] = [
                e
                for e in edges
                if not (str(e.get("source")) == source and str(e.get("target")) == target)
            ]
        else:
            if exists:
                raise OpError("These layers are connected already.")
            if _find_node(graph, source) is None or _find_node(graph, target) is None:
                raise OpError("One of the layers was deleted.")
            if source == target:
                raise OpError("A layer cannot be connected to itself.")
            if len(edges) >= MAX_EDGES:
                raise OpError(f"Networks are limited to {MAX_EDGES} connections.")
            edges.append({"source": source, "target": target})
        return {"op": kind, "source": source, "target": target}

    raise OpError(f"Unknown op: {kind}")


def _normalised(graph: Any) -> dict[str, Any]:
    graph = copy.deepcopy(graph) if isinstance(graph, dict) else {}
    graph.setdefault("version", 1)
    if not isinstance(graph.get("nodes"), list):
        graph["nodes"] = []
    if not isinstance(graph.get("edges"), list):
        graph["edges"] = []
    return graph


def apply_ops(project_id: int, user, ops: list[Any]) -> tuple[int, list[dict[str, Any]]]:
    """Apply ops to the stored project in one transaction. Ops that cannot be applied are
    left out. Returns the new version and the applied ops (the version only changes if at
    least one op was applied)."""
    with transaction.atomic():
        project = Project.objects.select_for_update().get(pk=project_id)
        graph = _normalised(project.graph)
        meta: dict[str, Any] = {}
        applied: list[dict[str, Any]] = []
        for op in ops:
            if isinstance(op, dict) and op.get("op") == "set_meta":
                change = _meta_change(op)
                if not change:
                    continue
                meta.update(change)
                applied.append({"op": "set_meta", **change})
                continue
            try:
                applied.append(apply_op(graph, op))
            except OpError:
                continue
        if not applied:
            return project.version, []
        project.graph = graph
        for field, value in meta.items():
            setattr(project, field, value)
        project.version += 1
        project.updated_by = user
        project.save(update_fields=["graph", *meta, "version", "updated_by", "updated_at"])
        from .history import record

        record(project, user)
        return project.version, applied


def _meta_change(op: dict[str, Any]) -> dict[str, Any]:
    change: dict[str, Any] = {}
    if isinstance(op.get("name"), str):
        change["name"] = op["name"].strip()[:120] or "Untitled network"
    if isinstance(op.get("description"), str):
        change["description"] = op["description"][:2000]
    return change


# --------------------------------------------------------------------------- #
# Notifications from the rest of the app (REST API, MCP, sharing)
# --------------------------------------------------------------------------- #


def _send(project_id: int, message: dict[str, Any]) -> None:
    """Tell everyone connected to the project. Never breaks the request that changed it."""
    layer = get_channel_layer()
    if layer is None:
        return
    try:
        async_to_sync(layer.group_send)(group_name(project_id), message)
    except Exception:
        logger.exception("Could not notify the live editors of project %s", project_id)


def notify_replaced(project: Project, user) -> None:
    """The project was saved as a whole (REST API, MCP): connected editors reload it."""
    _send(
        project.pk,
        {
            "type": "project.replaced",
            "project": _project_state(project),
            "by": _user_ref(user),
        },
    )


def notify_access(project_id: int, user_id: int, role: str | None) -> None:
    _send(project_id, {"type": "project.access", "user_id": user_id, "role": role})


def notify_deleted(project_id: int) -> None:
    _send(project_id, {"type": "project.deleted"})


def notify_history(project_id: int) -> None:
    """The history changed (a new or updated entry): open history panels reload it."""
    _send(project_id, {"type": "project.history"})


def notify_comment(project_id: int, action: str, comment: dict[str, Any]) -> None:
    _send(project_id, {"type": "project.comment", "action": action, "comment": comment})


def _user_ref(user) -> dict[str, Any]:
    return {"id": user.pk, "username": user.get_username()}


def _project_state(project: Project) -> dict[str, Any]:
    return {
        "id": project.pk,
        "name": project.name,
        "description": project.description,
        "graph": project.graph,
        "version": project.version,
    }


# --------------------------------------------------------------------------- #
# The WebSocket consumer
# --------------------------------------------------------------------------- #


def _user_from_token(token: Any):
    if not isinstance(token, str):
        return None
    try:
        access = AccessToken(token)
    except TokenError:
        return None
    return User.objects.filter(pk=access.get("user_id"), is_active=True).first()


def _load(project_id: int, user) -> tuple[dict[str, Any], str] | None:
    from .projects import projects_for, role_of

    project = projects_for(user).filter(pk=project_id).first()
    if project is None:
        return None
    return _project_state(project), role_of(user, project)


class ProjectConsumer(AsyncJsonWebsocketConsumer):
    async def connect(self) -> None:
        self.project_id = int(self.scope["url_route"]["kwargs"]["project_id"])
        self.user = None
        self.role: str | None = None
        self.client_id = uuid.uuid4().hex[:12]
        self.tokens = float(RATE_BURST)
        self.last_refill = time.monotonic()
        await self.accept()
        self.auth_timer = asyncio.get_running_loop().call_later(
            AUTH_TIMEOUT, lambda: asyncio.ensure_future(self._auth_timeout())
        )

    async def _auth_timeout(self) -> None:
        if self.user is None:
            await self.close(code=CLOSE_UNAUTHORIZED)

    async def disconnect(self, code: int) -> None:
        if getattr(self, "auth_timer", None) is not None:
            self.auth_timer.cancel()
        if getattr(self, "user", None) is not None:
            await self.channel_layer.group_discard(group_name(self.project_id), self.channel_name)
            await self._broadcast({"type": "peer.left", "client_id": self.client_id})

    def _within_rate(self) -> bool:
        now = time.monotonic()
        self.tokens = min(RATE_BURST, self.tokens + (now - self.last_refill) * RATE_PER_SECOND)
        self.last_refill = now
        if self.tokens < 1:
            return False
        self.tokens -= 1
        return True

    async def receive_json(self, content: Any, **kwargs) -> None:
        if not self._within_rate():
            await self.close(code=CLOSE_TOO_FAST)
            return
        if not isinstance(content, dict):
            return
        kind = content.get("type")
        if self.user is None:
            if kind == "auth":
                await self._authenticate(content.get("token"))
            else:
                await self.close(code=CLOSE_UNAUTHORIZED)
            return
        if kind == "ops":
            await self._ops(content)
        elif kind == "presence":
            state = content.get("state")
            if isinstance(state, dict):
                await self._broadcast(
                    {
                        "type": "peer.presence",
                        "client_id": self.client_id,
                        "user": _user_ref(self.user),
                        "state": state,
                    }
                )
        elif kind == "ping":
            await self.send_json({"type": "pong"})

    async def _authenticate(self, token: Any) -> None:
        user = await database_sync_to_async(_user_from_token)(token)
        if user is None:
            await self.close(code=CLOSE_UNAUTHORIZED)
            return
        loaded = await database_sync_to_async(_load)(self.project_id, user)
        if loaded is None:
            await self.close(code=CLOSE_NOT_FOUND)
            return
        self.auth_timer.cancel()
        self.user = user
        state, self.role = loaded
        await self.channel_layer.group_add(group_name(self.project_id), self.channel_name)
        await self.send_json(
            {
                "type": "hello",
                "project": state,
                "role": self.role,
                "client_id": self.client_id,
                "user": _user_ref(user),
            }
        )
        # The others answer with their presence, so the newcomer learns who is here.
        await self._broadcast(
            {
                "type": "peer.joined",
                "client_id": self.client_id,
                "channel": self.channel_name,
                "user": _user_ref(user),
            }
        )

    async def _ops(self, content: dict[str, Any]) -> None:
        message_id = content.get("id")
        ops = content.get("ops")
        if self.role not in ("owner", "editor"):
            await self.send_json({"type": "error", "id": message_id, "detail": "View only."})
            return
        if not isinstance(ops, list) or not 0 < len(ops) <= MAX_OPS_PER_MESSAGE:
            await self.send_json({"type": "error", "id": message_id, "detail": "Invalid ops."})
            return
        try:
            version, applied = await database_sync_to_async(apply_ops)(
                self.project_id, self.user, ops
            )
        except Project.DoesNotExist:
            await self.close(code=CLOSE_NOT_FOUND)
            return
        if not applied:
            await self.send_json({"type": "rejected", "id": message_id, "version": version})
            return
        await self._broadcast(
            {
                "type": "ops.applied",
                "ops": applied,
                "version": version,
                "by": _user_ref(self.user),
                "client_id": self.client_id,
                "id": message_id,
            }
        )

    async def _broadcast(self, message: dict[str, Any]) -> None:
        await self.channel_layer.group_send(group_name(self.project_id), message)

    # Messages from the group (names: "type" with "." replaced by "_").

    async def ops_applied(self, event: dict[str, Any]) -> None:
        await self.send_json({**event, "type": "ops"})

    async def peer_presence(self, event: dict[str, Any]) -> None:
        if event["client_id"] != self.client_id:
            await self.send_json({**event, "type": "presence"})

    async def peer_joined(self, event: dict[str, Any]) -> None:
        if event["client_id"] == self.client_id:
            return
        await self.send_json(
            {"type": "joined", "client_id": event["client_id"], "user": event["user"]}
        )
        # Introduce ourselves to the newcomer only.
        await self.channel_layer.send(
            event["channel"],
            {
                "type": "peer.here",
                "client_id": self.client_id,
                "user": _user_ref(self.user),
            },
        )

    async def peer_here(self, event: dict[str, Any]) -> None:
        await self.send_json({**event, "type": "joined"})

    async def peer_left(self, event: dict[str, Any]) -> None:
        if event["client_id"] != self.client_id:
            await self.send_json({"type": "left", "client_id": event["client_id"]})

    async def project_replaced(self, event: dict[str, Any]) -> None:
        await self.send_json({**event, "type": "replaced"})

    async def project_access(self, event: dict[str, Any]) -> None:
        if event["user_id"] != self.user.pk:
            return
        self.role = event["role"]
        await self.send_json({"type": "access", "role": self.role})
        if self.role is None:
            await self.close(code=CLOSE_FORBIDDEN)

    async def project_history(self, event: dict[str, Any]) -> None:
        await self.send_json({"type": "history"})

    async def project_comment(self, event: dict[str, Any]) -> None:
        await self.send_json({**event, "type": "comment"})

    async def project_deleted(self, event: dict[str, Any]) -> None:
        await self.send_json({"type": "deleted"})
        await self.close(code=CLOSE_NOT_FOUND)
