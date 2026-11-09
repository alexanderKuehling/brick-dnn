"""Live editing over WebSockets: ops, presence, access changes (brick_dnn_core.live)."""

from __future__ import annotations

import pytest
from asgiref.sync import async_to_sync
from channels.testing import WebsocketCommunicator
from django.contrib.auth import get_user_model
from rest_framework.test import APIClient
from rest_framework_simplejwt.tokens import AccessToken

from brick_dnn_core.live import OpError, apply_op
from brick_dnn_core.models import Project, ProjectMember
from brick_dnn_site.asgi import application

from .helpers import chain

GRAPH = chain(("Input", {"shape": [4]}), ("Linear", {"out_features": 2}))


# --------------------------------------------------------------------------- #
# Ops on a graph (no database)
# --------------------------------------------------------------------------- #


def fresh():
    return {"version": 1, "nodes": [dict(n) for n in GRAPH["nodes"]], "edges": list(GRAPH["edges"])}


def test_add_update_remove_nodes():
    graph = fresh()
    node = {"id": "n9", "type": "ReLU", "params": {}, "position": {"x": 1, "y": 2}}
    assert apply_op(graph, {"op": "add_node", "node": node})["node"]["id"] == "n9"
    with pytest.raises(OpError):
        apply_op(graph, {"op": "add_node", "node": node})

    applied = apply_op(
        graph, {"op": "update_node", "id": "n1", "set": {"params": {"out_features": 8}}}
    )
    assert applied["set"] == {"params": {"out_features": 8}}
    # Parameter by parameter: another change to another parameter keeps the first one.
    apply_op(graph, {"op": "update_node", "id": "n1", "set": {"params": {"bias": False}}})
    assert graph["nodes"][1]["params"] == {"out_features": 8, "bias": False}
    apply_op(graph, {"op": "update_node", "id": "n1", "set": {"params": {"bias": None}}})
    assert graph["nodes"][1]["params"] == {"out_features": 8}
    apply_op(graph, {"op": "update_node", "id": "n1", "set": {"position": {"x": 5, "y": 6}}})
    apply_op(graph, {"op": "update_node", "id": "n1", "set": {"name": "head"}})
    assert graph["nodes"][1]["position"] == {"x": 5, "y": 6} and graph["nodes"][1]["name"] == "head"

    apply_op(graph, {"op": "remove_node", "id": "n0"})
    assert [n["id"] for n in graph["nodes"]] == ["n1", "n9"]
    assert graph["edges"] == []  # its connections went with it
    with pytest.raises(OpError):  # a change to a deleted layer is dropped
        apply_op(graph, {"op": "update_node", "id": "n0", "set": {"params": {"x": 1}}})


def test_edges():
    graph = fresh()
    with pytest.raises(OpError):
        apply_op(graph, {"op": "add_edge", "source": "n0", "target": "n1"})  # exists
    apply_op(graph, {"op": "remove_edge", "source": "n0", "target": "n1"})
    assert graph["edges"] == []
    apply_op(graph, {"op": "add_edge", "source": "n0", "target": "n1"})
    for bad in [
        {"op": "add_edge", "source": "n0", "target": "gone"},
        {"op": "add_edge", "source": "n0", "target": "n0"},
        {"op": "remove_edge", "source": "n1", "target": "n0"},
    ]:
        with pytest.raises(OpError):
            apply_op(graph, bad)


@pytest.mark.parametrize(
    "op",
    [
        "nope",
        {"op": "explode"},
        {"op": "add_node", "node": {"id": "", "type": "Linear"}},
        {"op": "add_node", "node": {"id": "x", "type": "Linear", "params": []}},
        {"op": "add_node", "node": {"id": "x" * 101, "type": "Linear"}},
        {"op": "update_node", "id": "n1", "set": {}},
        {"op": "update_node", "id": "n1", "set": {"position": {"x": "a", "y": 1}}},
        {"op": "update_node", "id": "n1", "set": {"position": {"x": float("inf"), "y": 1}}},
    ],
)
def test_malformed_ops(op):
    with pytest.raises(OpError):
        apply_op(fresh(), op)


# --------------------------------------------------------------------------- #
# WebSocket
# --------------------------------------------------------------------------- #

pytestmark_db = pytest.mark.django_db(transaction=True)


def make_user(name: str):
    return get_user_model().objects.create_user(name, password="correct-horse-1")


def token(user) -> str:
    return str(AccessToken.for_user(user))


def connect(project_id: int):
    return WebsocketCommunicator(
        application,
        f"/api/ws/projects/{project_id}/",
        headers=[(b"origin", b"http://localhost:5173"), (b"host", b"localhost")],
    )


async def joined(project_id: int, user) -> tuple[WebsocketCommunicator, dict]:
    ws = connect(project_id)
    connected, _ = await ws.connect()
    assert connected
    await ws.send_json_to({"type": "auth", "token": token(user)})
    hello = await ws.receive_json_from(timeout=5)
    assert hello["type"] == "hello", hello
    return ws, hello


async def next_of(ws: WebsocketCommunicator, kind: str) -> dict:
    """The next message of a kind, skipping others (e.g. presence handshakes)."""
    for _ in range(20):
        message = await ws.receive_json_from(timeout=5)
        if message["type"] == kind:
            return message
    raise AssertionError(f"no {kind} message")


@pytest.fixture
def setup(settings):
    # The origin check uses the ALLOWED_HOSTS of the start (localhost by default).
    settings.CHANNEL_LAYERS = {"default": {"BACKEND": "channels.layers.InMemoryChannelLayer"}}
    ada, bob, cy = make_user("ada"), make_user("bob"), make_user("cy")
    project = Project.objects.create(owner=ada, name="Net", graph=GRAPH, updated_by=ada)
    ProjectMember.objects.create(project=project, user=bob, role="editor")
    ProjectMember.objects.create(project=project, user=cy, role="viewer")
    return project, ada, bob, cy


@pytestmark_db
def test_two_editors_see_each_others_changes(setup):
    project, ada, bob, _ = setup

    async def run():
        a, hello = await joined(project.pk, ada)
        assert hello["role"] == "owner" and hello["project"]["version"] == 1
        assert hello["project"]["graph"]["nodes"][0]["id"] == "n0"
        b, hello_b = await joined(project.pk, bob)
        assert hello_b["role"] == "editor"
        # They learn about each other.
        assert (await next_of(a, "joined"))["user"]["username"] == "bob"
        assert (await next_of(b, "joined"))["user"]["username"] == "ada"

        node = {"id": "n2", "type": "ReLU", "params": {}, "position": {"x": 0, "y": 0}}
        await a.send_json_to(
            {
                "type": "ops",
                "id": "m1",
                "ops": [
                    {"op": "add_node", "node": node},
                    {"op": "add_edge", "source": "n1", "target": "n2"},
                    {"op": "update_node", "id": "gone", "set": {"params": {"x": 1}}},
                ],
            }
        )
        for ws in (a, b):  # the sender gets it as confirmation, the other as news
            message = await next_of(ws, "ops")
            assert message["version"] == 2 and message["id"] == "m1"
            assert message["by"]["username"] == "ada"
            assert [op["op"] for op in message["ops"]] == ["add_node", "add_edge"]
            assert message["client_id"] == hello["client_id"]

        await b.send_json_to(
            {"type": "ops", "id": "m2", "ops": [{"op": "set_meta", "name": "Renamed"}]}
        )
        assert (await next_of(a, "ops"))["ops"] == [{"op": "set_meta", "name": "Renamed"}]

        await b.send_json_to({"type": "ops", "id": "m3", "ops": [{"op": "remove_node", "id": "x"}]})
        assert (await next_of(b, "rejected"))["id"] == "m3"

        # Presence goes to the others only.
        await b.send_json_to({"type": "presence", "state": {"selected": ["n2"]}})
        presence = await next_of(a, "presence")
        assert presence["state"] == {"selected": ["n2"]} and presence["user"]["username"] == "bob"

        await b.disconnect()
        assert (await next_of(a, "left"))["client_id"] == hello_b["client_id"]
        await a.disconnect()

    async_to_sync(run)()
    project.refresh_from_db()
    assert (
        project.version == 3 and project.name == "Renamed" and project.updated_by.username == "bob"
    )
    assert [n["id"] for n in project.graph["nodes"]] == ["n0", "n1", "n2"]
    assert {"source": "n1", "target": "n2"} in project.graph["edges"]


@pytestmark_db
def test_viewers_and_strangers(setup):
    project, _, _, cy = setup
    eve = make_user("eve")

    async def run():
        viewer, hello = await joined(project.pk, cy)
        assert hello["role"] == "viewer"
        await viewer.send_json_to(
            {"type": "ops", "id": "x", "ops": [{"op": "set_meta", "name": "Mine"}]}
        )
        assert (await next_of(viewer, "error"))["detail"] == "View only."
        await viewer.disconnect()

        stranger = connect(project.pk)
        await stranger.connect()
        await stranger.send_json_to({"type": "auth", "token": token(eve)})
        assert (await stranger.receive_output(timeout=5))["code"] == 4404

        anonymous = connect(project.pk)
        await anonymous.connect()
        await anonymous.send_json_to({"type": "ops", "ops": []})
        assert (await anonymous.receive_output(timeout=5))["code"] == 4401

        forged = connect(project.pk)
        await forged.connect()
        await forged.send_json_to({"type": "auth", "token": "not-a-jwt"})
        assert (await forged.receive_output(timeout=5))["code"] == 4401

    async_to_sync(run)()
    project.refresh_from_db()
    assert project.name == "Net" and project.version == 1


@pytestmark_db
def test_foreign_origin_is_refused(setup):
    project, *_ = setup

    async def run():
        ws = WebsocketCommunicator(
            application,
            f"/api/ws/projects/{project.pk}/",
            headers=[(b"origin", b"https://evil.example"), (b"host", b"localhost")],
        )
        connected, _ = await ws.connect()
        assert not connected

    async_to_sync(run)()


@pytestmark_db
def test_rest_saves_and_access_changes_reach_live_editors(setup):
    project, ada, bob, _ = setup

    async def run():
        b, _ = await joined(project.pk, bob)

        def save_and_demote():
            owner = APIClient()
            owner.force_authenticate(ada)
            response = owner.patch(
                f"/api/projects/{project.pk}/", {"name": "Saved via REST"}, format="json"
            )
            assert response.status_code == 200
            member = ProjectMember.objects.get(user=bob)
            owner.patch(
                f"/api/projects/{project.pk}/members/{member.pk}/",
                {"role": "viewer"},
                format="json",
            )
            return member.pk

        from channels.db import database_sync_to_async

        member_id = await database_sync_to_async(save_and_demote)()
        replaced = await next_of(b, "replaced")
        assert replaced["project"]["name"] == "Saved via REST"
        assert replaced["project"]["version"] == 2 and replaced["by"]["username"] == "ada"
        assert (await next_of(b, "access"))["role"] == "viewer"

        def remove():
            owner = APIClient()
            owner.force_authenticate(ada)
            owner.delete(f"/api/projects/{project.pk}/members/{member_id}/")

        await database_sync_to_async(remove)()
        assert (await next_of(b, "access"))["role"] is None
        assert (await b.receive_output(timeout=5))["type"] == "websocket.close"

        a, _ = await joined(project.pk, ada)

        def delete():
            owner = APIClient()
            owner.force_authenticate(ada)
            owner.delete(f"/api/projects/{project.pk}/")

        await database_sync_to_async(delete)()
        await next_of(a, "deleted")

    async_to_sync(run)()


@pytestmark_db
def test_comments_and_history_reach_live_editors(setup):
    project, ada, bob, _ = setup

    async def run():
        b, _ = await joined(project.pk, bob)

        def comment():
            client = APIClient()
            client.force_authenticate(ada)
            client.post(
                f"/api/projects/{project.pk}/comments/",
                {"text": "Hi", "node_id": "n1"},
                format="json",
            )

        from channels.db import database_sync_to_async

        await database_sync_to_async(comment)()
        message = await next_of(b, "comment")
        assert message["action"] == "created" and message["comment"]["text"] == "Hi"

        await b.send_json_to(
            {"type": "ops", "id": "m", "ops": [{"op": "set_meta", "name": "Changed"}]}
        )
        await next_of(b, "history")
        await b.disconnect()

    async_to_sync(run)()
    assert project.history.first().name == "Changed"


@pytestmark_db
def test_newcomers_see_presence_and_history_is_announced_once(setup):
    project, ada, bob, cy = setup

    async def run():
        a, _ = await joined(project.pk, ada)
        await a.send_json_to({"type": "presence", "state": {"selected": ["n1"]}})
        # Someone joining later learns what ada has selected, without ada sending it again.
        c, _ = await joined(project.pk, cy)
        presence = await next_of(c, "presence")
        assert presence["user"]["username"] == "ada" and presence["state"] == {"selected": ["n1"]}

        # A new history entry is announced; further changes by the same person in the same
        # session only update that entry and are not announced again.
        await a.send_json_to({"type": "ops", "id": "1", "ops": [{"op": "set_meta", "name": "A"}]})
        await next_of(c, "history")
        await a.send_json_to({"type": "ops", "id": "2", "ops": [{"op": "set_meta", "name": "B"}]})
        assert (await next_of(c, "ops"))["id"] == "1"
        assert (await next_of(c, "ops"))["id"] == "2"
        assert await c.receive_nothing(timeout=0.5)
        await a.disconnect()
        await c.disconnect()

    async_to_sync(run)()
