"""Approvals: rules per API token and app, calls that wait for the user, deciding them in
the web app, the server doing them as the agent, and the REST API refusing what it cannot
ask for."""

# ruff: noqa: F811 (fixtures are imported from other test modules)

from __future__ import annotations

import json
from datetime import timedelta

import pytest
from django.contrib.auth import get_user_model
from django.utils import timezone
from rest_framework.test import APIClient

from brick_dnn_core import approvals, budgets
from brick_dnn_core.models import (
    AgentAction,
    ApprovalRequest,
    ApprovalRule,
    DataPrep,
    Dataset,
    DataSource,
    Project,
    ProjectMember,
    TrainingRun,
)

from .test_budgets import (  # noqa: F401 (fixtures)
    app_client,
    controlled,
    limits,
    spend,
    token_client,
)
from .test_mcp import call
from .test_training_api import (  # noqa: F401 (fixtures)
    DEMO_GRAPH,
    QUICK,
    RANDOM,
    iris_csv,
    training_dir,
    wait_for,
)
from .test_training_queue import make_worker, queued, step_until  # noqa: F401

pytestmark = pytest.mark.django_db
User = get_user_model()

ALWAYS = approvals.Rule(training_cpu_minutes=0, publishing=True, deleting=True)
IRIS = iris_csv().decode()


@pytest.fixture
def ada(limits):
    return User.objects.create_user("ada", password="correct-horse-battery")


@pytest.fixture
def web(ada):
    client = APIClient()
    client.force_authenticate(ada)
    return client


def agent(user, rule: approvals.Rule | None = ALWAYS, name="agent"):
    client, token = token_client(user, name)
    if rule is not None:
        approvals.set_rule(user, rule, token=token)
    return client, token


def ok(client, tool, **arguments) -> dict:
    result, text = call(client, tool, **arguments)
    assert "isError" not in result, text
    return json.loads(text)


def error_of(client, tool, **arguments) -> str:
    result, text = call(client, tool, **arguments)
    assert result.get("isError") is True, text
    return text


def waiting(client, tool, **arguments) -> dict:
    """The answer of a call that waits for approval."""
    answer = ok(client, tool, **arguments)
    assert answer["approval_required"] is True, answer
    assert answer["approval"]["status"] == "pending"
    assert "Waiting for the user's approval" in answer["message"]
    assert f"get_approval with id {answer['approval']['id']}" in answer["message"]
    return answer["approval"]


def approve(web, approval_id, status=200) -> dict:
    response = web.post(f"/api/approvals/{approval_id}/approve/")
    assert response.status_code == status, response.data
    return response.data


def log(user, approval_id=None):
    entries = AgentAction.objects.filter(owner=user).order_by("id")
    if approval_id is not None:
        entries = entries.filter(approval_id=approval_id)
    return list(entries)


def start_args(**extra):
    return {"graph": DEMO_GRAPH, "dataset": RANDOM, "config": QUICK, "name": "Big", **extra}


# --------------------------------------------------------------------------- #
# Rules
# --------------------------------------------------------------------------- #


def test_new_tokens_and_apps_get_the_account_default(ada, web, settings):
    settings.PUBLIC_URL = ""
    old, old_token = token_client(ada, "from before")  # no rule: never asks
    rules = web.get("/api/approvals/rules/").data
    assert rules["default"] == approvals.BUILTIN_DEFAULT.to_dict()
    assert rules["default"] == {
        "training_cpu_minutes": 15.0,
        "publishing": True,
        "deleting": False,
    }
    assert rules["agents"][f"token:{old_token.pk}"] == approvals.OFF.to_dict()
    assert "get_weights_download" in rules["gated_tools"]["publishing"]

    created = web.post("/api/auth/api-tokens/", {"name": "new"}, format="json").data
    assert ApprovalRule.objects.get(api_token_id=created["id"]).publishing is True

    mine = {"training_cpu_minutes": 2.5, "publishing": False, "deleting": True}
    assert web.put("/api/approvals/rules/default/", mine, format="json").status_code == 200
    second = web.post("/api/auth/api-tokens/", {"name": "second"}, format="json").data
    rules = web.get("/api/approvals/rules/").data
    assert rules["default"] == mine
    assert rules["agents"][f"token:{second['id']}"] == mine
    assert rules["agents"][f"token:{created['id']}"]["publishing"] is True  # kept
    # A connected app (OAuth sign-in) gets the default too.
    _, app_connection = app_client(ada)
    approvals.adopt_default(ada, connection=app_connection)
    assert web.get("/api/approvals/rules/").data["agents"][f"app:{app_connection.pk}"] == mine

    # Per token and app.
    off = {"training_cpu_minutes": None, "publishing": False, "deleting": False}
    url = f"/api/approvals/rules/token/{second['id']}/"
    assert web.put(url, off, format="json").data == off
    assert approvals.rule_of(old_token) == approvals.OFF


def test_rules_are_checked_and_only_for_the_website_login(ada, web):
    other = User.objects.create_user("bob", password="correct-horse-battery")
    _, bobs = token_client(other, "bob's")
    assert web.put(f"/api/approvals/rules/token/{bobs.pk}/", {}, format="json").status_code == 404
    assert web.put("/api/approvals/rules/app/999/", {}, format="json").status_code == 404
    for bad in (
        {"training_cpu_minutes": -1},
        {"training_cpu_minutes": "ten"},
        {"training_cpu_minutes": True},
        {"publishing": "yes"},
    ):
        response = web.put("/api/approvals/rules/default/", bad, format="json")
        assert response.status_code == 400, bad
    client, _ = agent(ada)
    for path in ("/api/approvals/", "/api/approvals/rules/"):
        assert client.get(path).status_code == 403
    assert client.put("/api/approvals/rules/default/", {}, format="json").status_code == 403


def test_the_app_of_an_oauth_sign_in_gets_the_default(limits, settings):
    from .test_oauth import sign_in

    settings.PUBLIC_URL = "http://testserver"
    user = User.objects.create_user("ada", password="correct-horse-battery")
    approvals.set_rule(user, approvals.Rule(5.0, False, True))
    browser = APIClient()
    login = browser.post(
        "/api/auth/token/",
        {"username": "ada", "password": "correct-horse-battery"},
        format="json",
    )
    browser.credentials(HTTP_AUTHORIZATION=f"Bearer {login.data['access']}")
    sign_in(browser)
    rule = ApprovalRule.objects.get(owner=user, connection__isnull=False)
    assert (rule.training_cpu_minutes, rule.publishing, rule.deleting) == (5.0, False, True)


# --------------------------------------------------------------------------- #
# Trainings
# --------------------------------------------------------------------------- #


def test_a_training_above_the_limit_waits_and_runs_once_approved(ada, web, controlled):
    client, token = agent(ada)
    request = waiting(client, "start_training", **start_args())
    assert not TrainingRun.objects.exists()
    stored = ApprovalRequest.objects.get(pk=request["id"])
    assert stored.arguments == start_args()  # all of it, to start it later
    assert stored.category == "training" and stored.action == "start_training"
    assert stored.details["estimate"]["cpu_minutes"] > 0
    assert stored.details["network"].startswith("4 layers")
    assert "Every training needs approval" in stored.details["reason"]
    assert stored.expires_at - stored.created_at == timedelta(hours=approvals.EXPIRY_HOURS)

    # The same call again finds the same request.
    assert waiting(client, "start_training", **start_args())["id"] == request["id"]
    assert ApprovalRequest.objects.count() == 1
    assert ok(client, "get_approval", id=request["id"])["status"] == "pending"
    listed = ok(client, "list_approvals", status="pending")
    assert [a["id"] for a in listed["approvals"]] == [request["id"]]

    pending = web.get("/api/agent-activity/").data["approvals"]
    assert pending["pending_count"] == 1
    shown = pending["pending"][0]
    assert shown["summary"] == "Start training “Big” on random demo data"
    assert shown["agent"] == {
        "kind": "token",
        "id": token.pk,
        "key": f"token:{token.pk}",
        "name": "agent",
        "active": True,
    }
    assert shown["estimate"]["cpu_minutes"] == stored.details["estimate"]["cpu_minutes"]
    assert shown["arguments"]["graph"].startswith("4 layers")  # summarised for the page

    done = approve(web, request["id"])
    assert done["status"] == "executed"
    run = TrainingRun.objects.get()
    assert run.api_token == token and run.name == "Big"  # started as the agent
    answer = ok(client, "get_approval", id=request["id"])
    assert answer["status"] == "executed" and answer["result"]["id"] == str(run.pk)
    assert "result is what the tool returned" in answer["next"]

    # Approving again does nothing again.
    assert approve(web, request["id"])["status"] == "executed"
    assert TrainingRun.objects.count() == 1
    assert ApprovalRequest.objects.get(pk=request["id"]).status == "executed"

    entries = log(ada, request["id"])
    assert [e.outcome for e in entries] == ["pending", "pending", "approved"]
    assert entries[0].summary == (
        f"Asked for approval to start training “Big” on random demo data (request #{request['id']})"
    )
    assert entries[-1].summary.startswith("Started training “Big” on random demo data")
    assert entries[-1].summary.endswith(f"approved by you (request #{request['id']})")
    assert {"kind": "run", "id": str(run.pk), "name": "Big"} in entries[-1].targets
    assert entries[-1].agent_id == token.pk and entries[-1].agent_kind == "token"
    assert web.get("/api/agent-activity/").data["approvals"]["pending_count"] == 0
    (run_dir := controlled / "runs" / str(run.pk) / "finish").parent.mkdir(exist_ok=True)
    run_dir.touch()


def test_trainings_below_the_limit_start_right_away(ada, controlled):
    client, _ = agent(ada, approvals.Rule(training_cpu_minutes=10_000))
    run = ok(client, "start_training", **start_args())
    assert run["status"] == "starting" and "approval_required" not in run
    assert not ApprovalRequest.objects.exists()
    # Without a rule nothing waits either; invalid calls fail as before.
    other, _ = agent(ada, None, name="old")
    assert "approval" not in error_of(other, "start_training", graph=DEMO_GRAPH, dataset={})
    strict, _ = agent(ada, name="strict")
    assert "Unknown dataset type" in error_of(
        strict, "start_training", graph=DEMO_GRAPH, dataset={}
    )
    assert not ApprovalRequest.objects.exists()


def test_the_budgets_are_checked_again_when_approved(ada, web, controlled):
    client, token = agent(ada)
    request = waiting(client, "start_training", **start_args())
    token.budget_cpu_minutes = 1
    token.save()
    spend(ada, 120, api_token=token)  # the token's budget is used up meanwhile
    done = approve(web, request["id"])
    assert done["status"] == "failed"
    assert "budget" in done["error"] and "get_usage" in done["error"]
    assert not TrainingRun.objects.exists()
    answer = ok(client, "get_approval", id=request["id"])
    assert answer["status"] == "failed" and answer["error"] == done["error"]
    (failed,) = [e for e in log(ada, request["id"]) if e.outcome == "error"]
    assert failed.summary.startswith("Could not start training “Big”")
    # A budget that is used up already is said at once, without asking.
    assert "budget" in error_of(client, "start_training", **start_args(name="Another"))
    assert ApprovalRequest.objects.count() == 1


def test_an_approved_training_waits_in_the_queue(ada, web, make_worker):
    client, token = agent(ada)
    request = waiting(client, "start_training", **start_args())
    worker = make_worker()
    assert approve(web, request["id"])["status"] == "executed"
    run = TrainingRun.objects.get()
    assert run.status == "queued" and run.api_token == token
    answer = ok(client, "get_approval", id=request["id"])
    assert answer["result"]["status"] == "queued"
    step_until(worker, lambda: TrainingRun.objects.get().status == "running")
    entry = log(ada, request["id"])[-1]
    assert entry.outcome == "approved" and entry.summary.startswith("Queued training “Big”")


def test_apps_wait_too_and_act_as_the_app(ada, web, controlled):
    client, connection = app_client(ada)
    approvals.set_rule(ada, ALWAYS, connection=connection)
    request = waiting(client, "start_training", **start_args())
    stored = ApprovalRequest.objects.get()
    assert (stored.agent_kind, stored.agent_id, stored.agent_name) == (
        "app",
        connection.pk,
        "Claude",
    )
    assert approve(web, request["id"])["status"] == "executed"
    run = TrainingRun.objects.get()
    assert run.connection == connection and run.api_token is None
    assert ok(client, "get_approval", id=request["id"])["result"]["id"] == str(run.pk)


def test_a_revoked_token_cannot_act_any_more(ada, web, controlled):
    client, token = agent(ada)
    request = waiting(client, "start_training", **start_args())
    token.delete()
    done = approve(web, request["id"])
    assert done["status"] == "failed" and "was revoked" in done["error"]
    assert done["agent"]["active"] is False
    assert not TrainingRun.objects.exists()


# --------------------------------------------------------------------------- #
# Denying, expiry, permissions
# --------------------------------------------------------------------------- #


def test_deny_with_a_message(ada, web, controlled):
    client, _ = agent(ada)
    request = waiting(client, "start_training", **start_args())
    response = web.post(
        f"/api/approvals/{request['id']}/deny/", {"message": "Too big, use 5 epochs"}, format="json"
    )
    assert response.status_code == 200 and response.data["status"] == "denied"
    answer = ok(client, "get_approval", id=request["id"])
    assert answer["status"] == "denied" and answer["message"] == "Too big, use 5 epochs"
    assert "result" not in answer
    # Denied is final.
    response = approve(web, request["id"], status=409)
    assert response["detail"] == "You denied this request already."
    assert web.post(f"/api/approvals/{request['id']}/deny/").status_code == 200
    assert not TrainingRun.objects.exists()
    entry = log(ada, request["id"])[-1]
    assert entry.outcome == "denied" and entry.error == "Too big, use 5 epochs"
    assert (
        entry.summary
        == f"Denied: start training “Big” on random demo data (request #{request['id']})"
    )
    # Waiting again is a new request.
    assert waiting(client, "start_training", **start_args())["id"] != request["id"]


def test_requests_expire(ada, web, controlled):
    client, _ = agent(ada)
    request = waiting(client, "start_training", **start_args())
    ApprovalRequest.objects.filter(pk=request["id"]).update(
        expires_at=timezone.now() - timedelta(seconds=1)
    )
    assert ok(client, "get_approval", id=request["id"])["status"] == "expired"
    response = approve(web, request["id"], status=409)
    assert "expired" in response["detail"]
    assert web.get("/api/approvals/?status=pending").data["pending_count"] == 0
    assert not TrainingRun.objects.exists()


def test_only_the_owner_decides_and_agents_see_only_their_own(ada, web, controlled):
    client, _ = agent(ada)
    request = waiting(client, "start_training", **start_args())
    # Another agent of the same user cannot see it.
    other, _ = agent(ada, name="other")
    assert "No such approval request" in error_of(other, "get_approval", id=request["id"])
    assert ok(other, "list_approvals")["total"] == 0
    # Neither tokens nor other users can decide it, and MCP has no tool for it.
    for path in ("approve", "deny"):
        assert client.post(f"/api/approvals/{request['id']}/{path}/").status_code == 403
    bob = APIClient()
    bob.force_authenticate(User.objects.create_user("bob", password="correct-horse-battery"))
    assert bob.post(f"/api/approvals/{request['id']}/approve/").status_code == 404
    assert bob.get("/api/approvals/").data["results"] == []
    tools = client.post(
        "/api/mcp/",
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        format="json",
    ).json()["result"]["tools"]
    names = {tool["name"] for tool in tools}
    assert {"get_approval", "list_approvals"} <= names
    assert not any("approve" in name for name in names)
    assert ApprovalRequest.objects.get().status == "pending"


def test_a_flood_of_requests_is_refused(ada, controlled, monkeypatch):
    monkeypatch.setattr(approvals, "MAX_PENDING_PER_AGENT", 2)
    client, _ = agent(ada)
    waiting(client, "start_training", **start_args(name="one"))
    waiting(client, "start_training", **start_args(name="two"))
    assert "already wait" in error_of(client, "start_training", **start_args(name="three"))


def test_notify_hook_is_called(ada, controlled, monkeypatch):
    seen = []
    monkeypatch.setattr(approvals, "notify_approval_requested", seen.append)
    client, _ = agent(ada)
    request = waiting(client, "start_training", **start_args())
    waiting(client, "start_training", **start_args())  # the same one: no second notice
    assert [approval.pk for approval in seen] == [request["id"]]


# --------------------------------------------------------------------------- #
# Publishing
# --------------------------------------------------------------------------- #


@pytest.fixture
def finished(ada, web, training_dir):
    """A completed run of ada's, with weights."""
    response = web.post(
        "/api/training/runs/",
        {"name": "Done", "graph": DEMO_GRAPH, "dataset": RANDOM, "config": QUICK},
        format="json",
    )
    assert response.status_code == 201, response.data
    return wait_for(web, response.data["id"])


@pytest.fixture
def team(ada):
    project = Project.objects.create(owner=ada, name="Team", graph={}, updated_by=ada)
    bob = User.objects.create_user("bob", password="correct-horse-battery")
    ProjectMember.objects.create(project=project, user=bob, role="viewer")
    return project


def test_weights_links_wait_and_are_made_once_approved(ada, web, finished, settings):
    settings.PUBLIC_URL = ""
    client, _ = agent(ada, approvals.Rule(publishing=True))
    request = waiting(client, "get_weights_download", id=finished["id"])
    stored = ApprovalRequest.objects.get()
    assert stored.category == "publishing" and stored.arguments == {"id": finished["id"]}
    assert "without logging in" in stored.details["exposes"]
    done = approve(web, request["id"])
    assert done["status"] == "executed"
    answer = ok(client, "get_approval", id=request["id"])
    assert answer["result"]["url"].startswith("http://testserver/api/training/runs/")
    assert "sig=" in answer["result"]["url"]
    # The link works without logging in.
    assert APIClient().get(answer["result"]["url"]).status_code == 200

    second = waiting(client, "get_weights_download", id=finished["id"])
    web.post(f"/api/approvals/{second['id']}/deny/", {"message": "Not public"}, format="json")
    answer = ok(client, "get_approval", id=second["id"])
    assert answer["status"] == "denied" and answer["message"] == "Not public"
    assert "url" not in json.dumps(answer)


def test_uploads_are_kept_private_until_sharing_is_approved(ada, web, team, training_dir):
    client, _ = agent(ada, approvals.Rule(publishing=True))
    answer = ok(client, "upload_dataset", name="Iris", csv=IRIS, project_id=team.pk)
    dataset = Dataset.objects.get()
    assert answer["dataset"]["id"] == dataset.pk  # uploaded right away …
    assert not dataset.projects.exists()  # … but not shared yet
    request = answer["approval"]
    assert request["status"] == "pending" and request["action"] == "share_dataset"
    assert f"request #{request['id']}" in answer["approval_note"]
    stored = ApprovalRequest.objects.get()
    assert stored.arguments == {"dataset_id": dataset.pk, "project_id": team.pk}
    assert "setosa" not in json.dumps(stored.arguments)  # no file contents
    assert stored.summary == f"share dataset “Iris” (#{dataset.pk}) in network “Team” (#{team.pk})"
    assert "1 other person" in stored.details["exposes"]
    (entry,) = log(ada)
    assert entry.outcome == "pending" and entry.approval_id == stored.pk
    assert "project_id" not in entry.arguments
    assert entry.summary.startswith(f"Uploaded dataset “Iris” (#{dataset.pk})")

    assert approve(web, request["id"])["status"] == "executed"
    assert list(dataset.projects.all()) == [team]
    result = ok(client, "get_approval", id=request["id"])["result"]
    assert result["shared"] is True and result["project"]["id"] == team.pk


def test_data_preparations_are_shared_once_approved(ada, web, team, settings, tmp_path):
    settings.DATA_PREP_DIR = tmp_path / "prepare"
    settings.DATA_PREP_ENABLED = True
    settings.TRAINING_DIR = tmp_path / "training"
    client, _ = agent(ada, approvals.Rule(publishing=True))
    source = ok(client, "import_data", name="Sales", content="a,b\n1,2\n3,4\n5,6\n")["source"]
    answer = ok(client, "create_data_prep", source_id=source["id"], project_id=team.pk)
    prep = DataPrep.objects.get()
    assert answer["preparation"]["id"] == prep.pk and not prep.projects.exists()
    assert answer["approval"]["action"] == "share_data_prep"
    assert approve(web, answer["approval"]["id"])["status"] == "executed"
    assert list(prep.projects.all()) == [team]
    (entry,) = [e for e in log(ada) if e.tool == "create_data_prep" and e.outcome == "pending"]
    assert entry.summary.startswith(f"Created data preparation “Sales” (#{prep.pk}); to share")


def test_upload_links_into_a_project_wait(ada, web, team, training_dir, settings):
    settings.PUBLIC_URL = ""
    client, _ = agent(ada, approvals.Rule(publishing=True))
    request = waiting(
        client,
        "create_upload_link",
        purpose="training_dataset",
        filename="iris.csv",
        project_id=team.pk,
    )
    # Without a project nothing waits.
    plain = ok(client, "create_upload_link", purpose="training_dataset", filename="iris.csv")
    assert "url" in plain
    assert approve(web, request["id"])["status"] == "executed"
    assert "sig=" in ok(client, "get_approval", id=request["id"])["result"]["url"]


# --------------------------------------------------------------------------- #
# Deleting
# --------------------------------------------------------------------------- #


def test_deleting_waits_when_switched_on(ada, web, finished, settings, tmp_path):
    settings.DATA_PREP_DIR = tmp_path / "prepare"
    settings.DATA_PREP_ENABLED = True
    client, _ = agent(ada, approvals.Rule(deleting=True))
    project = Project.objects.create(owner=ada, name="Old", graph={}, updated_by=ada)
    dataset_id = ok(client, "upload_dataset", name="Iris", csv=IRIS)["dataset"]["id"]
    source = ok(client, "import_data", name="Sales", content="a,b\n1,2\n3,4\n")["source"]
    unused = ok(client, "import_data", name="Unused", content="a,b\n1,2\n3,4\n")["source"]
    prep = ok(client, "create_data_prep", source_id=source["id"])["preparation"]

    requests = {
        "delete_project": waiting(client, "delete_project", id=project.pk),
        "delete_training_run": waiting(client, "delete_training_run", id=finished["id"]),
        "delete_dataset": waiting(client, "delete_dataset", id=dataset_id),
        "delete_data_prep": waiting(client, "delete_data_prep", id=prep["id"]),
        "delete_data_source": waiting(client, "delete_data_source", id=unused["id"]),
    }
    assert Project.objects.filter(pk=project.pk).exists()
    assert TrainingRun.objects.filter(pk=finished["id"]).exists()
    stored = ApprovalRequest.objects.get(pk=requests["delete_project"]["id"])
    assert (
        stored.category == "deleting" and stored.summary == f"delete network “Old” (#{project.pk})"
    )
    assert stored.details["warning"] == approvals.DELETE_WARNING
    # A source that a preparation uses cannot be deleted anyway: no request, the error.
    assert "Used by" in error_of(client, "delete_data_source", id=source["id"])

    for request in requests.values():
        assert approve(web, request["id"])["status"] == "executed", request
    assert not Project.objects.filter(pk=project.pk).exists()
    assert not TrainingRun.objects.exists()
    assert not Dataset.objects.filter(pk=dataset_id).exists()
    assert not DataPrep.objects.exists()
    assert not DataSource.objects.filter(pk=unused["id"]).exists()
    answer = ok(client, "get_approval", id=requests["delete_project"]["id"])
    assert answer["result"]["deleted"] is True
    entry = log(ada, requests["delete_project"]["id"])[-1]
    assert entry.summary.startswith(f"Deleted network “Old” (#{project.pk})")


# --------------------------------------------------------------------------- #
# The REST API
# --------------------------------------------------------------------------- #


def test_the_rest_api_refuses_what_needs_approval(ada, web, team, controlled):
    client, token = agent(ada)
    response = client.post(f"/api/projects/{team.pk}/link/", {"enabled": True}, format="json")
    assert response.status_code == 403 and "approval" in response.data["detail"]
    response = client.post(
        f"/api/projects/{team.pk}/members/", {"user": "bob", "role": "editor"}, format="json"
    )
    assert response.status_code == 403
    assert client.delete(f"/api/projects/{team.pk}/").status_code == 403
    response = client.post("/api/training/runs/", start_args(), format="json")
    assert response.status_code == 403 and "MCP" in response.data["detail"]
    assert Project.objects.filter(pk=team.pk, link_access="none").exists()
    assert not TrainingRun.objects.exists()

    # Tokens without a rule (or with the switches off) work as before.
    approvals.set_rule(ada, approvals.OFF, token=token)
    response = client.post(f"/api/projects/{team.pk}/link/", {"enabled": True}, format="json")
    assert response.status_code == 200
    assert client.post("/api/training/runs/", start_args(), format="json").status_code == 201


def test_purge_deletes_old_decided_requests(ada, web, controlled):
    client, _ = agent(ada)
    old = waiting(client, "start_training", **start_args(name="old"))
    new = waiting(client, "start_training", **start_args(name="new"))
    web.post(f"/api/approvals/{old['id']}/deny/")
    long_ago = timezone.now() - timedelta(days=approvals.KEEP_DAYS + 1)
    ApprovalRequest.objects.update(created_at=long_ago, expires_at=long_ago)
    assert approvals.purge(timezone.now()) == 2  # denied and expired
    assert not ApprovalRequest.objects.filter(pk__in=[old["id"], new["id"]]).exists()


def test_budget_actor_of_the_approved_request_is_the_agent(ada, web, controlled):
    """Usage of an approved training counts for the token that asked."""
    client, token = agent(ada)
    request = waiting(client, "start_training", **start_args())
    approve(web, request["id"])
    run = TrainingRun.objects.get()
    assert budgets.Actor.of_run(run).token == token


def test_the_account_export_holds_rules_and_requests(ada, web, controlled):
    from .test_account import export, read_json

    client, token = agent(ada)
    request = waiting(client, "start_training", **start_args())
    data = read_json(export(web), "approvals.json")
    assert data["rules"] == [{"applies_to": f"API token #{token.pk}", **ALWAYS.to_dict()}]
    (exported,) = data["requests"]
    assert exported["id"] == request["id"] and exported["arguments"] == start_args()
