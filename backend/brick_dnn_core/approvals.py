"""Approvals: expensive or publishing actions of AI agents wait for the user's OK.

A user sets per API token and per connected app what the agent may do on its own
(:class:`~brick_dnn_core.models.ApprovalRule`):

- **trainings**: start freely up to N estimated CPU minutes, above that ask (0: ask before
  every training; empty: never ask). The estimate is the one check_training shows
  (``estimate.cpu_minutes``, the middle of its range);
- **publishing**: ask before something becomes visible to others or downloadable without
  logging in: a link to the trained weights (get_weights_download), a dataset or a data
  preparation shared in a project (upload_dataset, create_data_prep and create_upload_link
  with ``project_id``);
- **deleting**: ask before deleting a network, a training run, a dataset, a data file or a
  data preparation.

The account has a default rule that new tokens and apps get a copy of (without one of its
own, :data:`BUILTIN_DEFAULT`). Tokens and apps that existed before approvals have no rule and
never ask.

**Flow.** When an agent calls a gated MCP tool (``hold``, called by mcp.py before the tool
runs), nothing is done: an :class:`~brick_dnn_core.models.ApprovalRequest` keeps the validated
arguments (never file contents: an upload is stored for the user alone right away, only the
sharing waits) and the agent gets a structured answer with the request's id. It looks at the
request with the tool get_approval. The user approves or denies it in the web app (page
"Agent activity"); once approved, the server does it right away as that agent: with its
token or app, its budgets checked at that moment, the same checks as the tool. The result
(e.g. the run) is stored on the request for get_approval. Requests expire after
:data:`EXPIRY_HOURS` hours. Every step is in the activity log (agent_activity.py).

The REST API cannot ask: requests made with a token whose rule covers them are refused there
(``refuse_rest``, ``refuse_rest_training``) with a message that points to MCP or the web app.

``notify_approval_requested`` is the place to tell the owner by other means (e-mail) later.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from django.conf import settings
from django.db import IntegrityError, transaction
from django.utils import timezone
from rest_framework import status
from rest_framework.exceptions import NotFound, PermissionDenied, ValidationError
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from . import agent_activity as activity
from . import budgets
from .api_tokens import IsLoggedInUser
from .models import (
    ApiToken,
    ApprovalRequest,
    ApprovalRule,
    DataPrep,
    Dataset,
    DataSource,
    OAuthConnection,
    Project,
    ProjectMember,
)

logger = logging.getLogger(__name__)

EXPIRY_HOURS = 24
# Requests of one token or app that may wait at the same time.
MAX_PENDING_PER_AGENT = 20
# Decided and expired requests are deleted after this many days (purge_old_data).
KEEP_DAYS = 90
MAX_MESSAGE = 500
MAX_CPU_MINUTES = 10_000_000
LIST_LIMIT = 50


# --------------------------------------------------------------------------- #
# Rules
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Rule:
    """What needs approval. ``training_cpu_minutes``: ask for trainings estimated above it
    (0: every training; None: never)."""

    training_cpu_minutes: float | None = None
    publishing: bool = False
    deleting: bool = False

    @classmethod
    def of(cls, row: ApprovalRule | None) -> Rule:
        if row is None:
            return OFF
        return cls(row.training_cpu_minutes, row.publishing, row.deleting)

    def to_dict(self) -> dict[str, Any]:
        return {
            "training_cpu_minutes": self.training_cpu_minutes,
            "publishing": self.publishing,
            "deleting": self.deleting,
        }

    def fields(self) -> dict[str, Any]:
        return self.to_dict()


OFF = Rule()
# New tokens and apps of an account without a default of its own: trainings above
# 15 estimated CPU minutes and publishing need approval, deleting does not.
BUILTIN_DEFAULT = Rule(training_cpu_minutes=15.0, publishing=True, deleting=False)


def default_rule(user) -> Rule:
    row = ApprovalRule.objects.filter(
        owner=user, api_token__isnull=True, connection__isnull=True
    ).first()
    return Rule.of(row) if row is not None else BUILTIN_DEFAULT


def _rule_row(user, token: ApiToken | None = None, connection: OAuthConnection | None = None):
    rules = ApprovalRule.objects.filter(owner=user)
    if token is not None:
        return rules.filter(api_token=token).first()
    if connection is not None:
        return rules.filter(connection=connection).first()
    return rules.filter(api_token__isnull=True, connection__isnull=True).first()


def set_rule(
    user,
    rule: Rule,
    token: ApiToken | None = None,
    connection: OAuthConnection | None = None,
) -> Rule:
    """Store the rule of ``token``, ``connection`` or (neither) the account's default."""
    for _attempt in range(2):
        row = _rule_row(user, token, connection)
        if row is not None:
            for name, value in rule.fields().items():
                setattr(row, name, value)
            row.save()
            return rule
        try:
            with transaction.atomic():
                ApprovalRule.objects.create(
                    owner=user, api_token=token, connection=connection, **rule.fields()
                )
            return rule
        except IntegrityError:  # created at the same time by another request
            continue
    return rule


def adopt_default(user, token: ApiToken | None = None, connection=None) -> None:
    """Give a new token or app the account's default rule (called when it is created)."""
    try:
        set_rule(user, default_rule(user), token=token, connection=connection)
    except Exception:
        logger.exception("Could not give a new token or app the default approval rule")


def rule_of(token: ApiToken) -> Rule:
    """The rule of the token or app behind ``token`` (an app's access token: its app's)."""
    if token.connection_id is not None:
        row = ApprovalRule.objects.filter(connection_id=token.connection_id).first()
    elif token.pk is not None:
        row = ApprovalRule.objects.filter(api_token_id=token.pk).first()
    else:
        row = None
    return Rule.of(row)


def _who(token: ApiToken) -> str:
    return "this app" if token.connection_id is not None else "this API token"


# --------------------------------------------------------------------------- #
# Which calls need approval
# --------------------------------------------------------------------------- #


@dataclass
class Gate:
    """A call that needs approval: what the server will do once approved."""

    category: str  # training | publishing | deleting
    action: str  # a tool, or share_dataset / share_data_prep
    arguments: dict[str, Any]
    phrase: str  # infinitive: 'start training “GP 1” on random demo data'
    targets: list[dict[str, Any]]
    reason: str
    details: dict[str, Any] = field(default_factory=dict)
    # The tool runs first with these arguments (e.g. an upload without sharing), then
    # ``follow(result)`` gives the action's (arguments, phrase, targets, details), or None.
    first: dict[str, Any] | None = None
    follow: Callable[[Any], tuple[dict, str, list, dict] | None] | None = None


def _int(value: Any) -> int | None:
    if isinstance(value, str) and value.isdigit() and len(value) < 18:
        value = int(value)
    if isinstance(value, int) and not isinstance(value, bool) and 0 < value < 2**63:
        return value
    return None


def _people(project: Project, user) -> str:
    """Who can see what is shared in ``project``: '3 other people now and anyone added
    later'."""
    others = ProjectMember.objects.filter(project=project).exclude(user=user).count()
    if project.owner_id != user.pk:
        others += 1
    if others == 0:
        return "nobody else yet, but anyone added later"
    return f"{activity.count(others, 'other person', 'other people')} now and anyone added later"


def _editable_project(user, project_id: Any) -> Project | None:
    from .shared_data import editable_projects

    number = _int(project_id)
    if number is None or not isinstance(project_id, int):
        return None
    return editable_projects(user).filter(pk=number).first()


def _training_check(request, args: dict[str, Any]) -> dict[str, Any] | None:
    """The training check of a start (estimate, budget), or None when the start would be
    refused anyway: then the tool runs and says why."""
    from . import training_service as training

    if not settings.TRAINING_ENABLED:
        return None
    user = request.user
    graph = args.get("graph")
    name = args.get("name")
    if name is not None and (not isinstance(name, str) or len(name.strip()) > 120):
        return None
    if args.get("project_id") is not None and _editable_project(user, args["project_id"]) is None:
        return None
    try:
        training.validate_graph(graph)
        if training.unavailable_reason(training.graph_family(graph)):
            return None
        dataset = training.resolve_dataset(user, args.get("dataset"))
        config = training.parse_config(args.get("config"))
        search = training.parse_tuning(args.get("tuning"), graph)
        result = training.check(graph, dataset, config, search)
        if not result.ok or result.plan is None:
            return None
        data = training.budget_check(
            budgets.Actor.of(request), graph, dataset, config, result, search
        )
    except Exception:  # invalid input: start_training explains it
        logger.debug("No approval check for an invalid training", exc_info=True)
        return None
    if not (data.get("budget") or {}).get("allowed", True):
        return None  # refused by a budget: start_training says which
    return data


def _training_needs_approval(rule: Rule, check: dict[str, Any]) -> tuple[bool, str]:
    limit = rule.training_cpu_minutes
    assert limit is not None
    estimate = check.get("estimate")
    minutes = estimate.get("cpu_minutes") if isinstance(estimate, dict) else None
    if minutes is None:
        return True, "The CPU time of this training cannot be estimated."
    if minutes <= limit:
        return False, ""
    if limit == 0:
        return True, f"Every training needs approval (estimated {_minutes(minutes)})."
    return True, (
        f"The estimated CPU time ({_minutes(minutes)}, roughly "
        f"{_minutes(estimate.get('low_cpu_minutes'))} to "
        f"{_minutes(estimate.get('high_cpu_minutes'))}) is above the "
        f"{_minutes(limit)} that trainings may use without approval."
    )


def _minutes(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "an unknown number of CPU minutes"
    shown = f"{value:.2f}".rstrip("0").rstrip(".") if value < 10 else f"{value:,.0f}"
    return f"{shown} CPU min"


def _gate_training(request, args: dict[str, Any], rule: Rule) -> Gate | None:
    if rule.training_cpu_minutes is None:
        return None
    check = _training_check(request, args)
    if check is None:
        return None
    needed, reason = _training_needs_approval(rule, check)
    if not needed:
        return None
    user = request.user
    described = activity.describe_tool(user, "start_training", args, None)
    estimate = check.get("estimate") or None
    if estimate:
        estimate = {
            key: estimate.get(key)
            for key in ("cpu_minutes", "low_cpu_minutes", "high_cpu_minutes", "cost", "currency")
        }
    _, dataset_words = activity.training_data(user, args.get("dataset"))
    budget = check.get("budget") or {}
    details = {
        "estimate": estimate,
        "limit_cpu_minutes": rule.training_cpu_minutes,
        "network": activity.graph_summary(args.get("graph")),
        "dataset": dataset_words,
        "remaining_cpu_minutes": budget.get("remaining_cpu_minutes"),
        "warning": budget.get("warning"),
    }
    return Gate(
        "training",
        "start_training",
        _plain(args),
        described.phrase,
        described.targets,
        reason,
        details,
    )


def _gate_weights(request, args: dict[str, Any], rule: Rule) -> Gate | None:
    from . import training_service as training
    from .downloads import LINK_SECONDS
    from .shared_data import visible_runs

    if not rule.publishing:
        return None
    try:
        run = visible_runs(request.user).filter(pk=args.get("id")).first()
    except Exception:
        run = None
    if run is None or training.weights_path(run) is None:
        return None
    ref = activity.run_ref(request.user, str(run.pk), run.name)
    what = activity.phrase_of(ref, "run")
    return Gate(
        "publishing",
        "get_weights_download",
        {"id": str(run.pk)},
        f"get a download link for the weights of {what}",
        [ref] if ref else [],
        "Links that work without logging in need approval.",
        {
            "exposes": (
                f"A link to the trained weights of {what} that works without logging in for "
                f"{LINK_SECONDS // 60} minutes after you approve: anyone who gets the link can "
                "download them."
            )
        },
    )


def _share_follow(kind: str, user, project: Project):
    """After an upload or a new preparation: what sharing it in ``project`` would do."""
    project_ref = activity.project_ref(user, project.pk, project.name)
    where = activity.phrase_of(project_ref, "project")
    people = _people(project, user)

    def follow(result: Any):
        if not isinstance(result, dict):
            return None
        item = result.get("dataset" if kind == "dataset" else "preparation")
        item_id = _int(item.get("id")) if isinstance(item, dict) else None
        if item_id is None:
            return None
        if kind == "dataset":
            ref = activity.dataset_ref(user, item_id, item.get("name"))
            noun = activity.phrase_of(ref, "dataset")
            exposes = f"The people in {where} ({people}) can then see {noun} and train on it."
            action_args = {"dataset_id": item_id, "project_id": project.pk}
        else:
            ref = activity.prep_ref(user, item_id, item.get("name"))
            noun = activity.phrase_of(ref, "data_prep")
            exposes = (
                f"The people in {where} ({people}) can then see {noun}, its steps and the "
                "data file it prepares; the editors of the network can change it."
            )
            action_args = {"prep_id": item_id, "project_id": project.pk}
        targets = [t for t in (ref, project_ref) if t]
        return action_args, f"share {noun} in {where}", targets, {"exposes": exposes}

    return follow


def _gate_upload(request, args: dict[str, Any], rule: Rule) -> Gate | None:
    if not rule.publishing or args.get("project_id") is None:
        return None
    project = _editable_project(request.user, args["project_id"])
    if project is None:
        return None  # upload_dataset explains it
    first = {key: value for key, value in args.items() if key != "project_id"}
    return Gate(
        "publishing",
        "share_dataset",
        {},
        "",
        [],
        "Sharing in a project needs approval.",
        first=first,
        follow=_share_follow("dataset", request.user, project),
    )


def _gate_prep(request, args: dict[str, Any], rule: Rule) -> Gate | None:
    if not rule.publishing or args.get("project_id") is None:
        return None
    project = _editable_project(request.user, args["project_id"])
    if project is None:
        return None
    first = {key: value for key, value in args.items() if key != "project_id"}
    return Gate(
        "publishing",
        "share_data_prep",
        {},
        "",
        [],
        "Sharing in a project needs approval.",
        first=first,
        follow=_share_follow("prep", request.user, project),
    )


def _gate_upload_link(request, args: dict[str, Any], rule: Rule) -> Gate | None:
    from .upload_links import LINK_SECONDS

    if not rule.publishing or args.get("project_id") is None:
        return None
    if args.get("purpose") != "training_dataset" or not settings.TRAINING_ENABLED:
        return None
    project = _editable_project(request.user, args["project_id"])
    if project is None:
        return None
    user = request.user
    project_ref = activity.project_ref(user, project.pk, project.name)
    where = activity.phrase_of(project_ref, "project")
    filename = activity.clean_text(args.get("filename") or "a file", activity.MAX_NAME)
    return Gate(
        "publishing",
        "create_upload_link",
        _plain(args),
        f"create an upload link for “{filename}”, shared in {where}",
        [project_ref] if project_ref else [],
        "Sharing in a project needs approval.",
        {
            "exposes": (
                f"A one-time link that uploads a file without logging in, valid for "
                f"{LINK_SECONDS // 60} minutes after you approve. The dataset it creates is "
                f"shared in {where}: the people in it ({_people(project, user)}) can see it "
                "and train on it."
            )
        },
    )


DELETE_WARNING = "Deleting cannot be undone."


def _deleting(tool: str, phrase: str, targets: list, args: dict[str, Any]) -> Gate:
    return Gate(
        "deleting",
        tool,
        _plain(args),
        phrase,
        targets,
        "Deleting needs approval.",
        {"warning": DELETE_WARNING},
    )


def _gate_delete_project(request, args: dict[str, Any], rule: Rule) -> Gate | None:
    from .projects import projects_for, role_of

    project_id = _int(args.get("id"))
    if not rule.deleting or project_id is None:
        return None
    project = projects_for(request.user).filter(pk=project_id).first()
    if project is None or role_of(request.user, project) != "owner":
        return None
    described = activity.describe_tool(request.user, "delete_project", args, None)
    return _deleting("delete_project", described.phrase, described.targets, args)


def _gate_delete_run(request, args: dict[str, Any], rule: Rule) -> Gate | None:
    from .shared_data import can_delete_run, visible_runs

    if not rule.deleting:
        return None
    try:
        run = visible_runs(request.user).filter(pk=args.get("id")).first()
    except Exception:
        run = None
    if run is None or not can_delete_run(request.user, run):
        return None
    ref = activity.run_ref(request.user, str(run.pk), run.name)
    return _deleting(
        "delete_training_run",
        f"delete {activity.phrase_of(ref, 'run')}",
        [ref] if ref else [],
        {"id": str(run.pk)},
    )


def _gate_delete_dataset(request, args: dict[str, Any], rule: Rule) -> Gate | None:
    dataset_id = _int(args.get("id"))
    if not rule.deleting or dataset_id is None:
        return None
    dataset = Dataset.objects.filter(owner=request.user, pk=dataset_id).first()
    if dataset is None:
        return None
    ref = activity.dataset_ref(request.user, dataset.pk, dataset.name)
    return _deleting(
        "delete_dataset", f"delete {activity.phrase_of(ref, 'dataset')}", [ref], {"id": dataset.pk}
    )


def _gate_delete_source(request, args: dict[str, Any], rule: Rule) -> Gate | None:
    from . import data_service as data

    source_id = _int(args.get("id"))
    if not rule.deleting or source_id is None or data.unavailable_reason() is not None:
        return None
    source = DataSource.objects.filter(owner=request.user, pk=source_id).first()
    if source is None or data.preps_using(source):
        return None  # delete_data_source explains it
    ref = activity.source_ref(request.user, source.pk, source.name)
    return _deleting(
        "delete_data_source",
        f"delete {activity.phrase_of(ref, 'data_source')}",
        [ref],
        {"id": source.pk},
    )


def _gate_delete_prep(request, args: dict[str, Any], rule: Rule) -> Gate | None:
    from . import data_service as data

    prep_id = _int(args.get("id"))
    if not rule.deleting or prep_id is None or data.unavailable_reason() is not None:
        return None
    prep = DataPrep.objects.filter(owner=request.user, pk=prep_id).first()
    if prep is None:
        return None
    ref = activity.prep_ref(request.user, prep.pk, prep.name)
    return _deleting(
        "delete_data_prep",
        f"delete {activity.phrase_of(ref, 'data_prep')}",
        [ref],
        {"id": prep.pk},
    )


GATES: dict[str, Callable[[Request, dict[str, Any], Rule], Gate | None]] = {
    "start_training": _gate_training,
    "get_weights_download": _gate_weights,
    "upload_dataset": _gate_upload,
    "create_data_prep": _gate_prep,
    "create_upload_link": _gate_upload_link,
    "delete_project": _gate_delete_project,
    "delete_training_run": _gate_delete_run,
    "delete_dataset": _gate_delete_dataset,
    "delete_data_source": _gate_delete_source,
    "delete_data_prep": _gate_delete_prep,
}
# For the docs and the web app: which tools each switch covers.
GATED_TOOLS = {
    "training": ["start_training"],
    "publishing": [
        "get_weights_download",
        "upload_dataset (with project_id)",
        "create_data_prep (with project_id)",
        "create_upload_link (training_dataset with project_id)",
    ],
    "deleting": [
        "delete_project",
        "delete_training_run",
        "delete_dataset",
        "delete_data_source",
        "delete_data_prep",
    ],
}


def _plain(value: Any) -> Any:
    """A JSON copy of the arguments (as stored)."""
    return json.loads(json.dumps(value, default=str))


def _fingerprint(action: str, arguments: dict[str, Any]) -> str:
    text = json.dumps([action, arguments], sort_keys=True, default=str, ensure_ascii=True)
    return hashlib.sha256(text.encode()).hexdigest()


# --------------------------------------------------------------------------- #
# Holding a call (mcp.py)
# --------------------------------------------------------------------------- #


@dataclass
class Held:
    """What mcp.py answers instead of running the tool (and how it is logged)."""

    result: Any
    error: str | None
    outcome: str | None = None  # "pending" when a request waits
    approval: int | None = None
    summary: str | None = None
    targets: list[dict[str, Any]] | None = None
    arguments: dict[str, Any] | None = None  # logged instead of the call's arguments


class GateRefused(Exception):
    pass


def _agent_of(token: ApiToken) -> tuple[str, int, str]:
    if token.connection_id is not None:
        connection = OAuthConnection.objects.select_related("client").get(pk=token.connection_id)
        return "app", connection.pk, connection.client.name[:200]
    return "token", token.pk, token.name[:200]


def _agent_requests(token: ApiToken):
    kind, agent_id = (
        ("app", token.connection_id) if token.connection_id is not None else ("token", token.pk)
    )
    return ApprovalRequest.objects.filter(
        owner_id=token.owner_id, agent_kind=kind, agent_id=agent_id
    )


def _request_for(request, tool: str, gate: Gate) -> tuple[ApprovalRequest, bool]:
    """The waiting request for ``gate`` (the same one if the agent asks again)."""
    token: ApiToken = request.auth
    now = timezone.now()
    expire_due(request.user)
    fingerprint = _fingerprint(gate.action, gate.arguments)
    waiting = _agent_requests(token).filter(status="pending")
    same = waiting.filter(fingerprint=fingerprint).first()
    if same is not None:
        return same, False
    if waiting.count() >= MAX_PENDING_PER_AGENT:
        raise GateRefused(
            f"{MAX_PENDING_PER_AGENT} requests of {_who(token)} already wait for the user's "
            "approval. Wait until the user decides them (list_approvals) before asking for more."
        )
    kind, agent_id, name = _agent_of(token)
    details = {**gate.details, "reason": gate.reason, "targets": gate.targets}
    approval = ApprovalRequest.objects.create(
        owner=request.user,
        agent_kind=kind,
        api_token=token if kind == "token" else None,
        connection_id=token.connection_id if kind == "app" else None,
        agent_id=agent_id,
        agent_name=name,
        tool=tool,
        action=gate.action,
        category=gate.category,
        arguments=gate.arguments,
        fingerprint=fingerprint,
        summary=activity.clean_text(gate.phrase, 300),
        details=details,
        created_at=now,
        expires_at=now + timedelta(hours=EXPIRY_HOURS),
    )
    try:
        notify_approval_requested(approval)
    except Exception:
        logger.exception("Could not send the notice of approval request %s", approval.pk)
    return approval, True


def notify_approval_requested(approval: ApprovalRequest) -> None:
    """Tell the owner that ``approval`` waits for them, besides the web app.

    The seam for e-mail: for now it only logs. The web app shows waiting requests on the page
    "Agent activity" and as a badge in the header.
    """
    logger.info(
        "Approval request #%s of %s %s waits for user %s: %s",
        approval.pk,
        approval.agent_kind,
        approval.agent_name,
        approval.owner_id,
        approval.summary,
    )


def _iso(moment: datetime | None) -> str | None:
    return moment.isoformat(timespec="seconds") if moment is not None else None


def _waiting_message(approval: ApprovalRequest, tool: str, done: str = "") -> str:
    expires = approval.expires_at.strftime("%Y-%m-%d %H:%M UTC")
    reason = approval.details.get("reason") or ""
    nothing = done or "Nothing has been done yet."
    return (
        f"Waiting for the user's approval (request #{approval.pk}, expires {expires}). "
        f"{nothing} {reason} The user approves or denies it in the Brick DNN web app. Call "
        f"get_approval with id {approval.pk} to see the decision and, once approved, the "
        f"result. Do not call {tool} again for this."
    ).replace("  ", " ")


def hold(request, tool, args: dict[str, Any], run: Callable[[dict], tuple[Any, str | None]]):
    """Called by mcp.py before ``tool`` runs: ``None`` if it may run now, otherwise the
    answer (a waiting approval request). ``run(arguments)`` runs the tool (for calls whose
    first part is done right away, e.g. an upload that is shared once approved)."""
    gate_of = GATES.get(tool.name)
    token = getattr(request, "auth", None)
    if gate_of is None or not isinstance(token, ApiToken):
        return None
    if tool.writes and token.read_only:
        return None  # refused by the tool
    if any(args.get(name) is None for name in tool.required):
        return None
    rule = rule_of(token)
    if rule == OFF:
        return None
    gate = gate_of(request, args, rule)
    if gate is None:
        return None
    try:
        return _held(request, tool, args, gate, run)
    except GateRefused as exc:
        return Held(None, str(exc))


def _held(request, tool, args, gate: Gate, run) -> Held:
    user = request.user
    if gate.first is None:
        approval, _new = _request_for(request, tool.name, gate)
        answer = {
            "approval_required": True,
            "approval": agent_view(approval),
            "message": _waiting_message(approval, tool.name),
        }
        summary = f"Asked for approval to {approval.summary} (request #{approval.pk})"
        return Held(answer, None, "pending", approval.pk, summary, gate.targets)

    # Done right away without the part that needs approval, which waits.
    result, error = run(gate.first)
    if error is not None:
        return Held(None, error, arguments=gate.first)
    follow = gate.follow(result) if gate.follow is not None else None
    if follow is None:
        return Held(result, None, arguments=gate.first)
    action_args, phrase, targets, details = follow
    gate.arguments, gate.phrase, gate.targets = action_args, phrase, targets
    gate.details = {**gate.details, **details}
    approval, _new = _request_for(request, tool.name, gate)
    done, item_targets = _first_done(user, tool.name, gate.first, result)
    note = (
        f"Done for the user alone. To {approval.summary} the user's approval is needed: "
        f"request #{approval.pk}; get_approval with id {approval.pk} shows the decision."
    )
    answer = dict(result) if isinstance(result, dict) else {"result": result}
    answer["approval"] = agent_view(approval)
    answer["approval_note"] = note
    summary = f"{done}; to {approval.summary} it asked for approval (request #{approval.pk})"
    targets = list({json.dumps(t, sort_keys=True): t for t in [*item_targets, *targets]}.values())
    return Held(answer, None, "pending", approval.pk, summary, targets, gate.first)


def _first_done(user, tool: str, args: dict[str, Any], result: Any) -> tuple[str, list]:
    """What the part of a call that needs no approval did, and its targets."""
    if tool in activity.TOOL_DESCRIPTIONS:
        described = activity.describe_tool(user, tool, args, result)
        return described.summary(True), described.targets
    item = result.get("preparation") if isinstance(result, dict) else None
    if tool == "create_data_prep" and isinstance(item, dict):
        ref = activity.prep_ref(user, item.get("id"), item.get("name"))
        return f"Created {activity.phrase_of(ref, 'data_prep')}", [ref] if ref else []
    return f"Called {tool}", []


# --------------------------------------------------------------------------- #
# What the agent sees (get_approval, list_approvals)
# --------------------------------------------------------------------------- #

NEXT = {
    "pending": "The user has not decided yet. Call get_approval again later; do not repeat "
    "the original call.",
    "approved": "Approved: the server is doing it right now. Call get_approval again in a few "
    "seconds.",
    "executed": "Approved and done: result is what the tool returned.",
    "failed": "Approved, but it could not be done: error says why. You may fix the cause and "
    "call the tool again (it asks for approval again).",
    "denied": "The user denied it; message is their reason, if they gave one. Do not try the "
    "same again without asking the user.",
    "expired": "The user did not decide in time and nothing was done. Ask again only if it is "
    "still needed.",
}


def _sentence(phrase: str) -> str:
    return phrase[:1].upper() + phrase[1:] if phrase else phrase


def agent_view(approval: ApprovalRequest) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": approval.pk,
        "status": approval.status,
        "tool": approval.tool,
        "action": approval.action,
        "category": approval.category,
        "summary": _sentence(approval.summary),
        "reason": approval.details.get("reason"),
        "created_at": _iso(approval.created_at),
        "expires_at": _iso(approval.expires_at),
        "decided_at": _iso(approval.decided_at),
    }
    estimate = approval.details.get("estimate")
    if estimate:
        data["estimate"] = estimate
    if approval.status == "denied":
        data["message"] = approval.message or None
    if approval.status == "executed":
        data["result"] = approval.result
    if approval.status == "failed":
        data["error"] = approval.error
    data["next"] = NEXT[approval.status]
    return data


def _current(approval: ApprovalRequest) -> ApprovalRequest:
    if approval.status == "pending" and approval.expires_at <= timezone.now():
        expire_due(approval.owner)
        approval.refresh_from_db()
    return approval


def get_for_agent(request, approval_id: Any) -> dict[str, Any] | None:
    """A request of the calling token or app (None: no such request of theirs)."""
    number = _int(approval_id)
    token = getattr(request, "auth", None)
    if number is None or not isinstance(token, ApiToken):
        return None
    approval = _agent_requests(token).filter(pk=number).first()
    return agent_view(_current(approval)) if approval is not None else None


def list_for_agent(request, wanted: str | None, limit: int) -> dict[str, Any]:
    token: ApiToken = request.auth
    expire_due(request.user)
    requests = _agent_requests(token)
    if wanted:
        requests = requests.filter(status=wanted)
    total = requests.count()
    shown = [agent_view(approval) for approval in requests.order_by("-id")[:limit]]
    for item in shown:
        item.pop("result", None)  # get_approval shows it
    return {"total": total, "approvals": shown}


# --------------------------------------------------------------------------- #
# Deciding (the web app)
# --------------------------------------------------------------------------- #


def expire_due(user=None) -> int:
    requests = ApprovalRequest.objects.filter(status="pending", expires_at__lte=timezone.now())
    if user is not None:
        requests = requests.filter(owner=user)
    return requests.update(status="expired")


class Conflict(Exception):
    def __init__(self, message: str, approval: ApprovalRequest):
        super().__init__(message)
        self.message = message
        self.approval = approval


ALREADY = {
    "denied": "You denied this request already.",
    "expired": "This request has expired: it can no longer be approved.",
    "pending": "This request still waits.",
}


def _mine(user, approval_id: Any) -> ApprovalRequest:
    number = _int(approval_id)
    approval = ApprovalRequest.objects.filter(owner=user, pk=number).first() if number else None
    if approval is None:
        raise NotFound("No such approval request.")
    return approval


def approve(user, approval_id: Any, web_request) -> ApprovalRequest:
    """Approve and do it right away as the agent. Approving again does nothing again."""
    approval = _mine(user, approval_id)
    now = timezone.now()
    taken = ApprovalRequest.objects.filter(
        pk=approval.pk, status="pending", expires_at__gt=now
    ).update(status="approved", decided_at=now)
    if not taken:
        approval = _current(approval)
        approval.refresh_from_db()
        if approval.status in ("approved", "executed", "failed"):
            return approval  # approved before
        raise Conflict(ALREADY.get(approval.status, "This request cannot be approved."), approval)
    approval.refresh_from_db()
    _execute(approval, web_request)
    return approval


def deny(user, approval_id: Any, message: Any) -> ApprovalRequest:
    if message is not None and not isinstance(message, str):
        raise ValidationError({"message": "A text, or nothing."})
    text = activity.clean_text(message or "", MAX_MESSAGE) if message else ""
    approval = _mine(user, approval_id)
    now = timezone.now()
    taken = ApprovalRequest.objects.filter(
        pk=approval.pk, status="pending", expires_at__gt=now
    ).update(status="denied", decided_at=now, message=text)
    approval = _current(approval)
    approval.refresh_from_db()
    if not taken:
        if approval.status == "denied":
            return approval
        if approval.status in ("approved", "executed", "failed"):
            raise Conflict("You approved this request already.", approval)
        raise Conflict(ALREADY.get(approval.status, "This request cannot be denied."), approval)
    summary = f"Denied: {approval.summary} (request #{approval.pk})"
    activity.record_approval(approval, "denied", summary, error=text)
    return approval


def _agent_token(approval: ApprovalRequest) -> ApiToken:
    """The token to act with: the agent's own, or for an app one like its access tokens."""
    if approval.agent_kind == "token":
        token = approval.api_token
        if token is None:
            raise _Failed(f"The API token “{approval.agent_name}” was revoked.")
        if token.expires_at is not None and token.expires_at <= timezone.now():
            raise _Failed(f"The API token “{approval.agent_name}” has expired.")
        return token
    connection = approval.connection
    if connection is None:
        raise _Failed(f"The app “{approval.agent_name}” was disconnected.")
    return ApiToken(
        owner=approval.owner,
        connection=connection,
        name=connection.client.name[:100],
        prefix="",
        read_only=connection.read_only,
    )


class _Failed(Exception):
    pass


def _agent_request(web_request, user, token: ApiToken) -> Request:
    """A request as the agent's: the user's, with the agent's token (budgets, the activity
    log, rate limits), on the web request's host (links)."""
    django_request = copy.copy(web_request._request)
    django_request.__dict__.pop("_agent_calls", None)
    request = Request(django_request)
    request.user = user
    request.auth = token
    return request


def _share(kind: str):
    model = Dataset if kind == "dataset" else DataPrep
    key = "dataset_id" if kind == "dataset" else "prep_id"
    noun = "dataset" if kind == "dataset" else "data preparation"

    def run(request, args: dict[str, Any]) -> tuple[Any, str | None]:
        user = request.user
        item = model.objects.filter(owner=user, pk=_int(args.get(key))).first()
        if item is None:
            return None, f"The {noun} was deleted in the meantime."
        project = _editable_project(user, args.get("project_id"))
        if project is None:
            return None, "The user can no longer share data in this project."
        item.projects.add(project)
        return {
            "shared": True,
            kind: {"id": item.pk, "name": item.name},
            "project": {"id": project.pk, "name": project.name},
        }, None

    return run


SHARE_ACTIONS = {"share_dataset": _share("dataset"), "share_data_prep": _share("prep")}


def _execute(approval: ApprovalRequest, web_request) -> None:
    from . import mcp

    started = time.monotonic()
    result: Any = None
    try:
        request = _agent_request(web_request, approval.owner, _agent_token(approval))
        if approval.action in SHARE_ACTIONS:
            result, error = SHARE_ACTIONS[approval.action](request, approval.arguments)
        else:
            result, error = mcp._run_tool(mcp.TOOLS[approval.action], request, approval.arguments)
    except _Failed as exc:
        error = f"{exc} Nothing was done."
    except Exception:
        logger.exception("Approved request %s failed", approval.pk)
        error = "Internal error in the Brick DNN server."
    approval.status = "executed" if error is None else "failed"
    approval.result = _plain(result) if error is None else None
    approval.error = (error or "")[:1000]
    approval.finished_at = timezone.now()
    done = _done_summary(approval, result, error is None)
    approval.details = {**approval.details, "done": done}
    approval.save(update_fields=["status", "result", "error", "finished_at", "details"])
    activity.record_approval(
        approval,
        "approved" if error is None else "error",
        done,
        error=error or "",
        duration_ms=round((time.monotonic() - started) * 1000),
        result=result,
    )


def _done_summary(approval: ApprovalRequest, result: Any, ok: bool) -> str:
    if approval.action in activity.TOOL_DESCRIPTIONS:
        described = activity.describe_tool(
            approval.owner, approval.action, approval.arguments, result if ok else None
        )
        text = described.summary(ok)
    else:
        text = activity.past_tense(approval.summary) if ok else f"Could not {approval.summary}"
    return f"{text}, approved by you (request #{approval.pk})"


# --------------------------------------------------------------------------- #
# The REST API cannot ask
# --------------------------------------------------------------------------- #

REST_ROUTES = {
    ("project-link", "POST"): ("publishing", "Changing the share link of a network"),
    ("project-members", "POST"): ("publishing", "Inviting people to a network"),
    ("project-data", "POST"): ("publishing", "Sharing data in a network"),
    ("project-detail", "DELETE"): ("deleting", "Deleting a network"),
    ("training-run-detail", "DELETE"): ("deleting", "Deleting a training run"),
    ("training-dataset-detail", "DELETE"): ("deleting", "Deleting a dataset"),
    ("data-source-detail", "DELETE"): ("deleting", "Deleting a data file"),
    ("data-prep-detail", "DELETE"): ("deleting", "Deleting a data preparation"),
}
REST_HINT = (
    "The REST API cannot ask for approval: use the MCP server, whose tools ask the user, or "
    "ask the user to do it in the Brick DNN web app."
)


def refuse_rest(token: ApiToken, route: str | None, method: str | None) -> None:
    """Called for requests with a personal API token: raise ``PermissionDenied`` for a REST
    request its rule says needs approval."""
    entry = REST_ROUTES.get((route or "", method or ""))
    if entry is None:
        return
    category, what = entry
    if getattr(rule_of(token), category):
        raise PermissionDenied(f"{what} needs the user's approval with this API token. {REST_HINT}")


def refuse_rest_training(request, data: Any) -> None:
    """``POST /api/training/runs/`` with an API token: refused above the token's limit."""
    token = getattr(request, "auth", None)
    if not isinstance(token, ApiToken) or not isinstance(data, dict):
        return
    rule = rule_of(token)
    if rule.training_cpu_minutes is None:
        return
    args = {**data, "project_id": data.get("project")}
    args.pop("project", None)
    check = _training_check(request, args)
    if check is None:
        return
    needed, reason = _training_needs_approval(rule, check)
    if needed:
        raise PermissionDenied(
            f"This training needs the user's approval with this API token: {reason} {REST_HINT}"
        )


# --------------------------------------------------------------------------- #
# Housekeeping (purge_old_data) and export
# --------------------------------------------------------------------------- #


def purge(now: datetime, dry_run: bool = False) -> int:
    """Delete requests that ended more than KEEP_DAYS days ago."""
    expire_due()
    old = ApprovalRequest.objects.exclude(status__in=("pending", "approved")).filter(
        created_at__lt=now - timedelta(days=KEEP_DAYS)
    )
    count = old.count()
    if not dry_run:
        old.delete()
    return count


def export_approvals(user) -> dict[str, Any]:
    """approvals.json of the account export: the rules and the requests."""
    rules = []
    for row in ApprovalRule.objects.filter(owner=user).order_by("id"):
        if row.api_token_id is not None:
            applies = f"API token #{row.api_token_id}"
        elif row.connection_id is not None:
            applies = f"connected app #{row.connection_id}"
        else:
            applies = "default for new tokens and apps"
        rules.append({"applies_to": applies, **Rule.of(row).to_dict()})
    requests = [
        {
            **approval_data(approval),
            "arguments": approval.arguments,
            "result": approval.result,
        }
        for approval in ApprovalRequest.objects.filter(owner=user).order_by("id")
    ]
    return {"rules": rules, "requests": requests}


# --------------------------------------------------------------------------- #
# API for the web app (website login only)
# --------------------------------------------------------------------------- #


class WebsiteLogin(IsLoggedInUser):
    message = "Approvals are decided after logging in on the website; API tokens and apps cannot."


def _agent_active(approval: ApprovalRequest) -> bool:
    if approval.agent_kind == "token":
        return approval.api_token_id is not None
    return approval.connection_id is not None


def approval_data(approval: ApprovalRequest) -> dict[str, Any]:
    """``ApprovalRequest`` (frontend/src/api/types.ts)."""
    details = approval.details or {}
    return {
        "id": approval.pk,
        "status": approval.status,
        "category": approval.category,
        "tool": approval.tool,
        "action": approval.action,
        "agent": {
            "kind": approval.agent_kind,
            "id": approval.agent_id,
            "key": f"{approval.agent_kind}:{approval.agent_id}",
            "name": approval.agent_name,
            "active": _agent_active(approval),
        },
        "summary": _sentence(approval.summary),
        "reason": details.get("reason") or "",
        "estimate": details.get("estimate"),
        "limit_cpu_minutes": details.get("limit_cpu_minutes"),
        "network": details.get("network"),
        "dataset": details.get("dataset"),
        "exposes": details.get("exposes"),
        "warning": details.get("warning"),
        "targets": details.get("targets") or [],
        "arguments": activity.summarize(approval.arguments),
        "created_at": approval.created_at,
        "expires_at": approval.expires_at,
        "decided_at": approval.decided_at,
        "finished_at": approval.finished_at,
        "message": approval.message,
        "error": approval.error,
        "done": details.get("done"),
    }


def pending_for(user, limit: int = LIST_LIMIT) -> dict[str, Any]:
    """The waiting requests of ``user`` (for the activity page and the header)."""
    expire_due(user)
    waiting = ApprovalRequest.objects.filter(owner=user, status="pending")
    return {
        "pending_count": waiting.count(),
        "pending": [approval_data(a) for a in waiting.order_by("-id")[:limit]],
    }


STATUS_FILTERS = {
    "pending": ("pending",),
    "decided": ("approved", "executed", "failed", "denied", "expired"),
}


class ApprovalListView(APIView):
    """``GET /api/approvals/?status=pending|decided&limit=``: the user's requests, newest
    first, with ``pending_count``."""

    permission_classes = [WebsiteLogin]

    def get(self, request: Request) -> Response:
        expire_due(request.user)
        mine = ApprovalRequest.objects.filter(owner=request.user)
        wanted = request.query_params.get("status")
        requests = mine
        if wanted:
            if wanted not in STATUS_FILTERS:
                raise ValidationError({"status": "Use pending or decided."})
            requests = mine.filter(status__in=STATUS_FILTERS[wanted])
        limit = request.query_params.get("limit") or str(LIST_LIMIT)
        if not limit.isdigit() or int(limit) > 200:
            raise ValidationError({"limit": "Must be a whole number from 0 to 200."})
        page = requests.order_by("-id")[: int(limit)]
        return Response(
            {
                "results": [approval_data(a) for a in page],
                "pending_count": mine.filter(status="pending").count(),
            }
        )


class ApprovalDetailView(APIView):
    permission_classes = [WebsiteLogin]

    def get(self, request: Request, pk: int) -> Response:
        return Response(approval_data(_current(_mine(request.user, pk))))


def _conflict(exc: Conflict) -> Response:
    return Response(
        {"detail": exc.message, "approval": approval_data(exc.approval)},
        status=status.HTTP_409_CONFLICT,
    )


class ApproveView(APIView):
    """``POST /api/approvals/<id>/approve/``: done right away as the agent; the answer is the
    request with its outcome (``executed`` or ``failed``). 409 if denied or expired."""

    permission_classes = [WebsiteLogin]

    def post(self, request: Request, pk: int) -> Response:
        try:
            approval = approve(request.user, pk, request)
        except Conflict as exc:
            return _conflict(exc)
        return Response(approval_data(approval))


class DenyView(APIView):
    """``POST /api/approvals/<id>/deny/ {"message"?}``: the agent reads the message."""

    permission_classes = [WebsiteLogin]

    def post(self, request: Request, pk: int) -> Response:
        body = request.data if isinstance(request.data, dict) else {}
        try:
            approval = deny(request.user, pk, body.get("message"))
        except Conflict as exc:
            return _conflict(exc)
        return Response(approval_data(approval))


def _rule_from(body: Any) -> Rule:
    if not isinstance(body, dict):
        raise ValidationError({"detail": "Send training_cpu_minutes, publishing and deleting."})
    minutes = body.get("training_cpu_minutes")
    if minutes is not None:
        if isinstance(minutes, bool) or not isinstance(minutes, (int, float)):
            raise ValidationError({"training_cpu_minutes": "A number of CPU minutes, or null."})
        if not 0 <= minutes <= MAX_CPU_MINUTES:
            raise ValidationError(
                {"training_cpu_minutes": f"From 0 to {MAX_CPU_MINUTES:,} CPU minutes."}
            )
        minutes = float(minutes)
    flags = {}
    for name in ("publishing", "deleting"):
        value = body.get(name, False)
        if not isinstance(value, bool):
            raise ValidationError({name: "true or false."})
        flags[name] = value
    return Rule(minutes, **flags)


class ApprovalRulesView(APIView):
    """``GET /api/approvals/rules/``: the account's default and the rule of each token and
    app (``"token:<id>"``, ``"app:<id>"``)."""

    permission_classes = [WebsiteLogin]

    def get(self, request: Request) -> Response:
        user = request.user
        rows = ApprovalRule.objects.filter(owner=user)
        default = next(
            (r for r in rows if r.api_token_id is None and r.connection_id is None), None
        )
        by_token = {r.api_token_id: r for r in rows if r.api_token_id is not None}
        by_app = {r.connection_id: r for r in rows if r.connection_id is not None}
        agents: dict[str, Any] = {}
        tokens = ApiToken.objects.filter(owner=user, connection__isnull=True)
        for pk in tokens.values_list("pk", flat=True):
            agents[f"token:{pk}"] = Rule.of(by_token.get(pk)).to_dict()
        for pk in OAuthConnection.objects.filter(owner=user).values_list("pk", flat=True):
            agents[f"app:{pk}"] = Rule.of(by_app.get(pk)).to_dict()
        return Response(
            {
                "default": (Rule.of(default) if default else BUILTIN_DEFAULT).to_dict(),
                "builtin_default": BUILTIN_DEFAULT.to_dict(),
                "agents": agents,
                "gated_tools": GATED_TOOLS,
                "expiry_hours": EXPIRY_HOURS,
            }
        )


class ApprovalRuleView(APIView):
    """``PUT /api/approvals/rules/default/``, ``…/token/<id>/``, ``…/app/<id>/``:
    ``{"training_cpu_minutes", "publishing", "deleting"}``."""

    permission_classes = [WebsiteLogin]

    def put(self, request: Request, kind: str, pk: int | None = None) -> Response:
        user = request.user
        rule = _rule_from(request.data)
        token = connection = None
        if kind == "token":
            token = ApiToken.objects.filter(owner=user, connection__isnull=True, pk=pk).first()
            if token is None:
                raise NotFound("No such API token.")
        elif kind == "app":
            connection = OAuthConnection.objects.filter(owner=user, pk=pk).first()
            if connection is None:
                raise NotFound("No such app.")
        set_rule(user, rule, token=token, connection=connection)
        return Response(rule.to_dict())
