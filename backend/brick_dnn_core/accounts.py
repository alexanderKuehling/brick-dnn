"""The own account (the page /account): e-mail address, password, data export, deletion.

Only a website login (JWT) can use these endpoints, not an API token, and changes are rate
limited (``account`` scope). Deleting an account deletes everything it owns:

- its projects (members lose access; people editing them live are told), with their history,
  comments and the data shared in them;
- its data sources, preparations, datasets and training runs with their files (running
  trainings are stopped first);
- its memberships in other people's projects, its comments on them (replies of others to a
  thread it started go with it, as when it deletes the thread itself), API tokens,
  connected apps and the activity log of its agents;
- its usage records (the compute it used, see budgets.py) and budgets;
- what is kept about its paid plan (billing.py). A subscription at Paddle is cancelled first,
  at once; if Paddle cannot be reached, the account is not deleted (the request fails), so
  that nobody goes on paying for an account that is gone.

Other people's projects it worked on stay; their history keeps the versions it saved, without
saying who. All its logins end: its refresh tokens are blacklisted. See docs/privacy.md.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import IO, Any
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

from django.conf import settings
from django.contrib.auth import get_user_model
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ObjectDoesNotExist
from django.core.exceptions import ValidationError as DjangoValidationError
from django.core.serializers.json import DjangoJSONEncoder
from django.db import transaction
from django.db.models import Sum
from django.http import FileResponse
from django.utils import timezone
from django.utils.text import get_valid_filename, slugify
from rest_framework import serializers, status
from rest_framework.exceptions import AuthenticationFailed
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.throttling import ScopedRateThrottle
from rest_framework.views import APIView
from rest_framework_simplejwt.serializers import (
    TokenObtainPairSerializer,
    TokenRefreshSerializer,
)
from rest_framework_simplejwt.settings import api_settings as jwt_settings
from rest_framework_simplejwt.tokens import AccessToken, RefreshToken
from rest_framework_simplejwt.views import TokenRefreshView

from brickdnn.training import ACTIVE_STATUSES

from . import billing, budgets, data_service, live
from . import training_service as training
from .agent_activity import export_entries
from .api_tokens import IsLoggedInUser
from .models import (
    AgentAction,
    ApiToken,
    Comment,
    DataPrep,
    Dataset,
    DataSource,
    OAuthConnection,
    Project,
    ProjectMember,
    TrainingRun,
    UserProfile,
)
from .retention import record_activity
from .serializers import clean_email, email_field

logger = logging.getLogger(__name__)
User = get_user_model()


# --------------------------------------------------------------------------- #
# Logging in: remember the activity (for deleting inactive accounts)
# --------------------------------------------------------------------------- #


class LoginSerializer(TokenObtainPairSerializer):
    def validate(self, attrs):
        data = super().validate(attrs)
        record_activity(self.user.pk)
        return data


class RefreshSerializer(TokenRefreshSerializer):
    def validate(self, attrs):
        try:
            data = super().validate(attrs)
        except ObjectDoesNotExist:  # the account was deleted, e.g. in the Django admin
            raise AuthenticationFailed(
                self.error_messages["no_active_account"], "no_active_account"
            ) from None
        record_activity(AccessToken(data["access"]).payload.get(jwt_settings.USER_ID_CLAIM))
        return data


class RefreshView(TokenRefreshView):
    serializer_class = RefreshSerializer


def tokens_for(user) -> dict[str, str]:
    refresh = RefreshToken.for_user(user)
    return {"access": str(refresh.access_token), "refresh": str(refresh)}


def log_out_everywhere(user) -> None:
    """Blacklist all refresh tokens of ``user`` (access tokens expire within minutes)."""
    from rest_framework_simplejwt.token_blacklist.models import BlacklistedToken, OutstandingToken

    tokens = OutstandingToken.objects.filter(user=user, blacklistedtoken__isnull=True)
    BlacklistedToken.objects.bulk_create(
        [BlacklistedToken(token=token) for token in tokens], ignore_conflicts=True
    )


# --------------------------------------------------------------------------- #
# Deleting an account
# --------------------------------------------------------------------------- #


def delete_account(user, stop_trainings: bool = True) -> None:
    """Delete ``user`` and everything they own (see the module docstring). Files are removed
    and live editors told once the deletion is committed. ``stop_trainings``: kill active
    trainings first (the purge command skips accounts with active trainings instead).
    Raises ``billing.BillingUnavailable`` (and deletes nothing) if the account's subscription
    at Paddle cannot be cancelled."""
    billing.cancel_for_deletion(user)
    if stop_trainings:
        for run in TrainingRun.objects.filter(owner=user, status__in=ACTIVE_STATUSES):
            training.terminate_run(run)
    user_id = user.pk
    owned = list(Project.objects.filter(owner=user).values_list("pk", flat=True))
    member_of = list(ProjectMember.objects.filter(user=user).values_list("project_id", flat=True))
    comments = list(
        Comment.objects.filter(author=user)
        .exclude(project__owner=user)
        .values_list("project_id", "pk")
    )

    def notify() -> None:
        for project_id in owned:
            live.notify_deleted(project_id)
        for project_id in member_of:
            live.notify_access(project_id, user_id, None)
        for project_id, comment_id in comments:
            live.notify_comment(project_id, "deleted", {"id": comment_id})

    with transaction.atomic():
        log_out_everywhere(user)
        user.delete()  # cascades; the files go with post_delete signals
        transaction.on_commit(notify)


# --------------------------------------------------------------------------- #
# Exporting the data of an account
# --------------------------------------------------------------------------- #

README = """Brick DNN: your data
====================

Everything this server stores about the account "{username}", exported {date}.

account.json             your account: username, e-mail, when you signed up and last logged
                         in, the terms of use you accepted
projects.json            networks you own: graphs, descriptions, sharing settings and their
                         version history
memberships.json         networks of other people that were shared with you
comments.json            comments you wrote
api_tokens.json          your API tokens: name, the start of the token, dates (never the
                         token itself; the server only keeps a hash)
connected_apps.json      apps you connected with OAuth, e.g. AI assistants
agent_activity.json      what AI assistants and scripts did with your API tokens and
                         connected apps (summarised, as on the page "Agent activity")
data_sources.json        CSV files you imported in the Data tab
data_preparations.json   your data preparations: steps and settings
datasets.json            your training datasets
training_runs.json       your trainings: network, settings, status and metrics
usage.json               the compute (CPU time) your trainings and predictions used, per
                         month, with the API token or app that caused it, and your limits
{billing}files.json               the uploaded files: where they are in this ZIP file, or why not
files/                   the uploaded files themselves

Not included: trained weights (download them from a run in the app) and data of other people,
e.g. their comments on your networks.
"""


# Only in the export of an account that bought a plan.
BILLING_README = """billing.json             your paid plan: customer and subscription numbers at Paddle (the
                         seller), status and dates, and compute billed beyond the allowance.
                         Paddle keeps your payment data and invoices.
"""


def _json(archive: ZipFile, name: str, data: Any) -> None:
    text = json.dumps(data, cls=DjangoJSONEncoder, ensure_ascii=False, indent=2)
    archive.writestr(name, text + "\n", compress_type=ZIP_DEFLATED)


def _account(user) -> dict[str, Any]:
    profile = UserProfile.objects.filter(user=user).first()
    return {
        "id": user.pk,
        "username": user.get_username(),
        "email": user.email,
        "date_joined": user.date_joined,
        "last_login": user.last_login,
        "last_active_at": profile.last_active_at if profile else None,
        "terms_accepted_at": profile.terms_accepted_at if profile else None,
        "terms_version": (profile.terms_version or None) if profile else None,
        "privacy_version": (profile.privacy_version or None) if profile else None,
    }


def _projects(user) -> list[dict[str, Any]]:
    projects = []
    for project in Project.objects.filter(owner=user).order_by("pk"):
        history = [
            {
                "version": entry.version,
                "kind": entry.kind,
                "label": entry.label,
                "name": entry.name,
                "description": entry.description,
                "graph": entry.graph,
                "saved_by_you": entry.created_by_id == user.pk,
                "created_at": entry.created_at,
                "updated_at": entry.updated_at,
            }
            for entry in project.history.order_by("created_at", "pk")
        ]
        projects.append(
            {
                "id": project.pk,
                "name": project.name,
                "description": project.description,
                "graph": project.graph,
                "version": project.version,
                "created_at": project.created_at,
                "updated_at": project.updated_at,
                "link_access": project.link_access,
                "number_of_members": project.members.count(),
                "history": history,
            }
        )
    return projects


def _memberships(user) -> list[dict[str, Any]]:
    return [
        {
            "project": {"id": member.project_id, "name": member.project.name},
            "role": member.role,
            "since": member.created_at,
        }
        for member in ProjectMember.objects.filter(user=user).select_related("project")
    ]


def _comments(user) -> list[dict[str, Any]]:
    visible = set(Project.objects.filter(owner=user).values_list("pk", flat=True))
    visible |= set(ProjectMember.objects.filter(user=user).values_list("project_id", flat=True))
    comments = []
    for comment in Comment.objects.filter(author=user).select_related("project").order_by("pk"):
        project: dict[str, Any] = {"id": comment.project_id}
        if comment.project_id in visible:
            project["name"] = comment.project.name
        comments.append(
            {
                "id": comment.pk,
                "project": project,
                "layer": comment.node_id or None,
                "reply_to": comment.parent_id,
                "text": comment.text,
                "resolved": comment.resolved,
                "created_at": comment.created_at,
                "updated_at": comment.updated_at,
            }
        )
    return comments


def _api_tokens(user) -> list[dict[str, Any]]:
    return [
        {
            "name": token.name,
            "prefix": token.prefix,
            "read_only": token.read_only,
            "created_at": token.created_at,
            "last_used_at": token.last_used_at,
            "expires_at": token.expires_at,
        }
        for token in ApiToken.objects.filter(owner=user, connection__isnull=True)
    ]


def _connections(user) -> list[dict[str, Any]]:
    return [
        {
            "app": connection.client.name,
            "app_uri": connection.client.uri or None,
            "read_only": connection.read_only,
            "created_at": connection.created_at,
            "last_used_at": connection.last_used_at,
        }
        for connection in OAuthConnection.objects.filter(owner=user).select_related("client")
    ]


def _project_ids(item: Dataset | DataPrep) -> list[int]:
    return sorted(item.projects.values_list("pk", flat=True))


def _data_sources(user) -> list[dict[str, Any]]:
    return [
        {
            "id": source.pk,
            "name": source.name,
            "filename": source.filename,
            "size_bytes": source.size_bytes,
            "rows": source.n_rows,
            "columns": source.columns,
            "import_options": source.import_options,
            "created_at": source.created_at,
            "last_used_at": source.last_used_at,
        }
        for source in DataSource.objects.filter(owner=user).order_by("pk")
    ]


def _data_preps(user) -> list[dict[str, Any]]:
    return [
        {
            "id": prep.pk,
            "name": prep.name,
            "data_source": prep.source_id,
            "steps": prep.steps,
            "settings": prep.settings,
            "shared_in_projects": _project_ids(prep),
            "created_at": prep.created_at,
            "updated_at": prep.updated_at,
        }
        for prep in DataPrep.objects.filter(owner=user).order_by("pk")
    ]


def _datasets(user) -> list[dict[str, Any]]:
    return [
        {
            "id": dataset.pk,
            "name": dataset.name,
            "kind": dataset.kind,
            "size_bytes": dataset.size_bytes,
            "info": dataset.info,
            "prepared_from": dataset.prepared_from_id,
            "shared_in_projects": _project_ids(dataset),
            "created_at": dataset.created_at,
            "last_used_at": dataset.last_used_at,
        }
        for dataset in Dataset.objects.filter(owner=user).order_by("pk")
    ]


def _runs(user) -> list[dict[str, Any]]:
    return [
        {
            "id": str(run.pk),
            "name": run.name,
            "status": run.status,
            "message": run.message,
            "project": run.project_id,
            "dataset": run.dataset,
            "graph": run.graph,
            "config": run.config,
            "tuning": run.tuning,
            "total_params": run.total_params,
            "epochs_done": run.epochs_done,
            "trials_done": run.trials_done,
            "metrics": run.metrics,
            "best": run.best,
            "best_trial": run.best_trial,
            "has_weights": run.has_weights,
            "cpu_seconds": run.cpu_seconds,
            "stop_reason": run.stop_reason or None,
            "created_at": run.created_at,
            "started_at": run.started_at,
            "finished_at": run.finished_at,
            "last_used_at": run.last_used_at,
        }
        for run in TrainingRun.objects.filter(owner=user).order_by("created_at")
    ]


def _uploads(user) -> list[tuple[dict[str, Any], list[tuple[Path, str]]]]:
    """The uploaded files: ``(entry for files.json, [(path on disk, name in the ZIP)])``. An
    empty list of files means they are missing."""
    uploads = []
    for source in DataSource.objects.filter(owner=user).order_by("pk"):
        # The file as uploaded or downloaded: original.csv, original.xlsx, original.json.gz ...
        path = data_service.original_file(source.pk)
        name = get_valid_filename(Path(source.filename).name) if source.filename else ""
        if name in ("", ".", ".."):
            name = path.name if path else data_service.ORIGINAL_FILE
        target = f"files/data_sources/{source.pk}/{name}"
        entry = {"kind": "data_source", "id": source.pk, "name": source.name, "path": target}
        uploads.append((entry, [(path, target)] if path else []))
    for dataset in Dataset.objects.filter(owner=user).order_by("pk"):
        root = training.dataset_dir(dataset.pk)
        entry = {"kind": "dataset", "id": dataset.pk, "name": dataset.name}
        if dataset.kind == "csv":
            entry["path"] = f"files/datasets/{dataset.pk}/data.csv"
            path = root / "data.csv"
            files = [(path, entry["path"])] if path.is_file() else []
        else:
            entry["path"] = f"files/datasets/{dataset.pk}/images/"
            images = root / "images"
            files = [
                (path, entry["path"] + path.relative_to(images).as_posix())
                for path in sorted(images.rglob("*"))
                if path.is_file() and not path.is_symlink()
            ]
        uploads.append((entry, files))
    return uploads


def _write_files(user, archive: ZipFile) -> list[dict[str, Any]]:
    """Add the uploaded files up to EXPORT_MAX_FILES_MB; return what files.json lists."""
    budget = int(float(settings.EXPORT_MAX_FILES_MB) * 1024 * 1024)
    listed = []
    for entry, files in _uploads(user):
        size = sum(path.stat().st_size for path, _ in files)
        entry["size_bytes"] = size
        if not files:
            entry.update(included=False, note="The file is missing on the server.")
        elif size > budget:
            entry.update(
                included=False,
                note=(
                    f"Not included: the export holds at most {settings.EXPORT_MAX_FILES_MB:g} MB "
                    "of files. Ask the operator of this server for it."
                ),
            )
        else:
            for path, name in files:
                # Images are compressed already.
                stored = ZIP_STORED if entry["path"].endswith("/images/") else ZIP_DEFLATED
                archive.write(path, name, compress_type=stored)
            budget -= size
            entry["included"] = True
        if not entry["included"]:
            del entry["path"]
        listed.append(entry)
    return listed


def export_account(user, out: IO[bytes]) -> None:
    """Write a ZIP file with everything stored about ``user`` (see README) to ``out``."""
    with ZipFile(out, "w", compression=ZIP_DEFLATED) as archive:
        today = timezone.now().date().isoformat()
        paid = billing.export_records(user)
        archive.writestr(
            "README.txt",
            README.format(
                username=user.get_username(), date=today, billing=BILLING_README if paid else ""
            ),
        )
        _json(archive, "account.json", _account(user))
        _json(archive, "projects.json", _projects(user))
        _json(archive, "memberships.json", _memberships(user))
        _json(archive, "comments.json", _comments(user))
        _json(archive, "api_tokens.json", _api_tokens(user))
        _json(archive, "connected_apps.json", _connections(user))
        _json(archive, "agent_activity.json", export_entries(user))
        _json(archive, "data_sources.json", _data_sources(user))
        _json(archive, "data_preparations.json", _data_preps(user))
        _json(archive, "datasets.json", _datasets(user))
        _json(archive, "training_runs.json", _runs(user))
        _json(archive, "usage.json", budgets.export_records(user))
        if paid:
            _json(archive, "billing.json", paid)
        _json(archive, "files.json", _write_files(user, archive))


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #


class WebsiteLogin(IsLoggedInUser):
    message = "Manage your account after logging in on the website; API tokens cannot do this."


class AccountApiView(APIView):
    permission_classes = [WebsiteLogin]
    throttle_scope = "account"
    # Requests that only read count against the normal rate limits.
    unthrottled_methods: tuple[str, ...] = ()

    def get_throttles(self):
        throttles = super().get_throttles()
        if self.request.method not in self.unthrottled_methods:
            throttles.append(ScopedRateThrottle())
        return throttles


def account_data(user) -> dict[str, Any]:
    """``Account`` (frontend/src/api/types.ts): the account and how much is stored for it."""
    uploads = (DataSource.objects.filter(owner=user).aggregate(n=Sum("size_bytes"))["n"] or 0) + (
        Dataset.objects.filter(owner=user).aggregate(n=Sum("size_bytes"))["n"] or 0
    )
    return {
        **_account(user),
        "stored": {
            "projects": Project.objects.filter(owner=user).count(),
            "memberships": ProjectMember.objects.filter(user=user).count(),
            "comments": Comment.objects.filter(author=user).count(),
            "data_sources": DataSource.objects.filter(owner=user).count(),
            "data_preps": DataPrep.objects.filter(owner=user).count(),
            "datasets": Dataset.objects.filter(owner=user).count(),
            "training_runs": TrainingRun.objects.filter(owner=user).count(),
            "api_tokens": ApiToken.objects.filter(owner=user, connection__isnull=True).count(),
            "connected_apps": OAuthConnection.objects.filter(owner=user).count(),
            "agent_actions": AgentAction.objects.filter(owner=user).count(),
            "upload_bytes": uploads,
        },
    }


class EmailSerializer(serializers.Serializer):
    email = email_field()

    def validate_email(self, value: str) -> str:
        return clean_email(value, user=self.context["user"])


class AccountView(AccountApiView):
    """``GET``: the account. ``PATCH {"email"}``: change the e-mail address."""

    unthrottled_methods = ("GET", "HEAD", "OPTIONS")

    def get(self, request: Request) -> Response:
        return Response(account_data(request.user))

    def patch(self, request: Request) -> Response:
        serializer = EmailSerializer(data=request.data, context={"user": request.user})
        serializer.is_valid(raise_exception=True)
        request.user.email = serializer.validated_data["email"]
        request.user.save(update_fields=["email"])
        billing.email_changed(request.user)  # invoices of a paid plan go to this address
        return Response(account_data(request.user))


class PasswordSerializer(serializers.Serializer):
    current_password = serializers.CharField(write_only=True)
    new_password = serializers.CharField(write_only=True)

    def validate_current_password(self, value: str) -> str:
        if not self.context["user"].check_password(value):
            raise serializers.ValidationError("The current password is wrong.")
        return value

    def validate_new_password(self, value: str) -> str:
        try:
            validate_password(value, user=self.context["user"])
        except DjangoValidationError as exc:
            raise serializers.ValidationError(list(exc.messages)) from None
        return value


class PasswordView(AccountApiView):
    """``POST {"current_password", "new_password"}``: change the password. Logs out all other
    sessions and returns new tokens for this one: ``{"access", "refresh"}``."""

    def post(self, request: Request) -> Response:
        user = request.user
        serializer = PasswordSerializer(data=request.data, context={"user": user})
        serializer.is_valid(raise_exception=True)
        user.set_password(serializer.validated_data["new_password"])
        user.save(update_fields=["password"])
        log_out_everywhere(user)
        return Response(tokens_for(user))


class ExportView(AccountApiView):
    """``GET``: a ZIP file with everything stored about the account."""

    def get(self, request: Request) -> FileResponse:
        handle = tempfile.TemporaryFile()  # noqa: SIM115 (closed by the response)
        try:
            export_account(request.user, handle)
            handle.seek(0)
        except BaseException:
            handle.close()
            raise
        day = timezone.now().date().isoformat()
        name = slugify(request.user.get_username()) or "account"
        response = FileResponse(
            handle,
            as_attachment=True,
            filename=f"brick-dnn-{name}-{day}.zip",
            content_type="application/zip",
        )
        response["Cache-Control"] = "private, no-store"
        return response


class UsagePermission(IsLoggedInUser):
    """Reading the usage works with any login, also an API token (agents check their budget
    before training); changing the budget only with a website login."""

    message = "Change budgets after logging in on the website; API tokens cannot do this."

    def has_permission(self, request, view) -> bool:
        if request.method in ("GET", "HEAD", "OPTIONS"):
            return bool(request.user and request.user.is_authenticated)
        return super().has_permission(request, view)


class AccountBudgetSerializer(serializers.Serializer):
    """What users set for their own account; every field is optional."""

    budget_cpu_minutes = serializers.FloatField(
        min_value=0, max_value=10_000_000, allow_null=True, required=False
    )
    overage_enabled = serializers.BooleanField(required=False)
    overage_cap_cpu_hours = serializers.FloatField(
        min_value=0, max_value=1_000_000, allow_null=True, required=False
    )

    def validate_overage_enabled(self, value: bool) -> bool:
        user = self.context["user"]
        plan = budgets.plan_of(user)
        if value and (plan.overage_price_per_cpu_hour is None or budgets.is_unlimited(user)):
            raise serializers.ValidationError(
                "Your plan has no compute beyond its monthly allowance."
            )
        return value


class UsageView(AccountApiView):
    """``GET``: this month's usage, the plan, its limits and what is left
    (budgets.usage_summary; with an API token also that token's or app's own budget).
    ``PATCH {"budget_cpu_minutes"?, "overage_enabled"?, "overage_cap_cpu_hours"?}``: the
    monthly budget for the own account (null: none) and, with a plan that offers it, usage
    beyond the allowance (to be billed) up to an own cap in CPU hours (null: no cap)."""

    permission_classes = [UsagePermission]
    unthrottled_methods = ("GET", "HEAD", "OPTIONS")

    def get(self, request: Request) -> Response:
        return Response(usage_data(request))

    def patch(self, request: Request) -> Response:
        serializer = AccountBudgetSerializer(data=request.data, context={"user": request.user})
        serializer.is_valid(raise_exception=True)
        if serializer.validated_data:
            UserProfile.objects.update_or_create(
                user=request.user, defaults=serializer.validated_data
            )
        return Response(usage_data(request))


def usage_data(request: Request) -> dict[str, Any]:
    """budgets.usage_summary for the caller, after counting what its active runs used."""
    active = TrainingRun.objects.filter(owner=request.user, status__in=ACTIVE_STATUSES)
    training.sync_runs(active)  # count what the running trainings used until now
    return budgets.usage_summary(budgets.Actor.of(request))


class DeleteAccountSerializer(serializers.Serializer):
    password = serializers.CharField(write_only=True)
    username = serializers.CharField()

    def validate_password(self, value: str) -> str:
        if not self.context["user"].check_password(value):
            raise serializers.ValidationError("The password is wrong.")
        return value

    def validate_username(self, value: str) -> str:
        if value != self.context["user"].get_username():
            raise serializers.ValidationError("Type your username exactly to confirm.")
        return value


class DeleteAccountView(AccountApiView):
    """``POST {"password", "username"}``: delete the account and everything it owns."""

    def post(self, request: Request) -> Response:
        user = request.user
        serializer = DeleteAccountSerializer(data=request.data, context={"user": user})
        serializer.is_valid(raise_exception=True)
        user_id = user.pk
        delete_account(user)
        logger.info("Account #%s was deleted by its owner.", user_id)
        return Response(status=status.HTTP_204_NO_CONTENT)
