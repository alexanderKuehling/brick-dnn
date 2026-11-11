"""Short-lived links that upload a file without logging in: the way back of downloads.py.

AI assistants cannot put megabytes into a tool call, but those with a shell can run curl.
The MCP tool create_upload_link hands out ``/api/uploads/?sig=…``; a ``PUT`` of the file
(``curl -T data.csv``) or a multipart ``POST`` (``curl -F file=@data.csv``) to it imports the
file as a data source (Data tab) or stores it as a training dataset, through the same
functions, limits, checks and rate limits as an upload in the web app.

The signature (Django's ``TimestampSigner`` with the SECRET_KEY and a salt of its own) binds
the link to the user, its purpose, the file name, the import options and a random id, and
it is valid for :data:`LINK_SECONDS`. A link works once: its id is claimed in the cache
while the upload runs and kept as used after a successful one (an upload that fails, e.g. a
file that cannot be read, can be sent again). When it is used, the user must still exist
and be active. The signature is in the query string, which the access logs leave out.
"""

from __future__ import annotations

import secrets
from datetime import timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from django.conf import settings
from django.contrib.auth import get_user_model
from django.core import signing
from django.core.cache import cache
from django.core.files.uploadedfile import TemporaryUploadedFile
from django.urls import reverse
from django.utils import timezone
from rest_framework import permissions, status
from rest_framework.exceptions import PermissionDenied
from rest_framework.parsers import MultiPartParser
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.throttling import ScopedRateThrottle
from rest_framework.views import APIView

from . import data_service as data
from . import training_service as training
from .oauth import site_url

SALT = "brick_dnn_core.upload_links"
LINK_SECONDS = 15 * 60
PURPOSES = ("data_source", "training_dataset")
NEW_LINK = "Ask for a new one (MCP tool create_upload_link)."
CHUNK = 1024 * 1024


class UploadLinkError(ValueError):
    """The link cannot be created as asked; the message says why."""


def _signer() -> signing.TimestampSigner:
    return signing.TimestampSigner(salt=SALT)


def max_bytes(purpose: str) -> int:
    """The limit of the web upload for ``purpose`` (not the smaller MCP text limit)."""
    return data.max_upload_bytes() if purpose == "data_source" else training.max_upload_bytes()


def create_link(request: Request, user, purpose: Any, args: dict[str, Any]) -> dict[str, Any]:
    """A signed one-time upload link for ``user`` and what to do with it.

    ``args``: ``filename`` (decides the format), ``name``, ``project_id`` and ``kind``
    (training datasets: what a ZIP file holds, "images" or "segmentation") and the import
    options of the Data tab (data sources). Raises :class:`UploadLinkError`.
    """
    if purpose not in PURPOSES:
        raise UploadLinkError("'purpose' must be data_source or training_dataset.")
    filename = args.get("filename")
    if not isinstance(filename, str) or not Path(filename.strip()).name:
        raise UploadLinkError("'filename' must be the name of the file, e.g. sales.csv.")
    filename = Path(filename.strip()).name[:255]
    name = args.get("name")
    if name is not None and (not isinstance(name, str) or len(name.strip()) > 120):
        raise UploadLinkError("'name' must be a text of at most 120 characters.")
    payload: dict[str, Any] = {
        "u": user.pk,
        "p": purpose,
        "f": filename,
        "n": name,
        "id": secrets.token_urlsafe(12),
    }
    if purpose == "data_source":
        reason = data.unavailable_reason()
        if reason is not None:
            raise UploadLinkError(reason)
        if not data._known_suffix(filename):
            raise UploadLinkError(data.WRONG_FILE_MESSAGE)
        options = {k: args.get(k) for k in data.OPTION_FIELDS if args.get(k) not in (None, "")}
        _check_options(options)
        payload["o"] = options
        becomes = "a data source of the Data tab (then create_data_prep with its id)"
    else:
        if not settings.TRAINING_ENABLED:
            raise UploadLinkError(training.DISABLED_REASON)
        if Path(filename).suffix.lower() not in (".csv", ".zip"):
            raise UploadLinkError(training.WRONG_FILE_MESSAGE)
        kind = args.get("kind")
        if kind is not None:
            if kind not in training.UPLOAD_KINDS or Path(filename).suffix.lower() != ".zip":
                raise UploadLinkError(training.WRONG_KIND_MESSAGE)
            payload["k"] = kind
        payload["pr"] = _project_id(user, args.get("project_id"))
        becomes = "a training dataset (its reference goes to check_training / start_training)"
    signature = _signer().sign_object(payload, compress=True)
    url = f"{site_url(request)}{reverse('upload-link')}?{urlencode({'sig': signature})}"
    limit = max_bytes(purpose)
    expires_at = timezone.now() + timedelta(seconds=LINK_SECONDS)
    local = filename.replace('"', "")
    return {
        "url": url,
        "method": "PUT (the file as the body) or POST (multipart, field 'file')",
        "curl": f'curl --fail-with-body -T "{local}" "{url}"',
        "curl_multipart": f'curl --fail-with-body -F "file=@{local}" "{url}"',
        "expires_at": expires_at.isoformat(timespec="seconds"),
        "expires_in_minutes": LINK_SECONDS // 60,
        "max_mb": round(limit / 1024**2, 2),
        "becomes": becomes,
        "notes": [
            "Works once and without logging in until it expires; treat it like a password.",
            f"The file is read as {filename} (its ending decides the format), whatever the "
            "local file is called.",
            "The answer is JSON: the new item with its id, as the inline upload tools return it.",
        ],
    }


def _check_options(options: dict[str, Any]) -> None:
    from brickdnn.data.io import ImportOptions
    from brickdnn.data.table import DataError

    try:
        ImportOptions.from_dict(options)
    except DataError as exc:
        raise UploadLinkError(str(exc)) from None


def _project_id(user, project_id: Any) -> int | None:
    if project_id is None:
        return None
    from .shared_data import editable_projects

    if isinstance(project_id, bool) or not isinstance(project_id, int):
        raise UploadLinkError("'project_id' must be the number of a project.")
    if not editable_projects(user).filter(pk=project_id).exists():
        raise UploadLinkError(
            "Datasets can only be shared in projects the user owns or edits; list_projects "
            "shows the user's role in each project."
        )
    return project_id


def read_link(signature: Any) -> tuple[Any, dict[str, Any]]:
    """The user and payload of a link. Raises ``PermissionDenied`` (403)."""
    if not isinstance(signature, str) or not signature:
        raise PermissionDenied(f"This upload link is invalid. {NEW_LINK}")
    try:
        payload = _signer().unsign_object(signature, max_age=LINK_SECONDS)
    except signing.SignatureExpired:
        raise PermissionDenied(f"This upload link has expired. {NEW_LINK}") from None
    except signing.BadSignature:
        raise PermissionDenied(f"This upload link is invalid. {NEW_LINK}") from None
    if not isinstance(payload, dict) or payload.get("p") not in PURPOSES:
        raise PermissionDenied(f"This upload link is invalid. {NEW_LINK}")
    user = get_user_model().objects.filter(pk=payload.get("u"), is_active=True).first()
    if user is None:
        raise PermissionDenied("This upload link is no longer valid: the account is not active.")
    return user, payload


def _key(payload: dict[str, Any]) -> str:
    return f"brickdnn-upload-link:{payload.get('id')}"


class _UploadScope:
    throttle_scope = "uploads"


class TooLarge(Exception):
    pass


class UploadLinkView(APIView):
    """``PUT``/``POST /api/uploads/?sig=…``: the upload of a link (no login)."""

    authentication_classes: list = []
    permission_classes = [permissions.AllowAny]
    throttle_classes: list = []  # the uploads rate limit of the link's user, below
    # Multipart (curl -F); any other body (curl -T) is read as the file itself.
    parser_classes = [MultiPartParser]

    def put(self, request: Request) -> Response:
        return self._upload(request)

    def post(self, request: Request) -> Response:
        return self._upload(request)

    def _upload(self, request: Request) -> Response:
        user, payload = read_link(request.query_params.get("sig"))
        purpose = payload["p"]
        limit = max_bytes(purpose)
        too_large = Response(
            {
                "detail": f"The file is too large: upload links take at most "
                f"{round(limit / 1024**2, 2):g} MB."
            },
            status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        )
        try:
            length = int(request.META.get("CONTENT_LENGTH") or 0)
        except ValueError:
            length = 0
        if length > limit + 64 * 1024:
            return too_large
        # One use: claim the link while the upload runs.
        if not cache.add(_key(payload), "busy", timeout=LINK_SECONDS + 60):
            raise PermissionDenied(f"This upload link was already used. {NEW_LINK}")
        done = False
        try:
            request.user = user  # the rate limit counts for the link's user
            if not ScopedRateThrottle().allow_request(request, _UploadScope()):
                return Response(
                    {"detail": "Too many files were uploaded recently. Try again later."},
                    status=status.HTTP_429_TOO_MANY_REQUESTS,
                )
            try:
                upload = self._file(request, payload["f"], limit)
            except TooLarge:
                return too_large
            if upload is None:
                return Response(
                    {
                        "detail": "Send the file as the body (curl -T file) or as the multipart "
                        "field 'file' (curl -F file=@file)."
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )
            try:
                if upload.size is not None and upload.size > limit:
                    return too_large
                response = self._store(request, user, payload, upload)
            finally:
                upload.close()  # a temporary file is removed with it
            done = response.status_code < 400
            return response
        finally:
            if done:
                cache.set(_key(payload), "used", timeout=LINK_SECONDS + 60)
            else:
                cache.delete(_key(payload))

    def _file(self, request: Request, filename: str, limit: int):
        """The uploaded file: the multipart field ``file``, or the raw body (streamed into
        a temporary file, at most ``limit`` bytes). None if there is none."""
        content_type = request.content_type or ""
        if content_type.startswith("multipart/form-data"):
            upload = request.FILES.get("file")
            if upload is not None:
                upload.name = filename
            return upload
        upload = TemporaryUploadedFile(filename, "application/octet-stream", 0, None)
        size = 0
        stream = request._request
        while chunk := stream.read(CHUNK):
            size += len(chunk)
            if size > limit:
                upload.close()
                raise TooLarge
            upload.write(chunk)
        if size == 0:
            upload.close()
            return None
        upload.size = size
        upload.seek(0)
        return upload

    def _store(self, request: Request, user, payload: dict[str, Any], upload) -> Response:
        from . import mcp_compact, mcp_data

        if payload["p"] == "data_source":
            reason = data.unavailable_reason()
            if reason is not None:
                return Response({"detail": reason}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
            fields = {"name": payload.get("n"), **(payload.get("o") or {})}
            try:
                source = data.import_source(user, upload, fields)
            except data.DataInputError as exc:
                body: dict[str, Any] = {"detail": str(exc)}
                if exc.sheets:
                    body["sheets"] = exc.sheets
                return Response(body, status=status.HTTP_400_BAD_REQUEST)
            return Response(
                {
                    "source": mcp_data.source_detail(source, user),
                    "next": f"create_data_prep with source_id {source.pk} and the target column.",
                },
                status=status.HTTP_201_CREATED,
            )
        if not settings.TRAINING_ENABLED:
            return Response(
                {"detail": training.DISABLED_REASON}, status=status.HTTP_503_SERVICE_UNAVAILABLE
            )
        from brickdnn.training import DatasetError

        from .shared_data import editable_projects

        project = None
        if payload.get("pr") is not None:
            project = editable_projects(user).filter(pk=payload["pr"]).first()
            if project is None:
                return Response(
                    {"detail": "The user can no longer share datasets in this project."},
                    status=status.HTTP_403_FORBIDDEN,
                )
        try:
            dataset = training.save_upload(user, upload, payload.get("n"), payload.get("k"))
        except DatasetError as exc:
            return Response(
                {"detail": f"The file cannot be used: {exc}"}, status=status.HTTP_400_BAD_REQUEST
            )
        if project is not None:
            dataset.projects.add(project)
        return Response(
            {
                "dataset": mcp_compact.dataset_detail(dataset, user),
                "reference": {"kind": "upload", "id": dataset.pk},
                "next": "Pass the reference as dataset to check_training and start_training.",
            },
            status=status.HTTP_201_CREATED,
        )
