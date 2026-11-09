"""API of the Data tab: imports (data sources), preparations, previews and exports.

The logic lives in :mod:`brick_dnn_core.data_service`. Every endpoint except the status and
the step catalog needs a login, and all of them except the status answer 503 when data
preparation is switched off or pandas is not installed.
"""

from __future__ import annotations

from django.conf import settings
from django.db.models import Prefetch, Q, RestrictedError
from django.http import FileResponse, HttpResponse
from rest_framework import mixins, permissions, status, viewsets
from rest_framework.decorators import action, api_view
from rest_framework.exceptions import APIException, PermissionDenied, ValidationError
from rest_framework.negotiation import DefaultContentNegotiation
from rest_framework.parsers import MultiPartParser
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.throttling import ScopedRateThrottle

from . import data_service as data
from . import training_service as training
from .models import DataPrep, Dataset, DataSource
from .parsers import SafeJSONParser
from .serializers import DataPrepSerializer, DatasetSerializer, DataSourceSerializer
from .views import Conflict


class ServiceUnavailable(APIException):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    default_detail = "Data preparation is not available on this server."
    default_code = "unavailable"


class IgnoreFormatParam(DefaultContentNegotiation):
    """The download endpoint uses ``?format=`` for the file type, not to choose a renderer
    (files bypass rendering; errors are JSON)."""

    def select_renderer(self, request, renderers, format_suffix=None):
        return renderers[0], renderers[0].media_type


def _check_available() -> None:
    reason = data.unavailable_reason()
    if reason is not None:
        raise ServiceUnavailable(reason)


def _bad_request(exc: data.DataInputError) -> Response:
    body: dict = {"detail": str(exc)}
    if exc.issues is not None:
        body["issues"] = exc.issues
    return Response(body, status=status.HTTP_400_BAD_REQUEST)


def _first_message(errors) -> str:
    """The first message of DRF validation errors, e.g. "Name: This field is required."."""
    if isinstance(errors, list) and errors:
        return _first_message(errors[0])
    if isinstance(errors, dict) and errors:
        field, value = next(iter(errors.items()))
        message = _first_message(value)
        if field in ("non_field_errors", "steps", "settings"):
            return message  # these messages are complete sentences
        return f"{str(field).replace('_', ' ').capitalize()}: {message}"
    return str(errors)


@api_view(["GET"])
def data_status(request: Request) -> Response:
    """Whether data preparation works on this server, and its limits (public)."""
    return Response(data.status())


@api_view(["GET"])
def data_steps(request: Request) -> Response:
    """All step types with their settings, for the step menu and forms (public)."""
    _check_available()
    return Response(data.step_catalog())


class DataViewSet(viewsets.GenericViewSet):
    """Common behaviour: login, availability check, ``{"detail"}`` for validation errors."""

    permission_classes = [permissions.IsAuthenticated]
    http_method_names = ["get", "post", "patch", "delete", "head", "options"]

    def initial(self, request: Request, *args, **kwargs) -> None:
        super().initial(request, *args, **kwargs)  # authentication, permissions, throttles
        _check_available()
        if request.method in ("POST", "PATCH") and request.content_type.startswith(
            "application/json"
        ):
            from brickdnn.data.steps.base import valid_text

            if not valid_text(request.data):
                raise ValidationError(
                    {"detail": "The request contains characters that are not valid text."}
                )

    def handle_exception(self, exc):
        response = super().handle_exception(exc)
        errors = response.data
        if isinstance(exc, ValidationError) and isinstance(errors, dict) and "detail" not in errors:
            response.data = {"detail": _first_message(errors), **errors}
        return response

    def scoped_throttle(self, scope: str) -> list:
        """The default throttles plus ScopedRateThrottle for ``scope``."""
        self.throttle_scope = scope
        return [*super().get_throttles(), ScopedRateThrottle()]


class DataSourceViewSet(
    mixins.ListModelMixin,
    mixins.RetrieveModelMixin,
    mixins.UpdateModelMixin,
    mixins.DestroyModelMixin,
    DataViewSet,
):
    """Imported CSV files. POST is a multipart upload; PATCH changes the name."""

    serializer_class = DataSourceSerializer
    parser_classes = [SafeJSONParser, MultiPartParser]

    def get_queryset(self):
        user = self.request.user
        if self.action == "retrieve":
            # Also the sources of preparations shared with you (read only).
            from .shared_data import visible_preps

            used = visible_preps(user).values("source")
            return DataSource.objects.filter(Q(owner=user) | Q(pk__in=used))
        return DataSource.objects.filter(owner=user)

    def get_throttles(self):
        if self.action == "create":
            return self.scoped_throttle("uploads")
        return super().get_throttles()

    def create(self, request: Request, *args, **kwargs) -> Response:
        limit = data.max_upload_bytes()
        too_large = Response(
            {
                "detail": "The file is too large. Uploads are limited to "
                f"{settings.DATA_MAX_UPLOAD_MB:g} MB."
            },
            status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        )
        # Reject before the body is read (allowing a little room for the form fields).
        try:
            length = int(request.META.get("CONTENT_LENGTH") or 0)
        except ValueError:
            length = 0
        if length > limit + 64 * 1024:
            return too_large
        upload = request.FILES.get("file")
        if upload is None:
            return Response(
                {"detail": "Choose a file to upload."}, status=status.HTTP_400_BAD_REQUEST
            )
        if upload.size is not None and upload.size > limit:
            return too_large
        fields = {key: request.data.get(key) for key in ("name", *data.OPTION_FIELDS)}
        try:
            source = data.import_source(request.user, upload, fields)
        except data.DataInputError as exc:
            return _bad_request(exc)
        return Response(DataSourceSerializer(source).data, status=status.HTTP_201_CREATED)

    def perform_destroy(self, instance: DataSource) -> None:
        preps = data.preps_using(instance)
        if preps:
            raise Conflict(data.in_use_message(preps))
        try:
            instance.delete()  # the files are removed by a post_delete signal
        except RestrictedError:  # a preparation was created in the meantime
            raise Conflict(data.in_use_message(data.preps_using(instance))) from None


class DataPrepViewSet(
    mixins.ListModelMixin,
    mixins.CreateModelMixin,
    mixins.RetrieveModelMixin,
    mixins.UpdateModelMixin,
    mixins.DestroyModelMixin,
    DataViewSet,
):
    """Data preparations: a source, steps and settings; preview, export and downloads."""

    serializer_class = DataPrepSerializer

    def get_queryset(self):
        from .shared_data import visible_preps

        user = self.request.user
        # Your own preparations and the ones shared in your projects; only your own can be
        # deleted.
        preps = (
            DataPrep.objects.filter(owner=user) if self.action == "destroy" else visible_preps(user)
        )
        return preps.select_related("source", "owner").prefetch_related(
            Prefetch("datasets", queryset=Dataset.objects.order_by("-created_at", "-id"))
        )

    def get_throttles(self):
        if self.action in ("preview", "download"):
            return self.scoped_throttle("data")
        if self.action == "export":
            return self.scoped_throttle("uploads")
        return super().get_throttles()

    def perform_create(self, serializer) -> None:
        serializer.save(owner=self.request.user)

    def get_object(self) -> DataPrep:
        from .shared_data import can_edit_prep

        prep = super().get_object()
        if self.action in ("update", "partial_update", "export") and not can_edit_prep(
            self.request.user, prep
        ):
            raise PermissionDenied("This data preparation was shared with you read-only.")
        return prep

    @action(detail=True, methods=["post"])
    def preview(self, request: Request, pk=None) -> Response:
        """Run the (unsaved) pipeline: step reports, profile, sample and readiness."""
        prep = self.get_object()
        try:
            return Response(data.preview(prep, request.data))
        except data.DataInputError as exc:
            return _bad_request(exc)

    @action(detail=True, methods=["post"])
    def export(self, request: Request, pk=None) -> Response:
        """Create a training dataset from the saved preparation: ``{"name"?}``."""
        if not settings.TRAINING_ENABLED:
            return Response(
                {"detail": training.DISABLED_REASON}, status=status.HTTP_503_SERVICE_UNAVAILABLE
            )
        prep = self.get_object()
        body = request.data if isinstance(request.data, dict) else {}
        try:
            dataset = data.export_dataset(prep, body.get("name"))
        except data.DataInputError as exc:
            return _bad_request(exc)
        return Response(
            DatasetSerializer(dataset, context={"request": request}).data,
            status=status.HTTP_201_CREATED,
        )

    @action(detail=True, methods=["get"], content_negotiation_class=IgnoreFormatParam)
    def download(self, request: Request, pk=None) -> Response | HttpResponse:
        """``?format=csv|recipe|python``: the prepared data, the recipe or a pandas script."""
        prep = self.get_object()
        try:
            file = data.download(prep, request.query_params.get("format", "csv"))
        except data.DataInputError as exc:
            return _bad_request(exc)
        if file.handle is not None:
            return FileResponse(
                file.handle,
                as_attachment=True,
                filename=file.filename,
                content_type=file.content_type,
            )
        response = HttpResponse(file.content, content_type=file.content_type)
        response["Content-Disposition"] = f'attachment; filename="{file.filename}"'
        return response
