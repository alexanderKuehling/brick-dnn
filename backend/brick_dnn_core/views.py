import json
import re

from django.conf import settings
from django.http import FileResponse, HttpResponse
from django.utils.text import slugify
from rest_framework import mixins, permissions, renderers, status, viewsets
from rest_framework.decorators import action, api_view, permission_classes
from rest_framework.exceptions import APIException
from rest_framework.parsers import MultiPartParser
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.throttling import ScopedRateThrottle
from rest_framework.views import APIView
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework_simplejwt.views import TokenObtainPairView

import brickdnn
from brickdnn import EXAMPLES, GeneratorOptions, GraphValidationError, analyze, generate_code
from brickdnn.layers import CATEGORIES, layer_specs
from brickdnn.training import ACTIVE_STATUSES, DatasetError

from . import training_service as training
from .models import Dataset, Project, TrainingRun
from .serializers import (
    DatasetSerializer,
    ProjectSerializer,
    RegisterSerializer,
    RunDetailSerializer,
    RunSummarySerializer,
    UserSerializer,
)


def _graph_from_request(request: Request):
    data = request.data
    if isinstance(data, dict) and "graph" in data:
        return data["graph"]
    return data


# --------------------------------------------------------------------------- #
# Code generation (public, no login needed)
# --------------------------------------------------------------------------- #


@api_view(["GET"])
def health(request: Request) -> Response:
    return Response({"status": "ok", "version": brickdnn.__version__})


@api_view(["GET"])
def layers(request: Request) -> Response:
    """All layer types with their parameters: the UI builds its palette and forms from this."""
    return Response({"categories": list(CATEGORIES), "layers": layer_specs()})


@api_view(["GET"])
def examples(request: Request) -> Response:
    return Response(list(EXAMPLES.values()))


@api_view(["POST"])
def analyze_graph(request: Request) -> Response:
    """Validate a graph and return the inferred shape and parameter count of every layer."""
    return Response(analyze(_graph_from_request(request)).to_dict())


def _filename(raw) -> str:
    stem = str(raw or "model").strip().removesuffix(".py")
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")[:60] or "model"
    return stem + ".py"


@api_view(["POST"])
def generate(request: Request) -> Response | HttpResponse:
    """Generate a PyTorch script.

    Body: ``{"graph": {...}, "options": {"class_name": "...", "include_main": true,
    "include_training": true, "filename": "model"}}``. Add ``?download=1`` to get the
    file as an attachment instead of JSON.
    """
    graph = _graph_from_request(request)
    raw_options = request.data.get("options") if isinstance(request.data, dict) else None
    if raw_options is not None and not isinstance(raw_options, dict):
        return Response({"detail": "'options' must be an object."}, status=400)
    try:
        options = GeneratorOptions.from_dict(raw_options)
        code = generate_code(graph, options)
    except GraphValidationError as exc:
        return Response(
            {"detail": str(exc), "analysis": exc.analysis.to_dict()},
            status=status.HTTP_400_BAD_REQUEST,
        )
    except ValueError as exc:
        return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

    filename = _filename((raw_options or {}).get("filename"))
    if request.query_params.get("download"):
        response = HttpResponse(code, content_type="text/x-python; charset=utf-8")
        response["Content-Disposition"] = f'attachment; filename="{filename}"'
        return response
    return Response({"code": code, "filename": filename})


# --------------------------------------------------------------------------- #
# Accounts
# --------------------------------------------------------------------------- #


def _tokens_for(user) -> dict:
    refresh = RefreshToken.for_user(user)
    return {"access": str(refresh.access_token), "refresh": str(refresh)}


class LoginView(TokenObtainPairView):
    throttle_classes = [ScopedRateThrottle]
    throttle_scope = "auth"


class RegisterView(APIView):
    throttle_classes = [ScopedRateThrottle]
    throttle_scope = "auth"

    def post(self, request: Request) -> Response:
        if not settings.ALLOW_REGISTRATION:
            return Response(
                {"detail": "Registration is disabled on this server."},
                status=status.HTTP_403_FORBIDDEN,
            )
        serializer = RegisterSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()
        return Response(
            {"user": UserSerializer(user).data, **_tokens_for(user)},
            status=status.HTTP_201_CREATED,
        )


@api_view(["GET"])
@permission_classes([permissions.IsAuthenticated])
def me(request: Request) -> Response:
    return Response(UserSerializer(request.user).data)


# --------------------------------------------------------------------------- #
# Saved projects
# --------------------------------------------------------------------------- #


class ProjectViewSet(viewsets.ModelViewSet):
    serializer_class = ProjectSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return Project.objects.filter(owner=self.request.user)

    def perform_create(self, serializer):
        serializer.save(owner=self.request.user)


# --------------------------------------------------------------------------- #
# Local training
# --------------------------------------------------------------------------- #


class Conflict(APIException):
    status_code = status.HTTP_409_CONFLICT
    default_detail = "The request conflicts with the current state."
    default_code = "conflict"


class BinaryRenderer(renderers.BaseRenderer):
    """Lets clients ask for ``application/octet-stream`` (errors are still JSON)."""

    media_type = "application/octet-stream"
    format = "bin"
    charset = None
    render_style = "binary"

    def render(self, data, accepted_media_type=None, renderer_context=None):
        if isinstance(data, bytes):
            return data
        return json.dumps(data).encode()


def _disabled_response() -> Response | None:
    if not settings.TRAINING_ENABLED:
        return Response(
            {"detail": training.DISABLED_REASON}, status=status.HTTP_503_SERVICE_UNAVAILABLE
        )
    return None


@api_view(["GET"])
def training_info(request: Request) -> Response:
    """Whether training is possible on this server: PyTorch version, devices, datasets."""
    return Response(training.capabilities())


@api_view(["POST"])
@permission_classes([permissions.IsAuthenticated])
def training_check(request: Request) -> Response:
    """Check a network against a dataset before training: ``{"graph", "dataset"}``."""
    if (response := _disabled_response()) is not None:
        return response
    data = request.data if isinstance(request.data, dict) else {}
    graph = data.get("graph")
    try:
        training.validate_graph(graph)
        dataset = training.resolve_dataset(request.user, data.get("dataset"))
    except training.TrainingInputError as exc:
        return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
    return Response(training.check(graph, dataset).to_dict())


class DatasetViewSet(
    mixins.ListModelMixin,
    mixins.CreateModelMixin,
    mixins.RetrieveModelMixin,
    mixins.DestroyModelMixin,
    viewsets.GenericViewSet,
):
    """Uploaded training data: a CSV file or a ZIP file with one folder per class."""

    serializer_class = DatasetSerializer
    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [MultiPartParser]
    throttle_scope = "uploads"

    def get_queryset(self):
        return Dataset.objects.filter(owner=self.request.user)

    def get_throttles(self):
        throttles = super().get_throttles()
        if self.action == "create":
            throttles.append(ScopedRateThrottle())
        return throttles

    def create(self, request: Request, *args, **kwargs) -> Response:
        if (response := _disabled_response()) is not None:
            return response
        limit = training.max_upload_bytes()
        too_large = Response(
            {
                "detail": "The file is too large. Uploads are limited to "
                f"{settings.TRAINING_MAX_UPLOAD_MB} MB."
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
        name = request.data.get("name")
        try:
            dataset = training.save_upload(
                request.user, upload, name if isinstance(name, str) else None
            )
        except DatasetError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        return Response(DatasetSerializer(dataset).data, status=status.HTTP_201_CREATED)

    def perform_destroy(self, instance: Dataset) -> None:
        if training.dataset_in_use(instance):
            raise Conflict(
                "This dataset is used by a training that is still running. Stop it first."
            )
        instance.delete()  # the files are removed by a post_delete signal


class TrainingRunViewSet(
    mixins.ListModelMixin,
    mixins.RetrieveModelMixin,
    mixins.DestroyModelMixin,
    viewsets.GenericViewSet,
):
    """Training runs of the logged-in user. Training happens in a separate process."""

    permission_classes = [permissions.IsAuthenticated]
    throttle_scope = "training"

    def get_queryset(self):
        return TrainingRun.objects.filter(owner=self.request.user)

    def get_serializer_class(self):
        return RunSummarySerializer if self.action == "list" else RunDetailSerializer

    def get_throttles(self):
        throttles = super().get_throttles()
        if self.action == "create":
            throttles.append(ScopedRateThrottle())
        return throttles

    def get_object(self) -> TrainingRun:
        return training.sync_run(super().get_object())

    def list(self, request: Request, *args, **kwargs) -> Response:
        training.sync_runs(self.get_queryset().filter(status__in=ACTIVE_STATUSES))
        return super().list(request, *args, **kwargs)

    def create(self, request: Request) -> Response:
        """Start a training: ``{"name", "graph", "dataset", "config", "tuning"?}``."""
        data = request.data if isinstance(request.data, dict) else {}
        try:
            run = training.start_run(
                request.user,
                data.get("name"),
                data.get("graph"),
                data.get("dataset"),
                data.get("config"),
                data.get("tuning"),
            )
        except training.TrainingUnavailable as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        except training.TrainingInputError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        except training.IncompatibleNetwork as exc:
            result = exc.compatibility
            return Response(
                {"detail": str(exc), "errors": result.errors, "warnings": result.warnings},
                status=status.HTTP_400_BAD_REQUEST,
            )
        except training.TrainingBusy as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_409_CONFLICT)
        return Response(RunDetailSerializer(run).data, status=status.HTTP_201_CREATED)

    def perform_destroy(self, instance: TrainingRun) -> None:
        training.terminate_run(instance)
        instance.delete()  # the run directory is removed by a post_delete signal

    @action(detail=True, methods=["post"])
    def stop(self, request: Request, pk=None) -> Response:
        """Stop after the current batch; the weights trained so far are kept."""
        run = training.stop_run(self.get_object())
        return Response(RunDetailSerializer(run).data)

    @action(
        detail=True,
        methods=["get"],
        renderer_classes=[renderers.JSONRenderer, BinaryRenderer],
    )
    def weights(self, request: Request, pk=None) -> Response | FileResponse:
        """The trained weights (``model.state_dict()``) as a ``.pt`` file."""
        run = self.get_object()
        path = training.weights_path(run)
        if path is None:
            return Response(
                {"detail": "No weights available for this run."},
                status=status.HTTP_404_NOT_FOUND,
            )
        filename = (slugify(run.name) or "weights") + ".pt"
        return FileResponse(
            path.open("rb"),
            as_attachment=True,
            filename=filename,
            content_type="application/octet-stream",
        )


class ApiRoot(APIView):
    def get(self, request: Request) -> Response:
        base = request.build_absolute_uri("/api/")
        return Response(
            {
                "name": "Brick DNN API",
                "version": brickdnn.__version__,
                "endpoints": {
                    "layers": f"{base}layers/",
                    "examples": f"{base}examples/",
                    "analyze": f"{base}analyze/",
                    "generate": f"{base}generate/",
                    "register": f"{base}auth/register/",
                    "login": f"{base}auth/token/",
                    "projects": f"{base}projects/",
                    "training": f"{base}training/",
                    "training_check": f"{base}training/check/",
                    "training_datasets": f"{base}training/datasets/",
                    "training_runs": f"{base}training/runs/",
                },
            }
        )
