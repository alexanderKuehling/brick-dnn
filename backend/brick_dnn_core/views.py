import re

from django.conf import settings
from django.http import HttpResponse
from rest_framework import permissions, status, viewsets
from rest_framework.decorators import api_view, permission_classes
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.throttling import ScopedRateThrottle
from rest_framework.views import APIView
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework_simplejwt.views import TokenObtainPairView

import brickdnn
from brickdnn import EXAMPLES, GeneratorOptions, GraphValidationError, analyze, generate_code
from brickdnn.layers import CATEGORIES, layer_specs

from .models import Project
from .serializers import ProjectSerializer, RegisterSerializer, UserSerializer


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
                },
            }
        )
