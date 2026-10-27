from django.contrib.auth import get_user_model
from django.contrib.auth.password_validation import validate_password
from rest_framework import serializers

from brickdnn.graph import MAX_EDGES, MAX_NODES

from .models import Project

User = get_user_model()


def validate_graph_document(value):
    """Light structural check. Semantic errors are allowed: users may save work in progress."""
    if not isinstance(value, dict):
        raise serializers.ValidationError("The graph must be a JSON object.")
    nodes, edges = value.get("nodes", []), value.get("edges", [])
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise serializers.ValidationError("'nodes' and 'edges' must be lists.")
    if len(nodes) > MAX_NODES or len(edges) > MAX_EDGES:
        raise serializers.ValidationError(
            f"Graphs are limited to {MAX_NODES} layers and {MAX_EDGES} connections."
        )
    return value


class ProjectSerializer(serializers.ModelSerializer):
    num_layers = serializers.SerializerMethodField()

    class Meta:
        model = Project
        fields = ["id", "name", "description", "graph", "num_layers", "created_at", "updated_at"]
        read_only_fields = ["id", "created_at", "updated_at"]

    def get_num_layers(self, obj: Project) -> int:
        nodes = obj.graph.get("nodes", []) if isinstance(obj.graph, dict) else []
        return len(nodes)

    def validate_graph(self, value):
        return validate_graph_document(value)


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ["id", "username", "email", "date_joined"]


class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, style={"input_type": "password"})
    email = serializers.EmailField(required=False, allow_blank=True)

    class Meta:
        model = User
        fields = ["username", "email", "password"]

    def validate(self, attrs):
        candidate = User(username=attrs.get("username"), email=attrs.get("email", ""))
        validate_password(attrs["password"], user=candidate)
        return attrs

    def create(self, validated_data):
        return User.objects.create_user(**validated_data)
