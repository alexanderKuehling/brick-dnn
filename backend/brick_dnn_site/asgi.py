"""ASGI entry point: HTTP like WSGI, plus WebSockets for live editing (brick_dnn_core.live)."""

import os

from django.core.asgi import get_asgi_application

from brick_dnn_site.env_file import load_env_file

load_env_file()  # settings from the .env file in the repository root, if it exists
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "brick_dnn_site.settings")

# Set Django up before importing anything that uses models.
django_application = get_asgi_application()

from channels.routing import ProtocolTypeRouter, URLRouter  # noqa: E402
from channels.security.websocket import AllowedHostsOriginValidator  # noqa: E402
from django.urls import path  # noqa: E402

from brick_dnn_core.live import ProjectConsumer  # noqa: E402

application = ProtocolTypeRouter(
    {
        "http": django_application,
        # Only pages of this site may open the socket (the Origin must be an allowed host).
        "websocket": AllowedHostsOriginValidator(
            URLRouter([path("api/ws/projects/<int:project_id>/", ProjectConsumer.as_asgi())])
        ),
    }
)
