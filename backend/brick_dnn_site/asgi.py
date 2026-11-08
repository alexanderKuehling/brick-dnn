import os

from django.core.asgi import get_asgi_application

from brick_dnn_site.env_file import load_env_file

load_env_file()  # settings from the .env file in the repository root, if it exists
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "brick_dnn_site.settings")

application = get_asgi_application()
