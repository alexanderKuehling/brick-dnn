"""Django settings for the Brick DNN API.

All deployment-specific values come from environment variables, see
``.env.example`` in the repository root.
"""

from __future__ import annotations

import os
import secrets
from datetime import timedelta
from pathlib import Path

from django.core.exceptions import ImproperlyConfigured

from .database import database_settings

BASE_DIR = Path(__file__).resolve().parent.parent


def env_bool(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def env_list(name: str, default: str = "") -> list[str]:
    return [item.strip() for item in os.environ.get(name, default).split(",") if item.strip()]


DEBUG = env_bool("DJANGO_DEBUG", False)
DATA_DIR = Path(os.environ.get("BRICKDNN_DATA_DIR", BASE_DIR / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _secret_key() -> str:
    """Use DJANGO_SECRET_KEY, or create a random key once and keep it in DATA_DIR."""
    key = os.environ.get("DJANGO_SECRET_KEY")
    if key:
        return key
    key_file = DATA_DIR / "secret_key"
    if key_file.exists():
        return key_file.read_text().strip()
    key = secrets.token_urlsafe(50)
    try:
        key_file.write_text(key)
        key_file.chmod(0o600)
    except OSError as exc:  # pragma: no cover - read-only file systems
        raise ImproperlyConfigured("Set DJANGO_SECRET_KEY or make DATA_DIR writable.") from exc
    return key


SECRET_KEY = _secret_key()

# localhost is always allowed so that health checks inside the container work.
ALLOWED_HOSTS = [
    *env_list("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1,[::1]"),
    "localhost",
    "127.0.0.1",
]
CSRF_TRUSTED_ORIGINS = env_list("DJANGO_CSRF_TRUSTED_ORIGINS")
CORS_ALLOWED_ORIGINS = env_list(
    "CORS_ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
)

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "corsheaders",
    "rest_framework",
    "rest_framework_simplejwt",
    "rest_framework_simplejwt.token_blacklist",
    "brick_dnn_core",
]

MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "brick_dnn_site.urls"
WSGI_APPLICATION = "brick_dnn_site.wsgi.application"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

# A SQLite file in DATA_DIR, or PostgreSQL (e.g. Supabase) if DATABASE_URL is set:
# postgresql://USER:PASSWORD@HOST:PORT/NAME, see docs/database.md.
DATABASES = {
    "default": database_settings(
        os.environ.get("DATABASE_URL", ""),
        sqlite_path=os.environ.get("BRICKDNN_DB_PATH", DATA_DIR / "db.sqlite3"),
        conn_max_age=int(os.environ.get("BRICKDNN_DB_CONN_MAX_AGE", "60")),
    )
}

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STORAGES = {
    "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
    "staticfiles": {"BACKEND": "whitenoise.storage.CompressedStaticFilesStorage"},
}

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# A file cache is shared by all gunicorn workers, so rate limits are counted correctly.
CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.filebased.FileBasedCache",
        "LOCATION": DATA_DIR / "cache",
    }
}

# Graphs are small JSON documents; 1 MB is plenty and limits abuse. (File uploads do not
# count: they are streamed to temporary files and limited by TRAINING_MAX_UPLOAD_MB.)
DATA_UPLOAD_MAX_MEMORY_SIZE = 1024 * 1024

# Local training: networks are trained in a separate process per run (needs PyTorch).
# Set BRICKDNN_TRAINING=false to switch the feature off.
TRAINING_ENABLED = env_bool("BRICKDNN_TRAINING", True)
TRAINING_DIR = DATA_DIR / "training"
# Number of trainings that may run at the same time (counted across all users).
TRAINING_MAX_RUNS = int(os.environ.get("BRICKDNN_TRAINING_MAX_RUNS", "1"))
# Largest dataset upload (CSV or ZIP) in megabytes.
TRAINING_MAX_UPLOAD_MB = int(os.environ.get("BRICKDNN_MAX_UPLOAD_MB", "500"))
# Command that trains one run; the run directory is appended as the last argument.
# None means [sys.executable, "-m", "brickdnn.training.runner"] (tests use a fake runner).
TRAINING_RUNNER_COMMAND: list[str] | None = None

# Data preparation (the Data tab, needs pandas): imported CSV files and their tables live in
# DATA_PREP_DIR. Uploads are limited by DATA_MAX_UPLOAD_MB.
# Set BRICKDNN_DATA_PREP=false to switch the feature off.
DATA_PREP_ENABLED = env_bool("BRICKDNN_DATA_PREP", True)
DATA_PREP_DIR = DATA_DIR / "prepare"
# Largest table that can be imported.
DATA_MAX_ROWS = int(os.environ.get("BRICKDNN_DATA_MAX_ROWS", "2000000"))
# Most columns per imported file; 0 or unset: no limit.
DATA_MAX_COLUMNS = int(os.environ.get("BRICKDNN_DATA_MAX_COLUMNS", "0")) or None
# Largest CSV import in megabytes. Imports are parsed in memory (roughly 10-15 times the file
# size while reading), so this is lower than the training upload limit.
DATA_MAX_UPLOAD_MB = float(os.environ.get("BRICKDNN_DATA_MAX_UPLOAD_MB", "200"))
# Rows in a preview sample (the data table and the charts): default and maximum.
DATA_SAMPLE_ROWS = 5000
DATA_MAX_SAMPLE_ROWS = 20000

REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": [
        # Personal API tokens (bdnn_…) first; other bearer tokens are JWT logins.
        "brick_dnn_core.api_tokens.ApiTokenAuthentication",
        "rest_framework_simplejwt.authentication.JWTAuthentication",
    ],
    "DEFAULT_PERMISSION_CLASSES": ["rest_framework.permissions.AllowAny"],
    "DEFAULT_PARSER_CLASSES": ["brick_dnn_core.parsers.SafeJSONParser"],
    # Number of reverse proxies in front of Django (1 with the bundled nginx). Rate limits
    # then use the address added by the proxy instead of a client-supplied X-Forwarded-For.
    "NUM_PROXIES": int(os.environ.get("BRICKDNN_NUM_PROXIES", "0")),
    "DEFAULT_RENDERER_CLASSES": ["rest_framework.renderers.JSONRenderer"]
    + (["rest_framework.renderers.BrowsableAPIRenderer"] if DEBUG else []),
    "DEFAULT_THROTTLE_CLASSES": [
        "rest_framework.throttling.AnonRateThrottle",
        "rest_framework.throttling.UserRateThrottle",
    ],
    "DEFAULT_THROTTLE_RATES": {
        "anon": os.environ.get("BRICKDNN_ANON_RATE", "300/minute"),
        "user": os.environ.get("BRICKDNN_USER_RATE", "600/minute"),
        "auth": os.environ.get("BRICKDNN_AUTH_RATE", "20/minute"),
        "training": os.environ.get("BRICKDNN_TRAINING_RATE", "120/hour"),
        "uploads": os.environ.get("BRICKDNN_UPLOAD_RATE", "60/hour"),
        # Previews of data preparations (one per change in the Data tab).
        "data": os.environ.get("BRICKDNN_DATA_RATE", "1200/hour"),
        # Requests to the MCP server (one per tool call of an AI assistant).
        "mcp": os.environ.get("BRICKDNN_MCP_RATE", "600/hour"),
    },
}

SIMPLE_JWT = {
    "ACCESS_TOKEN_LIFETIME": timedelta(minutes=30),
    "REFRESH_TOKEN_LIFETIME": timedelta(days=7),
    "ROTATE_REFRESH_TOKENS": True,
    "BLACKLIST_AFTER_ROTATION": True,
    "UPDATE_LAST_LOGIN": True,
}

# Set to "false" to disable public sign-up (e.g. for a private instance).
ALLOW_REGISTRATION = env_bool("BRICKDNN_ALLOW_REGISTRATION", True)

if not DEBUG:
    SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
    SESSION_COOKIE_SECURE = env_bool("DJANGO_SECURE_COOKIES", False)
    CSRF_COOKIE_SECURE = SESSION_COOKIE_SECURE
    SECURE_CONTENT_TYPE_NOSNIFF = True
    X_FRAME_OPTIONS = "DENY"

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {"console": {"class": "logging.StreamHandler"}},
    "root": {"handlers": ["console"], "level": os.environ.get("DJANGO_LOG_LEVEL", "INFO")},
}
