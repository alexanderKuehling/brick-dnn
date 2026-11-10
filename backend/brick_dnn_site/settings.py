"""Django settings for the Brick DNN API.

All deployment-specific values come from environment variables, see
``.env.example`` in the repository root.
"""

from __future__ import annotations

import json
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


def load_site_settings(environ=os.environ) -> None:
    """Defaults from the file ``BRICKDNN_SITE_ENV`` names: ``BRICKDNN_…=value`` lines kept in
    the repository for one public server (``deploy/sites/<domain>/settings.env``, see
    deploy/sites/README.md). A variable that is set and not empty in the environment wins."""
    path = environ.get("BRICKDNN_SITE_ENV")
    if not path:
        return
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return  # no settings for this domain
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = (part.strip() for part in line.split("=", 1))
        if name.startswith("BRICKDNN_") and not environ.get(name):
            environ[name] = value


load_site_settings()

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
    # First: `manage.py runserver` then also serves WebSockets (live editing).
    "daphne",
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
    # Logs what AI agents do with API tokens (after the response, see agent_activity.py).
    "brick_dnn_core.agent_activity.AgentActivityMiddleware",
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
# Live editing (brick_dnn_core.live). Several server processes need Redis to reach each other;
# without REDIS_URL, messages stay inside one process (fine for `manage.py runserver`).
ASGI_APPLICATION = "brick_dnn_site.asgi.application"
REDIS_URL = os.environ.get("REDIS_URL", "")
CHANNEL_LAYERS = {
    "default": (
        {
            "BACKEND": "channels_redis.core.RedisChannelLayer",
            "CONFIG": {"hosts": [REDIS_URL], "capacity": 500, "expiry": 30},
        }
        if REDIS_URL
        else {"BACKEND": "channels.layers.InMemoryChannelLayer"}
    )
}

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
# Predictions with trained weights run in a separate process too, at most this many at the
# same time (counted across all users). The command gets the run directory as its last
# argument; None means [sys.executable, "-m", "brickdnn.training.predict"].
TRAINING_MAX_PREDICTIONS = int(os.environ.get("BRICKDNN_MAX_PREDICTIONS", "1"))
TRAINING_PREDICT_COMMAND: list[str] | None = None
# Largest CSV text that AI assistants can upload with the MCP tool upload_dataset, in MB (at
# most BRICKDNN_MAX_UPLOAD_MB). The MCP endpoint accepts requests of about twice this size
# (JSON escapes the text), unlike other JSON requests (DATA_UPLOAD_MAX_MEMORY_SIZE).
MCP_UPLOAD_MAX_MB = float(os.environ.get("BRICKDNN_MCP_UPLOAD_MB", "5"))


def env_number(name: str, default: float = 0.0) -> float:
    """A number from the environment; empty or unset: ``default``."""
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ImproperlyConfigured(f"{name} must be a number, not {value!r}.") from exc


def env_json(name: str):
    """JSON from the environment; empty or unset: None."""
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    try:
        return json.loads(value)
    except ValueError as exc:
        raise ImproperlyConfigured(f"{name} must be JSON: {exc}") from exc


# Budgets and cost limits per account (brick_dnn_core.budgets, docs/training.md). Compute is
# CPU time of trainings and predictions, counted per calendar month (UTC).
#
# An account's limits come from its plan. BRICKDNN_PLANS defines the plans as JSON, e.g.
#   {"free": {"name": "Free", "price": 0, "cpu_hours_per_month": 0.5, "storage_gb": 1,
#             "cpu_minutes_per_run": 15, "concurrent_trainings": 1, "priority": 0}, ...}
# (see docs/training.md; a missing or 0 limit is unlimited), BRICKDNN_DEFAULT_PLAN the plan of
# accounts that have none set (Django admin, `manage.py set_plan`). Without BRICKDNN_PLANS
# there is one plan with the limits below; 0 or unset: unlimited (the default, e.g. for a
# server of your own). Users can set lower budgets for their account and their API tokens.
PLANS = env_json("BRICKDNN_PLANS")
DEFAULT_PLAN = os.environ.get("BRICKDNN_DEFAULT_PLAN", "").strip()
BUDGET_CPU_MINUTES_PER_MONTH = env_number("BRICKDNN_BUDGET_CPU_MINUTES_PER_MONTH")
# The most CPU time one training may use; it is stopped like after a stop request.
BUDGET_CPU_MINUTES_PER_RUN = env_number("BRICKDNN_BUDGET_CPU_MINUTES_PER_RUN")
# What an account may store: uploads, data sources, datasets and the files of its runs.
BUDGET_STORAGE_MB = env_number("BRICKDNN_BUDGET_STORAGE_MB")
# Accounts of staff members and superusers have no limits unless the admin says otherwise.
STAFF_UNLIMITED = env_bool("BRICKDNN_STAFF_UNLIMITED", True)
# Optional prices, to show costs in money (and for billing later). 0 or unset: no price,
# usage is shown in CPU minutes and MB. Plans have prices of their own (per month).
PRICE_PER_CPU_HOUR = env_number("BRICKDNN_PRICE_PER_CPU_HOUR")
PRICE_PER_GB_MONTH = env_number("BRICKDNN_PRICE_PER_GB_MONTH")
PRICE_CURRENCY = os.environ.get("BRICKDNN_PRICE_CURRENCY", "").strip().upper() or "EUR"

# Data preparation (the Data tab, needs pandas): imported files (CSV, Excel, JSON, Parquet ...)
# and their tables live in DATA_PREP_DIR. Uploads are limited by DATA_MAX_UPLOAD_MB.
# Set BRICKDNN_DATA_PREP=false to switch the feature off.
DATA_PREP_ENABLED = env_bool("BRICKDNN_DATA_PREP", True)
DATA_PREP_DIR = DATA_DIR / "prepare"
# Largest table that can be imported.
DATA_MAX_ROWS = int(os.environ.get("BRICKDNN_DATA_MAX_ROWS", "2000000"))
# Most columns per imported file; 0 or unset: no limit.
DATA_MAX_COLUMNS = int(os.environ.get("BRICKDNN_DATA_MAX_COLUMNS", "0")) or None
# Largest import in megabytes (uploads, downloads from links and the unpacked content of .gz /
# .zip files). Imports are parsed in memory (roughly 10-15 times the file size while reading),
# so this is lower than the training upload limit.
DATA_MAX_UPLOAD_MB = float(os.environ.get("BRICKDNN_DATA_MAX_UPLOAD_MB", "200"))
# Imports from links (http/https): the server downloads the file. Only public internet
# addresses are allowed, unless BRICKDNN_DATA_URL_ALLOW_PRIVATE=true (for local installations
# that import from servers in their own network).
DATA_URL_IMPORT_ENABLED = env_bool("BRICKDNN_DATA_URL_IMPORT", True)
DATA_URL_ALLOW_PRIVATE = env_bool("BRICKDNN_DATA_URL_ALLOW_PRIVATE", False)
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
        # Predictions with trained weights (each one starts a process that loads PyTorch).
        "predict": os.environ.get("BRICKDNN_PREDICT_RATE", "120/hour"),
        # Previews of data preparations (one per change in the Data tab).
        "data": os.environ.get("BRICKDNN_DATA_RATE", "1200/hour"),
        # Requests to the MCP server (one per tool call of an AI assistant).
        "mcp": os.environ.get("BRICKDNN_MCP_RATE", "600/hour"),
        # Inviting people to a project and changing its link.
        "sharing": os.environ.get("BRICKDNN_SHARING_RATE", "120/hour"),
        # OAuth app registration, tokens and revocation (per IP address: apps like Claude
        # call it from their servers, for all their users).
        "oauth": os.environ.get("BRICKDNN_OAUTH_RATE", "120/minute"),
        # Changing the e-mail address or password, exporting the data, deleting the account.
        "account": os.environ.get("BRICKDNN_ACCOUNT_RATE", "30/hour"),
    },
}

SIMPLE_JWT = {
    "ACCESS_TOKEN_LIFETIME": timedelta(minutes=30),
    "REFRESH_TOKEN_LIFETIME": timedelta(days=7),
    "ROTATE_REFRESH_TOKENS": True,
    "BLACKLIST_AFTER_ROTATION": True,
    "UPDATE_LAST_LOGIN": True,
}

# The public address of the site (e.g. https://brickdnn.com), used as OAuth issuer. Empty:
# taken from each request (Host header and X-Forwarded-Proto behind a proxy).
PUBLIC_URL = os.environ.get("BRICKDNN_PUBLIC_URL", "").rstrip("/")

# Set to "false" to disable public sign-up (e.g. for a private instance).
ALLOW_REGISTRATION = env_bool("BRICKDNN_ALLOW_REGISTRATION", True)

# Privacy and legal pages (see docs/privacy.md). The imprint, privacy policy and terms of use
# are Markdown files in LEGAL_DIR (imprint.md, privacy.md, terms.md), read on every request:
# a page whose file is missing is not offered. With terms.md, sign-up asks to accept them.
LEGAL_DIR = Path(os.environ.get("BRICKDNN_LEGAL_DIR", DATA_DIR / "legal"))
# Automatic deletion by `manage.py purge_old_data` (the cleanup service of Docker Compose runs
# it daily). 0 or unset: off. Data sources, datasets and finished training runs nobody has
# used for this many days:
DELETE_UNUSED_UPLOADS_DAYS = int(os.environ.get("BRICKDNN_DELETE_UNUSED_UPLOADS_DAYS") or 0)
# Accounts (with everything they own) that nobody has logged in to or used for this many days.
# Staff accounts are never deleted automatically.
DELETE_INACTIVE_ACCOUNTS_DAYS = int(os.environ.get("BRICKDNN_DELETE_INACTIVE_ACCOUNTS_DAYS") or 0)
# The data export of an account includes uploaded files up to this many MB in total; larger
# files are listed only.
EXPORT_MAX_FILES_MB = float(os.environ.get("BRICKDNN_EXPORT_MAX_FILES_MB", "200"))
# The activity log of AI agents (MCP tool calls and REST requests with API tokens, shown on
# the page "Agent activity"): entries older than this many days are deleted by
# `manage.py purge_old_data`. 0: nothing is logged (and the purge deletes what is there).
AGENT_LOG_DAYS = max(0, int(os.environ.get("BRICKDNN_AGENT_LOG_DAYS") or 90))

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
