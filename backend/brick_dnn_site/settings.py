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


# Secrets are never read from the site settings file: it is in the public repository.
SITE_ENV_SECRETS = frozenset(
    {"BRICKDNN_PADDLE_API_KEY", "BRICKDNN_PADDLE_WEBHOOK_SECRET", "BRICKDNN_EMAIL_PASSWORD"}
)


def load_site_settings(environ=os.environ) -> None:
    """Defaults from the file ``BRICKDNN_SITE_ENV`` names: ``BRICKDNN_…=value`` lines kept in
    the repository for one public server (``deploy/sites/<domain>/settings.env``, see
    deploy/sites/README.md). A variable that is set and not empty in the environment wins.
    Secrets (``SITE_ENV_SECRETS``) are ignored there."""
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
        if name.startswith("BRICKDNN_") and name not in SITE_ENV_SECRETS and not environ.get(name):
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
    # The Django admin, whose login asks for the two-factor code (brick_dnn_core.admin_site).
    "brick_dnn_core.admin_apps.AdminConfig",
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
# Number of trainings that may run at the same time (counted across all users) when the web
# server trains itself; with a worker, its slots (BRICKDNN_WORKER_SLOTS) take that role.
TRAINING_MAX_RUNS = int(os.environ.get("BRICKDNN_TRAINING_MAX_RUNS", "1"))
# The job queue (brick_dnn_core.training_queue, docs/training.md): with
# BRICKDNN_TRAINING_WORKER=true the web server does not start trainings itself. New runs are
# queued in the database and `manage.py training_worker` (its own process or container) works
# them off, plans with a higher priority first. Off by default: without a worker process the
# web server trains itself as before (Docker Compose switches it on and starts the worker).
TRAINING_WORKER = env_bool("BRICKDNN_TRAINING_WORKER", False)
# The name of a worker (default: the host name) and the trainings it runs at the same time
# (0 or unset: from its CPUs and OMP_NUM_THREADS, see training_worker.default_slots).
WORKER_NAME = os.environ.get("BRICKDNN_WORKER_NAME", "").strip()
WORKER_SLOTS = int(os.environ.get("BRICKDNN_WORKER_SLOTS") or 0)
# A worker without a heartbeat for this many seconds is considered dead: its runs go back
# into the queue (once; after BRICKDNN_WORKER_MAX_RESTARTS restarts in all, a run fails).
WORKER_TIMEOUT = max(5, int(os.environ.get("BRICKDNN_WORKER_TIMEOUT") or 60))
WORKER_MAX_RESTARTS = max(0, int(os.environ.get("BRICKDNN_WORKER_MAX_RESTARTS") or 3))
# A worker that is asked to stop (a deploy) lets its trainings finish for this many seconds,
# then stops them and puts them back into the queue.
WORKER_DRAIN_SECONDS = max(0, int(os.environ.get("BRICKDNN_WORKER_DRAIN_SECONDS") or 30))
# Trainings of one account that may wait in the queue (0: no limit); a plan can set its own
# with "queued_trainings". Unset: 3 on a server with plans (BRICKDNN_PLANS), no limit without.
QUEUE_MAX_WAITING: int | None = (
    max(0, int(os.environ["BRICKDNN_QUEUE_MAX_WAITING"]))
    if os.environ.get("BRICKDNN_QUEUE_MAX_WAITING", "").strip()
    else None
)
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
        # Paid plans: opening the checkout, changing or cancelling the subscription.
        "billing": os.environ.get("BRICKDNN_BILLING_RATE", "60/hour"),
        # Paddle's notifications (per IP address: Paddle sends them from a few servers, and
        # several per payment).
        "billing_webhook": os.environ.get("BRICKDNN_BILLING_WEBHOOK_RATE", "600/minute"),
        # Asking for a password reset link (per IP address; each account also gets at most
        # a few links an hour, see recovery.py).
        "password_reset": os.environ.get("BRICKDNN_PASSWORD_RESET_RATE", "10/hour"),
        # Sending the link that verifies the e-mail address again (per account).
        "email_verify": os.environ.get("BRICKDNN_EMAIL_VERIFY_RATE", "5/hour"),
        # The second step of a login with two-factor authentication (per IP address; wrong
        # codes are also limited per account, see two_factor.py).
        "two_factor": os.environ.get("BRICKDNN_TWO_FACTOR_RATE", "30/hour"),
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
# Usernames of accounts that are administrators (staff and superuser): they get the rights
# when the backend starts, and nobody can register these names (see site_admins.py).
ADMIN_USERS = env_list("BRICKDNN_ADMIN_USERS")
# The name of the site in e-mails (with the domain of BRICKDNN_PUBLIC_URL).
SITE_NAME = os.environ.get("BRICKDNN_SITE_NAME", "").strip() or "Brick DNN"

# E-mail (brick_dnn_core.mail, docs/email.md): password reset links, verifying addresses,
# security notices, warnings before inactive accounts are deleted, and server errors for the
# operator. Off unless BRICKDNN_EMAIL_HOST is set: then nothing is sent and what needs e-mail
# says so (e.g. "Password reset is not available on this server"). Any SMTP provider works.
#   BRICKDNN_EMAIL_HOST, _PORT    the SMTP server, e.g. smtp-relay.brevo.com and 587
#   BRICKDNN_EMAIL_USER           its login
#   BRICKDNN_EMAIL_PASSWORD       SECRET: its password or key (only in the environment)
#   BRICKDNN_EMAIL_USE_TLS/_SSL   STARTTLS (default, port 587) or TLS from the start (465)
#   BRICKDNN_EMAIL_FROM           the sender, e.g. Brick DNN <no-reply@brickdnn.com>
#   BRICKDNN_EMAIL_REPLY_TO       where answers go (optional)
# BRICKDNN_EMAIL_BACKEND=console (to the log) or file (BRICKDNN_EMAIL_FILE_PATH) is for trying
# it out without a provider: e-mail then counts as working.
EMAIL_HOST = os.environ.get("BRICKDNN_EMAIL_HOST", "").strip()
EMAIL_PORT = int(os.environ.get("BRICKDNN_EMAIL_PORT") or 587)
EMAIL_HOST_USER = os.environ.get("BRICKDNN_EMAIL_USER", "").strip()
EMAIL_HOST_PASSWORD = os.environ.get("BRICKDNN_EMAIL_PASSWORD", "").strip()
# Docker Compose passes unset variables as empty strings: those keep the default.
EMAIL_USE_SSL = (
    env_bool("BRICKDNN_EMAIL_USE_SSL", EMAIL_PORT == 465)
    if os.environ.get("BRICKDNN_EMAIL_USE_SSL", "").strip()
    else EMAIL_PORT == 465
)
EMAIL_USE_TLS = not EMAIL_USE_SSL and (
    env_bool("BRICKDNN_EMAIL_USE_TLS")
    if os.environ.get("BRICKDNN_EMAIL_USE_TLS", "").strip()
    else True
)
# Seconds to wait for the SMTP server.
EMAIL_TIMEOUT = int(os.environ.get("BRICKDNN_EMAIL_TIMEOUT") or 10)
_EMAIL_BACKENDS = {
    "smtp": "django.core.mail.backends.smtp.EmailBackend",
    "console": "django.core.mail.backends.console.EmailBackend",
    "file": "django.core.mail.backends.filebased.EmailBackend",
}
_email_backend = os.environ.get("BRICKDNN_EMAIL_BACKEND", "").strip().lower() or "smtp"
if _email_backend not in _EMAIL_BACKENDS:
    raise ImproperlyConfigured("BRICKDNN_EMAIL_BACKEND must be smtp, console or file.")
EMAIL_BACKEND = _EMAIL_BACKENDS[_email_backend]
EMAIL_FILE_PATH = os.environ.get("BRICKDNN_EMAIL_FILE_PATH", "").strip() or str(DATA_DIR / "emails")
# Whether this server sends e-mail at all.
EMAIL_ENABLED = bool(EMAIL_HOST) if _email_backend == "smtp" else True
DEFAULT_FROM_EMAIL = os.environ.get("BRICKDNN_EMAIL_FROM", "").strip() or (
    f"{SITE_NAME} <no-reply@localhost>"
)
SERVER_EMAIL = DEFAULT_FROM_EMAIL
EMAIL_REPLY_TO = env_list("BRICKDNN_EMAIL_REPLY_TO")
EMAIL_SUBJECT_PREFIX = f"[{SITE_NAME}] "
# E-mails are handed to the SMTP server in a background thread, so that no request waits for
# it (and the answer to "forgot password" takes as long for any address). false: right away.
EMAIL_IN_BACKGROUND = env_bool("BRICKDNN_EMAIL_BACKGROUND", True)
# Unhandled server errors (HTTP 500) are mailed to these addresses (comma-separated), at most
# one e-mail per kind of error an hour (brick_dnn_core.error_mails).
ADMINS = [("Operator", address) for address in env_list("BRICKDNN_ADMIN_EMAIL")]
# Password reset links are valid this long (and only once).
PASSWORD_RESET_TIMEOUT = 3600
# Two-factor authentication (brick_dnn_core.two_factor): true lets staff accounts into the Django
# admin only once they have set it up. Turn it on in the account, then set this.
REQUIRE_2FA_FOR_STAFF = env_bool("BRICKDNN_REQUIRE_2FA_FOR_STAFF", False)

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
# With e-mail, accounts are warned this many days before (comma-separated), and an account with
# an e-mail address is only deleted once it was warned (see retention.py).
INACTIVE_ACCOUNT_WARNING_DAYS = sorted(
    {int(days) for days in env_list("BRICKDNN_INACTIVE_ACCOUNT_WARNING_DAYS", "14,3")} - {0},
    reverse=True,
)
# The data export of an account includes uploaded files up to this many MB in total; larger
# files are listed only.
EXPORT_MAX_FILES_MB = float(os.environ.get("BRICKDNN_EXPORT_MAX_FILES_MB", "200"))
# The activity log of AI agents (MCP tool calls and REST requests with API tokens, shown on
# the page "Agent activity"): entries older than this many days are deleted by
# `manage.py purge_old_data`. 0: nothing is logged (and the purge deletes what is there).
AGENT_LOG_DAYS = max(0, int(os.environ.get("BRICKDNN_AGENT_LOG_DAYS") or 90))

# Paid plans with Paddle Billing (brick_dnn_core.billing, docs/billing.md). Paddle sells the
# plans as the merchant of record; a paid subscription sets the account's plan (BRICKDNN_PLANS).
# Off unless all of it is configured: the environment, the client-side token, at least one
# price, the API key and the webhook secret. With it off, the app shows no prices or checkout.
#   BRICKDNN_PADDLE_ENV           sandbox or live
#   BRICKDNN_PADDLE_CLIENT_TOKEN  the client-side token for Paddle.js (test_… or live_…; public)
#   BRICKDNN_PADDLE_PRICES        the Paddle price of each paid plan: starter=pri_…,plus=pri_…
#                                 (or JSON; several prices of one plan separated by "|": the
#                                 first is sold, the others stay valid for who subscribed to them)
#   BRICKDNN_PADDLE_OVERAGE_PRODUCT  the Paddle product (pro_…) that compute beyond the
#                                 allowance is billed as; empty: overage is not billed
# The two secrets only come from the environment (the server's .env), never from the site file:
#   BRICKDNN_PADDLE_API_KEY         server-side API key
#   BRICKDNN_PADDLE_WEBHOOK_SECRET  secret of the notification destination (several while
#                                   rotating: separated by commas)
PADDLE_ENV = os.environ.get("BRICKDNN_PADDLE_ENV", "").strip().lower()
PADDLE_CLIENT_TOKEN = os.environ.get("BRICKDNN_PADDLE_CLIENT_TOKEN", "").strip()
PADDLE_PRICES = os.environ.get("BRICKDNN_PADDLE_PRICES", "").strip()
PADDLE_OVERAGE_PRODUCT = os.environ.get("BRICKDNN_PADDLE_OVERAGE_PRODUCT", "").strip()
PADDLE_API_KEY = os.environ.get("BRICKDNN_PADDLE_API_KEY", "").strip()
PADDLE_WEBHOOK_SECRETS = env_list("BRICKDNN_PADDLE_WEBHOOK_SECRET")
# A notification is accepted if it was signed at most this many seconds ago (0: any age).
# Every event is processed once, whatever its age (its id is remembered).
PADDLE_WEBHOOK_TOLERANCE = int(os.environ.get("BRICKDNN_PADDLE_WEBHOOK_TOLERANCE") or 300)
# After a failed payment the account keeps its plan for this many days while Paddle retries.
PADDLE_GRACE_DAYS = int(os.environ.get("BRICKDNN_PADDLE_GRACE_DAYS") or 7)
# Overage below this amount (in BRICKDNN_PRICE_CURRENCY) is not billed for a month.
PADDLE_OVERAGE_MIN_CHARGE = env_number("BRICKDNN_PADDLE_OVERAGE_MIN_CHARGE", 0.5)
# For tests and trying things out: another address for Paddle's API (a fake Paddle).
PADDLE_API_URL = os.environ.get("BRICKDNN_PADDLE_API_URL", "").strip()

if not DEBUG:
    SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
    SESSION_COOKIE_SECURE = env_bool("DJANGO_SECURE_COOKIES", False)
    CSRF_COOKIE_SECURE = SESSION_COOKIE_SECURE
    SECURE_CONTENT_TYPE_NOSNIFF = True
    X_FRAME_OPTIONS = "DENY"

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "console": {"class": "logging.StreamHandler"},
        # Server errors to ADMINS, if e-mail is set up (and at most one per kind an hour).
        "mail_admins": {
            "class": "brick_dnn_core.error_mails.ThrottledAdminEmailHandler",
            "level": "ERROR",
        },
    },
    "loggers": {
        # Instead of Django's own handlers (its unthrottled mail_admins among them); messages
        # still reach the console through the root logger.
        "django": {"handlers": [], "level": "INFO"},
        "django.request": {"handlers": ["mail_admins"]},
    },
    "root": {"handlers": ["console"], "level": os.environ.get("DJANGO_LOG_LEVEL", "INFO")},
}
