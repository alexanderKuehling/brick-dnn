"""The languages of the app: English (the default of the API) and German (docs/translations.md).

- **Requests.** :class:`ApiLocaleMiddleware` activates the language of ``Accept-Language``
  for the request (the web app sends the language the user chose with its switch), so error
  messages, validation errors and the descriptions of layers, training options and
  preparation steps come in that language. Without the header (scripts, curl) everything is
  English. The MCP server always answers in English: its tool descriptions and results are read
  by AI assistants, not people.
- **E-mails** go out in the language saved on the account (``UserProfile.language``, set at
  sign-up from the app and changed with its switch), else in the request's, else in English
  (:func:`user_language`).
- **Texts** are translated with gettext: ``locale/de/LC_MESSAGES/django.po``. Strings of the
  ``brickdnn`` package (layer, step and option descriptions) stay plain English there, so the
  package needs no Django; the API translates them when it returns them (``catalog.py``).
  ``manage.py update_translations`` adds new strings to the ``.po`` file and compiles the
  ``.mo`` file next to it, which is committed (no GNU gettext needed to build or run).
"""

from __future__ import annotations

from django.utils import translation
from django.utils.cache import patch_vary_headers

DEFAULT_LANGUAGE = "en"
LANGUAGES = ("en", "de")
LANGUAGE_CHOICES = [("en", "English"), ("de", "Deutsch")]

# Paths whose answers stay English whatever the request asks for.
ENGLISH_ONLY_PREFIXES = ("/api/mcp", "/.well-known/")


def normalize(code: str | None) -> str | None:
    """``"de-AT"`` → ``"de"``; None for a language the app does not have."""
    if not code:
        return None
    base = str(code).strip().lower().replace("_", "-").split("-")[0]
    return base if base in LANGUAGES else None


def request_language(request) -> str:
    """The language of ``Accept-Language`` (only languages of the app), else English."""
    if request is None:
        return DEFAULT_LANGUAGE
    return accepted_language(request.META.get("HTTP_ACCEPT_LANGUAGE", ""))


def accepted_language(header: str) -> str:
    """The app's language that ``Accept-Language`` prefers most (``"de-DE,de;q=0.9,en;q=0.8"``
    → ``"de"``), else English."""
    choices: list[tuple[float, int, str]] = []
    for index, part in enumerate(str(header or "").split(",")[:20]):
        code, _, params = part.strip().partition(";")
        quality = 1.0
        params = params.strip()
        if params.startswith("q="):
            try:
                quality = float(params[2:])
            except ValueError:
                quality = 0.0
        language = normalize(code)
        if language and quality > 0:
            choices.append((-quality, index, language))
    return min(choices)[2] if choices else DEFAULT_LANGUAGE


def saved_language(user) -> str | None:
    """The language saved on the account, or None."""
    if user is None or not getattr(user, "is_authenticated", False):
        return None
    try:
        profile = user.profile
    except Exception:  # no profile (yet)
        return None
    return normalize(getattr(profile, "language", ""))


def user_language(user, request=None) -> str:
    """The language of e-mails to ``user``: the account's, else the request's, else English."""
    return saved_language(user) or (request_language(request) if request else DEFAULT_LANGUAGE)


def save_language(user, language: str) -> None:
    from .models import UserProfile

    code = normalize(language)
    if code is None:
        return
    profile, _ = UserProfile.objects.get_or_create(user=user)
    if profile.language != code:
        profile.language = code
        profile.save(update_fields=["language"])


class ApiLocaleMiddleware:
    """Activates the request's language (``Accept-Language``) for the response.

    Unlike Django's LocaleMiddleware it reads no cookie or session and no language prefix in the
    URL: the app sends its language with every request. Responses vary on Accept-Language.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        path = request.path_info or ""
        if path.startswith(ENGLISH_ONLY_PREFIXES):
            language = DEFAULT_LANGUAGE
        else:
            language = request_language(request)
        translation.activate(language)
        request.LANGUAGE_CODE = language
        try:
            response = self.get_response(request)
        finally:
            translation.deactivate()
        patch_vary_headers(response, ("Accept-Language",))
        response.headers.setdefault("Content-Language", language)
        return response
