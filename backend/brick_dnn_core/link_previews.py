"""Link previews: the meta tags of a shared link, for chats and social networks.

A shared network (``/share/<token>``) and a shared comparison of runs (``/compare/<token>``)
are pages of the web app, which crawlers read without running its script. So that such a link
still unfolds with a title, a description and an image, nginx puts the answer of
``GET /api/link-preview/<share|compare>/<token>/`` into the ``<head>`` of the app's page
(a server-side include, see ``frontend/nginx.conf`` and ``frontend/index.html``): a fragment
of HTML with ``<title>``, the description and the Open Graph and Twitter card tags.

What the fragment says is less than the page behind the link shows: the **name** of the
network or comparison and **what kind** of thing it is. Never the owner, the description,
the layers or the runs. A link that does not exist or was turned off answers 404 with an
empty body, and nginx then keeps the app's general tags, so a preview cannot tell a
revoked link from a mistyped one any better than the page can.

The texts are in the language saved on the owner's account (crawlers send no
``Accept-Language``), else in the request's, else in English. The image is the general one
of the site (``frontend/public/brand/link-preview.png``).
"""

from __future__ import annotations

from django.conf import settings
from django.http import HttpRequest, HttpResponse
from django.utils import translation
from django.utils.html import format_html, format_html_join
from django.utils.translation import gettext
from django.views.decorators.http import require_GET

from . import languages
from .models import Project, RunComparison

# The pages of the web app that shared links open (frontend/src/App.tsx).
PAGES = {"share": "/share/", "compare": "/compare/"}
# The preview image, served by the frontend (frontend/public/), and its size in pixels.
IMAGE_PATH = "/brand/link-preview.png"
IMAGE_SIZE = (1200, 630)
# Chats cut long titles; a name can have 120 characters.
MAX_NAME = 70


def _shared(kind: str, token: str):
    """The shared network or comparison behind a token, with its owner (None: no such link)."""
    if not 8 <= len(token) <= 64:
        return None
    if kind == "share":
        return (
            Project.objects.select_related("owner")
            .filter(share_token=token, link_access="view")
            .first()
        )
    return RunComparison.objects.select_related("owner").filter(token=token).first()


def _short(name: str) -> str:
    name = " ".join(name.split())
    return name if len(name) <= MAX_NAME else f"{name[: MAX_NAME - 1].rstrip()}…"


def _texts(kind: str, name: str) -> tuple[str, str]:
    """Title and description of the preview, in the active language."""
    if kind == "share":
        return (
            gettext("%(name)s – a network built with Brick DNN") % {"name": name},
            gettext(
                "A neural network shared with Brick DNN. See its layers and the shape of every "
                "tensor, read the PyTorch code, or open a copy in the editor."
            ),
        )
    return (
        gettext("%(name)s – training runs compared in Brick DNN") % {"name": name},
        gettext(
            "Training runs compared side by side in Brick DNN: their settings, results and "
            "curves, read-only."
        ),
    )


def _base_url(request: HttpRequest) -> str:
    """The public address of the site: BRICKDNN_PUBLIC_URL, else the request's own."""
    return settings.PUBLIC_URL or f"{request.scheme}://{request.get_host()}"


def fragment(kind: str, token: str, name: str, base_url: str, language: str) -> str:
    """The tags of a shared link as HTML, every value escaped."""
    with translation.override(language):
        title, description = _texts(kind, _short(name))
        image_alt = gettext("Brick DNN: neural networks built from layers.")
    image = f"{base_url}{IMAGE_PATH}"
    tags = [
        ("property", "og:type", "website"),
        ("property", "og:site_name", "Brick DNN"),
        ("property", "og:title", title),
        ("property", "og:description", description),
        ("property", "og:url", f"{base_url}{PAGES[kind]}{token}"),
        ("property", "og:locale", "de_DE" if language == "de" else "en_US"),
        ("property", "og:image", image),
        ("property", "og:image:type", "image/png"),
        ("property", "og:image:width", str(IMAGE_SIZE[0])),
        ("property", "og:image:height", str(IMAGE_SIZE[1])),
        ("property", "og:image:alt", image_alt),
        ("name", "twitter:card", "summary_large_image"),
        ("name", "twitter:title", title),
        ("name", "twitter:description", description),
        ("name", "twitter:image", image),
        ("name", "twitter:image:alt", image_alt),
    ]
    return format_html(
        '<title>{}</title>\n<meta name="description" content="{}">\n'
        '<meta name="robots" content="noindex">\n{}\n',
        title,
        description,
        format_html_join("\n", '<meta {}="{}" content="{}">', tags),
    )


@require_GET
def link_preview(request: HttpRequest, kind: str, token: str) -> HttpResponse:
    """``GET /api/link-preview/<share|compare>/<token>/``: the meta tags of a shared link as
    an HTML fragment; 404 without a body for a link that does not exist (any more)."""
    shared = _shared(kind, token)
    if shared is None:
        response = HttpResponse(status=404, content_type="text/html; charset=utf-8")
    else:
        language = languages.user_language(shared.owner, request)
        response = HttpResponse(
            fragment(kind, token, shared.name, _base_url(request), language),
            content_type="text/html; charset=utf-8",
        )
        response.headers["Content-Language"] = language
    # Not a page of its own: never in a search index, never cached.
    response.headers["X-Robots-Tag"] = "noindex"
    response.headers["Cache-Control"] = "no-store"
    return response
