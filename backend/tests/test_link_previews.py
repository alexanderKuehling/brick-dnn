"""Link previews: the meta tags of shared links (link_previews.py)."""

from __future__ import annotations

import re
from html import unescape
from pathlib import Path

import pytest
from django.contrib.auth import get_user_model
from rest_framework.test import APIClient

from brick_dnn_core import link_previews
from brick_dnn_core.models import Project, RunComparison, UserProfile

from .helpers import chain

pytestmark = pytest.mark.django_db

REPO = Path(__file__).resolve().parents[2]
GRAPH = chain(("Input", {"shape": [4]}), ("Linear", {"out_features": 2}))


@pytest.fixture
def ada():
    return get_user_model().objects.create_user(
        "ada-lovelace", email="ada@example.com", password="correct-horse-1"
    )


@pytest.fixture
def shared(ada):
    return Project.objects.create(
        owner=ada,
        name="Small CNN",
        description="Secret notes about the thesis",
        graph=GRAPH,
        link_access="view",
        share_token="tok_SHARE-0123456789abcd",
    )


@pytest.fixture
def comparison(ada):
    return RunComparison.objects.create(
        owner=ada, name="LR sweep", token="tok_COMPARE-0123456789ab", run_ids=["1", "2"]
    )


def tags(html: str) -> dict[str, str]:
    """The meta tags of a fragment by name or property, values unescaped."""
    found = {
        name: unescape(content)
        for name, content in re.findall(
            r'<meta (?:name|property)="([^"]+)" content="([^"]*)">', html
        )
    }
    found["title"] = unescape(re.search(r"<title>([^<]*)</title>", html).group(1))
    return found


def test_a_shared_network_has_its_name_and_kind(shared, settings):
    settings.PUBLIC_URL = "https://brickdnn.com"
    response = APIClient().get(f"/api/link-preview/share/{shared.share_token}/")
    assert response.status_code == 200
    assert response["Content-Type"] == "text/html; charset=utf-8"
    assert response["X-Robots-Tag"] == "noindex" and response["Cache-Control"] == "no-store"
    html = response.content.decode()
    found = tags(html)
    assert found["title"] == "Small CNN – a network built with Brick DNN"
    assert found["og:title"] == found["twitter:title"] == found["title"]
    assert found["description"].startswith("A neural network shared with Brick DNN.")
    assert found["og:description"] == found["twitter:description"] == found["description"]
    assert found["og:url"] == f"https://brickdnn.com/share/{shared.share_token}"
    assert found["og:site_name"] == "Brick DNN" and found["og:type"] == "website"
    image = "https://brickdnn.com/brand/link-preview.png"
    assert found["og:image"] == found["twitter:image"] == image
    assert (found["og:image:width"], found["og:image:height"]) == ("1200", "630")
    assert found["og:image:alt"] and found["twitter:card"] == "summary_large_image"
    assert found["robots"] == "noindex"
    # Only a title and meta tags: nothing that runs or loads.
    assert set(re.findall(r"<(\w+)", html)) == {"title", "meta"}


def test_a_shared_comparison_has_its_name_and_kind(comparison, settings):
    settings.PUBLIC_URL = ""
    response = APIClient().get(
        f"/api/link-preview/compare/{comparison.token}/", HTTP_HOST="localhost:8080"
    )
    found = tags(response.content.decode())
    assert found["title"] == "LR sweep – training runs compared in Brick DNN"
    assert "side by side" in found["description"]
    # Without BRICKDNN_PUBLIC_URL the addresses are those of the request.
    assert found["og:url"] == f"http://localhost:8080/compare/{comparison.token}"
    assert found["og:image"] == "http://localhost:8080/brand/link-preview.png"


def test_nothing_but_the_name_leaks(shared, comparison):
    for path in (
        f"/api/link-preview/share/{shared.share_token}/",
        f"/api/link-preview/compare/{comparison.token}/",
    ):
        html = APIClient().get(path).content.decode()
        for private in ("ada", "lovelace", "example.com", "Secret notes", "thesis", "Linear"):
            assert private not in html, private


def test_unknown_and_revoked_links_answer_the_same_empty_404(shared, comparison):
    client = APIClient()
    unknown = client.get("/api/link-preview/share/tok_NOBODY-0123456789abc/")
    assert unknown.status_code == 404 and unknown.content == b""
    assert unknown["X-Robots-Tag"] == "noindex"

    # Turned off: the token stays on the project until a new link is made.
    Project.objects.filter(pk=shared.pk).update(link_access="none")
    revoked = client.get(f"/api/link-preview/share/{shared.share_token}/")
    assert (revoked.status_code, revoked.content) == (404, b"")
    assert dict(revoked.headers) == dict(unknown.headers)

    token = comparison.token
    comparison.delete()
    assert client.get(f"/api/link-preview/compare/{token}/").content == b""
    # A network's token is no comparison's.
    Project.objects.filter(pk=shared.pk).update(link_access="view")
    assert client.get(f"/api/link-preview/compare/{shared.share_token}/").status_code == 404
    # Too short or too long, another kind, another method.
    assert client.get("/api/link-preview/share/short/").status_code == 404
    assert client.get(f"/api/link-preview/share/{'x' * 65}/").status_code == 404
    assert client.get(f"/api/link-preview/project/{shared.share_token}/").status_code == 404
    assert client.post(f"/api/link-preview/share/{shared.share_token}/").status_code == 405


def test_names_are_escaped_and_shortened(ada, settings):
    settings.PUBLIC_URL = "https://brickdnn.com"
    name = '"><script>alert(1)</script><!--# include virtual="/api/auth/me/" --> & ' + "x" * 60
    project = Project.objects.create(
        owner=ada,
        name=name[:120],
        graph=GRAPH,
        link_access="view",
        share_token="tok_ESCAPE-0123456789abc",
    )
    html = APIClient().get(f"/api/link-preview/share/{project.share_token}/").content.decode()
    assert "<script" not in html and "<!--" not in html and '"">' not in html
    assert set(re.findall(r"<(\w+)", html)) == {"title", "meta"}
    title = tags(html)["title"]
    assert title.startswith('"><script>alert(1)</script><!--# include')
    assert title.endswith("… – a network built with Brick DNN")
    assert len(title) == link_previews.MAX_NAME + len(" – a network built with Brick DNN")


def test_the_language_is_the_owners_else_the_requests(shared, comparison, ada):
    path = f"/api/link-preview/share/{shared.share_token}/"
    german = APIClient().get(path, HTTP_ACCEPT_LANGUAGE="de-DE,de;q=0.9")
    assert tags(german.content.decode())["title"] == "Small CNN – ein Netz, gebaut mit Brick DNN"
    assert german["Content-Language"] == "de"

    # Crawlers send no Accept-Language: the language saved on the owner's account decides.
    UserProfile.objects.update_or_create(user=ada, defaults={"language": "de"})
    found = tags(APIClient().get(path).content.decode())
    assert found["title"] == "Small CNN – ein Netz, gebaut mit Brick DNN"
    assert found["og:locale"] == "de_DE" and "PyTorch-Code" in found["description"]
    found = tags(APIClient().get(f"/api/link-preview/compare/{comparison.token}/").content.decode())
    assert found["title"] == "LR sweep – Trainingsläufe im Vergleich in Brick DNN"
    english = APIClient().get(path, HTTP_ACCEPT_LANGUAGE="en")
    assert "ein Netz" in english.content.decode()  # the owner's language wins


def test_the_preview_image_and_the_include_exist():
    """What the fragment and nginx rely on in the frontend."""
    image = (REPO / "frontend/public" / link_previews.IMAGE_PATH.lstrip("/")).read_bytes()
    assert image[1:4] == b"PNG"
    width, height = int.from_bytes(image[16:20], "big"), int.from_bytes(image[20:24], "big")
    assert (width, height) == link_previews.IMAGE_SIZE
    shell = (REPO / "frontend/index.html").read_text(encoding="utf-8")
    assert '<!--# include virtual="/api/link-preview/$preview_kind/$preview_token/"' in shell
    nginx = (REPO / "frontend/nginx.conf").read_text(encoding="utf-8")
    kinds = re.search(r"\(\?<preview_kind>([a-z|]+)\)", nginx).group(1).split("|")
    assert sorted(kinds) == sorted(link_previews.PAGES)
