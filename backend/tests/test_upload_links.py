"""Uploads with a signed one-time link (MCP tool create_upload_link): curl -T or -F without
logging in, the same import, limits and rate limits as the web app, and the checks of the
link (expiry, one use, signature, purpose, the user)."""

# ruff: noqa: F811 (the fixture training_dir is imported from test_training_api)

from __future__ import annotations

import time
import types
from urllib.parse import parse_qs, urlsplit

import pytest
from django.contrib.auth import get_user_model
from django.core import signing
from django.core.cache import cache
from django.core.files.uploadedfile import SimpleUploadedFile
from rest_framework.test import APIClient
from rest_framework.throttling import ScopedRateThrottle

from brick_dnn_core import upload_links
from brick_dnn_core.models import Dataset, DataSource, Project, ProjectMember

from .test_mcp import call, mcp_client, rpc
from .test_mcp_data import sales_csv
from .test_training_api import iris_csv, training_dir  # noqa: F401 (a fixture)

pytestmark = pytest.mark.django_db


@pytest.fixture(autouse=True)
def data_dirs(settings, tmp_path):
    settings.DATA_PREP_DIR = tmp_path / "prepare"
    settings.DATA_PREP_ENABLED = True
    settings.DATA_MAX_UPLOAD_MB = 5
    settings.MCP_UPLOAD_MAX_MB = 0.001  # inline text is tiny; links are not limited by it
    return tmp_path


@pytest.fixture
def user():
    return get_user_model().objects.create_user("ada", password="correct-horse-1")


@pytest.fixture
def mcp(user):
    return mcp_client(user)


def link(client, **arguments) -> dict:
    result, text = call(client, "create_upload_link", **arguments)
    assert "isError" not in result, text
    return result["structuredContent"]


def error_of(client, **arguments) -> str:
    result, text = call(client, "create_upload_link", **arguments)
    assert result.get("isError") is True, text
    return text


def path_of(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.path}?{parts.query}"


def put(url: str, content: bytes, **headers):
    """curl -T: the file as the body, no login."""
    return APIClient().put(
        path_of(url), content, content_type="application/octet-stream", **headers
    )


def post_file(url: str, filename: str, content: bytes):
    """curl -F file=@…: multipart, no login."""
    return APIClient().post(
        path_of(url), {"file": SimpleUploadedFile(filename, content)}, format="multipart"
    )


def test_upload_a_data_source(mcp, user, data_dirs):
    info = link(mcp, purpose="data_source", filename="sales.csv", name="Sales 2024")
    assert info["url"].startswith("http://testserver/api/uploads/?sig=")
    assert info["curl"] == f'curl --fail-with-body -T "sales.csv" "{info["url"]}"'
    assert info["expires_in_minutes"] == 15 and info["max_mb"] == 5
    assert "data source" in info["becomes"]

    content = sales_csv(2000).encode()  # far more than the inline limit of this test
    response = put(info["url"], content)
    assert response.status_code == 201, response.data
    source = response.data["source"]
    assert source["name"] == "Sales 2024" and source["rows"] == 2000
    assert source["columns"]["temperature"].startswith("numeric")
    assert "create_data_prep" in response.data["next"]
    stored = DataSource.objects.get(pk=source["id"])
    assert stored.owner == user and stored.filename == "sales.csv"
    original = data_dirs / "prepare" / "sources" / str(stored.pk) / "original.csv"
    assert original.read_bytes() == content

    # Once only.
    response = put(info["url"], content)
    assert response.status_code == 403 and "already used" in response.data["detail"]
    assert DataSource.objects.count() == 1


def test_upload_with_multipart_and_import_options(mcp):
    info = link(mcp, purpose="data_source", filename="kunden.csv", delimiter=";", decimal=",")
    response = post_file(info["url"], "whatever.txt", b"a;b\n1,5;x\n2,5;y\n3,0;x\n")
    assert response.status_code == 201, response.data
    source = response.data["source"]
    # The name and options of the link count, not those of the local file.
    assert source["filename"] == "kunden.csv" and source["import"]["decimal"] == ","
    assert source["first_rows"][0] == {"a": 1.5, "b": "x"}


def test_upload_a_training_dataset(mcp, user, training_dir):
    project = Project.objects.create(owner=user, name="Shared", graph={}, updated_by=user)
    info = link(mcp, purpose="training_dataset", filename="iris.csv", project_id=project.pk)
    assert info["max_mb"] == 5 and "training dataset" in info["becomes"]
    response = put(info["url"], iris_csv())
    assert response.status_code == 201, response.data
    dataset = Dataset.objects.get()
    assert response.data["reference"] == {"kind": "upload", "id": dataset.pk}
    assert response.data["dataset"]["features"] == 4
    assert response.data["dataset"]["shared_in"] == [{"id": project.pk, "name": "Shared"}]
    assert dataset.owner == user and list(dataset.projects.all()) == [project]


def test_failed_uploads_can_be_sent_again(mcp):
    info = link(mcp, purpose="data_source", filename="sales.csv")
    response = put(info["url"], b"")
    assert response.status_code == 400 and "curl -T" in response.data["detail"]
    response = put(info["url"], b"\x00\x01 not a table")
    assert response.status_code == 400
    assert put(info["url"], sales_csv().encode()).status_code == 201
    assert put(info["url"], sales_csv().encode()).status_code == 403


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"purpose": "weights", "filename": "a.csv"}, "'purpose' must be"),
        ({"purpose": "data_source", "filename": " "}, "'filename'"),
        ({"purpose": "data_source", "filename": "a.docx"}, "Upload a CSV, Excel"),
        ({"purpose": "training_dataset", "filename": "a.xlsx"}, "Upload a .csv file"),
        ({"purpose": "data_source", "filename": "a.csv", "delimiter": "x"}, "Delimiter"),
        ({"purpose": "data_source", "filename": "a.csv", "name": "x" * 121}, "120 characters"),
        ({"purpose": "training_dataset", "filename": "a.csv", "project_id": 999}, "owns or edits"),
        ({"purpose": "data_source"}, "Missing argument: filename"),
    ],
)
def test_bad_link_requests(mcp, arguments, message):
    assert message in error_of(mcp, **arguments)


def test_read_only_tokens_cannot_create_links(user):
    client = mcp_client(user, read_only=True)
    names = {t["name"] for t in rpc(client, "tools/list").json()["result"]["tools"]}
    assert "create_upload_link" not in names
    assert "read-only" in error_of(client, purpose="data_source", filename="a.csv")


def test_expired_tampered_and_misused_links(mcp, user, monkeypatch):
    real = time.time()
    monkeypatch.setattr(signing, "time", types.SimpleNamespace(time=lambda: real - 16 * 60))
    old = link(mcp, purpose="data_source", filename="sales.csv")["url"]
    monkeypatch.setattr(signing, "time", time)
    response = put(old, sales_csv().encode())
    assert response.status_code == 403 and "expired" in response.data["detail"]

    url = link(mcp, purpose="data_source", filename="sales.csv")["url"]
    signature = parse_qs(urlsplit(url).query)["sig"][0]
    for forged in (signature[:-2] + "xx", "", "garbage"):
        response = put(f"http://testserver/api/uploads/?sig={forged}", sales_csv().encode())
        assert response.status_code == 403 and "invalid" in response.data["detail"], forged
    response = put("http://testserver/api/uploads/", sales_csv().encode())
    assert response.status_code == 403
    # A signature of another kind of link (a weights download) does not open uploads.
    other = signing.TimestampSigner(salt="brick_dnn_core.downloads.weights").sign_object(
        {"run": "x", "user": user.pk}
    )
    response = put(f"http://testserver/api/uploads/?sig={other}", b"a,b\n1,2\n")
    assert response.status_code == 403
    # Nor a signed payload with another purpose.
    forged = upload_links._signer().sign_object({"u": user.pk, "p": "weights", "f": "a.csv"})
    assert put(f"http://testserver/api/uploads/?sig={forged}", b"a,b\n").status_code == 403

    # Deactivated users' links stop working.
    user.is_active = False
    user.save()
    response = put(url, sales_csv().encode())
    assert response.status_code == 403 and "not active" in response.data["detail"]
    assert not DataSource.objects.exists()


def test_link_works_without_login_but_ignores_one(mcp, user):
    info = link(mcp, purpose="data_source", filename="sales.csv")
    client = APIClient()
    client.credentials(HTTP_AUTHORIZATION="Bearer nonsense")
    response = client.put(
        path_of(info["url"]), sales_csv().encode(), content_type="application/octet-stream"
    )
    assert response.status_code == 201, response.data  # no authentication on this endpoint
    assert DataSource.objects.get().owner == user


def test_size_limits_and_rate_limit(mcp, settings, monkeypatch, training_dir):
    settings.DATA_MAX_UPLOAD_MB = 0.01  # about 10 KB
    info = link(mcp, purpose="data_source", filename="sales.csv")
    assert info["max_mb"] == 0.01
    response = put(info["url"], sales_csv(1000).encode())
    assert response.status_code == 413 and "at most 0.01 MB" in response.data["detail"]
    settings.TRAINING_MAX_UPLOAD_MB = 0.001
    info = link(mcp, purpose="training_dataset", filename="iris.csv")
    assert put(info["url"], iris_csv(rows=200)).status_code == 413
    assert not DataSource.objects.exists() and not Dataset.objects.exists()

    # Uploads count against the rate limit of the web app (the ones above counted too).
    settings.DATA_MAX_UPLOAD_MB = 5
    cache.clear()
    monkeypatch.setitem(ScopedRateThrottle.THROTTLE_RATES, "uploads", "1/hour")
    first = link(mcp, purpose="data_source", filename="sales.csv")["url"]
    second = link(mcp, purpose="data_source", filename="sales.csv")["url"]
    assert put(first, sales_csv().encode()).status_code == 201
    response = put(second, sales_csv().encode())
    assert response.status_code == 429
    # The link was not used up by the refused upload.
    cache.clear()
    assert put(second, sales_csv().encode()).status_code == 201


def test_links_when_features_are_off(mcp, settings):
    settings.DATA_PREP_ENABLED = False
    assert "disabled" in error_of(mcp, purpose="data_source", filename="a.csv")
    settings.TRAINING_ENABLED = False
    assert "disabled" in error_of(mcp, purpose="training_dataset", filename="a.csv")


def test_shared_project_links_need_editing_rights(user, training_dir):
    owner = get_user_model().objects.create_user("bob", password="correct-horse-1")
    project = Project.objects.create(owner=owner, name="Bob's", graph={}, updated_by=owner)
    member = ProjectMember.objects.create(project=project, user=user, role="editor")
    mcp = mcp_client(user)
    info = link(mcp, purpose="training_dataset", filename="iris.csv", project_id=project.pk)
    member.role = "viewer"
    member.save()
    response = put(info["url"], iris_csv())
    assert response.status_code == 403 and not Dataset.objects.exists()
