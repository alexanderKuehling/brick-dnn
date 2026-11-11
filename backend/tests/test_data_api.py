"""Data preparation API (the Data tab): imports, preparations, previews, exports, downloads.

The data work itself is tested in the other ``test_data_*`` files; these tests check the
Django side: files, limits, ownership, response shapes and the hand-over to training.
"""

from __future__ import annotations

import json
import os
import time

import pytest
from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from rest_framework.parsers import MultiPartParser
from rest_framework.test import APIClient
from rest_framework.throttling import ScopedRateThrottle

from brick_dnn_core import data_service, training_service
from brick_dnn_core.models import DataPrep, Dataset, DataSource

from .helpers import chain

pytestmark = pytest.mark.django_db

PREVIEW_KEYS = {
    "steps",
    "split",
    "rows",
    "task",
    "profile",
    "before",
    "sample",
    "readiness",
    "warnings",
    "duration_ms",
}
SOURCE_KEYS = {
    "id",
    "name",
    "filename",
    "size_bytes",
    "n_rows",
    "n_columns",
    "columns",
    "import_options",
    "report",
    "created_at",
}
PREP_KEYS = {
    "id",
    "name",
    "source",
    "source_name",
    "steps",
    "settings",
    "datasets",
    "created_at",
    "updated_at",
    # Sharing in projects (see shared_data.py).
    "owner",
    "role",
    "shared_in",
}


# --------------------------------------------------------------------------- #
# Fixtures and helpers
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def data_dirs(settings, tmp_path):
    """Fresh folders for imports and training datasets, both features switched on."""
    settings.DATA_PREP_DIR = tmp_path / "prepare"
    settings.TRAINING_DIR = tmp_path / "training"
    settings.DATA_PREP_ENABLED = True
    settings.TRAINING_ENABLED = True
    settings.DATA_MAX_UPLOAD_MB = 5
    settings.DATA_SAMPLE_ROWS = 5000
    settings.DATA_MAX_SAMPLE_ROWS = 20000
    return tmp_path


@pytest.fixture
def user():
    return get_user_model().objects.create_user("ada", password="correct-horse-battery")


@pytest.fixture
def other_user():
    return get_user_model().objects.create_user("mallory", password="whatever-password-1")


@pytest.fixture
def client():
    return APIClient()


@pytest.fixture
def auth_client(user):
    client = APIClient()
    client.force_authenticate(user)
    return client


@pytest.fixture
def other_client(other_user):
    client = APIClient()
    client.force_authenticate(other_user)
    return client


def churn_csv(rows: int = 40) -> bytes:
    """Numeric features and a leave/stay target: ready for training without any step."""
    lines = ["customer,age,income,churn"]
    for i in range(rows):
        churn = "leave" if i % 3 == 0 else "stay"
        lines.append(f"{i + 1},{20 + (i * 7) % 50},{1000 + 37.5 * i},{churn}")
    return ("\n".join(lines) + "\n").encode()


def regions_csv(rows: int = 40) -> bytes:
    lines = ["customer,region"] + [f"{i + 1},{'north' if i % 2 else 'south'}" for i in range(rows)]
    return ("\n".join(lines) + "\n").encode()


def import_csv(client, filename: str, content: bytes, **fields):
    return client.post(
        "/api/data/sources/",
        {"file": SimpleUploadedFile(filename, content), **fields},
        format="multipart",
    )


def create_prep(client, source_id, name="Churn", steps=None, settings=None):
    body = {"name": name, "source": source_id}
    if steps is not None:
        body["steps"] = steps
    if settings is not None:
        body["settings"] = settings
    return client.post("/api/data/preps/", body, format="json")


def preview(client, prep_id, **body):
    return client.post(f"/api/data/preps/{prep_id}/preview/", body, format="json")


def export(client, prep_id, **body):
    return client.post(f"/api/data/preps/{prep_id}/export/", body, format="json")


def step(step_id, step_type, **params):
    return {"id": step_id, "type": step_type, "enabled": True, "params": params}


def entries(path):
    return sorted(p.name for p in path.iterdir()) if path.exists() else []


@pytest.fixture
def source(auth_client):
    response = import_csv(auth_client, "churn.csv", churn_csv())
    assert response.status_code == 201, response.data
    return response.data


@pytest.fixture
def prep(auth_client, source):
    response = create_prep(auth_client, source["id"], settings={"target": "churn"})
    assert response.status_code == 201, response.data
    return response.data


# --------------------------------------------------------------------------- #
# Status, step catalog, login and availability
# --------------------------------------------------------------------------- #


def test_status_is_public(client, settings):
    settings.DATA_MAX_ROWS = 1234
    response = client.get("/api/data/")
    assert response.status_code == 200
    assert response.data == {
        "enabled": True,
        "available": True,
        "reason": None,
        "max_upload_mb": 5,
        "max_rows": 1234,
        "max_columns": settings.DATA_MAX_COLUMNS,
        "default_sample_rows": 5000,
        "max_sample_rows": 20000,
        "login_required": True,
        "file_types": response.data["file_types"],
        "url_import": True,
    }
    assert {".csv", ".xlsx", ".json", ".parquet", ".zip"} <= set(response.data["file_types"])


def test_step_catalog_is_public(client):
    response = client.get("/api/data/steps/")
    assert response.status_code == 200
    groups = [g["id"] for g in response.data["groups"]]
    assert groups[0] == "combine" and "encode" in groups
    types = {s["type"] for s in response.data["steps"]}
    assert {"join", "append", "impute", "encode", "drop_columns"} <= types


def test_api_root_lists_data_endpoints(client):
    endpoints = client.get("/api/").data["endpoints"]
    assert endpoints["data"].endswith("/api/data/")
    assert endpoints["data_steps"].endswith("/api/data/steps/")
    assert endpoints["data_sources"].endswith("/api/data/sources/")
    assert endpoints["data_preps"].endswith("/api/data/preps/")


def test_data_endpoints_require_login(client):
    assert client.get("/api/data/sources/").status_code == 401
    assert import_csv(client, "churn.csv", churn_csv()).status_code == 401
    assert client.get("/api/data/sources/1/").status_code == 401
    assert client.get("/api/data/preps/").status_code == 401
    assert create_prep(client, 1).status_code == 401
    assert preview(client, 1).status_code == 401
    assert export(client, 1).status_code == 401
    assert client.get("/api/data/preps/1/download/?format=csv").status_code == 401


@pytest.mark.parametrize("cause", ["disabled", "no-pandas"])
def test_unavailable_data_preparation(client, auth_client, prep, settings, monkeypatch, cause):
    if cause == "disabled":
        settings.DATA_PREP_ENABLED = False
        reason = "Data preparation is disabled on this server."
    else:
        monkeypatch.setattr(data_service, "dependencies_installed", lambda: False)
        reason = data_service.MISSING_DEPENDENCIES
    status = client.get("/api/data/").data
    assert status["available"] is False
    assert status["reason"] == reason
    assert status["enabled"] is (cause != "disabled")

    responses = [
        client.get("/api/data/steps/"),
        auth_client.get("/api/data/sources/"),
        import_csv(auth_client, "churn.csv", churn_csv()),
        auth_client.get("/api/data/preps/"),
        create_prep(auth_client, prep["source"]),
        preview(auth_client, prep["id"]),
        export(auth_client, prep["id"]),
        auth_client.get(f"/api/data/preps/{prep['id']}/download/?format=csv"),
    ]
    for response in responses:
        assert response.status_code == 503, response.data
        assert response.data["detail"] == reason
    assert DataSource.objects.count() == 1 and DataPrep.objects.count() == 1


# --------------------------------------------------------------------------- #
# Imports (data sources)
# --------------------------------------------------------------------------- #


def test_import_csv(auth_client, data_dirs):
    content = churn_csv()
    response = import_csv(auth_client, "churn.csv", content)
    assert response.status_code == 201, response.data
    data = response.data
    assert set(data) == SOURCE_KEYS
    assert data["name"] == "churn" and data["filename"] == "churn.csv"
    assert data["size_bytes"] == len(content)
    assert (data["n_rows"], data["n_columns"]) == (40, 4)
    assert data["columns"] == [
        {"name": "customer", "kind": "numeric", "missing": 0},
        {"name": "age", "kind": "numeric", "missing": 0},
        {"name": "income", "kind": "numeric", "missing": 0},
        {"name": "churn", "kind": "categorical", "missing": 0},
    ]
    assert data["import_options"] == {
        "delimiter": "auto",
        "decimal": "auto",
        "header": "auto",
        "encoding": "auto",
        "missing_values": None,
        "sheet": None,
    }
    report = data["report"]
    assert report["delimiter"] == "," and report["header"] is True
    assert (report["rows"], report["columns"]) == (40, 4)
    assert isinstance(report["warnings"], list)

    directory = data_dirs / "prepare" / "sources" / str(data["id"])
    assert (directory / "original.csv").read_bytes() == content
    assert (directory / "table" / "table.json").is_file()
    assert entries(data_dirs / "prepare" / "sources") == [str(data["id"])]

    assert [s["id"] for s in auth_client.get("/api/data/sources/").data] == [data["id"]]
    assert auth_client.get(f"/api/data/sources/{data['id']}/").data == data


def test_import_with_options_and_name(auth_client):
    content = "a;b;label\n1,5;-999;x\n2,25;3;y\n-0,5;4;x\n7;-999;y\n".encode("cp1252")
    response = import_csv(
        auth_client,
        "messy.TXT",
        content,
        name="  Messy data ",
        delimiter=";",
        decimal=",",
        header="yes",
        encoding="cp1252",
        missing_values=json.dumps(["-999"]),
    )
    assert response.status_code == 201, response.data
    data = response.data
    assert data["name"] == "Messy data"
    assert data["import_options"] == {
        "delimiter": ";",
        "decimal": ",",
        "header": "yes",
        "encoding": "cp1252",
        "missing_values": ["-999"],
        "sheet": None,
    }
    assert data["report"]["delimiter"] == ";" and data["report"]["decimal"] == ","
    columns = {c["name"]: c for c in data["columns"]}
    assert columns["a"]["kind"] == "numeric"
    assert columns["b"]["missing"] == 2


@pytest.mark.parametrize(
    ("filename", "content", "fields", "message"),
    [
        ("data.pdf", b"a,b\n1,2\n", {}, "Upload a CSV, Excel, OpenDocument, JSON, Parquet"),
        ("empty.csv", b"", {}, None),
        ("ragged.csv", b"a,b,c\n1,2,3\n4,5\n6,7,8\n", {}, "line"),
        ("data.csv", b"a,b\n1,2\n", {"delimiter": "x"}, None),
        ("data.csv", b"a,b\n1,2\n", {"encoding": "ebcdic"}, None),
    ],
    ids=["wrong-extension", "empty", "ragged", "bad-delimiter", "bad-encoding"],
)
def test_bad_imports(auth_client, data_dirs, filename, content, fields, message):
    response = import_csv(auth_client, filename, content, **fields)
    assert response.status_code == 400, response.data
    assert isinstance(response.data["detail"], str) and response.data["detail"]
    if message is not None:
        assert message in response.data["detail"].lower() or message in response.data["detail"]
    assert DataSource.objects.count() == 0
    assert entries(data_dirs / "prepare" / "sources") == []  # no half-imported files


def test_import_without_file(auth_client):
    response = auth_client.post("/api/data/sources/", {"name": "x"}, format="multipart")
    assert response.status_code == 400
    assert response.data["detail"] == "Choose a file to upload."


def test_import_limits(auth_client, settings):
    settings.DATA_MAX_ROWS = 10
    response = import_csv(auth_client, "churn.csv", churn_csv(40))
    assert response.status_code == 400
    assert "10" in response.data["detail"]
    settings.DATA_MAX_ROWS = 1000
    settings.DATA_MAX_COLUMNS = 3
    response = import_csv(auth_client, "churn.csv", churn_csv(40))
    assert response.status_code == 400
    assert "3" in response.data["detail"]
    assert DataSource.objects.count() == 0


def test_import_too_large(auth_client, settings, monkeypatch):
    settings.DATA_MAX_UPLOAD_MB = 0.01  # about 10 KB
    # Slightly too large: detected from the size of the parsed file.
    response = import_csv(auth_client, "big.csv", b"1,2\n" * 5000)
    assert response.status_code == 413
    assert "limited to 0.01 MB" in response.data["detail"]

    # Far too large: rejected from the Content-Length header, before the body is parsed.
    def no_parsing(*args, **kwargs):
        raise AssertionError("the body must not be parsed")

    monkeypatch.setattr(MultiPartParser, "parse", no_parsing)
    response = import_csv(auth_client, "huge.csv", b"1,2\n" * 50_000)
    assert response.status_code == 413
    assert DataSource.objects.count() == 0


def test_imports_are_rate_limited(auth_client, monkeypatch):
    monkeypatch.setitem(ScopedRateThrottle.THROTTLE_RATES, "uploads", "2/hour")
    statuses = [import_csv(auth_client, "a.csv", churn_csv(8)).status_code for _ in range(3)]
    assert statuses == [201, 201, 429]
    assert auth_client.get("/api/data/sources/").status_code == 200  # only imports count


def workbook(sheets: dict[str, bytes]) -> bytes:
    """An .xlsx file with a sheet per CSV text."""
    import io

    import pandas as pd

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer) as writer:
        for name, content in sheets.items():
            frame = pd.read_csv(io.BytesIO(content)) if content else pd.DataFrame()
            frame.to_excel(writer, sheet_name=name, index=False)
    return buffer.getvalue()


def test_import_excel_sheets(auth_client, data_dirs):
    pytest.importorskip("openpyxl")
    content = workbook({"Regions": regions_csv(), "Churn": churn_csv()})
    response = import_csv(auth_client, "Kunden 2024.xlsx", content)
    assert response.status_code == 201, response.data
    data = response.data
    assert data["name"] == "Kunden 2024" and data["filename"] == "Kunden 2024.xlsx"
    report = data["report"]
    assert report["format"] == "xlsx" and report["sheet"] == "Regions"
    assert report["sheets"] == ["Regions", "Churn"] and report["delimiter"] is None
    directory = data_dirs / "prepare" / "sources" / str(data["id"])
    assert (directory / "original.xlsx").read_bytes() == content

    response = import_csv(auth_client, "Kunden 2024.xlsx", content, sheet="Churn")
    assert response.status_code == 201, response.data
    assert response.data["report"]["sheet"] == "Churn"
    assert response.data["import_options"]["sheet"] == "Churn"
    assert (response.data["n_rows"], response.data["n_columns"]) == (40, 4)

    # A wrong sheet: the answer lists the sheets to choose from.
    response = import_csv(auth_client, "Kunden 2024.xlsx", content, sheet="Q3")
    assert response.status_code == 400
    assert response.data["detail"] == "The workbook has no sheet “Q3”."
    assert response.data["sheets"] == ["Regions", "Churn"]
    assert DataSource.objects.count() == 2


@pytest.mark.parametrize("filename", ["churn.json", "churn.jsonl", "churn.parquet", "churn.csv.gz"])
def test_import_other_formats(auth_client, data_dirs, filename):
    import gzip
    import io

    import pandas as pd

    frame = pd.read_csv(io.BytesIO(churn_csv()))
    if filename.endswith(".json"):
        content = frame.to_json(orient="records").encode()
    elif filename.endswith(".jsonl"):
        content = frame.to_json(orient="records", lines=True).encode()
    elif filename.endswith(".parquet"):
        pytest.importorskip("pyarrow")
        content = frame.to_parquet()
    else:
        content = gzip.compress(churn_csv())
    response = import_csv(auth_client, filename, content)
    assert response.status_code == 201, response.data
    assert response.data["name"] == "churn"
    assert (response.data["n_rows"], response.data["n_columns"]) == (40, 4)
    assert [c["kind"] for c in response.data["columns"]] == [
        "numeric",
        "numeric",
        "numeric",
        "categorical",
    ]
    original = "original" + filename[len("churn") :]
    directory = data_dirs / "prepare" / "sources" / str(response.data["id"])
    assert (directory / original).read_bytes() == content


def test_python_script_of_an_excel_source(auth_client):
    pytest.importorskip("openpyxl")
    content = workbook({"Notes": b"", "Churn": churn_csv()})
    source = import_csv(auth_client, "churn.xlsx", content).data
    prep = create_prep(auth_client, source["id"], settings={"target": "churn"}).data
    response = auth_client.get(f"/api/data/preps/{prep['id']}/download/?format=python")
    assert response.status_code == 200
    script = b"".join(response.streaming_content if response.streaming else [response.content])
    assert b"'format': 'xlsx'" in script and b"'sheet': 'Churn'" in script
    assert b"'#DIV/0!'" in script  # error values of spreadsheets count as missing


@pytest.fixture
def file_server():
    from .test_url_import import Server

    server = Server()
    yield server
    server.close()


def import_url(client, url, **fields):
    return client.post("/api/data/sources/", {"url": url, **fields}, format="json")


def test_import_from_a_link(auth_client, settings, file_server, data_dirs):
    settings.DATA_URL_ALLOW_PRIVATE = True  # the test server runs on 127.0.0.1
    file_server.routes["/data/churn.csv"] = (200, {"Content-Type": "text/csv"}, churn_csv())
    file_server.routes["/sheet"] = (
        200,
        {"Content-Disposition": 'attachment; filename="Customers.xlsx"'},
        workbook({"A": regions_csv(), "B": churn_csv()}),
    )
    response = import_url(auth_client, f"{file_server.url}/data/churn.csv")
    assert response.status_code == 201, response.data
    data = response.data
    assert data["name"] == "churn" and data["filename"] == "churn.csv"
    assert data["report"]["url"] == f"{file_server.url}/data/churn.csv"
    assert data["size_bytes"] == len(churn_csv()) and data["n_rows"] == 40
    directory = data_dirs / "prepare" / "sources" / str(data["id"])
    assert (directory / "original.csv").read_bytes() == churn_csv()

    # The query of a link can hold secrets (signed links): it is not stored.
    file_server.routes["/data/churn.csv?token=secret"] = file_server.routes["/data/churn.csv"]
    response = import_url(auth_client, f"{file_server.url}/data/churn.csv?token=secret#top")
    assert response.status_code == 201, response.data
    assert response.data["report"]["url"] == f"{file_server.url}/data/churn.csv?…"

    response = import_url(auth_client, f"{file_server.url}/sheet", name="Kunden", sheet="B")
    assert response.status_code == 201, response.data
    assert response.data["name"] == "Kunden" and response.data["filename"] == "Customers.xlsx"
    assert response.data["report"]["sheet"] == "B" and response.data["n_rows"] == 40

    response = import_url(auth_client, f"{file_server.url}/missing.csv")
    assert response.status_code == 400
    assert "Nothing was found at this link" in response.data["detail"]
    assert entries(data_dirs / "prepare" / "sources") == sorted(
        str(s.pk) for s in DataSource.objects.all()
    )


def test_links_to_private_addresses_are_refused_by_default(auth_client, file_server):
    file_server.routes["/churn.csv"] = (200, {}, churn_csv())
    for url in (f"{file_server.url}/churn.csv", "http://169.254.169.254/latest/meta-data/"):
        response = import_url(auth_client, url)
        assert response.status_code == 400
        assert "not a public internet address" in response.data["detail"]
    assert file_server.requests == []
    response = import_url(auth_client, "https://user:pw@example.org/data.csv")
    assert "user name or password" in response.data["detail"]
    assert DataSource.objects.count() == 0


def test_link_imports_can_be_switched_off(client, auth_client, settings):
    settings.DATA_URL_IMPORT_ENABLED = False
    assert client.get("/api/data/").data["url_import"] is False
    response = import_url(auth_client, "https://example.org/data.csv")
    assert response.status_code == 403
    assert response.data["detail"] == "Importing from links is switched off on this server."


def test_link_import_requests(auth_client, client):
    assert import_url(client, "https://example.org/data.csv").status_code == 401
    response = auth_client.post("/api/data/sources/", {"name": "x"}, format="json")
    assert response.status_code == 400 and "link of a file" in response.data["detail"]
    response = import_url(auth_client, "https://example.org/data.csv", delimiter="x")
    assert response.status_code == 400 and response.data["detail"].startswith("Delimiter")


def test_link_imports_count_like_uploads(auth_client, settings, file_server, monkeypatch):
    settings.DATA_URL_ALLOW_PRIVATE = True
    file_server.routes["/a.csv"] = (200, {}, churn_csv(8))
    monkeypatch.setitem(ScopedRateThrottle.THROTTLE_RATES, "uploads", "2/hour")
    assert import_csv(auth_client, "a.csv", churn_csv(8)).status_code == 201
    assert import_url(auth_client, f"{file_server.url}/a.csv").status_code == 201
    assert import_url(auth_client, f"{file_server.url}/a.csv").status_code == 429


def test_abandoned_import_folders_are_removed(auth_client, data_dirs):
    root = data_dirs / "prepare" / "sources"
    abandoned = root / ".import-old"
    abandoned.mkdir(parents=True)
    os.utime(abandoned, (time.time() - 7 * 3600,) * 2)
    recent = root / ".import-recent"
    recent.mkdir()
    assert import_csv(auth_client, "churn.csv", churn_csv()).status_code == 201
    assert not abandoned.exists()
    assert recent.exists()  # may belong to an import that is still running


def test_rename_source(auth_client, source):
    url = f"/api/data/sources/{source['id']}/"
    response = auth_client.patch(url, {"name": "  Customers "}, format="json")
    assert response.status_code == 200, response.data
    assert response.data["name"] == "Customers"
    assert response.data["n_rows"] == 40
    # Everything but the name is read-only.
    response = auth_client.patch(url, {"n_rows": 1, "filename": "x.csv"}, format="json")
    assert response.data["n_rows"] == 40 and response.data["filename"] == "churn.csv"
    response = auth_client.patch(url, {"name": " "}, format="json")
    assert response.status_code == 400
    assert response.data["detail"].startswith("Name: ")
    assert auth_client.put(url, {"name": "x"}, format="json").status_code == 405


def test_delete_source(auth_client, source, data_dirs, django_capture_on_commit_callbacks):
    directory = data_dirs / "prepare" / "sources" / str(source["id"])
    assert directory.is_dir()
    with django_capture_on_commit_callbacks(execute=True):
        response = auth_client.delete(f"/api/data/sources/{source['id']}/")
    assert response.status_code == 204
    assert not directory.exists()
    assert auth_client.get(f"/api/data/sources/{source['id']}/").status_code == 404


def test_source_used_by_a_preparation_cannot_be_deleted(auth_client, source, prep):
    create_prep(auth_client, source["id"], name="Second")
    response = auth_client.delete(f"/api/data/sources/{source['id']}/")
    assert response.status_code == 409
    assert response.data["detail"] == (
        "Used by the preparations “Churn”, “Second”; delete them first."
    )
    for other in DataPrep.objects.filter(name="Second"):
        assert auth_client.delete(f"/api/data/preps/{other.pk}/").status_code == 204
    response = auth_client.delete(f"/api/data/sources/{source['id']}/")
    assert response.data["detail"] == "Used by the preparation “Churn”; delete it first."
    assert auth_client.delete(f"/api/data/preps/{prep['id']}/").status_code == 204
    assert auth_client.delete(f"/api/data/sources/{source['id']}/").status_code == 204


# --------------------------------------------------------------------------- #
# Ownership
# --------------------------------------------------------------------------- #


def test_users_only_see_their_own_data(auth_client, other_client, source, prep):
    assert other_client.get("/api/data/sources/").data == []
    assert other_client.get("/api/data/preps/").data == []
    source_url = f"/api/data/sources/{source['id']}/"
    assert other_client.get(source_url).status_code == 404
    assert other_client.patch(source_url, {"name": "x"}, format="json").status_code == 404
    assert other_client.delete(source_url).status_code == 404
    prep_url = f"/api/data/preps/{prep['id']}/"
    assert other_client.get(prep_url).status_code == 404
    assert other_client.patch(prep_url, {"name": "x"}, format="json").status_code == 404
    assert other_client.delete(prep_url).status_code == 404
    assert preview(other_client, prep["id"]).status_code == 404
    assert export(other_client, prep["id"]).status_code == 404
    assert other_client.get(f"{prep_url}download/?format=csv").status_code == 404

    # Preparations can only use own sources.
    response = create_prep(other_client, source["id"])
    assert response.status_code == 400
    assert response.data["detail"] == "Source: This data source does not exist (any more)."
    own = import_csv(other_client, "mine.csv", churn_csv(8)).data
    response = auth_client.patch(prep_url, {"source": own["id"]}, format="json")
    assert response.status_code == 400
    assert DataPrep.objects.get(pk=prep["id"]).source_id == source["id"]
    assert DataSource.objects.filter(pk=source["id"]).exists()


def test_join_only_reads_own_sources(auth_client, other_client, source, prep):
    own = import_csv(auth_client, "regions.csv", regions_csv()).data
    foreign = import_csv(other_client, "regions.csv", regions_csv()).data

    def join(source_id):
        return step(
            "j1",
            "join",
            source=source_id,
            how="left",
            left_on=["customer"],
            right_on=["customer"],
            suffix="_right",
        )

    response = preview(auth_client, prep["id"], steps=[join(own["id"])])
    assert response.status_code == 200, response.data
    report = response.data["steps"][0]
    assert report["status"] == "ok", report
    assert "region" in [c["name"] for c in report["columns"]]
    assert "region" in report["added"]

    response = preview(auth_client, prep["id"], steps=[join(foreign["id"])])
    assert response.status_code == 200, response.data
    report = response.data["steps"][0]
    assert report["status"] == "error"
    assert report["error"] == "This table does not exist (any more)."
    assert "region" not in [c["name"] for c in response.data["profile"]["columns"]]

    # A deleted join source fails the same way (only the main source is protected).
    auth_client.patch(f"/api/data/preps/{prep['id']}/", {"steps": [join(own["id"])]}, format="json")
    assert auth_client.delete(f"/api/data/sources/{own['id']}/").status_code == 204
    report = preview(auth_client, prep["id"]).data["steps"][0]
    assert report["status"] == "error"
    assert report["error"] == "This table does not exist (any more)."


# --------------------------------------------------------------------------- #
# Preparations
# --------------------------------------------------------------------------- #


def test_create_prep_with_defaults(auth_client, source):
    response = auth_client.post("/api/data/preps/", {"source": source["id"]}, format="json")
    assert response.status_code == 201, response.data
    data = response.data
    assert set(data) == PREP_KEYS
    assert data["name"] == "churn"  # the name of the source
    assert data["source"] == source["id"] and data["source_name"] == "churn"
    assert data["steps"] == [] and data["datasets"] == []
    assert data["settings"] == {
        "target": None,
        "task": "auto",
        "split": {"val": 0.15, "test": 0.15, "seed": 42, "stratify": True, "group_by": None},
    }


def test_prep_crud(auth_client, source, prep):
    url = f"/api/data/preps/{prep['id']}/"
    assert prep["settings"]["target"] == "churn"
    assert prep["settings"]["split"]["val"] == 0.15  # defaults filled in

    steps = [{"id": "s1", "type": "drop_columns", "params": {"columns": ["customer"]}}]
    response = auth_client.patch(url, {"steps": steps, "name": "Churn v2"}, format="json")
    assert response.status_code == 200, response.data
    assert response.data["name"] == "Churn v2"
    assert response.data["steps"] == [
        {"id": "s1", "type": "drop_columns", "enabled": True, "params": {"columns": ["customer"]}}
    ]
    assert response.data["settings"] == prep["settings"]  # unchanged

    split = {"val": 0.2, "test": 0, "seed": 7, "stratify": False}
    settings = {"target": "churn", "task": "classification", "split": split}
    response = auth_client.patch(url, {"settings": settings}, format="json")
    assert response.status_code == 200, response.data
    assert response.data["settings"]["split"]["seed"] == 7
    assert response.data["settings"]["task"] == "classification"

    assert auth_client.get(url).data["name"] == "Churn v2"
    listing = auth_client.get("/api/data/preps/").data
    assert [p["id"] for p in listing] == [prep["id"]]
    assert listing[0]["source_name"] == "churn"

    assert auth_client.delete(url).status_code == 204
    assert auth_client.get(url).status_code == 404
    assert DataSource.objects.filter(pk=source["id"]).exists()


@pytest.mark.parametrize(
    "body",
    [
        {"steps": "drop everything"},
        {"steps": [{"id": "s1", "type": "no_such_step", "params": {}}]},
        {"steps": [{"type": "drop_columns", "params": {}}]},
        {"steps": [step("s1", "drop_columns"), step("s1", "drop_columns")]},
        {"steps": [step("s1", "drop_columns"), step("s2", "append", source=1)]},
        {"settings": {"split": {"val": 0.95, "test": 0.1}}},
        {"settings": {"task": "clustering"}},
        {"settings": "nope"},
    ],
    ids=[
        "not-a-list",
        "unknown-type",
        "no-id",
        "duplicate-id",
        "combine-after-other",
        "split-too-large",
        "unknown-task",
        "settings-not-object",
    ],
)
def test_invalid_steps_and_settings(auth_client, prep, body):
    url = f"/api/data/preps/{prep['id']}/"
    response = auth_client.patch(url, body, format="json")
    assert response.status_code == 400, response.data
    assert isinstance(response.data["detail"], str) and response.data["detail"]
    assert next(iter(body)) in response.data  # the field errors are kept as well
    saved = auth_client.get(url).data
    assert saved["steps"] == prep["steps"] and saved["settings"] == prep["settings"]
    response = create_prep(auth_client, prep["source"], **body)
    assert response.status_code == 400


def test_create_prep_needs_a_source(auth_client):
    response = auth_client.post("/api/data/preps/", {"name": "x"}, format="json")
    assert response.status_code == 400
    assert response.data["detail"] == "Source: This field is required."
    response = auth_client.post("/api/data/preps/", {"source": "abc"}, format="json")
    assert response.status_code == 400
    assert response.data["detail"] == "Source: Choose one of your data sources."


# --------------------------------------------------------------------------- #
# Preview
# --------------------------------------------------------------------------- #


def test_preview_shape(auth_client, prep):
    response = preview(auth_client, prep["id"])
    assert response.status_code == 200, response.data
    data = response.data
    assert set(data) == PREVIEW_KEYS
    assert data["steps"] == []
    assert data["task"] == "classification"
    assert data["before"] is None and data["rows"] is None
    assert data["split"]["group_by"] is None and data["split"]["error"] is None
    assert [c["name"] for c in data["split"]["columns"]] == [
        c["name"] for c in data["profile"]["columns"]
    ]
    assert isinstance(data["duration_ms"], (int, float)) and isinstance(data["warnings"], list)
    profile = data["profile"]
    assert (profile["n_rows"], profile["n_columns"]) == (40, 4)
    assert sum(profile["split_counts"]) == 40
    assert profile["target"]["column"] == "churn"
    sample = data["sample"]
    assert sample["n_total"] == 40 and sample["rows"] == 40  # all rows (fewer than 5,000)
    assert len(sample["split"]) == 40
    assert [c["name"] for c in sample["columns"]] == ["customer", "age", "income", "churn"]
    readiness = data["readiness"]
    assert readiness["ready"] is True, readiness["issues"]
    assert readiness["target"] == "churn" and readiness["num_features"] == 3
    assert readiness["class_names"] == ["leave", "stay"]
    assert readiness["split_counts"] == profile["split_counts"]
    json.dumps(data, allow_nan=False)  # valid JSON: no NaN or infinity


def test_preview_sample_size_is_clamped(auth_client, prep, settings):
    assert preview(auth_client, prep["id"], sample_rows=3).data["sample"]["rows"] == 10
    settings.DATA_MAX_SAMPLE_ROWS = 25
    assert preview(auth_client, prep["id"], sample_rows=10**9).data["sample"]["rows"] == 25
    assert preview(auth_client, prep["id"], sample_rows=17).data["sample"]["rows"] == 17
    settings.DATA_SAMPLE_ROWS = 12
    assert preview(auth_client, prep["id"]).data["sample"]["rows"] == 12
    for bad in ("many", 2.5, True):
        response = preview(auth_client, prep["id"], sample_rows=bad)
        assert response.status_code == 400
        assert response.data["detail"] == "sample_rows must be a whole number."


def test_preview_of_unsaved_steps_upto_and_compare(auth_client, prep):
    steps = [
        step("s1", "drop_columns", columns=["customer"]),
        step("s2", "rename_columns", mapping={"age": "years"}),
    ]
    response = preview(auth_client, prep["id"], steps=steps)
    assert response.status_code == 200, response.data
    reports = response.data["steps"]
    assert [r["id"] for r in reports] == ["s1", "s2"]
    assert [r["status"] for r in reports] == ["ok", "ok"]
    assert reports[0]["removed"] == ["customer"]
    assert [c["name"] for c in response.data["profile"]["columns"]] == ["years", "income", "churn"]

    response = preview(auth_client, prep["id"], steps=steps, upto="s1", compare=True)
    assert response.status_code == 200, response.data
    assert [r["status"] for r in response.data["steps"]] == ["ok", "not_run"]
    assert [c["name"] for c in response.data["profile"]["columns"]] == ["age", "income", "churn"]
    before = response.data["before"]
    assert before is not None and before["n_columns"] == 4

    # The preview does not save anything.
    assert auth_client.get(f"/api/data/preps/{prep['id']}/").data["steps"] == []

    response = preview(auth_client, prep["id"], steps=steps, upto="s9")
    assert response.status_code == 400
    assert response.data["detail"] == "There is no step 's9' (any more)."


def test_preview_with_unsaved_settings_and_bad_requests(auth_client, prep):
    settings = {"target": "income", "task": "regression"}
    response = preview(auth_client, prep["id"], settings=settings)
    assert response.status_code == 200, response.data
    assert response.data["task"] == "regression"
    response = preview(auth_client, prep["id"], settings={"split": {"val": 2}})
    assert response.status_code == 400 and response.data["detail"]
    response = preview(auth_client, prep["id"], steps=[{"id": "x", "type": "nope"}])
    assert response.status_code == 400 and response.data["detail"]
    response = auth_client.post(f"/api/data/preps/{prep['id']}/preview/", [1], format="json")
    assert response.status_code == 400


def test_failing_step_keeps_the_preview_usable(auth_client, prep):
    steps = [step("s1", "drop_columns", columns=["nope"]), step("s2", "drop_columns", columns=[])]
    response = preview(auth_client, prep["id"], steps=steps)
    assert response.status_code == 200, response.data
    first, second = response.data["steps"]
    assert first["status"] == "error" and first["error"]
    assert second["status"] == "skipped"
    assert response.data["profile"]["n_columns"] == 4


def test_previews_are_rate_limited(auth_client, prep, monkeypatch):
    monkeypatch.setitem(ScopedRateThrottle.THROTTLE_RATES, "data", "2/hour")
    statuses = [preview(auth_client, prep["id"]).status_code for _ in range(3)]
    assert statuses == [200, 200, 429]
    assert auth_client.get(f"/api/data/preps/{prep['id']}/").status_code == 200


def test_preview_when_the_files_are_missing(auth_client, prep, data_dirs):
    import shutil

    shutil.rmtree(data_dirs / "prepare" / "sources" / str(prep["source"]))
    response = preview(auth_client, prep["id"])
    assert response.status_code == 400
    assert response.data["detail"] == data_service.MISSING_FILES_MESSAGE


# --------------------------------------------------------------------------- #
# Export into a training dataset
# --------------------------------------------------------------------------- #

GRAPH = chain(("Input", {"shape": [3]}), ("Linear", {"out_features": 2}))


def test_export_creates_a_training_dataset(auth_client, prep, data_dirs, monkeypatch):
    monkeypatch.setattr(training_service, "torch_installed", lambda: False)  # not needed
    expected = preview(auth_client, prep["id"]).data["readiness"]["split_counts"]
    response = export(auth_client, prep["id"])
    assert response.status_code == 201, response.data
    data = response.data
    assert data["name"] == "Churn" and data["kind"] == "csv"
    assert data["prepared_from"] == {"id": prep["id"], "name": "Churn"}
    info = data["info"]
    assert info["kind"] == "csv" and info["name"] == "Churn"
    assert info["predefined_split"] == expected and sum(expected) == 40
    assert info["num_samples"] == 40 and info["num_features"] == 3
    assert info["columns"] == ["customer", "age", "income", "churn"]
    assert info["class_names"] == ["leave", "stay"] and info["num_classes"] == 2
    assert info["target_kind"] == "integer"  # class indices

    directory = data_dirs / "training" / "datasets" / str(data["id"])
    # prepare/: the steps with what they learned, for predictions from raw rows
    assert entries(directory) == ["data.csv", "prepare", "split.bin"]
    assert entries(directory / "prepare") == ["recipe.json"]
    split = (directory / "split.bin").read_bytes()
    assert [split.count(i) for i in range(3)] == expected
    files = [p for p in directory.rglob("*") if p.is_file()]
    assert data["size_bytes"] == sum(p.stat().st_size for p in files)
    assert entries(data_dirs / "training" / "datasets") == [str(data["id"])]

    # It is listed with the training datasets, and the preparation lists it.
    listing = auth_client.get("/api/training/datasets/").data
    assert [d["id"] for d in listing] == [data["id"]]
    assert listing[0]["prepared_from"] == {"id": prep["id"], "name": "Churn"}
    datasets = auth_client.get(f"/api/data/preps/{prep['id']}/").data["datasets"]
    assert [d["id"] for d in datasets] == [data["id"]]
    assert set(datasets[0]) == {"id", "name", "created_at"}

    second = export(auth_client, prep["id"], name="  Churn (all rows) ").data
    assert second["name"] == "Churn (all rows)" and second["info"]["name"] == "Churn (all rows)"
    datasets = auth_client.get(f"/api/data/preps/{prep['id']}/").data["datasets"]
    assert [d["id"] for d in datasets] == [second["id"], data["id"]]  # newest first


def test_exported_dataset_in_the_training_check(auth_client, prep, data_dirs):
    dataset_id = export(auth_client, prep["id"]).data["id"]
    counts = Dataset.objects.get(pk=dataset_id).info["predefined_split"]
    response = auth_client.post(
        "/api/training/check/",
        {"graph": GRAPH, "dataset": {"kind": "upload", "id": dataset_id}},
        format="json",
    )
    assert response.status_code == 200, response.data
    data = response.data
    assert data["ok"] is True, data["errors"]
    assert data["plan"]["task"] == "classification"
    assert data["plan"]["class_names"] == ["leave", "stay"]
    train, val, test = counts
    assert data["warnings"][0] == (
        f"Uses the split of the data preparation ({train} training, {val} validation and "
        f"{test} test rows); the validation and test split settings are ignored."
    )


def test_uploaded_datasets_have_no_preparation(auth_client, data_dirs):
    response = auth_client.post(
        "/api/training/datasets/",
        {"file": SimpleUploadedFile("churn.csv", churn_csv())},
        format="multipart",
    )
    assert response.status_code == 201, response.data
    assert response.data["prepared_from"] is None
    assert response.data["info"]["predefined_split"] is None


def test_export_refuses_data_that_is_not_ready(auth_client, source, data_dirs):
    prep_id = create_prep(auth_client, source["id"]).data["id"]  # no target
    response = export(auth_client, prep_id)
    assert response.status_code == 400
    issues = response.data["issues"]
    errors = [i for i in issues if i["severity"] == "error"]
    assert errors and response.data["detail"] == errors[0]["message"]
    assert Dataset.objects.count() == 0
    assert entries(data_dirs / "training" / "datasets") == []


def test_export_refuses_a_failing_step(auth_client, prep, data_dirs):
    steps = [step("s1", "drop_columns", columns=["nope"])]
    auth_client.patch(f"/api/data/preps/{prep['id']}/", {"steps": steps}, format="json")
    response = export(auth_client, prep["id"])
    assert response.status_code == 400
    assert response.data["detail"].startswith("Step 1 (")
    assert response.data["detail"].endswith("Fix it before creating a training dataset.")
    response = auth_client.get(f"/api/data/preps/{prep['id']}/download/?format=csv")
    assert response.status_code == 400
    assert response.data["detail"].endswith("Fix it before downloading.")
    assert Dataset.objects.count() == 0


def test_export_name_and_training_switch(auth_client, prep, settings):
    response = export(auth_client, prep["id"], name="x" * 121)
    assert response.status_code == 400
    assert response.data["detail"] == "The name can be at most 120 characters long."
    settings.TRAINING_ENABLED = False
    response = export(auth_client, prep["id"])
    assert response.status_code == 503
    assert response.data["detail"] == "Training is disabled on this server."


def test_deleting_a_prep_keeps_its_datasets(auth_client, prep):
    dataset_id = export(auth_client, prep["id"]).data["id"]
    assert auth_client.delete(f"/api/data/preps/{prep['id']}/").status_code == 204
    data = auth_client.get(f"/api/training/datasets/{dataset_id}/").data
    assert data["prepared_from"] is None
    assert data["info"]["predefined_split"] is not None


def test_deleting_a_user_removes_their_files(
    auth_client, user, prep, data_dirs, django_capture_on_commit_callbacks
):
    # A dataset made from the preparation (written directly: the export is tested above).
    Dataset.objects.create(
        owner=user, name="Churn", kind="csv", prepared_from_id=prep["id"], info={}
    )
    source_dir = data_dirs / "prepare" / "sources" / str(prep["source"])
    assert source_dir.is_dir()
    with django_capture_on_commit_callbacks(execute=True):
        user.delete()  # the preparation protects its source, but both go together
    assert not source_dir.exists()
    assert DataSource.objects.count() == DataPrep.objects.count() == Dataset.objects.count() == 0


# --------------------------------------------------------------------------- #
# Downloads
# --------------------------------------------------------------------------- #


def download(client, prep_id, fmt):
    return client.get(f"/api/data/preps/{prep_id}/download/?format={fmt}")


def test_download_prepared_csv(auth_client, prep, data_dirs):
    response = download(auth_client, prep["id"], "csv")
    assert response.status_code == 200
    assert response["Content-Disposition"] == 'attachment; filename="churn.csv"'
    assert response["Content-Type"].startswith("text/csv")
    text = b"".join(response.streaming_content).decode()  # also closes the response
    lines = text.strip().splitlines()
    assert lines[0].split(",") == ["customer", "age", "income", "churn", "split"]
    assert len(lines) == 41
    assert entries(data_dirs / "prepare" / "tmp") == []  # the temporary file is gone


def test_download_recipe_and_script(auth_client, prep):
    steps = [step("s1", "drop_columns", columns=["customer"])]
    auth_client.patch(
        f"/api/data/preps/{prep['id']}/", {"steps": steps, "name": "My Churn"}, format="json"
    )
    response = download(auth_client, prep["id"], "recipe")
    assert response.status_code == 200
    assert response["Content-Disposition"] == 'attachment; filename="my-churn.recipe.json"'
    recipe = json.loads(response.content)
    assert recipe["format"] == "brickdnn-data-recipe"
    assert recipe["source"]["id"] == prep["source"]
    assert [s["id"] for s in recipe["steps"]] == ["s1"]

    response = download(auth_client, prep["id"], "python")
    assert response.status_code == 200
    assert response["Content-Disposition"] == 'attachment; filename="my_churn_prepare.py"'
    assert response["Content-Type"].startswith("text/x-python")
    script = response.content.decode()
    compile(script, "my_churn_prepare.py", "exec")
    assert "def prepare(" in script

    response = download(auth_client, prep["id"], "xlsx")
    assert response.status_code == 400
    assert response.data["detail"] == "Choose the format csv, recipe or python."


def test_invalid_unicode_is_rejected_not_a_server_error(auth_client, source, prep):
    """JSON allows lone surrogates ("\\ud800"); they cannot be stored or sent back."""
    body = '{"name": "\\ud800", "source": ' + str(source["id"]) + "}"
    response = auth_client.post("/api/data/preps/", body, content_type="application/json")
    assert response.status_code == 400
    assert "not valid text" in response.json()["detail"]
    steps = '[{"id": "s1", "type": "rename_columns", "params": {"mapping": {"age": "\\ud800"}}}]'
    response = auth_client.post(
        f"/api/data/preps/{prep['id']}/preview/",
        '{"steps": ' + steps + "}",
        content_type="application/json",
    )
    assert response.status_code == 400
    assert "not valid text" in response.json()["detail"]


def test_a_failed_step_makes_the_data_not_ready(auth_client, prep):
    steps = [step("s1", "drop_columns", columns=["nope"])]
    response = preview(auth_client, prep["id"], steps=steps)
    assert response.status_code == 200
    readiness = response.data["readiness"]
    assert readiness["ready"] is False
    assert readiness["issues"][0]["severity"] == "error"
    assert readiness["issues"][0]["message"].startswith("Step 1 (Drop columns) failed:")
    assert readiness["issues"][0]["code"] == "step_failed"
    assert readiness["issues"][0]["columns"] == []


def test_preview_lists_the_sanity_checks(auth_client):
    # 40 customers, each twice: rows of the validation/test part copy training rows
    lines = churn_csv().decode().splitlines()
    content = ("\n".join(lines + lines[1:]) + "\n").encode()
    source = import_csv(auth_client, "twice.csv", content).data
    prep = create_prep(auth_client, source["id"], settings={"target": "churn"}).data
    readiness = preview(auth_client, prep["id"]).data["readiness"]
    assert readiness["ready"] is True, readiness["issues"]
    issues = {i["code"]: i for i in readiness["issues"]}
    assert issues["split_overlap"]["severity"] == "warning"
    assert issues["split_overlap"]["columns"] == []
    assert issues["split_overlap"]["fix"]["type"] == "drop_duplicates"
    # income grows with the customer number: one of them is enough
    assert set(issues["redundant"]["columns"]) <= {"customer", "income"}
    assert all(i["severity"] in ("warning", "info") for i in readiness["issues"])

    # Oversampling repeats training rows on purpose: no note about duplicates, and no
    # "Remove duplicates" fix after it (it would undo the oversampling).
    steps = [step("s1", "balance_classes", method="oversample")]
    readiness = preview(auth_client, prep["id"], steps=steps).data["readiness"]
    issues = {i["code"]: i for i in readiness["issues"]}
    assert "duplicate_rows" not in issues
    assert issues["split_overlap"]["fix"] is None
    assert issues["split_overlap"]["message"].endswith(
        "Remove duplicates before balancing the classes."
    )


# --------------------------------------------------------------------------- #
# Row analysis (Analysis → Rows) and the split by groups
# --------------------------------------------------------------------------- #


def scans_csv() -> bytes:
    from .test_data_rows import people_frame

    frame, _, _ = people_frame(60)
    return frame.to_csv(index=False).encode()


def test_row_analysis_and_similar_rows(auth_client):
    source = import_csv(auth_client, "scans.csv", scans_csv()).data
    prep = create_prep(auth_client, source["id"], settings={"target": "age"}).data
    url = f"/api/data/preps/{prep['id']}"
    response = auth_client.post(f"{url}/rows/", {}, format="json")
    assert response.status_code == 200, response.data
    data = response.data
    assert data["analysed"] == data["n_total"] == source["n_rows"] and not data["sampled"]
    assert data["features"]["used"] == "partnered" and data["id_column"] == "id"
    assert data["target"] == "age" and data["task"] == "regression"
    assert data["suggested"]["atypical"] is not None and data["suggested"]["twins"] is not None
    assert data["groups"]["count"] > 10 and len(data["rows"]["spread"]) == source["n_rows"]
    # Another threshold and the columns of my choice.
    body = {"twin_threshold": 0.0001, "features": {"mode": "columns", "columns": ["x0", "x1"]}}
    other = auth_client.post(f"{url}/rows/", body, format="json").data
    assert other["features"]["count"] == 2 and other["groups"]["count"] == 0
    # Unsaved steps are analysed.
    steps = [step("s1", "remove_atypical")]
    fewer = auth_client.post(f"{url}/rows/", {"steps": steps}, format="json").data
    assert fewer["n_total"] == source["n_rows"] - 10
    similar = auth_client.post(f"{url}/similar/", {"row": 0}, format="json")
    assert similar.status_code == 200, similar.data
    assert len(similar.data["neighbors"]) == 10 and similar.data["pair"]["shared"] == 40
    bad = auth_client.post(f"{url}/similar/", {"row": "x"}, format="json")
    assert bad.status_code == 400
    bad = auth_client.post(f"{url}/rows/", {"features": {"mode": "nope"}}, format="json")
    assert bad.status_code == 400 and "features.mode" in bad.data["detail"]


def test_export_with_a_group_split(auth_client, settings):
    source = import_csv(auth_client, "scans.csv", scans_csv()).data
    steps = [
        step("s1", "remove_atypical"),
        step("s2", "find_groups"),
        step("s3", "drop_columns", columns=["group", "id"]),
    ]
    prep = create_prep(
        auth_client,
        source["id"],
        steps=steps,
        settings={"target": "age", "split": {"group_by": "group"}},
    ).data
    result = preview(auth_client, prep["id"]).data
    groups = result["split"]["groups"]
    assert result["split"]["group_by"] == "group" and groups["mixed"] == 0
    assert groups["multi"] > 10
    codes = {i["code"] for i in result["readiness"]["issues"]}
    assert "near_twins" not in codes and "atypical_rows" not in codes
    response = export(auth_client, prep["id"])
    assert response.status_code == 201, response.data
    info = response.data["info"]
    assert sum(info["predefined_split"]) == source["n_rows"] - 10
