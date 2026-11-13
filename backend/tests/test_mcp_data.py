"""MCP tools of data preparation (the Data tab): importing files and links, preparations
and their steps, previews with the readiness check and its fixes, exports to training
datasets and downloads, with the roles of shared projects and the limits of the web app.

The end-to-end test at the end goes from a CSV file to predictions with a trained network.
"""

# ruff: noqa: F811 (the fixture training_dir is imported from test_training_api)

from __future__ import annotations

import base64
import time

import pytest
from django.contrib.auth import get_user_model
from rest_framework.test import APIClient
from rest_framework.throttling import ScopedRateThrottle

from brick_dnn_core import mcp_data
from brick_dnn_core.mcp import TOOLS
from brick_dnn_core.models import DataPrep, Dataset, DataSource, Project, ProjectMember
from brickdnn.training import torch_installed

from .helpers import chain
from .test_data_api import churn_csv, regions_csv, workbook
from .test_mcp import call, mcp_client, rpc
from .test_training_api import training_dir  # noqa: F401 (a fixture)

pytestmark = pytest.mark.django_db

READ_TOOLS = {
    "list_data_sources",
    "get_data_source",
    "list_data_preps",
    "get_data_prep",
    "list_prep_steps",
    "preview_data_prep",
    "get_data_prep_script",
}
WRITE_TOOLS = {
    "import_data",
    "create_data_prep",
    "update_data_prep",
    "apply_readiness_fix",
    "export_training_dataset",
    "delete_data_prep",
    "delete_data_source",
}
DESTRUCTIVE = {"update_data_prep", "delete_data_prep", "delete_data_source"}


@pytest.fixture(autouse=True)
def data_dirs(settings, tmp_path):
    """Fresh folders for imports and training datasets, both features switched on."""
    settings.DATA_PREP_DIR = tmp_path / "prepare"
    settings.TRAINING_DIR = tmp_path / "training"
    settings.DATA_PREP_ENABLED = True
    settings.TRAINING_ENABLED = True
    settings.DATA_MAX_UPLOAD_MB = 5
    settings.MCP_UPLOAD_MAX_MB = 5
    return tmp_path


@pytest.fixture
def team():
    """ada owns the project, bob edits it, cy views it, eve is a stranger."""
    users = {
        name: get_user_model().objects.create_user(name, password="correct-horse-1")
        for name in ("ada", "bob", "cy", "eve")
    }
    project = Project.objects.create(
        owner=users["ada"], name="Shared", graph={}, updated_by=users["ada"]
    )
    ProjectMember.objects.create(project=project, user=users["bob"], role="editor")
    ProjectMember.objects.create(project=project, user=users["cy"], role="viewer")
    return project, users


@pytest.fixture
def mcps(team):
    _, users = team
    return {name: mcp_client(user) for name, user in users.items()}


@pytest.fixture
def ada(mcps):
    return mcps["ada"]


def ok(client, tool, **arguments):
    result, text = call(client, tool, **arguments)
    assert "isError" not in result, text
    return result.get("structuredContent", text)


def error_of(client, tool, **arguments) -> str:
    result, text = call(client, tool, **arguments)
    assert result.get("isError") is True, text
    return text


def sales_csv(rows: int = 60) -> str:
    """Daily sales of three stores: a date, a category, a yes/no column, a number with gaps
    and the target. Needs three fixes (encode, date parts, impute) before training."""
    stores = ["North", "South", "East"]
    lines = ["date,store,promo,temperature,sales"]
    for i in range(rows):
        date = f"2024-{1 + (i // 28) % 12:02d}-{1 + i % 28:02d}"
        promo = "yes" if i % 3 == 0 else "no"
        temperature = "" if i % 11 == 0 else f"{10 + (i % 20) * 0.7:.1f}"
        sales = 100 + (i % 7) * 12 + (30 if promo == "yes" else 0) + (i % 5)
        lines.append(f"{date},{stores[i % 4 % 3]},{promo},{temperature},{sales:.2f}")
    return "\n".join(lines) + "\n"


def imported(client, text=None, **arguments) -> dict:
    return ok(client, "import_data", content=text or sales_csv(), **arguments)["source"]


def created(client, source_id, **arguments) -> dict:
    return ok(client, "create_data_prep", source_id=source_id, **arguments)


@pytest.fixture
def prep(ada) -> dict:
    """ada's preparation of the sales data with the target "sales", not ready yet."""
    source = imported(ada, filename="sales.csv")
    return created(ada, source["id"], target="sales")


def fix_ids(readiness: dict, severity: str = "errors") -> dict[str, str]:
    return {
        issue["code"]: issue["fix_id"] for issue in readiness.get(severity, []) if "fix_id" in issue
    }


# --------------------------------------------------------------------------- #
# Tools, access and availability
# --------------------------------------------------------------------------- #


def test_data_tools_are_listed_with_their_hints(team):
    _, users = team
    tools = {
        t["name"]: t for t in rpc(mcp_client(users["ada"]), "tools/list").json()["result"]["tools"]
    }
    assert READ_TOOLS | WRITE_TOOLS <= set(tools) == set(TOOLS)
    for name in READ_TOOLS:
        assert tools[name]["annotations"]["readOnlyHint"] is True, name
    for name in WRITE_TOOLS:
        annotations = tools[name]["annotations"]
        assert annotations["readOnlyHint"] is False, name
        assert annotations["destructiveHint"] is (name in DESTRUCTIVE), name
    assert tools["import_data"]["inputSchema"]["additionalProperties"] is False
    assert tools["apply_readiness_fix"]["inputSchema"]["required"] == ["id", "fix_id"]

    # The instructions describe the flow from a file to a training dataset.
    instructions = rpc(mcp_client(users["ada"]), "initialize", {}).json()["result"]["instructions"]
    for tool in ("import_data", "create_data_prep", "preview_data_prep", "apply_readiness_fix"):
        assert tool in instructions
    assert "export_training_dataset" in instructions


def test_read_only_tokens_only_read_data(team, prep):
    _, users = team
    client = mcp_client(users["ada"], read_only=True)
    names = {t["name"] for t in rpc(client, "tools/list").json()["result"]["tools"]}
    assert names >= READ_TOOLS and not names & WRITE_TOOLS
    prep_id = prep["preparation"]["id"]
    assert ok(client, "get_data_prep", id=prep_id)["name"] == "sales"
    assert "read-only" in error_of(client, "import_data", content=sales_csv())
    assert "read-only" in error_of(client, "delete_data_prep", id=prep_id)
    assert "read-only" in error_of(client, "update_data_prep", id=prep_id, name="x")
    assert DataSource.objects.count() == 1 and DataPrep.objects.get().name == "sales"


@pytest.mark.parametrize("cause", ["disabled", "no pandas"])
def test_unavailable_data_preparation(ada, settings, monkeypatch, cause):
    from brick_dnn_core import data_service

    if cause == "disabled":
        settings.DATA_PREP_ENABLED = False
        message = "Data preparation is disabled"
    else:
        monkeypatch.setattr(data_service, "dependencies_installed", lambda: False)
        message = "pandas is not installed"
    for tool, arguments in [
        ("list_data_sources", {}),
        ("list_prep_steps", {}),
        ("import_data", {"content": sales_csv()}),
        ("preview_data_prep", {"id": 1}),
    ]:
        assert message in error_of(ada, tool, **arguments)
    assert not DataSource.objects.exists()


# --------------------------------------------------------------------------- #
# import_data and the data sources
# --------------------------------------------------------------------------- #


def test_import_csv_text(ada, data_dirs):
    source = imported(ada, filename="sales.csv")
    assert source["name"] == "sales" and source["filename"] == "sales.csv"
    assert source["rows"] == 60 and source["format"] == "csv" and source["mine"] is True
    assert source["columns"] == {
        "date": "datetime",
        "store": "categorical",
        "promo": "boolean",
        "temperature": "numeric (6 missing)",
        "sales": "numeric",
    }
    assert "columns_note" not in source
    assert source["import"] == {
        "format": "csv",
        "delimiter": ",",
        "decimal": ".",
        "header": True,
        "encoding": "utf-8",
    }
    # Dates as ISO text, missing values as null, the first rows in file order.
    assert source["first_rows"][:2] == [
        {"date": "2024-01-01", "store": "North", "promo": 1, "temperature": None, "sales": 130},
        {"date": "2024-01-02", "store": "South", "promo": 0, "temperature": 10.7, "sales": 113},
    ]
    assert len(source["first_rows"]) == 5 and source["preparations"] == []
    # Stored like an upload in the web app.
    directory = data_dirs / "prepare" / "sources" / str(source["id"])
    assert (directory / "original.csv").read_text() == sales_csv()

    listing = ok(ada, "list_data_sources")
    assert [s["id"] for s in listing["sources"]] == [source["id"]]
    assert listing["import_data"]["max_mb"] == 5 and listing["import_data"]["links"] is True
    assert ".parquet" in listing["import_data"]["file_types"]
    detail = ok(ada, "get_data_source", id=source["id"], sample_rows=2)
    assert len(detail["first_rows"]) == 2 and detail["columns"] == source["columns"]
    assert "first_rows" not in ok(ada, "get_data_source", id=source["id"], sample_rows=0)


def test_import_options_and_name(ada):
    text = "Kunde;Umsatz;Region\n1;1,5;Nord\n2;2,25;Süd\n3;-999;Nord\n4;4,0;Süd\n"
    source = imported(
        ada, text, name="  Kunden ", delimiter=";", decimal=",", missing_values=["-999"]
    )
    assert source["name"] == "Kunden" and source["filename"] == "data.csv"
    assert source["import"]["delimiter"] == ";" and source["import"]["decimal"] == ","
    assert source["columns"] == {
        "Kunde": "numeric",
        "Umsatz": "numeric (1 missing)",
        "Region": "categorical",
    }
    assert [row["Umsatz"] for row in source["first_rows"]] == [1.5, 2.25, None, 4]

    # JSON as text, named by its file name.
    rows = '[{"a": 1, "b": "x"}, {"a": 2, "b": "y"}, {"a": 3, "b": null}]'
    source = imported(ada, rows, filename="orders.json")
    assert source["format"] == "json" and source["name"] == "orders"
    assert source["first_rows"][2] == {"a": 3, "b": None}


def test_import_binary_files_as_base64(ada):
    pytest.importorskip("openpyxl")
    content = workbook({"Regions": regions_csv(), "Churn": churn_csv()})
    encoded = base64.b64encode(content).decode()
    result = ok(ada, "import_data", content_base64=encoded, filename="Kunden.xlsx", sheet="Churn")
    source = result["source"]
    assert source["format"] == "xlsx" and source["name"] == "Kunden"
    assert source["import"]["sheet"] == "Churn" and source["import"]["sheets"] == [
        "Regions",
        "Churn",
    ]
    assert source["rows"] == 40 and list(source["columns"])[-1] == "churn"

    # A sheet that does not exist: the error lists the sheets.
    text = error_of(ada, "import_data", content_base64=encoded, filename="k.xlsx", sheet="Q3")
    assert "Regions, Churn" in text and "sheet" in text
    # Base64 with line breaks (as many encoders write it) is fine.
    wrapped = "\n".join(encoded[i : i + 76] for i in range(0, len(encoded), 76))
    assert ok(ada, "import_data", content_base64=wrapped, filename="k.xlsx")["source"]["rows"] == 40


def test_import_parquet(ada):
    pytest.importorskip("pyarrow")
    import io

    import pandas as pd

    buffer = io.BytesIO()
    pd.DataFrame({"x": [1.5, 2.5, None], "label": ["a", "b", "a"]}).to_parquet(buffer)
    encoded = base64.b64encode(buffer.getvalue()).decode()
    source = ok(ada, "import_data", content_base64=encoded, filename="events.parquet")["source"]
    assert source["format"] == "parquet" and source["rows"] == 3
    assert source["columns"] == {"x": "numeric (1 missing)", "label": "categorical"}


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({}, "exactly one of content"),
        ({"content": "a,b\n1,2\n", "url": "https://example.org/a.csv"}, "exactly one of"),
        ({"content": "  "}, "content of the file as text"),
        ({"content": 5}, "content of the file as text"),
        ({"content": "a,b\n1,2\n", "encoding": "latin-1"}, "already decoded"),
        ({"content": "a,b\n1,2\n", "filename": "notes.docx"}, "Upload a CSV, Excel"),
        ({"content": "a,b\n1,2\n", "filename": " "}, "'filename'"),
        ({"content": "a,b\n1,2\n", "name": "x" * 121}, "120 characters"),
        ({"content": "a,b\n1,2\n", "delimiter": "x"}, "Delimiter"),
        ({"content_base64": "aGVsbG8="}, "need 'filename'"),
        ({"content_base64": "not base64!", "filename": "a.xlsx"}, "not valid base64"),
        ({"content_base64": "", "filename": "a.xlsx"}, "encoded as base64"),
        ({"content": "a\ud800,b\n1,2\n"}, "not valid text"),
    ],
)
def test_bad_imports(ada, data_dirs, arguments, message):
    assert message in error_of(ada, "import_data", **arguments)
    assert not DataSource.objects.exists()
    assert not list((data_dirs / "prepare" / "sources").glob(".import-*"))


def test_import_limits(ada, settings):
    settings.MCP_UPLOAD_MAX_MB = 0.001  # about 1 KB
    text = error_of(ada, "import_data", content=sales_csv())
    assert "at most 0.0 MB as content" in text and "link (url)" in text
    # The limit of the Data tab applies too.
    settings.MCP_UPLOAD_MAX_MB = 5
    settings.DATA_MAX_UPLOAD_MB = 0.001
    assert "at most" in error_of(ada, "import_data", content=sales_csv())
    assert ok(ada, "list_data_sources")["import_data"]["max_mb"] == 0.0
    settings.DATA_MAX_ROWS = 10
    settings.DATA_MAX_UPLOAD_MB = 5
    assert "rows" in error_of(ada, "import_data", content=sales_csv())
    assert not DataSource.objects.exists()


def test_mcp_requests_may_carry_larger_files(ada, settings):
    """Up to about twice the import limit (JSON escapes the text, base64 adds a third)."""
    settings.TRAINING_MAX_UPLOAD_MB = 0.5  # upload_dataset's limit must not decide
    rows = ["a,b,c,target"] + [f"{i},{i % 7}.5,{i % 3},{i % 2}" for i in range(100_000)]
    big = "\n".join(rows) + "\n"
    assert 1.2 * 1024**2 < len(big) < 2 * 1024**2
    assert imported(ada, big)["rows"] == 100_000
    encoded = base64.b64encode(big.encode()).decode()
    assert ok(ada, "import_data", content_base64=encoded, filename="big.csv")["source"]["rows"]

    settings.MCP_UPLOAD_MAX_MB = 0.5
    response = rpc(ada, "tools/call", {"name": "import_data", "arguments": {"content": big}})
    assert response.status_code == 413


def test_imports_count_against_the_upload_rate_limit(ada, monkeypatch):
    monkeypatch.setitem(ScopedRateThrottle.THROTTLE_RATES, "uploads", "2/hour")
    imported(ada)
    # Uploads of training datasets count too, as in the web app.
    ok(ada, "upload_dataset", name="Tiny", csv="a,b\n1,2\n3,4\n5,6\n7,8\n")
    assert "Too many" in error_of(ada, "import_data", content=sales_csv())
    assert DataSource.objects.count() == 1


@pytest.fixture
def file_server():
    from .test_url_import import Server

    server = Server()
    yield server
    server.close()


def test_import_from_a_link(ada, settings, file_server):
    settings.DATA_URL_ALLOW_PRIVATE = True  # the test server runs on 127.0.0.1
    file_server.routes["/data/sales.csv?token=secret"] = (
        200,
        {"Content-Type": "text/csv"},
        sales_csv().encode(),
    )
    url = f"{file_server.url}/data/sales.csv?token=secret"
    source = ok(ada, "import_data", url=url, name="Sales")["source"]
    assert source["name"] == "Sales" and source["filename"] == "sales.csv"
    assert source["rows"] == 60
    # The query of a link can hold secrets: it is not stored.
    assert source["import"]["url"] == f"{file_server.url}/data/sales.csv?…"

    text = error_of(ada, "import_data", url=f"{file_server.url}/missing.csv")
    assert "Nothing was found at this link" in text


def test_links_go_through_the_checks_against_private_addresses(ada, file_server):
    file_server.routes["/sales.csv"] = (200, {}, sales_csv().encode())
    for url in (f"{file_server.url}/sales.csv", "http://169.254.169.254/latest/meta-data/"):
        assert "not a public internet address" in error_of(ada, "import_data", url=url)
    assert file_server.requests == []  # nothing was requested
    assert "user name or password" in error_of(
        ada, "import_data", url="https://user:pw@example.org/data.csv"
    )
    assert "http" in error_of(ada, "import_data", url="file:///etc/passwd")
    assert not DataSource.objects.exists()


def test_link_imports_can_be_switched_off(ada, settings):
    settings.DATA_URL_IMPORT_ENABLED = False
    text = error_of(ada, "import_data", url="https://example.org/data.csv")
    assert "Importing from links is switched off on this server." in text
    assert ok(ada, "list_data_sources")["import_data"]["links"] is False


# --------------------------------------------------------------------------- #
# Preparations and the readiness check
# --------------------------------------------------------------------------- #


def test_create_data_prep_checks_the_readiness(ada, prep):
    preparation, check = prep["preparation"], prep["check"]
    assert preparation["name"] == "sales" and preparation["role"] == "owner"
    assert preparation["settings"] == {
        "target": "sales",
        "task": "auto",
        "split": {"val": 0.15, "test": 0.15, "seed": 42, "stratify": True, "group_by": None},
    }
    assert preparation["steps"] == [] and preparation["datasets"] == []
    readiness = check["readiness"]
    assert readiness["ready"] is False and readiness["task"] == "regression"
    assert readiness["split"] == {"train": 42, "validation": 9, "test": 9}
    assert set(fix_ids(readiness)) == {"categorical_features", "date_features", "missing_values"}
    issue = next(i for i in readiness["errors"] if i["code"] == "missing_values")
    assert issue["fix"] == {
        "label": "Impute temperature (median)",
        "type": "impute",
        "params": {"columns": ["temperature"], "strategy": "median"},
    }
    assert "fix_all" in readiness
    assert check["columns"] == {
        "date": "datetime",
        "store": "categorical",
        "promo": "boolean",
        "temperature": "numeric",
        "sales": "numeric",
    }
    # The source lists its preparation.
    source = ok(ada, "get_data_source", id=preparation["source"]["id"])
    assert source["preparations"] == [{"id": preparation["id"], "name": "sales"}]
    listing = ok(ada, "list_data_preps")["preparations"]
    assert listing[0]["id"] == preparation["id"] and listing[0]["target"] == "sales"


def test_create_data_prep_with_steps_split_and_without_target(ada):
    source = imported(ada)
    result = created(
        ada,
        source["id"],
        name="Sales v1",
        task="regression",
        split={"val": 0.2, "test": 0, "seed": 7},
        steps=[
            {"type": "impute", "params": {"strategy": "mean"}},
            {"type": "drop_columns", "params": {"columns": ["store"]}},
        ],
    )
    preparation = result["preparation"]
    assert preparation["name"] == "Sales v1"
    assert preparation["settings"]["split"] == {
        "val": 0.2,
        "test": 0.0,
        "seed": 7,
        "stratify": True,
        "group_by": None,
    }
    # New steps get ids and the defaults of their type, like steps added in the web app.
    first, second = preparation["steps"]
    assert first["id"] == "s1" and first["enabled"] is True
    assert first["params"] == {
        "columns": [],
        "strategy": "mean",
        "fill_value": None,
        "k": 5,
        "add_indicator": False,
    }
    assert second["id"] == "s2"
    readiness = result["check"]["readiness"]
    assert readiness["ready"] is False and readiness["errors"][0]["code"] == "no_target"
    assert readiness["choose_target_from"] == ["date", "promo", "temperature", "sales"]


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"source_id": 999}, "does not exist"),
        ({"source_id": "first"}, "Choose one of your data sources"),
        ({"task": "forecast"}, "Task: choose auto, classification or regression."),
        ({"split": {"val": -0.1}}, "Validation share"),
        ({"steps": [{"type": "nope"}]}, "Unknown step type 'nope'"),
        ({"steps": [{"id": "a", "type": "impute"}, {"id": "a", "type": "impute"}]}, "used by two"),
        ({"steps": "impute"}, "The steps must be a list."),
        ({"name": "x" * 121}, "120 characters"),
        ({"project_id": "1"}, "'project_id' must be the number"),
    ],
)
def test_create_data_prep_rejects(ada, arguments, message):
    source = imported(ada)
    arguments = {"source_id": source["id"], **arguments}
    assert message in error_of(ada, "create_data_prep", **arguments)
    assert not DataPrep.objects.exists()


def test_apply_readiness_fixes(ada, prep):
    prep_id = prep["preparation"]["id"]
    ids = fix_ids(prep["check"]["readiness"])
    result = ok(ada, "apply_readiness_fix", id=prep_id, fix_id=ids["missing_values"])
    assert result["applied"] == [
        {
            "fix_id": ids["missing_values"],
            "label": "Impute temperature (median)",
            "step": "s1",
            "type": "impute",
        }
    ]
    readiness = result["check"]["readiness"]
    assert set(fix_ids(readiness)) == {"categorical_features", "date_features"}
    assert (
        result["check"]["steps"][0]["summary"]
        == "Filled 6 missing values in 1 column with the median."
    )
    # Saved like a step added in the web app (with the defaults of its type).
    step = DataPrep.objects.get(pk=prep_id).steps[0]
    assert step["params"]["strategy"] == "median" and step["params"]["k"] == 5

    # Fix ids belong to the current state: an old one is not applied twice.
    text = error_of(ada, "apply_readiness_fix", id=prep_id, fix_id=ids["missing_values"])
    assert "No fix 'missing_values-" in text and ids["date_features"] in text
    assert "One-hot encode store" in text

    # All errors at once ("Fix all").
    result = ok(ada, "apply_readiness_fix", id=prep_id, fix_id="all")
    assert [a["type"] for a in result["applied"]] == ["encode", "datetime_features"]
    assert [s["id"] for s in result["steps"]] == ["s1", "s2", "s3"]
    readiness = result["check"]["readiness"]
    assert readiness["ready"] is True and "errors" not in readiness
    assert readiness["features"] == 8
    text = error_of(ada, "apply_readiness_fix", id=prep_id, fix_id="all")
    assert "No error has a fix" in text and "ready for training" in text
    # Warnings have fixes too.
    constant = fix_ids(readiness, "warnings")["constant"]
    result = ok(ada, "apply_readiness_fix", id=prep_id, fix_id=constant)
    assert result["applied"][0]["label"] == "Drop date_year"


def test_fixes_go_in_front_of_a_failing_step(ada, prep):
    prep_id = prep["preparation"]["id"]
    broken = {"type": "drop_columns", "params": {"columns": ["nope"]}}
    result = ok(ada, "update_data_prep", id=prep_id, add_steps=[broken])
    report = result["check"]["steps"][0]
    assert report["status"] == "error"
    assert report["error"] == "Drop columns: Columns: there is no column 'nope' (any more)."
    assert report["columns_before_step"] == ["date", "store", "promo", "temperature", "sales"]
    assert "list_prep_steps" in report["help"]
    # The readiness of the data before the failing step comes first.
    assert result["check"]["readiness"]["errors"][0]["code"] == "step_failed"
    fixes = fix_ids(result["check"]["readiness"])
    result = ok(ada, "apply_readiness_fix", id=prep_id, fix_id=fixes["missing_values"])
    assert [s["type"] for s in result["steps"]] == ["impute", "drop_columns"]
    assert [s["status"] for s in result["check"]["steps"]] == ["ok", "error"]


def test_update_data_prep_edits_steps(ada, prep):
    prep_id = prep["preparation"]["id"]
    result = ok(
        ada,
        "update_data_prep",
        id=prep_id,
        add_steps=[
            {"type": "impute"},
            {"type": "encode", "params": {"columns": ["store"]}},
            {"type": "datetime_features", "params": {"parts": ["month"]}, "after": "s1"},
        ],
    )
    steps = result["preparation"]["steps"]
    assert [(s["id"], s["type"]) for s in steps] == [
        ("s1", "impute"),
        ("s3", "datetime_features"),
        ("s2", "encode"),
    ]
    assert result["check"]["readiness"]["ready"] is True
    assert "date_month" in result["check"]["columns"]

    result = ok(
        ada,
        "update_data_prep",
        id=prep_id,
        edit_steps=[{"id": "s1", "params": {"strategy": "constant", "fill_value": 0}}],
        move_steps=[{"id": "s2", "after": None}],
        remove_steps=["s3"],
    )
    steps = result["preparation"]["steps"]
    assert [s["id"] for s in steps] == ["s2", "s1"]
    assert steps[1]["params"]["strategy"] == "constant" and steps[1]["params"]["fill_value"] == 0
    assert set(fix_ids(result["check"]["readiness"])) == {"date_features"}
    summary = result["check"]["steps"][1]["summary"]
    assert "with 0" in summary or "fixed value" in summary, summary

    # null resets a param to its default; enabled switches a step off.
    result = ok(
        ada,
        "update_data_prep",
        id=prep_id,
        edit_steps=[
            {"id": "s1", "params": {"fill_value": None, "strategy": None}, "enabled": False}
        ],
    )
    step = result["preparation"]["steps"][1]
    assert "strategy" not in step["params"] and step["enabled"] is False
    assert result["check"]["steps"][1]["status"] == "disabled"

    # The whole list at once.
    result = ok(ada, "update_data_prep", id=prep_id, steps=[{"type": "impute"}])
    assert [s["id"] for s in result["preparation"]["steps"]] == ["s1"]


def test_update_data_prep_settings_and_name(ada, prep):
    prep_id = prep["preparation"]["id"]
    result = ok(ada, "update_data_prep", id=prep_id, split={"test": 0.3}, name="Renamed")
    settings = result["preparation"]["settings"]
    assert settings["split"] == {
        "val": 0.15,
        "test": 0.3,
        "seed": 42,
        "stratify": True,
        "group_by": None,
    }
    assert settings["target"] == "sales" and result["preparation"]["name"] == "Renamed"
    result = ok(ada, "update_data_prep", id=prep_id, target="promo", task="classification")
    readiness = result["check"]["readiness"]
    assert readiness["target"] == "promo" and readiness["class_names"] == ["false", "true"]
    result = ok(ada, "update_data_prep", id=prep_id, settings={"target": "nope"})
    readiness = result["check"]["readiness"]
    assert result["preparation"]["settings"]["split"]["test"] == 0.15  # all settings replaced
    assert readiness["errors"][0]["code"] == "target_not_found"
    assert "sales" in readiness["choose_target_from"]


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({}, "Nothing to change"),
        ({"steps": [], "add_steps": [{"type": "impute"}]}, "not both"),
        ({"add_steps": [{"type": "nope"}]}, "Unknown step type 'nope'"),
        ({"add_steps": [{"type": "impute", "position": 1}]}, "unknown key 'position'"),
        ({"add_steps": [{"type": "impute", "after": "s9"}]}, "There is no step 's9'"),
        ({"add_steps": [{"type": "impute", "params": "median"}]}, "must be an object"),
        ({"remove_steps": ["s1"]}, "There is no step 's1'. The steps: none."),
        ({"edit_steps": [{"id": "s1"}]}, "There is no step 's1'"),
        ({"edit_steps": "s1"}, "must be a list"),
        ({"move_steps": [{"after": "s1"}]}, "{id, after}"),
        ({"add_steps": [{"type": "impute"}, {"type": "join", "params": {"source": 1}}]}, None),
        ({"split": {"val": 0.6, "test": 0.6}}, "at most 0.9"),
        ({"settings": "all"}, "'settings' must be an object"),
        ({"source_id": 999}, "does not exist"),
        ({"name": "x" * 121}, "120 characters"),
    ],
)
def test_update_data_prep_rejects(ada, prep, arguments, message):
    prep_id = prep["preparation"]["id"]
    before = DataPrep.objects.get(pk=prep_id)
    result, text = call(ada, "update_data_prep", id=prep_id, **arguments)
    if message is None:  # join goes before the other steps, so this is fine
        assert "isError" not in result, text
        types = [s["type"] for s in result["structuredContent"]["preparation"]["steps"]]
        assert types == ["join", "impute"]
        return
    assert result.get("isError") is True and message in text, text
    after = DataPrep.objects.get(pk=prep_id)
    assert (after.steps, after.settings, after.name) == (before.steps, before.settings, before.name)


def test_combine_steps_must_come_first(ada, prep):
    other = imported(ada, regions_csv().decode())
    join = {"id": "j", "type": "join", "params": {"source": other["id"]}}
    text = error_of(
        ada,
        "update_data_prep",
        id=prep["preparation"]["id"],
        steps=[{"id": "i", "type": "impute"}, join],
    )
    assert "must come before all other steps" in text


def test_preview_data_prep(ada, prep):
    prep_id = prep["preparation"]["id"]
    view = ok(ada, "preview_data_prep", id=prep_id, sample_rows=3)
    assert view["rows"] == 60 and view["task"] == "regression" and view["steps"] == []
    profiles = {p["name"]: p for p in view["profiles"]}
    assert profiles["date"] == {
        "name": "date",
        "kind": "datetime",
        "missing": 0,
        "unique": 60,
        "min": "2024-01-01",
        "max": "2024-03-04",
    }
    temperature = profiles["temperature"]
    assert temperature["missing"] == 6 and temperature["min"] == 10 and temperature["max"] == 23.3
    assert 0 <= temperature["target_association"] <= 1
    assert profiles["store"]["top"] == {"North": 30, "South": 15, "East": 15}
    assert view["target"]["column"] == "sales" and view["target"]["min"] == 100
    sample = view["sample"]
    assert len(sample["rows"]) == 3 and len(sample["split"]) == 3
    assert set(sample["rows"][0]) == {"date", "store", "promo", "temperature", "sales"}
    assert "unsaved" not in view

    # Unsaved steps and settings, only some columns, the data after one step.
    steps = [
        {"id": "a", "type": "impute"},
        {"id": "b", "type": "datetime_features", "params": {"parts": ["month"]}},
        {"id": "c", "type": "encode", "params": {"columns": ["store"]}},
    ]
    view = ok(
        ada,
        "preview_data_prep",
        id=prep_id,
        steps=steps,
        columns=["date_month", "sales"],
        split={"test": 0.2},
    )
    assert [p["name"] for p in view["profiles"]] == ["date_month", "sales"]
    assert set(view["sample"]["rows"][0]) == {"date_month", "sales"}
    assert view["readiness"]["ready"] is True and view["readiness"]["split"]["test"] == 12
    assert "without saving" in view["unsaved"]
    assert DataPrep.objects.get(pk=prep_id).steps == []
    view = ok(ada, "preview_data_prep", id=prep_id, steps=steps, upto="a")
    assert [s["status"] for s in view["steps"]] == ["ok", "not_run", "not_run"]
    assert "date" in view["columns"]

    assert "There is no column 'month'" in error_of(
        ada, "preview_data_prep", id=prep_id, columns=["month"]
    )
    assert "There is no step 'zz'" in error_of(ada, "preview_data_prep", id=prep_id, upto="zz")
    assert "No such data preparation" in error_of(ada, "preview_data_prep", id=999)
    assert "'sample_rows'" in error_of(ada, "preview_data_prep", id=prep_id, sample_rows=-1)


def test_previews_count_against_the_data_rate_limit(ada, prep, monkeypatch):
    # The check after create_data_prep was the first preview.
    monkeypatch.setitem(ScopedRateThrottle.THROTTLE_RATES, "data", "2/hour")
    prep_id = prep["preparation"]["id"]
    ok(ada, "preview_data_prep", id=prep_id)
    assert "Too many previews" in error_of(ada, "preview_data_prep", id=prep_id)
    # Changes are still saved; their check waits.
    result = ok(ada, "update_data_prep", id=prep_id, name="Later")
    assert result["preparation"]["name"] == "Later" and "skipped" in result["check"]
    assert "Too many" in error_of(ada, "apply_readiness_fix", id=prep_id, fix_id="all")


def test_list_prep_steps(ada):
    catalog = ok(ada, "list_prep_steps")
    steps = {s["type"]: s for s in catalog["steps"]}
    assert {"impute", "encode", "outliers", "datetime_features", "join", "keep_columns"} <= set(
        steps
    )
    impute = steps["impute"]
    assert impute["label"] == "Fill missing values" and impute["group"] == "Missing values"
    assert impute["params"]["strategy"].startswith("one of mean | median")
    assert 'default "median"' in impute["params"]["strategy"]
    assert (
        impute["params"]["k"]
        == "a whole number from 1 to 50; default 5; only used if strategy is knn"
    )
    assert "empty: All columns with missing values" in impute["params"]["columns"]
    assert "only used if add_source_column is true" in steps["append"]["params"]["source_column"]
    assert "Combine steps" in catalog["format"]
    group = steps["group_features"]
    assert group["label"] == "Group statistics" and group["group"] == "Feature engineering"
    assert "training rows" in group["description"]
    assert group["params"]["stats"].startswith("a list of mean | median")
    assert "(categorical or boolean columns)" in group["params"]["group_by"]
    assert steps["combine_columns"]["params"]["operations"].startswith("a list of sum | mean")
    assert "from 1 to 2000; default 200" in steps["interactions"]["params"]["max_columns"]

    detail = ok(ada, "list_prep_steps", type="impute")
    params = {p["name"]: p for p in detail["params"]}
    assert params["strategy"]["options"]["knn"] == "Nearest neighbours (KNN)"
    assert "training rows" in params["strategy"]["help"]
    assert detail["example"] == {
        "type": "impute",
        "params": {"columns": [], "strategy": "median", "add_indicator": False},
    }
    assert detail["learns_from_training_rows"] is True

    extract = steps["extract_text"]
    assert extract["group"] == "Columns" and "C85 → C" in extract["description"]
    assert extract["params"]["part"].startswith("one of first | last | pattern | word")
    assert "only used if part is pattern" in extract["params"]["pattern"]
    detail = ok(ada, "list_prep_steps", type="extract_text")
    assert detail["example"]["params"]["mapping"]["T"] == 8
    assert detail["example"]["params"]["column"] == "Cabin"
    assert "map(upper(left(`Cabin`, 1))" in steps["formula"]["description"]
    detail = ok(ada, "list_prep_steps", type="formula")
    assert detail["example"]["params"]["name"] == "Deck"
    assert detail["example"]["params"]["expression"].startswith("map(upper(left(`Cabin`, 1)), {")
    expression = next(p for p in detail["params"] if p["name"] == "expression")
    assert "split_part(x" in expression["help"] and "extract(x" in expression["help"]
    text = error_of(ada, "list_prep_steps", type="impute_all")
    assert "Unknown step type 'impute_all'" in text and "impute" in text


# --------------------------------------------------------------------------- #
# Export and downloads
# --------------------------------------------------------------------------- #


def ready(client, prep) -> int:
    prep_id = prep["preparation"]["id"]
    assert ok(client, "apply_readiness_fix", id=prep_id, fix_id="all")["check"]["readiness"][
        "ready"
    ]
    return prep_id


def test_export_training_dataset(ada, prep, training_dir):
    prep_id = prep["preparation"]["id"]
    text = error_of(ada, "export_training_dataset", id=prep_id)
    assert "not ready for training" in text and "categorical" in text
    assert "apply_readiness_fix" in text
    assert not Dataset.objects.exists()

    ready(ada, prep)
    result = ok(ada, "export_training_dataset", id=prep_id, name="Sales 2024")
    dataset = result["dataset"]
    assert result["reference"] == {"kind": "upload", "id": dataset["id"]}
    assert dataset["name"] == "Sales 2024" and dataset["prepared_from"]["id"] == prep_id
    assert result["task"] == "regression" and result["target"] == "sales"
    assert result["features"] == [
        "date_year",
        "date_month",
        "date_weekday",
        "store=East",
        "store=North",
        "store=South",
        "promo",
        "temperature",
    ]
    assert result["split"] == {"train": 42, "validation": 9, "test": 9}
    assert "Input of shape [8]" in result["next"] and "1 output" in result["next"]
    # Listed like an export in the web app, with the training datasets of the preparation.
    options = ok(ada, "get_training_options")
    assert [d["id"] for d in options["your_datasets"]] == [dataset["id"]]
    datasets = ok(ada, "get_data_prep", id=prep_id)["datasets"]
    assert datasets[0]["reference"] == result["reference"]

    graph = chain(("Input", {"shape": [8]}), ("Linear", {"out_features": 1}))
    check = ok(ada, "check_training", graph=graph, dataset=result["reference"])
    assert check["ok"] is True and check["plan"]["task"] == "regression"


def test_export_needs_training_and_counts_as_an_upload(
    ada, prep, training_dir, settings, monkeypatch
):
    prep_id = ready(ada, prep)
    settings.TRAINING_ENABLED = False
    assert "Training is disabled" in error_of(ada, "export_training_dataset", id=prep_id)
    settings.TRAINING_ENABLED = True
    monkeypatch.setitem(ScopedRateThrottle.THROTTLE_RATES, "uploads", "2/hour")  # 1 import
    ok(ada, "export_training_dataset", id=prep_id)
    assert "Too many" in error_of(ada, "export_training_dataset", id=prep_id)
    assert Dataset.objects.count() == 1


def test_downloads(ada, prep):
    prep_id = ready(ada, prep)
    script = ok(ada, "get_data_prep_script", id=prep_id)
    assert isinstance(script, str)
    assert "def prepare(df" in script and "TARGET = 'sales'" in script
    compile(script, "sales_prepare.py", "exec")

    recipe = ok(ada, "get_data_prep_script", id=prep_id, format="recipe")
    assert recipe["filename"] == "sales.recipe.json"
    assert [s["type"] for s in recipe["recipe"]["steps"]] == [
        "encode",
        "datetime_features",
        "impute",
    ]

    table = ok(ada, "get_data_prep_script", id=prep_id, format="csv", max_rows=2)
    assert table["rows_shown"] == 2 and table["truncated"] is True
    lines = table["csv"].splitlines()
    assert lines[0].startswith("date_year,date_month") and lines[0].endswith(",sales,split")
    assert len(lines) == 3
    everything = ok(ada, "get_data_prep_script", id=prep_id, format="csv", max_rows=1000)
    assert everything["rows_shown"] == 60 and everything["truncated"] is False

    assert "'format'" in error_of(ada, "get_data_prep_script", id=prep_id, format="xlsx")
    assert "'max_rows'" in error_of(ada, "get_data_prep_script", id=prep_id, max_rows=0)


def test_large_downloads_are_cut(ada, prep, monkeypatch):
    prep_id = ready(ada, prep)
    monkeypatch.setattr(mcp_data, "MAX_DOWNLOAD_BYTES", 500)
    script = ok(ada, "get_data_prep_script", id=prep_id)
    assert len(script) < 800 and "cut off here" in script and "web app" in script
    recipe = ok(ada, "get_data_prep_script", id=prep_id, format="recipe")
    assert recipe["truncated"] is True and "recipe" not in recipe
    table = ok(ada, "get_data_prep_script", id=prep_id, format="csv", max_rows=1000)
    assert table["truncated"] is True and table["rows_shown"] < 60


def test_downloads_of_a_failing_preparation(ada, prep):
    prep_id = prep["preparation"]["id"]
    ok(
        ada,
        "update_data_prep",
        id=prep_id,
        add_steps=[{"type": "drop_columns", "params": {"columns": ["x"]}}],
    )
    text = error_of(ada, "get_data_prep_script", id=prep_id)
    assert "Step 1 (Drop columns) failed" in text and "Fix it before downloading" in text


# --------------------------------------------------------------------------- #
# Sharing in projects and deleting
# --------------------------------------------------------------------------- #


def test_roles_of_a_shared_preparation(team, mcps, training_dir):
    project, _ = team
    ada, bob, cy, eve = (mcps[name] for name in ("ada", "bob", "cy", "eve"))
    source = imported(ada)
    result = created(ada, source["id"], target="sales", project_id=project.pk)
    prep_id = result["preparation"]["id"]
    assert result["preparation"]["shared_in"] == [{"id": project.pk, "name": "Shared"}]

    # Everybody in the project sees it, with their role; strangers don't.
    for client, role in ((bob, "editor"), (cy, "viewer")):
        assert [(p["id"], p["role"]) for p in ok(client, "list_data_preps")["preparations"]] == [
            (prep_id, role)
        ]
        assert ok(client, "get_data_prep", id=prep_id)["owner"] == "ada"
        assert ok(client, "preview_data_prep", id=prep_id)["rows"] == 60
        assert "def prepare(df" in ok(client, "get_data_prep_script", id=prep_id)
        # The source of a shared preparation can be looked at, but is not listed.
        assert ok(client, "get_data_source", id=source["id"])["mine"] is False
        assert ok(client, "list_data_sources")["sources"] == []
    assert ok(eve, "list_data_preps") == {"preparations": []}
    for tool in ("get_data_prep", "preview_data_prep", "update_data_prep", "delete_data_prep"):
        assert "No such data preparation" in error_of(eve, tool, id=prep_id)
    assert "No such data source" in error_of(eve, "get_data_source", id=source["id"])

    # Viewers cannot change it, export it or apply fixes.
    for tool, arguments in (
        ("update_data_prep", {"name": "Mine"}),
        ("apply_readiness_fix", {"fix_id": "all"}),
        ("export_training_dataset", {}),
    ):
        assert "read-only" in error_of(cy, tool, id=prep_id, **arguments)
    # Editors can, also with another source of the owner, but not with their own.
    ok(bob, "apply_readiness_fix", id=prep_id, fix_id="all")
    result = ok(bob, "update_data_prep", id=prep_id, name="Sales (Bob)")
    assert result["preparation"]["name"] == "Sales (Bob)"
    assert result["preparation"]["role"] == "editor"
    own = imported(bob)
    assert "does not exist" in error_of(bob, "update_data_prep", id=prep_id, source_id=own["id"])
    exported = ok(bob, "export_training_dataset", id=prep_id)
    # The dataset belongs to the owner of the preparation and is shared in its project.
    assert exported["dataset"]["owner"] == "ada" and exported["dataset"]["mine"] is False
    assert exported["dataset"]["shared_in"] == [{"id": project.pk, "name": "Shared"}]
    assert ok(cy, "get_training_options")["your_datasets"][0]["id"] == exported["reference"]["id"]

    # Only the owner deletes it, and its source.
    for client in (bob, cy):
        assert "Only its owner (ada)" in error_of(client, "delete_data_prep", id=prep_id)
        assert "Only its owner (ada)" in error_of(client, "delete_data_source", id=source["id"])
    # Sharing needs a project the user owns or edits.
    text = error_of(cy, "create_data_prep", source_id=imported(cy)["id"], project_id=project.pk)
    assert "projects the user owns or edits" in text


def test_delete_data_prep_and_source(
    ada, prep, training_dir, data_dirs, django_capture_on_commit_callbacks
):
    prep_id = ready(ada, prep)
    source_id = prep["preparation"]["source"]["id"]
    dataset_id = ok(ada, "export_training_dataset", id=prep_id)["reference"]["id"]

    # As in the web app: not while a preparation uses it.
    text = error_of(ada, "delete_data_source", id=source_id)
    assert (
        "Used by the preparation “sales”; delete it first." in text and "delete_data_prep" in text
    )

    result = ok(ada, "delete_data_prep", id=prep_id)
    assert result == {
        "deleted": True,
        "id": prep_id,
        "name": "sales",
        "note": "The 1 training dataset made from it is kept.",
    }
    assert not DataPrep.objects.exists()
    assert Dataset.objects.get(pk=dataset_id).prepared_from is None
    assert "No such data preparation" in error_of(ada, "delete_data_prep", id=prep_id)

    directory = data_dirs / "prepare" / "sources" / str(source_id)
    assert directory.is_dir()
    with django_capture_on_commit_callbacks(execute=True):
        result = ok(ada, "delete_data_source", id=source_id)
    assert result == {"deleted": True, "id": source_id, "name": "sales"}
    assert not DataSource.objects.exists() and not directory.exists()
    assert "No such data source" in error_of(ada, "delete_data_source", id=source_id)
    assert "number" in error_of(ada, "delete_data_source", id="first")


def test_rest_and_mcp_see_the_same_preparation(ada, prep, team):
    """Changes over MCP appear in the web app (the REST API) and the other way round."""
    _, users = team
    prep_id = prep["preparation"]["id"]
    ok(ada, "update_data_prep", id=prep_id, add_steps=[{"type": "impute"}])
    client = APIClient()
    client.force_authenticate(users["ada"])
    data = client.get(f"/api/data/preps/{prep_id}/").data
    assert [s["type"] for s in data["steps"]] == ["impute"]
    client.patch(f"/api/data/preps/{prep_id}/", {"name": "From the web"}, format="json")
    assert ok(ada, "get_data_prep", id=prep_id)["name"] == "From the web"


# --------------------------------------------------------------------------- #
# From a CSV file to predictions
# --------------------------------------------------------------------------- #

needs_torch = pytest.mark.skipif(not torch_installed(), reason="PyTorch is not installed")


def wait_for_run(client, run_id: str, timeout: float = 120) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        run = ok(client, "get_training_run", id=run_id)
        if run["status"] in ("completed", "stopped", "failed"):
            return run
        assert time.monotonic() < deadline, run
        time.sleep(0.25)


@needs_torch
def test_from_a_csv_file_to_predictions(team, training_dir, settings):
    """What an assistant does: import a CSV file, prepare it (a step of its own and the
    readiness fixes), export it, check and train a network and predict with it."""
    settings.TRAINING_RUNNER_COMMAND = None  # the real training and prediction processes
    settings.TRAINING_PREDICT_COMMAND = None
    _, users = team
    mcp = mcp_client(users["ada"])

    source = ok(mcp, "import_data", content=sales_csv(120), filename="sales.csv")["source"]
    prep = ok(mcp, "create_data_prep", source_id=source["id"], target="sales")
    prep_id = prep["preparation"]["id"]
    assert not prep["check"]["readiness"]["ready"]

    # A step of its own: clip outliers of the temperature.
    result = ok(
        mcp,
        "update_data_prep",
        id=prep_id,
        add_steps=[{"type": "clip", "params": {"columns": ["temperature"]}}],
    )
    assert result["check"]["steps"][0]["status"] == "ok"
    # And the fixes the readiness check suggests.
    fixes = fix_ids(result["check"]["readiness"])
    ok(mcp, "apply_readiness_fix", id=prep_id, fix_id=fixes["missing_values"])
    result = ok(mcp, "apply_readiness_fix", id=prep_id, fix_id="all")
    assert result["check"]["readiness"]["ready"] is True

    preview = ok(mcp, "preview_data_prep", id=prep_id, sample_rows=10)
    test_rows = [
        row
        for row, split in zip(preview["sample"]["rows"], preview["sample"]["split"], strict=True)
        if split != "train"
    ] or preview["sample"]["rows"]

    exported = ok(mcp, "export_training_dataset", id=prep_id)
    features = exported["features"]
    graph = chain(
        ("Input", {"shape": [len(features)]}),
        ("Linear", {"out_features": 16}),
        ("Activation", {"function": "relu"}),
        ("Linear", {"out_features": 1}),
    )
    check = ok(mcp, "check_training", graph=graph, dataset=exported["reference"])
    assert check["ok"] is True, check
    run = ok(
        mcp,
        "start_training",
        graph=graph,
        dataset=exported["reference"],
        name="Sales forecast",
        config={"epochs": 3, "batch_size": 16, "device": "cpu", "seed": 1},
    )
    run = wait_for_run(mcp, run["id"])
    assert run["status"] == "completed", run["message"]

    # Predict with prepared rows (the target in them is ignored).
    answer = ok(mcp, "predict", run_id=run["id"], inputs=test_rows[:3])
    assert answer["task"] == "regression" and len(answer["predictions"]) == len(test_rows[:3])
    assert all(isinstance(p["value"], float) for p in answer["predictions"])
    # Or a new row given by its prepared feature values.
    row = {name: test_rows[0][name] for name in features}
    assert len(ok(mcp, "predict", run_id=run["id"], inputs=[row])["predictions"]) == 1


# --------------------------------------------------------------------------- #
# Row structure: the summary in previews, the split by groups and its fix
# --------------------------------------------------------------------------- #


def test_row_analysis_summary_and_group_split(ada):
    from .test_data_rows import people_frame

    frame, _, fake = people_frame(60)
    source = imported(ada, frame.to_csv(index=False), filename="scans.csv")
    prep = created(ada, source["id"], target="age")
    prep_id = prep["preparation"]["id"]
    preview = ok(ada, "preview_data_prep", id=prep_id, sample_rows=0)
    rows = preview["row_analysis"]
    assert rows["atypical_rows"] == int(fake.sum()) and rows["groups"] > 10
    assert rows["features"]["used"] == "partnered"
    assert len(rows["nn_ratio_histogram"]["counts"]) == 11
    warnings = {i["code"]: i for i in preview["readiness"]["warnings"]}
    twins = warnings["near_twins"]
    assert twins["fix"]["settings"] == {"split": {"group_by": "group"}}
    # The fix adds the step and splits by its column.
    result = ok(ada, "apply_readiness_fix", id=prep_id, fix_id=twins["fix_id"])
    assert result["applied"][0]["type"] == "find_groups"
    saved = ok(ada, "get_data_prep", id=prep_id)
    assert saved["settings"]["split"]["group_by"] == "group"
    preview = ok(ada, "preview_data_prep", id=prep_id, sample_rows=0)
    assert preview["split_groups"]["group_by"] == "group"
    assert preview["split_groups"]["mixed"] == 0
    # group_by can also be set directly.
    updated = ok(ada, "update_data_prep", id=prep_id, split={"group_by": None})
    assert updated["preparation"]["settings"]["split"]["group_by"] is None
