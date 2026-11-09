"""Imports of other formats than CSV: Excel, OpenDocument, JSON, JSON Lines, Parquet,
Feather / Arrow and packed files (.gz, .bz2, .xz, .zip), with their limits. They all end in
the same typed table as a CSV file."""

from __future__ import annotations

import bz2
import gzip
import io
import json
import lzma
import zipfile

import numpy as np
import pandas as pd
import pytest

from brickdnn.data import file_types
from brickdnn.data.io import ImportOptions, read_csv
from brickdnn.data.readers import WorkbookError, read_file
from brickdnn.data.table import DataError

openpyxl = pytest.importorskip("openpyxl")
pa = pytest.importorskip("pyarrow")

MB = 1024 * 1024


def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "price": [1.5, None, 3.25, -2.0],
            "member": [True, False, None, True],
            "since": pd.to_datetime(
                ["2024-01-31 00:00", "2024-02-01 08:30", None, "2023-12-24 00:00"]
            ),
            "city": ["Berlin", "NA", "Rome", " Paris "],
        }
    )


EXPECTED_KINDS = {
    "id": "numeric",
    "price": "numeric",
    "member": "boolean",
    "since": "datetime",
    "city": "categorical",
}


def check_frame_table(table) -> None:
    assert table.kinds == EXPECTED_KINDS
    df = table.df
    assert df["id"].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert df["price"].tolist()[0] == 1.5 and np.isnan(df["price"][1])
    assert df["member"].tolist()[:2] == [1.0, 0.0] and np.isnan(df["member"][2])
    assert df["since"][1] == pd.Timestamp("2024-02-01 08:30") and pd.isna(df["since"][2])
    # Texts are stripped and missing-value markers apply, as in CSV files.
    assert df["city"].tolist() == ["Berlin", None, "Rome", "Paris"]


def read(path, max_bytes=10 * MB, **options):
    return read_file(path, ImportOptions.from_dict(options), max_bytes=max_bytes)


def write_xlsx(path, sheets: dict[str, pd.DataFrame], header=True) -> None:
    with pd.ExcelWriter(path) as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False, header=header)


def test_file_types():
    types = file_types()
    for suffix in (".csv", ".tsv", ".txt", ".xlsx", ".ods", ".json", ".jsonl", ".parquet"):
        assert suffix in types
    assert {".gz", ".bz2", ".xz", ".zip"} <= set(types)


# --------------------------------------------------------------------------- #
# Excel
# --------------------------------------------------------------------------- #


def test_excel_workbook_with_several_sheets(tmp_path):
    path = tmp_path / "book.xlsx"
    small = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    write_xlsx(path, {"Notes": pd.DataFrame(), "Data": frame(), "Small": small})

    # Without a choice: the first sheet with values, and a note about the empty one.
    table, report = read(path)
    check_frame_table(table)
    assert report.format == "xlsx" and report.header is True
    assert report.sheet == "Data" and report.sheets == ["Notes", "Data", "Small"]
    assert report.delimiter is None and report.encoding is None and report.decimal == "."
    assert report.warnings == ["The sheet “Notes” is empty; imported the sheet “Data”."]

    table, report = read(path, sheet="Small")
    assert report.sheet == "Small" and table.kinds == {"a": "numeric", "b": "categorical"}
    assert report.warnings == []

    with pytest.raises(WorkbookError, match="The workbook has no sheet “Q3”.") as error:
        read(path, sheet="Q3")
    assert error.value.sheets == ["Notes", "Data", "Small"]
    with pytest.raises(WorkbookError, match="The sheet “Notes” is empty.") as error:
        read(path, sheet="Notes")
    assert error.value.sheets == ["Notes", "Data", "Small"]


def test_excel_layout_header_errors_and_types(tmp_path):
    """Rows and columns around the table, a title without header detection trouble, error
    values, numbers stored as text, leading zeros and a row limit."""
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.title = "Export"
    sheet["B3"], sheet["C3"], sheet["D3"], sheet["E3"] = "zip", "amount", "ok", "note"
    rows = [
        ("01067", 12.5, True, "fine"),
        ("80331", "#DIV/0!", False, None),
        ("10115", 3, None, "x"),
        ("20095", -1, True, "y"),
    ]
    for number, values in enumerate(rows, start=4):
        for column, value in zip("BCDE", values, strict=True):
            sheet[f"{column}{number}"] = value
    sheet["Z20"] = None  # a formatted but empty cell far away
    book.save(tmp_path / "export.xlsx")

    table, report = read(tmp_path / "export.xlsx")
    assert table.columns == ["zip", "amount", "ok", "note"]
    assert table.kinds["zip"] == "categorical"  # leading zeros: codes, not numbers
    assert table.kinds["amount"] == "numeric" and np.isnan(table.df["amount"][1])
    assert table.kinds["ok"] == "boolean"
    assert report.header is True and report.rows == 4

    table, report = read(tmp_path / "export.xlsx", max_rows=2)
    assert table.n_rows == 2 and "Read only the first 2 data rows (row limit)." in report.warnings
    table, report = read(tmp_path / "export.xlsx", header="no")
    assert table.columns[0] == "column_1" and table.n_rows == 5


def test_excel_without_header_row(tmp_path):
    write_xlsx(tmp_path / "plain.xlsx", {"S": pd.DataFrame([[1, 2.5], [3, 4.5]])}, header=False)
    table, report = read(tmp_path / "plain.xlsx")
    assert report.header is False and table.columns == ["column_1", "column_2"]
    assert table.df["column_2"].tolist() == [2.5, 4.5]


def test_excel_limits(tmp_path):
    path = tmp_path / "big.xlsx"
    write_xlsx(path, {"S": pd.DataFrame({f"c{i}": range(20) for i in range(5)})})
    with pytest.raises(WorkbookError, match="at most 10 are supported") as error:
        read_file(path, max_rows=10)
    assert error.value.sheets == ["S"]
    with pytest.raises(WorkbookError, match="The file has 5 columns; at most 4"):
        read_file(path, max_columns=4)
    table, _ = read_file(path, max_rows=20, max_columns=5, max_bytes=10 * MB)
    assert table.n_rows == 20


def test_excel_zip_bomb_is_refused_before_reading(tmp_path):
    path = tmp_path / "bomb.xlsx"
    write_xlsx(path, {"S": pd.DataFrame({"a": [f"{i:05d}" + "x" * 1000 for i in range(2000)]})})
    assert path.stat().st_size < 100_000  # very repetitive: packs well
    with pytest.raises(DataError, match="The workbook is too large: unpacked it has"):
        read_file(path, max_bytes=200_000)  # unpacked 2 MB > 4 × 200 KB


def test_damaged_workbooks(tmp_path):
    with zipfile.ZipFile(tmp_path / "broken.xlsx", "w") as archive:
        archive.writestr("xl/workbook.xml", "<not really")
    with pytest.raises(DataError, match="The Excel workbook cannot be read"):
        read(tmp_path / "broken.xlsx")
    (tmp_path / "broken.xls").write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\0" * 600)
    with pytest.raises(DataError, match="The Excel 97-2003 workbook cannot be read"):
        read(tmp_path / "broken.xls")
    with zipfile.ZipFile(tmp_path / "data.xlsb", "w") as archive:
        archive.writestr("xl/workbook.bin", b"\0")
    with pytest.raises(DataError, match="xlsb"):
        read(tmp_path / "data.xlsb")


def test_text_with_a_spreadsheet_name_is_read_as_text(tmp_path):
    """Web applications often export tab-separated text as .xls."""
    (tmp_path / "export.xls").write_text("a\tb\n1\t2\n3\t4\n")
    table, report = read(tmp_path / "export.xls")
    assert report.format == "csv" and report.delimiter == "\t"
    assert table.df["b"].tolist() == [2.0, 4.0]


# --------------------------------------------------------------------------- #
# OpenDocument
# --------------------------------------------------------------------------- #

ODS_HEAD = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
    'xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" '
    'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">'
    "<office:body><office:spreadsheet>"
)
ODS_TAIL = "</office:spreadsheet></office:body></office:document-content>"


def ods_cell(value=None, repeat=1) -> str:
    attrs = f' table:number-columns-repeated="{repeat}"' if repeat > 1 else ""
    if value is None:
        return f"<table:table-cell{attrs}/>"
    if isinstance(value, bool):
        flag = "true" if value else "false"
        return (
            f'<table:table-cell office:value-type="boolean" office:boolean-value="{flag}"{attrs}>'
            f"<text:p>{'WAHR' if value else 'FALSCH'}</text:p></table:table-cell>"
        )
    if isinstance(value, (int, float)):
        return (
            f'<table:table-cell office:value-type="float" office:value="{value}"{attrs}>'
            f"<text:p>{value}</text:p></table:table-cell>"
        )
    if isinstance(value, tuple):  # ("date", "2024-01-31") or ("time", "PT08H30M00S")
        kind, text = value
        return (
            f'<table:table-cell office:value-type="{kind}" office:{kind}-value="{text}"{attrs}>'
            f"<text:p>{text}</text:p></table:table-cell>"
        )
    return (
        f'<table:table-cell office:value-type="string"{attrs}><text:p>{value}</text:p>'
        "</table:table-cell>"
    )


def ods_row(cells: list[str], repeat=1) -> str:
    attrs = f' table:number-rows-repeated="{repeat}"' if repeat > 1 else ""
    return f"<table:table-row{attrs}>{''.join(cells)}</table:table-row>"


def write_ods(path, tables: dict[str, list[str]]) -> None:
    body = "".join(
        f'<table:table table:name="{name}">{"".join(rows)}</table:table>'
        for name, rows in tables.items()
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("mimetype", "application/vnd.oasis.opendocument.spreadsheet")
        archive.writestr("content.xml", ODS_HEAD + body + ODS_TAIL)


def test_opendocument_spreadsheet(tmp_path):
    rows = [
        ods_row([ods_cell("name"), ods_cell("score"), ods_cell("ok"), ods_cell("day")]),
        ods_row([ods_cell("Ann"), ods_cell(1.5), ods_cell(True), ods_cell(("date", "2024-01-31"))]),
        ods_row([ods_cell("Bob"), ods_cell(2), ods_cell(False), ods_cell()], repeat=2),
        # LibreOffice pads sheets with huge repeated empty rows and cells: ignored.
        ods_row([ods_cell(repeat=16000)], repeat=1_048_000),
    ]
    other = [ods_row([ods_cell("x")]), ods_row([ods_cell(1)])]
    write_ods(tmp_path / "sheet.ods", {"Tabelle1": rows, "Other": other})

    table, report = read(tmp_path / "sheet.ods")
    assert report.format == "ods" and report.sheets == ["Tabelle1", "Other"]
    assert report.sheet == "Tabelle1" and report.rows == 3
    assert table.kinds == {
        "name": "categorical",
        "score": "numeric",
        "ok": "boolean",  # from the stored value, whatever the (localised) display text
        "day": "datetime",
    }
    assert table.df["score"].tolist() == [1.5, 2.0, 2.0]
    assert table.df["ok"].tolist() == [1.0, 0.0, 0.0]

    table, report = read(tmp_path / "sheet.ods", sheet="Other")
    assert table.columns == ["x"] and table.df["x"].tolist() == [1.0]


def test_opendocument_repeated_cells_cannot_blow_up(tmp_path):
    rows = [ods_row([ods_cell("a")]), ods_row([ods_cell(1, repeat=1_000_000)])]
    write_ods(tmp_path / "wide.ods", {"S": rows})
    with pytest.raises(DataError, match="more than 16,384 cells"):
        read(tmp_path / "wide.ods")
    rows = [ods_row([ods_cell("a")]), ods_row([ods_cell(1)], repeat=10**9)]
    write_ods(tmp_path / "long.ods", {"S": rows})
    with pytest.raises(DataError, match="at most 1,000 are supported"):
        read_file(tmp_path / "long.ods", max_rows=1000)


# --------------------------------------------------------------------------- #
# JSON
# --------------------------------------------------------------------------- #

RECORDS = [
    {"id": 1, "user": {"name": "Ann", "age": 30}, "tags": ["a", "b"], "price": "1.5"},
    {"id": 2, "user": {"name": "Bob", "age": None}, "tags": [], "price": "n/a"},
    {"id": 3, "user": {"name": "Cem"}, "joined": "2024-01-31", "price": "2"},
]


def write_json(tmp_path, document, name="data.json"):
    path = tmp_path / name
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("document", "json_path"),
    [
        (RECORDS, None),
        ({"data": RECORDS, "meta": {"count": 3}}, ["data"]),
        ({"status": "ok", "result": {"items": RECORDS}}, ["result", "items"]),
        ({"results": RECORDS, "errors": [{"code": 1}]}, ["results"]),
    ],
    ids=["records", "wrapper", "nested-wrapper", "usual-name-wins"],
)
def test_json_records_are_flattened(tmp_path, document, json_path):
    table, report = read(write_json(tmp_path, document))
    assert report.format == "json" and report.json_path == json_path
    assert report.encoding == "utf-8" and report.header is True
    assert table.columns == ["id", "user.name", "user.age", "tags", "price", "joined"]
    assert table.kinds["user.age"] == "numeric" and table.kinds["joined"] == "datetime"
    assert table.df["tags"].tolist() == ['["a", "b"]', "[]", None]
    # Numbers in texts are numbers, like in CSV files ("n/a" is a missing-value marker).
    assert table.kinds["price"] == "numeric" and table.df["price"].tolist()[0] == 1.5


def test_json_columns_split_rows_and_values(tmp_path):
    table, _ = read(write_json(tmp_path, {"a": [1, 2, 3], "b": ["x", None, "z"]}))
    assert table.columns == ["a", "b"] and table.df["b"].tolist() == ["x", None, "z"]

    split = {"columns": ["a", "b"], "index": [0, 1], "data": [[1, "x"], [2, "y"]]}
    table, _ = read(write_json(tmp_path, split))
    assert table.columns == ["a", "b"] and table.df["a"].tolist() == [1.0, 2.0]

    table, report = read(write_json(tmp_path, [["a", "b"], [1, True], [2, False]]))
    assert report.header is True and table.kinds == {"a": "numeric", "b": "boolean"}

    table, _ = read(write_json(tmp_path, [1.5, 2, None]))
    assert table.columns == ["value"] and table.df["value"].tolist()[:2] == [1.5, 2.0]


def test_json_lines(tmp_path):
    text = "\n".join(json.dumps(r) for r in RECORDS) + "\n\n"
    for name in ("data.jsonl", "data.ndjson", "data.json"):
        path = tmp_path / name
        path.write_text(text)
        table, report = read(path)
        assert report.format == "jsonl" and table.n_rows == 3
        assert table.columns[:3] == ["id", "user.name", "user.age"]
    (tmp_path / "bad.jsonl").write_text('{"a": 1}\n{"a": 2,\n')
    with pytest.raises(DataError, match="Line 2 is not valid JSON"):
        read(tmp_path / "bad.jsonl")


def test_json_in_a_txt_file_and_a_bom(tmp_path):
    path = tmp_path / "export.txt"
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps(RECORDS).encode())
    table, report = read(path)
    assert report.format == "json" and table.n_rows == 3


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ("{", "not valid JSON"),
        ("42", "a single value, not a table"),
        ('{"name": "x", "size": 3}', "No table was found"),
        ('{"a": [{"x": 1}], "b": [{"y": 2}]}', r"several lists \(“a”, “b”\)"),
        ('[{"a": 1}, 2]', "mixes objects"),
        ('{"a": [1, 2], "b": [1]}', "No table was found"),
        ('{"columns": ["a", "b"], "data": [[1]]}', "Row 1 under “data”"),
        ("[" * 5000 + "]" * 5000, "nested too deeply"),
    ],
)
def test_json_errors(tmp_path, document, message):
    path = tmp_path / "bad.json"
    path.write_text(document)
    with pytest.raises(DataError, match=message):
        read(path)


def test_json_limits(tmp_path):
    records = [{"a": i, "b": i * 2} for i in range(30)]
    path = write_json(tmp_path, records)
    with pytest.raises(DataError, match="The file has 30 rows; at most 10 are supported."):
        read_file(path, max_rows=10)
    table, report = read_file(path, ImportOptions(max_rows=5), max_rows=10)
    assert table.n_rows == 5 and "Read only the first 5 data rows (row limit)." in report.warnings
    with pytest.raises(DataError, match="more than 1 different fields"):
        read_file(path, max_columns=1)
    # Each record with other keys: the table would be huge, refused before it is built.
    sparse = write_json(tmp_path, [{f"k{i}": 1} for i in range(2000)], "sparse.json")
    with pytest.raises(DataError, match="cells this server can import"):
        read_file(sparse, max_bytes=1 * MB)


def test_custom_missing_values_apply_to_numbers(tmp_path):
    path = write_json(tmp_path, [{"a": -999, "b": "x"}, {"a": 5, "b": "-999"}])
    table, _ = read(path, missing_values=["-999"])
    assert np.isnan(table.df["a"][0]) and table.df["b"].tolist() == ["x", None]


# --------------------------------------------------------------------------- #
# Parquet and Feather / Arrow
# --------------------------------------------------------------------------- #


def test_parquet_and_feather(tmp_path):
    df = frame()
    df.to_parquet(tmp_path / "data.parquet")
    df.to_feather(tmp_path / "data.feather")
    for name, fmt in (("data.parquet", "parquet"), ("data.feather", "feather")):
        table, report = read(tmp_path / name)
        check_frame_table(table)
        assert report.format == fmt and report.encoding is None and report.warnings == []


def test_arrow_stream_nested_columns_and_time_zones(tmp_path):
    table = pa.table(
        {
            "when": pa.array(
                [pd.Timestamp("2024-01-01 12:00", tz="Europe/Berlin"), None],
                type=pa.timestamp("us", tz="Europe/Berlin"),
            ),
            "user": pa.array([{"name": "Ann", "age": 3}, {"name": "Bob", "age": None}]),
            "tags": pa.array([["a"], []]),
            "category": pa.array(["x", "y"]).dictionary_encode(),
            "seconds": pa.array([1500, None], type=pa.duration("ms")),
            "nothing": pa.array([None, None], type=pa.null()),
        }
    )
    sink = io.BytesIO()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    (tmp_path / "data.arrows").write_bytes(sink.getvalue())
    result, report = read(tmp_path / "data.arrows")
    assert report.format == "feather"
    assert result.columns == [
        "when",
        "user.name",
        "user.age",
        "tags",
        "category",
        "seconds",
        "nothing",
    ]
    assert result.df["when"][0] == pd.Timestamp("2024-01-01 11:00")  # UTC
    assert result.kinds["category"] == "categorical"
    assert result.df["tags"].tolist() == ['["a"]', "[]"]
    assert result.df["seconds"].tolist()[0] == 1.5
    assert "Column 'nothing' is empty (no values in any row)." in report.warnings


def test_typed_columns_give_the_same_table_as_csv(tmp_path):
    """Parquet keeps the types it has, CSV reads texts: the tables are the same."""
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "x": rng.normal(size=200).round(6),
            "n": rng.integers(-5, 5, size=200),
            "flag": rng.integers(0, 2, size=200).astype(bool),
            "day": pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 900, 200), "D"),
            "word": rng.choice(["red", "green", "blue"], size=200),
        }
    )
    df.loc[3, "x"] = np.nan
    df.to_parquet(tmp_path / "data.parquet")
    df.to_csv(tmp_path / "data.csv", index=False)
    from_parquet, _ = read(tmp_path / "data.parquet")
    from_csv, _ = read_csv(tmp_path / "data.csv")
    assert from_parquet.kinds == from_csv.kinds
    pd.testing.assert_frame_equal(from_parquet.df, from_csv.df)


def test_parquet_limits_are_checked_before_reading(tmp_path, monkeypatch):
    path = tmp_path / "data.parquet"
    pd.DataFrame({f"c{i}": range(100) for i in range(6)}).to_parquet(path, row_group_size=10)
    with pytest.raises(DataError, match="The file has 100 rows; at most 50 are supported."):
        read_file(path, max_rows=50)
    with pytest.raises(DataError, match="The file has 6 columns; at most 5 are supported."):
        read_file(path, max_columns=5)
    table, report = read_file(path, ImportOptions(max_rows=15), max_rows=50)
    assert table.n_rows == 15 and report.rows == 15


def test_columnar_files_cannot_expand_beyond_the_limits(tmp_path):
    """Parquet and Feather pack repetitive data very well: the cells and the uncompressed
    size are checked, not only the size of the file."""
    zeros = pd.DataFrame({f"c{i}": np.zeros(200_000) for i in range(3)})
    zeros.to_parquet(tmp_path / "zeros.parquet")
    zeros.to_feather(tmp_path / "zeros.feather", compression="zstd")
    for name in ("zeros.parquet", "zeros.feather"):
        assert (tmp_path / name).stat().st_size < 100_000
        with pytest.raises(DataError, match="cells this server can import|uncompressed"):
            read_file(tmp_path / name, max_bytes=200_000)


def test_feather_reads_only_what_it_needs(tmp_path):
    path = tmp_path / "data.feather"
    pd.DataFrame({"a": range(1000)}).to_feather(path, chunksize=100)
    with pytest.raises(DataError, match="more than 500 rows; at most 500"):
        read_file(path, max_rows=500)
    table, _ = read_file(path, ImportOptions(max_rows=150), max_rows=500)
    assert table.n_rows == 150


def test_formats_without_their_library(tmp_path, monkeypatch):
    from brickdnn.data import readers

    frame().to_parquet(tmp_path / "data.parquet")
    monkeypatch.setattr(readers, "_installed", lambda library: False)
    with pytest.raises(DataError, match=r"Parquet files cannot be read on this server \(pyarrow"):
        read(tmp_path / "data.parquet")


def test_unsupported_and_damaged_binary_files(tmp_path):
    (tmp_path / "old.feather").write_bytes(b"FEA1" + b"\0" * 100)
    with pytest.raises(DataError, match="old Feather file"):
        read(tmp_path / "old.feather")
    (tmp_path / "broken.parquet").write_bytes(b"PAR1" + b"\0" * 100)
    with pytest.raises(DataError, match="The Parquet file cannot be read"):
        read(tmp_path / "broken.parquet")


# --------------------------------------------------------------------------- #
# Packed files and limits for every format
# --------------------------------------------------------------------------- #

CSV = b"a;b\n1,5;x\n2,5;y\n"


@pytest.mark.parametrize(
    ("name", "pack", "compression"),
    [
        ("data.csv.gz", gzip.compress, "gzip"),
        ("data.csv.bz2", bz2.compress, "bz2"),
        ("data.csv.xz", lzma.compress, "xz"),
        ("data.gz", gzip.compress, "gzip"),
    ],
)
def test_packed_csv(tmp_path, name, pack, compression):
    (tmp_path / name).write_bytes(pack(CSV))
    table, report = read(tmp_path / name)
    assert report.compression == compression and report.format == "csv"
    assert report.decimal == "," and table.df["a"].tolist() == [1.5, 2.5]


def test_zip_with_one_data_file(tmp_path):
    with zipfile.ZipFile(tmp_path / "data.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("__MACOSX/._orders.json", b"junk")
        archive.writestr("export/.DS_Store", b"junk")
        archive.writestr("export/README.md", b"Orders of 2024")
        archive.writestr("export/orders.json", json.dumps(RECORDS))
    table, report = read(tmp_path / "data.zip")
    assert report.compression == "zip" and report.member == "orders.json"
    assert report.format == "json" and table.n_rows == 3

    buffer = io.BytesIO()
    frame().to_excel(buffer, index=False)
    with zipfile.ZipFile(tmp_path / "book.zip", "w") as archive:
        archive.writestr("book.xlsx", buffer.getvalue())
    table, report = read(tmp_path / "book.zip")
    assert report.format == "xlsx" and report.member == "book.xlsx"
    check_frame_table(table)


@pytest.mark.parametrize(
    ("files", "message"),
    [
        ({"a.csv": CSV, "b.csv": CSV}, r"several data files \(a.csv, b.csv\)"),
        ({"a.pdf": b"%PDF", "b.png": b"PNG"}, "holds no CSV, Excel, JSON or Parquet file"),
        ({"a.csv.gz": gzip.compress(CSV)}, "packed twice"),
    ],
)
def test_zip_problems(tmp_path, files, message):
    with zipfile.ZipFile(tmp_path / "data.zip", "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    with pytest.raises(DataError, match=message):
        read(tmp_path / "data.zip")


def test_zip_and_gzip_bombs_are_refused(tmp_path):
    big = b"1,2,3\n" * (2 * MB)  # 12 MB, packs to a few KB
    with zipfile.ZipFile(tmp_path / "bomb.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("bomb.csv", big)
    (tmp_path / "bomb.csv.gz").write_bytes(gzip.compress(big))
    assert (tmp_path / "bomb.zip").stat().st_size < 100_000
    for name in ("bomb.zip", "bomb.csv.gz"):
        with pytest.raises(DataError, match="The unpacked file is larger than 1 MB"):
            read_file(tmp_path / name, max_bytes=MB)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["bomb.csv.gz", "bomb.zip"]


def test_zip_with_a_wrong_size_in_its_directory_is_refused(tmp_path):
    """A ZIP file can claim a small size for a large member: unpacking stops anyway."""
    with zipfile.ZipFile(tmp_path / "liar.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("data.csv", b"1,2\n" * 300_000)
    data = bytearray((tmp_path / "liar.zip").read_bytes())
    real = (300_000 * 4).to_bytes(4, "little")
    fake = (1000).to_bytes(4, "little")
    assert data.count(real) == 2  # local header and central directory
    (tmp_path / "liar.zip").write_bytes(bytes(data).replace(real, fake))
    with pytest.raises(DataError):
        read_file(tmp_path / "liar.zip", max_bytes=100_000)


def test_web_pages_and_xml_are_refused(tmp_path):
    (tmp_path / "page.csv").write_text("<!DOCTYPE html><html><body>Login</body></html>")
    with pytest.raises(DataError, match="web page"):
        read(tmp_path / "page.csv")
    (tmp_path / "data.xml").write_text('<?xml version="1.0"?><rows/>')
    with pytest.raises(DataError, match="XML files are not supported"):
        read(tmp_path / "data.xml")


@pytest.mark.parametrize("name", ["data.xlsx", "data.json", "data.parquet", "data.csv.gz"])
def test_row_and_column_limits_for_every_format(tmp_path, name):
    df = pd.DataFrame({f"c{i}": range(30) for i in range(4)})
    path = tmp_path / name
    if name.endswith(".xlsx"):
        df.to_excel(path, index=False)
    elif name.endswith(".json"):
        df.to_json(path, orient="records")
    elif name.endswith(".parquet"):
        df.to_parquet(path)
    else:
        path.write_bytes(gzip.compress(df.to_csv(index=False).encode()))
    with pytest.raises(DataError, match="at most 20 are supported"):
        read_file(path, max_rows=20)
    with pytest.raises(DataError, match="at most 3 "):
        read_file(path, max_columns=3)
    with pytest.raises(DataError, match="larger than|too large|cells"):
        read_file(path, max_bytes=100)
    table, _ = read_file(path, max_rows=30, max_columns=4, max_bytes=MB)
    assert table.n_rows == 30 and len(table.columns) == 4
