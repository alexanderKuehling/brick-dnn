"""CSV import: detection of delimiter, decimal mark, header, encoding, missing values and
column kinds, and the error messages for broken files."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from brickdnn.data.io import (
    DEFAULT_MISSING,
    ImportOptions,
    ImportReport,
    parse_booleans,
    parse_dates,
    parse_numbers,
    read_csv,
)
from brickdnn.data.table import DataError


def write(tmp_path, text: str, name: str = "data.csv", encoding: str = "utf-8"):
    path = tmp_path / name
    path.write_bytes(text.encode(encoding))
    return path


def read(tmp_path, text: str, encoding: str = "utf-8", **options):
    return read_csv(write(tmp_path, text, encoding=encoding), ImportOptions(**options))


def column(table, name):
    return table.df[name].tolist()


# --------------------------------------------------------------------------- #
# Kinds
# --------------------------------------------------------------------------- #


def test_comma_file_with_all_kinds(tmp_path):
    text = (
        "id,age,price,active,signup,city,comment\n"
        "1,34,9.5,yes,2024-01-31,Berlin,A long free text comment about this customer\n"
        "2,NA,10.25,no,2024-02-01,Hamburg,Another fairly long comment about the customer\n"
        "3,51,,Yes,2024-02-02,Berlin,\n"
        "4,27,-3e2,NO,,N/A,Yet another very long comment written by someone\n"
    )
    table, report = read(tmp_path, text)
    assert report.delimiter == "," and report.decimal == "." and report.header is True
    assert report.encoding == "utf-8" and report.rows == 4 and report.columns == 7
    assert table.kinds == {
        "id": "numeric",
        "age": "numeric",
        "price": "numeric",
        "active": "boolean",
        "signup": "datetime",
        "city": "categorical",
        "comment": "text",
    }
    df = table.df
    assert df["age"].dtype == np.float64 and np.isnan(df["age"][1])
    assert column(table, "price")[:2] == [9.5, 10.25] and df["price"][3] == -300.0
    assert df["active"].tolist() == [1.0, 0.0, 1.0, 0.0]
    assert df["signup"].dtype == "datetime64[ns]" and pd.isna(df["signup"][3])
    assert df["signup"][0] == pd.Timestamp("2024-01-31")
    assert column(table, "city") == ["Berlin", "Hamburg", "Berlin", None]
    assert df["comment"][2] is None
    assert report.warnings == []


def test_semicolon_decimal_comma_thousands_and_german_dates(tmp_path):
    text = (
        "﻿Name;Preis;Menge;Datum;Aktiv\n"
        "Äpfel;1.234,5;3;31.01.2024;ja\n"
        "Birnen;2,25;-;01.02.2024 08:30;nein\n"
        ";;;;\n"
        "Kiwi;-0,5;1.000;15.03.2024;Ja\n"
    )
    table, report = read(tmp_path, text)
    assert (report.delimiter, report.decimal, report.header) == (";", ",", True)
    assert table.columns == ["Name", "Preis", "Menge", "Datum", "Aktiv"]  # BOM removed
    assert column(table, "Preis") == [1234.5, 2.25, -0.5]
    assert column(table, "Menge")[0] == 3 and np.isnan(column(table, "Menge")[1])
    assert column(table, "Menge")[2] == 1000  # "." groups thousands with a decimal comma
    assert column(table, "Datum") == [
        pd.Timestamp("2024-01-31"),
        pd.Timestamp("2024-02-01 08:30"),
        pd.Timestamp("2024-03-15"),
    ]
    assert column(table, "Aktiv") == [1.0, 0.0, 1.0]
    assert report.rows == 3
    assert "Skipped 1 empty row." in report.warnings


def test_tab_and_pipe_delimiters(tmp_path):
    table, report = read(tmp_path, "a\tb\n1\t2,5\n3\t4,5\n")
    assert report.delimiter == "\t" and report.decimal == ","
    assert column(table, "b") == [2.5, 4.5]
    table, report = read(tmp_path, "a|b|c\nx|1|2\ny|3|4\n")
    assert report.delimiter == "|" and table.kinds["b"] == "numeric"


def test_headerless_semicolon_file_with_decimal_commas(tmp_path):
    # "," splits these lines consistently too; ";" wins the tie.
    table, report = read(tmp_path, "1,5;2,5\n3,5;4,5\n")
    assert (report.delimiter, report.decimal, report.header) == (";", ",", False)
    assert table.columns == ["column_1", "column_2"]
    assert column(table, "column_1") == [1.5, 3.5]


def test_quoted_fields_and_grouped_numbers(tmp_path):
    text = 'id,text,amount\n1,"Hello, world","1,234.5"\n2,"two\nlines",2000\n3,"say ""hi""",-7\n'
    table, report = read(tmp_path, text)
    assert report.rows == 3
    assert column(table, "text") == ["Hello, world", "two\nlines", 'say "hi"']
    assert column(table, "amount") == [1234.5, 2000.0, -7.0]


def test_whitespace_is_stripped_and_crlf_lines(tmp_path):
    table, _ = read(tmp_path, "a ; b \r\n 1,5 ;  x \r\n2,5;y\r\n")
    assert table.columns == ["a", "b"]
    assert column(table, "a") == [1.5, 2.5] and column(table, "b") == ["x", "y"]


def test_almost_numeric_column_stays_categorical_with_examples(tmp_path):
    rows = "".join(f"{i},{i}\n" for i in range(30))
    table, report = read(tmp_path, "price,n\n" + rows + "n.a.,1\nfree,2\nn.a.,3\n")
    assert table.kinds == {"price": "categorical", "n": "numeric"}
    assert column(table, "price")[0] == "0"
    (warning,) = report.warnings
    assert warning.startswith("Column 'price' is categorical because 3 values are not numbers")
    assert "'n.a.', 'free'" in warning


def test_leading_zeros_stay_text(tmp_path):
    table, report = read(tmp_path, "zip,n\n01067,1\n10115,2\n04109,3\n")
    assert table.kinds["zip"] == "categorical"
    assert column(table, "zip") == ["01067", "10115", "04109"]
    assert "leading zeros" in report.warnings[0] and "'01067'" in report.warnings[0]
    table, _ = read(tmp_path, "x\n0.5\n0\n10\n")
    assert table.kinds["x"] == "numeric"


def test_booleans_and_zero_one_numbers(tmp_path):
    table, _ = read(tmp_path, "a,b,c,d\nTrue,1,y,t\nfalse,0,N,F\nTRUE,1,,t\n")
    assert table.kinds == {"a": "boolean", "b": "numeric", "c": "boolean", "d": "boolean"}
    assert column(table, "c")[:2] == [1.0, 0.0] and np.isnan(column(table, "c")[2])


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (["2024-01-31", "2024-01-31 12:00", "2024-02-01T08:30:00Z"], "2024-02-01 08:30"),
        (["2024/01/31", "2024/02/01 08:30"], "2024-02-01 08:30"),
        (["31-01-2024", "01-02-2024 08:30"], "2024-02-01 08:30"),
        (["01/31/2024", "02/01/2024 08:30"], "2024-02-01 08:30"),
        (["31/01/2024", "01/02/2024 08:30"], "2024-02-01 08:30"),
    ],
)
def test_date_formats(tmp_path, values, expected):
    table, _ = read(tmp_path, "when,n\n" + "".join(f"{v},1\n" for v in values))
    assert table.kinds["when"] == "datetime"
    assert table.df["when"].iloc[-1] == pd.Timestamp(expected)


def test_time_zones_become_utc(tmp_path):
    table, _ = read(tmp_path, "t\n2024-01-31T12:00:00+02:00\n2024-01-31T12:00:00Z\n")
    assert column(table, "t") == [
        pd.Timestamp("2024-01-31 10:00"),
        pd.Timestamp("2024-01-31 12:00"),
    ]


def test_ambiguous_and_invalid_dates(tmp_path):
    table, report = read(tmp_path, "d\n01/02/2024\n03/04/2024\n")
    assert table.df["d"][0] == pd.Timestamp("2024-01-02")  # month first
    assert "month/day/year" in report.warnings[0]
    dates = "".join(f"{day:02d}.01.2024\n" for day in range(1, 29))
    table, report = read(tmp_path, "d\n" + dates + "31.02.2024\n")
    assert table.kinds["d"] == "categorical"
    assert "1 value is not a valid date, e.g. '31.02.2024'" in report.warnings[0]


def test_text_and_empty_columns(tmp_path):
    long = "This is a rather long piece of free text with number {}"
    rows = "".join(f"{long.format(i)},,{i}\n" for i in range(5))
    table, report = read(tmp_path, "notes,empty,n\n" + rows)
    assert table.kinds == {"notes": "text", "empty": "categorical", "n": "numeric"}
    assert column(table, "empty") == [None] * 5
    assert "Column 'empty' is empty (no values in any row)." in report.warnings


def test_custom_missing_values(tmp_path):
    text = "a,b,c\nNA,-999,x\n1,2,y\n,unknown,z\n"
    table, _ = read(tmp_path, text)
    assert np.isnan(column(table, "a")[0]) and table.kinds["b"] == "categorical"
    table, _ = read(tmp_path, text, missing_values=["-999", "unknown"])
    # "NA" is a value now, the empty cell is still missing.
    assert column(table, "a") == ["NA", "1", None]
    assert table.kinds["b"] == "numeric" and column(table, "b")[1] == 2.0


# --------------------------------------------------------------------------- #
# Header, names and encodings
# --------------------------------------------------------------------------- #


def test_header_detection(tmp_path):
    _, report = read(tmp_path, "1.5,2,3\n4.5,5,6\n")
    assert report.header is False and "No header line found" in report.warnings[0]
    _, report = read(tmp_path, "name,city\nAlice,Berlin\nBob,Hamburg\n")
    assert report.header is True
    _, report = read(tmp_path, "Berlin,red\nHamburg,blue\nBerlin,red\n")
    assert report.header is False  # "Berlin" occurs again below
    _, report = read(tmp_path, ",name,value\n0,a,1.5\n1,b,2.5\n")
    assert report.header is True  # pandas index column without a name
    _, report = read(tmp_path, "1,2,3\n")
    assert report.header is False and report.rows == 1


def test_header_options_override_detection(tmp_path):
    table, report = read(tmp_path, "2019,2020\n5,6\n7,8\n", header="yes")
    assert report.header is True and table.columns == ["2019", "2020"]
    table, report = read(tmp_path, "a,b\nc,d\n", header="no")
    assert table.columns == ["column_1", "column_2"] and report.rows == 2
    assert report.warnings == []  # no "no header found" note when the user said so


def test_empty_and_repeated_column_names(tmp_path):
    table, report = read(tmp_path, "a,a,,b,a_2\n1,2,3,4,5\n")
    assert table.columns == ["a", "a_3", "column_3", "b", "a_2"]
    assert "Columns without a name were named column_3." in report.warnings
    assert "Repeated column names were renamed: a → a_3." in report.warnings


def test_trailing_delimiters_are_ignored(tmp_path):
    for text in ("a;b;\n1;2;\n3;4;\n", "a;b\n1;2;\n3;4;\n", "a;b;\n1;2\n3;4\n"):
        table, report = read(tmp_path, text)
        assert table.columns == ["a", "b"] and report.rows == 2
    assert "empty column at the end" in report.warnings[0]


def test_trailing_empty_values_in_a_row_are_fine(tmp_path):
    table, _ = read(tmp_path, "a;b;c\n1;2;\n3;4;5\n")
    assert np.isnan(column(table, "c")[0]) and column(table, "c")[1] == 5


def test_excel_sep_line(tmp_path):
    table, report = read(tmp_path, "sep=;\na;b\n1,5;2\n")
    assert report.delimiter == ";" and table.columns == ["a", "b"]


def test_windows_1252_and_latin_1(tmp_path):
    text = "name;city\nJosé;Köln\nAnna;München – Süd\n"
    table, report = read(tmp_path, text, encoding="cp1252")
    assert report.encoding == "cp1252"
    assert column(table, "city") == ["Köln", "München – Süd"]
    assert "read as Windows-1252" in report.warnings[0]
    # 0x81 is undefined in Windows-1252: Latin-1 reads everything.
    path = tmp_path / "latin.csv"
    path.write_bytes(b"a\n\x81x\n\xe4\n")
    table, report = read_csv(path)
    assert report.encoding == "latin-1" and column(table, "a") == ["\x81x", "ä"]


def test_explicit_encoding(tmp_path):
    path = write(tmp_path, "a\nKöln\n", encoding="cp1252")
    with pytest.raises(DataError, match="not valid UTF-8 text.*Windows-1252 or Latin-1"):
        read_csv(path, ImportOptions(encoding="utf-8"))
    table, report = read_csv(path, ImportOptions(encoding="latin-1"))
    assert report.encoding == "latin-1" and column(table, "a") == ["Köln"]


def test_utf16_with_bom(tmp_path):
    table, report = read(tmp_path, "a\tb\n1\t2\n", encoding="utf-16")
    assert report.encoding == "utf-16" and table.kinds == {"a": "numeric", "b": "numeric"}


def test_explicit_delimiter_and_decimal(tmp_path):
    table, report = read(tmp_path, "a;b\n1;2\n", delimiter=",", decimal=".")
    assert report.delimiter == "," and table.columns == ["a;b"]
    assert column(table, "a;b") == ["1;2"]
    table, report = read(tmp_path, '"x";"y"\n"1.5";"2"\n', delimiter=";", decimal=",")
    assert report.decimal == "," and table.kinds["x"] == "categorical"


# --------------------------------------------------------------------------- #
# Broken files and limits
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("a;b;c\n1;2;3\n4;5\n6;7;8\n", "Line 3 has 2 values, but the other lines have 3."),
        ("a,b\n1,2\n3,4,5\n", "Line 3 has 3 values, but the other lines have 2."),
        ("a,b\n1,2\n\n3,4,5\n", "Line 4 has 3 values"),
        ('a,b\n1,"x"\n2,"y, z",3\n', "Line 3 has 3 values"),
        ('a,b,c\n"1",2,\n"3",4\n"5",6,7\n', "Line 3 has 2 values, but the other lines have 3."),
        ('a,b,c\n1,"x\ny",\n3,4\n5,6,7\n', "Line 4 has 2 values"),
        ("Sales report\na;b;c\n1;2;3\n", "Line 1 has 1 value, but the other lines have 3."),
    ],
)
def test_ragged_rows_name_the_line(tmp_path, text, message):
    with pytest.raises(DataError, match=message) as error:
        read(tmp_path, text)
    assert "Check the delimiter" in str(error.value)


def test_title_line_hint(tmp_path):
    with pytest.raises(DataError, match="If the file starts with a title"):
        read(tmp_path, "Sales report\na;b;c\n1;2;3\n")


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"", "The file is empty."),
        (b"\n \n\n", "The file is empty."),
        (b"a,b\n", "header line but no data rows"),
        (b"a,b\n,\n", "no data rows"),
        (b'a,b\n1,"open\n2,3\n', "starts near line 2 is never closed"),
        (b"PK\x03\x04rest of a zip", "Excel workbook"),
        (b"a,b\n\x00\x01\x02", "not a text file"),
    ],
)
def test_broken_files(tmp_path, data, message):
    path = tmp_path / "broken.csv"
    path.write_bytes(data)
    with pytest.raises(DataError, match=message):
        read_csv(path)


def test_missing_file(tmp_path):
    with pytest.raises(DataError, match="cannot be opened"):
        read_csv(tmp_path / "nope.csv")


def test_limits(tmp_path):
    path = write(tmp_path, "a,b,c\n" + "1,2,3\n" * 10)
    with pytest.raises(DataError, match="The file has 3 columns; at most 2 are supported."):
        read_csv(path, max_columns=2)
    with pytest.raises(DataError, match="The file has 10 rows; at most 5 are supported."):
        read_csv(path, max_rows=5)
    table, report = read_csv(path, max_rows=10, max_columns=3)
    assert report.rows == 10
    # A row limit in the options reads the start of the file only.
    table, report = read_csv(path, ImportOptions(max_rows=4), max_rows=5)
    assert report.rows == 4 and table.n_rows == 4
    assert "Read only the first 4 data rows (row limit)." in report.warnings
    with pytest.raises(DataError, match="at most 5 are supported"):
        read_csv(path, ImportOptions(max_rows=8), max_rows=5)


# --------------------------------------------------------------------------- #
# Options and helpers
# --------------------------------------------------------------------------- #


def test_import_options_from_dict():
    assert ImportOptions.from_dict(None) == ImportOptions()
    options = ImportOptions.from_dict(
        {
            "delimiter": "\t",
            "decimal": ",",
            "header": "Yes",
            "encoding": "Windows-1252",
            "missing_values": '["-999", " ? ", 0]',
            "max_rows": "100",
            "name": "ignored",
            "file": object(),
        }
    )
    assert options == ImportOptions("\t", ",", "yes", "cp1252", ["-999", "?", "0"], 100)
    assert ImportOptions.from_dict({"delimiter": "tab", "header": False}).delimiter == "\t"
    assert ImportOptions.from_dict({"header": False}).header == "no"
    assert ImportOptions.from_dict({"delimiter": "", "missing_values": ""}) == ImportOptions()
    assert ImportOptions.from_dict({"missing_values": ["NA"]}).missing_values == ["NA"]
    assert ImportOptions.from_dict({"encoding": "UTF-8"}).encoding == "utf-8"
    assert ImportOptions.from_dict({"sheet": " Q1 "}).sheet == " Q1 "
    assert ImportOptions.from_dict({"sheet": "  "}).sheet is None
    round_trip = ImportOptions.from_dict(options.to_dict())
    assert round_trip == options


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"delimiter": ":"}, "Delimiter: choose"),
        ({"decimal": ";"}, "Decimal mark: choose"),
        ({"header": "maybe"}, "Header: choose"),
        ({"encoding": "utf-32"}, "Encoding: choose"),
        ({"delimiter": 5}, "Delimiter: invalid value"),
        ({"missing_values": "NA, ?"}, "Missing values: expected a list"),
        ({"missing_values": {"a": 1}}, "Missing values: expected a list"),
        ({"missing_values": [None]}, "every entry must be a text or a number"),
        ({"max_rows": 0}, "Row limit"),
        ({"max_rows": "ten"}, "Row limit"),
        ({"sheet": 2}, "Sheet: invalid value"),
        ({"sheet": "x" * 201}, "Sheet: the name can be at most 200"),
        ([1, 2], "must be an object"),
    ],
)
def test_import_options_errors(data, message):
    with pytest.raises(DataError, match=message):
        ImportOptions.from_dict(data)


def test_report_to_dict():
    report = ImportReport(",", ".", True, "utf-8", 3, 2, ["x"])
    assert report.to_dict() == {
        "delimiter": ",",
        "decimal": ".",
        "header": True,
        "encoding": "utf-8",
        "rows": 3,
        "columns": 2,
        "warnings": ["x"],
        "format": "csv",
        "compression": None,
        "member": None,
        "sheet": None,
        "sheets": [],
        "json_path": None,
    }
    assert "NA" in DEFAULT_MISSING and "" in DEFAULT_MISSING


@pytest.mark.parametrize(
    ("value", "decimal", "expected"),
    [
        ("1.5", ".", 1.5),
        ("-3e2", ".", -300.0),
        ("1,234.5", ".", 1234.5),
        ("1,2345", ".", None),
        ("1,5", ",", 1.5),
        ("-0,25", ",", -0.25),
        ("1.234,5", ",", 1234.5),
        ("1.234", ",", 1234.0),
        ("12.5", ",", None),
        ("1_000", ".", None),
        ("inf", ".", None),
        ("abc", ".", None),
        ("", ".", None),
    ],
)
def test_parse_numbers(value, decimal, expected):
    (result,) = parse_numbers([value], decimal)
    assert (np.isnan(result) and expected is None) or result == expected


def test_parse_booleans_and_dates():
    assert parse_booleans(pd.Series(["Yes", "no", "x", None])).tolist()[:2] == [1.0, 0.0]
    assert np.isnan(parse_booleans(["x"])[0])
    dates = parse_dates(["31.01.2024", "31.02.2024", None], "dmy_dot")
    assert dates[0] == np.datetime64("2024-01-31") and np.isnat(dates[1]) and np.isnat(dates[2])


def test_import_speed(tmp_path):
    """A 100,000 × 8 file with mixed kinds imports in a few seconds."""
    n = 100_000
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "id": np.arange(n),
            "x": rng.normal(size=n).round(4),
            "y": rng.integers(0, 100, n),
            "city": rng.choice(["Berlin", "Hamburg", "Köln"], n),
            "flag": rng.choice(["yes", "no"], n),
            "day": pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.integers(0, 900, n), "D"),
            "price": rng.normal(100, 30, n).round(2),
            "note": np.where(rng.random(n) < 0.3, "", "note"),
        }
    )
    path = tmp_path / "big.csv"
    frame.to_csv(path, index=False, sep=";", decimal=",", date_format="%d.%m.%Y")
    start = time.perf_counter()
    table, report = read_csv(path)
    elapsed = time.perf_counter() - start
    assert report.rows == n and report.decimal == ","
    assert table.kinds["day"] == "datetime" and table.kinds["flag"] == "boolean"
    assert elapsed < 10, f"import took {elapsed:.1f} s"
