"""Reading spreadsheets and JSON documents: the parts that the generated pandas script shares.

The functions and constants listed in ``SHARED`` are copied into the pandas script of a
preparation (see :mod:`brickdnn.data.export`), so that the script reads a file exactly like
the import did. They may therefore only use names the script defines as well: the modules
``datetime``, ``json``, ``math``, ``re`` and ``zipfile``, ``np`` (numpy), ``pd`` (pandas),
``iterparse`` (ElementTree's) and ``DataError``. Limits (rows, columns, sizes) are checked by
:mod:`brickdnn.data.readers`, which consumes the rows one by one.

Every cell becomes text with :func:`cell_text` before its column is typed, like a cell of a
CSV file: numbers in Python notation (``3``, ``0.25``, ``1e-07``), ``true`` / ``false``,
dates in ISO format, durations in seconds, lists and objects as JSON, ``""`` for missing
values. The import then detects the column kinds from these texts as for CSV files.
"""

from __future__ import annotations

import datetime
import json
import math
import re
import zipfile

import numpy as np
import pandas as pd

from .table import DataError

try:  # forbids DTDs and entity tricks in the XML of OpenDocument files
    from defusedxml.ElementTree import iterparse
except ImportError:  # pragma: no cover - defusedxml is a dependency of the server
    from xml.etree.ElementTree import iterparse

# Keys that usually hold the records in a JSON object ({"data": [...], "meta": {...}}).
RECORD_KEYS = ("data", "records", "rows", "items", "results", "entries", "values", "features")
# Nested JSON objects are flattened to "a.b.c" columns down to this depth; deeper ones stay
# JSON text.
MAX_NESTING = 10
# OpenDocument: most cells in a row (LibreOffice has at most 16,384 columns).
ODS_MAX_COLUMNS = 16384

_TABLE = "{urn:oasis:names:tc:opendocument:xmlns:table:1.0}"
_OFFICE = "{urn:oasis:names:tc:opendocument:xmlns:office:1.0}"
_TEXT = "{urn:oasis:names:tc:opendocument:xmlns:text:1.0}"


# --------------------------------------------------------------------------- #
# Cells as text
# --------------------------------------------------------------------------- #


def cell_text(value):
    """A cell as text, the way the CSV reader sees cells: "" for a missing value."""
    if value is None or value is pd.NaT or value is pd.NA:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            return ""
        if number.is_integer() and abs(number) < 2**53:
            return str(int(number))
        return repr(number)
    if isinstance(value, (datetime.timedelta, np.timedelta64)):
        seconds = pd.Timedelta(value).total_seconds()
        return "" if math.isnan(seconds) else cell_text(seconds)
    if isinstance(value, np.datetime64):
        return "" if np.isnat(value) else pd.Timestamp(value).isoformat()
    if isinstance(value, (datetime.date, datetime.time)):
        return value.isoformat()
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", "replace")
    if isinstance(value, (dict, list, tuple, np.ndarray)):
        return json.dumps(_json_value(value), ensure_ascii=False)
    return str(value)


def _json_value(value):
    """``value`` with numpy arrays and scalars turned into plain Python for ``json.dumps``."""
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (np.bool_, np.integer, np.floating)):
        return value.item()
    return cell_text(value)


# --------------------------------------------------------------------------- #
# JSON
# --------------------------------------------------------------------------- #


def _is_table_list(value):
    return isinstance(value, list) and len(value) > 0 and isinstance(value[0], (dict, list))


def _is_columns(document):
    """``{"a": [1, 2], "b": [3, 4]}``: lists of values (not of objects) of the same length."""
    lists = list(document.values())
    return (
        len(lists) > 0
        and all(isinstance(v, list) for v in lists)
        and len({len(v) for v in lists}) == 1
        and not any(_is_table_list(v) for v in lists)
    )


def _is_split(document):
    """pandas' "split" format: ``{"columns": [...], "data": [[...], ...]}``."""
    return isinstance(document.get("columns"), list) and isinstance(document.get("data"), list)


def find_json_table(document, path=None):
    """The part of a parsed JSON document that holds the table: ``(value, keys)``.

    ``path`` (the keys found by an earlier import) is followed directly. Otherwise a list,
    an object of columns (``{"a": [1, 2], ...}``) or pandas' split format is the table itself;
    in other objects, a list of objects or of rows up to three levels down (e.g.
    ``{"data": [...], "meta": {...}}``), preferring the usual names (data, records, rows ...)
    if there are several.
    """
    if path:
        for key in path:
            if not isinstance(document, dict) or key not in document:
                raise DataError("The JSON file has no “" + ".".join(path) + "” part.")
            document = document[key]
        return document, list(path)
    if isinstance(document, list):
        return document, []
    if not isinstance(document, dict):
        raise DataError("The JSON file holds a single value, not a table.")
    if _is_columns(document) or _is_split(document):
        return document, []
    level = [([], document)]
    for _ in range(3):
        found, deeper = [], []
        for keys, item in level:
            for key, value in item.items():
                if _is_table_list(value):
                    found.append(([*keys, str(key)], value))
                elif isinstance(value, dict):
                    deeper.append(([*keys, str(key)], value))
        usual = [f for f in found if f[0][-1].lower() in RECORD_KEYS]
        if len(found) > 1 and len(usual) == 1:
            found = usual
        if len(found) == 1:
            return found[0][1], found[0][0]
        if found:
            names = ", ".join("“" + ".".join(keys) + "”" for keys, _ in found[:5])
            raise DataError(
                f"The JSON file holds several lists ({names}); it is not clear which one is "
                "the table. Save the one you need as a file of its own."
            )
        level = deeper
    raise DataError(
        "No table was found in the JSON file. Supported are a list of objects "
        '([{"a": 1, "b": 2}, ...]), an object of columns ({"a": [1, 2], "b": [3, 4]}), '
        "such a list inside an object (e.g. under “data”) and JSON Lines (one object per line)."
    )


def flatten_record(record, prefix="", depth=0):
    """``[(name, value), ...]`` of a JSON object; nested objects become ``a.b`` names."""
    items = []
    for key, value in record.items():
        name = prefix + str(key)
        if isinstance(value, dict) and value and depth < MAX_NESTING:
            items.extend(flatten_record(value, name + ".", depth + 1))
        else:
            items.append((name, value))
    return items


def _cells_check(rows, columns, max_cells):
    if max_cells is not None and rows * columns > max_cells:
        raise DataError(
            f"The table is too large: {rows:,} rows × {columns:,} columns are more than the "
            f"{max_cells:,} cells this server can import."
        )


def json_columns(table, max_columns=None, max_cells=None):
    """``(names, columns)`` of a table found by :func:`find_json_table`: records (nested
    objects flattened), an object of columns, pandas' split format or a list of single
    values (one column "value"). For a list of rows (lists), ``names`` is None and
    ``columns`` holds the rows: their first row may hold the column names."""
    if isinstance(table, dict):
        if _is_split(table):
            names = [cell_text(name) for name in table["columns"]]
            rows = table["data"]
            for number, row in enumerate(rows, 1):
                if not isinstance(row, list) or len(row) != len(names):
                    raise DataError(
                        f"Row {number:,} under “data” does not have {len(names):,} values, "
                        "one per entry of “columns”."
                    )
            _cells_check(len(rows), len(names), max_cells)
            return names, [list(column) for column in zip(*rows, strict=True)] or [
                [] for _ in names
            ]
        if not _is_columns(table):
            raise DataError(
                "The JSON object is not a table: its entries must all be lists of the same "
                'length (one per column, e.g. {"a": [1, 2], "b": [3, 4]}).'
            )
        if max_columns is not None and len(table) > max_columns:
            raise DataError(
                f"The file has {len(table):,} columns; at most {max_columns:,} are supported."
            )
        _cells_check(len(next(iter(table.values()))), len(table), max_cells)
        return [str(key) for key in table], [list(column) for column in table.values()]
    if all(isinstance(item, dict) for item in table):
        index = {}
        columns = []
        for row, record in enumerate(table):
            for name, value in flatten_record(record):
                position = index.get(name)
                if position is None:
                    if max_columns is not None and len(index) >= max_columns:
                        raise DataError(
                            f"The records have more than {max_columns:,} different fields; "
                            f"at most {max_columns:,} columns are supported."
                        )
                    _cells_check(len(table), len(index) + 1, max_cells)
                    position = index[name] = len(columns)
                    columns.append([None] * len(table))
                columns[position][row] = value
        return list(index), columns
    if all(isinstance(item, list) for item in table):
        return None, table
    if not any(isinstance(item, (dict, list)) for item in table):
        return ["value"], [table]
    raise DataError(
        "The list in the JSON file mixes objects, lists and single values; every entry must "
        "be an object (one per row)."
    )


# --------------------------------------------------------------------------- #
# Workbooks (Excel .xlsx / .xls, OpenDocument .ods)
# --------------------------------------------------------------------------- #


def open_workbook(file, fmt):
    """Open a workbook (``fmt``: xlsx, xls or ods): ``(sheet names, rows, close)``.

    ``rows(name)`` yields the rows of a sheet as lists of values (None for an empty cell);
    ``close()`` closes the file. ``file`` is a path or a binary file object.
    """
    if fmt == "xlsx":
        import openpyxl

        book = openpyxl.load_workbook(file, read_only=True, data_only=True, keep_links=False)

        def xlsx_rows(name):
            sheet = book[name]
            sheet.reset_dimensions()  # the size stored in the file may be wrong
            for row in sheet.iter_rows(values_only=True):
                yield list(row)

        return [sheet.title for sheet in book.worksheets], xlsx_rows, book.close
    if fmt == "xls":
        import xlrd

        if hasattr(file, "read"):
            book = xlrd.open_workbook(file_contents=file.read(), on_demand=True)
        else:
            book = xlrd.open_workbook(file, on_demand=True)

        def xls_rows(name):
            sheet = book.sheet_by_name(name)
            for number in range(sheet.nrows):
                yield [_xls_value(cell, book.datemode) for cell in sheet.row(number)]

        return book.sheet_names(), xls_rows, book.release_resources
    archive = zipfile.ZipFile(file)
    return _ods_sheet_names(archive), lambda name: _ods_rows(archive, name), archive.close


def _xls_value(cell, datemode):
    import xlrd

    if cell.ctype == xlrd.XL_CELL_DATE:
        try:
            return xlrd.xldate_as_datetime(cell.value, datemode)
        except (ValueError, OverflowError, xlrd.xldate.XLDateError):
            return cell.value
    if cell.ctype == xlrd.XL_CELL_BOOLEAN:
        return bool(cell.value)
    if cell.ctype in (xlrd.XL_CELL_EMPTY, xlrd.XL_CELL_BLANK, xlrd.XL_CELL_ERROR):
        return None
    return cell.value


def _ods_elements(archive):
    """``(event, element)`` of content.xml; finished rows are removed to keep memory low."""
    with archive.open("content.xml") as xml:
        parents = []
        for event, element in iterparse(xml, events=("start", "end")):
            if event == "start":
                parents.append(element)
                continue
            parents.pop()
            yield event, element
            if element.tag == _TABLE + "table-row" and parents:
                parents[-1].remove(element)


def _ods_sheet_names(archive):
    names = []
    for _, element in _ods_elements(archive):
        if element.tag == _TABLE + "table":
            names.append(element.get(_TABLE + "name") or f"Sheet{len(names) + 1}")
            element.clear()
    return names


def _ods_rows(archive, name):
    # Tables end after their rows, so the rows of a table come before its name is known
    # here; count tables instead and look the name up once.
    position = _ods_sheet_names(archive).index(name)
    table = 0
    empty_rows = 0
    for _, element in _ods_elements(archive):
        if element.tag == _TABLE + "table":
            if table == position:
                return
            table += 1
            element.clear()
        elif element.tag == _TABLE + "table-row" and table == position:
            values = _ods_row(element)
            repeat = int(element.get(_TABLE + "number-rows-repeated") or 1)
            if values:
                # Empty rows are passed on only if values follow (trailing ones are padding).
                for _ in range(empty_rows):
                    yield []
                empty_rows = 0
                for _ in range(repeat):
                    yield values
            else:
                empty_rows += repeat


def _ods_row(row):
    values = []
    pending = 0  # empty cells, written only if a value follows
    for cell in row:
        if cell.tag not in (_TABLE + "table-cell", _TABLE + "covered-table-cell"):
            continue
        repeat = int(cell.get(_TABLE + "number-columns-repeated") or 1)
        value = _ods_value(cell) if cell.tag == _TABLE + "table-cell" else None
        if value is None or value == "":
            pending += repeat
            continue
        if len(values) + pending + repeat > ODS_MAX_COLUMNS:
            raise DataError(f"A row of the sheet has more than {ODS_MAX_COLUMNS:,} cells.")
        values.extend([None] * pending)
        pending = 0
        values.extend([value] * repeat)
    return values


def _ods_value(cell):
    kind = cell.get(_OFFICE + "value-type")
    try:
        if kind in ("float", "percentage", "currency"):
            return float(cell.get(_OFFICE + "value"))
        if kind == "boolean":
            return cell.get(_OFFICE + "boolean-value") == "true"
        if kind == "date":
            return pd.Timestamp(cell.get(_OFFICE + "date-value"))
        if kind == "time":
            return _ods_duration(cell.get(_OFFICE + "time-value"))
    except (TypeError, ValueError):
        pass
    if cell.get(_OFFICE + "string-value") is not None:
        return cell.get(_OFFICE + "string-value")
    paragraphs = [_ods_text(child) for child in cell if child.tag == _TEXT + "p"]
    return "\n".join(paragraphs) if paragraphs else None


def _ods_text(element):
    parts = [element.text or ""]
    for child in element:
        if child.tag == _TEXT + "s":
            parts.append(" " * int(child.get(_TEXT + "c") or 1))
        elif child.tag == _TEXT + "tab":
            parts.append("\t")
        elif child.tag == _TEXT + "line-break":
            parts.append("\n")
        elif child.tag != _OFFICE + "annotation":
            parts.append(_ods_text(child))
        parts.append(child.tail or "")
    return "".join(parts)


def _ods_duration(text):
    """``PT12H30M00S`` as a time of day (or a duration if it is not one)."""
    match = re.fullmatch(r"(-?)P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:([\d.]+)S)?)?", text)
    if not match:
        return text
    sign, days, hours, minutes, seconds = match.groups()
    duration = datetime.timedelta(
        days=int(days or 0),
        hours=int(hours or 0),
        minutes=int(minutes or 0),
        seconds=float(seconds or 0),
    )
    if sign:
        duration = -duration
    if datetime.timedelta(0) <= duration < datetime.timedelta(days=1):
        return (datetime.datetime.min + duration).time()
    return duration


# Copied into the pandas script, in this order (constants first).
SHARED = (
    "RECORD_KEYS",
    "MAX_NESTING",
    "ODS_MAX_COLUMNS",
    "_TABLE",
    "_OFFICE",
    "_TEXT",
    cell_text,
    _json_value,
    _is_table_list,
    _is_columns,
    _is_split,
    find_json_table,
    flatten_record,
    _cells_check,
    json_columns,
    open_workbook,
    _xls_value,
    _ods_elements,
    _ods_sheet_names,
    _ods_rows,
    _ods_row,
    _ods_value,
    _ods_text,
    _ods_duration,
)
