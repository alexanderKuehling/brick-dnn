"""Importing a data file of any supported format into a typed table.

:func:`read_file` recognises the format from the content (and the file name where the
content does not tell), unpacks compressed files and reads:

==========  ==============================================  ==========================
format      files                                           needs
==========  ==============================================  ==========================
csv         .csv .tsv .tab .psv .txt (also other text)      (see brickdnn.data.io)
xlsx        Excel workbooks .xlsx .xlsm                     openpyxl
xls         Excel 97-2003 workbooks .xls                    xlrd
ods         OpenDocument spreadsheets .ods                  (standard library)
json        JSON: records, columns, ``{"data": [...]}``     (standard library)
jsonl       JSON Lines .jsonl .ndjson (one object per line) (standard library)
parquet     Apache Parquet .parquet                         pyarrow
feather     Feather / Arrow IPC files and streams           pyarrow
==========  ==============================================  ==========================

Packed files (.gz, .bz2, .xz, or a .zip file with one data file inside) are unpacked first.
Every format ends in the same typed table as a CSV file (see :mod:`brickdnn.data.io`), with
the same limits: the unpacked size (``max_bytes``; the unpacked content of a workbook or the
uncompressed data of a Parquet / Feather file may be ``EXPANDED_SIZE_FACTOR`` times as
large), rows, columns and cells (rows × columns, at most ``max_bytes / 2`` like the cells of a
CSV file of that size). Sizes stored in a file are checked before it is read, and the import
stops as soon as a limit is exceeded while reading.
"""

from __future__ import annotations

import bz2
import contextlib
import gzip
import json
import lzma
import tempfile
import zipfile
import zlib
from collections.abc import Iterable, Iterator
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

import pandas as pd

from . import COMPRESSED_SUFFIXES, FORMAT_SUFFIXES
from .formats import cell_text, find_json_table, json_columns, open_workbook
from .io import (
    ImportOptions,
    ImportReport,
    missing_tokens,
    read_csv,
    table_from_columns,
    table_from_rows,
)
from .table import DataError, Table

FORMAT_NAMES = {
    "csv": "CSV file",
    "xlsx": "Excel workbook",
    "xls": "Excel 97-2003 workbook",
    "ods": "OpenDocument spreadsheet",
    "json": "JSON file",
    "jsonl": "JSON Lines file",
    "parquet": "Parquet file",
    "feather": "Feather / Arrow file",
}
WORKBOOK_FORMATS = ("xlsx", "xls", "ods")
# Error values of spreadsheet cells; they count as missing values.
SPREADSHEET_ERRORS = ("#N/A", "#DIV/0!", "#VALUE!", "#REF!", "#NAME?", "#NUM!", "#NULL!")
# Workbooks (unpacked XML) and Parquet / Feather data (uncompressed) may be this many times
# the size limit: their content is larger than the same data in a CSV file.
EXPANDED_SIZE_FACTOR = 4
# Rows of a sheet at most (including empty ones), so a sheet cannot make the import read
# forever. Excel has 1,048,576 rows, LibreOffice up to 16,777,216.
SHEET_MAX_ROWS = 2**24
# Rows read from Parquet / Feather files at a time.
BATCH_ROWS = 65536

_MAGIC = (
    (b"\x1f\x8b", "gzip"),
    (b"BZh", "bz2"),
    (b"\xfd7zXZ\x00", "xz"),
    (b"PK\x03\x04", "zip"),
    (b"PK\x05\x06", "zip"),
    (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", "xls"),
    (b"PAR1", "parquet"),
    (b"ARROW1", "feather"),
    (b"\xff\xff\xff\xff", "feather"),
)
_LIBRARIES = {"xlsx": "openpyxl", "xls": "xlrd", "parquet": "pyarrow", "feather": "pyarrow"}


class WorkbookError(DataError):
    """A sheet of a workbook cannot be imported; ``sheets`` lists all sheets to choose from."""

    def __init__(self, message: str, sheets: Iterable[str]):
        super().__init__(message)
        self.sheets = list(sheets)


def read_file(
    path: str | Path,
    options: ImportOptions | None = None,
    *,
    filename: str | None = None,
    max_rows: int | None = None,
    max_columns: int | None = None,
    max_bytes: int | None = None,
) -> tuple[Table, ImportReport]:
    """Read and type a data file of any supported format.

    ``filename`` (default: the name of ``path``) helps where the content does not tell the
    format (JSON or CSV text). ``max_rows`` / ``max_columns`` / ``max_bytes`` are server
    limits; exceeding them raises :class:`DataError`, as do unreadable files (with a message
    for the user). Workbook problems raise :class:`WorkbookError` with the sheet names.
    """
    options = options or ImportOptions()
    path = Path(path)
    name = PurePosixPath((filename or path.name).replace("\\", "/")).name
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise DataError(f"The file cannot be opened: {exc.strerror or exc}.") from exc
    if max_bytes is not None and size > max_bytes:
        raise DataError(f"The file is larger than {_megabytes(max_bytes)} MB.")
    kind = _magic(path)
    if kind in ("gzip", "bz2", "xz") or (kind == "zip" and _zip_kind(path) == "zip"):
        with tempfile.TemporaryDirectory(prefix=".unpack-", dir=path.parent) as folder:
            inner, member = _unpack(path, kind, name, Path(folder), max_bytes)
            inner_name = member or _strip_suffix(name)
            if _magic(inner) in ("gzip", "bz2", "xz") or _zip_kind(inner) == "zip":
                raise DataError("The file is packed twice. Unpack it once and import the result.")
            table, report = _read(inner, inner_name, options, max_rows, max_columns, max_bytes)
        report.compression = kind
        report.member = member
        return table, report
    return _read(path, name, options, max_rows, max_columns, max_bytes)


def _read(
    path: Path,
    name: str,
    options: ImportOptions,
    max_rows: int | None,
    max_columns: int | None,
    max_bytes: int | None,
) -> tuple[Table, ImportReport]:
    fmt = detect_format(path, name)
    library = _LIBRARIES.get(fmt)
    if library is not None and not _installed(library):
        raise DataError(
            f"{FORMAT_NAMES[fmt]}s cannot be read on this server ({library} is not "
            "installed). Save the file as CSV and import that."
        )
    limits = _Limits(max_rows, max_columns, max_bytes)
    warnings: list[str] = []
    if fmt == "csv":
        return read_csv(path, options, max_rows=max_rows, max_columns=max_columns)
    try:
        if fmt in WORKBOOK_FORMATS:
            with path.open("rb") as file:
                return _read_workbook(file, fmt, options, limits, warnings)
        if fmt in ("json", "jsonl"):
            return _read_json(path, fmt, options, limits, warnings)
        return _read_arrow(path, fmt, options, limits, warnings)
    except RecursionError:  # e.g. lists in lists in lists ... in JSON
        raise DataError("The file is nested too deeply to be read.") from None


# --------------------------------------------------------------------------- #
# Formats and packed files
# --------------------------------------------------------------------------- #


def _magic(path: Path) -> str | None:
    """The format announced by the first bytes of the file (None: text or unknown)."""
    with path.open("rb") as file:
        head = file.read(8)
    for magic, kind in _MAGIC:
        if head.startswith(magic):
            return kind
    return None


def _zip_kind(path: Path) -> str | None:
    """xlsx, ods or zip (any other ZIP file) for a ZIP file, else None."""
    if _magic(path) != "zip":
        return None
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            if "xl/workbook.xml" in names:
                return "xlsx"
            if "xl/workbook.bin" in names:
                return "xlsb"
            if "mimetype" in names and archive.getinfo("mimetype").file_size < 200:
                mimetype = archive.read("mimetype")
                if mimetype.startswith(b"application/vnd.oasis.opendocument.spreadsheet"):
                    return "ods"
                if mimetype.startswith(b"application/vnd.oasis.opendocument"):
                    return "odf"
    except (zipfile.BadZipFile, OSError, KeyError, EOFError, zlib.error) as exc:
        raise DataError(f"The ZIP file is damaged: {exc}.") from None
    return "zip"


def detect_format(path: Path, name: str) -> str:
    """The format of a (not packed) file: by its content, by its name if the content does
    not tell (JSON / JSON Lines or CSV text)."""
    kind = _magic(path)
    if kind == "zip":
        kind = _zip_kind(path)
        if kind == "xlsb":
            raise DataError(
                "Excel binary workbooks (.xlsb) are not supported. In Excel, save the file as "
                "“Excel Workbook (.xlsx)” or CSV and import that."
            )
        if kind == "odf":
            raise DataError(
                "This OpenDocument file is not a spreadsheet. Import an .ods spreadsheet, or "
                "save the data as CSV."
            )
    if kind in ("xlsx", "ods", "xls", "parquet", "feather"):
        return kind
    with path.open("rb") as file:
        head = file.read(4096)
    if head.startswith(b"FEA1"):
        raise DataError(
            "This is an old Feather file (version 1), which is not supported. Save it again "
            "with a current pandas or pyarrow (version 2) and import that."
        )
    start = _text_start(head)
    if start.startswith(("<!doctype html", "<html", "<head", "<body")):
        raise DataError(
            "This is a web page (HTML), not a data file. If you downloaded it from a website, "
            "download the file itself (e.g. “Raw” on GitHub, “Download” on Hugging Face)."
        )
    if start.startswith("<?xml"):
        raise DataError("XML files are not supported. Save the data as CSV and import that.")
    suffix = PurePosixPath(name.lower()).suffix
    by_name = FORMAT_SUFFIXES.get(suffix)
    if by_name in ("json", "jsonl"):
        return by_name
    if by_name == "csv" and suffix != ".txt":
        return "csv"
    # Text without a telling name (.txt, a link): JSON if it starts like JSON.
    if start.startswith(("{", "[")):
        return "json"
    return "csv"


def _text_start(head: bytes) -> str:
    """The start of a text file without byte order mark and leading space, lower case."""
    for bom, codec in (
        (b"\xef\xbb\xbf", "utf-8"),
        (b"\xff\xfe", "utf-16-le"),
        (b"\xfe\xff", "utf-16-be"),
    ):
        if head.startswith(bom):
            return head[len(bom) :].decode(codec, "ignore").lstrip().lower()
    return head.decode("latin-1").lstrip().lower()


def _strip_suffix(name: str) -> str:
    path = PurePosixPath(name)
    return path.stem if path.suffix.lower() in COMPRESSED_SUFFIXES else name


def _unpack(
    path: Path, kind: str, name: str, folder: Path, max_bytes: int | None
) -> tuple[Path, str | None]:
    """Unpack ``path`` into ``folder``: ``(unpacked file, name in the ZIP file or None)``."""
    target = folder / "unpacked"
    if kind != "zip":
        opener = {"gzip": gzip.open, "bz2": bz2.open, "xz": lzma.open}[kind]
        try:
            with opener(path, "rb") as source:
                _copy_limited(source, target, max_bytes)
        except (OSError, EOFError, zlib.error, lzma.LZMAError, ValueError) as exc:
            raise DataError(f"The packed file is damaged and cannot be unpacked: {exc}.") from None
        return target, None
    try:
        with zipfile.ZipFile(path) as archive:
            info = _zip_member(archive)
            if info.flag_bits & 0x1:
                raise DataError("The ZIP file is encrypted. Import it without a password.")
            if max_bytes is not None and info.file_size > max_bytes:
                raise DataError(_too_large_unpacked(max_bytes))
            with archive.open(info) as source:
                _copy_limited(source, target, max_bytes)
    except (zipfile.BadZipFile, OSError, EOFError, zlib.error, NotImplementedError) as exc:
        raise DataError(f"The ZIP file is damaged and cannot be unpacked: {exc}.") from None
    return target, PurePosixPath(info.filename).name


def _zip_member(archive: zipfile.ZipFile) -> zipfile.ZipInfo:
    """The one data file in a ZIP file (system files like __MACOSX are ignored)."""
    files = [
        info
        for info in archive.infolist()
        if not info.is_dir()
        and not info.filename.startswith("__MACOSX/")
        and not PurePosixPath(info.filename).name.startswith((".", "~$"))
    ]
    data = [
        info
        for info in files
        if PurePosixPath(info.filename.lower()).suffix in FORMAT_SUFFIXES
        and PurePosixPath(info.filename.lower()).suffix not in COMPRESSED_SUFFIXES
    ]
    if len(data) == 1:
        return data[0]
    if not data:
        if len(files) == 1:
            return files[0]
        listed = ", ".join(PurePosixPath(i.filename).name for i in files[:5]) or "nothing"
        raise DataError(
            f"The ZIP file holds no CSV, Excel, JSON or Parquet file (it holds {listed})."
        )
    listed = ", ".join(PurePosixPath(i.filename).name for i in data[:5])
    more = " …" if len(data) > 5 else ""
    raise DataError(
        f"The ZIP file holds several data files ({listed}{more}). Import them one by one, "
        "or put only one file into the ZIP file."
    )


def _copy_limited(source, target: Path, max_bytes: int | None) -> None:
    written = 0
    with target.open("wb") as out:
        while chunk := source.read(1024 * 1024):
            written += len(chunk)
            if max_bytes is not None and written > max_bytes:
                raise DataError(_too_large_unpacked(max_bytes))
            out.write(chunk)


def _too_large_unpacked(max_bytes: int) -> str:
    return (
        f"The unpacked file is larger than {_megabytes(max_bytes)} MB, the limit for imports "
        "on this server."
    )


def _megabytes(size: float) -> str:
    return f"{size / 1024 / 1024:g}" if size < 10 * 1024**2 else f"{size / 1024 / 1024:,.0f}"


def _installed(library: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(library) is not None


class _Limits:
    """The server limits of an import."""

    def __init__(self, max_rows: int | None, max_columns: int | None, max_bytes: int | None):
        self.rows = max_rows
        self.columns = max_columns
        self.bytes = max_bytes
        self.cells = max_bytes // 2 if max_bytes is not None else None
        self.expanded = max_bytes * EXPANDED_SIZE_FACTOR if max_bytes is not None else None

    def rows_to_read(self, options: ImportOptions) -> int | None:
        """Data rows worth reading: one more than the limit, to notice that it is exceeded."""
        limits = [n for n in (options.max_rows, self.rows) if n is not None]
        return min(limits) + 1 if limits else None

    def check_columns(self, columns: int) -> None:
        if self.columns is not None and columns > self.columns:
            raise DataError(
                f"The file has {columns:,} columns; at most {self.columns:,} are supported."
            )

    def check_rows(self, rows: int, options: ImportOptions, *, at_least: bool = False) -> None:
        """Refuse a file with more rows than the server allows, unless the import is limited
        to fewer rows anyway. ``at_least``: ``rows`` is a count so far."""
        if self.rows is None or rows <= self.rows:
            return
        if options.max_rows is not None and options.max_rows <= self.rows:
            return
        if at_least:
            raise DataError(
                f"The file has more than {self.rows:,} rows; at most {self.rows:,} are supported."
            )
        raise DataError(f"The file has {rows:,} rows; at most {self.rows:,} are supported.")

    def check_cells(self, rows: int, columns: int) -> None:
        if self.cells is not None and rows * columns > self.cells:
            raise DataError(
                f"The table is too large: {rows:,} rows × {columns:,} columns are more than the "
                f"{self.cells:,} cells this server can import."
            )


def _report(fmt: str, table: Table, header: bool, warnings: list[str], **extra) -> ImportReport:
    return ImportReport(
        delimiter=None,
        decimal=".",
        header=header,
        encoding="utf-8" if fmt in ("json", "jsonl") else None,
        rows=table.n_rows,
        columns=len(table.columns),
        warnings=warnings,
        format=fmt,
        **extra,
    )


# --------------------------------------------------------------------------- #
# Workbooks
# --------------------------------------------------------------------------- #


def _read_workbook(
    file: BinaryIO, fmt: str, options: ImportOptions, limits: _Limits, warnings: list[str]
) -> tuple[Table, ImportReport]:
    """Read a sheet of a workbook (``file``: opened binary; openpyxl would insist on a file
    name ending in .xlsx)."""
    if fmt in ("xlsx", "ods") and limits.expanded is not None:
        with zipfile.ZipFile(file) as archive:
            unpacked = sum(info.file_size for info in archive.infolist())
        if unpacked > limits.expanded:
            raise DataError(
                f"The workbook is too large: unpacked it has {_megabytes(unpacked)} MB, at most "
                f"{_megabytes(limits.expanded)} MB are supported."
            )
        file.seek(0)
    try:
        names, rows_of, close = open_workbook(file, fmt)
    except DataError:
        raise
    except Exception as exc:  # a damaged file: the libraries raise all kinds of errors
        if isinstance(exc, MemoryError):
            raise
        raise DataError(f"The {FORMAT_NAMES[fmt]} cannot be read: {_reason(exc)}") from None
    try:
        if not names:
            raise DataError("The workbook has no sheets.")
        if options.sheet is not None and options.sheet not in names:
            raise WorkbookError(f"The workbook has no sheet “{options.sheet}”.", names)
        tokens = missing_tokens(options, SPREADSHEET_ERRORS)
        candidates = [options.sheet] if options.sheet is not None else names
        for sheet in candidates:
            with contextlib.closing(rows_of(sheet)) as rows:
                grid, more = _read_rows(rows, tokens, limits, options, f"sheet “{sheet}”")
            if grid:
                break
        if not grid:
            if options.sheet is not None:
                raise WorkbookError(f"The sheet “{options.sheet}” is empty.", names)
            raise WorkbookError("The workbook is empty: none of its sheets has any values.", names)
        skipped = names[: names.index(sheet)] if options.sheet is None else []
        if skipped:
            listed = ", ".join(f"“{s}”" for s in skipped[:5])
            verb = "is" if len(skipped) == 1 else "are"
            warnings.append(f"The sheet {listed} {verb} empty; imported the sheet “{sheet}”.")
        table, header = table_from_rows(
            grid,
            options,
            tokens,
            max_rows=limits.rows,
            max_columns=limits.columns,
            more_rows=more,
            warnings=warnings,
        )
    except WorkbookError:
        raise
    except DataError as exc:
        raise WorkbookError(str(exc), names) from None
    except Exception as exc:
        if isinstance(exc, MemoryError):
            raise
        raise WorkbookError(
            f"The {FORMAT_NAMES[fmt]} cannot be read: {_reason(exc)}", names
        ) from None
    finally:
        close()
    return table, _report(fmt, table, header, warnings, sheet=sheet, sheets=names)


def _reason(exc: Exception) -> str:
    text = str(exc).strip().rstrip(".")
    return (text[:200] or type(exc).__name__) + "."


def _read_rows(
    rows: Iterator[list[Any]],
    tokens: set[str],
    limits: _Limits,
    options: ImportOptions,
    what: str,
) -> tuple[list[list[str]], bool]:
    """Rows (of a sheet or JSON) as texts without empty rows: ``(rows, more)``. Stops after
    the header and one data row more than the limits (``more``: there may be more rows)."""
    wanted = limits.rows_to_read(options)
    grid: list[list[str]] = []
    cells = 0
    meaningful = 0
    for count, row in enumerate(rows, 1):
        if count > SHEET_MAX_ROWS:
            raise DataError(f"The {what} has more than {SHEET_MAX_ROWS:,} rows.")
        texts = [cell_text(value) for value in row]
        while texts and not texts[-1].strip():
            texts.pop()
        if not texts:
            continue
        grid.append(texts)
        cells += len(texts)
        if limits.cells is not None and cells > limits.cells:
            raise DataError(
                f"The {what} is too large: it has more than the {limits.cells:,} cells "
                "this server can import."
            )
        if any(text.strip() not in tokens for text in texts):
            meaningful += 1
            # The header line plus one row more than the limit.
            if wanted is not None and meaningful >= wanted + 1:
                return grid, True
    return grid, False


# --------------------------------------------------------------------------- #
# JSON
# --------------------------------------------------------------------------- #


def _read_json(
    path: Path, fmt: str, options: ImportOptions, limits: _Limits, warnings: list[str]
) -> tuple[Table, ImportReport]:
    text = _decode_json(path.read_bytes(), options)
    document: Any = None
    if fmt == "json":
        try:
            document = json.loads(text)
        except RecursionError:
            raise DataError("The JSON file is nested too deeply.") from None
        except json.JSONDecodeError as exc:
            if exc.msg != "Extra data":
                raise DataError(
                    f"The file is not valid JSON: {exc.msg} (line {exc.lineno:,}, column "
                    f"{exc.colno:,})."
                ) from None
            fmt = "jsonl"  # several values: one per line
    if fmt == "jsonl":
        document = _json_lines(text, limits, options)
    del text
    table, json_path = find_json_table(document)
    del document
    table, total = _json_head(table, limits, options)
    names, columns = json_columns(table, limits.columns, limits.cells)
    del table
    if names is None:  # rows (lists): the first may hold the names
        tokens = missing_tokens(options)
        grid, more = _read_rows(iter(columns), tokens, limits, options, "JSON file")
        if not grid:
            raise DataError("The file has no data rows.")
        result, header = table_from_rows(
            grid,
            options,
            tokens,
            max_rows=limits.rows,
            max_columns=limits.columns,
            more_rows=more,
            warnings=warnings,
        )
    else:
        result = table_from_columns(
            names,
            columns,
            options,
            max_rows=limits.rows,
            total_rows=total,
            warnings=warnings,
        )
        header = True
    return result, _report(fmt, result, header, warnings, json_path=json_path or None)


def _json_head(table: Any, limits: _Limits, options: ImportOptions) -> tuple[Any, int]:
    """The rows of a JSON table worth reading (see ``_Limits.rows_to_read``) and the number
    of rows in the file. Refuses a table with more rows than the server allows."""
    wanted = limits.rows_to_read(options)
    if isinstance(table, list):
        total = len(table)
        if table and isinstance(table[0], list):
            return table, total  # rows with a header row maybe: _read_rows stops in time
        limits.check_rows(total, options)
        return (table[:wanted] if wanted is not None else table), total
    if isinstance(table, dict) and isinstance(table.get("data"), list) and "columns" in table:
        total = len(table["data"])
        limits.check_rows(total, options)
        if wanted is not None:
            table = {**table, "data": table["data"][:wanted]}
        return table, total
    lengths = [len(v) for v in table.values() if isinstance(v, list)]
    total = max(lengths, default=0)
    limits.check_rows(total, options)
    if wanted is not None:
        table = {k: (v[:wanted] if isinstance(v, list) else v) for k, v in table.items()}
    return table, total


def _decode_json(raw: bytes, options: ImportOptions) -> str:
    if options.encoding not in ("auto", "utf-8"):
        try:
            return raw.decode(options.encoding)
        except UnicodeDecodeError:
            raise DataError(f"The file is not valid {options.encoding} text.") from None
    for bom, codec in ((b"\xff\xfe", "utf-16"), (b"\xfe\xff", "utf-16")):
        if raw.startswith(bom):
            return raw.decode(codec, "strict")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DataError(
            f"The JSON file is not UTF-8 text (byte {exc.start + 1:,} cannot be read)."
        ) from None


def _json_lines(text: str, limits: _Limits, options: ImportOptions) -> list[Any]:
    """The values of a JSON Lines file (blank lines are skipped)."""
    lines = text.split("\n")
    wanted = limits.rows_to_read(options)
    items: list[Any] = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        if wanted is not None and len(items) > wanted + 1:  # +1: maybe a header row
            rows = sum(1 for line in lines if line.strip())
            limits.check_rows(rows - (1 if isinstance(items[0], list) else 0), options)
            break
        try:
            items.append(json.loads(line))
        except RecursionError:
            raise DataError(f"Line {number:,} is nested too deeply.") from None
        except json.JSONDecodeError as exc:
            raise DataError(
                f"Line {number:,} is not valid JSON: {exc.msg} (column {exc.colno:,})."
            ) from None
    if not items:
        raise DataError("The file is empty.")
    if any(isinstance(item, dict) for item in items) and not all(
        isinstance(item, dict) for item in items
    ):
        raise DataError("Every line of a JSON Lines file must hold an object (one row).")
    return items


# --------------------------------------------------------------------------- #
# Parquet and Feather / Arrow
# --------------------------------------------------------------------------- #


def _read_arrow(
    path: Path, fmt: str, options: ImportOptions, limits: _Limits, warnings: list[str]
) -> tuple[Table, ImportReport]:
    import pyarrow as pa

    wanted = limits.rows_to_read(options)
    try:
        if fmt == "parquet":
            table, total = _parquet_table(path, wanted, options, limits)
        else:
            table, total = _feather_table(path, wanted, options, limits)
        while any(pa.types.is_struct(field.type) for field in table.schema):
            table = table.flatten()  # nested records: "a.b" columns, like JSON
        limits.check_columns(table.num_columns)
        limits.check_cells(table.num_rows, table.num_columns)
        names = [str(name) for name in table.column_names]
        columns = [_column_series(table.column(i)) for i in range(table.num_columns)]
    except (pa.ArrowException, OSError) as exc:
        raise DataError(f"The {FORMAT_NAMES[fmt]} cannot be read: {_reason(exc)}") from None
    del table
    result = table_from_columns(
        names, columns, options, max_rows=limits.rows, total_rows=total, warnings=warnings
    )
    return result, _report(fmt, result, True, warnings)


def _column_series(column) -> pd.Series:
    return column.to_pandas(date_as_object=False)


def _parquet_table(path: Path, wanted: int | None, options: ImportOptions, limits: _Limits):
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    meta = parquet.metadata
    limits.check_columns(meta.num_columns)
    limits.check_rows(meta.num_rows, options)
    limits.check_cells(min(meta.num_rows, wanted or meta.num_rows), meta.num_columns)
    size = sum(meta.row_group(i).total_byte_size for i in range(meta.num_row_groups))
    if limits.expanded is not None and size > limits.expanded:
        raise DataError(_too_large_expanded(size, limits))
    batches = []
    rows = nbytes = 0
    for batch in parquet.iter_batches(batch_size=BATCH_ROWS):
        batches.append(batch)
        rows += batch.num_rows
        nbytes += batch.nbytes
        if limits.expanded is not None and nbytes > limits.expanded:
            raise DataError(_too_large_expanded(nbytes, limits))
        if wanted is not None and rows >= wanted:
            break
    table = pa.Table.from_batches(batches, schema=parquet.schema_arrow)
    return (table.slice(0, wanted) if wanted is not None else table), meta.num_rows


def _feather_table(path: Path, wanted: int | None, options: ImportOptions, limits: _Limits):
    import pyarrow as pa

    with path.open("rb") as file:
        file_format = file.read(6) == b"ARROW1"
    source = pa.memory_map(str(path))
    if file_format:
        reader = pa.ipc.open_file(source)
        batches_in = (reader.get_batch(i) for i in range(reader.num_record_batches))
    else:
        reader = pa.ipc.open_stream(source)
        batches_in = iter(reader)
    limits.check_columns(len(reader.schema))
    batches = []
    rows = nbytes = 0
    total: int | None = None
    for batch in batches_in:
        batches.append(batch)
        rows += batch.num_rows
        nbytes += batch.nbytes
        if limits.expanded is not None and nbytes > limits.expanded:
            raise DataError(_too_large_expanded(nbytes, limits))
        limits.check_rows(rows, options, at_least=True)
        if wanted is not None and rows >= wanted:
            break
    else:
        total = rows
    table = pa.Table.from_batches(batches, schema=reader.schema)
    return (table.slice(0, wanted) if wanted is not None else table), total


def _too_large_expanded(size: int, limits: _Limits) -> str:
    assert limits.expanded is not None
    return (
        f"The data is too large: uncompressed it has more than {_megabytes(size)} MB, at most "
        f"{_megabytes(limits.expanded)} MB are supported."
    )
