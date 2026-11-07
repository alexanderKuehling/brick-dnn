"""Reading CSV files into a :class:`~brickdnn.data.table.Table`.

The reader is forgiving, because real files are messy: it detects the delimiter
(``,`` ``;`` tab ``|``), decimal commas (``1,5``, also with thousands separators like
``1.234,5``), a header line, the text encoding (UTF-8 with or without BOM, UTF-16 with BOM,
otherwise Windows-1252 / Latin-1) and common placeholders for missing values (empty, ``NA``,
``N/A``, ``NaN``, ``null``, ``None``, ``-``, ``?`` ...). Cells are stripped of surrounding
whitespace, lines without any value are skipped. Every column gets a kind (see
:mod:`brickdnn.data.table`):

* ``numeric`` if every non-missing value is a number (with the detected decimal mark),
* ``boolean`` for true/false, yes/no, y/n, t/f values (also ja/nein, wahr/falsch;
  case-insensitive; 0/1 stays numeric),
* ``datetime`` if every non-missing value is a date or date-time in a common format
  (ISO ``2024-01-31``, ``2024-01-31 12:00[:00]``, ``31.01.2024``, ``2024/01/31``,
  ``01/31/2024``, ``31-01-2024``),
* ``text`` for long or mostly unique strings (see ``table.infer_kind``), otherwise
  ``categorical``.

Columns that are *almost* numeric (at least 90 % numbers) stay categorical, and the report
warns about them with a few of the offending values, so nothing is silently turned into
missing values. Whole numbers with leading zeros (ZIP codes, article numbers) stay
categorical too.
"""

from __future__ import annotations

import codecs
import csv
import io
import json
import re
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .table import TEXT_MEAN_LENGTH, TEXT_MIN_UNIQUE, TEXT_UNIQUE_SHARE, DataError, Table

DELIMITERS = (",", ";", "\t", "|")
DEFAULT_MISSING = ("", "NA", "N/A", "n/a", "na", "NaN", "nan", "null", "NULL", "None", "-", "?")
ENCODINGS = ("auto", "utf-8", "cp1252", "latin-1")

# Boolean words (compared in lower case). "0" / "1" are numbers.
TRUE_VALUES = ("true", "yes", "y", "t", "ja", "wahr")
FALSE_VALUES = ("false", "no", "n", "f", "nein", "falsch")

# Date formats by family, tried in this order (``ISO8601`` is pandas' ISO parser).
DATE_FORMATS: dict[str, tuple[str, ...]] = {
    "iso": ("ISO8601", "%Y-%m-%d", "%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"),
    "ymd": ("%Y/%m/%d", "%Y/%m/%d %H:%M", "%Y/%m/%d %H:%M:%S"),
    "dmy_dot": ("%d.%m.%Y", "%d.%m.%Y %H:%M", "%d.%m.%Y %H:%M:%S"),
    "mdy": ("%m/%d/%Y", "%m/%d/%Y %H:%M", "%m/%d/%Y %H:%M:%S"),
    "dmy": ("%d/%m/%Y", "%d/%m/%Y %H:%M", "%d/%m/%Y %H:%M:%S"),
    "dmy_dash": ("%d-%m-%Y", "%d-%m-%Y %H:%M", "%d-%m-%Y %H:%M:%S"),
}
_TIME = r"(?:[ T]\d{1,2}:\d{2}(?::\d{2}(?:[.,]\d+)?)?)?"
_DATE_PATTERNS = {
    "iso": re.compile(r"\d{4}-\d{1,2}-\d{1,2}" + _TIME + r"(?:Z|[+-]\d{2}(?::?\d{2})?)?"),
    "ymd": re.compile(r"\d{4}/\d{1,2}/\d{1,2}" + _TIME),
    "dmy_dot": re.compile(r"\d{1,2}\.\d{1,2}\.\d{4}" + _TIME),
    "mdy": re.compile(r"\d{1,2}/\d{1,2}/\d{4}" + _TIME),
    "dmy": re.compile(r"\d{1,2}/\d{1,2}/\d{4}" + _TIME),
    "dmy_dash": re.compile(r"\d{1,2}-\d{1,2}-\d{4}" + _TIME),
}

# A column with at least this share of numbers (or dates), but not all, stays categorical
# and gets a warning.
ALMOST_SHARE = 0.9
# Numbers with a leading zero ("01067") in at least this share of the distinct values are codes.
LEADING_ZERO_SHARE = 0.05
# Delimiter detection looks at the start of the file.
SNIFF_CHARS = 64 * 1024
SNIFF_RECORDS = 200
# Type checks look at this many values (spread over the column) before converting it all.
TYPE_SAMPLE = 500
MAX_MISSING_TOKENS = 100

_BOOL_TOKENS = frozenset(TRUE_VALUES + FALSE_VALUES)
# Decimal comma, optionally with "." thousands groups: 1,5  -0,25  1.234,5  1.234  ,5
_COMMA_NUMBER = re.compile(
    r"[+-]?(?:\d{1,3}(?:\.\d{3})+(?:,\d*)?|\d+(?:,\d*)?|,\d+)(?:[eE][+-]?\d+)?"
)
# Decimal point with "," thousands groups: 1,234.5  -12,000
_GROUPED_DOT_NUMBER = re.compile(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d*)?(?:[eE][+-]?\d+)?")
_COMMA_DECIMAL_CELL = re.compile(r"[+-]?(?:\d{1,3}(?:\.\d{3})+|\d+)?,\d+")
_DOT_DECIMAL_CELL = re.compile(r"[+-]?\d*\.\d+")
_DOT_GROUPS_ONLY = re.compile(r"[+-]?\d{1,3}(?:\.\d{3})+")
_LEADING_ZERO = re.compile(r"^[+-]?0\d", re.MULTILINE)
# Joins values for checks that run on the whole text at once (never part of a CSV value
# in practice; checked anyway).
_SEPARATOR = "\x00"
_SEP_LINE = re.compile(r"\s*sep=(.)\s*", re.IGNORECASE)

_DELIMITER_ALIASES = {
    "tab": "\t",
    "\\t": "\t",
    "comma": ",",
    "semicolon": ";",
    "pipe": "|",
}
_DECIMAL_ALIASES = {"comma": ",", "dot": ".", "point": ".", "period": "."}
_HEADER_ALIASES = {"true": "yes", "1": "yes", "false": "no", "0": "no"}
_ENCODING_ALIASES = {
    "utf8": "utf-8",
    "utf-8-sig": "utf-8",
    "windows-1252": "cp1252",
    "cp-1252": "cp1252",
    "latin1": "latin-1",
    "iso-8859-1": "latin-1",
    "iso8859-1": "latin-1",
}
_ENCODING_NAMES = {
    "utf-8": "UTF-8",
    "utf-16": "UTF-16",
    "cp1252": "Windows-1252",
    "latin-1": "Latin-1",
}
_DELIMITER_NAMES = {",": "“,”", ";": "“;”", "\t": "tab", "|": "“|”"}


@dataclass
class ImportOptions:
    """How to read a CSV file. ``"auto"`` / ``None`` means: detect it."""

    delimiter: str = "auto"  # auto | , | ; | \t | |
    decimal: str = "auto"  # auto | . | ,
    header: str = "auto"  # auto | yes | no
    encoding: str = "auto"  # see ENCODINGS
    # Cell values that mean "missing" (in addition to empty cells); None: DEFAULT_MISSING.
    missing_values: list[str] | None = None
    # Stop after this many data rows (None: read everything; limits are checked anyway).
    max_rows: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ImportOptions:
        """Build options from request data; raises :class:`DataError` for invalid values.

        Unknown keys (``name``, ``file`` ...) are ignored, empty values mean "auto".
        ``missing_values`` may be a list or a JSON string holding a list.
        """
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise DataError("The import settings must be an object.")
        options = cls()

        delimiter = _choice(data, "delimiter", "Delimiter", _DELIMITER_ALIASES)
        if delimiter is not None:
            if delimiter not in ("auto", *DELIMITERS):
                raise DataError("Delimiter: choose auto, comma, semicolon, tab or “|”.")
            options.delimiter = delimiter

        decimal = _choice(data, "decimal", "Decimal mark", _DECIMAL_ALIASES)
        if decimal is not None:
            if decimal not in ("auto", ".", ","):
                raise DataError("Decimal mark: choose auto, “.” or “,”.")
            options.decimal = decimal

        header = _choice(data, "header", "Header", _HEADER_ALIASES)
        if header is not None:
            if header not in ("auto", "yes", "no"):
                raise DataError("Header: choose auto, yes or no.")
            options.header = header

        encoding = _choice(data, "encoding", "Encoding", _ENCODING_ALIASES)
        if encoding is not None:
            if encoding not in ENCODINGS:
                raise DataError("Encoding: choose auto, UTF-8, Windows-1252 (cp1252) or Latin-1.")
            options.encoding = encoding

        options.missing_values = _missing_tokens(data.get("missing_values"))

        max_rows = data.get("max_rows")
        if max_rows not in (None, ""):
            if isinstance(max_rows, str) and max_rows.strip().isdigit():
                max_rows = int(max_rows.strip())
            if not isinstance(max_rows, int) or isinstance(max_rows, bool) or max_rows < 1:
                raise DataError("Row limit: enter a whole number of at least 1.")
            options.max_rows = max_rows
        return options

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ImportReport:
    """What the reader detected and anything the user should know. Stored with the source."""

    delimiter: str
    decimal: str
    header: bool
    encoding: str
    rows: int
    columns: int
    # Human-readable notes, e.g. "Column 'price' is categorical because 12 values are not
    # numbers, e.g. 'n.a.', 'free'. Use “Change type” to convert it."
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _choice(data: dict[str, Any], key: str, label: str, aliases: dict[str, str]) -> str | None:
    """A select value from request data: None if empty, aliases resolved, lower case."""
    value = data.get(key)
    if isinstance(value, bool) and key == "header":
        return "yes" if value else "no"
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise DataError(f"{label}: invalid value {value!r}.")
    text = value.strip().lower() or value  # a tab stays a tab
    return aliases.get(text, text)


def _missing_tokens(value: Any) -> list[str] | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            raise DataError(
                'Missing values: expected a list of texts, e.g. ["NA", "-", "?"].'
            ) from None
    if not isinstance(value, list):
        raise DataError('Missing values: expected a list of texts, e.g. ["NA", "-", "?"].')
    if len(value) > MAX_MISSING_TOKENS:
        raise DataError(f"Missing values: at most {MAX_MISSING_TOKENS} values are supported.")
    tokens: list[str] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (str, int, float)):
            raise DataError("Missing values: every entry must be a text or a number.")
        token = item.strip() if isinstance(item, str) else _number_token(item)
        if len(token) > 100:
            raise DataError("Missing values: entries can be at most 100 characters long.")
        if token not in tokens:
            tokens.append(token)
    return tokens


def _number_token(value: int | float) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


# --------------------------------------------------------------------------- #
# Numbers, booleans and dates (also used by the profile and the cast step)
# --------------------------------------------------------------------------- #


def _as_strings(values: pd.Series | np.ndarray | Sequence[Any]) -> list[str]:
    """Plain list of strings; anything else (None, NaN, numbers) becomes ""."""
    items = values.tolist() if hasattr(values, "tolist") else list(values)
    return [v if isinstance(v, str) else "" for v in items]


def parse_numbers(values: pd.Series | np.ndarray | Sequence[Any], decimal: str = ".") -> np.ndarray:
    """``values`` (stripped strings) as float64 numbers, NaN where a value is not a number.

    ``decimal="."``: ``1.5``, ``1e-3``, ``1,234.5``; ``decimal=","``: ``1,5``, ``1.234,5``,
    ``1.234`` (dots only as thousands separators in groups of three). Infinite values and
    non-strings count as "not a number".
    """
    return _parse_numbers(_as_strings(values), decimal)


def _parse_numbers(strings: list[str], decimal: str) -> np.ndarray:
    joined = _SEPARATOR.join(strings)
    # The joined text makes the common checks and replacements run in C, not per value.
    whole = joined.count(_SEPARATOR) == max(len(strings) - 1, 0)
    if decimal == ",":
        if whole and "." not in joined:
            strings = joined.replace(",", ".").split(_SEPARATOR) if strings else []
        else:
            strings = [
                v.replace(",", ".")
                if "." not in v
                else v.replace(".", "").replace(",", ".")
                if _COMMA_NUMBER.fullmatch(v)
                else ""
                for v in strings
            ]
            joined = _SEPARATOR.join(strings)
    result = _to_float(strings, "_" in joined)
    if decimal != "," and "," in joined:
        for i in np.flatnonzero(np.isnan(result)):
            v = strings[i]
            if "," in v and _GROUPED_DOT_NUMBER.fullmatch(v):
                result[i] = float(v.replace(",", ""))
    result[~np.isfinite(result)] = np.nan
    return result


def _to_float(strings: list[str], underscores: bool) -> np.ndarray:
    # float() accepts "1_000"; pandas does not, and neither do we.
    if not underscores:
        try:
            return np.array(strings, dtype=np.float64)
        except ValueError:
            pass
    return np.array([_float(v) for v in strings], dtype=np.float64)


def _float(value: str) -> float:
    if "_" in value:
        return np.nan
    try:
        return float(value)
    except ValueError:
        return np.nan


def parse_booleans(values: pd.Series | np.ndarray | Sequence[Any]) -> np.ndarray:
    """``values`` (stripped strings) as 1.0 / 0.0, NaN where a value is not a boolean word."""
    lookup = {**dict.fromkeys(TRUE_VALUES, 1.0), **dict.fromkeys(FALSE_VALUES, 0.0)}
    return np.array(
        [lookup.get(v.lower(), np.nan) for v in _as_strings(values)],
        dtype=np.float64,
    )


def parse_dates(values: pd.Series | np.ndarray | Sequence[Any], family: str) -> np.ndarray:
    """``values`` (stripped strings) parsed with the formats of ``DATE_FORMATS[family]``,
    as ``datetime64[ns]`` (time zones converted to UTC), NaT where a value does not fit."""
    strings = _as_strings(values)
    s = pd.Series(strings, dtype=object)
    result = np.full(len(s), np.datetime64("NaT"), dtype="datetime64[ns]")
    todo = np.array([v != "" for v in strings], dtype=bool)
    for fmt in DATE_FORMATS[family]:
        positions = np.flatnonzero(todo)
        if len(positions) == 0:
            break
        part = s.iloc[positions]
        if fmt == "ISO8601":
            parsed = pd.to_datetime(part, format="ISO8601", errors="coerce", utc=True)
            parsed = parsed.dt.tz_localize(None)
        else:
            parsed = pd.to_datetime(part, format=fmt, errors="coerce")
        ok = parsed.notna().to_numpy()
        result[positions[ok]] = parsed.to_numpy(dtype="datetime64[ns]")[ok]
        todo[positions[ok]] = False
    return result


def _date_families(sample: list[str]) -> list[str]:
    """Date families whose pattern every value of ``sample`` matches."""
    return [
        family
        for family, pattern in _DATE_PATTERNS.items()
        if all(pattern.fullmatch(v) for v in sample)
    ]


# --------------------------------------------------------------------------- #
# Reading the file
# --------------------------------------------------------------------------- #


@dataclass
class _Cells:
    """One column of the file: its distinct values (stripped) and a code per row."""

    values: np.ndarray  # object array of str
    missing: np.ndarray  # per value: a missing-value marker
    codes: np.ndarray  # per row: index into ``values``

    @classmethod
    def from_series(cls, raw: pd.Series, tokens: set[str]) -> _Cells:
        codes, uniques = pd.factorize(raw.to_numpy(dtype=object))
        stripped = [v.strip() for v in uniques.tolist()]
        if len(codes) and codes.min() < 0:  # cannot happen with na_filter=False; to be safe
            codes = np.where(codes < 0, len(stripped), codes)
            stripped.append("")
        values = np.empty(len(stripped), dtype=object)
        values[:] = stripped
        missing = pd.Series(values, dtype=object).isin(tokens).to_numpy(dtype=bool)
        return cls(values, missing, codes.astype(np.int32))

    def row_missing(self) -> np.ndarray:
        return self.missing[self.codes]

    def present(self) -> tuple[np.ndarray, list[str], np.ndarray]:
        """Indices, values and row counts of the non-missing values in use."""
        counts = np.bincount(self.codes, minlength=len(self.values))
        index = np.flatnonzero((counts > 0) & ~self.missing)
        return index, self.values[index].tolist(), counts[index]

    def sample_below_first(self, n: int) -> list[str]:
        """At most ``n`` distinct non-missing values of the rows after the first one."""
        candidates = ~self.missing
        first = self.codes[0]
        if candidates[first] and not (self.codes[1:] == first).any():
            candidates = candidates.copy()
            candidates[first] = False
        return _spread(self.values[candidates].tolist(), n)


def read_csv(
    path: str | Path,
    options: ImportOptions | None = None,
    *,
    max_rows: int | None = None,
    max_columns: int | None = None,
) -> tuple[Table, ImportReport]:
    """Read and type a CSV file.

    ``max_rows`` / ``max_columns`` are server limits: exceeding them raises
    :class:`DataError` ("The file has 2,400,000 rows; at most 2,000,000 are supported.").
    Also raises :class:`DataError` for empty files, files without data rows, unreadable
    encodings, duplicate or empty column names (empty names become ``column_3``, duplicates
    get ``_2``, ``_3`` suffixes and a warning instead) and rows with a wrong number of cells
    (the message names the line).
    """
    options = options or ImportOptions()
    warnings: list[str] = []
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise DataError(f"The file cannot be opened: {exc.strerror or exc}.") from exc

    _check_text_file(raw)
    text, encoding = _decode(raw, options.encoding, warnings)
    del raw
    text, sep_hint = _strip_sep_line(text)
    if not text.strip():
        raise DataError("The file is empty.")

    if options.delimiter != "auto":
        delimiter = options.delimiter
    else:
        delimiter = sep_hint if sep_hint in DELIMITERS else _detect_delimiter(text)

    width = _first_record_width(text, delimiter)
    if max_columns is not None and width > max_columns:
        raise DataError(f"The file has {width:,} columns; at most {max_columns:,} are supported.")

    # Read the header line and one line more than allowed, to notice too many rows.
    limits = [n for n in (options.max_rows, max_rows) if n is not None]
    nrows = min(limits) + 2 if limits else None
    frame = _parse(text, delimiter, nrows)
    tokens = set(DEFAULT_MISSING if options.missing_values is None else options.missing_values)
    tokens.add("")
    columns = [_Cells.from_series(frame.iloc[:, i], tokens) for i in range(frame.shape[1])]
    del frame

    # Every line ending with a delimiter gives an empty last column.
    dropped = 0
    while len(columns) > 1 and all(v == "" for v in columns[-1].values):
        columns.pop()
        dropped += 1
    if dropped:
        warnings.append(
            f"Ignored {_count(dropped, 'empty column')} at the end of the lines "
            "(the lines end with a delimiter)."
        )
    if max_columns is not None and len(columns) > max_columns:
        raise DataError(
            f"The file has {len(columns):,} columns; at most {max_columns:,} are supported."
        )

    decimal = options.decimal
    if decimal == "auto":
        decimal = _detect_decimal(columns, delimiter)
    if options.header == "auto":
        header = _detect_header(columns, decimal)
    else:
        header = options.header == "yes"
    if header:
        if len(columns[0].codes) <= 1:
            raise DataError("The file has a header line but no data rows.")
        names = [c.values[c.codes[0]] for c in columns]
        for c in columns:
            c.codes = c.codes[1:]
    else:
        names = [f"column_{i + 1}" for i in range(len(columns))]
        if options.header == "auto":
            warnings.append(
                f"No header line found; the columns are named column_1 … "
                f"column_{len(columns)}. Import again with “Header: yes” if the first line "
                "holds the column names."
            )
    names = _clean_names(names, warnings)

    # Lines without any value (e.g. ";;;" at the end of Excel exports) carry no data.
    empty_rows = np.logical_and.reduce([c.row_missing() for c in columns])
    if empty_rows.any():
        for c in columns:
            c.codes = c.codes[~empty_rows]
        warnings.append(f"Skipped {_count(int(empty_rows.sum()), 'empty row')}.")
    n_rows = len(columns[0].codes)
    if n_rows == 0:
        raise DataError("The file has no data rows.")

    if (
        options.max_rows is not None
        and n_rows > options.max_rows
        and (max_rows is None or options.max_rows <= max_rows)
    ):
        n_rows = options.max_rows
        for c in columns:
            c.codes = c.codes[:n_rows]
        warnings.append(f"Read only the first {n_rows:,} data rows (row limit).")
    elif max_rows is not None and n_rows > max_rows:
        total = _count_records(text) - (1 if header else 0)
        raise DataError(
            f"The file has {max(total, max_rows + 1):,} rows; at most {max_rows:,} are supported."
        )
    del text

    data: dict[str, np.ndarray] = {}
    kinds: dict[str, str] = {}
    for name, cells in zip(names, columns, strict=True):
        data[name], kinds[name] = _type_column(name, cells, decimal, warnings)
    df = pd.DataFrame(data, index=pd.RangeIndex(n_rows))
    report = ImportReport(
        delimiter=delimiter,
        decimal=decimal,
        header=header,
        encoding=encoding,
        rows=n_rows,
        columns=len(names),
        warnings=warnings,
    )
    return Table(df, kinds), report


def _count(n: int, word: str) -> str:
    return f"{n:,} {word}" + ("" if n == 1 else "s")


def _check_text_file(raw: bytes) -> None:
    if not raw.strip():
        raise DataError("The file is empty.")
    if raw.startswith((b"PK\x03\x04", b"\xd0\xcf\x11\xe0")):
        raise DataError(
            "This looks like an Excel workbook (or another binary file), not a CSV file. "
            "In Excel, use “Save as” → “CSV UTF-8 (comma delimited)” and import that file."
        )
    if raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return
    if b"\x00" in raw[:8192]:
        raise DataError(
            "This is not a text file. Import a CSV file (values separated by commas, "
            "semicolons or tabs)."
        )


def _decode(raw: bytes, encoding: str, warnings: list[str]) -> tuple[str, str]:
    """The text of the file and the name of the encoding that was used."""
    if encoding == "auto":
        if raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
            try:
                return raw.decode("utf-16"), "utf-16"
            except UnicodeDecodeError:
                raise DataError("The file starts like UTF-16 text but cannot be read.") from None
        try:
            return raw.decode("utf-8-sig"), "utf-8"
        except UnicodeDecodeError:
            pass
        for name in ("cp1252", "latin-1"):
            try:
                text = raw.decode(name)
            except UnicodeDecodeError:
                continue
            warnings.append(
                f"The file is not UTF-8 encoded, so it was read as {_ENCODING_NAMES[name]}. "
                "If letters like ä or é look wrong, import it again with another encoding."
            )
            return text, name
        raise DataError("The text encoding of the file cannot be detected.")  # pragma: no cover
    codec = "utf-8-sig" if encoding == "utf-8" else encoding
    try:
        return raw.decode(codec), encoding
    except UnicodeDecodeError as exc:
        others = [e for e in ("utf-8", "cp1252", "latin-1") if e != encoding]
        hint = " or ".join(_ENCODING_NAMES[e] for e in others)
        raise DataError(
            f"The file is not valid {_ENCODING_NAMES[encoding]} text (byte {exc.start + 1:,} "
            f"cannot be read). Choose {hint}, or let the encoding be detected (auto)."
        ) from None


def _strip_sep_line(text: str) -> tuple[str, str | None]:
    """Remove Excel's ``sep=;`` hint line and return the delimiter it names."""
    end = text.find("\n")
    first = text if end < 0 else text[:end]
    match = _SEP_LINE.fullmatch(first.rstrip("\r"))
    if not match:
        return text, None
    return ("" if end < 0 else text[end + 1 :]), match.group(1)


def _sniff_sample(text: str) -> str:
    if len(text) <= SNIFF_CHARS:
        return text
    sample = text[:SNIFF_CHARS]
    cut = sample.rfind("\n")
    return sample[:cut] if cut > 0 else sample


def _is_blank(record: list[str]) -> bool:
    return not record or (len(record) == 1 and not record[0].strip())


def _detect_delimiter(text: str) -> str:
    """The delimiter that splits the first lines into the same number (≥ 2) of values.

    Ties go to tab, ";" and "|" before ",": a file split consistently by ";" usually uses
    the comma as decimal mark.
    """
    sample = _sniff_sample(text)
    best, best_score = ",", (-1.0, 0)
    for priority, delimiter in enumerate(("\t", ";", "|", ",")):
        counts = []
        try:
            for record in csv.reader(io.StringIO(sample), delimiter=delimiter):
                if _is_blank(record):
                    continue
                counts.append(len(record))
                if len(counts) >= SNIFF_RECORDS:
                    break
        except csv.Error:
            continue
        if not counts:
            continue
        width, frequency = Counter(counts).most_common(1)[0]
        if width < 2:
            continue
        score = (round(frequency / len(counts), 2), -priority)
        if score > best_score:
            best, best_score = delimiter, score
    return best


def _first_record_width(text: str, delimiter: str) -> int:
    try:
        for record in csv.reader(io.StringIO(_sniff_sample(text)), delimiter=delimiter):
            if not _is_blank(record):
                return len(record)
    except csv.Error:
        pass
    return 1


def _read_kwargs(delimiter: str, nrows: int | None) -> dict[str, Any]:
    return {
        "sep": delimiter,
        "header": None,
        "dtype": str,
        "keep_default_na": False,
        "na_filter": False,
        "skip_blank_lines": True,
        "engine": "c",
        "nrows": nrows,
        "quotechar": '"',
        "doublequote": True,
    }


def _parse(text: str, delimiter: str, nrows: int | None) -> pd.DataFrame:
    """All cells as strings, plus checks that every line has the same number of values."""
    frame: pd.DataFrame | None
    try:
        frame = pd.read_csv(io.StringIO(text), **_read_kwargs(delimiter, nrows))
    except pd.errors.EmptyDataError:
        raise DataError("The file is empty.") from None
    except pd.errors.ParserError as exc:
        if "Expected" not in str(exc):
            raise _parser_error(exc) from None
        frame = None  # a line has more values than the first one

    # The parser pads short lines with empty cells, so look at the lines themselves when
    # that may have happened.
    suspicious = frame is None or (frame.shape[1] > 1 and frame.iloc[:, -1].eq("").to_numpy().any())
    if suspicious:
        widest = _check_record_widths(text, delimiter, nrows)
        if frame is None:
            kwargs = _read_kwargs(delimiter, nrows)
            try:
                frame = pd.read_csv(io.StringIO(text), names=list(range(widest)), **kwargs)
            except pd.errors.ParserError as exc:
                raise _parser_error(exc) from None
    assert frame is not None
    frame.columns = range(frame.shape[1])
    return frame


def _parser_error(exc: Exception) -> DataError:
    message = str(exc).replace("Error tokenizing data. C error: ", "").strip()
    match = re.search(r"EOF inside string starting at row (\d+)", message)
    if match:
        line = int(match.group(1)) + 1
        return DataError(
            f'A quoted value that starts near line {line:,} is never closed (a closing " is '
            "missing). Fix the quotes in that line and import the file again."
        )
    return DataError(f"The file cannot be read as CSV: {message.rstrip('.')}.")


def _records(text: str, delimiter: str) -> Iterator[tuple[int, list[str]]]:
    """``(line number, values)`` of every non-blank record, like the pandas parser sees them."""
    if '"' not in text:
        for number, line in enumerate(_lines(text), 1):
            if line.strip():
                yield number, line.split(delimiter)
        return
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter)
    start = 1
    try:
        for record in reader:
            if not _is_blank(record):
                yield start, record
            start = reader.line_num + 1
    except csv.Error:
        return


def _trimmed_width(record: list[str]) -> int:
    """Number of values without the empty ones at the end (at least 1)."""
    width = len(record)
    while width > 1 and not record[width - 1].strip():
        width -= 1
    return width


def _lines(text: str) -> list[str]:
    return text.split("\n") if "\r" not in text else re.split(r"\r\n|\r|\n", text)


def _check_record_widths(text: str, delimiter: str, nrows: int | None) -> int:
    """Raise :class:`DataError` naming the first line with too few or too many values (empty
    values at the end of a line do not count). Returns the largest number of values."""
    # Quick check on the physical lines: usually all have the same number of delimiters.
    lines = _lines(text)
    counts = np.fromiter((line.count(delimiter) for line in lines), np.int64, len(lines))
    used = np.flatnonzero(counts > 0)
    if len(used) < len(lines):  # lines without a delimiter: blank, or a single value
        single = [i for i in np.flatnonzero(counts == 0).tolist() if lines[i].strip()]
        used = np.sort(np.concatenate([used, np.array(single, dtype=np.int64)]))
    if nrows is not None:
        used = used[:nrows]
    widths = counts[used] + 1
    if len(widths) == 0:
        return 1
    if widths.min() == widths.max():
        return int(widths.max())
    if '"' not in text:
        low = int(widths.min())
        longer = used[widths > low].tolist()
        if all(_trimmed_width(lines[i].split(delimiter)) <= low for i in longer):
            return int(widths.max())
    return _check_records(text, delimiter, nrows)


def _check_records(text: str, delimiter: str, nrows: int | None) -> int:
    """:func:`_check_record_widths` on the records as the CSV parser sees them (quotes)."""
    raw: list[int] = []
    trimmed: list[int] = []
    lines: list[int] = []
    for line, record in _records(text, delimiter):
        raw.append(len(record))
        trimmed.append(_trimmed_width(record))
        lines.append(line)
        if nrows is not None and len(raw) >= nrows:
            break
    if not raw:
        return 1
    width = max(trimmed)
    if min(raw) >= width:
        return max(raw)
    # Blame the lines that differ from the most common number of values.
    usual = Counter(trimmed).most_common(1)[0][0]
    culprits = (
        (i, trimmed[i] if trimmed[i] > usual else raw[i], usual)
        for i in range(len(raw))
        if trimmed[i] > usual or raw[i] < usual
    )
    fallback = ((i, raw[i], width) for i in range(len(raw)) if raw[i] < width)
    bad, count, expected = next(culprits, None) or next(fallback)
    message = (
        f"Line {lines[bad]:,} has {_count(count, 'value')}, but the other lines have "
        f"{expected:,}. Check the delimiter (detected: "
        f"{_DELIMITER_NAMES.get(delimiter, delimiter)}) and the quotes in that line."
    )
    if bad == 0:
        message += (
            " If the file starts with a title or notes, remove those lines; if the header "
            "misses a column name, add it."
        )
    raise DataError(message)


def _count_records(text: str) -> int:
    return sum(1 for _ in _records(text, ","))


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #


def _spread(values: list[str], n: int) -> list[str]:
    """At most ``n`` values, evenly spread over ``values``."""
    if len(values) <= n:
        return values
    step = len(values) / n
    return [values[int(i * step)] for i in range(n)]


def _detect_decimal(columns: list[_Cells], delimiter: str) -> str:
    """The decimal mark: "," if numeric-looking cells use a comma more often than a point."""
    if delimiter == ",":
        return "."
    comma = dot = 0
    for cells in columns:
        for cell in cells.sample_below_first(TYPE_SAMPLE):
            if _COMMA_DECIMAL_CELL.fullmatch(cell):
                comma += 1
            elif _DOT_DECIMAL_CELL.fullmatch(cell) and not _DOT_GROUPS_ONLY.fullmatch(cell):
                dot += 1
    return "," if comma > dot else "."


def _sample_type(sample: list[str], decimal: str) -> str:
    """The type of a sample: "numeric", "boolean", "date:<family>", "string" or "empty"."""
    if not sample:
        return "empty"
    if not np.isnan(_parse_numbers(sample, decimal)).any():
        return "numeric"
    if all(v.lower() in _BOOL_TOKENS for v in sample):
        return "boolean"
    for family in _date_families(sample):
        if not np.isnat(parse_dates(sample, family)).any():
            return f"date:{family}"
    return "string"


def _fits(cell: str, kind: str, decimal: str) -> bool:
    if kind == "numeric":
        return not np.isnan(parse_numbers([cell], decimal)[0])
    if kind == "boolean":
        return cell.lower() in _BOOL_TOKENS
    if kind.startswith("date:"):
        return not np.isnat(parse_dates([cell], kind[5:])[0])
    return True


def _detect_header(columns: list[_Cells], decimal: str) -> bool:
    """Whether the first line holds column names.

    Yes if a cell of the first line does not fit the type of its column below (a word above
    numbers, dates or booleans). If the typed columns all fit, no. Files with text columns
    only have a header unless the first line has empty or repeated cells, or its cells also
    occur as values below.
    """
    if len(columns[0].codes) == 0:
        return False
    first = [None if c.missing[c.codes[0]] else c.values[c.codes[0]] for c in columns]
    if len(columns[0].codes) == 1:
        # A single line: a header without data, unless it is all numbers.
        cells = [c for c in first if c is not None]
        return bool(cells) and bool(np.isnan(parse_numbers(cells, decimal)).any())
    header_votes = data_votes = 0
    string_cells: list[tuple[str | None, _Cells]] = []
    for cells, cell in zip(columns, first, strict=True):
        kind = _sample_type(cells.sample_below_first(TYPE_SAMPLE), decimal)
        if kind in ("string", "empty"):
            string_cells.append((cell, cells))
        elif cell is not None:
            if _fits(cell, kind, decimal):
                data_votes += 1
            else:
                header_votes += 1
    if header_votes:
        return True
    if data_votes:
        return False
    present = [c for c, _ in string_cells if c is not None]
    if len(present) < len(string_cells) or len(set(present)) < len(present):
        return False
    # Column names rarely occur as values in their own column.
    repeated = sum(
        1
        for cell, cells in string_cells
        if (cells.codes[1:] == cells.codes[0]).any() or (cells.values == cell).sum() > 1
    )
    return repeated * 2 < len(present)


def _clean_names(names: list[str], warnings: list[str]) -> list[str]:
    """Strip names, name empty ones ``column_<i>`` and number repeated ones ``_2``, ``_3``."""
    cleaned = [re.sub(r"[\r\n\t]+", " ", n).strip() for n in names]
    empty = [i for i, n in enumerate(cleaned) if not n]
    for i in empty:
        cleaned[i] = f"column_{i + 1}"
    taken = set(cleaned)
    seen: set[str] = set()
    renamed: list[str] = []
    for i, name in enumerate(cleaned):
        if name not in seen:
            seen.add(name)
            continue
        number = 2
        while f"{name}_{number}" in taken:
            number += 1
        cleaned[i] = f"{name}_{number}"
        taken.add(cleaned[i])
        seen.add(cleaned[i])
        renamed.append(f"{name} → {cleaned[i]}")
    if empty:
        listed = ", ".join(cleaned[i] for i in empty[:5]) + (" …" if len(empty) > 5 else "")
        warnings.append(f"Columns without a name were named {listed}.")
    if renamed:
        listed = ", ".join(renamed[:5]) + (" …" if len(renamed) > 5 else "")
        warnings.append(f"Repeated column names were renamed: {listed}.")
    return cleaned


def _examples(values: list[str], counts: np.ndarray, limit: int = 3) -> str:
    """The most frequent of ``values``, quoted and shortened: "'n.a.', 'free'"."""
    shown = []
    for i in np.argsort(-counts, kind="stable")[:limit]:
        text = values[i]
        shown.append("'" + (text if len(text) <= 20 else text[:19] + "…") + "'")
    return ", ".join(shown)


def _string_kind(present: list[str], counts: np.ndarray) -> str:
    """``text`` or ``categorical``, by the same rule as ``table.infer_kind``."""
    rows = int(counts.sum())
    unique = len(set(present))
    mean_length = float(np.dot(np.fromiter(map(len, present), np.int64, len(present)), counts))
    mean_length /= rows
    if mean_length > TEXT_MEAN_LENGTH or (
        unique > TEXT_MIN_UNIQUE and unique / rows > TEXT_UNIQUE_SHARE and mean_length > 15
    ):
        return "text"
    return "categorical"


def _type_column(
    name: str, cells: _Cells, decimal: str, warnings: list[str]
) -> tuple[np.ndarray, str]:
    """The column in the storage format of its kind (see ``table``), and the kind."""
    index, present, counts = cells.present()
    rows = int(counts.sum())

    def expand(converted: np.ndarray, fill: Any) -> np.ndarray:
        per_value = np.full(len(cells.values), fill, dtype=converted.dtype)
        per_value[index] = converted
        return per_value[cells.codes]

    if rows == 0:
        warnings.append(f"Column '{name}' is empty (no values in any row).")
        return np.full(len(cells.codes), None, dtype=object), "categorical"
    sample = _spread(present, TYPE_SAMPLE)

    if np.isnan(_parse_numbers(sample, decimal)).mean() <= 1 - ALMOST_SHARE / 2:
        numbers = _parse_numbers(present, decimal)
        bad = np.isnan(numbers)
        if not bad.any():
            text = "\n".join(present)
            if len(_LEADING_ZERO.findall(text)) < LEADING_ZERO_SHARE * len(present):
                return expand(numbers, np.nan), "numeric"
            match = _LEADING_ZERO.search(text)
            assert match is not None
            example = text[match.start() :].split("\n", 1)[0]
            warnings.append(
                f"Column '{name}' was kept as text because values like '{example}' "
                "have leading zeros (codes such as ZIP codes or article numbers). Use "
                "“Change type” if they are amounts."
            )
        elif counts[~bad].sum() >= ALMOST_SHARE * rows:
            n_bad = int(counts[bad].sum())
            bad_values = [v for v, b in zip(present, bad, strict=True) if b]
            warnings.append(
                f"Column '{name}' is categorical because {_count(n_bad, 'value')} "
                f"{'is not a number' if n_bad == 1 else 'are not numbers'}, e.g. "
                f"{_examples(bad_values, counts[bad])}. Add them to the missing-value "
                "markers, or use “Change type” to convert the column."
            )
    elif all(v.lower() in _BOOL_TOKENS for v in sample):
        booleans = parse_booleans(present)
        if not np.isnan(booleans).any():
            return expand(booleans, np.nan), "boolean"
    else:
        dates = _type_dates(name, present, counts, sample, warnings)
        if dates is not None:
            return expand(dates, np.datetime64("NaT")), "datetime"

    strings = cells.values.copy()
    strings[cells.missing] = None
    return strings[cells.codes], _string_kind(present, counts)


def _type_dates(
    name: str, present: list[str], counts: np.ndarray, sample: list[str], warnings: list[str]
) -> np.ndarray | None:
    """The parsed dates if every value is a date of one family, else None (with a warning
    if almost all values are dates)."""
    rows = counts.sum()
    best: tuple[float, np.ndarray] | None = None
    families = _date_families(sample)
    for family in families:
        parsed = parse_dates(present, family)
        bad = np.isnat(parsed)
        if not bad.any():
            if (
                family == "mdy"
                and "dmy" in families
                and not np.isnat(parse_dates(present, "dmy")).any()
            ):
                warnings.append(
                    f"Column '{name}': dates like '{present[0]}' were read as month/day/year. "
                    "If they are day/month/year, use “Change type” with the format %d/%m/%Y."
                )
            return parsed
        share = counts[~bad].sum() / rows
        if best is None or share > best[0]:
            best = (share, bad)
    if best is not None and best[0] >= ALMOST_SHARE:
        bad = best[1]
        n_bad = int(counts[bad].sum())
        bad_values = [v for v, b in zip(present, bad, strict=True) if b]
        warnings.append(
            f"Column '{name}' is categorical because {_count(n_bad, 'value')} "
            f"{'is not a valid date' if n_bad == 1 else 'are not valid dates'}, e.g. "
            f"{_examples(bad_values, counts[bad])}. Use “Change type” to convert the column."
        )
    return None
