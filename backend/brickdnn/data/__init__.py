"""Data preparation: import data files (CSV, Excel, JSON, Parquet ...), combine, clean and
transform them, analyse them and turn them into training datasets.

Needs pandas and numpy (``pip install "brickdnn[data]"``). Nothing here imports PyTorch, and
nothing else in the package imports this module, so code generation keeps working without
these dependencies.

Modules:

``table``     :class:`Table`, the in-memory dataset every step works on
``io``        reading CSV files into a table (delimiter, decimal comma, header, column types)
``readers``   reading every other format (Excel, OpenDocument, JSON, Parquet, Feather, packed)
``formats``   the spreadsheet and JSON readers that the generated pandas script shares
``store``     saving imported tables to disk and loading them again
``profile``   statistics, histograms, correlations and warnings for the analysis views
``steps``     the preparation steps (impute, clip, encode, ...) and their catalog
``pipeline``  running a list of steps, the train/validation/test split, training readiness
``export``    training datasets, cleaned CSV files, recipes and pandas scripts
"""

from __future__ import annotations

import importlib.util


def dependencies_installed() -> bool:
    """True if pandas and numpy can be imported (cheap: does not import them)."""
    return all(importlib.util.find_spec(name) is not None for name in ("numpy", "pandas"))


MISSING_DEPENDENCIES = (
    "pandas is not installed on the server. Run `pip install pandas` in the backend "
    "environment to use data preparation."
)

# File name endings of the formats the import reads (see brickdnn.data.readers); the content
# decides in the end. Packed files (.gz, .bz2, .xz, .zip) hold one of them.
FORMAT_SUFFIXES = {
    ".csv": "csv",
    ".tsv": "csv",
    ".tab": "csv",
    ".psv": "csv",
    ".txt": "csv",
    ".xlsx": "xlsx",
    ".xlsm": "xlsx",
    ".xls": "xls",
    ".ods": "ods",
    ".json": "json",
    ".geojson": "json",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    ".parquet": "parquet",
    ".pq": "parquet",
    ".feather": "feather",
    ".arrow": "feather",
    ".arrows": "feather",
    ".ipc": "feather",
}
COMPRESSED_SUFFIXES = {".gz": "gzip", ".bz2": "bz2", ".xz": "xz", ".zip": "zip"}
# Libraries a format needs besides pandas.
FORMAT_LIBRARIES = {"xlsx": "openpyxl", "xls": "xlrd", "parquet": "pyarrow", "feather": "pyarrow"}


def file_types() -> list[str]:
    """The file name endings that can be imported here (formats whose library is missing
    are left out), e.g. for the ``accept`` attribute of a file picker."""
    available = {
        fmt
        for fmt in set(FORMAT_SUFFIXES.values())
        if fmt not in FORMAT_LIBRARIES or importlib.util.find_spec(FORMAT_LIBRARIES[fmt])
    }
    suffixes = [suffix for suffix, fmt in FORMAT_SUFFIXES.items() if fmt in available]
    return suffixes + list(COMPRESSED_SUFFIXES)
