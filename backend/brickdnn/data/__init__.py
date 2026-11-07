"""Data preparation: import CSV files, combine, clean and transform them, analyse them and turn
them into training datasets.

Needs pandas and numpy (``pip install "brickdnn[data]"``). Nothing here imports PyTorch, and
nothing else in the package imports this module, so code generation keeps working without
these dependencies.

Modules:

``table``     :class:`Table`, the in-memory dataset every step works on
``io``        reading CSV files into a table (delimiter, decimal comma, header, column types)
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
