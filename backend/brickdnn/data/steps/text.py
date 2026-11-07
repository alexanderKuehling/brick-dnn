"""Numbers and words for step summaries ("Clipped 57 values (1.2 %) in 2 columns.")."""

from __future__ import annotations

import math
from typing import Any


def fmt_count(n: int | float) -> str:
    """``1234`` -> ``"1,234"``."""
    return f"{int(n):,}"


def fmt_number(x: Any) -> str:
    """A number for a sentence: thousands separators, about four significant digits."""
    if x is None:
        return "–"
    x = float(x)
    if not math.isfinite(x):
        return "–"
    if x == 0:
        return "0"
    ax = abs(x)
    if ax >= 1e15 or ax < 1e-4:
        return f"{x:.3g}"
    if x.is_integer():
        return f"{int(x):,}"
    decimals = max(0, 3 - math.floor(math.log10(ax)))
    text = f"{x:,.{decimals}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def fmt_share(part: int | float, whole: int | float) -> str:
    """``part / whole`` as a percentage, e.g. ``"1.2 %"``."""
    if not whole:
        return "0 %"
    share = 100.0 * part / whole
    if 0 < share < 0.1:
        return "< 0.1 %"
    if 99.9 < share < 100:
        return "> 99.9 %"
    return f"{share:.1f} %".replace(".0 %", " %")


def plural(n: int, singular: str, plural_form: str | None = None) -> str:
    """``"1 value"``, ``"57 values"``."""
    return f"{fmt_count(n)} {singular if n == 1 else (plural_form or singular + 's')}"
