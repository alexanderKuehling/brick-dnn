"""Row structure: how typical every row is, its nearest other row and groups of near-twins.

Used by the Rows analysis view (``RowAnalysis`` in ``frontend/src/api/types.ts``), by the
readiness checks (:func:`brickdnn.data.checks.row_structure_issues`) and by the row steps
(``steps/row_structure.py``: remove atypical rows, find near-duplicate groups).

What is computed, on the chosen number columns (the *features*, see :func:`choose_features`):

* every column is put on a common scale with the **robust z-score**
  ``z = (x − median) / (1.4826 · MAD)`` (the median and MAD of the fit rows: the training
  rows, or all rows before the split; the standard deviation if the MAD is 0; columns
  without spread are left out), clipped to ±``Z_CLIP`` so that a single wild value cannot
  dominate a row;
* **spread** of a row: the median of ``|z|`` over its present values. Ordinary rows have
  about 0.67 (the median of ``|z|`` of a normal distribution); a row near 0 has every value
  close to its column median (filler or synthetic rows), a large value is an unusual row;
* **distance** between two rows: the root mean square of the z differences over the columns
  both rows have (``sqrt(mean((z_a − z_b)²))``); pairs that share fewer than half of
  the columns have no distance;
* the **nearest neighbours** of every row (brute force in float32, in chunks) and the
  **typical pair distance**: the median distance of random pairs of rows. The **NN ratio**
  of a row is its nearest distance divided by the typical pair distance: rows with a near
  twin (the same person or object measured twice) have a ratio far below that of the others;
* **groups**: rows connected by a neighbour link (each row to its ``K`` nearest rows) whose
  ratio is below a threshold, as connected components;
* a **projection** of the rows on the first two principal components (PCA) of the z-scores.

Suggested thresholds come from the shape of the distributions (:func:`suggest_threshold`): a
clear valley between a small cluster and the bulk, otherwise none. Ordinary data has no such
valley, so nothing is suggested there.

Sizes are bounded: at most ``MAX_ROWS`` rows (fewer for wide tables, see :func:`row_limit`)
are analysed; above that a seeded random sample, which the result says.
"""

from __future__ import annotations

import hashlib
import math
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .table import NUMERIC_KINDS, SPLIT_NAMES, DataError, Table

# Rows analysed at most (brute-force neighbour search: the work grows with rows²) ...
MAX_ROWS = 20_000
# ... and at most this many rows × feature columns (memory: float32 values) ...
MAX_CELLS = 10_000_000
# ... and at most rows² × feature columns (time: the neighbour search does this many
# multiply-adds up to four times; 2e10 is about 1-4 s of CPU on a small server, e.g. 9,000
# rows with 230 columns).
MAX_PAIR_WORK = 2e10
# Feature columns at most (the most strongly partnered / most complete ones are kept).
MAX_FEATURES = 1_000
# Neighbours kept per row (groups link a row to these).
K_NEIGHBORS = 10
# Random pairs for the typical pair distance, and how many of them are returned.
RANDOM_PAIRS = 8_000
RETURNED_PAIRS = 2_000
# Robust z-scores are clipped to this.
Z_CLIP = 10.0
MAD_SCALE = 1.4826
# Rows per chunk of the neighbour search (chunk × rows float32 matrices).
CHUNK_CELLS = 4_000_000
# "auto" features: tables with more number columns than this use the partnered columns.
AUTO_WIDE = 20
# A column has a strong partner if its largest |Spearman r| with another column is at least
# this (and clearly above chance for the number of rows, see partner_threshold).
PARTNER_MIN = 0.3
PARTNER_ROWS = 2_000
# Threshold suggestions (see suggest_threshold): only with this many rows and features.
MIN_ROWS = 30
MIN_FEATURES = 10
TWIN_MAX_RATIO = 0.7
TWIN_MIN_SHARE = 0.05
TWIN_MIN_ROWS = 10
TWIN_REST_SHARE = 0.05
TWIN_PEAK_ROWS = 15
TWIN_DEPTH = 0.35
TWIN_SEPARATION = 0.6
ATYPICAL_MIN_ROWS = 3
ATYPICAL_MIN_SHARE = 0.002
ATYPICAL_MAX_SHARE = 0.3
ATYPICAL_GAP = 2.0
ATYPICAL_MIN_GAP = 0.25
ATYPICAL_RELATIVE = 0.5
# Starting points of the view when nothing is suggested.
DEFAULT_TWIN_RATIO = 0.5
DEFAULT_SPREAD = 0.25
SEED = 0

FEATURE_MODES = ("auto", "all", "partnered", "columns")
_ID_NAME = re.compile(r"(?:^|[_\W])(?:id|uuid|guid|idx|key|nr|no)$|^(?:id|index|row)$", re.I)
_CAMEL_ID = re.compile(r"[a-z](?:Id|ID)$")


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #


@dataclass
class FeatureChoice:
    """The columns the distances use, and why."""

    # As asked: auto, all, partnered or columns.
    mode: str
    # What was used: all, partnered or columns.
    used: str
    columns: list[str]
    # Number columns that could be used (without the target, ID and group columns).
    candidates: int
    # Columns left out on purpose: [(name, reason)], reason in target / id / group /
    # constant / no_partner / limit.
    left_out: list[tuple[str, str]] = field(default_factory=list)
    note: str | None = None
    partner_threshold: float | None = None

    def to_dict(self, limit: int = 50) -> dict[str, Any]:
        reasons: dict[str, int] = {}
        for _, reason in self.left_out:
            reasons[reason] = reasons.get(reason, 0) + 1
        return {
            "mode": self.mode,
            "used": self.used,
            "count": len(self.columns),
            "columns": self.columns[:limit],
            "candidates": self.candidates,
            "left_out": [
                {"name": n, "reason": r} for n, r in self.left_out if r not in ("no_partner",)
            ][:limit],
            "left_out_counts": reasons,
            "partner_threshold": self.partner_threshold,
            "note": self.note,
        }


def is_id_column(table: Table, name: str) -> bool:
    """An ID column by name (``id``, ``customer_id``, ``userId`` ...) with unique values."""
    if not (_ID_NAME.search(name) or _CAMEL_ID.search(name)):
        return False
    series = table.df[name]
    present = series.dropna()
    return len(present) > 0 and present.is_unique


def id_column(table: Table) -> str | None:
    """The first ID column of the table (to name rows), if any."""
    for name in table.columns:
        if table.kinds[name] in ("numeric", "categorical", "text") and is_id_column(table, name):
            return name
    return None


def partner_threshold(rows: int) -> float:
    """``PARTNER_MIN``, or more for few rows (where chance correlations get large: the
    largest of hundreds of chance |r| is about 4 / sqrt(rows))."""
    return max(PARTNER_MIN, 4.5 / math.sqrt(max(rows, 1)))


def _ranks(values: np.ndarray) -> np.ndarray:
    """Column ranks scaled to mean 0 / std 1, missing values 0 (Spearman via Pearson)."""
    n = len(values)
    order = np.argsort(values, axis=0, kind="stable")
    ranks = np.empty_like(values, dtype=np.float32)
    columns = np.arange(values.shape[1])
    ranks[order, columns] = np.arange(n, dtype=np.float32)[:, None]
    missing = np.isnan(values)
    ranks[missing] = np.nan
    with np.errstate(all="ignore"):
        mean = np.nanmean(ranks, axis=0)
        std = np.nanstd(ranks, axis=0)
    std[~(std > 0)] = np.nan
    ranks = (ranks - mean) / std
    ranks[np.isnan(ranks)] = 0.0
    return ranks


def partner_strength(values: np.ndarray) -> np.ndarray:
    """The largest |Spearman r| of every column with any other column (missing values left
    out approximately: they get the mean rank)."""
    d = values.shape[1]
    if d < 2 or len(values) < 3:
        return np.zeros(d)
    ranks = _ranks(values)
    corr = (ranks.T @ ranks) / len(ranks)
    np.fill_diagonal(corr, 0.0)
    return np.nan_to_num(np.abs(corr).max(axis=0)).astype(np.float64)


def _floats(table: Table, columns: list[str], positions: np.ndarray | None = None) -> np.ndarray:
    if not columns:
        rows = table.n_rows if positions is None else len(positions)
        return np.empty((rows, 0), dtype=np.float64)
    values = table.df[columns].to_numpy(dtype=np.float64, na_value=np.nan)
    if positions is not None:
        values = values[positions]
    values[~np.isfinite(values)] = np.nan
    return values


def choose_features(
    table: Table,
    mode: str = "auto",
    columns: list[str] | None = None,
    *,
    exclude: list[str] | tuple[str, ...] = (),
    fit_positions: np.ndarray | None = None,
) -> FeatureChoice:
    """The number columns the row statistics use.

    ``all``: every number column except the target, ID and group columns (``exclude``) and
    constant ones. ``partnered``: of those, the columns with a strong partner (largest
    |Spearman r| with another column ≥ :func:`partner_threshold`): in wide tables most
    columns without any partner are noise, and noise columns drown the distances.
    ``auto``: partnered for tables with more than ``AUTO_WIDE`` number columns, else all.
    ``columns``: exactly ``columns``. Raises :class:`DataError` for unknown columns.
    """
    if mode not in FEATURE_MODES:
        raise DataError(f"Features: choose one of {', '.join(FEATURE_MODES)}.")
    excluded = {c for c in exclude if c}
    left_out: list[tuple[str, str]] = []
    if mode == "columns":
        chosen = list(dict.fromkeys(columns or []))
        if not chosen:
            raise DataError("Features: choose at least one number column.")
        for name in chosen:
            if name not in table.kinds:
                raise DataError(f"Features: there is no column '{name}' (any more).")
            if table.kinds[name] not in NUMERIC_KINDS:
                raise DataError(f"Features: '{name}' is not a number column.")
        candidates = chosen
    else:
        candidates = []
        for name in table.columns:
            if table.kinds[name] not in NUMERIC_KINDS:
                continue
            if name in excluded:
                left_out.append((name, "target_or_group"))
            elif is_id_column(table, name):
                left_out.append((name, "id"))
            else:
                candidates.append(name)
    fit = fit_positions if fit_positions is not None else np.arange(table.n_rows)
    if len(fit) > PARTNER_ROWS:
        rng = np.random.default_rng(SEED)
        fit = np.sort(rng.choice(fit, PARTNER_ROWS, replace=False))
    values = _floats(table, candidates, fit)
    with np.errstate(all="ignore"):
        spread = np.nanmax(values, axis=0) - np.nanmin(values, axis=0) if len(values) else None
    usable = []
    keep = []
    for i, name in enumerate(candidates):
        if spread is None or not (spread[i] > 0):
            left_out.append((name, "constant"))
        else:
            usable.append(name)
            keep.append(i)
    values = values[:, keep]
    choice = FeatureChoice(mode, "columns" if mode == "columns" else "all", usable, len(usable))
    choice.left_out = left_out
    wants_partners = mode == "partnered" or (mode == "auto" and len(usable) > AUTO_WIDE)
    strength = None
    if wants_partners or len(usable) > MAX_FEATURES:
        strength = partner_strength(values)
    if wants_partners and strength is not None:
        threshold = partner_threshold(len(values))
        choice.partner_threshold = round(threshold, 3)
        partnered = [name for name, s in zip(usable, strength, strict=True) if s >= threshold]
        if len(partnered) >= 2:
            choice.used = "partnered"
            choice.left_out += [(n, "no_partner") for n in usable if n not in set(partnered)]
            keep_idx = [i for i, s in enumerate(strength) if s >= threshold]
            strength = strength[keep_idx]
            choice.columns = partnered
        else:
            choice.note = (
                "Fewer than 2 columns have a strong partner, so all number columns are used."
            )
    if len(choice.columns) > MAX_FEATURES and strength is not None:
        order = np.argsort(-strength, kind="stable")[:MAX_FEATURES]
        kept = [choice.columns[i] for i in sorted(order)]
        choice.left_out += [(n, "limit") for n in choice.columns if n not in set(kept)]
        choice.columns = kept
        choice.note = (
            f"Only the {MAX_FEATURES:,} columns with the strongest partners are used "
            "(limit of the row analysis)."
        )
    return choice


# --------------------------------------------------------------------------- #
# Scaling and per-row statistics
# --------------------------------------------------------------------------- #


def fit_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Median and robust scale (1.4826 · MAD, or the standard deviation if the MAD is 0)
    of every column; scale NaN for columns without spread."""
    d = values.shape[1]
    if len(values) == 0:
        return np.full(d, np.nan), np.full(d, np.nan)
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            medians = np.nanmedian(values, axis=0)
            mad = np.nanmedian(np.abs(values - medians), axis=0) * MAD_SCALE
            std = np.nanstd(values, axis=0)
    scales = np.where(mad > 0, mad, std)
    scales = np.where(scales > 0, scales, np.nan)
    return medians, scales


def standardize(values: np.ndarray, medians: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Robust z-scores as float32, clipped to ±``Z_CLIP``; NaN where missing."""
    with np.errstate(all="ignore"):
        z = (values - medians) / scales
    z = np.clip(z, -Z_CLIP, Z_CLIP)
    return z.astype(np.float32)


def row_spread(z: np.ndarray) -> np.ndarray:
    """Median ``|z|`` of every row over its present values (NaN for rows without any)."""
    out = np.full(len(z), np.nan)
    if z.shape[1] == 0 or len(z) == 0:
        return out
    present = ~np.isnan(z)
    has = present.any(axis=1)
    if has.any():
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out[has] = np.nanmedian(np.abs(z[has]), axis=1)
    return out


def spreads(
    table: Table, columns: list[str], medians: np.ndarray, scales: np.ndarray
) -> np.ndarray:
    """The spread (median |z|) of every row of ``table``, in chunks of rows."""
    out = np.full(table.n_rows, np.nan)
    step = max(1, CHUNK_CELLS // max(len(columns), 1))
    for start in range(0, table.n_rows, step):
        rows = np.arange(start, min(table.n_rows, start + step))
        out[rows] = row_spread(standardize(_floats(table, columns, rows), medians, scales))
    return out


def fit_features(
    table: Table,
    mode: str = "auto",
    columns: list[str] | None = None,
    *,
    exclude: list[str] | tuple[str, ...] = (),
    seed: int = SEED,
) -> tuple[FeatureChoice, np.ndarray, np.ndarray]:
    """The feature columns and their medians and scales, learned from the training rows (all
    rows while there is no split; a seeded sample of 50,000 of them at most). Columns
    without spread are left out. Raises :class:`DataError` if none is left."""
    fit_positions = np.flatnonzero(table.train_mask())
    choice = choose_features(table, mode, columns, exclude=exclude, fit_positions=fit_positions)
    if len(fit_positions) > 50_000:
        rng = np.random.default_rng(seed)
        fit_positions = np.sort(rng.choice(fit_positions, 50_000, replace=False))
    medians, scales = fit_scale(_floats(table, choice.columns, fit_positions))
    usable = np.isfinite(scales)
    if not usable.all():
        dropped = [c for c, ok in zip(choice.columns, usable, strict=True) if not ok]
        choice.left_out += [(c, "constant") for c in dropped]
        choice.columns = [c for c, ok in zip(choice.columns, usable, strict=True) if ok]
        medians, scales = medians[usable], scales[usable]
    if not choice.columns:
        raise DataError("There are no number columns with different values to compare the rows on.")
    return choice, medians, scales


def min_shared(d: int) -> int:
    """Pairs of rows need at least this many columns in common to have a distance."""
    return max(1, math.ceil(d / 2))


# --------------------------------------------------------------------------- #
# Distances and neighbours
# --------------------------------------------------------------------------- #


def pair_distances(z: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distances of the row pairs ``(a[i], b[i])`` (NaN without enough shared columns)."""
    d = z.shape[1]
    out = np.full(len(a), np.nan)
    if d == 0:
        return out
    need = min_shared(d)
    step = max(1, CHUNK_CELLS // max(d, 1))
    for start in range(0, len(a), step):
        diff = z[a[start : start + step]] - z[b[start : start + step]]
        missing = np.isnan(diff)
        count = d - missing.sum(axis=1)
        diff[missing] = 0.0
        np.multiply(diff, diff, out=diff)
        total = diff.sum(axis=1, dtype=np.float64)
        with np.errstate(all="ignore"):
            dist = np.sqrt(total / count)
        dist[count < need] = np.nan
        out[start : start + step] = dist
    return out


def nearest_neighbors(z: np.ndarray, k: int = K_NEIGHBORS) -> tuple[np.ndarray, np.ndarray]:
    """The ``k`` nearest other rows of every row: ``(indices, distances)``, both ``(n, k)``,
    nearest first. Missing neighbours (fewer rows, no shared columns) are -1 / inf.

    Brute force in float32 and in chunks of rows; with missing values the squared
    differences and the shared column counts come from four matrix products.
    """
    n, d = z.shape
    k = max(1, min(k, n - 1)) if n > 1 else 1
    indices = np.full((n, k), -1, dtype=np.int64)
    distances = np.full((n, k), np.inf, dtype=np.float32)
    if n < 2 or d == 0:
        return indices, distances
    present = ~np.isnan(z)
    complete = bool(present.all())
    z0 = np.where(present, z, 0.0).astype(np.float32)
    mask = present.astype(np.float32)
    sq = z0 * z0
    need = min_shared(d)
    row_sq = sq.sum(axis=1)
    chunk = max(1, min(n, CHUNK_CELLS // n))
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        part = z0[start:stop]
        cross = part @ z0.T
        if complete:
            total = row_sq[start:stop, None] + row_sq[None, :] - 2.0 * cross
            mean = total / np.float32(d)
        else:
            total = sq[start:stop] @ mask.T + mask[start:stop] @ sq.T - 2.0 * cross
            count = mask[start:stop] @ mask.T
            with np.errstate(all="ignore"):
                mean = total / count
            mean[count < need - 0.5] = np.inf
        np.maximum(mean, 0.0, out=mean)
        rows = np.arange(stop - start)
        mean[rows, rows + start] = np.inf
        if k < n - 1:
            nearest = np.argpartition(mean, k - 1, axis=1)[:, :k]
        else:
            nearest = np.tile(np.arange(n), (stop - start, 1))
            nearest = nearest[nearest != (rows + start)[:, None]].reshape(stop - start, n - 1)
        values = np.take_along_axis(mean, nearest, axis=1)
        order = np.argsort(values, axis=1, kind="stable")
        nearest = np.take_along_axis(nearest, order, axis=1)
        values = np.take_along_axis(values, order, axis=1)
        distances[start:stop] = np.sqrt(values[:, :k])
        indices[start:stop] = nearest[:, :k]
    indices[~np.isfinite(distances)] = -1
    return indices, distances


def random_pairs(n: int, count: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """``count`` random pairs of different rows (all pairs if there are fewer)."""
    if n < 2:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    total = n * (n - 1) // 2
    if total <= count:
        a, b = np.triu_indices(n, 1)
        return a.astype(np.int64), b.astype(np.int64)
    a = rng.integers(0, n, count)
    b = rng.integers(0, n - 1, count)
    b = np.where(b >= a, b + 1, b)
    return a.astype(np.int64), b.astype(np.int64)


# --------------------------------------------------------------------------- #
# Groups
# --------------------------------------------------------------------------- #


def components(n: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Connected components of the graph with edges ``(a[i], b[i])``: a label per node,
    numbered 0, 1, … in the order of their first node."""
    parent = np.arange(n, dtype=np.int64)

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for x, y in zip(a.tolist(), b.tolist(), strict=True):
        rx, ry = find(x), find(y)
        if rx != ry:
            if rx < ry:
                parent[ry] = rx
            else:
                parent[rx] = ry
    roots = np.array([find(i) for i in range(n)], dtype=np.int64)
    _, labels = np.unique(roots, return_inverse=True)
    # np.unique sorts the roots; the root is the smallest node, so this is first-node order.
    return labels.astype(np.int64)


def group_labels(
    neighbors: np.ndarray, ratios: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """Groups of rows linked by a neighbour link with a ratio below ``threshold``:
    ``(label per row, size of the row's group)``."""
    n = len(neighbors)
    if n == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    linked = (neighbors >= 0) & (ratios < threshold)
    rows, cols = np.nonzero(linked)
    labels = components(n, rows, neighbors[rows, cols])
    sizes = np.bincount(labels)[labels]
    return labels, sizes


# --------------------------------------------------------------------------- #
# Thresholds
# --------------------------------------------------------------------------- #


def _density(values: np.ndarray, bins: int = 512) -> tuple[np.ndarray, np.ndarray, float] | None:
    """A smoothed histogram (Gaussian kernel, half the Silverman bandwidth) of ``values``:
    ``(grid, density, bandwidth)``, or None if the values have no spread."""
    n = len(values)
    if n < 10:
        return None
    sd = float(np.std(values))
    q75, q25 = np.quantile(values, [0.75, 0.25])
    spread = min(sd, float(q75 - q25) / 1.34) or sd
    if not spread > 0:
        return None
    bandwidth = 0.5 * 0.9 * spread * n ** (-0.2)
    lo, hi = float(values.min()) - 3 * bandwidth, float(values.max()) + 3 * bandwidth
    counts, edges = np.histogram(values, bins=bins, range=(lo, hi))
    width = edges[1] - edges[0]
    sigma = bandwidth / width
    radius = max(1, int(math.ceil(4 * sigma)))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    density = np.convolve(counts.astype(np.float64), kernel, mode="same") / n
    grid = (edges[:-1] + edges[1:]) / 2
    return grid, density, bandwidth


def _valleys(density: np.ndarray) -> list[int]:
    """Indices of the lowest point between every two neighbouring peaks."""
    rising = np.diff(density)
    peaks = [
        i
        for i in range(1, len(density) - 1)
        if rising[i - 1] > 0 and rising[i] <= 0 and density[i] > 1e-12
    ]
    result = []
    for left, right in zip(peaks, peaks[1:], strict=False):
        result.append(left + int(np.argmin(density[left : right + 1])))
    return result


def _twin_threshold(ratios: np.ndarray) -> float | None:
    """The valley between rows with a near twin and the others in the NN ratios (log scale),
    see :func:`suggest_threshold`."""
    n = len(ratios)
    logs = np.sort(np.log(np.maximum(ratios, 1e-4)))
    found = _density(logs)
    if found is None:
        return None
    grid, density, bandwidth = found

    def count(x: float) -> int:
        lo, hi = np.searchsorted(logs, [x - bandwidth / 2, x + bandwidth / 2])
        return int(hi - lo)

    best = None
    valleys = _valleys(density)
    for previous, v in zip([0, *valleys], valleys, strict=False):
        # The highest point since the previous valley, and the highest to the right.
        left = previous + int(np.argmax(density[previous : v + 1]))
        right = v + int(np.argmax(density[v:]))
        below = int(np.searchsorted(logs, grid[v]))
        peaks = min(count(grid[left]), count(grid[right]))
        if (
            below >= max(TWIN_MIN_ROWS, TWIN_MIN_SHARE * n)
            and n - below >= max(TWIN_MIN_ROWS, TWIN_REST_SHARE * n)
            and (
                peaks >= TWIN_PEAK_ROWS
                or (count(grid[v]) == 0 and min(below, n - below) >= TWIN_PEAK_ROWS)
            )
            and count(grid[v]) <= TWIN_DEPTH * peaks
            and math.exp(grid[v]) <= TWIN_MAX_RATIO
            and math.exp(grid[left]) <= TWIN_SEPARATION * math.exp(grid[right])
        ):
            best = grid[v]  # the rightmost one wins
    if best is None:
        return None
    # No row near the valley: the middle (on the log scale) of the empty stretch.
    low, high = logs[logs < best], logs[logs >= best]
    if count(best) == 0 and len(low) and len(high):
        best = (low.max() + high.min()) / 2
        value = _round(math.exp(best))
        return value if math.exp(low.max()) < value < math.exp(high.min()) else math.exp(best)
    return _round(math.exp(best))


def _atypical_threshold(spreads: np.ndarray) -> float | None:
    """The middle of a wide empty gap between a small cluster of low spreads and the bulk,
    see :func:`suggest_threshold`."""
    n = len(spreads)
    s = np.sort(spreads)
    first = max(ATYPICAL_MIN_ROWS, math.ceil(ATYPICAL_MIN_SHARE * n))
    last = int(ATYPICAL_MAX_SHARE * n)
    if last < first:
        return None
    gaps = s[first : last + 1] - s[first - 1 : last]
    # The cut between s[i - 1] and s[i] leaves i rows below; look at the widest gaps only.
    best = None
    for j in np.argsort(-gaps, kind="stable")[:20].tolist():
        i = first + j
        gap = float(gaps[j])
        if gap <= 0:
            break
        bulk = s[i:]
        middle = float(np.median(bulk))
        mad = float(np.median(np.abs(bulk - middle)))
        score = gap / max(mad, 1e-9)
        if (
            score >= ATYPICAL_GAP
            and gap >= ATYPICAL_MIN_GAP * middle
            and s[i - 1] <= ATYPICAL_RELATIVE * middle
            and (best is None or score > best[0])
        ):
            best = (score, float(s[i - 1]), float(s[i]))
    if best is None:
        return None
    _, low, high = best
    value = _round((low + high) / 2)
    return value if low < value < high else (low + high) / 2


def suggest_threshold(values: np.ndarray, kind: str, features: int | None = None) -> float | None:
    """A threshold where the distribution of ``values`` clearly splits, or None.

    ``kind="twins"`` (NN ratios): the valley between rows with a near twin (low ratios) and
    the rest, on a log scale. Taken is the rightmost valley with at least
    ``TWIN_MIN_SHARE`` of the rows below and ``TWIN_REST_SHARE`` above it, where the rows
    around the valley are at most ``TWIN_DEPTH`` × the rows around the lower of the two
    peaks (each with ``TWIN_PEAK_ROWS`` rows or more), below ratio ``TWIN_MAX_RATIO``, and
    whose twin peak lies below ``TWIN_SEPARATION`` × the ratio of the other peak.

    ``kind="atypical"`` (row spreads): a small cluster of rows clearly below the bulk (every
    value close to its median): the widest gap with 3 rows / 0.2 % to 30 % of the rows below
    it that is at least ``ATYPICAL_GAP`` × the MAD of the rows above it, at least
    ``ATYPICAL_MIN_GAP`` × their median, and whose lower end is below
    ``ATYPICAL_RELATIVE`` × their median.

    Ordinary data has neither, so nothing is suggested there; tables with fewer than
    ``MIN_FEATURES`` feature columns (``features``) get no suggestion either (with few
    columns, small spreads and close neighbours happen by chance).
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < MIN_ROWS or (features is not None and features < MIN_FEATURES):
        return None
    if kind == "twins":
        return _twin_threshold(values)
    if kind == "atypical":
        return _atypical_threshold(values)
    raise ValueError(kind)  # pragma: no cover


def _round(x: float) -> float:
    """Two significant digits."""
    if x <= 0 or not math.isfinite(x):
        return x
    digits = 1 - int(math.floor(math.log10(x)))
    return round(x, max(digits, 0))


# --------------------------------------------------------------------------- #
# The whole analysis
# --------------------------------------------------------------------------- #


def row_limit(features: int) -> int:
    """Rows analysed at most for this many feature columns."""
    d = max(features, 1)
    return int(max(2, min(MAX_ROWS, MAX_CELLS // d, math.sqrt(MAX_PAIR_WORK / d))))


@dataclass
class RowStats:
    """Row statistics of a table, see the module docstring. Arrays have one entry per
    analysed row (``positions`` in the table); neighbour indices refer to these rows."""

    n_total: int
    positions: np.ndarray
    sampled: bool
    features: FeatureChoice
    medians: np.ndarray
    scales: np.ndarray
    # Feature values of the analysed rows (float32, NaN missing) and their z-scores.
    values: np.ndarray
    z: np.ndarray
    spread: np.ndarray
    missing: np.ndarray
    neighbors: np.ndarray
    distances: np.ndarray
    pair_distance: float | None
    pair_a: np.ndarray
    pair_b: np.ndarray
    pair_dist: np.ndarray
    projection: np.ndarray
    explained: list[float]
    seed: int
    duration_ms: float = 0.0

    @property
    def rows(self) -> int:
        return len(self.positions)

    def ratios(self) -> np.ndarray:
        """Neighbour distances / typical pair distance (inf where none)."""
        if not self.pair_distance:
            return np.full(self.distances.shape, np.inf)
        return self.distances.astype(np.float64) / self.pair_distance

    def nn_ratio(self) -> np.ndarray:
        ratio = self.ratios()[:, 0] if self.rows else np.empty(0)
        return np.where(np.isfinite(ratio), ratio, np.nan)

    def groups(self, threshold: float) -> tuple[np.ndarray, np.ndarray]:
        return group_labels(self.neighbors, self.ratios(), threshold)

    def suggestions(self) -> dict[str, float | None]:
        features = len(self.features.columns)
        return {
            "atypical": suggest_threshold(self.spread, "atypical", features),
            "twins": suggest_threshold(self.nn_ratio(), "twins", features),
        }


def projection(z: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """The rows on the first two principal components of the z-scores (missing values
    count as the median, 0) and the share of the variance each explains."""
    n, d = z.shape
    out = np.zeros((n, 2))
    if n < 2 or d == 0:
        return out, [0.0, 0.0]
    filled = np.nan_to_num(z.astype(np.float64))
    filled -= filled.mean(axis=0)
    cov = (filled.T @ filled) / max(n - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1][:2]
    total = float(eigvals.clip(min=0).sum()) or 1.0
    comps = eigvecs[:, order]
    # A stable sign: the largest loading positive.
    for i in range(comps.shape[1]):
        if comps[np.argmax(np.abs(comps[:, i])), i] < 0:
            comps[:, i] *= -1
    out[:, : comps.shape[1]] = filled @ comps
    explained = [float(max(eigvals[i], 0) / total) for i in order]
    explained += [0.0] * (2 - len(explained))
    return out, explained


def compute(
    table: Table,
    *,
    target: str | None = None,
    exclude: list[str] | tuple[str, ...] = (),
    mode: str = "auto",
    columns: list[str] | None = None,
    max_rows: int | None = None,
    max_work: float | None = None,
    seed: int = SEED,
    fit: str = "train",
    k: int = K_NEIGHBORS,
    with_projection: bool = True,
) -> RowStats:
    """Row statistics of ``table`` (see the module docstring).

    The scaling is learned from the training rows (``fit="train"``; all rows while there
    is no split) or from all rows (``fit="all"``). At most :func:`row_limit` rows (and
    ``max_rows``) are analysed, a seeded random sample above that. ``exclude``: columns
    never used as features besides the target (e.g. the group column).
    Raises :class:`DataError` if no number column can be used.
    """
    started = time.perf_counter()
    n_total = table.n_rows
    skip = [c for c in (target, *exclude) if c]
    fit_table = table if fit == "train" or table.split is None else table.with_split(None)
    choice, medians, scales = fit_features(fit_table, mode, columns, exclude=skip, seed=seed)
    limit = row_limit(len(choice.columns))
    if max_rows is not None:
        limit = min(limit, max_rows)
    if max_work is not None:
        limit = max(2, min(limit, int(math.sqrt(max_work / len(choice.columns)))))
    sampled = n_total > limit
    if sampled:
        rng = np.random.default_rng(seed)
        positions = np.sort(rng.choice(n_total, limit, replace=False))
    else:
        positions = np.arange(n_total)
    values = _floats(table, choice.columns, positions).astype(np.float32)
    z = standardize(values, medians, scales)
    spread = row_spread(z)
    others = [c for c in table.columns if c != target]
    missing = (
        table.df[others].isna().to_numpy().sum(axis=1)[positions]
        if others
        else np.zeros(len(positions), dtype=np.int64)
    )
    neighbors, distances = nearest_neighbors(z, k)
    rng = np.random.default_rng(seed + 1)
    a, b = random_pairs(len(positions), RANDOM_PAIRS, rng)
    dist = pair_distances(z, a, b)
    finite = np.isfinite(dist)
    pair_distance = float(np.median(dist[finite])) if finite.any() else None
    if pair_distance is not None and not pair_distance > 0:
        pair_distance = None
    keep = np.flatnonzero(finite)[:RETURNED_PAIRS]
    proj, explained = projection(z) if with_projection else (np.zeros((len(z), 2)), [0.0, 0.0])
    return RowStats(
        n_total=n_total,
        positions=positions,
        sampled=sampled,
        features=choice,
        medians=medians,
        scales=scales,
        values=values,
        z=z,
        spread=spread,
        missing=np.asarray(missing, dtype=np.int64),
        neighbors=neighbors,
        distances=distances,
        pair_distance=pair_distance,
        pair_a=a[keep],
        pair_b=b[keep],
        pair_dist=dist[keep],
        projection=proj,
        explained=explained,
        seed=seed,
        duration_ms=round((time.perf_counter() - started) * 1000, 1),
    )


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #

# Row statistics of recent tables (per server process): the preview runs on every edit,
# and the Rows view asks again for every threshold or row, mostly for unchanged data.
CACHE_BYTES = 96 * 1024 * 1024
_cache: OrderedDict[str, RowStats] = OrderedDict()
_cache_lock = threading.Lock()


def _nbytes(stats: RowStats) -> int:
    arrays = (stats.values, stats.z, stats.neighbors, stats.distances, stats.projection)
    return sum(a.nbytes for a in arrays) + 64 * stats.rows


def table_digest(table: Table) -> str:
    """A digest of the content of ``table``: columns, kinds, values and split."""
    digest = hashlib.blake2b(digest_size=20)
    digest.update(repr(sorted(table.kinds.items())).encode())
    digest.update(repr(table.columns).encode())
    if table.n_rows and table.columns:
        digest.update(pd.util.hash_pandas_object(table.df, index=False).to_numpy().tobytes())
    if table.split is not None:
        digest.update(table.split.tobytes())
    return digest.hexdigest()


def cached_compute(table: Table, **options: Any) -> RowStats:
    """:func:`compute` with a small cache keyed by the table content and the options."""
    key = table_digest(table) + repr(sorted(options.items()))
    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    stats = compute(table, **options)
    with _cache_lock:
        _cache[key] = stats
        while len(_cache) > 1 and sum(_nbytes(s) for s in _cache.values()) > CACHE_BYTES:
            _cache.popitem(last=False)
    return stats


_fit_cache: OrderedDict[str, tuple[FeatureChoice, np.ndarray, np.ndarray]] = OrderedDict()
FIT_CACHE_ENTRIES = 16


def cached_fit_features(
    table: Table, mode: str = "auto", columns: list[str] | None = None, **options: Any
) -> tuple[FeatureChoice, np.ndarray, np.ndarray]:
    """:func:`fit_features` with a small cache (the steps before the split run with every
    preview; choosing partnered columns costs a correlation matrix). Returns copies."""
    key = table_digest(table) + repr((mode, columns, sorted(options.items())))
    with _cache_lock:
        found = _fit_cache.get(key)
        if found is not None:
            _fit_cache.move_to_end(key)
    if found is None:
        found = fit_features(table, mode, columns, **options)
        with _cache_lock:
            _fit_cache[key] = found
            while len(_fit_cache) > FIT_CACHE_ENTRIES:
                _fit_cache.popitem(last=False)
    choice, medians, scales = found
    copy = FeatureChoice(**{**choice.__dict__, "columns": list(choice.columns)})
    copy.left_out = list(choice.left_out)
    return copy, medians.copy(), scales.copy()


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()
        _fit_cache.clear()


# --------------------------------------------------------------------------- #
# JSON for the Rows view
# --------------------------------------------------------------------------- #


def _num(x: Any, digits: int = 4) -> float | None:
    if x is None:
        return None
    value = float(x)
    if not math.isfinite(value):
        return None
    if value == 0:
        return 0.0
    return float(f"{value:.{digits}g}")


def _nums(values: np.ndarray, digits: int = 4) -> list[float | None]:
    return [_num(v, digits) for v in np.asarray(values, dtype=np.float64).tolist()]


def target_values(table: Table, target: str | None, task: str | None, positions: np.ndarray):
    """Target values of the rows for the view: numbers for regression, class labels (text)
    for classification, None without a target."""
    if not target or target not in table.kinds or task is None:
        return None
    series = table.df[target].iloc[positions]
    if task == "regression":
        values = series.to_numpy(dtype=np.float64, na_value=np.nan)
        return [_num(v, 6) for v in values.tolist()]
    from .pipeline import class_labels  # (pipeline imports checks, which imports this module)

    names, codes = class_labels(table.df[target], table.kinds[target])
    return [names[c] if c >= 0 else None for c in codes[positions].tolist()]


def _cell(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (float, np.floating)):
        return _num(value, 8)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def analysis_view(
    stats: RowStats,
    table: Table,
    *,
    target: str | None,
    task: str | None,
    twin_threshold: float | None = None,
    atypical_threshold: float | None = None,
) -> dict[str, Any]:
    """``RowAnalysis`` (frontend/src/api/types.ts) for the Rows view."""
    positions = stats.positions
    suggested = stats.suggestions()
    twins = twin_threshold if twin_threshold is not None else suggested["twins"]
    twins = twins if twins is not None else DEFAULT_TWIN_RATIO
    atypical = atypical_threshold if atypical_threshold is not None else suggested["atypical"]
    atypical = atypical if atypical is not None else DEFAULT_SPREAD
    labels, sizes = stats.groups(twins)
    id_name = id_column(table)
    ids = [_cell(v) for v in table.df[id_name].to_numpy()[positions].tolist()] if id_name else None
    split = (
        [SPLIT_NAMES[int(s)] for s in table.split[positions]] if table.split is not None else None
    )
    targets = target_values(table, target, task, positions)
    ratio = stats.nn_ratio()
    neighbor = stats.neighbors[:, 0] if stats.rows else np.empty(0, dtype=np.int64)
    # Random pairs: their ratios and target differences (the baseline of the pairs view).
    pair_ratio = stats.pair_dist / stats.pair_distance if stats.pair_distance else stats.pair_dist
    pairs: dict[str, Any] = {
        "a": stats.pair_a.tolist(),
        "b": stats.pair_b.tolist(),
        "ratio": _nums(pair_ratio, 3),
    }
    in_group = sizes >= 2
    multi = np.unique(labels[in_group]) if in_group.any() else np.empty(0, dtype=np.int64)
    mixed = 0
    held_twin = held = 0
    if table.split is not None and stats.rows:
        parts = table.split[positions]
        for g in multi.tolist():
            if len(np.unique(parts[labels == g])) > 1:
                mixed += 1
        held_out = parts != 0
        held = int(held_out.sum())
        ratios = stats.ratios()
        linked = (stats.neighbors >= 0) & (ratios < twins)
        train_neighbor = np.zeros_like(linked)
        valid = stats.neighbors >= 0
        train_neighbor[valid] = parts[stats.neighbors[valid]] == 0
        held_twin = int((linked & train_neighbor).any(axis=1)[held_out].sum())
    return {
        "n_total": stats.n_total,
        "analysed": stats.rows,
        "sampled": stats.sampled,
        "limit": row_limit(len(stats.features.columns)),
        "seed": stats.seed,
        "features": stats.features.to_dict(),
        "id_column": id_name,
        "target": target if targets is not None else None,
        "task": task if targets is not None else None,
        "pair_distance": _num(stats.pair_distance),
        "suggested": suggested,
        "thresholds": {"twins": twins, "atypical": atypical},
        "rows": {
            "position": positions.tolist(),
            "id": ids,
            "split": split,
            "target": targets,
            "spread": _nums(stats.spread),
            "missing": stats.missing.tolist(),
            "nn_distance": _nums(stats.distances[:, 0] if stats.rows else []),
            "nn_ratio": _nums(ratio),
            "neighbor": neighbor.tolist(),
            "group": labels.tolist(),
            "group_size": sizes.tolist(),
            "pc1": _nums(stats.projection[:, 0], 4),
            "pc2": _nums(stats.projection[:, 1], 4),
        },
        "groups": {
            "threshold": twins,
            "count": len(multi),
            "rows": int(in_group.sum()),
            "largest": int(sizes.max()) if len(sizes) else 0,
            "mixed": mixed,
            "held_out": held,
            "held_out_with_twin": held_twin,
        },
        "random_pairs": pairs,
        "explained": [_num(v, 3) for v in stats.explained],
        "duration_ms": stats.duration_ms,
    }


def similar_view(
    stats: RowStats,
    table: Table,
    row: int,
    *,
    other: int | None = None,
    target: str | None,
    task: str | None,
    k: int = K_NEIGHBORS,
    columns_shown: int = 8,
) -> dict[str, Any]:
    """``SimilarRows``: the nearest rows of analysed row ``row`` (an index into the analysed
    rows) and a column comparison with ``other`` (default: its nearest row)."""
    if not 0 <= row < stats.rows:
        raise DataError("This row is not part of the analysis (any more).")
    ratios = stats.ratios()
    id_name = id_column(table)
    positions = stats.positions
    neighbors = [int(j) for j in stats.neighbors[row, :k].tolist() if j >= 0]
    split = table.split
    targets = target_values(table, target, task, positions)

    def describe(i: int, distance: float | None, ratio: float | None) -> dict[str, Any]:
        pos = int(positions[i])
        return {
            "index": i,
            "position": pos,
            "id": _cell(table.df[id_name].iloc[pos]) if id_name else None,
            "split": SPLIT_NAMES[int(split[pos])] if split is not None else None,
            "target": targets[i] if targets is not None else None,
            "spread": _num(stats.spread[i]),
            "distance": _num(distance),
            "ratio": _num(ratio),
        }

    result: dict[str, Any] = {
        "row": describe(row, None, None),
        "neighbors": [
            describe(j, float(stats.distances[row, n]), float(ratios[row, n]))
            for n, j in enumerate(neighbors)
        ],
        "pair": None,
    }
    if other is None:
        other = neighbors[0] if neighbors else None
    if other is not None:
        if not 0 <= other < stats.rows or other == row:
            raise DataError("Choose another row of the analysis to compare with.")
        za, zb = stats.z[row].astype(np.float64), stats.z[other].astype(np.float64)
        va, vb = stats.values[row], stats.values[other]
        diff = np.abs(za - zb)
        shared = np.flatnonzero(~np.isnan(diff))
        (distance,) = pair_distances(stats.z, np.array([row]), np.array([other]))
        order = shared[np.argsort(diff[shared], kind="stable")]

        def column(i: int) -> dict[str, Any]:
            return {
                "name": stats.features.columns[i],
                "a": _num(va[i], 6),
                "b": _num(vb[i], 6),
                "za": _num(za[i], 3),
                "zb": _num(zb[i], 3),
                "diff": _num(diff[i], 3),
            }

        result["pair"] = {
            "a": describe(row, None, None),
            "b": describe(other, None, None),
            "distance": _num(distance),
            "ratio": _num(distance / stats.pair_distance)
            if stats.pair_distance and np.isfinite(distance)
            else None,
            "shared": len(shared),
            "features": len(stats.features.columns),
            "closest": [column(i) for i in order[:columns_shown].tolist()],
            "farthest": [column(i) for i in order[::-1][:columns_shown].tolist()],
        }
    return result


def summary(stats: RowStats, table: Table) -> dict[str, Any]:
    """A compact summary (MCP previews): counts at the suggested thresholds and coarse
    histograms of the spreads and NN ratios."""
    suggested = stats.suggestions()
    ratio = stats.nn_ratio()
    view: dict[str, Any] = {
        "rows_analysed": stats.rows,
        "sampled": stats.sampled,
        "features": {
            "used": stats.features.used,
            "count": len(stats.features.columns),
            "candidates": stats.features.candidates,
        },
        "spread_median": _num(np.nanmedian(stats.spread)) if stats.rows else None,
        "nn_ratio_median": _num(np.nanmedian(ratio)) if np.isfinite(ratio).any() else None,
        "suggested": suggested,
    }
    if suggested["atypical"] is not None:
        view["atypical_rows"] = int(np.sum(stats.spread < suggested["atypical"]))
    if suggested["twins"] is not None:
        labels, sizes = stats.groups(suggested["twins"])
        view["rows_with_twin"] = int(np.sum(ratio < suggested["twins"]))
        view["groups"] = int(len(np.unique(labels[sizes >= 2])))
    for key, values, edges in (
        ("spread_histogram", stats.spread, [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1, 1.5, 2]),
        ("nn_ratio_histogram", ratio, [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.5]),
    ):
        finite = values[np.isfinite(values)]
        counts, _ = np.histogram(np.clip(finite, edges[0], edges[-1]), bins=edges)
        view[key] = {"edges": edges, "counts": counts.tolist()}
    return view
