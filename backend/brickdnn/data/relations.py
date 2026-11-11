"""How the number columns of a table relate to each other: for every column its strongest
partner (the other column it correlates with most), how many strong partners it has, and an
ordering that puts related columns next to each other.

Used by the Relationships view (``ColumnRelations`` in ``frontend/src/api/types.ts``), the
readiness check for columns that behave like noise (:func:`checks.relation_issues`) and the
"no strong partner" method of the *Select features* step. Real measurements of one object
(pixels, sensor channels, scan features) are related to some others; a column unrelated to
every other one is often noise.

Computed on the **training rows** (all rows before the split is assigned): a seeded random
sample of them on large tables, so that the work stays bounded (``RELATION_WORK``
multiply-adds at most; at least ``MIN_RELATION_ROWS`` rows). Tables with more than
``MAX_RELATION_COLUMNS`` number columns use a seeded sample of the columns.

Spearman ρ is the Pearson correlation of the ranks; both are computed for all pairs at once:
ranks (or values), standardised per column, missing values set to 0, then one matrix product
in float32, in chunks of columns. Pairs use the rows where both columns have a value (sums of
squares per pair from a second product with the presence mask). Means and ranks are those of
each column on its own rows, which differs from a pairwise-complete computation only in
columns with missing values, and only slightly.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

from .table import NUMERIC_KINDS, Table

# Rows: at most this many, at least MIN_RELATION_ROWS, and rows × columns² ≤ RELATION_WORK.
MAX_RELATION_ROWS = 20_000
MIN_RELATION_ROWS = 1_000
RELATION_WORK = 4_000_000_000
RELATION_SEED = 0
# More number columns than this: a seeded sample of them.
MAX_RELATION_COLUMNS = 3_000
# Cells (columns × columns) per block of the matrix product: bounds the memory of the
# intermediate products on very wide tables.
BLOCK_CELLS = 4_000_000
# Pairs with fewer rows where both have a value get no correlation.
MIN_PAIR_ROWS = 10
# A partner: |ρ| at least this.
PARTNER_LEVEL = 0.3
# Default threshold below which a column has no strong partner.
WEAK_LEVEL = 0.15
# Unrelated columns stay below the suggested threshold with this probability.
CHANCE_QUANTILE = 0.999
TOP_PARTNERS = 5
HISTOGRAM_STEP = 0.02
# The clustered order and the overview are computed up to this many columns.
MAX_CLUSTER_COLUMNS = 2_000
# The overview heatmap has at most this many blocks per side.
OVERVIEW_BLOCKS = 48
# Groups: clusters whose columns correlate with |ρ| ≥ PARTNER_LEVEL on average.
MIN_GROUP = 2
# The readiness check for columns without a strong partner (see noise_columns).
NOISE_MIN_COLUMNS = 20
NOISE_MIN_GROUP = 10
NOISE_MIN_SHARE = 0.1
NOISE_MAX_BETWEEN = 0.1

METHODS = ("spearman", "pearson")


@dataclass
class Relations:
    """Relations of the analysed number columns (indices refer to ``columns``)."""

    method: str
    columns: list[str]
    # Number features that could not be analysed (constant, or too few values).
    skipped: list[str]
    # Number features in the table (analysed, skipped and, with a column sample, left out).
    total: int
    rows: int
    # Rows the rows were drawn from (training rows, or all rows before the split).
    pool_rows: int
    training_rows: bool
    sampled_columns: bool
    # n × n correlations (float32, NaN on the diagonal and for undefined pairs); kept only
    # for the clustering (up to MAX_CLUSTER_COLUMNS columns).
    matrix: np.ndarray | None
    best: np.ndarray  # index of the strongest partner, -1 if none
    best_value: np.ndarray  # its signed correlation (NaN if none)
    partners: np.ndarray  # partners with |value| ≥ PARTNER_LEVEL
    top: np.ndarray  # n × k partner indices, strongest first (-1 = none)
    top_value: np.ndarray
    # Typical pair rows (rows where both have a value), for the chance level.
    pair_rows: int
    # Correlation of every column with the target (same method), if it is a number target.
    target_value: np.ndarray | None = None
    order: list[int] | None = None
    groups: list[int] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def strength(self) -> np.ndarray:
        """|correlation| with the strongest partner (0 if none)."""
        return np.nan_to_num(np.abs(self.best_value))

    def chance(self, quantile: float = 0.5) -> float | None:
        """The strongest |correlation| a column unrelated to all others reaches by chance
        with its best partner (median or another quantile), for the rows and columns here."""
        return chance_level(self.pair_rows, len(self.columns) - 1, quantile)


def chance_level(rows: int, partners: int, quantile: float = 0.5) -> float | None:
    """Quantile of max |r| over ``partners`` independent columns, with ``rows`` rows each
    (r is about normal with variance 1 / (rows - 1) for independent columns)."""
    if rows < 4 or partners < 1:
        return None
    # P(max |r| < t) = (1 - p)^partners with p = P(|r| ≥ t)
    p = 1 - quantile ** (1 / partners)
    z = NormalDist().inv_cdf(1 - p / 2)
    return min(z / math.sqrt(rows - 1), 1.0)


def suggested_level(relations: Relations) -> float:
    """Threshold for "no strong partner": ``WEAK_LEVEL``, or more where unrelated columns
    reach more by chance (few rows, many columns): the level 99.9 % of unrelated columns stay
    below (``CHANCE_QUANTILE``), rounded up to 0.01."""
    chance = relations.chance(CHANCE_QUANTILE) or 0.0
    return math.ceil(max(WEAK_LEVEL, chance) * 100 - 1e-9) / 100


@dataclass
class NoiseColumns:
    """Columns without a strong partner in a table whose other columns are clearly related."""

    columns: list[str]
    level: float  # strongest |ρ| of these columns below this
    strong: int  # columns with a partner at 2 × level or more
    analysed: int


def noise_columns(relations: Relations) -> NoiseColumns | None:
    """The columns without a strong partner, if the table shows the signature of injected or
    broken columns: many number columns (``NOISE_MIN_COLUMNS``), a clear group of columns
    without a partner above ``suggested_level`` and one with partners at twice that level
    (each at least ``NOISE_MIN_GROUP`` columns and ``NOISE_MIN_SHARE`` of them), and few
    columns in between (``NOISE_MAX_BETWEEN``). Ordinary tables with a few independent
    features stay quiet."""
    n = len(relations.columns)
    if n < NOISE_MIN_COLUMNS:
        return None
    level = suggested_level(relations)
    strength = relations.strength
    weak = strength < level
    strong = strength >= 2 * level
    enough = max(NOISE_MIN_GROUP, NOISE_MIN_SHARE * n)
    between = n - int(weak.sum()) - int(strong.sum())
    if weak.sum() < enough or strong.sum() < enough or between > NOISE_MAX_BETWEEN * n:
        return None
    names = [c for c, w in zip(relations.columns, weak, strict=True) if w]
    return NoiseColumns(names, level, int(strong.sum()), n)


def without_partner(relations: Relations, level: float) -> list[str]:
    """The analysed columns whose strongest partner has |correlation| below ``level``, and
    the columns that could not be analysed (constant)."""
    weak = [c for c, s in zip(relations.columns, relations.strength, strict=True) if s < level]
    return weak + relations.skipped


def numeric_features(table: Table, target: str | None) -> list[str]:
    return [c for c in table.columns if c != target and table.kinds[c] in NUMERIC_KINDS]


def _floats(series: pd.Series) -> np.ndarray:
    values = series.to_numpy(dtype=np.float64, na_value=np.nan).copy()
    values[~np.isfinite(values)] = np.nan
    return values


def _row_positions(table: Table, columns: int) -> tuple[np.ndarray, int, bool]:
    """Sorted rows to use, the rows they come from and whether those are training rows."""
    mask = table.train_mask() if table.split is not None else np.ones(table.n_rows, dtype=bool)
    positions = np.flatnonzero(mask)
    limit = int(RELATION_WORK // max(columns, 1) ** 2)
    limit = min(MAX_RELATION_ROWS, max(MIN_RELATION_ROWS, limit))
    pool = len(positions)
    if pool > limit:
        rng = np.random.default_rng(RELATION_SEED)
        positions = np.sort(rng.choice(positions, limit, replace=False))
    return positions, pool, table.split is not None


def _standardized(values: np.ndarray, method: str) -> np.ndarray:
    """Ranks (Spearman) or values, standardised per column over the present values (sum of
    squares 1); 0 where missing. float32 rows × columns."""
    if method == "spearman":
        values = pd.DataFrame(values).rank(method="average").to_numpy(dtype=np.float64)
    present = ~np.isnan(values)
    count = present.sum(axis=0)
    with np.errstate(all="ignore"):
        mean = np.nansum(values, axis=0) / np.maximum(count, 1)
        centred = np.where(present, values - mean, 0.0)
        scale = np.sqrt((centred * centred).sum(axis=0))
        z = centred / np.where(scale > 0, scale, 1.0)
    return z.astype(np.float32)


@dataclass
class _Standardized:
    z: np.ndarray  # rows × columns, float32, 0 where missing
    mask: np.ndarray | None  # presence as float32; None without missing values


def _prepare(values: np.ndarray, method: str) -> _Standardized:
    present = ~np.isnan(values)
    mask = None if present.all() else present.astype(np.float32)
    return _Standardized(_standardized(values, method), mask)


def _pair_rows(present: np.ndarray) -> tuple[int, bool]:
    """Typical rows per pair where both columns have a value, and whether every pair has at
    least ``MIN_PAIR_ROWS`` (|A ∩ B| ≥ |A| - |not B|, so no pair counts are needed then)."""
    rows = present.shape[0]
    count = present.sum(axis=0)
    missing = rows - count
    share = float(np.median(count)) / max(rows, 1)
    enough = len(count) < 2 or int(count.min()) - int(np.sort(missing)[-1]) >= MIN_PAIR_ROWS
    return round(rows * share * share), enough


def _block_size(n: int) -> int:
    """Columns per block of the matrix product: about ``BLOCK_CELLS`` cells per block."""
    return max(64, min(1024, BLOCK_CELLS // max(n, 1)))


def _blocks(data: _Standardized, enough: bool) -> Iterator[tuple[int, np.ndarray]]:
    """Rows ``start …`` of the correlation matrix, a block at a time (float32, NaN on the
    diagonal and for pairs with fewer than ``MIN_PAIR_ROWS`` rows)."""
    z, mask = data.z, data.mask
    n = z.shape[1]
    squares = None if mask is None else z * z
    size = _block_size(n)
    for start in range(0, n, size):
        stop = min(start + size, n)
        block = z[:, start:stop].T @ z  # (b × n): sums of products
        if mask is not None and squares is not None:
            # Σ z_i² over the rows where j has a value (z_i is 0 where i is missing), and
            # Σ z_j² over the rows where i has a value
            sq_i = squares[:, start:stop].T @ mask
            sq_j = mask[:, start:stop].T @ squares
            with np.errstate(all="ignore"):
                block /= np.sqrt(sq_i * sq_j)
            if not enough:
                counts = mask[:, start:stop].T @ mask
                block[counts < MIN_PAIR_ROWS] = np.nan
        np.clip(block, -1, 1, out=block)
        rows = np.arange(stop - start)
        block[rows, rows + start] = np.nan
        yield start, block


def correlation_matrix(values: np.ndarray, method: str = "spearman") -> tuple[np.ndarray, int]:
    """All pairwise correlations of the columns of ``values`` (rows × columns, NaN where
    missing; no constant columns): float32 matrix with NaN on the diagonal and for pairs with
    fewer than ``MIN_PAIR_ROWS`` common rows; and the typical rows per pair."""
    pair_rows, enough = _pair_rows(~np.isnan(values))
    n = values.shape[1]
    result = np.empty((n, n), dtype=np.float32)
    for start, block in _blocks(_prepare(values, method), enough):
        result[start : start + len(block)] = block
    return result, pair_rows


def _with_target(data: _Standardized, y: np.ndarray, method: str) -> np.ndarray:
    """Correlation of every column with ``y`` over the rows where both have a value."""
    ok = ~np.isnan(y)
    if ok.sum() < MIN_PAIR_ROWS:
        return np.full(data.z.shape[1], np.nan)
    zy = _standardized(y[:, None], method)[:, 0]
    with np.errstate(all="ignore"):
        products = zy @ data.z
        if data.mask is None and ok.all():
            r = products
        else:
            mask = data.mask if data.mask is not None else np.ones_like(data.z)
            sum_x = ok.astype(np.float32) @ (data.z * data.z)
            sum_y = (zy * zy) @ mask
            r = products / np.sqrt(sum_x * sum_y)
    r = np.clip(r.astype(np.float64), -1, 1)
    r[~np.isfinite(r)] = np.nan
    return r


def _values(table: Table, names: list[str], positions: np.ndarray) -> np.ndarray:
    """The columns as float64 rows × columns (Fortran order), NaN where missing."""
    values = table.df[names].take(positions).to_numpy(dtype=np.float64, na_value=np.nan)
    values = np.array(values, dtype=np.float64, order="F")  # a writable copy
    values[~np.isfinite(values)] = np.nan
    return values


def column_relations(
    table: Table,
    target: str | None = None,
    *,
    method: str = "spearman",
    cluster: bool = True,
    columns: list[str] | None = None,
    sample_columns: bool = True,
) -> Relations:
    """Relations of the number features (all number and yes/no columns but ``target``, or
    ``columns``) on the training rows, see the module docstring. ``cluster``: also the
    clustered order and groups (up to ``MAX_CLUSTER_COLUMNS`` columns). ``sample_columns``
    False: all columns however many (statistics per column only, block by block)."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}")
    if target not in table.kinds:
        target = None
    names = columns if columns is not None else numeric_features(table, target)
    total = len(names)
    sampled_columns = sample_columns and total > MAX_RELATION_COLUMNS
    if sampled_columns:
        rng = np.random.default_rng(RELATION_SEED)
        chosen = set(rng.choice(total, MAX_RELATION_COLUMNS, replace=False).tolist())
        names = [c for i, c in enumerate(names) if i in chosen]
    positions, pool, training = _row_positions(table, len(names))
    values = _values(table, names, positions) if names else np.empty((len(positions), 0))
    # Columns with at least two different values on these rows.
    present = ~np.isnan(values)
    highest = np.where(present, values, -np.inf).max(axis=0, initial=-np.inf)
    lowest = np.where(present, values, np.inf).min(axis=0, initial=np.inf)
    usable = highest > lowest
    analysed = [c for c, ok in zip(names, usable, strict=True) if ok]
    skipped = [c for c, ok in zip(names, usable, strict=True) if not ok]
    values = values[:, usable]
    n = len(analysed)
    empty_index = np.full(n, -1, dtype=np.int64)
    if n < 2:
        return Relations(
            method=method,
            columns=analysed,
            skipped=skipped,
            total=total,
            rows=len(positions),
            pool_rows=pool,
            training_rows=training,
            sampled_columns=sampled_columns,
            matrix=None,
            best=empty_index,
            best_value=np.full(n, np.nan),
            partners=np.zeros(n, dtype=np.int64),
            top=np.full((n, 0), -1, dtype=np.int64),
            top_value=np.zeros((n, 0)),
            pair_rows=len(positions),
        )
    pair_rows, enough = _pair_rows(~np.isnan(values))
    data = _prepare(values, method)
    keep = cluster and n <= MAX_CLUSTER_COLUMNS
    matrix = np.empty((n, n), dtype=np.float32) if keep else None
    k = min(TOP_PARTNERS, n - 1)
    best = np.empty(n, dtype=np.int64)
    best_value = np.empty(n)
    partners = np.empty(n, dtype=np.int64)
    top = np.empty((n, k), dtype=np.int64)
    top_value = np.empty((n, k))
    for start, block in _blocks(data, enough):
        stop = start + len(block)
        if matrix is not None:
            matrix[start:stop] = block
        filled = np.nan_to_num(np.abs(block), nan=-1.0)
        rows = np.arange(stop - start)
        chosen = filled.argmax(axis=1)
        found = filled[rows, chosen] >= 0
        best[start:stop] = np.where(found, chosen, -1)
        best_value[start:stop] = np.where(found, block[rows, chosen], np.nan)
        partners[start:stop] = (filled >= PARTNER_LEVEL).sum(axis=1)
        part = np.argpartition(-filled, k - 1, axis=1)[:, :k]
        part = np.take_along_axis(
            part, np.argsort(-np.take_along_axis(filled, part, axis=1), axis=1), axis=1
        )
        part_value = np.take_along_axis(block, part, axis=1).astype(np.float64)
        top[start:stop] = np.where(np.isnan(part_value), -1, part)
        top_value[start:stop] = part_value

    target_value = None
    extra: dict[str, Any] = {}
    if method == "spearman":
        extra["best_pearson"] = _pair_pearson(values, best)
    if target is not None and table.kinds[target] in NUMERIC_KINDS:
        y = _floats(table.df[target].iloc[positions])
        target_value = _with_target(data, y, method)

    relations = Relations(
        method=method,
        columns=analysed,
        skipped=skipped,
        total=total,
        rows=len(positions),
        pool_rows=pool,
        training_rows=training,
        sampled_columns=sampled_columns,
        matrix=matrix,
        best=best.astype(np.int64),
        best_value=best_value.astype(np.float64),
        partners=partners.astype(np.int64),
        top=top.astype(np.int64),
        top_value=top_value,
        pair_rows=pair_rows,
        target_value=target_value,
        extra=extra,
    )
    if matrix is not None:
        relations.order, relations.groups = cluster_columns(np.abs(matrix))
    return relations


def cluster_columns(strength: np.ndarray) -> tuple[list[int], list[int]]:
    """Average-linkage clustering of the columns by the distance 1 - |correlation|.

    Returns the leaf order (related columns next to each other) and a group per column:
    the clusters whose columns correlate with |ρ| ≥ ``PARTNER_LEVEL`` on average, numbered
    by size from 0 (largest); -1 for columns in no such cluster of at least ``MIN_GROUP``.
    Nearest-neighbour bookkeeping keeps it at about n vector operations of length n.
    """
    n = strength.shape[0]
    strength = np.nan_to_num(strength.astype(np.float64), nan=0.0)
    distance = 1.0 - strength
    np.fill_diagonal(distance, np.inf)
    sizes = np.ones(n)
    members: list[list[int] | None] = [[i] for i in range(n)]
    active = np.ones(n, dtype=bool)
    nearest = distance.argmin(axis=1)
    nearest_d = distance[np.arange(n), nearest]
    cut = 1.0 - PARTNER_LEVEL
    groups_at_cut: list[list[int]] | None = None
    for _ in range(n - 1):
        i = int(np.argmin(nearest_d))
        d = nearest_d[i]
        if groups_at_cut is None and d > cut:
            groups_at_cut = [list(m) for m in members if m is not None]
        j = int(nearest[i])
        if not math.isfinite(d):
            break
        # merge j into i
        merged = (sizes[i] * distance[i] + sizes[j] * distance[j]) / (sizes[i] + sizes[j])
        merged[i] = np.inf
        distance[i] = merged
        distance[:, i] = merged
        distance[j] = np.inf
        distance[:, j] = np.inf
        sizes[i] += sizes[j]
        members[i] = _joined(members[i], members[j], strength)  # type: ignore[arg-type]
        members[j] = None
        active[j] = False
        nearest_d[j] = np.inf
        # rows that pointed at i or j: search again; others may now be closer to i
        stale = np.flatnonzero(active & ((nearest == i) | (nearest == j)))
        for k in [i, *stale.tolist()]:
            nearest[k] = int(distance[k].argmin())
            nearest_d[k] = distance[k, nearest[k]]
        closer = active & (merged < nearest_d)
        closer[i] = False
        nearest[closer] = i
        nearest_d[closer] = merged[closer]
    if groups_at_cut is None:
        groups_at_cut = [list(m) for m in members if m is not None]
    order = [c for m in members if m is not None for c in m]
    groups = [-1] * n
    ranked = sorted(
        (g for g in groups_at_cut if len(g) >= MIN_GROUP), key=lambda g: (-len(g), g[0])
    )
    for number, group in enumerate(ranked):
        for c in group:
            groups[c] = number
    return order, groups


def _joined(a: list[int], b: list[int], strength: np.ndarray) -> list[int]:
    """Two leaf orders joined so that the most related ends touch."""
    s = strength
    options = (
        (s[a[-1], b[0]], a + b),
        (s[a[-1], b[-1]], a + b[::-1]),
        (s[a[0], b[0]], a[::-1] + b),
        (s[a[0], b[-1]], b + a),
    )
    return max(options, key=lambda o: o[0])[1]


def overview(relations: Relations, blocks: int = OVERVIEW_BLOCKS) -> dict[str, Any] | None:
    """The whole matrix in the clustered order, downsampled to at most ``blocks`` blocks per
    side: ``{"starts": [...], "values": [[mean |ρ| of the block pair]]}`` (``starts`` index
    into ``order``; one block per column when there are few). The diagonal of the matrix
    is left out of the means."""
    if relations.order is None or relations.matrix is None or len(relations.columns) < 2:
        return None
    order = np.asarray(relations.order)
    n = len(order)
    count = min(blocks, n)
    starts = np.floor(np.arange(count) * n / count).astype(np.int64)
    strength = np.abs(relations.matrix[np.ix_(order, order)]).astype(np.float64)
    finite = np.isfinite(strength)
    sums = np.add.reduceat(np.add.reduceat(np.where(finite, strength, 0.0), starts, 0), starts, 1)
    cells = np.add.reduceat(np.add.reduceat(finite.astype(np.float64), starts, 0), starts, 1)
    with np.errstate(all="ignore"):
        means = np.round(sums / cells, 2)
    return {"starts": starts.tolist(), "values": _plain(means)}


def histogram(strength: np.ndarray) -> dict[str, list]:
    """Counts of the strongest-partner |ρ| in bins of ``HISTOGRAM_STEP`` from 0 to 1."""
    bins = round(1 / HISTOGRAM_STEP)
    edges = [round(i * HISTOGRAM_STEP, 4) for i in range(bins + 1)]
    counts = np.histogram(np.clip(strength, 0, 1), bins=bins, range=(0.0, 1.0))[0]
    return {"edges": edges, "counts": counts.tolist()}


def _plain(values: np.ndarray, digits: int | None = None) -> list:
    """A float array as (nested) lists, NaN as None, optionally rounded."""
    if digits is not None:
        values = np.round(values, digits)
    return np.where(np.isfinite(values), values, None).tolist()


def _round(value: float | None, digits: int = 3) -> float | None:
    return round(float(value), digits) if value is not None and math.isfinite(value) else None


def to_json(relations: Relations) -> dict[str, Any]:
    """``ColumnRelations`` (frontend/src/api/types.ts)."""
    n = len(relations.columns)
    pearson = relations.extra.get("best_pearson")
    has_top = relations.top >= 0
    return {
        "method": relations.method,
        "columns": relations.columns,
        "skipped": relations.skipped,
        "total": relations.total,
        "rows": relations.rows,
        "pool_rows": relations.pool_rows,
        "training_rows": relations.training_rows,
        "sampled_columns": relations.sampled_columns,
        "best": np.where(relations.best >= 0, relations.best, None).tolist(),
        "best_rho": _plain(relations.best_value, 4),
        "best_r": _plain(pearson, 4) if pearson is not None else [None] * n,
        "partners": relations.partners.tolist(),
        "partner_level": PARTNER_LEVEL,
        "top": [row[ok].tolist() for row, ok in zip(relations.top, has_top, strict=True)],
        "top_rho": [
            row[ok].tolist()
            for row, ok in zip(np.round(relations.top_value, 3), has_top, strict=True)
        ],
        "histogram": histogram(relations.strength),
        "weak_level": WEAK_LEVEL,
        "suggested_level": suggested_level(relations),
        "chance": _round(relations.chance(0.5)),
        "chance_high": _round(relations.chance(CHANCE_QUANTILE)),
        "order": relations.order,
        "groups": relations.groups,
        "overview": overview(relations),
    }


def _pair_pearson(values: np.ndarray, partner: np.ndarray) -> np.ndarray:
    """Pearson r of every column of ``values`` with column ``partner[j]`` (-1: none), over
    the rows where both have a value."""
    result = np.full(values.shape[1], np.nan)
    if values.shape[1] < 2:
        return result
    a, b = values, values[:, np.maximum(partner, 0)]
    ok = ~np.isnan(a) & ~np.isnan(b)
    count = ok.sum(axis=0)
    with np.errstate(all="ignore"):
        a = np.where(ok, a, 0.0)
        b = np.where(ok, b, 0.0)
        da = np.where(ok, a - a.sum(axis=0) / np.maximum(count, 1), 0.0)
        db = np.where(ok, b - b.sum(axis=0) / np.maximum(count, 1), 0.0)
        r = (da * db).sum(axis=0) / np.sqrt((da * da).sum(axis=0) * (db * db).sum(axis=0))
    r[(partner < 0) | (count < MIN_PAIR_ROWS) | ~np.isfinite(r)] = np.nan
    return np.clip(r, -1, 1)
