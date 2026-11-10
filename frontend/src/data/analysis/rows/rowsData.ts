/** Pure helpers of the Rows view: filters, sorting, histograms and pairs of rows. */
import type { RowAnalysis } from '@/api/types';

export type RowFilter = 'all' | 'atypical' | 'twins' | 'group' | 'selected';

export type RowSortKey =
  'position' | 'split' | 'target' | 'spread' | 'missing' | 'nn_ratio' | 'group_size';

export interface RowSort {
  key: RowSortKey;
  descending: boolean;
}

export interface Thresholds {
  /** Spread below which a row is unusually typical. */
  atypical: number;
  /** NN ratio below which a row has a near twin. */
  twins: number;
}

const isNumber = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value);

/** Rows below `threshold` (missing values never count). */
export function countBelow(values: (number | null)[], threshold: number): number {
  let count = 0;
  for (const value of values) {
    if (isNumber(value) && value < threshold) {
      count += 1;
    }
  }
  return count;
}

/** `bins` equal bins from 0 to a round number above the largest value (at most `cap`). */
export function histogramEdges(values: (number | null)[], bins: number, cap: number): number[] {
  let max = 0;
  for (const value of values) {
    if (isNumber(value) && value > max) {
      max = value;
    }
  }
  const top = Math.min(cap, Math.max(0.1, Math.ceil(max * 10 * 1.02) / 10));
  return Array.from({ length: bins + 1 }, (_, i) => (top * i) / bins);
}

/** Counts per bin; values above the last edge go into the last bin. */
export function histogramCounts(values: (number | null)[], edges: number[]): number[] {
  const bins = edges.length - 1;
  const counts = new Array<number>(Math.max(bins, 0)).fill(0);
  if (bins <= 0) {
    return counts;
  }
  const lo = edges[0];
  const width = (edges[bins] - lo) / bins;
  for (const value of values) {
    if (!isNumber(value)) {
      continue;
    }
    const bin = width > 0 ? Math.floor((value - lo) / width) : 0;
    counts[Math.min(bins - 1, Math.max(0, bin))] += 1;
  }
  return counts;
}

export function median(values: number[]): number | null {
  if (values.length === 0) {
    return null;
  }
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

/** Each row with a near twin and its nearest row, every pair once. */
export function closePairs(analysis: RowAnalysis, threshold: number): [number, number][] {
  const { nn_ratio: ratio, neighbor } = analysis.rows;
  const seen = new Set<string>();
  const pairs: [number, number][] = [];
  ratio.forEach((value, i) => {
    const j = neighbor[i];
    if (!isNumber(value) || value >= threshold || j < 0) {
      return;
    }
    const key = i < j ? `${i}:${j}` : `${j}:${i}`;
    if (!seen.has(key)) {
      seen.add(key);
      pairs.push([i, j]);
    }
  });
  return pairs;
}

export interface PairTargets {
  /** |target difference| per pair (regression). */
  diffs: number[];
  /** Pairs with the same class / pairs with both classes known (classification). */
  same: number;
  known: number;
}

/** How the targets of pairs of rows differ (labels needed). */
export function pairTargets(analysis: RowAnalysis, pairs: [number, number][]): PairTargets {
  const target = analysis.rows.target;
  const result: PairTargets = { diffs: [], same: 0, known: 0 };
  if (!target) {
    return result;
  }
  for (const [a, b] of pairs) {
    const ta = target[a];
    const tb = target[b];
    if (ta === null || tb === null || ta === undefined || tb === undefined) {
      continue;
    }
    result.known += 1;
    if (ta === tb) {
      result.same += 1;
    }
    if (typeof ta === 'number' && typeof tb === 'number') {
      result.diffs.push(Math.abs(ta - tb));
    }
  }
  return result;
}

/** The random pairs of the analysis as index pairs. */
export function randomPairs(analysis: RowAnalysis): [number, number][] {
  const { a, b } = analysis.random_pairs;
  return a.map((value, i) => [value, b[i]] as [number, number]);
}

/** Name of a row for labels: "Row 12 (id 345)". */
export function rowName(analysis: RowAnalysis, index: number): string {
  const position = analysis.rows.position[index];
  const id = analysis.rows.id?.[index];
  const row = `Row ${(position ?? index).toLocaleString('en-US')}`;
  return id !== null && id !== undefined && analysis.id_column
    ? `${row} (${analysis.id_column} ${id})`
    : row;
}

/** Indices of the analysed rows that pass the filter and the search text, in table order. */
export function filterRows(
  analysis: RowAnalysis,
  filter: RowFilter,
  thresholds: Thresholds,
  options: { search?: string; selected?: ReadonlySet<number>; group?: number | null } = {}
): number[] {
  const { rows } = analysis;
  const search = (options.search ?? '').trim().toLowerCase();
  const result: number[] = [];
  for (let i = 0; i < rows.position.length; i += 1) {
    if (
      filter === 'atypical' &&
      !(isNumber(rows.spread[i]) && rows.spread[i]! < thresholds.atypical)
    ) {
      continue;
    }
    if (
      filter === 'twins' &&
      !(isNumber(rows.nn_ratio[i]) && rows.nn_ratio[i]! < thresholds.twins)
    ) {
      continue;
    }
    if (
      filter === 'group' &&
      (options.group === null || options.group === undefined || rows.group[i] !== options.group)
    ) {
      continue;
    }
    if (filter === 'selected' && !options.selected?.has(i)) {
      continue;
    }
    if (search) {
      const id = rows.id?.[i];
      const text = `${rows.position[i]} ${id ?? ''}`.toLowerCase();
      if (!text.split(' ').some((part) => part === search || part.startsWith(search))) {
        continue;
      }
    }
    result.push(i);
  }
  return result;
}

const SPLIT_ORDER = { train: 0, val: 1, test: 2 } as const;

function sortValue(analysis: RowAnalysis, key: RowSortKey, i: number): number | string | null {
  const { rows } = analysis;
  switch (key) {
    case 'position':
      return rows.position[i];
    case 'split':
      return rows.split ? SPLIT_ORDER[rows.split[i]] : null;
    case 'target':
      return rows.target ? rows.target[i] : null;
    case 'spread':
      return rows.spread[i];
    case 'missing':
      return rows.missing[i];
    case 'nn_ratio':
      return rows.nn_ratio[i];
    case 'group_size':
      return rows.group_size[i];
  }
}

/** `indices` sorted by `sort` (missing values last, ties in table order). */
export function sortRows(analysis: RowAnalysis, indices: number[], sort: RowSort): number[] {
  const sign = sort.descending ? -1 : 1;
  return [...indices].sort((a, b) => {
    const va = sortValue(analysis, sort.key, a);
    const vb = sortValue(analysis, sort.key, b);
    if (va === null || va === undefined) {
      return vb === null || vb === undefined ? a - b : 1;
    }
    if (vb === null || vb === undefined) {
      return -1;
    }
    if (va < vb) {
      return -sign;
    }
    if (va > vb) {
      return sign;
    }
    return a - b;
  });
}

/** A column name that is free: "group", else "group_2", … */
export function freeName(columns: string[], base = 'group'): string {
  const taken = new Set(columns);
  if (!taken.has(base)) {
    return base;
  }
  let n = 2;
  while (taken.has(`${base}_${n}`)) {
    n += 1;
  }
  return `${base}_${n}`;
}

/**
 * The "Drop rows" step for analysed rows: by the ID column if every row has an id, otherwise
 * by row number (rows of the table the analysis ran on).
 */
export function dropRowsStep(
  analysis: RowAnalysis,
  indices: number[]
): { type: 'drop_rows'; params: Record<string, unknown> } {
  const ids = analysis.rows.id;
  if (analysis.id_column && ids && indices.every((i) => ids[i] !== null && ids[i] !== undefined)) {
    return {
      type: 'drop_rows',
      params: { by: 'column', column: analysis.id_column, values: indices.map((i) => ids[i]) },
    };
  }
  return {
    type: 'drop_rows',
    params: { by: 'position', positions: indices.map((i) => analysis.rows.position[i]) },
  };
}
