/**
 * Helpers of the Relationships view: one row per analysed column (its strongest partner,
 * partner count, group …), filters and sorting, correlation matrices of chosen columns from
 * the sample, a clustered order for heatmaps, and parsing a pasted list of column names.
 */
import type { ColumnRelations, DataProfile, DataSample } from '@/api/types';
import { pearson, ranks } from '@/components/charts/stats';

export interface RelationRow {
  /** Index into `relations.columns`. */
  index: number;
  name: string;
  partner: string | null;
  /** Signed Spearman ρ with the strongest partner (null: none). */
  rho: number | null;
  /** |ρ| with the strongest partner, 0 if none. */
  strength: number;
  /** Pearson r with the same partner. */
  r: number | null;
  partners: number;
  group: number;
  /** Association with the target from the column profile (0 … 1), null if none. */
  target: number | null;
  /** Share of rows without a value. */
  missing: number;
}

export function relationRows(relations: ColumnRelations, profile: DataProfile): RelationRow[] {
  const byName = new Map(profile.columns.map((c) => [c.name, c]));
  return relations.columns.map((name, index) => {
    const best = relations.best[index];
    const rho = relations.best_rho[index];
    const column = byName.get(name);
    return {
      index,
      name,
      partner: best === null || best === undefined ? null : relations.columns[best],
      rho,
      strength: rho === null ? 0 : Math.abs(rho),
      r: relations.best_r[index] ?? null,
      partners: relations.partners[index] ?? 0,
      group: relations.groups?.[index] ?? -1,
      target: column?.target_association?.value ?? null,
      missing: column && profile.n_rows ? column.missing / profile.n_rows : 0,
    };
  });
}

export type StrengthFilter = 'all' | 'below' | 'above';

export interface RelationFilter {
  query: string;
  /** Relative to the threshold: all, strongest partner below it, at or above it. */
  strength: StrengthFilter;
  threshold: number;
  /** At least this many partners (null: no limit). */
  minPartners: number | null;
  /** Only this group (null: all, -1: columns in no group). */
  group: number | null;
}

export function filterRows(rows: RelationRow[], filter: RelationFilter): RelationRow[] {
  const q = filter.query.trim().toLowerCase();
  return rows.filter(
    (row) =>
      (!q || row.name.toLowerCase().includes(q)) &&
      (filter.strength === 'all' ||
        (filter.strength === 'below'
          ? row.strength < filter.threshold
          : row.strength >= filter.threshold)) &&
      (filter.minPartners === null || row.partners >= filter.minPartners) &&
      (filter.group === null || row.group === filter.group)
  );
}

export type RelationSortKey =
  'order' | 'name' | 'strength' | 'r' | 'partners' | 'group' | 'target' | 'missing';

function sortValue(
  row: RelationRow,
  key: RelationSortKey,
  position: number
): number | string | null {
  switch (key) {
    case 'order':
      return position;
    case 'name':
      return row.name.toLowerCase();
    case 'strength':
      return row.strength;
    case 'r':
      return row.r === null ? null : Math.abs(row.r);
    case 'partners':
      return row.partners;
    case 'group':
      return row.group < 0 ? null : row.group;
    case 'target':
      return row.target;
    case 'missing':
      return row.missing;
  }
}

/**
 * Rows sorted by `key`; "order" is the clustered order (related columns together) where the
 * profile has one, else the table order. Rows without a value go last in both directions.
 */
export function sortRows(
  rows: RelationRow[],
  key: RelationSortKey,
  desc: boolean,
  order: number[] | null
): RelationRow[] {
  const position = new Map<number, number>();
  (order ?? []).forEach((index, i) => position.set(index, i));
  const at = (row: RelationRow) => position.get(row.index) ?? row.index;
  return [...rows].sort((a, b) => {
    const va = sortValue(a, key, at(a));
    const vb = sortValue(b, key, at(b));
    if (va === null || vb === null) {
      return va === vb ? at(a) - at(b) : va === null ? 1 : -1;
    }
    if (va === vb) {
      return at(a) - at(b);
    }
    const result = va < vb ? -1 : 1;
    return desc ? -result : result;
  });
}

/** Columns whose strongest partner is below `threshold`, from the histogram-free rows. */
export function countBelow(relations: ColumnRelations, threshold: number): number {
  let count = 0;
  for (const rho of relations.best_rho) {
    if ((rho === null ? 0 : Math.abs(rho)) < threshold) {
      count += 1;
    }
  }
  return count;
}

// --------------------------------------------------------------------------
// Correlation matrices of chosen columns (from the sample) and a clustered order
// --------------------------------------------------------------------------

export interface SampleMatrix {
  columns: string[];
  values: (number | null)[][];
  /** Rows the values are based on (training rows of the sample where the split exists). */
  rows: number;
  trainingRows: boolean;
}

/**
 * Pearson or Spearman correlations of `columns` on the sample (its training rows if the
 * split is assigned), each pair over the rows where both have a value.
 */
export function sampleMatrix(
  sample: DataSample,
  columns: string[],
  method: 'pearson' | 'spearman'
): SampleMatrix {
  const keep: number[] = [];
  const split = sample.split;
  for (let i = 0; i < sample.rows; i += 1) {
    if (!split || split[i] === 'train') {
      keep.push(i);
    }
  }
  const byName = new Map(sample.columns.map((c) => [c.name, c]));
  const data = columns.map((name) => {
    const column = byName.get(name);
    return keep.map((i) => {
      const value = column?.values[i];
      return typeof value === 'number' && Number.isFinite(value) ? value : NaN;
    });
  });
  const values = columns.map(() => columns.map(() => null as number | null));
  for (let a = 0; a < columns.length; a += 1) {
    values[a][a] = 1;
    for (let b = a + 1; b < columns.length; b += 1) {
      const xs: number[] = [];
      const ys: number[] = [];
      const x = data[a];
      const y = data[b];
      for (let i = 0; i < x.length; i += 1) {
        if (!Number.isNaN(x[i]) && !Number.isNaN(y[i])) {
          xs.push(x[i]);
          ys.push(y[i]);
        }
      }
      const value =
        xs.length < 3
          ? null
          : method === 'pearson'
            ? pearson(xs, ys)
            : pearson(ranks(xs), ranks(ys));
      const rounded = value === null ? null : Math.round(value * 10_000) / 10_000;
      values[a][b] = rounded;
      values[b][a] = rounded;
    }
    if (data[a].every((v) => Number.isNaN(v) || v === data[a].find((w) => !Number.isNaN(w)))) {
      // constant (or empty): no correlation
      for (let b = 0; b < columns.length; b += 1) {
        values[a][b] = null;
        values[b][a] = null;
      }
    }
  }
  return { columns, values, rows: keep.length, trainingRows: !!split };
}

/**
 * Average-linkage order of the columns of a correlation matrix by 1 − |value|: related
 * columns end up next to each other, so blocks become visible in the heatmap.
 */
export function clusterOrder(values: (number | null)[][]): number[] {
  const n = values.length;
  const strength = (a: number, b: number) => Math.abs(values[a]?.[b] ?? 0);
  let clusters = Array.from({ length: n }, (_, i) => [i]);
  const distance = (a: number[], b: number[]) => {
    let sum = 0;
    for (const i of a) {
      for (const j of b) {
        sum += 1 - strength(i, j);
      }
    }
    return sum / (a.length * b.length);
  };
  while (clusters.length > 1) {
    let best = [0, 1];
    let bestDistance = Infinity;
    for (let a = 0; a < clusters.length; a += 1) {
      for (let b = a + 1; b < clusters.length; b += 1) {
        const d = distance(clusters[a], clusters[b]);
        if (d < bestDistance) {
          bestDistance = d;
          best = [a, b];
        }
      }
    }
    const [a, b] = best;
    const left = clusters[a];
    const right = clusters[b];
    // join so that the most related ends touch
    const options = [
      [...left, ...right],
      [...left, ...[...right].reverse()],
      [...[...left].reverse(), ...right],
      [...right, ...left],
    ];
    const joints = [
      strength(left[left.length - 1], right[0]),
      strength(left[left.length - 1], right[right.length - 1]),
      strength(left[0], right[0]),
      strength(right[right.length - 1], left[0]),
    ];
    const merged = options[joints.indexOf(Math.max(...joints))];
    clusters = [...clusters.filter((_, i) => i !== a && i !== b), merged];
  }
  return clusters[0] ?? [];
}

/** The columns of the strongest pairs (by strongest partner), at most `limit`. */
export function strongestPairColumns(relations: ColumnRelations, limit: number): string[] {
  const order = relations.columns
    .map((_, i) => i)
    .sort((a, b) => Math.abs(relations.best_rho[b] ?? 0) - Math.abs(relations.best_rho[a] ?? 0));
  const chosen: string[] = [];
  for (const i of order) {
    for (const j of [i, relations.best[i]]) {
      if (j !== null && j !== undefined && chosen.length < limit) {
        const name = relations.columns[j];
        if (!chosen.includes(name)) {
          chosen.push(name);
        }
      }
    }
    if (chosen.length >= limit) {
      break;
    }
  }
  return chosen;
}

// --------------------------------------------------------------------------
// Lists of column names
// --------------------------------------------------------------------------

/**
 * Column names from pasted text: separated by commas, semicolons, tabs, spaces or new lines,
 * optionally quoted (a Python list such as ['a', 'b'] works too). Names with spaces are
 * found when the text is split by commas/new lines only. Returns the known columns (in the
 * order given, once each) and the unknown names.
 */
export function parseColumnList(
  text: string,
  available: string[]
): { found: string[]; unknown: string[] } {
  const known = new Set(available);
  const clean = (part: string) =>
    part
      .trim()
      .replace(/^[[(]+|[\])]+$/g, '')
      .trim()
      .replace(/^(['"])(.*)\1$/, '$2');
  const byLines = text
    .split(/[,;\t\n\r]+/)
    .map(clean)
    .filter(Boolean);
  // Use whitespace as a separator too unless that breaks names that exist with spaces.
  const parts = byLines.every((p) => known.has(p))
    ? byLines
    : byLines.flatMap((p) => (known.has(p) ? [p] : p.split(/\s+/).map(clean).filter(Boolean)));
  const found: string[] = [];
  const unknown: string[] = [];
  for (const part of parts) {
    if (known.has(part)) {
      if (!found.includes(part)) {
        found.push(part);
      }
    } else if (!unknown.includes(part)) {
      unknown.push(part);
    }
  }
  return { found, unknown };
}
