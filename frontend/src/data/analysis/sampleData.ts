/**
 * Helpers that read the preview sample for the analysis views: values of a column, groups of
 * rows by target class, category counts and cross tables, x/y pairs.
 */
import type { ColumnKind, DataProfile, DataSample } from '@/api/types';
import { MAX_SERIES } from '@/components/charts/colors';
import { formatCount, formatDateTime } from '@/components/charts/format';

export type SampleColumn = DataSample['columns'][number];

export function sampleColumn(sample: DataSample, name: string | null): SampleColumn | undefined {
  return name === null ? undefined : sample.columns.find((c) => c.name === name);
}

/** Kinds drawn on a number line (booleans as 0/1, datetimes as timestamps). */
export function isNumericKind(kind: ColumnKind): boolean {
  return kind === 'numeric' || kind === 'boolean' || kind === 'datetime';
}

/** The finite numbers of a column. */
export function numericValues(column: SampleColumn): number[] {
  const out: number[] = [];
  for (const value of column.values) {
    if (typeof value === 'number' && Number.isFinite(value)) {
      out.push(value);
    }
  }
  return out;
}

/** Category key of a cell (booleans "0"/"1" like the profile), null if missing. */
export function categoryKey(value: number | string | null): string | null {
  if (value === null || value === undefined) {
    return null;
  }
  return typeof value === 'number' ? String(value) : value;
}

const BOOLEAN_LABELS: Record<string, string> = { '0': 'false', '1': 'true' };

/** Display label of a category value. */
export function valueLabel(value: string, kind: ColumnKind): string {
  if (kind === 'boolean') {
    return BOOLEAN_LABELS[value] ?? value;
  }
  return value === '' ? '(empty)' : value;
}

export interface RowGroup {
  key: string;
  label: string;
  /** Categorical colour slot; -1 for "Other". */
  slot: number;
  /** Sample rows in the group. */
  count: number;
}

export interface RowGroups {
  /** Name of the grouping column (the target). */
  column: string;
  groups: RowGroup[];
  /** Group of each sample row; -1 where the target is missing. */
  index: Int32Array;
}

export const OTHER_KEY = '__other__';

/**
 * A number as the backend names a class of numeric labels (`class_label` in
 * backend/brickdnn/data/pipeline.py): whole numbers without ".0" (3.0 → "3"), others like
 * Python's repr (2.5 → "2.5", 1e-7 → "1e-07", 1e16 → "1e+16").
 */
function numberClassName(value: number): string {
  if (Number.isInteger(value) && Math.abs(value) < 1e15) {
    return String(value === 0 ? 0 : value);
  }
  if (!Number.isFinite(value)) {
    return Number.isNaN(value) ? 'nan' : value > 0 ? 'inf' : '-inf';
  }
  // toExponential() gives the shortest digits that round-trip, like Python's repr.
  const [mantissa, exponent] = value.toExponential().split('e');
  const e = Number(exponent);
  if (e < -4 || e >= 16) {
    return `${mantissa}e${e < 0 ? '-' : '+'}${String(Math.abs(e)).padStart(2, '0')}`;
  }
  return Number.isInteger(value) ? `${value}.0` : String(value);
}

/** A timestamp as the backend names a date class: "2021-03-04" or "2021-03-04 13:05:09". */
function dateClassName(ms: number): string {
  return formatDateTime(Math.floor(ms / 1000) * 1000);
}

const DATE_TEXT = /^(\d{4})-(\d{2})-(\d{2})(?:[ T](\d{2}):(\d{2})(?::(\d{2})(?:\.\d+)?)?)?$/;

/**
 * A class name of the profile in the form `classKey` gives sample values, so that older or
 * differently written names still match: yes/no "0"/"1" → "false"/"true", "3.0" → "3",
 * "2021-03-04 00:00:00" → "2021-03-04".
 */
export function canonicalClassName(name: string, kind: ColumnKind): string {
  if (kind === 'boolean') {
    const lower = name.trim().toLowerCase();
    if (lower === '1' || lower === 'true') {
      return 'true';
    }
    return lower === '0' || lower === 'false' ? 'false' : name;
  }
  if (kind === 'numeric' && name.trim() !== '' && Number.isFinite(Number(name))) {
    return numberClassName(Number(name));
  }
  if (kind === 'datetime') {
    const m = DATE_TEXT.exec(name.trim());
    if (m) {
      const [y, mo, d, h, mi, sec] = m.slice(1).map((part) => Number(part ?? 0));
      return dateClassName(Date.UTC(y, mo - 1, d, h, mi, sec));
    }
  }
  return name;
}

/**
 * The class name of a sample value of a target column, as the backend names the classes
 * (`class_labels` in backend/brickdnn/data/pipeline.py): yes/no "false"/"true", numbers like
 * "3" or "2.5", dates "YYYY-MM-DD" or "YYYY-MM-DD HH:MM:SS" (UTC), strings as they are;
 * null if missing.
 */
export function classKey(value: number | string | null, kind: ColumnKind): string | null {
  if (value === null || value === undefined) {
    return null;
  }
  if (typeof value === 'string') {
    return canonicalClassName(value, kind);
  }
  if (kind === 'boolean') {
    return value ? 'true' : 'false';
  }
  return kind === 'datetime' ? dateClassName(value) : numberClassName(value);
}

/** Display name of a target class (yes/no classes as "false"/"true"). */
export function classLabel(name: string, kind: ColumnKind): string {
  if (kind === 'boolean') {
    return canonicalClassName(name, kind);
  }
  return name === '' ? '(empty)' : name;
}

type GroupCache = WeakMap<DataProfile, Map<string, RowGroups | null>>;
const groupCache = new WeakMap<DataSample, GroupCache>();

/**
 * Sample rows grouped by the target's classes: in class order (as in the profile, which is
 * the order of the exported class indices), at most `max` groups – the rest fold into "Other".
 * Computed once per sample, profile and target (the views share the result; do not change it).
 */
export function classGroups(
  sample: DataSample,
  profile: DataProfile,
  target: string | null,
  max = MAX_SERIES
): RowGroups | null {
  if (!target) {
    return null;
  }
  let byProfile = groupCache.get(sample);
  if (!byProfile) {
    byProfile = new WeakMap();
    groupCache.set(sample, byProfile);
  }
  let byTarget = byProfile.get(profile);
  if (!byTarget) {
    byTarget = new Map();
    byProfile.set(profile, byTarget);
  }
  const key = `${max}:${target}`;
  let groups = byTarget.get(key);
  if (groups === undefined) {
    groups = groupRows(sample, profile, target, max);
    byTarget.set(key, groups);
  }
  return groups;
}

function groupRows(
  sample: DataSample,
  profile: DataProfile,
  target: string,
  max: number
): RowGroups | null {
  const column = sampleColumn(sample, target);
  if (!column) {
    return null;
  }
  const kind = column.kind;
  const keys = column.values.map((value) => classKey(value, kind));
  const counts = new Map<string, number>();
  for (const key of keys) {
    if (key !== null) {
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
  }
  // Class key → display name, in class order.
  const order = new Map<string, string>();
  if (profile.target?.column === target && profile.target.task === 'classification') {
    for (const c of profile.target.classes ?? []) {
      const key = canonicalClassName(c.value, kind);
      if (!order.has(key)) {
        order.set(key, classLabel(c.value, kind));
      }
    }
  }
  // Without a class list: most frequent first. Classes only seen in the sample go last.
  const rest = [...counts.entries()].filter(([key]) => !order.has(key));
  if (order.size === 0) {
    rest.sort((a, b) => b[1] - a[1]);
  }
  for (const [key] of rest) {
    order.set(key, classLabel(key, kind));
  }

  const all = [...order.entries()];
  const fold = all.length > max;
  const own = fold ? all.slice(0, max - 1) : all;
  const slotOf = new Map(own.map(([key], i) => [key, i]));
  const groups: RowGroup[] = own.map(([key, label], i) => ({
    key,
    label,
    slot: i,
    count: counts.get(key) ?? 0,
  }));
  if (fold) {
    groups.push({
      key: OTHER_KEY,
      label: `Other (${formatCount(all.length - own.length)} classes)`,
      slot: -1,
      count: all.slice(max - 1).reduce((sum, [key]) => sum + (counts.get(key) ?? 0), 0),
    });
  }
  const other = fold ? groups.length - 1 : -1;
  const index = new Int32Array(keys.length);
  keys.forEach((key, row) => {
    index[row] = key === null ? -1 : (slotOf.get(key) ?? other);
  });
  return { column: target, groups, index };
}

/** The finite numbers of a column per group. */
export function valuesByGroup(column: SampleColumn, groups: RowGroups): number[][] {
  const out = groups.groups.map(() => [] as number[]);
  column.values.forEach((value, row) => {
    const g = groups.index[row];
    if (g >= 0 && typeof value === 'number' && Number.isFinite(value)) {
      out[g].push(value);
    }
  });
  return out;
}

/** Counts per category value (most frequent first) and missing cells. */
export function countValues(column: SampleColumn): {
  counts: { key: string; count: number }[];
  missing: number;
} {
  const counts = new Map<string, number>();
  let missing = 0;
  for (const value of column.values) {
    const key = categoryKey(value);
    if (key === null) {
      missing += 1;
    } else {
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
  }
  return {
    counts: [...counts.entries()]
      .map(([key, count]) => ({ key, count }))
      .sort((a, b) => b.count - a.count || a.key.localeCompare(b.key)),
    missing,
  };
}

/**
 * Rows per category (one entry per key of `keys`, plus the rest as "other") and group: the
 * class mix of each category in the sample. Yes/no keys match as "0"/"1" or "false"/"true".
 */
export function crossTab(
  column: SampleColumn,
  groups: RowGroups,
  keys: string[]
): { rows: number[][]; other: number[]; missing: number[] } {
  const boolean = column.kind === 'boolean';
  const canonical = (key: string) => (boolean ? canonicalClassName(key, 'boolean') : key);
  const position = new Map(keys.map((key, i) => [canonical(key), i]));
  const size = groups.groups.length;
  const rows = keys.map(() => new Array<number>(size).fill(0));
  const other = new Array<number>(size).fill(0);
  const missing = new Array<number>(size).fill(0);
  column.values.forEach((value, row) => {
    const g = groups.index[row];
    if (g < 0) {
      return;
    }
    const key = boolean ? classKey(value, 'boolean') : categoryKey(value);
    if (key === null) {
      missing[g] += 1;
      return;
    }
    const p = position.get(key);
    if (p === undefined) {
      other[g] += 1;
    } else {
      rows[p][g] += 1;
    }
  });
  return { rows, other, missing };
}

/** Rows where both columns have a number: their values and sample row positions. */
export function numericPairs(
  xColumn: SampleColumn,
  yColumn: SampleColumn
): { x: Float64Array; y: Float64Array; rows: Int32Array } {
  const length = Math.min(xColumn.values.length, yColumn.values.length);
  const rows: number[] = [];
  for (let i = 0; i < length; i += 1) {
    const a = xColumn.values[i];
    const b = yColumn.values[i];
    if (
      typeof a === 'number' &&
      Number.isFinite(a) &&
      typeof b === 'number' &&
      Number.isFinite(b)
    ) {
      rows.push(i);
    }
  }
  const x = new Float64Array(rows.length);
  const y = new Float64Array(rows.length);
  rows.forEach((row, k) => {
    x[k] = xColumn.values[row] as number;
    y[k] = yColumn.values[row] as number;
  });
  return { x, y, rows: Int32Array.from(rows) };
}

export const WEEKDAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
export const MONTHS = [
  'Jan',
  'Feb',
  'Mar',
  'Apr',
  'May',
  'Jun',
  'Jul',
  'Aug',
  'Sep',
  'Oct',
  'Nov',
  'Dec',
];

/** Timestamps per weekday (Monday first) and per month (UTC). */
export function calendarCounts(values: readonly number[]): { weekday: number[]; month: number[] } {
  const weekday = new Array<number>(7).fill(0);
  const month = new Array<number>(12).fill(0);
  for (const value of values) {
    const date = new Date(value);
    if (!Number.isNaN(date.getTime())) {
      weekday[(date.getUTCDay() + 6) % 7] += 1;
      month[date.getUTCMonth()] += 1;
    }
  }
  return { weekday, month };
}

/** "Based on a random sample of 5,000 of 1,234,567 rows." */
export function sampleNote(sample: DataSample): string {
  return sample.rows >= sample.n_total
    ? `Based on all ${formatCount(sample.n_total)} rows.`
    : `Based on a random sample of ${formatCount(sample.rows)} of ${formatCount(sample.n_total)} rows.`;
}

/** What the correlations, associations and duplicate counts are based on. */
export function analysisNote(profile: DataProfile): string {
  return profile.analysis_rows < profile.n_rows
    ? `Based on a random sample of ${formatCount(profile.analysis_rows)} of ${formatCount(profile.n_rows)} rows.`
    : `Based on all ${formatCount(profile.n_rows)} rows.`;
}

/** Text lengths of a text/categorical column (for a length histogram). */
export function textLengths(column: SampleColumn): number[] {
  const out: number[] = [];
  for (const value of column.values) {
    if (typeof value === 'string') {
      out.push(value.length);
    }
  }
  return out;
}

/** Name of sample row i: its row number in the table (1-based). */
export function rowName(sample: DataSample, i: number): string {
  const index = sample.index[i];
  return `Row ${formatCount((index ?? i) + 1)}`;
}
