/**
 * Groups the per-column warnings of a profile by kind ("Missing values in 828 columns
 * (6–9 % of the rows)"), so that wide tables get a few entries instead of hundreds. A group's
 * fix is one step for all its columns where the steps take a list of columns.
 */
import type { DataFix, DataProfile, DataWarning } from '@/api/types';
import { formatCount, formatNumber, formatPercent } from '@/components/charts/format';
import { i18n } from '@/i18n';

/** Warnings of one kind form a group from this many columns on. */
export const MIN_GROUP = 3;

export interface WarningGroup {
  key: string;
  code: string;
  severity: DataWarning['severity'];
  /** One sentence for the group, e.g. "Missing values in 828 columns (6–9 % of the rows)". */
  title: string;
  columns: string[];
  warnings: DataWarning[];
  /** One step for all columns, or null. */
  fix: DataFix | null;
}

export type WarningItem =
  | { type: 'single'; key: string; warning: DataWarning }
  | { type: 'group'; key: string; group: WarningGroup };

function fixKey(fix: DataFix | null): string {
  if (!fix) {
    return '';
  }
  const rest = Object.fromEntries(
    Object.entries(fix.params)
      .filter(([key]) => key !== 'columns')
      .sort(([a], [b]) => a.localeCompare(b))
  );
  return `${fix.type}:${JSON.stringify(rest)}`;
}

/** Whether the fix works on a list of columns (and on this warning's column only). */
function listFix(warning: DataWarning): boolean {
  const columns = warning.fix?.params.columns;
  return Array.isArray(columns) && columns.length === 1 && columns[0] === warning.column;
}

function range(values: number[], format: (value: number) => string): string {
  if (values.length === 0) {
    return '';
  }
  const low = Math.min(...values);
  const high = Math.max(...values);
  const a = format(low);
  const b = format(high);
  return a === b ? a : `${a.replace(/\s?%$/, '')}–${b}`;
}

const NUMBER = (value: number) => formatNumber(value, 2);

/** The title of a group of `code` warnings about `columns`. */
export function groupTitle(
  code: string,
  severity: DataWarning['severity'],
  columns: string[],
  profile: DataProfile | null
): string {
  const n = i18n.t('data.columns', { count: columns.length, value: formatCount(columns.length) });
  const byName = new Map((profile?.columns ?? []).map((c) => [c.name, c]));
  const found = columns.map((c) => byName.get(c)).filter((c) => c !== undefined);
  const rows = profile?.n_rows ?? 0;
  const detail = (text: string) => (text ? ` (${text})` : '');
  switch (code) {
    case 'high_missing': {
      const shares = rows ? found.map((c) => c.missing / rows) : [];
      const text = shares.length
        ? i18n.t('ui.data.warningGroups.ofTheRows', { range: range(shares, formatPercent) })
        : '';
      return severity === 'warning'
        ? i18n.t('ui.data.warningGroups.mostlyOrAllMissing', { n, detail: detail(text) })
        : i18n.t('ui.data.warningGroups.missingValuesIn', { n, detail: detail(text) });
    }
    case 'outliers': {
      const shares = found
        .filter((c) => c.stats && c.count)
        .map((c) => c.stats!.iqr_outliers / c.count);
      const text = shares.length
        ? i18n.t('ui.data.warningGroups.ofTheValues', { range: range(shares, formatPercent) })
        : '';
      return i18n.t('ui.data.warningGroups.outliersByTheIqrRule', { n, detail: detail(text) });
    }
    case 'skewed': {
      const skews = found.map((c) => c.stats?.skew).filter((s) => typeof s === 'number');
      const text = skews.length ? `skewness ${range(skews as number[], NUMBER)}` : '';
      return i18n.t('ui.data.warningGroups.stronglySkewed', { n, detail: detail(text) });
    }
    case 'many_zeros':
      return i18n.t('ui.data.warningGroups.mostlyZeros', { n });
    case 'constant':
      return i18n.t('ui.data.warningGroups.theSameValueInEvery', { n });
    case 'id_like':
      return i18n.t('ui.data.warningGroups.aDifferentValueInEvery', { n });
    case 'text':
      return i18n.t('ui.data.warningGroups.freeText', { n });
    case 'mixed_types':
      return i18n.t('ui.data.warningGroups.numbersStoredAsText', { n });
    case 'high_cardinality': {
      const counts = found.map((c) => c.unique);
      const text = counts.length ? `${range(counts, formatCount)} values` : '';
      return i18n.t('ui.data.warningGroups.manyCategories', { n, detail: detail(text) });
    }
    case 'high_correlation':
      return i18n.t('ui.data.warningGroups.almostCopiesOfOtherColumns', { n });
    default:
      return `${code.replace(/_/g, ' ')}: ${n}`;
  }
}

/** One fix for all warnings of a group: the same step with all their columns. */
function groupFix(warnings: DataWarning[]): DataFix | null {
  const first = warnings[0]?.fix;
  if (!first || !warnings.every(listFix)) {
    return null;
  }
  const columns = warnings.map((w) => w.column!);
  const name = warnings[0].column!;
  const label = first.label.includes(name)
    ? first.label.replace(
        name,
        i18n.t('data.columns', { count: columns.length, value: formatCount(columns.length) })
      )
    : i18n.t('ui.data.warningGroups.columns', {
        label: first.label,
        length: formatCount(columns.length),
      });
  return { type: first.type, params: { ...first.params, columns }, label };
}

/**
 * Warnings with a column and the same code, severity and kind of fix are grouped when there
 * are at least `MIN_GROUP` of them; the others stay single. Groups take the place of their
 * first warning, so the order (most serious first) is kept.
 */
export function groupWarnings(warnings: DataWarning[], profile: DataProfile | null): WarningItem[] {
  const keyOf = (w: DataWarning) =>
    w.column === null ? null : `${w.code}|${w.severity}|${fixKey(w.fix)}`;
  const members = new Map<string, DataWarning[]>();
  for (const warning of warnings) {
    const key = keyOf(warning);
    if (key !== null) {
      members.set(key, [...(members.get(key) ?? []), warning]);
    }
  }
  const items: WarningItem[] = [];
  const done = new Set<string>();
  warnings.forEach((warning, i) => {
    const key = keyOf(warning);
    const group = key === null ? undefined : members.get(key);
    if (key === null || !group || group.length < MIN_GROUP) {
      items.push({ type: 'single', key: `${warning.code}:${warning.column ?? ''}:${i}`, warning });
      return;
    }
    if (done.has(key)) {
      return;
    }
    done.add(key);
    const columns = group.map((w) => w.column!);
    items.push({
      type: 'group',
      key,
      group: {
        key,
        code: warning.code,
        severity: warning.severity,
        title: groupTitle(warning.code, warning.severity, columns, profile),
        columns,
        warnings: group,
        fix: groupFix(group),
      },
    });
  });
  return items;
}
