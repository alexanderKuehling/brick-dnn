/**
 * The columns of the runs table: what the run is, its main settings and its headline numbers
 * (backend/brick_dnn_core/experiments.py). A column id is `status`, `setting:<key>` or
 * `metric:<key>`; the server sorts by the same id.
 */
import type { RunRow } from '@/api';
import { i18n } from '@/i18n';
import {
  formatDate,
  formatDuration,
  formatMetric,
  formatPercent,
  formatSetting,
  LOSS_NAMES,
  SETTING_LABELS,
  STATUS_LABELS,
} from '@/training/format';

export type ColumnGroup = 'run' | 'settings' | 'metrics';

export interface Column {
  id: string;
  group: ColumnGroup;
  label: string;
  /** The sort key of the server; none: not sortable. */
  sort?: string;
  numeric: boolean;
  /** The value for charts (numbers) and filters (words); null: none. */
  value: (row: RunRow) => number | string | null;
  /** The text of the cell. */
  text: (row: RunRow) => string;
}

/** Columns of the run itself (the name is always shown first). */
const RUN_COLUMNS = [
  'status',
  'started',
  'duration',
  'dataset',
  'model',
  'network',
  'started_by',
  'tags',
] as const;

/** Settings in the order they are offered. */
export const SETTING_KEYS = [
  'optimizer',
  'learning_rate',
  'batch_size',
  'epochs',
  'weight_decay',
  'loss',
  'val_split',
  'test_split',
  'patience',
  'seed',
  'max_samples',
  'label_smoothing',
  'focal_gamma',
  'huber_delta',
  'feature_training',
  'method',
  'search',
  'trials',
] as const;

/** Metrics with a name of their own, in the order they are offered. */
export const KNOWN_METRICS = [
  'accuracy',
  'balanced_accuracy',
  'macro_f1',
  'weighted_f1',
  'roc_auc',
  'pr_auc',
  'log_loss',
  'loss',
  'rmse',
  'mae',
  'mse',
  'r2',
  'mape',
  'median_absolute_error',
  'max_error',
  'iou',
  'dice',
  'pixel_accuracy',
] as const;

/** Shares between 0 and 1, shown as percentages. */
const PERCENT_METRICS = new Set([
  'accuracy',
  'balanced_accuracy',
  'macro_f1',
  'weighted_f1',
  'roc_auc',
  'pr_auc',
  'iou',
  'dice',
  'pixel_accuracy',
  'mape',
]);
/** Smaller is better (for the scatter plot's reading direction). */
const LOWER_IS_BETTER = new Set([
  'loss',
  'log_loss',
  'rmse',
  'mae',
  'mse',
  'mape',
  'median_absolute_error',
  'max_error',
]);

export const SPLITS = ['train', 'val', 'test'] as const;
export type Split = (typeof SPLITS)[number];

/** `val_macro_f1` → `{split: "val", base: "macro_f1"}`; unknown names keep their base. */
export function splitMetric(key: string): { split: Split | null; base: string } {
  const match = /^(train|val|test)_(.+)$/.exec(key);
  return match ? { split: match[1] as Split, base: match[2] } : { split: null, base: key };
}

function hasKey<T extends string>(list: readonly T[], value: string): value is T {
  return (list as readonly string[]).includes(value);
}

/** "F1 (macro) · val." – a known metric in the app's language, others as they are named. */
export function metricLabel(key: string): string {
  if (key === 'best_epoch') {
    return i18n.t('experiments.metric.best_epoch');
  }
  const { split, base } = splitMetric(key);
  const name = hasKey(KNOWN_METRICS, base) ? i18n.t(`experiments.metric.${base}`) : base;
  return split
    ? i18n.t('experiments.metricOf', { metric: name, split: i18n.t(`experiments.split.${split}`) })
    : name;
}

export function isPercentMetric(key: string): boolean {
  return PERCENT_METRICS.has(splitMetric(key).base);
}

export function lowerIsBetter(key: string): boolean {
  return LOWER_IS_BETTER.has(splitMetric(key).base);
}

export function formatMetricOf(key: string, value: number | null | undefined): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return '–';
  }
  if (key === 'best_epoch') {
    return String(value);
  }
  if (splitMetric(key).base === 'r2') {
    return value.toFixed(3);
  }
  return isPercentMetric(key) ? formatPercent(value) : formatMetric(value);
}

export function settingLabel(key: string): string {
  if (key in SETTING_LABELS) {
    return SETTING_LABELS[key as keyof typeof SETTING_LABELS];
  }
  return hasKey(SETTING_KEYS, key) ? i18n.t(`experiments.setting.${key}`) : key;
}

export function formatSettingOf(key: string, value: number | string | undefined): string {
  if (value === undefined) {
    return '–';
  }
  if (key === 'loss' && typeof value === 'string') {
    return LOSS_NAMES[value] ?? value;
  }
  if (key === 'val_split' || key === 'test_split') {
    return typeof value === 'number' ? formatPercent(value, 0) : String(value);
  }
  if (key === 'search') {
    return i18n.t(`experiments.searchMethod.${value === 'grid' ? 'grid' : 'random'}`);
  }
  return formatSetting(key as Parameters<typeof formatSetting>[0], value);
}

function runColumn(id: (typeof RUN_COLUMNS)[number]): Column {
  const label = i18n.t(`experiments.column.${id}`);
  switch (id) {
    case 'status':
      return {
        id,
        group: 'run',
        label,
        sort: 'status',
        numeric: false,
        value: (row) => row.status,
        text: (row) => STATUS_LABELS[row.status],
      };
    case 'started':
      return {
        id,
        group: 'run',
        label,
        sort: 'created_at',
        numeric: false,
        value: (row) => Date.parse(row.created_at),
        text: (row) => formatDate(row.created_at),
      };
    case 'duration':
      return {
        id,
        group: 'run',
        label,
        sort: 'duration',
        numeric: true,
        value: (row) => row.duration_seconds,
        text: (row) => formatDuration(row.duration_seconds),
      };
    case 'dataset':
      return {
        id,
        group: 'run',
        label,
        sort: 'dataset',
        numeric: false,
        value: (row) => row.dataset.name,
        text: (row) => row.dataset.name ?? '–',
      };
    case 'model':
      return {
        id,
        group: 'run',
        label,
        sort: 'model',
        numeric: false,
        value: (row) => row.model,
        text: (row) => modelText(row),
      };
    case 'network':
      return {
        id,
        group: 'run',
        label,
        numeric: false,
        value: (row) => row.project_name ?? null,
        text: (row) => row.project_name ?? '–',
      };
    case 'started_by':
      return {
        id,
        group: 'run',
        label,
        numeric: false,
        value: (row) => row.started_by ?? null,
        text: (row) => row.started_by ?? '–',
      };
    case 'tags':
      return {
        id,
        group: 'run',
        label,
        numeric: false,
        value: (row) => (row.tags ?? []).join(', ') || null,
        text: (row) => (row.tags ?? []).join(', ') || '–',
      };
  }
}

/** The model of a run in the app's language where Brick DNN names it. */
export function modelText(row: RunRow): string {
  if (row.family === 'ensemble') {
    return i18n.t('experiments.family.ensemble');
  }
  if (row.model === 'Script') {
    return i18n.t('experiments.elsewhere.script');
  }
  if (row.model === 'Neural network') {
    return i18n.t('experiments.family.neural');
  }
  return row.model;
}

export function settingColumn(key: string): Column {
  return {
    id: `setting:${key}`,
    group: 'settings',
    label: settingLabel(key),
    sort: `setting:${key}`,
    numeric: !['optimizer', 'loss', 'feature_training', 'method', 'search'].includes(key),
    value: (row) => row.settings[key as keyof RunRow['settings']] ?? null,
    text: (row) => formatSettingOf(key, row.settings[key as keyof RunRow['settings']]),
  };
}

export function metricColumn(key: string): Column {
  return {
    id: `metric:${key}`,
    group: 'metrics',
    label: metricLabel(key),
    sort: `metric:${key}`,
    numeric: true,
    value: (row) => row.metrics[key] ?? null,
    text: (row) => formatMetricOf(key, row.metrics[key]),
  };
}

/** The order of metric keys: known metrics by split (val, test, train), others after. */
export function metricOrder(a: string, b: string): number {
  const rank = (key: string) => {
    if (key === 'best_epoch') {
      return 10_000;
    }
    const { split, base } = splitMetric(key);
    const known = (KNOWN_METRICS as readonly string[]).indexOf(base);
    const splitRank = split === 'val' ? 0 : split === 'test' ? 1 : split === 'train' ? 2 : 3;
    return (known < 0 ? 100 : known) * 4 + splitRank;
  };
  return rank(a) - rank(b) || a.localeCompare(b);
}

/** Every column the rows have values for (and the chosen ones), in their groups. */
export function availableColumns(rows: RunRow[], chosen: string[] = []): Column[] {
  const settings = new Set<string>();
  const metrics = new Set<string>();
  for (const row of rows) {
    Object.keys(row.settings).forEach((key) => settings.add(key));
    Object.keys(row.metrics).forEach((key) => metrics.add(key));
  }
  for (const id of chosen) {
    if (id.startsWith('setting:')) {
      settings.add(id.slice(8));
    } else if (id.startsWith('metric:')) {
      metrics.add(id.slice(7));
    }
  }
  const settingKeys = [...settings].sort((a, b) => {
    const rank = (key: string) => {
      const index = (SETTING_KEYS as readonly string[]).indexOf(key);
      return index < 0 ? 1000 : index;
    };
    return rank(a) - rank(b) || a.localeCompare(b);
  });
  return [
    ...RUN_COLUMNS.map(runColumn),
    ...settingKeys.map(settingColumn),
    ...[...metrics].sort(metricOrder).map(metricColumn),
  ];
}

export function columnById(id: string): Column | null {
  if ((RUN_COLUMNS as readonly string[]).includes(id)) {
    return runColumn(id as (typeof RUN_COLUMNS)[number]);
  }
  if (id.startsWith('setting:')) {
    return settingColumn(id.slice(8));
  }
  if (id.startsWith('metric:')) {
    return metricColumn(id.slice(7));
  }
  return null;
}

/** Shown until the user chooses: the run, its main settings and the main metrics. */
export const DEFAULT_COLUMNS = [
  'status',
  'started',
  'duration',
  'dataset',
  'model',
  'setting:optimizer',
  'setting:learning_rate',
  'setting:batch_size',
  'setting:epochs',
  'setting:loss',
  'metric:val_loss',
  'metric:val_accuracy',
  'metric:val_macro_f1',
  'metric:val_pr_auc',
  'metric:test_accuracy',
  'metric:val_rmse',
  'metric:val_r2',
  'metric:test_rmse',
  'metric:val_iou',
  'metric:test_iou',
];

/**
 * The columns to show: the user's choice, or the defaults that the rows have values for
 * (metrics of other tasks stay out of the way).
 */
export function shownColumns(rows: RunRow[], chosen: string[] | null): Column[] {
  if (chosen) {
    return chosen.map(columnById).filter((c): c is Column => c !== null);
  }
  return DEFAULT_COLUMNS.map(columnById)
    .filter((c): c is Column => c !== null)
    .filter(
      (column) =>
        column.group !== 'metrics' ||
        rows.some((row) => column.value(row) !== null && column.value(row) !== undefined)
    );
}
