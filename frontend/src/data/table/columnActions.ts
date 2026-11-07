/**
 * Quick actions of the column menu in the data table: each one adds a pipeline step with
 * prefilled params (the param names of the step catalog, see PLAN 4.5).
 */
import type { ColumnInfo, ColumnKind, ColumnProfile, NewStep } from '@/api/types';

export type ColumnActionKey =
  'impute' | 'clip' | 'log' | 'onehot' | 'date_parts' | 'drop' | 'drop_rows';

export interface ColumnStepAction {
  key: ColumnActionKey;
  label: string;
  step: NewStep;
}

/**
 * The step actions that make sense for a column, in menu order. `missing` is the number of
 * missing values (whole table if known, else in the sample). The target is not a feature:
 * encoding, filling, transforming or dropping it would change or remove what is predicted, so
 * it only offers to drop the rows without a target value.
 */
export function columnStepActions(
  column: ColumnInfo,
  profile: ColumnProfile | undefined,
  missing: number,
  isTarget = false
): ColumnStepAction[] {
  const columns = [column.name];
  if (isTarget) {
    return missing > 0
      ? [
          {
            key: 'drop_rows',
            label: 'Drop rows without a target value',
            step: { type: 'drop_missing', params: { columns, how: 'any' } },
          },
        ]
      : [];
  }
  const actions: ColumnStepAction[] = [];
  if (missing > 0) {
    actions.push(
      column.kind === 'numeric'
        ? {
            key: 'impute',
            label: 'Fill missing values with the median',
            step: { type: 'impute', params: { columns, strategy: 'median' } },
          }
        : {
            key: 'impute',
            label: 'Fill missing values with the most frequent value',
            step: { type: 'impute', params: { columns, strategy: 'most_frequent' } },
          }
    );
  }
  if (column.kind === 'numeric') {
    actions.push({
      key: 'clip',
      label: 'Clip to the 1st–99th percentile',
      step: {
        type: 'clip',
        params: { columns, mode: 'quantiles', lower_q: 0.01, upper_q: 0.99 },
      },
    });
    const min = profile?.stats?.min;
    actions.push(
      typeof min === 'number' && min < 0
        ? {
            key: 'log',
            label: 'Power transform (Yeo-Johnson)',
            step: { type: 'transform', params: { columns, function: 'yeo_johnson' } },
          }
        : {
            key: 'log',
            label: 'Log transform (log1p)',
            step: { type: 'transform', params: { columns, function: 'log1p' } },
          }
    );
  }
  if (column.kind === 'categorical') {
    actions.push({
      key: 'onehot',
      label: 'One-hot encode',
      step: { type: 'encode', params: { columns, method: 'onehot' } },
    });
  }
  if (column.kind === 'datetime') {
    actions.push({
      key: 'date_parts',
      label: 'Split into date parts',
      step: { type: 'datetime_features', params: { columns } },
    });
  }
  actions.push({
    key: 'drop',
    label: 'Drop column',
    step: { type: 'drop_columns', params: { columns } },
  });
  return actions;
}

export function renameStep(column: string, name: string): NewStep {
  return { type: 'rename_columns', params: { mapping: { [column]: name.trim() } } };
}

export function castStep(column: string, kind: ColumnKind): NewStep {
  return { type: 'cast', params: { columns: [column], to: kind } };
}

/** Why a new column name cannot be used, or null if it can. */
export function renameError(column: string, name: string, existing: string[]): string | null {
  const trimmed = name.trim();
  if (!trimmed) {
    return 'Enter a name.';
  }
  if (trimmed === column) {
    return 'That is the current name.';
  }
  if (existing.includes(trimmed)) {
    return `There is already a column “${trimmed}”.`;
  }
  return null;
}
