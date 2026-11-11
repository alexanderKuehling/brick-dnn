import { fireEvent, render, screen } from '@test-utils';
import type { DataProfile, DataWarning } from '@/api/types';
import { PROFILE } from '@/data/fixtures';
import { warningHeading } from './OverviewView';
import { groupWarnings } from './warningGroups';
import { WarningList } from './WarningList';

const N = 30;
const names = Array.from({ length: N }, (_, i) => `x${i}`);

const missing = (name: string): DataWarning => ({
  code: 'high_missing',
  severity: 'info',
  column: name,
  message: `${name} is missing in 8 % of the rows.`,
  fix: {
    type: 'impute',
    params: { columns: [name], strategy: 'median' },
    label: `Impute ${name} (median)`,
  },
});
const outliers = (name: string): DataWarning => ({
  code: 'outliers',
  severity: 'info',
  column: name,
  message: `${name} has outliers.`,
  fix: {
    type: 'clip',
    params: { columns: [name], mode: 'quantiles', lower_q: 0.01, upper_q: 0.99 },
    label: `Clip ${name} to the 1st–99th percentile`,
  },
});
const constant: DataWarning = {
  code: 'constant',
  severity: 'warning',
  column: 'flat',
  message: 'flat has the same value in every row (1).',
  fix: { type: 'drop_columns', params: { columns: ['flat'] }, label: 'Drop flat' },
};

const profile: DataProfile = {
  ...PROFILE,
  n_rows: 1000,
  columns: names.map((name, i) => ({
    ...PROFILE.columns[0],
    name,
    count: 1000 - (60 + i),
    missing: 60 + i,
    stats: { ...PROFILE.columns[0].stats!, iqr_outliers: 20 + i },
  })),
  warnings: [constant, ...names.map(missing), ...names.slice(0, 5).map(outliers)],
};

describe('grouped warnings', () => {
  it('groups warnings of one kind about many columns, with ranges', () => {
    const items = groupWarnings(profile.warnings, profile);
    expect(items).toHaveLength(3);
    expect(items[0]).toMatchObject({ type: 'single' });
    const [, missingGroup, outlierGroup] = items;
    if (missingGroup.type !== 'group' || outlierGroup.type !== 'group') {
      throw new Error('expected groups');
    }
    expect(missingGroup.group.title).toBe('Missing values in 30 columns (6–8.9% of the rows)');
    expect(missingGroup.group.fix).toEqual({
      type: 'impute',
      params: { columns: names, strategy: 'median' },
      label: 'Impute 30 columns (median)',
    });
    expect(outlierGroup.group.title).toBe(
      'Outliers by the IQR rule in 5 columns (2.1–2.6% of the values)'
    );
    expect(warningHeading(profile)).toBe('Warnings: 3 entries, 36 in all');
  });

  it('keeps few warnings and different fixes apart', () => {
    const skew = (name: string, fn: string): DataWarning => ({
      code: 'skewed',
      severity: 'warning',
      column: name,
      message: `${name} is skewed`,
      fix: { type: 'transform', params: { columns: [name], function: fn }, label: `T ${name}` },
    });
    const items = groupWarnings(
      [skew('a', 'log1p'), skew('b', 'log1p'), skew('c', 'yeo_johnson'), skew('d', 'log1p')],
      null
    );
    expect(items.map((i) => i.type)).toEqual(['group', 'single']);
    expect(warningHeading({ ...profile, warnings: [constant] })).toBe('Warnings (1)');
  });

  it('shows a group with one fix and expands to its columns', () => {
    const onAddStep = vi.fn();
    const onShowColumn = vi.fn();
    render(
      <WarningList
        warnings={profile.warnings}
        profile={profile}
        resetKey={1}
        onAddStep={onAddStep}
        onShowColumn={onShowColumn}
      />
    );
    expect(
      screen.getByText('Missing values in 30 columns (6–8.9% of the rows)')
    ).toBeInTheDocument();
    expect(screen.queryByText('x3 is missing in 8 % of the rows.')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Impute 30 columns (median)' }));
    expect(onAddStep).toHaveBeenCalledWith({
      type: 'impute',
      params: { columns: names, strategy: 'median' },
    });
    fireEvent.click(screen.getByRole('button', { name: /Show the 30 columns/ }));
    expect(screen.getByText('x3 is missing in 8 % of the rows.')).toBeInTheDocument();
    expect(screen.queryByText('x25 is missing in 8 % of the rows.')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Show all 30' }));
    expect(screen.getByText('x25 is missing in 8 % of the rows.')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'x25' }));
    expect(onShowColumn).toHaveBeenCalledWith('x25');
  });
});
