import { fireEvent, render, screen, within } from '@test-utils';
import type { ColumnProfile, DataProfile, DataSample } from '@/api/types';
import { PROFILE, RELATIONS, SAMPLE } from '@/data/fixtures';
import { AnalysisPanel, type AnalysisPanelProps } from './AnalysisPanel';

beforeAll(() => {
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
});

function setup(props: Partial<AnalysisPanelProps> = {}) {
  const handlers = {
    onRequestCompare: vi.fn(),
    onAddStep: vi.fn(),
    onColumnChange: vi.fn(),
  };
  const view = render(
    <AnalysisPanel
      profile={PROFILE}
      before={null}
      sample={SAMPLE}
      target="churn"
      {...handlers}
      {...props}
    />
  );
  return { ...handlers, ...view };
}

const openView = (label: string) => fireEvent.click(screen.getByRole('radio', { name: label }));

describe('AnalysisPanel', () => {
  it('starts with the overview: figures, column types and warnings', () => {
    setup();
    // ("Rows" is also the name of a view.)
    const rowsTile = screen.getAllByText('Rows').find((el) => !el.closest('label'));
    expect(rowsTile?.nextSibling).toHaveTextContent('1,200');
    expect(screen.getByText('840 / 180 / 180')).toBeInTheDocument();
    expect(
      screen.getByRole('img', { name: /Columns per type: Number 3 \(33%\)/ })
    ).toBeInTheDocument();
    expect(screen.getByText('Warnings (4)')).toBeInTheDocument();
    expect(screen.getByText(/income is strongly skewed/)).toBeInTheDocument();
    expect(
      screen.getByText(
        /Charts that split rows by class: based on a random sample of 60 of 1,200 rows/
      )
    ).toBeInTheDocument();
  });

  it('adds the fix of a warning as a step', () => {
    const { onAddStep } = setup();
    fireEvent.click(screen.getByRole('button', { name: 'Clip income to the 1st–99th percentile' }));
    expect(onAddStep).toHaveBeenCalledWith({
      type: 'clip',
      params: { columns: ['income'], mode: 'quantiles', lower_q: 0.01, upper_q: 0.99 },
    });
    expect(screen.getByRole('button', { name: 'Step added' })).toBeDisabled();
  });

  it('lists, searches and sorts the columns; a click opens the distribution', () => {
    const { onColumnChange } = setup();
    openView('Columns');
    const table = screen.getByRole('table');
    expect(within(table).getAllByRole('row')).toHaveLength(PROFILE.columns.length + 1);
    fireEvent.click(within(table).getByRole('button', { name: 'Missing' }));
    expect(within(table).getAllByRole('row')[1]).toHaveTextContent('notes');
    expect(within(table).getByRole('columnheader', { name: /Missing/ })).toHaveAttribute(
      'aria-sort',
      'descending'
    );
    fireEvent.change(screen.getByRole('textbox', { name: 'Search columns' }), {
      target: { value: 'inc' },
    });
    expect(within(table).getAllByRole('row')).toHaveLength(2);
    fireEvent.click(within(table).getByRole('button', { name: 'income' }));
    expect(onColumnChange).toHaveBeenCalledWith('income');
    expect(screen.getByText('income by churn')).toBeInTheDocument();
  });

  it('shows a numeric distribution split by the target, with quantiles', () => {
    setup({ column: 'income', view: 'distribution' });
    expect(screen.getByText('income by churn')).toBeInTheDocument();
    // One bin holds almost all rows (a few huge incomes), so the view zooms in by itself.
    const zoom = screen.getByRole('switch', { name: 'Zoom to 1st–99th percentile' });
    expect(zoom).toBeChecked();
    expect(
      screen.getByText(
        /Share of each class's rows per bin between 22,600 and 68,900 \(1st–99th percentile; 4 of 59 sample values outside\)\. Based on a random sample of 60 of 1,200 rows\./
      )
    ).toBeInTheDocument();
    fireEvent.click(zoom);
    expect(
      screen.getByText(/^Share of each class's rows per bin\. Based on a random sample of 60/)
    ).toBeInTheDocument();
    expect(screen.getByText('Box plots by class')).toBeInTheDocument();
    const quantiles = screen.getByRole('table', { name: 'Quantiles of income' });
    expect(within(quantiles).getByText('38,400')).toBeInTheDocument();
    expect(screen.getByRole('table', { name: 'Statistics of income' })).toHaveTextContent(
      'Skewness18.4'
    );
    // The column's warning comes with its fix.
    expect(
      screen.getByRole('button', { name: 'Clip income to the 1st–99th percentile' })
    ).toBeInTheDocument();
    fireEvent.click(screen.getByRole('switch', { name: 'Split by churn' }));
    expect(screen.getByText('Distribution of income')).toBeInTheDocument();
    expect(screen.getByText('Rows per bin, all 1,200 rows.')).toBeInTheDocument();
  });

  it('shows categories, dates and text by kind', () => {
    const { rerender } = render(
      <AnalysisPanel {...baseProps()} column="city" view="distribution" />
    );
    expect(screen.getByText('city by churn')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('switch', { name: 'Split by churn' }));
    expect(screen.getByText('Values of city')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Show Values of city as table' }));
    const table = screen.getByRole('table', { name: 'Values of city (table)' });
    expect(within(table).getByText('Berlin')).toBeInTheDocument();
    expect(within(table).getByText('Missing')).toBeInTheDocument();

    // The split setting is kept for the next column.
    rerender(<AnalysisPanel {...baseProps()} column="signup" view="distribution" />);
    expect(screen.getByText('signup over time')).toBeInTheDocument();
    expect(screen.getByText('By weekday')).toBeInTheDocument();
    expect(screen.getByRole('table', { name: 'Statistics of signup' })).toHaveTextContent(
      'First2021-01-01'
    );
    fireEvent.click(screen.getByRole('switch', { name: 'Split by churn' }));
    expect(screen.getByText('signup by churn')).toBeInTheDocument();

    rerender(<AnalysisPanel {...baseProps()} column="notes" view="distribution" />);
    expect(screen.getByText('notes by churn')).toBeInTheDocument();
    expect(screen.getByText('Text length')).toBeInTheDocument();
  });

  it('shows the time of day of dates that have one', () => {
    const sample: DataSample = {
      ...SAMPLE,
      columns: SAMPLE.columns.map((c) =>
        c.name === 'signup' ? { ...c, values: c.values.map((v) => (v as number) + 30_600_000) } : c
      ),
    };
    const profile: DataProfile = {
      ...PROFILE,
      columns: PROFILE.columns.map((c) =>
        c.name === 'signup' ? { ...c, stats: { ...c.stats!, min: Date.UTC(2021, 0, 1, 8, 30) } } : c
      ),
    };
    render(
      <AnalysisPanel
        {...baseProps()}
        sample={sample}
        profile={profile}
        column="signup"
        view="distribution"
      />
    );
    expect(screen.getByRole('table', { name: 'Statistics of signup' })).toHaveTextContent(
      'First2021-01-01 08:30:00'
    );
  });

  it('shows missing values with fill actions', () => {
    const { onAddStep } = setup({ view: 'missing' });
    expect(screen.getByText('Missing values per column')).toBeInTheDocument();
    expect(screen.getByText('Where values are missing')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Fill 2 number columns (median)' }));
    expect(onAddStep).toHaveBeenCalledWith({
      type: 'impute',
      params: { columns: ['age', 'income'], strategy: 'median' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Fill 1 category column (most frequent)' }));
    expect(onAddStep).toHaveBeenLastCalledWith({
      type: 'impute',
      params: { columns: ['city'], strategy: 'most_frequent' },
    });
  });

  it('shows correlations; a pair opens the scatter plot', () => {
    setup({ view: undefined });
    openView('Relationships');
    expect(screen.getByText('Spearman correlation')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('radio', { name: 'Pearson' }));
    expect(screen.getByText('Pearson correlation')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('radio', { name: 'Spearman' }));
    const pair = screen.getByRole('button', {
      name: /(age and income|income and age): Spearman ρ 0\.51/,
    });
    fireEvent.click(pair);
    expect(screen.getByText(/^(income vs age|age vs income)$/)).toBeInTheDocument();
    expect(screen.getByRole('radio', { name: 'Scatter' })).toBeChecked();
  });

  it('explores the relationships of the columns and acts on them', () => {
    const { onAddStep } = setup({ profile: { ...PROFILE, relations: RELATIONS } });
    openView('Relationships');
    expect(screen.getByText('Strongest partner of each column')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Drop the 2 without a partner' }));
    expect(onAddStep).toHaveBeenCalledWith({
      type: 'feature_selection',
      params: { method: 'no_partner', min_partner: 0.15, partner_method: 'spearman' },
    });
  });

  it('plots two columns of the sample against each other', () => {
    setup({ view: 'scatter' });
    // The strongest pair by default, coloured by the target.
    expect(screen.getByText('income vs age')).toBeInTheDocument();
    expect(screen.getByRole('combobox', { name: 'Colour' })).toHaveValue('churn');
    expect(screen.getByText(/of 60 sample rows have both values/)).toBeInTheDocument();
  });

  it('shows the class balance and the feature ranking', () => {
    setup({ view: 'target' });
    expect(screen.getByText('Classes of churn')).toBeInTheDocument();
    expect(screen.getByText('Majority baseline').nextSibling).toHaveTextContent('73%');
    expect(screen.getByText('2.7 : 1')).toBeInTheDocument();
    expect(screen.getByText('Class mix per split')).toBeInTheDocument();
    expect(screen.getByText('Features by association with the target')).toBeInTheDocument();
  });

  it('drops the features weakly associated with the target', () => {
    const { onAddStep } = setup({ view: 'target' });
    expect(
      screen.getByText(/1 of \d+ features are below 0\.05: customer_id\./)
    ).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Add a step: drop below this' }));
    expect(onAddStep).toHaveBeenLastCalledWith({
      type: 'feature_selection',
      params: { method: 'target_correlation', min_association: 0.05 },
    });
    expect(screen.getByRole('button', { name: 'Step added' })).toBeDisabled();

    fireEvent.change(screen.getByRole('textbox', { name: 'Drop below association' }), {
      target: { value: '0.1' },
    });
    expect(screen.getByText(/2 of \d+ features are below 0\.10/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Drop these 2 columns' }));
    expect(onAddStep).toHaveBeenLastCalledWith({
      type: 'drop_columns',
      params: { columns: ['city', 'customer_id'] },
    });
  });

  it('splits by a yes/no target into its two classes', () => {
    const { sample, profile } = booleanTarget();
    const props = { ...baseProps(), sample, profile };
    const legendLabels = () =>
      within(screen.getByRole('list', { name: 'Legend' }))
        .getAllByRole('listitem')
        .map((item) => item.textContent);
    const { rerender } = render(<AnalysisPanel {...props} column="income" view="distribution" />);
    expect(screen.getByText('income by churn')).toBeInTheDocument();
    expect(legendLabels()).toEqual(['false', 'true']);

    rerender(<AnalysisPanel {...props} column="city" view="distribution" />);
    expect(legendLabels()).toEqual(['false', 'true']);

    // Scatter: every point has its class colour, none is "Other" or missing.
    rerender(<AnalysisPanel {...props} view="scatter" />);
    expect(screen.getByRole('combobox', { name: 'Colour' })).toHaveValue('churn');
    expect(legendLabels()).toEqual(['false', 'true']);

    rerender(<AnalysisPanel {...props} view="target" />);
    expect(screen.getByText('Class mix per split')).toBeInTheDocument();
    expect(legendLabels()).toEqual(['false', 'true']);
    const mix = screen.getByRole('button', { name: 'Show Class mix per split as table' });
    fireEvent.click(mix);
    const table = screen.getByRole('table', { name: 'Class mix per split (table)' });
    expect(
      within(table)
        .getAllByRole('columnheader')
        .map((th) => th.textContent)
    ).toEqual(['Category', 'false', 'true', 'Total']);
  });

  it('asks the page to explain a missing target', () => {
    setup({ view: 'target', target: null, profile: { ...PROFILE, target: null } });
    expect(screen.getByText(/Choose a target column/)).toBeInTheDocument();
  });

  it('requests the profile before the steps and compares a column', () => {
    const { onRequestCompare, rerender } = setup({ view: 'compare', column: 'income' });
    expect(onRequestCompare).toHaveBeenCalledTimes(1);
    expect(screen.getByRole('status')).toHaveTextContent(
      'Profiling the data before the cleaning steps…'
    );

    const before: DataProfile = {
      ...PROFILE,
      n_columns: 10,
      columns: [...PROFILE.columns, { ...PROFILE.columns[1], name: 'legacy_id' }],
    };
    const after: DataProfile = {
      ...PROFILE,
      columns: PROFILE.columns.map((c) =>
        c.name === 'income'
          ? {
              ...c,
              stats: { ...c.stats!, max: 68_900, iqr_outliers: 0 },
              histogram: { edges: [22_100, 45_500, 68_900], counts: [700, 440] },
            }
          : c
      ),
    };
    rerender(
      <AnalysisPanel
        {...baseProps()}
        profile={after}
        before={before}
        view="compare"
        column="income"
      />
    );
    expect(
      screen.getByRole('table', { name: 'Data before and after the steps' })
    ).toHaveTextContent('Columns109−1');
    expect(screen.getByText('Removed (1):')).toBeInTheDocument();
    expect(screen.getByText('income before and after')).toBeInTheDocument();
    expect(screen.getByText(/counts spread evenly within the original bins/)).toBeInTheDocument();
    const stats = screen.getByRole('table', { name: 'income before and after' });
    expect(stats).toHaveTextContent('Max950,00068,900−881,100');
    expect(onRequestCompare).toHaveBeenCalledTimes(1);
  });

  it('opens the distribution of a column focused from outside', () => {
    const { rerender } = render(<AnalysisPanel {...baseProps()} column={null} />);
    expect(screen.getByText('Warnings (4)')).toBeInTheDocument();
    rerender(<AnalysisPanel {...baseProps()} column="age" />);
    expect(screen.getByText('age by churn')).toBeInTheDocument();
  });

  it('stays usable with many columns and a large sample', { timeout: 20_000 }, () => {
    const many: ColumnProfile[] = Array.from({ length: 250 }, (_, i) => ({
      ...PROFILE.columns[1],
      name: `feature_${i}`,
    }));
    const profile: DataProfile = {
      ...PROFILE,
      n_columns: 250,
      columns: many,
      correlations: [],
      target: null,
      warnings: [],
    };
    const rows = 20_000;
    const values = Array.from({ length: rows }, (_, i) => (i % 97) * 1.5);
    const sample: DataSample = {
      n_total: 1_234_567,
      rows,
      seed: 0,
      index: values.map((_, i) => i * 61),
      split: null,
      columns: many.slice(0, 3).map((c, k) => ({
        name: c.name,
        kind: 'numeric' as const,
        values: values.map((v, i) => (i % 50 === k ? null : v + k)),
      })),
    };
    const props = { ...baseProps(), profile, sample, target: null };
    const { rerender } = render(<AnalysisPanel {...props} view="columns" />);
    expect(screen.getAllByRole('row')).toHaveLength(101);
    fireEvent.click(screen.getByRole('button', { name: 'Show 100 more of 150' }));
    expect(screen.getAllByRole('row')).toHaveLength(201);
    rerender(<AnalysisPanel {...props} view="scatter" />);
    expect(
      screen.getByText(
        '19,200 of 20,000 sample rows have both values. Based on a random sample of 20,000 of 1,234,567 rows.'
      )
    ).toBeInTheDocument();
  });
});

/** The fixtures with a yes/no target: 0/1 in the sample, classes "false"/"true" in the profile. */
function booleanTarget(): { sample: DataSample; profile: DataProfile } {
  const sample: DataSample = {
    ...SAMPLE,
    columns: SAMPLE.columns.map((c) =>
      c.name === 'churn'
        ? { name: 'churn', kind: 'boolean', values: c.values.map((v) => (v === 'yes' ? 1 : 0)) }
        : c
    ),
  };
  const member = PROFILE.columns.find((c) => c.name === 'is_member')!;
  const profile: DataProfile = {
    ...PROFILE,
    columns: PROFILE.columns.map((c) =>
      c.name === 'churn'
        ? {
            ...member,
            name: 'churn',
            top: [
              { value: '0', count: 876 },
              { value: '1', count: 324 },
            ],
            target_association: null,
          }
        : c
    ),
    target: {
      ...PROFILE.target!,
      classes: [
        { value: 'false', count: 876 },
        { value: 'true', count: 324 },
      ],
    },
  };
  return { sample, profile };
}

function baseProps(): AnalysisPanelProps {
  return {
    profile: PROFILE,
    before: null,
    sample: SAMPLE,
    target: 'churn',
    onRequestCompare: () => {},
    onAddStep: () => {},
  };
}
