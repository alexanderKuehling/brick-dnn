import { fireEvent, render, screen, userEvent, within } from '@test-utils';
import { vi } from 'vitest';
import type { DataSample } from '@/api/types';
import { PROFILE, SAMPLE } from '../fixtures';
import { DataTable } from './DataTable';

function setup(overrides: Partial<React.ComponentProps<typeof DataTable>> = {}) {
  const props = {
    sample: SAMPLE,
    profile: PROFILE,
    target: 'churn',
    onAddStep: vi.fn(),
    onSetTarget: vi.fn(),
    onShowColumn: vi.fn(),
    ...overrides,
  };
  render(<DataTable {...props} />);
  return props;
}

const header = (name: string) =>
  screen
    .getAllByRole('columnheader')
    .find((th) => th.querySelector('button')?.textContent?.startsWith(name))!;

async function openMenu(column: string) {
  await userEvent.click(screen.getByRole('button', { name: `Actions for column ${column}` }));
}

describe('DataTable', () => {
  it('shows the sample with a caption, headers and formatted cells', () => {
    setup();
    expect(screen.getByText('Showing 60 random rows of 1,200 · 9 columns')).toBeInTheDocument();
    const table = screen.getByRole('table', { name: 'Data sample' });
    expect(table).toHaveAttribute('aria-rowcount', '61');
    expect(table).toHaveAttribute('aria-colcount', '11');

    // Missing share from the profile (whole table), unique count and top values.
    expect(within(header('age')).getByText('10 % missing')).toBeInTheDocument();
    expect(within(header('plan')).getByText('complete')).toBeInTheDocument();
    expect(within(header('city')).getByText(/· 5 unique/)).toBeInTheDocument();
    expect(within(header('city')).getByText(/Berlin 24 %/)).toBeInTheDocument();
    expect(within(header('is_member')).getByText('40 % true')).toBeInTheDocument();
    expect(within(header('churn')).getByText('Target')).toBeInTheDocument();

    // Row numbers (1-based), split badges, numbers right-aligned, ISO dates, missing cells.
    const rows = within(table).getAllByRole('row');
    const first = rows[1];
    expect(within(first).getByRole('rowheader')).toHaveTextContent('1');
    expect(within(first).getByText('train')).toBeInTheDocument();
    expect(within(first).getByText('1,000')).toHaveAttribute('data-align', 'right');
    const signup = SAMPLE.columns.find((c) => c.name === 'signup')!.values[0] as number;
    expect(
      within(first).getByText(new Date(signup).toISOString().slice(0, 10))
    ).toBeInTheDocument();
    expect(screen.getAllByText('missing').length).toBeGreaterThan(0);
    expect(screen.getByText('950,000')).toBeInTheDocument();
  });

  it('renders only a window of a large sample', () => {
    const rows = 5000;
    const sample: DataSample = {
      n_total: 1_000_000,
      rows,
      seed: 0,
      index: Array.from({ length: rows }, (_, i) => i * 200),
      split: null,
      columns: [
        { name: 'x', kind: 'numeric', values: Array.from({ length: rows }, (_, i) => i) },
        { name: 'label', kind: 'categorical', values: Array.from({ length: rows }, () => 'a') },
      ],
    };
    setup({ sample, profile: null, target: null });
    expect(
      screen.getByText('Showing 5,000 random rows of 1,000,000 · 2 columns')
    ).toBeInTheDocument();
    const rendered = screen.getAllByRole('row').length;
    expect(rendered).toBeGreaterThan(10);
    expect(rendered).toBeLessThan(100);
    expect(screen.queryByText('4,999')).not.toBeInTheDocument();

    const region = screen.getByRole('region');
    fireEvent.scroll(region, { target: { scrollTop: 4990 * 30 } });
    expect(screen.getByText('4,999')).toBeInTheDocument();
    expect(screen.queryByText('1,000')).not.toBeInTheDocument();
  });

  it('sorts the sample by a column', async () => {
    setup();
    const sortButton = within(header('customer_id')).getAllByRole('button')[0];
    await userEvent.click(sortButton);
    await userEvent.click(sortButton);
    expect(header('customer_id')).toHaveAttribute('aria-sort', 'descending');
    const firstRow = screen.getAllByRole('row')[1];
    expect(within(firstRow).getByText('1,059')).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: /Clear sorting/ }));
    expect(within(screen.getAllByRole('row')[1]).getByText('1,000')).toBeInTheDocument();
  });

  it('sets the target and opens a column in the analysis', async () => {
    const props = setup();
    await openMenu('age');
    await userEvent.click(screen.getByRole('menuitem', { name: 'Use as target' }));
    expect(props.onSetTarget).toHaveBeenCalledWith('age');

    await openMenu('age');
    await userEvent.click(screen.getByRole('menuitem', { name: 'Show in Analysis' }));
    expect(props.onShowColumn).toHaveBeenCalledWith('age');

    await openMenu('churn');
    expect(screen.getByRole('menuitem', { name: 'This is the target' })).toBeDisabled();
  });

  it('adds steps from the column menu with prefilled params', async () => {
    const props = setup();
    await openMenu('age');
    await userEvent.click(
      screen.getByRole('menuitem', { name: 'Fill missing values with the median' })
    );
    expect(props.onAddStep).toHaveBeenLastCalledWith({
      type: 'impute',
      params: { columns: ['age'], strategy: 'median' },
    });

    await openMenu('income');
    await userEvent.click(
      screen.getByRole('menuitem', { name: 'Clip to the 1st–99th percentile' })
    );
    expect(props.onAddStep).toHaveBeenLastCalledWith({
      type: 'clip',
      params: { columns: ['income'], mode: 'quantiles', lower_q: 0.01, upper_q: 0.99 },
    });

    await openMenu('income');
    await userEvent.click(screen.getByRole('menuitem', { name: 'Log transform (log1p)' }));
    expect(props.onAddStep).toHaveBeenLastCalledWith({
      type: 'transform',
      params: { columns: ['income'], function: 'log1p' },
    });

    await openMenu('city');
    await userEvent.click(screen.getByRole('menuitem', { name: 'One-hot encode' }));
    expect(props.onAddStep).toHaveBeenLastCalledWith({
      type: 'encode',
      params: { columns: ['city'], method: 'onehot' },
    });

    await openMenu('signup');
    expect(screen.queryByRole('menuitem', { name: /Fill missing/ })).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('menuitem', { name: 'Split into date parts' }));
    expect(props.onAddStep).toHaveBeenLastCalledWith({
      type: 'datetime_features',
      params: { columns: ['signup'] },
    });

    await openMenu('notes');
    await userEvent.click(screen.getByRole('menuitem', { name: 'Drop column' }));
    expect(props.onAddStep).toHaveBeenLastCalledWith({
      type: 'drop_columns',
      params: { columns: ['notes'] },
    });
  });

  it('offers only target-safe actions for the target column', async () => {
    const profile = {
      ...PROFILE,
      columns: PROFILE.columns.map((c) => (c.name === 'churn' ? { ...c, missing: 12 } : c)),
    };
    const props = setup({ profile });
    await openMenu('churn');
    const menu = screen.getByRole('menu');
    expect(within(menu).getByText(/^Target column/)).toBeInTheDocument();
    expect(within(menu).getByRole('menuitem', { name: 'This is the target' })).toBeDisabled();
    expect(
      within(menu)
        .getAllByRole('menuitem')
        .map((item) => item.textContent)
    ).toEqual([
      'This is the target',
      'Show in Analysis',
      'Drop rows without a target value',
      'Rename…',
      'Change type…',
    ]);
    await userEvent.click(
      within(menu).getByRole('menuitem', { name: 'Drop rows without a target value' })
    );
    expect(props.onAddStep).toHaveBeenLastCalledWith({
      type: 'drop_missing',
      params: { columns: ['churn'], how: 'any' },
    });
  });

  it('wraps the toolbar on narrow screens', () => {
    setup();
    const goTo = screen.getByRole('combobox', { name: 'Go to column' });
    const group = goTo.closest('.mantine-Group-root') as HTMLElement;
    expect(group.style.getPropertyValue('--group-wrap')).toBe('wrap');
  });

  it('renames a column in a dialog', async () => {
    const props = setup();
    await openMenu('age');
    await userEvent.click(screen.getByRole('menuitem', { name: 'Rename…' }));
    const dialog = screen.getByRole('dialog', { name: 'Rename “age”' });
    const input = within(dialog).getByLabelText('New name');
    await userEvent.clear(input);
    await userEvent.type(input, 'city');
    expect(within(dialog).getByText('There is already a column “city”.')).toBeInTheDocument();
    expect(within(dialog).getByRole('button', { name: 'Rename' })).toBeDisabled();
    await userEvent.clear(input);
    await userEvent.type(input, 'age_years{Enter}');
    expect(props.onAddStep).toHaveBeenCalledWith({
      type: 'rename_columns',
      params: { mapping: { age: 'age_years' } },
    });
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });

  it('changes the type of a column in a dialog', async () => {
    const props = setup();
    await openMenu('plan');
    await userEvent.click(screen.getByRole('menuitem', { name: 'Change type…' }));
    const dialog = screen.getByRole('dialog', { name: 'Change the type of “plan”' });
    expect(within(dialog).queryByRole('radio', { name: 'Categorical' })).not.toBeInTheDocument();
    await userEvent.click(within(dialog).getByRole('radio', { name: 'Text' }));
    await userEvent.click(within(dialog).getByRole('button', { name: 'Change type' }));
    expect(props.onAddStep).toHaveBeenCalledWith({
      type: 'cast',
      params: { columns: ['plan'], to: 'text' },
    });
  });

  it('works without a profile and without a split', () => {
    setup({ profile: null, sample: { ...SAMPLE, split: null } });
    expect(screen.queryByRole('columnheader', { name: 'Split' })).not.toBeInTheDocument();
    // Missing share from the sample.
    expect(within(header('notes')).getByText(/% missing/)).toBeInTheDocument();
  });

  it('says when no rows are left', () => {
    setup({
      sample: { ...SAMPLE, rows: 0, n_total: 0, index: [], split: [], columns: [] },
      profile: null,
    });
    expect(screen.getByText(/No rows left/)).toBeInTheDocument();
  });
});
