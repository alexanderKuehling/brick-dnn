import { fireEvent, render, screen, within } from '@test-utils';
import type { DataProfile, DataSample } from '@/api/types';
import { PROFILE, RELATIONS } from '@/data/fixtures';
import { RelationshipsView } from './RelationshipsView';

beforeAll(() => {
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
});

const values = (f: (i: number) => number | null) => Array.from({ length: 40 }, (_, i) => f(i));

const SAMPLE_ROWS: DataSample = {
  n_total: 40,
  rows: 40,
  seed: 0,
  index: values((i) => i) as number[],
  split: values(() => 0).map(() => 'train' as const),
  columns: [
    { name: 'a', kind: 'numeric', values: values((i) => i) },
    { name: 'b', kind: 'numeric', values: values((i) => 2 * i + (i % 3)) },
    { name: 'c', kind: 'numeric', values: values((i) => -i + (i % 7)) },
    { name: 'noise1', kind: 'numeric', values: values((i) => (i * 17) % 11) },
    { name: 'noise2', kind: 'numeric', values: values((i) => (i * 13) % 7) },
  ],
};

const profile: DataProfile = { ...PROFILE, relations: RELATIONS };

function setup(overrides: Partial<DataProfile> = {}) {
  const handlers = { onAddStep: vi.fn(), onScatter: vi.fn(), onMethodChange: vi.fn() };
  render(
    <RelationshipsView
      profile={{ ...profile, ...overrides }}
      sample={SAMPLE_ROWS}
      target="churn"
      method="spearman"
      {...handlers}
    />
  );
  return handlers;
}

const table = () => screen.getByRole('table', { name: 'Columns and their strongest partners' });
const names = () =>
  within(table())
    .getAllByRole('row')
    .slice(1)
    .map((row) => within(row).getAllByRole('button')[0]?.textContent);

describe('RelationshipsView', () => {
  it('lists every column with its strongest partner, in the clustered order', () => {
    setup();
    expect(names()).toEqual(['noise1', 'a', 'b', 'c', 'noise2']);
    const row = within(table()).getAllByRole('row')[2];
    expect(row).toHaveTextContent('a');
    expect(row).toHaveTextContent('b');
    expect(row).toHaveTextContent('0.90');
    expect(screen.getByText(/5 number and yes\/no columns relate/)).toBeInTheDocument();
  });

  it('sorts by the strongest partner: columns without one come first', () => {
    setup();
    fireEvent.click(within(table()).getByRole('button', { name: 'ρ' }));
    expect(names()).toEqual(['noise1', 'noise2', 'c', 'a', 'b']);
    fireEvent.click(within(table()).getByRole('button', { name: 'ρ' }));
    expect(names()[0]).toBe('a');
  });

  it('counts the columns below the threshold and updates it live', () => {
    setup();
    expect(screen.getAllByText(/2 of 5 columns have no partner at \|ρ\| ≥ 0.15/)).not.toHaveLength(
      0
    );
    const slider = screen.getByRole('slider', { name: /threshold/ });
    fireEvent.keyDown(slider, { key: 'ArrowDown' });
    fireEvent.keyDown(slider, { key: 'ArrowDown' });
    fireEvent.keyDown(slider, { key: 'ArrowDown' });
    fireEvent.keyDown(slider, { key: 'ArrowDown' });
    fireEvent.keyDown(slider, { key: 'ArrowDown' });
    fireEvent.keyDown(slider, { key: 'ArrowDown' });
    fireEvent.keyDown(slider, { key: 'ArrowDown' });
    expect(slider).toHaveAttribute('aria-valuenow', '0.08');
    expect(slider).toHaveAttribute(
      'aria-valuetext',
      'Threshold |ρ| 0.08: 1 of 5 columns have no partner above it'
    );
    fireEvent.change(screen.getByRole('textbox', { name: 'Threshold |ρ|' }), {
      target: { value: '0.7' },
    });
    expect(slider).toHaveAttribute('aria-valuenow', '0.7');
    expect(screen.getByRole('button', { name: 'Show the 3 in the table' })).toBeInTheDocument();
  });

  it('filters the table to the columns below the threshold and drops them in one step', () => {
    const { onAddStep } = setup();
    fireEvent.click(screen.getByRole('button', { name: 'Show the 2 in the table' }));
    expect(names()).toEqual(['noise1', 'noise2']);
    fireEvent.click(screen.getByRole('button', { name: 'Drop the 2 without a partner' }));
    expect(onAddStep).toHaveBeenCalledWith({
      type: 'feature_selection',
      params: { method: 'no_partner', min_partner: 0.15, partner_method: 'spearman' },
    });
    expect(screen.getByRole('button', { name: 'Step added' })).toBeDisabled();
  });

  it('selects all columns matching the filter and drops or keeps them', () => {
    const { onAddStep } = setup();
    fireEvent.change(screen.getByRole('textbox', { name: 'Search columns' }), {
      target: { value: 'noise' },
    });
    fireEvent.click(screen.getByRole('checkbox', { name: 'Select all 2 matching' }));
    expect(screen.getByText('2 selected')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Drop selected (2)' }));
    expect(onAddStep).toHaveBeenLastCalledWith({
      type: 'drop_columns',
      params: { columns: ['noise1', 'noise2'] },
    });
    // keep only: drops the other number columns, the constant one included
    fireEvent.click(screen.getByRole('button', { name: 'Keep only selected (drop 4)' }));
    expect(onAddStep).toHaveBeenLastCalledWith({
      type: 'drop_columns',
      params: { columns: ['a', 'b', 'c', 'flat'] },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Clear' }));
    expect(screen.getByText('0 selected')).toBeInTheDocument();
  });

  it('filters by partner count and group', () => {
    setup();
    fireEvent.change(screen.getByRole('textbox', { name: /Partners at/ }), {
      target: { value: '2' },
    });
    expect(names()).toEqual(['a']);
  });

  it('shows the partners of a column and opens the scatter plot', () => {
    const { onScatter } = setup();
    fireEvent.click(within(table()).getByRole('button', { name: 'a' }));
    expect(screen.getByText('Partners of a')).toBeInTheDocument();
    expect(screen.getByText('a against b')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /^c: Spearman ρ −0.60/ }));
    expect(onScatter).toHaveBeenCalledWith('c', 'a');
  });

  it('says how many columns the heatmap shows and where the values come from', () => {
    setup();
    expect(
      screen.getByText(/Showing 5 of 5 number and yes\/no columns, related ones next to each other/)
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Computed on the 40 training rows of the preview sample/)
    ).toBeInTheDocument();
    // the overview blocks: a click shows them
    expect(screen.getByText('All columns, related ones together')).toBeInTheDocument();
  });

  it('falls back to the plain heatmap without relations', () => {
    setup({ relations: null });
    expect(screen.queryByRole('table')).not.toBeInTheDocument();
    expect(screen.getByText(/correlation$/)).toBeInTheDocument();
  });
});
