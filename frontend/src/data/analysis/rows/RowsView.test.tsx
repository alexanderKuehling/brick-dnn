import { act, fireEvent, render, screen, waitFor, within } from '@test-utils';
import type { RowAnalysis } from '@/api/types';
import { rowAnalysis, SIMILAR } from './fixtures';
import { RowsView, type RowsSource } from './RowsView';

beforeAll(() => {
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
});

function setup(analysis: RowAnalysis | Error = rowAnalysis()) {
  const source: RowsSource = {
    revision: 1,
    load: vi.fn(() =>
      analysis instanceof Error ? Promise.reject(analysis) : Promise.resolve(analysis)
    ),
    similar: vi.fn(() => Promise.resolve(SIMILAR)),
    onChangeSplit: vi.fn(),
  };
  const onAddStep = vi.fn();
  render(
    <RowsView
      source={source}
      numberColumns={['x0', 'x1']}
      columns={['id', 'x0', 'x1', 'age']}
      onAddStep={onAddStep}
    />
  );
  return { source, onAddStep };
}

const belowCounts = () => screen.getAllByTestId('below-count').map((el) => el.textContent);

describe('RowsView', () => {
  it('shows both histograms with the rows below their thresholds', async () => {
    setup();
    await waitFor(() => expect(screen.getAllByTestId('below-count')).toHaveLength(2));
    expect(belowCounts()).toEqual(['2 rows are below', '6 rows have a near twin']);
    expect(screen.getByTestId('feature-note')).toHaveTextContent(
      'Compared on 12 of 30 number columns: the ones with a strong partner'
    );
    // Typed threshold.
    const [spreadInput] = screen.getAllByLabelText(/^Threshold/);
    fireEvent.change(spreadInput, { target: { value: '0.7' } });
    expect(belowCounts()[0]).toBe('5 rows are below');
    // Arrow keys move the threshold of the NN ratio histogram.
    const slider = screen.getByRole('slider', { name: /NN ratio\)?: threshold/ });
    fireEvent.keyDown(slider, { key: 'ArrowLeft' });
    fireEvent.keyDown(slider, { key: 'PageDown' });
    expect(Number(slider.getAttribute('aria-valuenow'))).toBeLessThan(0.32);
    // Bars below the threshold are marked.
    expect(screen.getAllByTestId('bar-below').length).toBeGreaterThan(0);
  });

  it('adds the steps for atypical rows and groups', async () => {
    const { onAddStep, source } = setup();
    await screen.findByRole('button', { name: 'Remove 2 rows (step)' });
    fireEvent.click(screen.getByRole('button', { name: 'Remove 2 rows (step)' }));
    expect(onAddStep).toHaveBeenCalledWith({
      type: 'remove_atypical',
      params: { features: 'auto', columns: [], side: 'low', threshold: 0.25 },
    });
    fireEvent.click(screen.getByRole('button', { name: /Find groups & split by them/ }));
    expect(onAddStep).toHaveBeenLastCalledWith({
      type: 'find_groups',
      params: { features: 'auto', columns: [], threshold: 0.32, name: 'group' },
    });
    expect(source.onChangeSplit).toHaveBeenCalledWith({ group_by: 'group' });
  });

  it('sorts, filters and selects rows, then drops the selected ones', async () => {
    const { onAddStep } = setup();
    await waitFor(() => expect(screen.getAllByTestId('row-table-row')).toHaveLength(8));
    const firstRow = () => screen.getAllByTestId('row-table-row')[0];
    expect(within(firstRow()).getAllByRole('cell')[1]).toHaveTextContent('0');
    fireEvent.click(screen.getByRole('button', { name: 'Spread' }));
    expect(within(firstRow()).getAllByRole('cell')[1]).toHaveTextContent('4');
    fireEvent.click(screen.getByRole('button', { name: 'Spread' }));
    expect(within(firstRow()).getAllByRole('cell')[1]).toHaveTextContent('7');

    fireEvent.click(screen.getAllByRole('button', { name: 'Show them' })[0]);
    expect(screen.getAllByTestId('row-table-row')).toHaveLength(2);
    expect(screen.getByTestId('row-count')).toHaveTextContent('2 of 8 rows · 0 selected');
    fireEvent.click(screen.getByRole('button', { name: 'Select all 2 shown' }));
    expect(screen.getByTestId('row-count')).toHaveTextContent('2 selected');
    fireEvent.click(screen.getByRole('button', { name: 'Drop 2 selected rows (step)' }));
    expect(onAddStep).toHaveBeenCalledWith({
      type: 'drop_rows',
      params: { by: 'column', column: 'id', values: [104, 105] },
    });

    fireEvent.change(screen.getByLabelText('Row or id'), { target: { value: '107' } });
    fireEvent.click(screen.getAllByRole('button', { name: 'Show them' })[1]);
    expect(screen.queryAllByTestId('row-table-row')).toHaveLength(0);
    fireEvent.change(screen.getByLabelText('Row or id'), { target: { value: '' } });
    expect(screen.getAllByTestId('row-table-row')).toHaveLength(6);
    fireEvent.click(screen.getByRole('checkbox', { name: 'Select row 2' }));
    expect(screen.getByTestId('row-count')).toHaveTextContent('1 selected');
  });

  it('shows the similar rows of a clicked row and compares a pair', async () => {
    const { source } = setup();
    await waitFor(() => expect(screen.getAllByTestId('row-table-row')).toHaveLength(8));
    fireEvent.click(screen.getAllByTestId('row-table-row')[0]);
    const panel = await screen.findByLabelText('Similar rows');
    expect(source.similar).toHaveBeenLastCalledWith(
      expect.objectContaining({ row: 0, other: null, features: { mode: 'auto' } }),
      expect.anything()
    );
    await within(panel).findByText('x3');
    expect(within(panel).getByText('Rows most similar to Row 0 (id 100)')).toBeInTheDocument();
    expect(within(panel).getByText('x7')).toBeInTheDocument();
    expect(within(panel).getByText('Row 6 (id 106)')).toBeInTheDocument();
    fireEvent.click(within(panel).getByRole('button', { name: 'Compare' }));
    expect(source.similar).toHaveBeenLastCalledWith(
      expect.objectContaining({ row: 0, other: 6 }),
      expect.anything()
    );
  });

  it('compares the targets of close and random pairs', async () => {
    setup();
    const medians = await screen.findByTestId('pair-medians');
    expect(medians).toHaveTextContent('Close pairs: median difference 1.5');
    expect(medians).toHaveTextContent('Random pairs: median difference 11.5');
    expect(screen.getByText(/This view uses the labels \(age\)/)).toBeInTheDocument();
    expect(screen.getByText('Rows on the two main directions (PCA)')).toBeInTheDocument();
  });

  it('notes sampling and reports errors', async () => {
    setup(rowAnalysis({ sampled: true, n_total: 50000, analysed: 8 }));
    expect(await screen.findByText(/A random sample of 8 of 50,000 rows/)).toBeInTheDocument();
  });

  it('offers to try again after an error', async () => {
    const { source } = setup(new Error('Too slow'));
    expect(await screen.findByText('Too slow')).toBeInTheDocument();
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Try again' }));
    });
    expect(source.load).toHaveBeenCalledTimes(2);
  });
});
