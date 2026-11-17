import { act, axeViolations, render, screen, userEvent, waitFor, within } from '@test-utils';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, vi } from 'vitest';
import { api } from '@/api';
import { i18n } from '@/i18n';
import { NO_SCOPE } from '@/training/access';
import { detail, DOC, summary } from '@/training/fixtures';
import { HistoryView } from '@/training/HistoryView';
import { RunView } from '@/training/RunView';
import { curve, row, table } from './fixtures';
import { POLL_MS, RunsPage } from './RunsPage';

vi.mock('@/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api')>();
  return {
    ...actual,
    api: {
      training: {
        runs: {
          table: vi.fn(),
          curves: vi.fn(),
          annotate: vi.fn(),
          columns: vi.fn(),
          setColumns: vi.fn(),
          get: vi.fn(),
        },
        comparisons: { list: vi.fn(), create: vi.fn(), remove: vi.fn() },
      },
    },
  };
});

const EXTERNAL = row({
  id: 'e1',
  name: 'On my GPU',
  kind: 'external',
  status: 'running',
  finished_at: null,
  model: 'Script',
  dataset: { key: 'external:iris.csv', kind: 'external', name: 'iris.csv' },
  external: { client: 'brickdnn-tracking/1', points: 7, reported_at: '2026-11-17T10:00:00Z' },
});
const HERE = row({ id: 'h1', name: 'Trained here' });

function renderPage(path = '/runs') {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <RunsPage />
    </MemoryRouter>
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(api.training.runs.table).mockResolvedValue(table([EXTERNAL, HERE]));
  vi.mocked(api.training.runs.columns).mockResolvedValue({ columns: null });
  vi.mocked(api.training.runs.curves).mockImplementation(async (ids) => ({
    runs: ids.map((id) => curve(id, id === 'e1' ? 'On my GPU' : 'Trained here')),
    missing: [],
  }));
});

afterEach(() => {
  vi.useRealTimers();
});

it('marks runs that ran elsewhere and filters by them', async () => {
  renderPage();
  const grid = await screen.findByRole('table', { name: 'Runs' });
  const external = within(grid).getAllByRole('row')[1];
  expect(within(external).getByText('ran elsewhere')).toBeInTheDocument();
  expect(within(external).getByText('Script')).toBeInTheDocument();
  expect(within(within(grid).getAllByRole('row')[2]).queryByText('ran elsewhere')).toBeNull();
  await userEvent.click(screen.getByRole('combobox', { name: 'Type' }));
  await userEvent.click(await screen.findByRole('option', { name: 'Ran elsewhere' }));
  await waitFor(() =>
    expect(api.training.runs.table).toHaveBeenLastCalledWith(
      expect.objectContaining({ family: 'external' }),
      expect.anything()
    )
  );
});

it('compares a run from elsewhere with one trained here, and follows it while it runs', async () => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  renderPage();
  await screen.findByRole('table', { name: 'Runs' });
  const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
  await user.click(screen.getByRole('checkbox', { name: 'Compare “On my GPU”' }));
  await user.click(screen.getByRole('checkbox', { name: 'Compare “Trained here”' }));
  await waitFor(() =>
    expect(api.training.runs.curves).toHaveBeenCalledWith(['e1', 'h1'], expect.anything())
  );
  expect(await screen.findByRole('slider', { name: /^Loss: use the arrow keys/ })).toHaveAttribute(
    'aria-valuetext',
    expect.stringContaining('On my GPU')
  );
  const tables = vi.mocked(api.training.runs.table).mock.calls.length;
  const curves = vi.mocked(api.training.runs.curves).mock.calls.length;
  // A run is active: the table and the curves are read again every few seconds.
  await act(async () => {
    await vi.advanceTimersByTimeAsync(POLL_MS + 100);
  });
  await waitFor(() =>
    expect(vi.mocked(api.training.runs.table).mock.calls.length).toBeGreaterThan(tables)
  );
  expect(vi.mocked(api.training.runs.curves).mock.calls.length).toBeGreaterThan(curves);
});

it('stops polling when nothing is active', async () => {
  vi.mocked(api.training.runs.table).mockResolvedValue(table([HERE]));
  vi.useFakeTimers({ shouldAdvanceTime: true });
  renderPage();
  await screen.findByRole('table', { name: 'Runs' });
  const calls = vi.mocked(api.training.runs.table).mock.calls.length;
  await act(async () => {
    await vi.advanceTimersByTimeAsync(POLL_MS * 3);
  });
  expect(vi.mocked(api.training.runs.table).mock.calls.length).toBe(calls);
});

it('shows the badge in the history too', async () => {
  render(
    <HistoryView
      runs={[
        {
          ...summary(),
          id: 'e2',
          name: 'Laptop run',
          kind: 'external',
          external: { client: null, points: 3, reported_at: null },
        },
      ]}
      error={null}
      onRetry={() => undefined}
      onOpen={() => undefined}
      onDeleted={() => undefined}
      onNewTraining={() => undefined}
      scope={NO_SCOPE}
      filter="mine"
      onFilterChange={() => undefined}
    />
  );
  expect(screen.getByText('ran elsewhere')).toBeInTheDocument();
  expect(screen.getByRole('link', { name: 'Compare runs' })).toHaveAttribute(
    'href',
    '/runs?mine=true'
  );
});

it('finds nothing serious with axe in German', async () => {
  await i18n.changeLanguage('de');
  renderPage();
  await screen.findByRole('table', { name: 'Läufe' });
  expect(screen.getByText('anderswo trainiert')).toBeInTheDocument();
  expect(await axeViolations()).toEqual([]);
});

it('shows a run from elsewhere with its curves, but no weights or script', () => {
  const base = detail();
  render(
    <RunView
      run={{
        ...base,
        kind: 'external',
        graph: { version: 1, nodes: [], edges: [] },
        has_weights: false,
        external: { client: 'brickdnn-tracking/1', points: 3, reported_at: null },
        state: base.state && { ...base.state, weights: false },
      }}
      doc={DOC}
      onBack={vi.fn()}
      onNewTraining={vi.fn()}
      onUpdate={vi.fn()}
      onDeleted={vi.fn()}
      onLoadNetwork={vi.fn()}
    />
  );
  expect(screen.getByText('ran elsewhere')).toBeInTheDocument();
  expect(screen.getByText(/a script logged 3 points of its curves here/)).toBeInTheDocument();
  expect(screen.queryByRole('button', { name: /Script \(.py\)|script/i })).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: /Weights/ })).not.toBeInTheDocument();
  expect(screen.getAllByRole('slider').length).toBeGreaterThan(0); // the curves
});
