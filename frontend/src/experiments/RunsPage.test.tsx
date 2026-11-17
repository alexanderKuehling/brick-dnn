import { axeViolations, render, screen, userEvent, waitFor, within } from '@test-utils';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, vi } from 'vitest';
import { api } from '@/api';
import { i18n } from '@/i18n';
import { curve, ROWS, table } from './fixtures';
import { RunsPage } from './RunsPage';

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

function renderPage(path = '/runs') {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <RunsPage />
    </MemoryRouter>
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(api.training.runs.table).mockResolvedValue(table());
  vi.mocked(api.training.runs.columns).mockResolvedValue({ columns: null });
  vi.mocked(api.training.runs.setColumns).mockImplementation(async (columns) => ({ columns }));
  vi.mocked(api.training.runs.curves).mockImplementation(async (ids) => ({
    runs: ids.map((id) =>
      curve(id, ROWS.find((r) => r.id === id)!.name, {
        loss: id === 'c3' ? 'focal' : 'cross_entropy',
      })
    ),
    missing: [],
  }));
  vi.mocked(api.training.comparisons.list).mockResolvedValue([]);
});

async function tableRows() {
  const grid = await screen.findByRole('table', { name: 'Runs' });
  return within(grid).getAllByRole('row').slice(1);
}

it('lists the runs with their settings and results, and sorts on the server', async () => {
  renderPage('/runs?project=3');
  const rows = await tableRows();
  expect(rows).toHaveLength(4);
  expect(api.training.runs.table).toHaveBeenCalledWith(
    expect.objectContaining({ project: 3, page_size: 50 }),
    expect.anything()
  );
  const first = within(rows[0]);
  expect(first.getByText('Adam 1e-3')).toBeInTheDocument();
  expect(first.getByText('90.0%')).toBeInTheDocument(); // validation accuracy
  expect(first.getByText('0.001')).toBeInTheDocument(); // learning rate
  expect(within(rows[3]).getByText('Search · 8 trials')).toBeInTheDocument();
  expect(screen.getByRole('columnheader', { name: /F1 \(macro\) · val\./ })).toBeInTheDocument();

  await userEvent.click(screen.getByRole('button', { name: 'Sort by F1 (macro) · val.' }));
  await waitFor(() =>
    expect(api.training.runs.table).toHaveBeenLastCalledWith(
      expect.objectContaining({ sort: 'metric:val_macro_f1', order: 'asc' }),
      expect.anything()
    )
  );
});

it('filters by tag and starred runs, and searches', async () => {
  renderPage();
  await tableRows();
  await userEvent.click(screen.getByRole('checkbox', { name: 'Starred only' }));
  await waitFor(() =>
    expect(api.training.runs.table).toHaveBeenLastCalledWith(
      expect.objectContaining({ starred: true }),
      expect.anything()
    )
  );
  await userEvent.click(screen.getByRole('combobox', { name: 'Tag' }));
  await userEvent.click(await screen.findByRole('option', { name: 'baseline (1)' }));
  await waitFor(() =>
    expect(api.training.runs.table).toHaveBeenLastCalledWith(
      expect.objectContaining({ tag: 'baseline', starred: true }),
      expect.anything()
    )
  );
  await userEvent.type(screen.getByRole('textbox', { name: 'Search' }), 'sgd');
  await waitFor(() =>
    expect(api.training.runs.table).toHaveBeenLastCalledWith(
      expect.objectContaining({ q: 'sgd' }),
      expect.anything()
    )
  );
});

it('remembers the chosen columns on the account', async () => {
  renderPage();
  await tableRows();
  await userEvent.click(screen.getByRole('button', { name: 'Columns' }));
  await userEvent.click(await screen.findByRole('checkbox', { name: 'Started by' }));
  await waitFor(() => expect(api.training.runs.setColumns).toHaveBeenCalled());
  const saved = vi.mocked(api.training.runs.setColumns).mock.calls[0][0]!;
  expect(saved).toContain('started_by');
  expect(saved).toContain('metric:val_macro_f1');
  expect(await screen.findByRole('columnheader', { name: 'Started by' })).toBeInTheDocument();
  await userEvent.click(screen.getByRole('checkbox', { name: 'Status' }));
  await waitFor(() =>
    expect(screen.queryByRole('columnheader', { name: /Status/ })).not.toBeInTheDocument()
  );
  await userEvent.click(screen.getByRole('button', { name: 'Back to the default columns' }));
  expect(api.training.runs.setColumns).toHaveBeenLastCalledWith(null);
});

it('starts with the columns the user chose before', async () => {
  vi.mocked(api.training.runs.columns).mockResolvedValue({
    columns: ['dataset', 'metric:val_loss'],
  });
  renderPage();
  await screen.findByRole('columnheader', { name: /Dataset/ });
  await waitFor(() =>
    expect(screen.queryByRole('columnheader', { name: /Status/ })).not.toBeInTheDocument()
  );
  expect(screen.getAllByRole('columnheader')).toHaveLength(5); // compare, star, name + 2
});

it('overlays the curves of the selected runs and warns about different losses', async () => {
  renderPage();
  await tableRows();
  expect(screen.getByText(/Select runs in the table/)).toBeInTheDocument();
  await userEvent.click(screen.getByRole('checkbox', { name: 'Compare “Adam 1e-3”' }));
  await userEvent.click(screen.getByRole('checkbox', { name: 'Compare “SGD 1e-2”' }));
  await waitFor(() =>
    expect(api.training.runs.curves).toHaveBeenLastCalledWith(['a1', 'b2'], expect.anything())
  );
  const loss = await screen.findByRole('slider', { name: /^Loss: use the arrow keys/ });
  expect(loss).toHaveAttribute('aria-valuetext', 'Epoch 5: Adam 1e-3 0.3, SGD 1e-2 0.3');
  expect(screen.getAllByRole('list', { name: 'Legend' })[0]).toHaveTextContent(
    /Adam 1e-3.*SGD 1e-2/
  );
  expect(screen.queryByText('The losses differ in kind')).not.toBeInTheDocument();

  loss.focus();
  await userEvent.keyboard('{Home}');
  expect(loss).toHaveAttribute('aria-valuetext', 'Epoch 1: Adam 1e-3 1.1, SGD 1e-2 1.1');

  await userEvent.click(screen.getByRole('checkbox', { name: 'Compare “Focal”' }));
  expect(await screen.findByText('The losses differ in kind')).toBeInTheDocument();
  expect(screen.getByText(/Cross-entropy, Focal loss/)).toBeInTheDocument();

  // The time axis and smoothing.
  await userEvent.click(screen.getAllByRole('radio', { name: 'Time' })[0]);
  expect(
    await screen.findByRole('slider', {
      name: /^Loss: use the arrow keys to read the values per Time/,
    })
  ).toHaveAttribute('aria-valuetext', expect.stringMatching(/^Time 1:00:/));
  await userEvent.click(screen.getAllByRole('switch', { name: 'Smooth' })[0]);
  expect(screen.getByText(/Smoothed with a moving average/)).toBeInTheDocument();
});

it('compares at most 10 runs', async () => {
  const many = Array.from({ length: 11 }, (_, i) => ({
    ...ROWS[0],
    id: `r${i}`,
    name: `Run ${i}`,
  }));
  vi.mocked(api.training.runs.table).mockResolvedValue(table(many));
  vi.mocked(api.training.runs.curves).mockResolvedValue({ runs: [], missing: [] });
  renderPage();
  await tableRows();
  for (let i = 0; i < 10; i += 1) {
    await userEvent.click(screen.getByRole('checkbox', { name: `Compare “Run ${i}”` }));
  }
  expect(screen.getByRole('checkbox', { name: 'Compare “Run 10”' })).toBeDisabled();
  expect(screen.getByText(/10 of 10 selected/)).toBeInTheDocument();
});

it('plots a setting against a result, and parallel coordinates', async () => {
  renderPage();
  await tableRows();
  for (const name of ['Adam 1e-3', 'SGD 1e-2', 'Focal']) {
    await userEvent.click(screen.getByRole('checkbox', { name: `Compare “${name}”` }));
  }
  await userEvent.click(screen.getByRole('tab', { name: 'Settings vs. result' }));
  expect(await screen.findByText('Loss · val. against Learning rate')).toBeInTheDocument();
  expect(
    screen.getByText(/3 runs with both values; the horizontal axis is logarithmic/)
  ).toBeInTheDocument();
  await userEvent.click(screen.getByRole('tab', { name: 'Parallel coordinates' }));
  const parallel = await screen.findByRole('slider', {
    name: /Settings and results: use the arrow keys/,
  });
  parallel.focus();
  expect(parallel).toHaveAttribute(
    'aria-valuetext',
    'Adam 1e-3: Learning rate 0.001, Batch size 64, Epochs 10, Loss · val. 0.4'
  );
  await userEvent.keyboard('{ArrowDown}');
  expect(parallel).toHaveAttribute(
    'aria-valuetext',
    expect.stringMatching(/^SGD 1e-2: Learning rate 0.01/)
  );
});

it('stars a run and edits its tags and note; runs of others cannot be annotated', async () => {
  vi.mocked(api.training.runs.annotate).mockImplementation(async (id, body) => ({
    id,
    tags: body.tags ?? [],
    note: body.note ?? '',
    starred: body.starred ?? false,
  }));
  renderPage();
  await tableRows();
  await userEvent.click(screen.getByRole('button', { name: 'Star “Adam 1e-3”' }));
  expect(api.training.runs.annotate).toHaveBeenCalledWith('a1', { starred: true });
  await waitFor(() =>
    expect(screen.getByRole('button', { name: 'Star “Adam 1e-3”' })).toHaveAttribute(
      'aria-pressed',
      'true'
    )
  );
  expect(screen.queryByRole('button', { name: 'Star “Focal”' })).not.toBeInTheDocument();
  expect(
    screen.queryByRole('button', { name: 'Tags and note of “Focal”' })
  ).not.toBeInTheDocument();

  await userEvent.click(screen.getByRole('button', { name: 'Tags and note of “SGD 1e-2”' }));
  const dialog = await screen.findByRole('dialog', { name: 'Tags and note of “SGD 1e-2”' });
  await userEvent.type(
    within(dialog)
      .getAllByLabelText('Tags')
      .find((el) => el.tagName === 'INPUT' && (el as HTMLInputElement).type !== 'hidden')!,
    'noisy{Enter}'
  );
  const note = within(dialog).getByRole('textbox', { name: 'Note' });
  await userEvent.clear(note);
  await userEvent.type(note, 'Lower the rate.');
  await userEvent.click(within(dialog).getByRole('button', { name: 'Save' }));
  expect(api.training.runs.annotate).toHaveBeenLastCalledWith('b2', {
    tags: ['noisy'],
    note: 'Lower the rate.',
    starred: true,
  });
});

it('shares the selected runs with a read-only link', async () => {
  vi.mocked(api.training.comparisons.create).mockResolvedValue({
    id: 5,
    name: 'LR sweep',
    token: 'tok-123456',
    runs: ['a1', 'b2'],
    columns: [],
    options: {},
    include_notes: false,
    created_at: '2026-11-17T10:00:00Z',
  });
  renderPage();
  await tableRows();
  expect(screen.getByRole('button', { name: 'Share comparison…' })).toBeDisabled();
  await userEvent.click(screen.getByRole('checkbox', { name: 'Compare “Adam 1e-3”' }));
  await userEvent.click(screen.getByRole('checkbox', { name: 'Compare “SGD 1e-2”' }));
  await userEvent.click(screen.getByRole('button', { name: 'Share comparison…' }));
  const dialog = await screen.findByRole('dialog', { name: 'Share a comparison' });
  expect(within(dialog).getByText(/does not show who started the runs/)).toBeInTheDocument();
  await userEvent.type(within(dialog).getByRole('textbox', { name: 'Name' }), 'LR sweep');
  await userEvent.click(within(dialog).getByRole('button', { name: 'Create the link' }));
  expect(api.training.comparisons.create).toHaveBeenCalledWith(
    expect.objectContaining({ name: 'LR sweep', runs: ['a1', 'b2'], include_notes: false })
  );
  expect(
    await within(dialog).findByRole('textbox', { name: 'Link to the comparison' })
  ).toHaveValue(`${window.location.origin}/compare/tok-123456`);
});

it('expands the trials of a search', async () => {
  vi.mocked(api.training.runs.get).mockResolvedValue({
    ...ROWS[3],
    tuning: { method: 'random', trials: 8, space: { learning_rate: { min: 0.0001, max: 0.01 } } },
    state: { trials: [], trials_total: 8, best_trial: null },
  } as never);
  renderPage();
  await tableRows();
  const toggle = screen.getByRole('button', { name: 'Trials of “Search”' });
  expect(toggle).toHaveAttribute('aria-expanded', 'false');
  await userEvent.click(toggle);
  expect(toggle).toHaveAttribute('aria-expanded', 'true');
  await waitFor(() => expect(api.training.runs.get).toHaveBeenCalledWith('d4', expect.anything()));
});

describe.each(['en', 'de'] as const)('in %s', (language) => {
  beforeEach(async () => {
    await i18n.changeLanguage(language);
  });

  it('finds nothing serious with axe, with runs compared', async () => {
    renderPage();
    await screen.findByRole('table', { name: language === 'de' ? 'Läufe' : 'Runs' });
    const name = language === 'de' ? '„Adam 1e-3“ vergleichen' : 'Compare “Adam 1e-3”';
    await userEvent.click(screen.getByRole('checkbox', { name }));
    await screen.findAllByRole('slider');
    expect(await axeViolations()).toEqual([]);
    if (language === 'de') {
      expect(screen.getByRole('heading', { name: 'Läufe' })).toBeInTheDocument();
      expect(screen.getByRole('columnheader', { name: /Lernrate/ })).toBeInTheDocument();
      expect(screen.getAllByText(/^90,0\s%$/).length).toBeGreaterThan(0);
    }
  });
});
