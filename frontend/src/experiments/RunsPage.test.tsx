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

/** `count` runs "Run 0", "Run 1", … (ids r0, r1, …), each with its own learning rate. */
function manyRuns(count: number, from = 0) {
  return Array.from({ length: count }, (_, i) => ({
    ...ROWS[0],
    id: `r${from + i}`,
    name: `Run ${from + i}`,
    settings: { ...ROWS[0].settings, learning_rate: 0.001 * (from + i + 1) },
  }));
}

it('selects and deselects all runs of the page with the checkbox in the header', async () => {
  renderPage();
  await tableRows();
  const all = screen.getByRole('checkbox', { name: 'Select all runs on this page' });
  const runs = () => screen.getAllByRole('checkbox', { name: /^Compare “/ });
  expect(all).not.toBeChecked();
  expect(all).not.toHaveAttribute('aria-checked');
  expect(screen.queryByText(/selected/)).not.toBeInTheDocument();

  // Some of them: indeterminate.
  await userEvent.click(screen.getByRole('checkbox', { name: 'Compare “SGD 1e-2”' }));
  expect(all).toBePartiallyChecked();
  expect(all).toHaveAttribute('aria-checked', 'mixed');
  expect(screen.getByText('4 runs · 1 selected')).toBeInTheDocument();

  // All of them, with the keyboard; the run selected before stays the first.
  all.focus();
  await userEvent.keyboard(' ');
  expect(all).toBeChecked();
  expect(all).not.toBePartiallyChecked();
  expect(all).not.toHaveAttribute('aria-checked');
  runs().forEach((box) => expect(box).toBeChecked());
  expect(screen.getByText('4 runs · 4 selected')).toBeInTheDocument();
  await waitFor(() =>
    expect(api.training.runs.curves).toHaveBeenLastCalledWith(
      ['b2', 'a1', 'c3', 'd4'],
      expect.anything()
    )
  );
  expect(screen.queryByText(/The charts show the first/)).not.toBeInTheDocument();

  // One less: indeterminate again; the header then selects the rest.
  await userEvent.click(screen.getByRole('checkbox', { name: 'Compare “Focal”' }));
  expect(all).toBePartiallyChecked();
  await userEvent.click(all);
  expect(all).toBeChecked();
  expect(screen.getByText('4 runs · 4 selected')).toBeInTheDocument();

  // None of them.
  await userEvent.click(all);
  expect(all).not.toBeChecked();
  runs().forEach((box) => expect(box).not.toBeChecked());
  expect(screen.queryByText(/selected/)).not.toBeInTheDocument();
  expect(screen.getByText(/Select runs in the table/)).toBeInTheDocument();

  // "Clear" next to the count.
  await userEvent.click(all);
  const clear = screen.getByRole('button', { name: 'Clear the selection' });
  expect(clear).toHaveTextContent('Clear');
  await userEvent.click(clear);
  expect(all).not.toBeChecked();
  expect(screen.queryByText(/selected/)).not.toBeInTheDocument();
});

it('keeps the runs selected on another page', async () => {
  vi.mocked(api.training.runs.table).mockImplementation(async (query) =>
    query?.page === 2
      ? table(manyRuns(2, 3), { count: 5, page: 2, pages: 2 })
      : table(manyRuns(3), { count: 5, pages: 2 })
  );
  renderPage();
  await tableRows();
  await userEvent.click(screen.getByRole('checkbox', { name: 'Select all runs on this page' }));
  expect(screen.getByText('5 runs · 3 selected')).toBeInTheDocument();

  await userEvent.click(screen.getByRole('button', { name: 'Page 2' }));
  await screen.findByRole('checkbox', { name: 'Compare “Run 4”' });
  const all = screen.getByRole('checkbox', { name: 'Select all runs on this page' });
  expect(all).not.toBeChecked();
  expect(all).not.toBePartiallyChecked();
  await userEvent.click(all);
  expect(screen.getByText('5 runs · 5 selected')).toBeInTheDocument();
  await userEvent.click(all);
  expect(screen.getByText('5 runs · 3 selected')).toBeInTheDocument();
  await waitFor(() =>
    expect(api.training.runs.curves).toHaveBeenLastCalledWith(['r0', 'r1', 'r2'], expect.anything())
  );
});

it('draws the curves of the first 10 selected runs, the other charts take all', async () => {
  vi.mocked(api.training.runs.table).mockResolvedValue(table(manyRuns(12)));
  vi.mocked(api.training.runs.curves).mockImplementation(async (ids) => ({
    runs: ids.map((id) => curve(id, `Run ${id.slice(1)}`)),
    missing: [],
  }));
  vi.mocked(api.training.comparisons.create).mockResolvedValue({
    id: 6,
    name: 'Sweep',
    token: 'tok-abcdef',
    runs: [],
    columns: [],
    options: {},
    include_notes: false,
    created_at: '2026-11-17T10:00:00Z',
  });
  renderPage();
  await tableRows();
  const all = screen.getByRole('checkbox', { name: 'Select all runs on this page' });
  await userEvent.click(all);
  expect(screen.getByText('12 runs · 12 selected')).toBeInTheDocument();
  screen
    .getAllByRole('checkbox', { name: /^Compare “/ })
    .forEach((box) => expect(box).toBeEnabled());

  // Curves: the first ten, and one line that says so.
  const firstTen = Array.from({ length: 10 }, (_, i) => `r${i}`);
  await waitFor(() =>
    expect(api.training.runs.curves).toHaveBeenLastCalledWith(firstTen, expect.anything())
  );
  expect(
    await screen.findByText('The charts show the first 10 of 12 selected runs.')
  ).toBeInTheDocument();
  await screen.findByRole('slider', { name: /^Loss: use the arrow keys/ });
  const legend = within(screen.getAllByRole('list', { name: 'Legend' })[0]);
  expect(legend.getAllByRole('listitem')).toHaveLength(10);
  expect(legend.getByText('Run 9')).toBeInTheDocument();
  expect(legend.queryByText('Run 10')).not.toBeInTheDocument();

  // Without the eleventh and twelfth run the same curves stay; nothing is loaded again.
  const calls = vi.mocked(api.training.runs.curves).mock.calls.length;
  await userEvent.click(screen.getByRole('checkbox', { name: 'Compare “Run 11”' }));
  expect(screen.getByText('The charts show the first 10 of 11 selected runs.')).toBeInTheDocument();
  await userEvent.click(screen.getByRole('checkbox', { name: 'Compare “Run 10”' }));
  expect(screen.queryByText(/The charts show the first/)).not.toBeInTheDocument();
  expect(api.training.runs.curves).toHaveBeenCalledTimes(calls);
  await userEvent.click(all);
  expect(screen.getByText('12 runs · 12 selected')).toBeInTheDocument();

  // The scatter plot and the parallel coordinates: all twelve.
  await userEvent.click(screen.getByRole('tab', { name: 'Settings vs. result' }));
  expect(await screen.findByText(/^12 runs with both values/)).toBeInTheDocument();
  await userEvent.click(screen.getByRole('tab', { name: 'Parallel coordinates' }));
  expect(
    await screen.findByRole('slider', { name: /Settings and results: use the arrow keys/ })
  ).toHaveAttribute('aria-valuemax', '12');

  // A shared link keeps its limit: the same ten runs.
  await userEvent.click(screen.getByRole('button', { name: 'Share comparison…' }));
  const dialog = await screen.findByRole('dialog', { name: 'Share a comparison' });
  expect(
    within(dialog).getByText(
      /^A read-only link to these 10 runs\..* They are the first 10 of the 12 selected runs\.$/
    )
  ).toBeInTheDocument();
  await userEvent.click(within(dialog).getByRole('button', { name: 'Create the link' }));
  expect(api.training.comparisons.create).toHaveBeenCalledWith(
    expect.objectContaining({ runs: firstTen })
  );
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
  const de = language === 'de';
  const selectAll = de ? 'Alle Läufe dieser Seite auswählen' : 'Select all runs on this page';

  beforeEach(async () => {
    await i18n.changeLanguage(language);
  });

  it('finds nothing serious with axe, with runs compared', async () => {
    renderPage();
    await screen.findByRole('table', { name: de ? 'Läufe' : 'Runs' });
    const name = de ? '„Adam 1e-3“ vergleichen' : 'Compare “Adam 1e-3”';
    await userEvent.click(screen.getByRole('checkbox', { name }));
    await screen.findAllByRole('slider');
    // Some runs are selected: the header's checkbox is indeterminate.
    expect(screen.getByRole('checkbox', { name: selectAll })).toHaveAttribute(
      'aria-checked',
      'mixed'
    );
    expect(await axeViolations()).toEqual([]);
    if (de) {
      expect(screen.getByRole('heading', { name: 'Läufe' })).toBeInTheDocument();
      expect(screen.getByRole('columnheader', { name: /Lernrate/ })).toBeInTheDocument();
      expect(screen.getAllByText(/^90,0\s%$/).length).toBeGreaterThan(0);
      expect(screen.getByText('4 Läufe · 1 ausgewählt')).toBeInTheDocument();
    }
  });

  it('finds nothing serious with axe, with more runs selected than the charts take', async () => {
    vi.mocked(api.training.runs.table).mockResolvedValue(table(manyRuns(12)));
    vi.mocked(api.training.runs.curves).mockImplementation(async (ids) => ({
      runs: ids.map((id) => curve(id, `Run ${id.slice(1)}`)),
      missing: [],
    }));
    renderPage();
    await screen.findByRole('table', { name: de ? 'Läufe' : 'Runs' });
    const all = screen.getByRole('checkbox', { name: selectAll });
    await userEvent.click(all);
    expect(all).toBeChecked();
    expect(
      screen.getByText(de ? '12 Läufe · 12 ausgewählt' : '12 runs · 12 selected')
    ).toBeInTheDocument();
    expect(
      await screen.findByText(
        de
          ? 'Die Diagramme zeigen die ersten 10 von 12 ausgewählten Läufen.'
          : 'The charts show the first 10 of 12 selected runs.'
      )
    ).toBeInTheDocument();
    await screen.findAllByRole('slider');
    expect(await axeViolations()).toEqual([]);

    await userEvent.click(
      screen.getByRole('button', { name: de ? 'Vergleich teilen …' : 'Share comparison…' })
    );
    const dialog = await screen.findByRole('dialog');
    expect(
      within(dialog).getByText(
        de
          ? /Es sind die ersten 10 der 12 ausgewählten Läufe\.$/
          : /They are the first 10 of the 12 selected runs\.$/
      )
    ).toBeInTheDocument();
    await userEvent.keyboard('{Escape}');
    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());

    const clear = screen.getByRole('button', {
      name: de ? 'Auswahl aufheben' : 'Clear the selection',
    });
    expect(clear).toHaveTextContent(de ? 'Aufheben' : 'Clear');
    await userEvent.click(clear);
    expect(all).not.toBeChecked();
    expect(screen.getByText(de ? '12 Läufe' : '12 runs')).toBeInTheDocument();
  });
});
