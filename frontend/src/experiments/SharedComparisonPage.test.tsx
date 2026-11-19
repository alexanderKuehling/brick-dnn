import { axeViolations, render, screen, userEvent, within } from '@test-utils';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { beforeEach, vi } from 'vitest';
import { api, ApiError, type SharedComparison } from '@/api';
import { i18n } from '@/i18n';
import { curve, row } from './fixtures';
import { SharedComparisonPage } from './SharedComparisonPage';

vi.mock('@/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api')>();
  return {
    ...actual,
    api: { training: { comparisons: { shared: vi.fn() } } },
  };
});

/** What the server sends: no starter, no network, no tags or notes (not chosen). */
function publicRow(id: string, name: string, accuracy: number) {
  const full = row({ id, name, metrics: { val_loss: 1 - accuracy, val_accuracy: accuracy } });
  const { started_by, project, project_name, mine, can_annotate, tags, note, ...shown } = full;
  void [started_by, project, project_name, mine, can_annotate, tags, note];
  return shown;
}

const SHARED: SharedComparison = {
  name: 'Learning rates',
  created_at: '2026-11-16T10:00:00Z',
  columns: ['dataset', 'setting:learning_rate', 'metric:val_accuracy'],
  options: { metric: 'val_accuracy' },
  include_notes: false,
  runs: [publicRow('a', 'Fast', 0.8), publicRow('b', 'Slow', 0.9)],
  curves: [curve('a', 'Fast'), curve('b', 'Slow')],
};

function renderAt(token = 'tok123456') {
  return render(
    <MemoryRouter initialEntries={[`/compare/${token}`]}>
      <Routes>
        <Route path="/compare/:token" element={<SharedComparisonPage />} />
      </Routes>
    </MemoryRouter>
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(api.training.comparisons.shared).mockResolvedValue(SHARED);
});

it('shows the shared runs read-only with the chosen columns and their curves', async () => {
  renderAt();
  expect(await screen.findByRole('heading', { name: 'Learning rates' })).toBeInTheDocument();
  expect(api.training.comparisons.shared).toHaveBeenCalledWith('tok123456', expect.anything());
  const grid = screen.getByRole('table', { name: 'Runs of “Learning rates”' });
  const headers = within(grid)
    .getAllByRole('columnheader')
    .map((h) => h.textContent);
  expect(headers).toEqual(['Star', 'Name', 'Dataset', 'Learning rate', 'Accuracy · val.']);
  // Nothing to change and nothing private.
  expect(screen.queryByRole('checkbox')).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: /Tags and note/ })).not.toBeInTheDocument();
  expect(screen.queryByText('ada')).not.toBeInTheDocument();
  expect(screen.queryByText('Iris net')).not.toBeInTheDocument();
  // Sorted in the browser.
  await userEvent.click(screen.getByRole('button', { name: 'Sort by Accuracy · val.' }));
  await userEvent.click(screen.getByRole('button', { name: 'Sort by Accuracy · val.' }));
  const names = within(grid)
    .getAllByRole('row')
    .slice(1)
    .map((r) => within(r).getAllByRole('cell')[1].textContent);
  expect(names).toEqual(['Slow', 'Fast']);
  // The metric chart the owner chose.
  expect(
    screen.getByRole('slider', { name: /^Accuracy · val\.: use the arrow keys/ })
  ).toBeInTheDocument();
});

it('says when the link was turned off', async () => {
  vi.mocked(api.training.comparisons.shared).mockRejectedValue(
    new ApiError(404, null, 'Not found')
  );
  renderAt('gone12345');
  expect(
    await screen.findByText('This link does not exist or was turned off.')
  ).toBeInTheDocument();
});

describe.each(['en', 'de'] as const)('in %s', (language) => {
  beforeEach(async () => {
    await i18n.changeLanguage(language);
  });

  it('finds nothing serious with axe', async () => {
    renderAt();
    await screen.findByRole('heading', { name: 'Learning rates' });
    if (language === 'de') {
      expect(screen.getByText(/2 Läufe, geteilt am/)).toBeInTheDocument();
      expect(screen.getByRole('tab', { name: 'Kurven' })).toBeInTheDocument();
    }
    expect(await axeViolations()).toEqual([]);
  });
});

describe.each(['en', 'de'] as const)('the "made with" block in %s', (language) => {
  it('says what Brick DNN is, with one link into the editor', async () => {
    await i18n.changeLanguage(language);
    renderAt();
    const about = await screen.findByRole('region', {
      name: language === 'de' ? 'Über Brick DNN' : 'About Brick DNN',
    });
    expect(
      within(about).getByText(language === 'de' ? 'Erstellt mit Brick DNN' : 'Made with Brick DNN')
    ).toBeInTheDocument();
    expect(
      within(about).getByText(
        language === 'de'
          ? /ist ein Editor für neuronale Netze im Browser/
          : /is an editor for neural networks in the browser/
      )
    ).toBeVisible();
    const links = within(about).getAllByRole('link');
    expect(links).toHaveLength(1);
    expect(links[0]).toHaveAccessibleName(
      language === 'de' ? 'Eigenes Netz bauen' : 'Build your own network'
    );
    expect(links[0]).toHaveAttribute('href', '/app');
    // The logo is decoration: the words next to it name it.
    expect(about.querySelector('img')).toHaveAttribute('alt', '');
    expect(await axeViolations()).toEqual([]);
  });
});

it('shows no "made with" block for a link that was turned off', async () => {
  vi.mocked(api.training.comparisons.shared).mockRejectedValue(new ApiError(404, 'Not found'));
  renderAt();
  expect(await screen.findByRole('alert')).toBeInTheDocument();
  expect(screen.queryByRole('region', { name: 'About Brick DNN' })).not.toBeInTheDocument();
});
