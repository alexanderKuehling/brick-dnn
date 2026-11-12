/** The training panel at the full width of the window and back. */
import { render, screen, userEvent, waitFor } from '@test-utils';
import { vi } from 'vitest';
import { ModalsProvider } from '@mantine/modals';
import { api } from '@/api';
import { NO_SCOPE } from './access';
import { ANALYSIS, DOC, STATUS } from './fixtures';
import { DEFAULT_SOURCE } from './settings';
import TrainingDrawer, { WIDE_PANEL_KEY } from './TrainingDrawer';
import type { TrainingController } from './useTraining';

vi.mock('@/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api')>();
  return {
    ...actual,
    api: {
      training: { runs: { list: vi.fn() }, datasets: { list: vi.fn() } },
    },
  };
});

function controller(overrides: Partial<TrainingController> = {}): TrainingController {
  return {
    status: STATUS,
    statusError: null,
    reloadStatus: vi.fn(),
    runs: [],
    runsError: null,
    refreshRuns: vi.fn(),
    activeRun: null,
    scope: NO_SCOPE,
    opened: true,
    open: vi.fn(),
    close: vi.fn(),
    view: 'history',
    showSetup: vi.fn(),
    showHistory: vi.fn(),
    openRun: vi.fn(),
    selectedRunId: null,
    detail: null,
    detailError: null,
    retryDetail: vi.fn(),
    applyRun: vi.fn(),
    forgetRun: vi.fn(),
    data: {
      source: DEFAULT_SOURCE,
      setSource: vi.fn(),
      selection: 0,
      datasets: [],
      datasetsError: null,
      reloadDatasets: vi.fn(),
      uploading: null,
      uploadProgress: null,
      uploadError: null,
      dismissUploadError: vi.fn(),
      upload: vi.fn(),
      cancelUpload: vi.fn(),
      remove: vi.fn(),
    },
    ...overrides,
  };
}

function renderPanel(training = controller()) {
  return render(
    <ModalsProvider>
      <TrainingDrawer
        training={training}
        loggedIn
        doc={DOC}
        networkName="Tiny MLP"
        analysis={ANALYSIS}
        onLogin={vi.fn()}
        onShowProblems={vi.fn()}
        onLoadNetwork={vi.fn()}
      />
    </ModalsProvider>
  );
}

const panelSize = () =>
  (document.querySelector('.mantine-Drawer-root') as HTMLElement).style.getPropertyValue(
    '--drawer-size'
  );
const toggle = () => screen.getByRole('button', { name: 'Full width' });

beforeEach(() => {
  localStorage.clear();
  vi.mocked(api.training.datasets.list).mockResolvedValue([]);
});

afterEach(() => {
  vi.mocked(window.matchMedia).mockImplementation(
    (query: string) =>
      ({
        matches: false,
        media: query,
        onchange: null,
        addListener: vi.fn(),
        removeListener: vi.fn(),
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        dispatchEvent: vi.fn(),
      }) as unknown as MediaQueryList
  );
});

describe('The width of the training panel', () => {
  it('switches to the full width and back, and remembers it', async () => {
    const { unmount } = renderPanel();
    expect(await screen.findByRole('dialog', { name: 'Train this network' })).toBeInTheDocument();
    expect(toggle()).toHaveAttribute('aria-pressed', 'false');
    expect(panelSize()).not.toBe('100%');

    await userEvent.click(toggle());
    expect(toggle()).toHaveAttribute('aria-pressed', 'true');
    expect(panelSize()).toBe('100%');
    expect(localStorage.getItem(WIDE_PANEL_KEY)).toBe('1');

    // Opened again later: still wide.
    unmount();
    renderPanel();
    expect(toggle()).toHaveAttribute('aria-pressed', 'true');
    expect(panelSize()).toBe('100%');

    // With the keyboard: from the close button (focused first) back to the toggle.
    const close = document.querySelector('.mantine-Drawer-close') as HTMLElement;
    await waitFor(() => expect(close).toHaveFocus());
    await userEvent.tab({ shift: true });
    expect(toggle()).toHaveFocus();
    await userEvent.keyboard('{Enter}');
    expect(toggle()).toHaveAttribute('aria-pressed', 'false');
    expect(panelSize()).not.toBe('100%');
    expect(localStorage.getItem(WIDE_PANEL_KEY)).toBeNull();
  });

  it('keeps closing and the first focus as they were', async () => {
    const training = controller();
    renderPanel(training);
    const close = document.querySelector('.mantine-Drawer-close') as HTMLElement;
    await waitFor(() => expect(close).toHaveFocus());
    await userEvent.keyboard('{Escape}');
    expect(training.close).toHaveBeenCalled();
  });

  it('works without storage', async () => {
    const getItem = vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
      throw new Error('blocked');
    });
    const setItem = vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new Error('blocked');
    });
    renderPanel();
    await userEvent.click(toggle());
    expect(toggle()).toHaveAttribute('aria-pressed', 'true');
    getItem.mockRestore();
    setItem.mockRestore();
  });

  it('has no toggle on phones, where the panel fills the screen anyway', () => {
    vi.mocked(window.matchMedia).mockImplementation(
      (query: string) =>
        ({
          matches: true,
          media: query,
          onchange: null,
          addListener: vi.fn(),
          removeListener: vi.fn(),
          addEventListener: vi.fn(),
          removeEventListener: vi.fn(),
          dispatchEvent: vi.fn(),
        }) as unknown as MediaQueryList
    );
    renderPanel();
    expect(screen.queryByRole('button', { name: 'Full width' })).not.toBeInTheDocument();
    expect(panelSize()).toBe('100%');
  });
});
