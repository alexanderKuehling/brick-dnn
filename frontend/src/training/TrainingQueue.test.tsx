/**
 * The training queue in the training panel: servers with a training worker put new runs into
 * a queue (`queued`), and the panel shows where they wait and when they probably start.
 */
import { act, cleanup, render, renderHook, screen, userEvent, waitFor, within } from '@test-utils';
import { afterEach, beforeEach, vi } from 'vitest';
import { ModalsProvider } from '@mantine/modals';
import { notifications } from '@mantine/notifications';
import { api, ApiError, type Compatibility, type QueuePreview } from '@/api';
import { Toolbar } from '@/builder/Toolbar';
import { NO_SCOPE } from './access';
import {
  ANALYSIS,
  detail,
  DOC,
  queue,
  QUEUE_STATUS,
  queuedDetail,
  queuedSummary,
  STATUS,
  summary,
} from './fixtures';
import { isActive, progressLabel, queueText, startsIn, STATUS_LABELS } from './format';
import { DEFAULT_SOURCE } from './settings';
import { SUPPORT_HINT_KEY } from './supportHint';
import TrainingDrawer, { type TrainingDrawerProps } from './TrainingDrawer';
import type { DataController } from './useDatasets';
import {
  DETAIL_INTERVAL,
  LIST_INTERVAL,
  useTraining,
  type TrainingController,
} from './useTraining';

vi.mock('@/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api')>();
  return {
    ...actual,
    api: {
      generate: vi.fn(),
      training: {
        status: vi.fn(),
        datasets: { list: vi.fn(), upload: vi.fn(), remove: vi.fn() },
        check: vi.fn(),
        estimate: vi.fn(),
        runs: {
          list: vi.fn(),
          get: vi.fn(),
          create: vi.fn(),
          stop: vi.fn(),
          remove: vi.fn(),
          downloadWeights: vi.fn(),
        },
      },
    },
  };
});

const data: DataController = {
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
};

function controller(overrides: Partial<TrainingController> = {}): TrainingController {
  return {
    status: QUEUE_STATUS,
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
    view: 'setup',
    showSetup: vi.fn(),
    showHistory: vi.fn(),
    openRun: vi.fn(),
    selectedRunId: null,
    detail: null,
    detailError: null,
    retryDetail: vi.fn(),
    applyRun: vi.fn(),
    forgetRun: vi.fn(),
    data,
    ...overrides,
  };
}

function drawer(training: TrainingController, props: Partial<TrainingDrawerProps> = {}) {
  return (
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
        {...props}
      />
    </ModalsProvider>
  );
}

/** The panel showing one run. */
function showRun(run = queuedDetail(), overrides: Partial<TrainingController> = {}) {
  const training = controller({ view: 'run', selectedRunId: run.id, detail: run, ...overrides });
  return { training, ...render(drawer(training)) };
}

const READY: Compatibility = { ok: true, errors: [], warnings: [], plan: null };

function preview(overrides: Partial<QueuePreview> = {}): QueuePreview {
  return { ...queue(), running: 1, message: null, ...overrides };
}

const startButton = () => screen.getByRole('button', { name: 'Start training' });

beforeEach(() => {
  vi.clearAllMocks();
  localStorage.clear();
  vi.mocked(api.training.datasets.list).mockResolvedValue([]);
  vi.spyOn(notifications, 'show').mockImplementation(() => '');
});

describe('texts of the queue', () => {
  it('describes where a run waits and when it starts', () => {
    expect(isActive('queued')).toBe(true);
    expect(STATUS_LABELS.queued).toBe('Queued');
    expect(queueText(queue())).toBe('Queued — position 3, starts in about 4 minutes');
    expect(queueText(queue({ position: 1, estimated_wait_seconds: 0 }))).toBe(
      'Queued — position 1, starts in a moment'
    );
    expect(queueText(queue({ estimated_wait_seconds: null }))).toBe('Queued — position 3');
    expect(queueText(queue({ worker_online: false, estimated_wait_seconds: null }))).toBe(
      'Queued — position 3, waiting for a training worker'
    );
    expect(queueText(queue({ blocked_by: 'account' }))).toBe(
      'Queued — position 3, starts after your running training'
    );
    expect(queueText(null)).toBe('Queued');
  });

  it('rounds the estimated wait', () => {
    expect(startsIn(5)).toBe('starts in a moment');
    expect(startsIn(60)).toBe('starts in about a minute');
    expect(startsIn(150)).toBe('starts in about 3 minutes');
    expect(startsIn(3600)).toBe('starts in about 60 minutes');
    expect(startsIn(2 * 3600)).toBe('starts in about 2 hours');
    expect(startsIn(3600 * 1.6)).toBe('starts in about 2 hours');
    expect(startsIn(null)).toBeNull();
    expect(startsIn(undefined)).toBeNull();
  });

  it('keeps the toolbar text short', () => {
    expect(progressLabel(queuedSummary())).toBe('Queued #3');
    // An older server (or a run that was just taken by a worker) sends no position.
    expect(progressLabel(queuedSummary({ queue: null }))).toBe('Queued');
  });
});

describe('a queued run', () => {
  it('shows its position and estimated start instead of results', () => {
    showRun();
    expect(screen.getByRole('heading', { name: 'Small CNN' })).toBeInTheDocument();
    expect(screen.getByText('Queued')).toBeInTheDocument();
    const box = screen.getByRole('region', { name: 'Queue' });
    expect(
      within(box).getByText('Queued — position 3, starts in about 4 minutes')
    ).toBeInTheDocument();
    expect(box).toHaveTextContent('The start time is a rough estimate.');
    expect(box).toHaveTextContent('5 trainings are waiting.');
    expect(box).toHaveTextContent('Paid plans start first.');
    expect(box).toHaveTextContent('You can close this panel');
    expect(screen.getByText(/queued Oct 28/)).toBeInTheDocument();
    // Nothing was trained yet: no tiles, charts or weights.
    expect(screen.queryByText('Validation loss')).not.toBeInTheDocument();
    expect(screen.queryByText(/The charts appear/)).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /Weights/ })).not.toBeInTheDocument();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('follows the queue while it is polled', () => {
    const { training, rerender } = showRun();
    const next = queuedDetail({
      queue: queue({ position: 1, waiting: 1, estimated_wait_seconds: 10, hint: null }),
    });
    rerender(drawer({ ...training, detail: next }));
    const box = screen.getByRole('region', { name: 'Queue' });
    expect(within(box).getByText('Queued — position 1, starts in a moment')).toBeInTheDocument();
    expect(box).not.toHaveTextContent('Paid plans start first.');
    expect(box).not.toHaveTextContent('trainings are waiting');
    // A worker took it: the usual progress takes over.
    rerender(drawer({ ...training, detail: detail() }));
    expect(screen.queryByRole('region', { name: 'Queue' })).not.toBeInTheDocument();
    expect(screen.getByText('Epoch 4 of 10')).toBeInTheDocument();
  });

  it('says when no training worker is running', () => {
    showRun(
      queuedDetail({
        queue: queue({
          worker_online: false,
          estimated_wait_seconds: null,
          estimated_start_at: null,
        }),
      })
    );
    expect(
      screen.getByText('Queued — position 3, waiting for a training worker')
    ).toBeInTheDocument();
    const alert = screen.getByRole('alert');
    expect(alert).toHaveTextContent('No training worker is running');
    expect(alert).toHaveTextContent('keeps its place and starts as soon as a worker is back');
    expect(screen.queryByText(/rough estimate/)).not.toBeInTheDocument();
  });

  it('explains a run that waits for the plan’s trainings at a time', () => {
    showRun(queuedDetail({ queue: queue({ position: 1, blocked_by: 'account', hint: null }) }));
    const box = screen.getByRole('region', { name: 'Queue' });
    expect(
      within(box).getByText('Queued — position 1, starts after your running training')
    ).toBeInTheDocument();
    expect(box).toHaveTextContent('Your plan limits how many trainings run at the same time');
  });

  it('says why a run starts again', () => {
    const message = 'Waiting to start again from the beginning: the training worker was restarted.';
    showRun(queuedDetail({ message }));
    expect(screen.getByRole('region', { name: 'Queue' })).toHaveTextContent(message);
  });

  it('can be cancelled', async () => {
    const run = queuedDetail();
    const cancelled = queuedDetail({
      status: 'stopped',
      message: 'Cancelled before it started.',
      queue: null,
    });
    vi.mocked(api.training.runs.stop).mockResolvedValue(cancelled);
    const { training } = showRun(run);
    await userEvent.click(screen.getByRole('button', { name: 'Cancel' }));
    expect(await screen.findByText('Cancel the training?')).toBeInTheDocument();
    expect(
      screen.getByText('“Small CNN” leaves the queue without having trained. It uses no compute.')
    ).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Keep waiting' })).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Cancel training' }));
    expect(api.training.runs.stop).toHaveBeenCalledWith('r1');
    await waitFor(() => expect(training.applyRun).toHaveBeenCalledWith(cancelled));

    cleanup();
    showRun(cancelled);
    expect(screen.getByRole('alert')).toHaveTextContent('Cancelled before it started.');
    expect(screen.queryByRole('button', { name: 'Cancel' })).not.toBeInTheDocument();
  });

  it('says that it leaves the queue when it is deleted', async () => {
    showRun();
    await userEvent.click(screen.getByRole('button', { name: 'Delete' }));
    expect(
      await screen.findByText('The training leaves the queue and is deleted.')
    ).toBeInTheDocument();
  });

  it('explains a run that could not start because its budget was gone', () => {
    showRun(
      queuedDetail({
        status: 'failed',
        message: 'Not started: The monthly compute allowance of this account is used up.',
        queue: null,
        usage: {
          cpu_minutes: 0,
          limit_cpu_minutes: 15,
          estimated_cpu_minutes: 2,
          stop_reason: 'budget',
        },
      })
    );
    const alert = screen.getByRole('alert');
    expect(alert).toHaveTextContent('Not started: no compute budget left');
    expect(alert).toHaveTextContent('allowance of this account is used up');
    expect(within(alert).getByRole('link', { name: 'Usage & budget' })).toBeInTheDocument();
  });

  it('is listed with its place in the queue', async () => {
    const training = controller({
      view: 'history',
      runs: [
        queuedSummary({ id: 'q2', name: 'Second', queue: queue({ position: 4 }) }),
        queuedSummary({
          id: 'q1',
          name: 'First',
          queue: queue({ position: 2, worker_online: false, estimated_wait_seconds: null }),
        }),
        summary({ id: 'r0', name: 'Training now' }),
      ],
    });
    render(drawer(training));
    const second = screen.getByText('Second').closest('button')!;
    expect(within(second).getByText('Queued')).toBeInTheDocument();
    expect(within(second).getByText('Position 4')).toBeInTheDocument();
    expect(within(second).getByText('starts in about 4 minutes')).toBeInTheDocument();
    const first = screen.getByText('First').closest('button')!;
    expect(within(first).getByText('Position 2')).toBeInTheDocument();
    expect(within(first).getByText('in the queue')).toBeInTheDocument();
    expect(screen.getByText('Epoch 4/10')).toBeInTheDocument();
    await userEvent.click(screen.getByText('Second'));
    expect(training.openRun).toHaveBeenCalledWith('q2');
  });
});

describe('starting a training on a server with a queue', () => {
  it('does not wait for the running training', async () => {
    vi.mocked(api.training.check).mockResolvedValue(READY);
    vi.mocked(api.training.estimate).mockResolvedValue({ ...READY, queue: preview() });
    const created = queuedDetail({ id: 'new' });
    vi.mocked(api.training.runs.create).mockResolvedValue(created);
    const training = controller({ activeRun: summary() });
    render(drawer(training));
    // Without a queue the panel would say "A training is running" and block the start.
    await waitFor(() => expect(startButton()).toBeEnabled());
    expect(screen.queryByText('A training is running')).not.toBeInTheDocument();
    expect(
      await screen.findByText(
        'The server is busy: this training will wait in the queue (position 3, starts in about 4 minutes; an estimate). Paid plans start first.',
        {},
        { timeout: 3000 }
      )
    ).toBeInTheDocument();
    await userEvent.click(startButton());
    expect(training.applyRun).toHaveBeenCalledWith(created);
  });

  it('says nothing about the queue when the training would start at once', async () => {
    vi.mocked(api.training.check).mockResolvedValue(READY);
    vi.mocked(api.training.estimate).mockResolvedValue({
      ...READY,
      queue: preview({ position: 1, waiting: 0, running: 0, estimated_wait_seconds: 0 }),
    });
    render(drawer(controller()));
    await waitFor(() => expect(api.training.estimate).toHaveBeenCalled(), { timeout: 3000 });
    await waitFor(() => expect(startButton()).toBeEnabled());
    expect(screen.queryByText(/will wait in the queue/)).not.toBeInTheDocument();
    expect(screen.queryByText(/No training worker/)).not.toBeInTheDocument();
  });

  it('warns that no training worker is running', async () => {
    vi.mocked(api.training.check).mockResolvedValue(READY);
    vi.mocked(api.training.estimate).mockResolvedValue({
      ...READY,
      queue: preview({ position: 1, worker_online: false, estimated_wait_seconds: null }),
    });
    render(drawer(controller()));
    const alert = await screen.findByRole('alert', {}, { timeout: 3000 });
    expect(alert).toHaveTextContent('No training worker is running');
    expect(alert).toHaveTextContent('waits in the queue until a worker is back');
    expect(startButton()).toBeEnabled(); // it can still be queued
  });

  it('explains a start that is refused because too many trainings wait', async () => {
    vi.mocked(api.training.check).mockResolvedValue(READY);
    vi.mocked(api.training.estimate).mockResolvedValue(READY);
    vi.mocked(api.training.runs.create).mockRejectedValue(
      new ApiError(409, null, 'Your plan (Free) lets 3 trainings of an account wait in the queue.')
    );
    const training = controller();
    render(drawer(training));
    await waitFor(() => expect(startButton()).toBeEnabled());
    await userEvent.click(startButton());
    expect(await screen.findByText('Too many trainings are waiting')).toBeInTheDocument();
    expect(screen.getByText(/lets 3 trainings of an account wait/)).toBeInTheDocument();
    expect(training.refreshRuns).toHaveBeenCalled();
  });

  it('still blocks the start on a server that trains one at a time', async () => {
    vi.mocked(api.training.check).mockResolvedValue(READY);
    render(drawer(controller({ status: STATUS, activeRun: summary() })));
    expect(await screen.findByText('A training is running')).toBeInTheDocument();
    expect(startButton()).toBeDisabled();
  });
});

describe('the toolbar', () => {
  function renderToolbar(run: React.ComponentProps<typeof Toolbar>['trainingRun']) {
    const noop = vi.fn();
    render(
      <Toolbar
        name="MLP"
        onNameChange={noop}
        analysis={null}
        analyzing={false}
        analysisError={null}
        empty={false}
        examples={[]}
        saving={false}
        isSaved
        onLoadExample={noop}
        onNew={noop}
        onImport={noop}
        onExportJson={noop}
        onTidy={noop}
        onSave={noop}
        onShowCode={noop}
        onExportPython={noop}
        onTrain={noop}
        trainingRun={run}
        onShowSummary={noop}
        onOpenPalette={noop}
        onOpenInspector={noop}
      />
    );
  }

  it('shows the place of a queued training on the train button', async () => {
    renderToolbar(queuedSummary());
    const button = screen.getByRole('button', { name: 'Queued #3' });
    await userEvent.hover(button);
    expect(
      await screen.findByText('“Small CNN”: Queued — position 3, starts in about 4 minutes')
    ).toBeInTheDocument();
    // The icon-only button of phones names it too.
    expect(screen.getByRole('button', { name: 'Training: Queued #3' })).toBeInTheDocument();
  });
});

describe('useTraining with a queue', () => {
  const flush = () => act(() => vi.advanceTimersByTimeAsync(0));
  const wait = (ms: number) => act(() => vi.advanceTimersByTimeAsync(ms));

  beforeEach(() => {
    vi.useFakeTimers();
    vi.mocked(api.training.status).mockResolvedValue(QUEUE_STATUS);
    localStorage.setItem(SUPPORT_HINT_KEY, 'shown');
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('follows a queued run until it has trained and reports the result', async () => {
    vi.mocked(api.training.runs.list)
      .mockResolvedValueOnce([queuedSummary()])
      .mockResolvedValueOnce([queuedSummary({ queue: queue({ position: 1 }) })])
      .mockResolvedValueOnce([summary()])
      .mockResolvedValue([summary({ status: 'completed', epochs_done: 10 })]);
    const { result } = renderHook(() => useTraining(true));
    await flush();
    expect(result.current.activeRun?.status).toBe('queued');
    expect(result.current.activeRun?.queue?.position).toBe(3);

    await wait(LIST_INTERVAL);
    expect(result.current.activeRun?.queue?.position).toBe(1);
    await wait(LIST_INTERVAL);
    expect(result.current.activeRun?.status).toBe('running');
    expect(notifications.show).not.toHaveBeenCalled(); // starting is no news
    await wait(LIST_INTERVAL);
    expect(result.current.activeRun).toBeNull();
    expect(notifications.show).toHaveBeenCalledWith(
      expect.objectContaining({ title: 'Training finished', color: 'green' })
    );
  });

  it('asks for a coffee after the first training also when it had to wait', async () => {
    localStorage.removeItem(SUPPORT_HINT_KEY);
    vi.mocked(api.training.runs.list)
      .mockResolvedValueOnce([queuedSummary()])
      .mockResolvedValue([summary({ status: 'completed', epochs_done: 10 })]);
    renderHook(() => useTraining(true, { username: 'ada', project: null }));
    await flush();
    await wait(LIST_INTERVAL);
    expect(notifications.show).toHaveBeenCalledWith(
      expect.objectContaining({ id: 'support-hint', title: 'Your first training is done!' })
    );
    expect(localStorage.getItem(SUPPORT_HINT_KEY)).not.toBeNull();
  });

  it('reports a queued run that could not start', async () => {
    vi.mocked(api.training.runs.list)
      .mockResolvedValueOnce([queuedSummary()])
      .mockResolvedValue([
        queuedSummary({
          status: 'failed',
          message: 'Not started: the budget is used up.',
          queue: null,
        }),
      ]);
    renderHook(() => useTraining(true));
    await flush();
    await wait(LIST_INTERVAL);
    expect(notifications.show).toHaveBeenCalledWith(
      expect.objectContaining({ title: 'Training failed', color: 'red' })
    );
  });

  it('puts the training that runs before the ones that wait', async () => {
    vi.mocked(api.training.runs.list).mockResolvedValue([
      queuedSummary({ id: 'q2', queue: queue({ position: 2 }) }),
      queuedSummary({ id: 'q1', queue: queue({ position: 1 }) }),
      summary({ id: 'r0' }),
    ]);
    const { result } = renderHook(() => useTraining(true));
    await flush();
    expect(result.current.activeRun?.id).toBe('r0');
  });

  it('shows the queued run that starts first when none runs', async () => {
    vi.mocked(api.training.runs.list).mockResolvedValue([
      queuedSummary({ id: 'q3', queue: queue({ position: 5 }) }),
      queuedSummary({ id: 'q2', queue: queue({ position: 2 }) }),
      queuedSummary({ id: 'q1', queue: queue({ position: 4 }) }),
    ]);
    vi.mocked(api.training.runs.get).mockResolvedValue(queuedDetail({ id: 'q2' }));
    const { result } = renderHook(() => useTraining(true));
    await flush();
    expect(result.current.activeRun?.id).toBe('q2');

    // Opening the panel shows it, and it is polled like a running one.
    act(() => result.current.open());
    await flush();
    expect(result.current.selectedRunId).toBe('q2');
    expect(result.current.detail?.status).toBe('queued');
    await wait(DETAIL_INTERVAL * 2);
    expect(vi.mocked(api.training.runs.get).mock.calls.length).toBeGreaterThanOrEqual(3);
  });
});
