import { render, screen, userEvent, waitFor, within } from '@test-utils';
import { beforeEach, vi } from 'vitest';
import { ModalsProvider } from '@mantine/modals';
import { api, ApiError } from '@/api';
import { ANALYSIS, detail, DOC, searchDetail, STATUS, summary, TUNING } from './fixtures';
import { DEFAULT_SETTINGS, DEFAULT_SOURCE, saveSettings } from './settings';
import TrainingDrawer, { type TrainingDrawerProps } from './TrainingDrawer';
import type { DataController } from './useDatasets';
import type { TrainingController } from './useTraining';

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

function dataController(overrides: Partial<DataController> = {}): DataController {
  return {
    source: DEFAULT_SOURCE,
    setSource: vi.fn(),
    datasets: [],
    datasetsError: null,
    reloadDatasets: vi.fn(),
    uploading: null,
    uploadError: null,
    dismissUploadError: vi.fn(),
    upload: vi.fn(),
    remove: vi.fn(),
    ...overrides,
  };
}

function controller(overrides: Partial<TrainingController> = {}): TrainingController {
  return {
    status: STATUS,
    statusError: null,
    reloadStatus: vi.fn(),
    runs: [],
    runsError: null,
    refreshRuns: vi.fn(),
    activeRun: null,
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
    data: dataController(),
    ...overrides,
  };
}

function renderDrawer(props: Partial<TrainingDrawerProps> = {}) {
  const all: TrainingDrawerProps = {
    training: controller(),
    loggedIn: true,
    doc: DOC,
    networkName: 'Tiny MLP',
    analysis: ANALYSIS,
    onLogin: vi.fn(),
    onShowProblems: vi.fn(),
    onLoadNetwork: vi.fn(),
    ...props,
  };
  render(
    <ModalsProvider>
      <TrainingDrawer {...all} />
    </ModalsProvider>
  );
  return all;
}

const startButton = () => screen.getByRole('button', { name: 'Start training' });

beforeEach(() => {
  vi.clearAllMocks();
  localStorage.clear();
  vi.mocked(api.training.datasets.list).mockResolvedValue([]);
});

describe('TrainingDrawer', () => {
  it('explains how to install PyTorch when training is unavailable', async () => {
    const training = controller({
      status: {
        ...STATUS,
        available: false,
        reason: 'PyTorch is not installed on the server.',
        devices: [],
      },
    });
    renderDrawer({ training });
    expect(screen.getByText('PyTorch is not installed on the server.')).toBeInTheDocument();
    expect(screen.getByText('pip install torch torchvision')).toBeInTheDocument();
    expect(screen.getByText(/BRICKDNN_INSTALL_TORCH=true/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Check again' }));
    expect(training.reloadStatus).toHaveBeenCalled();
  });

  it('asks visitors to log in', async () => {
    const props = renderDrawer({ loggedIn: false });
    expect(screen.getByText('Log in to train')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Start training' })).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Log in' }));
    expect(props.onLogin).toHaveBeenCalled();
  });

  it('points to the problems of an invalid network', async () => {
    const props = renderDrawer({
      analysis: { ...ANALYSIS, valid: false, errors: [{ node: 'fc', message: 'Bad' }] },
    });
    expect(screen.getByText('The network has problems')).toBeInTheDocument();
    expect(startButton()).toBeDisabled();
    expect(api.training.check).not.toHaveBeenCalled();
    await userEvent.click(screen.getByRole('button', { name: 'Show problems' }));
    expect(props.onShowProblems).toHaveBeenCalled();
  });

  it('blocks the start when the network does not fit the data', async () => {
    vi.mocked(api.training.check).mockResolvedValue({
      ok: false,
      errors: ['The Input expects shape 4, but the CSV file has 7 feature columns.'],
      warnings: [],
      plan: null,
    });
    renderDrawer();
    expect(await screen.findByText('Does not fit the data')).toBeInTheDocument();
    expect(screen.getByText(/but the CSV file has 7 feature columns/)).toBeInTheDocument();
    expect(startButton()).toBeDisabled();
    expect(api.training.check).toHaveBeenCalledWith(
      expect.objectContaining({ nodes: expect.any(Array) }),
      { kind: 'random' },
      expect.any(AbortSignal)
    );
  });

  it('starts a training when the check passes', async () => {
    vi.mocked(api.training.check).mockResolvedValue({
      ok: true,
      errors: [],
      warnings: ['The network outputs 12 values, but MNIST has only 10 classes.'],
      plan: {
        task: 'demo',
        summary: 'Demo data: random inputs and a hidden rule to learn (classification).',
        inputs: [],
        targets: [],
        image: null,
        num_classes: 3,
        class_names: null,
        notes: ['The numbers say nothing about real-world accuracy.'],
      },
    });
    const created = detail({ status: 'starting' });
    vi.mocked(api.training.runs.create).mockResolvedValue(created);
    const training = controller();
    renderDrawer({ training });

    expect(await screen.findByText(/Demo data: random inputs/)).toBeInTheDocument();
    expect(screen.getByText(/only 10 classes/)).toBeInTheDocument();
    await waitFor(() => expect(startButton()).toBeEnabled());
    await userEvent.click(startButton());

    expect(api.training.runs.create).toHaveBeenCalledWith({
      name: 'Tiny MLP',
      graph: DOC,
      dataset: { kind: 'random' },
      config: expect.objectContaining({ epochs: 10, batch_size: 64, optimizer: 'adam' }),
    });
    expect(training.applyRun).toHaveBeenCalledWith(created);
  });

  it('offers to open the running training when the server is busy', async () => {
    vi.mocked(api.training.check).mockResolvedValue({
      ok: true,
      errors: [],
      warnings: [],
      plan: null,
    });
    vi.mocked(api.training.runs.create).mockRejectedValue(
      new ApiError(409, null, 'Another training is still running.')
    );
    const training = controller();
    renderDrawer({ training });
    await waitFor(() => expect(startButton()).toBeEnabled());
    await userEvent.click(startButton());
    expect(await screen.findByText('Another training is still running.')).toBeInTheDocument();
    expect(training.refreshRuns).toHaveBeenCalled();
  });

  it('lists runs with their results', async () => {
    const training = controller({
      view: 'history',
      runs: [
        summary({
          id: 'r2',
          name: 'Done',
          status: 'completed',
          finished_at: '2026-10-28T10:02:00Z',
        }),
        summary({ id: 'r1', name: 'Broken', status: 'failed', metrics: null }),
      ],
    });
    renderDrawer({ training });
    expect(screen.getByText('Done')).toBeInTheDocument();
    expect(screen.getByText('95.0%')).toBeInTheDocument();
    expect(screen.getByText(/2 min/)).toBeInTheDocument();
    expect(screen.getByText('Failed')).toBeInTheDocument();
    await userEvent.click(screen.getByText('Done'));
    expect(training.openRun).toHaveBeenCalledWith('r2');
  });

  it('shows progress, metrics and charts of a run', async () => {
    const run = detail();
    renderDrawer({
      training: controller({ view: 'run', selectedRunId: run.id, detail: run, activeRun: run }),
    });
    expect(screen.getByRole('heading', { name: 'Small CNN' })).toBeInTheDocument();
    expect(screen.getByText('Epoch 4 of 10')).toBeInTheDocument();
    expect(screen.getByText(/Step 469 of 938/)).toBeInTheDocument();
    expect(screen.getByText('Validation accuracy').nextSibling).toHaveTextContent('95.0%');
    expect(screen.getByText('Best epoch').nextSibling).toHaveTextContent('3');
    expect(screen.getByText('5.4K/s')).toBeInTheDocument();
    expect(screen.getByRole('slider', { name: /^Loss/ })).toHaveAttribute('aria-valuenow', '3');
    expect(screen.getByRole('slider', { name: /^Accuracy/ })).toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: 'Values per epoch' }));
    const table = screen.getByRole('table');
    expect(within(table).getAllByRole('row')).toHaveLength(4);

    await userEvent.click(screen.getByRole('button', { name: /^Log/ }));
    expect(screen.getByText(/Downloading MNIST/)).toBeInTheDocument();
  });

  it('shows why the server ended a run, not the last state of the process', () => {
    const base = detail();
    // The process died: the API marked the run failed, state.json still says "running".
    const run = detail({
      status: 'failed',
      message: 'The training process stopped unexpectedly. Last output: Killed',
      finished_at: '2026-10-28T10:05:00Z',
      state: { ...base.state!, status: 'running', message: 'Epoch 4 of 10' },
    });
    renderDrawer({ training: controller({ view: 'run', selectedRunId: run.id, detail: run }) });
    expect(screen.getByRole('alert')).toHaveTextContent('stopped unexpectedly');
    expect(screen.queryByText('Epoch 4 of 10')).not.toBeInTheDocument();
  });

  it('stops a running training after confirmation', async () => {
    const run = detail();
    const stopped = detail({ stop_requested: true });
    vi.mocked(api.training.runs.stop).mockResolvedValue(stopped);
    const training = controller({ view: 'run', selectedRunId: run.id, detail: run });
    renderDrawer({ training });

    await userEvent.click(screen.getByRole('button', { name: 'Stop' }));
    await userEvent.click(await screen.findByRole('button', { name: 'Stop training' }));
    expect(api.training.runs.stop).toHaveBeenCalledWith('r1');
    await waitFor(() => expect(training.applyRun).toHaveBeenCalledWith(stopped));
  });

  it('offers the weights and loading the network of a finished run', async () => {
    const run = detail({
      status: 'completed',
      has_weights: true,
      graph: { ...DOC, nodes: DOC.nodes.slice(0, 1), edges: [] },
    });
    const props = renderDrawer({
      training: controller({ view: 'run', selectedRunId: run.id, detail: run }),
    });
    expect(screen.queryByRole('button', { name: /Stop/ })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Weights \(\.pt\)/ })).toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: 'How to use the weights' }));
    expect(screen.getByText(/load_state_dict/)).toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: 'Load into editor' }));
    expect(props.onLoadNetwork).toHaveBeenCalledWith(run.graph, 'Small CNN');
  });

  it('starts a hyperparameter search with a test set', async () => {
    vi.mocked(api.training.check).mockResolvedValue({
      ok: true,
      errors: [],
      warnings: [],
      plan: null,
    });
    saveSettings({ ...DEFAULT_SETTINGS, search: true, test_enabled: true, test_split: 10 });
    vi.mocked(api.training.runs.create).mockResolvedValue(searchDetail());
    renderDrawer();
    const start = screen.getByRole('button', { name: 'Start search' });
    await waitFor(() => expect(start).toBeEnabled());
    await userEvent.click(start);
    expect(api.training.runs.create).toHaveBeenCalledWith(
      expect.objectContaining({
        config: expect.objectContaining({ test_split: 0.1, epochs: 10 }),
        tuning: {
          method: 'random',
          trials: 10,
          space: { learning_rate: { min: 0.0001, max: 0.01 }, batch_size: [32, 64, 128] },
        },
      })
    );
    const config = vi.mocked(api.training.runs.create).mock.calls[0][0].config;
    expect(config).not.toHaveProperty('learning_rate', expect.anything());
  });

  it('shows the trials of a running search', async () => {
    const run = searchDetail();
    renderDrawer({
      training: controller({ view: 'run', selectedRunId: run.id, detail: run, activeRun: run }),
    });
    expect(screen.getByText('Trial 4 of 4 · Epoch 3 of 5')).toBeInTheDocument();
    // 3 finished trials and half of the fourth one's third epoch: (3 + 2.5 / 5) / 4.
    expect(screen.getByText('87%')).toBeInTheDocument();
    expect(screen.getByText(/search · 4 trials/)).toBeInTheDocument();
    expect(screen.getByText('Best trial so far').nextSibling).toHaveTextContent('3');
    expect(
      screen.getByText(/Best so far: trial 3 \(learning rate 0.0011, batch size 32\)/)
    ).toBeInTheDocument();
    expect(screen.getByText('Curves of trial 4 (running)')).toBeInTheDocument();

    const table = screen.getByRole('table', { name: 'Trials' });
    const rows = within(table).getAllByRole('row');
    expect(rows).toHaveLength(5);
    expect(rows[0]).toHaveTextContent('TrialLearning rateBatch sizeVal. lossVal. acc.Status');
    expect(rows[1]).toHaveTextContent('10.00312640.293.0%Done');
    expect(rows[3]).toHaveTextContent('Best');
    expect(rows[4]).toHaveTextContent('Epoch 3/5');
    await userEvent.hover(within(rows[2]).getByText('Failed'));
    expect(await screen.findByText(/The loss became NaN in epoch 2/)).toBeInTheDocument();
  });

  it('shows the test result and the best trial of a finished search', () => {
    const base = searchDetail();
    const run = searchDetail({
      status: 'completed',
      message: 'Finished 4 trials (1 failed). Best: trial 3 …',
      has_weights: true,
      trials_done: 4,
      metrics: {
        train_loss: 0.1,
        val_loss: 0.12,
        val_accuracy: 0.96,
        val_rmse: null,
        test_loss: 0.14,
        test_accuracy: 0.955,
        test_rmse: null,
      },
      state: {
        ...base.state!,
        status: 'completed',
        trial: null,
        progress: null,
        test_size: 10000,
        test: { loss: 0.14, accuracy: 0.955, rmse: null },
        trials: base.state!.trials!.map((t) =>
          t.trial === 4 ? { ...t, status: 'completed' as const } : t
        ),
      },
    });
    renderDrawer({ training: controller({ view: 'run', selectedRunId: run.id, detail: run }) });
    expect(screen.getByText('Test accuracy').nextSibling).toHaveTextContent('95.5%');
    expect(
      screen.getByText('loss 0.14 · 10,000 test samples, measured once after training')
    ).toBeInTheDocument();
    expect(screen.getByText('Best trial').nextSibling).toHaveTextContent('3');
    expect(screen.getByText(/Its weights are kept/)).toBeInTheDocument();
    expect(screen.getByText('Curves of the best trial (3)')).toBeInTheDocument();
  });

  it('says that the test set waits until the end', () => {
    const base = detail();
    const run = detail({
      config: { ...base.config, test_split: 0.1 },
      state: { ...base.state!, test_size: 6000 },
    });
    renderDrawer({ training: controller({ view: 'run', selectedRunId: run.id, detail: run }) });
    expect(screen.getByText('Test accuracy').nextSibling).toHaveTextContent('–');
    expect(
      screen.getByText('6,000 test samples are held out until training has finished.')
    ).toBeInTheDocument();
  });

  it('lists searches and test results', () => {
    const training = controller({
      view: 'history',
      runs: [
        summary({
          id: 'r3',
          name: 'Search',
          status: 'completed',
          tuning: TUNING,
          metrics: {
            train_loss: 0.1,
            val_loss: 0.1,
            val_accuracy: 0.97,
            val_rmse: null,
            test_loss: 0.12,
            test_accuracy: 0.961,
            test_rmse: null,
          },
        }),
      ],
    });
    renderDrawer({ training });
    expect(screen.getByText(/MNIST · search · 4 trials/)).toBeInTheDocument();
    expect(screen.getByText('96.1%')).toBeInTheDocument();
    expect(screen.getByText('test accuracy')).toBeInTheDocument();
  });
});
