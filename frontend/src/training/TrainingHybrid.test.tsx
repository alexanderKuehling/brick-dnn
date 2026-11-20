/**
 * Training hybrid networks: neural layers learn the features of a classical model. The
 * settings have both parts; the run view shows the curves of the features (phase 1) and the
 * results of the classical model (phase 2).
 */
import { render, screen, userEvent, waitFor } from '@test-utils';
import { beforeEach, vi } from 'vitest';
import { ModalsProvider } from '@mantine/modals';
import { api, type ClassicalTuningLimits, type Compatibility } from '@/api';
import { downloadBlob } from '@/builder/download';
import { NO_SCOPE } from './access';
import { CLASSICAL_STATUS, STATUS, summary } from './fixtures';
import { describeParams, paramLabel, progressLabel, searchedParams } from './format';
import { HYBRID_ANALYSIS, HYBRID_DOC, hybridDetail } from './hybridFixtures';
import {
  DEFAULT_SETTINGS,
  DEFAULT_SOURCE,
  NEURAL_SEARCH_NO_VALIDATION,
  saveSettings,
  toConfig,
  toTuning,
  validateSettings,
} from './settings';
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
vi.mock('@/builder/download', () => ({ downloadBlob: vi.fn(), downloadText: vi.fn() }));

const TUNING: ClassicalTuningLimits = { models: {}, cv_folds: { min: 2, max: 10, default: 5 } };
// A server with PyTorch and scikit-learn (a hybrid network needs both).
const BOTH = { ...STATUS, classical: { ...CLASSICAL_STATUS.classical!, tuning: TUNING } };

function dataController(): DataController {
  return {
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
}

function controller(overrides: Partial<TrainingController> = {}): TrainingController {
  return {
    status: BOTH,
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
    data: dataController(),
    ...overrides,
  };
}

function renderDrawer(props: Partial<TrainingDrawerProps> = {}) {
  render(
    <ModalsProvider>
      <TrainingDrawer
        training={controller()}
        loggedIn
        doc={HYBRID_DOC}
        networkName="MLP features + trees"
        analysis={HYBRID_ANALYSIS}
        onLogin={vi.fn()}
        onShowProblems={vi.fn()}
        onLoadNetwork={vi.fn()}
        {...props}
      />
    </ModalsProvider>
  );
}

const READY: Compatibility = {
  ok: true,
  errors: [],
  warnings: [],
  plan: {
    task: 'demo',
    summary:
      'Demo data: random inputs and a hidden rule to learn (classification): neural layers learn 3 features, HistGradientBoostingClassifier is fitted on them.',
    inputs: [{ shape: [4], dtype: 'float', vocab: null }],
    targets: [{ kind: 'class', num_classes: 3 }],
    image: null,
    num_classes: 3,
    class_names: null,
    notes: [],
  },
};

beforeEach(() => {
  vi.clearAllMocks();
  localStorage.clear();
  vi.mocked(api.training.datasets.list).mockResolvedValue([]);
  vi.mocked(api.training.check).mockResolvedValue(READY);
});

describe('Setting up a hybrid network', () => {
  it('has the network settings for the features and the search of the classical model', async () => {
    saveSettings({ ...DEFAULT_SETTINGS, epochs: 5, seed: 3 });
    vi.mocked(api.training.runs.create).mockResolvedValue(hybridDetail({ status: 'starting' }));
    renderDrawer();

    expect(await screen.findByText(/neural layers learn 3 features/)).toBeInTheDocument();
    // The two phases are explained next to the heading of the network settings.
    await userEvent.click(screen.getByRole('button', { name: 'More about “Model and optimizer”' }));
    expect(await screen.findByText(/temporary linear head on the target/)).toBeVisible();
    await userEvent.keyboard('{Escape}');
    for (const label of ['Epochs', 'Batch size', 'Learning rate']) {
      expect(screen.getByLabelText(label)).toBeInTheDocument();
    }
    expect(screen.getByRole('combobox', { name: 'Features' })).toHaveValue(
      'Learn them (temporary head on the target)'
    );
    // The search varies the classical model; the training settings only when asked.
    await userEvent.click(screen.getByRole('switch', { name: /Hyperparameter search/ }));
    expect(screen.getByRole('checkbox', { name: 'Search learning rate' })).toBeChecked();
    expect(screen.queryByRole('checkbox', { name: 'Search batch size' })).not.toBeInTheDocument();
    expect(
      screen.getByRole('checkbox', { name: /Search the training settings of the neural layers/ })
    ).not.toBeChecked();
    expect(screen.getByRole('textbox', { name: 'Learning rate' })).toBeEnabled();

    const start = screen.getByRole('button', { name: 'Start search' });
    await waitFor(() => expect(start).toBeEnabled());
    await userEvent.click(start);
    const body = vi.mocked(api.training.runs.create).mock.calls[0][0];
    expect(body.config).toMatchObject({ epochs: 5, seed: 3, feature_training: 'pretrain' });
    expect(body.config?.learning_rate).toBe(0.001); // the network's, not searched
    expect(body.tuning).toMatchObject({ method: 'random', trials: 10 });
    expect(Object.keys(body.tuning?.space ?? {})).toEqual(
      expect.arrayContaining(['learning_rate', 'max_leaf_nodes', 'min_samples_leaf'])
    );
  });

  it('can keep random features instead of learning them', async () => {
    // Even an invalid learning rate does not matter: it is not used.
    saveSettings({ ...DEFAULT_SETTINGS, feature_training: 'frozen', learning_rate: 0 });
    vi.mocked(api.training.runs.create).mockResolvedValue(hybridDetail({ status: 'starting' }));
    renderDrawer();
    await screen.findByText(/neural layers learn 3 features/);
    expect(screen.getByLabelText('Epochs')).toBeDisabled();
    expect(screen.getByText('Not used: the features stay random')).toBeInTheDocument();
    // Every training setting of the neural layers, and why.
    for (const label of ['Batch size', 'Learning rate', 'Weight decay', 'Early stopping']) {
      expect(screen.getByLabelText(label)).toBeDisabled();
    }
    expect(screen.getByRole('combobox', { name: 'Optimizer' })).toBeDisabled();
    expect(screen.getByLabelText(/Keep the weights of the best/)).toBeDisabled();
    expect(screen.getByText(/Nothing is trained: the training settings below/)).toBeVisible();
    expect(screen.getAllByText('Not used: frozen features are not trained').length).toBeGreaterThan(
      3
    );
    // The device and the seed still apply: the features are computed with the random weights.
    await userEvent.click(screen.getByRole('button', { name: 'Advanced settings' }));
    expect(await screen.findByRole('combobox', { name: 'Device' })).toBeEnabled();
    const start = screen.getByRole('button', { name: 'Start training' });
    await waitFor(() => expect(start).toBeEnabled());
    await userEvent.click(start);
    const config = vi.mocked(api.training.runs.create).mock.calls[0][0].config;
    expect(config).toMatchObject({ feature_training: 'frozen' });
    for (const key of ['epochs', 'batch_size', 'learning_rate', 'optimizer', 'patience']) {
      expect(config).not.toHaveProperty(key);
    }
  });

  it('can search the training settings of the neural layers too', async () => {
    saveSettings({ ...DEFAULT_SETTINGS, epochs: 5, search: true, search_method: 'grid' });
    vi.mocked(api.training.runs.create).mockResolvedValue(hybridDetail({ status: 'starting' }));
    renderDrawer();
    await screen.findByText(/neural layers learn 3 features/);
    await userEvent.click(
      screen.getByRole('checkbox', { name: /Search the training settings of the neural layers/ })
    );
    // The rows of the neural settings (the learning rate of the boosting has its own).
    expect(screen.getByRole('checkbox', { name: 'Search neural learning rate' })).toBeChecked();
    expect(screen.getByRole('checkbox', { name: 'Search neural batch size' })).toBeChecked();
    expect(screen.getByRole('checkbox', { name: 'Search learning rate' })).toBeChecked();
    expect(screen.getByRole('textbox', { name: 'Learning rate' })).toBeDisabled(); // searched
    expect(screen.getByText(/compared on the validation rows: the features/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole('checkbox', { name: 'Search learning rate' }));
    await userEvent.click(screen.getByRole('checkbox', { name: 'Search max. leaves per tree' }));
    await userEvent.click(screen.getByRole('checkbox', { name: 'Search min. rows per leaf' }));
    // 3 learning rates × 3 batch sizes, nothing of the model.
    expect(screen.getByText(/^9 trials: each learns the features \(5 epochs\)/)).toBeVisible();

    const start = screen.getByRole('button', { name: 'Start search' });
    await waitFor(() => expect(start).toBeEnabled());
    await userEvent.click(start);
    const body = vi.mocked(api.training.runs.create).mock.calls[0][0];
    expect(body.tuning).toEqual({
      method: 'grid',
      space: {},
      neural_space: { learning_rate: [0.0001, 0.001, 0.01], batch_size: [32, 64, 128] },
    });
    // Left out: the search sets them.
    expect(body.config?.learning_rate).toBeUndefined();
    expect(body.config?.batch_size).toBeUndefined();
    expect(body.config).toMatchObject({ epochs: 5, feature_training: 'pretrain' });
  });

  it('needs PyTorch as well as scikit-learn', () => {
    renderDrawer({ training: controller({ status: CLASSICAL_STATUS }) });
    expect(screen.getByText('PyTorch is not installed on the server.')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Start training' })).not.toBeInTheDocument();
  });

  it('names scikit-learn when only it is missing', () => {
    const reason = 'scikit-learn is not installed on the server.';
    const status = {
      ...BOTH,
      classical: { ...BOTH.classical!, available: false, reason, sklearn_version: null },
    };
    renderDrawer({ training: controller({ status }) });
    expect(screen.getByText(reason)).toBeInTheDocument();
    expect(screen.getByText(/This network is hybrid: its neural layers learn/)).toBeVisible();
    expect(screen.getByText('pip install -r requirements.txt')).toBeInTheDocument();
    expect(screen.queryByText('pip install torch torchvision')).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Start training' })).not.toBeInTheDocument();
  });

  it('validates the network settings and the classical search together', () => {
    const values = { ...DEFAULT_SETTINGS, epochs: '', search: true };
    const errors = validateSettings(values, {
      hybrid: { model: 'HistGradientBoostingClassifier' },
    });
    expect(errors.epochs).toBe('Required');
    // A search of the classical model may cross-validate: no validation data is needed.
    expect(
      validateSettings(
        { ...DEFAULT_SETTINGS, val_split: 0, search: true },
        { hybrid: { model: 'SVC' } }
      ).search
    ).toBeUndefined();
    expect(toConfig(DEFAULT_SETTINGS, { hybrid: true })).toMatchObject({
      epochs: 10,
      feature_training: 'pretrain',
    });
  });

  it('validates a search of the neural settings', () => {
    const neural = { ...DEFAULT_SETTINGS, search: true, search_neural: true };
    const hybrid = { model: 'SVC' };
    expect(validateSettings(neural, { hybrid })).toEqual({});
    // It needs validation rows (no cross-validation), and some values.
    expect(validateSettings({ ...neural, val_split: 0 }, { hybrid }).search).toBe(
      NEURAL_SEARCH_NO_VALIDATION()
    );
    expect(validateSettings({ ...neural, batch_sizes: [] }, { hybrid }).batch_sizes).toBe(
      'Choose at least one'
    );
    // Neural settings alone are enough; with nothing ticked at all it says so.
    const none = {
      ...neural,
      classical_params: { SVC: { C: { on: false }, gamma: { on: false } } },
    };
    expect(validateSettings(none, { hybrid })).toEqual({});
    expect(
      validateSettings({ ...none, tune_learning_rate: false, tune_batch_size: false }, { hybrid })
        .search
    ).toBe('Choose at least one setting or parameter to search.');
    // A grid of both parts has at most 100 trials.
    const grid = {
      ...neural,
      search_method: 'grid' as const,
      lr_values: [0.1, 0.01, 0.001, 0.0001, 0.00001],
      batch_sizes: [8, 16, 32, 64, 128],
      classical_params: {
        SVC: {
          C: { on: true, mode: 'list' as const, text: '1, 2, 3, 4, 5' },
          gamma: { on: false },
        },
      },
    };
    expect(validateSettings(grid, { hybrid }).search).toBe(
      'The grid has 125 combinations, at most 100 are possible. Choose fewer values.'
    );
    // Cross-validation is not sent with the neural settings; frozen features have none.
    const tuning = toTuning({ ...neural, cv_enabled: true }, 'SVC', { hybrid: true });
    expect(tuning).not.toHaveProperty('cv_folds');
    expect(tuning?.neural_space).toEqual({
      learning_rate: { min: 0.0001, max: 0.01 },
      batch_size: [32, 64, 128],
    });
    expect(
      toTuning({ ...neural, feature_training: 'frozen' }, 'SVC', { hybrid: true })
    ).not.toHaveProperty('neural_space');
  });
});

describe('The progress of a hybrid run', () => {
  const running = (overrides = {}) =>
    summary({ status: 'running', family: 'hybrid', epochs: 10, ...overrides });
  const search = { method: 'random' as const, trials: 4, space: { C: { min: 1, max: 10 } } };

  it('learns the features, then fits or searches the classical model', () => {
    expect(progressLabel(running({ epochs_done: 2 }))).toBe('Epoch 3/10');
    expect(progressLabel(running({ epochs_done: 10 }))).toBe('Fitting');
    // A search of the classical model: the features first, then its trials.
    expect(progressLabel(running({ tuning: search, epochs_done: 2 }))).toBe('Epoch 3/10');
    expect(progressLabel(running({ tuning: search, epochs_done: 10 }))).toBe('Trial 1/4');
    // Early stopping ended phase 1 before its last epoch: the trials count.
    expect(progressLabel(running({ tuning: search, epochs_done: 6, trials_done: 2 }))).toBe(
      'Trial 3/4'
    );
    // A search of the neural settings learns the features in every trial.
    const neural = { ...search, space: {}, neural_space: { batch_size: [16, 32] } };
    expect(progressLabel(running({ tuning: neural, epochs_done: 2, trials_done: 1 }))).toBe(
      'Trial 2/4'
    );
  });

  it('names the neural settings of the trials', () => {
    const tuning = {
      method: 'grid' as const,
      trials: 4,
      space: { C: [1, 10] },
      neural_space: { batch_size: [16, 32], learning_rate: [0.001] },
    };
    expect(searchedParams(tuning)).toEqual(['neural.learning_rate', 'neural.batch_size', 'C']);
    expect(paramLabel('neural.batch_size')).toBe('Neural batch size');
    expect(describeParams({ 'neural.learning_rate': 0.001, 'neural.batch_size': 32, C: 10 })).toBe(
      'neural layers: learning rate 0.001, batch size 32; C 10'
    );
    expect(describeParams({ 'neural.optimizer': 'sgd' })).toBe('neural layers: SGD');
  });
});

describe('A run of a hybrid network', () => {
  const show = (run: ReturnType<typeof hybridDetail>) =>
    renderDrawer({
      training: controller({ view: 'run', selectedRunId: run.id, detail: run }),
    });

  it('shows the curves of the features and the results of the classical model', () => {
    show(hybridDetail());
    expect(screen.getByText('Phase 1 · The neural layers learn the features')).toBeInTheDocument();
    expect(
      screen.getByText(/They learn 3 features with a temporary linear head/)
    ).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Values per epoch' })).toBeInTheDocument();
    expect(
      screen.getByText('Phase 2 · Gradient boosting classification on the learned features')
    ).toBeInTheDocument();
    expect(screen.getByText('Boosting iterations').nextSibling).toHaveTextContent('50');
    const table = screen.getByRole('table', { name: 'Results of the fitted model' });
    expect(table).toHaveTextContent('91.0%');
    expect(table).toHaveTextContent('97.0%');
  });

  it('learns the features before the trials of a search of the classical model', () => {
    const tuning = {
      method: 'random' as const,
      trials: 4,
      space: { learning_rate: { min: 0.01, max: 0.3 } },
      metric: 'loss' as const,
      cv_folds: null,
    };
    const run = hybridDetail({ status: 'running', tuning, epochs_done: 1, has_weights: false });
    const state = { ...run.state!, status: 'running' as const, trial: null, trials: [] };
    show({ ...run, state: { ...state, history: state.history.slice(0, 1), trials_total: 4 } });
    expect(screen.getByText('Epoch 2 of 3')).toBeInTheDocument();
    expect(screen.queryByText(/Trial 1 of 4/)).not.toBeInTheDocument();
  });

  it('shows the trials of a search of the neural settings', () => {
    const tuning = {
      method: 'grid' as const,
      trials: 2,
      space: {},
      neural_space: { batch_size: [16, 32] },
      metric: 'loss' as const,
      cv_folds: null,
    };
    const trial = (number: number, status: 'completed' | 'running', epochs: number) => ({
      trial: number,
      params: { 'neural.batch_size': 16 * number },
      status,
      message: '',
      epochs_done: epochs,
      train_loss: null,
      val_loss: status === 'completed' ? 0.3 : null,
      val_accuracy: status === 'completed' ? 0.9 : null,
      val_rmse: null,
      score: status === 'completed' ? 0.3 : null,
      folds: null,
      seconds: status === 'completed' ? 1.5 : null,
    });
    const run = hybridDetail({ status: 'running', tuning, trials_done: 1, has_weights: false });
    show({
      ...run,
      state: {
        ...run.state!,
        status: 'running',
        tuning,
        trial: 2,
        trials_total: 2,
        best_trial: 1,
        history: run.state!.history.slice(0, 3),
        trials: [trial(1, 'completed', 3), trial(2, 'running', 3)],
      },
    });
    const table = screen.getByRole('table', { name: 'Trials' });
    expect(table).toHaveTextContent('Neural batch size');
    expect(table).toHaveTextContent('Fitting'); // its features are learned
    expect(screen.getByText(/Best so far: trial 1 \(neural layers: batch size 16\)/)).toBeVisible();
  });

  it('says that frozen features were not trained', () => {
    const run = hybridDetail({ config: { ...hybridDetail().config, feature_training: 'frozen' } });
    show({ ...run, state: { ...run.state!, history: [] } });
    expect(screen.getByText(/Frozen random features: the neural layers keep/)).toBeInTheDocument();
  });

  it('downloads both parts as a ZIP file and explains it', async () => {
    vi.mocked(api.training.runs.downloadWeights).mockResolvedValue({
      blob: new Blob(['zip']),
      filename: 'mlp-features-trees.zip',
    });
    show(hybridDetail());
    await userEvent.click(screen.getByRole('button', { name: 'Model (.zip)' }));
    await waitFor(() =>
      expect(downloadBlob).toHaveBeenCalledWith('mlp_features_trees.zip', expect.any(Blob))
    );
    await userEvent.click(screen.getByRole('button', { name: 'How to use the weights' }));
    expect(
      screen.getByText(
        'python mlp_features_trees.py --predict new.csv --weights mlp_features_trees.features.pt --save mlp_features_trees.joblib'
      )
    ).toBeInTheDocument();
  });
});
