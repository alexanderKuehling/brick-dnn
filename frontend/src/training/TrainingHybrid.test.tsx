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
import { CLASSICAL_STATUS, STATUS } from './fixtures';
import { HYBRID_ANALYSIS, HYBRID_DOC, hybridDetail } from './hybridFixtures';
import {
  DEFAULT_SETTINGS,
  DEFAULT_SOURCE,
  saveSettings,
  toConfig,
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
    expect(screen.getByText(/temporary linear head on the target/)).toBeInTheDocument();
    for (const label of ['Epochs', 'Batch size', 'Learning rate']) {
      expect(screen.getByLabelText(label)).toBeInTheDocument();
    }
    expect(screen.getByRole('combobox', { name: 'Features' })).toHaveValue(
      'Learn them (temporary head on the target)'
    );
    // The search varies the classical model, not the training settings.
    await userEvent.click(
      screen.getByRole('switch', { name: /Hyperparameter search of Gradient boosting/ })
    );
    expect(screen.getByRole('checkbox', { name: 'Search learning rate' })).toBeChecked();
    expect(screen.queryByRole('checkbox', { name: 'Search batch size' })).not.toBeInTheDocument();

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
    saveSettings({ ...DEFAULT_SETTINGS, feature_training: 'frozen' });
    vi.mocked(api.training.runs.create).mockResolvedValue(hybridDetail({ status: 'starting' }));
    renderDrawer();
    await screen.findByText(/neural layers learn 3 features/);
    expect(screen.getByLabelText('Epochs')).toBeDisabled();
    expect(screen.getByText('Not used: the features stay random')).toBeInTheDocument();
    const start = screen.getByRole('button', { name: 'Start training' });
    await waitFor(() => expect(start).toBeEnabled());
    await userEvent.click(start);
    expect(vi.mocked(api.training.runs.create).mock.calls[0][0].config).toMatchObject({
      feature_training: 'frozen',
    });
  });

  it('needs PyTorch as well as scikit-learn', () => {
    renderDrawer({ training: controller({ status: CLASSICAL_STATUS }) });
    expect(screen.getByText('PyTorch is not installed on the server.')).toBeInTheDocument();
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
