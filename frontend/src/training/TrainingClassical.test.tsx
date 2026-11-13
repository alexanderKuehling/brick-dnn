/** Training classical models (scikit-learn): fitted once, with their own settings and results. */
import { render, screen, userEvent, waitFor, within } from '@test-utils';
import { beforeEach, vi } from 'vitest';
import { ModalsProvider } from '@mantine/modals';
import { api, type Compatibility } from '@/api';
import { GP_ANALYSIS } from '@/builder/classicalFixtures';
import { downloadBlob } from '@/builder/download';
import { NO_SCOPE } from './access';
import {
  ANALYSIS,
  boostingDetail,
  CLASSICAL_STATUS,
  DOC,
  GP_DOC,
  gpDetail,
  summary,
  svcDetail,
  votingDetail,
} from './fixtures';
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
    status: CLASSICAL_STATUS,
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
        doc={GP_DOC}
        networkName="House prices GP"
        analysis={GP_ANALYSIS}
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
    summary: 'Demo data: random features and a hidden rule to learn (regression).',
    inputs: [],
    targets: [],
    image: null,
    num_classes: null,
    class_names: null,
    notes: [],
  },
};

const startButton = () => screen.getByRole('button', { name: 'Start training' });

beforeEach(() => {
  vi.clearAllMocks();
  localStorage.clear();
  vi.mocked(api.training.datasets.list).mockResolvedValue([]);
  vi.mocked(api.training.check).mockResolvedValue(READY);
});

describe('Setting up a classical model', () => {
  it('trains without PyTorch, with only the settings that apply', async () => {
    // A search left on for neural networks does not apply here.
    saveSettings({ ...DEFAULT_SETTINGS, search: true, seed: 7 });
    vi.mocked(api.training.runs.create).mockResolvedValue(gpDetail({ status: 'starting' }));
    renderDrawer();

    expect(await screen.findByText(/Demo data: random features/)).toBeInTheDocument();
    expect(screen.queryByText('PyTorch is not installed on the server.')).not.toBeInTheDocument();
    for (const label of ['Epochs', 'Batch size', 'Learning rate', 'Optimizer', 'Device']) {
      expect(screen.queryByLabelText(label)).not.toBeInTheDocument();
    }
    expect(screen.queryByText('Hyperparameter search')).not.toBeInTheDocument();
    expect(screen.queryByText('Advanced settings')).not.toBeInTheDocument();
    expect(screen.getByLabelText('Validation split')).toBeInTheDocument();
    expect(screen.getByText(/fitted on at most 5,000 training rows on this server/)).toBeVisible();
    expect(screen.getByText(/fitted once with scikit-learn/)).toBeInTheDocument();

    await waitFor(() => expect(startButton()).toBeEnabled());
    await userEvent.click(startButton());
    expect(api.training.runs.create).toHaveBeenCalledWith({
      name: 'House prices GP',
      graph: GP_DOC,
      dataset: { kind: 'random' },
      config: { val_split: 0.2, test_split: 0, max_samples: null, seed: 7 },
    });
  });

  it('caps Max. samples at the row limit of the model', async () => {
    saveSettings({ ...DEFAULT_SETTINGS, max_samples: 6000 });
    renderDrawer();
    expect(await screen.findByText('At most 5,000 on this server')).toBeInTheDocument();
    await screen.findByText(/Demo data: random features/);
    expect(startButton()).toBeDisabled();
  });

  it('shows why a classical model does not fit image data', async () => {
    const message =
      'Support vector classification is a classical model: it trains on tabular data (a CSV ' +
      'file or a dataset from the Data tab) or on demo data, not on images like MNIST.';
    vi.mocked(api.training.check).mockResolvedValue({
      ok: false,
      errors: [message],
      warnings: [],
      plan: null,
    });
    renderDrawer();
    expect(await screen.findByText(message)).toBeInTheDocument();
    expect(screen.getByText('Does not fit the data')).toBeInTheDocument();
    expect(startButton()).toBeDisabled();
  });

  it('needs PyTorch for neural networks on the same server', () => {
    renderDrawer({ doc: DOC, analysis: ANALYSIS });
    expect(screen.getByText('PyTorch is not installed on the server.')).toBeInTheDocument();
    expect(screen.getByText('pip install torch torchvision')).toBeInTheDocument();
  });

  it('explains how to install scikit-learn when it is missing', () => {
    const reason =
      'scikit-learn is not installed on the server, which classical models need. Install it ' +
      'with `pip install scikit-learn` and restart the backend.';
    renderDrawer({
      training: controller({
        status: {
          ...CLASSICAL_STATUS,
          available: true,
          reason: null,
          classical: { ...CLASSICAL_STATUS.classical!, available: false, reason },
        },
      }),
    });
    expect(screen.getByText(reason)).toBeInTheDocument();
    expect(screen.getByText('pip install -r requirements.txt')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Start training' })).not.toBeInTheDocument();
  });
});

describe('A run of a classical model', () => {
  const show = (run: ReturnType<typeof gpDetail>, active = false) =>
    renderDrawer({
      training: controller({
        view: 'run',
        selectedRunId: run.id,
        detail: run,
        activeRun: active ? run : null,
      }),
    });

  it('says what is being fitted, without epochs or curves', () => {
    const base = gpDetail();
    const run = gpDetail({
      status: 'running',
      message: 'Fitting the Gaussian process on 640 rows …',
      metrics: null,
      has_weights: false,
      state: {
        ...base.state!,
        status: 'running',
        history: [],
        test: null,
        classical: { ...base.state!.classical!, fit_seconds: null, kernel: null, train: null },
      },
    });
    show(run, true);
    expect(screen.getByText('Fitting the Gaussian process on 640 rows …')).toBeInTheDocument();
    expect(screen.getByText(/without progress in between/)).toBeInTheDocument();
    expect(screen.queryByRole('progressbar')).not.toBeInTheDocument();
    expect(screen.queryByText(/Epoch/)).not.toBeInTheDocument();
    expect(screen.getByText('The results appear when the fit is done.')).toBeInTheDocument();
    expect(screen.getByText('Rows fitted on').nextSibling).toHaveTextContent('640');
  });

  it('shows the fit, the kernel and the scores of a Gaussian process', async () => {
    const run = gpDetail();
    show(run);
    expect(screen.getByText('Fit time').nextSibling).toHaveTextContent('0.8 s');
    expect(screen.getByText('Log marginal likelihood').nextSibling).toHaveTextContent('-117.7');
    expect(screen.getByText('at most 5,000 on this server')).toBeInTheDocument();

    const table = screen.getByRole('table', { name: 'Results of the fitted model' });
    const rows = within(table).getAllByRole('row');
    expect(rows.map((row) => row.textContent)).toEqual([
      'RowsRMSER²',
      'Training · 6403.8990.994',
      'Validation · 1609.6820.965',
      'Test · 20010.210.951',
    ]);
    expect(screen.getByLabelText('Fitted kernel').textContent).toBe(
      '2.49**2 * RBF(length_scale=3.38)\n+ WhiteKernel(noise_level=0.0141)'
    );
    expect(screen.getByLabelText('Starting kernel')).toHaveTextContent('ConstantKernel');
    // Nothing about epochs, weights or loss curves.
    expect(screen.queryByRole('slider', { name: /^Loss/ })).not.toBeInTheDocument();
    expect(screen.queryByText('Best epoch')).not.toBeInTheDocument();
    expect(screen.queryByText(/Weights from epoch/)).not.toBeInTheDocument();
  });

  it('downloads the model as a joblib file and explains how to use it', async () => {
    const blob = new Blob(['model']);
    vi.mocked(api.training.runs.downloadWeights).mockResolvedValue({
      blob,
      filename: 'House prices GP.joblib',
    });
    show(gpDetail());
    await userEvent.click(screen.getByRole('button', { name: 'Model (.joblib)' }));
    expect(downloadBlob).toHaveBeenCalledWith('house_prices_gp.joblib', blob);

    await userEvent.click(screen.getByRole('button', { name: 'How to use the model' }));
    expect(
      screen.getByText('python house_prices_gp.py --predict new.csv --save house_prices_gp.joblib')
    ).toBeInTheDocument();
    const snippet = [...document.querySelectorAll('pre')].map((pre) => pre.textContent).join('\n');
    expect(snippet).toContain('bundle = joblib.load("house_prices_gp.joblib")');
    expect(snippet).toContain('mean, std = model.predict(rows, return_std=True)');
    expect(snippet).not.toContain('load_state_dict');
  });

  it('shows the support vectors and accuracy of an SVM', () => {
    show(svcDetail());
    expect(screen.getByText('Support vectors').nextSibling).toHaveTextContent('200');
    expect(screen.getByText('25% of the rows')).toBeInTheDocument();
    expect(screen.getByText('subsampled from 1,600 (limit 800)')).toBeInTheDocument();
    const table = screen.getByRole('table', { name: 'Results of the fitted model' });
    expect(
      within(table)
        .getAllByRole('row')
        .map((row) => row.textContent)
    ).toEqual(['RowsAccuracyLoss', 'Training · 80095.0%0.2', 'Validation · 20093.0%0.25']);
    expect(screen.queryByLabelText('Fitted kernel')).not.toBeInTheDocument();
  });

  it('shows the boosting iterations of gradient boosting', () => {
    show(boostingDetail());
    expect(screen.getByText('Boosting iterations').nextSibling).toHaveTextContent('87');
    expect(screen.getByText('of at most 200: early stopping')).toBeInTheDocument();
    expect(screen.getByText('at most 1,000,000 on this server')).toBeInTheDocument();
    expect(screen.queryByText('Support vectors')).not.toBeInTheDocument();
    expect(screen.queryByText('Log marginal likelihood')).not.toBeInTheDocument();
  });

  it('shows the models of an ensemble with their own scores', () => {
    show(votingDetail());
    expect(screen.getByText('Ensemble').nextSibling).toHaveTextContent('2 models');
    expect(screen.queryByText('Support vectors')).not.toBeInTheDocument();
    const table = screen.getByRole('table', { name: 'Models of the ensemble' });
    expect(
      within(table)
        .getAllByRole('row')
        .map((row) => row.textContent)
    ).toEqual([
      'ModelAccuracyLoss',
      'gb · Gradient boosting classification90.0%0.3',
      'svc · Support vector classification88.0%0.3',
    ]);
    expect(
      screen.getByText(/Each model of the ensemble on its own, on the validation/)
    ).toBeVisible();
  });

  it('warns that stopping ends the fit at once', async () => {
    const run = gpDetail({ status: 'running' });
    show(run, true);
    await userEvent.click(screen.getByRole('button', { name: 'Stop' }));
    expect(await screen.findByText(/scikit-learn cannot pause a fit/)).toBeInTheDocument();
  });

  it('lists classical runs with R² or accuracy', () => {
    renderDrawer({
      training: controller({
        view: 'history',
        runs: [
          summary({ ...gpDetail(), id: 'gp', status: 'completed' }),
          summary({ ...svcDetail(), id: 'svc', status: 'completed' }),
          summary({ id: 'fit', name: 'Fitting now', status: 'running', family: 'classical' }),
        ],
      }),
    });
    expect(screen.getByText('0.951')).toBeInTheDocument();
    expect(screen.getByText('test R²')).toBeInTheDocument();
    expect(screen.getByText('93.0%')).toBeInTheDocument();
    expect(screen.getByText('Fitting')).toBeInTheDocument();
  });
});
