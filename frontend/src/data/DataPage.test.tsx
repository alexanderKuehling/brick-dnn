/** The Data page with its hooks; the API and the editor/result components are stubbed. */
import { act, fireEvent, render, screen, userEvent, waitFor, within } from '@test-utils';
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom';
import { afterEach, beforeEach, vi } from 'vitest';
import { ModalsProvider } from '@mantine/modals';
import { notifications } from '@mantine/notifications';
import { api, ApiError, type Dataset, type DataSource, type PreviewResult, type User } from '@/api';
import { useAuth } from '@/auth/AuthContext';
import { downloadBlob } from '@/builder/download';
import type { AnalysisPanelProps } from './analysis/AnalysisPanel';
import { DataPage } from './DataPage';
import { CATALOG, DATA_STATUS, OTHER_SOURCE, PREP, PREVIEW, READINESS, SOURCE } from './fixtures';
import type { PipelineEditorProps } from './pipeline/PipelineEditor';
import type { ReadinessPanelProps } from './readiness/ReadinessPanel';
import type { DataTableProps } from './table/DataTable';

vi.mock('@/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api')>();
  return {
    ...actual,
    api: {
      data: {
        status: vi.fn(),
        steps: vi.fn(),
        sources: {
          list: vi.fn(),
          get: vi.fn(),
          upload: vi.fn(),
          rename: vi.fn(),
          remove: vi.fn(),
        },
        preps: {
          list: vi.fn(),
          get: vi.fn(),
          create: vi.fn(),
          update: vi.fn(),
          remove: vi.fn(),
          preview: vi.fn(),
          export: vi.fn(),
          download: vi.fn(),
        },
      },
    },
  };
});
vi.mock('@/auth/AuthContext', () => ({ useAuth: vi.fn() }));
vi.mock('@/builder/download', () => ({ downloadBlob: vi.fn(), downloadText: vi.fn() }));

// The editor and the result views have their own tests; these stubs expose their callbacks.
vi.mock('./pipeline/PipelineEditor', () => ({
  PipelineEditor: ({ steps, onStepsChange, settings, onSettingsChange }: PipelineEditorProps) => (
    <div>
      <p>Steps: {steps.map((step) => step.id).join(',')}</p>
      <p>Target: {String(settings.target)}</p>
      <button type="button" onClick={() => onStepsChange(steps.slice(1))}>
        Remove first step
      </button>
      <button type="button" onClick={() => onSettingsChange({ ...settings, target: 'plan' })}>
        Predict plan
      </button>
    </div>
  ),
}));
vi.mock('./table/DataTable', () => ({
  DataTable: ({ sample, onAddStep, onSetTarget, onShowColumn }: DataTableProps) => (
    <div>
      <p>Table of {sample.rows} rows</p>
      <button
        type="button"
        onClick={() => onAddStep({ type: 'drop_columns', params: { columns: ['notes'] } })}
      >
        Drop notes
      </button>
      <button type="button" onClick={() => onSetTarget('age')}>
        Use age as target
      </button>
      <button type="button" onClick={() => onShowColumn?.('income')}>
        Analyse income
      </button>
    </div>
  ),
}));
vi.mock('./analysis/AnalysisPanel', () => ({
  AnalysisPanel: ({ column, onRequestCompare }: AnalysisPanelProps) => (
    <div>
      <p>Analysis of {column ?? 'everything'}</p>
      <button type="button" onClick={onRequestCompare}>
        Compare
      </button>
    </div>
  ),
}));
vi.mock('./readiness/ReadinessPanel', () => ({
  ReadinessPanel: ({ readiness, exportBlocked, fixesDisabled }: ReadinessPanelProps) => (
    <>
      <p>{readiness.ready ? 'Ready to train' : 'Not ready'}</p>
      {exportBlocked && <p>Export blocked: {exportBlocked}</p>}
      {fixesDisabled && <p>Fixes wait for the preview</p>}
    </>
  ),
}));

const USER: User = { id: 1, username: 'ada', email: '', date_joined: '2026-01-01T00:00:00Z' };

const READY: PreviewResult = { ...PREVIEW, readiness: { ...READINESS, ready: true, issues: [] } };

const DATASET: Dataset = {
  id: 42,
  name: 'Churn model data',
  kind: 'csv',
  size_bytes: 51_200,
  created_at: '2026-11-07T11:00:00Z',
  info: {
    kind: 'csv',
    name: 'Churn model data',
    num_samples: 1200,
    builtin: null,
    sample_shape: null,
    train_size: null,
    test_size: null,
    num_features: 14,
    columns: [],
    has_header: true,
    delimiter: ',',
    target_kind: 'text',
    target_range: null,
    class_names: ['no', 'yes'],
    class_counts: null,
    subdir: null,
    num_classes: 2,
    predefined_split: [840, 180, 180],
  },
  prepared_from: { id: 10, name: 'Churn model data' },
  owner: 'ada',
  mine: true,
  shared_in: [],
};

function LocationProbe() {
  const location = useLocation();
  return (
    <output data-testid="location">
      {location.pathname}
      {location.state ? ` ${JSON.stringify(location.state)}` : ''}
    </output>
  );
}

function renderPage(path = '/data/10') {
  return render(
    <ModalsProvider>
      <MemoryRouter initialEntries={[path]}>
        <Routes>
          <Route path="/data/:prepId?" element={<DataPage />} />
          <Route path="*" element={<p>Another page</p>} />
        </Routes>
        <LocationProbe />
      </MemoryRouter>
    </ModalsProvider>
  );
}

const location = () => screen.getByTestId('location');

const originalMatchMedia = window.matchMedia;
/** Desktop: every `min-width` query matches; phone: none does. */
function setScreen(size: 'desktop' | 'phone') {
  window.matchMedia = vi.fn().mockImplementation((query: string) => ({
    matches: size === 'desktop' && query.includes('min-width'),
    media: query,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  }));
}

/** Waits until the preparation is open and previewed (the stubs show steps and rows). */
async function opened() {
  const steps = await screen.findByText('Steps: s1,s2,s3');
  await screen.findByText('Table of 60 rows');
  return steps;
}

beforeEach(() => {
  vi.clearAllMocks();
  localStorage.clear();
  setScreen('desktop');
  vi.spyOn(notifications, 'show').mockImplementation(() => 'id');
  vi.mocked(useAuth).mockReturnValue({
    user: USER,
    ready: true,
    login: vi.fn(),
    register: vi.fn(),
    logout: vi.fn(),
  });
  vi.mocked(api.data.status).mockResolvedValue(DATA_STATUS);
  vi.mocked(api.data.steps).mockResolvedValue(CATALOG);
  vi.mocked(api.data.sources.list).mockResolvedValue([SOURCE, OTHER_SOURCE]);
  vi.mocked(api.data.preps.list).mockResolvedValue([PREP]);
  vi.mocked(api.data.preps.get).mockResolvedValue(PREP);
  vi.mocked(api.data.preps.update).mockImplementation(async (_id, data) => ({ ...PREP, ...data }));
  vi.mocked(api.data.preps.preview).mockResolvedValue(PREVIEW);
});

afterEach(() => {
  window.matchMedia = originalMatchMedia;
});

describe('DataPage', () => {
  it('asks visitors to log in and brings them back afterwards', async () => {
    vi.mocked(useAuth).mockReturnValue({
      user: null,
      ready: true,
      login: vi.fn(),
      register: vi.fn(),
      logout: vi.fn(),
    });
    renderPage('/data/10');
    expect(await screen.findByText('Log in to prepare data')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Sign up' })).toHaveAttribute('href', '/register');
    await userEvent.click(screen.getByRole('link', { name: 'Log in' }));
    expect(location()).toHaveTextContent('/login {"from":"/data/10"}');
    expect(api.data.preps.list).not.toHaveBeenCalled();
  });

  it('explains why data preparation is not available', async () => {
    vi.mocked(api.data.status).mockResolvedValue({
      ...DATA_STATUS,
      available: false,
      reason: 'pandas is not installed on the server.',
    });
    renderPage();
    expect(await screen.findByText('Data preparation is not available')).toBeInTheDocument();
    expect(screen.getByText('pandas is not installed on the server.')).toBeInTheDocument();
    expect(screen.getByText('pip install -r requirements.txt')).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Check again' }));
    expect(api.data.status).toHaveBeenCalledTimes(2);
    expect(api.data.preps.get).not.toHaveBeenCalled();
  });

  it('imports a CSV file and opens a new preparation for it', async () => {
    vi.mocked(api.data.sources.list).mockResolvedValue([]);
    vi.mocked(api.data.preps.list).mockResolvedValue([]);
    vi.mocked(api.data.sources.upload).mockResolvedValue(SOURCE);
    vi.mocked(api.data.preps.create).mockResolvedValue(PREP);
    renderPage('/data');
    expect(await screen.findByText('Prepare data for training')).toBeInTheDocument();

    await userEvent.click(screen.getAllByRole('button', { name: 'Import CSV file' })[0]);
    const dialog = await screen.findByRole('dialog', { name: 'Import a CSV file' });
    const file = new File(['id;age\n1;34\n'], 'customers.csv', { type: 'text/csv' });
    await userEvent.upload(dialog.querySelector('input[type=file]') as HTMLInputElement, file);
    expect(within(dialog).getByText('customers.csv')).toBeInTheDocument();
    expect(within(dialog).getByLabelText('Name')).toHaveValue('customers');

    await userEvent.click(within(dialog).getByRole('button', { name: 'Import options' }));
    await userEvent.click(within(dialog).getByRole('combobox', { name: 'Separator' }));
    await userEvent.click(await screen.findByRole('option', { name: 'Semicolon ( ; )' }));
    await userEvent.click(within(dialog).getByRole('button', { name: 'Import' }));

    expect(api.data.sources.upload).toHaveBeenCalledWith(
      file,
      { delimiter: ';', name: 'customers' },
      expect.any(AbortSignal)
    );
    await waitFor(() =>
      expect(api.data.preps.create).toHaveBeenCalledWith({ name: 'customers', source: 1 })
    );
    await waitFor(() => expect(location()).toHaveTextContent('/data/10'));
    expect(await opened()).toBeInTheDocument();
    expect(notifications.show).toHaveBeenCalledWith(
      expect.objectContaining({ title: 'Imported “customers”' })
    );
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    expect(api.data.preps.preview).toHaveBeenCalledWith(
      10,
      expect.objectContaining({ steps: PREP.steps }),
      expect.any(AbortSignal)
    );
  });

  it('shows what the reader noticed, and can only add the table', async () => {
    vi.mocked(api.data.sources.upload).mockResolvedValue({
      ...OTHER_SOURCE,
      id: 3,
      name: 'orders',
      report: { ...OTHER_SOURCE.report, warnings: ['Line 17 has 4 cells instead of 3.'] },
    });
    renderPage();
    await opened();
    await userEvent.click(screen.getByRole('button', { name: 'Import CSV file' }));
    const dialog = await screen.findByRole('dialog', { name: 'Import a CSV file' });
    const file = new File(['a,b,c\n'], 'orders.csv', { type: 'text/csv' });
    await userEvent.upload(dialog.querySelector('input[type=file]') as HTMLInputElement, file);
    // Offered while a preparation is open: add the table to join it there.
    await userEvent.click(
      within(dialog).getByRole('checkbox', { name: /Start a new preparation/ })
    );
    await userEvent.click(within(dialog).getByRole('button', { name: 'Import' }));

    expect(
      await within(dialog).findByText('Line 17 has 4 cells instead of 3.')
    ).toBeInTheDocument();
    expect(api.data.preps.create).not.toHaveBeenCalled();
    await userEvent.click(within(dialog).getByRole('button', { name: 'Continue' }));
    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
    expect(screen.getByRole('list', { name: 'Imported tables' })).toHaveTextContent('orders');
    expect(location()).toHaveTextContent('/data/10');
  });

  /** Starts an import whose upload only finishes when `finish` is called. */
  async function startSlowImport() {
    const upload: { finish?: (source: DataSource) => void; signal?: AbortSignal } = {};
    vi.mocked(api.data.sources.upload).mockImplementation((_file, _options, signal) => {
      upload.signal = signal;
      return new Promise((resolve) => {
        upload.finish = resolve;
      });
    });
    vi.mocked(api.data.preps.create).mockResolvedValue({ ...PREP, id: 13, name: 'orders' });
    const view = renderPage();
    await opened();
    await userEvent.click(screen.getByRole('button', { name: 'Import CSV file' }));
    const dialog = await screen.findByRole('dialog', { name: 'Import a CSV file' });
    const file = new File(['a;b\n1;2\n'], 'orders.csv', { type: 'text/csv' });
    await userEvent.upload(dialog.querySelector('input[type=file]') as HTMLInputElement, file);
    await userEvent.click(within(dialog).getByRole('button', { name: 'Import' }));
    expect(await within(dialog).findByText(/Reading “orders.csv”/)).toBeInTheDocument();
    return { dialog, upload, view };
  }

  const ORDERS: DataSource = { ...OTHER_SOURCE, id: 3, name: 'orders' };

  it('cancels a running import and ignores its late answer', async () => {
    const { dialog, upload } = await startSlowImport();
    await userEvent.click(within(dialog).getByRole('button', { name: 'Cancel' }));
    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
    expect(upload.signal?.aborted).toBe(true);

    await act(async () => upload.finish?.(ORDERS));
    expect(api.data.preps.create).not.toHaveBeenCalled();
    expect(location()).toHaveTextContent('/data/10');
  });

  it('cancels a running import with Escape', async () => {
    const { upload } = await startSlowImport();
    await userEvent.keyboard('{Escape}');
    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
    expect(upload.signal?.aborted).toBe(true);
  });

  it('stops a running import when the page is left', async () => {
    const { upload, view } = await startSlowImport();
    view.unmount();
    expect(upload.signal?.aborted).toBe(true);
    await act(async () => upload.finish?.(ORDERS));
    expect(api.data.preps.create).not.toHaveBeenCalled();
    expect(notifications.show).not.toHaveBeenCalledWith(
      expect.objectContaining({ title: 'Imported “orders”' })
    );
  });

  it('imports a file dropped on the page', async () => {
    renderPage();
    const target = await opened();
    const file = new File(['a;b\n1;2\n'], 'orders.csv', { type: 'text/csv' });
    const dataTransfer = { types: ['Files'], files: [file], dropEffect: 'none' };
    fireEvent.dragEnter(target, { dataTransfer });
    expect(screen.getByText('Drop the CSV file to import it')).toBeInTheDocument();
    fireEvent.drop(target, { dataTransfer });
    expect(screen.queryByText('Drop the CSV file to import it')).not.toBeInTheDocument();
    const dialog = await screen.findByRole('dialog', { name: 'Import a CSV file' });
    expect(within(dialog).getByText('orders.csv')).toBeInTheDocument();
    expect(within(dialog).getByLabelText('Name')).toHaveValue('orders');
  });

  it('refuses files that are not CSV files', async () => {
    renderPage();
    await opened();
    await userEvent.click(screen.getByRole('button', { name: 'Import CSV file' }));
    const dialog = await screen.findByRole('dialog', { name: 'Import a CSV file' });
    const input = dialog.querySelector('input[type=file]') as HTMLInputElement;
    await userEvent.upload(input, new File(['x'], 'photo.png', { type: 'image/png' }), {
      applyAccept: false,
    });
    expect(within(dialog).getByText(/is not a CSV file/)).toBeInTheDocument();
    expect(within(dialog).getByRole('button', { name: 'Import' })).toBeDisabled();
  });

  it('opens the last used preparation', async () => {
    const second = { ...PREP, id: 12, name: 'Second try' };
    vi.mocked(api.data.preps.list).mockResolvedValue([PREP, second]);
    vi.mocked(api.data.preps.get).mockResolvedValue(second);
    localStorage.setItem('brickdnn:data-prep', '12');
    renderPage('/data');
    await waitFor(() => expect(location()).toHaveTextContent('/data/12'));
    expect(await screen.findByRole('textbox', { name: 'Preparation name' })).toHaveValue(
      'Second try'
    );
    expect(screen.getByRole('link', { name: /Second try/ })).toHaveAttribute(
      'aria-current',
      'page'
    );
  });

  it('previews an edit after a short pause and saves it', async () => {
    renderPage();
    await opened();
    expect(screen.getByText('Table of 60 rows')).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Remove first step' }));
    expect(screen.getByText('Steps: s2,s3')).toBeInTheDocument();
    // The last result stays visible while the new one is computed.
    expect(screen.getByText('Table of 60 rows')).toBeInTheDocument();
    expect(screen.getByText('Saving…')).toBeInTheDocument();

    await waitFor(() =>
      expect(api.data.preps.preview).toHaveBeenLastCalledWith(
        10,
        { steps: PREP.steps.slice(1), settings: PREP.settings, upto: null },
        expect.any(AbortSignal)
      )
    );
    await waitFor(
      () =>
        expect(api.data.preps.update).toHaveBeenCalledWith(10, {
          name: 'Churn model data',
          steps: PREP.steps.slice(1),
          settings: PREP.settings,
        }),
      { timeout: 2000 }
    );
    expect(await screen.findByText('Saved')).toBeInTheDocument();
  });

  it('adds steps, sets the target and opens columns from the result views', async () => {
    renderPage();
    await opened();
    await userEvent.click(screen.getByRole('button', { name: 'Drop notes' }));
    // Step 2 failed in the preview, so the new step goes in front of it, where it runs.
    expect(screen.getByText('Steps: s1,s4,s2,s3')).toBeInTheDocument();
    // Clicked twice before a preview ran it: added once.
    await userEvent.click(screen.getByRole('button', { name: 'Drop notes' }));
    expect(screen.getByText('Steps: s1,s4,s2,s3')).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Use age as target' }));
    expect(screen.getByText('Target: age')).toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: 'Analyse income' }));
    expect(screen.getByRole('tab', { name: 'Analysis' })).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByText('Analysis of income')).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Compare' }));
    await waitFor(() =>
      expect(api.data.preps.preview).toHaveBeenLastCalledWith(
        10,
        expect.objectContaining({ compare: true }),
        expect.any(AbortSignal)
      )
    );
  });

  it('shows the readiness problems instead of exporting data that is not ready', async () => {
    renderPage();
    await opened();
    const tab = screen.getByRole('tab', { name: 'Readiness, 4 problems' });
    await userEvent.click(screen.getByRole('button', { name: 'Create training dataset' }));
    expect(api.data.preps.export).not.toHaveBeenCalled();
    expect(tab).toHaveAttribute('aria-selected', 'true');
    expect(notifications.show).toHaveBeenCalledWith(
      expect.objectContaining({ title: 'Not ready for training yet' })
    );
  });

  it('blocks the export in the Readiness tab while a step failed', async () => {
    // An older check says ready, but step 2 failed in the preview.
    vi.mocked(api.data.preps.preview).mockResolvedValue(READY);
    renderPage();
    await opened();
    await userEvent.click(screen.getByRole('tab', { name: 'Readiness' }));
    expect(screen.getByText('Ready to train')).toBeInTheDocument();
    expect(screen.getByText('Export blocked: Step 2 “Clip values” failed.')).toBeInTheDocument();
    expect(screen.queryByText('Fixes wait for the preview')).not.toBeInTheDocument();

    // While the next preview is on its way, the fixes wait for it.
    await userEvent.click(screen.getByRole('button', { name: 'Remove first step' }));
    expect(screen.getByText('Fixes wait for the preview')).toBeInTheDocument();
  });

  it('creates a training dataset', async () => {
    vi.mocked(api.data.preps.preview).mockResolvedValue(READY);
    vi.mocked(api.data.preps.export).mockResolvedValue(DATASET);
    renderPage();
    await opened();
    await userEvent.click(screen.getByRole('button', { name: 'Create training dataset' }));
    await waitFor(() => expect(api.data.preps.export).toHaveBeenCalledWith(10));
    expect(notifications.show).toHaveBeenCalledWith(
      expect.objectContaining({ title: 'Training dataset created', color: 'green' })
    );
    expect(location()).toHaveTextContent('/data/10');
  });

  it('saves pending changes, creates the dataset and opens the training panel', async () => {
    vi.mocked(api.data.preps.preview).mockResolvedValue(READY);
    vi.mocked(api.data.preps.export).mockResolvedValue(DATASET);
    renderPage();
    await opened();
    await userEvent.click(screen.getByRole('button', { name: 'Remove first step' }));
    await userEvent.click(screen.getByRole('button', { name: 'Train with this data' }));

    await waitFor(() =>
      expect(location()).toHaveTextContent('/ {"openTraining":true,"dataset":42}')
    );
    const update = vi.mocked(api.data.preps.update);
    expect(update).toHaveBeenCalledWith(
      10,
      expect.objectContaining({ steps: PREP.steps.slice(1) })
    );
    expect(update.mock.invocationCallOrder[0]).toBeLessThan(
      vi.mocked(api.data.preps.export).mock.invocationCallOrder[0]
    );
    expect(JSON.parse(localStorage.getItem('brickdnn:training-source')!)).toEqual({
      tab: 'upload',
      builtin: null,
      upload: 42,
    });
    expect(notifications.show).toHaveBeenCalledWith(
      expect.objectContaining({
        message: '“Churn model data” is selected in the training panel (14 features, 2 classes).',
      })
    );
  });

  it('reports problems the server finds on export', async () => {
    vi.mocked(api.data.preps.preview).mockResolvedValue(READY);
    vi.mocked(api.data.preps.export).mockRejectedValue(
      new ApiError(
        400,
        { detail: 'The target column is missing.', issues: [] },
        'The target column is missing.'
      )
    );
    renderPage();
    await opened();
    await userEvent.click(screen.getByRole('button', { name: 'Train with this data' }));
    await waitFor(() =>
      expect(notifications.show).toHaveBeenCalledWith(
        expect.objectContaining({
          title: 'Not ready for training',
          message: 'The target column is missing.',
        })
      )
    );
    expect(screen.getByRole('tab', { name: /Readiness/ })).toHaveAttribute('aria-selected', 'true');
    expect(location()).toHaveTextContent('/data/10');
  });

  it('downloads the pandas script', async () => {
    const blob = new Blob(['import pandas as pd\n']);
    vi.mocked(api.data.preps.download).mockResolvedValue({ blob, filename: null });
    renderPage();
    await opened();
    await userEvent.click(screen.getByRole('button', { name: 'Download' }));
    await userEvent.click(await screen.findByRole('menuitem', { name: /Python script \(\.py\)/ }));
    await waitFor(() =>
      expect(downloadBlob).toHaveBeenCalledWith('churn_model_data_prepare.py', blob)
    );
    expect(api.data.preps.download).toHaveBeenCalledWith(10, 'python');
  });

  it('explains why a table that is in use cannot be deleted', async () => {
    vi.mocked(api.data.sources.remove).mockRejectedValue(
      new ApiError(409, null, 'Used by the preparations “Churn model data”; delete them first.')
    );
    renderPage();
    await opened();
    await userEvent.click(screen.getByRole('button', { name: 'Actions for customers' }));
    await userEvent.click(await screen.findByRole('menuitem', { name: 'Delete' }));
    const confirm = await screen.findByRole('dialog', { name: 'Delete “customers”?' });
    await userEvent.click(within(confirm).getByRole('button', { name: 'Delete' }));
    await waitFor(() =>
      expect(notifications.show).toHaveBeenCalledWith(
        expect.objectContaining({
          title: 'Cannot delete “customers”',
          message: 'Used by the preparations “Churn model data”; delete them first.',
        })
      )
    );
    expect(screen.getByRole('list', { name: 'Imported tables' })).toHaveTextContent('customers');
  });

  it('deletes the open preparation', async () => {
    vi.mocked(api.data.preps.remove).mockResolvedValue(undefined);
    renderPage();
    await opened();
    await userEvent.click(screen.getByRole('button', { name: 'Actions for Churn model data' }));
    await userEvent.click(await screen.findByRole('menuitem', { name: 'Delete' }));
    const confirm = await screen.findByRole('dialog', { name: 'Delete “Churn model data”?' });
    await userEvent.click(within(confirm).getByRole('button', { name: 'Delete' }));
    await waitFor(() => expect(location()).toHaveTextContent(/^\/data$/));
    expect(api.data.preps.remove).toHaveBeenCalledWith(10);
    expect(await screen.findByText('Prepare data for training')).toBeInTheDocument();
    expect(api.data.preps.update).not.toHaveBeenCalled();
  });

  it('duplicates a preparation with its steps', async () => {
    vi.mocked(api.data.preps.create).mockResolvedValue({ ...PREP, id: 13, name: 'Copy' });
    renderPage();
    await opened();
    await userEvent.click(screen.getByRole('button', { name: 'Actions for Churn model data' }));
    await userEvent.click(await screen.findByRole('menuitem', { name: 'Duplicate' }));
    await waitFor(() =>
      expect(api.data.preps.create).toHaveBeenCalledWith({
        name: 'Churn model data (copy)',
        source: 1,
        steps: PREP.steps,
        settings: PREP.settings,
      })
    );
    await waitFor(() => expect(location()).toHaveTextContent('/data/13'));
  });

  it('says when a preparation does not exist', async () => {
    vi.mocked(api.data.preps.get).mockRejectedValue(new ApiError(404, null, 'Not found.'));
    renderPage('/data/77');
    expect(await screen.findByText('Preparation not found')).toBeInTheDocument();
    await userEvent.click(screen.getByRole('link', { name: 'Show my preparations' }));
    await waitFor(() => expect(location()).toHaveTextContent('/data/10'));
  });

  it('describes the save state and the tables for keyboard and screen reader users', async () => {
    renderPage();
    await opened();
    // Below `sm` the save state is only an icon; its label is there for screen readers.
    expect(screen.getByRole('img', { name: 'Saved' })).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Remove first step' }));
    expect(screen.getByRole('img', { name: 'Saving…' })).toBeInTheDocument();

    // The details of a table (shown as a tooltip) can be reached with the keyboard.
    const tables = screen.getByRole('list', { name: 'Imported tables' });
    const row = within(tables).getByText('customers').closest('[tabindex]');
    expect(row).toHaveAttribute('tabindex', '0');
    expect(row).toHaveAccessibleDescription(
      /^customers\.csv, semicolon-separated · decimal comma · UTF-8 · header row, imported /
    );
  });

  it('shows the steps and the results as tabs on small screens', async () => {
    setScreen('phone');
    renderPage();
    const steps = await screen.findByRole('tab', { name: /Steps/ });
    expect(await screen.findByText('Table of 60 rows')).toBeInTheDocument();
    expect(screen.queryByText('Steps: s1,s2,s3')).not.toBeInTheDocument();
    await userEvent.click(steps);
    expect(await opened()).toBeInTheDocument();

    // The preparations and tables are in a drawer.
    expect(screen.queryByRole('list', { name: 'Imported tables' })).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Show preparations and tables' }));
    expect(await screen.findByRole('list', { name: 'Imported tables' })).toBeInTheDocument();
  });
});
