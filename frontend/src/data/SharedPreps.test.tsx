/** Preparations shared in the user's networks: listed apart, read-only for viewers. */
import { render, screen, userEvent, waitFor, within } from '@test-utils';
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom';
import { afterEach, beforeEach, vi } from 'vitest';
import { ModalsProvider } from '@mantine/modals';
import { notifications } from '@mantine/notifications';
import { api, type DataPrep, type User } from '@/api';
import { useAuth } from '@/auth/AuthContext';
import { DataPage } from './DataPage';
import { CATALOG, DATA_STATUS, OTHER_SOURCE, PREP, PREVIEW, SOURCE } from './fixtures';
import type { PipelineEditorProps } from './pipeline/PipelineEditor';
import { usePrepAccess } from './prepAccess';
import type { DataTableProps } from './table/DataTable';
import { AUTOSAVE_DELAY } from './useDataPrep';

vi.mock('@/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api')>();
  return {
    ...actual,
    api: {
      data: {
        status: vi.fn(),
        steps: vi.fn(),
        sources: { list: vi.fn(), get: vi.fn(), upload: vi.fn(), rename: vi.fn(), remove: vi.fn() },
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

// The editor and the result views have their own tests; these stubs show what they get.
vi.mock('./pipeline/PipelineEditor', () => ({
  PipelineEditor: ({ steps, onStepsChange, sources }: PipelineEditorProps) => {
    const { readOnly, sharedBy } = usePrepAccess();
    return (
      <div>
        <p>Steps: {steps.map((step) => step.id).join(',')}</p>
        <p>Editor: {readOnly ? 'read-only' : 'editable'}</p>
        <p>Shared by: {sharedBy ?? 'nobody'}</p>
        <p>Tables: {sources.map((source) => source.name).join(',')}</p>
        <button type="button" onClick={() => onStepsChange(steps.slice(1))}>
          Remove first step
        </button>
      </div>
    );
  },
}));
vi.mock('./table/DataTable', () => ({
  DataTable: ({ sample, onAddStep }: DataTableProps) => (
    <div>
      <p>Table of {sample.rows} rows</p>
      <button type="button" onClick={() => onAddStep({ type: 'drop_columns', params: {} })}>
        Drop notes
      </button>
    </div>
  ),
}));
vi.mock('./analysis/AnalysisPanel', () => ({ AnalysisPanel: () => <p>Analysis</p> }));
vi.mock('./readiness/ReadinessPanel', () => ({ ReadinessPanel: () => <p>Readiness</p> }));

const USER: User = { id: 1, username: 'ada', email: '', date_joined: '2026-01-01T00:00:00Z' };

const NETWORK = { id: 5, name: 'Churn net' };
const THEIR_SOURCE = { ...SOURCE, id: 31, name: 'bobs customers' };
const SHARED: DataPrep = {
  ...PREP,
  id: 20,
  name: 'Bob’s churn',
  source: THEIR_SOURCE.id,
  source_name: THEIR_SOURCE.name,
  owner: 'bob',
  role: 'viewer',
  shared_in: [NETWORK],
};
const EDITABLE: DataPrep = { ...SHARED, id: 21, name: 'Team churn', role: 'editor' };

function LocationProbe() {
  return <output data-testid="location">{useLocation().pathname}</output>;
}

function renderPage(path: string) {
  return render(
    <ModalsProvider>
      <MemoryRouter initialEntries={[path]}>
        <Routes>
          <Route path="/data/:prepId?" element={<DataPage />} />
        </Routes>
        <LocationProbe />
      </MemoryRouter>
    </ModalsProvider>
  );
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

const originalMatchMedia = window.matchMedia;

beforeEach(() => {
  vi.clearAllMocks();
  localStorage.clear();
  // Desktop: the sidebar, the editor and the results side by side.
  window.matchMedia = vi.fn().mockImplementation((query: string) => ({
    matches: query.includes('min-width'),
    media: query,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  }));
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
  vi.mocked(api.data.sources.get).mockResolvedValue(THEIR_SOURCE);
  vi.mocked(api.data.preps.list).mockResolvedValue([PREP, SHARED, EDITABLE]);
  vi.mocked(api.data.preps.get).mockImplementation(async (id) =>
    [PREP, SHARED, EDITABLE].find((prep) => prep.id === id)!
  );
  vi.mocked(api.data.preps.update).mockImplementation(async (id, data) => ({
    ...[PREP, SHARED, EDITABLE].find((prep) => prep.id === id)!,
    ...data,
  }));
  vi.mocked(api.data.preps.preview).mockResolvedValue(PREVIEW);
  vi.mocked(api.data.preps.download).mockResolvedValue({
    blob: new Blob(['a,b']),
    filename: 'churn.csv',
  });
});

afterEach(() => {
  window.matchMedia = originalMatchMedia;
});

describe('Shared preparations in the Data tab', () => {
  it('lists them apart from the own ones, with owner and role', async () => {
    renderPage('/data/10');
    const shared = await screen.findByRole('list', { name: 'Shared with you' });
    expect(within(shared).getByRole('link', { name: /Bob’s churn/ })).toHaveTextContent(
      'by bob · can view'
    );
    expect(within(shared).getByRole('link', { name: /Team churn/ })).toHaveTextContent(
      'by bob · can edit'
    );
    // Viewers have no actions; editors may only rename.
    expect(
      within(shared).queryByRole('button', { name: 'Actions for Bob’s churn' })
    ).not.toBeInTheDocument();
    await userEvent.click(within(shared).getByRole('button', { name: 'Actions for Team churn' }));
    expect(await screen.findByRole('menuitem', { name: 'Rename' })).toBeInTheDocument();
    expect(screen.queryByRole('menuitem', { name: 'Delete' })).not.toBeInTheDocument();
    expect(screen.queryByRole('menuitem', { name: 'Duplicate' })).not.toBeInTheDocument();
  });

  it('opens an own preparation first', async () => {
    vi.mocked(api.data.preps.list).mockResolvedValue([SHARED, PREP]);
    renderPage('/data');
    await waitFor(() => expect(screen.getByTestId('location')).toHaveTextContent('/data/10'));
  });

  it('shows viewers the data without anything that changes it', async () => {
    renderPage('/data/20');
    expect(await screen.findByText('Editor: read-only')).toBeInTheDocument();
    await screen.findByText('Table of 60 rows');
    expect(screen.getByText('View only')).toBeInTheDocument();
    expect(screen.getByText(/You can look at the data/)).toHaveTextContent(
      'Shared by bob in “Churn net”.'
    );
    expect(screen.getByRole('textbox', { name: 'Preparation name' })).toHaveAttribute('readonly');
    expect(screen.queryByRole('button', { name: /Train/ })).not.toBeInTheDocument();
    expect(
      screen.queryByRole('button', { name: 'Create training dataset' })
    ).not.toBeInTheDocument();
    // The main table belongs to bob: it is loaded on its own, combining is left to him.
    await waitFor(() => expect(screen.getByText('Tables: bobs customers')).toBeInTheDocument());
    expect(api.data.sources.get).toHaveBeenCalledWith(31, expect.any(AbortSignal));
    expect(screen.getByText('Shared by: bob')).toBeInTheDocument();

    // Nothing changes, nothing is saved.
    await userEvent.click(screen.getByRole('button', { name: 'Remove first step' }));
    await userEvent.click(screen.getByRole('button', { name: 'Drop notes' }));
    expect(screen.getByText('Steps: s1,s2,s3')).toBeInTheDocument();
    await userEvent.keyboard('{Control>}s{/Control}');
    await sleep(AUTOSAVE_DELAY + 200);
    expect(api.data.preps.update).not.toHaveBeenCalled();

    // Downloads work.
    await userEvent.click(screen.getByRole('button', { name: 'Download' }));
    await userEvent.click(await screen.findByRole('menuitem', { name: /Prepared data/ }));
    await waitFor(() => expect(api.data.preps.download).toHaveBeenCalledWith(20, 'csv'));
    expect(api.data.preps.update).not.toHaveBeenCalled();
  });

  it('lets editors change it', async () => {
    renderPage('/data/21');
    expect(await screen.findByText('Editor: editable')).toBeInTheDocument();
    expect(screen.queryByText('View only')).not.toBeInTheDocument();
    expect(screen.getByText(/your changes are saved for everyone/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Train/ })).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Remove first step' }));
    await waitFor(
      () =>
        expect(api.data.preps.update).toHaveBeenCalledWith(
          21,
          expect.objectContaining({ steps: PREP.steps.slice(1) })
        ),
      { timeout: AUTOSAVE_DELAY + 1000 }
    );
    // The source is never sent: editors cannot change it.
    expect(vi.mocked(api.data.preps.update).mock.calls[0][1]).not.toHaveProperty('source');
  });
});
