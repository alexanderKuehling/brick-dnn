/** The builder with a saved project: roles (view only), versions and save conflicts. */
import { render, screen, userEvent, waitFor, within } from '@test-utils';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, vi } from 'vitest';
import { notifications } from '@mantine/notifications';
import {
  api,
  ApiError,
  type GraphDocument,
  type LayerCatalog,
  type ParamSpec,
  type Project,
} from '@/api';
import { useAuth } from '@/auth/AuthContext';
import { useTraining, type TrainingController } from '@/training/useTraining';
import { BuilderContext, type BuilderContextValue } from './BuilderContext';
import { BuilderPage } from './BuilderPage';
import { saveDraft, type Draft } from './draft';
import { fromDocument } from './graph';
import { Inspector } from './Inspector';

vi.mock('@/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api')>();
  return {
    ...actual,
    api: {
      projects: {
        get: vi.fn(),
        create: vi.fn(),
        update: vi.fn(),
        duplicate: vi.fn(),
        members: { list: vi.fn() },
      },
    },
  };
});
vi.mock('@/auth/AuthContext', () => ({ useAuth: vi.fn() }));
vi.mock('@/training/useTraining', () => ({ useTraining: vi.fn() }));
vi.mock('./useAnalysis', () => ({
  useAnalysis: () => ({ analysis: null, loading: false, error: null }),
}));

const param = (over: Partial<ParamSpec> & Pick<ParamSpec, 'name' | 'label' | 'kind'>) => ({
  default: null,
  help: '',
  options: [],
  min: null,
  max: null,
  auto: false,
  length: null,
  show_if: null,
  ...over,
});

const CATALOG: LayerCatalog = {
  categories: ['Input', 'Basic'],
  layers: [
    {
      type: 'Input',
      label: 'Input',
      category: 'Input',
      description: 'Network input',
      params: [param({ name: 'shape', label: 'Shape', kind: 'ints', default: [4] })],
      min_inputs: 0,
      max_inputs: 0,
    },
    {
      type: 'Linear',
      label: 'Linear',
      category: 'Basic',
      description: 'Fully connected layer',
      params: [param({ name: 'out_features', label: 'Output features', kind: 'int', default: 10 })],
      min_inputs: 1,
      max_inputs: 1,
    },
  ],
};

vi.mock('./useCatalog', () => ({
  useCatalog: () => ({ catalog: CATALOG, examples: [], error: null, reload: vi.fn() }),
}));

const GRAPH: GraphDocument = {
  version: 1,
  nodes: [
    { id: 'in', type: 'Input', params: { shape: [4] }, position: { x: 0, y: 0 } },
    { id: 'fc', type: 'Linear', params: { out_features: 2 }, position: { x: 300, y: 0 } },
  ],
  edges: [{ source: 'in', target: 'fc' }],
};

function project(over: Partial<Project> = {}): Project {
  return {
    id: 5,
    name: 'Shared MLP',
    description: '',
    graph: GRAPH,
    num_layers: 2,
    created_at: '2026-10-01T09:00:00Z',
    updated_at: '2026-11-08T16:20:00Z',
    owner: { id: 2, username: 'maria' },
    role: 'editor',
    version: 3,
    updated_by: 'maria',
    shared: true,
    link_access: 'none',
    share_token: null,
    ...over,
  };
}

function openDraft(over: Partial<Draft> = {}) {
  saveDraft({
    doc: GRAPH,
    name: 'Shared MLP',
    projectId: 5,
    version: 3,
    role: 'editor',
    saved: true,
    ...over,
  });
}

function renderBuilder() {
  render(
    <MemoryRouter>
      <BuilderPage />
    </MemoryRouter>
  );
}

const nameInput = () => screen.getByRole('textbox', { name: 'Network name' });
const saveButton = () => screen.getByRole('button', { name: 'Save to account' });

beforeEach(() => {
  vi.clearAllMocks();
  localStorage.clear();
  vi.spyOn(notifications, 'show').mockImplementation(() => 'id');
  vi.mocked(useAuth).mockReturnValue({
    user: { id: 1, username: 'alex', email: '', date_joined: '2026-01-01T00:00:00Z' },
    ready: true,
    login: vi.fn(),
    register: vi.fn(),
    logout: vi.fn(),
  });
  vi.mocked(useTraining).mockReturnValue({
    open: vi.fn(),
    close: vi.fn(),
    opened: false,
    activeRun: null,
    data: { setSource: vi.fn(), reloadDatasets: vi.fn() },
  } as unknown as TrainingController);
  vi.mocked(api.projects.members.list).mockResolvedValue({
    owner: { id: 2, username: 'maria' },
    members: [{ id: 9, user: { id: 1, username: 'alex' }, role: 'editor', created_at: '' }],
  });
});

describe('Builder for viewers', () => {
  beforeEach(() => {
    openDraft({ role: 'viewer' });
    vi.mocked(api.projects.get).mockResolvedValue(project({ role: 'viewer' }));
  });

  it('is read only', async () => {
    renderBuilder();
    expect(await screen.findByText('View only')).toBeInTheDocument();
    expect(api.projects.get).toHaveBeenCalledWith(5);
    expect(screen.queryByRole('button', { name: 'Save to account' })).not.toBeInTheDocument();
    expect(screen.queryByRole('textbox', { name: 'Search layers' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Show layers' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Train' })).not.toBeInTheDocument();
    expect(nameInput()).toHaveAttribute('readonly');
    expect(screen.getByText('You can view this network, but not change it.')).toBeInTheDocument();
    // Viewing the code and exporting it still work.
    expect(screen.getByRole('button', { name: 'Export .py' })).toBeInTheDocument();
    // Who has access, from the members list.
    expect(
      await screen.findByRole('button', { name: /Shared: maria \(owner\), alex/ })
    ).toBeInTheDocument();
  });

  it('duplicates the network into the own account and continues with the copy', async () => {
    const copy = project({ id: 8, name: 'Shared MLP (copy)', role: 'owner', shared: false });
    vi.mocked(api.projects.duplicate).mockResolvedValue(copy);
    renderBuilder();
    await screen.findByText('View only');

    await userEvent.click(screen.getAllByRole('button', { name: 'Duplicate to my networks' })[0]);
    expect(api.projects.duplicate).toHaveBeenCalledWith(5);
    await waitFor(() => expect(nameInput()).toHaveValue('Shared MLP (copy)'));
    expect(screen.queryByText('View only')).not.toBeInTheDocument();
    expect(saveButton()).toBeInTheDocument();
    expect(screen.getByRole('textbox', { name: 'Search layers' })).toBeInTheDocument();
  });
});

describe('Inspector', () => {
  const node = fromDocument(GRAPH).nodes[1];
  const context = (readOnly: boolean): BuilderContextValue => ({
    specs: Object.fromEntries(CATALOG.layers.map((spec) => [spec.type, spec])),
    analysis: null,
    errorsByNode: {},
    warningsByNode: {},
    readOnly,
    updateNode: vi.fn(),
    deleteNode: vi.fn(),
    duplicateNode: vi.fn(),
    focusNode: vi.fn(),
  });

  it('shows the settings of viewers disabled', () => {
    render(
      <BuilderContext.Provider value={context(true)}>
        <Inspector node={node} nodes={[node]} />
      </BuilderContext.Provider>
    );
    expect(screen.getByRole('textbox', { name: 'Output features' })).toBeDisabled();
    expect(screen.getByRole('textbox', { name: 'Output features' })).toHaveValue('2');
    expect(screen.getByRole('textbox', { name: 'Name in code' })).toBeDisabled();
    expect(screen.queryByRole('button', { name: 'Delete layer' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Duplicate layer' })).not.toBeInTheDocument();
  });

  it('lets editors change the settings', () => {
    render(
      <BuilderContext.Provider value={context(false)}>
        <Inspector node={node} nodes={[node]} />
      </BuilderContext.Provider>
    );
    expect(screen.getByRole('textbox', { name: 'Output features' })).toBeEnabled();
    expect(screen.getByRole('button', { name: 'Delete layer' })).toBeInTheDocument();
  });

  it('shows the tips it is given instead of the ones for shared networks', () => {
    render(
      <BuilderContext.Provider value={context(true)}>
        <Inspector node={null} nodes={[node]} tips={['An earlier version, read only.']} />
      </BuilderContext.Provider>
    );
    expect(screen.getByText('An earlier version, read only.')).toBeInTheDocument();
    expect(screen.queryByText(/makes a copy that you can change/)).not.toBeInTheDocument();
  });

  it('tells viewers of shared networks how to get their own copy', () => {
    render(
      <BuilderContext.Provider value={context(true)}>
        <Inspector node={null} nodes={[node]} />
      </BuilderContext.Provider>
    );
    expect(screen.getByText(/makes a copy that you can change/)).toBeInTheDocument();
  });
});

describe('Saving a shared network', () => {
  const theirs = project({
    version: 4,
    name: 'Their version',
    updated_by: 'maria',
    updated_at: '2026-11-09T11:42:00Z',
  });

  async function saveAndConflict() {
    openDraft();
    vi.mocked(api.projects.get).mockResolvedValue(project());
    vi.mocked(api.projects.update).mockRejectedValueOnce(
      new ApiError(409, { detail: 'Changed by someone else.', project: theirs }, 'Changed.')
    );
    renderBuilder();
    await waitFor(() => expect(api.projects.get).toHaveBeenCalled());
    await userEvent.clear(nameInput());
    await userEvent.type(nameInput(), 'My version');
    await userEvent.click(saveButton());

    expect(api.projects.update).toHaveBeenCalledWith(5, {
      name: 'My version',
      graph: expect.objectContaining({ nodes: expect.any(Array) }),
      version: 3,
    });
    const dialog = await screen.findByRole('dialog', { name: 'This network was changed' });
    expect(
      within(dialog).getByText('maria changed this network since you opened it.')
    ).toBeInTheDocument();
    return dialog;
  }

  it('loads their version', async () => {
    const dialog = await saveAndConflict();
    await userEvent.click(within(dialog).getByRole('button', { name: 'Load their version' }));
    await waitFor(() => expect(nameInput()).toHaveValue('Their version'));
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();

    // Later saves are based on their version.
    vi.mocked(api.projects.update).mockResolvedValueOnce(project({ version: 5 }));
    await userEvent.type(nameInput(), '!');
    await userEvent.click(saveButton());
    expect(api.projects.update).toHaveBeenLastCalledWith(
      5,
      expect.objectContaining({ name: 'Their version!', version: 4 })
    );
  });

  it('keeps mine by saving over their version', async () => {
    const dialog = await saveAndConflict();
    vi.mocked(api.projects.update).mockResolvedValueOnce(
      project({ name: 'My version', version: 5, updated_by: 'alex' })
    );
    await userEvent.click(within(dialog).getByRole('button', { name: 'Keep mine' }));
    await waitFor(() => expect(api.projects.update).toHaveBeenCalledTimes(2));
    expect(api.projects.update).toHaveBeenLastCalledWith(
      5,
      expect.objectContaining({ name: 'My version', version: 4 })
    );
    expect(nameInput()).toHaveValue('My version');
    await waitFor(() =>
      expect(notifications.show).toHaveBeenLastCalledWith(
        expect.objectContaining({ message: 'Saved “My version”.' })
      )
    );
  });

  it('saves mine as a copy', async () => {
    const dialog = await saveAndConflict();
    vi.mocked(api.projects.create).mockResolvedValue(
      project({ id: 12, name: 'My version (copy)', role: 'owner', version: 1, shared: false })
    );
    await userEvent.click(within(dialog).getByRole('button', { name: 'Save mine as a copy' }));
    expect(api.projects.create).toHaveBeenCalledWith({
      name: 'My version (copy)',
      graph: expect.objectContaining({ nodes: expect.any(Array) }),
    });
    await waitFor(() => expect(nameInput()).toHaveValue('My version (copy)'));
    expect(api.projects.update).toHaveBeenCalledTimes(1);
  });
});

describe('Opening a saved network', () => {
  it('shows the latest version when there are no local changes', async () => {
    openDraft();
    vi.mocked(api.projects.get).mockResolvedValue(
      project({ name: 'Renamed by maria', version: 6 })
    );
    renderBuilder();
    await waitFor(() => expect(nameInput()).toHaveValue('Renamed by maria'));

    vi.mocked(api.projects.update).mockResolvedValue(project({ version: 7 }));
    await userEvent.click(saveButton());
    expect(api.projects.update).toHaveBeenCalledWith(5, expect.objectContaining({ version: 6 }));
  });

  it('keeps local changes, based on the version they started from', async () => {
    openDraft({ name: 'My unsaved name', saved: false });
    vi.mocked(api.projects.get).mockResolvedValue(project({ version: 6 }));
    renderBuilder();
    await waitFor(() =>
      expect(notifications.show).toHaveBeenCalledWith(
        expect.objectContaining({ title: 'maria changed “Shared MLP”' })
      )
    );
    expect(nameInput()).toHaveValue('My unsaved name');

    vi.mocked(api.projects.update).mockResolvedValue(project({ version: 7 }));
    await userEvent.click(saveButton());
    expect(api.projects.update).toHaveBeenCalledWith(5, expect.objectContaining({ version: 3 }));
  });

  it('detaches from a network the user can no longer open', async () => {
    openDraft();
    vi.mocked(api.projects.get).mockRejectedValue(new ApiError(404, null, 'Not found.'));
    renderBuilder();
    await waitFor(() =>
      expect(notifications.show).toHaveBeenCalledWith(
        expect.objectContaining({ title: 'No access any more' })
      )
    );

    // Saving now creates a new network in the own account.
    vi.mocked(api.projects.create).mockResolvedValue(project({ id: 20, role: 'owner' }));
    await userEvent.click(saveButton());
    expect(api.projects.create).toHaveBeenCalled();
    expect(api.projects.update).not.toHaveBeenCalled();
  });
});
