/** The builder with a live connection to the open project (fake WebSocket as the server). */
import {
  act,
  FakeWebSocket,
  fireEvent,
  render,
  screen,
  userEvent,
  waitFor,
  within,
} from '@test-utils';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, vi } from 'vitest';
import { notifications } from '@mantine/notifications';
import {
  api,
  setTokens,
  type GraphDocument,
  type LayerCatalog,
  type ParamSpec,
  type Project,
} from '@/api';
import { useAuth } from '@/auth/AuthContext';
import { useTraining, type TrainingController } from '@/training/useTraining';
import { BuilderPage } from './BuilderPage';
import { loadDraft, saveDraft, type Draft } from './draft';

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
    {
      type: 'Dropout',
      label: 'Dropout',
      category: 'Basic',
      description: 'Randomly zeroes values',
      params: [param({ name: 'p', label: 'Drop probability', kind: 'float', default: 0.5 })],
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

const MARIA = { id: 2, username: 'maria' };
const ALEX = { id: 1, username: 'alex' };

function project(over: Partial<Project> = {}): Project {
  return {
    id: 5,
    name: 'Shared MLP',
    description: '',
    graph: GRAPH,
    num_layers: 2,
    created_at: '2026-10-01T09:00:00Z',
    updated_at: '2026-11-08T16:20:00Z',
    owner: MARIA,
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

function hello(
  over: { name?: string; graph?: GraphDocument; version?: number; role?: string } = {}
) {
  return {
    type: 'hello',
    project: {
      id: 5,
      name: over.name ?? 'Shared MLP',
      description: '',
      graph: over.graph ?? GRAPH,
      version: over.version ?? 3,
    },
    role: over.role ?? 'editor',
    client_id: 'me',
    user: ALEX,
  };
}

function renderBuilder() {
  render(
    <MemoryRouter>
      <BuilderPage />
    </MemoryRouter>
  );
}

/** Wait for the builder to connect, then answer as the server. */
async function connect(message: object = hello()) {
  await waitFor(() => expect(FakeWebSocket.instances.length).toBeGreaterThan(0));
  const socket = FakeWebSocket.last;
  act(() => {
    socket.open();
    socket.receive(message);
  });
  return socket;
}

function receive(socket: FakeWebSocket, message: object) {
  act(() => socket.receive(message));
}

/** The ops the builder sent (all messages together). */
const sentOps = (socket: FakeWebSocket) => socket.messages('ops').flatMap((m) => m.ops as object[]);

/** Select a layer like a keyboard user (d3-drag cannot handle jsdom's mouse events). */
function select(testId: string) {
  const layer = screen.getByTestId(testId).closest('.react-flow__node');
  fireEvent.keyDown(layer!, { key: 'Enter' });
}

const nameInput = () => screen.getByRole('textbox', { name: 'Network name' });
const liveStatus = () => screen.getByRole('status');

beforeEach(() => {
  vi.clearAllMocks();
  localStorage.clear();
  FakeWebSocket.reset();
  vi.stubGlobal('WebSocket', FakeWebSocket);
  setTokens({ access: 'access-token', refresh: 'refresh-token' });
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
  vi.mocked(api.projects.get).mockResolvedValue(project());
  vi.mocked(api.projects.members.list).mockResolvedValue({
    owner: MARIA,
    members: [{ id: 9, user: ALEX, role: 'editor', created_at: '' }],
  });
});

afterEach(() => {
  vi.unstubAllGlobals();
  setTokens(null);
});

describe('Live editing in the builder', () => {
  it('shows the server state and a live status instead of the save button', async () => {
    openDraft();
    renderBuilder();
    const graph: GraphDocument = {
      ...GRAPH,
      nodes: [
        ...GRAPH.nodes,
        { id: 'd', type: 'Dropout', params: { p: 0.2 }, position: { x: 600, y: 0 } },
      ],
    };
    const socket = await connect(hello({ name: 'Renamed by maria', graph, version: 6 }));
    expect(socket.url).toBe('ws://localhost:3000/api/ws/projects/5/');
    expect(socket.sent[0]).toEqual({ type: 'auth', token: 'access-token' });

    await waitFor(() => expect(nameInput()).toHaveValue('Renamed by maria'));
    expect(screen.getByTestId('layer-Dropout')).toBeInTheDocument();
    expect(liveStatus()).toHaveTextContent('Live');
    expect(screen.queryByRole('button', { name: 'Save to account' })).not.toBeInTheDocument();
    // Nothing was changed here, so nothing is sent.
    expect(socket.messages('ops')).toEqual([]);
  });

  it('sends local changes as ops', async () => {
    openDraft();
    renderBuilder();
    const socket = await connect();
    await screen.findByRole('status');

    select('layer-Linear');
    const field = await screen.findByRole('textbox', { name: 'Output features' });
    await userEvent.clear(field);
    await userEvent.type(field, '64');
    await userEvent.type(nameInput(), '!');

    await waitFor(() =>
      expect(sentOps(socket)).toEqual(
        expect.arrayContaining([
          { op: 'update_node', id: 'fc', set: { params: { out_features: 64 } } },
          { op: 'set_meta', name: 'Shared MLP!' },
        ])
      )
    );
    // The draft keeps what the server has not confirmed yet.
    await waitFor(() => expect(loadDraft()?.pending?.length).toBeGreaterThan(0));
    expect(loadDraft()?.saved).toBe(false);

    // The server confirms: nothing is pending any more.
    for (const message of socket.messages('ops')) {
      receive(socket, {
        type: 'ops',
        ops: message.ops,
        version: 3 + socket.messages('ops').indexOf(message) + 1,
        by: ALEX,
        client_id: 'me',
        id: message.id,
      });
    }
    await waitFor(() => expect(loadDraft()?.saved).toBe(true));
    expect(loadDraft()?.pending).toBeUndefined();
    // The own echo is not applied again (the field keeps what was typed).
    expect(field).toHaveValue('64');
  });

  it('shows changes of others and keeps the own selection', async () => {
    openDraft();
    renderBuilder();
    const socket = await connect();
    select('layer-Linear');
    expect(await screen.findByRole('textbox', { name: 'Output features' })).toHaveValue('2');

    receive(socket, {
      type: 'ops',
      ops: [
        { op: 'update_node', id: 'fc', set: { params: { out_features: 32 } } },
        {
          op: 'add_node',
          node: { id: 'd', type: 'Dropout', params: { p: 0.1 }, position: { x: 600, y: 0 } },
        },
        { op: 'add_edge', source: 'fc', target: 'd' },
      ],
      version: 4,
      by: MARIA,
      client_id: 'other',
      id: 'm1',
    });
    expect(await screen.findByTestId('layer-Dropout')).toBeInTheDocument();
    // Still selected, with the new value.
    expect(screen.getByRole('textbox', { name: 'Output features' })).toHaveValue('32');
    // Applying them is not a local change.
    expect(socket.messages('ops')).toEqual([]);
  });

  it('closes the settings of a layer someone else deleted', async () => {
    openDraft();
    renderBuilder();
    const socket = await connect();
    select('layer-Linear');
    await screen.findByRole('textbox', { name: 'Output features' });

    receive(socket, {
      type: 'ops',
      ops: [{ op: 'remove_node', id: 'fc' }],
      version: 4,
      by: MARIA,
      client_id: 'other',
      id: 'm1',
    });
    await waitFor(() => expect(screen.queryByTestId('layer-Linear')).not.toBeInTheDocument());
    expect(screen.queryByRole('textbox', { name: 'Output features' })).not.toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Model' })).toBeInTheDocument();
    expect(notifications.show).toHaveBeenCalledWith(
      expect.objectContaining({ message: 'maria deleted the layer you had selected.' })
    );
  });

  it('shows who is here and what they have selected', async () => {
    openDraft();
    renderBuilder();
    const socket = await connect();
    receive(socket, { type: 'joined', client_id: 'p1', user: MARIA });
    receive(socket, {
      type: 'presence',
      client_id: 'p1',
      user: MARIA,
      state: { selected: ['fc'] },
    });
    // Phones and wide screens have their own avatars.
    expect(await screen.findAllByRole('button', { name: /Here now: maria, alex/ })).toHaveLength(2);
    const layer = screen.getByTestId('layer-Linear');
    expect(within(layer).getByText('maria')).toBeInTheDocument();
    expect(layer).toHaveAttribute('data-remote');

    // We tell the newcomer what we have selected.
    await waitFor(() => expect(socket.messages('presence').length).toBeGreaterThan(0));
    expect(socket.messages('presence').at(-1)).toEqual({
      type: 'presence',
      state: { selected: [], dragging: null },
    });

    receive(socket, { type: 'left', client_id: 'p1' });
    await waitFor(() => expect(layer).not.toHaveAttribute('data-remote'));
  });

  it('is read only, but live, for viewers', async () => {
    openDraft({ role: 'viewer' });
    vi.mocked(api.projects.get).mockResolvedValue(project({ role: 'viewer' }));
    renderBuilder();
    const socket = await connect(hello({ role: 'viewer' }));
    expect(await screen.findByText('View only')).toBeInTheDocument();
    expect(liveStatus()).toHaveTextContent('Live');
    expect(nameInput()).toHaveAttribute('readonly');

    receive(socket, {
      type: 'ops',
      ops: [{ op: 'set_meta', name: 'Changed by maria' }],
      version: 4,
      by: MARIA,
      client_id: 'other',
      id: 'm1',
    });
    await waitFor(() => expect(nameInput()).toHaveValue('Changed by maria'));
  });

  it('becomes read only when the role changes to viewer', async () => {
    openDraft();
    renderBuilder();
    const socket = await connect();
    await screen.findByRole('status');
    receive(socket, { type: 'access', role: 'viewer' });
    expect(await screen.findByText('View only')).toBeInTheDocument();
    // It gets the current state again (changes that were not confirmed are dropped).
    await waitFor(() => expect(FakeWebSocket.instances).toHaveLength(2));
  });

  it('loads a replaced network', async () => {
    openDraft();
    renderBuilder();
    const socket = await connect();
    receive(socket, {
      type: 'replaced',
      project: { id: 5, name: 'Restored', description: '', graph: GRAPH, version: 9 },
      by: MARIA,
    });
    await waitFor(() => expect(nameInput()).toHaveValue('Restored'));
    expect(notifications.show).toHaveBeenCalledWith(
      expect.objectContaining({ title: 'maria replaced the network' })
    );
  });

  it('keeps the canvas as an unsaved network when the project is deleted', async () => {
    openDraft();
    renderBuilder();
    const socket = await connect();
    await screen.findByRole('status');
    act(() => {
      socket.receive({ type: 'deleted' });
      socket.drop(4404);
    });
    await waitFor(() =>
      expect(notifications.show).toHaveBeenCalledWith(
        expect.objectContaining({ title: '“Shared MLP” was deleted' })
      )
    );
    expect(screen.queryByRole('status')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Save to account' })).toBeInTheDocument();
    expect(screen.getByTestId('layer-Linear')).toBeInTheDocument();
  });

  it('keeps changes made offline and sends them after reconnecting', async () => {
    openDraft();
    renderBuilder();
    const first = await connect();
    await screen.findByRole('status');
    act(() => first.drop(1006));
    await waitFor(() => expect(liveStatus()).toHaveTextContent('Offline – changes will sync'));

    await userEvent.clear(nameInput());
    await userEvent.type(nameInput(), 'Offline name');
    // The reconnect (after a short backoff) gets the newest state from the server ...
    await waitFor(() => expect(FakeWebSocket.instances).toHaveLength(2), { timeout: 2000 });
    const second = FakeWebSocket.last;
    const graph: GraphDocument = {
      ...GRAPH,
      nodes: [...GRAPH.nodes, { id: 'd', type: 'Dropout', params: {}, position: { x: 600, y: 0 } }],
    };
    act(() => {
      second.open();
      second.receive(hello({ name: 'Their name', graph, version: 5 }));
    });
    // ... with the own change on top, sent again.
    expect(await screen.findByTestId('layer-Dropout')).toBeInTheDocument();
    expect(nameInput()).toHaveValue('Offline name');
    expect(sentOps(second)).toEqual([{ op: 'set_meta', name: 'Offline name' }]);
    expect(liveStatus()).toHaveTextContent(/Live|Saving/);
  });

  it('sends unsaved changes of the version the server still has', async () => {
    openDraft({ name: 'Changed before', saved: false });
    renderBuilder();
    const socket = await connect(hello({ version: 3 }));
    await waitFor(() =>
      expect(sentOps(socket)).toEqual([{ op: 'set_meta', name: 'Changed before' }])
    );
    expect(nameInput()).toHaveValue('Changed before');
    expect(liveStatus()).toBeInTheDocument();
  });

  it('works as before when unsaved changes are based on an older version', async () => {
    openDraft({ name: 'Changed before', saved: false });
    vi.mocked(api.projects.get).mockResolvedValue(project({ version: 6 }));
    renderBuilder();
    const socket = await connect(hello({ version: 6 }));
    await waitFor(() => expect(api.projects.get).toHaveBeenCalled());
    // Saving lets the user choose between the versions (no ops are sent).
    expect(screen.getByRole('button', { name: 'Save to account' })).toBeInTheDocument();
    expect(nameInput()).toHaveValue('Changed before');
    expect(socket.messages('ops')).toEqual([]);
  });

  it('leaves the project for a new network', async () => {
    openDraft();
    renderBuilder();
    const socket = await connect();
    await screen.findByRole('status');
    await userEvent.click(screen.getByRole('button', { name: 'File' }));
    // Everything is saved already, so it does not ask.
    await userEvent.click(await screen.findByRole('menuitem', { name: 'New network' }));
    await waitFor(() => expect(socket.closedByClient).toBe(1000));
    expect(screen.queryByTestId('layer-Linear')).not.toBeInTheDocument();
    expect(screen.queryByRole('status')).not.toBeInTheDocument();
    expect(nameInput()).toHaveValue('Untitled network');
    expect(socket.messages('ops')).toEqual([]);
    await waitFor(() => expect(loadDraft()?.projectId).toBeNull());
  });

  it('does not connect for networks that are not saved', async () => {
    saveDraft({ doc: GRAPH, name: 'Local', projectId: null });
    renderBuilder();
    await screen.findByRole('button', { name: 'Save to account' });
    expect(FakeWebSocket.instances).toHaveLength(0);
  });
});
