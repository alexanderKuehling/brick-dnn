/** History and comments in the builder, with the live connection (fake WebSocket as server). */
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
import { BuilderPage } from '@/builder/BuilderPage';
import { saveDraft } from '@/builder/draft';
import { useTraining, type TrainingController } from '@/training/useTraining';
import { ALICE, BOB, comment, version } from './fixtures';

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
        versions: {
          list: vi.fn(),
          get: vi.fn(),
          create: vi.fn(),
          rename: vi.fn(),
          restore: vi.fn(),
        },
        comments: { list: vi.fn(), create: vi.fn(), update: vi.fn(), remove: vi.fn() },
      },
    },
  };
});
vi.mock('@/auth/AuthContext', () => ({ useAuth: vi.fn() }));
vi.mock('@/training/useTraining', () => ({ useTraining: vi.fn() }));
vi.mock('@/builder/useAnalysis', () => ({
  useAnalysis: () => ({ analysis: null, loading: false, error: null }),
}));
vi.mock('@/builder/NetworkViewer', () => ({
  NetworkViewer: () => <div data-testid="viewer" />,
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

vi.mock('@/builder/useCatalog', () => ({
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

const WITH_DROPOUT: GraphDocument = {
  ...GRAPH,
  nodes: [
    ...GRAPH.nodes,
    { id: 'd', type: 'Dropout', params: { p: 0.2 }, position: { x: 600, y: 0 } },
  ],
  edges: [...GRAPH.edges, { source: 'fc', target: 'd' }],
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
    owner: BOB,
    role: 'editor',
    version: 3,
    updated_by: 'bob',
    shared: true,
    link_access: 'none',
    share_token: null,
    ...over,
  };
}

const hello = {
  type: 'hello',
  project: { id: 5, name: 'Shared MLP', description: '', graph: GRAPH, version: 3 },
  role: 'editor',
  client_id: 'me',
  user: ALICE,
};

const OPEN_ON_LINEAR = comment({ id: 1, node_id: 'fc', author: BOB, text: 'Is 2 enough?' });

function renderBuilder({ saved = true } = {}) {
  if (saved) {
    saveDraft({
      doc: GRAPH,
      name: 'Shared MLP',
      projectId: 5,
      version: 3,
      role: 'editor',
      saved: true,
    });
  }
  render(
    <MemoryRouter>
      <BuilderPage />
    </MemoryRouter>
  );
}

async function connect() {
  await waitFor(() => expect(FakeWebSocket.instances.length).toBeGreaterThan(0));
  const socket = FakeWebSocket.last;
  act(() => {
    socket.open();
    socket.receive(hello);
  });
  await screen.findByRole('status'); // "Live"
  return socket;
}

const receive = (socket: FakeWebSocket, message: object) => act(() => socket.receive(message));
const markerOf = (testId: string) =>
  within(screen.getByTestId(testId)).queryByTestId('comment-marker');

beforeEach(() => {
  vi.clearAllMocks();
  localStorage.clear();
  FakeWebSocket.reset();
  vi.stubGlobal('WebSocket', FakeWebSocket);
  setTokens({ access: 'access-token', refresh: 'refresh-token' });
  vi.spyOn(notifications, 'show').mockImplementation(() => 'id');
  vi.mocked(useAuth).mockReturnValue({
    user: { ...ALICE, email: '', date_joined: '2026-01-01T00:00:00Z' },
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
    owner: BOB,
    members: [{ id: 9, user: ALICE, role: 'editor', created_at: '' }],
  });
  vi.mocked(api.projects.comments.list).mockResolvedValue([OPEN_ON_LINEAR]);
  vi.mocked(api.projects.versions.list).mockResolvedValue([
    version({ id: 3, version: 3, created_by: 'bob' }),
    version({ id: 1, version: 1, kind: 'named', label: 'Baseline', num_layers: 3 }),
  ]);
  vi.mocked(api.projects.versions.get).mockResolvedValue({
    ...version({ id: 1, version: 1, kind: 'named', label: 'Baseline', num_layers: 3 }),
    description: '',
    graph: WITH_DROPOUT,
  });
});

afterEach(() => {
  vi.unstubAllGlobals();
  setTokens(null);
});

describe('History and comments in the builder', () => {
  it('are there for saved networks, with markers on layers that have open comments', async () => {
    renderBuilder();
    expect(await screen.findByRole('button', { name: 'Comments (1 open)' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'History' })).toBeInTheDocument();
    expect(api.projects.comments.list).toHaveBeenCalledWith(5);
    expect(markerOf('layer-Linear')).toHaveTextContent('1');
    expect(markerOf('layer-Input')).toBeNull();
  });

  it('are not there for networks that are not saved', async () => {
    renderBuilder({ saved: false });
    await screen.findByRole('textbox', { name: 'Network name' });
    expect(screen.queryByRole('button', { name: 'History' })).not.toBeInTheDocument();
    expect(api.projects.comments.list).not.toHaveBeenCalled();
  });

  it('follow live comment messages', async () => {
    renderBuilder();
    const socket = await connect();
    await screen.findByRole('button', { name: 'Comments (1 open)' });

    const onInput = comment({ id: 2, node_id: 'in', author: BOB, text: 'Wrong shape' });
    receive(socket, { type: 'comment', action: 'created', comment: onInput });
    expect(markerOf('layer-Input')).toHaveTextContent('1');
    expect(screen.getByRole('button', { name: 'Comments (2 open)' })).toBeInTheDocument();

    receive(socket, {
      type: 'comment',
      action: 'updated',
      comment: { ...OPEN_ON_LINEAR, resolved: true, resolved_by: 'bob' },
    });
    expect(markerOf('layer-Linear')).toBeNull();

    receive(socket, { type: 'comment', action: 'deleted', comment: { id: 2 } });
    expect(markerOf('layer-Input')).toBeNull();
    expect(screen.getByRole('button', { name: 'Comments' })).toBeInTheDocument();
  });

  it('shows the threads of a layer from its marker and goes back to the layer', async () => {
    renderBuilder();
    await screen.findByRole('button', { name: 'Comments (1 open)' });
    // jsdom cannot measure layers, so React Flow keeps them hidden (no role queries), and
    // d3-drag cannot handle its mouse events (a click only).
    fireEvent.click(markerOf('layer-Linear')!);

    const panel = await screen.findByRole('tabpanel', { name: /Comments/ });
    expect(within(panel).getByText('On Linear')).toBeInTheDocument();
    expect(within(panel).getByText('Is 2 enough?')).toBeInTheDocument();

    await userEvent.click(within(panel).getByRole('button', { name: 'All comments' }));
    await userEvent.click(within(panel).getByRole('button', { name: 'Show layer Linear' }));
    await waitFor(() =>
      expect(screen.getByTestId('layer-Linear').closest('.react-flow__node')).toHaveClass(
        'selected'
      )
    );
  });

  it('comment on a layer from the inspector', async () => {
    vi.mocked(api.projects.comments.create).mockResolvedValue(
      comment({ id: 3, node_id: 'fc', author: ALICE, text: 'Make it 64' })
    );
    renderBuilder();
    await screen.findByRole('button', { name: 'Comments (1 open)' });
    const layer = screen.getByTestId('layer-Linear').closest('.react-flow__node');
    fireEvent.keyDown(layer!, { key: 'Enter' });

    await userEvent.click(await screen.findByRole('button', { name: 'Add comment' }));
    await userEvent.type(
      screen.getByRole('textbox', { name: 'Comment on this layer' }),
      'Make it 64'
    );
    await userEvent.click(screen.getByRole('button', { name: 'Comment' }));
    expect(api.projects.comments.create).toHaveBeenCalledWith(5, {
      text: 'Make it 64',
      node_id: 'fc',
    });
    await waitFor(() => expect(markerOf('layer-Linear')).toHaveTextContent('2'));
  });

  it('load the history when it is opened and again on live history messages', async () => {
    renderBuilder();
    const socket = await connect();
    await userEvent.click(await screen.findByRole('button', { name: 'History' }));
    expect(await screen.findByText('Baseline')).toBeInTheDocument();
    expect(api.projects.versions.list).toHaveBeenCalledTimes(1);

    receive(socket, { type: 'history' });
    await waitFor(() => expect(api.projects.versions.list).toHaveBeenCalledTimes(2));

    // Both panels are tabs of one side panel.
    await userEvent.click(screen.getByRole('tab', { name: /Comments/ }));
    expect(await screen.findByText('Is 2 enough?')).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Close the panel' }));
    await waitFor(() => expect(screen.queryByRole('tab')).not.toBeInTheDocument());
  });

  async function restoreBaseline() {
    await userEvent.click(await screen.findByRole('button', { name: 'History' }));
    await userEvent.click(await screen.findByRole('button', { name: /Named version: Baseline/ }));
    const dialog = await screen.findByRole('dialog', { name: /Baseline/ });
    await within(dialog).findByTestId('viewer');
    await userEvent.click(within(dialog).getByRole('button', { name: 'Restore this version' }));
    await userEvent.click(within(dialog).getByRole('button', { name: 'Restore' }));
    expect(api.projects.versions.restore).toHaveBeenCalledWith(5, 1);
    await waitFor(() => expect(screen.queryByRole('dialog', { name: /Baseline/ })).toBeNull());
  }

  it('restores a version: live, the canvas follows the server', async () => {
    const restored = project({ graph: WITH_DROPOUT, version: 4, num_layers: 3 });
    vi.mocked(api.projects.versions.restore).mockResolvedValue(restored);
    renderBuilder();
    const socket = await connect();
    await restoreBaseline();
    // The live connection brings the new state (not loaded twice).
    expect(screen.queryByTestId('layer-Dropout')).not.toBeInTheDocument();
    receive(socket, {
      type: 'replaced',
      project: { id: 5, name: 'Shared MLP', description: '', graph: WITH_DROPOUT, version: 4 },
      by: ALICE,
    });
    expect(await screen.findByTestId('layer-Dropout')).toBeInTheDocument();
    expect(notifications.show).toHaveBeenCalledWith(
      expect.objectContaining({ id: 'live-replaced', title: 'Restored “Baseline”' })
    );
  });

  it('restores a version: without a live connection, the builder loads it', async () => {
    vi.mocked(api.projects.versions.restore).mockResolvedValue(
      project({ graph: WITH_DROPOUT, version: 4, num_layers: 3 })
    );
    renderBuilder(); // The connection never opens.
    await restoreBaseline();
    expect(await screen.findByTestId('layer-Dropout')).toBeInTheDocument();
  });
});
