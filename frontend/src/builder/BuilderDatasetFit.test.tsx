/**
 * The Input layer follows the dataset chosen for training, in a live project: the changes go
 * out as normal ops, and viewers' networks are never changed.
 */
import { useState } from 'react';
import { act, FakeWebSocket, render, screen, waitFor } from '@test-utils';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, vi } from 'vitest';
import { notifications } from '@mantine/notifications';
import {
  api,
  setTokens,
  type Analysis,
  type GraphDocument,
  type LayerCatalog,
  type Project,
} from '@/api';
import { useAuth } from '@/auth/AuthContext';
import { dataset, STATUS } from '@/training/fixtures';
import type { DataController } from '@/training/useDatasets';
import { useTraining, type TrainingController } from '@/training/useTraining';
import { BuilderPage } from './BuilderPage';
import { saveDraft } from './draft';

vi.mock('@/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api')>();
  return {
    ...actual,
    api: {
      projects: { get: vi.fn(), members: { list: vi.fn() } },
      training: { check: vi.fn() },
    },
  };
});
vi.mock('@/auth/AuthContext', () => ({ useAuth: vi.fn() }));
vi.mock('@/training/useTraining', () => ({ useTraining: vi.fn() }));

const ANALYSIS: Analysis = {
  valid: true,
  errors: [],
  warnings: [],
  nodes: {},
  inputs: ['in'],
  outputs: ['fc'],
  total_params: 0,
};
vi.mock('./useAnalysis', () => ({
  useAnalysis: () => ({ analysis: ANALYSIS, loading: false, error: null }),
}));

const param = (name: string, label: string, kind: string, value: unknown) => ({
  name,
  label,
  kind,
  default: value,
  help: '',
  options: [],
  min: null,
  max: null,
  auto: false,
  length: null,
  show_if: null,
});

const CATALOG = {
  categories: ['Input', 'Basic'],
  layers: [
    {
      type: 'Input',
      label: 'Input',
      category: 'Input',
      description: 'Network input',
      params: [param('shape', 'Shape', 'ints', [784])],
      min_inputs: 0,
      max_inputs: 0,
    },
    {
      type: 'Linear',
      label: 'Linear',
      category: 'Basic',
      description: 'Fully connected layer',
      params: [param('out_features', 'Output features', 'int', 10)],
      min_inputs: 1,
      max_inputs: 1,
    },
  ],
} as unknown as LayerCatalog;
vi.mock('./useCatalog', () => ({
  useCatalog: () => ({ catalog: CATALOG, examples: [], error: null, reload: vi.fn() }),
}));

const GRAPH: GraphDocument = {
  version: 1,
  nodes: [
    { id: 'in', type: 'Input', params: { shape: [784] }, position: { x: 0, y: 0 } },
    { id: 'fc', type: 'Linear', params: { out_features: 10 }, position: { x: 300, y: 0 } },
  ],
  edges: [{ source: 'in', target: 'fc' }],
};

const ALEX = { id: 1, username: 'alex' };
const MARIA = { id: 2, username: 'maria' };

function project(role: 'editor' | 'viewer'): Project {
  return {
    id: 5,
    name: 'Shared MLP',
    description: '',
    graph: GRAPH,
    num_layers: 2,
    created_at: '2026-10-01T09:00:00Z',
    updated_at: '2026-11-08T16:20:00Z',
    owner: MARIA,
    role,
    version: 3,
    updated_by: 'maria',
    shared: true,
    link_access: 'none',
    share_token: null,
  };
}

/** Chooses a dataset in the training panel (set by the mocked useTraining). */
let chooseDataset: () => void = () => undefined;

function mockTraining(role: 'editor' | 'viewer') {
  vi.mocked(useTraining).mockImplementation(() => {
    // A real counter, as in useDatasets: every choice is a new selection.
    const [selection, setSelection] = useState(0);
    chooseDataset = () => setSelection((n) => n + 1);
    const data = {
      source: { tab: 'upload', builtin: null, upload: 7 },
      setSource: vi.fn(),
      selection,
      datasets: [dataset({ id: 7, name: 'iris' })],
      reloadDatasets: vi.fn(),
    } as unknown as DataController;
    return {
      status: STATUS,
      scope: { username: 'alex', project: { id: 5, role, shared: true } },
      opened: false,
      view: 'setup',
      open: vi.fn(),
      close: vi.fn(),
      activeRun: null,
      data,
    } as unknown as TrainingController;
  });
}

async function openLive(role: 'editor' | 'viewer') {
  saveDraft({ doc: GRAPH, name: 'Shared MLP', projectId: 5, version: 3, role, saved: true });
  vi.mocked(api.projects.get).mockResolvedValue(project(role));
  mockTraining(role);
  render(
    <MemoryRouter>
      <BuilderPage />
    </MemoryRouter>
  );
  await waitFor(() => expect(FakeWebSocket.instances.length).toBeGreaterThan(0));
  const socket = FakeWebSocket.last;
  act(() => {
    socket.open();
    socket.receive({
      type: 'hello',
      project: { id: 5, name: 'Shared MLP', description: '', graph: GRAPH, version: 3 },
      role,
      client_id: 'me',
      user: ALEX,
    });
  });
  await screen.findByRole('status');
  return socket;
}

const sentOps = (socket: FakeWebSocket) => socket.messages('ops').flatMap((m) => m.ops as object[]);

beforeEach(() => {
  vi.clearAllMocks();
  localStorage.clear();
  FakeWebSocket.reset();
  vi.stubGlobal('WebSocket', FakeWebSocket);
  setTokens({ access: 'access-token', refresh: 'refresh-token' });
  vi.spyOn(notifications, 'show').mockImplementation(() => 'id');
  vi.mocked(useAuth).mockReturnValue({
    user: { ...ALEX, email: '', date_joined: '2026-01-01T00:00:00Z' },
    ready: true,
    login: vi.fn(),
    register: vi.fn(),
    logout: vi.fn(),
  });
  vi.mocked(api.projects.members.list).mockResolvedValue({ owner: MARIA, members: [] });
  vi.mocked(api.training.check).mockResolvedValue({
    ok: false,
    errors: ['The Input expects shape (784), but the CSV file has 4 feature columns.'],
    warnings: [],
    plan: null,
    fixes: [
      {
        node: 'in',
        type: 'Input',
        params: { shape: [4] },
        previous: { shape: [784] },
        summary: 'Input shape 784 → 4',
      },
      {
        node: 'fc',
        type: 'Linear',
        params: { out_features: 3 },
        previous: { out_features: 10 },
        summary: 'Outputs of the last layer 10 → 3',
      },
    ],
  });
});

afterEach(() => {
  vi.unstubAllGlobals();
  setTokens(null);
});

describe('The Input follows the training data in a live network', () => {
  it('sends the changes to the others as ops', async () => {
    const socket = await openLive('editor');
    expect(api.training.check).not.toHaveBeenCalled();

    act(() => chooseDataset());
    await waitFor(() =>
      expect(sentOps(socket)).toEqual(
        expect.arrayContaining([
          { op: 'update_node', id: 'in', set: { params: { shape: [4] } } },
          { op: 'update_node', id: 'fc', set: { params: { out_features: 3 } } },
        ])
      )
    );
    expect(notifications.show).toHaveBeenCalledWith(
      expect.objectContaining({ id: expect.stringMatching(/^dataset-fit-/) })
    );
  });

  it('leaves the network of a viewer alone', async () => {
    const socket = await openLive('viewer');
    act(() => chooseDataset());
    await act(() => new Promise((resolve) => setTimeout(resolve, 700)));
    expect(api.training.check).not.toHaveBeenCalled();
    expect(sentOps(socket)).toEqual([]);
    expect(notifications.show).not.toHaveBeenCalledWith(
      expect.objectContaining({ id: expect.stringMatching(/^dataset-fit-/) })
    );
  });
});
