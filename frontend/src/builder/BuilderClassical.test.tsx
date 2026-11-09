/** Building classical models on the canvas: kernels go into the bricks that take them. */
import { fireEvent, render, screen, userEvent, waitFor } from '@test-utils';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, vi } from 'vitest';
import type { GraphDocument } from '@/api';
import { useAuth } from '@/auth/AuthContext';
import { useTraining, type TrainingController } from '@/training/useTraining';
import { BuilderPage } from './BuilderPage';
import { CLASSICAL_CATALOG } from './classicalFixtures';
import { loadDraft, saveDraft } from './draft';

vi.mock('@/auth/AuthContext', () => ({ useAuth: vi.fn() }));
vi.mock('@/training/useTraining', () => ({ useTraining: vi.fn() }));
vi.mock('./useCatalog', () => ({
  useCatalog: () => ({ catalog: CLASSICAL_CATALOG, examples: [], error: null, reload: vi.fn() }),
}));
vi.mock('./useAnalysis', () => ({
  useAnalysis: () => ({ analysis: null, loading: false, error: null }),
}));

const GRAPH: GraphDocument = {
  version: 1,
  nodes: [
    { id: 'in', type: 'Input', params: { shape: [8] }, position: { x: 0, y: 0 } },
    { id: 'gp', type: 'GaussianProcessRegressor', params: {}, position: { x: 300, y: 0 } },
  ],
  edges: [{ id: 'e-in-gp', source: 'in', target: 'gp' }],
};

function renderBuilder(graph: GraphDocument = GRAPH) {
  saveDraft({ doc: graph, name: 'GP', projectId: null });
  render(
    <MemoryRouter>
      <BuilderPage />
    </MemoryRouter>
  );
}

/** Select a layer like a keyboard user (d3-drag cannot handle jsdom's mouse events). */
function select(id: string) {
  const layer = document.querySelector(`.react-flow__node[data-id="${id}"]`);
  fireEvent.keyDown(layer!, { key: 'Enter' });
}

/** The connections as saved in the draft, as "source → target" with types for new layers. */
async function savedEdges(count: number): Promise<string[]> {
  let edges: string[] = [];
  await waitFor(() => {
    const doc = loadDraft()!.doc;
    const name = (id: string) =>
      ['in', 'gp'].includes(id) ? id : doc.nodes.find((node) => node.id === id)!.type;
    edges = doc.edges.map((edge) => `${name(edge.source)} → ${name(edge.target)}`);
    expect(edges).toHaveLength(count);
  });
  return edges;
}

beforeEach(() => {
  vi.clearAllMocks();
  localStorage.clear();
  vi.mocked(useAuth).mockReturnValue({
    user: null,
    ready: true,
    login: vi.fn(),
    register: vi.fn(),
    logout: vi.fn(),
  });
  vi.mocked(useTraining).mockReturnValue({
    open: vi.fn(),
    opened: false,
    data: { setSource: vi.fn(), reloadDatasets: vi.fn() },
  } as unknown as TrainingController);
});

describe('Classical models in the builder', () => {
  it('draws one input per slot, labelled, with kernel handles', async () => {
    renderBuilder();
    const gp = await screen.findByTestId('layer-GaussianProcessRegressor');
    const handles = gp.querySelectorAll('.react-flow__handle.target');
    expect([...handles].map((h) => h.getAttribute('data-handleid'))).toEqual([
      'features',
      'kernel',
    ]);
    expect([...handles].map((h) => h.getAttribute('data-kind'))).toEqual(['tensor', 'kernel']);
    expect(gp).toHaveTextContent('features');
    expect(gp).toHaveTextContent('default kernel');
  });

  it('puts a kernel from the palette into the selected Gaussian process', async () => {
    renderBuilder();
    await screen.findByTestId('layer-GaussianProcessRegressor');
    select('gp');
    await userEvent.click(screen.getByRole('button', { name: 'RBF kernel' }));
    expect(await savedEdges(2)).toEqual(['in → gp', 'RBF → gp']);

    // The new kernel is selected: a kernel sum is appended after it.
    await userEvent.click(screen.getByRole('button', { name: 'Kernel sum' }));
    expect(await savedEdges(3)).toEqual(['in → gp', 'RBF → gp', 'RBF → KernelSum']);
  });

  it('does not connect a kernel where it does not fit', async () => {
    renderBuilder();
    await screen.findByTestId('layer-Input');
    select('in');
    // The Input puts out data: a kernel cannot go into it, and a Linear layer is appended.
    await userEvent.click(screen.getByRole('button', { name: 'White noise kernel' }));
    await userEvent.click(screen.getByRole('button', { name: 'Linear' }));
    await waitFor(() => expect(loadDraft()!.doc.nodes).toHaveLength(4));
    // The kernel was selected when Linear was added: nothing is connected.
    expect(await savedEdges(1)).toEqual(['in → gp']);
  });

  it('gives a Gaussian process with a kernel no second one', async () => {
    renderBuilder({
      ...GRAPH,
      nodes: [...GRAPH.nodes, { id: 'rbf', type: 'RBF', params: {}, position: { x: 0, y: 170 } }],
      edges: [...GRAPH.edges, { id: 'e-rbf-gp', source: 'rbf', target: 'gp' }],
    });
    await screen.findByTestId('layer-GaussianProcessRegressor');
    expect(screen.getByTestId('layer-GaussianProcessRegressor')).toHaveTextContent('kernel');
    select('gp');
    await userEvent.click(screen.getByRole('button', { name: 'White noise kernel' }));
    await waitFor(() => expect(loadDraft()!.doc.nodes).toHaveLength(4));
    expect(await savedEdges(2)).toEqual(['in → gp', 'RBF → gp']);
  });
});
