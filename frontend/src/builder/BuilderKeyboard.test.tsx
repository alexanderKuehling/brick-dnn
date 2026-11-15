/** Building a network with the keyboard alone (docs/accessibility.md). */
import {
  act,
  axeViolations,
  fireEvent,
  render,
  screen,
  userEvent,
  waitFor,
  within,
} from '@test-utils';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, vi } from 'vitest';
import type { GraphDocument } from '@/api';
import { useAuth } from '@/auth/AuthContext';
import { i18n } from '@/i18n';
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
    { id: 'fc1', type: 'Linear', name: 'hidden', params: {}, position: { x: 300, y: 0 } },
    { id: 'fc2', type: 'Linear', name: 'head', params: {}, position: { x: 600, y: 0 } },
  ],
  edges: [{ id: 'e-in-fc1', source: 'in', target: 'fc1' }],
};

function renderBuilder(graph: GraphDocument = GRAPH) {
  saveDraft({ doc: graph, name: 'Keys', projectId: null });
  render(
    <MemoryRouter>
      <BuilderPage />
    </MemoryRouter>
  );
}

const layer = (id: string) =>
  document.querySelector<HTMLElement>(`.react-flow__node[data-id="${id}"]`)!;

/** A key pressed and released (React Flow tracks the keys that are down). */
function press(element: HTMLElement, key: string) {
  fireEvent.keyDown(element, { key });
  fireEvent.keyUp(element, { key });
}

/** Focus a layer on the canvas (Tab would get there) and select it with Enter. */
async function select(id: string) {
  act(() => layer(id).focus());
  press(layer(id), 'Enter');
  await waitFor(() => expect(layer(id)).toHaveClass('selected'));
}

/** The connections as saved in the draft: "source → target" by id or, for new layers, type. */
async function savedEdges(count: number): Promise<string[]> {
  let edges: string[] = [];
  await waitFor(() => {
    const doc = loadDraft()!.doc;
    const name = (id: string) =>
      GRAPH.nodes.some((node) => node.id === id)
        ? id
        : doc.nodes.find((node) => node.id === id)!.type;
    edges = doc.edges.map((edge) => `${name(edge.source)} → ${name(edge.target)}`);
    expect(edges).toHaveLength(count);
  });
  return edges;
}

const announced = () => screen.getByTestId('announcer');

/** The text field of a select (its list has the same label). */
const input = (container: HTMLElement, label: string) =>
  within(container)
    .getAllByLabelText(label)
    .find((element) => element.tagName === 'INPUT')!;

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

describe('The builder with the keyboard', () => {
  it('names layers and connections on the canvas and describes the keys', async () => {
    renderBuilder();
    await screen.findByTestId('layer-Input');
    expect(screen.getByRole('application', { name: 'Network canvas' })).toBeInTheDocument();
    // React Flow shows layers once it has measured them, which jsdom cannot (so no role
    // queries); it draws no connections without sizes (the browser check covers their names).
    expect(layer('fc1')).toHaveAttribute('role', 'group');
    expect(layer('fc1')).toHaveAttribute('aria-label', 'hidden, Linear');
    expect(layer('fc1')).toHaveAttribute('tabindex', '0');
    const description = layer('fc1').getAttribute('aria-describedby')!;
    expect(document.getElementById(description)).toHaveTextContent(/Delete removes it/);
    expect(screen.getByRole('button', { name: 'Zoom in' })).toBeInTheDocument();
  });

  it('adds a layer from the palette with Enter and says where it went', async () => {
    renderBuilder();
    await screen.findByTestId('layer-Input');
    await select('in');
    act(() => screen.getByRole('button', { name: 'Linear' }).focus());
    await userEvent.keyboard('{Enter}');
    expect(await savedEdges(2)).toEqual(['in → fc1', 'in → Linear']);
    expect(announced()).toHaveTextContent('Added Linear after Input.');
  });

  it('moves through the palette with the arrow keys; Enter in the search adds the first match', async () => {
    renderBuilder();
    await screen.findByTestId('layer-Input');
    const palette = screen.getAllByRole('group', { name: 'Layers' })[0];
    const items = within(palette).getAllByRole('button');
    // One stop of the Tab key for the whole list.
    expect(items.filter((item) => item.tabIndex === 0)).toHaveLength(1);
    act(() => items[0].focus());
    await userEvent.keyboard('{ArrowDown}');
    expect(items[1]).toHaveFocus();
    expect(items[1]).toHaveAttribute('tabindex', '0');
    await userEvent.keyboard('{End}');
    expect(items[items.length - 1]).toHaveFocus();

    await userEvent.type(screen.getAllByLabelText('Search layers')[0], 'white{Enter}');
    await waitFor(() => expect(loadDraft()!.doc.nodes.map((n) => n.type)).toContain('WhiteKernel'));
  });

  it('connects and disconnects layers in the properties panel', async () => {
    renderBuilder();
    await screen.findByTestId('layer-Input');
    await select('fc2');
    const panel = screen.getAllByTestId('connections')[0];
    expect(within(panel).getAllByText('None')).toHaveLength(2);

    // "Add input from …": only layers that can feed it are offered.
    await userEvent.click(input(panel, 'Add input from …'));
    const options = screen.getAllByRole('option').map((option) => option.textContent);
    expect(options).toEqual(['Input', 'hidden (Linear)']);
    await userEvent.click(screen.getByRole('option', { name: 'hidden (Linear)' }));
    expect(await savedEdges(2)).toEqual(['in → fc1', 'fc1 → fc2']);
    expect(announced()).toHaveTextContent('Connected hidden to head.');

    // Removing it again.
    const panelNow = screen.getAllByTestId('connections')[0];
    await userEvent.click(
      within(panelNow).getByRole('button', { name: 'Remove the connection from hidden to head' })
    );
    expect(await savedEdges(1)).toEqual(['in → fc1']);
    expect(announced()).toHaveTextContent('Removed the connection from hidden to head.');
  });

  it('deletes the selected layer with Delete and keeps the focus on the canvas', async () => {
    renderBuilder();
    await screen.findByTestId('layer-Input');
    await select('fc1');
    press(layer('fc1'), 'Delete');
    await waitFor(() => expect(loadDraft()!.doc.nodes.map((n) => n.id)).toEqual(['in', 'fc2']));
    expect(await savedEdges(0)).toEqual([]);
    expect(announced()).toHaveTextContent('Deleted hidden.');
    // On to the layer it was connected to.
    await waitFor(() => expect(layer('in')).toHaveFocus());
  });

  it('deletes from the properties panel and moves the focus to a neighbour', async () => {
    renderBuilder();
    await screen.findByTestId('layer-Input');
    await select('fc1');
    await userEvent.click(screen.getAllByRole('button', { name: 'Delete layer' })[0]);
    await waitFor(() => expect(loadDraft()!.doc.nodes).toHaveLength(2));
    expect(announced()).toHaveTextContent('Deleted hidden.');
    await waitFor(() => expect(layer('in')).toHaveFocus());
  });

  it('lists the layers in the summary; choosing one selects it on the canvas', async () => {
    renderBuilder();
    await screen.findByTestId('layer-Input');
    const list = screen.getAllByRole('list', { name: /Layers of the network/ })[0];
    expect(
      within(list)
        .getAllByRole('listitem')
        .map((item) => item.textContent)
    ).toEqual([
      'Input · no inputs',
      'hidden (Linear) · input from Input',
      'head (Linear) · no inputs',
    ]);
    await userEvent.click(within(list).getByRole('button', { name: /^head/ }));
    await waitFor(() => expect(layer('fc2')).toHaveClass('selected'));
    await waitFor(() => expect(layer('fc2')).toHaveFocus());
  });

  it('has no serious axe violations, with and without a selected layer', async () => {
    renderBuilder();
    await screen.findByTestId('layer-Input');
    expect(await axeViolations()).toEqual([]);
    await select('fc1');
    await screen.findAllByTestId('connections');
    expect(await axeViolations()).toEqual([]);
  });

  it('speaks German', async () => {
    await i18n.changeLanguage('de');
    renderBuilder();
    await screen.findByTestId('layer-Input');
    expect(
      screen.getByRole('application', { name: 'Arbeitsfläche des Netzes' })
    ).toBeInTheDocument();
    expect(layer('fc2')).toHaveAttribute('aria-label', 'head, Linear');
    expect(screen.getByRole('button', { name: 'Vergrößern' })).toBeInTheDocument();
    await select('fc2');
    expect(
      input(screen.getAllByTestId('connections')[0], 'Eingang hinzufügen von …')
    ).toBeInTheDocument();
  });
});
