/** The guide for a first visit (shown once, skippable, re-openable) and the empty canvas. */
import { render, screen, userEvent, waitFor } from '@test-utils';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, vi } from 'vitest';
import type { Example, GraphDocument } from '@/api';
import { useAuth } from '@/auth/AuthContext';
import { useTraining, type TrainingController } from '@/training/useTraining';
import { BuilderPage } from './BuilderPage';
import { CLASSICAL_CATALOG } from './classicalFixtures';
import { loadDraft, saveDraft } from './draft';
import { GUIDE_KEY, GUIDE_STEPS, guideSeen, OnboardingGuide, startExamples } from './Onboarding';

vi.mock('@/auth/AuthContext', () => ({ useAuth: vi.fn() }));
vi.mock('@/training/useTraining', () => ({ useTraining: vi.fn() }));
const catalogState: { examples: Example[] } = { examples: [] };
vi.mock('./useCatalog', () => ({
  useCatalog: () => ({
    catalog: CLASSICAL_CATALOG,
    examples: catalogState.examples,
    error: null,
    reload: vi.fn(),
  }),
}));
vi.mock('./useAnalysis', () => ({
  useAnalysis: () => ({ analysis: null, loading: false, error: null }),
}));

const GRAPH: GraphDocument = {
  version: 1,
  nodes: [{ id: 'in', type: 'Input', params: { shape: [8] }, position: { x: 0, y: 0 } }],
  edges: [],
};

function example(id: string, name: string): Example {
  return { id, name, description: `About ${name}.`, order: 1, graph: GRAPH };
}

const EXAMPLES = [
  example('tabular_regression', 'Tabular regression'),
  example('mlp_mnist', 'MLP classifier (MNIST)'),
  example('cnn_mnist', 'Small CNN (MNIST)'),
  example('resnet_block', 'Residual block'),
  example('unet', 'U-Net (image segmentation)'),
  example('gp_regression', 'Gaussian process regression'),
];

function renderBuilder(graph: GraphDocument = GRAPH) {
  saveDraft({ doc: graph, name: 'Mine', projectId: null });
  return render(
    <MemoryRouter>
      <BuilderPage />
    </MemoryRouter>
  );
}

const firstStep = () =>
  screen.findByRole('dialog', { name: GUIDE_STEPS[0].title }, { timeout: 3000 });

beforeEach(() => {
  vi.clearAllMocks();
  localStorage.clear();
  catalogState.examples = [];
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

afterEach(() => {
  vi.unstubAllEnvs();
});

describe('The guide to the builder', () => {
  it('walks through the steps and is remembered when finished', async () => {
    const onClose = vi.fn();
    render(<OnboardingGuide opened onClose={onClose} />);
    expect(await firstStep()).toHaveTextContent('Drag a layer onto the canvas');
    expect(screen.getByText('1 of 5')).toBeInTheDocument();

    for (const step of GUIDE_STEPS.slice(1)) {
      await userEvent.click(screen.getByRole('button', { name: 'Next' }));
      expect(await screen.findByRole('dialog', { name: step.title })).toBeInTheDocument();
    }
    expect(screen.getByText(/The \? button shows this guide again/)).toBeInTheDocument();
    expect(guideSeen()).toBe(false);
    await userEvent.click(screen.getByRole('button', { name: 'Done' }));
    expect(onClose).toHaveBeenCalled();
    expect(guideSeen()).toBe(true);
  });

  it('can be skipped, also with Escape', async () => {
    const onClose = vi.fn();
    const { unmount } = render(<OnboardingGuide opened onClose={onClose} />);
    await firstStep();
    await userEvent.click(screen.getByRole('button', { name: 'Skip' }));
    expect(onClose).toHaveBeenCalledTimes(1);
    expect(localStorage.getItem(GUIDE_KEY)).not.toBeNull();
    unmount();

    localStorage.clear();
    render(<OnboardingGuide opened onClose={onClose} />);
    await firstStep();
    await userEvent.keyboard('{Escape}');
    expect(onClose).toHaveBeenCalledTimes(2);
    expect(guideSeen()).toBe(true);
  });

  it('is not shown in tests unless asked', async () => {
    renderBuilder();
    await screen.findByLabelText('Network name');
    await new Promise((resolve) => setTimeout(resolve, 900));
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });

  it('starts once on a first visit and opens again from the ? button', async () => {
    vi.stubEnv('MODE', 'development');
    const { unmount } = renderBuilder();
    await firstStep();
    await userEvent.click(screen.getByRole('button', { name: 'Skip' }));
    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
    unmount();

    // The next visit in this browser: no guide …
    renderBuilder();
    await screen.findByLabelText('Network name');
    await new Promise((resolve) => setTimeout(resolve, 900));
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    // … until it is asked for.
    await userEvent.click(screen.getByRole('button', { name: 'Show the guide' }));
    expect(await firstStep()).toBeInTheDocument();
  });

  it('survives a browser without storage', async () => {
    const getItem = vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
      throw new Error('denied');
    });
    const setItem = vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new Error('denied');
    });
    expect(guideSeen()).toBe(true); // never, rather than on every visit
    const onClose = vi.fn();
    render(<OnboardingGuide opened onClose={onClose} />);
    await firstStep();
    await userEvent.click(screen.getByRole('button', { name: 'Skip' }));
    expect(onClose).toHaveBeenCalled();
    getItem.mockRestore();
    setItem.mockRestore();
  });
});

describe('The empty canvas', () => {
  it('offers one example of each kind', () => {
    expect(startExamples(EXAMPLES).map((e) => e.id)).toEqual([
      'mlp_mnist',
      'cnn_mnist',
      'unet',
      'gp_regression',
    ]);
    // A server with other examples: the first four.
    expect(startExamples(EXAMPLES.slice(3, 4)).map((e) => e.id)).toEqual(['resnet_block']);
  });

  it('starts from an example card', async () => {
    catalogState.examples = EXAMPLES;
    renderBuilder({ version: 1, nodes: [], edges: [] });
    expect(await screen.findByText('Start building')).toBeInTheDocument();
    expect(screen.getByText('Or start from an example')).toBeInTheDocument();
    const cards = screen.getAllByRole('button', { name: /^Start from the example/ });
    expect(cards.map((card) => card.getAttribute('aria-label'))).toEqual([
      'Start from the example MLP classifier (MNIST)',
      'Start from the example Small CNN (MNIST)',
      'Start from the example U-Net (image segmentation)',
      'Start from the example Gaussian process regression',
    ]);
    await userEvent.click(cards[2]);
    await waitFor(() => expect(loadDraft()?.name).toBe('U-Net (image segmentation)'));
    expect(screen.queryByText('Start building')).not.toBeInTheDocument();
  });
});
