/** Links to /app?example=<id>, from the example pages of the website (frontend/landing/). */
import { render, screen, userEvent, waitFor } from '@test-utils';
import { MemoryRouter, useLocation } from 'react-router-dom';
import { beforeEach, vi } from 'vitest';
import type { Example, GraphDocument } from '@/api';
import { useAuth } from '@/auth/AuthContext';
import { LocalizedModalsProvider } from '@/i18n/mantine';
import { useTraining, type TrainingController } from '@/training/useTraining';
import { BuilderPage } from './BuilderPage';
import { CLASSICAL_CATALOG } from './classicalFixtures';
import { saveDraft } from './draft';

vi.mock('@/auth/AuthContext', () => ({ useAuth: vi.fn() }));
vi.mock('@/training/useTraining', () => ({ useTraining: vi.fn() }));
vi.mock('./useCatalog', () => ({
  useCatalog: () => ({
    catalog: CLASSICAL_CATALOG,
    examples: EXAMPLES,
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

const EXAMPLES = [example('mlp_mnist', 'MLP classifier (MNIST)'), example('unet', 'U-Net')];

function Address() {
  const location = useLocation();
  return <output data-testid="address">{location.pathname + location.search}</output>;
}

function renderAt(url: string) {
  return render(
    <LocalizedModalsProvider>
      <MemoryRouter initialEntries={[url]}>
        <BuilderPage />
        <Address />
      </MemoryRouter>
    </LocalizedModalsProvider>
  );
}

const networkName = () => screen.findByLabelText('Network name');

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

describe('A link with ?example=', () => {
  it('opens that example on a first visit and drops the parameter', async () => {
    renderAt('/app?example=unet&x=1');
    await waitFor(async () => expect(await networkName()).toHaveValue('U-Net'));
    expect(screen.getByTestId('address')).toHaveTextContent(/^\/app\?x=1$/);
  });

  it('asks before replacing a network on the canvas', async () => {
    saveDraft({ doc: GRAPH, name: 'Mine', projectId: null });
    renderAt('/app?example=unet');
    expect(await screen.findByText('Replace the current network?')).toBeInTheDocument();
    expect(await networkName()).toHaveValue('Mine');
    await userEvent.click(screen.getByRole('button', { name: 'Replace' }));
    await waitFor(async () => expect(await networkName()).toHaveValue('U-Net'));
    expect(screen.getByTestId('address')).toHaveTextContent(/^\/app$/);
  });

  it('shows the first example for an unknown id', async () => {
    renderAt('/app?example=nope');
    await waitFor(async () => expect(await networkName()).toHaveValue('MLP classifier (MNIST)'));
    expect(screen.getByTestId('address')).toHaveTextContent(/^\/app$/);
  });
});
