import { render, screen, userEvent, waitFor } from '@test-utils';
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom';
import { beforeEach, vi } from 'vitest';
import { ModalsProvider } from '@mantine/modals';
import { notifications } from '@mantine/notifications';
import { api, ApiError, type Project, type SharedNetwork } from '@/api';
import { useAuth } from '@/auth/AuthContext';
import { loadDraft } from '@/builder/draft';
import { SharedNetworkPage } from './SharedNetworkPage';

vi.mock('@/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api')>();
  return {
    ...actual,
    api: { shared: { get: vi.fn(), duplicate: vi.fn() }, generate: vi.fn() },
  };
});
vi.mock('@/auth/AuthContext', () => ({ useAuth: vi.fn() }));
// The canvas itself is the builder's (tested there).
vi.mock('@/builder/NetworkViewer', () => ({
  NetworkViewer: ({ doc }: { doc: { nodes: unknown[] } }) => (
    <div data-testid="viewer">{doc.nodes.length} layers</div>
  ),
}));

const NETWORK: SharedNetwork = {
  name: 'Small CNN',
  description: 'Two conv blocks.',
  graph: { version: 1, nodes: [{ id: 'in', type: 'Input', params: {} }], edges: [] },
  owner: 'alex',
  updated_at: '2026-11-08T16:20:00Z',
};

const COPY: Project = {
  id: 9,
  name: 'Small CNN (copy)',
  description: '',
  graph: NETWORK.graph,
  num_layers: 1,
  created_at: '2026-11-09T10:00:00Z',
  updated_at: '2026-11-09T10:00:00Z',
  owner: { id: 2, username: 'maria' },
  role: 'owner',
  version: 1,
  updated_by: 'maria',
  shared: false,
  link_access: 'none',
  share_token: null,
};

function Where() {
  const location = useLocation();
  return (
    <output data-testid="location">
      {location.pathname}
      {location.state ? ` ${JSON.stringify(location.state)}` : ''}
    </output>
  );
}

function renderPage(entry: string | { pathname: string; state: unknown } = '/share/tok1') {
  return render(
    <MemoryRouter initialEntries={[entry]}>
      <ModalsProvider>
        <Routes>
          <Route path="/share/:token" element={<SharedNetworkPage />} />
          <Route path="*" element={<p>Elsewhere</p>} />
        </Routes>
        <Where />
      </ModalsProvider>
    </MemoryRouter>
  );
}

function loggedIn(yes: boolean) {
  vi.mocked(useAuth).mockReturnValue({
    user: yes ? { id: 2, username: 'maria', email: '', date_joined: '2026-01-01T00:00:00Z' } : null,
    ready: true,
    login: vi.fn(),
    register: vi.fn(),
    logout: vi.fn(),
  });
}

beforeEach(() => {
  vi.clearAllMocks();
  localStorage.clear();
  vi.spyOn(notifications, 'show').mockImplementation(() => 'id');
  loggedIn(false);
  vi.mocked(api.shared.get).mockResolvedValue(NETWORK);
});

describe('SharedNetworkPage', () => {
  it('shows the network read only, with its owner', async () => {
    renderPage();
    expect(await screen.findByRole('heading', { name: 'Small CNN' })).toBeInTheDocument();
    expect(api.shared.get).toHaveBeenCalledWith('tok1');
    expect(screen.getByText(/^by alex · updated/)).toBeInTheDocument();
    expect(screen.getByText('View only')).toBeInTheDocument();
    expect(screen.getByText('Two conv blocks.')).toBeInTheDocument();
    expect(screen.getByTestId('viewer')).toHaveTextContent('1 layers');
  });

  it('shows the code', async () => {
    vi.mocked(api.generate).mockResolvedValue({ code: 'class SmallCNN: ...', filename: 'x.py' });
    renderPage();
    await userEvent.click(await screen.findByRole('button', { name: 'View code' }));
    const dialog = await screen.findByRole('dialog', { name: 'PyTorch code' });
    // Highlighted, so the code is split into several elements.
    await waitFor(() => expect(dialog).toHaveTextContent('class SmallCNN: ...'));
    expect(api.generate).toHaveBeenCalledWith(
      NETWORK.graph,
      expect.objectContaining({ filename: 'small_cnn' })
    );
  });

  it('asks to log in before duplicating, then comes back and duplicates', async () => {
    renderPage();
    await userEvent.click(await screen.findByRole('button', { name: 'Duplicate to my networks' }));
    expect(api.shared.duplicate).not.toHaveBeenCalled();
    expect(screen.getByTestId('location')).toHaveTextContent(
      '/login {"from":"/share/tok1","returnState":{"duplicate":true}}'
    );
  });

  it('duplicates after logging in and opens the copy in the builder', async () => {
    loggedIn(true);
    vi.mocked(api.shared.duplicate).mockResolvedValue(COPY);
    // The login page sends the user back with the state it was given.
    renderPage({ pathname: '/share/tok1', state: { duplicate: true } });
    await waitFor(() => expect(api.shared.duplicate).toHaveBeenCalledWith('tok1'));
    expect(api.shared.duplicate).toHaveBeenCalledTimes(1);
    await waitFor(() => expect(screen.getByTestId('location')).toHaveTextContent(/^\/$/));
    expect(loadDraft()).toMatchObject({ projectId: 9, name: 'Small CNN (copy)', role: 'owner' });
  });

  it('duplicates right away when logged in', async () => {
    loggedIn(true);
    vi.mocked(api.shared.duplicate).mockResolvedValue(COPY);
    renderPage();
    await userEvent.click(await screen.findByRole('button', { name: 'Duplicate to my networks' }));
    expect(api.shared.duplicate).toHaveBeenCalledWith('tok1');
    await waitFor(() => expect(screen.getByTestId('location')).toHaveTextContent(/^\/$/));
  });

  it('explains a link that does not work', async () => {
    vi.mocked(api.shared.get).mockRejectedValue(new ApiError(404, null, 'Not found.'));
    renderPage('/share/old');
    expect(
      await screen.findByRole('heading', { name: 'This link does not exist or was turned off' })
    ).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Duplicate to my networks' })).toBeNull();
  });

  it('offers to try again after other errors', async () => {
    vi.mocked(api.shared.get).mockRejectedValueOnce(new ApiError(0, null, 'Cannot reach it.'));
    renderPage();
    expect(await screen.findByText('Cannot reach it.')).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Try again' }));
    expect(await screen.findByRole('heading', { name: 'Small CNN' })).toBeInTheDocument();
  });
});
