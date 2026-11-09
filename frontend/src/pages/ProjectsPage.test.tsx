import { render, screen, userEvent, waitFor, within } from '@test-utils';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, vi } from 'vitest';
import { ModalsProvider } from '@mantine/modals';
import { notifications } from '@mantine/notifications';
import { api, type Project } from '@/api';
import { useAuth } from '@/auth/AuthContext';
import { loadDraft } from '@/builder/draft';
import { ProjectsPage } from './ProjectsPage';

vi.mock('@/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api')>();
  return {
    ...actual,
    api: {
      projects: {
        list: vi.fn(),
        update: vi.fn(),
        remove: vi.fn(),
        duplicate: vi.fn(),
        members: { list: vi.fn(), remove: vi.fn() },
        data: { list: vi.fn() },
      },
    },
  };
});

vi.mock('@/auth/AuthContext', () => ({ useAuth: vi.fn() }));

const GRAPH = { version: 1 as const, nodes: [], edges: [] };

function project(over: Partial<Project> & Pick<Project, 'id' | 'name'>): Project {
  return {
    description: '',
    graph: GRAPH,
    num_layers: 3,
    created_at: '2026-10-01T09:00:00Z',
    updated_at: '2026-11-08T16:20:00Z',
    owner: { id: 1, username: 'alex' },
    role: 'owner',
    version: 1,
    updated_by: 'alex',
    shared: false,
    link_access: 'none',
    share_token: null,
    ...over,
  };
}

const MINE = project({ id: 1, name: 'Small CNN', shared: true, version: 4 });
const PRIVATE = project({ id: 2, name: 'Tabular regression' });
const EDITING = project({
  id: 3,
  name: 'Residual CNN',
  owner: { id: 2, username: 'maria' },
  role: 'editor',
  shared: true,
  updated_by: 'maria',
  version: 9,
});
const VIEWING = project({
  id: 4,
  name: 'LSTM classifier',
  owner: { id: 3, username: 'jonas' },
  role: 'viewer',
  shared: true,
  updated_by: 'jonas',
});

function renderPage(path = '/projects') {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <ModalsProvider>
        <ProjectsPage />
      </ModalsProvider>
    </MemoryRouter>
  );
}

async function openMenu(name: string) {
  const card = await screen.findByRole('article', { name });
  await userEvent.click(within(card).getByRole('button', { name: 'Network actions' }));
  return screen.findByRole('menu');
}

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
  vi.mocked(api.projects.list).mockResolvedValue([MINE, PRIVATE, EDITING, VIEWING]);
});

describe('ProjectsPage', () => {
  it('lists own networks, with a mark on shared ones', async () => {
    renderPage();
    const mine = await screen.findByRole('article', { name: 'Small CNN' });
    expect(within(mine).getByText('Shared')).toBeInTheDocument();
    const other = screen.getByRole('article', { name: 'Tabular regression' });
    expect(within(other).queryByText('Shared')).not.toBeInTheDocument();
    expect(screen.queryByRole('article', { name: 'Residual CNN' })).not.toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /Mine/ })).toHaveAttribute('aria-selected', 'true');
  });

  it('lists networks shared with the user, with owner and role', async () => {
    renderPage();
    await userEvent.click(await screen.findByRole('tab', { name: /Shared with me/ }));
    const editing = screen.getByRole('article', { name: 'Residual CNN' });
    expect(within(editing).getByText('by maria')).toBeInTheDocument();
    expect(within(editing).getByText('Can edit')).toBeInTheDocument();
    expect(within(editing).getByText(/Updated .* by maria/)).toBeInTheDocument();
    const viewing = screen.getByRole('article', { name: 'LSTM classifier' });
    expect(within(viewing).getByText('by jonas')).toBeInTheDocument();
    expect(within(viewing).getByText('Can view')).toBeInTheDocument();
    expect(screen.queryByRole('article', { name: 'Small CNN' })).not.toBeInTheDocument();
  });

  it('opens the shared tab from the address', async () => {
    renderPage('/projects?tab=shared');
    expect(await screen.findByRole('article', { name: 'Residual CNN' })).toBeInTheDocument();
  });

  it('only offers deleting to the owner and leaving to members', async () => {
    renderPage();
    let menu = await openMenu('Small CNN');
    expect(within(menu).getByRole('menuitem', { name: 'Delete' })).toBeInTheDocument();
    expect(within(menu).queryByRole('menuitem', { name: 'Leave' })).not.toBeInTheDocument();
    await userEvent.keyboard('{Escape}');

    await userEvent.click(screen.getByRole('tab', { name: /Shared with me/ }));
    menu = await openMenu('LSTM classifier');
    expect(within(menu).getByRole('menuitem', { name: 'Leave' })).toBeInTheDocument();
    expect(within(menu).queryByRole('menuitem', { name: 'Delete' })).not.toBeInTheDocument();
    // Viewers cannot rename.
    expect(within(menu).queryByRole('menuitem', { name: 'Rename' })).not.toBeInTheDocument();
  });

  it('leaves a shared network after a confirmation', async () => {
    vi.mocked(api.projects.members.list).mockResolvedValue({
      owner: { id: 2, username: 'maria' },
      members: [
        { id: 30, user: { id: 3, username: 'jonas' }, role: 'viewer', created_at: '' },
        { id: 31, user: { id: 1, username: 'alex' }, role: 'editor', created_at: '' },
      ],
    });
    vi.mocked(api.projects.members.remove).mockResolvedValue(undefined);
    renderPage('/projects?tab=shared');

    const menu = await openMenu('Residual CNN');
    await userEvent.click(within(menu).getByRole('menuitem', { name: 'Leave' }));
    const dialog = await screen.findByRole('dialog', { name: 'Leave “Residual CNN”?' });
    expect(within(dialog).getByText(/Only maria can invite you again/)).toBeInTheDocument();
    await userEvent.click(within(dialog).getByRole('button', { name: 'Leave' }));

    expect(api.projects.members.list).toHaveBeenCalledWith(3);
    await waitFor(() => expect(api.projects.members.remove).toHaveBeenCalledWith(3, 31));
    await waitFor(() =>
      expect(screen.queryByRole('article', { name: 'Residual CNN' })).not.toBeInTheDocument()
    );
    expect(screen.getByRole('article', { name: 'LSTM classifier' })).toBeInTheDocument();
  });

  it('duplicates a shared network into the own networks', async () => {
    const copy = project({ id: 7, name: 'LSTM classifier (copy)' });
    vi.mocked(api.projects.duplicate).mockResolvedValue(copy);
    renderPage('/projects?tab=shared');

    const menu = await openMenu('LSTM classifier');
    await userEvent.click(within(menu).getByRole('menuitem', { name: 'Duplicate' }));
    expect(api.projects.duplicate).toHaveBeenCalledWith(4);
    await userEvent.click(screen.getByRole('tab', { name: /Mine/ }));
    expect(await screen.findByRole('article', { name: 'LSTM classifier (copy)' })).toBeVisible();
  });

  it('opens a network in the builder with its version and role', async () => {
    renderPage('/projects?tab=shared');
    const card = await screen.findByRole('article', { name: 'Residual CNN' });
    await userEvent.click(within(card).getByRole('button', { name: 'Open in builder' }));
    expect(loadDraft()).toEqual({
      doc: GRAPH,
      name: 'Residual CNN',
      projectId: 3,
      version: 9,
      role: 'editor',
      saved: true,
    });
  });

  it('opens the share dialog from the menu', async () => {
    vi.mocked(api.projects.members.list).mockResolvedValue({
      owner: { id: 1, username: 'alex' },
      members: [],
    });
    renderPage();
    const menu = await openMenu('Tabular regression');
    await userEvent.click(within(menu).getByRole('menuitem', { name: 'Share…' }));
    const dialog = await screen.findByRole('dialog', { name: 'Share “Tabular regression”' });
    expect(await within(dialog).findByText(/Only you can open this network/)).toBeInTheDocument();
  });

  it('shows the data shared in a network from the menu', async () => {
    vi.mocked(api.projects.data.list).mockResolvedValue({
      datasets: [],
      preps: [{ id: 10, name: 'Churn', owner: 'jonas', created_at: '2026-11-01T10:00:00Z' }],
    });
    renderPage('/projects?tab=shared');
    const menu = await openMenu('LSTM classifier');
    await userEvent.click(within(menu).getByRole('menuitem', { name: 'Shared data…' }));
    const dialog = await screen.findByRole('dialog', { name: 'Share “LSTM classifier”' });
    expect(within(dialog).getByRole('tab', { name: 'Data' })).toHaveAttribute(
      'aria-selected',
      'true'
    );
    const preps = await within(dialog).findByRole('list', { name: 'Data preparations' });
    expect(within(preps).getByRole('link', { name: 'Churn' })).toHaveAttribute('href', '/data/10');
    expect(api.projects.data.list).toHaveBeenCalledWith(4, expect.any(AbortSignal));
  });
});
