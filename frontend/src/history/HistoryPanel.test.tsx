import { act, render, screen, userEvent, waitFor, within } from '@test-utils';
import { beforeEach, vi } from 'vitest';
import { notifications } from '@mantine/notifications';
import { api, ApiError, type Project, type ProjectRole, type ProjectVersion } from '@/api';
import { useAuth } from '@/auth/AuthContext';
import type { ProjectEvent } from '@/collab/events';
import { minutesAgo, version } from '@/collab/fixtures';
import { HistoryPanel } from './HistoryPanel';

vi.mock('@/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api')>();
  return {
    ...actual,
    api: {
      projects: {
        versions: {
          list: vi.fn(),
          get: vi.fn(),
          create: vi.fn(),
          rename: vi.fn(),
          restore: vi.fn(),
        },
      },
    },
  };
});
vi.mock('@/auth/AuthContext', () => ({ useAuth: vi.fn() }));
// The canvas itself is tested with the builder; here it only shows what it would draw.
vi.mock('@/builder/NetworkViewer', () => ({
  NetworkViewer: ({ doc, tips }: { doc: { nodes: unknown[] }; tips?: string[] }) => (
    <div data-testid="viewer">
      <p>{doc.nodes.length} layers on the canvas</p>
      <ul aria-label="Tips">
        {tips?.map((tip) => (
          <li key={tip}>{tip}</li>
        ))}
      </ul>
    </div>
  ),
}));

const DAY = 24 * 60;
const HISTORY: ProjectVersion[] = [
  version({ id: 14, version: 9, created_by: 'bob', updated_at: minutesAgo(2), num_layers: 4 }),
  version({
    id: 13,
    version: 8,
    kind: 'restore',
    label: 'Restored Baseline',
    created_by: 'alice',
    updated_at: minutesAgo(20),
  }),
  version({ id: 12, version: 7, created_by: 'alice', updated_at: minutesAgo(DAY + 30) }),
  version({
    id: 10,
    version: 5,
    kind: 'named',
    label: 'Baseline',
    created_by: 'bob',
    num_layers: 1,
    updated_at: minutesAgo(5 * DAY),
  }),
];

const RESTORED: Project = {
  id: 5,
  name: 'Shared MLP',
  description: '',
  graph: { version: 1, nodes: [], edges: [] },
  num_layers: 1,
  created_at: '',
  updated_at: '',
  owner: { id: 1, username: 'alice' },
  role: 'editor',
  version: 10,
  updated_by: 'alice',
  shared: true,
  link_access: 'none',
  share_token: null,
};

function renderPanel(
  role: ProjectRole = 'editor',
  props: Partial<React.ComponentProps<typeof HistoryPanel>> = {}
) {
  const onRestored = vi.fn();
  render(<HistoryPanel projectId={5} role={role} onRestored={onRestored} {...props} />);
  return { onRestored };
}

const entry = (name: RegExp | string) => screen.getByRole('button', { name });

beforeEach(() => {
  vi.clearAllMocks();
  vi.spyOn(notifications, 'show').mockImplementation(() => 'id');
  vi.mocked(useAuth).mockReturnValue({
    user: { id: 1, username: 'alice', email: '', date_joined: '' },
    ready: true,
    login: vi.fn(),
    register: vi.fn(),
    logout: vi.fn(),
  });
  vi.mocked(api.projects.versions.list).mockResolvedValue(HISTORY);
  vi.mocked(api.projects.versions.get).mockImplementation(async (_project, id) => ({
    ...HISTORY.find((v) => v.id === id)!,
    description: '',
    graph: { version: 1, nodes: [{ id: 'in', type: 'Input', params: {} }], edges: [] },
  }));
});

describe('HistoryPanel', () => {
  it('lists the history by day with names, authors and layer counts', async () => {
    renderPanel();
    await screen.findByText('Edited by bob');
    expect(api.projects.versions.list).toHaveBeenCalledWith(5);

    const today = screen.getByRole('region', { name: 'Today' });
    expect(within(today).getByText('Edited by bob')).toBeInTheDocument();
    expect(within(today).getByText('Restored Baseline')).toBeInTheDocument();
    expect(within(today).getByText('Current')).toBeInTheDocument();
    expect(within(today).getByText('2 min ago · 4 layers')).toBeInTheDocument();
    expect(within(today).getByText('you · 20 min ago · 2 layers')).toBeInTheDocument();

    const yesterday = screen.getByRole('region', { name: 'Yesterday' });
    expect(within(yesterday).getByText('Edited by you')).toBeInTheDocument();

    const regions = screen.getAllByRole('region');
    expect(regions).toHaveLength(3);
    expect(within(regions[2]).getByText('Baseline')).toBeInTheDocument();
    expect(within(regions[2]).getByText(/^bob · .* · 1 layer$/)).toBeInTheDocument();
    // Screen readers hear the kind of each entry.
    expect(entry(/Named version: Baseline/)).toBeInTheDocument();
    expect(entry(/Automatic version: Edited by bob/)).toBeInTheDocument();
  });

  it('names the current version', async () => {
    const named = version({ id: 15, version: 9, kind: 'named', label: 'Wider layers' });
    vi.mocked(api.projects.versions.create).mockResolvedValue(named);
    renderPanel('editor', { unsaved: true });
    await screen.findByText('Edited by bob');

    await userEvent.click(screen.getByRole('button', { name: 'Name current version' }));
    expect(screen.getByText(/Your unsaved changes are not part of it/)).toBeInTheDocument();
    const field = screen.getByRole('textbox', { name: /Name the current version/ });
    await userEvent.click(screen.getByRole('button', { name: 'Save name' }));
    expect(screen.getByText('Give the version a name.')).toBeInTheDocument();

    vi.mocked(api.projects.versions.list).mockResolvedValue([named, ...HISTORY]);
    await userEvent.type(field, 'Wider layers{Enter}');
    expect(api.projects.versions.create).toHaveBeenCalledWith(5, 'Wider layers');
    expect(await screen.findByText('Wider layers')).toBeInTheDocument();
    expect(screen.queryByRole('textbox')).not.toBeInTheDocument();
  });

  it('shows why a name was rejected', async () => {
    vi.mocked(api.projects.versions.create).mockRejectedValue(
      new ApiError(400, { label: ['Give the version a name.'] }, 'Label: Give the version a name.')
    );
    renderPanel('owner');
    await screen.findByText('Edited by bob');
    await userEvent.click(screen.getByRole('button', { name: 'Name current version' }));
    await userEvent.type(screen.getByRole('textbox'), '  x {Enter}');
    expect(await screen.findByText('Give the version a name.')).toBeInTheDocument();
  });

  it('renames an entry', async () => {
    vi.mocked(api.projects.versions.rename).mockResolvedValue({
      ...HISTORY[2],
      kind: 'named',
      label: 'Before the dropout',
    });
    renderPanel();
    await screen.findByText('Edited by you');

    await userEvent.click(screen.getByRole('button', { name: 'Actions for “Edited by you”' }));
    await userEvent.click(await screen.findByRole('menuitem', { name: 'Name this version' }));
    await userEvent.type(
      screen.getByRole('textbox', { name: 'Name this version' }),
      'Before the dropout{Enter}'
    );
    expect(api.projects.versions.rename).toHaveBeenCalledWith(5, 12, 'Before the dropout');
    expect(await screen.findByText('Before the dropout')).toBeInTheDocument();
    expect(screen.queryByText('Edited by you')).not.toBeInTheDocument();

    // Named versions are renamed.
    await userEvent.click(screen.getByRole('button', { name: 'Actions for “Baseline”' }));
    expect(await screen.findByRole('menuitem', { name: 'Rename' })).toBeInTheDocument();
  });

  it('previews a version and restores it', async () => {
    vi.mocked(api.projects.versions.restore).mockResolvedValue(RESTORED);
    const { onRestored } = renderPanel('editor', { unsaved: true });
    await userEvent.click(await screen.findByRole('button', { name: /Baseline.*1 layer/ }));

    const dialog = await screen.findByRole('dialog');
    expect(api.projects.versions.get).toHaveBeenCalledWith(5, 10, expect.any(AbortSignal));
    expect(await within(dialog).findByTestId('viewer')).toHaveTextContent('1 layers on the canvas');
    // The tips of the canvas fit the preview (not those of a shared network).
    const tips = within(within(dialog).getByRole('list', { name: 'Tips' })).getAllByRole(
      'listitem'
    );
    expect(tips.map((tip) => tip.textContent)).toEqual([
      'This is an earlier version of the network, read only.',
      'Click a layer to see its settings here.',
      'Restore this version to continue from it. The current state stays in the history.',
    ]);
    await userEvent.click(within(dialog).getByRole('button', { name: 'Restore this version' }));
    expect(within(dialog).getByText('Restore “Baseline”?')).toBeInTheDocument();
    expect(within(dialog).getByText(/the current state stays in the history/)).toBeInTheDocument();
    expect(
      within(dialog).getByText(/unsaved changes on the canvas are replaced/)
    ).toBeInTheDocument();

    await userEvent.click(within(dialog).getByRole('button', { name: 'Restore' }));
    expect(api.projects.versions.restore).toHaveBeenCalledWith(5, 10);
    await waitFor(() => expect(onRestored).toHaveBeenCalledWith(RESTORED));
    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
    expect(notifications.show).toHaveBeenCalledWith(
      expect.objectContaining({ id: 'live-replaced', title: 'Restored “Baseline”' })
    );
    // The list is loaded again (it has the restore now).
    expect(api.projects.versions.list).toHaveBeenCalledTimes(2);
  });

  it('asks before restoring from the menu and keeps the preview open on errors', async () => {
    vi.mocked(api.projects.versions.restore).mockRejectedValue(new Error('View only.'));
    const { onRestored } = renderPanel('owner');
    await screen.findByText('Edited by you');
    await userEvent.click(screen.getByRole('button', { name: 'Actions for “Edited by you”' }));
    await userEvent.click(await screen.findByRole('menuitem', { name: 'Restore…' }));

    const dialog = await screen.findByRole('dialog');
    expect(within(dialog).getByText('Restore “Edited by you”?')).toBeInTheDocument();
    await within(dialog).findByTestId('viewer');
    await userEvent.click(within(dialog).getByRole('button', { name: 'Restore' }));
    expect(await within(dialog).findByText('View only.')).toBeInTheDocument();
    expect(onRestored).not.toHaveBeenCalled();
  });

  it('does not offer to restore the current state', async () => {
    renderPanel('owner');
    await screen.findByText('Edited by bob');
    await userEvent.click(screen.getByRole('button', { name: 'Actions for “Edited by bob”' }));
    expect(await screen.findByRole('menuitem', { name: 'Preview' })).toBeInTheDocument();
    expect(screen.queryByRole('menuitem', { name: 'Restore…' })).not.toBeInTheDocument();
  });

  it('is read only for viewers', async () => {
    renderPanel('viewer');
    await screen.findByText('Edited by bob');
    expect(screen.queryByRole('button', { name: 'Name current version' })).not.toBeInTheDocument();
    expect(screen.getByText(/The owner and editors can name and restore them/)).toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: 'Actions for “Baseline”' }));
    expect(await screen.findByRole('menuitem', { name: 'Preview' })).toBeInTheDocument();
    expect(screen.queryByRole('menuitem', { name: 'Rename' })).not.toBeInTheDocument();
    expect(screen.queryByRole('menuitem', { name: 'Restore…' })).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole('menuitem', { name: 'Preview' }));
    const dialog = await screen.findByRole('dialog');
    await within(dialog).findByTestId('viewer');
    expect(
      within(dialog).queryByRole('button', { name: 'Restore this version' })
    ).not.toBeInTheDocument();
    expect(
      within(dialog).getByText(/The owner and editors can restore earlier versions/)
    ).toBeInTheDocument();
    expect(within(dialog).getByText('The owner and editors can restore it.')).toBeInTheDocument();
  });

  it('loads the history again on live messages', async () => {
    const handlers = new Set<(event: ProjectEvent) => void>();
    const subscribe = (handler: (event: ProjectEvent) => void) => {
      handlers.add(handler);
      return () => handlers.delete(handler);
    };
    const editor: ProjectRole = 'editor';
    const { unmount } = render(
      <HistoryPanel projectId={5} role={editor} subscribe={subscribe} live onRestored={vi.fn()} />
    );
    await screen.findByText('Edited by bob');
    expect(api.projects.versions.list).toHaveBeenCalledTimes(1);

    vi.mocked(api.projects.versions.list).mockResolvedValue([
      version({ id: 16, kind: 'named', label: 'From maria' }),
      ...HISTORY,
    ]);
    act(() => handlers.forEach((handler) => handler({ type: 'history' })));
    expect(await screen.findByText('From maria')).toBeInTheDocument();
    // Other messages are no reason to load it at once.
    act(() => handlers.forEach((handler) => handler({ type: 'presence' })));
    expect(api.projects.versions.list).toHaveBeenCalledTimes(2);

    unmount();
    expect(handlers.size).toBe(0);
  });
});
