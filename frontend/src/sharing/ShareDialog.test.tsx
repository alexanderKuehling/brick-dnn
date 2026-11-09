import { render, screen, userEvent, waitFor, within } from '@test-utils';
import { beforeEach, vi } from 'vitest';
import { notifications } from '@mantine/notifications';
import { api, ApiError, type ProjectData, type ProjectMember, type ProjectMembers } from '@/api';
import { useAuth } from '@/auth/AuthContext';
import { PREP } from '@/data/fixtures';
import { dataset } from '@/training/fixtures';
import { ShareDialog, type ShareTab, type ShareTarget } from './ShareDialog';

vi.mock('@/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api')>();
  return {
    ...actual,
    api: {
      projects: {
        members: { list: vi.fn(), add: vi.fn(), update: vi.fn(), remove: vi.fn() },
        data: { list: vi.fn(), add: vi.fn(), remove: vi.fn() },
        setLink: vi.fn(),
      },
      training: { datasets: { list: vi.fn() } },
      data: { preps: { list: vi.fn() } },
    },
  };
});

vi.mock('@/auth/AuthContext', () => ({ useAuth: vi.fn() }));

const ALEX = { id: 1, username: 'alex' };
const MARIA = { id: 2, username: 'maria' };
const JONAS = { id: 3, username: 'jonas' };

const member = (id: number, user: typeof MARIA, role: ProjectMember['role']): ProjectMember => ({
  id,
  user,
  role,
  created_at: '2026-11-01T10:00:00Z',
});

const OWNED: ProjectMembers = { owner: ALEX, members: [member(11, MARIA, 'editor')] };

const PROJECT: ShareTarget = {
  id: 5,
  name: 'Small CNN',
  role: 'owner',
  link_access: 'none',
  share_token: null,
};

function loginAs(user: typeof ALEX) {
  vi.mocked(useAuth).mockReturnValue({
    user: { ...user, email: '', date_joined: '2026-01-01T00:00:00Z' },
    ready: true,
    login: vi.fn(),
    register: vi.fn(),
    logout: vi.fn(),
  });
}

function renderDialog(project: ShareTarget = PROJECT, initialTab?: ShareTab) {
  const onClose = vi.fn();
  const onChange = vi.fn();
  const onLeave = vi.fn();
  render(
    <ShareDialog
      project={project}
      onClose={onClose}
      onChange={onChange}
      onLeave={onLeave}
      initialTab={initialTab}
    />
  );
  return { onClose, onChange, onLeave };
}

const people = () => screen.getByRole('list', { name: 'People with access' });

beforeEach(() => {
  vi.clearAllMocks();
  vi.spyOn(notifications, 'show').mockImplementation(() => 'id');
  loginAs(ALEX);
  vi.mocked(api.projects.members.list).mockResolvedValue(OWNED);
});

describe('ShareDialog for the owner', () => {
  it('shows the owner first, then the members', async () => {
    renderDialog();
    const owner = await screen.findByRole('listitem', { name: 'alex' });
    expect(within(owner).getByText('(you)')).toBeInTheDocument();
    expect(within(owner).getByText('Owner')).toBeInTheDocument();
    const maria = screen.getByRole('listitem', { name: 'maria' });
    expect(within(maria).getByRole('combobox', { name: 'Role of maria' })).toHaveValue('Can edit');
    expect(within(people()).getAllByRole('listitem')).toHaveLength(2);
    expect(api.projects.members.list).toHaveBeenCalledWith(5);
  });

  it('invites people by username or e-mail', async () => {
    const { onChange } = renderDialog();
    await screen.findByRole('listitem', { name: 'maria' });
    vi.mocked(api.projects.members.add).mockResolvedValue(member(12, JONAS, 'viewer'));

    await userEvent.type(screen.getByRole('textbox', { name: 'Username or e-mail' }), ' jonas ');
    await userEvent.click(screen.getByRole('combobox', { name: 'Role of the invited person' }));
    await userEvent.click(await screen.findByRole('option', { name: 'Can view' }));
    await userEvent.click(screen.getByRole('button', { name: 'Invite' }));

    expect(api.projects.members.add).toHaveBeenCalledWith(5, 'jonas', 'viewer');
    const jonas = await screen.findByRole('listitem', { name: 'jonas' });
    expect(within(jonas).getByRole('combobox', { name: 'Role of jonas' })).toHaveValue('Can view');
    expect(screen.getByRole('textbox', { name: 'Username or e-mail' })).toHaveValue('');
    expect(onChange).toHaveBeenLastCalledWith(
      expect.objectContaining({
        shared: true,
        members: {
          owner: ALEX,
          members: [member(11, MARIA, 'editor'), member(12, JONAS, 'viewer')],
        },
      })
    );
  });

  it('shows why an invitation failed below the field', async () => {
    renderDialog();
    await screen.findByRole('listitem', { name: 'maria' });
    const field = screen.getByRole('textbox', { name: 'Username or e-mail' });

    await userEvent.click(screen.getByRole('button', { name: 'Invite' }));
    expect(await screen.findByText('Enter a username or an e-mail address.')).toBeInTheDocument();
    expect(api.projects.members.add).not.toHaveBeenCalled();

    vi.mocked(api.projects.members.add).mockRejectedValueOnce(
      new ApiError(400, { user: ['There is no user “nobody”.'] }, 'User: There is no user.')
    );
    await userEvent.type(field, 'nobody');
    await userEvent.click(screen.getByRole('button', { name: 'Invite' }));
    expect(await screen.findByText('There is no user “nobody”.')).toBeInTheDocument();
    expect(field).toHaveAttribute('aria-invalid', 'true');

    // The member limit comes as `detail`.
    vi.mocked(api.projects.members.add).mockRejectedValueOnce(
      new ApiError(400, { detail: 'A network can have at most 50 members.' }, 'limit')
    );
    await userEvent.click(screen.getByRole('button', { name: 'Invite' }));
    expect(await screen.findByText('A network can have at most 50 members.')).toBeInTheDocument();
    expect(notifications.show).not.toHaveBeenCalled();
  });

  it('changes roles and removes members', async () => {
    const { onChange } = renderDialog();
    const maria = await screen.findByRole('listitem', { name: 'maria' });
    vi.mocked(api.projects.members.update).mockResolvedValue(member(11, MARIA, 'viewer'));

    await userEvent.click(within(maria).getByRole('combobox', { name: 'Role of maria' }));
    await userEvent.click(await screen.findByRole('option', { name: 'Can view' }));
    expect(api.projects.members.update).toHaveBeenCalledWith(5, 11, 'viewer');
    await waitFor(() =>
      expect(within(maria).getByRole('combobox', { name: 'Role of maria' })).toHaveValue('Can view')
    );

    vi.mocked(api.projects.members.remove).mockResolvedValue(undefined);
    await userEvent.click(within(maria).getByRole('button', { name: 'Remove maria' }));
    expect(api.projects.members.remove).toHaveBeenCalledWith(5, 11);
    await waitFor(() =>
      expect(screen.queryByRole('listitem', { name: 'maria' })).not.toBeInTheDocument()
    );
    expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ shared: false }));
  });

  it('turns the view link on, copies it and replaces it', async () => {
    const user = userEvent.setup(); // With a stub of the clipboard.
    const { onChange } = renderDialog();
    await screen.findByRole('listitem', { name: 'maria' });
    expect(screen.queryByRole('textbox', { name: 'Share link' })).not.toBeInTheDocument();

    vi.mocked(api.projects.setLink).mockResolvedValueOnce({
      link_access: 'view',
      share_token: 'tok1',
    });
    await user.click(screen.getByRole('switch', { name: /Anyone with the link can view/ }));
    expect(api.projects.setLink).toHaveBeenCalledWith(5, true, false);
    const link = await screen.findByRole('textbox', { name: 'Share link' });
    expect(link).toHaveValue(`${window.location.origin}/share/tok1`);
    expect(onChange).toHaveBeenLastCalledWith(
      expect.objectContaining({ link_access: 'view', share_token: 'tok1', shared: true })
    );

    await user.click(screen.getByRole('button', { name: 'Copy link' }));
    expect(await navigator.clipboard.readText()).toBe(`${window.location.origin}/share/tok1`);
    expect(screen.getByRole('button', { name: 'Copied' })).toBeInTheDocument();

    // A new link needs a confirmation: the old one stops working.
    await user.click(screen.getByRole('button', { name: 'Create new link' }));
    expect(screen.getByText(/The current link stops working/)).toBeInTheDocument();
    expect(api.projects.setLink).toHaveBeenCalledTimes(1);
    vi.mocked(api.projects.setLink).mockResolvedValueOnce({
      link_access: 'view',
      share_token: 'tok2',
    });
    const buttons = screen.getAllByRole('button', { name: 'Create new link' });
    await user.click(buttons[buttons.length - 1]);
    expect(api.projects.setLink).toHaveBeenLastCalledWith(5, true, true);
    await waitFor(() => expect(link).toHaveValue(`${window.location.origin}/share/tok2`));
    expect(screen.queryByText(/The current link stops working/)).not.toBeInTheDocument();

    vi.mocked(api.projects.setLink).mockResolvedValueOnce({
      link_access: 'none',
      share_token: null,
    });
    await user.click(screen.getByRole('switch', { name: /Anyone with the link can view/ }));
    expect(api.projects.setLink).toHaveBeenLastCalledWith(5, false, false);
    await waitFor(() =>
      expect(screen.queryByRole('textbox', { name: 'Share link' })).not.toBeInTheDocument()
    );
  });

  it('reports errors of the link as a notification', async () => {
    renderDialog({ ...PROJECT, link_access: 'view', share_token: 'tok1' });
    await screen.findByRole('listitem', { name: 'maria' });
    vi.mocked(api.projects.setLink).mockRejectedValue(new ApiError(500, null, 'Server error.'));
    await userEvent.click(screen.getByRole('switch', { name: /Anyone with the link can view/ }));
    await waitFor(() =>
      expect(notifications.show).toHaveBeenCalledWith(
        expect.objectContaining({ color: 'red', message: 'Server error.' })
      )
    );
    expect(screen.getByRole('switch', { name: /Anyone with the link can view/ })).toBeChecked();
  });
});

describe('ShareDialog for members', () => {
  const SHARED: ProjectMembers = {
    owner: MARIA,
    members: [member(31, ALEX, 'editor'), member(32, JONAS, 'viewer')],
  };

  beforeEach(() => {
    vi.mocked(api.projects.members.list).mockResolvedValue(SHARED);
  });

  it('shows who has access, read only', async () => {
    renderDialog({ ...PROJECT, role: 'editor', link_access: 'view' });
    const owner = await screen.findByRole('listitem', { name: 'maria' });
    expect(within(owner).getByText('Owner')).toBeInTheDocument();
    const jonas = screen.getByRole('listitem', { name: 'jonas' });
    expect(within(jonas).getByText('Can view')).toBeInTheDocument();
    expect(screen.queryByRole('textbox', { name: 'Username or e-mail' })).not.toBeInTheDocument();
    expect(screen.queryByRole('combobox', { name: /Role of/ })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /Remove/ })).not.toBeInTheDocument();
    expect(screen.queryByRole('switch')).not.toBeInTheDocument();
    expect(screen.getByText(/Anyone with the link can view this network/)).toBeInTheDocument();
    // Only the own row has "Leave".
    expect(within(jonas).queryByRole('button')).not.toBeInTheDocument();
  });

  it('leaves the network after a confirmation', async () => {
    const { onLeave, onClose } = renderDialog({ ...PROJECT, role: 'editor' });
    const own = await screen.findByRole('listitem', { name: 'alex' });
    expect(within(own).getByText('Can edit')).toBeInTheDocument();

    await userEvent.click(within(own).getByRole('button', { name: 'Leave' }));
    expect(screen.getByText(/You lose access to it/)).toBeInTheDocument();
    expect(api.projects.members.remove).not.toHaveBeenCalled();
    vi.mocked(api.projects.members.remove).mockResolvedValue(undefined);
    const buttons = screen.getAllByRole('button', { name: 'Leave' });
    await userEvent.click(buttons[buttons.length - 1]);

    expect(api.projects.members.remove).toHaveBeenCalledWith(5, 31);
    await waitFor(() => expect(onLeave).toHaveBeenCalled());
    expect(onClose).toHaveBeenCalled();
  });
});

describe('ShareDialog, data of the network', () => {
  const SHARED_DATA: ProjectData = {
    datasets: [
      {
        id: 42,
        name: 'Churn v1',
        owner: 'maria',
        created_at: '2026-11-01T10:00:00Z',
        kind: 'csv',
        size_bytes: 51_200,
      },
    ],
    preps: [{ id: 10, name: 'Churn', owner: 'alex', created_at: '2026-11-01T10:00:00Z' }],
  };
  const dataList = (name: string) => screen.getByRole('list', { name });

  beforeEach(() => {
    vi.mocked(api.projects.data.list).mockResolvedValue(SHARED_DATA);
    vi.mocked(api.training.datasets.list).mockResolvedValue([
      dataset({ id: 7, name: 'Iris' }),
      dataset({ id: 42, name: 'Churn v1', mine: false, owner: 'maria' }),
    ]);
    vi.mocked(api.data.preps.list).mockResolvedValue([
      { ...PREP, name: 'Churn' },
      { ...PREP, id: 11, name: 'Fraud', owner: 'maria', role: 'editor' },
      { ...PREP, id: 12, name: 'Leads' },
    ]);
  });

  it('lists what is shared there and lets the owner share more or remove it', async () => {
    renderDialog();
    await userEvent.click(screen.getByRole('tab', { name: 'Data' }));
    const datasets = await screen.findByRole('list', { name: 'Training datasets' });
    expect(api.projects.data.list).toHaveBeenCalledWith(5, expect.any(AbortSignal));
    expect(within(datasets).getByRole('listitem', { name: 'Churn v1' })).toHaveTextContent(
      'by maria · CSV · 50 KB'
    );
    expect(dataList('Data preparations')).toHaveTextContent('Churnby you');

    // Only own items that are not shared there yet are offered.
    await userEvent.click(
      screen.getByRole('combobox', { name: 'Dataset or preparation to share' })
    );
    expect(await screen.findByRole('option', { name: 'Iris' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'Leads' })).toBeInTheDocument();
    expect(screen.queryByRole('option', { name: 'Churn' })).not.toBeInTheDocument();
    expect(screen.queryByRole('option', { name: 'Churn v1' })).not.toBeInTheDocument();
    expect(screen.queryByRole('option', { name: 'Fraud' })).not.toBeInTheDocument();

    vi.mocked(api.projects.data.add).mockResolvedValue({
      ...SHARED_DATA,
      preps: [...SHARED_DATA.preps, { id: 12, name: 'Leads', owner: 'alex', created_at: '' }],
    });
    await userEvent.click(screen.getByRole('option', { name: 'Leads' }));
    await userEvent.click(screen.getByRole('button', { name: 'Share' }));
    expect(api.projects.data.add).toHaveBeenCalledWith(5, 'prep', 12);
    expect(await within(dataList('Data preparations')).findByText('Leads')).toBeInTheDocument();

    // The owner of the network removes anything.
    vi.mocked(api.projects.data.remove).mockResolvedValue(undefined);
    await userEvent.click(screen.getByRole('button', { name: 'Stop sharing Churn v1' }));
    expect(api.projects.data.remove).toHaveBeenCalledWith(5, 'dataset', 42);
    await waitFor(() =>
      expect(screen.queryByRole('list', { name: 'Training datasets' })).not.toBeInTheDocument()
    );
  });

  it('lets editors share their own data and remove only that', async () => {
    renderDialog({ ...PROJECT, role: 'editor' }, 'data');
    const datasets = await screen.findByRole('list', { name: 'Training datasets' });
    expect(
      screen.getByRole('combobox', { name: 'Dataset or preparation to share' })
    ).toBeInTheDocument();
    expect(within(datasets).queryByRole('button')).not.toBeInTheDocument(); // maria's
    expect(
      within(dataList('Data preparations')).getByRole('button', { name: 'Stop sharing Churn' })
    ).toBeInTheDocument();
  });

  it('shows viewers what is shared, read only', async () => {
    vi.mocked(api.projects.data.list).mockResolvedValue({ datasets: [], preps: [] });
    renderDialog({ ...PROJECT, role: 'viewer' }, 'data');
    expect(await screen.findByText(/Nothing is shared here yet/)).toHaveTextContent(
      'The owner and the editors of the network can share their data here.'
    );
    expect(
      screen.queryByRole('combobox', { name: 'Dataset or preparation to share' })
    ).not.toBeInTheDocument();
    expect(api.training.datasets.list).not.toHaveBeenCalled();
  });

  it('reports what could not be shared', async () => {
    vi.mocked(api.projects.data.add).mockRejectedValue(
      new ApiError(400, null, 'Choose one of your own datasets or data preparations.')
    );
    renderDialog(PROJECT, 'data');
    await screen.findByRole('list', { name: 'Training datasets' });
    await userEvent.click(
      screen.getByRole('combobox', { name: 'Dataset or preparation to share' })
    );
    await userEvent.click(await screen.findByRole('option', { name: 'Iris' }));
    await userEvent.click(screen.getByRole('button', { name: 'Share' }));
    await waitFor(() =>
      expect(notifications.show).toHaveBeenCalledWith(
        expect.objectContaining({ title: 'Could not share it' })
      )
    );
  });
});
