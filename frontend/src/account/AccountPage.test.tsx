import { render, screen, userEvent, waitFor, within } from '@test-utils';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { afterEach, beforeEach, vi } from 'vitest';
import { notifications } from '@mantine/notifications';
import { api, ApiError, getTokens, setTokens, type Account } from '@/api';
import { AuthProvider } from '@/auth/AuthContext';
import { RequireAuth } from '@/components/RequireAuth';
import { resetLegalInfo } from '@/legal/useLegalInfo';
import { AccountPage, storedSummary } from './AccountPage';

vi.mock('@/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api')>();
  return {
    ...actual,
    api: {
      me: vi.fn(),
      logout: vi.fn(),
      legal: { info: vi.fn(), page: vi.fn() },
      account: {
        get: vi.fn(),
        changeEmail: vi.fn(),
        changePassword: vi.fn(),
        exportData: vi.fn(),
        remove: vi.fn(),
      },
    },
  };
});

vi.mock('@/builder/download', () => ({ downloadBlob: vi.fn() }));

const ACCOUNT: Account = {
  id: 1,
  username: 'ada',
  email: 'ada@example.com',
  date_joined: '2026-01-02T10:00:00Z',
  last_login: '2026-11-08T10:00:00Z',
  last_active_at: '2026-11-08T10:00:00Z',
  terms_accepted_at: '2026-01-02T10:00:00Z',
  terms_version: 'abc',
  privacy_version: 'def',
  stored: {
    projects: 3,
    memberships: 1,
    comments: 0,
    data_sources: 1,
    data_preps: 1,
    datasets: 2,
    training_runs: 5,
    api_tokens: 0,
    connected_apps: 0,
    upload_bytes: 3 * 1024 * 1024,
  },
};

function renderPage() {
  return render(
    <AuthProvider>
      <MemoryRouter initialEntries={['/account']}>
        <Routes>
          <Route
            path="/account"
            element={
              <RequireAuth>
                <AccountPage />
              </RequireAuth>
            }
          />
          <Route path="/" element={<p>Home page</p>} />
          <Route path="/login" element={<p>Login page</p>} />
        </Routes>
      </MemoryRouter>
    </AuthProvider>
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  resetLegalInfo();
  setTokens({ access: 'old-access', refresh: 'old-refresh' });
  vi.mocked(api.me).mockResolvedValue({
    id: 1,
    username: 'ada',
    email: 'ada@example.com',
    date_joined: ACCOUNT.date_joined,
  });
  vi.mocked(api.logout).mockResolvedValue(undefined);
  vi.mocked(api.legal.info).mockResolvedValue({
    pages: { imprint: true, privacy: true, terms: false },
    terms_version: null,
    privacy_version: 'def',
    accept_terms_required: false,
    retention: { unused_uploads_days: 90, inactive_accounts_days: 0 },
  });
  vi.mocked(api.account.get).mockResolvedValue(ACCOUNT);
  vi.spyOn(notifications, 'show').mockImplementation(() => '');
});

afterEach(() => {
  setTokens(null);
});

test('storedSummary', () => {
  expect(storedSummary(ACCOUNT.stored)).toEqual([
    '3 networks',
    'access to 1 network of other people',
    '1 imported table',
    '1 data preparation',
    '2 training datasets',
    '5 training runs',
  ]);
});

test('shows the account and what is stored', async () => {
  renderPage();
  expect(await screen.findByRole('heading', { name: 'Account' })).toBeInTheDocument();
  expect(screen.getByLabelText('Username')).toHaveValue('ada');
  expect(screen.getByLabelText(/E-mail address/)).toHaveValue('ada@example.com');
  const data = screen.getByRole('region', { name: 'Your data' });
  expect(within(data).getByText('3 networks')).toBeInTheDocument();
  expect(within(data).getByText('3.0 MB of uploaded files')).toBeInTheDocument();
  expect(
    await within(data).findByText(/nobody has used for 90 days are deleted automatically/)
  ).toBeInTheDocument();
  expect(screen.getByRole('navigation', { name: 'Legal' })).toBeInTheDocument();
});

test('changes the e-mail address', async () => {
  vi.mocked(api.account.changeEmail)
    .mockRejectedValueOnce(
      new ApiError(400, { email: ['An account with this e-mail address exists already.'] }, 'x')
    )
    .mockResolvedValueOnce({ ...ACCOUNT, email: 'new@example.com' });
  renderPage();
  const input = await screen.findByLabelText(/E-mail address/);
  const save = screen.getByRole('button', { name: 'Save e-mail address' });
  expect(save).toBeDisabled();
  await userEvent.clear(input);
  await userEvent.type(input, 'taken@example.com');
  await userEvent.click(save);
  expect(
    await screen.findByText('An account with this e-mail address exists already.')
  ).toBeInTheDocument();
  await userEvent.clear(input);
  await userEvent.type(input, 'new@example.com');
  await userEvent.click(save);
  await waitFor(() => expect(api.account.changeEmail).toHaveBeenLastCalledWith('new@example.com'));
  await waitFor(() => expect(save).toBeDisabled());
  expect(input).toHaveValue('new@example.com');
});

test('asks accounts without an e-mail address for one', async () => {
  vi.mocked(api.account.get).mockResolvedValue({ ...ACCOUNT, email: '' });
  renderPage();
  expect(await screen.findByText(/Your account has no e-mail address yet/)).toBeInTheDocument();
});

test('changes the password and keeps the new login', async () => {
  vi.mocked(api.account.changePassword)
    .mockRejectedValueOnce(
      new ApiError(400, { current_password: ['The current password is wrong.'] }, 'x')
    )
    .mockResolvedValueOnce({ access: 'new-access', refresh: 'new-refresh' });
  renderPage();
  const current = await screen.findByLabelText(/Current password/);
  await userEvent.type(current, 'wrong');
  await userEvent.type(screen.getByLabelText(/New password/), 'a-new-long-password');
  await userEvent.click(screen.getByRole('button', { name: 'Change password' }));
  expect(await screen.findByText('The current password is wrong.')).toBeInTheDocument();
  expect(getTokens()?.refresh).toBe('old-refresh');

  await userEvent.clear(current);
  await userEvent.type(current, 'right');
  await userEvent.click(screen.getByRole('button', { name: 'Change password' }));
  await waitFor(() => expect(getTokens()?.refresh).toBe('new-refresh'));
  expect(api.account.changePassword).toHaveBeenLastCalledWith('right', 'a-new-long-password');
  expect(current).toHaveValue('');
});

test('downloads the data', async () => {
  const { downloadBlob } = await import('@/builder/download');
  const blob = new Blob(['zip']);
  vi.mocked(api.account.exportData).mockResolvedValue({ blob, filename: 'brick-dnn-ada.zip' });
  renderPage();
  await userEvent.click(await screen.findByRole('button', { name: 'Download my data' }));
  await waitFor(() => expect(downloadBlob).toHaveBeenCalledWith('brick-dnn-ada.zip', blob));
});

test('deletes the account after confirming with password and username', async () => {
  vi.mocked(api.account.remove)
    .mockRejectedValueOnce(new ApiError(400, { password: ['The password is wrong.'] }, 'x'))
    .mockResolvedValueOnce(undefined);
  renderPage();
  await userEvent.click(await screen.findByRole('button', { name: 'Delete account…' }));
  const dialog = await screen.findByRole('dialog', { name: 'Delete your account?' });
  const confirm = within(dialog).getByRole('button', { name: 'Delete my account' });
  expect(confirm).toBeDisabled();

  await userEvent.type(within(dialog).getByLabelText(/Your password/), 'guess');
  const username = within(dialog).getByLabelText(/Type your username/);
  await userEvent.type(username, 'Ada');
  expect(confirm).toBeDisabled(); // the username must match exactly
  await userEvent.clear(username);
  await userEvent.type(username, 'ada');
  await userEvent.click(confirm);
  expect(await within(dialog).findByText('The password is wrong.')).toBeInTheDocument();
  expect(getTokens()).not.toBeNull();

  await userEvent.click(confirm);
  // To the home page, not to the login page RequireAuth sends logged-out visitors to.
  expect(await screen.findByText('Home page')).toBeInTheDocument();
  expect(screen.queryByText('Login page')).toBeNull();
  expect(api.account.remove).toHaveBeenLastCalledWith('guess', 'ada');
  expect(getTokens()).toBeNull();
  // The server ended the logins already: no logout request.
  expect(api.logout).not.toHaveBeenCalled();
});
