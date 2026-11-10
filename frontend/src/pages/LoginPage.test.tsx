import { render, screen, userEvent, waitFor } from '@test-utils';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { afterEach, beforeEach, vi } from 'vitest';
import { api, ApiError, getTokens, setTokens } from '@/api';
import { AuthProvider } from '@/auth/AuthContext';
import { LoginPage } from './LoginPage';

vi.mock('@/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api')>();
  return {
    ...actual,
    api: {
      login: vi.fn(),
      loginWithCode: vi.fn(),
      me: vi.fn(),
      logout: vi.fn(),
      legal: { info: vi.fn(), page: vi.fn() },
    },
  };
});

const USER = { id: 1, username: 'ada', email: 'ada@example.com', date_joined: '2026-01-02' };

function renderLogin() {
  return render(
    <AuthProvider>
      <MemoryRouter initialEntries={[{ pathname: '/login', state: { from: '/projects' } }]}>
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route path="/projects" element={<p>Projects page</p>} />
          <Route path="/forgot-password" element={<p>Forgot page</p>} />
        </Routes>
      </MemoryRouter>
    </AuthProvider>
  );
}

async function enterPassword() {
  await userEvent.type(screen.getByLabelText(/Username/), 'ada');
  await userEvent.type(screen.getByLabelText(/^Password/), 'secret-password');
  await userEvent.click(screen.getByRole('button', { name: 'Log in' }));
}

beforeEach(() => {
  vi.clearAllMocks();
  setTokens(null);
  vi.mocked(api.legal.info).mockResolvedValue({
    pages: { imprint: false, privacy: false, terms: false },
    terms_version: null,
    privacy_version: null,
    accept_terms_required: false,
    retention: { unused_uploads_days: 0, inactive_accounts_days: 0 },
  });
  vi.mocked(api.me).mockResolvedValue(USER);
});

afterEach(() => setTokens(null));

test('logs in with the password alone', async () => {
  vi.mocked(api.login).mockResolvedValue({ access: 'a', refresh: 'r' });
  renderLogin();
  await enterPassword();
  expect(await screen.findByText('Projects page')).toBeInTheDocument();
  expect(getTokens()).toEqual({ access: 'a', refresh: 'r' });
  expect(api.loginWithCode).not.toHaveBeenCalled();
});

test('asks for the code when two-factor authentication is on', async () => {
  vi.mocked(api.login).mockResolvedValue({
    two_factor_required: true,
    challenge: 'signed-challenge',
    expires_in: 300,
  });
  vi.mocked(api.loginWithCode)
    .mockRejectedValueOnce(
      new ApiError(400, { code: ['The code is wrong or was used already.'] }, 'Code: wrong')
    )
    .mockResolvedValueOnce({ access: 'a2', refresh: 'r2' });
  renderLogin();
  await enterPassword();
  const code = await screen.findByLabelText(/^Code/);
  expect(getTokens()).toBeNull(); // no tokens before the code
  expect(screen.getByText('Logging in as ada')).toBeInTheDocument();
  expect(code).toHaveAttribute('autocomplete', 'one-time-code');

  await userEvent.type(code, '000000');
  await userEvent.click(screen.getByRole('button', { name: 'Verify' }));
  expect(await screen.findByText('Code: wrong')).toBeInTheDocument();

  await userEvent.type(screen.getByLabelText(/^Code/), '123456');
  await userEvent.click(screen.getByRole('button', { name: 'Verify' }));
  expect(await screen.findByText('Projects page')).toBeInTheDocument();
  expect(api.loginWithCode).toHaveBeenLastCalledWith('signed-challenge', '123456');
  expect(getTokens()).toEqual({ access: 'a2', refresh: 'r2' });
});

test('takes a recovery code instead', async () => {
  vi.mocked(api.login).mockResolvedValue({
    two_factor_required: true,
    challenge: 'c',
    expires_in: 300,
  });
  vi.mocked(api.loginWithCode).mockResolvedValue({ access: 'a', refresh: 'r' });
  renderLogin();
  await enterPassword();
  await userEvent.click(await screen.findByRole('button', { name: 'Use a recovery code' }));
  await userEvent.type(screen.getByLabelText(/Recovery code/), 'abcd-efgh-ijkl-mnop');
  await userEvent.click(screen.getByRole('button', { name: 'Verify' }));
  await waitFor(() => expect(api.loginWithCode).toHaveBeenCalledWith('c', 'abcd-efgh-ijkl-mnop'));
});

test('an expired challenge goes back to the password', async () => {
  vi.mocked(api.login).mockResolvedValue({
    two_factor_required: true,
    challenge: 'c',
    expires_in: 300,
  });
  vi.mocked(api.loginWithCode).mockRejectedValue(
    new ApiError(400, { detail: 'The login has expired.', code: 'challenge_expired' }, 'x')
  );
  renderLogin();
  await enterPassword();
  await userEvent.type(await screen.findByLabelText(/^Code/), '123456');
  await userEvent.click(screen.getByRole('button', { name: 'Verify' }));
  expect(await screen.findByText(/Enter your password again/)).toBeInTheDocument();
  expect(screen.getByLabelText(/^Password/)).toHaveValue('');
});

test('links to the forgotten password page', async () => {
  renderLogin();
  await userEvent.click(screen.getByRole('link', { name: 'Forgot password?' }));
  expect(await screen.findByText('Forgot page')).toBeInTheDocument();
});
