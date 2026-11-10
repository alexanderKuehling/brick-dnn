import { render, screen } from '@test-utils';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { afterEach, beforeEach, vi } from 'vitest';
import { api, ApiError, setTokens } from '@/api';
import { AuthProvider } from '@/auth/AuthContext';
import { RequireAuth } from '@/components/RequireAuth';
import { VerifyEmailPage } from './VerifyEmailPage';

vi.mock('@/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api')>();
  return {
    ...actual,
    api: {
      me: vi.fn(),
      logout: vi.fn(),
      legal: { info: vi.fn(), page: vi.fn() },
      account: { verifyEmail: vi.fn() },
    },
  };
});

function renderPage(path = '/verify-email#token=signed') {
  return render(
    <AuthProvider>
      <MemoryRouter initialEntries={[path]}>
        <Routes>
          <Route
            path="/verify-email"
            element={
              <RequireAuth>
                <VerifyEmailPage />
              </RequireAuth>
            }
          />
          <Route path="/login" element={<p>Login page</p>} />
        </Routes>
      </MemoryRouter>
    </AuthProvider>
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(api.me).mockResolvedValue({
    id: 1,
    username: 'ada',
    email: 'ada@example.com',
    date_joined: '2026-01-02',
  });
  vi.mocked(api.logout).mockResolvedValue(undefined);
  vi.mocked(api.legal.info).mockResolvedValue({
    pages: { imprint: false, privacy: false, terms: false },
    terms_version: null,
    privacy_version: null,
    accept_terms_required: false,
    retention: { unused_uploads_days: 0, inactive_accounts_days: 0 },
  });
});

afterEach(() => setTokens(null));

test('asks to log in first', async () => {
  setTokens(null);
  renderPage();
  expect(await screen.findByText('Login page')).toBeInTheDocument();
  expect(api.account.verifyEmail).not.toHaveBeenCalled();
});

test('confirms the address of the logged-in account', async () => {
  setTokens({ access: 'a', refresh: 'r' });
  vi.mocked(api.account.verifyEmail).mockResolvedValue({ email: 'ada@example.com' });
  renderPage();
  expect(await screen.findByText(/ada@example.com is confirmed/)).toBeInTheDocument();
  expect(api.account.verifyEmail).toHaveBeenCalledTimes(1);
  expect(api.account.verifyEmail).toHaveBeenCalledWith('signed');
});

test('a link for another account offers to log out', async () => {
  setTokens({ access: 'a', refresh: 'r' });
  vi.mocked(api.account.verifyEmail).mockRejectedValue(
    new ApiError(
      400,
      { detail: 'This link is for another account.', code: 'other_account' },
      'This link is for another account.'
    )
  );
  renderPage();
  expect(await screen.findByText('This link is for another account.')).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Log out of ada' })).toBeInTheDocument();
});

test('an expired link points to the account page', async () => {
  setTokens({ access: 'a', refresh: 'r' });
  vi.mocked(api.account.verifyEmail).mockRejectedValue(
    new ApiError(400, { detail: 'Expired.', code: 'invalid_link' }, 'Expired.')
  );
  renderPage();
  expect(await screen.findByText('Expired.')).toBeInTheDocument();
  expect(screen.getByRole('link', { name: 'Send a new link from your account' })).toHaveAttribute(
    'href',
    '/account'
  );
});
