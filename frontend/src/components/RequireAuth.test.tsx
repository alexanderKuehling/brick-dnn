import { render, screen, userEvent } from '@test-utils';
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom';
import { afterEach, beforeEach, vi } from 'vitest';
import { api, setTokens, type User } from '@/api';
import { AuthProvider } from '@/auth/AuthContext';
import { LoginPage } from '@/pages/LoginPage';
import { RegisterPage } from '@/pages/RegisterPage';
import { RequireAuth } from './RequireAuth';

vi.mock('@/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api')>();
  return {
    ...actual,
    api: { login: vi.fn(), register: vi.fn(), logout: vi.fn(), me: vi.fn() },
  };
});

const USER: User = { id: 1, username: 'alex', email: '', date_joined: '2026-01-01T00:00:00Z' };
// An OAuth sign-in request: the page needs all of its query string.
const PAGE = '/oauth/authorize?response_type=code&client_id=c1&state=s%2B1&scope=read+write';

function Protected() {
  const location = useLocation();
  return <p data-testid="protected">{`${location.pathname}${location.search}`}</p>;
}

function renderAt(path: string) {
  return render(
    <AuthProvider>
      <MemoryRouter initialEntries={[path]}>
        <Routes>
          <Route
            path="/oauth/authorize"
            element={
              <RequireAuth>
                <Protected />
              </RequireAuth>
            }
          />
          <Route path="/login" element={<LoginPage />} />
          <Route path="/register" element={<RegisterPage />} />
        </Routes>
      </MemoryRouter>
    </AuthProvider>
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(api.login).mockResolvedValue({ access: 'a', refresh: 'r' });
  vi.mocked(api.me).mockResolvedValue(USER);
  vi.mocked(api.register).mockResolvedValue({ user: USER, access: 'a', refresh: 'r' });
  vi.mocked(api.logout).mockResolvedValue(undefined);
});

afterEach(() => {
  setTokens(null);
});

describe('RequireAuth', () => {
  it('returns to the page with its query string after logging in', async () => {
    renderAt(PAGE);
    expect(await screen.findByRole('heading', { name: 'Welcome back' })).toBeInTheDocument();
    await userEvent.type(screen.getByLabelText(/Username/), 'alex');
    await userEvent.type(screen.getByLabelText(/Password/), 'secret-password');
    await userEvent.click(screen.getByRole('button', { name: 'Log in' }));
    expect(await screen.findByTestId('protected')).toHaveTextContent(PAGE);
  });

  it('returns to the page after creating an account instead', async () => {
    renderAt(PAGE);
    await userEvent.click(await screen.findByRole('link', { name: 'Create one' }));
    expect(await screen.findByRole('heading', { name: 'Create an account' })).toBeInTheDocument();
    await userEvent.type(screen.getByLabelText(/Username/), 'alex');
    await userEvent.type(screen.getByLabelText(/^Password/), 'secret-password');
    await userEvent.click(screen.getByRole('button', { name: 'Create account' }));
    expect(api.register).toHaveBeenCalledWith('alex', '', 'secret-password');
    expect(await screen.findByTestId('protected')).toHaveTextContent(PAGE);
  });

  it('shows the page right away when logged in', async () => {
    setTokens({ access: 'a', refresh: 'r' });
    renderAt(PAGE);
    expect(await screen.findByTestId('protected')).toHaveTextContent(PAGE);
    expect(api.me).toHaveBeenCalled();
  });
});
