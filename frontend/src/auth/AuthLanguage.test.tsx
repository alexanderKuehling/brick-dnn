import { render, screen, userEvent, waitFor } from '@test-utils';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, vi } from 'vitest';
import { api, setTokens, type User } from '@/api';
import { AuthProvider } from '@/auth/AuthContext';
import { currentLanguage, LANGUAGE_STORAGE_KEY, setLanguage } from '@/i18n';
import { LanguageSwitch } from '@/i18n/LanguageSwitch';
import { LoginPage } from '@/pages/LoginPage';

vi.mock('@/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api')>();
  return {
    ...actual,
    api: {
      login: vi.fn(),
      register: vi.fn(),
      logout: vi.fn(),
      me: vi.fn(),
      account: { setLanguage: vi.fn() },
    },
  };
});

const USER: User = { id: 1, username: 'alex', email: '', date_joined: '2026-01-01T00:00:00Z' };

function renderLogin() {
  return render(
    <AuthProvider>
      <MemoryRouter initialEntries={['/login']}>
        <LanguageSwitch />
        <LoginPage />
      </MemoryRouter>
    </AuthProvider>
  );
}

async function logIn() {
  await userEvent.type(screen.getByLabelText(/username|benutzername/i), 'alex');
  await userEvent.type(screen.getByLabelText(/^password|^passwort/i), 'secret-password');
  await userEvent.click(screen.getByRole('button', { name: /log in|anmelden/i }));
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(api.login).mockResolvedValue({ access: 'a', refresh: 'r' });
  vi.mocked(api.logout).mockResolvedValue(undefined);
  vi.mocked(api.account.setLanguage).mockResolvedValue({ language: 'de' });
});

afterEach(() => {
  setTokens(null);
});

describe('the language of the account', () => {
  it('follows the account after logging in', async () => {
    vi.mocked(api.me).mockResolvedValue({ ...USER, language: 'de' });
    renderLogin();
    expect(screen.getByText('Welcome back')).toBeInTheDocument();
    await logIn();
    await waitFor(() => expect(currentLanguage()).toBe('de'));
    expect(localStorage.getItem(LANGUAGE_STORAGE_KEY)).toBe('de');
    // Taking over the account's language saves nothing back.
    expect(api.account.setLanguage).not.toHaveBeenCalled();
  });

  it('gives an account without a language the current one', async () => {
    vi.mocked(api.me).mockResolvedValue({ ...USER, language: '' });
    await setLanguage('de', { save: false });
    renderLogin();
    await logIn();
    await waitFor(() => expect(api.account.setLanguage).toHaveBeenCalledWith('de'));
    expect(currentLanguage()).toBe('de');
  });

  it('saves a language chosen while logged in on the account', async () => {
    vi.mocked(api.me).mockResolvedValue({ ...USER, language: 'en' });
    renderLogin();
    await logIn();
    await waitFor(() => expect(api.me).toHaveBeenCalled());
    await userEvent.click(screen.getAllByText('DE')[0]);
    await waitFor(() => expect(api.account.setLanguage).toHaveBeenCalledWith('de'));
    expect(localStorage.getItem(LANGUAGE_STORAGE_KEY)).toBe('de');
  });

  it('only keeps the choice in the browser while logged out', async () => {
    renderLogin();
    await userEvent.click(screen.getAllByText('DE')[0]);
    expect(await screen.findByText('Willkommen zurück')).toBeInTheDocument();
    expect(localStorage.getItem(LANGUAGE_STORAGE_KEY)).toBe('de');
    expect(api.account.setLanguage).not.toHaveBeenCalled();
  });
});
