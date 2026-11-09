/** The account page and the header with and without paid plans on the server. */
import { render, screen, within } from '@test-utils';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { beforeEach, expect, test, vi } from 'vitest';
import { AccountPage } from '@/account/AccountPage';
import { api, setTokens, type Account } from '@/api';
import { AuthProvider } from '@/auth/AuthContext';
import { AppHeader } from '@/components/AppHeader';
import { resetLegalInfo } from '@/legal/useLegalInfo';
import { BILLING, subscribed } from './testData';
import { resetBillingInfo } from './useBillingInfo';

vi.mock('@/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api')>();
  return {
    ...actual,
    api: {
      me: vi.fn(),
      logout: vi.fn(),
      legal: { info: vi.fn(), page: vi.fn() },
      account: { get: vi.fn(), usage: vi.fn() },
      billing: { info: vi.fn(), state: vi.fn() },
    },
  };
});

const ACCOUNT: Account = {
  id: 1,
  username: 'ada',
  email: 'ada@example.com',
  date_joined: '2026-01-02T10:00:00Z',
  last_login: '2026-11-08T10:00:00Z',
  last_active_at: '2026-11-08T10:00:00Z',
  terms_accepted_at: null,
  terms_version: null,
  privacy_version: null,
  stored: {
    projects: 0,
    memberships: 0,
    comments: 0,
    data_sources: 0,
    data_preps: 0,
    datasets: 0,
    training_runs: 0,
    api_tokens: 0,
    connected_apps: 0,
    upload_bytes: 0,
  },
};

function renderAccount() {
  return render(
    <AuthProvider>
      <MemoryRouter initialEntries={['/account']}>
        <AppHeader />
        <Routes>
          <Route path="/account" element={<AccountPage />} />
        </Routes>
      </MemoryRouter>
    </AuthProvider>
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  resetLegalInfo();
  resetBillingInfo();
  setTokens({ access: 'access', refresh: 'refresh' });
  vi.mocked(api.me).mockResolvedValue({
    id: 1,
    username: 'ada',
    email: 'ada@example.com',
    date_joined: ACCOUNT.date_joined,
  });
  vi.mocked(api.legal.info).mockRejectedValue(new Error('no legal pages'));
  vi.mocked(api.account.get).mockResolvedValue(ACCOUNT);
  // The usage section is not what these tests are about.
  vi.mocked(api.account.usage).mockReturnValue(new Promise(() => {}));
});

test('without paid plans the account page and the header are as before', async () => {
  vi.mocked(api.billing.info).mockResolvedValue({ enabled: false });
  renderAccount();
  expect(await screen.findByRole('region', { name: 'Profile' })).toBeInTheDocument();
  expect(screen.queryByRole('region', { name: 'Plan & billing' })).not.toBeInTheDocument();
  expect(screen.queryByRole('link', { name: 'Pricing' })).not.toBeInTheDocument();
  expect(api.billing.state).not.toHaveBeenCalled();
  expect(document.querySelectorAll('script[src*="paddle"]')).toHaveLength(0);
});

test('with paid plans the account page has a plan section and the header a pricing link', async () => {
  vi.mocked(api.billing.info).mockResolvedValue(BILLING);
  vi.mocked(api.billing.state).mockResolvedValue(subscribed());
  renderAccount();
  const section = within(await screen.findByRole('region', { name: 'Plan & billing' }));
  expect(await section.findByText('Plus plan')).toBeInTheDocument();
  expect(section.getByText(/Renews on 9 December 2026/)).toBeInTheDocument();
  expect(screen.getByRole('link', { name: 'Pricing' })).toHaveAttribute('href', '/pricing');
  // The page shows the plan without loading anything from Paddle.
  expect(document.querySelectorAll('script[src*="paddle"]')).toHaveLength(0);
});
