/**
 * The header's account menu shows the plan and this month's compute; the link to the
 * project's donation page is in the footer row, not next to "Pricing".
 */
import { render, screen, userEvent, waitFor } from '@test-utils';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { beforeEach, expect, test, vi } from 'vitest';
import { api, setTokens, type Usage } from '@/api';
import { AuthProvider } from '@/auth/AuthContext';
import { BILLING, FREE } from '@/billing/testData';
import { resetBillingInfo } from '@/billing/useBillingInfo';
import { LegalLinks } from '@/legal/LegalLinks';
import { resetLegalInfo } from '@/legal/useLegalInfo';
import { SUPPORT_URL } from '@/links';
import { AppHeader } from './AppHeader';

vi.mock('@/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api')>();
  return {
    ...actual,
    api: {
      me: vi.fn(),
      logout: vi.fn(),
      legal: { info: vi.fn(), page: vi.fn() },
      billing: { info: vi.fn(), state: vi.fn() },
      account: { usage: vi.fn() },
      agentActivity: { summary: vi.fn() },
    },
  };
});
vi.mock('@/approvals/pending', () => ({ usePendingApprovals: () => 0 }));

const USAGE = {
  period: { start: '2026-11-01', resets_at: '2026-12-01T00:00:00Z' },
  plan: FREE,
  unlimited: false,
  compute: { used: 12, limit: 30, allowance: 30, own_budget: null },
  storage: { used_bytes: 0, limit_bytes: null },
} as unknown as Usage;

function renderHeader() {
  return render(
    <AuthProvider>
      <MemoryRouter initialEntries={['/app']}>
        <AppHeader />
        <Routes>
          <Route path="/pricing" element={<p>Pricing page</p>} />
          <Route path="/account" element={<p>Account page</p>} />
          <Route path="*" element={<LegalLinks />} />
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
    date_joined: '2026-01-02T10:00:00Z',
  });
  vi.mocked(api.legal.info).mockResolvedValue({
    pages: { imprint: true, privacy: true, terms: true, refund: false, accessibility: false },
    terms_version: 'a',
    privacy_version: 'b',
    accept_terms_required: true,
    retention: { unused_uploads_days: 0, inactive_accounts_days: 0 },
  });
  vi.mocked(api.billing.info).mockResolvedValue(BILLING);
  vi.mocked(api.agentActivity.summary).mockResolvedValue({ latest_at: null } as never);
  vi.mocked(api.account.usage).mockResolvedValue(USAGE);
});

test('the account menu shows the plan and usage and leads to the plans', async () => {
  renderHeader();
  await userEvent.click(await screen.findByRole('button', { name: /ada/ }));
  const item = await screen.findByRole('menuitem', {
    name: 'Your plan and usage: Free plan · 12 of 30 CPU min',
  });
  expect(item).toHaveTextContent('Free plan · 12 of 30 CPU min');
  await userEvent.click(item);
  expect(await screen.findByText('Pricing page')).toBeInTheDocument();
});

test('the usage is only asked for when the menu is opened', async () => {
  renderHeader();
  await screen.findByRole('button', { name: /ada/ });
  expect(api.account.usage).not.toHaveBeenCalled();
});

test('on a server that sells no plans the line leads to the account page', async () => {
  vi.mocked(api.billing.info).mockResolvedValue({ enabled: false });
  vi.mocked(api.account.usage).mockResolvedValue({
    ...USAGE,
    plan: { ...FREE, key: 'default', name: 'Default' },
  });
  renderHeader();
  await userEvent.click(await screen.findByRole('button', { name: /ada/ }));
  const item = await screen.findByRole('menuitem', {
    name: 'Your plan and usage: Compute this month: 12 of 30 CPU min',
  });
  await userEvent.click(item);
  expect(await screen.findByText('Account page')).toBeInTheDocument();
});

test('accounts without limits have no such line', async () => {
  vi.mocked(api.account.usage).mockResolvedValue({ ...USAGE, unlimited: true });
  renderHeader();
  await userEvent.click(await screen.findByRole('button', { name: /ada/ }));
  await screen.findByRole('menuitem', { name: 'Account' });
  await waitFor(() => expect(api.account.usage).toHaveBeenCalled());
  expect(screen.queryByTestId('menu-plan-usage')).not.toBeInTheDocument();
});

test('the donation link is in the footer row, not in the header', async () => {
  renderHeader();
  const support = await screen.findByRole('link', { name: 'Support this project' });
  expect(support).toHaveAttribute('href', SUPPORT_URL);
  expect(support.closest('nav')).toHaveAccessibleName('Legal');
  expect(screen.queryByRole('link', { name: 'Buy me a coffee' })).not.toBeInTheDocument();
  expect(screen.getByRole('link', { name: 'Pricing' })).toBeInTheDocument();
});
