import { render, screen, userEvent, waitFor } from '@test-utils';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, vi } from 'vitest';
import { api, setTokens } from '@/api';
import { AuthProvider } from '@/auth/AuthContext';
import { resetBillingInfo } from '@/billing/useBillingInfo';
import { AppHeader } from '@/components/AppHeader';
import { resetLegalInfo } from '@/legal/useLegalInfo';
import { setPendingCount } from './pending';

vi.mock('@/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api')>();
  return {
    ...actual,
    api: {
      me: vi.fn(),
      logout: vi.fn(),
      legal: { info: vi.fn() },
      billing: { info: vi.fn() },
      agentActivity: { summary: vi.fn() },
      approvals: { list: vi.fn() },
    },
  };
});

function renderHeader() {
  return render(
    <AuthProvider>
      <MemoryRouter>
        <AppHeader />
      </MemoryRouter>
    </AuthProvider>
  );
}

describe('the approvals badge in the header', () => {
  beforeEach(() => {
    resetLegalInfo();
    resetBillingInfo();
    vi.mocked(api.legal.info).mockResolvedValue({
      pages: { imprint: false, privacy: false, terms: false },
    } as never);
    vi.mocked(api.billing.info).mockResolvedValue({ enabled: false } as never);
    vi.mocked(api.agentActivity.summary).mockResolvedValue({
      agents: [],
      tools: [],
      total: 0,
      latest_id: 0,
      latest_at: null,
      retention_days: 90,
    });
    vi.mocked(api.me).mockResolvedValue({
      id: 1,
      username: 'ada',
      email: 'ada@example.com',
      date_joined: '2026-01-01T00:00:00Z',
    } as never);
  });

  afterEach(() => {
    setTokens(null);
    setPendingCount(0);
    vi.mocked(api.approvals.list).mockReset();
  });

  it('shows how many requests wait, also in the account menu', async () => {
    setTokens({ access: 'access', refresh: 'refresh' });
    vi.mocked(api.approvals.list).mockResolvedValue({ results: [], pending_count: 2 });
    renderHeader();
    const account = await screen.findByRole('button', {
      name: 'ada, 2 waiting for your approval',
    });
    expect(api.approvals.list).toHaveBeenCalledWith({ status: 'pending', limit: 0 });
    await userEvent.click(account);
    expect(await screen.findByText('2 waiting')).toBeInTheDocument();
  });

  it('follows the count the activity page reports', async () => {
    setTokens({ access: 'access', refresh: 'refresh' });
    vi.mocked(api.approvals.list).mockResolvedValue({ results: [], pending_count: 1 });
    renderHeader();
    await screen.findByRole('button', { name: 'ada, 1 waiting for your approval' });
    setPendingCount(0);
    await waitFor(() => expect(screen.getByRole('button', { name: 'ada' })).toBeInTheDocument());
  });

  it('asks nothing when logged out', async () => {
    renderHeader();
    expect(await screen.findByRole('link', { name: 'Log in' })).toBeInTheDocument();
    expect(api.approvals.list).not.toHaveBeenCalled();
  });
});
