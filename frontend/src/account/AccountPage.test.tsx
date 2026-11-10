import { render, screen, userEvent, waitFor, within } from '@test-utils';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { afterEach, beforeEach, vi } from 'vitest';
import { notifications } from '@mantine/notifications';
import { api, ApiError, getTokens, setTokens, type Account, type Plan, type Usage } from '@/api';
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
        usage: vi.fn(),
        setBudget: vi.fn(),
        setOverage: vi.fn(),
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
    agent_actions: 0,
    upload_bytes: 3 * 1024 * 1024,
  },
};

const USAGE: Usage = {
  period: { start: '2026-11-01', resets_at: '2026-12-01T00:00:00Z' },
  compute: {
    unit: 'cpu_minutes',
    used: 75,
    limit: 300,
    remaining: 225,
    available: 225,
    allowance: 300,
    own_budget: null,
    run_limit: 30,
    reserved_by_active_runs: 0,
    by_kind: { training: 74.5, prediction: 0.5 },
    by_source: [
      {
        source: 'token',
        id: 3,
        name: 'agent',
        deleted: false,
        used_cpu_minutes: 50,
        budget_cpu_minutes: 60,
        cost: null,
      },
      {
        source: 'web',
        id: null,
        name: 'Web app',
        deleted: false,
        used_cpu_minutes: 25,
        budget_cpu_minutes: null,
        cost: null,
      },
    ],
  },
  storage: {
    used_bytes: 512 * 1024 * 1024,
    limit_bytes: 1024 * 1024 * 1024,
    remaining_bytes: 512 * 1024 * 1024,
    by_kind: { data_sources: 0, datasets: 500 * 1024 * 1024, training_runs: 12 * 1024 * 1024 },
  },
  prices: {
    currency: null,
    per_cpu_hour: null,
    per_gb_month: null,
    compute_cost: null,
    compute_limit_cost: null,
    storage_cost_per_month: null,
  },
  caller: null,
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
  vi.mocked(api.account.usage).mockResolvedValue(USAGE);
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
  expect(storedSummary({ ...ACCOUNT.stored, agent_actions: 1234 })).toContain(
    '1,234 entries in the agent activity log'
  );
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

test('shows the usage this month and sets the own budget', async () => {
  vi.mocked(api.account.setBudget).mockResolvedValue({
    ...USAGE,
    compute: { ...USAGE.compute, own_budget: 100, limit: 100, remaining: 25 },
  });
  renderPage();
  const section = await screen.findByRole('region', { name: 'Usage & budget' });
  expect(await within(section).findByText('75 of 300 CPU min')).toBeInTheDocument();
  expect(
    within(section).getByRole('progressbar', { name: 'Compute this month: 25% used' })
  ).toBeInTheDocument();
  expect(within(section).getByText(/Resets on 1 December 2026 \(UTC\)/)).toBeInTheDocument();
  expect(within(section).getByText(/“agent” 50 CPU min/)).toBeInTheDocument();
  expect(within(section).getByText('512 MB of 1.0 GB')).toBeInTheDocument();
  expect(within(section).getByText(/30 CPU min per training/)).toBeInTheDocument();

  await userEvent.click(
    within(section).getByRole('button', { name: 'Set the monthly budget of your account' })
  );
  const input = await screen.findByLabelText('Monthly budget');
  await userEvent.type(input, '100');
  await userEvent.click(screen.getByRole('button', { name: 'Save' }));
  await waitFor(() => expect(api.account.setBudget).toHaveBeenCalledWith(100));
  expect(await within(section).findByText('75 of 100 CPU min')).toBeInTheDocument();
  expect(within(section).getByText(/Your monthly budget: 100 CPU min/)).toBeInTheDocument();
});

test('usage without limits and with prices', async () => {
  vi.mocked(api.account.usage).mockResolvedValue({
    ...USAGE,
    compute: { ...USAGE.compute, limit: null, remaining: null, allowance: null, run_limit: null },
    storage: { ...USAGE.storage, limit_bytes: null, remaining_bytes: null },
    prices: { ...USAGE.prices, currency: 'EUR', per_cpu_hour: 0.6, compute_cost: 0.75 },
  });
  renderPage();
  const section = await screen.findByRole('region', { name: 'Usage & budget' });
  expect(await within(section).findByText('75 CPU min · no limit (€0.75)')).toBeInTheDocument();
  expect(within(section).queryByRole('progressbar')).not.toBeInTheDocument();
  expect(within(section).getByText(/No monthly limit applies\./)).toBeInTheDocument();
});

const FREE: Plan = {
  key: 'free',
  name: 'Free',
  price: 0,
  currency: 'EUR',
  cpu_hours_per_month: 0.5,
  storage_gb: 1,
  cpu_minutes_per_run: 15,
  concurrent_trainings: 1,
  priority: 0,
  overage_price_per_cpu_hour: null,
};
const STARTER: Plan = {
  ...FREE,
  key: 'starter',
  name: 'Starter',
  price: 5,
  cpu_hours_per_month: 10,
  storage_gb: 5,
  cpu_minutes_per_run: 60,
  priority: 1,
  overage_price_per_cpu_hour: 0.4,
};
const OVERAGE = {
  available: false,
  enabled: false,
  price_per_cpu_hour: null,
  currency: 'EUR',
  cap_cpu_hours: null,
  used_cpu_minutes: 0,
  cost: null,
};

test('shows the plan and the other plans, without buying them', async () => {
  vi.mocked(api.account.usage).mockResolvedValue({
    ...USAGE,
    plan: FREE,
    unlimited: false,
    plans: [FREE, STARTER],
    overage: OVERAGE,
  });
  renderPage();
  const section = await screen.findByRole('region', { name: 'Usage & budget' });
  expect(await within(section).findByText('Free plan')).toBeInTheDocument();
  expect(
    within(section).getByText(
      '0.5 CPU h per month · 1 GB · 15 CPU min per training · 1 training at a time'
    )
  ).toBeInTheDocument();
  expect(
    within(section).getByText(/Other plans: Starter \(€5\.00 per month\): 10 CPU h, 5 GB\./)
  ).toBeInTheDocument();
  expect(within(section).queryByRole('button', { name: /upgrade|buy/i })).not.toBeInTheDocument();
  expect(within(section).queryByText(/beyond the monthly allowance/)).not.toBeInTheDocument();
});

test('unlimited accounts', async () => {
  vi.mocked(api.account.usage).mockResolvedValue({
    ...USAGE,
    compute: { ...USAGE.compute, limit: null, remaining: null, allowance: null, run_limit: null },
    plan: FREE,
    unlimited: true,
    plans: [FREE, STARTER],
    overage: OVERAGE,
  });
  renderPage();
  const section = await screen.findByRole('region', { name: 'Usage & budget' });
  expect(await within(section).findByText('Unlimited')).toBeInTheDocument();
  expect(within(section).getByText(/No limits apply to your account/)).toBeInTheDocument();
  expect(within(section).queryByText(/Other plans/)).not.toBeInTheDocument();
  expect(within(section).getByText(/75 CPU min · no limit/)).toBeInTheDocument();
});

test('switches compute beyond the allowance on, with a cap', async () => {
  const usage: Usage = {
    ...USAGE,
    plan: STARTER,
    unlimited: false,
    plans: [FREE, STARTER],
    overage: { ...OVERAGE, available: true, price_per_cpu_hour: 0.4 },
  };
  vi.mocked(api.account.usage).mockResolvedValue(usage);
  vi.mocked(api.account.setOverage).mockResolvedValue({
    ...usage,
    overage: { ...usage.overage!, enabled: true, cap_cpu_hours: 2 },
  });
  renderPage();
  expect(await screen.findByText('Allow compute beyond the monthly allowance')).toBeInTheDocument();
  const toggle = screen.getByRole('switch');
  expect(screen.getByText(/Billed at €0\.40 per CPU hour/)).toBeInTheDocument();
  await userEvent.click(toggle);
  await userEvent.type(screen.getByLabelText('Most beyond the allowance'), '2');
  await userEvent.click(screen.getByRole('button', { name: 'Save' }));
  await waitFor(() => expect(api.account.setOverage).toHaveBeenCalledWith(true, 2));
  await waitFor(() => expect(screen.getByRole('switch')).toBeChecked());
});
