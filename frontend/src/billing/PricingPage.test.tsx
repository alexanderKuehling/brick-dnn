import { render, screen, userEvent, waitFor, within } from '@test-utils';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import { Notifications, notifications } from '@mantine/notifications';
import { api, ApiError, setTokens, type BillingState } from '@/api';
import { AuthProvider } from '@/auth/AuthContext';
import { AppHeader } from '@/components/AppHeader';
import { resetLegalInfo } from '@/legal/useLegalInfo';
import { loadPaddle, openCheckout } from './paddle';
import { PricingPage } from './PricingPage';
import { BILLING, NO_SUBSCRIPTION, PRO, subscribed } from './testData';
import { resetBillingInfo } from './useBillingInfo';
import { waitForPlan } from './useSubscribe';

vi.mock('@/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api')>();
  return {
    ...actual,
    api: {
      me: vi.fn(),
      logout: vi.fn(),
      legal: { info: vi.fn(), page: vi.fn() },
      billing: {
        info: vi.fn(),
        state: vi.fn(),
        checkout: vi.fn(),
        previewChange: vi.fn(),
        changePlan: vi.fn(),
        cancel: vi.fn(),
        reactivate: vi.fn(),
        portal: vi.fn(),
        refresh: vi.fn(),
      },
    },
  };
});

// The real one would load Paddle.js; paddle.test.ts covers it.
vi.mock('./paddle', () => ({ openCheckout: vi.fn(), loadPaddle: vi.fn() }));

const CHECKOUT = {
  plan: 'plus',
  price_id: 'pri_plus',
  customer_id: 'ctm_ada',
  custom_data: { user_id: '1' },
};

function renderPage(path = '/pricing') {
  return render(
    <AuthProvider>
      <Notifications />
      <MemoryRouter initialEntries={[path]}>
        <AppHeader />
        <Routes>
          <Route path="/pricing" element={<PricingPage />} />
          <Route path="/login" element={<p>Login page</p>} />
          <Route path="/register" element={<p>Sign-up page</p>} />
          <Route path="/account" element={<p>Account page</p>} />
          <Route path="*" element={<p>Another page</p>} />
        </Routes>
      </MemoryRouter>
    </AuthProvider>
  );
}

function logIn(state: BillingState) {
  setTokens({ access: 'access', refresh: 'refresh' });
  vi.mocked(api.me).mockResolvedValue({
    id: 1,
    username: 'ada',
    email: 'ada@example.com',
    date_joined: '2026-01-02T10:00:00Z',
  });
  vi.mocked(api.billing.state).mockResolvedValue(state);
}

function card(name: string) {
  return within(screen.getByRole('region', { name: `${name} plan` }));
}

function paddleScripts() {
  return document.querySelectorAll('script[src*="paddle"]');
}

beforeEach(() => {
  vi.clearAllMocks();
  resetLegalInfo();
  resetBillingInfo();
  setTokens(null);
  vi.mocked(api.legal.info).mockResolvedValue({
    pages: { imprint: true, privacy: true, terms: true, refund: false },
    terms_version: 'a',
    privacy_version: 'b',
    accept_terms_required: true,
    retention: { unused_uploads_days: 0, inactive_accounts_days: 0 },
  });
  vi.mocked(api.billing.info).mockResolvedValue(BILLING);
  vi.mocked(loadPaddle).mockResolvedValue({} as Awaited<ReturnType<typeof loadPaddle>>);
});

afterEach(() => {
  notifications.clean();
  vi.useRealTimers();
});

test('a server that sells no plans has no pricing page and no link to it', async () => {
  vi.mocked(api.billing.info).mockResolvedValue({ enabled: false });
  renderPage();
  expect(await screen.findByText('Page not found')).toBeInTheDocument();
  expect(screen.queryByRole('link', { name: 'Pricing' })).not.toBeInTheDocument();
  expect(screen.queryByText(/per month/)).not.toBeInTheDocument();
  expect(api.billing.state).not.toHaveBeenCalled();
  expect(paddleScripts()).toHaveLength(0);
});

test('no answer about billing means no pricing page either', async () => {
  vi.mocked(api.billing.info).mockRejectedValue(new ApiError(500, null, 'Server error'));
  renderPage();
  expect(await screen.findByText('Page not found')).toBeInTheDocument();
});

test('shows the plans with prices, what they include and the overage price', async () => {
  renderPage();
  expect(await screen.findByRole('heading', { name: 'Plans and prices' })).toBeInTheDocument();
  expect(screen.getByRole('link', { name: 'Pricing' })).toHaveAttribute('href', '/pricing');

  expect(card('Free').getByText('Free', { selector: 'span' })).toBeInTheDocument();
  expect(card('Free').getByText(/0.5 CPU hours per month/)).toBeInTheDocument();
  expect(card('Free').getByText('1 GB storage for data and trained models')).toBeInTheDocument();
  expect(card('Free').getByText('Up to 15 CPU minutes per training')).toBeInTheDocument();
  expect(card('Free').queryByText(/More compute/)).not.toBeInTheDocument();

  expect(card('Starter').getByText('€5')).toBeInTheDocument();
  expect(card('Starter').getByText(/10 CPU hours per month/)).toBeInTheDocument();
  expect(card('Starter').getByText('Up to 1 CPU hour per training')).toBeInTheDocument();
  expect(card('Plus').getByText('€14')).toBeInTheDocument();
  expect(card('Plus').getByText('20 GB storage for data and trained models')).toBeInTheDocument();
  expect(card('Plus').getByText('One training at a time')).toBeInTheDocument();
  expect(
    card('Plus').getByText('More compute if you switch it on: €0.40 per CPU hour')
  ).toBeInTheDocument();
  expect(card('Pro').getByText('€45')).toBeInTheDocument();
  expect(card('Pro').getByText(/120 CPU hours per month/)).toBeInTheDocument();
  expect(card('Pro').getByText('Up to 8 CPU hours per training')).toBeInTheDocument();
  expect(card('Pro').getByText('2 trainings at the same time')).toBeInTheDocument();

  expect(screen.getByText(/Prices are per month and include VAT/)).toBeInTheDocument();
  expect(screen.getByText(/only used and billed if you switch it on yourself/)).toBeInTheDocument();
  expect(screen.getByText(/sold by Paddle \(Paddle.com Market Ltd\)/)).toBeInTheDocument();
  expect(screen.getByText(/Nothing is deleted when a plan ends/)).toBeInTheDocument();
  expect(screen.queryByText('Test mode')).not.toBeInTheDocument();
  // The terms and the privacy policy are one click away.
  expect(screen.getByRole('link', { name: 'Terms of use' })).toHaveAttribute('href', '/terms');
  expect(screen.getByRole('link', { name: 'Privacy policy' })).toHaveAttribute('href', '/privacy');
  // Looking at prices loads nothing from Paddle.
  expect(paddleScripts()).toHaveLength(0);
  expect(openCheckout).not.toHaveBeenCalled();
  expect(loadPaddle).not.toHaveBeenCalled();
});

test('a payment link from Paddle loads Paddle.js, which opens its checkout', async () => {
  renderPage('/pricing?_ptxn=txn_01abc');
  await screen.findByRole('heading', { name: 'Plans and prices' });
  await waitFor(() => expect(loadPaddle).toHaveBeenCalledWith(BILLING));
});

test('a payment link does nothing on a server that sells no plans', async () => {
  vi.mocked(api.billing.info).mockResolvedValue({ enabled: false });
  renderPage('/pricing?_ptxn=txn_01abc');
  await screen.findByText('Page not found');
  expect(loadPaddle).not.toHaveBeenCalled();
});

test('overage is not advertised when the server cannot bill it', async () => {
  vi.mocked(api.billing.info).mockResolvedValue({ ...BILLING, overage_billed: false });
  renderPage();
  await screen.findByRole('heading', { name: 'Plans and prices' });
  expect(screen.queryByText(/More compute/)).not.toBeInTheDocument();
  expect(screen.queryByText(/only used and billed/)).not.toBeInTheDocument();
});

test('the sandbox is marked as test mode', async () => {
  vi.mocked(api.billing.info).mockResolvedValue({ ...BILLING, environment: 'sandbox' });
  renderPage();
  expect(await screen.findByText('Test mode')).toBeInTheDocument();
  expect(screen.getByText(/nothing is charged/)).toBeInTheDocument();
});

test('visitors are sent to log in or sign up; no checkout opens', async () => {
  renderPage();
  await screen.findByRole('heading', { name: 'Plans and prices' });
  expect(card('Free').getByRole('link', { name: 'Sign up for free' })).toHaveAttribute(
    'href',
    '/register'
  );
  await userEvent.click(card('Plus').getByRole('link', { name: 'Log in to choose Plus' }));
  expect(await screen.findByText('Login page')).toBeInTheDocument();
  expect(api.billing.checkout).not.toHaveBeenCalled();
  expect(api.billing.state).not.toHaveBeenCalled();
});

test('choosing a plan asks the server and opens the checkout it prepared', async () => {
  logIn(NO_SUBSCRIPTION);
  vi.mocked(api.billing.checkout).mockResolvedValue(CHECKOUT);
  renderPage();
  const choose = await screen.findByRole('button', { name: 'Choose Plus' });
  expect(card('Free').getByText('Your plan')).toBeInTheDocument();
  expect(openCheckout).not.toHaveBeenCalled();

  await userEvent.click(choose);
  await waitFor(() => expect(openCheckout).toHaveBeenCalledTimes(1));
  expect(api.billing.checkout).toHaveBeenCalledWith('plus');
  const [info, checkout, theme, handlers] = vi.mocked(openCheckout).mock.calls[0];
  expect(info).toBe(BILLING);
  expect(checkout).toEqual(CHECKOUT);
  expect(theme).toBe('light');
  expect(typeof handlers?.onCompleted).toBe('function');
});

test('a checkout that cannot be opened says why', async () => {
  logIn(NO_SUBSCRIPTION);
  vi.mocked(api.billing.checkout).mockRejectedValue(
    new ApiError(409, null, 'Add an e-mail address to your account first: Paddle needs one.')
  );
  renderPage();
  await userEvent.click(await screen.findByRole('button', { name: 'Choose Pro' }));
  expect(await screen.findByText('Could not open the checkout')).toBeInTheDocument();
  expect(screen.getByText(/Add an e-mail address to your account first/)).toBeInTheDocument();
  expect(openCheckout).not.toHaveBeenCalled();
  expect(screen.getByRole('button', { name: 'Choose Pro' })).toBeEnabled();
});

test('after the payment the page waits until the plan has arrived', async () => {
  logIn(NO_SUBSCRIPTION);
  vi.mocked(api.billing.checkout).mockResolvedValue(CHECKOUT);
  renderPage();
  await userEvent.click(await screen.findByRole('button', { name: 'Choose Plus' }));
  await waitFor(() => expect(openCheckout).toHaveBeenCalledTimes(1));

  // Paddle reports the payment; the server learns of the subscription a moment later.
  vi.useFakeTimers({ shouldAdvanceTime: true });
  vi.mocked(api.billing.state)
    .mockResolvedValueOnce(NO_SUBSCRIPTION)
    .mockResolvedValue(subscribed());
  const { onCompleted } = vi.mocked(openCheckout).mock.calls[0][3]!;
  const done = onCompleted!();
  expect(await screen.findByText(/Payment received. Activating your plan/)).toBeInTheDocument();
  await vi.advanceTimersByTimeAsync(4500);
  await done;
  vi.useRealTimers();

  expect(await screen.findByText('You are on the Plus plan now')).toBeInTheDocument();
  expect(card('Plus').getByText('Your plan')).toBeInTheDocument();
  expect(screen.queryByRole('button', { name: /Choose/ })).not.toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Switch to Pro' })).toBeInTheDocument();
});

test('waiting for the plan asks Paddle through the server when it takes long', async () => {
  vi.useFakeTimers();
  vi.mocked(api.billing.state).mockResolvedValue(NO_SUBSCRIPTION);
  vi.mocked(api.billing.refresh).mockResolvedValue(subscribed());
  const waiting = waitForPlan('plus', { intervalMs: 1000, timeoutMs: 20_000 });
  await vi.advanceTimersByTimeAsync(4000);
  expect(await waiting).toEqual(subscribed());
  expect(api.billing.state).toHaveBeenCalledTimes(3);
  expect(api.billing.refresh).toHaveBeenCalledTimes(1);
});

test('waiting for the plan gives up after a while', async () => {
  vi.useFakeTimers();
  vi.mocked(api.billing.state).mockResolvedValue(NO_SUBSCRIPTION);
  vi.mocked(api.billing.refresh).mockRejectedValue(new ApiError(503, null, 'Paddle is down'));
  const waiting = waitForPlan('plus', { intervalMs: 1000, timeoutMs: 6000 });
  await vi.advanceTimersByTimeAsync(7000);
  expect(await waiting).toBeNull();
});

test('subscribers switch plans after seeing the price', async () => {
  logIn(subscribed());
  vi.mocked(api.billing.previewChange).mockResolvedValue({
    plan: PRO,
    upgrade: true,
    now: { action: 'charge', amount: 15.5, currency: 'EUR' },
    next_billed_at: '2026-12-09T10:00:00Z',
  });
  vi.mocked(api.billing.changePlan).mockResolvedValue(
    subscribed({ plan: 'pro', plan_name: 'Pro' }, PRO)
  );
  renderPage();
  expect(await screen.findByText('Your plan')).toBeInTheDocument();
  expect(card('Plus').getByText('Your plan')).toBeInTheDocument();
  expect(card('Plus').getByRole('link', { name: 'Manage on your account page' })).toHaveAttribute(
    'href',
    '/account'
  );
  expect(
    card('Free').getByText(/Cancel your subscription on your account page/)
  ).toBeInTheDocument();
  expect(screen.queryByRole('button', { name: /Choose/ })).not.toBeInTheDocument();

  await userEvent.click(card('Pro').getByRole('button', { name: 'Switch to Pro' }));
  const dialog = within(await screen.findByRole('dialog', { name: 'Switch to the Pro plan?' }));
  expect(dialog.getByText('€45 per month')).toBeInTheDocument();
  expect(dialog.getByText(/VAT included/)).toBeInTheDocument();
  expect(
    await dialog.findByText(/€15.50 is charged now for the rest of the month you paid for/)
  ).toBeInTheDocument();
  expect(dialog.getByText(/Next payment: 9 December 2026/)).toBeInTheDocument();
  expect(api.billing.previewChange).toHaveBeenCalledWith('pro');
  expect(api.billing.changePlan).not.toHaveBeenCalled();

  await userEvent.click(dialog.getByRole('button', { name: 'Switch to Pro' }));
  await waitFor(() => expect(api.billing.changePlan).toHaveBeenCalledWith('pro'));
  expect(await screen.findByText('You are on the Pro plan now')).toBeInTheDocument();
  await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
  expect(card('Pro').getByText('Your plan')).toBeInTheDocument();
  expect(openCheckout).not.toHaveBeenCalled();
});

test('a downgrade says what is credited, and a refusal keeps the dialog open', async () => {
  logIn(subscribed({ plan: 'pro', plan_name: 'Pro' }, PRO));
  vi.mocked(api.billing.previewChange).mockResolvedValue({
    plan: BILLING.plans[1],
    upgrade: false,
    now: { action: 'credit', amount: 31, currency: 'EUR' },
    next_billed_at: null,
  });
  vi.mocked(api.billing.changePlan).mockRejectedValue(
    new ApiError(503, null, 'The payment provider (Paddle) did not answer.')
  );
  renderPage();
  await userEvent.click(await screen.findByRole('button', { name: 'Switch to Starter' }));
  const dialog = within(await screen.findByRole('dialog'));
  expect(await dialog.findByText(/€31 is credited to your next invoice/)).toBeInTheDocument();
  await userEvent.click(dialog.getByRole('button', { name: 'Switch to Starter' }));
  expect(await dialog.findByText(/Paddle\) did not answer/)).toBeInTheDocument();
  await userEvent.click(dialog.getByRole('button', { name: 'Keep Pro' }));
  await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
});

test('a subscription that ends or has an open payment cannot change its plan here', async () => {
  logIn(
    subscribed({ ends_at: '2026-12-09T10:00:00Z', can_change_plan: false, can_reactivate: true })
  );
  renderPage();
  await screen.findByText('Your plan');
  expect(
    screen.queryByRole('button', { name: /^Switch to (Starter|Plus|Pro)/ })
  ).not.toBeInTheDocument();
  expect(
    card('Pro').getByText(/Continue your subscription on your account page/)
  ).toBeInTheDocument();
});

test('unlimited accounts are told that they need no plan', async () => {
  logIn({ ...NO_SUBSCRIPTION, unlimited: true, can_subscribe: false });
  renderPage();
  expect(await screen.findByText(/you do not need a plan/)).toBeInTheDocument();
  expect(
    screen.queryByRole('button', { name: /^(Choose|Switch to) (Starter|Plus|Pro)/ })
  ).not.toBeInTheDocument();
  expect(screen.queryByText('Your plan')).not.toBeInTheDocument();
});
