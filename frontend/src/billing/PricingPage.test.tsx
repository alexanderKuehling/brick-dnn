import { axeViolations, render, screen, userEvent, waitFor, within } from '@test-utils';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import { Notifications, notifications } from '@mantine/notifications';
import { api, ApiError, setTokens, type BillingState } from '@/api';
import { AuthProvider } from '@/auth/AuthContext';
import { AppHeader } from '@/components/AppHeader';
import { setLanguage } from '@/i18n';
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
    pages: { imprint: true, privacy: true, terms: true, refund: false, accessibility: false },
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

test('a server that sells no plans says so, without prices, a link to it or a 404', async () => {
  vi.mocked(api.billing.info).mockResolvedValue({ enabled: false });
  renderPage();
  expect(
    await screen.findByRole('heading', { name: 'No plans are sold here' })
  ).toBeInTheDocument();
  expect(screen.queryByText('Page not found')).not.toBeInTheDocument();
  expect(screen.queryByRole('link', { name: 'Pricing' })).not.toBeInTheDocument();
  expect(screen.queryByText(/per month|€/)).not.toBeInTheDocument();
  expect(screen.getByRole('link', { name: 'Open the editor' })).toHaveAttribute('href', '/app');
  // Logged out: nothing about an account page either.
  expect(screen.queryByRole('link', { name: /account page/ })).not.toBeInTheDocument();
  expect(api.billing.state).not.toHaveBeenCalled();
  expect(paddleScripts()).toHaveLength(0);
});

test('no answer about billing offers to try again instead of a 404', async () => {
  vi.mocked(api.billing.info).mockRejectedValueOnce(new ApiError(500, null, 'Server error'));
  renderPage();
  expect(await screen.findByText('The plans could not be loaded')).toBeInTheDocument();
  expect(screen.queryByText('Page not found')).not.toBeInTheDocument();
  expect(screen.queryByText('No plans are sold here')).not.toBeInTheDocument();
  await userEvent.click(screen.getByRole('button', { name: 'Try again' }));
  expect(await screen.findByRole('heading', { name: 'Plans and prices' })).toBeInTheDocument();
  expect(api.billing.info).toHaveBeenCalledTimes(2);
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
  // Overage in one sentence; the price per CPU hour is on each card.
  expect(
    screen.getByText(/trainings stop unless you allow more yourself on your account page/)
  ).toBeInTheDocument();
  expect(screen.getByText(/sold by Paddle \(Paddle.com Market Ltd\)/)).toBeInTheDocument();
  expect(screen.getByText(/Nothing is deleted when a plan ends/)).toBeInTheDocument();
  expect(screen.queryByText('Test mode')).not.toBeInTheDocument();
  // The terms and the privacy policy are one click away.
  expect(screen.getByRole('link', { name: 'Terms' })).toHaveAttribute('href', '/terms');
  expect(screen.getByRole('link', { name: 'Privacy' })).toHaveAttribute('href', '/privacy');
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
  await screen.findByText('No plans are sold here');
  expect(loadPaddle).not.toHaveBeenCalled();
});

test('overage is not advertised when the server cannot bill it', async () => {
  vi.mocked(api.billing.info).mockResolvedValue({ ...BILLING, overage_billed: false });
  renderPage();
  await screen.findByRole('heading', { name: 'Plans and prices' });
  expect(screen.queryByText(/More compute/)).not.toBeInTheDocument();
  expect(screen.queryByText(/unless you allow more yourself/)).not.toBeInTheDocument();
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
  expect(api.billing.checkout).toHaveBeenCalledWith('plus', 'month');
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
    await dialog.findByText(/€15.50 is charged now for the rest of the period you paid for/)
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

// --------------------------------------------------------------------------
// Comparing plans, the plan a limit led here for, paying per year
// --------------------------------------------------------------------------

const PRIORITIES = BILLING.plans.map((plan, index) => ({ ...plan, priority: index }));

test('says what changes against the own plan, and which plans start first', async () => {
  vi.mocked(api.billing.info).mockResolvedValue({ ...BILLING, plans: PRIORITIES });
  logIn(NO_SUBSCRIPTION);
  renderPage();
  await screen.findByRole('button', { name: 'Choose Plus' });
  expect(card('Free').getByText('Your plan')).toBeInTheDocument();
  expect(card('Free').queryByText(/\(now/)).not.toBeInTheDocument();
  // 30 CPU hours instead of 0.5, 20 GB instead of 1, three hours per training instead of 15 min.
  expect(card('Plus').getByText('(now 0.5)')).toBeInTheDocument();
  expect(card('Plus').getByText('(now 1 GB)')).toBeInTheDocument();
  expect(card('Plus').getByText('(now 15 min)')).toBeInTheDocument();
  // One training at a time on both: nothing changes there.
  expect(card('Plus').getAllByText(/\(now/)).toHaveLength(3);
  expect(card('Pro').getByText('(now 1)')).toBeInTheDocument();
  // The queue: paid plans start before cheaper ones, the free plan does not.
  const line = 'Starts before trainings of cheaper plans when the server is busy';
  expect(card('Free').queryByText(line)).not.toBeInTheDocument();
  expect(card('Starter').getByText(line)).toBeInTheDocument();
  expect(card('Pro').getByText(line)).toBeInTheDocument();
});

test('plans without a queue priority of their own do not claim one', async () => {
  renderPage();
  await screen.findByRole('heading', { name: 'Plans and prices' });
  expect(screen.queryByText(/Starts before trainings/)).not.toBeInTheDocument();
});

test('the plan a limit message led here for is marked and its button is the main one', async () => {
  logIn(NO_SUBSCRIPTION);
  renderPage('/pricing?plan=starter');
  const choose = await screen.findByRole('button', { name: 'Choose Starter' });
  expect(screen.getByRole('region', { name: 'Starter plan' })).toHaveAttribute(
    'aria-current',
    'true'
  );
  expect(screen.getByRole('region', { name: 'Plus plan' })).not.toHaveAttribute('aria-current');
  expect(choose).toHaveAttribute('data-variant', 'filled');
  expect(screen.getByRole('button', { name: 'Choose Plus' })).toHaveAttribute(
    'data-variant',
    'light'
  );
  // Nothing opens by itself.
  expect(openCheckout).not.toHaveBeenCalled();
  expect(api.billing.checkout).not.toHaveBeenCalled();
});

test('visitors are offered a free account first, and come back to the plan', async () => {
  renderPage('/pricing?plan=plus');
  await screen.findByRole('heading', { name: 'Plans and prices' });
  const create = card('Plus').getByRole('link', { name: 'Create a free account to choose Plus' });
  expect(create).toHaveAttribute('href', '/register');
  expect(create).toHaveAttribute('data-variant', 'filled');
  expect(card('Plus').getByRole('link', { name: 'Log in to choose Plus' })).toBeInTheDocument();
  await userEvent.click(create);
  expect(await screen.findByText('Sign-up page')).toBeInTheDocument();
});

test('teaching and research are pointed to the operator, as on the landing page', async () => {
  renderPage();
  await screen.findByRole('heading', { name: 'Plans and prices' });
  expect(
    screen.getByText(/Discounts for teaching and research are available on request/)
  ).toBeInTheDocument();
  expect(screen.getByRole('link', { name: 'write to us' })).toHaveAttribute('href', '/imprint');
});

test('without yearly prices there is no switch and nothing about years', async () => {
  renderPage();
  await screen.findByRole('heading', { name: 'Plans and prices' });
  expect(screen.queryByRole('radio', { name: 'Yearly' })).not.toBeInTheDocument();
  expect(screen.queryByText(/per year|Yearly/)).not.toBeInTheDocument();
});

const YEARLY = {
  ...BILLING,
  yearly: true,
  plans: BILLING.plans.map((plan) =>
    plan.key === 'plus' || plan.key === 'pro' ? { ...plan, price_per_year: plan.price! * 10 } : plan
  ),
};

test('with yearly prices the page offers both and buys the chosen one', async () => {
  vi.mocked(api.billing.info).mockResolvedValue(YEARLY);
  logIn(NO_SUBSCRIPTION);
  vi.mocked(api.billing.checkout).mockResolvedValue({ ...CHECKOUT, interval: 'year' });
  renderPage();
  await screen.findByRole('button', { name: 'Choose Plus' });
  expect(card('Plus').getByText('€14')).toBeInTheDocument();
  expect(screen.getByText(/Prices are per month or per year and include VAT/)).toBeInTheDocument();

  await userEvent.click(screen.getByRole('radio', { name: 'Yearly' }));
  expect(card('Plus').getByText('€140')).toBeInTheDocument();
  expect(card('Plus').getByText('per year')).toBeInTheDocument();
  // What that is per month, and how much less than twelve months: from the two prices.
  expect(
    card('Plus').getByText('€11.67 per month · 17 % less than paying monthly')
  ).toBeInTheDocument();
  // A plan without a yearly price keeps its monthly one and says so.
  expect(card('Starter').getByText('€5')).toBeInTheDocument();
  expect(card('Starter').getByText('Only monthly')).toBeInTheDocument();

  await userEvent.click(screen.getByRole('button', { name: 'Choose Plus' }));
  await waitFor(() => expect(api.billing.checkout).toHaveBeenCalledWith('plus', 'year'));
  await userEvent.click(screen.getByRole('radio', { name: 'Monthly' }));
  expect(card('Plus').getByText('€14')).toBeInTheDocument();
});

test('a subscription keeps the way it is paid: no switch, its prices', async () => {
  vi.mocked(api.billing.info).mockResolvedValue(YEARLY);
  const state = subscribed();
  logIn({ ...state, subscription: { ...state.subscription!, interval: 'year' } });
  renderPage();
  await screen.findByText('Your plan');
  expect(screen.queryByRole('radio', { name: 'Yearly' })).not.toBeInTheDocument();
  expect(card('Pro').getByText('€450')).toBeInTheDocument();
  expect(card('Pro').getByRole('button', { name: 'Switch to Pro' })).toBeInTheDocument();
  // Starter has no yearly price: this subscription cannot move there.
  expect(card('Starter').queryByRole('button')).not.toBeInTheDocument();
  expect(card('Starter').getAllByText('Only monthly').length).toBeGreaterThan(0);
});

test('the page is German in the German app', async () => {
  await setLanguage('de', { save: false });
  vi.mocked(api.billing.info).mockResolvedValue({ ...BILLING, plans: PRIORITIES });
  logIn(NO_SUBSCRIPTION);
  renderPage('/pricing?plan=starter');
  expect(await screen.findByRole('heading', { name: 'Tarife und Preise' })).toBeInTheDocument();
  const starter = within(screen.getByRole('region', { name: 'Tarif Starter' }));
  expect(
    starter.getByText('10 CPU-Stunden pro Monat für Trainings und Vorhersagen')
  ).toBeInTheDocument();
  expect(starter.getByText('5 GB Speicher für Daten und trainierte Modelle')).toBeInTheDocument();
  expect(starter.getByText('Bis zu 1 CPU-Stunde pro Training')).toBeInTheDocument();
  expect(starter.getByText('Ein Training zur selben Zeit')).toBeInTheDocument();
  expect(
    starter.getByText('Mehr Rechenzeit, wenn Sie sie einschalten: 0,40 € pro CPU-Stunde')
  ).toBeInTheDocument();
  expect(await starter.findByText('(jetzt 0,5)')).toBeInTheDocument();
  expect(starter.getByRole('button', { name: 'Starter wählen' })).toBeInTheDocument();
  expect(
    screen.getByText(/Für Lehre und Forschung gibt es auf Anfrage einen Rabatt/)
  ).toBeInTheDocument();
  expect(
    screen.getByText(
      /stoppen Trainings, sofern Sie auf Ihrer Kontoseite nicht selbst mehr erlauben/
    )
  ).toBeInTheDocument();
  // No English plan lines are left.
  expect(
    screen.queryByText(/CPU hours per month|storage for data|per training$/)
  ).not.toBeInTheDocument();
});

test('a preselected plan and the yearly switch have no accessibility violations', async () => {
  vi.mocked(api.billing.info).mockResolvedValue(YEARLY);
  logIn(NO_SUBSCRIPTION);
  renderPage('/pricing?plan=plus');
  await screen.findByRole('button', { name: 'Choose Plus' });
  await userEvent.click(screen.getByRole('radio', { name: 'Yearly' }));
  expect(await axeViolations()).toEqual([]);
});

test('no plans, and no answer: both without accessibility violations', async () => {
  vi.mocked(api.billing.info).mockResolvedValue({ enabled: false });
  const { unmount } = renderPage();
  await screen.findByText('No plans are sold here');
  expect(await axeViolations()).toEqual([]);
  unmount();
  resetBillingInfo();
  vi.mocked(api.billing.info).mockRejectedValue(new ApiError(500, null, 'Server error'));
  renderPage();
  await screen.findByText('The plans could not be loaded');
  expect(await axeViolations()).toEqual([]);
});
