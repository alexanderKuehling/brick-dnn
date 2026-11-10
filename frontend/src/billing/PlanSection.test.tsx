import { render, screen, userEvent, waitFor, within } from '@test-utils';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import { Notifications, notifications } from '@mantine/notifications';
import { api, ApiError, type BillingState } from '@/api';
import { planFeatures } from './format';
import { PlanSection } from './PlanSection';
import { BILLING, FREE, NO_SUBSCRIPTION, PLUS, subscribed } from './testData';

vi.mock('@/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api')>();
  return {
    ...actual,
    api: {
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

function renderSection(state: BillingState, info = BILLING) {
  vi.mocked(api.billing.state).mockResolvedValue(state);
  return render(
    <MemoryRouter initialEntries={['/account']}>
      <Notifications />
      <Routes>
        <Route path="/account" element={<PlanSection info={info} />} />
        <Route path="/pricing" element={<p>Pricing page</p>} />
      </Routes>
    </MemoryRouter>
  );
}

beforeEach(() => {
  vi.clearAllMocks();
});

afterEach(() => {
  notifications.clean();
  vi.unstubAllGlobals();
});

test('the free plan: an invitation to see the plans, nothing to manage', async () => {
  renderSection(NO_SUBSCRIPTION);
  expect(await screen.findByText('Free plan')).toBeInTheDocument();
  expect(screen.getByText(/You are on the free plan/)).toBeInTheDocument();
  expect(screen.queryByRole('button', { name: /Cancel subscription/ })).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: /Payment method/ })).not.toBeInTheDocument();
  expect(screen.queryByText(/customer portal/)).not.toBeInTheDocument();
  expect(screen.queryByText('Test mode')).not.toBeInTheDocument();
  await userEvent.click(screen.getByRole('link', { name: 'See plans' }));
  expect(await screen.findByText('Pricing page')).toBeInTheDocument();
});

test('an active subscription: when it renews and for how much', async () => {
  renderSection(subscribed());
  expect(await screen.findByText('Plus plan')).toBeInTheDocument();
  expect(screen.getByText('Active')).toBeInTheDocument();
  expect(screen.getByText('Renews on 9 December 2026 for €14 per month.')).toBeInTheDocument();
  expect(screen.getByRole('link', { name: 'Change plan' })).toHaveAttribute('href', '/pricing');
  expect(screen.getByRole('button', { name: 'Payment method and invoices' })).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Cancel subscription' })).toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Continue subscription' })).not.toBeInTheDocument();
  expect(screen.queryByRole('link', { name: 'See plans' })).not.toBeInTheDocument();
});

test('cancelling asks first and says until when the plan stays', async () => {
  const cancelled = subscribed({
    renews_at: null,
    ends_at: '2026-12-09T10:00:00Z',
    can_change_plan: false,
    can_cancel: false,
    can_reactivate: true,
  });
  vi.mocked(api.billing.cancel).mockResolvedValue(cancelled);
  renderSection(subscribed());
  await userEvent.click(await screen.findByRole('button', { name: 'Cancel subscription' }));
  const dialog = within(await screen.findByRole('dialog', { name: 'Cancel your subscription?' }));
  expect(dialog.getByText(/You keep Plus until 9 December 2026/)).toBeInTheDocument();
  expect(dialog.getByText(/Nothing is deleted/)).toBeInTheDocument();
  expect(api.billing.cancel).not.toHaveBeenCalled();

  await userEvent.click(dialog.getByRole('button', { name: 'Cancel subscription' }));
  await waitFor(() => expect(api.billing.cancel).toHaveBeenCalledTimes(1));
  expect(await screen.findByText('Your subscription was cancelled.')).toBeInTheDocument();
  await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
  expect(screen.getByText('Cancelled')).toBeInTheDocument();
  expect(screen.getByText(/you keep Plus until 9 December 2026/)).toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Cancel subscription' })).not.toBeInTheDocument();
  expect(screen.queryByRole('link', { name: 'Change plan' })).not.toBeInTheDocument();
});

test('keeping the plan in the cancel dialog changes nothing; errors stay in it', async () => {
  vi.mocked(api.billing.cancel).mockRejectedValue(
    new ApiError(503, null, 'The payment provider (Paddle) did not answer.')
  );
  renderSection(subscribed());
  await userEvent.click(await screen.findByRole('button', { name: 'Cancel subscription' }));
  const dialog = within(await screen.findByRole('dialog'));
  await userEvent.click(dialog.getByRole('button', { name: 'Cancel subscription' }));
  expect(await dialog.findByText(/Paddle\) did not answer/)).toBeInTheDocument();
  await userEvent.click(dialog.getByRole('button', { name: 'Keep Plus' }));
  await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
  expect(screen.getByText('Active')).toBeInTheDocument();
});

test('a cancelled subscription can be continued', async () => {
  vi.mocked(api.billing.reactivate).mockResolvedValue(subscribed());
  renderSection(
    subscribed({
      renews_at: null,
      ends_at: '2026-12-09T10:00:00Z',
      can_change_plan: false,
      can_cancel: false,
      can_reactivate: true,
    })
  );
  await userEvent.click(await screen.findByRole('button', { name: 'Continue subscription' }));
  expect(await screen.findByText('Your subscription continues.')).toBeInTheDocument();
  expect(api.billing.reactivate).toHaveBeenCalledTimes(1);
  expect(screen.getByText('Active')).toBeInTheDocument();
  expect(screen.getByText(/Renews on 9 December 2026/)).toBeInTheDocument();
});

test('a failed payment is a warning with the way to fix it', async () => {
  renderSection(
    subscribed({
      status: 'past_due',
      payment_failed: true,
      renews_at: null,
      grace_until: '2026-11-16T10:00:00Z',
      can_change_plan: false,
    })
  );
  expect(await screen.findByText('Payment failed')).toBeInTheDocument();
  const alert = within(screen.getByRole('alert'));
  expect(alert.getByText(/The last payment did not go through/)).toBeInTheDocument();
  expect(alert.getByText(/you keep Plus until 16 November 2026 at least/)).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Payment method and invoices' })).toBeInTheDocument();
  expect(screen.queryByRole('link', { name: 'Change plan' })).not.toBeInTheDocument();
});

test('an ended subscription: back on the free plan, nothing deleted', async () => {
  renderSection({
    ...subscribed(
      {
        status: 'canceled',
        renews_at: null,
        ended_at: '2026-11-01T10:00:00Z',
        current_period_end: null,
        can_change_plan: false,
        can_cancel: false,
      },
      { ...FREE }
    ),
    can_subscribe: true,
  });
  expect(await screen.findByText('Free plan')).toBeInTheDocument();
  expect(screen.getByText(/Your Plus subscription ended on 1 November 2026/)).toBeInTheDocument();
  expect(screen.getByText(/Nothing was deleted/)).toBeInTheDocument();
  expect(screen.queryByText('Ended')).not.toBeInTheDocument();
  expect(screen.getByRole('link', { name: 'See plans' })).toBeInTheDocument();
  // The invoices of the past are still in Paddle's portal.
  expect(screen.getByRole('button', { name: 'Payment method and invoices' })).toBeInTheDocument();
});

test('unlimited accounts need no plan', async () => {
  renderSection({ ...NO_SUBSCRIPTION, unlimited: true, can_subscribe: false });
  expect(await screen.findByText('No plan needed')).toBeInTheDocument();
  expect(screen.getByText(/No limits apply to your account/)).toBeInTheDocument();
  expect(screen.queryByRole('link')).not.toBeInTheDocument();
  expect(screen.queryByRole('button')).not.toBeInTheDocument();
});

test('the customer portal opens in a new tab with a link made for it', async () => {
  const tab = { opener: {} as unknown, location: { href: '' }, close: vi.fn() };
  const open = vi.fn(() => tab);
  vi.stubGlobal('open', open);
  vi.mocked(api.billing.portal).mockResolvedValue({ url: 'https://portal.paddle.example/x' });
  renderSection(subscribed());
  await userEvent.click(await screen.findByRole('button', { name: 'Payment method and invoices' }));
  await waitFor(() => expect(tab.location.href).toBe('https://portal.paddle.example/x'));
  expect(open).toHaveBeenCalledWith('', '_blank');
  expect(tab.opener).toBeNull();

  // Without an answer the empty tab is closed again.
  vi.mocked(api.billing.portal).mockRejectedValue(new ApiError(503, null, 'Paddle is down.'));
  await userEvent.click(screen.getByRole('button', { name: 'Payment method and invoices' }));
  expect(await screen.findByText('Paddle is down.')).toBeInTheDocument();
  expect(tab.close).toHaveBeenCalledTimes(1);
});

test('the sandbox is marked, and an error loading the state is shown', async () => {
  const view = renderSection(subscribed(), { ...BILLING, environment: 'sandbox' });
  expect(await screen.findByText('Test mode')).toBeInTheDocument();
  view.unmount();
  vi.mocked(api.billing.state).mockRejectedValue(new ApiError(500, null, 'Server error'));
  render(
    <MemoryRouter>
      <PlanSection info={BILLING} />
    </MemoryRouter>
  );
  expect(await screen.findByText('Server error')).toBeInTheDocument();
});

test('plan features in words', () => {
  expect(planFeatures(PLUS, true)).toEqual([
    '30 CPU hours per month for trainings and predictions',
    '20 GB storage for data and trained models',
    'Up to 3 CPU hours per training',
    'One training at a time',
    'More compute if you switch it on: €0.40 per CPU hour',
  ]);
  expect(planFeatures(PLUS, false)).toHaveLength(4);
  expect(
    planFeatures(
      {
        ...FREE,
        cpu_hours_per_month: null,
        storage_gb: null,
        cpu_minutes_per_run: 90,
        concurrent_trainings: null,
      },
      true
    )
  ).toEqual([
    'Unlimited compute for trainings and predictions',
    'Unlimited storage',
    'Up to 90 CPU minutes per training',
  ]);
});
