/** Test data: the plans of brickdnn.com and the states of an account's subscription. */
import type { BillingEnabled, BillingPlan, BillingState, BillingSubscription } from '@/api';

function plan(
  key: string,
  name: string,
  price: number,
  cpuHours: number,
  storageGb: number,
  minutesPerRun: number,
  extra: Partial<BillingPlan> = {}
): BillingPlan {
  return {
    key,
    name,
    price,
    currency: 'EUR',
    cpu_hours_per_month: cpuHours,
    storage_gb: storageGb,
    cpu_minutes_per_run: minutesPerRun,
    concurrent_trainings: 1,
    priority: 0,
    overage_price_per_cpu_hour: price > 0 ? 0.4 : null,
    purchasable: price > 0,
    default: price === 0,
    ...extra,
  };
}

export const FREE = plan('free', 'Free', 0, 0.5, 1, 15);
export const STARTER = plan('starter', 'Starter', 5, 10, 5, 60);
export const PLUS = plan('plus', 'Plus', 14, 30, 20, 180);
export const PRO = plan('pro', 'Pro', 45, 120, 100, 480, { concurrent_trainings: 2 });

export const BILLING: BillingEnabled = {
  enabled: true,
  provider: 'paddle',
  environment: 'live',
  client_token: 'live_0123456789abcdef',
  currency: 'EUR',
  prices_include_tax: true,
  overage_billed: true,
  plans: [FREE, STARTER, PLUS, PRO],
};

export const NO_SUBSCRIPTION: BillingState = {
  plan: FREE,
  unlimited: false,
  subscription: null,
  can_subscribe: true,
  has_billing_profile: false,
  email_missing: false,
};

export function subscription(extra: Partial<BillingSubscription> = {}): BillingSubscription {
  return {
    status: 'active',
    plan: 'plus',
    plan_name: 'Plus',
    renews_at: '2026-12-09T10:00:00Z',
    ends_at: null,
    ended_at: null,
    current_period_end: '2026-12-09T10:00:00Z',
    payment_failed: false,
    grace_until: null,
    can_change_plan: true,
    can_cancel: true,
    can_reactivate: false,
    ...extra,
  };
}

export function subscribed(
  extra: Partial<BillingSubscription> = {},
  current: BillingPlan = PLUS
): BillingState {
  return {
    plan: current,
    unlimited: false,
    subscription: subscription(extra),
    can_subscribe: false,
    has_billing_profile: true,
    email_missing: false,
  };
}
