import { axeViolations, render, screen, userEvent, waitFor } from '@test-utils';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { Alert } from '@mantine/core';
import { api, ApiError, type PlanOffer, type Usage } from '@/api';
import { setLanguage } from '@/i18n';
import {
  LimitNotice,
  limitOf,
  limitText,
  limitTitle,
  pricingPath,
  type Limit,
} from './LimitNotice';
import { STARTER } from './testData';

vi.mock('@/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api')>();
  return { ...actual, api: { account: { usage: vi.fn(), setOverage: vi.fn() } } };
});

const OFFER: PlanOffer = {
  ...STARTER,
  storage_gb: 3,
  url: 'https://brickdnn.com/pricing?plan=starter',
};

/** A free account whose compute for the month is used up, as the server answers. */
const CPU_TIME: Limit = {
  code: 'budget_exceeded',
  limit: {
    kind: 'cpu_time',
    used: 30,
    limit: 30,
    unit: 'cpu_minutes',
    resets_at: '2026-12-01T00:00:00Z',
  },
  plan: { key: 'free', name: 'Free' },
  upgrade: OFFER,
  overage: null,
};

function show(info: Limit, props: Partial<Parameters<typeof LimitNotice>[0]> = {}) {
  return render(
    <MemoryRouter>
      <Alert title={limitTitle(info) ?? 'Limit'}>
        <LimitNotice info={info} fallback="The server's own words." {...props} />
      </Alert>
    </MemoryRouter>
  );
}

beforeEach(() => {
  vi.clearAllMocks();
});

describe('the limit in the answer of the server', () => {
  it('is read from a refused request and from a part of an answer', () => {
    expect(limitOf(new ApiError(403, CPU_TIME, 'Used up'))).toEqual(CPU_TIME);
    expect(limitOf({ allowed: false, ...CPU_TIME })?.limit.kind).toBe('cpu_time');
    // Older servers and other errors have none.
    expect(limitOf(new ApiError(403, { detail: 'No', code: 'budget_exceeded' }, 'No'))).toBeNull();
    expect(limitOf(new Error('offline'))).toBeNull();
    expect(limitOf(null)).toBeNull();
  });

  it('leads to the pricing page of this app with the plan preselected', () => {
    expect(pricingPath('starter')).toBe('/pricing?plan=starter');
    expect(pricingPath()).toBe('/pricing');
  });
});

describe('a used-up month', () => {
  it('names the limit, the next plan with its price and what it includes, and one button', () => {
    show(CPU_TIME);
    expect(screen.getByText('This month’s compute is used up')).toBeInTheDocument();
    expect(
      screen.getByText(
        'This month’s compute of the Free plan is used up: 30 CPU min of 30 CPU min. It is back on 1 December 2026.'
      )
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        'The Starter plan includes 10 CPU hours per month, 3 GB of storage, and up to 1 CPU hour per training for €5 per month.'
      )
    ).toBeInTheDocument();
    const buttons = screen.getAllByRole('link');
    expect(buttons).toHaveLength(1);
    expect(buttons[0]).toHaveAccessibleName('Upgrade to Starter');
    expect(buttons[0]).toHaveAttribute('href', '/pricing?plan=starter');
    // The server's text (with its address that cannot be clicked) is not shown.
    expect(screen.queryByText(/server's own words/)).not.toBeInTheDocument();
  });

  it('opens the pricing page in a new tab inside the training panel, and says so', () => {
    show(CPU_TIME, { newTab: true });
    const button = screen.getByRole('link', {
      name: /Upgrade to Starter\s*\(opens in a new tab\)/,
    });
    expect(button).toHaveAttribute('target', '_blank');
    expect(button).toHaveAttribute('href', '/pricing?plan=starter');
  });

  it('shows no plan and no price on a server that sells none', () => {
    show({ ...CPU_TIME, upgrade: null });
    expect(screen.getByText(/It is back on 1 December 2026/)).toBeInTheDocument();
    expect(screen.queryByRole('link')).not.toBeInTheDocument();
    expect(screen.queryByText(/€|per month for|Upgrade/)).not.toBeInTheDocument();
  });

  it('is in German in the German app', async () => {
    await setLanguage('de', { save: false });
    show(CPU_TIME);
    expect(
      screen.getByText(
        'Die Rechenzeit des Tarifs Free für diesen Monat ist aufgebraucht (30 CPU-Min. von 30 CPU-Min.). Am 1. Dezember 2026 steht sie wieder bereit.'
      )
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        'Der Tarif Starter enthält 10 CPU-Stunden pro Monat, 3 GB Speicher und bis zu 1 CPU-Stunde pro Training für 5 € pro Monat.'
      )
    ).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Zu Starter wechseln' })).toBeInTheDocument();
  });

  it('has no accessibility violations', async () => {
    show(CPU_TIME, { newTab: true });
    expect(await axeViolations()).toEqual([]);
  });
});

describe('a paid plan that ran out', () => {
  const PAID: Limit = {
    ...CPU_TIME,
    limit: { ...CPU_TIME.limit, used: 600, limit: 600 },
    plan: { key: 'starter', name: 'Starter' },
    upgrade: null,
    overage: { price_per_cpu_hour: 0.4, currency: 'EUR' },
  };
  const USAGE = {
    overage: {
      available: true,
      enabled: false,
      price_per_cpu_hour: 0.4,
      currency: 'EUR',
      cap_cpu_hours: null,
      used_cpu_minutes: 0,
      cost: null,
    },
  } as Usage;

  it('offers compute beyond the plan, which the user switches on with a limit of their own', async () => {
    vi.mocked(api.account.usage).mockResolvedValue(USAGE);
    vi.mocked(api.account.setOverage).mockResolvedValue({
      ...USAGE,
      overage: { ...USAGE.overage!, enabled: true, cap_cpu_hours: 5 },
    });
    const onChanged = vi.fn();
    show(PAID, { onChanged });
    expect(
      screen.getByText(
        'You can also allow compute beyond your plan at €0.40 per CPU hour, up to a limit you set.'
      )
    ).toBeInTheDocument();
    // Nothing is switched on by looking at it.
    expect(api.account.setOverage).not.toHaveBeenCalled();
    await userEvent.click(screen.getByRole('button', { name: 'Allow extra compute' }));
    const dialog = await screen.findByRole('dialog', { name: 'Compute beyond your plan' });
    expect(dialog).toBeInTheDocument();
    await screen.findByText('Allow compute beyond the monthly allowance');
    const toggle = screen.getByRole('switch');
    expect(toggle).toBeChecked();
    await userEvent.type(screen.getByLabelText('Most beyond the allowance'), '5');
    await userEvent.click(screen.getByRole('button', { name: 'Save' }));
    await waitFor(() => expect(api.account.setOverage).toHaveBeenCalledWith(true, 5));
    await waitFor(() => expect(onChanged).toHaveBeenCalled());
    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
  });

  it('offers the next plan first when there is one', () => {
    show({ ...PAID, upgrade: { ...OFFER, key: 'plus', name: 'Plus', price: 14 } });
    expect(screen.getByText(/The Plus plan includes/)).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Upgrade to Plus' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Allow extra compute' })).toBeInTheDocument();
  });
});

describe('other limits', () => {
  it('says what one training may use against what this one needs', () => {
    const info: Limit = {
      limit: { kind: 'run_time', limit: 15, unit: 'cpu_minutes' },
      plan: { key: 'free', name: 'Free' },
      upgrade: OFFER,
    };
    expect(limitText(info, { estimate: 42 })).toBe(
      'With the Free plan one training may use up to 15 CPU min; this one needs about 42 CPU min.'
    );
    expect(limitText(info, { stopped: true })).toBe(
      'This training reached 15 CPU min, the most one training may use with the Free plan.'
    );
    expect(limitText({ ...info, plan: null }, { estimate: 42 })).toBe(
      'One training may use up to 15 CPU min; this one needs about 42 CPU min.'
    );
    expect(limitTitle(info)).toBe('Too long for one training');
  });

  it('says how full the storage is', () => {
    const info: Limit = {
      limit: { kind: 'storage', used: 536870912, limit: 536870912, unit: 'bytes' },
      plan: { key: 'free', name: 'Free' },
      upgrade: OFFER,
    };
    expect(limitText(info)).toBe(
      'The storage of the Free plan is full: 512 MB of 512 MB. Delete data or training runs you no longer need.'
    );
  });

  it('says how many trainings run at a time, and which plan trains more', () => {
    const info: Limit = {
      code: 'concurrent_trainings_limit',
      limit: { kind: 'concurrent_trainings', used: 1, limit: 1, unit: 'trainings' },
      plan: { key: 'free', name: 'Free' },
      upgrade: { ...OFFER, key: 'pro', name: 'Pro', price: 45, concurrent_trainings: 2 },
    };
    show(info);
    expect(screen.getByText('A training is already running')).toBeInTheDocument();
    expect(
      screen.getByText(
        'The Free plan trains one network at a time. Wait until your running training finishes, or stop it.'
      )
    ).toBeInTheDocument();
    expect(
      screen.getByText(/The Pro plan includes .* 2 trainings at a time for €45/)
    ).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Upgrade to Pro' })).toHaveAttribute(
      'href',
      '/pricing?plan=pro'
    );
  });

  it('leads to the account page for budgets the user set, not to a plan', () => {
    show({
      limit: { kind: 'own_budget', used: 10, limit: 10, unit: 'cpu_minutes' },
      plan: { key: 'free', name: 'Free' },
      upgrade: null,
    });
    expect(
      screen.getByText(/The monthly budget you set for your account is used up/)
    ).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Usage & budget' })).toHaveAttribute(
      'href',
      '/account'
    );
    expect(screen.queryByText(/Upgrade/)).not.toBeInTheDocument();
  });

  it('suggests nothing while running trainings reserve the rest', () => {
    show({ ...CPU_TIME, reserved: true });
    expect(
      screen.getByText(/reserved by your trainings that are still running/)
    ).toBeInTheDocument();
    expect(screen.queryByRole('link')).not.toBeInTheDocument();
  });

  it('keeps the words of the server for limits it does not know', () => {
    show({ limit: { kind: 'token_budget', used: 1, limit: 1, unit: 'cpu_minutes' } });
    expect(screen.getByText("The server's own words.")).toBeInTheDocument();
  });
});
