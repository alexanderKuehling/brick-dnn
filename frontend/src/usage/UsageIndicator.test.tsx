import { axeViolations, render, screen } from '@test-utils';
import { MemoryRouter } from 'react-router-dom';
import { describe, expect, it } from 'vitest';
import type { PlanOffer, Usage } from '@/api';
import { FREE, STARTER } from '@/billing/testData';
import { setLanguage } from '@/i18n';
import { budgetText, trainingsThatFit } from './CostEstimate';
import { computeText, UsageIndicator, usageLevel } from './UsageIndicator';

const MB = 1024 * 1024;
const OFFER: PlanOffer = { ...STARTER, url: '/pricing?plan=starter' };

/** A free account on a server that sells plans: 30 CPU minutes and 512 MB a month. */
function usage(usedMinutes: number, usedBytes = 100 * MB, extra: Partial<Usage> = {}): Usage {
  return {
    period: { start: '2026-11-01', resets_at: '2026-12-01T00:00:00Z' },
    plan: FREE,
    unlimited: false,
    plans: [FREE, STARTER],
    upgrade_url: 'https://brickdnn.com/pricing',
    next_plans: { compute: OFFER, storage: OFFER },
    compute: {
      unit: 'cpu_minutes',
      used: usedMinutes,
      limit: 30,
      remaining: Math.max(0, 30 - usedMinutes),
      available: Math.max(0, 30 - usedMinutes),
      allowance: 30,
      own_budget: null,
      run_limit: 15,
      reserved_by_active_runs: 0,
      by_kind: { training: usedMinutes, prediction: 0 },
      by_source: [],
    },
    storage: {
      used_bytes: usedBytes,
      limit_bytes: 512 * MB,
      remaining_bytes: 512 * MB - usedBytes,
      by_kind: { data_sources: 0, datasets: usedBytes, training_runs: 0 },
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
    ...extra,
  };
}

function show(value: Usage | null, newTab = false) {
  return render(
    <MemoryRouter>
      <UsageIndicator usage={value} newTab={newTab} />
    </MemoryRouter>
  );
}

const indicator = () => screen.getByTestId('usage-indicator');

describe('the usage where trainings are started', () => {
  it('is one quiet line while there is room', () => {
    show(usage(18));
    expect(indicator()).toHaveAttribute('data-level', 'ok');
    expect(
      screen.getByText('Usage this month: 18 of 30 CPU min · 100 MB of 512 MB storage')
    ).toBeInTheDocument();
    expect(screen.queryByRole('link')).not.toBeInTheDocument();
    expect(screen.queryByText(/€|plan/)).not.toBeInTheDocument();
  });

  it('stays quiet just below 80 %', () => {
    show(usage(23.9));
    expect(indicator()).toHaveAttribute('data-level', 'ok');
    expect(screen.queryByRole('link')).not.toBeInTheDocument();
  });

  it('warns from 80 % with the next plan and a link to the plans', () => {
    show(usage(24));
    expect(indicator()).toHaveAttribute('data-level', 'nearly');
    expect(
      screen.getByText(
        /80 % of this month’s compute is used\. Starter includes 10 CPU hours per month, 5 GB of storage, and up to 1 CPU hour per training for €5 per month\./
      )
    ).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'See plans' })).toHaveAttribute(
      'href',
      '/pricing?plan=starter'
    );
  });

  it('says at 100 % that the month is used up and when it is back', () => {
    show(usage(30), true);
    expect(indicator()).toHaveAttribute('data-level', 'full');
    expect(
      screen.getByText(/This month’s compute is used up; it is back on 1 December 2026\./)
    ).toBeInTheDocument();
    const link = screen.getByRole('link', { name: /See plans\s*\(opens in a new tab\)/ });
    expect(link).toHaveAttribute('target', '_blank');
  });

  it('warns about the storage when that is what runs out', () => {
    show(usage(3, 460 * MB));
    expect(indicator()).toHaveAttribute('data-level', 'nearly');
    expect(screen.getByText(/89 % of your storage is used\./)).toBeInTheDocument();
    show(usage(3, 512 * MB));
    expect(screen.getByText(/Your storage is full\./)).toBeInTheDocument();
  });

  it('leads to the account page, without a plan or price, where no plans are sold', () => {
    show(usage(30, 100 * MB, { upgrade_url: null, next_plans: { compute: null, storage: null } }));
    expect(indicator()).toHaveAttribute('data-level', 'full');
    expect(screen.getByRole('link', { name: 'Usage & budget' })).toHaveAttribute(
      'href',
      '/account'
    );
    expect(screen.queryByText(/€|Starter|See plans/)).not.toBeInTheDocument();
  });

  it('names no plan when a budget of the user is the limit', () => {
    const own = usage(8);
    own.compute = { ...own.compute, limit: 10, own_budget: 10 };
    show(own);
    expect(screen.getByText(/80 % of this month’s compute is used\./)).toBeInTheDocument();
    expect(screen.queryByText(/Starter includes/)).not.toBeInTheDocument();
  });

  it('shows nothing for an account without limits', () => {
    show(usage(18, 100 * MB, { unlimited: true }));
    const none = usage(18);
    none.compute = { ...none.compute, limit: null, allowance: null };
    none.storage = { ...none.storage, limit_bytes: null };
    show(none);
    show(null);
    expect(screen.queryByTestId('usage-indicator')).not.toBeInTheDocument();
    expect(screen.queryByText(/Usage this month/)).not.toBeInTheDocument();
  });

  it('is in German in the German app', async () => {
    await setLanguage('de', { save: false });
    show(usage(24));
    expect(
      screen.getByText('Nutzung in diesem Monat: 24 von 30 CPU-Min. · 100 MB von 512 MB Speicher')
    ).toBeInTheDocument();
    expect(
      screen.getByText(/80 % der Rechenzeit dieses Monats sind verbraucht\./)
    ).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Tarife ansehen' })).toBeInTheDocument();
  });

  it('has no accessibility violations at any level', async () => {
    for (const used of [18, 24, 30]) {
      const { unmount } = show(usage(used), true);
      expect(await axeViolations()).toEqual([]);
      unmount();
    }
  });
});

describe('numbers', () => {
  it('turns into a warning at 80 % and into "full" at 100 %', () => {
    expect([0, 79.9, 80, 99.9, 100, 140].map(usageLevel)).toEqual([
      'ok',
      'ok',
      'nearly',
      'nearly',
      'full',
      'full',
    ]);
  });

  it('writes small allowances in CPU minutes and larger ones in CPU hours', () => {
    expect(computeText(18, 30)).toBe('18 of 30 CPU min');
    expect(computeText(18, 600)).toBe('0.3 of 10 CPU h');
  });

  it('says how many trainings like the planned one fit into what is left', () => {
    const state = {
      estimate: { cpu_minutes: 9.9 } as never,
      budget: {
        allowed: true,
        reason: null,
        warning: null,
        run_limit_cpu_minutes: 15,
        limit_reason: 'run_limit' as const,
        remaining_cpu_minutes: 30,
        resets_at: '2026-12-01T00:00:00Z',
      },
    };
    expect(trainingsThatFit(state)).toBe(3);
    expect(budgetText(state)).toBe(
      'It may use up to 15 CPU min (the limit per training). 30 CPU min left this month, until 1 December 2026. About 3 trainings like this one fit into what is left.'
    );
    const one = { ...state, estimate: { cpu_minutes: 20 } as never };
    expect(budgetText(one)).toMatch(/About 1 training like this one fits into what is left\.$/);
    // Less than one fits: the warning of the check says so, not this line.
    expect(budgetText({ ...state, estimate: { cpu_minutes: 40 } as never })).not.toMatch(/fit/);
    expect(trainingsThatFit({ budget: state.budget })).toBeNull();
  });
});
