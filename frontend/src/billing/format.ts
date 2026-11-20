/** What plans include and cost, in words. */
import type { BillingInterval, Plan } from '@/api';
import { currentLocale, i18n, longDateLocale } from '@/i18n';

/** 14, "EUR" → "€14", 4.5 → "€4.50". */
export function formatPrice(amount: number, currency: string | null): string {
  const digits = Number.isInteger(amount) ? 0 : 2;
  try {
    return amount.toLocaleString(currentLocale(), {
      style: 'currency',
      currency: currency ?? 'EUR',
      minimumFractionDigits: digits,
      maximumFractionDigits: 2,
    });
  } catch {
    return `${amount.toFixed(digits)} ${currency ?? ''}`.trim();
  }
}

function count(value: number): string {
  return value.toLocaleString(currentLocale(), { maximumFractionDigits: 2 });
}

/** "9 December 2026": a day a subscription renews or ends. */
export function formatDay(iso: string | null | undefined): string {
  if (!iso) {
    return '–';
  }
  return new Date(iso).toLocaleDateString(longDateLocale(), {
    day: 'numeric',
    month: 'long',
    year: 'numeric',
  });
}

/** What the plan costs per `interval`; null where it is not sold that way. */
export function planAmount(plan: Plan, interval: BillingInterval = 'month'): number | null {
  return interval === 'year' ? (plan.price_per_year ?? null) : plan.price;
}

/** "€14 per month" or "€140 per year" ("" for a plan without that price). */
export function priceFor(
  plan: Plan,
  interval: BillingInterval = 'month',
  currency: string | null = plan.currency
): string {
  const amount = planAmount(plan, interval);
  if (!amount) {
    return '';
  }
  return i18n.t(interval === 'year' ? 'plans.short.perYear' : 'plans.short.perMonth', {
    price: formatPrice(amount, plan.currency ?? currency),
  });
}

/** One thing a plan includes: the line, and what it is about (to compare two plans). */
export interface PlanLine {
  key: 'compute' | 'storage' | 'perRun' | 'concurrent' | 'priority' | 'overage';
  text: string;
  /** The amount alone ("10", "3 GB", "1 h"), for "(now …)" next to another plan's line. */
  amount: string | null;
}

function perRun(plan: Plan, where: 'features' | 'short'): { text: string; amount: string } | null {
  const minutes = plan.cpu_minutes_per_run;
  if (minutes === null) {
    return null;
  }
  if (minutes >= 60 && minutes % 60 === 0) {
    const value = count(minutes / 60);
    return {
      text: i18n.t(`plans.${where}.perRunHours`, { count: minutes / 60, value }),
      amount: i18n.t('plans.short.hours', { value }),
    };
  }
  const value = count(minutes);
  return { text: i18n.t(`plans.${where}.perRunMinutes`, { value }), amount: `${value} min` };
}

/**
 * What a plan includes, one line each. `overage`: the server bills compute beyond the
 * allowance; `priority`: the plan's trainings start before those of cheaper plans.
 */
export function planLines(
  plan: Plan,
  { overage = false, priority = false }: { overage?: boolean; priority?: boolean } = {}
): PlanLine[] {
  const lines: PlanLine[] = [];
  const hours = plan.cpu_hours_per_month;
  lines.push({
    key: 'compute',
    text:
      hours !== null
        ? i18n.t('plans.features.compute', { count: hours, value: count(hours) })
        : i18n.t('plans.features.computeUnlimited'),
    amount: hours !== null ? count(hours) : null,
  });
  lines.push({
    key: 'storage',
    text:
      plan.storage_gb !== null
        ? i18n.t('plans.features.storage', { value: count(plan.storage_gb) })
        : i18n.t('plans.features.storageUnlimited'),
    amount: plan.storage_gb !== null ? `${count(plan.storage_gb)} GB` : null,
  });
  const run = perRun(plan, 'features');
  if (run) {
    lines.push({ key: 'perRun', ...run });
  }
  if (plan.concurrent_trainings !== null) {
    lines.push({
      key: 'concurrent',
      text: i18n.t('plans.features.concurrent', { count: plan.concurrent_trainings }),
      amount: String(plan.concurrent_trainings),
    });
  }
  if (priority) {
    lines.push({ key: 'priority', text: i18n.t('plans.features.priority'), amount: null });
  }
  if (overage && plan.overage_price_per_cpu_hour !== null) {
    lines.push({
      key: 'overage',
      text: i18n.t('plans.features.overage', {
        price: formatPrice(plan.overage_price_per_cpu_hour, plan.currency),
      }),
      amount: null,
    });
  }
  return lines;
}

/** The lines of `planLines` as text. */
export function planFeatures(plan: Plan, overage: boolean): string[] {
  return planLines(plan, { overage }).map((line) => line.text);
}

/** "A, B and C" in the language of the app. */
export function listText(parts: string[]): string {
  try {
    return new Intl.ListFormat(currentLocale(), { style: 'long', type: 'conjunction' }).format(
      parts
    );
  } catch {
    return parts.join(', ');
  }
}

/**
 * "10 CPU hours per month, 3 GB of storage and up to 1 CPU hour per training": what a plan
 * includes as part of a sentence.
 */
export function planIncludes(plan: Plan): string {
  const hours = plan.cpu_hours_per_month;
  const parts = [
    hours !== null
      ? i18n.t('plans.short.compute', { count: hours, value: count(hours) })
      : i18n.t('plans.short.computeUnlimited'),
    plan.storage_gb !== null
      ? i18n.t('plans.short.storage', { value: count(plan.storage_gb) })
      : i18n.t('plans.short.storageUnlimited'),
  ];
  const run = perRun(plan, 'short');
  if (run) {
    parts.push(run.text);
  }
  if (plan.concurrent_trainings !== null && plan.concurrent_trainings > 1) {
    parts.push(i18n.t('plans.short.concurrent', { count: plan.concurrent_trainings }));
  }
  return listText(parts);
}

/** "0.5 CPU h per month · 1 GB · 15 CPU min per training · 1 training at a time". */
export function planLimits(plan: Plan): string {
  const hours = plan.cpu_hours_per_month;
  const parts = [
    hours !== null
      ? i18n.t('plans.short.compute', { count: hours, value: count(hours) })
      : i18n.t('plans.short.computeUnlimited'),
    plan.storage_gb !== null
      ? i18n.t('plans.short.storage', { value: count(plan.storage_gb) })
      : i18n.t('plans.short.storageUnlimited'),
  ];
  const run = perRun(plan, 'short');
  if (run) {
    parts.push(run.text);
  }
  if (plan.concurrent_trainings !== null) {
    parts.push(i18n.t('plans.short.concurrent', { count: plan.concurrent_trainings }));
  }
  return parts.join(' · ');
}
