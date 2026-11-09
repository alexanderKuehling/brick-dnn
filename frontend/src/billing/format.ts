/** What plans include and cost, in words. */
import type { Plan } from '@/api';

/** 14, "EUR" → "€14", 4.5 → "€4.50". */
export function formatPrice(amount: number, currency: string | null): string {
  const digits = Number.isInteger(amount) ? 0 : 2;
  try {
    return amount.toLocaleString('en-US', {
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
  return value.toLocaleString('en-US', { maximumFractionDigits: 2 });
}

/** "9 December 2026": a day a subscription renews or ends. */
export function formatDay(iso: string | null | undefined): string {
  if (!iso) {
    return '–';
  }
  return new Date(iso).toLocaleDateString('en-GB', {
    day: 'numeric',
    month: 'long',
    year: 'numeric',
  });
}

/** What a plan includes, one line each; `overage`: the server bills compute beyond it. */
export function planFeatures(plan: Plan, overage: boolean): string[] {
  const lines: string[] = [];
  lines.push(
    plan.cpu_hours_per_month !== null
      ? `${count(plan.cpu_hours_per_month)} CPU ${plan.cpu_hours_per_month === 1 ? 'hour' : 'hours'} per month for trainings and predictions`
      : 'Unlimited compute for trainings and predictions'
  );
  lines.push(
    plan.storage_gb !== null
      ? `${count(plan.storage_gb)} GB storage for data and trained models`
      : 'Unlimited storage'
  );
  if (plan.cpu_minutes_per_run !== null) {
    const minutes = plan.cpu_minutes_per_run;
    const length =
      minutes >= 60 && minutes % 60 === 0
        ? `${count(minutes / 60)} CPU ${minutes === 60 ? 'hour' : 'hours'}`
        : `${count(minutes)} CPU minutes`;
    lines.push(`Up to ${length} per training`);
  }
  if (plan.concurrent_trainings !== null) {
    lines.push(
      plan.concurrent_trainings === 1
        ? 'One training at a time'
        : `${plan.concurrent_trainings} trainings at the same time`
    );
  }
  if (overage && plan.overage_price_per_cpu_hour !== null) {
    lines.push(
      `More compute if you switch it on: ${formatPrice(plan.overage_price_per_cpu_hour, plan.currency)} per CPU hour`
    );
  }
  return lines;
}
