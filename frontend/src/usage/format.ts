/** Formatting of compute usage (CPU minutes), money and budget dates. */

/** 0.0123 → "0.01 CPU min", 2.5 → "2.5 CPU min", 312.4 → "312 CPU min". */
export function formatCpu(minutes: number | null | undefined, unit = 'CPU min'): string {
  if (typeof minutes !== 'number' || !Number.isFinite(minutes)) {
    return '–';
  }
  const digits = minutes >= 100 ? 0 : minutes >= 10 ? 1 : 2;
  const value = Number(minutes.toFixed(digits)).toLocaleString('en-US', {
    maximumFractionDigits: digits,
  });
  return unit ? `${value} ${unit}` : value;
}

/** 0.27, "EUR" → "€0.27" (an amount in the configured currency). */
export function formatMoney(amount: number | null | undefined, currency: string | null): string {
  if (typeof amount !== 'number' || !Number.isFinite(amount) || !currency) {
    return '';
  }
  try {
    return amount.toLocaleString('en-US', {
      style: 'currency',
      currency,
      minimumFractionDigits: 2,
      maximumFractionDigits: amount > 0 && amount < 0.01 ? 4 : 2,
    });
  } catch {
    return `${amount.toFixed(2)} ${currency}`;
  }
}

/** When a budget resets: "1 December 2026". Shown in UTC (budgets follow UTC months). */
export function formatResetDate(iso: string): string {
  return new Date(iso).toLocaleDateString('en-GB', {
    day: 'numeric',
    month: 'long',
    year: 'numeric',
    timeZone: 'UTC',
  });
}

/** Share of a limit in percent for a progress bar (0–100). */
export function usagePercent(used: number, limit: number | null | undefined): number {
  if (typeof limit !== 'number' || limit <= 0) {
    return used > 0 ? 100 : 0;
  }
  return Math.min(100, Math.max(0, (used / limit) * 100));
}

/** The colour of a usage bar: green, yellow from 75 %, red once used up. */
export function usageColor(percent: number): string {
  return percent >= 100 ? 'red' : percent >= 75 ? 'yellow' : 'brick';
}
