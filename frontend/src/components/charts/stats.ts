/**
 * Descriptive statistics for sample data, computed like pandas (sample standard deviation,
 * linear quantiles, bias-corrected skewness and excess kurtosis), plus box plot summaries,
 * correlations and least-squares fits.
 */
import type { NumericStats } from '@/api/types';

/** The finite numbers of `values`, sorted ascending. */
export function sortedFinite(values: Iterable<unknown>): number[] {
  const out: number[] = [];
  for (const value of values) {
    if (typeof value === 'number' && Number.isFinite(value)) {
      out.push(value);
    }
  }
  return out.sort((a, b) => a - b);
}

/** Quantile of sorted values with linear interpolation (pandas/numpy default). */
export function quantileSorted(sorted: readonly number[], q: number): number {
  const n = sorted.length;
  if (n === 0) {
    return NaN;
  }
  const position = (n - 1) * Math.min(1, Math.max(0, q));
  const lo = Math.floor(position);
  const hi = Math.min(n - 1, lo + 1);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (position - lo);
}

export function mean(values: readonly number[]): number {
  let sum = 0;
  for (const value of values) {
    sum += value;
  }
  return values.length ? sum / values.length : NaN;
}

const orNull = (value: number) => (Number.isFinite(value) ? value : null);

/** NumericStats of the finite numbers in `values` (null where undefined). */
export function describe(values: Iterable<unknown>): NumericStats {
  const sorted = sortedFinite(values);
  const n = sorted.length;
  const m = mean(sorted);
  let m2 = 0;
  let m3 = 0;
  let m4 = 0;
  let zeros = 0;
  let negative = 0;
  for (const value of sorted) {
    const d = value - m;
    m2 += d * d;
    m3 += d * d * d;
    m4 += d * d * d * d;
    zeros += value === 0 ? 1 : 0;
    negative += value < 0 ? 1 : 0;
  }
  const variance = n > 1 ? m2 / (n - 1) : NaN;
  const std = Math.sqrt(variance);
  const constant = n > 0 && sorted[0] === sorted[n - 1];
  // pandas: adjusted Fisher–Pearson skewness and bias-corrected excess kurtosis.
  const skew = n >= 3 && !constant ? ((n * Math.sqrt(n - 1)) / (n - 2)) * (m3 / m2 ** 1.5) : NaN;
  const kurtosis =
    n >= 4 && !constant
      ? ((n + 1) * n * (n - 1) * m4) / ((n - 2) * (n - 3) * m2 * m2) -
        (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))
      : NaN;
  const q = (p: number) => orNull(quantileSorted(sorted, p));
  const q25 = quantileSorted(sorted, 0.25);
  const q75 = quantileSorted(sorted, 0.75);
  const fence = 1.5 * (q75 - q25);
  let iqrOutliers = 0;
  let zOutliers = 0;
  for (const value of sorted) {
    iqrOutliers += value < q25 - fence || value > q75 + fence ? 1 : 0;
    zOutliers += std > 0 && Math.abs(value - m) / std > 3 ? 1 : 0;
  }
  return {
    mean: orNull(m),
    std: orNull(std),
    min: n ? sorted[0] : null,
    max: n ? sorted[n - 1] : null,
    q01: q(0.01),
    q05: q(0.05),
    q25: q(0.25),
    median: q(0.5),
    q75: q(0.75),
    q95: q(0.95),
    q99: q(0.99),
    skew: orNull(skew),
    kurtosis: orNull(kurtosis),
    zeros,
    negative,
    iqr_outliers: iqrOutliers,
    z_outliers: zOutliers,
  };
}

/** What a box plot draws for one group. */
export interface BoxSummary {
  min: number;
  q1: number;
  median: number;
  q3: number;
  max: number;
  /** Whisker ends: the most extreme values within 1.5 × IQR of the box. */
  low: number;
  high: number;
  mean: number | null;
  /** Values in the group (null if unknown). */
  n: number | null;
  /** Values beyond the whiskers to draw (at most MAX_OUTLIER_DOTS). */
  outliers: number[];
  /** All values beyond the whiskers (null if unknown). */
  outlierCount: number | null;
  /** False if the whiskers are only estimated (from quantiles, not the values). */
  exact: boolean;
}

export const MAX_OUTLIER_DOTS = 300;

/**
 * Box plot from profile statistics. Without the values the whiskers end at the 1.5 × IQR
 * fences (or at min/max if those are closer), so they are marked as estimated.
 */
export function boxFromStats(stats: NumericStats, n: number | null = null): BoxSummary | null {
  const { min, q25, median, q75, max } = stats;
  if (min === null || q25 === null || median === null || q75 === null || max === null) {
    return null;
  }
  const fence = 1.5 * (q75 - q25);
  return {
    min,
    q1: q25,
    median,
    q3: q75,
    max,
    low: Math.max(min, q25 - fence),
    high: Math.min(max, q75 + fence),
    mean: stats.mean,
    n,
    outliers: [],
    outlierCount: stats.iqr_outliers,
    exact: min >= q25 - fence && max <= q75 + fence,
  };
}

/** Box plot of the finite numbers in `values` (Tukey whiskers, exact). */
export function boxFromValues(values: Iterable<unknown>): BoxSummary | null {
  const sorted = sortedFinite(values);
  const n = sorted.length;
  if (n === 0) {
    return null;
  }
  const q1 = quantileSorted(sorted, 0.25);
  const q3 = quantileSorted(sorted, 0.75);
  const fence = 1.5 * (q3 - q1);
  let low = sorted[0];
  let high = sorted[n - 1];
  const outliers: number[] = [];
  let outlierCount = 0;
  for (const value of sorted) {
    if (value < q1 - fence || value > q3 + fence) {
      outlierCount += 1;
      outliers.push(value);
    }
  }
  for (const value of sorted) {
    if (value >= q1 - fence) {
      low = value;
      break;
    }
  }
  for (let i = n - 1; i >= 0; i -= 1) {
    if (sorted[i] <= q3 + fence) {
      high = sorted[i];
      break;
    }
  }
  // Keep the most extreme outliers when there are too many to draw.
  const shown =
    outliers.length > MAX_OUTLIER_DOTS
      ? [
          ...outliers.slice(0, MAX_OUTLIER_DOTS / 2),
          ...outliers.slice(outliers.length - MAX_OUTLIER_DOTS / 2),
        ]
      : outliers;
  return {
    min: sorted[0],
    q1,
    median: quantileSorted(sorted, 0.5),
    q3,
    max: sorted[n - 1],
    low,
    high,
    mean: mean(sorted),
    n,
    outliers: shown,
    outlierCount,
    exact: true,
  };
}

/** A least-squares line y = intercept + slope · x with Pearson's r. */
export interface LinearFit {
  slope: number;
  intercept: number;
  r: number;
  r2: number;
  n: number;
}

interface Moments {
  n: number;
  mx: number;
  my: number;
  sxx: number;
  syy: number;
  sxy: number;
}

/** Means and centred sums of squares of the pairs where both values are finite. */
function moments(x: ArrayLike<number>, y: ArrayLike<number>): Moments {
  const length = Math.min(x.length, y.length);
  let n = 0;
  let mx = 0;
  let my = 0;
  for (let i = 0; i < length; i += 1) {
    if (Number.isFinite(x[i]) && Number.isFinite(y[i])) {
      n += 1;
      mx += (x[i] - mx) / n;
      my += (y[i] - my) / n;
    }
  }
  // Centred sums (two passes), so timestamps and other large numbers keep their precision.
  let sxx = 0;
  let syy = 0;
  let sxy = 0;
  for (let i = 0; i < length; i += 1) {
    if (Number.isFinite(x[i]) && Number.isFinite(y[i])) {
      const dx = x[i] - mx;
      const dy = y[i] - my;
      sxx += dx * dx;
      syy += dy * dy;
      sxy += dx * dy;
    }
  }
  return { n, mx, my, sxx, syy, sxy };
}

/** Ordinary least squares of y on x (pairs with a non-finite value are skipped). */
export function ols(x: ArrayLike<number>, y: ArrayLike<number>): LinearFit | null {
  const { n, mx, my, sxx, syy, sxy } = moments(x, y);
  if (n < 2 || sxx === 0) {
    return null;
  }
  const slope = sxy / sxx;
  const r = syy === 0 ? 0 : sxy / Math.sqrt(sxx * syy);
  return { slope, intercept: my - slope * mx, r, r2: r * r, n };
}

/** Pearson correlation (null for fewer than 2 pairs or a constant variable). */
export function pearson(x: ArrayLike<number>, y: ArrayLike<number>): number | null {
  const { n, sxx, syy, sxy } = moments(x, y);
  if (n < 2 || sxx === 0 || syy === 0) {
    return null;
  }
  return Math.max(-1, Math.min(1, sxy / Math.sqrt(sxx * syy)));
}

/** Ranks 1 … n, ties get their average rank. */
export function ranks(values: readonly number[]): number[] {
  const order = values.map((_, i) => i).sort((a, b) => values[a] - values[b]);
  const out = new Array<number>(values.length);
  for (let i = 0; i < order.length;) {
    let j = i;
    while (j + 1 < order.length && values[order[j + 1]] === values[order[i]]) {
      j += 1;
    }
    const rank = (i + j) / 2 + 1;
    for (let k = i; k <= j; k += 1) {
      out[order[k]] = rank;
    }
    i = j + 1;
  }
  return out;
}

/** Spearman rank correlation of the pairs where both values are finite. */
export function spearman(x: ArrayLike<number>, y: ArrayLike<number>): number | null {
  const xs: number[] = [];
  const ys: number[] = [];
  for (let i = 0; i < Math.min(x.length, y.length); i += 1) {
    if (Number.isFinite(x[i]) && Number.isFinite(y[i])) {
      xs.push(x[i]);
      ys.push(y[i]);
    }
  }
  return pearson(ranks(xs), ranks(ys));
}
