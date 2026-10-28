/** Scale and tick helpers for the small SVG charts (no chart library needed). */

/** Round a raw step up to 1, 2 or 5 × 10^k. */
export function niceStep(rawStep: number): number {
  if (!Number.isFinite(rawStep) || rawStep <= 0) {
    return 1;
  }
  const base = 10 ** Math.floor(Math.log10(rawStep));
  const fraction = rawStep / base;
  const nice = fraction <= 1 ? 1 : fraction <= 2 ? 2 : fraction <= 5 ? 5 : 10;
  return nice * base;
}

/** Number of decimals needed to print multiples of `step` exactly (at most 100). */
export function stepDecimals(step: number): number {
  if (!Number.isFinite(step) || step <= 0) {
    return 0;
  }
  return Math.min(100, Math.max(0, -Math.floor(Math.log10(step) + 1e-9)));
}

export interface NiceScale {
  min: number;
  max: number;
  step: number;
  ticks: number[];
}

/** At most this many ticks, whatever the input. */
const MAX_TICKS = 50;

/**
 * `value / step` rounded to a whole number if it is one up to floating point noise. The
 * tolerance grows with the quotient (noise is relative), so values far from zero keep their
 * first tick, but stays far below 0.5, so floor and ceil still cover the data.
 */
function snap(quotient: number): number {
  const nearest = Math.round(quotient);
  return Math.abs(quotient - nearest) <= 1e-9 + Math.abs(nearest) * 1e-12 ? nearest : quotient;
}

/**
 * A domain covering [lo, hi] that starts and ends on round numbers, with about `count`
 * ticks. `bounds` clamps the result (e.g. [0, 1] for accuracies, [0, Infinity] for losses).
 */
export function niceScale(
  lo: number,
  hi: number,
  count = 5,
  bounds: [number, number] = [-Infinity, Infinity]
): NiceScale {
  let low = Number.isFinite(lo) ? lo : 0;
  let high = Number.isFinite(hi) ? hi : low;
  if (low > high) {
    [low, high] = [high, low];
  }
  // Values that differ only by rounding noise count as one value; a step that small could
  // not be represented between them.
  if (high - low <= Math.max(Math.abs(low), Math.abs(high)) * 1e-12) {
    // A single value: give it some room so it sits in the middle of the chart.
    const pad = Math.abs(high) * 0.1 || 1;
    low = high - pad;
    high += pad;
  }
  const step = niceStep((high - low) / Math.max(1, count - 1));
  const decimals = stepDecimals(step);
  const round = (value: number) => Number(value.toFixed(decimals));
  let min = round(Math.floor(snap(low / step)) * step);
  let max = round(Math.ceil(snap(high / step)) * step);
  min = Math.max(min, bounds[0]);
  max = Math.min(max, bounds[1]);
  if (max <= min) {
    max = min + step;
  }
  const first = Math.ceil(snap(min / step));
  const last = Math.floor(snap(max / step));
  const ticks: number[] = [];
  for (let k = 0; k <= Math.min(last - first, MAX_TICKS - 1); k += 1) {
    ticks.push(round((first + k) * step));
  }
  return { min, max, step, ticks };
}

/** Whole-number ticks (e.g. epochs) between min and max, about `count` of them. */
export function integerTicks(min: number, max: number, count = 6): number[] {
  if (!Number.isFinite(min) || !Number.isFinite(max) || max < min) {
    return [];
  }
  const step = Math.max(1, niceStep((max - min) / Math.max(1, count - 1)));
  const ticks: number[] = [];
  for (let value = Math.ceil(min / step) * step; value <= max; value += step) {
    ticks.push(value);
  }
  return ticks;
}

/** Map a domain linearly onto a pixel range. A zero-width domain maps to the middle. */
export function linearScale(
  [d0, d1]: [number, number],
  [r0, r1]: [number, number]
): (value: number) => number {
  if (d1 === d0) {
    return () => (r0 + r1) / 2;
  }
  const k = (r1 - r0) / (d1 - d0);
  return (value) => r0 + (value - d0) * k;
}

const compact = new Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 1 });

/**
 * Axis label for a tick: as many decimals as the step needs, compact from 1,000 up (1K, 2.5K)
 * as long as one decimal of the compact unit is fine enough for the step (1,210 stays 1,210).
 */
export function formatTick(value: number, step: number): string {
  const size = Math.abs(value);
  if (size >= 1000) {
    const unit = 1000 ** Math.floor(Math.log10(size) / 3);
    if (step >= unit / 10) {
      return compact.format(value);
    }
    return value.toLocaleString('en-US', { maximumFractionDigits: stepDecimals(step) });
  }
  const text = value.toFixed(stepDecimals(step));
  return text === '-0' ? '0' : text;
}

/** Axis label for a fraction shown as percent (0.95 → "95%"). */
export function formatPercentTick(value: number, step: number): string {
  return `${(value * 100).toFixed(stepDecimals(step * 100))}%`;
}

/** Smallest and largest finite value, or null if there is none. */
export function extent(values: Iterable<number | null | undefined>): [number, number] | null {
  let min = Infinity;
  let max = -Infinity;
  for (const value of values) {
    if (typeof value === 'number' && Number.isFinite(value)) {
      min = Math.min(min, value);
      max = Math.max(max, value);
    }
  }
  return min <= max ? [min, max] : null;
}

/** Index of the entry of the sorted array `xs` closest to `x` (-1 for an empty array). */
export function nearestIndex(xs: number[], x: number): number {
  if (xs.length === 0) {
    return -1;
  }
  let lo = 0;
  let hi = xs.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] <= x) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return Math.abs(xs[hi] - x) < Math.abs(xs[lo] - x) ? hi : lo;
}

type Point = [number, number];

/** SVG path for a line through the points; a null value breaks the line into segments. */
export function linePath(points: (Point | null)[]): string {
  let path = '';
  let pen = false;
  for (const point of points) {
    if (!point) {
      pen = false;
      continue;
    }
    path += `${pen ? 'L' : 'M'}${point[0].toFixed(1)},${point[1].toFixed(1)}`;
    pen = true;
  }
  return path;
}
