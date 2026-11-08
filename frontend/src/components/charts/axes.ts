/** Axis helpers on top of `chartScale.ts`: ticks inside a fixed domain, log and time axes. */
import { formatTick, integerTicks, niceScale } from '@/components/chartScale';
import { formatDate, timeUnitFor, type TimeUnit } from './format';

export interface Axis {
  /** Domain of the axis (data units). */
  domain: [number, number];
  ticks: number[];
  format: (value: number) => string;
}

/** Round ticks inside [lo, hi] without widening the domain (histograms start at their edge). */
export function innerTicks(lo: number, hi: number, count: number, integer = false): Axis {
  if (integer) {
    const ticks = integerTicks(Math.ceil(lo), Math.floor(hi), count);
    const step = ticks.length > 1 ? ticks[1] - ticks[0] : 1;
    return { domain: [lo, hi], ticks, format: (v) => formatTick(v, step) };
  }
  // niceScale rounds the step up, which can leave a single tick inside a fixed domain; try
  // finer steps and keep the one closest to `count` ticks.
  const eps = (hi - lo) * 1e-9;
  let best = { ticks: [] as number[], step: 1 };
  for (const k of [1, 2, 2.5, 4, 5]) {
    const scale = niceScale(lo, hi, Math.max(2, Math.round(count * k)));
    const ticks = scale.ticks.filter((tick) => tick >= lo - eps && tick <= hi + eps);
    if (ticks.length > count + 1) {
      break;
    }
    if (Math.abs(ticks.length - count) < Math.abs(best.ticks.length - count)) {
      best = { ticks, step: scale.step };
    }
  }
  return { domain: [lo, hi], ticks: best.ticks, format: (v) => formatTick(v, best.step) };
}

/** A linear value axis from round number to round number (e.g. counts from 0). */
export function valueAxis(
  lo: number,
  hi: number,
  count: number,
  bounds?: [number, number]
): Axis & { step: number } {
  const scale = niceScale(lo, hi, count, bounds);
  return {
    domain: [scale.min, scale.max],
    ticks: scale.ticks,
    step: scale.step,
    format: (v) => formatTick(v, scale.step),
  };
}

/** Maps a positive domain logarithmically onto a pixel range. */
export function logScale(
  [d0, d1]: [number, number],
  [r0, r1]: [number, number]
): (value: number) => number {
  const l0 = Math.log10(d0);
  const l1 = Math.log10(d1);
  if (!(l1 > l0)) {
    return () => (r0 + r1) / 2;
  }
  return (value) => r0 + ((Math.log10(Math.max(value, d0)) - l0) / (l1 - l0)) * (r1 - r0);
}

/**
 * A log axis for positive values from `minPositive` to `max`: powers of ten, with 2 and 5 in
 * between when the range spans at most two decades. The domain starts half a step below the
 * smallest value so that it still shows as a bar.
 */
export function logAxis(minPositive: number, max: number, percent = false): Axis {
  const top = Math.max(max, minPositive);
  const lowDecade = Math.floor(Math.log10(minPositive));
  const highDecade = Math.ceil(Math.log10(top) - 1e-9);
  const d0 = 10 ** lowDecade / 2;
  const d1 = Math.max(10 ** highDecade, d0 * 10);
  const ticks: number[] = [];
  const decades = highDecade - lowDecade;
  for (let e = lowDecade; e <= Math.max(highDecade, lowDecade + 1); e += 1) {
    for (const k of decades <= 2 ? [1, 2, 5] : [1]) {
      const tick = k * 10 ** e;
      if (tick >= d0 && tick <= d1 * (1 + 1e-9)) {
        ticks.push(Number(tick.toPrecision(12)));
      }
    }
  }
  const format = (v: number) => {
    if (percent) {
      return `${Number((v * 100).toPrecision(3))}%`;
    }
    return formatTick(v, 10 ** Math.floor(Math.log10(v)));
  };
  return { domain: [d0, d1], ticks, format };
}

const SECOND = 1000;
const MINUTE = 60 * SECOND;
const HOUR = 60 * MINUTE;
const DAY = 24 * HOUR;

/** Fixed-length tick intervals; months and years are added by calendar. */
const FIXED_STEPS = [
  SECOND,
  5 * SECOND,
  15 * SECOND,
  30 * SECOND,
  MINUTE,
  5 * MINUTE,
  15 * MINUTE,
  30 * MINUTE,
  HOUR,
  3 * HOUR,
  6 * HOUR,
  12 * HOUR,
  DAY,
  2 * DAY,
  7 * DAY,
  14 * DAY,
];
const MONTH_STEPS = [1, 2, 3, 6];
const YEAR_STEPS = [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000];

/** Calendar-aligned ticks (UTC) for timestamps in ms, about `count` of them. */
export function timeTicks(lo: number, hi: number, count = 5): Axis & { unit: TimeUnit } {
  const span = Math.max(hi - lo, 1);
  const target = span / Math.max(1, count);
  const make = (ticks: number[], unit: TimeUnit) => ({
    domain: [lo, hi] as [number, number],
    ticks,
    unit,
    format: (v: number) => formatDate(v, unit),
  });
  for (const step of FIXED_STEPS) {
    if (step >= target) {
      const ticks: number[] = [];
      for (let t = Math.ceil(lo / step) * step; t <= hi; t += step) {
        ticks.push(t);
      }
      return make(ticks, timeUnitFor(step));
    }
  }
  for (const months of MONTH_STEPS) {
    if (months * 30.44 * DAY >= target) {
      const start = new Date(lo);
      let y = start.getUTCFullYear();
      let m = start.getUTCMonth();
      if (Date.UTC(y, m, 1) < lo) {
        m += 1;
      }
      m = Math.ceil(m / months) * months;
      const ticks: number[] = [];
      for (let t = Date.UTC(y, m, 1); t <= hi; t = Date.UTC(y, m, 1)) {
        ticks.push(t);
        m += months;
        if (m >= 12) {
          y += Math.floor(m / 12);
          m %= 12;
        }
      }
      return make(ticks, 'month');
    }
  }
  const years = YEAR_STEPS.find((k) => k * 365.25 * DAY >= target) ?? 1000;
  const first = new Date(lo).getUTCFullYear();
  const ticks: number[] = [];
  for (let y = Math.ceil(first / years) * years; Date.UTC(y, 0, 1) <= hi; y += years) {
    if (Date.UTC(y, 0, 1) >= lo) {
      ticks.push(Date.UTC(y, 0, 1));
    }
  }
  return make(ticks, 'year');
}
