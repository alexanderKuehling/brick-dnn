/** Histogram bins: edges on round numbers, counting, and moving counts onto other bins. */
import type { HistogramData } from '@/api/types';
import { niceStep, stepDecimals } from '@/components/chartScale';
import { quantileSorted, sortedFinite } from './stats';

/**
 * Bin of `value` for the given edges: bin i covers edges[i] ≤ value < edges[i + 1], the last
 * bin includes its end (like numpy and the backend). -1 outside the edges.
 */
export function binIndex(edges: readonly number[], value: number): number {
  const last = edges.length - 1;
  if (last < 1 || !Number.isFinite(value) || value < edges[0] || value > edges[last]) {
    return -1;
  }
  if (value === edges[last]) {
    return last - 1;
  }
  let lo = 0;
  let hi = last;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (edges[mid] <= value) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return lo;
}

/** Count the finite numbers of `values` per bin (values outside the edges are ignored). */
export function countBins(values: Iterable<unknown>, edges: readonly number[]): number[] {
  const counts = new Array<number>(Math.max(0, edges.length - 1)).fill(0);
  for (const value of values) {
    if (typeof value === 'number') {
      const index = binIndex(edges, value);
      if (index >= 0) {
        counts[index] += 1;
      }
    }
  }
  return counts;
}

/**
 * Number of bins by the Freedman–Diaconis rule (bin width 2·IQR/∛n), Sturges' rule when the
 * IQR is 0, clamped to [minBins, maxBins].
 */
export function binCount(sorted: readonly number[], minBins = 5, maxBins = 50): number {
  const n = sorted.length;
  if (n < 2) {
    return 1;
  }
  const range = sorted[n - 1] - sorted[0];
  if (range <= 0) {
    return 1;
  }
  const iqr = quantileSorted(sorted, 0.75) - quantileSorted(sorted, 0.25);
  const raw = iqr > 0 ? range / ((2 * iqr) / Math.cbrt(n)) : Math.log2(n) + 1;
  return Math.min(maxBins, Math.max(minBins, Math.ceil(raw)));
}

export interface EdgeOptions {
  /** Whole numbers: bins of whole-number width, one per value when there are few. */
  integer?: boolean;
  minBins?: number;
  maxBins?: number;
}

/** Edges on multiples of a round step from `lo` to (at least) `hi`. */
export function roundEdges(lo: number, hi: number, bins: number, integer = false): number[] {
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) {
    return [];
  }
  if (hi <= lo) {
    const pad = integer ? 0.5 : Math.abs(lo) * 0.05 || 0.5;
    return [lo - pad, lo + pad];
  }
  let step = niceStep((hi - lo) / Math.max(1, bins));
  if (integer) {
    step = Math.max(1, Math.round(step));
  }
  const decimals = stepDecimals(step);
  const round = (value: number) => Number(value.toFixed(decimals));
  const first = Math.floor(lo / step + 1e-9);
  const edges = [round(first * step)];
  for (let k = first + 1; edges[edges.length - 1] < hi && edges.length <= 1000; k += 1) {
    edges.push(round(k * step));
  }
  return edges;
}

/** Edges for the values: one per whole number for small integer ranges, else round bins. */
export function autoEdges(values: Iterable<unknown>, options: EdgeOptions = {}): number[] {
  const { integer = false, minBins = 5, maxBins = 50 } = options;
  const sorted = sortedFinite(values);
  if (sorted.length === 0) {
    return [];
  }
  const lo = sorted[0];
  const hi = sorted[sorted.length - 1];
  if (integer && hi - lo + 1 <= maxBins) {
    return Array.from({ length: hi - lo + 2 }, (_, i) => lo - 0.5 + i);
  }
  return roundEdges(lo, hi, binCount(sorted, minBins, maxBins), integer);
}

/** A histogram of the finite numbers in `values`. */
export function histogram(values: readonly unknown[], options: EdgeOptions = {}): HistogramData {
  const edges = autoEdges(values, options);
  return { edges, counts: countBins(values, edges) };
}

/**
 * The counts of `hist` moved onto other `edges`, assuming the values of each bin are spread
 * evenly over it. Exact when the edges coincide; an estimate otherwise.
 */
export function rebin(hist: HistogramData, edges: readonly number[]): number[] {
  const counts = new Array<number>(Math.max(0, edges.length - 1)).fill(0);
  for (let i = 0; i < hist.counts.length; i += 1) {
    const count = hist.counts[i];
    const a = hist.edges[i];
    const b = hist.edges[i + 1];
    if (!count || !Number.isFinite(a) || !Number.isFinite(b)) {
      continue;
    }
    if (b <= a) {
      const index = binIndex(edges, a);
      if (index >= 0) {
        counts[index] += count;
      }
      continue;
    }
    for (let j = 0; j < counts.length; j += 1) {
      const overlap = Math.min(b, edges[j + 1]) - Math.max(a, edges[j]);
      if (overlap > 0) {
        counts[j] += (count * overlap) / (b - a);
      }
    }
  }
  return counts;
}

export function sameEdges(a: readonly number[], b: readonly number[]): boolean {
  return a.length === b.length && a.every((edge, i) => edge === b[i]);
}

/** Shared edges for several histograms: their own if all agree, else round bins over all. */
export function commonEdges(hists: HistogramData[], bins = 30): number[] {
  const usable = hists.filter((h) => h.edges.length > 1);
  if (usable.length === 0) {
    return [];
  }
  if (usable.every((h) => sameEdges(h.edges, usable[0].edges))) {
    return usable[0].edges;
  }
  const lo = Math.min(...usable.map((h) => h.edges[0]));
  const hi = Math.max(...usable.map((h) => h.edges[h.edges.length - 1]));
  return roundEdges(lo, hi, bins);
}
