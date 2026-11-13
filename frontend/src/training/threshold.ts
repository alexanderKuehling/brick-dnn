/**
 * Binary classification: reading the threshold curves of a run's evaluation (see
 * `ThresholdCurve`). A row is predicted positive when its probability of the positive class is
 * at least the threshold; every split of a run has its curve at the same thresholds.
 */
import type { ThresholdCurve, ThresholdPoint } from '@/api';

/** The threshold a run predicts with until its owner picks another one. */
export const DEFAULT_THRESHOLD = 0.5;

/** The index of the curve threshold closest to `value` (the first of two equally close). */
export function nearestIndex(thresholds: number[], value: number): number {
  let best = 0;
  thresholds.forEach((threshold, i) => {
    if (Math.abs(threshold - value) < Math.abs(thresholds[best] - value)) {
      best = i;
    }
  });
  return best;
}

function ratio(part: number, whole: number): number | null {
  return whole > 0 ? part / whole : null;
}

/** Precision, recall, F1, the false-positive rate and the counts at point `index`. */
export function pointAt(curve: ThresholdCurve, index: number): ThresholdPoint {
  const { points } = curve;
  const tp = points.tp[index] ?? 0;
  const fp = points.fp[index] ?? 0;
  const tn = points.tn[index] ?? 0;
  const fn = points.fn[index] ?? 0;
  return {
    threshold: points.threshold[index],
    precision: ratio(tp, tp + fp),
    recall: ratio(tp, tp + fn),
    f1: ratio(2 * tp, 2 * tp + fp + fn),
    fpr: ratio(fp, fp + tn),
    tp,
    fp,
    tn,
    fn,
  };
}

export interface CurveLine {
  x: number[];
  y: number[];
  /** The curve threshold of every point (for the table and the tooltip). */
  threshold: number[];
}

/**
 * The precision-recall curve (x recall, y precision) from the highest threshold to the lowest;
 * thresholds that predict nothing positive have no precision and are left out.
 */
export function prLine(curve: ThresholdCurve): CurveLine {
  const line: CurveLine = { x: [], y: [], threshold: [] };
  const { points } = curve;
  for (let i = points.threshold.length - 1; i >= 0; i -= 1) {
    const recall = points.recall[i];
    const precision = points.precision[i];
    if (recall !== null && precision !== null) {
      line.x.push(recall);
      line.y.push(precision);
      line.threshold.push(points.threshold[i]);
    }
  }
  return line;
}

/**
 * The ROC curve (x false-positive rate, y recall) from the highest threshold to the lowest,
 * starting at (0, 0): above every probability nothing is predicted positive.
 */
export function rocLine(curve: ThresholdCurve): CurveLine {
  const line: CurveLine = { x: [0], y: [0], threshold: [Infinity] };
  const { points } = curve;
  for (let i = points.threshold.length - 1; i >= 0; i -= 1) {
    const recall = points.recall[i];
    const fpr = points.fpr[i];
    if (recall !== null && fpr !== null) {
      line.x.push(fpr);
      line.y.push(recall);
      line.threshold.push(points.threshold[i]);
    }
  }
  return line;
}

/** A threshold for display: 0.5 → "0.5", 0.123456 → "0.1235", 0.000123 → "0.000123". */
export function formatThreshold(value: number | null | undefined): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return '–';
  }
  if (value === 0 || value >= 0.001) {
    return String(Number(value.toFixed(4)));
  }
  return String(Number(value.toPrecision(3)));
}
