/** The curves of several runs in one chart: what to draw and when losses differ in kind. */
import type { RunCurve, TrainingTask } from '@/api';
import type { OverlaySeries } from '@/components/charts/OverlayChart';
import { i18n } from '@/i18n';
import { LOSS_NAMES } from '@/training/format';
import { formatMetricOf, metricLabel } from './columns';

export type XAxis = 'epoch' | 'time';

/**
 * Exponential moving average with the start corrected (as TensorBoard smooths): `weight` 0
 * keeps the values, 0.9 smooths strongly. Gaps (null) stay gaps and do not reset it.
 */
export function smooth(values: (number | null)[], weight: number): (number | null)[] {
  if (!(weight > 0)) {
    return values;
  }
  const w = Math.min(0.99, weight);
  let last = 0;
  let steps = 0;
  return values.map((value) => {
    if (typeof value !== 'number' || !Number.isFinite(value)) {
      return null;
    }
    last = last * w + (1 - w) * value;
    steps += 1;
    return last / (1 - w ** steps);
  });
}

/** The metric curves the runs can show besides the loss: their main metric first. */
export function curveMetrics(curves: RunCurve[]): string[] {
  const found = new Set<string>();
  for (const curve of curves) {
    for (const point of curve.points) {
      Object.entries(point).forEach(([key, value]) => {
        if (!['x', 'epoch', 'step', 'time'].includes(key) && typeof value === 'number') {
          found.add(key);
        }
      });
    }
  }
  const main = [
    'val_loss',
    'train_loss',
    'val_accuracy',
    'train_accuracy',
    'val_rmse',
    'val_r2',
    'val_iou',
    'val_dice',
    'val_pixel_accuracy',
  ];
  return [
    ...main.filter((key) => found.has(key)),
    ...[...found].filter((k) => !main.includes(k)).sort(),
  ];
}

/** The main metric of a task's curves: accuracy, RMSE or IoU. */
export function mainMetric(curves: RunCurve[]): string | null {
  const available = curveMetrics(curves);
  const tasks = new Set(curves.map((c) => c.task));
  const preferred: Record<string, string> = {
    classification: 'val_accuracy',
    regression: 'val_rmse',
    segmentation: 'val_iou',
  };
  for (const task of tasks) {
    const key = task ? preferred[task] : undefined;
    if (key && available.includes(key)) {
      return key;
    }
  }
  return available.find((key) => !key.endsWith('_loss')) ?? null;
}

/** The series of one metric per run, on the epoch or the time axis, smoothed. */
export function overlaySeries(
  curves: RunCurve[],
  metric: string,
  xAxis: XAxis,
  smoothing: number
): OverlaySeries[] {
  return curves.map((curve) => {
    const points = curve.points.filter(
      (p) => xAxis === 'epoch' || (typeof p.time === 'number' && Number.isFinite(p.time))
    );
    const raw = points.map((p) => {
      const value = p[metric];
      return typeof value === 'number' && Number.isFinite(value) ? value : null;
    });
    return {
      key: curve.id,
      label: curve.name,
      x: points.map((p) => (xAxis === 'time' ? (p.time as number) : p.x)),
      y: smooth(raw, smoothing),
    };
  });
}

/** The loss names of runs that trained with different kinds of loss, or null if they agree. */
export function differentLosses(curves: RunCurve[]): string[] | null {
  const names = Array.from(
    new Set(curves.map((c) => c.loss).filter((loss): loss is string => !!loss))
  );
  if (names.length < 2) {
    return null;
  }
  return names.map((name) => LOSS_NAMES[name] ?? name);
}

/** The tasks of the runs if they differ (a classification next to a regression). */
export function differentTasks(curves: RunCurve[]): TrainingTask[] | null {
  const tasks = Array.from(
    new Set(curves.map((c) => c.task).filter((task): task is TrainingTask => !!task))
  );
  return tasks.length > 1 ? tasks : null;
}

/** "Loss" for the loss curves, the metric's label for the others. */
export function curveLabel(metric: string): string {
  if (metric === 'val_loss') {
    return i18n.t('experiments.curves.valLoss');
  }
  if (metric === 'train_loss') {
    return i18n.t('experiments.curves.trainLoss');
  }
  return metricLabel(metric);
}

export function formatCurveValue(metric: string, value: number): string {
  return formatMetricOf(metric, value);
}

/** 75 → "1:15", 3725 → "1:02:05" (seconds since the start of a run). */
export function formatElapsed(seconds: number): string {
  const total = Math.max(0, Math.round(seconds));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const pad = (n: number) => String(n).padStart(2, '0');
  return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${m}:${pad(s)}`;
}

/** The table view of the overlay: one row per x value and a column per run. */
export function overlayTable(
  series: OverlaySeries[],
  xLabel: string,
  formatX: (x: number) => string,
  formatY: (y: number) => string
) {
  const xs = Array.from(new Set(series.flatMap((s) => s.x))).sort((a, b) => a - b);
  return {
    head: [xLabel, ...series.map((s) => s.label)],
    rows: xs.map((x) => [
      formatX(x),
      ...series.map((s) => {
        const index = s.x.indexOf(x);
        const value = index >= 0 ? s.y[index] : null;
        return typeof value === 'number' ? formatY(value) : '–';
      }),
    ]),
  };
}
