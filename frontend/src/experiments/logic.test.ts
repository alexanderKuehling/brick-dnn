import { i18n } from '@/i18n';
import {
  availableColumns,
  formatMetricOf,
  metricLabel,
  metricOrder,
  shownColumns,
  splitMetric,
} from './columns';
import {
  curveMetrics,
  differentLosses,
  differentTasks,
  formatElapsed,
  mainMetric,
  overlaySeries,
  overlayTable,
  smooth,
} from './curves';
import { curve, row, ROWS } from './fixtures';
import { sortRows } from './SharedComparisonPage';

describe('columns', () => {
  it('names metrics by split and formats shares as percentages', () => {
    expect(splitMetric('val_macro_f1')).toEqual({ split: 'val', base: 'macro_f1' });
    expect(splitMetric('my_metric')).toEqual({ split: null, base: 'my_metric' });
    expect(metricLabel('val_macro_f1')).toBe('F1 (macro) · val.');
    expect(metricLabel('test_pr_auc')).toBe('PR AUC · test');
    expect(metricLabel('best_epoch')).toBe('Kept epoch');
    expect(metricLabel('custom')).toBe('custom');
    expect(formatMetricOf('val_accuracy', 0.912)).toBe('91.2%');
    expect(formatMetricOf('val_loss', 0.123456)).toBe('0.1235');
    expect(formatMetricOf('val_r2', 0.91234)).toBe('0.912');
    expect(formatMetricOf('val_loss', null)).toBe('–');
  });

  it('names them in German too', async () => {
    await i18n.changeLanguage('de');
    expect(metricLabel('val_accuracy')).toBe('Genauigkeit · Val.');
    expect(formatMetricOf('val_accuracy', 0.912)).toMatch(/^91,2\s%$/);
  });

  it('orders metrics by kind and split', () => {
    const keys = ['test_accuracy', 'best_epoch', 'zzz', 'val_loss', 'val_accuracy'];
    expect([...keys].sort(metricOrder)).toEqual([
      'val_accuracy',
      'test_accuracy',
      'val_loss',
      'zzz',
      'best_epoch',
    ]);
  });

  it('shows the defaults that the rows have values for, or the chosen columns', () => {
    const ids = shownColumns(ROWS, null).map((c) => c.id);
    expect(ids).toContain('metric:val_macro_f1');
    expect(ids).not.toContain('metric:val_rmse'); // no regression among the rows
    expect(ids.slice(0, 2)).toEqual(['status', 'started']);
    expect(shownColumns(ROWS, ['metric:val_rmse', 'dataset']).map((c) => c.id)).toEqual([
      'metric:val_rmse',
      'dataset',
    ]);
    const available = availableColumns(ROWS).map((c) => c.id);
    expect(available).toContain('setting:loss');
    expect(available).toContain('metric:test_accuracy');
  });
});

describe('curves', () => {
  it('smooths with a corrected start and keeps gaps', () => {
    expect(smooth([1, 1, 1], 0.9)).toEqual([1, 1, 1].map(() => expect.closeTo(1, 9)));
    const smoothed = smooth([0, 10, null, 10], 0.5);
    expect(smoothed[0]).toBe(0);
    expect(smoothed[1]).toBeCloseTo(6.667, 2);
    expect(smoothed[2]).toBeNull();
    expect(smooth([3, 4], 0)).toEqual([3, 4]);
  });

  it('finds the metrics and the main one of the task', () => {
    const curves = [curve('a', 'A'), curve('b', 'B')];
    expect(curveMetrics(curves)).toEqual(['val_loss', 'train_loss', 'val_accuracy']);
    expect(mainMetric(curves)).toBe('val_accuracy');
    const regression = curve('r', 'R', {
      task: 'regression',
      points: [{ x: 1, epoch: 1, time: 1, val_loss: 1, val_rmse: 2 }],
    });
    expect(mainMetric([regression])).toBe('val_rmse');
  });

  it('warns when losses or tasks differ', () => {
    const a = curve('a', 'A');
    const b = curve('b', 'B', { loss: 'focal' });
    expect(differentLosses([a, a])).toBeNull();
    expect(differentLosses([a, b])).toEqual(['Cross-entropy', 'Focal loss']);
    expect(differentTasks([a, curve('c', 'C', { task: 'regression' })])).toEqual([
      'classification',
      'regression',
    ]);
  });

  it('builds a series per run on the epoch or the time axis, and a table', () => {
    const series = overlaySeries([curve('a', 'A')], 'val_accuracy', 'time', 0);
    expect(series[0].x).toEqual([12, 24, 36, 48, 60]);
    expect(series[0].y[0]).toBeCloseTo(0.6);
    const byEpoch = overlaySeries([curve('a', 'A'), curve('b', 'B')], 'val_loss', 'epoch', 0);
    const view = overlayTable(byEpoch, 'Epoch', String, (v) => v.toFixed(2));
    expect(view.head).toEqual(['Epoch', 'A', 'B']);
    expect(view.rows[0]).toEqual(['1', '1.10', '1.10']);
    expect(formatElapsed(75)).toBe('1:15');
    expect(formatElapsed(3725)).toBe('1:02:05');
  });
});

describe('sorting a shared comparison', () => {
  it('sorts by a metric with missing values last', () => {
    const rows = [row({ id: '1', name: 'one', metrics: {} }), ...ROWS];
    const column = availableColumns(rows).find((c) => c.id === 'metric:val_accuracy')!;
    expect(sortRows(rows, column, 'desc').map((r) => r.name)).toEqual([
      'Focal',
      'Adam 1e-3',
      'SGD 1e-2',
      'one',
      'Search',
    ]);
  });
});
