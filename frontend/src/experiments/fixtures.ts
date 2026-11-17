/** Rows and curves of the experiment tracker for tests (and screenshots). */
import type { RunCurve, RunRow, RunTable } from '@/api';

export function row(over: Partial<RunRow> & Pick<RunRow, 'id' | 'name'>): RunRow {
  return {
    status: 'completed',
    kind: 'training',
    family: 'neural',
    task: 'classification',
    model: 'Neural network',
    dataset: { key: 'upload:7', kind: 'upload', name: 'iris.csv' },
    created_at: '2026-11-10T09:00:00Z',
    started_at: '2026-11-10T09:00:05Z',
    finished_at: '2026-11-10T09:05:05Z',
    duration_seconds: 300,
    epochs: 10,
    epochs_done: 10,
    trials_done: 0,
    search: false,
    total_params: 1234,
    loss: 'cross_entropy',
    settings: { optimizer: 'adam', learning_rate: 0.001, batch_size: 64, epochs: 10 },
    metrics: { val_loss: 0.4, val_accuracy: 0.9, val_macro_f1: 0.88, test_accuracy: 0.89 },
    starred: false,
    tags: [],
    note: '',
    project: 3,
    project_name: 'Iris net',
    started_by: 'ada',
    mine: true,
    can_annotate: true,
    ...over,
  };
}

export const ROWS: RunRow[] = [
  row({ id: 'a1', name: 'Adam 1e-3', tags: ['baseline'] }),
  row({
    id: 'b2',
    name: 'SGD 1e-2',
    settings: { optimizer: 'sgd', learning_rate: 0.01, batch_size: 32, epochs: 10 },
    metrics: { val_loss: 0.6, val_accuracy: 0.82, val_macro_f1: 0.8, test_accuracy: 0.8 },
    starred: true,
    note: 'Too noisy.',
  }),
  row({
    id: 'c3',
    name: 'Focal',
    loss: 'focal',
    settings: {
      optimizer: 'adam',
      learning_rate: 0.0003,
      batch_size: 64,
      epochs: 10,
      loss: 'focal',
    },
    metrics: { val_loss: 0.2, val_accuracy: 0.91, val_macro_f1: 0.9, test_accuracy: 0.9 },
    started_by: 'bob',
    mine: false,
    can_annotate: false,
  }),
  row({
    id: 'd4',
    name: 'Search',
    search: true,
    trials_done: 8,
    status: 'running',
    finished_at: null,
    metrics: { val_loss: 0.5 },
  }),
];

export function table(results: RunRow[] = ROWS, over: Partial<RunTable> = {}): RunTable {
  return {
    count: results.length,
    page: 1,
    pages: 1,
    page_size: 50,
    sort: 'created_at',
    order: 'desc',
    results,
    facets: {
      tags: [{ name: 'baseline', count: 1 }],
      datasets: [{ key: 'upload:7', name: 'iris.csv' }],
      projects: [{ id: 3, name: 'Iris net' }],
      families: { neural: results.length },
    },
    truncated: false,
    limits: { compare: 10, tags: 20, note: 2000 },
    ...over,
  };
}

export function curve(id: string, name: string, over: Partial<RunCurve> = {}): RunCurve {
  const points = Array.from({ length: 5 }, (_, i) => ({
    x: i + 1,
    epoch: i + 1,
    time: (i + 1) * 12,
    train_loss: 1 / (i + 1),
    val_loss: 1 / (i + 1) + 0.1,
    val_accuracy: 0.6 + i * 0.05,
  }));
  return {
    id,
    name,
    status: 'completed',
    kind: 'training',
    family: 'neural',
    task: 'classification',
    loss: 'cross_entropy',
    loss_label: 'cross-entropy',
    x_unit: 'epoch',
    points,
    ...over,
  };
}
