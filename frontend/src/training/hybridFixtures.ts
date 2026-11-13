/**
 * A hybrid network for tests: an MLP learns 3 features, gradient boosting is fitted on them
 * (Input → Linear → ReLU → Linear(3) → HistGradientBoostingClassifier). Only imported by tests.
 */
import type { Analysis, GraphDocument, NodeAnalysis, RunDetail } from '@/api/types';
import { CONFIG, svcDetail } from './fixtures';

export const HYBRID_DOC: GraphDocument = {
  version: 1,
  nodes: [
    { id: 'in', type: 'Input', params: { shape: [4] }, position: { x: 0, y: 0 } },
    { id: 'fc', type: 'Linear', params: { out_features: 16 }, position: { x: 280, y: 0 } },
    { id: 'act', type: 'Activation', params: { function: 'relu' }, position: { x: 560, y: 0 } },
    { id: 'fc2', type: 'Linear', params: { out_features: 3 }, position: { x: 840, y: 0 } },
    { id: 'gb', type: 'HistGradientBoostingClassifier', params: {}, position: { x: 1120, y: 0 } },
  ],
  edges: [
    { id: 'e1', source: 'in', target: 'fc' },
    { id: 'e2', source: 'fc', target: 'act' },
    { id: 'e3', source: 'act', target: 'fc2' },
    { id: 'e4', source: 'fc2', target: 'gb' },
  ],
};

const tensor = (type: string, shape: number[], params: number): NodeAnalysis => ({
  type,
  output_shape: shape,
  dtype: 'float',
  num_params: params,
  params: {},
  output_kind: 'tensor',
  kernel: null,
  task: null,
});

export const HYBRID_ANALYSIS: Analysis = {
  valid: true,
  errors: [],
  warnings: [],
  nodes: {
    in: tensor('Input', [4], 0),
    fc: tensor('Linear', [16], 80),
    act: tensor('Activation', [16], 0),
    fc2: tensor('Linear', [3], 51),
    gb: {
      type: 'HistGradientBoostingClassifier',
      output_shape: null,
      dtype: null,
      num_params: null,
      params: {},
      output_kind: 'estimator',
      kernel: null,
      task: 'classification',
    },
  },
  inputs: ['in'],
  outputs: ['gb'],
  total_params: null,
  family: 'hybrid',
  classical: {
    node: 'gb',
    type: 'HistGradientBoostingClassifier',
    estimator: 'HistGradientBoostingClassifier',
    task: 'classification',
    kernel: null,
  },
  hybrid: { boundary: 'fc2', features: 3, neural_params: 131 },
};

/** A finished hybrid run on demo data: 3 epochs of features, then gradient boosting. */
export function hybridDetail(overrides: Partial<RunDetail> = {}): RunDetail {
  const base = svcDetail();
  return {
    ...base,
    name: 'MLP features + trees',
    message:
      'Finished: 3 features learned in 3 epochs; HistGradientBoostingClassifier on them: validation accuracy 91.0%.',
    graph: HYBRID_DOC,
    family: 'hybrid',
    config: { ...CONFIG, epochs: 3 },
    epochs: 3,
    epochs_done: 3,
    metrics: {
      train_loss: 0.1,
      val_loss: 0.28,
      val_accuracy: 0.91,
      val_rmse: null,
      val_r2: null,
      test_loss: null,
      test_accuracy: null,
      test_rmse: null,
      test_r2: null,
    },
    total_params: 131,
    plan: {
      ...base.plan!,
      classical: {
        ...base.plan!.classical!,
        node: 'gb',
        type: 'HistGradientBoostingClassifier',
        estimator: 'HistGradientBoostingClassifier',
        label: 'Gradient boosting classification',
        scale: 'gradient_boosting',
        max_rows: 1_000_000,
      },
      hybrid: {
        boundary: 'fc2',
        features: 3,
        head_outputs: 3,
        feature_training: 'pretrain',
        neural_params: 131,
      },
    },
    state: {
      ...base.state!,
      family: 'hybrid',
      epochs: 3,
      weights_epoch: 3,
      history: [1, 2, 3].map((epoch) => ({
        epoch,
        train_loss: 1 / epoch,
        val_loss: 1.1 / epoch,
        train_accuracy: 0.6 + epoch / 10,
        val_accuracy: 0.55 + epoch / 10,
        seconds: 0.2,
      })),
      classical: {
        ...base.state!.classical!,
        estimator: 'HistGradientBoostingClassifier',
        label: 'Gradient boosting classification',
        max_rows: 1_000_000,
        rows: 800,
        subsampled_from: null,
        fit_seconds: 0.4,
        support_vectors: null,
        iterations: 50,
        max_iterations: 200,
        train: { loss: 0.1, accuracy: 0.97, rmse: null, r2: null },
        validation: { loss: 0.28, accuracy: 0.91, rmse: null, r2: null },
        hybrid: { boundary: 'fc2', features: 3, feature_training: 'pretrain', epochs: 3 },
      },
    },
    ...overrides,
  };
}
