/** Test data for the training panel (only imported by tests). */
import type {
  Analysis,
  GraphDocument,
  RunDetail,
  RunSummary,
  TrainingConfig,
  TrainingStatus,
} from '@/api/types';

export const STATUS: TrainingStatus = {
  enabled: true,
  available: true,
  reason: null,
  torch_version: '2.2.2',
  torchvision: true,
  devices: [
    { id: 'cpu', label: 'CPU' },
    { id: 'mps', label: 'Apple GPU (MPS)' },
  ],
  default_device: 'mps',
  builtin_datasets: [
    {
      id: 'mnist',
      name: 'MNIST',
      description: 'Handwritten digits 0-9, grayscale 28×28.',
      sample_shape: [1, 28, 28],
      num_classes: 10,
      class_names: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
      train_size: 60000,
      val_size: 10000,
      download_mb: 12,
      available: true,
    },
  ],
  max_upload_mb: 500,
  max_concurrent_runs: 1,
  login_required: true,
};

export const DOC: GraphDocument = {
  version: 1,
  nodes: [
    { id: 'in', type: 'Input', params: { shape: [4] }, position: { x: 0, y: 0 } },
    { id: 'fc', type: 'Linear', params: { out_features: 3 }, position: { x: 300, y: 0 } },
  ],
  edges: [{ id: 'e', source: 'in', target: 'fc' }],
};

export const ANALYSIS: Analysis = {
  valid: true,
  errors: [],
  warnings: [],
  nodes: {},
  inputs: ['in'],
  outputs: ['fc'],
  total_params: 15,
};

export const CONFIG: TrainingConfig = {
  epochs: 10,
  batch_size: 64,
  learning_rate: 0.001,
  optimizer: 'adam',
  weight_decay: 0,
  val_split: 0.2,
  device: 'auto',
  max_samples: null,
  seed: null,
};

export function summary(overrides: Partial<RunSummary> = {}): RunSummary {
  return {
    id: 'r1',
    name: 'Small CNN',
    status: 'running',
    message: 'Epoch 4 of 10',
    dataset: { kind: 'builtin', id: 'mnist', name: 'MNIST' },
    config: CONFIG,
    task: 'classification',
    epochs: 10,
    epochs_done: 3,
    metrics: { train_loss: 0.2, val_loss: 0.15, val_accuracy: 0.95, val_rmse: null },
    best: { epoch: 3, val_loss: 0.15, val_accuracy: 0.95 },
    has_weights: false,
    stop_requested: false,
    total_params: 101770,
    created_at: '2026-10-28T10:00:00Z',
    finished_at: null,
    ...overrides,
  };
}

export function detail(overrides: Partial<RunDetail> = {}): RunDetail {
  const base = summary(overrides);
  return {
    ...base,
    graph: DOC,
    plan: null,
    state: {
      status: base.status,
      message: base.message,
      updated_at: 1761645630,
      started_at: 1761645600,
      finished_at: null,
      device: 'mps',
      device_label: 'Apple GPU (MPS)',
      task: 'classification',
      num_classes: 10,
      class_names: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
      train_size: 60000,
      val_size: 10000,
      epochs: 10,
      steps_per_epoch: 938,
      total_params: 101770,
      history: [
        {
          epoch: 1,
          train_loss: 0.6,
          val_loss: 0.3,
          train_accuracy: 0.8,
          val_accuracy: 0.9,
          seconds: 12,
        },
        {
          epoch: 2,
          train_loss: 0.3,
          val_loss: 0.2,
          train_accuracy: 0.9,
          val_accuracy: 0.93,
          seconds: 11,
        },
        {
          epoch: 3,
          train_loss: 0.2,
          val_loss: 0.15,
          train_accuracy: 0.94,
          val_accuracy: 0.95,
          seconds: 11,
        },
      ],
      progress: { epoch: 4, step: 469, steps: 938, loss: 0.18, samples_per_second: 5400 },
      best: { epoch: 3, val_loss: 0.15, val_accuracy: 0.95 },
      weights: false,
      preprocessing: null,
      log: ['Downloading MNIST …', 'Epoch 1 of 10'],
    },
    ...overrides,
  };
}
