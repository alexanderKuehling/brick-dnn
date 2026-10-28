/** Test data for the training panel (only imported by tests). */
import type {
  Analysis,
  GraphDocument,
  RunDetail,
  RunSummary,
  TrainingConfig,
  TrainingStatus,
  TrialEntry,
  TuningConfig,
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
  test_split: 0,
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
    metrics: {
      train_loss: 0.2,
      val_loss: 0.15,
      val_accuracy: 0.95,
      val_rmse: null,
      test_loss: null,
      test_accuracy: null,
      test_rmse: null,
    },
    best: { epoch: 3, val_loss: 0.15, val_accuracy: 0.95 },
    tuning: null,
    trials_done: 0,
    best_trial: null,
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
      test_size: null,
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
      test: null,
      weights: false,
      preprocessing: null,
      log: ['Downloading MNIST …', 'Epoch 1 of 10'],
      tuning: null,
      trials_total: null,
      trial: null,
      trials: [],
      best_trial: null,
    },
    ...overrides,
  };
}

export const TUNING: TuningConfig = {
  method: 'random',
  trials: 4,
  space: { learning_rate: { min: 0.0001, max: 0.01 }, batch_size: [32, 64] },
};

/** One trial of a search (completed unless overridden). */
export function trial(overrides: Partial<TrialEntry> = {}): TrialEntry {
  return {
    trial: 1,
    params: { learning_rate: 0.00312, batch_size: 64 },
    status: 'completed',
    message: '',
    epochs_done: 5,
    train_loss: 0.3,
    val_loss: 0.2,
    val_accuracy: 0.93,
    val_rmse: null,
    seconds: 40,
    ...overrides,
  };
}

/**
 * A search of 4 trials with 5 epochs each: trial 1 done, trial 2 failed, trial 3 the best so
 * far, trial 4 running in its third epoch.
 */
export function searchDetail(overrides: Partial<RunDetail> = {}): RunDetail {
  const base = detail({
    message: 'Trial 4 of 4: epoch 3 of 5',
    epochs: 5,
    epochs_done: 2,
    config: { ...CONFIG, epochs: 5 },
    tuning: TUNING,
    trials_done: 3,
    best_trial: {
      trial: 3,
      params: { learning_rate: 0.0011, batch_size: 32 },
      val_loss: 0.12,
      val_accuracy: 0.96,
      val_rmse: null,
    },
  });
  const trials = [
    trial(),
    trial({
      trial: 2,
      params: { learning_rate: 0.0095, batch_size: 32 },
      status: 'failed',
      message: 'The loss became NaN in epoch 2. Try a lower learning rate.',
      epochs_done: 1,
      val_loss: null,
      val_accuracy: null,
    }),
    trial({
      trial: 3,
      params: { learning_rate: 0.0011, batch_size: 32 },
      val_loss: 0.12,
      val_accuracy: 0.96,
    }),
    trial({
      trial: 4,
      params: { learning_rate: 0.00042, batch_size: 64 },
      status: 'running',
      epochs_done: 2,
      val_loss: 0.4,
      val_accuracy: 0.88,
      seconds: null,
    }),
  ];
  return {
    ...base,
    state: {
      ...base.state!,
      message: 'Trial 4 of 4: epoch 3 of 5',
      epochs: 5,
      history: base.state!.history.slice(0, 2),
      progress: { epoch: 3, step: 469, steps: 938, loss: 0.18, samples_per_second: 5400 },
      tuning: TUNING,
      trials_total: 4,
      trial: 4,
      trials,
      best_trial: 3,
    },
    ...overrides,
  };
}
