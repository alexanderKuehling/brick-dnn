/** Test data for the training panel (only imported by tests). */
import type {
  Analysis,
  Dataset,
  GraphDocument,
  RunDetail,
  RunQueue,
  RunSummary,
  TrainingConfig,
  TrainingStatus,
  TrialEntry,
  TuningConfig,
} from '@/api/types';

// ---------------------------------------------------------------------------
// The training queue (servers with a training worker); the rest is below.
// ---------------------------------------------------------------------------

export function queue(overrides: Partial<RunQueue> = {}): RunQueue {
  return {
    position: 3,
    waiting: 5,
    estimated_wait_seconds: 240,
    estimated_start_at: '2026-10-28T10:04:00Z',
    worker_online: true,
    blocked_by: null,
    priority: 0,
    hint: 'Paid plans start first.',
    ...overrides,
  };
}

const NOT_TRAINED = {
  status: 'queued',
  message: 'Waiting for a free training slot.',
  epochs_done: 0,
  metrics: null,
  best: null,
} as const;

/** A run that waits for a training worker. */
export function queuedSummary(overrides: Partial<RunSummary> = {}): RunSummary {
  return summary({ ...NOT_TRAINED, queue: queue(), ...overrides });
}

export function queuedDetail(overrides: Partial<RunDetail> = {}): RunDetail {
  return detail({ ...NOT_TRAINED, queue: queue(), state: null, ...overrides });
}

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
      test_size: 10000,
      download_mb: 12,
      available: true,
    },
  ],
  max_upload_mb: 500,
  max_concurrent_runs: 1,
  login_required: true,
};

/** The server trains with a worker: one slot, one training runs, two wait. */
export const QUEUE_STATUS: TrainingStatus = {
  ...STATUS,
  queue: {
    online: true,
    workers: 1,
    slots: 1,
    running: 1,
    waiting: 2,
    message: null,
    priority_hint: 'Paid plans start first.',
  },
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
    project: null,
    started_by: 'ada',
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

/** An uploaded CSV dataset of the user. */
export function dataset(overrides: Partial<Dataset> = {}): Dataset {
  const name = overrides.name ?? 'iris';
  return {
    id: 7,
    name,
    kind: 'csv',
    size_bytes: 4551,
    created_at: '2026-10-28T10:00:00Z',
    info: {
      kind: 'csv',
      name,
      num_samples: 150,
      builtin: null,
      sample_shape: null,
      train_size: null,
      test_size: null,
      num_features: 4,
      columns: ['a', 'b', 'c', 'd', 'species'],
      has_header: true,
      delimiter: ',',
      target_kind: 'text',
      target_range: null,
      class_names: ['setosa', 'versicolor', 'virginica'],
      class_counts: null,
      subdir: null,
      num_classes: 3,
    },
    owner: 'ada',
    mine: true,
    shared_in: [],
    ...overrides,
  };
}

/**
 * A finished regression of a neural network on a CSV file with a test set: 3 epochs, the
 * weights of epoch 2 (lowest validation loss) are kept.
 */
export function regressionDetail(overrides: Partial<RunDetail> = {}): RunDetail {
  const base = detail({
    name: 'House prices MLP',
    status: 'completed',
    message: 'Finished 3 epochs; kept the weights of epoch 2 (lowest validation loss). …',
    dataset: { kind: 'upload', id: 1, name: 'houses' },
    config: { ...CONFIG, epochs: 3, test_split: 0.2 },
    task: 'regression',
    epochs: 3,
    epochs_done: 3,
    metrics: {
      train_loss: 0.21,
      val_loss: 0.18,
      val_accuracy: null,
      val_rmse: 41.27,
      val_r2: 0.8234,
      test_loss: 0.2,
      test_accuracy: null,
      test_rmse: 43.9,
      test_r2: 0.7961,
    },
    best: { epoch: 2, val_loss: 0.18, val_accuracy: null },
    has_weights: true,
    total_params: 161,
    finished_at: '2026-10-28T10:00:30Z',
  });
  const epoch = (n: number, loss: number, rmse: number, r2: number | null) => ({
    epoch: n,
    train_loss: loss + 0.03,
    val_loss: loss,
    train_accuracy: null,
    val_accuracy: null,
    val_rmse: rmse,
    val_r2: r2,
    seconds: 0.4,
  });
  return {
    ...base,
    state: {
      ...base.state!,
      status: 'completed',
      message: base.message,
      finished_at: 1761645630,
      device: 'cpu',
      device_label: 'CPU',
      task: 'regression',
      num_classes: null,
      class_names: null,
      train_size: 600,
      val_size: 200,
      test_size: 200,
      epochs: 3,
      steps_per_epoch: 10,
      total_params: 161,
      history: [
        epoch(1, 0.62, 76.4, -0.4512),
        epoch(2, 0.18, 41.27, 0.8234),
        epoch(3, 0.2, 43.5, 0.8),
      ],
      progress: null,
      best: { epoch: 2, val_loss: 0.18, val_accuracy: null },
      weights_epoch: 2,
      stopped_early: false,
      test: { loss: 0.2, accuracy: null, rmse: 43.9, r2: 0.7961 },
      weights: true,
      log: [],
    },
    ...overrides,
  };
}

// --------------------------------------------------------------------------
// Classical models (scikit-learn)
// --------------------------------------------------------------------------

/** A server without PyTorch: only classical models can be trained. */
export const CLASSICAL_STATUS: TrainingStatus = {
  ...STATUS,
  available: false,
  reason: 'PyTorch is not installed on the server.',
  torch_version: null,
  torchvision: false,
  devices: [],
  default_device: null,
  classical: {
    available: true,
    reason: null,
    sklearn_version: '1.9.1',
    max_rows: {
      GaussianProcessRegressor: 5000,
      GaussianProcessClassifier: 2000,
      SVR: 50000,
      SVC: 50000,
    },
    tuning: false,
  },
};

/** Input → Gaussian process regression with RBF + WhiteKernel (see GP_ANALYSIS). */
export const GP_DOC: GraphDocument = {
  version: 1,
  nodes: [
    { id: 'in', type: 'Input', params: { shape: [8] }, position: { x: 0, y: 0 } },
    { id: 'rbf', type: 'RBF', params: { length_scale: 1 }, position: { x: 0, y: 170 } },
    { id: 'noise', type: 'WhiteKernel', params: { noise_level: 0.1 }, position: { x: 0, y: 340 } },
    { id: 'sum', type: 'KernelSum', params: {}, position: { x: 300, y: 250 } },
    { id: 'gp', type: 'GaussianProcessRegressor', params: {}, position: { x: 600, y: 0 } },
  ],
  edges: [
    { id: 'e1', source: 'in', target: 'gp' },
    { id: 'e2', source: 'rbf', target: 'sum' },
    { id: 'e3', source: 'noise', target: 'sum' },
    { id: 'e4', source: 'sum', target: 'gp' },
  ],
};

const GP_KERNEL =
  'ConstantKernel(constant_value=1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)';

/** A finished Gaussian process regression on a CSV file (as the server reports it). */
export function gpDetail(overrides: Partial<RunDetail> = {}): RunDetail {
  const metrics = {
    train_loss: 0.0064,
    val_loss: 0.0397,
    val_accuracy: null,
    val_rmse: 9.6824,
    val_r2: 0.96496,
    test_loss: 0.05,
    test_accuracy: null,
    test_rmse: 10.21,
    test_r2: 0.9512,
  };
  const base = detail({
    name: 'House prices GP',
    status: 'completed',
    message: 'Finished: validation RMSE 9.682, R² 0.965. Fitted on 640 rows in 0.8 s.',
    dataset: { kind: 'upload', id: 1, name: 'houses' },
    config: { ...CONFIG, test_split: 0.2 },
    task: 'regression',
    family: 'classical',
    epochs: 1,
    epochs_done: 1,
    metrics,
    best: { epoch: 1, val_loss: 0.0397, val_accuracy: null },
    has_weights: true,
    total_params: 3,
    finished_at: '2026-10-28T10:00:02Z',
  });
  return {
    ...base,
    graph: GP_DOC,
    plan: {
      task: 'regression',
      summary:
        "Regression: predict 'price' from 8 features (1,000 rows) with GaussianProcessRegressor.",
      inputs: [{ shape: [8], dtype: 'float', vocab: null }],
      targets: [{ kind: 'value', shape: [1] }],
      image: null,
      num_classes: null,
      class_names: null,
      notes: [],
      classical: {
        node: 'gp',
        type: 'GaussianProcessRegressor',
        estimator: 'GaussianProcessRegressor',
        label: 'Gaussian process regression',
        task: 'regression',
        kernel: GP_KERNEL,
        scale: 'gaussian_process',
        max_rows: 5000,
      },
    },
    state: {
      ...base.state!,
      status: 'completed',
      message: base.message,
      finished_at: 1761645602,
      device: 'cpu',
      device_label: 'CPU (scikit-learn)',
      family: 'classical',
      task: 'regression',
      num_classes: null,
      class_names: null,
      train_size: 640,
      val_size: 160,
      test_size: 200,
      epochs: 1,
      steps_per_epoch: 1,
      total_params: 3,
      history: [
        {
          epoch: 1,
          train_loss: 0.0064,
          val_loss: 0.0397,
          train_accuracy: null,
          val_accuracy: null,
          val_rmse: 9.6824,
          val_r2: 0.96496,
          seconds: 0.85,
        },
      ],
      progress: null,
      best: { epoch: 1, val_loss: 0.0397, val_accuracy: null },
      weights_epoch: 1,
      stopped_early: false,
      test: { loss: 0.05, accuracy: null, rmse: 10.21, r2: 0.9512 },
      weights: true,
      preprocessing: {
        kind: 'tabular',
        columns: ['rooms', 'area', 'age', 'x4', 'x5', 'x6', 'x7', 'x8'],
        target: 'price',
        mean: [3.2, 120.5, 30, 0, 0, 0, 0, 0],
        std: [1, 40, 12, 1, 1, 1, 1, 1],
        target_mean: null,
        target_std: null,
        class_names: null,
        estimator: 'GaussianProcessRegressor',
      },
      log: ['Fitted the Gaussian process on 640 rows in 0.8 s.'],
      classical: {
        estimator: 'GaussianProcessRegressor',
        label: 'Gaussian process regression',
        max_rows: 5000,
        rows: 640,
        subsampled_from: null,
        fit_seconds: 0.838,
        initial_kernel: GP_KERNEL,
        kernel: '2.49**2 * RBF(length_scale=3.38) + WhiteKernel(noise_level=0.0141)',
        log_marginal_likelihood: -117.7,
        support_vectors: null,
        train: { loss: 0.0064, accuracy: null, rmse: 3.8988, r2: 0.99357 },
      },
    },
    ...overrides,
  };
}

/** A finished SVC on demo data with 3 classes, subsampled to the row limit. */
export function svcDetail(overrides: Partial<RunDetail> = {}): RunDetail {
  const base = gpDetail();
  return {
    ...base,
    name: 'SVM classifier',
    message: 'Finished: validation accuracy 93.0%. Fitted on 800 rows in 0.1 s.',
    dataset: { kind: 'random', id: null, name: 'Demo data' },
    config: CONFIG,
    task: 'demo',
    total_params: null,
    metrics: {
      train_loss: 0.2,
      val_loss: 0.25,
      val_accuracy: 0.93,
      val_rmse: null,
      val_r2: null,
      test_loss: null,
      test_accuracy: null,
      test_rmse: null,
      test_r2: null,
    },
    plan: {
      ...base.plan!,
      task: 'demo',
      summary: 'Demo data: random features and a hidden rule to learn (3 classes).',
      targets: [{ kind: 'class', num_classes: 3 }],
      num_classes: 3,
      class_names: ['class 0', 'class 1', 'class 2'],
      classical: {
        ...base.plan!.classical!,
        node: 'svc',
        type: 'SVC',
        estimator: 'SVC',
        label: 'Support vector classification',
        task: 'classification',
        kernel: null,
        scale: 'svm',
        max_rows: 800,
      },
    },
    state: {
      ...base.state!,
      task: 'demo',
      num_classes: 3,
      class_names: ['class 0', 'class 1', 'class 2'],
      train_size: 800,
      val_size: 200,
      test_size: null,
      total_params: null,
      history: [
        {
          epoch: 1,
          train_loss: 0.2,
          val_loss: 0.25,
          train_accuracy: 0.95,
          val_accuracy: 0.93,
          seconds: 0.1,
        },
      ],
      test: null,
      preprocessing: { kind: 'demo' },
      classical: {
        estimator: 'SVC',
        label: 'Support vector classification',
        max_rows: 800,
        rows: 800,
        subsampled_from: 1600,
        fit_seconds: 0.12,
        initial_kernel: null,
        kernel: null,
        log_marginal_likelihood: null,
        support_vectors: 200,
        train: { loss: 0.2, accuracy: 0.95, rmse: null, r2: null },
      },
    },
    ...overrides,
  };
}
