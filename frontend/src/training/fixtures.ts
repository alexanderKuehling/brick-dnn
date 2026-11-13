/** Test data for the training panel (only imported by tests). */
import type {
  Analysis,
  ClassificationEvaluation,
  Dataset,
  GraphDocument,
  RegressionEvaluation,
  RunDetail,
  RunPreviews,
  RunQueue,
  RunSummary,
  ThresholdCurve,
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

// --------------------------------------------------------------------------
// Segmentation (a class for every pixel)
// --------------------------------------------------------------------------

const SHAPE_CLASSES = ['background', 'square', 'circle', 'triangle'];
const SHAPE_COLORS = ['#1a1a19', '#3987e5', '#d95926', '#199e70'];

/** A server with segmentation: the generated dataset Shapes and the limits. */
export const SEGMENTATION_STATUS: TrainingStatus = {
  ...STATUS,
  builtin_datasets: [
    { ...STATUS.builtin_datasets[0], task: 'classification' },
    {
      id: 'shapes',
      name: 'Shapes',
      description:
        'Segmentation: squares, circles and triangles on noisy backgrounds, with a class for ' +
        'every pixel. Generated, RGB 64×64.',
      task: 'segmentation',
      sample_shape: [3, 64, 64],
      num_classes: 4,
      class_names: SHAPE_CLASSES,
      train_size: 400,
      test_size: 100,
      download_mb: 0,
      available: true,
    },
  ],
  segmentation: {
    max_classes: 32,
    max_images: 20000,
    max_input_side: 512,
    batch_memory_mb: 900,
    losses: ['cross_entropy', 'cross_entropy_dice'],
  },
};

/** An uploaded ZIP file with images and masks. */
export function segmentationDataset(overrides: Partial<Dataset> = {}): Dataset {
  const name = overrides.name ?? 'cells';
  return {
    id: 9,
    name,
    kind: 'segmentation',
    size_bytes: 2_400_000,
    created_at: '2026-11-10T10:00:00Z',
    info: {
      kind: 'segmentation',
      name,
      num_samples: 120,
      builtin: null,
      sample_shape: [3, 256, 256],
      train_size: null,
      test_size: null,
      num_features: null,
      columns: [],
      has_header: null,
      delimiter: null,
      target_kind: null,
      target_range: null,
      class_names: ['background', 'cell', 'nucleus'],
      class_counts: null,
      subdir: null,
      num_classes: 3,
      task: 'segmentation',
      segmentation: {
        mask_format: 'indices',
        ignore_index: null,
        class_pixel_share: [0.81, 0.154, 0.036],
        image_sizes: { most_common: [256, 256], min: [256, 256], max: [256, 256], different: 1 },
        unmatched_images: 2,
        unmatched_masks: 0,
        unmatched_examples: ['images/a.png', 'images/b.png'],
        mask_size_mismatches: 0,
      },
    },
    owner: 'ada',
    mine: true,
    shared_in: [],
    ...overrides,
  };
}

/**
 * A finished segmentation run (a U-Net on Shapes): 3 epochs, the weights of epoch 3 are kept,
 * measured on the test images, with eight preview images.
 */
export function segmentationDetail(overrides: Partial<RunDetail> = {}): RunDetail {
  const base = detail({
    name: 'U-Net',
    status: 'completed',
    message: 'Finished: validation mean IoU 78.2%, pixel accuracy 97.2%; test mean IoU 77.8%, …',
    dataset: { kind: 'builtin', id: 'shapes', name: 'Shapes' },
    config: { ...CONFIG, epochs: 3, batch_size: 16, loss: 'cross_entropy' },
    task: 'segmentation',
    epochs: 3,
    epochs_done: 3,
    metrics: {
      train_loss: 0.14,
      val_loss: 0.08,
      val_accuracy: null,
      val_rmse: null,
      val_iou: 0.7821,
      val_dice: 0.8702,
      val_pixel_accuracy: 0.9718,
      test_loss: 0.09,
      test_accuracy: null,
      test_rmse: null,
      test_iou: 0.7779,
      test_dice: 0.8671,
      test_pixel_accuracy: 0.9694,
    },
    best: { epoch: 3, val_loss: 0.08, val_accuracy: null, val_iou: 0.7821 },
    has_weights: true,
    total_params: 1925700,
    finished_at: '2026-10-28T10:01:30Z',
  });
  const epoch = (n: number, loss: number, iou: number, dice: number, accuracy: number) => ({
    epoch: n,
    train_loss: loss + 0.06,
    val_loss: loss,
    train_accuracy: null,
    val_accuracy: null,
    val_iou: iou,
    val_dice: dice,
    val_pixel_accuracy: accuracy,
    seconds: 8,
  });
  const split = (iou: number, dice: number, accuracy: number, classIou: number[]) => ({
    iou,
    dice,
    pixel_accuracy: accuracy,
    class_iou: classIou,
    class_dice: classIou.map((value) => (2 * value) / (1 + value)),
  });
  return {
    ...base,
    plan: {
      task: 'segmentation',
      summary:
        'Segmentation into 4 classes at 64×64 pixels (Shapes, 400 training and 100 test images).',
      inputs: [{ shape: [3, 64, 64], dtype: 'float', vocab: null }],
      targets: [{ kind: 'mask', num_classes: 4, binary: false }],
      image: { channels: 3, height: 64, width: 64, layout: 'chw' },
      num_classes: 4,
      class_names: SHAPE_CLASSES,
      notes: [],
      segmentation: {
        binary: false,
        ignore_index: null,
        fixed_bytes: 30_811_200,
        sample_bytes: 17_118_003,
      },
    },
    state: {
      ...base.state!,
      status: 'completed',
      message: base.message,
      finished_at: 1761645690,
      device: 'cpu',
      device_label: 'CPU',
      task: 'segmentation',
      num_classes: 4,
      class_names: SHAPE_CLASSES,
      train_size: 320,
      val_size: 80,
      test_size: 100,
      epochs: 3,
      steps_per_epoch: 20,
      total_params: 1925700,
      history: [
        epoch(1, 0.46, 0.2163, 0.2319, 0.865),
        epoch(2, 0.19, 0.5401, 0.6372, 0.9331),
        epoch(3, 0.08, 0.7821, 0.8702, 0.9718),
      ],
      progress: null,
      best: { epoch: 3, val_loss: 0.08, val_accuracy: null, val_iou: 0.7821 },
      weights_epoch: 3,
      stopped_early: false,
      test: {
        loss: 0.09,
        accuracy: null,
        rmse: null,
        r2: null,
        ...split(0.7779, 0.8671, 0.9694, [0.981, 0.749, 0.809, 0.573]),
      },
      weights: true,
      preprocessing: {
        kind: 'image',
        task: 'segmentation',
        channels: 3,
        height: 64,
        width: 64,
        layout: 'chw',
        mean: [0.5556, 0.5575, 0.5419],
        std: [0.1903, 0.192, 0.2007],
        class_names: SHAPE_CLASSES,
        mask: { resize: 'nearest', binary: false, ignore_index: null },
      },
      log: [],
      segmentation: {
        loss: 'cross_entropy',
        binary: false,
        ignore_index: null,
        class_names: SHAPE_CLASSES,
        colors: SHAPE_COLORS,
        validation: { epoch: 3, ...split(0.7821, 0.8702, 0.9718, [0.984, 0.802, 0.78, 0.563]) },
        test: split(0.7779, 0.8671, 0.9694, [0.981, 0.749, 0.809, 0.573]),
        previews: { count: 2, epoch: 3, split: 'validation' },
      },
    },
    ...overrides,
  };
}

/** The preview images of `segmentationDetail` (two samples; the pictures are 1×1 PNGs). */
export function previews(overrides: Partial<RunPreviews> = {}): RunPreviews {
  const pixel =
    'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==';
  const sample = (index: number, iou: number, accuracy: number, classes: number[]) => ({
    index,
    image: { file: `${index}_image.png`, url: pixel },
    truth: { file: `${index}_truth.png`, url: pixel },
    prediction: { file: `${index}_prediction.png`, url: pixel },
    overlay: { file: `${index}_overlay.png`, url: pixel },
    iou,
    pixel_accuracy: accuracy,
    classes,
  });
  return {
    epoch: 3,
    split: 'validation',
    width: 64,
    height: 64,
    class_names: SHAPE_CLASSES,
    colors: SHAPE_COLORS,
    ignore_color: '#8a8f98',
    samples: [sample(0, 0.9206, 0.9744, [0, 1, 3]), sample(1, 0.4512, 0.9012, [0, 2, 3])],
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// The final evaluation of a run (state.evaluation)
// ---------------------------------------------------------------------------

/** Three classes; validation rows 30 (10 per class), test rows 20. */
export function classificationEvaluation(
  overrides: Partial<ClassificationEvaluation> = {}
): ClassificationEvaluation {
  return {
    task: 'classification',
    epoch: 3,
    trial: null,
    train: { loss: 0.2, accuracy: 0.94 },
    class_names: ['cat', 'dog', 'bird'],
    matrix_classes_max: 50,
    val: {
      size: 30,
      loss: 0.31,
      accuracy: 0.8,
      balanced_accuracy: 0.8,
      macro_f1: 0.7968,
      weighted_f1: 0.7968,
      log_loss: 0.31,
      top_k_accuracy: null,
      per_class: {
        precision: [0.75, 0.7778, 0.9],
        recall: [0.9, 0.7, 0.8],
        f1: [0.8182, 0.7368, 0.8471],
        support: [10, 10, 10],
      },
      // rows: true class, columns: predicted class
      confusion_matrix: [
        [9, 1, 0],
        [2, 7, 1],
        [1, 1, 8],
      ],
      most_confused: [
        { true: 1, predicted: 0, count: 2 },
        { true: 0, predicted: 1, count: 1 },
      ],
    },
    test: {
      size: 20,
      loss: 0.35,
      accuracy: 0.75,
      balanced_accuracy: 0.5167,
      macro_f1: 0.4889,
      weighted_f1: 0.6667,
      log_loss: 0.35,
      top_k_accuracy: null,
      per_class: {
        precision: [0.8, 0.6, null],
        recall: [0.8, 0.75, 0],
        f1: [0.8, 0.6667, 0],
        support: [10, 8, 2],
      },
      confusion_matrix: [
        [8, 2, 0],
        [2, 6, 0],
        [0, 2, 0],
      ],
      most_confused: [{ true: 0, predicted: 1, count: 2 }],
    },
    ...overrides,
  };
}

/** A threshold curve from true and false positives per threshold (thresholds ascending). */
export function thresholdCurve(
  positives: number,
  negatives: number,
  thresholds: number[],
  tp: number[],
  fp: number[],
  aucs: { roc: number; pr: number }
): ThresholdCurve {
  const ratio = (a: number, b: number) => (b > 0 ? a / b : null);
  return {
    positive_class: 1,
    positives,
    negatives,
    roc_auc: aucs.roc,
    pr_auc: aucs.pr,
    points: {
      threshold: thresholds,
      precision: tp.map((t, i) => ratio(t, t + fp[i])),
      recall: tp.map((t) => ratio(t, positives)),
      fpr: fp.map((f) => ratio(f, negatives)),
      tp,
      fp,
      tn: fp.map((f) => negatives - f),
      fn: tp.map((t) => positives - t),
    },
  };
}

const THRESHOLDS = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1];

/**
 * Two classes (ok, fraud): validation 10 fraud and 90 ok rows, test 5 and 45. At 0.5 the
 * validation rows find 4 of 10 frauds; the best F1 is at 0.3, 80% recall at 0.2.
 */
export function binaryEvaluation(
  overrides: Partial<ClassificationEvaluation> = {}
): ClassificationEvaluation {
  const base = classificationEvaluation();
  const split = (size: number, matrix: number[][], curve: ThresholdCurve) => ({
    ...base.val!,
    size,
    per_class: {
      precision: [0.9, 0.67],
      recall: [0.98, 0.4],
      f1: [0.94, 0.5],
      support: [matrix[0][0] + matrix[0][1], matrix[1][0] + matrix[1][1]],
    },
    confusion_matrix: matrix,
    most_confused: [],
    threshold_curve: curve,
  });
  return {
    ...base,
    class_names: ['ok', 'fraud'],
    val: split(
      100,
      [
        [88, 2],
        [6, 4],
      ],
      thresholdCurve(10, 90, THRESHOLDS, [10, 10, 9, 7, 4, 3, 0], [90, 40, 20, 6, 2, 0, 0], {
        roc: 0.912,
        pr: 0.634,
      })
    ),
    test: split(
      50,
      [
        [44, 1],
        [3, 2],
      ],
      thresholdCurve(5, 45, THRESHOLDS, [5, 5, 5, 4, 2, 1, 0], [45, 20, 10, 6, 1, 0, 0], {
        roc: 0.871,
        pr: 0.552,
      })
    ),
    binary: {
      positive_class: 1,
      recall_target: 0.8,
      source: 'val',
      best_f1: {
        threshold: 0.3,
        precision: 0.5385,
        recall: 0.7,
        f1: 0.6087,
        fpr: 0.0667,
        tp: 7,
        fp: 6,
        tn: 84,
        fn: 3,
      },
      recall: {
        threshold: 0.2,
        precision: 0.3103,
        recall: 0.9,
        f1: 0.4615,
        fpr: 0.2222,
        tp: 9,
        fp: 20,
        tn: 70,
        fn: 1,
      },
    },
    ...overrides,
  };
}

export function regressionEvaluation(
  overrides: Partial<RegressionEvaluation> = {}
): RegressionEvaluation {
  return {
    task: 'regression',
    epoch: 3,
    trial: null,
    train: { loss: 0.18, accuracy: null },
    val: {
      size: 4,
      loss: 0.2,
      mae: 30.5,
      mse: 1703.2,
      rmse: 41.27,
      r2: 0.823,
      mape: 0.0412,
      median_absolute_error: 25.1,
      max_error: 98.4,
      residuals: { q05: -70.2, q25: -20.1, median: 1.5, q75: 22.4, q95: 80.3 },
      sample: { actual: [100, 200, 300, 400], predicted: [110, 190, 320, 380] },
    },
    test: {
      size: 3,
      loss: 0.22,
      mae: 33.1,
      mse: 1927.2,
      rmse: 43.9,
      r2: 0.796,
      mape: null,
      median_absolute_error: 28,
      max_error: 101.2,
      residuals: { q05: -80, q25: -25, median: 0, q75: 25, q95: 85 },
      sample: { actual: [0, 150, 250], predicted: [20, 140, 260] },
    },
    ...overrides,
  };
}
