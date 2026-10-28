export type ParamKind = 'int' | 'float' | 'bool' | 'select' | 'ints';
export type ParamValue = number | boolean | string | number[] | null;
export type Params = Record<string, ParamValue>;

export interface ParamSpec {
  name: string;
  label: string;
  kind: ParamKind;
  default: ParamValue;
  help: string;
  options: string[];
  min: number | null;
  max: number | null;
  /** If true the value may be left empty and is inferred from the input shape. */
  auto: boolean;
  /** Fixed number of values for `ints` params (e.g. 2 for a 2-D kernel size). */
  length: number | null;
  show_if: { param: string; value: ParamValue } | null;
}

export interface LayerSpec {
  type: string;
  label: string;
  category: string;
  description: string;
  params: ParamSpec[];
  min_inputs: number;
  max_inputs: number | null;
}

export interface LayerCatalog {
  categories: string[];
  layers: LayerSpec[];
}

export interface GraphNode {
  id: string;
  type: string;
  params: Params;
  name?: string;
  position?: { x: number; y: number };
}

export interface GraphEdge {
  id?: string;
  source: string;
  target: string;
}

export interface GraphDocument {
  version: 1;
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface Issue {
  node: string | null;
  message: string;
}

export interface NodeAnalysis {
  type: string;
  output_shape: number[] | null;
  dtype: 'float' | 'long' | null;
  num_params: number | null;
  params: Params | null;
}

export interface Analysis {
  valid: boolean;
  errors: Issue[];
  warnings: Issue[];
  nodes: Record<string, NodeAnalysis>;
  inputs: string[];
  outputs: string[];
  total_params: number | null;
}

export interface Example {
  id: string;
  name: string;
  description: string;
  order: number;
  graph: GraphDocument;
}

export interface GenerateOptions {
  class_name?: string;
  include_main?: boolean;
  include_training?: boolean;
  filename?: string;
}

export interface GenerateResult {
  code: string;
  filename: string;
}

export interface User {
  id: number;
  username: string;
  email: string;
  date_joined: string;
}

export interface Tokens {
  access: string;
  refresh: string;
}

export interface Project {
  id: number;
  name: string;
  description: string;
  graph: GraphDocument;
  num_layers: number;
  created_at: string;
  updated_at: string;
}

// --------------------------------------------------------------------------
// Training (runs on the machine of the backend, see /api/training/)
// --------------------------------------------------------------------------

export type RunStatus = 'starting' | 'preparing' | 'running' | 'completed' | 'stopped' | 'failed';

export interface TrainingDevice {
  id: string;
  label: string;
}

export interface BuiltinDataset {
  id: string;
  name: string;
  description: string;
  /** Channels, height, width. */
  sample_shape: number[];
  num_classes: number;
  class_names: string[];
  train_size: number;
  val_size: number;
  download_mb: number;
  /** False when torchvision is missing on the server. */
  available: boolean;
}

/** Capabilities of the server: GET /api/training/. */
export interface TrainingStatus {
  enabled: boolean;
  available: boolean;
  reason: string | null;
  torch_version: string | null;
  torchvision: boolean;
  devices: TrainingDevice[];
  default_device: string | null;
  builtin_datasets: BuiltinDataset[];
  max_upload_mb: number;
  max_concurrent_runs: number;
  login_required: boolean;
  /** Limits of the hyperparameter search (missing on servers without it). */
  tuning?: TuningLimits;
}

export type DatasetKind = 'random' | 'builtin' | 'csv' | 'images';

/** What the server knows about a dataset without loading it. */
export interface DatasetInfo {
  kind: DatasetKind;
  name: string;
  num_samples: number | null;
  builtin: string | null;
  sample_shape: number[] | null;
  train_size: number | null;
  val_size: number | null;
  num_features: number | null;
  /** Feature columns followed by the target column. */
  columns: string[];
  has_header: boolean | null;
  delimiter: string | null;
  target_kind: 'integer' | 'number' | 'text' | null;
  target_range: number[] | null;
  class_names: string[] | null;
  class_counts: number[] | null;
  subdir: string | null;
  num_classes: number | null;
}

/** An uploaded dataset (CSV file or ZIP of image folders). */
export interface Dataset {
  id: number;
  name: string;
  kind: 'csv' | 'images';
  size_bytes: number;
  created_at: string;
  info: DatasetInfo;
}

export type DatasetRef =
  { kind: 'random' } | { kind: 'builtin'; id: string } | { kind: 'upload'; id: number };

export type TrainingTask = 'classification' | 'regression' | 'demo';

export interface TrainingPlan {
  task: TrainingTask;
  summary: string;
  inputs: { shape: number[]; dtype: 'float' | 'long'; vocab: number | null }[];
  targets: ({ kind: 'class'; num_classes: number } | { kind: 'value'; shape: number[] })[];
  image: {
    channels: number;
    height: number;
    width: number;
    layout: 'chw' | 'hw' | 'flat';
  } | null;
  num_classes: number | null;
  class_names: string[] | null;
  notes: string[];
}

export interface Compatibility {
  ok: boolean;
  errors: string[];
  warnings: string[];
  plan: TrainingPlan | null;
}

export type Optimizer = 'adam' | 'adamw' | 'sgd';

export interface TrainingConfig {
  epochs: number;
  batch_size: number;
  learning_rate: number;
  optimizer: Optimizer;
  weight_decay: number;
  /**
   * Share of the data held out for validation. Built-in datasets without a test set validate
   * on their official test images and ignore it.
   */
  val_split: number;
  /**
   * Share of the data held out as a test set, measured once after training (0: none).
   * Built-in datasets use their official test images when it is above 0. Missing in runs
   * from before test sets existed.
   */
  test_split?: number;
  device: string;
  max_samples: number | null;
  seed: number | null;
}

// --------------------------------------------------------------------------
// Hyperparameter search
// --------------------------------------------------------------------------

export type TuningMethod = 'random' | 'grid';

/** Settings a search can vary, in display order. */
export type TunableSetting = 'learning_rate' | 'batch_size' | 'optimizer' | 'weight_decay';

/** Numbers between `min` and `max` (both > 0), drawn log-uniformly. */
export interface TuningRange {
  min: number;
  max: number;
}

/** What a search tries. Settings that are missing keep their value from the config. */
export interface TuningSpace {
  learning_rate?: TuningRange | number[];
  batch_size?: number[];
  optimizer?: Optimizer[];
  weight_decay?: TuningRange | number[];
}

/**
 * A hyperparameter search: random search draws `trials` combinations, grid search tries
 * every combination of the listed values. As stored on a run, `trials` is the planned
 * number of trials.
 */
export interface TuningConfig {
  method: TuningMethod;
  trials: number;
  space: TuningSpace;
}

/** Search settings sent to the API (`trials` is ignored by grid search). */
export interface TuningInput {
  method: TuningMethod;
  trials?: number;
  space: TuningSpace;
}

/** GET /api/training/ `tuning`. */
export interface TuningLimits {
  methods: TuningMethod[];
  tunable: TunableSetting[];
  max_trials: number;
  max_values: number;
  default: TuningConfig;
}

/** Values of the tuned settings of one trial. */
export interface TrialParams {
  learning_rate?: number;
  batch_size?: number;
  optimizer?: Optimizer;
  weight_decay?: number;
}

export type TrialStatus = 'running' | 'completed' | 'failed' | 'stopped';

/** One trial of a search. The metrics are those of its last (or latest) epoch. */
export interface TrialEntry {
  /** Counts from 1. */
  trial: number;
  params: TrialParams;
  status: TrialStatus;
  /** Why the trial failed, e.g. "The loss became NaN in epoch 2. …". */
  message: string;
  epochs_done: number;
  train_loss: number | null;
  val_loss: number | null;
  val_accuracy: number | null;
  val_rmse: number | null;
  seconds: number | null;
}

/** The trial with the lowest final validation loss; its weights are kept. */
export interface BestTrial {
  trial: number;
  params: TrialParams;
  val_loss: number | null;
  val_accuracy: number | null;
  val_rmse: number | null;
}

/** The final model measured on the test set. */
export interface TestResult {
  loss: number | null;
  accuracy: number | null;
  rmse: number | null;
}

/** Metrics of the last finished epoch (of the best trial once a search has ended). */
export interface RunMetrics {
  train_loss: number | null;
  val_loss: number | null;
  val_accuracy: number | null;
  val_rmse: number | null;
  /** Test set results, null without a test set or before the end (missing in old runs). */
  test_loss?: number | null;
  test_accuracy?: number | null;
  test_rmse?: number | null;
}

export interface BestEpoch {
  epoch: number;
  val_loss: number | null;
  val_accuracy: number | null;
}

export interface RunSummary {
  id: string;
  name: string;
  status: RunStatus;
  message: string;
  dataset: { kind: 'random' | 'builtin' | 'upload'; id: string | number | null; name: string };
  config: TrainingConfig;
  task: TrainingTask | null;
  /** Planned number of epochs (per trial in a search). */
  epochs: number;
  /** Finished epochs (of the running trial in a search). */
  epochs_done: number;
  /** Metrics of the last finished epoch. */
  metrics: RunMetrics | null;
  best: BestEpoch | null;
  /** Set for a hyperparameter search. */
  tuning: TuningConfig | null;
  /** Finished trials of a search (completed or failed). */
  trials_done: number;
  best_trial: BestTrial | null;
  has_weights: boolean;
  stop_requested: boolean;
  total_params: number | null;
  created_at: string;
  finished_at: string | null;
}

/** One finished epoch. Accuracies are null for regression, which reports val_rmse instead. */
export interface HistoryEntry {
  epoch: number;
  train_loss: number | null;
  val_loss: number | null;
  train_accuracy?: number | null;
  val_accuracy?: number | null;
  val_rmse?: number | null;
  seconds?: number | null;
}

export interface RunProgress {
  epoch: number;
  step: number;
  steps: number;
  loss: number | null;
  samples_per_second: number | null;
}

/** Live state written by the training process (state.json). Times are unix seconds. */
export interface RunState {
  status: RunStatus;
  message: string;
  updated_at: number;
  started_at: number | null;
  finished_at: number | null;
  device: string | null;
  device_label: string | null;
  task: TrainingTask | null;
  num_classes: number | null;
  class_names: string[] | null;
  train_size: number | null;
  val_size: number | null;
  /** Null without a test set. The new keys below are missing in runs of older versions. */
  test_size?: number | null;
  epochs: number;
  steps_per_epoch: number | null;
  total_params: number | null;
  /** Epochs of the running trial in a search; of the best trial once it has ended. */
  history: HistoryEntry[];
  progress: RunProgress | null;
  best: BestEpoch | null;
  /** The final model on the test set. */
  test?: TestResult | null;
  weights: boolean;
  /** How inputs were prepared; needed to reuse the weights. Shape defined by the engine. */
  preprocessing: Record<string, unknown> | null;
  log: string[];
  tuning?: TuningConfig | null;
  trials_total?: number | null;
  /** The trial that is running (from 1), null outside of trials. */
  trial?: number | null;
  /** Every started trial, in order. */
  trials?: TrialEntry[];
  best_trial?: number | null;
}

export interface RunDetail extends RunSummary {
  graph: GraphDocument;
  plan: TrainingPlan | null;
  state: RunState | null;
}

export interface RunInput {
  name: string;
  graph: GraphDocument;
  dataset: DatasetRef;
  config: Partial<TrainingConfig>;
  /** A hyperparameter search instead of a single training. */
  tuning?: TuningInput;
}
