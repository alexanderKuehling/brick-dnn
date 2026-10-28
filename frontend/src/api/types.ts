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
  /** Share of the data held out for validation (ignored for built-in datasets). */
  val_split: number;
  device: string;
  max_samples: number | null;
  seed: number | null;
}

export interface RunMetrics {
  train_loss: number | null;
  val_loss: number | null;
  val_accuracy: number | null;
  val_rmse: number | null;
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
  /** Planned number of epochs. */
  epochs: number;
  epochs_done: number;
  /** Metrics of the last finished epoch. */
  metrics: RunMetrics | null;
  best: BestEpoch | null;
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
  epochs: number;
  steps_per_epoch: number | null;
  total_params: number | null;
  history: HistoryEntry[];
  progress: RunProgress | null;
  best: BestEpoch | null;
  weights: boolean;
  /** How inputs were prepared; needed to reuse the weights. Shape defined by the engine. */
  preprocessing: Record<string, unknown> | null;
  log: string[];
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
}
