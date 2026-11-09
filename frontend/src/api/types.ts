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

// --------------------------------------------------------------------------
// The own account and the legal pages
// --------------------------------------------------------------------------

/** The logged-in user's account (page /account) and how much the server stores for it. */
export interface Account extends User {
  last_login: string | null;
  last_active_at: string | null;
  /** When the terms of use were accepted at sign-up; null if the server had none then. */
  terms_accepted_at: string | null;
  terms_version: string | null;
  privacy_version: string | null;
  stored: {
    projects: number;
    memberships: number;
    comments: number;
    data_sources: number;
    data_preps: number;
    datasets: number;
    training_runs: number;
    api_tokens: number;
    connected_apps: number;
    /** Imported CSV files and training datasets together. */
    upload_bytes: number;
  };
}

export type LegalPageName = 'imprint' | 'privacy' | 'terms';

/** Which legal pages the server has (Markdown files the operator puts on it). */
export interface LegalInfo {
  pages: Record<LegalPageName, boolean>;
  terms_version: string | null;
  privacy_version: string | null;
  /** Sign-up needs `accept_terms: true` (the server has terms of use). */
  accept_terms_required: boolean;
  /** Automatic deletion after this many days without use; 0: off. */
  retention: { unused_uploads_days: number; inactive_accounts_days: number };
}

export interface LegalPage {
  page: LegalPageName;
  title: string;
  /** Markdown, rendered without raw HTML (see legal/markdown.tsx). */
  content: string;
  version: string;
  updated_at: string;
}

/** Days until a new API token expires; null means it never does. */
export type ApiTokenExpiry = 7 | 30 | 90 | 365;

/** A personal API token for AI assistants (MCP) and scripts. The secret is never listed. */
export interface ApiToken {
  id: number;
  name: string;
  /** The start of the token (e.g. "bdnn_Ab3dEf9h"), to recognise it. */
  prefix: string;
  read_only: boolean;
  created_at: string;
  last_used_at: string | null;
  expires_at: string | null;
}

export interface ApiTokenInput {
  name: string;
  read_only?: boolean;
  expires_in_days?: ApiTokenExpiry | null;
}

/** The answer to creating a token: the only time the server sends the token itself. */
export interface NewApiToken extends ApiToken {
  token: string;
}

// --------------------------------------------------------------------------
// OAuth: apps like the Claude apps connect to the MCP server by signing in
// --------------------------------------------------------------------------

/** A valid authorization request (/oauth/authorize), checked by the server. */
export interface OAuthRequest {
  /** As the app registered itself: the name and homepage are not verified. */
  client: { name: string; uri: string | null };
  /** Where the browser returns to after the decision. */
  redirect_host: string;
  /** The app asks to make changes (scope "write"), not only to read. */
  write_requested: boolean;
}

/** A rejected authorization request (HTTP 400), with the OAuth error code. */
export interface OAuthRequestError {
  error: string;
  error_description: string;
  /**
   * Set when the app itself can be told about the error. Without it, the app or its redirect
   * URI is unknown, so the browser must not be sent anywhere.
   */
  redirect_to?: string;
}

/** The answer to allowing or denying a request: where to send the browser back to. */
export interface OAuthDecision {
  redirect_to: string;
}

/** An app the user allowed to use their account; disconnecting it revokes its access. */
export interface OAuthConnection {
  id: number;
  client_name: string;
  client_uri: string | null;
  read_only: boolean;
  created_at: string;
  last_used_at: string | null;
}

// --------------------------------------------------------------------------
// Saved networks (projects) and sharing
// --------------------------------------------------------------------------

/** owner: one per project, manages sharing · editor: changes the network · viewer: reads it. */
export type ProjectRole = 'owner' | 'editor' | 'viewer';

/** The roles the owner can give to other people. */
export type MemberRole = Exclude<ProjectRole, 'owner'>;

/** 'view': anyone with the share link can view the network without an account. */
export type LinkAccess = 'none' | 'view';

export interface UserRef {
  id: number;
  username: string;
}

export interface Project {
  id: number;
  name: string;
  description: string;
  graph: GraphDocument;
  num_layers: number;
  created_at: string;
  updated_at: string;
  owner: UserRef;
  /** The role of the current user. */
  role: ProjectRole;
  /** Goes up with every change; saves send it to detect changes by others (HTTP 409). */
  version: number;
  /** Who made the last change. */
  updated_by: string | null;
  /** Has members or an active share link. */
  shared: boolean;
  link_access: LinkAccess;
  /** Only sent to the owner (null for everyone else and while the link is off). */
  share_token: string | null;
}

/** The answer to a save based on an old version (HTTP 409). */
export interface ProjectConflict {
  detail: string;
  /** The current state, including who changed it. */
  project: Project;
}

export interface ProjectMember {
  id: number;
  user: UserRef;
  role: MemberRole;
  created_at: string;
}

export interface ProjectMembers {
  owner: UserRef;
  members: ProjectMember[];
}

export interface ShareLink {
  link_access: LinkAccess;
  share_token: string | null;
}

/** A project as other objects refer to it. */
export interface ProjectRef {
  id: number;
  name: string;
}

export type ProjectDataKind = 'dataset' | 'prep';

/** A training dataset or data preparation shared in a project. */
export interface ProjectDataItem {
  id: number;
  name: string;
  /** Username of the owner. */
  owner: string;
  created_at: string;
  /** Datasets only. */
  kind?: 'csv' | 'images';
  /** Datasets only. */
  size_bytes?: number;
}

/** GET /api/projects/<id>/data/: what is shared in a project, by name. */
export interface ProjectData {
  datasets: ProjectDataItem[];
  preps: ProjectDataItem[];
}

/** A network opened through its share link (no login needed). */
export interface SharedNetwork {
  name: string;
  description: string;
  graph: GraphDocument;
  /** Username of the owner. */
  owner: string;
  updated_at: string;
}

// --------------------------------------------------------------------------
// History and comments of saved networks
// --------------------------------------------------------------------------

/**
 * auto: kept after a change (changes by the same person within ten minutes update the same
 * entry) · named: named by someone, kept for good · restore: an older version made current.
 */
export type VersionKind = 'auto' | 'named' | 'restore';

/** An entry in the history of a project (GET /api/projects/<id>/versions/, newest first). */
export interface ProjectVersion {
  id: number;
  /** The project's `version` this entry holds. */
  version: number;
  kind: VersionKind;
  /** The name of named versions; "Restored …" for restores; empty for automatic entries. */
  label: string;
  /** The network's name at that time. */
  name: string;
  num_layers: number;
  /** Username of who made the change (or named it); null if the account was deleted. */
  created_by: string | null;
  created_at: string;
  /** The last change merged into this entry. */
  updated_at: string;
}

/** One version with its content (GET /api/projects/<id>/versions/<vid>/). */
export interface ProjectVersionDetail extends ProjectVersion {
  description: string;
  graph: GraphDocument;
}

/** A comment on a project or on one of its layers; replies have a `parent` (one level). */
export interface ProjectComment {
  id: number;
  /** The first comment of the thread; null for the first comment itself. */
  parent: number | null;
  /** The layer the thread is about; empty for comments on the whole network. */
  node_id: string;
  text: string;
  author: UserRef;
  /** Set on the first comment of a thread. */
  resolved: boolean;
  /** Username of who resolved the thread. */
  resolved_by: string | null;
  created_at: string;
  updated_at: string;
}

export interface CommentInput {
  text: string;
  /** The layer to comment on (new threads only; replies belong to the layer of their thread). */
  node_id?: string;
  /** Reply to this thread. */
  parent?: number;
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
  /** The official test images: always the test set. */
  test_size: number;
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
  /** Built-in datasets: training and official test images. */
  train_size: number | null;
  test_size?: number | null;
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
  /**
   * Datasets made in the Data tab come with their own split: [train, validation, test] rows.
   * Training then uses it instead of `val_split` / `test_split`.
   */
  predefined_split?: number[] | null;
}

/** An uploaded dataset (CSV file or ZIP of image folders). */
export interface Dataset {
  id: number;
  name: string;
  kind: 'csv' | 'images';
  size_bytes: number;
  created_at: string;
  info: DatasetInfo;
  /** The data preparation (Data tab) this dataset was created from, if any. */
  prepared_from?: { id: number; name: string } | null;
  /** Username of the owner: datasets shared in the user's projects belong to others. */
  owner: string;
  /** Owned by the current user (only they can delete it and share it further). */
  mine: boolean;
  /** Projects it is shared in (those the current user can see), by name. */
  shared_in: ProjectRef[];
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

/**
 * A suggested change that makes the network fit the data: set `params` on the layer `node`.
 * Only the Input layer (`shape`, `dtype`) and the last Linear layer (`out_features`) are ever
 * changed.
 */
export interface CompatibilityFix {
  node: string;
  type: string;
  params: Params;
  /** The current values of `params`. */
  previous: Params;
  /** E.g. "Input shape 784 → 4". */
  summary: string;
}

export interface Compatibility {
  ok: boolean;
  errors: string[];
  warnings: string[];
  plan: TrainingPlan | null;
  /** Changes that make the network fit (applied together). Missing from older servers. */
  fixes?: CompatibilityFix[];
}

export type Optimizer = 'adam' | 'adamw' | 'sgd';

export interface TrainingConfig {
  epochs: number;
  batch_size: number;
  learning_rate: number;
  optimizer: Optimizer;
  weight_decay: number;
  /** Share of the data held out for validation (built-in datasets: of the training images). */
  val_split: number;
  /**
   * Share of the data held out as a test set, measured once after training (0: none). Not
   * used for built-in datasets: they are always tested on their official test images.
   * Missing in runs from before test sets existed.
   */
  test_split?: number;
  device: string;
  max_samples: number | null;
  seed: number | null;
  /**
   * Early stopping: end the training when the validation loss has not improved for this many
   * epochs (null: off). Missing in runs from before early stopping existed.
   */
  patience?: number | null;
  /**
   * Test and save the weights of the epoch with the lowest validation loss instead of those
   * of the last epoch (default true; missing in older runs, which kept the last epoch).
   */
  restore_best?: boolean;
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

/**
 * One trial of a search. The metrics are those of the epoch whose weights the trial keeps
 * (`weights_epoch`); while it runs, and for failed or stopped trials, of its latest epoch.
 */
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
  /** The epoch whose weights the trial keeps (missing in older runs). */
  weights_epoch?: number | null;
  /** Ended by early stopping. */
  stopped_early?: boolean;
  seconds: number | null;
}

/** The trial whose kept weights have the lowest validation loss; its weights are kept. */
export interface BestTrial {
  trial: number;
  params: TrialParams;
  val_loss: number | null;
  val_accuracy: number | null;
  val_rmse: number | null;
  weights_epoch?: number | null;
}

/** The kept weights measured on the test set. */
export interface TestResult {
  loss: number | null;
  accuracy: number | null;
  rmse: number | null;
}

/**
 * Metrics of the last finished epoch while a run is active, then of the epoch whose weights
 * are kept (of the best trial once a search has ended).
 */
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
  /** Metrics of the last finished epoch, at the end of the kept one. */
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
  /** The project (saved network) it was started in; its members see it. */
  project: number | null;
  /** Username of who started it. */
  started_by: string;
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
  /**
   * The epoch whose weights are tested and saved (the best one, or the last one without
   * `restore_best`); null before the epochs are over. Missing in runs of older versions.
   */
  weights_epoch?: number | null;
  /** True if early stopping ended the training (in a search: the best trial). */
  stopped_early?: boolean;
  /** The kept weights on the test set. */
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
  /** Train in this project (owner or editor of it): its members see the run. */
  project?: number;
}

// --------------------------------------------------------------------------
// Data preparation (the Data tab). The backend is `brickdnn.data` + `/api/data/`.
// --------------------------------------------------------------------------

/**
 * How a column is stored and analysed. numeric: numbers; boolean: 0/1; datetime: dates and
 * times; categorical: a limited set of labels; text: free text (long or mostly unique).
 */
export type ColumnKind = 'numeric' | 'boolean' | 'datetime' | 'categorical' | 'text';

export interface ColumnInfo {
  name: string;
  kind: ColumnKind;
}

/** GET /api/data/: whether data preparation works on this server (public). */
export interface DataStatus {
  enabled: boolean;
  /** False if disabled or pandas is missing; `reason` says why. */
  available: boolean;
  reason: string | null;
  max_upload_mb: number;
  max_rows: number;
  /** null: no limit. */
  max_columns: number | null;
  /** Rows in a preview sample unless the request asks for another number. */
  default_sample_rows: number;
  max_sample_rows: number;
  login_required: boolean;
}

/** How to read a CSV file; 'auto' means detect. Sent as form fields with the upload. */
export interface ImportOptions {
  delimiter: 'auto' | ',' | ';' | '\t' | '|';
  decimal: 'auto' | '.' | ',';
  header: 'auto' | 'yes' | 'no';
  encoding: 'auto' | 'utf-8' | 'cp1252' | 'latin-1';
  /** Cell values that mean "missing"; null: the defaults (NA, N/A, null, ? …). */
  missing_values: string[] | null;
}

/** What the CSV reader detected. */
export interface ImportReport {
  delimiter: string;
  decimal: string;
  header: boolean;
  encoding: string;
  rows: number;
  columns: number;
  warnings: string[];
}

/** An imported CSV file (raw data). Several sources can be combined in one preparation. */
export interface DataSource {
  id: number;
  name: string;
  filename: string;
  size_bytes: number;
  n_rows: number;
  n_columns: number;
  columns: (ColumnInfo & { missing: number })[];
  import_options: ImportOptions;
  report: ImportReport;
  created_at: string;
}

// ---- step catalog (GET /api/data/steps/) ---------------------------------

export type StepParamKind =
  | 'column'
  | 'columns'
  | 'source'
  | 'source_columns'
  | 'select'
  | 'multiselect'
  | 'number'
  | 'int'
  | 'bool'
  | 'string'
  | 'value'
  | 'list'
  | 'mapping';

export type CellValue = string | number | boolean | null;

export interface StepParamSpec {
  name: string;
  /**
   * column: one column name · columns: list of column names (often [] = all suitable
   * columns, see `empty_means`) · source: id of another DataSource · source_columns: column
   * names of the source chosen in `source_param` · select / multiselect: from `options` ·
   * number / int / bool / string · value: one cell value · list: list of cell values ·
   * mapping: {"old value": new value}.
   */
  kind: StepParamKind;
  label: string;
  default: unknown;
  help: string | null;
  optional: boolean;
  options?: { value: string; label: string }[];
  min?: number;
  max?: number;
  step?: number;
  /** column / columns: only offer columns of these kinds. */
  column_kinds?: ColumnKind[];
  /** columns: what an empty selection means, e.g. "All numeric columns". */
  empty_means?: string;
  /** Only shown (and only used) when param `param` has one of `values`. */
  visible_if?: { param: string; values: unknown[] };
  /** source_columns: the param that holds the source. */
  source_param?: string;
  /** mapping: the keys are column names (renaming) or cell values (replacing). */
  key_kind?: 'column' | 'value';
}

export type StepGroup =
  | 'combine'
  | 'columns'
  | 'rows'
  | 'missing'
  | 'outliers'
  | 'transform'
  | 'encode'
  | 'datetime'
  | 'target'
  | 'selection';

export interface StepSpec {
  type: string;
  label: string;
  group: StepGroup;
  description: string;
  /** Learns values (means, bounds, categories …) from the training rows only. */
  fitted: boolean;
  /** Combines data sources: must come before all other steps. */
  combine: boolean;
  /** Only changes training rows (e.g. class balancing); not applied to new data. */
  training_only: boolean;
  params: StepParamSpec[];
}

export interface StepCatalog {
  groups: { id: StepGroup; label: string }[];
  steps: StepSpec[];
}

// ---- preparations ---------------------------------------------------------

export interface PipelineStep {
  /** Unique within the preparation (made by the UI, e.g. "s7"). */
  id: string;
  type: string;
  enabled: boolean;
  params: Record<string, unknown>;
}

/** A step to add, before it gets an id (quick fixes, column actions). */
export interface NewStep {
  type: string;
  params: Record<string, unknown>;
}

export type PrepTask = 'auto' | 'classification' | 'regression';

export interface SplitSettings {
  /** Shares of all rows; val > 0 and val + test ≤ 0.9. */
  val: number;
  test: number;
  seed: number;
  /** Keep the class proportions in every part (classification only). */
  stratify: boolean;
}

export interface PrepSettings {
  /** The column to predict; all other columns are features. */
  target: string | null;
  task: PrepTask;
  split: SplitSettings;
}

/** A data preparation: a source, a list of steps and the target/split settings. */
export interface DataPrep {
  id: number;
  name: string;
  source: number;
  source_name: string;
  steps: PipelineStep[];
  settings: PrepSettings;
  /** Training datasets exported from this preparation, newest first. */
  datasets: { id: number; name: string; created_at: string }[];
  created_at: string;
  updated_at: string;
  /** Username of the owner. */
  owner: string;
  /**
   * owner: may do everything · editor (shared in a project the user edits): may change the
   * steps and settings (not the source) and export · viewer: may look, preview and download.
   */
  role: ProjectRole;
  /** Projects it is shared in (those the current user can see), by name. */
  shared_in: ProjectRef[];
}

export type DataPrepInput = Pick<DataPrep, 'name' | 'source'> &
  Partial<Pick<DataPrep, 'steps' | 'settings'>>;

// ---- preview (POST /api/data/preps/<id>/preview/) ---------------------------

export interface PreviewRequest {
  /** Unsaved steps/settings to preview; the saved ones if omitted. */
  steps?: PipelineStep[];
  settings?: PrepSettings;
  /** Show the data after this step (later steps are 'not_run'). */
  upto?: string | null;
  /** Rows in the sample (default DataStatus.default_sample_rows). */
  sample_rows?: number;
  /** Also profile the data before the cleaning steps (for before/after comparisons). */
  compare?: boolean;
}

export type StepStatus = 'ok' | 'error' | 'skipped' | 'disabled' | 'not_run';

export interface StepReport {
  id: string;
  type: string;
  status: StepStatus;
  /** One sentence, e.g. "Clipped 57 values (1.2 %) in 2 columns." Empty unless 'ok'. */
  summary: string;
  error: string | null;
  warnings: string[];
  /** Per-column facts, e.g. [{column: 'age', text: 'median 34'}]. */
  details: { column?: string; text: string }[];
  rows_before: number;
  rows_after: number;
  /** The columns after this step (before it for steps that did not run). */
  columns: ColumnInfo[];
  added: string[];
  removed: string[];
  /** What the step learned from the training rows. */
  fitted: Record<string, unknown> | null;
  duration_ms: number;
}

export interface HistogramData {
  /** Bin edges: counts[i] covers edges[i] … edges[i + 1] (the last bin includes its end). */
  edges: number[];
  counts: number[];
}

export interface NumericStats {
  mean: number | null;
  std: number | null;
  min: number | null;
  max: number | null;
  q01: number | null;
  q05: number | null;
  q25: number | null;
  median: number | null;
  q75: number | null;
  q95: number | null;
  q99: number | null;
  skew: number | null;
  kurtosis: number | null;
  zeros: number;
  negative: number;
  /** Values outside [q25 − 1.5·IQR, q75 + 1.5·IQR]. */
  iqr_outliers: number;
  /** Values with |z| > 3. */
  z_outliers: number;
}

export interface ColumnProfile {
  name: string;
  kind: ColumnKind;
  /** numeric: all values are whole numbers. */
  integer: boolean;
  /** Non-missing values. */
  count: number;
  missing: number;
  unique: number;
  /** numeric, boolean and datetime (datetime values in ms since the epoch). */
  stats: NumericStats | null;
  /** numeric and datetime (datetime edges in ms since the epoch). */
  histogram: HistogramData | null;
  /** categorical, text and boolean: most frequent values (booleans as "0"/"1"). */
  top: { value: string; count: number }[] | null;
  /** Non-missing values not listed in `top`. */
  other_count: number | null;
  /** categorical and text: mean length in characters. */
  mean_length: number | null;
  /**
   * Association with the target (absent for the target itself): pearson (|r|, numeric vs
   * numeric target), eta (correlation ratio, numeric vs classes or classes vs numeric) or
   * cramers_v (categories vs classes). 0 … 1.
   */
  target_association: { method: 'pearson' | 'eta' | 'cramers_v'; value: number } | null;
}

export interface CorrelationMatrix {
  method: 'pearson' | 'spearman';
  columns: string[];
  /** Symmetric, 1 on the diagonal, null where undefined (constant columns). */
  matrix: (number | null)[][];
}

export interface TargetProfile {
  column: string;
  task: 'classification' | 'regression';
  /** classification: class counts, in class index order. */
  classes: { value: string; count: number }[] | null;
  /** classification: largest class count / smallest class count. */
  imbalance_ratio: number | null;
  /** regression: distribution of the target. */
  histogram: HistogramData | null;
  stats: NumericStats | null;
}

export interface DataFix {
  type: string;
  params: Record<string, unknown>;
  /** Button text, e.g. "Impute 3 columns (median)". */
  label: string;
}

export interface DataWarning {
  /** e.g. high_missing, constant, high_cardinality, id_like, skewed, many_zeros,
   * duplicates, imbalanced, mixed_types, high_correlation. */
  code: string;
  severity: 'info' | 'warning';
  column: string | null;
  message: string;
  fix: DataFix | null;
}

export interface DataProfile {
  n_rows: number;
  n_columns: number;
  missing_cells: number;
  duplicate_rows: number;
  memory_bytes: number;
  /** [train, validation, test] rows, null before the split is assigned. */
  split_counts: number[] | null;
  /** Rows the correlations, associations and duplicate check are based on. */
  analysis_rows: number;
  columns: ColumnProfile[];
  /** Pearson and Spearman, for numeric and boolean columns (empty with fewer than 2). */
  correlations: CorrelationMatrix[];
  target: TargetProfile | null;
  warnings: DataWarning[];
}

export interface DataSample {
  /** Rows of the whole table. */
  n_total: number;
  /** Rows in this sample (in table order). */
  rows: number;
  seed: number;
  /** Row numbers in the table (0-based), one per sample row. */
  index: number[];
  /** Split of every sample row, null before the split is assigned. */
  split: ('train' | 'val' | 'test')[] | null;
  /** Numbers (booleans as 0/1), datetimes as ms since the epoch, strings, null = missing. */
  columns: (ColumnInfo & { values: (number | string | null)[] })[];
}

export interface ReadinessIssue {
  /**
   * The check, e.g. missing_values, categorical_features, numbers_as_text, leakage,
   * split_overlap, duplicate_rows, rare_classes, near_constant, redundant, skewed, scales,
   * few_rows (see backend/brickdnn/data/checks.py); step_failed for a failed step.
   */
  code: string;
  /** error: blocks the export; warning: likely hurts the network or its scores; info: a hint. */
  severity: 'error' | 'warning' | 'info';
  /** The first affected column; null for rows or the whole table. */
  column: string | null;
  /** All affected columns (empty for rows or the whole table). */
  columns: string[];
  message: string;
  fix: DataFix | null;
}

/** Can the prepared data be used for training? */
export interface Readiness {
  ready: boolean;
  task: 'classification' | 'regression' | null;
  target: string | null;
  rows: number;
  num_features: number;
  num_classes: number | null;
  class_names: string[] | null;
  split_counts: number[] | null;
  issues: ReadinessIssue[];
}

export interface PreviewResult {
  steps: StepReport[];
  /** Resolved task ('auto' decided). */
  task: 'classification' | 'regression' | null;
  /** The data after the last step that ran. */
  profile: DataProfile;
  /** The data before the cleaning steps (after combining, with the split); with `compare`. */
  before: DataProfile | null;
  sample: DataSample;
  readiness: Readiness;
  /** Pipeline-level notes, e.g. "Stratified split not possible: …". */
  warnings: string[];
  duration_ms: number;
}

export type PrepDownloadFormat = 'csv' | 'recipe' | 'python';
