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

/** What flows along a connection: data (a tensor) or the kernel of a Gaussian process. */
export type OutputKind = 'tensor' | 'kernel';

/** neural: PyTorch layer · classical: kernel or classical model (scikit-learn) · shared: Input. */
export type LayerFamily = 'neural' | 'classical' | 'shared';

/** One kind of incoming connection of a brick (e.g. "features" and "kernel" of a GP). */
export interface InputSlot {
  name: string;
  label: string;
  kind: OutputKind;
  /** Connections of this kind the brick needs. */
  min: number;
  /** Connections of this kind it accepts; null: any number. */
  max: number | null;
}

export interface LayerSpec {
  type: string;
  label: string;
  category: string;
  description: string;
  params: ParamSpec[];
  /** All incoming connections together. */
  min_inputs: number;
  max_inputs: number | null;
  /** The fields below are missing from older servers (see builder/kinds.ts for defaults). */
  family?: LayerFamily;
  output_kind?: OutputKind;
  /** [] for bricks without inputs. */
  input_slots?: InputSlot[];
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

export type ClassicalTask = 'regression' | 'classification';

export interface NodeAnalysis {
  type: string;
  /** Null for kernels and classifiers (the classes come from the data) and on errors. */
  output_shape: number[] | null;
  dtype: 'float' | 'long' | null;
  /**
   * Trainable parameters; kernels count the hyperparameters a Gaussian process optimises.
   * Null when unknown before training (SVR, SVC, gradient boosting).
   */
  num_params: number | null;
  params: Params | null;
  /** What its outgoing connections carry (missing from older servers: tensor). */
  output_kind?: OutputKind;
  /** Kernel bricks and Gaussian processes: the kernel as scikit-learn code. */
  kernel?: string | null;
  /** Classical models: what they predict. */
  task?: ClassicalTask | null;
}

/** The classical model of a network (Gaussian process, support vector machine, gradient boosting). */
export interface ClassicalModelInfo {
  node: string;
  type: string;
  estimator: string;
  task: ClassicalTask;
  /** Gaussian processes: the kernel they start from (the default one without kernel bricks). */
  kernel: string | null;
}

export interface Analysis {
  valid: boolean;
  errors: Issue[];
  warnings: Issue[];
  nodes: Record<string, NodeAnalysis>;
  inputs: string[];
  outputs: string[];
  /** Null when a layer has an unknown count (SVR, SVC, gradient boosting) or on errors. */
  total_params: number | null;
  /** classical: the network contains kernels or a classical model (missing: neural). */
  family?: 'neural' | 'classical';
  classical?: ClassicalModelInfo | null;
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

/** The answer of the password login when the account has two-factor authentication on. */
export interface TwoFactorChallenge {
  two_factor_required: true;
  /** Signed, valid for `expires_in` seconds: send it with the code. */
  challenge: string;
  expires_in: number;
}

export type LoginResult = Tokens | TwoFactorChallenge;

export function isTwoFactorChallenge(result: LoginResult): result is TwoFactorChallenge {
  return (result as TwoFactorChallenge).two_factor_required === true;
}

export interface TwoFactorState {
  enabled: boolean;
  enabled_at: string | null;
  recovery_codes_left: number;
}

/** The start of the setup: shown once, never again. */
export interface TwoFactorSetup {
  secret: string;
  /** For the QR code. */
  otpauth_uri: string;
}

/** Two-factor authentication is on: the recovery codes (shown once) and new tokens. */
export interface TwoFactorEnabled extends Tokens {
  recovery_codes: string[];
  state: TwoFactorState;
}

export interface RecoveryCodes {
  recovery_codes: string[];
  state: TwoFactorState;
}

export interface PasswordResetLink {
  valid: boolean;
  username: string | null;
}

// --------------------------------------------------------------------------
// The own account and the legal pages
// --------------------------------------------------------------------------

/** The logged-in user's account (page /account) and how much the server stores for it. */
export interface Account extends User {
  /** The address was confirmed with a link sent to it (and has not changed since). */
  email_verified: boolean;
  email_verified_at: string | null;
  /** Whether the server sends e-mail at all. */
  email_sending: boolean;
  /** Whether a forgotten password could be reset by e-mail (needs a confirmed address). */
  password_reset_available: boolean;
  two_factor: TwoFactorState;
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
    /** Entries of the agent activity log. */
    agent_actions: number;
    /** Imported CSV files and training datasets together. */
    upload_bytes: number;
  };
}

export type LegalPageName = 'imprint' | 'privacy' | 'terms' | 'refund';

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
  /** Monthly compute budget of the token in CPU minutes; null: none. Missing on older servers. */
  budget_cpu_minutes?: number | null;
  /** CPU minutes its trainings and predictions used this month. */
  used_cpu_minutes?: number;
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
  /** Monthly compute budget of the app in CPU minutes; null: none. Missing on older servers. */
  budget_cpu_minutes?: number | null;
  used_cpu_minutes?: number;
}

// --------------------------------------------------------------------------
// Usage and budgets (compute is CPU time, counted per calendar month in UTC)
// --------------------------------------------------------------------------

/** This month's usage of the web app, one API token or one connected app. */
export interface UsageSource {
  source: 'web' | 'token' | 'app';
  /** The token's or connection's id (null for the web app). */
  id: number | null;
  name: string;
  /** The token was revoked or the app disconnected; its usage still counts. */
  deleted: boolean;
  used_cpu_minutes: number;
  budget_cpu_minutes: number | null;
  /** In `Usage.prices.currency`; null without a price. */
  cost: number | null;
}

/** A plan (subscription tier) of the server; null limits are unlimited. */
export interface Plan {
  key: string;
  name: string;
  /** Per month, in `currency`, incl. VAT (with paid plans, Paddle charges it). */
  price: number | null;
  currency: string | null;
  cpu_hours_per_month: number | null;
  storage_gb: number | null;
  cpu_minutes_per_run: number | null;
  concurrent_trainings: number | null;
  /** Trainings of an account that may wait in the queue (null: the server's default). */
  queued_trainings?: number | null;
  /** For the training queue: runs of plans with a higher priority start first. */
  priority: number;
  /** Compute beyond the allowance, if the plan offers it (per CPU hour). */
  overage_price_per_cpu_hour: number | null;
}

/** GET /api/account/usage/: this month's usage, the limits and what is left. */
export interface Usage {
  period: { start: string; resets_at: string };
  /** The plan the limits come from. Missing on older servers. */
  plan?: Plan;
  /** No limit of the plan applies (the usage is still measured). */
  unlimited?: boolean;
  /** The plans with prices, when the server offers several (else empty). */
  plans?: Plan[];
  /** Where plans can be bought, on a server that sells them (the pricing page); else null. */
  upgrade_url?: string | null;
  /** Compute beyond the monthly allowance (paid plans), switched on by the user. */
  overage?: {
    available: boolean;
    enabled: boolean;
    price_per_cpu_hour: number | null;
    currency: string;
    /** The user's own cap in CPU hours (null: none). */
    cap_cpu_hours: number | null;
    used_cpu_minutes: number;
    cost: number | null;
  };
  compute: {
    unit: 'cpu_minutes';
    used: number;
    /** The account's limit this month: the allowance or the own budget, whichever is lower. */
    limit: number | null;
    remaining: number | null;
    /** What the caller can still use (with its token's or app's budget). */
    available: number | null;
    /** The server's monthly allowance for the account (null: unlimited). */
    allowance: number | null;
    /** The budget the user set for the account (null: none). */
    own_budget: number | null;
    /** The most one training may use (null: unlimited). */
    run_limit: number | null;
    reserved_by_active_runs: number;
    by_kind: { training: number; prediction: number };
    by_source: UsageSource[];
  };
  storage: {
    used_bytes: number;
    limit_bytes: number | null;
    remaining_bytes: number | null;
    by_kind: { data_sources: number; datasets: number; training_runs: number };
  };
  prices: {
    currency: string | null;
    per_cpu_hour: number | null;
    per_gb_month: number | null;
    compute_cost: number | null;
    compute_limit_cost: number | null;
    storage_cost_per_month: number | null;
  };
  /** With an API token or app: its own usage and budget (null in the web app). */
  caller: {
    source: 'token' | 'app';
    id: number;
    name: string;
    used_cpu_minutes: number;
    budget_cpu_minutes: number | null;
    remaining_cpu_minutes: number | null;
  } | null;
}

/** A rough estimate of the CPU time a planned training will use. */
export interface CostEstimate {
  cpu_minutes: number;
  low_cpu_minutes: number;
  high_cpu_minutes: number;
  /** With a configured price: what `cpu_minutes` cost, in `currency`. */
  cost: number | null;
  currency: string | null;
  /** E.g. "48,000 training and 22,000 validation/test samples, 10 epochs, …". */
  basis: string;
  note: string;
  details: Record<string, unknown>;
}

/** Whether the budgets allow a planned training, and how much it may use. */
export interface BudgetCheck {
  allowed: boolean;
  /** Why a start would be refused. */
  reason: string | null;
  /** E.g. the estimate exceeds what the run may use: it would probably be stopped. */
  warning: string | null;
  /** The most the run may use (null: unlimited). */
  run_limit_cpu_minutes: number | null;
  limit_reason: 'budget' | 'run_limit' | null;
  /** What is left this month of the budgets that apply (null: no budget applies). */
  remaining_cpu_minutes?: number | null;
  resets_at: string;
  /** budget_exceeded | estimate_too_high | storage_quota_exceeded, when not allowed. */
  code?: string;
}

/** The answer of the API when a budget or quota does not allow a request (HTTP 403). */
export interface BudgetErrorBody {
  detail: string;
  code: 'budget_exceeded' | 'estimate_too_high' | 'storage_quota_exceeded';
}

// --------------------------------------------------------------------------
// Agent activity: what assistants and scripts did with the user's tokens and apps
// --------------------------------------------------------------------------

/** "token:<ApiToken id>" or "app:<OAuthConnection id>". */
export type AgentKey = `${'token' | 'app'}:${number}`;

/** The API token or connected app an action was made with. */
export interface AgentRef {
  kind: 'token' | 'app';
  id: number;
  key: AgentKey;
  name: string;
  /** The start of a personal token (apps get a new token every hour: null). */
  prefix: string | null;
}

export type AgentTargetKind = 'project' | 'run' | 'dataset' | 'data_source' | 'data_prep';

/** Something an action touched. `name` is missing when the user cannot see it (any more). */
export interface AgentTarget {
  kind: AgentTargetKind;
  /** A number; a text for training runs (and for ids the agent got wrong). */
  id: number | string;
  name?: string;
}

type ArgumentValue = string | number | boolean | null | (number | boolean | null)[];

/** A summarised argument: sizes and counts instead of contents, secrets "[redacted]". */
export type AgentArgument = ArgumentValue | Record<string, ArgumentValue>;

/** One entry of the activity log: an MCP tool call or a REST request with an API token. */
export interface AgentAction {
  id: number;
  /** When it finished (it started `duration_ms` before). */
  created_at: string;
  source: 'mcp' | 'api';
  agent: AgentRef;
  user_agent: string;
  /** The MCP tool, or the REST route (e.g. "training-run-stop") with `method`. */
  tool: string;
  method: string;
  /** Changes something rather than only reading. */
  writes: boolean;
  /** pending: waited for the user's approval; approved: approved and done; denied. */
  outcome: AgentOutcome;
  /** What the agent was told went wrong ("" when it worked). */
  error: string;
  /** REST requests: the HTTP status. */
  status_code: number | null;
  duration_ms: number;
  /** A sentence, e.g. 'Started training “GP 1” on dataset “houses.csv” (#3)'. */
  summary: string;
  arguments: Record<string, AgentArgument>;
  targets: AgentTarget[];
  /** The network it was about, if any. */
  project: number | null;
  /** The approval request it belongs to, if any. Missing on older servers. */
  approval?: number | null;
}

export type AgentOutcome = 'ok' | 'error' | 'pending' | 'approved' | 'denied';

export interface AgentActivityQuery {
  agent?: AgentKey;
  tool?: string;
  outcome?: AgentOutcome;
  project?: number;
  /** Older entries than this id (the next page). */
  before?: number;
  /** Newer entries than this id (polling). */
  after?: number;
  limit?: number;
}

export interface AgentActivityPage {
  /** Newest first. */
  results: AgentAction[];
  /** More entries match: older ones, or with `after` more new ones than `limit`. */
  has_more: boolean;
  /** The user's newest entry, whatever the filters: poll with `after` this. 0: none. */
  latest_id: number;
  /** How long entries are kept; 0: this server keeps no log. */
  retention_days: number;
  /** Requests waiting for the user's approval, with every poll. Missing on older servers. */
  approvals?: PendingApprovals;
}

// --------------------------------------------------------------------------
// Approvals: actions of agents that wait for the user's OK
// --------------------------------------------------------------------------

/** What a token or app may do without asking. */
export interface ApprovalRule {
  /** Trainings estimated above this many CPU minutes ask (0: every one; null: none). */
  training_cpu_minutes: number | null;
  /** Sharing in projects and links that work without logging in ask. */
  publishing: boolean;
  /** Deleting networks, runs, datasets and data asks. */
  deleting: boolean;
}

export interface ApprovalRules {
  /** What new tokens and apps get. */
  default: ApprovalRule;
  builtin_default: ApprovalRule;
  /** By "token:<id>" and "app:<id>". */
  agents: Partial<Record<AgentKey, ApprovalRule>>;
  gated_tools: Record<ApprovalCategory, string[]>;
  expiry_hours: number;
}

export type ApprovalCategory = 'training' | 'publishing' | 'deleting';
export type ApprovalStatus = 'pending' | 'approved' | 'executed' | 'failed' | 'denied' | 'expired';

export interface ApprovalEstimate {
  cpu_minutes: number | null;
  low_cpu_minutes: number | null;
  high_cpu_minutes: number | null;
  cost: number | null;
  currency: string | null;
}

export interface ApprovalRequest {
  id: number;
  status: ApprovalStatus;
  category: ApprovalCategory;
  /** The tool the agent called, and what the server does once approved. */
  tool: string;
  action: string;
  agent: AgentRef & { active: boolean };
  /** 'Start training “GP 1” on dataset “houses.csv” (#3)'. */
  summary: string;
  /** Why it needs approval. */
  reason: string;
  estimate: ApprovalEstimate | null;
  limit_cpu_minutes: number | null;
  /** Trainings: the network ("5 layers (…), 4 connections") and the data. */
  network: string | null;
  dataset: string | null;
  /** Publishing: what would become visible, and to whom. */
  exposes: string | null;
  warning: string | null;
  targets: AgentTarget[];
  arguments: Record<string, AgentArgument>;
  created_at: string;
  expires_at: string;
  decided_at: string | null;
  finished_at: string | null;
  /** The user's message with a denial. */
  message: string;
  /** Why it failed once approved. */
  error: string;
  /** What was done once approved. */
  done: string | null;
}

export interface PendingApprovals {
  pending_count: number;
  /** Newest first. */
  pending: ApprovalRequest[];
}

export interface ApprovalList {
  results: ApprovalRequest[];
  pending_count: number;
}

/** A token or app that acted, for filters and the API tokens page. */
export interface ActivityAgent extends AgentRef {
  /** The token or connection still exists (it was not revoked or disconnected). */
  active: boolean;
  actions: number;
  errors: number;
  last_at: string;
  last_id: number;
}

export interface AgentActivitySummary {
  /** The most recently active first. */
  agents: ActivityAgent[];
  tools: { tool: string; actions: number }[];
  total: number;
  latest_id: number;
  latest_at: string | null;
  retention_days: number;
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
  kind?: 'csv' | 'images' | 'segmentation';
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

/** `queued`: waiting for a training worker (servers with a training queue). */
export type RunStatus =
  'queued' | 'starting' | 'preparing' | 'running' | 'completed' | 'stopped' | 'failed';

/**
 * Where a queued run waits (`queue` of a run). The start is a rough estimate from the CPU-time
 * estimates of the trainings ahead and the worker's slots.
 */
export interface RunQueue {
  /** Its place among all waiting trainings, from 1, and how many wait. */
  position: number;
  waiting: number;
  /** Null when it cannot be told (e.g. without a worker). */
  estimated_wait_seconds: number | null;
  estimated_start_at: string | null;
  /** False: no training worker is running; the run waits until one is back. */
  worker_online: boolean;
  /** `account`: it waits for a running training of the same account to finish. */
  blocked_by: 'account' | null;
  priority: number;
  /** One line on what the plan means for the place, e.g. "Paid plans start first." */
  hint: string | null;
}

/** The training queue of the server (`queue` of GET /api/training/). */
export interface QueueStatus {
  /** A training worker is running. */
  online: boolean;
  workers: number;
  /** Trainings the workers run at the same time. */
  slots: number;
  running: number;
  waiting: number;
  /** Set when no worker is running. */
  message: string | null;
  /** Logged in: what the plan means for the place in the queue (null: nothing). */
  priority_hint: string | null;
}

/** Where a training started now would wait (`queue` of the training check). */
export interface QueuePreview extends RunQueue {
  running: number;
  message: string | null;
}

export interface TrainingDevice {
  id: string;
  label: string;
}

export interface BuiltinDataset {
  id: string;
  name: string;
  description: string;
  /**
   * `segmentation`: a class for every pixel (Shapes, generated on the server); otherwise one
   * label per image. Missing on older servers.
   */
  task?: 'classification' | 'segmentation';
  /** Channels, height, width. */
  sample_shape: number[];
  num_classes: number;
  class_names: string[];
  train_size: number;
  /** The official test images: always the test set. */
  test_size: number;
  download_mb: number;
  /** False when torchvision is missing on the server (generated datasets do not need it). */
  available: boolean;
}

/** What a segmentation dataset and training may be on this server. */
export interface SegmentationLimits {
  max_classes: number;
  max_images: number;
  /** The largest Input height and width. */
  max_input_side: number;
  /** The memory a batch may need (every layer's output for all its images). */
  batch_memory_mb: number;
  losses: SegmentationLoss[];
}

/** The loss of a segmentation: the cross-entropy per pixel, optionally plus a Dice term. */
export type SegmentationLoss = 'cross_entropy' | 'cross_entropy_dice';

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
  /**
   * With a training worker: new trainings wait in a queue instead of being refused while
   * the server is busy. Null (or missing on older servers): the web server trains itself.
   */
  queue?: QueueStatus | null;
  login_required: boolean;
  /** Limits of the hyperparameter search (missing on servers without it). */
  tuning?: TuningLimits;
  /**
   * Classical models (scikit-learn). `available` and `reason` above are about PyTorch;
   * classical models train without it. Missing on older servers.
   */
  classical?: ClassicalCapabilities;
  /** Segmentation (a class for every pixel). Missing on servers without it. */
  segmentation?: SegmentationLimits;
}

export interface ClassicalCapabilities {
  available: boolean;
  reason: string | null;
  sklearn_version: string | null;
  /** Training rows each model type is fitted on at most (larger sets are subsampled). */
  max_rows: Record<string, number>;
  /** Hyperparameter search for classical models (not yet). */
  tuning: boolean;
}

export type DatasetKind = 'random' | 'builtin' | 'csv' | 'images' | 'segmentation';

/** What the inspection of a segmentation dataset found (images with one mask each). */
export interface SegmentationInfo {
  /** How the masks are read: black and white, class indices, grey levels or colours. */
  mask_format: 'binary' | 'indices' | 'levels' | 'colors';
  mask_values?: unknown[] | null;
  /** 255 if the masks mark pixels that are ignored. */
  ignore_index: number | null;
  ignored_share?: number;
  /** The share of the pixels of every class. */
  class_pixel_share: number[];
  image_sizes?: {
    /** Width, height. */
    most_common: number[];
    min: number[];
    max: number[];
    different: number;
  };
  unmatched_images?: number;
  unmatched_masks?: number;
  unmatched_examples?: string[];
  /** Masks with another size than their image (they are resized). */
  mask_size_mismatches?: number;
}

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
   * Training then uses it instead of `val_split` / `test_split`. Segmentation datasets with
   * train/, val/ (and test/) folders have one too: their images.
   */
  predefined_split?: number[] | null;
  /** `segmentation` for data with a class for every pixel. */
  task?: 'segmentation' | null;
  segmentation?: SegmentationInfo | null;
}

/** An uploaded dataset (CSV file, ZIP of image folders or ZIP of images with masks). */
export interface Dataset {
  id: number;
  name: string;
  kind: 'csv' | 'images' | 'segmentation';
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

export type TrainingTask = 'classification' | 'regression' | 'segmentation' | 'demo';

export interface TrainingPlan {
  task: TrainingTask;
  summary: string;
  inputs: { shape: number[]; dtype: 'float' | 'long'; vocab: number | null }[];
  targets: (
    | { kind: 'class'; num_classes: number }
    | { kind: 'value'; shape: number[] }
    /** Segmentation; `binary`: one output channel, the score of class 1. */
    | { kind: 'mask'; num_classes: number; binary: boolean }
  )[];
  image: {
    channels: number;
    height: number;
    width: number;
    layout: 'chw' | 'hw' | 'flat';
  } | null;
  num_classes: number | null;
  class_names: string[] | null;
  notes: string[];
  /** Classical models: what is fitted (missing for neural networks and older runs). */
  classical?: ClassicalPlan | null;
  /**
   * Segmentation: a training step needs about `fixed_bytes + batch size × sample_bytes` of
   * memory (null otherwise).
   */
  segmentation?: {
    binary: boolean;
    ignore_index: number | null;
    fixed_bytes: number;
    sample_bytes: number;
  } | null;
}

export interface ClassicalPlan {
  node: string;
  type: string;
  estimator: string;
  label: string;
  task: ClassicalTask;
  kernel: string | null;
  scale: string;
  max_rows: number;
}

/**
 * A suggested change that makes the network fit the data: set `params` on the layer `node`.
 * Only the Input layer (`shape`, `dtype`) and the last layer are ever changed: `out_features`
 * of a Linear layer, for segmentation `out_channels` of a convolution.
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
  /** The estimated CPU time of the run (null if it cannot be estimated). */
  estimate?: CostEstimate | null;
  budget?: BudgetCheck | null;
  /** Servers with a training queue: where a training started now would wait. */
  queue?: QueuePreview | null;
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
  /** Segmentation only (missing in older runs). */
  loss?: SegmentationLoss;
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
  /** Regression only (missing in older runs). */
  val_r2?: number | null;
  /** Segmentation only. */
  val_iou?: number | null;
  val_dice?: number | null;
  val_pixel_accuracy?: number | null;
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
  /** Regression only (missing in older runs). */
  val_r2?: number | null;
  /** Segmentation only. */
  val_iou?: number | null;
  weights_epoch?: number | null;
}

/** The kept weights measured on the test set. */
export interface TestResult {
  loss: number | null;
  accuracy: number | null;
  rmse: number | null;
  /** Coefficient of determination: regression only (missing in older runs). */
  r2?: number | null;
  /** Segmentation only: mean IoU and Dice over the classes, and the share of right pixels. */
  iou?: number | null;
  dice?: number | null;
  pixel_accuracy?: number | null;
  /** Per class; null for a class that occurs neither in the masks nor the predictions. */
  class_iou?: (number | null)[];
  class_dice?: (number | null)[];
}

/**
 * Metrics of the last finished epoch while a run is active, then of the epoch whose weights
 * are kept (of the best trial once a search has ended).
 */
export interface RunMetrics {
  train_loss: number | null;
  /** The kept epoch on the training rows (classification; missing in older runs). */
  train_accuracy?: number | null;
  /** The validation values are null for a run without validation data. */
  val_loss: number | null;
  val_accuracy: number | null;
  val_rmse: number | null;
  /** Test set results, null without a test set or before the end (missing in old runs). */
  test_loss?: number | null;
  test_accuracy?: number | null;
  test_rmse?: number | null;
  /**
   * Coefficient of determination: regression only, null otherwise and when the targets of the
   * split are all equal (missing in older runs).
   */
  val_r2?: number | null;
  test_r2?: number | null;
  /**
   * Segmentation only (other runs do not have the keys): mean IoU and Dice over the classes
   * that occur, and the share of pixels with the right class.
   */
  val_iou?: number | null;
  val_dice?: number | null;
  val_pixel_accuracy?: number | null;
  test_iou?: number | null;
  test_dice?: number | null;
  test_pixel_accuracy?: number | null;
}

export interface BestEpoch {
  epoch: number;
  val_loss: number | null;
  val_accuracy: number | null;
  /** Segmentation only. */
  val_iou?: number | null;
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
  /**
   * classical: a scikit-learn model, fitted once (epochs 1); ensemble: finished runs combined.
   * Missing on older servers.
   */
  family?: 'neural' | 'classical' | 'ensemble';
  /** The API token or connected app that started it (null: the web app). */
  started_via?: { source: 'token' | 'app'; name: string | null } | null;
  usage?: RunUsage;
  /** While the run is `queued`: its position and estimated start. Null otherwise. */
  queue?: RunQueue | null;
  /**
   * Binary classification: predictions are the second class when its probability is at least
   * this (default 0.5). Null for other runs; missing on older servers.
   */
  decision_threshold?: number | null;
  /** "ensemble": finished runs combined into one model (see `ensemble`). */
  kind?: RunKind;
  /** An ensemble's method and members; null for other runs. */
  ensemble?: EnsembleInfo | null;
}

export type RunKind = 'training' | 'ensemble';

/** average: the weighted average of the members' predictions; stacking: a fitted model. */
export type EnsembleMethod = 'average' | 'stacking';

/** A run an ensemble combines. */
export interface EnsembleMember {
  /** Null once the run was deleted. */
  id: string | null;
  name: string;
  /** Its share of the weighted average (the weights add up to 1); null for stacking. */
  weight: number | null;
  family: 'neural' | 'classical' | null;
  /** "Neural network" or the classical model's name. */
  model: string | null;
  status: RunStatus | null;
  /** The user can open its run. */
  visible: boolean;
  deleted: boolean;
}

export interface EnsembleInfo {
  method: EnsembleMethod;
  method_label: string;
  members: EnsembleMember[];
  /** Why it cannot predict any more (a member was deleted); null if it can. */
  broken: string | null;
}

/** Headline metrics of one split, for the comparison of the members with the ensemble. */
export interface EnsembleSplitMetrics {
  size: number;
  accuracy?: number | null;
  balanced_accuracy?: number | null;
  macro_f1?: number | null;
  log_loss?: number | null;
  /** Two classes only. */
  roc_auc?: number | null;
  pr_auc?: number | null;
  rmse?: number | null;
  mae?: number | null;
  r2?: number | null;
}

export type EnsembleSplit = 'val' | 'test' | 'training';

/** One row of the comparison: a member (`member`: its index) or the ensemble (null). */
export interface EnsembleComparisonRow {
  name: string;
  member: number | null;
  val?: EnsembleSplitMetrics | null;
  test?: EnsembleSplitMetrics | null;
  training?: EnsembleSplitMetrics | null;
}

/** What an ensemble's state.json says about the combination. */
export interface EnsembleState {
  method: EnsembleMethod;
  weights: number[] | null;
  /** Stacking: the fitted model as numbers. */
  meta: Record<string, unknown> | null;
  /** Stacking: the split its model was fitted on. */
  fitted_on: 'val' | null;
  /** Measured on rows it was fitted on: stacking without test rows, or training rows. */
  optimistic: boolean;
  splits: EnsembleSplit[];
  rows: Partial<Record<EnsembleSplit, number>>;
  comparison: EnsembleComparisonRow[];
}

export interface EnsembleInput {
  members: string[];
  method?: EnsembleMethod;
  /** Average only: one weight per member (scaled to add up to 1). */
  weights?: number[];
  name?: string;
  /** A network the user owns or edits: its members see the ensemble. */
  project?: number | null;
}

/** Why a stopped run stopped: the user, a used-up budget or the limit per training. */
export type StopReason = 'user' | 'budget' | 'run_limit';

export interface RunUsage {
  /** CPU minutes counted so far. */
  cpu_minutes: number;
  /** The most it may use (its budget or the limit per training); null: unlimited. */
  limit_cpu_minutes: number | null;
  /** The estimate made before it started. */
  estimated_cpu_minutes: number | null;
  stop_reason: StopReason | null;
}

/**
 * One finished epoch. Accuracies are null for regression, which reports val_rmse and val_r2
 * instead.
 */
export interface HistoryEntry {
  epoch: number;
  train_loss: number | null;
  val_loss: number | null;
  train_accuracy?: number | null;
  val_accuracy?: number | null;
  val_rmse?: number | null;
  /** Coefficient of determination; null when the validation targets are all equal. */
  val_r2?: number | null;
  /** Segmentation only. */
  val_iou?: number | null;
  val_dice?: number | null;
  val_pixel_accuracy?: number | null;
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
  family?: 'neural' | 'classical' | 'ensemble';
  /** An ensemble: its method, weights and the comparison with its members. */
  ensemble?: EnsembleState | null;
  /** What the fit of a classical model found (null for neural networks). */
  classical?: ClassicalRunState | null;
  /** Segmentation: the results per class and the preview images, once the weights are saved. */
  segmentation?: SegmentationRunState | null;
  /**
   * Completed classification and regression runs: the kept weights evaluated once more on the
   * validation and the test rows. Missing in runs of older versions.
   */
  evaluation?: RunEvaluation | null;
}

/** Precision, recall, F1 and support per class (in the order of `class_names`). */
export interface PerClassMetrics {
  /** Null: the class was never predicted. */
  precision: (number | null)[];
  /** Null: the class has no rows in the split. */
  recall: (number | null)[];
  /** Null: neither. */
  f1: (number | null)[];
  support: number[];
}

export interface ClassificationSplit {
  size: number;
  /** The run's loss (cross-entropy). */
  loss: number | null;
  accuracy: number | null;
  /** Mean recall over the classes that occur. */
  balanced_accuracy: number | null;
  macro_f1: number | null;
  weighted_f1: number | null;
  log_loss: number | null;
  /** {"3": 0.97, "5": 0.99}; only for more classes than k. */
  top_k_accuracy: Record<string, number> | null;
  per_class: PerClassMetrics;
  /** Counts; rows: true class, columns: predicted class. Null above matrix_classes_max. */
  confusion_matrix: number[][] | null;
  /** The largest off-diagonal cells (class indices). */
  most_confused: { true: number; predicted: number; count: number }[];
  /** Two classes only (missing in older runs): see ThresholdCurve. */
  threshold_curve?: ThresholdCurve;
}

/**
 * Binary classification: the split at a few hundred thresholds, ascending; a row is predicted
 * positive (class `positive_class`) when its probability of that class is at least the
 * threshold. The thresholds are the same for every split of a run.
 */
export interface ThresholdCurve {
  positive_class: number;
  positives: number;
  negatives: number;
  /** Null without rows of both classes. */
  roc_auc: number | null;
  /** Average precision; null without positive rows. */
  pr_auc: number | null;
  points: {
    threshold: number[];
    /** Null where nothing is predicted positive. */
    precision: (number | null)[];
    recall: (number | null)[];
    /** False-positive rate. */
    fpr: (number | null)[];
    tp: number[];
    fp: number[];
    tn: number[];
    fn: number[];
  };
}

/** One threshold with what it gives on a split. */
export interface ThresholdPoint {
  threshold: number;
  precision: number | null;
  recall: number | null;
  f1: number | null;
  fpr: number | null;
  tp: number;
  fp: number;
  tn: number;
  fn: number;
}

/** Binary classification: where the thresholds come from and the two suggestions. */
export interface BinaryAnalysis {
  positive_class: number;
  recall_target: number;
  /** The split the thresholds and suggestions come from (validation, else test, else training). */
  source: 'val' | 'test' | 'training';
  /** The threshold with the best F1 score; null without positive rows. */
  best_f1: ThresholdPoint | null;
  /** The highest threshold whose recall is at least recall_target. */
  recall: ThresholdPoint | null;
}

export interface RegressionSplit {
  size: number;
  /** The run's loss (MSE of the standardised target). */
  loss: number | null;
  mae: number | null;
  mse: number | null;
  rmse: number | null;
  r2: number | null;
  /** A share (0.04 = 4 %); null when a target is close to 0. */
  mape: number | null;
  median_absolute_error?: number | null;
  max_error?: number | null;
  /** Quantiles of predicted − actual. */
  residuals?: Record<'q05' | 'q25' | 'median' | 'q75' | 'q95', number | null>;
  /** At most 500 rows of the split. */
  sample?: { actual: (number | null)[]; predicted: (number | null)[] };
}

interface EvaluationBase {
  /** The epoch of the kept weights (classical models: 1). */
  epoch: number | null;
  /** Hyperparameter search: the best trial. */
  trial: number | null;
  /** The kept epoch on the training rows. */
  train: { loss: number | null; accuracy: number | null } | null;
}

export interface ClassificationEvaluation extends EvaluationBase {
  task: 'classification';
  class_names: string[];
  matrix_classes_max: number;
  /** Null for a run without validation rows. */
  val: ClassificationSplit | null;
  test: ClassificationSplit | null;
  /**
   * Only for a run without validation and test rows: the model on its training rows
   * (optimistic: it has seen them). Missing in older runs.
   */
  training?: ClassificationSplit | null;
  /** Two classes only (missing in older runs). */
  binary?: BinaryAnalysis;
}

export interface RegressionEvaluation extends EvaluationBase {
  task: 'regression';
  /** Null for a run without validation rows. */
  val: RegressionSplit | null;
  test: RegressionSplit | null;
  /** Only for a run without validation and test rows, see ClassificationEvaluation. */
  training?: RegressionSplit | null;
}

export type RunEvaluation = ClassificationEvaluation | RegressionEvaluation;

/** The results of the kept weights of a segmentation on one split, with the values per class. */
export interface SegmentationSplit {
  /** Validation only: the epoch of the kept weights. */
  epoch?: number | null;
  iou: number | null;
  dice: number | null;
  pixel_accuracy: number | null;
  class_iou: (number | null)[];
  class_dice: (number | null)[];
}

export interface SegmentationRunState {
  loss: SegmentationLoss;
  /** One output channel (two classes). */
  binary: boolean;
  ignore_index: number | null;
  class_names: string[];
  /** One colour per class, as in the preview images. */
  colors: string[];
  /** Null if the kept weights are not those of a finished epoch. */
  validation: SegmentationSplit | null;
  test: SegmentationSplit | null;
  /**
   * How many preview images there are (GET …/runs/<id>/previews/), or null. `split`:
   * "validation", or "test" / "training" for a run without validation data.
   */
  previews: { count: number; epoch: number | null; split: string } | null;
}

/** One picture of a preview: its file name and, embedded, a data URL. */
export interface PreviewImage {
  file: string;
  url?: string;
}

/** A validation image of a segmentation run with what the kept weights predict for it. */
export interface PreviewSample {
  index: number;
  image: PreviewImage;
  /** The true mask and the predicted one: one colour per class. */
  truth: PreviewImage;
  prediction: PreviewImage;
  /** The prediction drawn over the image. */
  overlay: PreviewImage;
  /** Mean IoU over the classes in its mask or prediction. */
  iou: number | null;
  pixel_accuracy: number | null;
  /** The classes in its mask or prediction. */
  classes: number[];
}

/** GET /api/training/runs/<id>/previews/. */
export interface RunPreviews {
  epoch: number | null;
  split: string;
  /** Size of the pictures (large inputs are scaled down). */
  width: number;
  height: number;
  class_names: string[];
  colors: string[];
  ignore_color: string;
  samples: PreviewSample[];
}

/** The metrics of a fitted classical model on some rows. */
export interface ClassicalMetrics {
  loss: number | null;
  accuracy: number | null;
  rmse: number | null;
  r2: number | null;
}

export interface ClassicalRunState {
  estimator: string;
  /** E.g. "Gaussian process regression". */
  label: string;
  /** Training rows the model is fitted on at most on this server. */
  max_rows: number;
  /** Training rows it was fitted on. */
  rows: number | null;
  /** Training rows before the row limit applied (null: not subsampled). */
  subsampled_from: number | null;
  fit_seconds: number | null;
  /** Gaussian processes: the kernel the fit started from and the fitted one. */
  initial_kernel: string | null;
  /** A classifier with more than two classes: one per class ("setosa: …; versicolor: …"). */
  kernel: string | null;
  log_marginal_likelihood: number | null;
  /** SVMs: the number of support vectors. */
  support_vectors: number | null;
  /** Gradient boosting: the boosting iterations (early stopping may end sooner; older servers: none). */
  iterations?: number | null;
  /** Gradient boosting: at most this many iterations (its max_iter). */
  max_iterations?: number | null;
  /** The metrics on the training rows. */
  train: ClassicalMetrics | null;
  /** Gaussian processes: kernel hyperparameters the fit left at a bound (older servers: none). */
  at_bounds?: KernelBound[] | null;
}

/** A kernel hyperparameter that ended at (or very near) one of its bounds after the fit. */
export interface KernelBound {
  /** The kernel brick (null: the default kernel of a Gaussian process without kernel bricks). */
  node: string | null;
  /** Its name in the network, e.g. "white_1". */
  name: string;
  type: string;
  /** E.g. "noise_level", with its label "Noise level". */
  parameter: string;
  label: string;
  value: number;
  bound: number;
  side: 'lower' | 'upper';
  /** Classifiers with one kernel per class: the classes whose kernel hit the bound. */
  classes: string[] | null;
  /** New parameters for the brick: custom bounds that move this bound. */
  suggestion: Params | null;
  /** "Noise level of white_1 ended at its lower bound 1e-05: the model may be overfitting; …" */
  message: string;
}

export interface RunDetail extends RunSummary {
  graph: GraphDocument;
  plan: TrainingPlan | null;
  state: RunState | null;
  /** In the answer to starting it: it will probably be stopped by its compute limit. */
  budget_warning?: string | null;
}

// --------------------------------------------------------------------------
// Predictions with a trained run (POST /api/training/runs/<id>/predict/)
// --------------------------------------------------------------------------

/** A cell value of an input row: numbers, text, true/false; null for a missing value. */
export type PredictionCell = string | number | boolean | null;

/** A row from the data with its actual target value, to try the model with. */
export interface PredictionSample {
  values: Record<string, PredictionCell>;
  target?: PredictionCell;
}

/** A column of the source table of a data preparation (raw rows), from its training rows. */
export interface RawColumn {
  name: string;
  kind: ColumnKind;
  /** Text columns: the values of the training rows, most frequent first. */
  categories?: string[];
  /** There are more categories than listed. */
  more_categories?: boolean;
  /** Numbers and dates (ISO text): the range of the training rows. */
  min?: number | string | null;
  max?: number | string | null;
  /** Numbers that were all whole numbers. */
  integer?: boolean;
  /** The training rows have missing values here. */
  missing?: boolean;
  /** false: the preparation drops the column before anything reads it. */
  used?: boolean;
}

/** GET /api/training/runs/<id>/predict-form/: what the prediction panel shows. */
export interface PredictionForm {
  available: boolean;
  /** Why the run cannot predict (no weights, unknown preprocessing …). */
  reason: string | null;
  kind: 'tabular' | 'image' | 'demo' | null;
  task: 'classification' | 'regression' | null;
  class_names: string[] | null;
  /** Rows per request; more rows are sent in several requests. */
  max_rows: number;
  /** Rows of a pasted or uploaded CSV file at most. */
  max_form_rows: number;
  max_top_k: number;
  /** Set when a compute budget is used up (the server refuses predictions then). */
  budget_note: string | null;
  /** Tabular: the target column. */
  target?: string | null;
  gaussian_process?: boolean;
  /** Tabular: the feature columns of the training dataset (standardised by the server). */
  prepared?: {
    columns: { name: string; mean: number; std: number }[];
    samples: PredictionSample[];
  };
  /** Runs on a dataset from a data preparation: rows in the columns of its source table. */
  raw?: {
    prep: { id: number; name: string } | null;
    source: string | null;
    target: string | null;
    columns: RawColumn[];
    samples: PredictionSample[];
    /** Steps that only change the training rows and do not apply to new rows. */
    skipped_steps: string[];
    /** Tables joined by the preparation (stored with the dataset). */
    joined: string[];
  };
  /** From a preparation, but exported before raw rows were supported. */
  raw_note?: string;
  image?: { channels: 1 | 3; height: number; width: number };
}

export interface PredictedClass {
  class: string;
  index?: number;
  probability: number;
}

/** One prediction: a class with its probability and the top classes, or a value. */
export interface Prediction {
  class?: string | null;
  index?: number | null;
  probability?: number | null;
  top?: PredictedClass[];
  /** Regression: in the units of the target. */
  value?: number | number[] | null;
  /** Gaussian processes: the predictive standard deviation. */
  std?: number | null;
  /** Binary classification: the probability of the positive class, compared with the threshold. */
  positive_probability?: number | null;
}

export interface PredictionRequest {
  inputs: unknown;
  input_format?: 'prepared' | 'raw';
  /** Images: 'raw' for pixels 0–255. */
  input_scale?: 'raw' | 'unit' | 'normalized';
  top_k?: number;
  /** Ensembles: also return what each member predicts. */
  members?: boolean;
}

/** What one member of an ensemble predicted (`members: true`). */
export interface MemberPredictions {
  id: string;
  name: string;
  predictions: Prediction[];
}

export interface PredictionResult {
  run: { id: string; name: string };
  task: 'classification' | 'regression' | null;
  predictions: Prediction[];
  target?: string | null;
  note?: string;
  std_note?: string;
  /** Raw rows: unknown categories, rows unlike the training rows … */
  warnings?: string[];
  /** Binary classification: the run's decision threshold and the class it decides for. */
  decision_threshold?: number;
  positive_class?: string;
  threshold_note?: string;
  /** Ensembles: the method and the members that predicted. */
  ensemble?: {
    method: EnsembleMethod;
    members: { id: string; name: string; weight: number | null }[];
  };
  /** Ensembles, with `members: true`: what each member predicted, in the same order. */
  members?: MemberPredictions[];
}

/** The answer when inputs are invalid (HTTP 400). */
export interface PredictionErrorBody {
  detail: string;
  /** What the run takes. */
  inputs?: string | null;
  /** Raw rows: every problem, per row and column. */
  errors?: string[];
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
  /** File name endings the server can import (e.g. ".csv", ".xlsx", ".gz"). */
  file_types?: string[];
  /** Whether files can be imported from a link (the server downloads them). */
  url_import?: boolean;
}

/**
 * How to read a file; 'auto' means detect. Sent as form fields with the upload (or as JSON
 * with a link). Delimiter, decimal mark and encoding apply to text (CSV) files only.
 */
export interface ImportOptions {
  delimiter: 'auto' | ',' | ';' | '\t' | '|';
  decimal: 'auto' | '.' | ',';
  header: 'auto' | 'yes' | 'no';
  encoding: 'auto' | 'utf-8' | 'cp1252' | 'latin-1';
  /** Cell values that mean "missing"; null: the defaults (NA, N/A, null, ? …). */
  missing_values: string[] | null;
  /** The sheet of a workbook (Excel, OpenDocument); null: the first one with values. */
  sheet?: string | null;
}

export type ImportFormat =
  'csv' | 'xlsx' | 'xls' | 'ods' | 'json' | 'jsonl' | 'parquet' | 'feather';

/** What the reader detected. Sources imported before other formats existed lack `format`. */
export interface ImportReport {
  /** null: not a CSV file. */
  delimiter: string | null;
  decimal: string;
  header: boolean;
  /** null: a binary file (Excel, Parquet …). */
  encoding: string | null;
  rows: number;
  columns: number;
  warnings: string[];
  format?: ImportFormat;
  /** gzip, bz2, xz or zip if the file was packed; `member`: the file inside the ZIP file. */
  compression?: string | null;
  member?: string | null;
  /** Workbooks: the imported sheet and all sheets. */
  sheet?: string | null;
  sheets?: string[];
  /** JSON: the keys that lead to the table, e.g. ["data"]. */
  json_path?: string[] | null;
  /** The link the file was downloaded from. */
  url?: string;
}

/** An imported data file (raw data). Several sources can be combined in one preparation. */
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
  | 'features'
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
  /**
   * Runs before the split, right after the combine steps (remove atypical rows, find
   * near-duplicate groups). Missing on older servers.
   */
  pre_split?: boolean;
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
  /**
   * Keep the rows of a group together: all rows with the same value of this column go to the
   * same part. It must exist where the split is assigned (see SplitInfo). Missing = null.
   */
  group_by?: string | null;
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
  /** Settings the fix changes as well, e.g. {split: {group_by: 'group'}}. */
  settings?: { split?: Partial<SplitSettings> };
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
  /** How the number columns relate (null where not computed, e.g. the "before" profile). */
  relations?: ColumnRelations | null;
}

// --- Column relations (backend/brickdnn/data/relations.py) -------------------------------

/**
 * Every number/yes-no feature's strongest partner: the other feature it correlates with most
 * (Spearman ρ), on the training rows (a seeded sample on large tables). Indices refer to
 * `columns`.
 */
export interface ColumnRelations {
  method: 'spearman' | 'pearson';
  /** The analysed features (number and yes/no columns but the target, not constant). */
  columns: string[];
  /** Number features that could not be analysed (constant on these rows). */
  skipped: string[];
  /** Number features in the table (with `sampled_columns`, more than were analysed). */
  total: number;
  /** Rows used, and the rows they were drawn from. */
  rows: number;
  pool_rows: number;
  /** The rows are training rows (false before the split is assigned). */
  training_rows: boolean;
  sampled_columns: boolean;
  /** Index of each column's strongest partner (null: none). */
  best: (number | null)[];
  /** Signed Spearman ρ with it, and Pearson r with the same partner. */
  best_rho: (number | null)[];
  best_r: (number | null)[];
  /** Partners with |ρ| ≥ partner_level. */
  partners: number[];
  partner_level: number;
  /** Up to 5 partners per column (indices), strongest first, and their ρ. */
  top: number[][];
  top_rho: number[][];
  /** Histogram of the strongest-partner |ρ| (bins of 0.02 from 0 to 1). */
  histogram: { edges: number[]; counts: number[] };
  /** Default threshold of "no strong partner". */
  weak_level: number;
  /** Threshold suggested for these rows: weak_level, more where chance correlations are large. */
  suggested_level: number;
  /** Strongest |ρ| an unrelated column reaches by chance: median, and 99.9 % stay below. */
  chance: number | null;
  chance_high: number | null;
  /** Clustered order of the columns (related ones together); null for very wide tables. */
  order: number[] | null;
  /** Group of related columns per column (0 = largest), -1 for none. */
  groups: number[] | null;
  /**
   * The whole matrix in the clustered order, in blocks: `starts` index into `order`;
   * `values[a][b]` is the mean |ρ| of the columns of block a with those of block b.
   */
  overview: { starts: number[]; values: (number | null)[][] } | null;
}

// --- end of column relations ---------------------------------------------------------------

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
  /** Where the split was assigned; missing on older servers. */
  split?: SplitInfo;
  /** Pipeline-level notes, e.g. "Stratified split not possible: …". */
  warnings: string[];
  duration_ms: number;
}

export type PrepDownloadFormat = 'csv' | 'recipe' | 'python';

// --------------------------------------------------------------------------
// Paid plans (Paddle), see docs/billing.md
// --------------------------------------------------------------------------

/** A plan on the pricing page. */
export interface BillingPlan extends Plan {
  /** It can be bought (the others are the free plan). */
  purchasable: boolean;
  /** The plan of accounts without a subscription. */
  default: boolean;
}

/** GET /api/billing/ (public): whether the server sells plans, and what the checkout needs. */
export type BillingInfo =
  | { enabled: false }
  | {
      enabled: true;
      provider: 'paddle';
      /** `sandbox`: test mode, no real payments. */
      environment: 'sandbox' | 'live';
      /** The client-side token for Paddle.js. */
      client_token: string;
      currency: string;
      prices_include_tax: boolean;
      /** Compute beyond the allowance can be billed (plans offer overage). */
      overage_billed: boolean;
      plans: BillingPlan[];
    };

/** The server's billing settings when it sells plans. */
export type BillingEnabled = Extract<BillingInfo, { enabled: true }>;

/** The account's subscription as Paddle last reported it. */
export interface BillingSubscription {
  status: 'active' | 'trialing' | 'past_due' | 'paused' | 'canceled' | (string & {});
  /** The plan it pays for. */
  plan: string | null;
  plan_name: string | null;
  /** The next payment (null: it does not renew). */
  renews_at: string | null;
  /** Cancelled to the end of the paid period: when it ends. */
  ends_at: string | null;
  /** When it ended, once it has. */
  ended_at: string | null;
  current_period_end: string | null;
  payment_failed: boolean;
  /** After a failed payment: the plan is kept until then while Paddle retries. */
  grace_until: string | null;
  can_change_plan: boolean;
  can_cancel: boolean;
  can_reactivate: boolean;
}

/** GET /api/billing/subscription/: the account's plan and subscription. */
export interface BillingState {
  /** The plan the account has now. */
  plan: Plan;
  /** No limits apply: the account needs no plan. */
  unlimited: boolean;
  subscription: BillingSubscription | null;
  /** A checkout can be opened (no subscription yet, not unlimited). */
  can_subscribe: boolean;
  /** The account has a customer at Paddle (payment method, invoices). */
  has_billing_profile: boolean;
  email_missing: boolean;
}

/** POST /api/billing/checkout/: what Paddle's checkout is opened with. */
export interface BillingCheckout {
  plan: string;
  price_id: string;
  customer_id: string;
  custom_data: Record<string, string>;
}

/** POST /api/billing/change-plan/ with `preview`: what changing the plan would cost. */
export interface PlanChangePreview {
  plan: Plan;
  upgrade: boolean;
  /** Charged now, or credited to the next invoice; null if Paddle did not say. */
  now: { action: 'charge' | 'credit'; amount: number; currency: string } | null;
  next_billed_at: string | null;
}
// ---- Row analysis (Analysis → Rows) and the split by groups -----------------------------

/** Where the split was assigned and, with `group_by`, how the groups were split. */
export interface SplitInfo {
  group_by: string | null;
  /** Columns where the split is assigned (after the combine and pre-split steps). */
  columns: ColumnInfo[];
  /** Why the groups could not be used (the column is missing there); a random split instead. */
  error: string | null;
  groups: {
    /** Groups in all, including single rows. */
    count: number;
    /** Groups with more than one row. */
    multi: number;
    largest: number;
    /** Rows without a group value (each is its own group). */
    missing: number;
    /** Groups in training, validation and test. */
    per_part: [number, number, number];
    /** Groups in more than one part (0). */
    mixed: number;
  } | null;
}

/** Which number columns the row statistics compare rows on. */
export type RowFeatureMode = 'auto' | 'all' | 'partnered' | 'columns';

export interface RowFeatureChoice {
  mode: RowFeatureMode;
  /** What was used: auto resolves to all or partnered. */
  used: 'all' | 'partnered' | 'columns';
  count: number;
  /** The first 50 columns used. */
  columns: string[];
  /** Number columns that could be used (without target, ID and group columns). */
  candidates: number;
  left_out: { name: string; reason: string }[];
  /** Columns left out per reason: target_or_group, id, constant, no_partner, limit. */
  left_out_counts: Record<string, number>;
  /** |Spearman r| a partner needs (partnered only). */
  partner_threshold: number | null;
  note: string | null;
}

export interface RowAnalysisRequest {
  steps?: PipelineStep[];
  settings?: PrepSettings;
  upto?: string | null;
  features?: { mode: RowFeatureMode; columns?: string[] };
  /** NN ratio below which rows are near twins (default: suggested, else 0.5). */
  twin_threshold?: number | null;
  /** Spread below which rows are unusually typical (default: suggested, else 0.25). */
  atypical_threshold?: number | null;
}

/**
 * Row statistics of the previewed data: one entry per analysed row in `rows` (all rows,
 * or a seeded sample when `sampled`). Neighbour and pair indices refer to these entries.
 */
export interface RowAnalysis {
  n_total: number;
  analysed: number;
  sampled: boolean;
  /** Rows analysed at most for this many feature columns. */
  limit: number;
  seed: number;
  features: RowFeatureChoice;
  /** An ID column that names the rows, if the table has one. */
  id_column: string | null;
  target: string | null;
  task: 'classification' | 'regression' | null;
  /** Median distance of random pairs of rows (the unit of the NN ratio). */
  pair_distance: number | null;
  /** Thresholds at clear gaps of the distributions; null if there is none. */
  suggested: { atypical: number | null; twins: number | null };
  /** Thresholds used for `groups` (asked, else suggested, else defaults). */
  thresholds: { twins: number; atypical: number };
  rows: {
    /** Row number in the table (0-based). */
    position: number[];
    id: (string | number | null)[] | null;
    split: ('train' | 'val' | 'test')[] | null;
    /** Numbers (regression) or class labels (classification). */
    target: (number | string | null)[] | null;
    /** Median |robust z| over the row's values: ~0.67 ordinary, near 0 too typical. */
    spread: (number | null)[];
    /** Missing values in the row (all columns except the target). */
    missing: number[];
    nn_distance: (number | null)[];
    /** Nearest distance / typical pair distance. */
    nn_ratio: (number | null)[];
    /** Index (into these rows) of the nearest row, -1 if none. */
    neighbor: number[];
    group: number[];
    group_size: number[];
    pc1: (number | null)[];
    pc2: (number | null)[];
  };
  groups: {
    threshold: number;
    /** Groups with at least two rows. */
    count: number;
    rows: number;
    largest: number;
    /** Groups whose rows are in more than one part of the split. */
    mixed: number;
    held_out: number;
    /** Validation/test rows with a near twin in the training rows. */
    held_out_with_twin: number;
  };
  /** Random pairs (the baseline): indices into `rows` and their distance ratios. */
  random_pairs: { a: number[]; b: number[]; ratio: (number | null)[] };
  /** Share of the variance on the two projection axes. */
  explained: (number | null)[];
  split_group_by: string | null;
  duration_ms: number;
  stats_ms: number;
}

export interface SimilarRow {
  /** Index into RowAnalysis.rows. */
  index: number;
  position: number;
  id: string | number | null;
  split: 'train' | 'val' | 'test' | null;
  target: number | string | null;
  spread: number | null;
  distance: number | null;
  ratio: number | null;
}

export interface PairColumn {
  name: string;
  a: number | null;
  b: number | null;
  za: number | null;
  zb: number | null;
  /** |za − zb| */
  diff: number | null;
}

export interface SimilarRows {
  row: SimilarRow;
  /** The nearest rows, nearest first. */
  neighbors: SimilarRow[];
  pair: {
    a: SimilarRow;
    b: SimilarRow;
    distance: number | null;
    ratio: number | null;
    /** Columns both rows have, of `features`. */
    shared: number;
    features: number;
    closest: PairColumn[];
    farthest: PairColumn[];
  } | null;
}

export type SimilarRowsRequest = RowAnalysisRequest & { row: number; other?: number | null };
