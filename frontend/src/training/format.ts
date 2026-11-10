import type {
  Dataset,
  HistoryEntry,
  Optimizer,
  RunDetail,
  RunMetrics,
  RunQueue,
  RunStatus,
  RunSummary,
  TrainingPlan,
  TrialParams,
  TunableSetting,
  TuningConfig,
} from '@/api/types';
import { isClassicalRun } from './classical';

/** Not finished yet: waiting in the queue or training. */
export const ACTIVE_STATUSES: readonly RunStatus[] = ['queued', 'starting', 'preparing', 'running'];

export function isActive(status: RunStatus | null | undefined): boolean {
  return !!status && ACTIVE_STATUSES.includes(status);
}

export const STATUS_LABELS: Record<RunStatus, string> = {
  queued: 'Queued',
  starting: 'Starting',
  preparing: 'Preparing',
  running: 'Running',
  completed: 'Completed',
  stopped: 'Stopped',
  failed: 'Failed',
};

export const STATUS_COLORS: Record<RunStatus, string> = {
  queued: 'grape',
  starting: 'gray',
  preparing: 'cyan',
  running: 'blue',
  completed: 'green',
  stopped: 'orange',
  failed: 'red',
};

/** 0.9812 → "98.1%". */
export function formatPercent(value: number | null | undefined, digits = 1): string {
  return typeof value === 'number' && Number.isFinite(value)
    ? `${(value * 100).toFixed(digits)}%`
    : '–';
}

/** Losses and other metrics: four significant digits, no exponent. */
export function formatMetric(value: number | null | undefined): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return '–';
  }
  return value.toLocaleString('en-US', { maximumSignificantDigits: 4 });
}

export function formatBytes(bytes: number | null | undefined): string {
  if (typeof bytes !== 'number' || !Number.isFinite(bytes)) {
    return '–';
  }
  const units = ['bytes', 'KB', 'MB', 'GB'];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return unit === 0 ? `${value} bytes` : `${value.toFixed(value < 10 ? 1 : 0)} ${units[unit]}`;
}

/** Progress of an upload, both in the unit of the total: "12.3 MB of 48.0 MB". */
export function formatBytesOf(sent: number, total: number): string {
  const units = ['bytes', 'KB', 'MB', 'GB'];
  let scale = 1;
  let unit = 0;
  while (total / scale >= 1024 && unit < units.length - 1) {
    scale *= 1024;
    unit += 1;
  }
  const format = (bytes: number) =>
    `${unit === 0 ? bytes : (bytes / scale).toFixed(1)} ${units[unit]}`;
  return `${format(sent)} of ${format(total)}`;
}

/** R² (coefficient of determination): 0.91234 → "0.912". */
export function formatR2(value: number | null | undefined): string {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(3) : '–';
}

/** A fit time: 0.42 → "0.4 s", 7.25 → "7.3 s", longer ones like formatDuration. */
export function formatFitTime(seconds: number | null | undefined): string {
  if (typeof seconds === 'number' && Number.isFinite(seconds) && seconds >= 0 && seconds < 10) {
    return `${seconds < 0.05 ? '< 0.1' : seconds.toFixed(1)} s`;
  }
  return formatDuration(seconds);
}

/** 42 → "42 s", 192 → "3 min 12 s", 3900 → "1 h 5 min". */
export function formatDuration(seconds: number | null | undefined): string {
  if (typeof seconds !== 'number' || !Number.isFinite(seconds) || seconds < 0) {
    return '–';
  }
  const total = Math.round(seconds);
  if (total < 60) {
    return `${total} s`;
  }
  const minutes = Math.floor(total / 60);
  if (minutes < 60) {
    return total % 60 ? `${minutes} min ${total % 60} s` : `${minutes} min`;
  }
  const hours = Math.floor(minutes / 60);
  return minutes % 60 ? `${hours} h ${minutes % 60} min` : `${hours} h`;
}

export function formatDate(iso: string | null | undefined): string {
  if (!iso) {
    return '–';
  }
  return new Date(iso).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });
}

export function formatCompact(value: number | null | undefined): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return '–';
  }
  return value.toLocaleString('en-US', {
    notation: 'compact',
    maximumFractionDigits: Math.abs(value) < 1000 ? 0 : 1,
  });
}

/** "150 rows · 4 features · 3 classes" for an uploaded dataset. */
export function datasetDetails(dataset: Dataset): string {
  const { info } = dataset;
  const parts: string[] = [];
  const count = (n: number | null | undefined, word: string) =>
    typeof n === 'number'
      ? `${n.toLocaleString('en-US')} ${word}${n === 1 ? '' : word.endsWith('s') ? 'es' : 's'}`
      : null;
  if (dataset.kind === 'csv') {
    parts.push(count(info.num_samples, 'row') ?? '', count(info.num_features, 'feature') ?? '');
    if (info.num_classes && info.target_kind !== 'number') {
      parts.push(count(info.num_classes, 'class') ?? '');
    } else if (info.target_kind === 'number') {
      parts.push('numeric target');
    }
  } else {
    parts.push(count(info.num_samples, 'image') ?? '', count(info.num_classes, 'class') ?? '');
  }
  parts.push(formatBytes(dataset.size_bytes));
  return parts.filter(Boolean).join(' · ');
}

/** Rows of a predefined split (datasets from the Data tab): "840 / 180 / 180". */
export function formatSplit(counts: number[]): string {
  return counts.map((count) => count.toLocaleString('en-US')).join(' / ');
}

/** Shares of a predefined split in percent, rounded to one decimal: [70, 15, 15]. */
export function splitPercents(counts: number[]): number[] {
  const total = counts.reduce((sum, count) => sum + count, 0);
  return counts.map((count) => (total > 0 ? Math.round((count / total) * 1000) / 10 : 0));
}

// --------------------------------------------------------------------------
// Hyperparameter search
// --------------------------------------------------------------------------

/** Settings a search can vary, in display order. */
export const TUNABLE_SETTINGS: TunableSetting[] = [
  'learning_rate',
  'batch_size',
  'optimizer',
  'weight_decay',
];

export const SETTING_LABELS: Record<TunableSetting, string> = {
  learning_rate: 'Learning rate',
  batch_size: 'Batch size',
  optimizer: 'Optimizer',
  weight_decay: 'Weight decay',
};

const OPTIMIZER_LABELS: Record<Optimizer, string> = { adam: 'Adam', adamw: 'AdamW', sgd: 'SGD' };

/** A setting value as people write it: 0.00312, 64, "Adam". */
export function formatSetting(name: TunableSetting, value: number | string | undefined): string {
  if (value === undefined) {
    return '–';
  }
  if (name === 'optimizer') {
    return OPTIMIZER_LABELS[value as Optimizer] ?? String(value);
  }
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return String(value);
  }
  return name === 'batch_size' ? String(value) : String(Number(value.toPrecision(3)));
}

/** "learning rate 0.00312, batch size 64, Adam" (like the engine's messages). */
export function describeParams(params: TrialParams): string {
  return TUNABLE_SETTINGS.filter((name) => params[name] !== undefined)
    .map((name) => {
      const value = formatSetting(name, params[name]);
      return name === 'optimizer' ? value : `${SETTING_LABELS[name].toLowerCase()} ${value}`;
    })
    .join(', ');
}

/** The settings a search varies, in display order. */
export function searchedSettings(tuning: TuningConfig | null | undefined): TunableSetting[] {
  return tuning ? TUNABLE_SETTINGS.filter((name) => tuning.space[name] !== undefined) : [];
}

/** "search · 10 trials" or "grid search · 6 trials". */
export function searchLabel(tuning: TuningConfig): string {
  const method = tuning.method === 'grid' ? 'grid search' : 'search';
  return `${method} · ${tuning.trials} ${tuning.trials === 1 ? 'trial' : 'trials'}`;
}

/** "starts in about 4 minutes": the estimated wait of a queued run; null if unknown. */
export function startsIn(seconds: number | null | undefined): string | null {
  if (typeof seconds !== 'number' || !Number.isFinite(seconds) || seconds < 0) {
    return null;
  }
  if (seconds < 20) {
    return 'starts in a moment';
  }
  if (seconds < 90) {
    return 'starts in about a minute';
  }
  if (seconds < 90 * 60) {
    return `starts in about ${Math.round(seconds / 60)} minutes`;
  }
  const hours = Math.round(seconds / 3600);
  return `starts in about ${hours} ${hours === 1 ? 'hour' : 'hours'}`;
}

/**
 * Where a queued run waits, e.g. "Queued — position 3, starts in about 4 minutes" (the start
 * is the server's rough estimate).
 */
export function queueText(queue: RunQueue | null | undefined): string {
  if (!queue) {
    return 'Queued';
  }
  const place = `Queued — position ${queue.position}`;
  if (!queue.worker_online) {
    return `${place}, waiting for a training worker`;
  }
  if (queue.blocked_by === 'account') {
    return `${place}, starts after your running training`;
  }
  const start = startsIn(queue.estimated_wait_seconds);
  return start ? `${place}, ${start}` : place;
}

/**
 * Short progress text for the toolbar, e.g. "Epoch 3/10", "Trial 2/10", "Preparing", "Queued
 * #3" or "Fitting" (classical models have no epochs).
 */
export function progressLabel(run: RunSummary): string {
  if (run.status === 'queued' && run.queue) {
    return `Queued #${run.queue.position}`;
  }
  if (run.status === 'running' && isClassicalRun(run)) {
    return 'Fitting';
  }
  if (run.status === 'running' && run.tuning && run.tuning.trials > 0) {
    const total = run.tuning.trials;
    return `Trial ${Math.min(run.trials_done + 1, total)}/${total}`;
  }
  if (run.status === 'running' && run.epochs > 0) {
    return `Epoch ${Math.min(run.epochs_done + 1, run.epochs)}/${run.epochs}`;
  }
  return STATUS_LABELS[run.status];
}

/** Whether the run is about classes (accuracy) rather than values (RMSE). */
export function isClassification(run: RunSummary, history: HistoryEntry[] = []): boolean {
  if (run.task === 'classification') {
    return true;
  }
  if (run.task === 'regression') {
    return false;
  }
  return (
    typeof run.metrics?.val_accuracy === 'number' ||
    history.some((entry) => typeof entry.val_accuracy === 'number')
  );
}

/**
 * Whether a run of a neural network shows its R² next to the RMSE: regressions. Their history
 * entries carry `val_r2` (null when the validation targets are all equal); runs of older
 * versions, which did not report it, have no R² anywhere and show none.
 */
export function reportsR2(run: RunSummary, history: HistoryEntry[]): boolean {
  if (isClassification(run, history)) {
    return false;
  }
  const values = [run.metrics?.val_r2, run.metrics?.test_r2, ...history.map((h) => h.val_r2)];
  if (values.some((value) => typeof value === 'number')) {
    return true;
  }
  return run.task === 'regression' && history.every((entry) => 'val_r2' in entry);
}

/** Whether a test set result exists (only after the end of a run with a test set). */
export function hasTestResult(metrics: RunMetrics | null | undefined): boolean {
  return (
    typeof metrics?.test_accuracy === 'number' ||
    typeof metrics?.test_rmse === 'number' ||
    typeof metrics?.test_loss === 'number'
  );
}

/**
 * The most telling number of a run and what it is, e.g. ["98.1%", "test accuracy"]: the test
 * result if there is one, otherwise the validation result.
 */
export function headlineMetric(metrics: RunMetrics | null | undefined): [string, string] | null {
  if (!metrics) {
    return null;
  }
  if (typeof metrics.test_accuracy === 'number') {
    return [formatPercent(metrics.test_accuracy), 'test accuracy'];
  }
  // Regression: R² says more than an RMSE in the units of the target.
  if (typeof metrics.test_r2 === 'number') {
    return [formatR2(metrics.test_r2), 'test R²'];
  }
  if (typeof metrics.test_rmse === 'number') {
    return [formatMetric(metrics.test_rmse), 'test RMSE'];
  }
  if (typeof metrics.test_loss === 'number') {
    return [formatMetric(metrics.test_loss), 'test loss'];
  }
  if (typeof metrics.val_accuracy === 'number') {
    return [formatPercent(metrics.val_accuracy), 'validation accuracy'];
  }
  if (typeof metrics.val_r2 === 'number') {
    return [formatR2(metrics.val_r2), 'validation R²'];
  }
  if (typeof metrics.val_rmse === 'number') {
    return [formatMetric(metrics.val_rmse), 'validation RMSE'];
  }
  if (typeof metrics.val_loss === 'number') {
    return [formatMetric(metrics.val_loss), 'validation loss'];
  }
  return null;
}

/**
 * The headline result of a run, e.g. "validation accuracy 98.1%", "test accuracy 97.5%" or
 * "best trial 3, test accuracy 97.5%".
 */
export function resultText(run: RunSummary): string | null {
  const headline = headlineMetric(run.metrics);
  if (!headline) {
    return null;
  }
  const text = `${headline[1]} ${headline[0]}`;
  return run.tuning && run.best_trial ? `best trial ${run.best_trial.trial}, ${text}` : text;
}

/**
 * Which weights a finished run keeps, e.g. "Weights from epoch 7 (lowest validation loss) ·
 * stopped early after epoch 12". Null while the epochs are not over and for older runs.
 */
export function weightsNote(run: RunDetail): string | null {
  if (isClassicalRun(run)) {
    return null; // Fitted once: there is only one model.
  }
  const state = run.state;
  const epoch = state?.weights_epoch;
  if (typeof epoch !== 'number') {
    return null;
  }
  const last = state?.history.at(-1)?.epoch ?? run.epochs_done;
  const best = epoch !== last && run.config.restore_best !== false;
  const parts = [`Weights from epoch ${epoch}${best ? ' (lowest validation loss)' : ''}`];
  if (state?.stopped_early) {
    parts.push(`stopped early after epoch ${last}`);
  } else if (epoch !== last) {
    parts.push(`${last} epochs trained`);
  }
  return parts.join(' · ');
}

/**
 * How the inputs were prepared (`state.preprocessing`) as a JSON file to download with the
 * weights: normalisation statistics of the training data, image size or CSV columns and class
 * names. Null for demo data and runs without it.
 */
export function preprocessingJson(run: RunDetail): string | null {
  const pre = run.state?.preprocessing;
  if (!pre || !pre.kind || pre.kind === 'demo') {
    return null;
  }
  return `${JSON.stringify(pre, null, 2)}\n`;
}

// --------------------------------------------------------------------------
// "How to use the weights" snippet
// --------------------------------------------------------------------------

function numberList(value: unknown): number[] | null {
  return Array.isArray(value) && value.every((v) => typeof v === 'number') ? value : null;
}

function stringList(value: unknown): string[] | null {
  return Array.isArray(value) && value.every((v) => typeof v === 'string') ? value : null;
}

function pyList(values: number[]): string {
  return `[${values.map((v) => Number(v.toPrecision(6))).join(', ')}]`;
}

function pyStrings(values: string[], max = 20): string {
  const shown = values.slice(0, max).map((value) => JSON.stringify(value));
  return `[${shown.join(', ')}${values.length > max ? ', ...' : ''}]`;
}

/**
 * Python code that loads the trained weights and prepares inputs the way the training did.
 * Follows ``state.preprocessing`` written by the engine: ``{"kind": "image", channels, height,
 * width, layout, mean, std, class_names}``, ``{"kind": "tabular", columns, mean, std,
 * target_mean, target_std, class_names}`` or ``{"kind": "demo"}``.
 */
export function weightsSnippet(run: RunDetail, scriptName: string, weightsName: string): string {
  if (isClassicalRun(run)) {
    return classicalSnippet(run, weightsName);
  }
  const pre = run.state?.preprocessing ?? {};
  const plan: Partial<TrainingPlan> = run.plan ?? {};
  const module = scriptName.replace(/\.py$/, '');
  const mean = numberList(pre.mean);
  const std = numberList(pre.std);
  const classNames =
    stringList(pre.class_names) ?? run.state?.class_names ?? plan.class_names ?? null;
  const image =
    pre.kind === 'image'
      ? (pre as { channels?: number; height?: number; width?: number; layout?: string })
      : (plan.image ?? null);

  const lines = [
    'import torch',
    ...(image ? ['import torchvision.transforms.functional as TF', 'from PIL import Image'] : []),
    '',
    `from ${module} import NeuralNetwork  # the downloaded script`,
    '',
    'model = NeuralNetwork()',
    `model.load_state_dict(torch.load("${weightsName}", map_location="cpu"))`,
    'model.eval()',
    '',
  ];

  if (pre.kind === 'demo' || run.dataset.kind === 'random') {
    const shape = plan.inputs?.[0]?.shape;
    lines.push(
      '# Trained on random demo data: good for testing, not for real predictions.',
      `x = torch.randn(1, ${shape && shape.length > 0 ? shape.join(', ') : '...'})`
    );
  } else if (image) {
    const mode = image.channels === 3 ? 'RGB' : 'L';
    lines.push(
      '# Prepare an image exactly like during training:',
      `image = Image.open("example.png").convert("${mode}")`,
      `image = image.resize((${image.width}, ${image.height}), Image.BILINEAR)`,
      'x = TF.to_tensor(image)  # channels × height × width, pixels scaled to 0..1'
    );
    if (mean && std) {
      lines.push(`x = TF.normalize(x, mean=${pyList(mean)}, std=${pyList(std)})`);
    }
    if (image.layout === 'flat') {
      lines.push('x = x.flatten()');
    } else if (image.layout === 'hw') {
      lines.push('x = x[0]  # the network expects height × width');
    }
    lines.push('x = x.unsqueeze(0)  # a batch of one image');
  } else if (pre.kind === 'tabular') {
    const columns = stringList(pre.columns) ?? [];
    lines.push(
      `# One row of features in the order of the CSV columns:`,
      `# ${columns.join(', ')}`,
      `features = [0.0] * ${columns.length || mean?.length || 1}  # replace with your values`
    );
    if (mean && std) {
      lines.push(
        '# Standardise them like during training:',
        `mean = torch.tensor(${pyList(mean)})`,
        `std = torch.tensor(${pyList(std)})`,
        'x = (torch.tensor([features]) - mean) / std'
      );
    } else {
      lines.push('x = torch.tensor([features])');
    }
  } else {
    lines.push('x = ...  # a batch of inputs, prepared like the training data');
    if (Object.keys(pre).length > 0) {
      lines.push(`# Preprocessing used in training: ${JSON.stringify(pre)}`);
    }
  }

  lines.push('', 'with torch.no_grad():', '    output = model(x)');
  const targetMean = pre.target_mean;
  const targetStd = pre.target_std;
  if (classNames && classNames.length > 0 && pre.kind !== 'demo') {
    lines.push(
      '',
      `classes = ${pyStrings(classNames)}`,
      'print(classes[output.argmax(dim=1).item()])'
    );
  } else if (typeof targetMean === 'number' && typeof targetStd === 'number') {
    lines.push(
      '',
      '# The network learned standardised targets; convert back to the units of the CSV:',
      `print(output.item() * ${Number(targetStd.toPrecision(6))} + ${Number(targetMean.toPrecision(6))})`
    );
  } else {
    lines.push('print(output)');
  }
  return `${lines.join('\n')}\n`;
}

/**
 * Python code that loads a fitted classical model (the scikit-learn pipeline saved with
 * joblib) and predicts one row, like `inference.usage_snippet` on the server. The pipeline
 * standardises the features itself; `state.preprocessing` gives their order.
 */
export function classicalSnippet(run: RunDetail, modelName: string): string {
  const pre = run.state?.preprocessing ?? {};
  const classical = run.plan?.classical ?? null;
  const estimator = classical?.estimator ?? run.state?.classical?.estimator ?? 'the model';
  const four = (values: number[]) => `[${values.map((v) => Number(v.toPrecision(4))).join(', ')}]`;
  const columns = stringList(pre.columns);
  const mean = numberList(pre.mean);
  const lines = [
    'import joblib',
    '',
    `bundle = joblib.load("${modelName}")  # a pickle: only load files you trust`,
    `model = bundle["model"]  # scikit-learn Pipeline: StandardScaler, then ${estimator}`,
    '',
  ];
  if (pre.kind === 'tabular' && columns && mean) {
    lines.push(
      `# Features in this order: ${columns.join(', ')}`,
      `rows = [${four(mean)}]  # one list of ${columns.length} values per sample, as in the CSV`
    );
  } else {
    const n = run.plan?.inputs?.[0]?.shape?.[0] ?? 1;
    lines.push(
      '# Trained on random demo data: the predictions mean nothing.',
      `rows = [${four(Array<number>(n).fill(0))}]  # one list of ${n} values per sample`
    );
  }
  const target = typeof pre.target === 'string' ? pre.target : 'target';
  const classification =
    run.task === 'classification' ||
    classical?.task === 'classification' ||
    (stringList(pre.class_names)?.length ?? 0) > 0;
  if (classification) {
    lines.push(
      "probabilities = model.predict_proba(rows)  # one column per bundle['class_names']",
      'print([bundle["class_names"][i] for i in probabilities.argmax(axis=1)])'
    );
  } else if ((classical?.type ?? estimator) === 'GaussianProcessRegressor') {
    lines.push(
      'mean, std = model.predict(rows, return_std=True)',
      `print(mean.tolist(), std.tolist())  # predicted ${target} and its uncertainty`
    );
  } else {
    lines.push(`print(model.predict(rows).tolist())  # predicted ${target}`);
  }
  return `${lines.join('\n')}\n`;
}
