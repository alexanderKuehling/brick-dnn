import type {
  Dataset,
  HistoryEntry,
  Optimizer,
  RunDetail,
  RunMetrics,
  RunStatus,
  RunSummary,
  TrainingPlan,
  TrialParams,
  TunableSetting,
  TuningConfig,
} from '@/api/types';

export const ACTIVE_STATUSES: readonly RunStatus[] = ['starting', 'preparing', 'running'];

export function isActive(status: RunStatus | null | undefined): boolean {
  return !!status && ACTIVE_STATUSES.includes(status);
}

export const STATUS_LABELS: Record<RunStatus, string> = {
  starting: 'Starting',
  preparing: 'Preparing',
  running: 'Running',
  completed: 'Completed',
  stopped: 'Stopped',
  failed: 'Failed',
};

export const STATUS_COLORS: Record<RunStatus, string> = {
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

/** Short progress text for the toolbar, e.g. "Epoch 3/10", "Trial 2/10" or "Preparing". */
export function progressLabel(run: RunSummary): string {
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
  if (typeof metrics.test_rmse === 'number') {
    return [formatMetric(metrics.test_rmse), 'test RMSE'];
  }
  if (typeof metrics.test_loss === 'number') {
    return [formatMetric(metrics.test_loss), 'test loss'];
  }
  if (typeof metrics.val_accuracy === 'number') {
    return [formatPercent(metrics.val_accuracy), 'validation accuracy'];
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
