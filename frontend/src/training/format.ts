import type {
  Dataset,
  HistoryEntry,
  RunDetail,
  RunStatus,
  RunSummary,
  TrainingPlan,
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

/** Short progress text for the toolbar, e.g. "Epoch 3/10" or "Preparing". */
export function progressLabel(run: RunSummary): string {
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

/** The headline result of a run, e.g. "validation accuracy 98.1%". */
export function resultText(run: RunSummary): string | null {
  const metrics = run.metrics;
  if (!metrics) {
    return null;
  }
  if (typeof metrics.val_accuracy === 'number') {
    return `validation accuracy ${formatPercent(metrics.val_accuracy)}`;
  }
  if (typeof metrics.val_rmse === 'number') {
    return `validation RMSE ${formatMetric(metrics.val_rmse)}`;
  }
  if (typeof metrics.val_loss === 'number') {
    return `validation loss ${formatMetric(metrics.val_loss)}`;
  }
  return null;
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
