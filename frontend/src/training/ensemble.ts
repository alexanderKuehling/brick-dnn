import type {
  EnsembleMethod,
  EnsembleSplit,
  EnsembleSplitMetrics,
  RunSummary,
  TrainingTask,
} from '@/api';

/** The most runs one ensemble combines (as on the server). */
export const MAX_MEMBERS = 10;

export const METHOD_LABELS: Record<EnsembleMethod, string> = {
  average: 'Weighted average',
  stacking: 'Stacking',
};

export const SPLIT_LABELS: Record<EnsembleSplit, string> = {
  val: 'Validation',
  test: 'Test',
  training: 'Training',
};

export function isEnsemble(run: Pick<RunSummary, 'kind' | 'family'>): boolean {
  return run.kind === 'ensemble' || run.family === 'ensemble';
}

/** A finished classification or regression on an uploaded dataset, with weights. */
export function canBeMember(run: RunSummary): boolean {
  return (
    !isEnsemble(run) &&
    run.has_weights &&
    (run.status === 'completed' || run.status === 'stopped') &&
    (run.task === 'classification' || run.task === 'regression') &&
    run.dataset.kind === 'upload'
  );
}

/**
 * Why the selected runs cannot be combined, as far as the list tells (null: try it; the server
 * checks the rest, e.g. the classes and the split).
 */
export function combineProblem(runs: RunSummary[]): string | null {
  if (runs.length < 2) {
    return 'Select at least 2 finished runs.';
  }
  if (runs.length > MAX_MEMBERS) {
    return `An ensemble combines at most ${MAX_MEMBERS} runs.`;
  }
  if (new Set(runs.map((run) => String(run.dataset.id))).size > 1) {
    return 'The runs were trained on different datasets. An ensemble needs runs of one dataset, so that their validation and test rows line up.';
  }
  if (new Set(runs.map((run) => run.task)).size > 1) {
    return 'The runs solve different tasks (classification and regression).';
  }
  return null;
}

/** "Ensemble of MLP and Trees", "Ensemble of MLP, Trees and 2 more". */
export function defaultEnsembleName(runs: RunSummary[]): string {
  const names = runs.map((run) => run.name);
  const text =
    names.length <= 2
      ? names.join(' and ')
      : names.length === 3
        ? `${names[0]}, ${names[1]} and ${names[2]}`
        : `${names[0]}, ${names[1]} and ${names.length - 2} more`;
  return `Ensemble of ${text}`.slice(0, 120);
}

/** What a method does, in one or two sentences. */
export function methodText(method: EnsembleMethod, task: TrainingTask | null | undefined): string {
  const regression = task === 'regression';
  if (method === 'average') {
    return regression
      ? 'The weighted average of the values the members predict.'
      : 'The weighted average of the class probabilities the members predict.';
  }
  return regression
    ? 'A ridge regression learns from the validation rows how to combine the members’ predicted values; it is measured on the test rows.'
    : 'A logistic regression learns from the validation rows how to combine the members’ probabilities; it is measured on the test rows.';
}

/** The weights as shares that add up to 1 (all equal if they add up to 0). */
export function shares(weights: number[]): number[] {
  const total = weights.reduce((sum, w) => sum + (Number.isFinite(w) && w > 0 ? w : 0), 0);
  return weights.map((w) =>
    total > 0 ? (Number.isFinite(w) && w > 0 ? w / total : 0) : 1 / weights.length
  );
}

export interface ComparisonColumn {
  key: keyof EnsembleSplitMetrics;
  label: string;
  /** Higher is better (else lower is). */
  higher: boolean;
  /** percent: 94.0%; score: 0.871 (F1, AUC, log loss); value: in the units of the target. */
  format: 'percent' | 'score' | 'value' | 'r2';
}

/** The columns of the comparison table of a task. */
export function comparisonColumns(
  task: TrainingTask | null | undefined,
  binary: boolean
): ComparisonColumn[] {
  if (task === 'regression') {
    return [
      { key: 'rmse', label: 'RMSE', higher: false, format: 'value' },
      { key: 'mae', label: 'MAE', higher: false, format: 'value' },
      { key: 'r2', label: 'R²', higher: true, format: 'r2' },
    ];
  }
  return [
    { key: 'accuracy', label: 'Accuracy', higher: true, format: 'percent' },
    { key: 'balanced_accuracy', label: 'Balanced acc.', higher: true, format: 'percent' },
    { key: 'macro_f1', label: 'Macro F1', higher: true, format: 'score' },
    ...(binary
      ? [{ key: 'roc_auc' as const, label: 'ROC AUC', higher: true, format: 'score' as const }]
      : []),
    { key: 'log_loss', label: 'Log loss', higher: false, format: 'score' },
  ];
}
