/**
 * Hyperparameter search: the objective metrics that rank the trials, and what a search of a
 * classical model can vary (see docs/training.md#hyperparameter-search). The same as
 * backend/brickdnn/training/objectives.py and classical_tuning.py; the server checks again.
 */
import type {
  ClassicalTuningSpace,
  ClassicalValue,
  TrainingPlan,
  TrainingTask,
  TuningMetric,
  TuningRange,
} from '@/api/types';

// --------------------------------------------------------------------------
// Objective metrics
// --------------------------------------------------------------------------

export interface MetricInfo {
  name: TuningMetric;
  label: string;
  /** True: higher is better. */
  maximize: boolean;
  /** null: any task. */
  task: 'classification' | 'regression' | null;
  /** Needs exactly two classes. */
  binary: boolean;
  /** Shown as a percentage. */
  percent: boolean;
}

export const METRICS: MetricInfo[] = [
  { name: 'loss', label: 'Loss', maximize: false, task: null, binary: false, percent: false },
  {
    name: 'accuracy',
    label: 'Accuracy',
    maximize: true,
    task: 'classification',
    binary: false,
    percent: true,
  },
  {
    name: 'balanced_accuracy',
    label: 'Balanced accuracy',
    maximize: true,
    task: 'classification',
    binary: false,
    percent: true,
  },
  {
    name: 'macro_f1',
    label: 'Macro F1',
    maximize: true,
    task: 'classification',
    binary: false,
    percent: false,
  },
  {
    name: 'roc_auc',
    label: 'ROC AUC',
    maximize: true,
    task: 'classification',
    binary: true,
    percent: false,
  },
  {
    name: 'pr_auc',
    label: 'PR AUC',
    maximize: true,
    task: 'classification',
    binary: true,
    percent: false,
  },
  {
    name: 'rmse',
    label: 'RMSE',
    maximize: false,
    task: 'regression',
    binary: false,
    percent: false,
  },
  { name: 'mae', label: 'MAE', maximize: false, task: 'regression', binary: false, percent: false },
  { name: 'r2', label: 'R²', maximize: true, task: 'regression', binary: false, percent: false },
];

export function metricInfo(name: TuningMetric | null | undefined): MetricInfo {
  return METRICS.find((metric) => metric.name === name) ?? METRICS[0];
}

/**
 * The metrics a search can rank its trials by: the loss always; the others need a single
 * class output (classification) or a single value (regression), ROC and PR AUC two classes.
 * `task` comes from the training check (null: not known yet, only the loss).
 */
export function availableMetrics(
  task: TrainingTask | 'classification' | 'regression' | null | undefined,
  numClasses: number | null | undefined
): MetricInfo[] {
  return METRICS.filter(
    (metric) =>
      metric.task === null || (metric.task === task && (!metric.binary || numClasses === 2))
  );
}

/**
 * "classification" or "regression" for a single class output or a single value (the runs
 * with a final evaluation), otherwise null (segmentation, several outputs).
 */
export function evaluationTask(plan: TrainingPlan): 'classification' | 'regression' | null {
  if (plan.targets.length !== 1) {
    return null;
  }
  const target = plan.targets[0];
  if (target.kind === 'class') {
    return 'classification';
  }
  if (target.kind === 'value' && target.shape.reduce((a, b) => a * b, 1) === 1) {
    return 'regression';
  }
  return null;
}

/** The metrics a search on the data of `plan` can rank by (null plan: not known yet). */
export function planMetrics(plan: TrainingPlan | null | undefined): MetricInfo[] | null {
  return plan ? availableMetrics(evaluationTask(plan), plan.num_classes) : null;
}

/** The metric inside a sentence: "balanced accuracy", "macro F1", "ROC AUC", "the loss". */
export function metricInSentence(name: TuningMetric | null | undefined): string {
  const { label } = metricInfo(name);
  if (label === 'Loss') {
    return 'the loss';
  }
  return /^[A-Z][a-z]/.test(label) ? label[0].toLowerCase() + label.slice(1) : label;
}

/** "higher is better" or "lower is better". */
export function metricDirection(name: TuningMetric | null | undefined): string {
  return metricInfo(name).maximize ? 'higher is better' : 'lower is better';
}

/** A value of the metric as shown in tables: "93.1%", "0.912" or "–". */
export function formatMetricValue(
  name: TuningMetric | null | undefined,
  value: number | null | undefined
): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return '–';
  }
  if (metricInfo(name).percent) {
    return `${(value * 100).toFixed(1)}%`;
  }
  return value.toLocaleString('en-US', { maximumSignificantDigits: 4 });
}

// --------------------------------------------------------------------------
// Classical models
// --------------------------------------------------------------------------

/** A brick parameter a classical search can vary. */
export interface Tunable {
  name: string;
  label: string;
  kind: 'float' | 'int' | 'choice' | 'bool' | 'gamma';
  min: number;
  max: number;
  /** `min` itself is not allowed (e.g. C > 0). */
  minExclusive: boolean;
  /** How a range is drawn by default. */
  scale: 'log' | 'linear';
  /** choice and bool: the values to tick; gamma: the names besides numbers. */
  options: ClassicalValue[];
  /** A list may hold "none" (null: no limit). */
  none: boolean;
  /** What the form suggests. */
  default: TuningRange | ClassicalValue[];
  help?: string;
}

const tunable = (
  name: string,
  label: string,
  kind: Tunable['kind'],
  defaults: Tunable['default'],
  extra: Partial<Tunable> = {}
): Tunable => ({
  name,
  label,
  kind,
  min: 0,
  max: 1e6,
  minExclusive: false,
  scale: 'log',
  options: [],
  none: false,
  default: defaults,
  ...extra,
});

const CLASS_WEIGHT = tunable('class_weight', 'Class weights', 'choice', ['none', 'balanced'], {
  options: ['none', 'balanced'],
});
const BOOSTING = [
  tunable(
    'learning_rate',
    'Learning rate',
    'float',
    { min: 0.01, max: 0.3 },
    { max: 10, minExclusive: true }
  ),
  tunable('max_iter', 'Max. iterations', 'int', [100, 200, 500], { min: 1, max: 10_000 }),
  tunable(
    'max_leaf_nodes',
    'Max. leaves per tree',
    'int',
    { min: 15, max: 127 },
    { min: 2, max: 4096 }
  ),
  tunable('max_depth', 'Max. depth', 'int', [null, 3, 6], {
    min: 1,
    max: 100,
    none: true,
    help: '"none": no limit',
  }),
  tunable('min_samples_leaf', 'Min. rows per leaf', 'int', { min: 5, max: 100 }, { min: 1 }),
  tunable('l2_regularization', 'L2 regularisation', 'float', [0, 0.1, 1, 10], { max: 1e4 }),
];
const SVM = [
  tunable('C', 'C', 'float', { min: 0.1, max: 100 }, { minExclusive: true }),
  tunable(
    'gamma',
    'Gamma',
    'gamma',
    { min: 0.001, max: 1 },
    {
      max: 1e4,
      minExclusive: true,
      options: ['scale', 'auto'],
      help: 'Numbers, or scale and auto in a list',
    }
  ),
  tunable('kernel', 'Kernel', 'choice', ['rbf', 'linear'], {
    options: ['linear', 'poly', 'rbf', 'sigmoid'],
  }),
];
const RESTARTS = tunable('n_restarts_optimizer', 'Optimizer restarts', 'int', [0, 2, 5], {
  max: 20,
  scale: 'linear',
});

/** What each classical model can tune, by brick type. */
export const CLASSICAL_TUNABLES: Record<string, Tunable[]> = {
  HistGradientBoostingRegressor: BOOSTING,
  HistGradientBoostingClassifier: [...BOOSTING, CLASS_WEIGHT],
  SVR: [...SVM, tunable('epsilon', 'Epsilon', 'float', { min: 0.01, max: 1 })],
  SVC: [...SVM, CLASS_WEIGHT],
  GaussianProcessRegressor: [
    tunable('alpha', 'Alpha', 'float', { min: 1e-10, max: 0.01 }, { max: 1e3 }),
    tunable('normalize_y', 'Normalize targets', 'bool', [true, false], {
      options: [true, false],
    }),
    RESTARTS,
  ],
  GaussianProcessClassifier: [RESTARTS],
};

/** What a search of each model varies unless the user chooses otherwise. */
export const DEFAULT_CLASSICAL_PARAMS: Record<string, string[]> = {
  HistGradientBoostingRegressor: ['learning_rate', 'max_leaf_nodes', 'min_samples_leaf'],
  HistGradientBoostingClassifier: ['learning_rate', 'max_leaf_nodes', 'min_samples_leaf'],
  SVR: ['C', 'gamma'],
  SVC: ['C', 'gamma'],
  GaussianProcessRegressor: ['alpha'],
  GaussianProcessClassifier: ['n_restarts_optimizer'],
};

/** Labels of brick parameters (trial tables), by name over all classical models. */
export const PARAM_LABELS: Record<string, string> = Object.fromEntries(
  Object.values(CLASSICAL_TUNABLES)
    .flat()
    .map((item) => [item.name, item.label])
);

export const FOLDS = { min: 2, max: 10, default: 5 } as const;

/** A tick-able parameter in the settings form. */
export interface ParamState {
  on: boolean;
  /** Numbers: a range (random search only) or a list. */
  mode: 'range' | 'list';
  min: number | string;
  max: number | string;
  scale: 'log' | 'linear';
  /** Numbers as typed, separated by commas ("0, 0.1, 1"; "none" where allowed). */
  text: string;
  /** choice and bool parameters: the ticked options (as strings). */
  choices: string[];
}

const valueText = (value: ClassicalValue) => (value === null ? 'none' : String(value));

/** The form state of a parameter from its defaults (ticked if `on`). */
export function defaultParamState(item: Tunable, on: boolean): ParamState {
  const range = !Array.isArray(item.default) ? item.default : null;
  const list = Array.isArray(item.default) ? item.default : null;
  return {
    on,
    mode: range ? 'range' : 'list',
    min: range ? range.min : '',
    max: range ? range.max : '',
    scale: item.scale,
    // A grid needs lists: a range starts as a list of its two ends.
    text: range
      ? `${range.min}, ${range.max}`
      : list && item.kind !== 'choice' && item.kind !== 'bool'
        ? list.map(valueText).join(', ')
        : '',
    choices: list && (item.kind === 'choice' || item.kind === 'bool') ? list.map(String) : [],
  };
}

/** The form state of every parameter of `model`, with what was saved before. */
export function paramStates(
  model: string,
  saved: Record<string, Partial<ParamState>> | undefined
): Record<string, ParamState> {
  const defaults = DEFAULT_CLASSICAL_PARAMS[model] ?? [];
  return Object.fromEntries(
    (CLASSICAL_TUNABLES[model] ?? []).map((item) => {
      const base = defaultParamState(item, defaults.includes(item.name));
      const kept = saved?.[item.name];
      return [item.name, kept && typeof kept === 'object' ? { ...base, ...kept } : base];
    })
  );
}

/** A number field: '' for empty, NaN for not a number. */
function parseNumber(value: number | string): number | '' {
  if (typeof value === 'number') {
    return value;
  }
  const text = value.trim().replace(/[\s,]/g, '');
  return text === '' ? '' : Number(text);
}

function checkNumber(item: Tunable, value: number): string | null {
  if (!Number.isFinite(value)) {
    return 'Enter numbers';
  }
  if (item.kind === 'int' && !Number.isInteger(value)) {
    return 'Whole numbers';
  }
  const low = item.minExclusive ? value <= item.min : value < item.min;
  if (low || value > item.max) {
    const bound = item.minExclusive ? 'Greater than' : 'At least';
    return `${bound} ${item.min.toLocaleString('en-US')}, at most ${item.max.toLocaleString('en-US')}`;
  }
  return null;
}

/** The values of a list as the API takes them; an error message if one is wrong. */
export function parseList(item: Tunable, state: ParamState): ClassicalValue[] | string {
  if (item.kind === 'choice' || item.kind === 'bool') {
    if (state.choices.length === 0) {
      return 'Choose at least one';
    }
    return item.options.filter((option) => state.choices.includes(String(option)));
  }
  const parts = state.text
    .split(/[,;\s]+/)
    .map((part) => part.trim())
    .filter(Boolean);
  if (parts.length === 0) {
    return 'Enter at least one value';
  }
  const values: ClassicalValue[] = [];
  for (const part of parts) {
    const word = part.toLowerCase();
    if (item.none && word === 'none') {
      values.push(null);
    } else if (item.kind === 'gamma' && item.options.includes(word)) {
      values.push(word);
    } else {
      const number = Number(part);
      const error = checkNumber(item, number);
      if (error) {
        return `${part}: ${error.toLowerCase()}`;
      }
      values.push(number);
    }
  }
  return [...new Set(values)];
}

/** Problems with the range of a parameter (null: fine). */
export function rangeError(item: Tunable, state: ParamState): string | null {
  const lo = parseNumber(state.min);
  const hi = parseNumber(state.max);
  if (lo === '' || hi === '') {
    return 'Enter both ends';
  }
  const error = checkNumber(item, lo) ?? checkNumber(item, hi);
  if (error) {
    return error;
  }
  if (lo > hi) {
    return 'The minimum is larger than the maximum';
  }
  if (state.scale === 'log' && lo <= 0) {
    return 'A log range needs a minimum above 0: use linear, or a list';
  }
  return null;
}

/** Whether a parameter is given as a range (numbers in a random search only). */
export function usesRange(item: Tunable, state: ParamState, grid: boolean): boolean {
  return !grid && state.mode === 'range' && item.kind !== 'choice' && item.kind !== 'bool';
}

/** The search space of the ticked parameters for the API. */
export function classicalSpace(
  model: string,
  states: Record<string, ParamState>,
  grid: boolean
): ClassicalTuningSpace {
  const space: ClassicalTuningSpace = {};
  for (const item of CLASSICAL_TUNABLES[model] ?? []) {
    const state = states[item.name];
    if (!state?.on) {
      continue;
    }
    if (usesRange(item, state, grid)) {
      const lo = parseNumber(state.min);
      const hi = parseNumber(state.max);
      space[item.name] = {
        min: typeof lo === 'number' ? lo : 0,
        max: typeof hi === 'number' ? hi : 0,
        ...(state.scale === 'linear' ? { scale: 'linear' as const } : {}),
      };
    } else {
      const values = parseList(item, state);
      space[item.name] = typeof values === 'string' ? [] : values;
    }
  }
  return space;
}

/** Combinations of the listed values, or null if a range makes them unlimited. */
export function classicalCombinations(
  model: string,
  states: Record<string, ParamState>,
  grid: boolean
): number | null {
  let count = 1;
  for (const item of CLASSICAL_TUNABLES[model] ?? []) {
    const state = states[item.name];
    if (!state?.on) {
      continue;
    }
    if (usesRange(item, state, grid)) {
      return null;
    }
    const values = parseList(item, state);
    count *= typeof values === 'string' ? 0 : values.length;
  }
  return count;
}
