import type {
  BuiltinDataset,
  ClassWeights,
  Dataset,
  DatasetRef,
  FeatureTraining,
  LossName,
  LossTask,
  Optimizer,
  TrainingConfig,
  TrainingPlan,
  TrainingStatus,
  TuningInput,
  TuningMethod,
  TuningMetric,
  TuningSpace,
} from '@/api/types';
import { currentLocale } from '@/i18n';
import { isOwnDataset } from './access';
import {
  CLASSICAL_TUNABLES,
  classicalCombinations,
  classicalSpace,
  FOLDS,
  METRICS,
  paramStates,
  parseList,
  rangeError,
  usesRange,
  type ParamState,
} from './search';

/**
 * A number field as typed. NumberInput reports unfinished input such as "0.00" or "12." as a
 * string; it is kept as it is so typing is not interrupted, and only converted for validation
 * and the API.
 */
export type NumberField = number | string;

/** Form state of the training settings; '' is an empty field. */
export interface SettingsValues {
  epochs: NumberField;
  batch_size: NumberField;
  learning_rate: NumberField;
  optimizer: Optimizer;
  device: string;
  /**
   * Percent (0–50), sent as a fraction. 0: no validation data, so the weights of the last
   * epoch are kept and early stopping and a search are not possible.
   */
  val_split: NumberField;
  /** Hold out a test set (built-in datasets always test on their official test images). */
  test_enabled: boolean;
  /** Percent (5–50) of the data for the test set, sent as a fraction. */
  test_split: NumberField;
  weight_decay: NumberField;
  /** Empty: use all samples. */
  max_samples: NumberField;
  /** Empty: a random seed. */
  seed: NumberField;
  /** Early stopping: epochs without improvement of the validation loss. Empty: off. */
  patience: NumberField;
  /** Keep the weights of the epoch with the lowest validation loss (not the last one). */
  restore_best: boolean;
  /**
   * Segmentation has a batch size of its own: every layer keeps its output for all pixels of
   * a batch, so batches are much smaller than for other data.
   */
  segmentation_batch_size: NumberField;
  /**
   * The loss of a neural network ('auto': the default of the task, see `LOSS_OPTIONS`); a
   * loss of another task (the data changed) counts as 'auto'.
   */
  loss: LossName;
  /** weighted_cross_entropy and focal: balanced from the class counts, given, or none. */
  class_weights_mode: ClassWeightsMode;
  /** The given class weights: "1, 10" (one per class) or "fraud: 10, ok: 1". */
  class_weights_text: string;
  /** focal: 0 to 10 (0 is the cross-entropy). */
  focal_gamma: NumberField;
  /** cross_entropy and weighted_cross_entropy of a classification: 0 to 0.3. */
  label_smoothing: NumberField;
  /** huber: in standard deviations of the target. */
  huber_delta: NumberField;
  /**
   * Hybrid networks: the neural layers learn the features with a temporary head first
   * ('pretrain'), or keep their random weights ('frozen', a quick baseline).
   */
  feature_training: FeatureTraining;

  /** Hyperparameter search instead of a single training. */
  search: boolean;
  /**
   * Hybrid networks: the search also varies the training settings of the neural layers (the
   * `tune_*` fields below); every trial then learns the features again.
   */
  search_neural: boolean;
  search_method: TuningMethod;
  /** Random search: number of trials. */
  search_trials: NumberField;
  tune_learning_rate: boolean;
  /** Random search: learning rates are drawn log-uniformly between these two. */
  lr_min: NumberField;
  lr_max: NumberField;
  /** Grid search: the learning rates to try. */
  lr_values: number[];
  tune_batch_size: boolean;
  batch_sizes: number[];
  tune_optimizer: boolean;
  optimizers: Optimizer[];
  tune_weight_decay: boolean;
  weight_decays: number[];
  /** The losses to try (with a known task); needs another objective metric than the loss. */
  tune_loss: boolean;
  losses: LossName[];
  tune_focal_gamma: boolean;
  focal_gammas: number[];
  tune_label_smoothing: boolean;
  label_smoothings: number[];
  tune_huber_delta: boolean;
  huber_deltas: number[];
  /** What ranks the trials (networks and classical models). */
  metric: TuningMetric;
  /** Classical models: compare the trials by cross-validation on the training rows. */
  cv_enabled: boolean;
  cv_folds: NumberField;
  /**
   * Classical models: the parameters of the model brick to search, by model type and name
   * (only what differs from the defaults needs to be there).
   */
  classical_params: Record<string, Record<string, Partial<ParamState>>>;
}

const NUMBER_FIELDS = [
  'epochs',
  'batch_size',
  'segmentation_batch_size',
  'learning_rate',
  'val_split',
  'test_split',
  'weight_decay',
  'max_samples',
  'seed',
  'patience',
  'search_trials',
  'lr_min',
  'lr_max',
  'cv_folds',
  'focal_gamma',
  'label_smoothing',
  'huber_delta',
] as const;

const BOOLEAN_FIELDS = [
  'test_enabled',
  'restore_best',
  'search',
  'search_neural',
  'tune_learning_rate',
  'tune_batch_size',
  'tune_optimizer',
  'tune_weight_decay',
  'tune_loss',
  'tune_focal_gamma',
  'tune_label_smoothing',
  'tune_huber_delta',
  'cv_enabled',
] as const;

/** Values offered by the search form. */
export const LR_PRESETS = [0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001];
export const BATCH_SIZE_PRESETS = [8, 16, 32, 64, 128, 256, 512, 1024];
export const WEIGHT_DECAY_PRESETS = [0, 0.00001, 0.0001, 0.001, 0.01, 0.1];
export const FOCAL_GAMMA_PRESETS = [0, 0.5, 1, 2, 3, 5];
export const LABEL_SMOOTHING_PRESETS = [0, 0.05, 0.1, 0.2, 0.3];
export const HUBER_DELTA_PRESETS = [0.1, 0.25, 0.5, 1, 2, 5];

export const DEFAULT_SETTINGS: SettingsValues = {
  epochs: 10,
  batch_size: 64,
  learning_rate: 0.001,
  optimizer: 'adam',
  device: 'auto',
  val_split: 20,
  test_enabled: false,
  test_split: 20,
  weight_decay: 0,
  max_samples: '',
  seed: '',
  patience: '',
  restore_best: true,
  segmentation_batch_size: 8,
  loss: 'auto',
  class_weights_mode: 'balanced',
  class_weights_text: '',
  focal_gamma: 2,
  label_smoothing: 0,
  huber_delta: 1,
  feature_training: 'pretrain',
  search: false,
  search_neural: false,
  search_method: 'random',
  search_trials: 10,
  tune_learning_rate: true,
  lr_min: 0.0001,
  lr_max: 0.01,
  lr_values: [0.01, 0.001, 0.0001],
  tune_batch_size: true,
  batch_sizes: [32, 64, 128],
  tune_optimizer: false,
  optimizers: ['adam', 'sgd'],
  tune_weight_decay: false,
  weight_decays: [0, 0.0001, 0.001],
  tune_loss: false,
  losses: ['cross_entropy', 'focal'],
  tune_focal_gamma: false,
  focal_gammas: [1, 2, 3],
  tune_label_smoothing: false,
  label_smoothings: [0, 0.1],
  tune_huber_delta: false,
  huber_deltas: [0.5, 1, 2],
  metric: 'loss',
  cv_enabled: false,
  cv_folds: FOLDS.default,
  classical_params: {},
};

/** Same limits as backend/brickdnn/training/config.py. */
export const LIMITS = {
  epochs: { min: 1, max: 1000 },
  batch_size: { min: 1, max: 8192 },
  learning_rate: { min: 0, max: 10 },
  weight_decay: { min: 0, max: 1 },
  val_split: { min: 0, max: 50 },
  test_split: { min: 5, max: 50 },
  /** Validation and test split together, percent. */
  held_out: { max: 80 },
  max_samples: { min: 10, max: 1_000_000_000 },
  seed: { min: 0, max: 2 ** 31 - 1 },
  patience: { min: 1, max: 1000 },
  /** Same limits as backend/brickdnn/training/losses.py. */
  focal_gamma: { min: 0, max: 10 },
  label_smoothing: { min: 0, max: 0.3 },
  /** Greater than the minimum. */
  huber_delta: { min: 0, max: 10 },
  class_weight: { max: 1000 },
  /** Same limits as backend/brickdnn/training/tuning.py. */
  search_trials: { min: 1, max: 100 },
  cv_folds: { min: FOLDS.min, max: FOLDS.max },
} as const;

export const OPTIMIZERS: { value: Optimizer; label: string }[] = [
  { value: 'adam', label: 'Adam' },
  { value: 'adamw', label: 'AdamW' },
  { value: 'sgd', label: 'SGD' },
];

export const FEATURE_TRAININGS: { value: FeatureTraining; label: string }[] = [
  { value: 'pretrain', label: 'Learn them (temporary head on the target)' },
  { value: 'frozen', label: 'Frozen random features (no training)' },
];

export type ClassWeightsMode = 'balanced' | 'custom' | 'none';

export interface LossOption {
  value: Exclude<LossName, 'auto'>;
  label: string;
  /** When to use it, below the field. */
  help: string;
}

/**
 * The losses per task, the default first (like backend/brickdnn/training/losses.py). The
 * curves, early stopping and the kept epoch use the chosen loss.
 */
export const LOSS_OPTIONS: Record<LossTask, LossOption[]> = {
  classification: [
    {
      value: 'cross_entropy',
      label: 'Cross-entropy',
      help: 'The default: the log loss of the predicted probabilities.',
    },
    {
      value: 'weighted_cross_entropy',
      label: 'Weighted cross-entropy',
      help: 'Rare classes count more (balanced: by their counts in the training rows). For imbalanced data such as fraud, instead of oversampling.',
    },
    {
      value: 'focal',
      label: 'Focal loss',
      help: 'Well-classified samples count less, so the training concentrates on the hard and rare ones.',
    },
  ],
  regression: [
    {
      value: 'mse',
      label: 'Mean squared error',
      help: 'The default, of the standardised target: large errors count most.',
    },
    {
      value: 'mae',
      label: 'Mean absolute error',
      help: 'Every error counts by its size: less sensitive to outliers.',
    },
    {
      value: 'huber',
      label: 'Huber loss',
      help: 'Squared for small errors, linear for large ones (beyond delta).',
    },
  ],
  segmentation: [
    { value: 'cross_entropy', label: 'Cross-entropy', help: 'The cross-entropy of every pixel.' },
    {
      value: 'cross_entropy_dice',
      label: 'Cross-entropy + Dice',
      help: 'The Dice term helps classes that cover only a few pixels.',
    },
    {
      value: 'weighted_cross_entropy',
      label: 'Weighted cross-entropy',
      help: 'Pixels of rare classes count more (balanced: by the pixel counts of the training images).',
    },
  ],
};

/** All losses with their labels (for runs and trials). */
export const LOSS_LABELS: Record<string, string> = Object.fromEntries(
  Object.values(LOSS_OPTIONS).flatMap((list) => list.map((option) => [option.value, option.label]))
);

/** Which losses use which setting (the others ignore it). */
export const LOSS_PARAMS: Record<LossParam, readonly LossName[]> = {
  class_weights: ['weighted_cross_entropy', 'focal'],
  focal_gamma: ['focal'],
  label_smoothing: ['cross_entropy', 'weighted_cross_entropy'],
  huber_delta: ['huber'],
};

export type LossParam = 'class_weights' | 'focal_gamma' | 'label_smoothing' | 'huber_delta';

/** Whether `loss` (of `task`) uses `param`; label smoothing is for classification only. */
export function usesParam(param: LossParam, loss: LossName, task: LossTask): boolean {
  if (param === 'label_smoothing' && task !== 'classification') {
    return false;
  }
  return LOSS_PARAMS[param].includes(loss);
}

/** The loss the form trains with for `task`: the chosen one, or the task's default. */
export function effectiveLoss(values: SettingsValues, task: LossTask): Exclude<LossName, 'auto'> {
  const options = LOSS_OPTIONS[task];
  return options.find((option) => option.value === values.loss)?.value ?? options[0].value;
}

/** The losses of a search for `task`, in the order of the options ('auto': the default). */
export function searchedLosses(
  values: SettingsValues,
  task: LossTask
): Exclude<LossName, 'auto'>[] {
  const options = LOSS_OPTIONS[task].map((option) => option.value);
  const chosen = values.losses.map((name) => (name === 'auto' ? options[0] : name));
  return options.filter((name) => chosen.includes(name));
}

/**
 * The losses the trials use: those of a search over the loss, otherwise the chosen one. No
 * task: none (the loss is not offered then).
 */
export function lossesInUse(
  values: SettingsValues,
  task: LossTask | null
): Exclude<LossName, 'auto'>[] {
  if (task === null) {
    return [];
  }
  if (values.search && values.tune_loss) {
    const searched = searchedLosses(values, task);
    return searched.length > 0 ? searched : [effectiveLoss(values, task)];
  }
  return [effectiveLoss(values, task)];
}

/** Whether a loss the run uses takes `param`. */
export function lossUses(values: SettingsValues, task: LossTask | null, param: LossParam) {
  return task !== null && lossesInUse(values, task).some((loss) => usesParam(param, loss, task));
}

/**
 * The class weight modes: a weighted cross-entropy always has weights, the focal loss may
 * have none (alpha 1); with both in a search, the weighted cross-entropy has none as balanced.
 */
export function classWeightModes(losses: LossName[]): { value: ClassWeightsMode; label: string }[] {
  const given = [
    { value: 'balanced' as const, label: 'Balanced (from the class counts)' },
    { value: 'custom' as const, label: 'Custom' },
  ];
  return losses.length === 1 && losses[0] === 'focal'
    ? [{ value: 'none' as const, label: 'None' }, ...given]
    : given;
}

/**
 * Parses given class weights: "1, 10" (one per class) or "fraud: 10, ok: 1" (by class name or
 * index; missing classes get 1). Returns the weights or a message. `classes`: the class names
 * of the data (null: not known, not checked).
 */
export function parseClassWeights(
  text: string,
  classes: string[] | null = null
): number[] | Record<string, number> | string {
  const items = text
    .split(/[,;\n]/)
    .map((item) => item.trim())
    .filter(Boolean);
  if (items.length === 0) {
    return 'Enter a weight per class, e.g. "1, 10", or "fraud: 10"';
  }
  const number = (value: string) => {
    const weight = value.trim() === '' ? NaN : Number(value.trim());
    return Number.isFinite(weight) && weight > 0 && weight <= LIMITS.class_weight.max
      ? weight
      : null;
  };
  const invalid = `Weights greater than 0 and at most ${LIMITS.class_weight.max}`;
  if (items.some((item) => /[:=]/.test(item))) {
    const weights: Record<string, number> = {};
    for (const item of items) {
      const [name, value = ''] = item.split(/[:=]/, 2);
      const weight = number(value);
      if (!name.trim() || weight === null) {
        return invalid;
      }
      const key = name.trim();
      const known =
        classes === null ||
        classes.some((label) => label.toLowerCase() === key.toLowerCase()) ||
        (/^\d+$/.test(key) && Number(key) < classes.length);
      if (!known) {
        return `'${key}' is not a class: ${(classes ?? []).slice(0, 8).join(', ')}`;
      }
      weights[key] = weight;
    }
    return weights;
  }
  const weights = items.map(number);
  if (weights.some((weight) => weight === null)) {
    return invalid;
  }
  if (classes !== null && weights.length !== classes.length) {
    return `One weight per class: ${classes.length} (${classes.slice(0, 8).join(', ')})`;
  }
  return weights as number[];
}

/**
 * The class weights sent with the settings for the `losses` of the run: null for none,
 * undefined while the given ones are not valid.
 */
export function classWeightsOf(
  values: SettingsValues,
  losses: LossName[],
  classes: string[] | null = null
): ClassWeights | null | undefined {
  const modes = classWeightModes(losses);
  const mode = modes.some((option) => option.value === values.class_weights_mode)
    ? values.class_weights_mode
    : modes[0].value;
  if (mode === 'none') {
    return null;
  }
  if (mode === 'balanced') {
    return 'balanced';
  }
  const parsed = parseClassWeights(values.class_weights_text, classes);
  return typeof parsed === 'string' ? undefined : parsed;
}

/**
 * The task whose losses the form offers, and the class names of the data (from the training
 * check; null: not known, the given class weights are not checked against them).
 */
export interface LossScope {
  task: LossTask;
  classes?: string[] | null;
}

/**
 * The loss scope of the data of a training check: a network with a single output of classes,
 * values or a class per pixel (hybrid networks: of their temporary head). Null without a plan
 * and for several outputs (demo data), which keep the default of their tasks.
 */
export function lossScopeFor(plan: TrainingPlan | null | undefined): LossScope | null {
  if (!plan || plan.targets.length !== 1 || (plan.classical && !plan.hybrid)) {
    return null;
  }
  const target = plan.targets[0];
  const names = (count: number) =>
    Array.from({ length: count }, (_, index) => plan.class_names?.[index] ?? String(index));
  if (target.kind === 'mask') {
    return { task: 'segmentation', classes: names(target.num_classes) };
  }
  if (target.kind === 'class') {
    return { task: 'classification', classes: names(target.num_classes) };
  }
  return { task: 'regression', classes: null };
}

/**
 * A segmentation training: its batch size is a setting of its own, and `maxBatch` is the
 * largest batch that fits into the memory of a training on this server (null: not known).
 */
export interface SegmentationScope {
  maxBatch: number | null;
}

/**
 * The largest batch of a segmentation that fits into `memoryMb` (the server's limit), from the
 * plan of the training check: a step needs about `fixed_bytes + batch × sample_bytes`.
 */
export function maxSegmentationBatch(
  plan: { fixed_bytes: number; sample_bytes: number } | null | undefined,
  memoryMb: number | null | undefined
): number | null {
  if (!plan || !memoryMb || !(plan.sample_bytes > 0)) {
    return null;
  }
  return Math.floor((memoryMb * 1024 * 1024 - plan.fixed_bytes) / plan.sample_bytes);
}

export type SettingsErrors = Partial<Record<keyof SettingsValues, string>> & {
  /** Classical search: problems with a parameter, by name. */
  params?: Record<string, string>;
};

/** '' for an empty field, the number, or null if the text is not a number. */
export function parseField(value: NumberField): number | '' | null {
  if (typeof value === 'number') {
    return Number.isFinite(value) ? value : null;
  }
  const text = value.trim().replace(/[\s,%]/g, '');
  if (text === '') {
    return '';
  }
  const number = Number(text);
  return Number.isFinite(number) ? number : null;
}

function wholeNumber(
  field: NumberField,
  { min, max }: { min: number; max: number },
  required: boolean
): string | null {
  const value = parseField(field);
  if (value === '') {
    return required ? 'Required' : null;
  }
  if (value === null || !Number.isInteger(value)) {
    return 'Enter a whole number';
  }
  if (value < min || value > max) {
    return `Between ${min.toLocaleString(currentLocale())} and ${max.toLocaleString(currentLocale())}`;
  }
  return null;
}

function decimal(
  field: NumberField,
  check: (value: number) => boolean,
  message: string,
  required: boolean
): string | null {
  const value = parseField(field);
  if (value === '') {
    return required ? 'Required' : null;
  }
  if (value === null) {
    return 'Enter a number';
  }
  return check(value) ? null : message;
}

/**
 * Which settings the search varies (none when the search is off). The loss and its settings
 * need the task of the data (`task`, null: not known): a setting of a loss only counts when a
 * loss of the trials uses it.
 */
export function tunedSettings(values: SettingsValues, task: LossTask | null = null) {
  const on = values.search;
  const param = (tune: boolean, name: LossParam) => on && tune && lossUses(values, task, name);
  return {
    learning_rate: on && values.tune_learning_rate,
    batch_size: on && values.tune_batch_size,
    optimizer: on && values.tune_optimizer,
    weight_decay: on && values.tune_weight_decay,
    loss: on && task !== null && values.tune_loss,
    focal_gamma: param(values.tune_focal_gamma, 'focal_gamma'),
    label_smoothing: param(values.tune_label_smoothing, 'label_smoothing'),
    huber_delta: param(values.tune_huber_delta, 'huber_delta'),
  };
}

/**
 * Whether a hybrid network's search varies the training settings of its neural layers too
 * (not with frozen features, which are not trained).
 */
export function searchesNeural(values: SettingsValues): boolean {
  return values.search && values.search_neural && values.feature_training !== 'frozen';
}

/** The training settings of the neural layers, which frozen features do not use. */
export const FROZEN_UNUSED = [
  'epochs',
  'batch_size',
  'segmentation_batch_size',
  'learning_rate',
  'optimizer',
  'weight_decay',
  'patience',
  'restore_best',
  'loss',
  'class_weights_text',
  'focal_gamma',
  'label_smoothing',
  'huber_delta',
] as const;

/** Number of combinations of the listed values, or null if a range makes it unlimited. */
function combinations(values: SettingsValues, task: LossTask | null = null): number | null {
  const tuned = tunedSettings(values, task);
  if (tuned.learning_rate && values.search_method === 'random') {
    return null;
  }
  let count = 1;
  if (tuned.learning_rate) {
    count *= values.lr_values.length;
  }
  if (tuned.batch_size) {
    count *= values.batch_sizes.length;
  }
  if (tuned.optimizer) {
    count *= values.optimizers.length;
  }
  if (tuned.weight_decay) {
    count *= values.weight_decays.length;
  }
  if (tuned.loss && task !== null) {
    count *= Math.max(searchedLosses(values, task).length, 1);
  }
  if (tuned.focal_gamma) {
    count *= values.focal_gammas.length;
  }
  if (tuned.label_smoothing) {
    count *= values.label_smoothings.length;
  }
  if (tuned.huber_delta) {
    count *= values.huber_deltas.length;
  }
  return count;
}

/**
 * How many trials the search will run: every combination for a grid, otherwise the chosen
 * number (but each combination of lists at most once, like the server). Null if unknown.
 */
export function plannedTrials(values: SettingsValues, task: LossTask | null = null): number | null {
  const combos = combinations(values, task);
  if (values.search_method === 'grid') {
    return combos;
  }
  const trials = parseField(values.search_trials);
  if (typeof trials !== 'number' || !Number.isInteger(trials) || trials < 1) {
    return null;
  }
  return combos === null ? trials : Math.min(trials, combos);
}

/**
 * Classical models (scikit-learn) are fitted once: only the splits, Max. samples, the seed
 * and a search over the parameters of the model brick apply. `maxRows` is the most training
 * rows the model is fitted on on this server; `model` the type of the model brick.
 */
export interface ClassicalScope {
  maxRows: number | null;
  model?: string | null;
}

/** The form state of the parameters of a classical `model` (defaults where none is saved). */
export function classicalParams(values: SettingsValues, model: string) {
  return paramStates(model, values.classical_params[model]);
}

/**
 * How many trials a classical search runs: every combination for a grid, otherwise the
 * chosen number (each combination of lists at most once). Null if unknown.
 */
export function plannedClassicalTrials(values: SettingsValues, model: string): number | null {
  const grid = values.search_method === 'grid';
  const combos = classicalCombinations(model, classicalParams(values, model), grid);
  if (grid) {
    return combos;
  }
  const trials = parseField(values.search_trials);
  if (typeof trials !== 'number' || !Number.isInteger(trials) || trials < 1) {
    return null;
  }
  return combos === null ? trials : Math.min(trials, combos);
}

/**
 * How many trials the search of a hybrid network runs: the combinations of its neural
 * settings (if searched) times those of the parameters of its classical `model`, like the
 * server. Null if unknown.
 */
export function plannedHybridTrials(
  values: SettingsValues,
  model: string,
  task: LossTask | null = null
): number | null {
  const grid = values.search_method === 'grid';
  const neural = searchesNeural(values) ? combinations(values, task) : 1;
  const classical = classicalCombinations(model, classicalParams(values, model), grid);
  const combos = neural === null || classical === null ? null : neural * classical;
  if (grid) {
    return combos;
  }
  const trials = parseField(values.search_trials);
  if (typeof trials !== 'number' || !Number.isInteger(trials) || trials < 1) {
    return null;
  }
  return combos === null ? trials : Math.min(trials, combos);
}

/** Shown when a setting needs validation data that the run will not have. */
export const NO_VALIDATION_PATIENCE =
  'Early stopping needs validation data: set a validation split above 0, or leave this empty.';
export const NO_VALIDATION_SEARCH =
  'A hyperparameter search needs validation data to compare its trials: set a validation split above 0.';
const PREPARED_PATIENCE =
  'Early stopping needs validation data, and this dataset has no validation rows: leave this empty, or give it a validation share in the Data tab.';
const PREPARED_SEARCH =
  'A hyperparameter search needs validation data to compare its trials, and this dataset has no validation rows: give it a validation share in the Data tab.';
export const LOSS_SEARCH_METRIC =
  'A search over losses cannot rank its trials by the loss: losses of different kinds are not comparable. Choose another objective metric, e.g. PR AUC or balanced accuracy (regression: RMSE or MAE).';
export const NEURAL_SEARCH_NO_VALIDATION =
  'Searching the neural settings needs validation data: every trial is ranked by its classical model on the validation rows. Set a validation split above 0, or search only the classical model.';

/**
 * Whether the run gets validation data: the validation rows of a dataset that brings its own
 * split (`validationRows`), otherwise a validation split above 0.
 */
export function hasValidation(
  values: SettingsValues,
  validationRows: number | null = null
): boolean {
  if (validationRows !== null) {
    return validationRows > 0;
  }
  const value = parseField(values.val_split);
  return !(typeof value === 'number' && value === 0);
}

function validateSplits(values: SettingsValues, errors: SettingsErrors) {
  const split = (value: number) => value >= 0 && value <= 50;
  const testShare = (value: number) => value >= 5 && value <= 50;
  const set = (key: keyof SettingsValues, message: string | null) => {
    if (message) {
      errors[key] = message;
    }
  };
  set('val_split', decimal(values.val_split, split, 'Between 0% and 50%', true));
  if (values.test_enabled) {
    set('test_split', decimal(values.test_split, testShare, 'Between 5% and 50%', true));
    const val = parseField(values.val_split);
    const test = parseField(values.test_split);
    if (
      !errors.val_split &&
      !errors.test_split &&
      typeof val === 'number' &&
      typeof test === 'number' &&
      val + test > LIMITS.held_out.max
    ) {
      errors.test_split = 'Validation and test together: at most 80%';
    }
  }
}

function validateClassicalSearch(
  values: SettingsValues,
  model: string,
  errors: SettingsErrors,
  hybrid = false,
  task: LossTask | null = null
) {
  const grid = values.search_method === 'grid';
  // A hybrid network may search its neural settings alone, without cross-validation.
  const neural = hybrid && searchesNeural(values);
  if (!grid) {
    const trials = wholeNumber(values.search_trials, LIMITS.search_trials, true);
    if (trials) {
      errors.search_trials = trials;
    }
  }
  if (values.cv_enabled && !neural) {
    const folds = wholeNumber(values.cv_folds, LIMITS.cv_folds, true);
    if (folds) {
      errors.cv_folds = folds;
    }
  }
  const states = classicalParams(values, model);
  const params: Record<string, string> = {};
  let ticked = 0;
  for (const item of CLASSICAL_TUNABLES[model] ?? []) {
    const state = states[item.name];
    if (!state?.on) {
      continue;
    }
    ticked += 1;
    const problem = usesRange(item, state, grid) ? rangeError(item, state) : null;
    const list = usesRange(item, state, grid) ? [] : parseList(item, state);
    if (problem || typeof list === 'string') {
      params[item.name] = problem ?? (list as string);
    }
  }
  if (Object.keys(params).length > 0) {
    errors.params = params;
  }
  if (neural) {
    validateNeuralSearch(values, errors, ticked, task);
    return;
  }
  const combos = classicalCombinations(model, states, grid);
  if (ticked === 0) {
    errors.search = 'Choose at least one parameter to search.';
  } else if (grid && combos !== null && combos > LIMITS.search_trials.max) {
    errors.search = `The grid has ${combos} combinations, at most ${LIMITS.search_trials.max} are possible. Choose fewer values.`;
  }
}

/**
 * The search of a hybrid network over its neural settings: they need validation data, and
 * with the `classicalTicked` parameters of its model at least one thing must be searched;
 * a grid has at most the most trials.
 */
function validateNeuralSearch(
  values: SettingsValues,
  errors: SettingsErrors,
  classicalTicked: number,
  task: LossTask | null = null
) {
  const set = (key: keyof SettingsValues, message: string | null) => {
    if (message) {
      errors[key] = message;
    }
  };
  validateSearchSettings(values, null, set, task);
  const neural = Object.values(tunedSettings(values, task)).filter(Boolean).length;
  if (neural + classicalTicked === 0) {
    errors.search = 'Choose at least one setting or parameter to search.';
  }
}

function validateClassical(
  values: SettingsValues,
  { maxRows, model = null }: ClassicalScope
): SettingsErrors {
  const errors: SettingsErrors = {};
  validateSplits(values, errors);
  if (values.search && model) {
    validateClassicalSearch(values, model, errors);
  }
  const limit = { min: LIMITS.max_samples.min, max: maxRows ?? LIMITS.max_samples.max };
  const samples = wholeNumber(values.max_samples, limit, false);
  if (samples) {
    const value = parseField(values.max_samples);
    errors.max_samples =
      maxRows !== null && typeof value === 'number' && value > maxRows
        ? `At most ${maxRows.toLocaleString(currentLocale())} on this server`
        : samples;
  }
  const seed = wholeNumber(values.seed, LIMITS.seed, false);
  if (seed) {
    errors.seed = seed;
  }
  return errors;
}

/**
 * Problems with the settings, by field. `validationRows`: the validation rows of a dataset that
 * brings its own split (null: the validation split decides); without validation data early
 * stopping and a search are errors.
 */
export function validateSettings(
  values: SettingsValues,
  {
    classical = null,
    hybrid = null,
    segmentation = null,
    validationRows = null,
    metrics = null,
    loss = null,
  }: {
    classical?: ClassicalScope | null;
    /**
     * A hybrid network: the neural settings apply to its features, a search varies the
     * parameters of its classical `model` (a classical search, which may cross-validate).
     */
    hybrid?: { model: string | null } | null;
    segmentation?: SegmentationScope | null;
    validationRows?: number | null;
    /** The objective metrics the data allows (null: not known yet, not checked). */
    metrics?: TuningMetric[] | null;
    /** The task whose losses are offered (a segmentation's without it; null: none). */
    loss?: LossScope | null;
  } = {}
): SettingsErrors {
  const scope = lossScopeOf(loss, segmentation !== null);
  const task = scope?.task ?? null;
  if (hybrid) {
    const errors = validateSettings(
      { ...values, search: false },
      { segmentation, validationRows, loss }
    );
    if (values.feature_training === 'frozen') {
      // Not used (and not sent): the neural layers keep their random weights.
      for (const key of FROZEN_UNUSED) {
        delete errors[key];
      }
    }
    if (values.search && hybrid.model) {
      validateClassicalSearch(values, hybrid.model, errors, true, task);
      if (searchesNeural(values)) {
        const grid = values.search_method === 'grid';
        const combos = plannedHybridTrials(values, hybrid.model, task);
        if (!errors.search && !hasValidation(values, validationRows)) {
          errors.search = NEURAL_SEARCH_NO_VALIDATION;
        } else if (!errors.search && grid && combos !== null && combos > LIMITS.search_trials.max) {
          errors.search = `The grid has ${combos} combinations, at most ${LIMITS.search_trials.max} are possible. Choose fewer values.`;
        }
      }
      checkMetric(values, metrics, errors);
    }
    return errors;
  }
  if (classical) {
    const errors = validateClassical(values, classical);
    if (values.search && classical.model) {
      checkMetric(values, metrics, errors);
    }
    return errors;
  }
  const errors: SettingsErrors = {};
  const set = (key: keyof SettingsValues, message: string | null) => {
    if (message) {
      errors[key] = message;
    }
  };
  const tuned = tunedSettings(values, task);
  set('epochs', wholeNumber(values.epochs, LIMITS.epochs, true));
  const maxBatch = segmentation?.maxBatch ?? null;
  const tooLarge = (size: number) =>
    maxBatch !== null && size > maxBatch
      ? maxBatch >= 1
        ? `At most ${maxBatch.toLocaleString(currentLocale())}: larger batches do not fit into memory`
        : 'No batch fits into memory: make the Input smaller'
      : null;
  if (!tuned.batch_size && segmentation) {
    const size = parseField(values.segmentation_batch_size);
    set(
      'segmentation_batch_size',
      wholeNumber(values.segmentation_batch_size, LIMITS.batch_size, true) ??
        (typeof size === 'number' ? tooLarge(size) : null)
    );
  } else if (!tuned.batch_size) {
    set('batch_size', wholeNumber(values.batch_size, LIMITS.batch_size, true));
  }
  const learningRate = (lr: number) => lr > 0 && lr <= LIMITS.learning_rate.max;
  if (!tuned.learning_rate) {
    set(
      'learning_rate',
      decimal(values.learning_rate, learningRate, 'Greater than 0, at most 10', true)
    );
  }
  if (!tuned.weight_decay) {
    set(
      'weight_decay',
      decimal(values.weight_decay, (wd) => wd >= 0 && wd <= 1, 'Between 0 and 1', false)
    );
  }
  validateSplits(values, errors);
  set('max_samples', wholeNumber(values.max_samples, LIMITS.max_samples, false));
  set('seed', wholeNumber(values.seed, LIMITS.seed, false));
  set('patience', wholeNumber(values.patience, LIMITS.patience, false));
  const validates = hasValidation(values, validationRows);
  const prepared = validationRows !== null;
  if (!validates && !errors.patience && parseField(values.patience) !== '') {
    errors.patience = prepared ? PREPARED_PATIENCE : NO_VALIDATION_PATIENCE;
  }
  if (scope) {
    validateLoss(values, scope, set);
  }

  if (values.search) {
    const grid = values.search_method === 'grid';
    validateSearchSettings(values, tooLarge, set, task);
    const combos = combinations(values, task);
    if (!validates) {
      errors.search = prepared ? PREPARED_SEARCH : NO_VALIDATION_SEARCH;
    } else if (!Object.values(tuned).some(Boolean)) {
      errors.search = 'Choose at least one setting to search.';
    } else if (grid && combos !== null && combos > LIMITS.search_trials.max) {
      errors.search = `The grid has ${combos} combinations, at most ${LIMITS.search_trials.max} are possible. Choose fewer values.`;
    }
    checkMetric(values, metrics, errors);
    // A hybrid network ranks its trials by its classical model, whose loss does not change.
    if (
      !errors.metric &&
      tuned.loss &&
      task !== null &&
      searchedLosses(values, task).length > 1 &&
      values.metric === 'loss'
    ) {
      errors.metric = LOSS_SEARCH_METRIC;
    }
  }
  return errors;
}

/** The loss scope of the form: the given one, or a segmentation's. */
function lossScopeOf(loss: LossScope | null, segmentation: boolean): LossScope | null {
  return loss ?? (segmentation ? { task: 'segmentation' } : null);
}

/** The settings of the losses the run uses (those a search varies are checked there). */
function validateLoss(
  values: SettingsValues,
  scope: LossScope,
  set: (key: keyof SettingsValues, message: string | null) => void
) {
  const { task } = scope;
  const tuned = tunedSettings(values, task);
  const losses = lossesInUse(values, task);
  const uses = (param: LossParam) => losses.some((loss) => usesParam(param, loss, task));
  if (uses('class_weights') && classWeightsOf(values, losses) !== null) {
    const modes = classWeightModes(losses).map((option) => option.value);
    const mode = modes.includes(values.class_weights_mode) ? values.class_weights_mode : modes[0];
    if (mode === 'custom') {
      const parsed = parseClassWeights(values.class_weights_text, scope.classes ?? null);
      set('class_weights_text', typeof parsed === 'string' ? parsed : null);
    }
  }
  if (uses('focal_gamma') && !tuned.focal_gamma) {
    const { min, max } = LIMITS.focal_gamma;
    const inRange = (value: number) => value >= min && value <= max;
    set('focal_gamma', decimal(values.focal_gamma, inRange, `Between ${min} and ${max}`, true));
  }
  if (uses('label_smoothing') && !tuned.label_smoothing) {
    const { min, max } = LIMITS.label_smoothing;
    const inRange = (value: number) => value >= min && value <= max;
    set(
      'label_smoothing',
      decimal(values.label_smoothing, inRange, `Between ${min} and ${max}`, false)
    );
  }
  if (uses('huber_delta') && !tuned.huber_delta) {
    const { min, max } = LIMITS.huber_delta;
    const inRange = (value: number) => value > min && value <= max;
    set(
      'huber_delta',
      decimal(values.huber_delta, inRange, `Greater than ${min}, at most ${max}`, true)
    );
  }
}

/**
 * The values of the searched training settings (and the trials of a random search).
 * `tooLarge`: a segmentation's limit of the batch size (null: none).
 */
function validateSearchSettings(
  values: SettingsValues,
  tooLarge: ((size: number) => string | null) | null,
  set: (key: keyof SettingsValues, message: string | null) => void,
  task: LossTask | null = null
) {
  const tuned = tunedSettings(values, task);
  const grid = values.search_method === 'grid';
  const learningRate = (lr: number) => lr > 0 && lr <= LIMITS.learning_rate.max;
  if (!grid) {
    set('search_trials', wholeNumber(values.search_trials, LIMITS.search_trials, true));
  }
  if (tuned.learning_rate && grid && values.lr_values.length === 0) {
    set('lr_values', 'Choose at least one');
  }
  if (tuned.learning_rate && !grid) {
    const range = (field: NumberField) =>
      decimal(field, learningRate, 'Greater than 0, at most 10', true);
    const lowError = range(values.lr_min);
    const highError = range(values.lr_max);
    set('lr_min', lowError);
    set('lr_max', highError);
    const lo = parseField(values.lr_min);
    const hi = parseField(values.lr_max);
    if (!lowError && !highError && typeof lo === 'number' && typeof hi === 'number' && lo > hi) {
      set('lr_min', 'At most the maximum');
    }
  }
  if (tuned.batch_size && values.batch_sizes.length === 0) {
    set('batch_sizes', 'Choose at least one');
  } else if (tuned.batch_size && tooLarge) {
    set('batch_sizes', tooLarge(Math.max(...values.batch_sizes)));
  }
  if (tuned.optimizer && values.optimizers.length === 0) {
    set('optimizers', 'Choose at least one');
  }
  if (tuned.weight_decay && values.weight_decays.length === 0) {
    set('weight_decays', 'Choose at least one');
  }
  if (tuned.loss && task !== null && searchedLosses(values, task).length === 0) {
    set('losses', 'Choose at least one');
  }
  if (tuned.focal_gamma && values.focal_gammas.length === 0) {
    set('focal_gammas', 'Choose at least one');
  }
  if (tuned.label_smoothing && values.label_smoothings.length === 0) {
    set('label_smoothings', 'Choose at least one');
  }
  if (tuned.huber_delta && values.huber_deltas.length === 0) {
    set('huber_deltas', 'Choose at least one');
  }
}

/** The chosen objective metric must fit the data (e.g. ROC AUC needs two classes). */
function checkMetric(
  values: SettingsValues,
  metrics: TuningMetric[] | null,
  errors: SettingsErrors
) {
  if (metrics !== null && !metrics.includes(values.metric)) {
    errors.metric = 'Not available for this data: choose another metric.';
  }
}

/** Fractions with at most four decimals: 12.5 % → 0.125. */
const fraction = (percent: number) => Number((percent / 100).toFixed(4));

/**
 * The training settings for the API. Built-in datasets (`builtin`) always test on their
 * official test images, so they need no test split. Settings varied by a search are left
 * out; the search sets them. A `classical` model only gets the settings that apply to it; a
 * `segmentation` uses its own batch size and has a loss to choose.
 */
export function toConfig(
  values: SettingsValues,
  {
    builtin = false,
    classical = false,
    hybrid = false,
    segmentation = false,
    loss = null,
  }: {
    builtin?: boolean;
    classical?: boolean;
    hybrid?: boolean;
    segmentation?: boolean;
    /** The task whose losses are offered (a segmentation's without it; null: none). */
    loss?: LossScope | null;
  } = {}
): Partial<TrainingConfig> {
  if (hybrid) {
    // The neural settings learn the features; those a search varies are left out, and
    // frozen features use none of them.
    const config: Partial<TrainingConfig> = {
      ...toConfig({ ...values, search: searchesNeural(values) }, { builtin, segmentation, loss }),
      feature_training: values.feature_training,
    };
    if (values.feature_training === 'frozen') {
      for (const key of [...FROZEN_UNUSED, 'class_weights']) {
        delete config[key as keyof TrainingConfig];
      }
    }
    return config;
  }
  const number = (field: NumberField) => {
    const value = parseField(field);
    return value === '' || value === null ? null : value;
  };
  const scope = lossScopeOf(loss, segmentation);
  const tuned = tunedSettings(values, scope?.task ?? null);
  const split = number(values.val_split);
  const testSplit = number(values.test_split);
  const test = values.test_enabled && !builtin && testSplit !== null ? fraction(testSplit) : 0;
  if (classical) {
    return {
      val_split: split === null ? undefined : fraction(split),
      test_split: test,
      max_samples: number(values.max_samples),
      seed: number(values.seed),
    };
  }
  const batchSize = segmentation ? values.segmentation_batch_size : values.batch_size;
  return {
    epochs: number(values.epochs) ?? undefined,
    batch_size: tuned.batch_size ? undefined : (number(batchSize) ?? undefined),
    learning_rate: tuned.learning_rate ? undefined : (number(values.learning_rate) ?? undefined),
    optimizer: tuned.optimizer ? undefined : values.optimizer,
    weight_decay: tuned.weight_decay ? undefined : (number(values.weight_decay) ?? 0),
    val_split: split === null ? undefined : fraction(split),
    test_split: test,
    device: values.device,
    max_samples: number(values.max_samples),
    seed: number(values.seed),
    patience: number(values.patience),
    restore_best: values.restore_best,
    ...(scope ? lossConfig(values, scope, tuned) : {}),
  };
}

/**
 * The loss and the settings it uses: the loss only when one was chosen ('auto' is the
 * default), the settings only for the losses of the run and when no search varies them; a
 * label smoothing of 0 is the default.
 */
function lossConfig(
  values: SettingsValues,
  scope: LossScope,
  tuned: ReturnType<typeof tunedSettings>
): Partial<TrainingConfig> {
  const { task } = scope;
  const config: Partial<TrainingConfig> = {};
  const loss = effectiveLoss(values, task);
  if (!tuned.loss && values.loss === loss) {
    config.loss = loss;
  }
  const losses = lossesInUse(values, task);
  const uses = (param: LossParam) => losses.some((name) => usesParam(param, name, task));
  const number = (field: NumberField) => {
    const value = parseField(field);
    return value === '' || value === null ? undefined : value;
  };
  if (uses('class_weights')) {
    const weights = classWeightsOf(values, losses, scope.classes ?? null);
    if (weights !== undefined) {
      config.class_weights = weights;
    }
  }
  if (uses('focal_gamma') && !tuned.focal_gamma) {
    config.focal_gamma = number(values.focal_gamma);
  }
  if (uses('label_smoothing') && !tuned.label_smoothing && number(values.label_smoothing)) {
    config.label_smoothing = number(values.label_smoothing);
  }
  if (uses('huber_delta') && !tuned.huber_delta) {
    config.huber_delta = number(values.huber_delta);
  }
  return config;
}

const ascending = (a: number, b: number) => a - b;

/**
 * The search settings for the API, or null for a single training. `classicalModel`: the
 * type of a classical model brick, whose parameters the search varies.
 */
export function toTuning(
  values: SettingsValues,
  classicalModel: string | null = null,
  { hybrid = false, loss = null }: { hybrid?: boolean; loss?: LossScope | null } = {}
): TuningInput | null {
  if (!values.search) {
    return null;
  }
  const task = loss?.task ?? null;
  const metric = values.metric !== 'loss' ? { metric: values.metric } : {};
  if (classicalModel) {
    const grid = values.search_method === 'grid';
    const trials = parseField(values.search_trials);
    const folds = parseField(values.cv_folds);
    // A hybrid network may also search its neural settings (then never cross-validated).
    const neural = hybrid && searchesNeural(values);
    return {
      method: values.search_method,
      ...(!grid && typeof trials === 'number' ? { trials } : {}),
      ...metric,
      ...(values.cv_enabled && !neural && typeof folds === 'number' ? { cv_folds: folds } : {}),
      space: classicalSpace(classicalModel, classicalParams(values, classicalModel), grid),
      ...(neural ? { neural_space: settingsSpace(values, task) } : {}),
    };
  }
  const grid = values.search_method === 'grid';
  const trials = parseField(values.search_trials);
  return {
    method: values.search_method,
    ...(!grid && typeof trials === 'number' ? { trials } : {}),
    ...metric,
    space: settingsSpace(values, task),
  };
}

/** The search space of the ticked training settings (the loss needs the `task`). */
function settingsSpace(values: SettingsValues, task: LossTask | null = null): TuningSpace {
  const tuned = tunedSettings({ ...values, search: true }, task);
  const grid = values.search_method === 'grid';
  const space: TuningSpace = {};
  if (tuned.learning_rate) {
    const lo = parseField(values.lr_min);
    const hi = parseField(values.lr_max);
    space.learning_rate = grid
      ? [...values.lr_values].sort(ascending)
      : { min: typeof lo === 'number' ? lo : 0, max: typeof hi === 'number' ? hi : 0 };
  }
  if (tuned.batch_size) {
    space.batch_size = [...values.batch_sizes].sort(ascending);
  }
  if (tuned.optimizer) {
    space.optimizer = OPTIMIZERS.map((option) => option.value).filter((name) =>
      values.optimizers.includes(name)
    );
  }
  if (tuned.weight_decay) {
    space.weight_decay = [...values.weight_decays].sort(ascending);
  }
  if (tuned.loss && task !== null) {
    space.loss = searchedLosses(values, task);
  }
  if (tuned.focal_gamma) {
    space.focal_gamma = [...values.focal_gammas].sort(ascending);
  }
  if (tuned.label_smoothing) {
    space.label_smoothing = [...values.label_smoothings].sort(ascending);
  }
  if (tuned.huber_delta) {
    space.huber_delta = [...values.huber_deltas].sort(ascending);
  }
  return space;
}

// --------------------------------------------------------------------------
// Data source
// --------------------------------------------------------------------------

export type SourceTab = 'random' | 'builtin' | 'upload';

export interface SourceChoice {
  tab: SourceTab;
  builtin: string | null;
  upload: number | null;
}

export const DEFAULT_SOURCE: SourceChoice = { tab: 'random', builtin: null, upload: null };

export function datasetRef(choice: SourceChoice): DatasetRef | null {
  if (choice.tab === 'random') {
    return { kind: 'random' };
  }
  if (choice.tab === 'builtin') {
    return choice.builtin ? { kind: 'builtin', id: choice.builtin } : null;
  }
  return choice.upload !== null ? { kind: 'upload', id: choice.upload } : null;
}

/** The data source as it is used for training (see `effectiveSource`). */
export interface EffectiveSource {
  choice: SourceChoice;
  /** null: no dataset to use (yet), e.g. while the uploads are loading. */
  ref: DatasetRef | null;
  builtin: BuiltinDataset | null;
  upload: Dataset | null;
  /** Name of the dataset in use; null for demo data or none. */
  name: string | null;
}

/**
 * A remembered upload may have been deleted, a built-in dataset may be unavailable. Without a
 * (valid) choice, the first built-in dataset and the newest own upload are used.
 */
export function effectiveSource(
  status: TrainingStatus,
  choice: SourceChoice,
  datasets: Dataset[] | null
): EffectiveSource {
  const available = status.builtin_datasets.filter((d) => d.available);
  const builtin = available.find((d) => d.id === choice.builtin) ?? available[0] ?? null;
  const upload =
    datasets?.find((d) => d.id === choice.upload) ??
    datasets?.find(isOwnDataset) ??
    datasets?.[0] ??
    null;
  const effective: SourceChoice = {
    ...choice,
    builtin: builtin ? builtin.id : null,
    upload: upload ? upload.id : null,
  };
  const name =
    choice.tab === 'builtin'
      ? (builtin?.name ?? null)
      : choice.tab === 'upload'
        ? (upload?.name ?? null)
        : null;
  return { choice: effective, ref: datasetRef(effective), builtin, upload, name };
}

// --------------------------------------------------------------------------
// Persistence (per browser)
// --------------------------------------------------------------------------

const SETTINGS_KEY = 'brickdnn:training-settings';
const SOURCE_KEY = 'brickdnn:training-source';

function read<T extends object>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    const data = raw ? (JSON.parse(raw) as Partial<T>) : null;
    if (!data || typeof data !== 'object') {
      return fallback;
    }
    // Only keep known keys with plain values (booleans and lists where those are expected).
    const result = { ...fallback };
    for (const name of Object.keys(fallback) as (keyof T)[]) {
      const value = data[name];
      const expected = fallback[name];
      const object = (item: unknown) =>
        typeof item === 'object' && item !== null && !Array.isArray(item);
      if (
        value === null ||
        ['string', 'number'].includes(typeof value) ||
        (typeof expected === 'boolean' && typeof value === 'boolean') ||
        (Array.isArray(expected) && Array.isArray(value)) ||
        (object(expected) && object(value))
      ) {
        result[name] = value as T[keyof T];
      }
    }
    return result;
  } catch {
    return fallback;
  }
}

function write(key: string, value: unknown): void {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Storage full or unavailable: the choice simply is not remembered.
  }
}

export function loadSettings(): SettingsValues {
  const values = read(SETTINGS_KEY, DEFAULT_SETTINGS);
  for (const key of NUMBER_FIELDS) {
    const value = values[key] === null ? null : parseField(values[key]);
    values[key] = value === null ? DEFAULT_SETTINGS[key] : value;
  }
  if (!OPTIMIZERS.some((option) => option.value === values.optimizer)) {
    values.optimizer = DEFAULT_SETTINGS.optimizer;
  }
  if (typeof values.device !== 'string' || !values.device) {
    values.device = DEFAULT_SETTINGS.device;
  }
  const losses = new Set(['auto', ...Object.keys(LOSS_LABELS)]);
  if (!losses.has(values.loss)) {
    values.loss = DEFAULT_SETTINGS.loss;
  }
  if (!['balanced', 'custom', 'none'].includes(values.class_weights_mode)) {
    values.class_weights_mode = DEFAULT_SETTINGS.class_weights_mode;
  }
  if (typeof values.class_weights_text !== 'string') {
    values.class_weights_text = '';
  }
  if (!FEATURE_TRAININGS.some((option) => option.value === values.feature_training)) {
    values.feature_training = DEFAULT_SETTINGS.feature_training;
  }
  for (const key of BOOLEAN_FIELDS) {
    if (typeof values[key] !== 'boolean') {
      values[key] = DEFAULT_SETTINGS[key];
    }
  }
  if (!['random', 'grid'].includes(values.search_method)) {
    values.search_method = DEFAULT_SETTINGS.search_method;
  }
  if (!METRICS.some((metric) => metric.name === values.metric)) {
    values.metric = DEFAULT_SETTINGS.metric;
  }
  if (typeof values.classical_params !== 'object' || values.classical_params === null) {
    values.classical_params = {};
  }
  // Lists: only values the form offers, without duplicates.
  const only = <V>(list: unknown, allowed: readonly V[], fallback: V[]): V[] =>
    Array.isArray(list) ? allowed.filter((value) => list.includes(value)) : fallback;
  values.lr_values = only(values.lr_values, LR_PRESETS, DEFAULT_SETTINGS.lr_values);
  values.batch_sizes = only(values.batch_sizes, BATCH_SIZE_PRESETS, DEFAULT_SETTINGS.batch_sizes);
  values.weight_decays = only(
    values.weight_decays,
    WEIGHT_DECAY_PRESETS,
    DEFAULT_SETTINGS.weight_decays
  );
  values.optimizers = only(
    values.optimizers,
    OPTIMIZERS.map((option) => option.value),
    DEFAULT_SETTINGS.optimizers
  );
  values.losses = only(values.losses, [...losses] as LossName[], DEFAULT_SETTINGS.losses);
  values.focal_gammas = only(
    values.focal_gammas,
    FOCAL_GAMMA_PRESETS,
    DEFAULT_SETTINGS.focal_gammas
  );
  values.label_smoothings = only(
    values.label_smoothings,
    LABEL_SMOOTHING_PRESETS,
    DEFAULT_SETTINGS.label_smoothings
  );
  values.huber_deltas = only(
    values.huber_deltas,
    HUBER_DELTA_PRESETS,
    DEFAULT_SETTINGS.huber_deltas
  );
  return values;
}
export const saveSettings = (values: SettingsValues) => write(SETTINGS_KEY, values);
export function loadSource(): SourceChoice {
  const choice = read(SOURCE_KEY, DEFAULT_SOURCE);
  return ['random', 'builtin', 'upload'].includes(choice.tab) ? choice : DEFAULT_SOURCE;
}
export const saveSource = (choice: SourceChoice) => write(SOURCE_KEY, choice);
