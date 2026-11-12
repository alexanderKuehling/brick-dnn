import type {
  BuiltinDataset,
  Dataset,
  DatasetRef,
  Optimizer,
  SegmentationLoss,
  TrainingConfig,
  TrainingStatus,
  TuningInput,
  TuningMethod,
  TuningSpace,
} from '@/api/types';
import { isOwnDataset } from './access';

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
  /** Segmentation: the cross-entropy per pixel, or that plus a Dice term. */
  loss: SegmentationLoss;

  /** Hyperparameter search instead of a single training. */
  search: boolean;
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
] as const;

const BOOLEAN_FIELDS = [
  'test_enabled',
  'restore_best',
  'search',
  'tune_learning_rate',
  'tune_batch_size',
  'tune_optimizer',
  'tune_weight_decay',
] as const;

/** Values offered by the search form. */
export const LR_PRESETS = [0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001];
export const BATCH_SIZE_PRESETS = [8, 16, 32, 64, 128, 256, 512, 1024];
export const WEIGHT_DECAY_PRESETS = [0, 0.00001, 0.0001, 0.001, 0.01, 0.1];

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
  loss: 'cross_entropy',
  search: false,
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
  /** Same limits as backend/brickdnn/training/tuning.py. */
  search_trials: { min: 1, max: 100 },
} as const;

export const OPTIMIZERS: { value: Optimizer; label: string }[] = [
  { value: 'adam', label: 'Adam' },
  { value: 'adamw', label: 'AdamW' },
  { value: 'sgd', label: 'SGD' },
];

export const LOSSES: { value: SegmentationLoss; label: string }[] = [
  { value: 'cross_entropy', label: 'Cross-entropy' },
  { value: 'cross_entropy_dice', label: 'Cross-entropy + Dice' },
];

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

export type SettingsErrors = Partial<Record<keyof SettingsValues, string>>;

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
    return `Between ${min.toLocaleString('en-US')} and ${max.toLocaleString('en-US')}`;
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

/** Which settings the search varies (none when the search is off). */
export function tunedSettings(values: SettingsValues) {
  const on = values.search;
  return {
    learning_rate: on && values.tune_learning_rate,
    batch_size: on && values.tune_batch_size,
    optimizer: on && values.tune_optimizer,
    weight_decay: on && values.tune_weight_decay,
  };
}

/** Number of combinations of the listed values, or null if a range makes it unlimited. */
function combinations(values: SettingsValues): number | null {
  const tuned = tunedSettings(values);
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
  return count;
}

/**
 * How many trials the search will run: every combination for a grid, otherwise the chosen
 * number (but each combination of lists at most once, like the server). Null if unknown.
 */
export function plannedTrials(values: SettingsValues): number | null {
  const combos = combinations(values);
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
 * Classical models (scikit-learn) are fitted once: only the splits, Max. samples and the
 * seed apply. `maxRows` is the most training rows the model is fitted on on this server.
 */
export interface ClassicalScope {
  maxRows: number | null;
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

function validateClassical(values: SettingsValues, { maxRows }: ClassicalScope): SettingsErrors {
  const errors: SettingsErrors = {};
  validateSplits(values, errors);
  const limit = { min: LIMITS.max_samples.min, max: maxRows ?? LIMITS.max_samples.max };
  const samples = wholeNumber(values.max_samples, limit, false);
  if (samples) {
    const value = parseField(values.max_samples);
    errors.max_samples =
      maxRows !== null && typeof value === 'number' && value > maxRows
        ? `At most ${maxRows.toLocaleString('en-US')} on this server`
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
    segmentation = null,
    validationRows = null,
  }: {
    classical?: ClassicalScope | null;
    segmentation?: SegmentationScope | null;
    validationRows?: number | null;
  } = {}
): SettingsErrors {
  if (classical) {
    return validateClassical(values, classical);
  }
  const errors: SettingsErrors = {};
  const set = (key: keyof SettingsValues, message: string | null) => {
    if (message) {
      errors[key] = message;
    }
  };
  const tuned = tunedSettings(values);
  set('epochs', wholeNumber(values.epochs, LIMITS.epochs, true));
  const maxBatch = segmentation?.maxBatch ?? null;
  const tooLarge = (size: number) =>
    maxBatch !== null && size > maxBatch
      ? maxBatch >= 1
        ? `At most ${maxBatch.toLocaleString('en-US')}: larger batches do not fit into memory`
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

  if (values.search) {
    const grid = values.search_method === 'grid';
    if (!grid) {
      set('search_trials', wholeNumber(values.search_trials, LIMITS.search_trials, true));
    }
    if (tuned.learning_rate && grid && values.lr_values.length === 0) {
      errors.lr_values = 'Choose at least one';
    }
    if (tuned.learning_rate && !grid) {
      const range = (field: NumberField) =>
        decimal(field, learningRate, 'Greater than 0, at most 10', true);
      set('lr_min', range(values.lr_min));
      set('lr_max', range(values.lr_max));
      const lo = parseField(values.lr_min);
      const hi = parseField(values.lr_max);
      if (!errors.lr_min && !errors.lr_max && typeof lo === 'number' && typeof hi === 'number') {
        if (lo > hi) {
          errors.lr_min = 'At most the maximum';
        }
      }
    }
    if (tuned.batch_size && values.batch_sizes.length === 0) {
      errors.batch_sizes = 'Choose at least one';
    } else if (tuned.batch_size && segmentation) {
      set('batch_sizes', tooLarge(Math.max(...values.batch_sizes)));
    }
    if (tuned.optimizer && values.optimizers.length === 0) {
      errors.optimizers = 'Choose at least one';
    }
    if (tuned.weight_decay && values.weight_decays.length === 0) {
      errors.weight_decays = 'Choose at least one';
    }
    const combos = combinations(values);
    if (!validates) {
      errors.search = prepared ? PREPARED_SEARCH : NO_VALIDATION_SEARCH;
    } else if (!Object.values(tuned).some(Boolean)) {
      errors.search = 'Choose at least one setting to search.';
    } else if (grid && combos !== null && combos > LIMITS.search_trials.max) {
      errors.search = `The grid has ${combos} combinations, at most ${LIMITS.search_trials.max} are possible. Choose fewer values.`;
    }
  }
  return errors;
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
    segmentation = false,
  }: { builtin?: boolean; classical?: boolean; segmentation?: boolean } = {}
): Partial<TrainingConfig> {
  const number = (field: NumberField) => {
    const value = parseField(field);
    return value === '' || value === null ? null : value;
  };
  const tuned = tunedSettings(values);
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
    ...(segmentation ? { loss: values.loss } : {}),
  };
}

const ascending = (a: number, b: number) => a - b;

/** The search settings for the API, or null for a single training. */
export function toTuning(values: SettingsValues): TuningInput | null {
  if (!values.search) {
    return null;
  }
  const tuned = tunedSettings(values);
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
  const trials = parseField(values.search_trials);
  return {
    method: values.search_method,
    ...(!grid && typeof trials === 'number' ? { trials } : {}),
    space,
  };
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
      if (
        value === null ||
        ['string', 'number'].includes(typeof value) ||
        (typeof expected === 'boolean' && typeof value === 'boolean') ||
        (Array.isArray(expected) && Array.isArray(value))
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
  if (!LOSSES.some((option) => option.value === values.loss)) {
    values.loss = DEFAULT_SETTINGS.loss;
  }
  for (const key of BOOLEAN_FIELDS) {
    if (typeof values[key] !== 'boolean') {
      values[key] = DEFAULT_SETTINGS[key];
    }
  }
  if (!['random', 'grid'].includes(values.search_method)) {
    values.search_method = DEFAULT_SETTINGS.search_method;
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
  return values;
}
export const saveSettings = (values: SettingsValues) => write(SETTINGS_KEY, values);
export function loadSource(): SourceChoice {
  const choice = read(SOURCE_KEY, DEFAULT_SOURCE);
  return ['random', 'builtin', 'upload'].includes(choice.tab) ? choice : DEFAULT_SOURCE;
}
export const saveSource = (choice: SourceChoice) => write(SOURCE_KEY, choice);
