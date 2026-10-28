import type {
  DatasetRef,
  Optimizer,
  TrainingConfig,
  TuningInput,
  TuningMethod,
  TuningSpace,
} from '@/api/types';

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
  /** Percent (5–50), sent as a fraction. */
  val_split: NumberField;
  /** Hold out a test set (built-in datasets: their official test images). */
  test_enabled: boolean;
  /** Percent (5–50) of the data for the test set, sent as a fraction. */
  test_split: NumberField;
  weight_decay: NumberField;
  /** Empty: use all samples. */
  max_samples: NumberField;
  /** Empty: a random seed. */
  seed: NumberField;

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
  'learning_rate',
  'val_split',
  'test_split',
  'weight_decay',
  'max_samples',
  'seed',
  'search_trials',
  'lr_min',
  'lr_max',
] as const;

const BOOLEAN_FIELDS = [
  'test_enabled',
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
  val_split: { min: 5, max: 50 },
  test_split: { min: 5, max: 50 },
  /** Validation and test split together, percent. */
  held_out: { max: 80 },
  max_samples: { min: 10, max: 1_000_000_000 },
  seed: { min: 0, max: 2 ** 31 - 1 },
  /** Same limits as backend/brickdnn/training/tuning.py. */
  search_trials: { min: 1, max: 100 },
} as const;

/** Built-in datasets only switch their official test images on; the share is not used. */
const BUILTIN_TEST_SPLIT = 0.2;

export const OPTIMIZERS: { value: Optimizer; label: string }[] = [
  { value: 'adam', label: 'Adam' },
  { value: 'adamw', label: 'AdamW' },
  { value: 'sgd', label: 'SGD' },
];

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

export function validateSettings(values: SettingsValues): SettingsErrors {
  const errors: SettingsErrors = {};
  const set = (key: keyof SettingsValues, message: string | null) => {
    if (message) {
      errors[key] = message;
    }
  };
  const tuned = tunedSettings(values);
  set('epochs', wholeNumber(values.epochs, LIMITS.epochs, true));
  if (!tuned.batch_size) {
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
  const split = (value: number) => value >= 5 && value <= 50;
  set('val_split', decimal(values.val_split, split, 'Between 5% and 50%', true));
  if (values.test_enabled) {
    set('test_split', decimal(values.test_split, split, 'Between 5% and 50%', true));
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
  set('max_samples', wholeNumber(values.max_samples, LIMITS.max_samples, false));
  set('seed', wholeNumber(values.seed, LIMITS.seed, false));

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
    }
    if (tuned.optimizer && values.optimizers.length === 0) {
      errors.optimizers = 'Choose at least one';
    }
    if (tuned.weight_decay && values.weight_decays.length === 0) {
      errors.weight_decays = 'Choose at least one';
    }
    const combos = combinations(values);
    if (!Object.values(tuned).some(Boolean)) {
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
 * The training settings for the API. Built-in datasets only switch their official test
 * images on (`builtin`). Settings varied by a search are left out; the search sets them.
 */
export function toConfig(
  values: SettingsValues,
  { builtin = false }: { builtin?: boolean } = {}
): Partial<TrainingConfig> {
  const number = (field: NumberField) => {
    const value = parseField(field);
    return value === '' || value === null ? null : value;
  };
  const tuned = tunedSettings(values);
  const split = number(values.val_split);
  const testSplit = number(values.test_split);
  let test = 0;
  if (values.test_enabled) {
    test = builtin ? BUILTIN_TEST_SPLIT : testSplit === null ? 0 : fraction(testSplit);
  }
  return {
    epochs: number(values.epochs) ?? undefined,
    batch_size: tuned.batch_size ? undefined : (number(values.batch_size) ?? undefined),
    learning_rate: tuned.learning_rate ? undefined : (number(values.learning_rate) ?? undefined),
    optimizer: tuned.optimizer ? undefined : values.optimizer,
    weight_decay: tuned.weight_decay ? undefined : (number(values.weight_decay) ?? 0),
    val_split: split === null ? undefined : fraction(split),
    test_split: test,
    device: values.device,
    max_samples: number(values.max_samples),
    seed: number(values.seed),
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
