import type { DatasetRef, Optimizer, TrainingConfig } from '@/api/types';

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
  weight_decay: NumberField;
  /** Empty: use all samples. */
  max_samples: NumberField;
  /** Empty: a random seed. */
  seed: NumberField;
}

const NUMBER_FIELDS = [
  'epochs',
  'batch_size',
  'learning_rate',
  'val_split',
  'weight_decay',
  'max_samples',
  'seed',
] as const;

export const DEFAULT_SETTINGS: SettingsValues = {
  epochs: 10,
  batch_size: 64,
  learning_rate: 0.001,
  optimizer: 'adam',
  device: 'auto',
  val_split: 20,
  weight_decay: 0,
  max_samples: '',
  seed: '',
};

/** Same limits as backend/brickdnn/training/config.py. */
export const LIMITS = {
  epochs: { min: 1, max: 1000 },
  batch_size: { min: 1, max: 8192 },
  learning_rate: { min: 0, max: 10 },
  weight_decay: { min: 0, max: 1 },
  val_split: { min: 5, max: 50 },
  max_samples: { min: 10, max: 1_000_000_000 },
  seed: { min: 0, max: 2 ** 31 - 1 },
} as const;

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

export function validateSettings(values: SettingsValues): SettingsErrors {
  const errors: SettingsErrors = {};
  const set = (key: keyof SettingsValues, message: string | null) => {
    if (message) {
      errors[key] = message;
    }
  };
  set('epochs', wholeNumber(values.epochs, LIMITS.epochs, true));
  set('batch_size', wholeNumber(values.batch_size, LIMITS.batch_size, true));
  set(
    'learning_rate',
    decimal(
      values.learning_rate,
      (lr) => lr > 0 && lr <= LIMITS.learning_rate.max,
      'Greater than 0, at most 10',
      true
    )
  );
  set(
    'weight_decay',
    decimal(values.weight_decay, (wd) => wd >= 0 && wd <= 1, 'Between 0 and 1', false)
  );
  set(
    'val_split',
    decimal(values.val_split, (split) => split >= 5 && split <= 50, 'Between 5% and 50%', true)
  );
  set('max_samples', wholeNumber(values.max_samples, LIMITS.max_samples, false));
  set('seed', wholeNumber(values.seed, LIMITS.seed, false));
  return errors;
}

export function toConfig(values: SettingsValues): Partial<TrainingConfig> {
  const number = (field: NumberField) => {
    const value = parseField(field);
    return value === '' || value === null ? null : value;
  };
  const split = number(values.val_split);
  return {
    epochs: number(values.epochs) ?? undefined,
    batch_size: number(values.batch_size) ?? undefined,
    learning_rate: number(values.learning_rate) ?? undefined,
    optimizer: values.optimizer,
    weight_decay: number(values.weight_decay) ?? 0,
    val_split: split === null ? undefined : Number((split / 100).toFixed(4)),
    device: values.device,
    max_samples: number(values.max_samples),
    seed: number(values.seed),
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
    // Only keep known keys with plain values.
    const result = { ...fallback };
    for (const name of Object.keys(fallback) as (keyof T)[]) {
      const value = data[name];
      if (value === null || ['string', 'number'].includes(typeof value)) {
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
  return values;
}
export const saveSettings = (values: SettingsValues) => write(SETTINGS_KEY, values);
export function loadSource(): SourceChoice {
  const choice = read(SOURCE_KEY, DEFAULT_SOURCE);
  return ['random', 'builtin', 'upload'].includes(choice.tab) ? choice : DEFAULT_SOURCE;
}
export const saveSource = (choice: SourceChoice) => write(SOURCE_KEY, choice);
