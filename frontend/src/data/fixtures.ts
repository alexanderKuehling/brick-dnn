/**
 * Test and development data for the Data tab (only imported by tests). A small customer
 * churn dataset with the usual problems: missing values, a skewed column with outliers,
 * categories, a date, free text and an ID column. Shared by all Data tab tests, so keep it
 * consistent with `frontend/src/api/types.ts`.
 */
import type {
  ColumnProfile,
  DataPrep,
  DataProfile,
  DataSample,
  DataSource,
  DataStatus,
  PreviewResult,
  Readiness,
  StepCatalog,
  StepReport,
} from '@/api/types';

export const DATA_STATUS: DataStatus = {
  enabled: true,
  available: true,
  reason: null,
  max_upload_mb: 500,
  max_rows: 2_000_000,
  max_columns: 500,
  default_sample_rows: 5000,
  max_sample_rows: 20000,
  login_required: true,
};

export const CATALOG: StepCatalog = {
  groups: [
    { id: 'combine', label: 'Combine data' },
    { id: 'columns', label: 'Columns' },
    { id: 'rows', label: 'Rows' },
    { id: 'missing', label: 'Missing values' },
    { id: 'outliers', label: 'Outliers & clipping' },
    { id: 'transform', label: 'Transform numbers' },
    { id: 'encode', label: 'Encode categories' },
    { id: 'datetime', label: 'Dates & times' },
    { id: 'target', label: 'Target & balance' },
    { id: 'selection', label: 'Feature selection' },
  ],
  steps: [
    {
      type: 'join',
      label: 'Join another table',
      group: 'combine',
      description: 'Add the columns of another imported table by matching key columns.',
      fitted: false,
      combine: true,
      training_only: false,
      params: [
        {
          name: 'source',
          kind: 'source',
          label: 'Table',
          default: null,
          help: null,
          optional: false,
        },
        {
          name: 'how',
          kind: 'select',
          label: 'Keep',
          default: 'left',
          help: 'Which rows to keep when a key has no match.',
          optional: false,
          options: [
            { value: 'left', label: 'All rows of this table (left join)' },
            { value: 'inner', label: 'Only matching rows (inner join)' },
            { value: 'outer', label: 'All rows of both tables (outer join)' },
            { value: 'right', label: 'All rows of the other table (right join)' },
          ],
        },
        {
          name: 'left_on',
          kind: 'columns',
          label: 'Key columns here',
          default: [],
          help: null,
          optional: false,
        },
        {
          name: 'right_on',
          kind: 'source_columns',
          label: 'Key columns there',
          default: [],
          help: 'Same number and order as the key columns here.',
          optional: false,
          source_param: 'source',
        },
        {
          name: 'suffix',
          kind: 'string',
          label: 'Suffix for duplicate names',
          default: '_right',
          help: null,
          optional: false,
        },
      ],
    },
    {
      type: 'drop_columns',
      label: 'Drop columns',
      group: 'columns',
      description: 'Remove columns that should not be used.',
      fitted: false,
      combine: false,
      training_only: false,
      params: [
        {
          name: 'columns',
          kind: 'columns',
          label: 'Columns',
          default: [],
          help: null,
          optional: false,
        },
      ],
    },
    {
      type: 'filter_rows',
      label: 'Filter rows',
      group: 'rows',
      description: 'Keep only the rows that meet a condition.',
      fitted: false,
      combine: false,
      training_only: false,
      params: [
        {
          name: 'column',
          kind: 'column',
          label: 'Column',
          default: null,
          help: null,
          optional: false,
        },
        {
          name: 'operator',
          kind: 'select',
          label: 'Keep rows where the value',
          default: '>=',
          help: null,
          optional: false,
          options: [
            { value: '==', label: 'equals' },
            { value: '!=', label: 'does not equal' },
            { value: '>', label: 'is greater than' },
            { value: '>=', label: 'is at least' },
            { value: '<', label: 'is less than' },
            { value: '<=', label: 'is at most' },
            { value: 'between', label: 'is between' },
            { value: 'in', label: 'is one of' },
            { value: 'not_missing', label: 'is not missing' },
          ],
        },
        {
          name: 'value',
          kind: 'value',
          label: 'Value',
          default: null,
          help: null,
          optional: true,
          visible_if: { param: 'operator', values: ['==', '!=', '>', '>=', '<', '<=', 'between'] },
        },
        {
          name: 'value2',
          kind: 'value',
          label: 'and',
          default: null,
          help: null,
          optional: true,
          visible_if: { param: 'operator', values: ['between'] },
        },
        {
          name: 'values',
          kind: 'list',
          label: 'Values',
          default: [],
          help: null,
          optional: true,
          visible_if: { param: 'operator', values: ['in'] },
        },
      ],
    },
    {
      type: 'impute',
      label: 'Fill missing values',
      group: 'missing',
      description: 'Replace missing values, e.g. with the median of the training rows.',
      fitted: true,
      combine: false,
      training_only: false,
      params: [
        {
          name: 'columns',
          kind: 'columns',
          label: 'Columns',
          default: [],
          help: null,
          optional: true,
          empty_means: 'All columns with missing values',
        },
        {
          name: 'strategy',
          kind: 'select',
          label: 'Fill with',
          default: 'median',
          help: null,
          optional: false,
          options: [
            { value: 'mean', label: 'Mean' },
            { value: 'median', label: 'Median' },
            { value: 'most_frequent', label: 'Most frequent value' },
            { value: 'constant', label: 'A fixed value' },
            { value: 'knn', label: 'Nearest neighbours (KNN)' },
          ],
        },
        {
          name: 'fill_value',
          kind: 'value',
          label: 'Value',
          default: null,
          help: null,
          optional: true,
          visible_if: { param: 'strategy', values: ['constant'] },
        },
        {
          name: 'k',
          kind: 'int',
          label: 'Neighbours',
          default: 5,
          help: null,
          optional: false,
          min: 1,
          max: 50,
          visible_if: { param: 'strategy', values: ['knn'] },
        },
        {
          name: 'add_indicator',
          kind: 'bool',
          label: 'Add a column that marks filled values',
          default: false,
          help: null,
          optional: false,
        },
      ],
    },
    {
      type: 'clip',
      label: 'Clip values',
      group: 'outliers',
      description: 'Limit values to a range, e.g. the 1st to 99th percentile of the training rows.',
      fitted: true,
      combine: false,
      training_only: false,
      params: [
        {
          name: 'columns',
          kind: 'columns',
          label: 'Columns',
          default: [],
          help: null,
          optional: true,
          column_kinds: ['numeric'],
          empty_means: 'All numeric columns',
        },
        {
          name: 'mode',
          kind: 'select',
          label: 'Bounds',
          default: 'quantiles',
          help: null,
          optional: false,
          options: [
            { value: 'quantiles', label: 'Quantiles' },
            { value: 'values', label: 'Fixed values' },
            { value: 'std', label: 'Mean ± k standard deviations' },
          ],
        },
        {
          name: 'lower_q',
          kind: 'number',
          label: 'Lower quantile',
          default: 0.01,
          help: null,
          optional: false,
          min: 0,
          max: 0.5,
          step: 0.01,
          visible_if: { param: 'mode', values: ['quantiles'] },
        },
        {
          name: 'upper_q',
          kind: 'number',
          label: 'Upper quantile',
          default: 0.99,
          help: null,
          optional: false,
          min: 0.5,
          max: 1,
          step: 0.01,
          visible_if: { param: 'mode', values: ['quantiles'] },
        },
      ],
    },
    {
      type: 'encode',
      label: 'Encode categories',
      group: 'encode',
      description: 'Turn categories into numbers, e.g. one column per category (one-hot).',
      fitted: true,
      combine: false,
      training_only: false,
      params: [
        {
          name: 'columns',
          kind: 'columns',
          label: 'Columns',
          default: [],
          help: null,
          optional: true,
          column_kinds: ['categorical', 'boolean'],
          empty_means: 'All categorical columns',
        },
        {
          name: 'method',
          kind: 'select',
          label: 'Method',
          default: 'onehot',
          help: null,
          optional: false,
          options: [
            { value: 'onehot', label: 'One-hot' },
            { value: 'ordinal', label: 'Ordinal (category index)' },
            { value: 'frequency', label: 'Frequency' },
            { value: 'target', label: 'Target mean' },
          ],
        },
      ],
    },
    {
      type: 'rename_columns',
      label: 'Rename columns',
      group: 'columns',
      description: 'Give columns new names.',
      fitted: false,
      combine: false,
      training_only: false,
      params: [
        {
          name: 'mapping',
          kind: 'mapping',
          label: 'New names',
          default: {},
          help: null,
          optional: false,
        },
      ],
    },
    {
      type: 'datetime_features',
      label: 'Date parts',
      group: 'datetime',
      description: 'Split dates into year, month, weekday … columns.',
      fitted: false,
      combine: false,
      training_only: false,
      params: [
        {
          name: 'columns',
          kind: 'columns',
          label: 'Columns',
          default: [],
          help: null,
          optional: true,
          column_kinds: ['datetime'],
          empty_means: 'All date columns',
        },
        {
          name: 'parts',
          kind: 'multiselect',
          label: 'Parts',
          default: ['year', 'month', 'weekday'],
          help: null,
          optional: false,
          options: [
            { value: 'year', label: 'Year' },
            { value: 'month', label: 'Month' },
            { value: 'day', label: 'Day' },
            { value: 'weekday', label: 'Weekday' },
            { value: 'hour', label: 'Hour' },
          ],
        },
        {
          name: 'drop_original',
          kind: 'bool',
          label: 'Drop the original column',
          default: true,
          help: null,
          optional: false,
        },
      ],
    },
  ],
};

// ---- the data -------------------------------------------------------------

const CITIES = ['Berlin', 'Hamburg', 'Munich', 'Cologne', 'Leipzig'];
const PLANS = ['basic', 'plus', 'pro'];

/** Small seeded PRNG (mulberry32), so the fixture is the same in every test run. */
function random(seed: number) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export const SAMPLE_ROWS = 60;

function makeSample(): DataSample {
  const next = random(7);
  const id: number[] = [];
  const age: (number | null)[] = [];
  const income: (number | null)[] = [];
  const city: (string | null)[] = [];
  const plan: string[] = [];
  const signup: number[] = [];
  const member: number[] = [];
  const notes: (string | null)[] = [];
  const churn: string[] = [];
  const split: ('train' | 'val' | 'test')[] = [];
  for (let i = 0; i < SAMPLE_ROWS; i += 1) {
    id.push(1000 + i);
    age.push(next() < 0.1 ? null : Math.round(18 + next() * 60));
    const base = Math.exp(10 + next() * 1.2);
    income.push(next() < 0.05 ? null : Math.round(i === 7 ? 950_000 : base));
    city.push(next() < 0.08 ? null : CITIES[Math.floor(next() * CITIES.length)]);
    plan.push(PLANS[Math.floor(next() * PLANS.length)]);
    signup.push(Date.UTC(2021, 0, 1) + Math.floor(next() * 3 * 365) * 86_400_000);
    member.push(next() < 0.4 ? 1 : 0);
    notes.push(next() < 0.5 ? null : `Customer note number ${i} with some free text.`);
    churn.push(next() < 0.27 ? 'yes' : 'no');
    split.push(i % 10 < 7 ? 'train' : i % 10 < 9 ? 'val' : 'test');
  }
  return {
    n_total: 1200,
    rows: SAMPLE_ROWS,
    seed: 0,
    index: id.map((_, i) => i * 20),
    split,
    columns: [
      { name: 'customer_id', kind: 'numeric', values: id },
      { name: 'age', kind: 'numeric', values: age },
      { name: 'income', kind: 'numeric', values: income },
      { name: 'city', kind: 'categorical', values: city },
      { name: 'plan', kind: 'categorical', values: plan },
      { name: 'signup', kind: 'datetime', values: signup },
      { name: 'is_member', kind: 'boolean', values: member },
      { name: 'notes', kind: 'text', values: notes },
      { name: 'churn', kind: 'categorical', values: churn },
    ],
  };
}

export const SAMPLE: DataSample = makeSample();

const numericColumn = (
  name: string,
  overrides: Partial<ColumnProfile> & Pick<ColumnProfile, 'stats' | 'histogram'>
): ColumnProfile => ({
  name,
  kind: 'numeric',
  integer: true,
  count: 1200,
  missing: 0,
  unique: 1200,
  top: null,
  other_count: null,
  mean_length: null,
  target_association: null,
  ...overrides,
});

export const PROFILE: DataProfile = {
  n_rows: 1200,
  n_columns: 9,
  missing_cells: 120 + 60 + 96 + 600,
  duplicate_rows: 3,
  memory_bytes: 312_000,
  split_counts: [840, 180, 180],
  analysis_rows: 1200,
  columns: [
    numericColumn('customer_id', {
      stats: {
        mean: 1599.5,
        std: 346.6,
        min: 1000,
        max: 2199,
        q01: 1012,
        q05: 1060,
        q25: 1300,
        median: 1599.5,
        q75: 1899,
        q95: 2139,
        q99: 2187,
        skew: 0,
        kurtosis: -1.2,
        zeros: 0,
        negative: 0,
        iqr_outliers: 0,
        z_outliers: 0,
      },
      histogram: { edges: [1000, 1240, 1480, 1720, 1960, 2200], counts: [240, 240, 240, 240, 240] },
      target_association: { method: 'eta', value: 0.02 },
    }),
    numericColumn('age', {
      count: 1080,
      missing: 120,
      unique: 61,
      stats: {
        mean: 47.6,
        std: 17.2,
        min: 18,
        max: 78,
        q01: 18,
        q05: 21,
        q25: 33,
        median: 48,
        q75: 62,
        q95: 75,
        q99: 78,
        skew: 0.02,
        kurtosis: -1.19,
        zeros: 0,
        negative: 0,
        iqr_outliers: 0,
        z_outliers: 0,
      },
      histogram: {
        edges: [18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78],
        counts: [101, 110, 105, 112, 108, 109, 104, 111, 106, 114],
      },
      target_association: { method: 'eta', value: 0.31 },
    }),
    numericColumn('income', {
      count: 1140,
      missing: 60,
      unique: 1098,
      stats: {
        mean: 41_200,
        std: 39_800,
        min: 22_100,
        max: 950_000,
        q01: 22_600,
        q05: 23_900,
        q25: 29_800,
        median: 38_400,
        q75: 49_700,
        q95: 63_100,
        q99: 68_900,
        skew: 18.4,
        kurtosis: 402.1,
        zeros: 0,
        negative: 0,
        iqr_outliers: 9,
        z_outliers: 2,
      },
      histogram: {
        edges: [
          0, 95_000, 190_000, 285_000, 380_000, 475_000, 570_000, 665_000, 760_000, 855_000,
          950_000,
        ],
        counts: [1137, 1, 0, 0, 0, 0, 0, 0, 0, 2],
      },
      target_association: { method: 'eta', value: 0.18 },
    }),
    {
      name: 'city',
      kind: 'categorical',
      integer: false,
      count: 1104,
      missing: 96,
      unique: 5,
      stats: null,
      histogram: null,
      top: [
        { value: 'Berlin', count: 260 },
        { value: 'Hamburg', count: 231 },
        { value: 'Munich', count: 228 },
        { value: 'Cologne', count: 199 },
        { value: 'Leipzig', count: 186 },
      ],
      other_count: 0,
      mean_length: 6.6,
      target_association: { method: 'cramers_v', value: 0.07 },
    },
    {
      name: 'plan',
      kind: 'categorical',
      integer: false,
      count: 1200,
      missing: 0,
      unique: 3,
      stats: null,
      histogram: null,
      top: [
        { value: 'basic', count: 520 },
        { value: 'plus', count: 410 },
        { value: 'pro', count: 270 },
      ],
      other_count: 0,
      mean_length: 4.0,
      target_association: { method: 'cramers_v', value: 0.24 },
    },
    {
      name: 'signup',
      kind: 'datetime',
      integer: false,
      count: 1200,
      missing: 0,
      unique: 870,
      stats: {
        mean: Date.UTC(2022, 6, 1),
        std: 2.7e10,
        min: Date.UTC(2021, 0, 1),
        max: Date.UTC(2023, 11, 31),
        q01: Date.UTC(2021, 0, 12),
        q05: Date.UTC(2021, 2, 1),
        q25: Date.UTC(2021, 9, 1),
        median: Date.UTC(2022, 6, 1),
        q75: Date.UTC(2023, 3, 1),
        q95: Date.UTC(2023, 10, 1),
        q99: Date.UTC(2023, 11, 20),
        skew: 0,
        kurtosis: -1.2,
        zeros: 0,
        negative: 0,
        iqr_outliers: 0,
        z_outliers: 0,
      },
      histogram: {
        edges: [
          Date.UTC(2021, 0, 1),
          Date.UTC(2022, 0, 1),
          Date.UTC(2023, 0, 1),
          Date.UTC(2024, 0, 1),
        ],
        counts: [395, 402, 403],
      },
      top: null,
      other_count: null,
      mean_length: null,
      target_association: null,
    },
    {
      name: 'is_member',
      kind: 'boolean',
      integer: true,
      count: 1200,
      missing: 0,
      unique: 2,
      stats: {
        mean: 0.4,
        std: 0.49,
        min: 0,
        max: 1,
        q01: 0,
        q05: 0,
        q25: 0,
        median: 0,
        q75: 1,
        q95: 1,
        q99: 1,
        skew: 0.41,
        kurtosis: -1.83,
        zeros: 720,
        negative: 0,
        iqr_outliers: 0,
        z_outliers: 0,
      },
      histogram: null,
      top: [
        { value: '0', count: 720 },
        { value: '1', count: 480 },
      ],
      other_count: 0,
      mean_length: null,
      target_association: { method: 'cramers_v', value: 0.12 },
    },
    {
      name: 'notes',
      kind: 'text',
      integer: false,
      count: 600,
      missing: 600,
      unique: 600,
      stats: null,
      histogram: null,
      top: [{ value: 'Customer note number 1 with some free text.', count: 1 }],
      other_count: 599,
      mean_length: 44.2,
      target_association: null,
    },
    {
      name: 'churn',
      kind: 'categorical',
      integer: false,
      count: 1200,
      missing: 0,
      unique: 2,
      stats: null,
      histogram: null,
      top: [
        { value: 'no', count: 876 },
        { value: 'yes', count: 324 },
      ],
      other_count: 0,
      mean_length: 2.3,
      target_association: null,
    },
  ],
  correlations: [
    {
      method: 'pearson',
      columns: ['customer_id', 'age', 'income', 'is_member'],
      matrix: [
        [1, 0.01, -0.02, 0.0],
        [0.01, 1, 0.42, 0.11],
        [-0.02, 0.42, 1, 0.05],
        [0.0, 0.11, 0.05, 1],
      ],
    },
    {
      method: 'spearman',
      columns: ['customer_id', 'age', 'income', 'is_member'],
      matrix: [
        [1, 0.0, -0.01, 0.0],
        [0.0, 1, 0.51, 0.1],
        [-0.01, 0.51, 1, 0.06],
        [0.0, 0.1, 0.06, 1],
      ],
    },
  ],
  target: {
    column: 'churn',
    task: 'classification',
    classes: [
      { value: 'no', count: 876 },
      { value: 'yes', count: 324 },
    ],
    imbalance_ratio: 2.7,
    histogram: null,
    stats: null,
  },
  warnings: [
    {
      code: 'id_like',
      severity: 'warning',
      column: 'customer_id',
      message: 'customer_id has a different value in every row; it looks like an ID.',
      fix: {
        type: 'drop_columns',
        params: { columns: ['customer_id'] },
        label: 'Drop customer_id',
      },
    },
    {
      code: 'skewed',
      severity: 'warning',
      column: 'income',
      message: 'income is strongly skewed (skewness 18.4) and has 9 outliers.',
      fix: {
        type: 'clip',
        params: { columns: ['income'], mode: 'quantiles', lower_q: 0.01, upper_q: 0.99 },
        label: 'Clip income to the 1st–99th percentile',
      },
    },
    {
      code: 'high_missing',
      severity: 'warning',
      column: 'notes',
      message: 'notes is missing in 50 % of the rows.',
      fix: { type: 'drop_columns', params: { columns: ['notes'] }, label: 'Drop notes' },
    },
    {
      code: 'duplicates',
      severity: 'info',
      column: null,
      message: '3 rows are exact duplicates of other rows.',
      fix: {
        type: 'drop_duplicates',
        params: { columns: [], keep: 'first' },
        label: 'Remove duplicates',
      },
    },
  ],
};

export const READINESS: Readiness = {
  ready: false,
  task: 'classification',
  target: 'churn',
  rows: 1200,
  num_features: 8,
  num_classes: 2,
  class_names: ['no', 'yes'],
  split_counts: [840, 180, 180],
  issues: [
    {
      severity: 'error',
      column: 'age',
      message: '2 feature columns have missing values (age, income).',
      fix: {
        type: 'impute',
        params: { columns: ['age', 'income'], strategy: 'median' },
        label: 'Impute 2 columns (median)',
      },
    },
    {
      severity: 'error',
      column: 'city',
      message: '2 feature columns are categorical (city, plan). Networks need numbers.',
      fix: {
        type: 'encode',
        params: { columns: ['city', 'plan'], method: 'onehot' },
        label: 'One-hot encode 2 columns',
      },
    },
    {
      severity: 'error',
      column: 'notes',
      message: 'notes is free text and cannot be used as a feature.',
      fix: { type: 'drop_columns', params: { columns: ['notes'] }, label: 'Drop notes' },
    },
    {
      severity: 'error',
      column: 'signup',
      message: 'signup is a date. Split it into numbers first.',
      fix: {
        type: 'datetime_features',
        params: { columns: ['signup'], parts: ['year', 'month', 'weekday'], drop_original: true },
        label: 'Split signup into year, month, weekday',
      },
    },
    {
      severity: 'warning',
      column: null,
      message: 'The classes are imbalanced (2.7 : 1).',
      fix: null,
    },
  ],
};

const COLUMNS = SAMPLE.columns.map(({ name, kind }) => ({ name, kind }));

export const STEP_REPORTS: StepReport[] = [
  {
    id: 's1',
    type: 'impute',
    status: 'ok',
    summary: 'Filled 180 missing values in 2 columns with the median.',
    error: null,
    warnings: [],
    details: [
      { column: 'age', text: 'median 48' },
      { column: 'income', text: 'median 38,400' },
    ],
    rows_before: 1200,
    rows_after: 1200,
    columns: COLUMNS,
    added: [],
    removed: [],
    fitted: { values: { age: 48, income: 38400 } },
    duration_ms: 3,
  },
  {
    id: 's2',
    type: 'clip',
    status: 'error',
    summary: '',
    error: "Columns: 'city' is a categorical column; numeric columns needed.",
    warnings: [],
    details: [],
    rows_before: 1200,
    rows_after: 1200,
    columns: COLUMNS,
    added: [],
    removed: [],
    fitted: null,
    duration_ms: 0,
  },
  {
    id: 's3',
    type: 'encode',
    status: 'skipped',
    summary: '',
    error: null,
    warnings: [],
    details: [],
    rows_before: 1200,
    rows_after: 1200,
    columns: COLUMNS,
    added: [],
    removed: [],
    fitted: null,
    duration_ms: 0,
  },
];

export const SOURCE: DataSource = {
  id: 1,
  name: 'customers',
  filename: 'customers.csv',
  size_bytes: 98_304,
  n_rows: 1200,
  n_columns: 9,
  columns: PROFILE.columns.map(({ name, kind, missing }) => ({ name, kind, missing })),
  import_options: {
    delimiter: 'auto',
    decimal: 'auto',
    header: 'auto',
    encoding: 'auto',
    missing_values: null,
  },
  report: {
    delimiter: ';',
    decimal: ',',
    header: true,
    encoding: 'utf-8',
    rows: 1200,
    columns: 9,
    warnings: [],
  },
  created_at: '2026-11-07T10:00:00Z',
};

export const OTHER_SOURCE: DataSource = {
  ...SOURCE,
  id: 2,
  name: 'regions',
  filename: 'regions.csv',
  size_bytes: 512,
  n_rows: 5,
  n_columns: 3,
  columns: [
    { name: 'city', kind: 'categorical', missing: 0 },
    { name: 'region', kind: 'categorical', missing: 0 },
    { name: 'population', kind: 'numeric', missing: 0 },
  ],
};

export const PREP: DataPrep = {
  id: 10,
  name: 'Churn model data',
  source: 1,
  source_name: 'customers',
  steps: [
    {
      id: 's1',
      type: 'impute',
      enabled: true,
      params: { columns: [], strategy: 'median', fill_value: null, k: 5, add_indicator: false },
    },
    {
      id: 's2',
      type: 'clip',
      enabled: true,
      params: { columns: ['city'], mode: 'quantiles', lower_q: 0.01, upper_q: 0.99 },
    },
    { id: 's3', type: 'encode', enabled: true, params: { columns: [], method: 'onehot' } },
  ],
  settings: {
    target: 'churn',
    task: 'auto',
    split: { val: 0.15, test: 0.15, seed: 42, stratify: true },
  },
  datasets: [],
  created_at: '2026-11-07T10:00:00Z',
  updated_at: '2026-11-07T10:05:00Z',
};

export const PREVIEW: PreviewResult = {
  steps: STEP_REPORTS,
  task: 'classification',
  profile: PROFILE,
  before: PROFILE,
  sample: SAMPLE,
  readiness: READINESS,
  warnings: [],
  duration_ms: 42,
};
