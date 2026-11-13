import { IconChevronDown, IconChevronRight } from '@tabler/icons-react';
import {
  Box,
  Button,
  Checkbox,
  Collapse,
  Group,
  Input,
  MultiSelect,
  NumberInput,
  Paper,
  SegmentedControl,
  Select,
  SimpleGrid,
  Stack,
  Switch,
  Text,
  TextInput,
} from '@mantine/core';
import { useDisclosure } from '@mantine/hooks';
import type {
  FeatureTraining,
  Optimizer,
  SegmentationLoss,
  TrainingDevice,
  TuningMethod,
  TuningMetric,
} from '@/api';
import { formatSetting, formatSplit, splitPercents } from './format';
import {
  CLASSICAL_TUNABLES,
  FOLDS,
  metricDirection,
  METRICS,
  type MetricInfo,
  type ParamState,
  type Tunable,
} from './search';
import {
  BATCH_SIZE_PRESETS,
  classicalParams,
  FEATURE_TRAININGS,
  hasValidation,
  LIMITS,
  LOSSES,
  LR_PRESETS,
  OPTIMIZERS,
  parseField,
  plannedClassicalTrials,
  plannedTrials,
  tunedSettings,
  WEIGHT_DECAY_PRESETS,
  type SegmentationScope,
  type SettingsErrors,
  type SettingsValues,
} from './settings';

interface SettingsFormProps {
  values: SettingsValues;
  onChange: (values: SettingsValues) => void;
  errors: SettingsErrors;
  name: string;
  namePlaceholder: string;
  onNameChange: (name: string) => void;
  devices: TrainingDevice[];
  defaultDevice: string | null;
  /**
   * Built-in datasets come with official test images. They are always the test set;
   * validation data comes from the training images.
   */
  builtin: boolean;
  /**
   * Rows for training, validation and test of a dataset made in the Data tab, or the images
   * in the train/, val/ and test/ folders of a segmentation dataset. Training uses this split,
   * so the share fields are shown but cannot be changed.
   */
  predefinedSplit?: number[] | null;
  /**
   * The data has a class for every pixel: the batch size is the one for segmentation (with
   * the largest that fits into memory, if known) and the loss can be chosen.
   */
  segmentation?: SegmentationScope | null;
  /**
   * A classical model (scikit-learn) is fitted once: only the splits, Max. samples and the
   * seed apply. `maxRows`: the most training rows it is fitted on on this server.
   */
  classical?: ClassicalSettings | null;
  /**
   * A hybrid network: the network settings apply to its neural layers, which learn the
   * features; the search varies the parameters of its classical model.
   */
  hybrid?: HybridSettings | null;
  /**
   * The objective metrics a search can rank by on this data (from the training check; null
   * while it is not known: all are offered and the server checks).
   */
  metrics?: MetricInfo[] | null;
}

export interface ClassicalSettings {
  /** E.g. "Gaussian process regression". */
  model: string;
  maxRows: number | null;
  /** Gaussian processes optimise their kernel hyperparameters during the fit. */
  optimisesKernel: boolean;
  /**
   * The type of the model brick when a search over its parameters is possible (null: the
   * server does not offer it).
   */
  type?: string | null;
}

export interface HybridSettings {
  /** The classical model, e.g. "Gradient boosting classification". */
  model: string;
  /** The length of the feature vector the neural layers learn (null if unknown). */
  features: number | null;
  /** The most training rows the classical model is fitted on on this server. */
  maxRows: number | null;
  /** The type of the classical model brick when its search is possible (null: not). */
  type?: string | null;
}

/** Shown in the fixed fields while the search varies them. */
const SEARCHED = 'Set by the search';

export function SettingsForm({
  values,
  onChange,
  errors,
  name,
  namePlaceholder,
  onNameChange,
  devices,
  defaultDevice,
  builtin,
  predefinedSplit = null,
  classical = null,
  hybrid = null,
  segmentation = null,
  metrics = null,
}: SettingsFormProps) {
  const [advanced, { toggle }] = useDisclosure(false);
  const set =
    <K extends keyof SettingsValues>(key: K) =>
    (value: SettingsValues[K]) =>
      onChange({ ...values, [key]: value });
  // Keep unfinished input ("0.00") as typed; converting it would reset the field.
  const setNumber = (key: keyof SettingsValues) => (value: number | string) =>
    onChange({ ...values, [key]: value });
  // A hybrid network's search varies its classical model, not the training settings.
  const tuned = tunedSettings(hybrid ? { ...values, search: false } : values);
  const frozen = hybrid !== null && values.feature_training === 'frozen';
  const maxBatch = segmentation?.maxBatch ?? null;

  const automatic = devices.find((device) => device.id === defaultDevice)?.label;
  const deviceOptions = [
    { value: 'auto', label: automatic ? `Automatic: ${automatic}` : 'Automatic' },
    ...devices.map((device) => ({ value: device.id, label: device.label })),
  ];
  const device = deviceOptions.some((option) => option.value === values.device)
    ? values.device
    : 'auto';
  const fixed = predefinedSplit && predefinedSplit.length === 3 ? predefinedSplit : null;
  const fixedPercents = fixed ? splitPercents(fixed) : null;
  const testEnabled = fixed ? fixed[2] > 0 : builtin || values.test_enabled;
  const validates = hasValidation(values, fixed ? fixed[1] : null);
  // Without validation data an early stopping setting is an error: show it even when the
  // advanced settings are folded.
  const advancedOpen = advanced || Boolean(errors.patience);

  const nameField = (
    <TextInput
      label="Name"
      placeholder={namePlaceholder}
      value={name}
      onChange={(event) => onNameChange(event.currentTarget.value)}
      maxLength={120}
    />
  );
  const validationField = (
    <NumberInput
      label="Validation split"
      description={
        validates
          ? undefined
          : classical
            ? 'No validation data: the model is measured on the test rows only (without them, on its training rows).'
            : 'No validation data: the last epoch is kept, no early stopping or search; scores only on test (or training) data.'
      }
      value={fixedPercents ? fixedPercents[1] : values.val_split}
      disabled={fixed !== null}
      onChange={setNumber('val_split')}
      min={LIMITS.val_split.min}
      max={LIMITS.val_split.max}
      step={5}
      suffix="%"
      allowNegative={false}
      decimalScale={1}
      error={fixed ? undefined : errors.val_split}
    />
  );
  const testFields = (
    <div>
      <Switch
        label={builtin && !fixed ? 'Test on the official test images' : 'Hold out a test set'}
        description={
          fixed && segmentation
            ? `This dataset brings its own split, the folders train, val and test: ${formatSplit(fixed)} images for training, validation and test. Training uses it as it is.`
            : fixed
              ? `This dataset brings its own split from the Data tab: ${formatSplit(fixed)} rows for training, validation and test. Training uses it as it is; change it in the Data tab.`
              : builtin
                ? 'Always on for built-in datasets: the official test images are never trained on and not used to choose the weights; the kept weights are measured on them once, after training. The validation split is taken from the training images.'
                : 'Never trained on and not used to compare settings: the final model is measured on it once, after training.'
        }
        checked={testEnabled}
        disabled={fixed !== null || builtin}
        onChange={(event) => set('test_enabled')(event.currentTarget.checked)}
      />
      {testEnabled && (fixed !== null || !builtin) && (
        <SimpleGrid cols={{ base: 1, xs: 2 }} spacing="sm" mt="xs">
          <NumberInput
            label="Test split"
            value={fixedPercents ? fixedPercents[2] : values.test_split}
            disabled={fixed !== null}
            onChange={setNumber('test_split')}
            min={LIMITS.test_split.min}
            max={LIMITS.test_split.max}
            step={5}
            suffix="%"
            allowNegative={false}
            decimalScale={1}
            error={fixed ? undefined : errors.test_split}
          />
        </SimpleGrid>
      )}
    </div>
  );
  const maxRows = classical?.maxRows ?? null;
  const maxSamplesField = (
    <NumberInput
      label="Max. samples"
      description={
        classical
          ? maxRows !== null
            ? `${classical.model} is fitted on at most ${maxRows.toLocaleString('en-US')} training rows on this server; larger training sets are subsampled.`
            : 'Training rows to fit on (empty: all).'
          : undefined
      }
      placeholder={maxRows !== null ? `All, up to ${maxRows.toLocaleString('en-US')}` : 'All'}
      value={values.max_samples}
      onChange={setNumber('max_samples')}
      min={LIMITS.max_samples.min}
      max={maxRows ?? LIMITS.max_samples.max}
      allowDecimal={false}
      allowNegative={false}
      thousandSeparator=","
      error={errors.max_samples}
    />
  );
  const seedField = (
    <NumberInput
      label="Seed"
      placeholder="Random"
      value={values.seed}
      onChange={setNumber('seed')}
      min={LIMITS.seed.min}
      max={LIMITS.seed.max}
      allowDecimal={false}
      allowNegative={false}
      error={errors.seed}
    />
  );

  if (classical) {
    const type = classical.type ?? null;
    return (
      <Stack gap="sm">
        {nameField}
        <Text size="xs" c="dimmed">
          {classical.model} is fitted once with scikit-learn, on the CPU: only the split, the number
          of rows and the seed apply
          {classical.optimisesKernel
            ? '; the fit optimises the kernel hyperparameters itself.'
            : '.'}
        </Text>
        <SimpleGrid cols={{ base: 1, xs: 2 }} spacing="sm" verticalSpacing="sm">
          {validationField}
          {seedField}
        </SimpleGrid>
        {testFields}
        {maxSamplesField}
        {type && (
          <div>
            <Switch
              label="Hyperparameter search"
              description="Fits the model several times with other values of its parameters and keeps the best one."
              checked={values.search}
              onChange={(event) => set('search')(event.currentTarget.checked)}
            />
            {values.search && (
              <ClassicalSearchFields
                values={values}
                onChange={onChange}
                errors={errors}
                model={type}
                validates={validates}
                metrics={metrics}
              />
            )}
          </div>
        )}
      </Stack>
    );
  }

  return (
    <Stack gap="sm">
      {nameField}
      {hybrid && (
        <>
          <Text size="xs" c="dimmed">
            Two phases: the neural layers learn{' '}
            {hybrid.features !== null ? `${hybrid.features} features` : 'the features'} with a
            temporary linear head on the target (the settings below), then {hybrid.model} is fitted
            on the features of the training rows
            {hybrid.maxRows !== null
              ? ` (at most ${hybrid.maxRows.toLocaleString('en-US')} on this server)`
              : ''}
            .
          </Text>
          <Select
            label="Features"
            data={FEATURE_TRAININGS}
            value={values.feature_training}
            onChange={(value) => value && set('feature_training')(value as FeatureTraining)}
            allowDeselect={false}
          />
        </>
      )}
      <SimpleGrid cols={{ base: 1, xs: 2 }} spacing="sm" verticalSpacing="sm">
        <NumberInput
          label="Epochs"
          description={
            frozen
              ? 'Not used: the features stay random'
              : values.search && !hybrid
                ? 'Per trial of the search'
                : 'Passes over the training data'
          }
          disabled={frozen}
          value={values.epochs}
          onChange={setNumber('epochs')}
          min={LIMITS.epochs.min}
          max={LIMITS.epochs.max}
          allowDecimal={false}
          allowNegative={false}
          error={errors.epochs}
        />
        {segmentation ? (
          <NumberInput
            label="Batch size"
            description={
              maxBatch !== null && maxBatch >= 1
                ? `Images per optimizer step; up to ${maxBatch.toLocaleString('en-US')} fit into memory`
                : 'Images per optimizer step (4 to 16 is usual for segmentation)'
            }
            value={tuned.batch_size ? '' : values.segmentation_batch_size}
            placeholder={tuned.batch_size ? SEARCHED : undefined}
            disabled={tuned.batch_size}
            onChange={setNumber('segmentation_batch_size')}
            min={LIMITS.batch_size.min}
            max={LIMITS.batch_size.max}
            allowDecimal={false}
            allowNegative={false}
            error={errors.segmentation_batch_size}
          />
        ) : (
          <NumberInput
            label="Batch size"
            description="Samples per optimizer step"
            value={tuned.batch_size ? '' : values.batch_size}
            placeholder={tuned.batch_size ? SEARCHED : undefined}
            disabled={tuned.batch_size}
            onChange={setNumber('batch_size')}
            min={LIMITS.batch_size.min}
            max={LIMITS.batch_size.max}
            allowDecimal={false}
            allowNegative={false}
            error={errors.batch_size}
          />
        )}
        <NumberInput
          label="Learning rate"
          value={tuned.learning_rate ? '' : values.learning_rate}
          placeholder={tuned.learning_rate ? SEARCHED : undefined}
          disabled={tuned.learning_rate}
          onChange={setNumber('learning_rate')}
          min={0}
          max={LIMITS.learning_rate.max}
          step={0.0001}
          decimalScale={6}
          allowNegative={false}
          error={errors.learning_rate}
        />
        <Select
          label="Optimizer"
          data={OPTIMIZERS}
          value={tuned.optimizer ? null : values.optimizer}
          placeholder={tuned.optimizer ? SEARCHED : undefined}
          disabled={tuned.optimizer}
          onChange={(value) => value && set('optimizer')(value as Optimizer)}
          allowDeselect={false}
        />
        <Select
          label="Device"
          data={deviceOptions}
          value={device}
          onChange={(value) => value && set('device')(value)}
          allowDeselect={false}
        />
        {validationField}
        {segmentation && (
          <Select
            label="Loss"
            description="The Dice term helps classes that cover only a few pixels"
            data={LOSSES}
            value={values.loss}
            onChange={(value) => value && set('loss')(value as SegmentationLoss)}
            allowDeselect={false}
          />
        )}
      </SimpleGrid>

      {testFields}

      {hybrid ? (
        hybrid.type && (
          <div>
            <Switch
              label={`Hyperparameter search of ${hybrid.model}`}
              description="Learns the features once, then fits the classical model several times with other values of its parameters and keeps the best one."
              checked={values.search}
              onChange={(event) => set('search')(event.currentTarget.checked)}
            />
            {values.search && (
              <ClassicalSearchFields
                values={values}
                onChange={onChange}
                errors={errors}
                model={hybrid.type}
                validates={validates}
                metrics={metrics}
              />
            )}
          </div>
        )
      ) : (
        <div>
          <Switch
            label="Hyperparameter search"
            description="Trains the network several times with different settings and keeps the best one (by default the one with the lowest validation loss)."
            checked={values.search}
            onChange={(event) => set('search')(event.currentTarget.checked)}
          />
          {values.search && (
            <SearchFields values={values} onChange={onChange} errors={errors} metrics={metrics} />
          )}
        </div>
      )}

      <Button
        variant="subtle"
        color="gray"
        size="compact-sm"
        leftSection={advanced ? <IconChevronDown size={14} /> : <IconChevronRight size={14} />}
        onClick={toggle}
        aria-expanded={advancedOpen}
        style={{ alignSelf: 'flex-start' }}
      >
        Advanced settings
      </Button>
      <Collapse expanded={advancedOpen}>
        <Stack gap="sm">
          <SimpleGrid cols={{ base: 1, xs: 3 }} spacing="sm" verticalSpacing="sm">
            <NumberInput
              label="Weight decay"
              value={tuned.weight_decay ? '' : values.weight_decay}
              placeholder={tuned.weight_decay ? SEARCHED : undefined}
              disabled={tuned.weight_decay}
              onChange={setNumber('weight_decay')}
              min={LIMITS.weight_decay.min}
              max={LIMITS.weight_decay.max}
              step={0.0001}
              decimalScale={6}
              allowNegative={false}
              error={errors.weight_decay}
            />
            {maxSamplesField}
            {seedField}
          </SimpleGrid>
          <NumberInput
            label="Early stopping"
            description="Stop when the validation loss has not improved for this many epochs"
            placeholder="Off"
            value={values.patience}
            onChange={setNumber('patience')}
            min={LIMITS.patience.min}
            max={LIMITS.patience.max}
            allowDecimal={false}
            allowNegative={false}
            error={errors.patience}
          />
          <Checkbox
            label="Keep the weights of the best epoch"
            description={
              validates
                ? 'Test and save the weights with the lowest validation loss, not those of the last epoch.'
                : 'Not used without validation data: the weights of the last epoch are kept.'
            }
            disabled={!validates}
            checked={values.restore_best && validates}
            onChange={(event) => set('restore_best')(event.currentTarget.checked)}
          />
        </Stack>
      </Collapse>
    </Stack>
  );
}

// --------------------------------------------------------------------------
// Hyperparameter search
// --------------------------------------------------------------------------

const METHODS: { value: TuningMethod; label: string }[] = [
  { value: 'random', label: 'Random search' },
  { value: 'grid', label: 'Grid search' },
];

const options = (values: number[], format: (value: number) => string) =>
  values.map((value) => ({ value: String(value), label: format(value) }));

const LR_OPTIONS = options(LR_PRESETS, (v) => formatSetting('learning_rate', v));
const BATCH_SIZE_OPTIONS = options(BATCH_SIZE_PRESETS, String);
const WEIGHT_DECAY_OPTIONS = options(WEIGHT_DECAY_PRESETS, (v) => formatSetting('weight_decay', v));

interface SearchFieldsProps {
  values: SettingsValues;
  onChange: (values: SettingsValues) => void;
  errors: SettingsErrors;
  metrics: MetricInfo[] | null;
}

/** What ranks the trials; only the metrics the data allows (all while that is not known). */
function MetricField({
  values,
  onChange,
  errors,
  metrics,
  where,
}: SearchFieldsProps & { where: string }) {
  const offered = metrics ?? METRICS;
  const data = offered.map((metric) => ({ value: metric.name, label: metric.label }));
  if (!offered.some((metric) => metric.name === values.metric)) {
    const chosen = METRICS.find((metric) => metric.name === values.metric);
    data.push({ value: values.metric, label: chosen?.label ?? values.metric });
  }
  return (
    <Select
      label="Objective metric"
      description={`Ranks the trials on ${where}; ${metricDirection(values.metric)}.`}
      data={data}
      value={values.metric}
      onChange={(value) => value && onChange({ ...values, metric: value as TuningMetric })}
      allowDeselect={false}
      error={errors.metric}
      w={{ base: '100%', xs: 260 }}
    />
  );
}

function SearchFields({ values, onChange, errors, metrics }: SearchFieldsProps) {
  const update = (changes: Partial<SettingsValues>) => onChange({ ...values, ...changes });
  const numbers = (list: string[]) => list.map(Number);
  const grid = values.search_method === 'grid';
  const trials = plannedTrials(values);
  const epochs = parseField(values.epochs);
  const requested = parseField(values.search_trials);

  let summary: string | null = null;
  if (trials !== null && typeof epochs === 'number' && epochs > 0) {
    const count = `${trials} ${trials === 1 ? 'trial' : 'trials'}`;
    summary = `${count} of ${epochs} ${epochs === 1 ? 'epoch' : 'epochs'} each.`;
    if (!grid && typeof requested === 'number' && trials < requested) {
      summary += ` Only ${trials} different combinations: each is tried once.`;
    }
  }

  return (
    <Paper withBorder p="sm" radius="sm" mt="xs">
      <Stack gap="sm">
        <Group gap="sm" align="flex-end" wrap="wrap">
          <Input.Wrapper label="Method">
            <SegmentedControl
              display="flex"
              data={METHODS}
              value={values.search_method}
              onChange={(value) => update({ search_method: value as TuningMethod })}
            />
          </Input.Wrapper>
          {!grid && (
            <NumberInput
              label="Trials"
              w={110}
              value={values.search_trials}
              onChange={(value) => update({ search_trials: value })}
              min={LIMITS.search_trials.min}
              max={LIMITS.search_trials.max}
              allowDecimal={false}
              allowNegative={false}
              error={errors.search_trials}
            />
          )}
        </Group>

        <SearchRow
          label="Learning rate"
          checked={values.tune_learning_rate}
          onToggle={(checked) => update({ tune_learning_rate: checked })}
          fixed={formatField('learning_rate', values.learning_rate)}
        >
          {grid ? (
            <MultiSelect
              aria-label="Learning rates to try"
              data={LR_OPTIONS}
              value={values.lr_values.map(String)}
              onChange={(list) => update({ lr_values: numbers(list) })}
              error={errors.lr_values}
            />
          ) : (
            <Group gap="xs" wrap="nowrap" align="flex-start">
              <NumberInput
                aria-label="Smallest learning rate"
                value={values.lr_min}
                onChange={(value) => update({ lr_min: value })}
                min={0}
                max={LIMITS.learning_rate.max}
                decimalScale={6}
                allowNegative={false}
                error={errors.lr_min}
                style={{ flex: 1 }}
              />
              <Text size="sm" mt={7}>
                to
              </Text>
              <NumberInput
                aria-label="Largest learning rate"
                value={values.lr_max}
                onChange={(value) => update({ lr_max: value })}
                min={0}
                max={LIMITS.learning_rate.max}
                decimalScale={6}
                allowNegative={false}
                error={errors.lr_max}
                style={{ flex: 1 }}
              />
            </Group>
          )}
        </SearchRow>
        <SearchRow
          label="Batch size"
          checked={values.tune_batch_size}
          onToggle={(checked) => update({ tune_batch_size: checked })}
          fixed={formatField('batch_size', values.batch_size)}
        >
          <MultiSelect
            aria-label="Batch sizes to try"
            data={BATCH_SIZE_OPTIONS}
            value={values.batch_sizes.map(String)}
            onChange={(list) => update({ batch_sizes: numbers(list) })}
            error={errors.batch_sizes}
          />
        </SearchRow>
        <SearchRow
          label="Optimizer"
          checked={values.tune_optimizer}
          onToggle={(checked) => update({ tune_optimizer: checked })}
          fixed={formatSetting('optimizer', values.optimizer)}
        >
          <MultiSelect
            aria-label="Optimizers to try"
            data={OPTIMIZERS}
            value={values.optimizers}
            onChange={(list) => update({ optimizers: list as Optimizer[] })}
            error={errors.optimizers}
          />
        </SearchRow>
        <SearchRow
          label="Weight decay"
          checked={values.tune_weight_decay}
          onToggle={(checked) => update({ tune_weight_decay: checked })}
          fixed={formatField('weight_decay', values.weight_decay)}
        >
          <MultiSelect
            aria-label="Weight decays to try"
            data={WEIGHT_DECAY_OPTIONS}
            value={values.weight_decays.map(String)}
            onChange={(list) => update({ weight_decays: numbers(list) })}
            error={errors.weight_decays}
          />
        </SearchRow>

        <MetricField
          values={values}
          onChange={onChange}
          errors={errors}
          metrics={metrics}
          where="the validation data"
        />

        {errors.search && (
          <Text size="sm" c="red" role="alert">
            {errors.search}
          </Text>
        )}
        <Text size="xs" c="dimmed">
          {summary && `${summary} `}
          {grid
            ? 'Every combination of the chosen values is trained.'
            : 'Learning rates are drawn evenly on a log scale.'}{' '}
          All trials use the same data and starting weights; the trial with the best objective
          metric wins (each trial keeps the weights of its epoch with the lowest validation loss).
        </Text>
      </Stack>
    </Paper>
  );
}

function formatField(
  name: 'learning_rate' | 'batch_size' | 'weight_decay',
  field: string | number
) {
  const value = parseField(field);
  return typeof value === 'number' ? formatSetting(name, value) : '–';
}

function SearchRow({
  label,
  checked,
  onToggle,
  fixed,
  children,
}: {
  label: string;
  checked: boolean;
  onToggle: (checked: boolean) => void;
  /** The value used when the setting is not searched. */
  fixed: string;
  children: React.ReactNode;
}) {
  return (
    // On phones the values move below the checkbox, where they have the full width.
    <Group align="flex-start" gap="sm" wrap="wrap">
      <Checkbox
        label={label}
        aria-label={`Search ${label.toLowerCase()}`}
        checked={checked}
        onChange={(event) => onToggle(event.currentTarget.checked)}
        w={120}
        mt={7}
        style={{ flexShrink: 0 }}
      />
      <Box style={{ flex: 1, minWidth: 220 }}>
        {checked ? (
          children
        ) : (
          <Text size="sm" c="dimmed" mt={6}>
            Fixed: {fixed}
          </Text>
        )}
      </Box>
    </Group>
  );
}

// --------------------------------------------------------------------------
// Hyperparameter search of a classical model
// --------------------------------------------------------------------------

const MODEL_NOTES: Record<string, string> = {
  GaussianProcessRegressor:
    'The fit optimises the kernel hyperparameters itself: a search can only vary alpha, the normalisation of the targets and the optimizer restarts.',
  GaussianProcessClassifier:
    'The fit optimises the kernel hyperparameters itself: a search can only vary the optimizer restarts.',
};

const RANGE_MODES = [
  { value: 'range', label: 'Range' },
  { value: 'list', label: 'List' },
];
const SCALES = [
  { value: 'log', label: 'Log' },
  { value: 'linear', label: 'Linear' },
];

interface ClassicalSearchFieldsProps extends SearchFieldsProps {
  /** The type of the model brick, e.g. "SVC". */
  model: string;
  /** False: no validation rows, so the trials are always cross-validated. */
  validates: boolean;
}

function ClassicalSearchFields({
  values,
  onChange,
  errors,
  model,
  validates,
  metrics,
}: ClassicalSearchFieldsProps) {
  const update = (changes: Partial<SettingsValues>) => onChange({ ...values, ...changes });
  const grid = values.search_method === 'grid';
  const states = classicalParams(values, model);
  const setParam = (name: string, changes: Partial<ParamState>) =>
    update({
      classical_params: {
        ...values.classical_params,
        [model]: { ...states, [name]: { ...states[name], ...changes } },
      },
    });
  const trials = plannedClassicalTrials(values, model);
  const requested = parseField(values.search_trials);
  const crossValidates = values.cv_enabled || !validates;
  const folds = parseField(values.cv_folds);
  const foldCount =
    values.cv_enabled && typeof folds === 'number' && Number.isInteger(folds)
      ? folds
      : FOLDS.default;

  let summary: string | null = null;
  if (trials !== null) {
    summary = `${trials} ${trials === 1 ? 'trial' : 'trials'}`;
    summary += crossValidates
      ? ` × ${foldCount} folds, then the best one is fitted on all training rows.`
      : ', each fitted on the training rows and measured on the validation rows.';
    if (!grid && typeof requested === 'number' && trials < requested) {
      summary += ` Only ${trials} different combinations: each is tried once.`;
    }
  }

  return (
    <Paper withBorder p="sm" radius="sm" mt="xs">
      <Stack gap="sm">
        <Group gap="sm" align="flex-end" wrap="wrap">
          <Input.Wrapper label="Method">
            <SegmentedControl
              display="flex"
              data={METHODS}
              value={values.search_method}
              onChange={(value) => update({ search_method: value as TuningMethod })}
            />
          </Input.Wrapper>
          {!grid && (
            <NumberInput
              label="Trials"
              w={110}
              value={values.search_trials}
              onChange={(value) => update({ search_trials: value })}
              min={LIMITS.search_trials.min}
              max={LIMITS.search_trials.max}
              allowDecimal={false}
              allowNegative={false}
              error={errors.search_trials}
            />
          )}
        </Group>

        {MODEL_NOTES[model] && (
          <Text size="xs" c="dimmed">
            {MODEL_NOTES[model]}
          </Text>
        )}

        {(CLASSICAL_TUNABLES[model] ?? []).map((item) => (
          <ParamRow
            key={item.name}
            item={item}
            state={states[item.name]}
            grid={grid}
            error={errors.params?.[item.name]}
            onChange={(changes) => setParam(item.name, changes)}
          />
        ))}

        <MetricField
          values={values}
          onChange={onChange}
          errors={errors}
          metrics={metrics}
          where={crossValidates ? 'the cross-validation' : 'the validation rows'}
        />

        {validates ? (
          <Group gap="sm" align="flex-start" wrap="wrap">
            <Checkbox
              label="Cross-validate on the training rows"
              description="Instead of the validation rows, which the search then does not see."
              checked={values.cv_enabled}
              onChange={(event) => update({ cv_enabled: event.currentTarget.checked })}
              style={{ flex: 1, minWidth: 220 }}
            />
            {values.cv_enabled && (
              <NumberInput
                label="Folds"
                w={110}
                value={values.cv_folds}
                onChange={(value) => update({ cv_folds: value })}
                min={LIMITS.cv_folds.min}
                max={LIMITS.cv_folds.max}
                allowDecimal={false}
                allowNegative={false}
                error={errors.cv_folds}
              />
            )}
          </Group>
        ) : (
          <Text size="xs" c="dimmed">
            No validation rows: the trials are compared by {FOLDS.default}-fold cross-validation on
            the training rows.
          </Text>
        )}

        {errors.search && (
          <Text size="sm" c="red" role="alert">
            {errors.search}
          </Text>
        )}
        <Text size="xs" c="dimmed">
          {summary && `${summary} `}
          {grid
            ? 'Every combination of the listed values is fitted.'
            : 'Ranges are drawn evenly on their scale; whole-number parameters give whole numbers.'}{' '}
          The best trial is measured on the test rows like a single fit.
        </Text>
      </Stack>
    </Paper>
  );
}

function ParamRow({
  item,
  state,
  grid,
  error,
  onChange,
}: {
  item: Tunable;
  state: ParamState;
  grid: boolean;
  error: string | undefined;
  onChange: (changes: Partial<ParamState>) => void;
}) {
  const choice = item.kind === 'choice' || item.kind === 'bool';
  const range = !grid && !choice && state.mode === 'range';
  let editor: React.ReactNode;
  if (choice) {
    editor = (
      <MultiSelect
        aria-label={`${item.label} to try`}
        data={item.options.map((option) => ({ value: String(option), label: String(option) }))}
        value={state.choices}
        onChange={(choices) => onChange({ choices })}
        error={error}
      />
    );
  } else {
    editor = (
      <Stack gap={4}>
        {!grid && (
          <SegmentedControl
            size="xs"
            data={RANGE_MODES}
            value={state.mode}
            onChange={(mode) => onChange({ mode: mode as ParamState['mode'] })}
            aria-label={`${item.label}: range or list`}
            style={{ alignSelf: 'flex-start' }}
          />
        )}
        {range ? (
          <Group gap="xs" wrap="nowrap" align="flex-start">
            <NumberInput
              aria-label={`Smallest ${item.label.toLowerCase()}`}
              value={state.min}
              onChange={(min) => onChange({ min })}
              allowNegative={false}
              allowDecimal={item.kind !== 'int'}
              decimalScale={12}
              error={error}
              style={{ flex: 1 }}
            />
            <Text size="sm" mt={7}>
              to
            </Text>
            <NumberInput
              aria-label={`Largest ${item.label.toLowerCase()}`}
              value={state.max}
              onChange={(max) => onChange({ max })}
              allowNegative={false}
              allowDecimal={item.kind !== 'int'}
              decimalScale={12}
              style={{ flex: 1 }}
            />
            <Select
              aria-label={`Scale of the ${item.label.toLowerCase()}`}
              data={SCALES}
              value={state.scale}
              onChange={(scale) => scale && onChange({ scale: scale as ParamState['scale'] })}
              allowDeselect={false}
              w={90}
            />
          </Group>
        ) : (
          <TextInput
            aria-label={`${item.label} to try`}
            placeholder={item.none ? 'e.g. none, 3, 6' : 'e.g. 0.1, 1, 10'}
            description={item.help}
            value={state.text}
            onChange={(event) => onChange({ text: event.currentTarget.value })}
            error={error}
          />
        )}
      </Stack>
    );
  }
  return (
    <SearchRow
      label={item.label}
      checked={state.on}
      onToggle={(on) => onChange({ on })}
      fixed="the value of the brick"
    >
      {editor}
    </SearchRow>
  );
}
