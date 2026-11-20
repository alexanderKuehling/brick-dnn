import { useId, type ReactNode } from 'react';
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
  LossName,
  LossTask,
  Optimizer,
  TrainingDevice,
  TuningMethod,
  TuningMetric,
} from '@/api';
import { HintedLabel, InfoHint } from '@/components/InfoHint';
import { currentLanguage, currentLocale, i18n, useTranslation } from '@/i18n';
import { numberSeparators } from '@/i18n/mantine';
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
  classWeightModes,
  effectiveLoss,
  FEATURE_TRAININGS,
  FOCAL_GAMMA_PRESETS,
  hasValidation,
  HUBER_DELTA_PRESETS,
  LABEL_SMOOTHING_PRESETS,
  LIMITS,
  LOSS_OPTIONS,
  lossesInUse,
  lossUses,
  LR_PRESETS,
  OPTIMIZERS,
  parseField,
  plannedClassicalTrials,
  plannedHybridTrials,
  plannedTrials,
  searchesNeural,
  tunedSettings,
  usesParam,
  WEIGHT_DECAY_PRESETS,
  type ClassWeightsMode,
  type LossParam,
  type LossScope,
  type SegmentationScope,
  type SettingsErrors,
  type SettingsValues,
} from './settings';
import classes from './Training.module.css';

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
   * features; the search varies the parameters of its classical model and/or the training
   * settings of the neural layers.
   */
  hybrid?: HybridSettings | null;
  /**
   * The objective metrics a search can rank by on this data (from the training check; null
   * while it is not known: all are offered and the server checks).
   */
  metrics?: MetricInfo[] | null;
  /**
   * The task of the data, whose losses the form offers, and its classes (from the training
   * check; a segmentation's without it). Null: not known, the loss is the task's default.
   */
  loss?: LossScope | null;
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

/**
 * Select options with their labels in the current language
 * (`training.settings.options.<group>.<value>`); options without a translation keep theirs.
 */
function optionsIn<T extends string>(
  options: { value: T; label: string }[],
  group: 'featureTraining' | 'loss'
): { value: T; label: string }[] {
  return options.map((option) => {
    const key = `training.settings.options.${group}.${option.value}`;
    return i18n.exists(key)
      ? { ...option, label: String(i18n.t(key as 'training.settings.options.loss.cross_entropy')) }
      : option;
  });
}

/** Shown in the fixed fields while the search varies them. */
const searchedText = () => i18n.t('training.settings.searched');
/** The training settings of a hybrid network with frozen features. */
const notTrained = () => i18n.t('training.settings.notTrained');

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
  loss = null,
}: SettingsFormProps) {
  const { t } = useTranslation();
  const [advanced, { toggle }] = useDisclosure(false);
  const set =
    <K extends keyof SettingsValues>(key: K) =>
    (value: SettingsValues[K]) =>
      onChange({ ...values, [key]: value });
  // Keep unfinished input ("0.00") as typed; converting it would reset the field.
  const setNumber = (key: keyof SettingsValues) => (value: number | string) =>
    onChange({ ...values, [key]: value });
  // A hybrid network's search varies the training settings only if it searches the neural
  // layers too; frozen features use none of them.
  const lossScope: LossScope | null = loss ?? (segmentation ? { task: 'segmentation' } : null);
  const task = lossScope?.task ?? null;
  const tuned = tunedSettings(
    hybrid ? { ...values, search: searchesNeural(values) } : values,
    task
  );
  const frozen = hybrid !== null && values.feature_training === 'frozen';
  const maxBatch = segmentation?.maxBatch ?? null;

  const automatic = devices.find((device) => device.id === defaultDevice)?.label;
  const deviceOptions = [
    {
      value: 'auto',
      label: automatic
        ? t('training.settings.automaticDevice', { device: automatic })
        : t('training.settings.automatic'),
    },
    ...devices.map((device) => ({ value: device.id, label: device.label })),
  ];
  const device = deviceOptions.some((option) => option.value === values.device)
    ? values.device
    : 'auto';
  const fixed = predefinedSplit && predefinedSplit.length === 3 ? predefinedSplit : null;
  const fixedPercents = fixed ? splitPercents(fixed) : null;
  const testEnabled = fixed ? fixed[2] > 0 : builtin || values.test_enabled;
  const validates = hasValidation(values, fixed ? fixed[1] : null);
  const advancedId = useId();
  const lossId = useId();
  const patienceId = useId();

  const nameField = (
    <TextInput
      label={t('training.settings.name')}
      placeholder={namePlaceholder}
      value={name}
      onChange={(event) => onNameChange(event.currentTarget.value)}
      maxLength={120}
    />
  );
  const validationField = (
    <NumberInput
      label={t('training.settings.valSplit')}
      description={
        validates
          ? undefined
          : classical
            ? t('training.settings.noValClassical')
            : t('training.settings.noVal')
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
        label={
          builtin && !fixed ? t('training.settings.officialTest') : t('training.settings.holdOut')
        }
        description={
          fixed && segmentation
            ? t('training.settings.ownSplitImages', { split: formatSplit(fixed) })
            : fixed
              ? t('training.settings.ownSplitRows', { split: formatSplit(fixed) })
              : builtin
                ? t('training.settings.builtinTest')
                : undefined
        }
        checked={testEnabled}
        disabled={fixed !== null || builtin}
        onChange={(event) => set('test_enabled')(event.currentTarget.checked)}
      />
      {testEnabled && (fixed !== null || !builtin) && (
        <SimpleGrid cols={{ base: 1, xs: 2 }} spacing="sm" mt="xs">
          <NumberInput
            label={t('training.settings.testSplit')}
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
      label={t('training.settings.maxSamples')}
      description={
        classical
          ? maxRows !== null
            ? t('training.settings.maxRowsHint', {
                model: classical.model,
                rows: maxRows.toLocaleString(currentLocale()),
              })
            : undefined
          : undefined
      }
      placeholder={
        maxRows !== null
          ? t('training.settings.allUpTo', { rows: maxRows.toLocaleString(currentLocale()) })
          : t('training.settings.all')
      }
      value={values.max_samples}
      onChange={setNumber('max_samples')}
      min={LIMITS.max_samples.min}
      max={maxRows ?? LIMITS.max_samples.max}
      allowDecimal={false}
      allowNegative={false}
      thousandSeparator={numberSeparators(currentLanguage()).thousandSeparator}
      error={errors.max_samples}
    />
  );
  const seedField = (
    <NumberInput
      label={t('training.settings.seed')}
      placeholder={t('training.settings.random')}
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
      <Stack gap="xl">
        {nameField}
        <FormSection
          title={t('training.settings.sections.fit')}
          info={
            classical.optimisesKernel
              ? t('training.settings.classicalNoteKernel', { model: classical.model })
              : t('training.settings.classicalNote', { model: classical.model })
          }
        >
          <SimpleGrid cols={{ base: 1, xs: 2 }} spacing="sm" verticalSpacing="sm">
            {maxSamplesField}
            {seedField}
          </SimpleGrid>
        </FormSection>
        <FormSection
          title={t('training.settings.sections.split')}
          info={builtin ? undefined : t('training.settings.testHint')}
        >
          <SimpleGrid cols={{ base: 1, xs: 2 }} spacing="sm" verticalSpacing="sm">
            {validationField}
          </SimpleGrid>
          {testFields}
        </FormSection>
        {type && (
          <FormSection title={t('training.settings.sections.search')}>
            <Switch
              label={t('training.settings.search')}
              description={t('training.settings.searchClassical')}
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
          </FormSection>
        )}
      </Stack>
    );
  }

  const epochsField = (
    <NumberInput
      label={t('training.settings.epochs')}
      description={
        frozen
          ? t('training.settings.epochsFrozen')
          : values.search && !hybrid
            ? t('training.settings.perTrial')
            : undefined
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
  );
  const batchSizeField = segmentation ? (
    <NumberInput
      label={t('training.settings.batchSize')}
      description={
        maxBatch !== null && maxBatch >= 1
          ? t('training.settings.imagesPerStepMax', {
              max: maxBatch.toLocaleString(currentLocale()),
            })
          : t('training.settings.imagesPerStep')
      }
      value={tuned.batch_size ? '' : values.segmentation_batch_size}
      placeholder={tuned.batch_size ? searchedText() : undefined}
      disabled={tuned.batch_size || frozen}
      onChange={setNumber('segmentation_batch_size')}
      min={LIMITS.batch_size.min}
      max={LIMITS.batch_size.max}
      allowDecimal={false}
      allowNegative={false}
      error={errors.segmentation_batch_size}
    />
  ) : (
    <NumberInput
      label={t('training.settings.batchSize')}
      description={frozen ? notTrained() : undefined}
      value={tuned.batch_size ? '' : values.batch_size}
      placeholder={tuned.batch_size ? searchedText() : undefined}
      disabled={tuned.batch_size || frozen}
      onChange={setNumber('batch_size')}
      min={LIMITS.batch_size.min}
      max={LIMITS.batch_size.max}
      allowDecimal={false}
      allowNegative={false}
      error={errors.batch_size}
    />
  );
  // A hybrid network can search only when the server offers a search of its model.
  const searchable = !hybrid || Boolean(hybrid.type);

  return (
    <Stack gap="xl">
      {nameField}

      <FormSection
        title={t('training.settings.sections.model')}
        info={
          hybrid
            ? t(
                hybrid.maxRows !== null
                  ? 'training.settings.hybridNoteMax'
                  : 'training.settings.hybridNote',
                {
                  features:
                    hybrid.features !== null
                      ? t('training.settings.nFeatures', { count: hybrid.features })
                      : t('training.settings.theFeatures'),
                  model: hybrid.model,
                  rows: hybrid.maxRows?.toLocaleString(currentLocale()) ?? '',
                }
              )
            : undefined
        }
      >
        {hybrid && (
          <>
            <Select
              label={t('training.settings.features')}
              description={frozen ? t('training.settings.frozenHint') : undefined}
              data={optionsIn(FEATURE_TRAININGS, 'featureTraining')}
              value={values.feature_training}
              onChange={(value) => value && set('feature_training')(value as FeatureTraining)}
              allowDeselect={false}
            />
          </>
        )}
        <SimpleGrid cols={{ base: 1, xs: 2 }} spacing="sm" verticalSpacing="sm">
          <NumberInput
            label={t('training.settings.learningRate')}
            description={frozen ? notTrained() : undefined}
            value={tuned.learning_rate ? '' : values.learning_rate}
            placeholder={tuned.learning_rate ? searchedText() : undefined}
            disabled={tuned.learning_rate || frozen}
            onChange={setNumber('learning_rate')}
            min={0}
            max={LIMITS.learning_rate.max}
            step={0.0001}
            decimalScale={6}
            allowNegative={false}
            error={errors.learning_rate}
          />
          <Select
            label={t('training.settings.optimizer')}
            description={frozen ? notTrained() : undefined}
            data={OPTIMIZERS}
            value={tuned.optimizer ? null : values.optimizer}
            placeholder={tuned.optimizer ? searchedText() : undefined}
            disabled={tuned.optimizer || frozen}
            onChange={(value) => value && set('optimizer')(value as Optimizer)}
            allowDeselect={false}
          />
          {batchSizeField}
          {task && (
            <div>
              <HintedLabel id={lossId} label={t('training.settings.loss')}>
                {i18n.t('ui.training.SettingsForm.theCurvesAndEarlyStopping', {
                  value:
                    LOSS_OPTIONS[task].find((o) => o.value === effectiveLoss(values, task))?.help ??
                    '',
                })}
              </HintedLabel>
              <Select
                id={lossId}
                description={
                  frozen
                    ? notTrained()
                    : tuned.loss
                      ? i18n.t('ui.training.SettingsForm.theSearchTriesTheLosses')
                      : undefined
                }
                data={LOSS_OPTIONS[task].map(({ value, label }) => ({ value, label }))}
                value={tuned.loss ? null : effectiveLoss(values, task)}
                placeholder={tuned.loss ? searchedText() : undefined}
                disabled={tuned.loss || frozen}
                onChange={(value) => value && set('loss')(value as LossName)}
                allowDeselect={false}
              />
            </div>
          )}
        </SimpleGrid>
        {lossScope && !frozen && (
          <LossFields
            values={values}
            onChange={onChange}
            errors={errors}
            task={lossScope.task}
            tuned={tuned}
            searched={hybrid ? searchesNeural(values) : values.search}
          />
        )}
      </FormSection>

      <FormSection title={t('training.settings.sections.schedule')}>
        <SimpleGrid cols={{ base: 1, xs: 2 }} spacing="sm" verticalSpacing="sm">
          {epochsField}
          <div>
            <HintedLabel id={patienceId} label={t('training.settings.earlyStopping')}>
              {t('training.settings.earlyStoppingHint')}
            </HintedLabel>
            <NumberInput
              id={patienceId}
              description={frozen ? notTrained() : undefined}
              placeholder={t('common.off')}
              disabled={frozen}
              value={values.patience}
              onChange={setNumber('patience')}
              min={LIMITS.patience.min}
              max={LIMITS.patience.max}
              allowDecimal={false}
              allowNegative={false}
              error={errors.patience}
            />
          </div>
        </SimpleGrid>
        <Checkbox
          label={t('training.settings.restoreBest')}
          description={
            frozen ? notTrained() : validates ? undefined : t('training.settings.restoreBestNoVal')
          }
          disabled={!validates || frozen}
          checked={values.restore_best && validates && !frozen}
          onChange={(event) => set('restore_best')(event.currentTarget.checked)}
        />
      </FormSection>

      <FormSection
        title={t('training.settings.sections.split')}
        info={builtin ? undefined : t('training.settings.testHint')}
      >
        <SimpleGrid cols={{ base: 1, xs: 2 }} spacing="sm" verticalSpacing="sm">
          {validationField}
        </SimpleGrid>
        {testFields}
      </FormSection>

      {searchable && (
        <FormSection title={t('training.settings.sections.search')}>
          {hybrid?.type ? (
            <div>
              <Switch
                label={t('training.settings.search')}
                description={t('training.settings.searchHybrid', { model: hybrid.model })}
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
                  hybrid={{ model: hybrid.model, frozen }}
                  task={task}
                />
              )}
            </div>
          ) : (
            <div>
              <Switch
                label={t('training.settings.search')}
                description={t('training.settings.searchNeural')}
                checked={values.search}
                onChange={(event) => set('search')(event.currentTarget.checked)}
              />
              {values.search && (
                <SearchFields
                  values={values}
                  onChange={onChange}
                  errors={errors}
                  metrics={metrics}
                  task={task}
                />
              )}
            </div>
          )}
        </FormSection>
      )}

      <section aria-label={t('training.settings.advanced')}>
        <Button
          variant="subtle"
          color="gray"
          size="compact-sm"
          leftSection={advanced ? <IconChevronDown size={14} /> : <IconChevronRight size={14} />}
          onClick={toggle}
          aria-expanded={advanced}
          aria-controls={advancedId}
          className={classes.advancedToggle}
        >
          {t('training.settings.advanced')}
        </Button>
        <Collapse expanded={advanced} id={advancedId}>
          <SimpleGrid cols={{ base: 1, xs: 2 }} spacing="sm" verticalSpacing="sm" pt="sm">
            <Select
              label={t('training.settings.device')}
              data={deviceOptions}
              value={device}
              onChange={(value) => value && set('device')(value)}
              allowDeselect={false}
            />
            <NumberInput
              label={t('training.settings.weightDecay')}
              description={frozen ? notTrained() : undefined}
              value={tuned.weight_decay ? '' : values.weight_decay}
              placeholder={tuned.weight_decay ? searchedText() : undefined}
              disabled={tuned.weight_decay || frozen}
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
        </Collapse>
      </section>
    </Stack>
  );
}

/** A labelled group of settings (model and optimizer, schedule …). */
function FormSection({
  title,
  info,
  children,
}: {
  title: string;
  /** An explanation behind an (i) button next to the title. */
  info?: ReactNode;
  children: ReactNode;
}) {
  const id = useId();
  return (
    <section aria-labelledby={id}>
      <Group gap={4} wrap="nowrap" className={classes.sectionHead}>
        <Text component="h3" id={id} className={classes.sectionTitle}>
          {title}
        </Text>
        {info && <InfoHint topic={title}>{info}</InfoHint>}
      </Group>
      <Stack gap="sm">{children}</Stack>
    </section>
  );
}

// --------------------------------------------------------------------------
// Hyperparameter search
// --------------------------------------------------------------------------

const METHODS: { value: TuningMethod; label: string }[] = [
  {
    value: 'random',
    get label() {
      return i18n.t('ui.training.SettingsForm.randomSearch');
    },
  },
  {
    value: 'grid',
    get label() {
      return i18n.t('ui.training.SettingsForm.gridSearch');
    },
  },
];

const options = (values: number[], format: (value: number) => string) =>
  values.map((value) => ({ value: String(value), label: format(value) }));

const LR_OPTIONS = options(LR_PRESETS, (v) => formatSetting('learning_rate', v));
const BATCH_SIZE_OPTIONS = options(BATCH_SIZE_PRESETS, String);
const WEIGHT_DECAY_OPTIONS = options(WEIGHT_DECAY_PRESETS, (v) => formatSetting('weight_decay', v));
const FOCAL_GAMMA_OPTIONS = options(FOCAL_GAMMA_PRESETS, String);
const LABEL_SMOOTHING_OPTIONS = options(LABEL_SMOOTHING_PRESETS, String);
const HUBER_DELTA_OPTIONS = options(HUBER_DELTA_PRESETS, String);

interface SearchFieldsProps {
  values: SettingsValues;
  onChange: (values: SettingsValues) => void;
  errors: SettingsErrors;
  metrics: MetricInfo[] | null;
  /** The task whose losses a search can try (null: not known, no loss rows). */
  task?: LossTask | null;
}

/**
 * The settings of the losses the run uses: class weights, focal gamma, label smoothing and
 * Huber's delta (those a search varies show that).
 */
function LossFields({
  values,
  onChange,
  errors,
  task,
  tuned,
  searched,
}: {
  values: SettingsValues;
  onChange: (values: SettingsValues) => void;
  errors: SettingsErrors;
  task: LossTask;
  tuned: ReturnType<typeof tunedSettings>;
  /** Whether a search varies the training settings (a hybrid network's: its neural ones). */
  searched: boolean;
}) {
  useTranslation();
  const ids = { focal_gamma: useId(), label_smoothing: useId(), huber_delta: useId() };
  const losses = lossesInUse({ ...values, search: searched }, task);
  const uses = (param: LossParam) => losses.some((name) => usesParam(param, name, task));
  const update = (changes: Partial<SettingsValues>) => onChange({ ...values, ...changes });
  const modes = classWeightModes(losses);
  const mode = modes.some((option) => option.value === values.class_weights_mode)
    ? values.class_weights_mode
    : modes[0].value;
  if (!(['class_weights', 'focal_gamma', 'label_smoothing', 'huber_delta'] as const).some(uses)) {
    return null;
  }
  const pixels = task === 'segmentation';
  return (
    <SimpleGrid cols={{ base: 1, xs: 2 }} spacing="sm" verticalSpacing="sm">
      {uses('class_weights') && (
        <Select
          label={i18n.t('ui.training.SettingsForm.classWeights')}
          description={
            mode === 'balanced'
              ? i18n.t('ui.training.SettingsForm.nKCountFromThe', {
                  value: pixels
                    ? i18n.t('ui.training.SettingsForm.pixelsOfTheTrainingImages')
                    : i18n.t('ui.training.SettingsForm.trainingRows'),
                })
              : mode === 'custom'
                ? i18n.t('ui.training.SettingsForm.oneWeightPerClassOr')
                : i18n.t('ui.training.SettingsForm.everyClassCountsTheSame')
          }
          data={modes}
          value={mode}
          onChange={(value) => value && update({ class_weights_mode: value as ClassWeightsMode })}
          allowDeselect={false}
        />
      )}
      {uses('class_weights') && mode === 'custom' && (
        <TextInput
          label={i18n.t('ui.training.SettingsForm.weights')}
          placeholder={i18n.t('ui.training.SettingsForm.text110OrFraud10')}
          description={i18n.t('ui.training.SettingsForm.classesThatAreNotNamed')}
          value={values.class_weights_text}
          onChange={(event) => update({ class_weights_text: event.currentTarget.value })}
          error={errors.class_weights_text}
        />
      )}
      {uses('focal_gamma') && (
        <div>
          <HintedLabel id={ids.focal_gamma} label={i18n.t('ui.training.SettingsForm.focalGamma')}>
            {i18n.t('ui.training.SettingsForm.howMuchLessWellClassified')}
          </HintedLabel>
          <NumberInput
            id={ids.focal_gamma}
            value={tuned.focal_gamma ? '' : values.focal_gamma}
            placeholder={tuned.focal_gamma ? searchedText() : undefined}
            disabled={tuned.focal_gamma}
            onChange={(value) => update({ focal_gamma: value })}
            min={LIMITS.focal_gamma.min}
            max={LIMITS.focal_gamma.max}
            step={0.5}
            decimalScale={3}
            allowNegative={false}
            error={errors.focal_gamma}
          />
        </div>
      )}
      {uses('label_smoothing') && (
        <div>
          <HintedLabel
            id={ids.label_smoothing}
            label={i18n.t('ui.training.SettingsForm.labelSmoothing')}
          >
            {i18n.t('ui.training.SettingsForm.shareOfTheTargetSpread')}
          </HintedLabel>
          <NumberInput
            id={ids.label_smoothing}
            value={tuned.label_smoothing ? '' : values.label_smoothing}
            placeholder={tuned.label_smoothing ? searchedText() : undefined}
            disabled={tuned.label_smoothing}
            onChange={(value) => update({ label_smoothing: value })}
            min={LIMITS.label_smoothing.min}
            max={LIMITS.label_smoothing.max}
            step={0.05}
            decimalScale={3}
            allowNegative={false}
            error={errors.label_smoothing}
          />
        </div>
      )}
      {uses('huber_delta') && (
        <div>
          <HintedLabel id={ids.huber_delta} label={i18n.t('ui.training.SettingsForm.huberDelta')}>
            {i18n.t('ui.training.SettingsForm.whereTheLossTurnsFrom')}
          </HintedLabel>
          <NumberInput
            id={ids.huber_delta}
            value={tuned.huber_delta ? '' : values.huber_delta}
            placeholder={tuned.huber_delta ? searchedText() : undefined}
            disabled={tuned.huber_delta}
            onChange={(value) => update({ huber_delta: value })}
            min={LIMITS.huber_delta.min}
            max={LIMITS.huber_delta.max}
            step={0.25}
            decimalScale={3}
            allowNegative={false}
            error={errors.huber_delta}
          />
        </div>
      )}
    </SimpleGrid>
  );
}

/** What ranks the trials; only the metrics the data allows (all while that is not known). */
function MetricField({
  values,
  onChange,
  errors,
  metrics,
  where,
}: SearchFieldsProps & { where: string }) {
  useTranslation();
  const offered = metrics ?? METRICS;
  const data = offered.map((metric) => ({ value: metric.name, label: metric.label }));
  if (!offered.some((metric) => metric.name === values.metric)) {
    const chosen = METRICS.find((metric) => metric.name === values.metric);
    data.push({ value: values.metric, label: chosen?.label ?? values.metric });
  }
  return (
    <Select
      label={i18n.t('ui.training.SettingsForm.objectiveMetric')}
      description={i18n.t('ui.training.SettingsForm.ranksTheTrialsOn', {
        where,
        metricDirection: metricDirection(values.metric),
      })}
      data={data}
      value={values.metric}
      onChange={(value) => value && onChange({ ...values, metric: value as TuningMetric })}
      allowDeselect={false}
      error={errors.metric}
      w={{ base: '100%', xs: 260 }}
    />
  );
}

function SearchFields({ values, onChange, errors, metrics, task = null }: SearchFieldsProps) {
  useTranslation();
  const update = (changes: Partial<SettingsValues>) => onChange({ ...values, ...changes });
  const grid = values.search_method === 'grid';
  const trials = plannedTrials(values, task);
  const epochs = parseField(values.epochs);
  const requested = parseField(values.search_trials);

  let summary: string | null = null;
  if (trials !== null && typeof epochs === 'number' && epochs > 0) {
    const count = i18n.t('ui.training.format.trials', { count: trials });
    summary = i18n.t('ui.training.SettingsForm.ofEach', {
      count,
      epochs,
      value:
        epochs === 1
          ? i18n.t('ui.training.SettingsForm.epoch')
          : i18n.t('ui.training.SettingsForm.epochs'),
    });
    if (!grid && typeof requested === 'number' && trials < requested) {
      summary += i18n.t('ui.training.SettingsForm.onlyDifferentCombinations', { trials });
    }
  }

  return (
    <Paper withBorder p="sm" radius="sm" mt="xs">
      <Stack gap="sm">
        <Group gap="sm" align="flex-end" wrap="wrap">
          <Input.Wrapper label={i18n.t('ui.training.SettingsForm.method')}>
            <SegmentedControl
              display="flex"
              data={METHODS}
              value={values.search_method}
              onChange={(value) => update({ search_method: value as TuningMethod })}
            />
          </Input.Wrapper>
          {!grid && (
            <NumberInput
              label={i18n.t('ui.training.SettingsForm.trials')}
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

        <NeuralSearchRows values={values} onChange={onChange} errors={errors} task={task} />

        <MetricField
          values={values}
          onChange={onChange}
          errors={errors}
          metrics={metrics}
          where={i18n.t('ui.training.SettingsForm.whereValidationData')}
        />

        {errors.search && (
          <Text size="sm" c="red" role="alert">
            {errors.search}
          </Text>
        )}
        <Text size="xs" c="dimmed">
          {i18n.t('ui.training.SettingsForm.allTrialsUseTheSame', {
            value: summary && `${summary} `,
            value2: grid
              ? i18n.t('ui.training.SettingsForm.everyCombinationOfTheChosen')
              : i18n.t('ui.training.SettingsForm.learningRatesAreDrawnEvenly'),
          })}
        </Text>
      </Stack>
    </Paper>
  );
}

/** The training settings a search can vary, each with its values. */
function NeuralSearchRows({
  values,
  onChange,
  errors,
  neural = false,
  task = null,
}: Omit<SearchFieldsProps, 'metrics'> & {
  /** Those of a hybrid network's neural layers (beside the parameters of its model). */
  neural?: boolean;
}) {
  useTranslation();
  const update = (changes: Partial<SettingsValues>) => onChange({ ...values, ...changes });
  const numbers = (list: string[]) => list.map(Number);
  const grid = values.search_method === 'grid';
  const what = (setting: string) =>
    neural ? i18n.t('ui.training.SettingsForm.whatNeural', { setting }) : setting;
  return (
    <>
      <SearchRow
        label={i18n.t('ui.training.SettingsForm.learningRate')}
        what={what(i18n.t('ui.training.SettingsForm.whatLearningRate'))}
        checked={values.tune_learning_rate}
        onToggle={(checked) => update({ tune_learning_rate: checked })}
        fixed={formatField('learning_rate', values.learning_rate)}
      >
        {grid ? (
          <MultiSelect
            aria-label={i18n.t('ui.training.SettingsForm.learningRatesToTry')}
            data={LR_OPTIONS}
            value={values.lr_values.map(String)}
            onChange={(list) => update({ lr_values: numbers(list) })}
            error={errors.lr_values}
          />
        ) : (
          <Group gap="xs" wrap="nowrap" align="flex-start">
            <NumberInput
              aria-label={i18n.t('ui.training.SettingsForm.smallestLearningRate')}
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
              {i18n.t('ui.training.SettingsForm.to')}
            </Text>
            <NumberInput
              aria-label={i18n.t('ui.training.SettingsForm.largestLearningRate')}
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
        label={i18n.t('ui.training.SettingsForm.batchSize')}
        what={what(i18n.t('ui.training.SettingsForm.whatBatchSize'))}
        checked={values.tune_batch_size}
        onToggle={(checked) => update({ tune_batch_size: checked })}
        fixed={formatField('batch_size', values.batch_size)}
      >
        <MultiSelect
          aria-label={i18n.t('ui.training.SettingsForm.batchSizesToTry')}
          data={BATCH_SIZE_OPTIONS}
          value={values.batch_sizes.map(String)}
          onChange={(list) => update({ batch_sizes: numbers(list) })}
          error={errors.batch_sizes}
        />
      </SearchRow>
      <SearchRow
        label={i18n.t('ui.training.SettingsForm.optimizer')}
        what={what(i18n.t('ui.training.SettingsForm.whatOptimizer'))}
        checked={values.tune_optimizer}
        onToggle={(checked) => update({ tune_optimizer: checked })}
        fixed={formatSetting('optimizer', values.optimizer)}
      >
        <MultiSelect
          aria-label={i18n.t('ui.training.SettingsForm.optimizersToTry')}
          data={OPTIMIZERS}
          value={values.optimizers}
          onChange={(list) => update({ optimizers: list as Optimizer[] })}
          error={errors.optimizers}
        />
      </SearchRow>
      <SearchRow
        label={i18n.t('ui.training.SettingsForm.weightDecay')}
        what={what(i18n.t('ui.training.SettingsForm.whatWeightDecay'))}
        checked={values.tune_weight_decay}
        onToggle={(checked) => update({ tune_weight_decay: checked })}
        fixed={formatField('weight_decay', values.weight_decay)}
      >
        <MultiSelect
          aria-label={i18n.t('ui.training.SettingsForm.weightDecaysToTry')}
          data={WEIGHT_DECAY_OPTIONS}
          value={values.weight_decays.map(String)}
          onChange={(list) => update({ weight_decays: numbers(list) })}
          error={errors.weight_decays}
        />
      </SearchRow>
      {task && (
        <SearchRow
          label={i18n.t('ui.training.SettingsForm.loss')}
          what={what(i18n.t('ui.training.SettingsForm.whatLoss'))}
          checked={values.tune_loss}
          onToggle={(checked) => update({ tune_loss: checked })}
          fixed={
            LOSS_OPTIONS[task].find((option) => option.value === effectiveLoss(values, task))
              ?.label ?? '–'
          }
        >
          <MultiSelect
            aria-label={i18n.t('ui.training.SettingsForm.lossesToTry')}
            data={LOSS_OPTIONS[task].map(({ value, label }) => ({ value, label }))}
            value={LOSS_OPTIONS[task]
              .map((option) => option.value)
              .filter((name) => values.losses.includes(name))}
            onChange={(list) => update({ losses: list as LossName[] })}
            error={errors.losses}
          />
        </SearchRow>
      )}
      {lossUses(values, task, 'focal_gamma') && (
        <SearchRow
          label={i18n.t('ui.training.SettingsForm.focalGamma')}
          what={what(i18n.t('ui.training.SettingsForm.whatFocalGamma'))}
          checked={values.tune_focal_gamma}
          onToggle={(checked) => update({ tune_focal_gamma: checked })}
          fixed={formatNumber(values.focal_gamma)}
        >
          <MultiSelect
            aria-label={i18n.t('ui.training.SettingsForm.focalGammasToTry')}
            data={FOCAL_GAMMA_OPTIONS}
            value={values.focal_gammas.map(String)}
            onChange={(list) => update({ focal_gammas: numbers(list) })}
            error={errors.focal_gammas}
          />
        </SearchRow>
      )}
      {lossUses(values, task, 'label_smoothing') && (
        <SearchRow
          label={i18n.t('ui.training.SettingsForm.labelSmoothing')}
          what={what(i18n.t('ui.training.SettingsForm.whatLabelSmoothing'))}
          checked={values.tune_label_smoothing}
          onToggle={(checked) => update({ tune_label_smoothing: checked })}
          fixed={formatNumber(values.label_smoothing)}
        >
          <MultiSelect
            aria-label={i18n.t('ui.training.SettingsForm.labelSmoothingsToTry')}
            data={LABEL_SMOOTHING_OPTIONS}
            value={values.label_smoothings.map(String)}
            onChange={(list) => update({ label_smoothings: numbers(list) })}
            error={errors.label_smoothings}
          />
        </SearchRow>
      )}
      {lossUses(values, task, 'huber_delta') && (
        <SearchRow
          label={i18n.t('ui.training.SettingsForm.huberDelta')}
          what={what(i18n.t('ui.training.SettingsForm.whatHuberDelta'))}
          checked={values.tune_huber_delta}
          onToggle={(checked) => update({ tune_huber_delta: checked })}
          fixed={formatNumber(values.huber_delta)}
        >
          <MultiSelect
            aria-label={i18n.t('ui.training.SettingsForm.huberDeltasToTry')}
            data={HUBER_DELTA_OPTIONS}
            value={values.huber_deltas.map(String)}
            onChange={(list) => update({ huber_deltas: numbers(list) })}
            error={errors.huber_deltas}
          />
        </SearchRow>
      )}
    </>
  );
}

/** A number field as text, "–" while it is empty or not a number. */
function formatNumber(field: string | number) {
  const value = parseField(field);
  return typeof value === 'number' ? String(value) : '–';
}

/** "10 epochs". */
function epochCount(values: SettingsValues): string {
  const epochs = parseField(values.epochs);
  return typeof epochs === 'number' && epochs > 0
    ? i18n.t('ui.training.SettingsForm.epochCount', { count: epochs })
    : i18n.t('ui.training.SettingsForm.itsEpochs');
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
  what,
}: {
  label: string;
  checked: boolean;
  onToggle: (checked: boolean) => void;
  /** The value used when the setting is not searched. */
  fixed: string;
  children: React.ReactNode;
  /** What is searched, for the checkbox's accessible name (default: the label). */
  what?: string;
}) {
  useTranslation();
  return (
    // On phones the values move below the checkbox, where they have the full width.
    <Group align="flex-start" gap="sm" wrap="wrap">
      <Checkbox
        label={label}
        aria-label={i18n.t('ui.training.SettingsForm.search', {
          value: what ?? (currentLanguage() === 'en' ? label.toLowerCase() : label),
        })}
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
            {i18n.t('ui.training.SettingsForm.fixed', { fixed })}
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
  get GaussianProcessRegressor() {
    return i18n.t('ui.training.SettingsForm.theFitOptimisesTheKernel');
  },
  get GaussianProcessClassifier() {
    return i18n.t('ui.training.SettingsForm.theFitOptimisesTheKernel2');
  },
};

const RANGE_MODES = [
  {
    value: 'range',
    get label() {
      return i18n.t('ui.training.SettingsForm.range');
    },
  },
  {
    value: 'list',
    get label() {
      return i18n.t('ui.training.SettingsForm.list');
    },
  },
];
const SCALES = [
  {
    value: 'log',
    get label() {
      return i18n.t('ui.training.SettingsForm.log');
    },
  },
  {
    value: 'linear',
    get label() {
      return i18n.t('ui.training.SettingsForm.linear');
    },
  },
];

interface ClassicalSearchFieldsProps extends SearchFieldsProps {
  /** The type of the model brick, e.g. "SVC". */
  model: string;
  /** False: no validation rows, so the trials are always cross-validated. */
  validates: boolean;
  /**
   * A hybrid network: its search can also vary the training settings of the neural layers
   * (not with `frozen` features). `model`: the classical model, e.g. "Support vector
   * classification".
   */
  hybrid?: { model: string; frozen: boolean } | null;
}

function ClassicalSearchFields({
  values,
  onChange,
  errors,
  model,
  validates,
  metrics,
  hybrid = null,
  task = null,
}: ClassicalSearchFieldsProps) {
  useTranslation();
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
  const neural = hybrid !== null && searchesNeural(values);
  const trials = hybrid
    ? plannedHybridTrials(values, model, task)
    : plannedClassicalTrials(values, model);
  const requested = parseField(values.search_trials);
  // The neural settings are compared on the validation rows (their features of the training
  // rows were learned from those rows).
  const crossValidates = !neural && (values.cv_enabled || !validates);
  const folds = parseField(values.cv_folds);
  const foldCount =
    values.cv_enabled && typeof folds === 'number' && Number.isInteger(folds)
      ? folds
      : FOLDS.default;

  let summary: string | null = null;
  if (trials !== null) {
    const count = i18n.t('ui.training.format.trials', { count: trials });
    summary = neural
      ? i18n.t('ui.training.SettingsForm.trialsLearnTheFeatures', {
          count,
          epochs: epochCount(values),
        })
      : crossValidates
        ? i18n.t('ui.training.SettingsForm.trialsTimesFolds', { count, folds: foldCount })
        : i18n.t('ui.training.SettingsForm.trialsEachFitted', { count });
    if (!grid && typeof requested === 'number' && trials < requested) {
      summary += i18n.t('ui.training.SettingsForm.onlyDifferentCombinations', { trials });
    }
  }

  return (
    <Paper withBorder p="sm" radius="sm" mt="xs">
      <Stack gap="sm">
        <Group gap="sm" align="flex-end" wrap="wrap">
          <Input.Wrapper label={i18n.t('ui.training.SettingsForm.method')}>
            <SegmentedControl
              display="flex"
              data={METHODS}
              value={values.search_method}
              onChange={(value) => update({ search_method: value as TuningMethod })}
            />
          </Input.Wrapper>
          {!grid && (
            <NumberInput
              label={i18n.t('ui.training.SettingsForm.trials')}
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

        {hybrid && (
          <>
            <Text size="sm" fw={600}>
              {i18n.t('ui.training.SettingsForm.neuralLayers')}
            </Text>
            <Checkbox
              label={i18n.t('ui.training.SettingsForm.searchTheTrainingSettingsOf')}
              description={
                hybrid.frozen
                  ? i18n.t('ui.training.SettingsForm.notWithFrozenFeaturesThey')
                  : i18n.t('ui.training.SettingsForm.everyTrialLearnsTheFeatures')
              }
              disabled={hybrid.frozen}
              checked={neural}
              onChange={(event) => update({ search_neural: event.currentTarget.checked })}
            />
            {neural && (
              <NeuralSearchRows
                values={values}
                onChange={onChange}
                errors={errors}
                neural
                task={task}
              />
            )}
            <Text size="sm" fw={600}>
              {hybrid.model}
            </Text>
          </>
        )}

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
          where={
            crossValidates
              ? i18n.t('ui.training.SettingsForm.whereCrossValidation')
              : i18n.t('ui.training.SettingsForm.whereValidationRows')
          }
        />

        {neural ? (
          <Text size="xs" c={validates ? 'dimmed' : 'red'}>
            {i18n.t('ui.training.SettingsForm.theTrialsAreComparedOn')}
          </Text>
        ) : validates ? (
          <Group gap="sm" align="flex-start" wrap="wrap">
            <Checkbox
              label={i18n.t('ui.training.SettingsForm.crossValidateOnTheTraining')}
              description={i18n.t('ui.training.SettingsForm.insteadOfTheValidationRows')}
              checked={values.cv_enabled}
              onChange={(event) => update({ cv_enabled: event.currentTarget.checked })}
              style={{ flex: 1, minWidth: 220 }}
            />
            {values.cv_enabled && (
              <NumberInput
                label={i18n.t('ui.training.SettingsForm.folds')}
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
            {i18n.t('ui.training.SettingsForm.noValidationRowsTheTrials', {
              default: FOLDS.default,
            })}
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
            ? i18n.t('ui.training.SettingsForm.everyCombinationOfTheListed')
            : i18n.t('ui.training.SettingsForm.rangesAreDrawnEvenlyOn')}{' '}
          {neural
            ? i18n.t('ui.training.SettingsForm.parametersThatAreNotTicked')
            : i18n.t('ui.training.SettingsForm.theBestTrialIsMeasured')}
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
  useTranslation();
  const choice = item.kind === 'choice' || item.kind === 'bool';
  const range = !grid && !choice && state.mode === 'range';
  let editor: React.ReactNode;
  if (choice) {
    editor = (
      <MultiSelect
        aria-label={i18n.t('ui.training.SettingsForm.toTry', { label: item.label })}
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
            aria-label={i18n.t('ui.training.SettingsForm.rangeOrList', { label: item.label })}
            style={{ alignSelf: 'flex-start' }}
          />
        )}
        {range ? (
          <Group gap="xs" wrap="nowrap" align="flex-start">
            <NumberInput
              aria-label={i18n.t('ui.training.SettingsForm.smallest', {
                label: item.label.toLowerCase(),
              })}
              value={state.min}
              onChange={(min) => onChange({ min })}
              allowNegative={false}
              allowDecimal={item.kind !== 'int'}
              decimalScale={12}
              error={error}
              style={{ flex: 1 }}
            />
            <Text size="sm" mt={7}>
              {i18n.t('ui.training.SettingsForm.to')}
            </Text>
            <NumberInput
              aria-label={i18n.t('ui.training.SettingsForm.largest', {
                label: item.label.toLowerCase(),
              })}
              value={state.max}
              onChange={(max) => onChange({ max })}
              allowNegative={false}
              allowDecimal={item.kind !== 'int'}
              decimalScale={12}
              style={{ flex: 1 }}
            />
            <Select
              aria-label={i18n.t('ui.training.SettingsForm.scaleOfThe', {
                label: item.label.toLowerCase(),
              })}
              data={SCALES}
              value={state.scale}
              onChange={(scale) => scale && onChange({ scale: scale as ParamState['scale'] })}
              allowDeselect={false}
              w={90}
            />
          </Group>
        ) : (
          <TextInput
            aria-label={i18n.t('ui.training.SettingsForm.toTry', { label: item.label })}
            placeholder={
              item.none ? i18n.t('ui.training.SettingsForm.eGNone36') : 'e.g. 0.1, 1, 10'
            }
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
      fixed={i18n.t('ui.training.SettingsForm.theValueOfTheBrick')}
    >
      {editor}
    </SearchRow>
  );
}
