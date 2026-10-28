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
import type { Optimizer, TrainingDevice, TuningMethod } from '@/api';
import { formatSetting } from './format';
import {
  BATCH_SIZE_PRESETS,
  LIMITS,
  LR_PRESETS,
  OPTIMIZERS,
  parseField,
  plannedTrials,
  tunedSettings,
  WEIGHT_DECAY_PRESETS,
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
   * Built-in datasets come with official test images: without a test set they validate on
   * them, with one they are the test set and validation data comes from the training images.
   */
  builtin: boolean;
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
}: SettingsFormProps) {
  const [advanced, { toggle }] = useDisclosure(false);
  const set =
    <K extends keyof SettingsValues>(key: K) =>
    (value: SettingsValues[K]) =>
      onChange({ ...values, [key]: value });
  // Keep unfinished input ("0.00") as typed; converting it would reset the field.
  const setNumber = (key: keyof SettingsValues) => (value: number | string) =>
    onChange({ ...values, [key]: value });
  const tuned = tunedSettings(values);

  const automatic = devices.find((device) => device.id === defaultDevice)?.label;
  const deviceOptions = [
    { value: 'auto', label: automatic ? `Automatic: ${automatic}` : 'Automatic' },
    ...devices.map((device) => ({ value: device.id, label: device.label })),
  ];
  const device = deviceOptions.some((option) => option.value === values.device)
    ? values.device
    : 'auto';
  const showValSplit = !builtin || values.test_enabled;

  return (
    <Stack gap="sm">
      <TextInput
        label="Name"
        placeholder={namePlaceholder}
        value={name}
        onChange={(event) => onNameChange(event.currentTarget.value)}
        maxLength={120}
      />
      <SimpleGrid cols={{ base: 1, xs: 2 }} spacing="sm" verticalSpacing="sm">
        <NumberInput
          label="Epochs"
          description={values.search ? 'Per trial of the search' : 'Passes over the training data'}
          value={values.epochs}
          onChange={setNumber('epochs')}
          min={LIMITS.epochs.min}
          max={LIMITS.epochs.max}
          allowDecimal={false}
          allowNegative={false}
          error={errors.epochs}
        />
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
        {showValSplit && (
          <NumberInput
            label="Validation split"
            value={values.val_split}
            onChange={setNumber('val_split')}
            min={LIMITS.val_split.min}
            max={LIMITS.val_split.max}
            step={5}
            suffix="%"
            allowNegative={false}
            decimalScale={1}
            error={errors.val_split}
          />
        )}
      </SimpleGrid>

      <div>
        <Switch
          label={builtin ? 'Test on the official test images' : 'Hold out a test set'}
          description={
            builtin
              ? 'The final model is measured once on the official test images. Validation data then comes from the training images.'
              : 'Never trained on and not used to compare settings: the final model is measured on it once, after training.'
          }
          checked={values.test_enabled}
          onChange={(event) => set('test_enabled')(event.currentTarget.checked)}
        />
        {values.test_enabled && !builtin && (
          <SimpleGrid cols={{ base: 1, xs: 2 }} spacing="sm" mt="xs">
            <NumberInput
              label="Test split"
              value={values.test_split}
              onChange={setNumber('test_split')}
              min={LIMITS.test_split.min}
              max={LIMITS.test_split.max}
              step={5}
              suffix="%"
              allowNegative={false}
              decimalScale={1}
              error={errors.test_split}
            />
          </SimpleGrid>
        )}
      </div>

      <div>
        <Switch
          label="Hyperparameter search"
          description="Trains the network several times with different settings and keeps the one with the lowest validation loss."
          checked={values.search}
          onChange={(event) => set('search')(event.currentTarget.checked)}
        />
        {values.search && <SearchFields values={values} onChange={onChange} errors={errors} />}
      </div>

      <Button
        variant="subtle"
        color="gray"
        size="compact-sm"
        leftSection={advanced ? <IconChevronDown size={14} /> : <IconChevronRight size={14} />}
        onClick={toggle}
        aria-expanded={advanced}
        style={{ alignSelf: 'flex-start' }}
      >
        Advanced settings
      </Button>
      <Collapse expanded={advanced}>
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
          <NumberInput
            label="Max. samples"
            placeholder="All"
            value={values.max_samples}
            onChange={setNumber('max_samples')}
            min={LIMITS.max_samples.min}
            max={LIMITS.max_samples.max}
            allowDecimal={false}
            allowNegative={false}
            thousandSeparator=","
            error={errors.max_samples}
          />
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
        </SimpleGrid>
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
}

function SearchFields({ values, onChange, errors }: SearchFieldsProps) {
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
          All trials use the same data and starting weights; the lowest validation loss after the
          last epoch wins.
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
    <Group align="flex-start" gap="sm" wrap="nowrap">
      <Checkbox
        label={label}
        aria-label={`Search ${label.toLowerCase()}`}
        checked={checked}
        onChange={(event) => onToggle(event.currentTarget.checked)}
        w={120}
        mt={7}
        style={{ flexShrink: 0 }}
      />
      <Box style={{ flex: 1, minWidth: 0 }}>
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
