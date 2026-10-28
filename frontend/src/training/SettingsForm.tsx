import { IconChevronDown, IconChevronRight } from '@tabler/icons-react';
import { Button, Collapse, NumberInput, Select, SimpleGrid, Stack, TextInput } from '@mantine/core';
import { useDisclosure } from '@mantine/hooks';
import type { Optimizer, TrainingDevice } from '@/api';
import { LIMITS, OPTIMIZERS, type SettingsErrors, type SettingsValues } from './settings';

interface SettingsFormProps {
  values: SettingsValues;
  onChange: (values: SettingsValues) => void;
  errors: SettingsErrors;
  name: string;
  namePlaceholder: string;
  onNameChange: (name: string) => void;
  devices: TrainingDevice[];
  defaultDevice: string | null;
  /** Built-in datasets come with their own validation images. */
  showValSplit: boolean;
}

export function SettingsForm({
  values,
  onChange,
  errors,
  name,
  namePlaceholder,
  onNameChange,
  devices,
  defaultDevice,
  showValSplit,
}: SettingsFormProps) {
  const [advanced, { toggle }] = useDisclosure(false);
  const set =
    <K extends keyof SettingsValues>(key: K) =>
    (value: SettingsValues[K]) =>
      onChange({ ...values, [key]: value });
  // Keep unfinished input ("0.00") as typed; converting it would reset the field.
  const setNumber = (key: keyof SettingsValues) => (value: number | string) =>
    onChange({ ...values, [key]: value });

  const automatic = devices.find((device) => device.id === defaultDevice)?.label;
  const deviceOptions = [
    { value: 'auto', label: automatic ? `Automatic: ${automatic}` : 'Automatic' },
    ...devices.map((device) => ({ value: device.id, label: device.label })),
  ];
  const device = deviceOptions.some((option) => option.value === values.device)
    ? values.device
    : 'auto';

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
          description="Passes over the training data"
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
          value={values.batch_size}
          onChange={setNumber('batch_size')}
          min={LIMITS.batch_size.min}
          max={LIMITS.batch_size.max}
          allowDecimal={false}
          allowNegative={false}
          error={errors.batch_size}
        />
        <NumberInput
          label="Learning rate"
          value={values.learning_rate}
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
          value={values.optimizer}
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
      <Collapse in={advanced}>
        <SimpleGrid cols={{ base: 1, xs: 3 }} spacing="sm" verticalSpacing="sm">
          <NumberInput
            label="Weight decay"
            value={values.weight_decay}
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
