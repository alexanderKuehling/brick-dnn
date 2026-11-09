import { useState } from 'react';
import { IconCheck, IconTarget } from '@tabler/icons-react';
import {
  Anchor,
  Group,
  Input,
  NumberInput,
  Paper,
  SegmentedControl,
  Select,
  SimpleGrid,
  Stack,
  Switch,
  Text,
  Title,
  Tooltip,
  type ComboboxItem,
  type ComboboxLikeRenderOptionInput,
} from '@mantine/core';
import type { ColumnInfo, PrepSettings, PrepTask } from '@/api/types';
import { formatCount } from '../table/format';
import { KIND_LABELS, KindIcon } from '../table/kinds';
import { displayedSplit, MAX_SEED, percentToShare, shareToPercent, splitError } from './split';
import classes from './PipelineEditor.module.css';

interface TargetSplitCardProps {
  settings: PrepSettings;
  onSettingsChange: (settings: PrepSettings) => void;
  /** Columns at the end of the pipeline. */
  columns: ColumnInfo[];
  task: 'classification' | 'regression' | null;
  splitCounts: number[] | null;
  /** Shown, not changed (a preparation shared with the user to look at). */
  readOnly?: boolean;
  /** Adds a "Join another table" step; missing when no join can be added. */
  onAddJoin?: () => void;
  /** Why no join can be added (e.g. no other table imported yet). */
  joinUnavailable?: string | null;
}

const TASKS: { value: PrepTask; label: string }[] = [
  { value: 'auto', label: 'Auto' },
  { value: 'classification', label: 'Classification' },
  { value: 'regression', label: 'Regression' },
];

const TASK_NAMES = { classification: 'classification', regression: 'regression' } as const;

type ShareKey = 'val' | 'test';

/** Target column, task and the train / validation / test split. */
export function TargetSplitCard({
  settings,
  onSettingsChange,
  columns,
  task,
  splitCounts,
  readOnly = false,
  onAddJoin,
  joinUnavailable = null,
}: TargetSplitCardProps) {
  const { split } = settings;
  const kinds = new Map(columns.map((column) => [column.name, column.kind]));
  const target = settings.target;
  const targetMissing = target !== null && !kinds.has(target);
  const data = columns.map((column) => ({ value: column.name, label: column.name }));
  if (target && targetMissing) {
    data.unshift({ value: target, label: target });
  }

  // Share fields keep what is typed; only valid splits are sent to the page.
  const [draft, setDraft] = useState<Record<ShareKey, number | string>>({
    val: shareToPercent(split.val),
    test: shareToPercent(split.test),
  });
  const [synced, setSynced] = useState({ val: split.val, test: split.test });
  if (synced.val !== split.val || synced.test !== split.test) {
    setSynced({ val: split.val, test: split.test });
    setDraft({ val: shareToPercent(split.val), test: shareToPercent(split.test) });
  }
  const draftShare = (key: ShareKey) => {
    const raw = draft[key];
    return raw === '' ? Number.NaN : percentToShare(Number(raw));
  };
  const draftVal = draftShare('val');
  const draftTest = draft.test === '' ? 0 : draftShare('test');
  const errors = Number.isNaN(draftVal)
    ? { val: 'Enter a share.' }
    : Number.isNaN(draftTest)
      ? { test: 'Enter a share.' }
      : splitError(draftVal, draftTest);

  const setShare = (key: ShareKey) => (raw: number | string) => {
    const next = { ...draft, [key]: raw };
    setDraft(next);
    const val = next.val === '' ? Number.NaN : percentToShare(Number(next.val));
    const test = next.test === '' ? 0 : percentToShare(Number(next.test));
    if (!Number.isNaN(val) && !Number.isNaN(test) && splitError(val, test) === null) {
      setSynced({ val, test });
      onSettingsChange({ ...settings, split: { ...split, val, test } });
    }
  };

  const [seedDraft, setSeedDraft] = useState<number | string>(split.seed);
  const [seedSynced, setSeedSynced] = useState(split.seed);
  if (seedSynced !== split.seed) {
    setSeedSynced(split.seed);
    setSeedDraft(split.seed);
  }
  const seedValid = (raw: number | string) =>
    typeof raw === 'number' && Number.isInteger(raw) && raw >= 0 && raw <= MAX_SEED;

  const regression =
    settings.task === 'regression' || (settings.task === 'auto' && task === 'regression');
  // While a share field has an error, the bar shows the last valid split.
  const shownVal = errors ? split.val : draftVal;
  const shownTest = errors ? split.test : draftTest;
  const shown = displayedSplit(splitCounts, shownVal, shownTest);

  const renderOption = ({ option, checked }: ComboboxLikeRenderOptionInput<ComboboxItem>) => {
    const kind = kinds.get(option.value);
    return (
      <Group gap={8} wrap="nowrap" style={{ flex: 1, minWidth: 0 }}>
        {kind ? <KindIcon kind={kind} /> : <span style={{ width: 14 }} />}
        <Text size="sm" truncate style={{ flex: 1 }}>
          {option.label}
        </Text>
        {kind && (
          <Text size="xs" c="dimmed">
            {KIND_LABELS[kind]}
          </Text>
        )}
        {checked && <IconCheck size={14} aria-hidden />}
      </Group>
    );
  };

  let taskHint: string;
  if (settings.task === 'classification') {
    taskHint = 'Predicts a class for every row.';
  } else if (settings.task === 'regression') {
    taskHint = 'Predicts a number for every row.';
  } else if (!target) {
    taskHint = 'Choose a target to detect the task.';
  } else if (task !== null) {
    taskHint = `Detected: ${TASK_NAMES[task]}.`;
  } else {
    taskHint = targetMissing
      ? 'No task: the target column is missing.'
      : 'Detected once the preview has run.';
  }

  return (
    <Paper withBorder radius="md" p="sm" className={classes.settingsCard}>
      <Stack gap="sm">
        <Group gap={8} wrap="nowrap">
          <IconTarget size={18} aria-hidden color="var(--mantine-color-green-light-color)" />
          <Title order={3} size="h5">
            Target &amp; split
          </Title>
        </Group>

        <Select
          label="Target column"
          description={
            <>
              The column to predict; all other columns are features.
              {!readOnly && (
                <>
                  {' '}
                  Is it in another table?{' '}
                  {onAddJoin ? (
                    <>
                      <Anchor component="button" type="button" size="xs" onClick={onAddJoin}>
                        Add a join step
                      </Anchor>{' '}
                      to bring it in through a column both tables share, such as an id.
                    </>
                  ) : (
                    (joinUnavailable ??
                    'Bring it in with a “Join another table” step through a column both tables share.')
                  )}
                </>
              )}
            </>
          }
          placeholder="Choose a column"
          data={data}
          value={target}
          readOnly={readOnly}
          searchable={!readOnly}
          clearable={!readOnly}
          nothingFoundMessage="No matching column"
          renderOption={renderOption}
          leftSection={target && kinds.has(target) ? <KindIcon kind={kinds.get(target)!} /> : null}
          comboboxProps={{ withinPortal: true, shadow: 'md' }}
          error={
            targetMissing
              ? `“${target}” does not exist after the steps. Choose the target again.`
              : undefined
          }
          onChange={(next) => onSettingsChange({ ...settings, target: next })}
        />

        <Input.Wrapper label="Task" labelElement="div" description={taskHint}>
          <SegmentedControl
            fullWidth
            size="xs"
            mt={6}
            aria-label="Task"
            data={TASKS}
            value={settings.task}
            readOnly={readOnly}
            onChange={(next) => onSettingsChange({ ...settings, task: next as PrepTask })}
          />
        </Input.Wrapper>

        <SimpleGrid cols={2} spacing="sm">
          <NumberInput
            label="Validation"
            value={draft.val}
            onChange={setShare('val')}
            min={0}
            max={90}
            step={5}
            suffix=" %"
            readOnly={readOnly}
            hideControls={readOnly}
            decimalScale={1}
            allowNegative={false}
            clampBehavior="none"
            error={errors?.val}
          />
          <NumberInput
            label="Test"
            value={draft.test}
            onChange={setShare('test')}
            min={0}
            max={90}
            step={5}
            suffix=" %"
            readOnly={readOnly}
            hideControls={readOnly}
            decimalScale={1}
            allowNegative={false}
            clampBehavior="none"
            error={errors?.test}
          />
        </SimpleGrid>

        <SplitBar val={shownVal} test={shownTest} shown={shown} />

        <SimpleGrid cols={2} spacing="sm">
          <NumberInput
            label="Seed"
            description="Same seed, same split"
            value={seedDraft}
            min={0}
            max={MAX_SEED}
            allowDecimal={false}
            allowNegative={false}
            readOnly={readOnly}
            hideControls={readOnly}
            error={seedValid(seedDraft) ? undefined : 'Enter a whole number ≥ 0.'}
            onChange={(raw) => {
              setSeedDraft(raw);
              if (seedValid(raw)) {
                setSeedSynced(raw as number);
                onSettingsChange({ ...settings, split: { ...split, seed: raw as number } });
              }
            }}
          />
          <Tooltip
            label="Stratifying needs a classification target."
            disabled={!regression}
            multiline
            w={220}
          >
            <div className={classes.stratify}>
              <Switch
                label="Stratify"
                description="Same class shares in every part"
                checked={split.stratify && !regression}
                disabled={regression || readOnly}
                onChange={(event) =>
                  onSettingsChange({
                    ...settings,
                    split: { ...split, stratify: event.currentTarget.checked },
                  })
                }
              />
            </div>
          </Tooltip>
        </SimpleGrid>
      </Stack>
    </Paper>
  );
}

interface SplitBarProps {
  val: number;
  test: number;
  shown: ReturnType<typeof displayedSplit>;
}

/** The three parts as a bar with their shares and row counts. */
function SplitBar({ val, test, shown }: SplitBarProps) {
  const parts = [
    { key: 'train', label: 'Train', share: Math.max(0, 1 - val - test) },
    { key: 'val', label: 'Validation', share: val },
    { key: 'test', label: 'Test', share: test },
  ];
  const counts = shown?.counts;
  const prefix = shown?.estimated ? '≈ ' : '';
  return (
    <div>
      <div className={classes.splitBar} aria-hidden>
        {parts.map((part) => (
          <span
            key={part.key}
            className={classes.splitPart}
            data-part={part.key}
            style={{ flexGrow: part.share }}
          />
        ))}
      </div>
      <Group gap="md" mt={6} wrap="wrap">
        {parts.map((part, index) => (
          <Text key={part.key} size="xs" c="dimmed">
            <span className={classes.splitDot} data-part={part.key} aria-hidden />
            {part.label} {shareToPercent(part.share)}&nbsp;%
            {counts ? ` · ${prefix}${formatCount(counts[index])} rows` : ''}
          </Text>
        ))}
      </Group>
    </div>
  );
}
