import { useId, useState, type ReactNode } from 'react';
import { IconCheck, IconChevronRight, IconTarget } from '@tabler/icons-react';
import {
  Anchor,
  Collapse,
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
  UnstyledButton,
  type ComboboxItem,
  type ComboboxLikeRenderOptionInput,
} from '@mantine/core';
import type { ColumnInfo, PrepSettings, PrepTask, SplitInfo } from '@/api/types';
import { HintedLabel } from '@/components/InfoHint';
import { i18n, Trans, useTranslation } from '@/i18n';
import { formatCount } from '../table/format';
import { KIND_LABELS, KindIcon } from '../table/kinds';
import {
  displayedSplit,
  heldOutHint,
  MAX_SEED,
  percentToShare,
  shareToPercent,
  splitError,
} from './split';
import classes from './PipelineEditor.module.css';

interface TargetSplitCardProps {
  settings: PrepSettings;
  onSettingsChange: (settings: PrepSettings) => void;
  /** Columns at the end of the pipeline. */
  columns: ColumnInfo[];
  task: 'classification' | 'regression' | null;
  splitCounts: number[] | null;
  /**
   * Where the last preview assigned the split: the columns there (the group column must be one
   * of them) and how the groups were split.
   */
  splitInfo?: SplitInfo | null;
  /** Shown, not changed (a preparation shared with the user to look at). */
  readOnly?: boolean;
  /** Adds a "Join another table" step; missing when no join can be added. */
  onAddJoin?: () => void;
  /** Why no join can be added (e.g. no other table imported yet). */
  joinUnavailable?: string | null;
}

const TASKS: { value: PrepTask; label: string }[] = [
  {
    value: 'auto',
    get label() {
      return i18n.t('ui.data.TargetSplitCard.auto');
    },
  },
  {
    value: 'classification',
    get label() {
      return i18n.t('ui.data.TargetSplitCard.classification');
    },
  },
  {
    value: 'regression',
    get label() {
      return i18n.t('ui.data.TargetSplitCard.regression');
    },
  },
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
  splitInfo = null,
  readOnly = false,
  onAddJoin,
  joinUnavailable = null,
}: TargetSplitCardProps) {
  useTranslation();
  const { split } = settings;
  const kinds = new Map(columns.map((column) => [column.name, column.kind]));
  const target = settings.target;
  const targetMissing = target !== null && !kinds.has(target);
  const data = columns.map((column) => ({ value: column.name, label: column.name }));
  // Folded to a one-line summary once a target is chosen; it opens by itself when the target
  // is missing (none chosen yet, or a step removed it).
  const bodyId = useId();
  const targetId = useId();
  const moreId = useId();
  const needsTarget = target === null || targetMissing;
  const [open, setOpen] = useState(needsTarget);
  const [neededTarget, setNeededTarget] = useState(needsTarget);
  if (neededTarget !== needsTarget) {
    setNeededTarget(needsTarget);
    if (needsTarget) {
      setOpen(true);
    }
  }
  // Seed, stratification and groups: folded unless rows are kept together by a group.
  const [moreOpen, setMoreOpen] = useState(Boolean(settings.split.group_by));
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
    ? { val: i18n.t('ui.data.TargetSplitCard.enterAShare') }
    : Number.isNaN(draftTest)
      ? { test: i18n.t('ui.data.TargetSplitCard.enterAShare') }
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
  const hint = heldOutHint(shownVal, shownTest);

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

  // Only "Auto" needs a word: what was detected (a chosen task is shown by the control).
  let taskHint: string | undefined;
  if (settings.task !== 'auto') {
    taskHint = undefined;
  } else if (!target) {
    taskHint = i18n.t('ui.data.TargetSplitCard.chooseATargetToDetect');
  } else if (task !== null) {
    taskHint = i18n.t('ui.data.TargetSplitCard.detected', { value: TASK_NAMES[task] });
  } else {
    taskHint = targetMissing
      ? i18n.t('ui.data.TargetSplitCard.noTaskTheTargetColumn')
      : i18n.t('ui.data.TargetSplitCard.detectedOnceThePreviewHas');
  }

  const resolvedTask =
    settings.task !== 'auto' ? settings.task : target && !targetMissing ? task : null;
  const summaryText = [
    target ?? i18n.t('ui.data.TargetSplitCard.noTargetYet'),
    resolvedTask === 'classification'
      ? i18n.t('ui.data.TargetSplitCard.classification')
      : resolvedTask === 'regression'
        ? i18n.t('ui.data.TargetSplitCard.regression')
        : null,
    i18n.t('ui.data.TargetSplitCard.splitSummary', {
      train: shareToPercent(Math.max(0, 1 - split.val - split.test)),
      val: shareToPercent(split.val),
      test: shareToPercent(split.test),
    }),
    split.group_by
      ? i18n.t('ui.data.TargetSplitCard.groupsSummary', { column: split.group_by })
      : null,
  ]
    .filter(Boolean)
    .join(' · ');

  return (
    <Paper withBorder radius="md" p="sm" className={classes.settingsCard}>
      <Title order={3} size="h5">
        <UnstyledButton
          className={classes.cardToggle}
          onClick={() => setOpen((value) => !value)}
          aria-expanded={open}
          aria-controls={bodyId}
        >
          <IconChevronRight
            size={14}
            className={classes.chevron}
            data-expanded={open || undefined}
            aria-hidden
          />
          <IconTarget size={18} aria-hidden color="var(--mantine-color-green-light-color)" />
          <span>{i18n.t('ui.data.TargetSplitCard.targetSplit')}</span>
        </UnstyledButton>
      </Title>
      {!open && (
        <Text size="sm" c="dimmed" className={classes.cardSummary} data-testid="target-summary">
          {summaryText}
        </Text>
      )}
      <Collapse expanded={open} id={bodyId}>
        <Stack gap="sm" pt="sm">
          <div>
            <HintedLabel id={targetId} label={i18n.t('ui.data.TargetSplitCard.targetColumn')}>
              {i18n.t('ui.data.TargetSplitCard.theColumnToPredictAll')}
              {!readOnly && (
                <>
                  {' '}
                  {i18n.t('ui.data.TargetSplitCard.isItInAnotherTable')}{' '}
                  {onAddJoin ? (
                    <Trans
                      i18nKey="ui.data.TargetSplitCard.addAJoinStepTo"
                      components={[
                        <Anchor component="button" type="button" size="sm" onClick={onAddJoin} />,
                      ]}
                    />
                  ) : (
                    (joinUnavailable ?? i18n.t('ui.data.TargetSplitCard.bringItInWithA'))
                  )}
                </>
              )}
            </HintedLabel>
            <Select
              id={targetId}
              placeholder={i18n.t('ui.data.TargetSplitCard.chooseAColumn')}
              data={data}
              value={target}
              readOnly={readOnly}
              searchable={!readOnly}
              clearable={!readOnly}
              nothingFoundMessage={i18n.t('ui.data.TargetSplitCard.noMatchingColumn')}
              renderOption={renderOption}
              leftSection={
                target && kinds.has(target) ? <KindIcon kind={kinds.get(target)!} /> : null
              }
              comboboxProps={{ withinPortal: true, shadow: 'md' }}
              error={
                targetMissing
                  ? i18n.t('ui.data.TargetSplitCard.doesNotExistAfterThe', { target })
                  : undefined
              }
              onChange={(next) => onSettingsChange({ ...settings, target: next })}
            />
          </div>

          <Input.Wrapper
            label={i18n.t('ui.data.TargetSplitCard.task')}
            labelElement="div"
            description={taskHint}
          >
            <SegmentedControl
              fullWidth
              size="xs"
              mt={6}
              aria-label={i18n.t('ui.data.TargetSplitCard.task')}
              data={TASKS}
              value={settings.task}
              readOnly={readOnly}
              onChange={(next) => onSettingsChange({ ...settings, task: next as PrepTask })}
            />
          </Input.Wrapper>

          <SimpleGrid cols={2} spacing="sm">
            <NumberInput
              label={i18n.t('ui.data.TargetSplitCard.validation')}
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
              label={i18n.t('ui.data.TargetSplitCard.test')}
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
          {hint && (
            <Text size="xs" c="dimmed" data-testid="held-out-hint">
              {hint}
            </Text>
          )}

          <div>
            <UnstyledButton
              className={classes.moreToggle}
              onClick={() => setMoreOpen((value) => !value)}
              aria-expanded={moreOpen}
              aria-controls={moreId}
            >
              <IconChevronRight
                size={14}
                className={classes.chevron}
                data-expanded={moreOpen || undefined}
                aria-hidden
              />
              {i18n.t('ui.data.TargetSplitCard.moreSplitOptions')}
            </UnstyledButton>
            <Collapse expanded={moreOpen} id={moreId}>
              <Stack gap="sm" pt="xs">
                <SimpleGrid cols={2} spacing="sm">
                  <NumberInput
                    label={i18n.t('ui.data.TargetSplitCard.seed')}
                    value={seedDraft}
                    min={0}
                    max={MAX_SEED}
                    allowDecimal={false}
                    allowNegative={false}
                    readOnly={readOnly}
                    hideControls={readOnly}
                    error={
                      seedValid(seedDraft)
                        ? undefined
                        : i18n.t('ui.data.TargetSplitCard.enterAWholeNumber0')
                    }
                    onChange={(raw) => {
                      setSeedDraft(raw);
                      if (seedValid(raw)) {
                        setSeedSynced(raw as number);
                        onSettingsChange({ ...settings, split: { ...split, seed: raw as number } });
                      }
                    }}
                  />
                  <Tooltip
                    label={i18n.t('ui.data.TargetSplitCard.stratifyingNeedsAClassificationTarget')}
                    disabled={!regression}
                    multiline
                    w={220}
                  >
                    <div className={classes.stratify}>
                      <Switch
                        label={i18n.t('ui.data.TargetSplitCard.stratify')}
                        description={i18n.t('ui.data.TargetSplitCard.sameClassSharesInEvery')}
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

                <GroupSplitField
                  settings={settings}
                  onSettingsChange={onSettingsChange}
                  columns={splitInfo?.columns ?? columns}
                  splitInfo={splitInfo}
                  readOnly={readOnly}
                  renderOption={renderOption}
                />
              </Stack>
            </Collapse>
          </div>
        </Stack>
      </Collapse>
    </Paper>
  );
}

interface GroupSplitFieldProps {
  settings: PrepSettings;
  onSettingsChange: (settings: PrepSettings) => void;
  columns: ColumnInfo[];
  splitInfo: SplitInfo | null;
  readOnly: boolean;
  renderOption: (input: ComboboxLikeRenderOptionInput<ComboboxItem>) => ReactNode;
}

/** "Keep rows of a group together": the group column and how the groups were split. */
export function GroupSplitField({
  settings,
  onSettingsChange,
  columns,
  splitInfo,
  readOnly,
  renderOption,
}: GroupSplitFieldProps) {
  useTranslation();
  const { split } = settings;
  const groupBy = split.group_by ?? null;
  const fieldId = useId();
  const target = settings.target;
  const data = columns
    .filter((column) => column.name !== target)
    .map((column) => ({ value: column.name, label: column.name }));
  if (groupBy && !data.some((item) => item.value === groupBy)) {
    data.unshift({ value: groupBy, label: groupBy });
  }
  const groups = splitInfo && splitInfo.group_by === groupBy ? splitInfo.groups : null;
  const error = splitInfo && splitInfo.group_by === groupBy ? splitInfo.error : null;
  let summary: ReactNode = null;
  if (groupBy && groups) {
    summary = (
      <Text size="xs" c="dimmed" data-testid="group-split-summary">
        {i18n.t('ui.data.TargetSplitCard.withSeveralRowsTheLargest', {
          count: formatCount(groups.count),
          value:
            groups.count === 1
              ? i18n.t('ui.data.TargetSplitCard.group')
              : i18n.t('ui.data.TargetSplitCard.groups'),
          multi: formatCount(groups.multi),
          largest: formatCount(groups.largest),
          value2:
            groups.mixed === 0
              ? i18n.t('ui.data.TargetSplitCard.noGroupIsInTwo')
              : i18n.t('ui.data.TargetSplitCard.groupsAreSplit', { mixed: groups.mixed }),
          map: groups.per_part.map(formatCount).join(' / '),
        })}
      </Text>
    );
  }
  return (
    <Stack gap={4}>
      <div>
        <HintedLabel id={fieldId} label={i18n.t('ui.data.TargetSplitCard.keepRowsOfAGroup')}>
          {i18n.t('ui.data.TargetSplitCard.groupHint')}
        </HintedLabel>
        <Select
          id={fieldId}
          placeholder={i18n.t('ui.data.TargetSplitCard.offRowsAreSplitOne')}
          data={data}
          value={groupBy}
          readOnly={readOnly}
          searchable={!readOnly}
          clearable={!readOnly}
          nothingFoundMessage={i18n.t('ui.data.TargetSplitCard.noMatchingColumn')}
          renderOption={renderOption}
          comboboxProps={{ withinPortal: true, shadow: 'md' }}
          error={error ?? undefined}
          onChange={(next) =>
            onSettingsChange({ ...settings, split: { ...split, group_by: next ?? null } })
          }
        />
      </div>
      {summary}
    </Stack>
  );
}

interface SplitBarProps {
  val: number;
  test: number;
  shown: ReturnType<typeof displayedSplit>;
}

/** The three parts as a bar with their shares and row counts. */
function SplitBar({ val, test, shown }: SplitBarProps) {
  useTranslation();
  const parts = [
    {
      key: 'train',
      label: i18n.t('ui.data.TargetSplitCard.train'),
      share: Math.max(0, 1 - val - test),
    },
    { key: 'val', label: i18n.t('ui.data.TargetSplitCard.validation'), share: val },
    { key: 'test', label: i18n.t('ui.data.TargetSplitCard.test'), share: test },
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
            {part.key !== 'train' && part.share <= 0 ? (
              i18n.t('ui.data.TargetSplitCard.none', { label: part.label })
            ) : (
              <>
                {part.label} {shareToPercent(part.share)}&nbsp;%
                {counts
                  ? i18n.t('ui.data.TargetSplitCard.rows', {
                      prefix,
                      value: formatCount(counts[index]),
                    })
                  : ''}
              </>
            )}
          </Text>
        ))}
      </Group>
    </div>
  );
}
