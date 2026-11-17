import { useState } from 'react';
import {
  IconAlertCircle,
  IconAlertTriangle,
  IconBulb,
  IconCheck,
  IconChevronDown,
  IconChevronUp,
  IconCircleCheck,
  IconDatabaseExport,
  IconWand,
} from '@tabler/icons-react';
import {
  Alert,
  Badge,
  Button,
  Group,
  Paper,
  SimpleGrid,
  Stack,
  Text,
  ThemeIcon,
  Title,
  VisuallyHidden,
} from '@mantine/core';
import type { DataFix, NewStep, Readiness, ReadinessIssue, SplitSettings } from '@/api/types';
import { i18n, useTranslation } from '@/i18n';
import { usePrepAccess } from '../prepAccess';
import { formatCount } from '../table/format';
import classes from './ReadinessPanel.module.css';

export interface ReadinessPanelProps {
  readiness: Readiness;
  onAddStep: (step: NewStep) => void;
  /** Settings a fix changes besides adding its step (e.g. split by a new group column). */
  onChangeSplit?: (split: Partial<SplitSettings>) => void;
  onExport: () => void;
  exporting: boolean;
  /** A new preview is on its way: fixes wait for it (they could be outdated or added twice). */
  fixesDisabled?: boolean;
  /**
   * Why the data cannot be exported although `readiness` (maybe from an older preview) says it
   * is ready, e.g. "Step 2 “Clip values” failed."; null if nothing else blocks it.
   */
  exportBlocked?: string | null;
}

const CLASSES_SHOWN = 12;
/** Affected columns listed under an issue; the rest are counted. */
const COLUMNS_SHOWN = 6;

const TASK_LABELS = {
  get classification() {
    return i18n.t('ui.data.ReadinessPanel.classification');
  },
  get regression() {
    return i18n.t('ui.data.ReadinessPanel.regression');
  },
} as const;

type Severity = ReadinessIssue['severity'];

/** The lists of the panel: errors first; long lists of hints show their first items. */
const SECTIONS: { severity: Severity; title: string; shown: number }[] = [
  {
    severity: 'error',
    get title() {
      return i18n.t('ui.data.ReadinessPanel.mustBeFixed');
    },
    shown: Infinity,
  },
  {
    severity: 'warning',
    get title() {
      return i18n.t('ui.data.ReadinessPanel.worthALook');
    },
    shown: 8,
  },
  {
    severity: 'info',
    get title() {
      return i18n.t('ui.data.ReadinessPanel.suggestions');
    },
    shown: 3,
  },
];

const SEVERITY = {
  error: { color: 'red', Icon: IconAlertCircle, prefix: 'Error: ' },
  warning: { color: 'yellow', Icon: IconAlertTriangle, prefix: 'Warning: ' },
  info: { color: 'blue', Icon: IconBulb, prefix: 'Suggestion: ' },
} as const;

/** Short names of the checks (`ReadinessIssue.code`), shown as a badge. */
const CHECK_LABELS: Record<string, string> = {
  get target_missing() {
    return i18n.t('ui.data.ReadinessPanel.target');
  },
  get target_not_numeric() {
    return i18n.t('ui.data.ReadinessPanel.target');
  },
  get missing_values() {
    return i18n.t('ui.data.ReadinessPanel.missingValues');
  },
  get categorical_features() {
    return i18n.t('ui.data.ReadinessPanel.categories');
  },
  get high_cardinality() {
    return i18n.t('ui.data.ReadinessPanel.manyCategories');
  },
  get inconsistent_categories() {
    return i18n.t('ui.data.ReadinessPanel.spellings');
  },
  get numbers_as_text() {
    return i18n.t('ui.data.ReadinessPanel.numbersAsText');
  },
  get dates_as_text() {
    return i18n.t('ui.data.ReadinessPanel.datesAsText');
  },
  get text_features() {
    return i18n.t('ui.data.ReadinessPanel.freeText');
  },
  get date_features() {
    return i18n.t('ui.data.ReadinessPanel.dates');
  },
  get infinite_values() {
    return i18n.t('ui.data.ReadinessPanel.infiniteValues');
  },
  get leakage() {
    return i18n.t('ui.data.ReadinessPanel.targetLeakage');
  },
  get split_overlap() {
    return i18n.t('ui.data.ReadinessPanel.trainTestOverlap');
  },
  get duplicate_rows() {
    return i18n.t('ui.data.ReadinessPanel.duplicates');
  },
  get conflicting_labels() {
    return i18n.t('ui.data.ReadinessPanel.conflictingLabels');
  },
  get rare_classes() {
    return i18n.t('ui.data.ReadinessPanel.rareClasses');
  },
  get classes_not_in_split() {
    return i18n.t('ui.data.ReadinessPanel.split');
  },
  get imbalanced() {
    return i18n.t('ui.data.ReadinessPanel.imbalance');
  },
  id_like: 'ID column',
  get constant() {
    return i18n.t('ui.data.ReadinessPanel.constant');
  },
  get near_constant() {
    return i18n.t('ui.data.ReadinessPanel.nearConstant');
  },
  get redundant() {
    return i18n.t('ui.data.ReadinessPanel.redundant');
  },
  get no_partner() {
    return i18n.t('ui.data.ReadinessPanel.noStrongPartner');
  },
  get skewed() {
    return i18n.t('ui.data.ReadinessPanel.skewed');
  },
  get skewed_target() {
    return i18n.t('ui.data.ReadinessPanel.skewedTarget');
  },
  get scales() {
    return i18n.t('ui.data.ReadinessPanel.scales');
  },
  get few_rows() {
    return i18n.t('ui.data.ReadinessPanel.fewRows');
  },
  get many_features() {
    return i18n.t('ui.data.ReadinessPanel.manyFeatures');
  },
};

function fixKey(fix: DataFix): string {
  return JSON.stringify([fix.type, fix.params]);
}

function plural(count: number, word: string): string {
  return count === 1 ? `1 ${word}` : `${formatCount(count)} ${word}s`;
}

/** Can the prepared data be used for training? Summary, problems with fixes, export. */
export function ReadinessPanel({
  readiness,
  onAddStep,
  onChangeSplit,
  onExport,
  exporting,
  fixesDisabled = false,
  exportBlocked = null,
}: ReadinessPanelProps) {
  useTranslation();
  // Viewers of a shared preparation see the problems, but neither fix them nor export.
  const { readOnly } = usePrepAccess();
  // Fixes added since the last check; they disappear with the next preview.
  const [applied, setApplied] = useState<ReadonlySet<string>>(() => new Set());
  const [checked, setChecked] = useState(readiness);
  if (checked !== readiness) {
    setChecked(readiness);
    setApplied(new Set());
  }

  const ready = readiness.ready && !exportBlocked;
  const bySeverity = (severity: Severity) =>
    readiness.issues.filter((issue) => issue.severity === severity);
  const errors = bySeverity('error');
  const warnings = bySeverity('warning');
  const hints = bySeverity('info');
  const pendingFixes: DataFix[] = [];
  for (const issue of readOnly ? [] : errors) {
    if (issue.fix && !applied.has(fixKey(issue.fix))) {
      if (!pendingFixes.some((fix) => fixKey(fix) === fixKey(issue.fix!))) {
        pendingFixes.push(issue.fix);
      }
    }
  }

  const apply = (fixes: DataFix[]) => {
    const next = new Set(applied);
    for (const fix of fixes) {
      onAddStep({ type: fix.type, params: structuredClone(fix.params) });
      if (fix.settings?.split) {
        onChangeSplit?.(fix.settings.split);
      }
      next.add(fixKey(fix));
    }
    setApplied(next);
  };

  const split = readiness.split_counts;
  const stats: { label: string; value: string; hint?: string }[] = [
    { label: i18n.t('ui.data.ReadinessPanel.target'), value: readiness.target ?? 'Not set' },
    {
      label: i18n.t('ui.data.ReadinessPanel.task'),
      value: readiness.task ? TASK_LABELS[readiness.task] : '–',
    },
    {
      label: i18n.t('ui.data.ReadinessPanel.features'),
      value: formatCount(readiness.num_features),
    },
    {
      label: i18n.t('ui.data.ReadinessPanel.classes'),
      value: readiness.num_classes === null ? '–' : formatCount(readiness.num_classes),
    },
    { label: i18n.t('ui.data.ReadinessPanel.rows'), value: formatCount(readiness.rows) },
    {
      label: i18n.t('ui.data.ReadinessPanel.split'),
      value: split ? split.map(formatCount).join(' / ') : '–',
      hint: split ? i18n.t('ui.data.ReadinessPanel.trainValidationTest') : undefined,
    },
  ];
  const classNames = readiness.class_names ?? [];
  const notes = [
    warnings.length > 0 ? plural(warnings.length, 'warning') : '',
    hints.length > 0 ? plural(hints.length, 'suggestion') : '',
  ].filter(Boolean);
  const lists = { error: errors, warning: warnings, info: hints };

  return (
    <Stack gap="md" className={classes.root}>
      {ready ? (
        <Alert
          color="green"
          variant="light"
          icon={<IconCircleCheck size={20} />}
          title={i18n.t('ui.data.ReadinessPanel.readyForTraining')}
        >
          {notes.length > 0
            ? i18n.t('ui.data.ReadinessPanel.lookAtTheNotesBelow', { notes: notes.join(' and ') })
            : null}
        </Alert>
      ) : (
        <Alert
          color="red"
          variant="light"
          icon={<IconAlertCircle size={20} />}
          title={i18n.t('ui.data.ReadinessPanel.notReadyForTrainingYet')}
        >
          {errors.length === 0
            ? (exportBlocked ?? i18n.t('ui.data.ReadinessPanel.theDataCannotBeUsed'))
            : errors.length === 1
              ? i18n.t('ui.data.ReadinessPanel.text1ProblemMustBeFixed')
              : i18n.t('ui.data.ReadinessPanel.problemsMustBeFixedFirst', {
                  length: errors.length,
                })}
          {pendingFixes.length > 0 && i18n.t('ui.data.ReadinessPanel.eachFixAddsAStep')}
        </Alert>
      )}

      <SimpleGrid cols={{ base: 2, xs: 3 }} spacing="xs" verticalSpacing="xs">
        {stats.map((stat) => (
          <Paper key={stat.label} withBorder p="xs" radius="sm">
            <Text size="xs" c="dimmed">
              {stat.label}
            </Text>
            <Text fw={600} size="sm" truncate title={stat.value}>
              {stat.value}
            </Text>
            {stat.hint && (
              <Text size="xs" c="dimmed">
                {stat.hint}
              </Text>
            )}
          </Paper>
        ))}
      </SimpleGrid>

      {classNames.length > 0 && (
        <div>
          <Text size="xs" c="dimmed" mb={4}>
            {i18n.t('ui.data.ReadinessPanel.classOrderClassIndex0')}
          </Text>
          <Group gap={4}>
            {classNames.slice(0, CLASSES_SHOWN).map((name, index) => (
              <Badge key={`${index}-${name}`} variant="light" color="gray" tt="none">
                {index}: {name}
              </Badge>
            ))}
            {classNames.length > CLASSES_SHOWN && (
              <Text size="xs" c="dimmed">
                {i18n.t('ui.data.ReadinessPanel.more', {
                  value: classNames.length - CLASSES_SHOWN,
                })}
              </Text>
            )}
          </Group>
        </div>
      )}

      {SECTIONS.map(
        ({ severity, title, shown }) =>
          lists[severity].length > 0 && (
            <IssueList
              key={severity}
              title={title}
              issues={lists[severity]}
              shown={shown}
              applied={applied}
              disabled={fixesDisabled}
              onFix={readOnly ? undefined : (fix) => apply([fix])}
            />
          )
      )}
      {readiness.issues.length === 0 && (
        <Text size="sm" c="dimmed">
          {i18n.t('ui.data.ReadinessPanel.noProblemsFound')}
        </Text>
      )}

      <Group gap="sm" className={classes.actions}>
        {pendingFixes.length > 0 && (
          <Button
            variant="light"
            leftSection={<IconWand size={16} />}
            disabled={fixesDisabled}
            onClick={() => apply(pendingFixes)}
          >
            {i18n.t('ui.data.ReadinessPanel.fixAll', { length: pendingFixes.length })}
          </Button>
        )}
        {!readOnly && (
          <Button
            leftSection={<IconDatabaseExport size={16} />}
            onClick={onExport}
            loading={exporting}
            disabled={!ready}
          >
            {i18n.t('ui.data.ReadinessPanel.createTrainingDataset')}
          </Button>
        )}
      </Group>
      {!readOnly && (
        <Text size="xs" c="dimmed" mt={-8}>
          {ready
            ? i18n.t('ui.data.ReadinessPanel.savesThePreparedDataWith')
            : exportBlocked && errors.length > 0
              ? i18n.t('ui.data.ReadinessPanel.availableOnceItAndThe', { exportBlocked })
              : exportBlocked
                ? i18n.t('ui.data.ReadinessPanel.availableOnceItIsFixed', { exportBlocked })
                : i18n.t('ui.data.ReadinessPanel.availableOnceTheProblemsAbove')}
        </Text>
      )}
    </Stack>
  );
}

interface IssueListProps {
  title: string;
  issues: ReadinessIssue[];
  /** Issues shown before “Show more”. */
  shown: number;
  applied: ReadonlySet<string>;
  disabled: boolean;
  /** Without it, the fixes are not offered (read-only views). */
  onFix?: (fix: DataFix) => void;
}

function IssueList({ title, issues, shown, applied, disabled, onFix }: IssueListProps) {
  useTranslation();
  const [expanded, setExpanded] = useState(false);
  const hidden = expanded ? 0 : Math.max(issues.length - shown, 0);
  const visible = hidden > 0 ? issues.slice(0, shown) : issues;
  return (
    <section>
      <Title order={3} size="h6" mb="xs">
        {title}
      </Title>
      <ul className={classes.list}>
        {visible.map((issue, index) => (
          <IssueItem
            key={`${index}-${issue.message}`}
            issue={issue}
            done={issue.fix ? applied.has(fixKey(issue.fix)) : false}
            disabled={disabled}
            onFix={onFix}
          />
        ))}
      </ul>
      {issues.length > shown && (
        <Button
          variant="subtle"
          size="compact-sm"
          mt={6}
          leftSection={expanded ? <IconChevronUp size={14} /> : <IconChevronDown size={14} />}
          onClick={() => setExpanded(!expanded)}
          aria-expanded={expanded}
        >
          {expanded
            ? i18n.t('ui.data.ReadinessPanel.showFewer')
            : i18n.t('ui.data.ReadinessPanel.showMore', { hidden })}
        </Button>
      )}
    </section>
  );
}

interface IssueItemProps {
  issue: ReadinessIssue;
  done: boolean;
  disabled: boolean;
  onFix?: (fix: DataFix) => void;
}

function IssueItem({ issue, done, disabled, onFix }: IssueItemProps) {
  useTranslation();
  const { color, Icon, prefix } = SEVERITY[issue.severity];
  const label = CHECK_LABELS[issue.code];
  const { columns } = issue;
  const more = columns.length - COLUMNS_SHOWN;
  return (
    <Paper
      component="li"
      withBorder
      radius="md"
      className={classes.issue}
      data-severity={issue.severity}
      data-code={issue.code}
    >
      <ThemeIcon variant="light" color={color} size="md" radius="xl" aria-hidden>
        <Icon size={16} />
      </ThemeIcon>
      <div className={classes.message}>
        <Text size="sm">
          <VisuallyHidden>{prefix}</VisuallyHidden>
          {issue.message}
        </Text>
        {(label || columns.length > 0) && (
          <Group gap={6} mt={4} className={classes.meta}>
            {label && (
              <Badge size="xs" variant="light" color={color} tt="none" className={classes.kind}>
                {label}
              </Badge>
            )}
            {columns.length > 0 && (
              <Text size="xs" c="dimmed" ff="monospace" className={classes.columns}>
                <VisuallyHidden>
                  {columns.length === 1
                    ? i18n.t('ui.data.ReadinessPanel.column')
                    : i18n.t('ui.data.ReadinessPanel.columns')}
                </VisuallyHidden>
                {columns.slice(0, COLUMNS_SHOWN).join(', ')}
                {more > 0 && i18n.t('ui.data.ReadinessPanel.more2', { more })}
              </Text>
            )}
          </Group>
        )}
      </div>
      {issue.fix && onFix && (
        <Button
          size="compact-sm"
          variant={done ? 'subtle' : 'light'}
          color={done ? 'green' : undefined}
          leftSection={done ? <IconCheck size={14} /> : <IconWand size={14} />}
          disabled={done || disabled}
          classNames={{ root: classes.fix, label: classes.fixLabel }}
          onClick={() => onFix(issue.fix!)}
        >
          {done ? i18n.t('ui.data.ReadinessPanel.stepAdded') : issue.fix.label}
        </Button>
      )}
    </Paper>
  );
}
