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
import type { DataFix, NewStep, Readiness, ReadinessIssue } from '@/api/types';
import { usePrepAccess } from '../prepAccess';
import { formatCount } from '../table/format';
import classes from './ReadinessPanel.module.css';

export interface ReadinessPanelProps {
  readiness: Readiness;
  onAddStep: (step: NewStep) => void;
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

const TASK_LABELS = { classification: 'Classification', regression: 'Regression' } as const;

type Severity = ReadinessIssue['severity'];

/** The lists of the panel: errors first; long lists of hints show their first items. */
const SECTIONS: { severity: Severity; title: string; shown: number }[] = [
  { severity: 'error', title: 'Must be fixed', shown: Infinity },
  { severity: 'warning', title: 'Worth a look', shown: 8 },
  { severity: 'info', title: 'Suggestions', shown: 3 },
];

const SEVERITY = {
  error: { color: 'red', Icon: IconAlertCircle, prefix: 'Error: ' },
  warning: { color: 'yellow', Icon: IconAlertTriangle, prefix: 'Warning: ' },
  info: { color: 'blue', Icon: IconBulb, prefix: 'Suggestion: ' },
} as const;

/** Short names of the checks (`ReadinessIssue.code`), shown as a badge. */
const CHECK_LABELS: Record<string, string> = {
  target_missing: 'Target',
  target_not_numeric: 'Target',
  missing_values: 'Missing values',
  categorical_features: 'Categories',
  high_cardinality: 'Many categories',
  inconsistent_categories: 'Spellings',
  numbers_as_text: 'Numbers as text',
  dates_as_text: 'Dates as text',
  text_features: 'Free text',
  date_features: 'Dates',
  infinite_values: 'Infinite values',
  leakage: 'Target leakage',
  split_overlap: 'Train/test overlap',
  duplicate_rows: 'Duplicates',
  conflicting_labels: 'Conflicting labels',
  rare_classes: 'Rare classes',
  classes_not_in_split: 'Split',
  imbalanced: 'Imbalance',
  id_like: 'ID column',
  constant: 'Constant',
  near_constant: 'Near-constant',
  redundant: 'Redundant',
  no_partner: 'No strong partner',
  skewed: 'Skewed',
  skewed_target: 'Skewed target',
  scales: 'Scales',
  few_rows: 'Few rows',
  many_features: 'Many features',
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
  onExport,
  exporting,
  fixesDisabled = false,
  exportBlocked = null,
}: ReadinessPanelProps) {
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
      next.add(fixKey(fix));
    }
    setApplied(next);
  };

  const split = readiness.split_counts;
  const stats: { label: string; value: string; hint?: string }[] = [
    { label: 'Target', value: readiness.target ?? 'Not set' },
    { label: 'Task', value: readiness.task ? TASK_LABELS[readiness.task] : '–' },
    { label: 'Features', value: formatCount(readiness.num_features) },
    {
      label: 'Classes',
      value: readiness.num_classes === null ? '–' : formatCount(readiness.num_classes),
    },
    { label: 'Rows', value: formatCount(readiness.rows) },
    {
      label: 'Split',
      value: split ? split.map(formatCount).join(' / ') : '–',
      hint: split ? 'train / validation / test' : undefined,
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
          title="Ready for training"
        >
          {notes.length > 0
            ? `The data can be used for training. Have a look at the ${notes.join(' and ')} below.`
            : 'The data can be used for training.'}
        </Alert>
      ) : (
        <Alert
          color="red"
          variant="light"
          icon={<IconAlertCircle size={20} />}
          title="Not ready for training yet"
        >
          {errors.length === 0
            ? (exportBlocked ?? 'The data cannot be used for training yet.')
            : errors.length === 1
              ? '1 problem must be fixed first.'
              : `${errors.length} problems must be fixed first.`}
          {pendingFixes.length > 0 && ' Each fix adds a step to the pipeline.'}
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
            Class order (class index 0, 1, …)
          </Text>
          <Group gap={4}>
            {classNames.slice(0, CLASSES_SHOWN).map((name, index) => (
              <Badge key={`${index}-${name}`} variant="light" color="gray" tt="none">
                {index}: {name}
              </Badge>
            ))}
            {classNames.length > CLASSES_SHOWN && (
              <Text size="xs" c="dimmed">
                +{classNames.length - CLASSES_SHOWN} more
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
          No problems found.
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
            Fix all ({pendingFixes.length})
          </Button>
        )}
        {!readOnly && (
          <Button
            leftSection={<IconDatabaseExport size={16} />}
            onClick={onExport}
            loading={exporting}
            disabled={!ready}
          >
            Create training dataset
          </Button>
        )}
      </Group>
      {!readOnly && (
        <Text size="xs" c="dimmed" mt={-8}>
          {ready
            ? 'Saves the prepared data with its split; it appears in the Train panel under “Your data”.'
            : exportBlocked && errors.length > 0
              ? `${exportBlocked} Available once it and the problems above are fixed.`
              : exportBlocked
                ? `${exportBlocked} Available once it is fixed.`
                : 'Available once the problems above are fixed.'}
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
          {expanded ? 'Show fewer' : `Show ${hidden} more`}
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
                <VisuallyHidden>{columns.length === 1 ? 'Column: ' : 'Columns: '}</VisuallyHidden>
                {columns.slice(0, COLUMNS_SHOWN).join(', ')}
                {more > 0 && ` +${more} more`}
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
          {done ? 'Step added' : issue.fix.label}
        </Button>
      )}
    </Paper>
  );
}
