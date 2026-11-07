import { useState } from 'react';
import {
  IconAlertCircle,
  IconAlertTriangle,
  IconCheck,
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

const TASK_LABELS = { classification: 'Classification', regression: 'Regression' } as const;

function fixKey(fix: DataFix): string {
  return JSON.stringify([fix.type, fix.params]);
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
  // Fixes added since the last check; they disappear with the next preview.
  const [applied, setApplied] = useState<ReadonlySet<string>>(() => new Set());
  const [checked, setChecked] = useState(readiness);
  if (checked !== readiness) {
    setChecked(readiness);
    setApplied(new Set());
  }

  const ready = readiness.ready && !exportBlocked;
  const errors = readiness.issues.filter((issue) => issue.severity === 'error');
  const warnings = readiness.issues.filter((issue) => issue.severity !== 'error');
  const pendingFixes: DataFix[] = [];
  for (const issue of errors) {
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

  return (
    <Stack gap="md" className={classes.root}>
      {ready ? (
        <Alert
          color="green"
          variant="light"
          icon={<IconCircleCheck size={20} />}
          title="Ready for training"
        >
          {warnings.length > 0
            ? `The data can be used for training. Have a look at the ${warnings.length === 1 ? 'warning' : `${warnings.length} warnings`} below.`
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

      {errors.length > 0 && (
        <IssueList
          title="Must be fixed"
          issues={errors}
          applied={applied}
          disabled={fixesDisabled}
          onFix={(fix) => apply([fix])}
        />
      )}
      {warnings.length > 0 && (
        <IssueList
          title="Worth a look"
          issues={warnings}
          applied={applied}
          disabled={fixesDisabled}
          onFix={(fix) => apply([fix])}
        />
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
        <Button
          leftSection={<IconDatabaseExport size={16} />}
          onClick={onExport}
          loading={exporting}
          disabled={!ready}
        >
          Create training dataset
        </Button>
      </Group>
      <Text size="xs" c="dimmed" mt={-8}>
        {ready
          ? 'Saves the prepared data with its split; it appears in the Train panel under “Your data”.'
          : exportBlocked && errors.length > 0
            ? `${exportBlocked} Available once it and the problems above are fixed.`
            : exportBlocked
              ? `${exportBlocked} Available once it is fixed.`
              : 'Available once the problems above are fixed.'}
      </Text>
    </Stack>
  );
}

interface IssueListProps {
  title: string;
  issues: ReadinessIssue[];
  applied: ReadonlySet<string>;
  disabled: boolean;
  onFix: (fix: DataFix) => void;
}

function IssueList({ title, issues, applied, disabled, onFix }: IssueListProps) {
  return (
    <section>
      <Title order={3} size="h6" mb="xs">
        {title}
      </Title>
      <ul className={classes.list}>
        {issues.map((issue, index) => {
          const error = issue.severity === 'error';
          const done = issue.fix ? applied.has(fixKey(issue.fix)) : false;
          return (
            <Paper
              component="li"
              key={`${index}-${issue.message}`}
              withBorder
              radius="md"
              className={classes.issue}
              data-severity={issue.severity}
            >
              <ThemeIcon
                variant="light"
                color={error ? 'red' : 'yellow'}
                size="md"
                radius="xl"
                aria-hidden
              >
                {error ? <IconAlertCircle size={16} /> : <IconAlertTriangle size={16} />}
              </ThemeIcon>
              <div className={classes.message}>
                <Text size="sm">
                  <VisuallyHidden>{error ? 'Error: ' : 'Warning: '}</VisuallyHidden>
                  {issue.message}
                </Text>
                {issue.column && (
                  <Text size="xs" c="dimmed" ff="monospace">
                    {issue.column}
                  </Text>
                )}
              </div>
              {issue.fix && (
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
        })}
      </ul>
    </section>
  );
}
