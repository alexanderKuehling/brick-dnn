import { IconTrash } from '@tabler/icons-react';
import {
  ActionIcon,
  Alert,
  Button,
  Center,
  Group,
  Loader,
  SegmentedControl,
  Stack,
  Text,
  Tooltip,
  UnstyledButton,
} from '@mantine/core';
import type { RunSummary } from '@/api';
import {
  canDeleteRun,
  canTrain,
  filterRuns,
  otherStarter,
  type RunFilter,
  type TrainingScope,
} from './access';
import {
  formatDate,
  formatDuration,
  headlineMetric,
  isActive,
  progressLabel,
  searchLabel,
} from './format';
import { confirmDeleteRun, StatusBadge } from './runActions';
import classes from './Training.module.css';

interface HistoryViewProps {
  runs: RunSummary[] | null;
  error: string | null;
  onRetry: () => void;
  onOpen: (id: string) => void;
  onDeleted: (id: string) => void;
  onNewTraining: () => void;
  /** The user and the open network: decides what is listed and what they may do. */
  scope: TrainingScope;
  /** With an open network: its runs (from everyone) or all of the user's own. */
  filter: RunFilter;
  onFilterChange: (filter: RunFilter) => void;
}

const FILTERS: { value: RunFilter; label: string }[] = [
  { value: 'network', label: 'This network' },
  { value: 'mine', label: 'All my runs' },
];

/** The headline number of a run and what it is (the test result if there is one). */
function headline(run: RunSummary): [string, string] | null {
  if (isActive(run.status)) {
    return [progressLabel(run), 'in progress'];
  }
  const result = headlineMetric(run.metrics);
  return result ? [result[0], result[1].replace('validation', 'val.')] : null;
}

export function HistoryView({
  runs,
  error,
  onRetry,
  onOpen,
  onDeleted,
  onNewTraining,
  scope,
  filter,
  onFilterChange,
}: HistoryViewProps) {
  if (error && !runs) {
    return (
      <Alert color="red">
        {error}{' '}
        <Button size="compact-xs" variant="light" color="red" onClick={onRetry}>
          Try again
        </Button>
      </Alert>
    );
  }
  if (!runs) {
    return (
      <Center h={120}>
        <Loader />
      </Center>
    );
  }
  const network = scope.project !== null;
  const shownFilter: RunFilter = network ? filter : 'mine';
  const shown = filterRuns(runs, scope, shownFilter);
  const trainable = canTrain(scope);
  const toggle = network && (
    <SegmentedControl
      size="xs"
      fullWidth
      aria-label="Show runs"
      data={FILTERS}
      value={shownFilter}
      onChange={(value) => onFilterChange(value as RunFilter)}
      mb="xs"
    />
  );

  if (shown.length === 0) {
    return (
      <>
        {toggle}
        <Stack align="center" gap="xs" py="xl">
          <Text c="dimmed" size="sm" ta="center">
            {shownFilter === 'network'
              ? 'Nobody has trained this network yet.'
              : 'No trainings yet.'}
          </Text>
          {trainable && (
            <Button variant="light" onClick={onNewTraining}>
              Set up a training
            </Button>
          )}
        </Stack>
      </>
    );
  }

  return (
    <Stack gap={4}>
      {toggle}
      {shown.map((run) => {
        const by = otherStarter(run, scope);
        const result = headline(run);
        const duration = run.finished_at
          ? formatDuration((Date.parse(run.finished_at) - Date.parse(run.created_at)) / 1000)
          : null;
        return (
          <Group key={run.id} wrap="nowrap" gap={4}>
            <UnstyledButton
              className={classes.row}
              onClick={() => onOpen(run.id)}
              style={{ flex: 1, minWidth: 0 }}
            >
              <Group justify="space-between" wrap="nowrap" gap="sm">
                <div style={{ minWidth: 0 }}>
                  <Group gap="xs" wrap="nowrap">
                    <Text size="sm" fw={600} truncate>
                      {run.name}
                    </Text>
                    <StatusBadge status={run.status} />
                  </Group>
                  <Text size="xs" c="dimmed" truncate>
                    {[
                      by && `by ${by}`,
                      run.dataset.name,
                      run.tuning && searchLabel(run.tuning),
                      formatDate(run.created_at),
                      duration,
                    ]
                      .filter(Boolean)
                      .join(' · ')}
                  </Text>
                </div>
                {result && (
                  <div style={{ textAlign: 'right', flexShrink: 0 }}>
                    <Text size="sm" fw={600}>
                      {result[0]}
                    </Text>
                    <Text size="xs" c="dimmed">
                      {result[1]}
                    </Text>
                  </div>
                )}
              </Group>
            </UnstyledButton>
            {canDeleteRun(run, scope) ? (
              <Tooltip label="Delete">
                <ActionIcon
                  variant="subtle"
                  color="gray"
                  onClick={() => confirmDeleteRun(run, onDeleted)}
                  aria-label={`Delete ${run.name}`}
                >
                  <IconTrash size={16} />
                </ActionIcon>
              </Tooltip>
            ) : (
              // Keeps the rows aligned.
              <span className={classes.actionSpace} aria-hidden />
            )}
          </Group>
        );
      })}
    </Stack>
  );
}
