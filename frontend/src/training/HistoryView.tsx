import { useState } from 'react';
import { IconStack2, IconTrash } from '@tabler/icons-react';
import {
  ActionIcon,
  Alert,
  Badge,
  Button,
  Center,
  Checkbox,
  Group,
  Loader,
  SegmentedControl,
  Stack,
  Text,
  Tooltip,
  UnstyledButton,
} from '@mantine/core';
import type { RunDetail, RunSummary } from '@/api';
import {
  canDeleteRun,
  canTrain,
  filterRuns,
  otherStarter,
  type RunFilter,
  type TrainingScope,
} from './access';
import { CombineDialog } from './CombineDialog';
import { canBeMember, combineProblem, isEnsemble } from './ensemble';
import {
  formatDate,
  formatDuration,
  headlineMetric,
  isActive,
  progressLabel,
  searchLabel,
  startsIn,
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
  /** Finished runs can be combined into an ensemble (missing: not offered). */
  onCombined?: (run: RunDetail) => void;
  /** The network a new ensemble belongs to (one the user owns or edits). */
  combineProject?: number | null;
}

const FILTERS: { value: RunFilter; label: string }[] = [
  { value: 'network', label: 'This network' },
  { value: 'mine', label: 'All my runs' },
];

/** The headline number of a run and what it is (the test result if there is one). */
function headline(run: RunSummary): [string, string] | null {
  if (run.status === 'queued') {
    // Its place in the queue and the server's rough estimate of the start.
    const queue = run.queue;
    const start = queue?.worker_online ? startsIn(queue.estimated_wait_seconds) : null;
    return [queue ? `Position ${queue.position}` : 'Waiting', start ?? 'in the queue'];
  }
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
  onCombined,
  combineProject = null,
}: HistoryViewProps) {
  // Selecting runs to combine: their ids, in the order they were picked.
  const [selecting, setSelecting] = useState(false);
  const [selected, setSelected] = useState<string[]>([]);
  const [combining, setCombining] = useState(false);
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

  const members = shown.filter(canBeMember);
  const picked = selected
    .map((id) => members.find((run) => run.id === id))
    .filter((run): run is RunSummary => run !== undefined);
  const problem = combineProblem(picked);
  const offerCombine = onCombined !== undefined && members.length >= 2;
  const toggleRun = (id: string) =>
    setSelected((current) =>
      current.includes(id) ? current.filter((other) => other !== id) : [...current, id]
    );
  const stopSelecting = () => {
    setSelecting(false);
    setSelected([]);
  };

  return (
    <Stack gap={4}>
      {toggle}
      {offerCombine && (
        <Group justify="space-between" gap="xs" mb={4} wrap="nowrap">
          <Text size="xs" c="dimmed">
            {selecting
              ? (problem ?? `${picked.length} runs selected.`)
              : 'Finished runs of one dataset can be combined into an ensemble.'}
          </Text>
          {selecting ? (
            <Group gap={4} wrap="nowrap">
              <Button size="compact-xs" variant="subtle" color="gray" onClick={stopSelecting}>
                Cancel
              </Button>
              <Button
                size="compact-xs"
                leftSection={<IconStack2 size={14} />}
                disabled={problem !== null}
                onClick={() => setCombining(true)}
              >
                Combine into an ensemble
              </Button>
            </Group>
          ) : (
            <Button
              size="compact-xs"
              variant="light"
              leftSection={<IconStack2 size={14} />}
              onClick={() => setSelecting(true)}
            >
              Combine runs
            </Button>
          )}
        </Group>
      )}
      {shown.map((run) => {
        const by = otherStarter(run, scope);
        const result = headline(run);
        const duration = run.finished_at
          ? formatDuration((Date.parse(run.finished_at) - Date.parse(run.created_at)) / 1000)
          : null;
        const selectable = selecting && canBeMember(run);
        return (
          <Group key={run.id} wrap="nowrap" gap={4}>
            {selecting &&
              (selectable ? (
                <Checkbox
                  aria-label={`Select ${run.name}`}
                  checked={selected.includes(run.id)}
                  onChange={() => toggleRun(run.id)}
                />
              ) : (
                <span style={{ width: 20, flexShrink: 0 }} aria-hidden />
              ))}
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
                    {isEnsemble(run) && (
                      <Badge variant="outline" color="grape" tt="none" style={{ flexShrink: 0 }}>
                        Ensemble
                      </Badge>
                    )}
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
      {onCombined && (
        <CombineDialog
          opened={combining}
          runs={picked}
          project={combineProject}
          onClose={() => setCombining(false)}
          onCreated={(run) => {
            stopSelecting();
            onCombined(run);
          }}
        />
      )}
    </Stack>
  );
}
