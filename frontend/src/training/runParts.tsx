import {
  IconAlertTriangle,
  IconChevronDown,
  IconChevronRight,
  IconHourglassHigh,
} from '@tabler/icons-react';
import { Alert, Button, Group, Paper, Stack, Text } from '@mantine/core';
import type { RunDetail } from '@/api';
import { formatCpu } from '@/usage/format';
import { isActive, queueText } from './format';

/** A button that opens or closes a section below it. */
export function ToggleButton({
  opened,
  onClick,
  children,
}: {
  opened: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <Button
      variant="subtle"
      color="gray"
      size="compact-sm"
      leftSection={opened ? <IconChevronDown size={14} /> : <IconChevronRight size={14} />}
      onClick={onClick}
      aria-expanded={opened}
      mb={4}
    >
      {children}
    </Button>
  );
}

/** "2.3 CPU min" used (of its limit while it runs); null before anything was counted. */
export function cpuUsage(run: RunDetail): string | null {
  const usage = run.usage;
  if (!usage || (usage.cpu_minutes <= 0 && usage.limit_cpu_minutes === null)) {
    return null;
  }
  return isActive(run.status) && usage.limit_cpu_minutes !== null
    ? `${formatCpu(usage.cpu_minutes, '')} of ${formatCpu(usage.limit_cpu_minutes)}`
    : formatCpu(usage.cpu_minutes);
}

const QUEUED_MESSAGE = 'Waiting for a free training slot.';

/**
 * A run that waits for a training worker: its position, when it probably starts (the
 * server's rough estimate, updated with every poll) and what holds it up.
 */
export function QueueNotice({ run }: { run: RunDetail }) {
  const queue = run.queue ?? null;
  const notes: string[] = [];
  if (run.message && run.message !== QUEUED_MESSAGE) {
    notes.push(run.message); // e.g. it starts again after the worker was restarted
  }
  if (queue?.blocked_by === 'account') {
    notes.push(
      'Your plan limits how many trainings run at the same time: this one starts when your running training has ended.'
    );
  } else if (queue?.worker_online && queue.estimated_wait_seconds !== null) {
    notes.push('The start time is a rough estimate.');
  }
  if (queue && queue.waiting > 1) {
    notes.push(`${queue.waiting} trainings are waiting.`);
  }
  if (queue?.hint) {
    notes.push(queue.hint);
  }
  notes.push('You can close this panel: the training starts by itself.');
  return (
    <Stack gap="xs">
      <Paper component="section" withBorder p="sm" radius="md" aria-label="Queue">
        <Group gap="xs" wrap="nowrap" align="flex-start">
          <IconHourglassHigh
            size={18}
            color="var(--mantine-color-grape-6)"
            style={{ flexShrink: 0, marginTop: 2 }}
            aria-hidden
          />
          <div style={{ minWidth: 0 }}>
            <Text size="sm" fw={600}>
              {queueText(queue)}
            </Text>
            <Text size="xs" c="dimmed" mt={4}>
              {notes.join(' ')}
            </Text>
          </div>
        </Group>
      </Paper>
      {queue && !queue.worker_online && (
        <Alert
          color="yellow"
          icon={<IconAlertTriangle size={18} />}
          title="No training worker is running"
        >
          This server cannot train right now. Your training keeps its place and starts as soon as a
          worker is back.
        </Alert>
      )}
    </Stack>
  );
}
