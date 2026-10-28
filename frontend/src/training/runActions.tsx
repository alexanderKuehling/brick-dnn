import { Badge, Loader, Text } from '@mantine/core';
import { modals } from '@mantine/modals';
import { notifications } from '@mantine/notifications';
import { api, type RunDetail, type RunStatus, type RunSummary } from '@/api';
import { isActive, STATUS_COLORS, STATUS_LABELS } from './format';

export function StatusBadge({ status }: { status: RunStatus }) {
  const color = STATUS_COLORS[status];
  return (
    <Badge
      color={color}
      variant="light"
      tt="none"
      style={{ flexShrink: 0 }}
      leftSection={isActive(status) ? <Loader size={10} color={color} /> : undefined}
    >
      {STATUS_LABELS[status]}
    </Badge>
  );
}

export function confirmStopRun(run: RunSummary, onStopped: (run: RunDetail) => void) {
  modals.openConfirmModal({
    title: 'Stop the training?',
    children: (
      <Text size="sm">
        “{run.name}” stops after the current batch. The weights trained so far are kept.
      </Text>
    ),
    labels: { confirm: 'Stop training', cancel: 'Keep training' },
    confirmProps: { color: 'red' },
    onConfirm: async () => {
      try {
        onStopped(await api.training.runs.stop(run.id));
      } catch (error) {
        notifications.show({
          color: 'red',
          title: 'Cannot stop the training',
          message: (error as Error).message,
        });
      }
    },
  });
}

export function confirmDeleteRun(run: RunSummary, onDeleted: (id: string) => void) {
  modals.openConfirmModal({
    title: `Delete “${run.name}”?`,
    children: (
      <Text size="sm">
        {isActive(run.status)
          ? 'The training is stopped and everything it produced is deleted, including the weights.'
          : 'The results and the trained weights are deleted. This cannot be undone.'}
      </Text>
    ),
    labels: { confirm: 'Delete', cancel: 'Cancel' },
    confirmProps: { color: 'red' },
    onConfirm: async () => {
      try {
        await api.training.runs.remove(run.id);
        onDeleted(run.id);
      } catch (error) {
        notifications.show({
          color: 'red',
          title: 'Cannot delete the training',
          message: (error as Error).message,
        });
      }
    },
  });
}
