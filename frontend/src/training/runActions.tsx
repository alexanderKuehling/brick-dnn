import { Badge, Loader, Text } from '@mantine/core';
import { modals } from '@mantine/modals';
import { notifications } from '@mantine/notifications';
import { api, ApiError, type RunDetail, type RunStatus, type RunSummary } from '@/api';
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

/** Runs of shared networks: the server refuses what the user's role does not allow (403). */
function notAllowed(error: unknown): boolean {
  return error instanceof ApiError && error.status === 403;
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
          title: notAllowed(error) ? 'You cannot stop this training' : 'Cannot stop the training',
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
        {run.project !== null && ' It is deleted for everyone in the network.'}
      </Text>
    ),
    labels: { confirm: 'Delete', cancel: 'Cancel' },
    confirmProps: { color: 'red' },
    onConfirm: async () => {
      try {
        await api.training.runs.remove(run.id);
      } catch (error) {
        if (!(error instanceof ApiError && error.status === 404)) {
          notifications.show({
            color: 'red',
            title: notAllowed(error)
              ? 'You cannot delete this training'
              : 'Cannot delete the training',
            message: (error as Error).message,
          });
          return;
        }
        // Deleted already (e.g. by the owner of its network).
      }
      onDeleted(run.id);
    },
  });
}
