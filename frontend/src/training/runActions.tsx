import { Badge, Loader, Text } from '@mantine/core';
import { modals } from '@mantine/modals';
import { notifications } from '@mantine/notifications';
import { api, ApiError, type RunDetail, type RunStatus, type RunSummary } from '@/api';
import { i18n } from '@/i18n';
import { isClassicalRun, isHybridRun } from './classical';
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
  // A run that still waits in the queue is cancelled: nothing was trained, nothing is lost.
  const queued = run.status === 'queued';
  modals.openConfirmModal({
    title: queued
      ? i18n.t('ui.training.runActions.cancelTheTraining')
      : i18n.t('ui.training.runActions.stopTheTraining'),
    children: (
      <Text size="sm">
        {queued
          ? i18n.t('ui.training.runActions.leavesTheQueueWithoutHaving', { name: run.name })
          : isClassicalRun(run)
            ? i18n.t('ui.training.runActions.scikitLearnCannotPauseA', { name: run.name })
            : isHybridRun(run)
              ? i18n.t('ui.training.runActions.stopsAtOnceAHybrid', { name: run.name })
              : i18n.t('ui.training.runActions.stopsAfterTheCurrentBatch', { name: run.name })}
      </Text>
    ),
    labels: queued
      ? {
          confirm: i18n.t('ui.training.runActions.cancelTraining'),
          cancel: i18n.t('ui.training.runActions.keepWaiting'),
        }
      : {
          confirm: i18n.t('ui.training.runActions.stopTraining'),
          cancel: i18n.t('ui.training.runActions.keepTraining'),
        },
    confirmProps: { color: 'red' },
    onConfirm: async () => {
      try {
        onStopped(await api.training.runs.stop(run.id));
      } catch (error) {
        const what = queued ? 'cancel' : 'stop';
        notifications.show({
          color: 'red',
          title: notAllowed(error)
            ? i18n.t('ui.training.runActions.youCannotThisTraining', { what })
            : i18n.t('ui.training.runActions.cannotTheTraining', { what }),
          message: (error as Error).message,
        });
      }
    },
  });
}

export function confirmDeleteRun(run: RunSummary, onDeleted: (id: string) => void) {
  modals.openConfirmModal({
    title: i18n.t('ui.training.runActions.delete', { name: run.name }),
    children: (
      <Text size="sm">
        {run.status === 'queued'
          ? i18n.t('ui.training.runActions.theTrainingLeavesTheQueue')
          : isActive(run.status)
            ? i18n.t('ui.training.runActions.theTrainingIsStoppedAnd')
            : i18n.t('ui.training.runActions.theResultsAndTheTrained')}
        {run.project !== null && i18n.t('ui.training.runActions.itIsDeletedForEveryone')}
      </Text>
    ),
    labels: {
      confirm: i18n.t('ui.training.runActions.delete2'),
      cancel: i18n.t('ui.training.runActions.cancel'),
    },
    confirmProps: { color: 'red' },
    onConfirm: async () => {
      try {
        await api.training.runs.remove(run.id);
      } catch (error) {
        if (!(error instanceof ApiError && error.status === 404)) {
          notifications.show({
            color: 'red',
            title: notAllowed(error)
              ? i18n.t('ui.training.runActions.youCannotDeleteThisTraining')
              : i18n.t('ui.training.runActions.cannotDeleteTheTraining'),
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
