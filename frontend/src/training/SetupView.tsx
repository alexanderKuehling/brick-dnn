import { useEffect, useMemo, useState } from 'react';
import {
  IconAlertTriangle,
  IconCircleCheck,
  IconHourglassHigh,
  IconInfoCircle,
  IconPlayerPlay,
  IconUsers,
  IconWand,
} from '@tabler/icons-react';
import { Alert, Button, Divider, Group, List, Loader, Paper, Stack, Text } from '@mantine/core';
import { modals } from '@mantine/modals';
import { notifications } from '@mantine/notifications';
import {
  api,
  ApiError,
  type Analysis,
  type Dataset,
  type GraphDocument,
  type QueuePreview,
  type RunDetail,
  type RunSummary,
  type TrainingStatus,
} from '@/api';
import { AccountLink, CostEstimate } from '@/usage/CostEstimate';
import { useCostEstimate } from '@/usage/useCostEstimate';
import type { TrainingProject } from './access';
import { classicalMaxRows, MODEL_KINDS, trainingFamily } from './classical';
import { DataSourcePicker } from './DataSourcePicker';
import { isSegmentationSource } from './fit';
import { startsIn } from './format';
import { CLASSICAL_TUNABLES, planMetrics } from './search';
import {
  effectiveSource,
  loadSettings,
  maxSegmentationBatch,
  saveSettings,
  toConfig,
  toTuning,
  validateSettings,
  type SettingsValues,
  type SourceChoice,
} from './settings';
import { SettingsForm, type ClassicalSettings, type HybridSettings } from './SettingsForm';
import { useCompatibility, type CompatibilityState } from './useCompatibility';
import type { DatasetFitController } from './useDatasetFit';
import type { DataController } from './useDatasets';

interface SetupViewProps {
  status: TrainingStatus;
  doc: GraphDocument;
  networkName: string;
  /** The saved network the trainings are started in (its members see them), if any. */
  project?: TrainingProject | null;
  analysis: Analysis | null;
  activeRun: RunSummary | null;
  /** Who started `activeRun`, if it was someone else. */
  activeRunBy?: string | null;
  /** Data source and uploads; kept by useTraining so they survive closing the panel. */
  data: DataController;
  onStarted: (run: RunDetail) => void;
  onOpenRun: (id: string) => void;
  onShowProblems: () => void;
  /**
   * Called when the server says training is unavailable (503) or busy (409: another training
   * runs or, with a training queue, too many of the account's trainings are waiting).
   */
  onRefresh: () => void;
  /**
   * The builder's check of the network against the chosen dataset, with the action that adapts
   * the network to it (see useDatasetFit). Without it, the view checks the network itself.
   */
  fit?: DatasetFitController | null;
}

// Mantine's list items do not wrap next to the bullet on narrow screens without this.
const WRAPPING_LIST = {
  item: { whiteSpace: 'normal' },
  itemWrapper: { display: 'inline' },
} as const;

interface StartError {
  message: string;
  errors?: string[];
  busy?: boolean;
  /** A budget or the storage quota does not allow it (HTTP 403 with a budget code). */
  budget?: string;
}

const BUDGET_TITLES: Record<string, string> = {
  budget_exceeded: 'Not enough compute budget',
  estimate_too_high: 'Too much compute for the budget',
  storage_quota_exceeded: 'The storage of your account is full',
};

export function SetupView({
  status,
  doc,
  networkName,
  project = null,
  analysis,
  activeRun,
  activeRunBy = null,
  data,
  onStarted,
  onOpenRun,
  onShowProblems,
  onRefresh,
  fit = null,
}: SetupViewProps) {
  const [settings, setSettings] = useState<SettingsValues>(loadSettings);
  const [runName, setRunName] = useState('');
  const [starting, setStarting] = useState(false);
  const [startError, setStartError] = useState<StartError | null>(null);
  const { source: choice, datasets, reloadDatasets } = data;

  // Refresh the list whenever the view is shown (it is kept while the panel is closed).
  useEffect(reloadDatasets, [reloadDatasets]);

  const choose = (next: SourceChoice) => {
    data.setSource(next);
    setStartError(null);
  };
  const changeSettings = (next: SettingsValues) => {
    setSettings(next);
    saveSettings(next);
  };

  const source = effectiveSource(status, choice, datasets);
  const { choice: effective, ref, upload, name: datasetName } = source;

  const empty = doc.nodes.length === 0;
  const networkValid = !empty && analysis?.valid === true;
  const ownCheck = useCompatibility(doc, ref, networkValid && !fit);
  const compatibility = fit ? fit.compatibility : ownCheck;
  const check = compatibility.result;

  // A classical model (scikit-learn) is fitted once: fewer settings; its search varies the
  // parameters of the model brick (if the server offers it). A hybrid network has the
  // settings of a network for its features, and the search of its classical model.
  const family = trainingFamily(analysis);
  const model = family !== 'neural' ? (analysis?.classical ?? null) : null;
  const maxRows = classicalMaxRows(status, analysis);
  const classicalSearch = Boolean(
    model && status.classical?.tuning && CLASSICAL_TUNABLES[model.type]
  );
  const classical: ClassicalSettings | null =
    model && family === 'classical'
      ? {
          model: MODEL_KINDS[model.type] ?? model.estimator,
          maxRows,
          optimisesKernel: model.kernel !== null,
          type: classicalSearch ? model.type : null,
        }
      : null;
  const hybrid: HybridSettings | null =
    model && family === 'hybrid'
      ? {
          model: MODEL_KINDS[model.type] ?? model.estimator,
          features: analysis?.hybrid?.features ?? null,
          maxRows,
          type: classicalSearch ? model.type : null,
        }
      : null;
  const searchType = classical?.type ?? hybrid?.type ?? null;
  const search = settings.search && ((!classical && !hybrid) || searchType !== null);
  // The objective metrics the data allows (from the check; null while it is not known).
  const metrics = compatibility.fresh ? planMetrics(check?.plan) : null;

  const builtinSource = choice.tab === 'builtin';
  // A class for every pixel: the batch size is limited by the memory a batch needs, which the
  // check of the network reports (plan.segmentation) for the limit of this server.
  const segmentationSource = !classical && !hybrid && isSegmentationSource(source);
  const maxBatch = segmentationSource
    ? maxSegmentationBatch(
        compatibility.fresh ? check?.plan?.segmentation : null,
        status.segmentation?.batch_memory_mb
      )
    : null;
  const segmentation = useMemo(
    () => (segmentationSource ? { maxBatch } : null),
    [segmentationSource, maxBatch]
  );
  // Datasets made in the Data tab bring their own split; the share fields are not used.
  const predefinedSplit = choice.tab === 'upload' ? (upload?.info.predefined_split ?? null) : null;
  const errors = validateSettings(settings, {
    classical: classical && { maxRows, model: searchType },
    hybrid: hybrid && { model: searchType },
    segmentation,
    validationRows: predefinedSplit && predefinedSplit.length === 3 ? predefinedSplit[1] : null,
    metrics: metrics && metrics.map((metric) => metric.name),
  });
  if (builtinSource) {
    // Built-in datasets always test on their official test images: the test share is not used.
    delete errors.test_split;
  }
  if (predefinedSplit) {
    delete errors.val_split;
    delete errors.test_split;
  }
  // With a training queue (a worker trains) a new training simply waits for its turn;
  // without one the server refuses a start while it is busy.
  const queueing = status.queue != null;
  const blockedByActive = !queueing && activeRun !== null && status.max_concurrent_runs <= 1;
  const settingsValid = Object.keys(errors).length === 0;

  // The settings as they are sent: for the start and for the estimate of its CPU time.
  const knownDevice = status.devices.some((d) => d.id === settings.device);
  const isClassical = classical !== null;
  const isHybrid = hybrid !== null;
  const config = useMemo(
    () =>
      // A remembered device may not exist on this server (any more).
      toConfig(
        { ...settings, device: knownDevice ? settings.device : 'auto' },
        {
          builtin: builtinSource,
          classical: isClassical,
          hybrid: isHybrid,
          segmentation: segmentationSource,
        }
      ),
    [settings, knownDevice, builtinSource, isClassical, isHybrid, segmentationSource]
  );
  const tuning = useMemo(
    () => ((isClassical || isHybrid) && !searchType ? null : toTuning(settings, searchType)),
    [settings, isClassical, isHybrid, searchType]
  );
  const cost = useCostEstimate(
    doc,
    ref,
    config,
    tuning,
    networkValid && ref !== null && check?.ok === true && settingsValid
  );
  const refusedByBudget = cost.fresh && cost.budget?.allowed === false;

  const canStart =
    networkValid &&
    ref !== null &&
    !compatibility.loading &&
    check?.ok !== false &&
    settingsValid &&
    !blockedByActive &&
    !refusedByBudget;

  const deleteDataset = (dataset: Dataset) =>
    modals.openConfirmModal({
      title: `Delete “${dataset.name}”?`,
      children: (
        <Text size="sm">
          The uploaded data is removed from the server. Finished trainings keep their results.
        </Text>
      ),
      labels: { confirm: 'Delete', cancel: 'Cancel' },
      confirmProps: { color: 'red' },
      onConfirm: async () => {
        try {
          await data.remove(dataset);
        } catch (error) {
          notifications.show({
            color: 'red',
            title: 'Cannot delete the data',
            message: (error as Error).message,
          });
        }
      },
    });

  const start = async () => {
    if (!ref || starting) {
      return;
    }
    setStarting(true);
    setStartError(null);
    try {
      const run = await api.training.runs.create({
        name: runName.trim() || networkName.trim() || 'Untitled network',
        graph: doc,
        dataset: ref,
        config,
        ...(tuning ? { tuning } : {}),
        ...(project ? { project: project.id } : {}),
      });
      if (run.budget_warning) {
        notifications.show({ color: 'yellow', message: run.budget_warning, autoClose: 10000 });
      }
      onStarted(run);
    } catch (error) {
      const message = (error as Error).message;
      const code =
        error instanceof ApiError ? (error.data as { code?: unknown } | null)?.code : null;
      if (error instanceof ApiError && error.status === 403 && typeof code === 'string') {
        setStartError({ message, budget: code });
      } else if (error instanceof ApiError && error.status === 409) {
        setStartError({ message, busy: true });
        onRefresh();
      } else if (error instanceof ApiError && error.status === 503) {
        setStartError({ message });
        onRefresh();
      } else if (error instanceof ApiError && error.status === 400) {
        const data = error.data as { errors?: unknown } | null;
        const list = Array.isArray(data?.errors) ? data.errors.map(String) : undefined;
        setStartError({ message, errors: list && list.length > 1 ? list : undefined });
      } else {
        setStartError({ message });
      }
    } finally {
      setStarting(false);
    }
  };

  return (
    <Stack gap="md">
      {empty && (
        <Alert color="gray" icon={<IconInfoCircle size={18} />}>
          The canvas is empty. Build a network (or load an example) to train it.
        </Alert>
      )}
      {!empty && analysis && !analysis.valid && (
        <Alert color="red" icon={<IconAlertTriangle size={18} />} title="The network has problems">
          <Text size="sm">Fix the layers marked in red first, then come back here.</Text>
          <Button size="compact-sm" variant="light" color="red" mt="xs" onClick={onShowProblems}>
            Show problems
          </Button>
        </Alert>
      )}
      {blockedByActive && activeRun && (
        <Alert color="blue" icon={<Loader size={16} />} title="A training is running">
          <Text size="sm">
            “{activeRun.name}”{activeRunBy ? ` (started by ${activeRunBy})` : ''} is still training.
            You can start a new one when it has finished.
          </Text>
          <Button size="compact-sm" variant="light" mt="xs" onClick={() => onOpenRun(activeRun.id)}>
            Show it
          </Button>
        </Alert>
      )}

      <Divider label="Data" labelPosition="left" />
      <DataSourcePicker
        status={status}
        choice={effective}
        onChange={choose}
        datasets={datasets}
        datasetsError={data.datasetsError}
        onReloadDatasets={reloadDatasets}
        uploading={data.uploading}
        uploadProgress={data.uploadProgress}
        uploadError={data.uploadError}
        onDismissUploadError={data.dismissUploadError}
        onUpload={data.upload}
        onCancelUpload={data.cancelUpload}
        onDelete={deleteDataset}
      />

      {networkValid && ref && (
        <CheckResult
          {...compatibility}
          datasetName={datasetName}
          canAdapt={fit !== null}
          onAdapt={fit?.adapt ?? null}
        />
      )}

      <Divider label="Settings" labelPosition="left" />
      <SettingsForm
        values={settings}
        onChange={changeSettings}
        errors={errors}
        name={runName}
        namePlaceholder={networkName || 'Untitled network'}
        onNameChange={setRunName}
        devices={status.devices}
        defaultDevice={status.default_device}
        builtin={builtinSource}
        predefinedSplit={predefinedSplit}
        classical={classical}
        hybrid={hybrid}
        segmentation={segmentation}
        metrics={metrics}
      />

      {networkValid && ref && check?.ok && settingsValid && <CostEstimate state={cost} />}
      {networkValid && ref && check?.ok && settingsValid && cost.fresh && (
        <QueueHint queue={cost.queue} />
      )}

      {startError && (
        <Alert
          color={startError.busy ? 'yellow' : 'red'}
          icon={<IconAlertTriangle size={18} />}
          title={
            startError.busy
              ? queueing
                ? 'Too many trainings are waiting'
                : 'Another training is running'
              : startError.budget
                ? (BUDGET_TITLES[startError.budget] ?? 'Not enough budget')
                : 'Cannot start the training'
          }
        >
          <Text size="sm">{startError.message}</Text>
          {startError.budget && <AccountLink />}
          {startError.errors && (
            <List styles={WRAPPING_LIST} size="sm" mt={4}>
              {startError.errors.map((message) => (
                <List.Item key={message}>{message}</List.Item>
              ))}
            </List>
          )}
          {startError.busy && activeRun && (
            <Button
              size="compact-sm"
              variant="light"
              color="yellow"
              mt="xs"
              onClick={() => onOpenRun(activeRun.id)}
            >
              Open it
            </Button>
          )}
        </Alert>
      )}

      <Group justify="flex-end" gap="sm">
        {project?.shared && (
          <Group gap={6} wrap="nowrap" mr="auto">
            <IconUsers size={14} color="var(--mantine-color-dimmed)" aria-hidden />
            <Text size="xs" c="dimmed">
              Visible to everyone in this network
            </Text>
          </Group>
        )}
        <Button
          leftSection={<IconPlayerPlay size={16} />}
          onClick={() => void start()}
          loading={starting}
          disabled={!canStart}
        >
          {search ? 'Start search' : 'Start training'}
        </Button>
      </Group>
    </Stack>
  );
}

/**
 * Servers with a training queue: says before the start that the training will wait, where
 * and roughly how long (the server's estimate), or that no worker is running at all. Nothing
 * when it would start right away.
 */
function QueueHint({ queue }: { queue: QueuePreview | null }) {
  if (!queue) {
    return null;
  }
  if (!queue.worker_online) {
    return (
      <Alert
        color="yellow"
        icon={<IconAlertTriangle size={18} />}
        title="No training worker is running"
      >
        <Text size="sm">
          This server cannot train right now. A training you start waits in the queue until a worker
          is back.
        </Text>
      </Alert>
    );
  }
  const wait = queue.estimated_wait_seconds;
  const blocked = queue.blocked_by === 'account';
  if (!blocked && queue.position <= 1 && (wait === null || wait < 20)) {
    return null;
  }
  const start = blocked ? 'starts after your running training' : startsIn(wait);
  return (
    <Group gap={6} wrap="nowrap" align="flex-start">
      <IconHourglassHigh
        size={16}
        color="var(--mantine-color-dimmed)"
        style={{ flexShrink: 0, marginTop: 2 }}
        aria-hidden
      />
      <Text size="xs" c="dimmed">
        The server is busy: this training will wait in the queue (position {queue.position}
        {start ? `, ${start}` : ''}; an estimate).{queue.hint ? ` ${queue.hint}` : ''}
      </Text>
    </Group>
  );
}

interface CheckResultProps extends CompatibilityState {
  datasetName: string | null;
  /** The network can be adapted here (the fixes of the result are offered). */
  canAdapt: boolean;
  /** Applies the suggested fixes; null while they are not known for the current network. */
  onAdapt: (() => void) | null;
}

function CheckResult({ result, loading, error, datasetName, canAdapt, onAdapt }: CheckResultProps) {
  if (error) {
    return (
      <Alert color="red" icon={<IconAlertTriangle size={18} />} title="Cannot check the data">
        {error}
      </Alert>
    );
  }
  if (!result) {
    return (
      <Group gap="xs">
        <Loader size="xs" />
        <Text size="sm" c="dimmed">
          Checking the network against the data …
        </Text>
      </Group>
    );
  }
  return (
    <Stack gap="xs" style={{ opacity: loading ? 0.6 : 1, transition: 'opacity 150ms' }}>
      {result.errors.length > 0 && (
        <Alert color="red" icon={<IconAlertTriangle size={18} />} title="Does not fit the data">
          <List styles={WRAPPING_LIST} size="sm" spacing={4}>
            {result.errors.map((message) => (
              <List.Item key={message}>{message}</List.Item>
            ))}
          </List>
        </Alert>
      )}
      {result.ok && result.plan && (
        <Alert color="green" icon={<IconCircleCheck size={18} />} title="Ready to train">
          <Text size="sm">{result.plan.summary}</Text>
          {result.plan.notes.length > 0 && (
            <List styles={WRAPPING_LIST} size="xs" c="dimmed" mt={4} spacing={2}>
              {result.plan.notes.map((note) => (
                <List.Item key={note}>{note}</List.Item>
              ))}
            </List>
          )}
        </Alert>
      )}
      {result.warnings.map((message) => (
        <Alert key={message} color="yellow" icon={<IconInfoCircle size={18} />}>
          {message}
        </Alert>
      ))}
      {canAdapt && result.fixes && result.fixes.length > 0 && (
        <Paper withBorder p="sm" radius="md">
          <Stack gap={6}>
            <Text size="sm">
              {result.ok
                ? `To fit “${datasetName ?? 'the data'}” exactly, change:`
                : `To fit “${datasetName ?? 'the data'}”, change:`}
            </Text>
            <List styles={WRAPPING_LIST} size="sm" spacing={2}>
              {result.fixes.map((fix) => (
                <List.Item key={fix.node}>{fix.summary}</List.Item>
              ))}
            </List>
            <Button
              size="sm"
              variant={result.ok ? 'light' : 'filled'}
              leftSection={<IconWand size={16} />}
              onClick={onAdapt ?? undefined}
              disabled={loading || !onAdapt}
              style={{ alignSelf: 'flex-start' }}
            >
              Adapt network to “{datasetName ?? 'the data'}”
            </Button>
          </Stack>
        </Paper>
      )}
    </Stack>
  );
}
