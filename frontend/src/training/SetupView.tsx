import { useEffect, useState } from 'react';
import {
  IconAlertTriangle,
  IconCircleCheck,
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
  type RunDetail,
  type RunSummary,
  type TrainingStatus,
} from '@/api';
import type { TrainingProject } from './access';
import { DataSourcePicker } from './DataSourcePicker';
import {
  effectiveSource,
  loadSettings,
  saveSettings,
  toConfig,
  toTuning,
  validateSettings,
  type SettingsValues,
  type SourceChoice,
} from './settings';
import { SettingsForm } from './SettingsForm';
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
  /** Called when the server says training is unavailable (503) or busy (409). */
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
}

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

  const {
    choice: effective,
    ref,
    upload,
    name: datasetName,
  } = effectiveSource(status, choice, datasets);

  const empty = doc.nodes.length === 0;
  const networkValid = !empty && analysis?.valid === true;
  const ownCheck = useCompatibility(doc, ref, networkValid && !fit);
  const compatibility = fit ? fit.compatibility : ownCheck;
  const check = compatibility.result;

  const builtinSource = choice.tab === 'builtin';
  const errors = validateSettings(settings);
  if (builtinSource) {
    // Built-in datasets always test on their official test images: the test share is not used.
    delete errors.test_split;
  }
  // Datasets made in the Data tab bring their own split; the share fields are not used.
  const predefinedSplit = choice.tab === 'upload' ? (upload?.info.predefined_split ?? null) : null;
  if (predefinedSplit) {
    delete errors.val_split;
    delete errors.test_split;
  }
  const blockedByActive = activeRun !== null && status.max_concurrent_runs <= 1;
  const canStart =
    networkValid &&
    ref !== null &&
    !compatibility.loading &&
    check?.ok !== false &&
    Object.keys(errors).length === 0 &&
    !blockedByActive;

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
      const tuning = toTuning(settings);
      const run = await api.training.runs.create({
        name: runName.trim() || networkName.trim() || 'Untitled network',
        graph: doc,
        dataset: ref,
        // A remembered device may not exist on this server (any more).
        config: toConfig(
          {
            ...settings,
            device: status.devices.some((d) => d.id === settings.device) ? settings.device : 'auto',
          },
          { builtin: builtinSource }
        ),
        ...(tuning ? { tuning } : {}),
        ...(project ? { project: project.id } : {}),
      });
      onStarted(run);
    } catch (error) {
      const message = (error as Error).message;
      if (error instanceof ApiError && error.status === 409) {
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
      />

      {startError && (
        <Alert
          color={startError.busy ? 'yellow' : 'red'}
          icon={<IconAlertTriangle size={18} />}
          title={startError.busy ? 'Another training is running' : 'Cannot start the training'}
        >
          <Text size="sm">{startError.message}</Text>
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
          {settings.search ? 'Start search' : 'Start training'}
        </Button>
      </Group>
    </Stack>
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
