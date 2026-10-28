import { useEffect, useState } from 'react';
import {
  IconAlertTriangle,
  IconCircleCheck,
  IconInfoCircle,
  IconPlayerPlay,
} from '@tabler/icons-react';
import { Alert, Button, Divider, Group, List, Loader, Stack, Text } from '@mantine/core';
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
import { DataSourcePicker } from './DataSourcePicker';
import {
  datasetRef,
  loadSettings,
  saveSettings,
  toConfig,
  toTuning,
  validateSettings,
  type SettingsValues,
  type SourceChoice,
} from './settings';
import { SettingsForm } from './SettingsForm';
import { useCompatibility } from './useCompatibility';
import type { DataController } from './useDatasets';

interface SetupViewProps {
  status: TrainingStatus;
  doc: GraphDocument;
  networkName: string;
  analysis: Analysis | null;
  activeRun: RunSummary | null;
  /** Data source and uploads; kept by useTraining so they survive closing the panel. */
  data: DataController;
  onStarted: (run: RunDetail) => void;
  onOpenRun: (id: string) => void;
  onShowProblems: () => void;
  /** Called when the server says training is unavailable (503) or busy (409). */
  onRefresh: () => void;
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
  analysis,
  activeRun,
  data,
  onStarted,
  onOpenRun,
  onShowProblems,
  onRefresh,
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

  // A remembered upload may have been deleted, a built-in dataset may be unavailable.
  // Without a (valid) choice, the first built-in dataset and the newest upload are used.
  const availableBuiltins = status.builtin_datasets.filter((d) => d.available);
  const builtin =
    availableBuiltins.find((d) => d.id === choice.builtin) ?? availableBuiltins[0] ?? null;
  const upload = datasets?.find((d) => d.id === choice.upload) ?? datasets?.[0] ?? null;
  const effective: SourceChoice = {
    ...choice,
    builtin: builtin ? builtin.id : null,
    upload: upload ? upload.id : null,
  };
  const ref = datasetRef(effective);

  const empty = doc.nodes.length === 0;
  const networkValid = !empty && analysis?.valid === true;
  const compatibility = useCompatibility(doc, ref, networkValid);
  const check = compatibility.result;

  const builtinSource = choice.tab === 'builtin';
  const errors = validateSettings(settings);
  if (builtinSource) {
    // Built-in datasets have official test images: the test share is not used, and without
    // a test set they validate on those images instead of a share of the training data.
    delete errors.test_split;
    if (!settings.test_enabled) {
      delete errors.val_split;
    }
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
            “{activeRun.name}” is still training. You can start a new one when it has finished.
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
        uploadError={data.uploadError}
        onDismissUploadError={data.dismissUploadError}
        onUpload={data.upload}
        onDelete={deleteDataset}
      />

      {networkValid && ref && <CheckResult {...compatibility} />}

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

      <Group justify="flex-end">
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

function CheckResult({ result, loading, error }: ReturnType<typeof useCompatibility>) {
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
    </Stack>
  );
}
