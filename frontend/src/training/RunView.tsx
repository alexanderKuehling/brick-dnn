import { useEffect, useMemo, useRef, useState } from 'react';
import {
  IconAlertTriangle,
  IconArrowLeft,
  IconChevronDown,
  IconChevronRight,
  IconCircleCheck,
  IconDownload,
  IconFileCode,
  IconInfoCircle,
  IconPlayerPlay,
  IconPlayerStop,
  IconReplace,
  IconTrash,
} from '@tabler/icons-react';
import {
  Alert,
  Box,
  Button,
  Collapse,
  Group,
  Loader,
  Paper,
  Progress,
  ScrollArea,
  SimpleGrid,
  Stack,
  Table,
  Text,
  Title,
} from '@mantine/core';
import { useDisclosure } from '@mantine/hooks';
import { notifications } from '@mantine/notifications';
import { api, type GraphDocument, type HistoryEntry, type RunDetail } from '@/api';
import { CodeView } from '@/builder/CodeView';
import { downloadBlob, downloadText } from '@/builder/download';
import { formatCount } from '@/builder/format';
import { semanticKey, slugify } from '@/builder/graph';
import { formatPercentTick } from '@/components/chartScale';
import { LineChart, SERIES_COLORS, type LineSeries } from '@/components/LineChart';
import {
  formatCompact,
  formatDate,
  formatDuration,
  formatMetric,
  formatPercent,
  isActive,
  isClassification,
  weightsSnippet,
} from './format';
import { confirmDeleteRun, confirmStopRun, StatusBadge } from './runActions';
import classes from './Training.module.css';

interface RunViewProps {
  run: RunDetail;
  doc: GraphDocument;
  onBack: () => void;
  onNewTraining: () => void;
  onUpdate: (run: RunDetail) => void;
  onDeleted: (id: string) => void;
  onLoadNetwork: (graph: GraphDocument, name: string) => void;
}

/** Re-renders every second while `running` so elapsed times stay current. */
function useNow(running: boolean): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!running) {
      return undefined;
    }
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [running]);
  return now;
}

function Tile({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <Paper withBorder p="xs" radius="sm">
      <Text size="xs" c="dimmed">
        {label}
      </Text>
      <Text fw={600} size="lg" lh={1.3}>
        {value}
      </Text>
      {hint && (
        <Text size="xs" c="dimmed">
          {hint}
        </Text>
      )}
    </Paper>
  );
}

/** Module name for the downloaded script (must be a valid Python identifier). */
function scriptBase(name: string): string {
  const slug = slugify(name);
  return /^[a-z_]/.test(slug) ? slug : `model_${slug}`;
}

export function RunView({
  run,
  doc,
  onBack,
  onNewTraining,
  onUpdate,
  onDeleted,
  onLoadNetwork,
}: RunViewProps) {
  const state = run.state;
  const active = isActive(run.status);
  const history = useMemo(() => state?.history ?? [], [state?.history]);
  const classification = isClassification(run, history);
  const now = useNow(active);
  const [logOpened, log] = useDisclosure(false);
  const [tableOpened, table] = useDisclosure(false);
  const [usageOpened, usage] = useDisclosure(false);
  const [downloading, setDownloading] = useState<'weights' | 'script' | null>(null);
  const logViewport = useRef<HTMLDivElement>(null);

  const base = scriptBase(run.name);
  const sameNetwork = useMemo(() => semanticKey(run.graph) === semanticKey(doc), [run.graph, doc]);

  const logLines = state?.log ?? [];
  useEffect(() => {
    const viewport = logViewport.current;
    if (logOpened && viewport) {
      viewport.scrollTop = viewport.scrollHeight;
    }
  }, [logOpened, logLines.length]);

  // ---------------------------------------------------------------- derived values
  const startedAt = state?.started_at ? state.started_at * 1000 : Date.parse(run.created_at);
  const finishedAt = state?.finished_at
    ? state.finished_at * 1000
    : run.finished_at
      ? Date.parse(run.finished_at)
      : null;
  const elapsed = ((finishedAt ?? now) - startedAt) / 1000;
  const progress = state?.progress ?? null;
  const epochs = state?.epochs || run.epochs;
  const done = run.epochs_done;
  const stepShare = progress && progress.steps > 0 ? progress.step / progress.steps : 0;
  const overall = epochs > 0 ? Math.min(100, ((done + stepShare) / epochs) * 100) : 0;
  const currentEpoch = progress?.epoch ?? Math.min(done + 1, epochs);
  const metrics = run.metrics;
  const best = run.best ?? state?.best ?? null;
  const epochSeconds = history.map((h) => h.seconds).filter((s): s is number => !!s);
  const meanEpoch = epochSeconds.length
    ? epochSeconds.reduce((a, b) => a + b, 0) / epochSeconds.length
    : null;

  const charts = useMemo(() => chartSeries(history, classification), [history, classification]);

  // ---------------------------------------------------------------- actions
  const downloadWeights = async () => {
    setDownloading('weights');
    try {
      const file = await api.training.runs.downloadWeights(run.id);
      // Same base name as the script, so the snippet below works as it is.
      downloadBlob(`${base}.pt`, file.blob);
    } catch (error) {
      notifications.show({
        color: 'red',
        title: 'Download failed',
        message: (error as Error).message,
      });
    } finally {
      setDownloading(null);
    }
  };

  const downloadScript = async () => {
    setDownloading('script');
    try {
      const result = await api.generate(run.graph, {
        class_name: 'NeuralNetwork',
        include_main: true,
        include_training: false,
        filename: base,
      });
      downloadText(`${base}.py`, result.code, 'text/x-python');
    } catch (error) {
      notifications.show({
        color: 'red',
        title: 'Download failed',
        message: (error as Error).message,
      });
    } finally {
      setDownloading(null);
    }
  };

  // ---------------------------------------------------------------- render
  return (
    <Stack gap="md">
      <Group justify="space-between" wrap="nowrap" gap="xs">
        <Button
          variant="subtle"
          color="gray"
          size="compact-sm"
          leftSection={<IconArrowLeft size={14} />}
          onClick={onBack}
        >
          All runs
        </Button>
        <Button
          variant="subtle"
          size="compact-sm"
          leftSection={<IconPlayerPlay size={14} />}
          onClick={onNewTraining}
        >
          New training
        </Button>
      </Group>

      <div>
        <Group gap="xs" wrap="nowrap" justify="space-between" align="flex-start">
          <Title order={4} style={{ minWidth: 0, overflowWrap: 'anywhere' }}>
            {run.name}
          </Title>
          <StatusBadge status={run.status} />
        </Group>
        <Text size="xs" c="dimmed" mt={4}>
          {[
            run.dataset.name,
            state?.device_label,
            `started ${formatDate(new Date(startedAt).toISOString())}`,
            formatDuration(elapsed),
          ]
            .filter(Boolean)
            .join(' · ')}
        </Text>
      </div>

      <RunProgress
        run={run}
        currentEpoch={currentEpoch}
        epochs={epochs}
        overall={overall}
        stepShare={stepShare}
      />

      <SimpleGrid cols={{ base: 2, xs: 3 }} spacing="xs" verticalSpacing="xs">
        <Tile label="Training loss" value={formatMetric(metrics?.train_loss)} />
        <Tile label="Validation loss" value={formatMetric(metrics?.val_loss)} />
        {classification ? (
          <Tile label="Validation accuracy" value={formatPercent(metrics?.val_accuracy)} />
        ) : (
          <Tile label="Validation RMSE" value={formatMetric(metrics?.val_rmse)} />
        )}
        <Tile
          label="Best epoch"
          value={best ? String(best.epoch) : '–'}
          hint={
            best
              ? typeof best.val_accuracy === 'number'
                ? `accuracy ${formatPercent(best.val_accuracy)}`
                : `loss ${formatMetric(best.val_loss)}`
              : undefined
          }
        />
        {active && progress?.samples_per_second ? (
          <Tile
            label="Speed"
            value={`${formatCompact(progress.samples_per_second)}/s`}
            hint="samples"
          />
        ) : (
          <Tile label="Time per epoch" value={formatDuration(meanEpoch)} />
        )}
        <Tile label="Parameters" value={formatCount(run.total_params ?? state?.total_params)} />
      </SimpleGrid>

      {history.length > 0 ? (
        <Stack gap="lg">
          <LineChart title="Loss" x={charts.x} series={charts.loss} yBounds={[0, Infinity]} />
          {classification ? (
            <LineChart
              title="Accuracy"
              x={charts.x}
              series={charts.accuracy}
              yBounds={[0, 1]}
              formatYTick={formatPercentTick}
              formatValue={(value) => formatPercent(value)}
            />
          ) : (
            charts.rmse.length > 0 && (
              <LineChart
                title="Validation RMSE"
                x={charts.x}
                series={charts.rmse}
                yBounds={[0, Infinity]}
              />
            )
          )}
          <div>
            <ToggleButton opened={tableOpened} onClick={table.toggle}>
              Values per epoch
            </ToggleButton>
            <Collapse in={tableOpened}>
              <HistoryTable history={history} classification={classification} />
            </Collapse>
          </div>
        </Stack>
      ) : (
        active && (
          <Text size="sm" c="dimmed">
            The charts appear when the first epoch is done.
          </Text>
        )
      )}

      {logLines.length > 0 && (
        <div>
          <ToggleButton opened={logOpened} onClick={log.toggle}>
            Log ({logLines.length} {logLines.length === 1 ? 'line' : 'lines'})
          </ToggleButton>
          <Collapse in={logOpened}>
            <ScrollArea.Autosize mah={240} type="auto" viewportRef={logViewport}>
              <pre className={classes.log}>{logLines.join('\n')}</pre>
            </ScrollArea.Autosize>
          </Collapse>
        </div>
      )}

      <Group gap="xs">
        {active && (
          <Button
            color="red"
            variant="light"
            leftSection={<IconPlayerStop size={16} />}
            disabled={run.stop_requested}
            onClick={() => confirmStopRun(run, onUpdate)}
          >
            {run.stop_requested ? 'Stopping …' : 'Stop'}
          </Button>
        )}
        {run.has_weights && (
          <Button
            leftSection={<IconDownload size={16} />}
            loading={downloading === 'weights'}
            onClick={() => void downloadWeights()}
          >
            Weights (.pt)
          </Button>
        )}
        <Button
          variant="default"
          leftSection={<IconFileCode size={16} />}
          loading={downloading === 'script'}
          onClick={() => void downloadScript()}
        >
          Script (.py)
        </Button>
        {!sameNetwork && (
          <Button
            variant="default"
            leftSection={<IconReplace size={16} />}
            onClick={() => onLoadNetwork(run.graph, run.name)}
          >
            Load into editor
          </Button>
        )}
        <Button
          variant="subtle"
          color="red"
          leftSection={<IconTrash size={16} />}
          onClick={() => confirmDeleteRun(run, onDeleted)}
        >
          Delete
        </Button>
      </Group>

      {run.has_weights && (
        <div>
          <ToggleButton opened={usageOpened} onClick={usage.toggle}>
            How to use the weights
          </ToggleButton>
          <Collapse in={usageOpened}>
            <Text size="sm" c="dimmed" mb="xs">
              Download the weights and the script, put both in one folder and load them in Python:
            </Text>
            <ScrollArea type="auto">
              <CodeView code={weightsSnippet(run, `${base}.py`, `${base}.pt`)} />
            </ScrollArea>
          </Collapse>
        </div>
      )}
    </Stack>
  );
}

function ToggleButton({
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

function RunProgress({
  run,
  currentEpoch,
  epochs,
  overall,
  stepShare,
}: {
  run: RunDetail;
  currentEpoch: number;
  epochs: number;
  overall: number;
  stepShare: number;
}) {
  const state = run.state;
  // The API keeps run.message in sync with state.json while the process runs, and sets it
  // itself when the process died or had to be killed (state.json is stale then).
  const message = run.message || state?.message;
  const progress = state?.progress ?? null;

  if (run.status === 'starting' || run.status === 'preparing') {
    return (
      <Group gap="xs" wrap="nowrap">
        <Loader size="sm" />
        <Text size="sm">{message || 'Preparing the data …'}</Text>
      </Group>
    );
  }
  if (run.status === 'running') {
    return (
      <Box>
        <Group justify="space-between" mb={4}>
          <Text size="sm" fw={600}>
            Epoch {currentEpoch} of {epochs}
          </Text>
          <Text size="xs" c="dimmed">
            {Math.floor(overall)}%
          </Text>
        </Group>
        <Progress value={overall} aria-label="Training progress" />
        {progress && (
          <>
            <Text size="xs" c="dimmed" mt="xs" mb={4}>
              Step {progress.step.toLocaleString('en-US')} of{' '}
              {progress.steps.toLocaleString('en-US')} in this epoch
              {typeof progress.loss === 'number' && ` · loss ${formatMetric(progress.loss)}`}
            </Text>
            <Progress
              value={stepShare * 100}
              size="xs"
              color="gray"
              aria-label="Progress of the current epoch"
            />
          </>
        )}
        {run.stop_requested && (
          <Text size="sm" c="dimmed" mt="xs">
            Stopping after the current batch …
          </Text>
        )}
      </Box>
    );
  }
  if (run.status === 'completed') {
    return (
      <Alert color="green" icon={<IconCircleCheck size={18} />}>
        {message || 'Training finished.'}
      </Alert>
    );
  }
  if (run.status === 'stopped') {
    return (
      <Alert color="orange" icon={<IconInfoCircle size={18} />}>
        {message || 'Training was stopped.'}
      </Alert>
    );
  }
  return (
    <Alert color="red" icon={<IconAlertTriangle size={18} />} title="Training failed">
      {message || 'Something went wrong.'}
      {(state?.log.length ?? 0) > 0 && (
        <Text size="xs" mt={4}>
          The log below may tell more.
        </Text>
      )}
    </Alert>
  );
}

function chartSeries(history: HistoryEntry[], classification: boolean) {
  const x = history.map((entry) => entry.epoch);
  const has = (values: (number | null | undefined)[]) =>
    values.some((value) => typeof value === 'number');
  const series = (
    id: string,
    label: string,
    color: string,
    values: (number | null | undefined)[]
  ) => ({ id, label, color, values }) as LineSeries;

  const loss = [
    series(
      'train_loss',
      'Training',
      SERIES_COLORS[0],
      history.map((h) => h.train_loss)
    ),
    series(
      'val_loss',
      'Validation',
      SERIES_COLORS[1],
      history.map((h) => h.val_loss)
    ),
  ].filter((s) => has(s.values));
  const accuracy = classification
    ? [
        series(
          'train_acc',
          'Training',
          SERIES_COLORS[0],
          history.map((h) => h.train_accuracy)
        ),
        series(
          'val_acc',
          'Validation',
          SERIES_COLORS[1],
          history.map((h) => h.val_accuracy)
        ),
      ].filter((s) => has(s.values))
    : [];
  const rmse = [
    series(
      'val_rmse',
      'Validation',
      SERIES_COLORS[1],
      history.map((h) => h.val_rmse)
    ),
  ].filter((s) => has(s.values));
  return { x, loss, accuracy, rmse };
}

function HistoryTable({
  history,
  classification,
}: {
  history: HistoryEntry[];
  classification: boolean;
}) {
  return (
    <ScrollArea.Autosize mah={260} type="auto">
      <Table striped stickyHeader fz="xs" ff="monospace" style={{ whiteSpace: 'nowrap' }}>
        <Table.Thead>
          <Table.Tr>
            <Table.Th>Epoch</Table.Th>
            <Table.Th>Train loss</Table.Th>
            <Table.Th>Val. loss</Table.Th>
            {classification ? (
              <>
                <Table.Th>Train acc.</Table.Th>
                <Table.Th>Val. acc.</Table.Th>
              </>
            ) : (
              <Table.Th>Val. RMSE</Table.Th>
            )}
            <Table.Th>Time</Table.Th>
          </Table.Tr>
        </Table.Thead>
        <Table.Tbody>
          {history.map((entry) => (
            <Table.Tr key={entry.epoch}>
              <Table.Td>{entry.epoch}</Table.Td>
              <Table.Td>{formatMetric(entry.train_loss)}</Table.Td>
              <Table.Td>{formatMetric(entry.val_loss)}</Table.Td>
              {classification ? (
                <>
                  <Table.Td>{formatPercent(entry.train_accuracy)}</Table.Td>
                  <Table.Td>{formatPercent(entry.val_accuracy)}</Table.Td>
                </>
              ) : (
                <Table.Td>{formatMetric(entry.val_rmse)}</Table.Td>
              )}
              <Table.Td>{formatDuration(entry.seconds)}</Table.Td>
            </Table.Tr>
          ))}
        </Table.Tbody>
      </Table>
    </ScrollArea.Autosize>
  );
}
