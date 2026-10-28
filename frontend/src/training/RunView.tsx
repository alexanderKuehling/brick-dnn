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
  Badge,
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
  Tooltip,
} from '@mantine/core';
import { useDisclosure } from '@mantine/hooks';
import { notifications } from '@mantine/notifications';
import {
  api,
  type GraphDocument,
  type HistoryEntry,
  type RunDetail,
  type TestResult,
  type TrialEntry,
  type TuningConfig,
} from '@/api';
import { CodeView } from '@/builder/CodeView';
import { downloadBlob, downloadText } from '@/builder/download';
import { formatCount } from '@/builder/format';
import { semanticKey, slugify } from '@/builder/graph';
import { formatPercentTick } from '@/components/chartScale';
import { LineChart, SERIES_COLORS, type LineSeries } from '@/components/LineChart';
import {
  describeParams,
  formatCompact,
  formatDate,
  formatDuration,
  formatMetric,
  formatPercent,
  formatSetting,
  isActive,
  isClassification,
  searchedSettings,
  searchLabel,
  SETTING_LABELS,
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
  const tuning = run.tuning ?? state?.tuning ?? null;
  const trials = state?.trials ?? [];
  const trialsTotal = state?.trials_total ?? tuning?.trials ?? 0;
  const runningTrial = state?.trial ?? null;
  // In a search the history belongs to the running trial (the database only learns about a
  // new trial after its first epoch).
  const done = tuning && state ? history.length : run.epochs_done;
  const stepShare = progress && progress.steps > 0 ? progress.step / progress.steps : 0;
  const currentEpoch = progress?.epoch ?? Math.min(done + 1, epochs);
  let overall = epochs > 0 ? Math.min(100, ((done + stepShare) / epochs) * 100) : 0;
  let headline = `Epoch ${currentEpoch} of ${epochs}`;
  if (tuning && trialsTotal > 0) {
    const share =
      runningTrial !== null && epochs > 0 ? Math.min(1, (done + stepShare) / epochs) : 0;
    overall = Math.min(100, ((run.trials_done + share) / trialsTotal) * 100);
    headline =
      runningTrial !== null
        ? `Trial ${runningTrial} of ${trialsTotal} · Epoch ${currentEpoch} of ${epochs}`
        : `Trial ${Math.min(run.trials_done + 1, trialsTotal)} of ${trialsTotal}`;
  }
  if (run.message.startsWith('Testing')) {
    headline = run.message;
  }
  const bestTrial = run.best_trial;
  const bestTrialNumber = state?.best_trial ?? bestTrial?.trial ?? null;
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
            tuning && searchLabel(tuning),
            state?.device_label,
            `started ${formatDate(new Date(startedAt).toISOString())}`,
            formatDuration(elapsed),
          ]
            .filter(Boolean)
            .join(' · ')}
        </Text>
      </div>

      <RunProgress run={run} headline={headline} overall={overall} stepShare={stepShare} />

      <SimpleGrid cols={{ base: 2, xs: 3 }} spacing="xs" verticalSpacing="xs">
        <Tile label="Training loss" value={formatMetric(metrics?.train_loss)} />
        <Tile label="Validation loss" value={formatMetric(metrics?.val_loss)} />
        {classification ? (
          <Tile label="Validation accuracy" value={formatPercent(metrics?.val_accuracy)} />
        ) : (
          <Tile label="Validation RMSE" value={formatMetric(metrics?.val_rmse)} />
        )}
        {tuning ? (
          <Tile
            label={active ? 'Best trial so far' : 'Best trial'}
            value={bestTrial ? String(bestTrial.trial) : '–'}
            hint={
              bestTrial
                ? typeof bestTrial.val_accuracy === 'number'
                  ? `accuracy ${formatPercent(bestTrial.val_accuracy)}`
                  : `loss ${formatMetric(bestTrial.val_loss)}`
                : undefined
            }
          />
        ) : (
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
        )}
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

      <TestResultCard run={run} classification={classification} active={active} />

      {tuning && (
        <TrialsTable
          tuning={tuning}
          trials={trials}
          total={trialsTotal}
          epochs={epochs}
          best={bestTrialNumber}
          active={active}
          classification={classification}
        />
      )}

      {history.length > 0 ? (
        <Stack gap="lg">
          {tuning && (
            <Text size="xs" c="dimmed" mb={-8}>
              {runningTrial !== null
                ? `Curves of trial ${runningTrial} (running)`
                : bestTrialNumber !== null
                  ? `Curves of the best trial (${bestTrialNumber})`
                  : null}
            </Text>
          )}
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
            <Collapse expanded={tableOpened}>
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
          <Collapse expanded={logOpened}>
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
          <Collapse expanded={usageOpened}>
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
  headline,
  overall,
  stepShare,
}: {
  run: RunDetail;
  /** E.g. "Epoch 3 of 10" or "Trial 2 of 10 · Epoch 3 of 5". */
  headline: string;
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
            {headline}
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

// --------------------------------------------------------------------------
// Test set and hyperparameter search
// --------------------------------------------------------------------------

/** The final model on the test set; before the end, how many samples are held out. */
function TestResultCard({
  run,
  classification,
  active,
}: {
  run: RunDetail;
  classification: boolean;
  active: boolean;
}) {
  const state = run.state;
  const metrics = run.metrics;
  const fromMetrics: TestResult | null =
    metrics &&
    [metrics.test_loss, metrics.test_accuracy, metrics.test_rmse].some((v) => typeof v === 'number')
      ? {
          loss: metrics.test_loss ?? null,
          accuracy: metrics.test_accuracy ?? null,
          rmse: metrics.test_rmse ?? null,
        }
      : null;
  const result = state?.test ?? fromMetrics;
  const size = state?.test_size ?? null;
  if (!result && size === null && !((run.config.test_split ?? 0) > 0)) {
    return null;
  }

  const rmse = typeof result?.rmse === 'number';
  const label = classification ? 'Test accuracy' : rmse ? 'Test RMSE' : 'Test loss';
  const value = classification
    ? formatPercent(result?.accuracy)
    : formatMetric(rmse ? result?.rmse : result?.loss);
  const samples = size !== null ? `${size.toLocaleString('en-US')} test samples` : 'The test set';
  let note: string;
  if (result) {
    const loss = classification || rmse ? `loss ${formatMetric(result.loss)} · ` : '';
    note = `${loss}${samples}, measured once after training`;
  } else if (active) {
    note = `${samples} ${size !== null ? 'are' : 'is'} held out until training has finished.`;
  } else {
    note = 'Not measured: the training did not finish.';
  }
  return (
    <Paper withBorder p="xs" radius="sm">
      <Group justify="space-between" wrap="nowrap" gap="sm">
        <div style={{ flexShrink: 0 }}>
          <Text size="xs" c="dimmed">
            {label}
          </Text>
          <Text fw={600} size="lg" lh={1.3}>
            {value}
          </Text>
        </div>
        <Text size="xs" c="dimmed" ta="right">
          {note}
        </Text>
      </Group>
    </Paper>
  );
}

const STATUS_TEXT = { completed: 'Done', running: 'Running', stopped: 'Stopped', failed: 'Failed' };

function TrialsTable({
  tuning,
  trials,
  total,
  epochs,
  best,
  active,
  classification,
}: {
  tuning: TuningConfig;
  trials: TrialEntry[];
  total: number;
  epochs: number;
  best: number | null;
  active: boolean;
  classification: boolean;
}) {
  const settings = searchedSettings(tuning);
  const showRmse = !classification && trials.some((t) => typeof t.val_rmse === 'number');
  const winner = trials.find((t) => t.trial === best) ?? null;
  const finished = trials.filter((t) => t.status === 'completed' || t.status === 'failed').length;

  return (
    <div>
      <Group justify="space-between" gap="xs" mb={4}>
        <Text size="sm" fw={600}>
          Trials
        </Text>
        <Text size="xs" c="dimmed">
          {tuning.method === 'grid' ? 'Grid search' : 'Random search'} · {finished} of {total}{' '}
          finished
        </Text>
      </Group>
      {winner && (
        <Text size="sm" mb="xs">
          {active ? 'Best so far' : 'Best'}: trial {winner.trial} ({describeParams(winner.params)})
          {active ? '' : '. Its weights are kept.'}
        </Text>
      )}
      {trials.length === 0 ? (
        <Text size="sm" c="dimmed">
          The first trial starts when the data is ready.
        </Text>
      ) : (
        <ScrollArea.Autosize mah={260} type="auto">
          <Table stickyHeader fz="xs" style={{ whiteSpace: 'nowrap' }} aria-label="Trials">
            <Table.Thead>
              <Table.Tr>
                <Table.Th>Trial</Table.Th>
                {settings.map((name) => (
                  <Table.Th key={name}>{SETTING_LABELS[name]}</Table.Th>
                ))}
                <Table.Th>Val. loss</Table.Th>
                {classification && <Table.Th>Val. acc.</Table.Th>}
                {showRmse && <Table.Th>Val. RMSE</Table.Th>}
                <Table.Th>Status</Table.Th>
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {trials.map((entry) => {
                const isBest = entry.trial === best;
                return (
                  <Table.Tr
                    key={entry.trial}
                    fw={isBest ? 700 : undefined}
                    bg={isBest ? 'var(--mantine-primary-color-light)' : undefined}
                  >
                    <Table.Td>{entry.trial}</Table.Td>
                    {settings.map((name) => (
                      <Table.Td key={name}>{formatSetting(name, entry.params[name])}</Table.Td>
                    ))}
                    <Table.Td>{formatMetric(entry.val_loss)}</Table.Td>
                    {classification && <Table.Td>{formatPercent(entry.val_accuracy)}</Table.Td>}
                    {showRmse && <Table.Td>{formatMetric(entry.val_rmse)}</Table.Td>}
                    <Table.Td>
                      <TrialStatus entry={entry} best={isBest} epochs={epochs} active={active} />
                    </Table.Td>
                  </Table.Tr>
                );
              })}
            </Table.Tbody>
          </Table>
        </ScrollArea.Autosize>
      )}
    </div>
  );
}

function TrialStatus({
  entry,
  best,
  epochs,
  active,
}: {
  entry: TrialEntry;
  best: boolean;
  epochs: number;
  /** False when the run has ended: a trial still marked as running was interrupted. */
  active: boolean;
}) {
  if (entry.status === 'failed') {
    return (
      <Tooltip label={entry.message || 'Failed'} multiline w={260} withArrow>
        <Text span size="xs" c="red" td="underline" style={{ textDecorationStyle: 'dotted' }}>
          Failed
        </Text>
      </Tooltip>
    );
  }
  if (entry.status === 'running' && !active) {
    return (
      <Text span size="xs" c="dimmed">
        Interrupted
      </Text>
    );
  }
  if (entry.status === 'running') {
    return (
      <Text span size="xs" c="blue">
        Epoch {Math.min(entry.epochs_done + 1, epochs)}/{epochs}
      </Text>
    );
  }
  if (best) {
    return (
      <Badge size="xs" variant="filled">
        Best
      </Badge>
    );
  }
  return (
    <Text span size="xs" c="dimmed">
      {STATUS_TEXT[entry.status]}
    </Text>
  );
}
