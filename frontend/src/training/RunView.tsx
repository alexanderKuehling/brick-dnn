import { useEffect, useMemo, useRef, useState } from 'react';
import {
  IconAlertTriangle,
  IconArrowLeft,
  IconChevronDown,
  IconChevronRight,
  IconCircleCheck,
  IconDownload,
  IconFileCode,
  IconHourglassHigh,
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
  Code,
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
  type ClassicalMetrics,
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
import { kernelLines, kernelTerms } from '@/builder/kinds';
import { seriesColor } from '@/components/charts/colors';
import { formatPercentTick } from '@/components/chartScale';
import { LineChart, SERIES_COLORS, type LineSeries } from '@/components/LineChart';
import { ExplainButton } from '@/explain/ExplainButton';
import { AccountLink } from '@/usage/CostEstimate';
import { formatCpu } from '@/usage/format';
import {
  canDeleteRun,
  canStopRun,
  canTrain,
  NO_SCOPE,
  otherStarter,
  type TrainingScope,
} from './access';
import { isClassicalRun } from './classical';
import {
  describeParams,
  formatCompact,
  formatDate,
  formatDuration,
  formatFitTime,
  formatMetric,
  formatPercent,
  formatR2,
  formatSetting,
  isActive,
  isClassification,
  isSegmentation,
  preprocessingJson,
  queueText,
  reportsR2,
  searchedSettings,
  searchLabel,
  SETTING_LABELS,
  weightsNote,
  weightsSnippet,
} from './format';
import { PredictionPanel } from './predict/PredictionPanel';
import { KernelBoundsHint } from './KernelBounds';
import { confirmDeleteRun, confirmStopRun, StatusBadge } from './runActions';
import { ClassTable, PreviewGallery } from './SegmentationResults';
import classes from './Training.module.css';

interface RunViewProps {
  run: RunDetail;
  doc: GraphDocument;
  /** The user and the open network: who started the run and what the user may do. */
  scope?: TrainingScope;
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

/** A number with its name above it. */
function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <Text size="xs" c="dimmed">
        {label}
      </Text>
      <Text fw={600} size="lg" lh={1.3}>
        {value}
      </Text>
    </div>
  );
}

function Tile({
  label,
  value,
  hint,
  next,
}: {
  label: string;
  value: string;
  hint?: string;
  /** A second number next to the first one, e.g. the R² next to the RMSE. */
  next?: { label: string; value: string };
}) {
  return (
    <Paper withBorder p="xs" radius="sm">
      {next ? (
        <Group gap="xs" align="flex-start" justify="space-between">
          <Metric label={label} value={value} />
          <Metric label={next.label} value={next.value} />
        </Group>
      ) : (
        <Metric label={label} value={value} />
      )}
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
  scope = NO_SCOPE,
  onBack,
  onNewTraining,
  onUpdate,
  onDeleted,
  onLoadNetwork,
}: RunViewProps) {
  const state = run.state;
  const active = isActive(run.status);
  // Waiting for a training worker: nothing was trained yet.
  const queued = run.status === 'queued';
  const history = useMemo(() => state?.history ?? [], [state?.history]);
  const classification = isClassification(run, history);
  // A class for every pixel: mean IoU, Dice and pixel accuracy instead of accuracy or RMSE.
  const segmentation = isSegmentation(run, history);
  const r2 = reportsR2(run, history);
  const now = useNow(active);
  const [logOpened, log] = useDisclosure(false);
  const [tableOpened, table] = useDisclosure(false);
  const [usageOpened, usage] = useDisclosure(false);
  const [predictOpened, predict] = useDisclosure(false);
  const [downloading, setDownloading] = useState<'weights' | 'script' | null>(null);
  const logViewport = useRef<HTMLDivElement>(null);

  const base = scriptBase(run.name);
  // A classical model (scikit-learn) is fitted once and saved as a joblib file.
  const classicalRun = isClassicalRun(run);
  const modelFile = `${base}.${classicalRun ? 'joblib' : 'pt'}`;
  const by = otherStarter(run, scope);
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
  const keptEpoch = state?.weights_epoch ?? null;
  const keptNote = run.has_weights ? weightsNote(run) : null;
  const preprocessing = run.has_weights ? preprocessingJson(run) : null;

  // ---------------------------------------------------------------- actions
  const downloadWeights = async () => {
    setDownloading('weights');
    try {
      const file = await api.training.runs.downloadWeights(run.id);
      // Same base name as the script, so the snippet below works as it is.
      downloadBlob(modelFile, file.blob);
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
        {canTrain(scope) && (
          <Button
            variant="subtle"
            size="compact-sm"
            leftSection={<IconPlayerPlay size={14} />}
            onClick={onNewTraining}
          >
            New training
          </Button>
        )}
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
            by && `by ${by}`,
            run.dataset.name,
            tuning && searchLabel(tuning),
            state?.device_label,
            run.started_via?.name && `via “${run.started_via.name}”`,
            `${queued ? 'queued' : 'started'} ${formatDate(new Date(startedAt).toISOString())}`,
            formatDuration(elapsed),
            cpuUsage(run),
          ]
            .filter(Boolean)
            .join(' · ')}
        </Text>
      </div>

      <RunProgress
        run={run}
        headline={headline}
        overall={overall}
        stepShare={stepShare}
        classical={classicalRun}
      />

      {queued ? null : classicalRun ? (
        <ClassicalResults run={run} classification={classification} active={active} />
      ) : (
        <>
          <SimpleGrid cols={{ base: 2, xs: 3 }} spacing="xs" verticalSpacing="xs">
            <Tile label="Training loss" value={formatMetric(metrics?.train_loss)} />
            <Tile label="Validation loss" value={formatMetric(metrics?.val_loss)} />
            {segmentation ? (
              <>
                <Tile
                  label="Validation IoU"
                  value={formatPercent(metrics?.val_iou)}
                  hint="mean over the classes"
                />
                <Tile
                  label="Validation Dice"
                  value={formatPercent(metrics?.val_dice)}
                  hint="mean over the classes"
                />
                <Tile
                  label="Pixel accuracy"
                  value={formatPercent(metrics?.val_pixel_accuracy)}
                  hint="validation"
                />
              </>
            ) : classification ? (
              <Tile label="Validation accuracy" value={formatPercent(metrics?.val_accuracy)} />
            ) : (
              <Tile
                label="Validation RMSE"
                value={formatMetric(metrics?.val_rmse)}
                next={r2 ? { label: 'R²', value: formatR2(metrics?.val_r2) } : undefined}
              />
            )}
            {tuning ? (
              <Tile
                label={active ? 'Best trial so far' : 'Best trial'}
                value={bestTrial ? String(bestTrial.trial) : '–'}
                hint={
                  bestTrial
                    ? typeof bestTrial.val_iou === 'number'
                      ? `IoU ${formatPercent(bestTrial.val_iou)}`
                      : typeof bestTrial.val_accuracy === 'number'
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
                    ? typeof best.val_iou === 'number'
                      ? `IoU ${formatPercent(best.val_iou)}`
                      : typeof best.val_accuracy === 'number'
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

          <TestResultCard
            run={run}
            classification={classification}
            segmentation={segmentation}
            r2={r2}
            active={active}
          />

          {tuning && (
            <TrialsTable
              tuning={tuning}
              trials={trials}
              total={trialsTotal}
              epochs={epochs}
              best={bestTrialNumber}
              active={active}
              classification={classification}
              segmentation={segmentation}
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
              {charts.iou.length > 0 && (
                <LineChart
                  title="Validation IoU"
                  x={charts.x}
                  series={charts.iou}
                  yBounds={[0, 1]}
                  formatYTick={formatPercentTick}
                  formatValue={(value) => formatPercent(value)}
                />
              )}
              {segmentation ? null : classification ? (
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
              {charts.r2.length > 0 && (
                <LineChart
                  title="Validation R²"
                  x={charts.x}
                  series={charts.r2}
                  yBounds={[-Infinity, 1]}
                  formatValue={formatR2}
                />
              )}
              <div>
                <ToggleButton opened={tableOpened} onClick={table.toggle}>
                  Values per epoch
                </ToggleButton>
                <Collapse expanded={tableOpened}>
                  <HistoryTable
                    history={history}
                    classification={classification}
                    segmentation={segmentation}
                    r2={r2}
                    kept={keptEpoch}
                  />
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

          {state?.segmentation && <ClassTable results={state.segmentation} />}
          <PreviewGallery run={run} />
        </>
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
        {active && canStopRun(run, scope) && (
          <Button
            color="red"
            variant="light"
            leftSection={<IconPlayerStop size={16} />}
            disabled={run.stop_requested}
            onClick={() => confirmStopRun(run, onUpdate)}
          >
            {queued ? 'Cancel' : run.stop_requested ? 'Stopping …' : 'Stop'}
          </Button>
        )}
        {run.has_weights && (
          <Button
            leftSection={<IconDownload size={16} />}
            loading={downloading === 'weights'}
            onClick={() => void downloadWeights()}
          >
            {classicalRun ? 'Model (.joblib)' : 'Weights (.pt)'}
          </Button>
        )}
        {preprocessing && (
          <Button
            variant="default"
            leftSection={<IconDownload size={16} />}
            onClick={() =>
              downloadText(`${base}.preprocessing.json`, preprocessing, 'application/json')
            }
          >
            Preprocessing (.json)
          </Button>
        )}
        {/* Explanations of the trained model (src/explain). */}
        {run.has_weights && !active && <ExplainButton run={run} />}
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
        {canDeleteRun(run, scope) && (
          <Button
            variant="subtle"
            color="red"
            leftSection={<IconTrash size={16} />}
            onClick={() => confirmDeleteRun(run, onDeleted)}
          >
            Delete
          </Button>
        )}
      </Group>
      {keptNote && (
        <Text size="xs" c="dimmed" mt={-8}>
          {keptNote}
        </Text>
      )}

      {run.has_weights && !active && (
        <div>
          <ToggleButton opened={predictOpened} onClick={predict.toggle}>
            Try the model: predict
          </ToggleButton>
          <Collapse expanded={predictOpened}>
            {predictOpened && <PredictionPanel runId={run.id} runName={run.name} />}
          </Collapse>
        </div>
      )}

      {run.has_weights && (
        <div>
          <ToggleButton opened={usageOpened} onClick={usage.toggle}>
            {classicalRun ? 'How to use the model' : 'How to use the weights'}
          </ToggleButton>
          <Collapse expanded={usageOpened}>
            {classicalRun ? (
              <Text size="sm" c="dimmed" mb="xs">
                Download the model and load it in Python with joblib (it needs scikit-learn). It is
                the whole pipeline: it standardises the features itself. The script predicts with it
                too:{' '}
                <Code>
                  python {base}.py --predict new.csv --save {modelFile}
                </Code>
              </Text>
            ) : (
              <Text size="sm" c="dimmed" mb="xs">
                Download the weights and the script, put both in one folder and load them in Python
                {preprocessing
                  ? '. New inputs need the same preprocessing as the training data; its values are also in the preprocessing download (.json).'
                  : ':'}
              </Text>
            )}
            <ScrollArea type="auto">
              <CodeView code={weightsSnippet(run, `${base}.py`, modelFile)} />
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

const STOP_TITLES = {
  budget: 'Stopped by the compute budget',
  run_limit: 'Stopped at the limit per training',
};

/** "2.3 CPU min" used (of its limit while it runs); null before anything was counted. */
function cpuUsage(run: RunDetail): string | null {
  const usage = run.usage;
  if (!usage || (usage.cpu_minutes <= 0 && usage.limit_cpu_minutes === null)) {
    return null;
  }
  return isActive(run.status) && usage.limit_cpu_minutes !== null
    ? `${formatCpu(usage.cpu_minutes, '')} of ${formatCpu(usage.limit_cpu_minutes)}`
    : formatCpu(usage.cpu_minutes);
}

function RunProgress({
  run,
  headline,
  overall,
  stepShare,
  classical = false,
}: {
  run: RunDetail;
  /** E.g. "Epoch 3 of 10" or "Trial 2 of 10 · Epoch 3 of 5". */
  headline: string;
  overall: number;
  stepShare: number;
  /** A classical model: fitted once, without progress. */
  classical?: boolean;
}) {
  const state = run.state;
  // The API keeps run.message in sync with state.json while the process runs, and sets it
  // itself when the process died or had to be killed (state.json is stale then).
  const message = run.message || state?.message;
  const progress = state?.progress ?? null;

  if (run.status === 'queued') {
    return <QueueNotice run={run} />;
  }
  if (run.status === 'starting' || run.status === 'preparing') {
    return (
      <Group gap="xs" wrap="nowrap">
        <Loader size="sm" />
        <Text size="sm">{message || 'Preparing the data …'}</Text>
      </Group>
    );
  }
  if (run.status === 'running' && classical) {
    return (
      <Box>
        <Group gap="xs" wrap="nowrap">
          <Loader size="sm" />
          <Text size="sm" fw={600}>
            {message || 'Fitting the model …'}
          </Text>
        </Group>
        <Text size="xs" c="dimmed" mt={4}>
          {run.stop_requested
            ? 'Stopping …'
            : 'scikit-learn fits the model in one go, without progress in between. Stopping ends the fit at once; nothing is saved then.'}
        </Text>
      </Box>
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
    const reason = run.usage?.stop_reason;
    return (
      <Alert
        color="orange"
        icon={<IconInfoCircle size={18} />}
        title={reason === 'budget' || reason === 'run_limit' ? STOP_TITLES[reason] : undefined}
      >
        {message || 'Training was stopped.'}
      </Alert>
    );
  }
  // A queued run whose budget was gone when its turn came never started.
  const noBudget = run.usage?.stop_reason === 'budget';
  return (
    <Alert
      color="red"
      icon={<IconAlertTriangle size={18} />}
      title={noBudget ? 'Not started: no compute budget left' : 'Training failed'}
    >
      {message || 'Something went wrong.'}
      {noBudget && (
        <div>
          <AccountLink />
        </div>
      )}
      {(state?.log.length ?? 0) > 0 && (
        <Text size="xs" mt={4}>
          The log below may tell more.
        </Text>
      )}
    </Alert>
  );
}

/** What the server says about a run by default while it waits. */
const QUEUED_MESSAGE = 'Waiting for a free training slot.';

/**
 * A run that waits for a training worker: its position, when it probably starts (the
 * server's rough estimate, updated with every poll) and what holds it up.
 */
function QueueNotice({ run }: { run: RunDetail }) {
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

/**
 * The results of a classical model, fitted once: how long the fit took and on how many
 * rows, what it found (the kernel and log marginal likelihood of a Gaussian process, the
 * support vectors of an SVM) and its scores on the training, validation and test rows.
 */
function ClassicalResults({
  run,
  classification,
  active,
}: {
  run: RunDetail;
  classification: boolean;
  active: boolean;
}) {
  const state = run.state;
  const fit = state?.classical ?? null;
  const entry = state?.history[0] ?? null;
  const metrics = run.metrics;
  const validation: ClassicalMetrics | null = entry
    ? {
        loss: entry.val_loss,
        accuracy: entry.val_accuracy ?? null,
        rmse: entry.val_rmse ?? null,
        r2: entry.val_r2 ?? null,
      }
    : metrics && (metrics.val_loss !== null || metrics.val_accuracy !== null)
      ? {
          loss: metrics.val_loss,
          accuracy: metrics.val_accuracy,
          rmse: metrics.val_rmse,
          r2: metrics.val_r2 ?? null,
        }
      : null;
  const test: ClassicalMetrics | null = state?.test
    ? {
        loss: state.test.loss,
        accuracy: state.test.accuracy,
        rmse: state.test.rmse,
        r2: state.test.r2 ?? null,
      }
    : metrics &&
        [metrics.test_loss, metrics.test_accuracy, metrics.test_rmse].some(
          (v) => typeof v === 'number'
        )
      ? {
          loss: metrics.test_loss ?? null,
          accuracy: metrics.test_accuracy ?? null,
          rmse: metrics.test_rmse ?? null,
          r2: metrics.test_r2 ?? null,
        }
      : null;
  const testSize = state?.test_size ?? null;
  const gaussianProcess = Boolean(
    fit && (fit.initial_kernel !== null || fit.kernel !== null || fit.estimator.startsWith('Gauss'))
  );
  const rows = fit?.rows ?? state?.train_size ?? null;
  const sizeText = (size: number | null | undefined) =>
    typeof size === 'number' ? size.toLocaleString('en-US') : '–';

  const splits: { label: string; metrics: ClassicalMetrics | null; size: number | null }[] = [
    { label: 'Training', metrics: fit?.train ?? null, size: rows },
    { label: 'Validation', metrics: validation, size: state?.val_size ?? null },
  ];
  if (test || testSize !== null) {
    splits.push({ label: 'Test', metrics: test, size: testSize });
  }
  const supportShare = fit?.support_vectors != null && rows ? fit.support_vectors / rows : null;

  return (
    <Stack gap="sm">
      <SimpleGrid cols={{ base: 2, xs: 3 }} spacing="xs" verticalSpacing="xs">
        <Tile label="Fit time" value={formatFitTime(fit?.fit_seconds)} />
        <Tile
          label="Rows fitted on"
          value={sizeText(rows)}
          hint={
            fit?.subsampled_from
              ? `subsampled from ${sizeText(fit.subsampled_from)} (limit ${sizeText(fit.max_rows)})`
              : fit
                ? `at most ${sizeText(fit.max_rows)} on this server`
                : undefined
          }
        />
        {gaussianProcess ? (
          <Tile
            label="Log marginal likelihood"
            value={formatMetric(fit?.log_marginal_likelihood)}
            hint="of the fitted kernel; higher fits better"
          />
        ) : (
          <Tile
            label="Support vectors"
            value={fit?.support_vectors != null ? sizeText(fit.support_vectors) : '–'}
            hint={
              supportShare !== null ? `${formatPercent(supportShare, 0)} of the rows` : undefined
            }
          />
        )}
      </SimpleGrid>

      {entry || test ? (
        <Table fz="sm" aria-label="Results of the fitted model" style={{ whiteSpace: 'nowrap' }}>
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Rows</Table.Th>
              {classification ? (
                <>
                  <Table.Th>Accuracy</Table.Th>
                  <Table.Th>Loss</Table.Th>
                </>
              ) : (
                <>
                  <Table.Th>RMSE</Table.Th>
                  <Table.Th>R²</Table.Th>
                </>
              )}
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {splits.map(({ label, metrics: values, size }) => (
              <Table.Tr key={label}>
                <Table.Td>
                  {label}
                  <Text span size="xs" c="dimmed">
                    {' '}
                    · {sizeText(size)}
                  </Text>
                </Table.Td>
                {classification ? (
                  <>
                    <Table.Td fw={600}>{formatPercent(values?.accuracy)}</Table.Td>
                    <Table.Td>{formatMetric(values?.loss)}</Table.Td>
                  </>
                ) : (
                  <>
                    <Table.Td fw={600}>{formatMetric(values?.rmse)}</Table.Td>
                    <Table.Td fw={600}>{formatR2(values?.r2)}</Table.Td>
                  </>
                )}
              </Table.Tr>
            ))}
          </Table.Tbody>
        </Table>
      ) : (
        active && (
          <Text size="sm" c="dimmed">
            The results appear when the fit is done.
          </Text>
        )
      )}
      {(entry || test) && (
        <Text size="xs" c="dimmed" mt={-6}>
          {classification
            ? 'Loss: cross-entropy of the predicted class probabilities.'
            : 'RMSE in the units of the target; R² = 1 is a perfect fit, 0 is no better than the mean.'}
          {test ? ' The test rows were measured once, after the fit.' : ''}
        </Text>
      )}

      {gaussianProcess && fit?.at_bounds?.length ? (
        <KernelBoundsHint bounds={fit.at_bounds} />
      ) : null}
      {gaussianProcess && fit?.kernel && <KernelBlock label="Fitted kernel" code={fit.kernel} />}
      {gaussianProcess && fit?.initial_kernel && (
        <KernelBlock label="Starting kernel" code={fit.initial_kernel} dimmed />
      )}
    </Stack>
  );
}

/** A kernel as scikit-learn writes it: one line per class (classifiers) and per term. */
function KernelBlock({
  label,
  code,
  dimmed = false,
}: {
  label: string;
  code: string;
  dimmed?: boolean;
}) {
  const lines = kernelLines(code).flatMap((line) => kernelTerms(line));
  return (
    <div>
      <Text size="xs" c="dimmed" mb={4}>
        {label}
      </Text>
      <Code
        block
        aria-label={label}
        c={dimmed ? 'dimmed' : undefined}
        style={{ whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}
      >
        {lines.join('\n')}
      </Code>
    </div>
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
  const r2 = [
    series(
      'val_r2',
      'Validation',
      SERIES_COLORS[1],
      history.map((h) => h.val_r2)
    ),
  ].filter((s) => has(s.values));
  // Segmentation: three shares of the validation pixels, on one scale from 0 to 100 %.
  const iou = [
    series(
      'val_iou',
      'Mean IoU',
      SERIES_COLORS[1],
      history.map((h) => h.val_iou)
    ),
    series(
      'val_dice',
      'Dice',
      seriesColor(2),
      history.map((h) => h.val_dice)
    ),
    series(
      'val_pixel_accuracy',
      'Pixel accuracy',
      seriesColor(3),
      history.map((h) => h.val_pixel_accuracy)
    ),
  ].filter((s) => has(s.values));
  return { x, loss, accuracy, rmse, r2, iou };
}

function HistoryTable({
  history,
  classification,
  segmentation = false,
  r2,
  kept,
}: {
  history: HistoryEntry[];
  classification: boolean;
  /** Mean IoU, Dice and pixel accuracy instead of an RMSE. */
  segmentation?: boolean;
  /** Add the validation R² (regressions). */
  r2: boolean;
  /** The epoch whose weights are kept (shown in bold). */
  kept: number | null;
}) {
  return (
    <ScrollArea.Autosize mah={260} type="auto">
      <Table striped stickyHeader fz="xs" ff="monospace" style={{ whiteSpace: 'nowrap' }}>
        <Table.Thead>
          <Table.Tr>
            <Table.Th>Epoch</Table.Th>
            <Table.Th>Train loss</Table.Th>
            <Table.Th>Val. loss</Table.Th>
            {segmentation ? (
              <>
                <Table.Th>Val. IoU</Table.Th>
                <Table.Th>Val. Dice</Table.Th>
                <Table.Th>Pixel acc.</Table.Th>
              </>
            ) : classification ? (
              <>
                <Table.Th>Train acc.</Table.Th>
                <Table.Th>Val. acc.</Table.Th>
              </>
            ) : (
              <Table.Th>Val. RMSE</Table.Th>
            )}
            {r2 && <Table.Th>Val. R²</Table.Th>}
            <Table.Th>Time</Table.Th>
          </Table.Tr>
        </Table.Thead>
        <Table.Tbody>
          {history.map((entry) => (
            <Table.Tr key={entry.epoch} fw={entry.epoch === kept ? 700 : undefined}>
              <Table.Td>{entry.epoch}</Table.Td>
              <Table.Td>{formatMetric(entry.train_loss)}</Table.Td>
              <Table.Td>{formatMetric(entry.val_loss)}</Table.Td>
              {segmentation ? (
                <>
                  <Table.Td>{formatPercent(entry.val_iou)}</Table.Td>
                  <Table.Td>{formatPercent(entry.val_dice)}</Table.Td>
                  <Table.Td>{formatPercent(entry.val_pixel_accuracy)}</Table.Td>
                </>
              ) : classification ? (
                <>
                  <Table.Td>{formatPercent(entry.train_accuracy)}</Table.Td>
                  <Table.Td>{formatPercent(entry.val_accuracy)}</Table.Td>
                </>
              ) : (
                <Table.Td>{formatMetric(entry.val_rmse)}</Table.Td>
              )}
              {r2 && <Table.Td>{formatR2(entry.val_r2)}</Table.Td>}
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
  segmentation = false,
  r2,
  active,
}: {
  run: RunDetail;
  classification: boolean;
  /** Show the mean IoU with Dice and pixel accuracy. */
  segmentation?: boolean;
  /** Show the R² next to the RMSE (regressions). */
  r2: boolean;
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
          r2: metrics.test_r2 ?? null,
          iou: metrics.test_iou ?? null,
          dice: metrics.test_dice ?? null,
          pixel_accuracy: metrics.test_pixel_accuracy ?? null,
        }
      : null;
  const result = state?.test ?? fromMetrics;
  const size = state?.test_size ?? null;
  if (!result && size === null && !((run.config.test_split ?? 0) > 0)) {
    return null;
  }

  const rmse = typeof result?.rmse === 'number';
  const label = segmentation
    ? 'Test IoU'
    : classification
      ? 'Test accuracy'
      : rmse
        ? 'Test RMSE'
        : 'Test loss';
  const value = segmentation
    ? formatPercent(result?.iou)
    : classification
      ? formatPercent(result?.accuracy)
      : formatMetric(rmse ? result?.rmse : result?.loss);
  const samples = size !== null ? `${size.toLocaleString('en-US')} test samples` : 'The test set';
  let note: string;
  if (result) {
    const loss =
      segmentation || classification || rmse ? `loss ${formatMetric(result.loss)} · ` : '';
    note = `${loss}${samples}, measured once after training`;
  } else if (active) {
    note = `${samples} ${size !== null ? 'are' : 'is'} held out until training has finished.`;
  } else {
    note = 'Not measured: the training did not finish.';
  }
  return (
    <Paper withBorder p="xs" radius="sm">
      <Group justify="space-between" wrap="nowrap" gap="sm">
        <Group gap="md" wrap="nowrap" align="flex-start" style={{ flexShrink: 0 }}>
          <Metric label={label} value={value} />
          {r2 && <Metric label="R²" value={formatR2(result?.r2)} />}
          {segmentation && <Metric label="Dice" value={formatPercent(result?.dice)} />}
          {segmentation && (
            <Metric label="Pixel acc." value={formatPercent(result?.pixel_accuracy)} />
          )}
        </Group>
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
  segmentation = false,
}: {
  tuning: TuningConfig;
  trials: TrialEntry[];
  total: number;
  epochs: number;
  best: number | null;
  active: boolean;
  classification: boolean;
  /** Show the mean IoU of every trial. */
  segmentation?: boolean;
}) {
  const settings = searchedSettings(tuning);
  const showRmse = !classification && trials.some((t) => typeof t.val_rmse === 'number');
  const showR2 = !classification && trials.some((t) => typeof t.val_r2 === 'number');
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
                {showR2 && <Table.Th>Val. R²</Table.Th>}
                {segmentation && <Table.Th>Val. IoU</Table.Th>}
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
                    {showR2 && <Table.Td>{formatR2(entry.val_r2)}</Table.Td>}
                    {segmentation && <Table.Td>{formatPercent(entry.val_iou)}</Table.Td>}
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
