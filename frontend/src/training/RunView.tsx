import { useEffect, useId, useMemo, useRef, useState, type ReactNode } from 'react';
import {
  IconAlertTriangle,
  IconArrowLeft,
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
  type ClassicalRunState,
  type GraphDocument,
  type HistoryEntry,
  type HubCapabilities,
  type RunDetail,
  type TestResult,
  type TrialEntry,
  type TuningConfig,
} from '@/api';
import { CodeView } from '@/builder/CodeView';
import { downloadBlob, downloadText } from '@/builder/download';
import { formatCount } from '@/builder/format';
import { semanticKey, slugify } from '@/builder/graph';
import { isBoosting, kernelLines, kernelTerms } from '@/builder/kinds';
import { seriesColor } from '@/components/charts/colors';
import { formatPercentTick } from '@/components/chartScale';
import { InfoHint } from '@/components/InfoHint';
import { LineChart, SERIES_COLORS, type LineSeries } from '@/components/LineChart';
import { ElsewhereBadge } from '@/experiments/ElsewhereBadge';
import { ExplainButton } from '@/explain/ExplainButton';
import { RunExplanations } from '@/explain/RunExplanations';
import { PublishButton } from '@/huggingface/PublishButton';
import { currentLocale, i18n, Trans, useTranslation } from '@/i18n';
import { AccountLink } from '@/usage/CostEstimate';
import {
  canDeleteRun,
  canSetThreshold,
  canStopRun,
  canTrain,
  NO_SCOPE,
  otherStarter,
  type TrainingScope,
} from './access';
import { isClassicalRun, isHybridRun, MODEL_KINDS } from './classical';
import { isEnsemble } from './ensemble';
import { EnsembleRunView } from './EnsembleRunView';
import { EvaluationResults, type ThresholdProps } from './EvaluationResults';
import {
  describeParams,
  formatCompact,
  formatDate,
  formatDuration,
  formatFitTime,
  formatMetric,
  formatParam,
  formatPercent,
  formatR2,
  isActive,
  isClassification,
  isSegmentation,
  lossText,
  paramLabel,
  preprocessingJson,
  reportsR2,
  searchedParams,
  searchLabel,
  weightsNote,
  weightsSnippet,
} from './format';
import { KernelBoundsHint } from './KernelBounds';
import { PredictionPanel } from './predict/PredictionPanel';
import { confirmDeleteRun, confirmStopRun, StatusBadge } from './runActions';
import { cpuUsage, QueueNotice, ToggleButton } from './runParts';
import { formatMetricValue, metricDirection, metricInfo, metricInSentence } from './search';
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
  /** Opens another run (the members of an ensemble). */
  onOpenRun?: (id: string) => void;
  /** Publishing to the Hugging Face Hub, if the server offers it. */
  hub?: HubCapabilities;
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
    <Paper p="xs" radius="sm" className={classes.tile}>
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

/** The heading of a part of the results (training curves, files …). */
export function RunSectionTitle({ children, id }: { children: ReactNode; id?: string }) {
  return (
    <Text component="h5" id={id} size="sm" fw={600} m={0}>
      {children}
    </Text>
  );
}

/** Binary classification: the class its decision threshold decides for (the second one). */
function positiveClass(run: RunDetail): string | undefined {
  const evaluation = run.state?.evaluation;
  const names =
    evaluation?.task === 'classification' ? evaluation.class_names : run.state?.class_names;
  return names?.[1];
}

/** Module name for the downloaded script (must be a valid Python identifier). */
function scriptBase(name: string): string {
  const slug = slugify(name);
  return /^[a-z_]/.test(slug) ? slug : `model_${slug}`;
}

/** A run: a training of a network or a classical model, or an ensemble of runs. */
export function RunView(props: RunViewProps) {
  return isEnsemble(props.run) ? <EnsembleRunView {...props} /> : <NetworkRunView {...props} />;
}

function NetworkRunView({
  run,
  doc,
  scope = NO_SCOPE,
  onBack,
  onNewTraining,
  onUpdate,
  onDeleted,
  onLoadNetwork,
  hub,
}: RunViewProps) {
  useTranslation();
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
  const curvesHeading = useId();

  const base = scriptBase(run.name);
  // A classical model (scikit-learn) is fitted once and saved as a joblib file; a hybrid
  // network downloads as a ZIP file with its neural layers and its classical model.
  const classicalRun = isClassicalRun(run);
  const hybridRun = isHybridRun(run);
  const modelFile = `${base}.${classicalRun ? 'joblib' : hybridRun ? 'zip' : 'pt'}`;
  const hybridFit = hybridRun ? (state?.classical ?? null) : null;
  const frozenFeatures = hybridRun && run.config.feature_training === 'frozen';
  const hybridModel =
    MODEL_KINDS[run.plan?.classical?.type ?? ''] ?? run.plan?.classical?.estimator ?? 'the model';
  const learnedFeatures = run.plan?.hybrid?.features ?? hybridFit?.hybrid?.features ?? null;
  const by = otherStarter(run, scope);
  // Binary classification: the threshold its predictions use; only who started it saves another.
  const threshold: ThresholdProps = {
    decisionThreshold: run.decision_threshold,
    onSaveThreshold: canSetThreshold(run, scope)
      ? async (value) => {
          const saved = await api.training.runs.setDecisionThreshold(run.id, value);
          onUpdate({ ...run, decision_threshold: saved.decision_threshold });
        }
      : undefined,
  };
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
  // A hybrid network that searches only its classical model learns the features once, before
  // the trials (which have no epochs); a search of its neural settings learns them per trial.
  const fittedTrials = hybridRun && !tuning?.neural_space;
  const learningFirst = fittedTrials && runningTrial === null && run.trials_done === 0;
  if (tuning && trialsTotal > 0 && !learningFirst) {
    const share =
      runningTrial !== null && epochs > 0 && !fittedTrials
        ? Math.min(1, (done + stepShare) / epochs)
        : 0;
    overall = Math.min(100, ((run.trials_done + share) / trialsTotal) * 100);
    headline =
      runningTrial !== null && !fittedTrials
        ? `Trial ${runningTrial} of ${trialsTotal} · Epoch ${currentEpoch} of ${epochs}`
        : `Trial ${runningTrial ?? Math.min(run.trials_done + 1, trialsTotal)} of ${trialsTotal}`;
  }
  if (run.message.startsWith('Testing') || run.message.startsWith('Evaluating')) {
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
  // Trained without validation data: only training values per epoch, the last epoch is kept.
  const noValidation = state?.val_size === 0;
  const trainAccuracy = metrics?.train_accuracy ?? history.at(-1)?.train_accuracy ?? null;
  const keptNote = run.has_weights ? weightsNote(run) : null;
  const loss = lossText(run);
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
        title: i18n.t('ui.training.RunView.downloadFailed'),
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
        title: i18n.t('ui.training.RunView.downloadFailed'),
        message: (error as Error).message,
      });
    } finally {
      setDownloading(null);
    }
  };

  // ---------------------------------------------------------------- render
  return (
    <Stack gap="lg">
      <Group justify="space-between" wrap="nowrap" gap="xs">
        <Button
          variant="subtle"
          color="gray"
          size="compact-sm"
          leftSection={<IconArrowLeft size={14} />}
          onClick={onBack}
        >
          {i18n.t('ui.training.RunView.allRuns')}
        </Button>
        {canTrain(scope) && (
          <Button
            variant="subtle"
            size="compact-sm"
            leftSection={<IconPlayerPlay size={14} />}
            onClick={onNewTraining}
          >
            {i18n.t('ui.training.RunView.newTraining')}
          </Button>
        )}
      </Group>

      <div>
        <Group gap="xs" wrap="nowrap" justify="space-between" align="flex-start">
          <Title order={4} style={{ minWidth: 0, overflowWrap: 'anywhere' }}>
            {run.name}
          </Title>
          <Group gap={6} wrap="nowrap">
            {run.external && <ElsewhereBadge external={run.external} />}
            <StatusBadge status={run.status} />
          </Group>
        </Group>
        {run.external && (
          <Text size="xs" c="dimmed" mt={4}>
            {i18n.t('experiments.elsewhere.runNote', { count: run.external.points })}
          </Text>
        )}
        <Text size="xs" c="dimmed" mt={4}>
          {[
            by && i18n.t('training.run.by', { user: by }),
            run.dataset.name,
            tuning && searchLabel(tuning),
            state?.device_label,
            run.started_via?.name && i18n.t('training.run.via', { name: run.started_via.name }),
            i18n.t(queued ? 'training.run.queued' : 'training.run.started', {
              date: formatDate(new Date(startedAt).toISOString()),
            }),
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
        <ClassicalResults
          run={run}
          classification={classification}
          active={active}
          threshold={threshold}
        />
      ) : (
        <>
          {hybridRun && (
            <div>
              <Title order={5}>{i18n.t('ui.training.RunView.phase1TheNeuralLayers')}</Title>
              <Text size="xs" c="dimmed">
                {frozenFeatures
                  ? i18n.t('ui.training.RunView.frozenRandomFeaturesTheNeural', {
                      value: learnedFeatures ?? i18n.t('ui.training.RunView.the'),
                    })
                  : i18n.t('ui.training.RunView.theyLearnFeaturesWithA', {
                      value: learnedFeatures ?? i18n.t('ui.training.RunView.the'),
                    })}
              </Text>
            </div>
          )}
          {!hybridRun && (
            <SimpleGrid cols={{ base: 2, xs: 3 }} spacing="xs" verticalSpacing="xs">
              <Tile
                label={i18n.t('ui.training.RunView.trainingLoss')}
                value={formatMetric(metrics?.train_loss)}
              />
              {noValidation ? (
                <Tile
                  label={i18n.t('ui.training.RunView.validation')}
                  value="None"
                  hint={i18n.t('ui.training.RunView.noValidationData')}
                />
              ) : (
                <Tile
                  label={i18n.t('ui.training.RunView.validationLoss')}
                  value={formatMetric(metrics?.val_loss)}
                />
              )}
              {noValidation ? (
                classification ? (
                  <Tile
                    label={i18n.t('ui.training.RunView.trainingAccuracy')}
                    value={formatPercent(trainAccuracy)}
                    hint={i18n.t('ui.training.RunView.onTheTrainingDataOptimistic')}
                  />
                ) : null
              ) : segmentation ? (
                <>
                  <Tile
                    label={i18n.t('ui.training.RunView.validationIou')}
                    value={formatPercent(metrics?.val_iou)}
                    hint={i18n.t('ui.training.RunView.meanOverTheClasses')}
                  />
                  <Tile
                    label={i18n.t('ui.training.RunView.validationDice')}
                    value={formatPercent(metrics?.val_dice)}
                    hint={i18n.t('ui.training.RunView.meanOverTheClasses')}
                  />
                  <Tile
                    label={i18n.t('ui.training.RunView.pixelAccuracy')}
                    value={formatPercent(metrics?.val_pixel_accuracy)}
                    hint="validation"
                  />
                </>
              ) : classification ? (
                <Tile
                  label={i18n.t('ui.training.RunView.validationAccuracy')}
                  value={formatPercent(metrics?.val_accuracy)}
                />
              ) : (
                <Tile
                  label={i18n.t('ui.training.RunView.validationRmse')}
                  value={formatMetric(metrics?.val_rmse)}
                  next={r2 ? { label: 'R²', value: formatR2(metrics?.val_r2) } : undefined}
                />
              )}
              {tuning ? (
                <Tile
                  label={
                    active
                      ? i18n.t('ui.training.RunView.bestTrialSoFar')
                      : i18n.t('ui.training.RunView.bestTrial')
                  }
                  value={bestTrial ? String(bestTrial.trial) : '–'}
                  hint={
                    bestTrial
                      ? tuning.metric && tuning.metric !== 'loss'
                        ? `${metricInfo(tuning.metric).label} ${formatMetricValue(tuning.metric, bestTrial.score)}`
                        : typeof bestTrial.val_iou === 'number'
                          ? `IoU ${formatPercent(bestTrial.val_iou)}`
                          : typeof bestTrial.val_accuracy === 'number'
                            ? `accuracy ${formatPercent(bestTrial.val_accuracy)}`
                            : `loss ${formatMetric(bestTrial.val_loss)}`
                      : undefined
                  }
                />
              ) : noValidation ? (
                <Tile
                  label={i18n.t('ui.training.RunView.keptEpoch')}
                  value={keptEpoch !== null ? String(keptEpoch) : '–'}
                  hint={i18n.t('ui.training.RunView.theLastOneNoValidation')}
                />
              ) : (
                <Tile
                  label={i18n.t('ui.training.RunView.bestEpoch')}
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
                  label={i18n.t('ui.training.RunView.speed')}
                  value={`${formatCompact(progress.samples_per_second)}/s`}
                  hint="samples"
                />
              ) : (
                <Tile
                  label={i18n.t('ui.training.RunView.timePerEpoch')}
                  value={formatDuration(meanEpoch)}
                />
              )}
              <Tile
                label={i18n.t('ui.training.RunView.parameters')}
                value={formatCount(run.total_params ?? state?.total_params)}
              />
            </SimpleGrid>
          )}

          {!hybridRun && (
            <TestResultCard
              run={run}
              classification={classification}
              segmentation={segmentation}
              r2={r2}
              active={active}
            />
          )}

          {tuning && !hybridRun && (
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
            <Stack gap="lg" component="section" aria-labelledby={curvesHeading}>
              <Group gap={4} wrap="nowrap">
                <RunSectionTitle id={curvesHeading}>
                  {i18n.t('ui.training.RunView.trainingCurves')}
                </RunSectionTitle>
                {/* Which loss the curves show matters when runs are compared: on demand. */}
                {loss && (
                  <InfoHint topic={i18n.t('ui.training.RunView.trainingCurves')}>
                    {i18n.t('ui.training.RunView.lossTheCurvesEarlyStopping', { loss })}
                  </InfoHint>
                )}
              </Group>
              {tuning && (
                <Text size="xs" c="dimmed" mb={-8}>
                  {runningTrial !== null
                    ? i18n.t('ui.training.RunView.curvesOfTrialRunning', { runningTrial })
                    : bestTrialNumber !== null
                      ? i18n.t('ui.training.RunView.curvesOfTheBestTrial', { bestTrialNumber })
                      : null}
                </Text>
              )}

              <LineChart
                title={i18n.t('ui.training.RunView.loss')}
                x={charts.x}
                series={charts.loss}
                yBounds={[0, Infinity]}
              />
              {charts.iou.length > 0 && (
                <LineChart
                  title={i18n.t('ui.training.RunView.validationIou')}
                  x={charts.x}
                  series={charts.iou}
                  yBounds={[0, 1]}
                  formatYTick={formatPercentTick}
                  formatValue={(value) => formatPercent(value)}
                />
              )}
              {segmentation ? null : classification ? (
                <LineChart
                  title={i18n.t('ui.training.RunView.accuracy')}
                  x={charts.x}
                  series={charts.accuracy}
                  yBounds={[0, 1]}
                  formatYTick={formatPercentTick}
                  formatValue={(value) => formatPercent(value)}
                />
              ) : (
                charts.rmse.length > 0 && (
                  <LineChart
                    title={i18n.t('ui.training.RunView.validationRmse')}
                    x={charts.x}
                    series={charts.rmse}
                    yBounds={[0, Infinity]}
                  />
                )
              )}
              {charts.r2.length > 0 && (
                <LineChart
                  title={i18n.t('ui.training.RunView.validationR')}
                  x={charts.x}
                  series={charts.r2}
                  yBounds={[-Infinity, 1]}
                  formatValue={formatR2}
                />
              )}
              <div>
                <ToggleButton opened={tableOpened} onClick={table.toggle}>
                  {i18n.t('ui.training.RunView.valuesPerEpoch')}
                </ToggleButton>
                <Collapse expanded={tableOpened}>
                  <HistoryTable
                    history={history}
                    classification={classification}
                    segmentation={segmentation}
                    r2={r2}
                    kept={keptEpoch}
                    validation={!noValidation}
                  />
                </Collapse>
              </div>
            </Stack>
          ) : (
            active &&
            !frozenFeatures && (
              <Text size="sm" c="dimmed">
                {i18n.t('ui.training.RunView.theChartsAppearWhenThe')}
              </Text>
            )
          )}

          {hybridRun && (
            <>
              <div>
                <Title order={5}>
                  {i18n.t('ui.training.RunView.phase2OnTheLearned', { hybridModel })}
                </Title>
                <Text size="xs" c="dimmed">
                  {tuning?.neural_space
                    ? i18n.t('ui.training.RunView.everyTrialOfTheSearch')
                    : i18n.t('ui.training.RunView.fittedOnTheFeaturesOf')}
                </Text>
              </div>
              <ClassicalResults
                run={run}
                classification={classification}
                active={active}
                threshold={threshold}
                hybrid
              />
            </>
          )}

          {!active && !hybridRun && (
            <EvaluationResults evaluation={state?.evaluation} threshold={threshold} />
          )}
          {state?.segmentation && <ClassTable results={state.segmentation} />}
          <PreviewGallery run={run} />
        </>
      )}

      {/* Previews of the explanations; beside the results when the panel is wide. */}
      {run.has_weights && !active && <RunExplanations run={run} />}

      {logLines.length > 0 && (
        <div>
          <ToggleButton opened={logOpened} onClick={log.toggle}>
            {i18n.t('ui.training.RunView.log', {
              length: logLines.length,
              value:
                logLines.length === 1
                  ? i18n.t('ui.training.RunView.line')
                  : i18n.t('ui.training.RunView.lines'),
            })}
          </ToggleButton>
          <Collapse expanded={logOpened}>
            <ScrollArea.Autosize mah={240} type="auto" viewportRef={logViewport}>
              <pre className={classes.log}>{logLines.join('\n')}</pre>
            </ScrollArea.Autosize>
          </Collapse>
        </div>
      )}

      {!active && (run.has_weights || run.graph.nodes.length > 0) && (
        <RunSectionTitle>{i18n.t('ui.training.RunView.modelAndFiles')}</RunSectionTitle>
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
            {queued
              ? i18n.t('ui.training.RunView.cancel')
              : run.stop_requested
                ? i18n.t('ui.training.RunView.stopping')
                : i18n.t('ui.training.RunView.stop')}
          </Button>
        )}
        {run.has_weights && (
          <Button
            leftSection={<IconDownload size={16} />}
            loading={downloading === 'weights'}
            onClick={() => void downloadWeights()}
          >
            {classicalRun
              ? i18n.t('ui.training.RunView.modelJoblib')
              : hybridRun
                ? i18n.t('ui.training.RunView.modelZip')
                : i18n.t('ui.training.RunView.weightsPt')}
          </Button>
        )}
        {/* Explanations of the trained model (src/explain). */}
        {run.has_weights && !active && <ExplainButton run={run} />}
        {preprocessing && (
          <Button
            variant="default"
            leftSection={<IconDownload size={16} />}
            onClick={() =>
              downloadText(`${base}.preprocessing.json`, preprocessing, 'application/json')
            }
          >
            {i18n.t('ui.training.RunView.preprocessingJson')}
          </Button>
        )}
        {run.graph.nodes.length > 0 && (
          <Button
            variant="default"
            leftSection={<IconFileCode size={16} />}
            loading={downloading === 'script'}
            onClick={() => void downloadScript()}
          >
            {i18n.t('ui.training.RunView.scriptPy')}
          </Button>
        )}
        {/* Who may stop a run may publish it (src/huggingface). */}
        {hub?.publish &&
          run.has_weights &&
          (run.status === 'completed' || run.status === 'stopped') &&
          canStopRun(run, scope) && <PublishButton run={run} hub={hub} />}
        {!sameNetwork && run.graph.nodes.length > 0 && (
          <Button
            variant="default"
            leftSection={<IconReplace size={16} />}
            onClick={() => onLoadNetwork(run.graph, run.name)}
          >
            {i18n.t('ui.training.RunView.loadIntoEditor')}
          </Button>
        )}
        {canDeleteRun(run, scope) && (
          <Button
            variant="subtle"
            color="red"
            leftSection={<IconTrash size={16} />}
            onClick={() => confirmDeleteRun(run, onDeleted)}
          >
            {i18n.t('ui.training.RunView.delete')}
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
            {i18n.t('ui.training.RunView.tryTheModelPredict')}
          </ToggleButton>
          <Collapse expanded={predictOpened}>
            {predictOpened && (
              <PredictionPanel
                runId={run.id}
                runName={run.name}
                decisionThreshold={run.decision_threshold}
                positiveClass={positiveClass(run)}
              />
            )}
          </Collapse>
        </div>
      )}

      {run.has_weights && (
        <div>
          <ToggleButton opened={usageOpened} onClick={usage.toggle}>
            {classicalRun
              ? i18n.t('ui.training.RunView.howToUseTheModel')
              : i18n.t('ui.training.RunView.howToUseTheWeights')}
          </ToggleButton>
          <Collapse expanded={usageOpened}>
            {hybridRun ? (
              <Text size="sm" c="dimmed" mb="xs">
                {i18n.t('ui.training.RunView.theZipFileHoldsThe', { base, base2: base })}
                <Code>
                  {i18n.t('ui.training.RunView.pythonPyPredictNewCsv', {
                    base,
                    base2: base,
                    base3: base,
                  })}
                </Code>
              </Text>
            ) : classicalRun ? (
              <Text size="sm" c="dimmed" mb="xs">
                <Trans
                  i18nKey="ui.training.RunView.downloadTheModelAndLoad"
                  values={{ base, modelFile }}
                  components={[<Code />]}
                />
              </Text>
            ) : (
              <Text size="sm" c="dimmed" mb="xs">
                {i18n.t('ui.training.RunView.downloadTheWeightsAndThe', {
                  value: preprocessing ? i18n.t('ui.training.RunView.newInputsNeedTheSame') : ':',
                })}
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

const STOP_TITLES = {
  get budget() {
    return i18n.t('ui.training.RunView.stoppedByTheComputeBudget');
  },
  get run_limit() {
    return i18n.t('ui.training.RunView.stoppedAtTheLimitPer');
  },
};

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
  useTranslation();
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
        <Text size="sm">{message || i18n.t('ui.training.RunView.preparingTheData')}</Text>
      </Group>
    );
  }
  if (run.status === 'running' && classical) {
    return (
      <Box>
        <Group gap="xs" wrap="nowrap">
          <Loader size="sm" />
          <Text size="sm" fw={600}>
            {message || i18n.t('ui.training.RunView.fittingTheModel')}
          </Text>
        </Group>
        <Text size="xs" c="dimmed" mt={4}>
          {run.stop_requested
            ? i18n.t('ui.training.RunView.stopping')
            : i18n.t('ui.training.RunView.scikitLearnFitsTheModel')}
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
        <Progress value={overall} aria-label={i18n.t('ui.training.RunView.trainingProgress')} />
        {progress && (
          <>
            <Text size="xs" c="dimmed" mt="xs" mb={4}>
              {i18n.t('ui.training.RunView.step')} {progress.step.toLocaleString(currentLocale())}{' '}
              {i18n.t('ui.training.RunView.of')} {progress.steps.toLocaleString(currentLocale())}{' '}
              {i18n.t('ui.training.RunView.inThisEpoch')}
              {typeof progress.loss === 'number' &&
                i18n.t('ui.training.RunView.loss3', { loss: formatMetric(progress.loss) })}
            </Text>
            <Progress
              value={stepShare * 100}
              size="xs"
              color="gray"
              aria-label={i18n.t('ui.training.RunView.progressOfTheCurrentEpoch')}
            />
          </>
        )}
        {run.stop_requested && (
          <Text size="sm" c="dimmed" mt="xs">
            {i18n.t('ui.training.RunView.stoppingAfterTheCurrentBatch')}
          </Text>
        )}
      </Box>
    );
  }
  if (run.status === 'completed') {
    return (
      <Alert color="green" icon={<IconCircleCheck size={18} />}>
        {message || i18n.t('ui.training.RunView.trainingFinished')}
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
        {message || i18n.t('ui.training.RunView.trainingWasStopped')}
      </Alert>
    );
  }
  // A queued run whose budget was gone when its turn came never started.
  const noBudget = run.usage?.stop_reason === 'budget';
  return (
    <Alert
      color="red"
      icon={<IconAlertTriangle size={18} />}
      title={
        noBudget
          ? i18n.t('ui.training.RunView.notStartedNoComputeBudget')
          : i18n.t('ui.training.RunView.trainingFailed')
      }
    >
      {message || i18n.t('ui.training.RunView.somethingWentWrong')}
      {noBudget && (
        <div>
          <AccountLink />
        </div>
      )}
      {(state?.log.length ?? 0) > 0 && (
        <Text size="xs" mt={4}>
          {i18n.t('ui.training.RunView.theLogBelowMayTell')}
        </Text>
      )}
    </Alert>
  );
}

/** What the server says about a run by default while it waits. */
/**
 * The models of an ensemble, each measured on its own (on the validation rows, or the
 * training rows without them), to compare them with the ensemble.
 */
function EnsembleMembers({
  members,
  classification,
  validation,
}: {
  members: NonNullable<ClassicalRunState['members']>;
  classification: boolean;
  validation: boolean;
}) {
  useTranslation();
  return (
    <Stack gap={4}>
      <Table
        fz="sm"
        aria-label={i18n.t('ui.training.RunView.modelsOfTheEnsemble')}
        style={{ whiteSpace: 'nowrap' }}
      >
        <Table.Thead>
          <Table.Tr>
            <Table.Th>{i18n.t('ui.training.RunView.model')}</Table.Th>
            {classification ? (
              <>
                <Table.Th>{i18n.t('ui.training.RunView.accuracy')}</Table.Th>
                <Table.Th>{i18n.t('ui.training.RunView.loss')}</Table.Th>
              </>
            ) : (
              <>
                <Table.Th>{i18n.t('ui.training.RunView.rmse')}</Table.Th>
                <Table.Th>R²</Table.Th>
              </>
            )}
          </Table.Tr>
        </Table.Thead>
        <Table.Tbody>
          {members.map((member) => (
            <Table.Tr key={member.node}>
              <Table.Td>
                <Text span size="sm" ff="monospace">
                  {member.name}
                </Text>
                <Text span size="xs" c="dimmed">
                  {' '}
                  · {member.label}
                </Text>
              </Table.Td>
              {classification ? (
                <>
                  <Table.Td>{formatPercent(member.metrics?.accuracy)}</Table.Td>
                  <Table.Td>{formatMetric(member.metrics?.loss)}</Table.Td>
                </>
              ) : (
                <>
                  <Table.Td>{formatMetric(member.metrics?.rmse)}</Table.Td>
                  <Table.Td>{formatR2(member.metrics?.r2)}</Table.Td>
                </>
              )}
            </Table.Tr>
          ))}
        </Table.Tbody>
      </Table>
      <Text size="xs" c="dimmed" mt={-6}>
        {i18n.t('ui.training.RunView.eachModelOfTheEnsemble', {
          value: validation
            ? i18n.t('ui.training.RunView.validation2')
            : i18n.t('ui.training.RunView.training2'),
        })}
      </Text>
    </Stack>
  );
}

/**
 * The results of a classical model, fitted once: how long the fit took and on how many
 * rows, what it found (the kernel and log marginal likelihood of a Gaussian process, the
 * support vectors of an SVM, the boosting iterations of gradient boosting, the models of an
 * ensemble) and its scores on the training, validation and test rows.
 */
function ClassicalResults({
  run,
  classification,
  active,
  threshold,
  hybrid = false,
}: {
  run: RunDetail;
  classification: boolean;
  active: boolean;
  threshold: ThresholdProps;
  /** The classical model of a hybrid network: the history is that of its features. */
  hybrid?: boolean;
}) {
  useTranslation();
  const state = run.state;
  const fit = state?.classical ?? null;
  // Once fitted: the model on the validation rows (a hybrid network keeps them apart from
  // the history, which has the curves of its features).
  const entry = hybrid
    ? fit?.train
      ? {
          val_loss: fit.validation?.loss ?? null,
          val_accuracy: fit.validation?.accuracy ?? null,
          val_rmse: fit.validation?.rmse ?? null,
          val_r2: fit.validation?.r2 ?? null,
        }
      : null
    : (state?.history[0] ?? null);
  const tuning = run.tuning ?? state?.tuning ?? null;
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
    typeof size === 'number' ? size.toLocaleString(currentLocale()) : '–';

  const splits: { label: string; metrics: ClassicalMetrics | null; size: number | null }[] = [
    { label: i18n.t('ui.training.RunView.training'), metrics: fit?.train ?? null, size: rows },
  ];
  // A validation split of 0: no validation rows to show.
  if (state?.val_size !== 0) {
    splits.push({
      label: i18n.t('ui.training.RunView.validation'),
      metrics: validation,
      size: state?.val_size ?? null,
    });
  }
  if (test || testSize !== null) {
    splits.push({ label: i18n.t('ui.training.RunView.test'), metrics: test, size: testSize });
  }
  const supportShare = fit?.support_vectors != null && rows ? fit.support_vectors / rows : null;
  const boosting = Boolean(fit && (fit.iterations != null || isBoosting(fit.estimator)));
  const iterations = fit?.iterations ?? null;
  const maxIterations = fit?.max_iterations ?? null;
  const members = fit?.members?.length ? fit.members : null;

  return (
    <Stack gap="sm">
      <SimpleGrid cols={{ base: 2, xs: 3 }} spacing="xs" verticalSpacing="xs">
        <Tile
          label={i18n.t('ui.training.RunView.fitTime')}
          value={formatFitTime(fit?.fit_seconds)}
        />
        <Tile
          label={i18n.t('ui.training.RunView.rowsFittedOn')}
          value={sizeText(rows)}
          hint={
            fit?.subsampled_from
              ? i18n.t('ui.training.RunView.subsampledFromLimit', {
                  sizeText: sizeText(fit.subsampled_from),
                  sizeText2: sizeText(fit.max_rows),
                })
              : fit
                ? i18n.t('ui.training.RunView.atMostOnThisServer', {
                    sizeText: sizeText(fit.max_rows),
                  })
                : undefined
          }
        />
        {members ? (
          <Tile
            label={i18n.t('ui.training.RunView.ensemble')}
            value={`${members.length} models`}
            hint={fit?.ensemble === 'stacking' ? 'stacking' : 'voting'}
          />
        ) : gaussianProcess ? (
          <Tile
            label={i18n.t('ui.training.RunView.logMarginalLikelihood')}
            value={formatMetric(fit?.log_marginal_likelihood)}
            hint={i18n.t('ui.training.RunView.ofTheFittedKernelHigher')}
          />
        ) : boosting ? (
          <Tile
            label={i18n.t('ui.training.RunView.boostingIterations')}
            value={sizeText(iterations)}
            hint={
              iterations !== null && maxIterations !== null
                ? iterations < maxIterations
                  ? i18n.t('ui.training.RunView.ofAtMostEarlyStopping', {
                      sizeText: sizeText(maxIterations),
                    })
                  : i18n.t('ui.training.RunView.ofAtMost', { sizeText: sizeText(maxIterations) })
                : undefined
            }
          />
        ) : (
          <Tile
            label={i18n.t('ui.training.RunView.supportVectors')}
            value={fit?.support_vectors != null ? sizeText(fit.support_vectors) : '–'}
            hint={
              supportShare !== null
                ? i18n.t('ui.training.RunView.ofTheRows', {
                    supportShare: formatPercent(supportShare, 0),
                  })
                : undefined
            }
          />
        )}
      </SimpleGrid>

      {entry || test ? (
        <Table
          fz="sm"
          aria-label={i18n.t('ui.training.RunView.resultsOfTheFittedModel')}
          style={{ whiteSpace: 'nowrap' }}
        >
          <Table.Thead>
            <Table.Tr>
              <Table.Th>{i18n.t('ui.training.RunView.rows')}</Table.Th>
              {classification ? (
                <>
                  <Table.Th>{i18n.t('ui.training.RunView.accuracy')}</Table.Th>
                  <Table.Th>{i18n.t('ui.training.RunView.loss')}</Table.Th>
                </>
              ) : (
                <>
                  <Table.Th>{i18n.t('ui.training.RunView.rmse')}</Table.Th>
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
            {i18n.t('ui.training.RunView.theResultsAppearWhenThe')}
          </Text>
        )
      )}
      {(entry || test) && (
        <Text size="xs" c="dimmed" mt={-6}>
          {classification
            ? i18n.t('ui.training.RunView.lossCrossEntropyOfThe')
            : i18n.t('ui.training.RunView.rmseInTheUnitsOf')}
          {test ? i18n.t('ui.training.RunView.theTestRowsWereMeasured') : ''}
        </Text>
      )}

      {tuning && (
        <TrialsTable
          tuning={tuning}
          trials={state?.trials ?? []}
          total={state?.trials_total ?? tuning.trials}
          epochs={tuning.neural_space ? state?.epochs || run.epochs : 1}
          best={state?.best_trial ?? run.best_trial?.trial ?? null}
          active={active}
          classification={classification}
          classical
          features={Boolean(tuning.neural_space)}
        />
      )}

      {members && members.some((member) => member.metrics) && (
        <EnsembleMembers
          members={members}
          classification={classification}
          validation={state?.val_size !== 0}
        />
      )}

      {!active && (
        <EvaluationResults evaluation={state?.evaluation} classical threshold={threshold} />
      )}

      {gaussianProcess && fit?.at_bounds?.length ? (
        <KernelBoundsHint bounds={fit.at_bounds} />
      ) : null}
      {gaussianProcess && fit?.kernel && (
        <KernelBlock label={i18n.t('ui.training.RunView.fittedKernel')} code={fit.kernel} />
      )}
      {gaussianProcess && fit?.initial_kernel && (
        <KernelBlock
          label={i18n.t('ui.training.RunView.startingKernel')}
          code={fit.initial_kernel}
          dimmed
        />
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
      i18n.t('training.run.training'),
      SERIES_COLORS[0],
      history.map((h) => h.train_loss)
    ),
    series(
      'val_loss',
      i18n.t('training.run.validation'),
      SERIES_COLORS[1],
      history.map((h) => h.val_loss)
    ),
  ].filter((s) => has(s.values));
  const accuracy = classification
    ? [
        series(
          'train_acc',
          i18n.t('training.run.training'),
          SERIES_COLORS[0],
          history.map((h) => h.train_accuracy)
        ),
        series(
          'val_acc',
          i18n.t('training.run.validation'),
          SERIES_COLORS[1],
          history.map((h) => h.val_accuracy)
        ),
      ].filter((s) => has(s.values))
    : [];
  const rmse = [
    series(
      'val_rmse',
      i18n.t('training.run.validation'),
      SERIES_COLORS[1],
      history.map((h) => h.val_rmse)
    ),
  ].filter((s) => has(s.values));
  const r2 = [
    series(
      'val_r2',
      i18n.t('training.run.validation'),
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
  validation = true,
}: {
  history: HistoryEntry[];
  classification: boolean;
  /** Mean IoU, Dice and pixel accuracy instead of an RMSE. */
  segmentation?: boolean;
  /** Add the validation R² (regressions). */
  r2: boolean;
  /** The epoch whose weights are kept (shown in bold). */
  kept: number | null;
  /** False for a run without validation data: only the training columns. */
  validation?: boolean;
}) {
  useTranslation();
  if (!validation) {
    return (
      <ScrollArea.Autosize mah={260} type="auto">
        <Table striped stickyHeader fz="xs" ff="monospace" style={{ whiteSpace: 'nowrap' }}>
          <Table.Thead>
            <Table.Tr>
              <Table.Th>{i18n.t('ui.training.RunView.epoch')}</Table.Th>
              <Table.Th>{i18n.t('ui.training.RunView.trainLoss')}</Table.Th>
              {classification && <Table.Th>{i18n.t('ui.training.RunView.trainAcc')}</Table.Th>}
              <Table.Th>{i18n.t('ui.training.RunView.time')}</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {history.map((entry) => (
              <Table.Tr key={entry.epoch} fw={entry.epoch === kept ? 700 : undefined}>
                <Table.Td>{entry.epoch}</Table.Td>
                <Table.Td>{formatMetric(entry.train_loss)}</Table.Td>
                {classification && <Table.Td>{formatPercent(entry.train_accuracy)}</Table.Td>}
                <Table.Td>{formatDuration(entry.seconds)}</Table.Td>
              </Table.Tr>
            ))}
          </Table.Tbody>
        </Table>
      </ScrollArea.Autosize>
    );
  }
  return (
    <ScrollArea.Autosize mah={260} type="auto">
      <Table striped stickyHeader fz="xs" ff="monospace" style={{ whiteSpace: 'nowrap' }}>
        <Table.Thead>
          <Table.Tr>
            <Table.Th>{i18n.t('ui.training.RunView.epoch')}</Table.Th>
            <Table.Th>{i18n.t('ui.training.RunView.trainLoss')}</Table.Th>
            <Table.Th>{i18n.t('ui.training.RunView.valLoss')}</Table.Th>
            {segmentation ? (
              <>
                <Table.Th>{i18n.t('ui.training.RunView.valIou')}</Table.Th>
                <Table.Th>{i18n.t('ui.training.RunView.valDice')}</Table.Th>
                <Table.Th>{i18n.t('ui.training.RunView.pixelAcc')}</Table.Th>
              </>
            ) : classification ? (
              <>
                <Table.Th>{i18n.t('ui.training.RunView.trainAcc')}</Table.Th>
                <Table.Th>{i18n.t('ui.training.RunView.valAcc')}</Table.Th>
              </>
            ) : (
              <Table.Th>{i18n.t('ui.training.RunView.valRmse')}</Table.Th>
            )}
            {r2 && <Table.Th>{i18n.t('ui.training.RunView.valR')}</Table.Th>}
            <Table.Th>{i18n.t('ui.training.RunView.time')}</Table.Th>
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
  useTranslation();
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
    ? i18n.t('ui.training.RunView.testIou')
    : classification
      ? i18n.t('ui.training.RunView.testAccuracy')
      : rmse
        ? i18n.t('ui.training.RunView.testRmse')
        : i18n.t('ui.training.RunView.testLoss');
  const value = segmentation
    ? formatPercent(result?.iou)
    : classification
      ? formatPercent(result?.accuracy)
      : formatMetric(rmse ? result?.rmse : result?.loss);
  const samples =
    size !== null ? `${size.toLocaleString(currentLocale())} test samples` : 'The test set';
  let note: string;
  if (result) {
    const loss =
      segmentation || classification || rmse ? `loss ${formatMetric(result.loss)} · ` : '';
    note = i18n.t('ui.training.RunView.measuredOnceAfterTraining', { loss, samples });
  } else if (active) {
    note = i18n.t('ui.training.RunView.heldOutUntilTrainingHas', {
      samples,
      value: size !== null ? i18n.t('ui.training.RunView.are') : i18n.t('ui.training.RunView.is'),
    });
  } else {
    note = i18n.t('ui.training.RunView.notMeasuredTheTrainingDid');
  }
  return (
    <Paper p="xs" radius="sm" className={classes.tile}>
      <Group justify="space-between" wrap="nowrap" gap="sm">
        <Group gap="md" wrap="nowrap" align="flex-start" style={{ flexShrink: 0 }}>
          <Metric label={label} value={value} />
          {r2 && <Metric label="R²" value={formatR2(result?.r2)} />}
          {segmentation && (
            <Metric
              label={i18n.t('ui.training.RunView.dice')}
              value={formatPercent(result?.dice)}
            />
          )}
          {segmentation && (
            <Metric
              label={i18n.t('ui.training.RunView.pixelAcc')}
              value={formatPercent(result?.pixel_accuracy)}
            />
          )}
        </Group>
        <Text size="xs" c="dimmed" ta="right">
          {note}
        </Text>
      </Group>
    </Paper>
  );
}

const STATUS_TEXT = {
  get completed() {
    return i18n.t('ui.training.RunView.done');
  },
  get running() {
    return i18n.t('ui.training.RunView.running');
  },
  get stopped() {
    return i18n.t('ui.training.RunView.stopped');
  },
  get failed() {
    return i18n.t('ui.training.RunView.failed');
  },
};

export function TrialsTable({
  tuning,
  trials,
  total,
  epochs,
  best,
  active,
  classification,
  segmentation = false,
  classical = false,
  features = false,
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
  /** A classical model: brick parameters, fitted once per trial (and fold). */
  classical?: boolean;
  /**
   * A hybrid network that searches its neural settings: every trial learns the features in
   * `epochs` epochs, then fits its classical model on them.
   */
  features?: boolean;
}) {
  useTranslation();
  const settings = searchedParams(tuning);
  const metric = tuning.metric ?? 'loss';
  // The loss, accuracy, RMSE and R² have columns of their own; other metrics get one.
  const scoreColumn = !['loss', 'accuracy', 'rmse', 'r2'].includes(metric);
  const showRmse = !classification && trials.some((t) => typeof t.val_rmse === 'number');
  const showR2 = !classification && trials.some((t) => typeof t.val_r2 === 'number');
  const winner = trials.find((t) => t.trial === best) ?? null;
  const finished = trials.filter((t) => t.status === 'completed' || t.status === 'failed').length;
  const folds = tuning.cv_folds ?? trials.find((t) => t.folds)?.folds ?? null;
  const where = folds ? `${folds}-fold cross-validation` : 'the validation data';
  const prefix = folds ? 'CV' : 'Val.';

  return (
    <div>
      <Group justify="space-between" gap="xs" mb={4}>
        <Text size="sm" fw={600}>
          {i18n.t('ui.training.RunView.trials')}
        </Text>
        <Text size="xs" c="dimmed">
          {i18n.t('ui.training.RunView.ofFinished', {
            value:
              tuning.method === 'grid'
                ? i18n.t('ui.training.RunView.gridSearch')
                : i18n.t('ui.training.RunView.randomSearch'),
            finished,
            total,
          })}
        </Text>
      </Group>
      <Text size="xs" c="dimmed" mb={4}>
        {i18n.t('ui.training.RunView.rankedByOn', {
          metricInSentence: metricInSentence(metric),
          where,
          metricDirection: metricDirection(metric),
        })}
      </Text>
      {winner && (
        <Text size="sm" mb="xs">
          {i18n.t('ui.training.RunView.trial', {
            value: active
              ? i18n.t('ui.training.RunView.bestSoFar')
              : i18n.t('ui.training.RunView.best'),
            trial: winner.trial,
            describeParams: describeParams(winner.params),
            value2: active
              ? ''
              : features
                ? i18n.t('ui.training.RunView.itsFeaturesAndModelAre')
                : classical
                  ? i18n.t('ui.training.RunView.itsModelIsKept')
                  : i18n.t('ui.training.RunView.itsWeightsAreKept'),
          })}
        </Text>
      )}
      {trials.length === 0 ? (
        <Text size="sm" c="dimmed">
          {i18n.t('ui.training.RunView.theFirstTrialStartsWhen')}
        </Text>
      ) : (
        <ScrollArea.Autosize mah={260} type="auto">
          <Table
            stickyHeader
            fz="xs"
            style={{ whiteSpace: 'nowrap' }}
            aria-label={i18n.t('ui.training.RunView.trials')}
          >
            <Table.Thead>
              <Table.Tr>
                <Table.Th>{i18n.t('ui.training.RunView.trial2')}</Table.Th>
                {settings.map((name) => (
                  <Table.Th key={name}>{paramLabel(name)}</Table.Th>
                ))}
                <Table.Th>{i18n.t('ui.training.RunView.loss2', { prefix })}</Table.Th>
                {classification && (
                  <Table.Th>{i18n.t('ui.training.RunView.acc', { prefix })}</Table.Th>
                )}
                {showRmse && <Table.Th>{i18n.t('ui.training.RunView.rmse2', { prefix })}</Table.Th>}
                {showR2 && <Table.Th>{prefix} R²</Table.Th>}
                {segmentation && (
                  <Table.Th>{i18n.t('ui.training.RunView.iou', { prefix })}</Table.Th>
                )}
                {scoreColumn && (
                  <Table.Th>
                    {prefix} {metricInfo(metric).label}
                  </Table.Th>
                )}
                <Table.Th>{i18n.t('ui.training.RunView.status')}</Table.Th>
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
                      <Table.Td key={name}>{formatParam(name, entry.params[name])}</Table.Td>
                    ))}
                    <Table.Td>{formatMetric(entry.val_loss)}</Table.Td>
                    {classification && <Table.Td>{formatPercent(entry.val_accuracy)}</Table.Td>}
                    {showRmse && <Table.Td>{formatMetric(entry.val_rmse)}</Table.Td>}
                    {showR2 && <Table.Td>{formatR2(entry.val_r2)}</Table.Td>}
                    {segmentation && <Table.Td>{formatPercent(entry.val_iou)}</Table.Td>}
                    {scoreColumn && <Table.Td>{formatMetricValue(metric, entry.score)}</Table.Td>}
                    <Table.Td>
                      <TrialStatus
                        entry={entry}
                        best={isBest}
                        epochs={epochs}
                        active={active}
                        classical={classical && !features}
                        features={features}
                      />
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
  classical = false,
  features = false,
}: {
  entry: TrialEntry;
  best: boolean;
  epochs: number;
  /** False when the run has ended: a trial still marked as running was interrupted. */
  active: boolean;
  /** Fitted once, without epochs. */
  classical?: boolean;
  /** Learns the features in `epochs` epochs, then fits a classical model. */
  features?: boolean;
}) {
  useTranslation();
  if (entry.status === 'failed') {
    return (
      <Tooltip
        label={entry.message || i18n.t('ui.training.RunView.failed')}
        multiline
        w={260}
        withArrow
      >
        <Text span size="xs" c="red" td="underline" style={{ textDecorationStyle: 'dotted' }}>
          {i18n.t('ui.training.RunView.failed')}
        </Text>
      </Tooltip>
    );
  }
  if (entry.status === 'running' && !active) {
    return (
      <Text span size="xs" c="dimmed">
        {i18n.t('ui.training.RunView.interrupted')}
      </Text>
    );
  }
  if (entry.status === 'running') {
    return (
      <Text span size="xs" c="blue">
        {classical || (features && entry.epochs_done >= epochs)
          ? i18n.t('ui.training.RunView.fitting')
          : i18n.t('ui.training.RunView.epoch2', {
              min: Math.min(entry.epochs_done + 1, epochs),
              epochs,
            })}
      </Text>
    );
  }
  if (best) {
    return (
      <Badge size="xs" variant="filled">
        {i18n.t('ui.training.RunView.best')}
      </Badge>
    );
  }
  return (
    <Text span size="xs" c="dimmed">
      {STATUS_TEXT[entry.status]}
    </Text>
  );
}
