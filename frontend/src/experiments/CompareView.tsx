import { useMemo, useState } from 'react';
import {
  Alert,
  Group,
  MultiSelect,
  SegmentedControl,
  Select,
  Stack,
  Switch,
  Tabs,
  Text,
} from '@mantine/core';
import type { ComparisonOptions, RunCurve, RunRow } from '@/api';
import { OverlayChart } from '@/components/charts/OverlayChart';
import { ParallelCoordinates, type ParallelAxis } from '@/components/charts/ParallelCoordinates';
import { ScatterPlot } from '@/components/charts/ScatterPlot';
import { useTranslation } from '@/i18n';
import { availableColumns, type Column } from './columns';
import {
  curveLabel,
  curveMetrics,
  differentLosses,
  differentTasks,
  formatCurveValue,
  formatElapsed,
  mainMetric,
  overlaySeries,
  overlayTable,
  type XAxis,
} from './curves';

/** The weight of the moving average when smoothing is on. */
export const SMOOTHING = 0.6;
/** Settings whose values span decades: the scatter plot shows them on a log axis. */
const LOG_SETTINGS = new Set(['setting:learning_rate', 'setting:weight_decay']);

export interface CompareViewProps {
  /** The runs to compare (the order gives their colours). */
  rows: RunRow[];
  curves: RunCurve[] | null;
  /** How many runs the curve charts take: the first ones of `rows` (default: all of them). */
  curveLimit?: number;
  /** All runs on the page, for the scatter plot over all of them. */
  allRows?: RunRow[];
  options: ComparisonOptions;
  onOptionsChange: (options: ComparisonOptions) => void;
  error?: string | null;
}

function numericColumns(rows: RunRow[]): Column[] {
  return availableColumns(rows).filter(
    (column) =>
      column.numeric &&
      column.group !== 'run' &&
      rows.some((row) => typeof column.value(row) === 'number')
  );
}

/**
 * Curves, settings against results and parallel coordinates of the chosen runs. The curve
 * charts stop at `curveLimit` runs and say so; the other two take all of them.
 */
export function CompareView({
  rows,
  curves,
  curveLimit,
  allRows,
  options,
  onOptionsChange,
  error,
}: CompareViewProps) {
  const { t } = useTranslation();
  const [tab, setTab] = useState<string | null>('curves');
  const [scope, setScope] = useState<'selected' | 'all'>('selected');
  const xAxis: XAxis = options.x_axis ?? 'epoch';
  const smoothing = options.smoothing ?? 0;
  const set = (change: Partial<ComparisonOptions>) => onOptionsChange({ ...options, ...change });

  // ------------------------------------------------------------------ curves
  const curveRows = useMemo(
    () => (curveLimit === undefined ? rows : rows.slice(0, curveLimit)),
    [rows, curveLimit]
  );
  const shownCurves = useMemo(
    () => (curves ?? []).filter((curve) => curveRows.some((row) => row.id === curve.id)),
    [curves, curveRows]
  );
  const metrics = curveMetrics(shownCurves).filter((key) => !key.endsWith('_loss'));
  const metric =
    options.metric && metrics.includes(options.metric)
      ? options.metric
      : (mainMetric(shownCurves) ?? metrics[0] ?? null);
  const [lossKind, setLossKind] = useState<'val_loss' | 'train_loss'>('val_loss');
  const losses = differentLosses(shownCurves);
  const tasks = differentTasks(shownCurves);
  const xLabel = xAxis === 'time' ? t('experiments.curves.time') : t('experiments.curves.epoch');
  const formatX = xAxis === 'time' ? formatElapsed : (value: number) => String(Math.round(value));
  const lossSeries = overlaySeries(shownCurves, lossKind, xAxis, smoothing);
  const metricSeries = metric ? overlaySeries(shownCurves, metric, xAxis, smoothing) : [];
  const percentLike = metric !== null && /accuracy|iou|dice|f1|auc/.test(metric);

  const curveControls = (
    <Group gap="sm" wrap="wrap">
      <SegmentedControl
        size="xs"
        aria-label={t('experiments.curves.xAxis')}
        data={[
          { value: 'epoch', label: t('experiments.curves.epoch') },
          { value: 'time', label: t('experiments.curves.time') },
        ]}
        value={xAxis}
        onChange={(value) => set({ x_axis: value as XAxis })}
      />
      <Switch
        size="sm"
        label={t('experiments.curves.smooth')}
        checked={smoothing > 0}
        onChange={(event) => set({ smoothing: event.currentTarget.checked ? SMOOTHING : 0 })}
      />
    </Group>
  );

  // ------------------------------------------------------------------ scatter
  const scatterRows = scope === 'all' && allRows ? allRows : rows;
  const candidates = numericColumns(scatterRows);
  const settingCandidates = candidates.filter((c) => c.group === 'settings');
  const metricCandidates = candidates.filter((c) => c.group === 'metrics');
  const scatterX =
    candidates.find((c) => c.id === options.scatter_x) ??
    settingCandidates.find((c) => c.id === 'setting:learning_rate') ??
    settingCandidates[0] ??
    candidates[0];
  const scatterY =
    candidates.find((c) => c.id === options.scatter_y) ??
    metricCandidates.find((c) => c.id === 'metric:val_loss') ??
    metricCandidates[0] ??
    candidates[1];
  const logX = scatterX ? LOG_SETTINGS.has(scatterX.id) : false;
  const points = useMemo(() => {
    if (!scatterX || !scatterY) {
      return [];
    }
    return scatterRows
      .map((row) => ({ row, x: scatterX.value(row), y: scatterY.value(row) }))
      .filter(
        (p): p is { row: RunRow; x: number; y: number } =>
          typeof p.x === 'number' &&
          typeof p.y === 'number' &&
          Number.isFinite(p.x) &&
          Number.isFinite(p.y) &&
          (!logX || p.x > 0)
      );
  }, [scatterRows, scatterX, scatterY, logX]);
  const columnOptions = (list: Column[]) =>
    [
      {
        group: t('experiments.columns.settings'),
        items: list
          .filter((c) => c.group === 'settings')
          .map((c) => ({ value: c.id, label: c.label })),
      },
      {
        group: t('experiments.columns.metrics'),
        items: list
          .filter((c) => c.group === 'metrics')
          .map((c) => ({ value: c.id, label: c.label })),
      },
    ].filter((group) => group.items.length > 0);

  // ------------------------------------------------------------------ parallel
  const defaultAxes = [
    ...settingCandidates
      .filter((c) =>
        [
          'setting:learning_rate',
          'setting:batch_size',
          'setting:epochs',
          'setting:weight_decay',
        ].includes(c.id)
      )
      .map((c) => c.id),
    ...(scatterY ? [scatterY.id] : []),
  ];
  const [axisIds, setAxisIds] = useState<string[] | null>(null);
  const parallelAvailable = availableColumns(rows).filter(
    (c) => c.group !== 'run' && rows.some((row) => c.value(row) !== null)
  );
  const chosenAxes = (axisIds ?? defaultAxes)
    .map((id) => parallelAvailable.find((c) => c.id === id))
    .filter((c): c is Column => c !== undefined);
  const parallelAxes: ParallelAxis[] = chosenAxes.map((column) => ({
    key: column.id,
    label: column.label,
    values: rows.map((row) => column.value(row)),
    format: (value: number) => column.text(withValue(rows[0], column, value)),
  }));

  if (rows.length === 0) {
    return (
      <Text size="sm" c="dimmed">
        {t('experiments.compare.selectRuns')}
      </Text>
    );
  }

  return (
    <Tabs value={tab} onChange={setTab} keepMounted={false}>
      <Tabs.List>
        <Tabs.Tab value="curves">{t('experiments.compare.curves')}</Tabs.Tab>
        <Tabs.Tab value="scatter">{t('experiments.compare.scatter')}</Tabs.Tab>
        <Tabs.Tab value="parallel">{t('experiments.compare.parallel')}</Tabs.Tab>
      </Tabs.List>

      <Tabs.Panel value="curves" pt="md">
        <Stack gap="lg">
          {error && <Alert color="red">{error}</Alert>}
          {curveRows.length < rows.length && (
            <Text size="xs" c="dimmed">
              {t('experiments.curves.firstOnly', {
                shown: curveRows.length,
                total: rows.length,
              })}
            </Text>
          )}
          {tasks && (
            <Alert color="yellow" title={t('experiments.curves.tasksDifferTitle')}>
              {t('experiments.curves.tasksDiffer')}
            </Alert>
          )}
          {losses && (
            <Alert color="yellow" title={t('experiments.curves.lossesDifferTitle')}>
              {t('experiments.curves.lossesDiffer', { losses: losses.join(', ') })}
            </Alert>
          )}
          <OverlayChart
            title={t('experiments.curves.loss')}
            subtitle={t('experiments.curves.lossSubtitle')}
            series={lossSeries}
            xLabel={xLabel}
            yLabel={curveLabel(lossKind)}
            formatX={formatX}
            yBounds={[0, Infinity]}
            controls={
              <Group gap="sm" wrap="wrap">
                <SegmentedControl
                  size="xs"
                  aria-label={t('experiments.curves.whichLoss')}
                  data={[
                    { value: 'val_loss', label: t('experiments.curves.validation') },
                    { value: 'train_loss', label: t('experiments.curves.training') },
                  ]}
                  value={lossKind}
                  onChange={(value) => setLossKind(value as 'val_loss' | 'train_loss')}
                />
                {curveControls}
              </Group>
            }
            table={() =>
              overlayTable(lossSeries, xLabel, formatX, (v) => formatCurveValue(lossKind, v))
            }
            caption={smoothing > 0 ? t('experiments.curves.smoothedNote') : undefined}
          />
          {metric && (
            <OverlayChart
              title={curveLabel(metric)}
              series={metricSeries}
              xLabel={xLabel}
              yLabel={curveLabel(metric)}
              formatX={formatX}
              formatY={(v) => formatCurveValue(metric, v)}
              yBounds={percentLike ? [0, 1] : undefined}
              controls={
                <Select
                  size="xs"
                  w={200}
                  aria-label={t('experiments.curves.metric')}
                  data={metrics.map((key) => ({ value: key, label: curveLabel(key) }))}
                  value={metric}
                  onChange={(value) => value && set({ metric: value })}
                  allowDeselect={false}
                />
              }
              table={() =>
                overlayTable(metricSeries, xLabel, formatX, (v) => formatCurveValue(metric, v))
              }
            />
          )}
        </Stack>
      </Tabs.Panel>

      <Tabs.Panel value="scatter" pt="md">
        <Stack gap="sm">
          <Group gap="sm" wrap="wrap" align="flex-end">
            <Select
              size="xs"
              w={220}
              label={t('experiments.scatter.x')}
              data={columnOptions(candidates)}
              value={scatterX?.id ?? null}
              onChange={(value) => value && set({ scatter_x: value })}
              allowDeselect={false}
            />
            <Select
              size="xs"
              w={220}
              label={t('experiments.scatter.y')}
              data={columnOptions(candidates)}
              value={scatterY?.id ?? null}
              onChange={(value) => value && set({ scatter_y: value })}
              allowDeselect={false}
            />
            {allRows && (
              <SegmentedControl
                size="xs"
                aria-label={t('experiments.scatter.which')}
                data={[
                  { value: 'selected', label: t('experiments.scatter.selected') },
                  { value: 'all', label: t('experiments.scatter.all') },
                ]}
                value={scope}
                onChange={(value) => setScope(value as 'selected' | 'all')}
              />
            )}
          </Group>
          {scatterX && scatterY && points.length > 0 ? (
            <ScatterPlot
              title={t('experiments.scatter.title', { y: scatterY.label, x: scatterX.label })}
              subtitle={
                logX
                  ? t('experiments.scatter.logSubtitle', { count: points.length })
                  : t('experiments.scatter.subtitle', { count: points.length })
              }
              x={points.map((p) => (logX ? Math.log10(p.x) : p.x))}
              y={points.map((p) => p.y)}
              xLabel={
                logX ? t('experiments.scatter.logOf', { label: scatterX.label }) : scatterX.label
              }
              yLabel={scatterY.label}
              formatX={(v) =>
                logX
                  ? scatterX.text(withValue(points[0].row, scatterX, 10 ** v))
                  : scatterX.text(withValue(points[0].row, scatterX, v))
              }
              formatY={(v) => scatterY.text(withValue(points[0].row, scatterY, v))}
              pointLabel={(i) => points[i].row.name}
              trend={points.length >= 3}
            />
          ) : (
            <Text size="sm" c="dimmed">
              {t('experiments.scatter.nothing')}
            </Text>
          )}
        </Stack>
      </Tabs.Panel>

      <Tabs.Panel value="parallel" pt="md">
        <Stack gap="sm">
          <MultiSelect
            size="xs"
            label={t('experiments.parallel.axes')}
            data={columnOptions(parallelAvailable)}
            value={chosenAxes.map((c) => c.id)}
            onChange={setAxisIds}
            maxValues={8}
            searchable
          />
          <ParallelCoordinates
            title={t('experiments.parallel.title')}
            subtitle={t('experiments.parallel.subtitle')}
            axes={parallelAxes}
            items={rows.map((row) => ({ key: row.id, label: row.name }))}
          />
        </Stack>
      </Tabs.Panel>
    </Tabs>
  );
}

/** A copy of `row` with `value` in the column, to format an axis tick like a cell. */
function withValue(row: RunRow, column: Column, value: number): RunRow {
  if (column.id.startsWith('setting:')) {
    return { ...row, settings: { ...row.settings, [column.id.slice(8)]: value } };
  }
  if (column.id.startsWith('metric:')) {
    return { ...row, metrics: { ...row.metrics, [column.id.slice(7)]: value } };
  }
  return { ...row, duration_seconds: value };
}
