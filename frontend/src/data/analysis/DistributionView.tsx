import { useMemo } from 'react';
import { IconChartDots } from '@tabler/icons-react';
import { Button, Group, SimpleGrid, Stack, Switch, Table, Text } from '@mantine/core';
import type { ColumnProfile, DataProfile, DataSample, NewStep, NumericStats } from '@/api/types';
import { BarChart, type BarRow, type BarSeries } from '@/components/charts/BarChart';
import { autoEdges, countBins, histogram } from '@/components/charts/binning';
import { BoxPlot, type BoxGroup } from '@/components/charts/BoxPlot';
import { OTHER_COLOR, seriesColor } from '@/components/charts/colors';
import {
  dateUnitForValues,
  formatCount,
  formatDate,
  formatDateTime,
  formatDuration,
  formatNumber,
  formatPercent,
} from '@/components/charts/format';
import { Histogram, type HistogramSeries } from '@/components/charts/Histogram';
import { boxFromStats, boxFromValues } from '@/components/charts/stats';
import {
  calendarCounts,
  classGroups,
  countValues,
  crossTab,
  MONTHS,
  numericValues,
  sampleColumn,
  sampleNote,
  textLengths,
  valueLabel,
  valuesByGroup,
  WEEKDAYS,
  type RowGroups,
} from './sampleData';
import { ColumnSelect, KIND_LABELS, Note } from './shared';
import { WarningList } from './WarningList';
import classes from './AnalysisPanel.module.css';

export interface DistributionOptions {
  log: boolean;
  split: boolean;
  /** Zoom set for a column (1st–99th percentile); other columns decide by themselves. */
  zoom: { column: string; on: boolean } | null;
}

/**
 * Whether a histogram is dominated by a few extreme values: one bin holds ≥ 80 % of the values
 * while the 1st–99th percentile covers less than half of the range.
 */
export function needsZoom(column: ColumnProfile): boolean {
  const s = column.stats;
  const h = column.histogram;
  if (!s || !h || s.q01 === null || s.q99 === null || s.min === null || s.max === null) {
    return false;
  }
  const total = h.counts.reduce((sum, c) => sum + c, 0);
  const crowded = total > 0 && Math.max(...h.counts) / total >= 0.8;
  return crowded && s.q99 > s.q01 && s.q99 - s.q01 < 0.5 * (s.max - s.min);
}

export interface DistributionViewProps {
  profile: DataProfile;
  sample: DataSample;
  target: string | null;
  column: string | null;
  onColumnChange: (column: string) => void;
  options: DistributionOptions;
  onOptionsChange: (options: DistributionOptions) => void;
  onAddStep: (step: NewStep) => void;
  /** Open the scatter plot of two columns. */
  onScatter: (x: string, y: string) => void;
}

/** Category bars before "Other": the profile keeps the 20 most frequent values. */
const SPLIT_CATEGORIES = 12;

export function KeyValueTable({ rows, label }: { rows: [string, string][]; label: string }) {
  return (
    <Table
      fz="xs"
      verticalSpacing={3}
      withRowBorders={false}
      aria-label={label}
      className={classes.kv}
    >
      <Table.Tbody>
        {rows.map(([key, value]) => (
          <Table.Tr key={key}>
            <Table.Td c="dimmed">{key}</Table.Td>
            <Table.Td ta="right" className={classes.number}>
              {value}
            </Table.Td>
          </Table.Tr>
        ))}
      </Table.Tbody>
    </Table>
  );
}

function quantileRows(
  stats: NumericStats,
  format: (v: number | null) => string
): [string, string][] {
  return [
    ['Min', format(stats.min)],
    ['1st percentile', format(stats.q01)],
    ['5th percentile', format(stats.q05)],
    ['25th percentile (Q1)', format(stats.q25)],
    ['Median', format(stats.median)],
    ['75th percentile (Q3)', format(stats.q75)],
    ['95th percentile', format(stats.q95)],
    ['99th percentile', format(stats.q99)],
    ['Max', format(stats.max)],
  ];
}

const num = (v: number | null) => formatNumber(v);

export function columnFacts(column: ColumnProfile, rows: number): [string, string][] {
  return [
    ['Values', formatCount(column.count)],
    [
      'Missing',
      `${formatCount(column.missing)} (${formatPercent(rows ? column.missing / rows : 0)})`,
    ],
    ['Unique', formatCount(column.unique)],
  ];
}

function numericFacts(column: ColumnProfile, rows: number): [string, string][] {
  const s = column.stats;
  if (!s) {
    return columnFacts(column, rows);
  }
  return [
    ...columnFacts(column, rows),
    ['Mean', num(s.mean)],
    ['Standard deviation', num(s.std)],
    ['Skewness', num(s.skew)],
    ['Excess kurtosis', num(s.kurtosis)],
    ['Zeros', formatCount(s.zeros)],
    ['Negative', formatCount(s.negative)],
    ['Outliers (1.5 × IQR)', formatCount(s.iqr_outliers)],
    ['Outliers (|z| > 3)', formatCount(s.z_outliers)],
  ];
}

/** Series per class from the sample, on the profile's bins. */
function classHistograms(
  edges: number[],
  values: number[][],
  groups: RowGroups
): HistogramSeries[] {
  return groups.groups.map((g, i) => ({
    key: g.key,
    label: g.label,
    color: seriesColor(g.slot),
    counts: countBins(values[i], edges),
  }));
}

/** The distribution of one column: chart(s) by kind, statistics and its warnings. */
export function DistributionView({
  profile,
  sample,
  target,
  column: name,
  onColumnChange,
  options,
  onOptionsChange,
  onAddStep,
  onScatter,
}: DistributionViewProps) {
  const column = profile.columns.find((c) => c.name === name) ?? profile.columns[0];
  const groups = useMemo(
    () =>
      profile.target?.task === 'classification' && profile.target.column === target
        ? classGroups(sample, profile, target)
        : null,
    [sample, profile, target]
  );
  if (!column) {
    return <Text size="sm">No columns.</Text>;
  }
  const isTarget = column.name === target;
  const canSplit = !!groups && !isTarget && groups.groups.length > 1;
  const split = canSplit && options.split;
  const regressionTarget =
    profile.target?.task === 'regression' && !isTarget && target ? target : null;
  const warnings = profile.warnings.filter((w) => w.column === column.name);
  const stats = column.stats;
  const zoomable =
    column.kind === 'numeric' &&
    !!stats &&
    stats.q01 !== null &&
    stats.q99 !== null &&
    stats.min !== null &&
    stats.max !== null &&
    stats.q99 > stats.q01 &&
    (stats.q01 > stats.min || stats.q99 < stats.max);
  const zoom =
    zoomable && (options.zoom?.column === column.name ? options.zoom.on : needsZoom(column));

  return (
    <Stack gap="md">
      <Group gap="sm" align="flex-end" wrap="wrap">
        <ColumnSelect
          label="Column"
          columns={profile.columns}
          value={column.name}
          onChange={(value) => value && onColumnChange(value)}
          w={240}
        />
        {(column.kind === 'numeric' || column.kind === 'datetime') && (
          <Switch
            size="xs"
            label="Log scale (counts)"
            checked={options.log}
            onChange={(event) => onOptionsChange({ ...options, log: event.currentTarget.checked })}
          />
        )}
        {zoomable && (
          <Switch
            size="xs"
            label="Zoom to 1st–99th percentile"
            checked={zoom}
            onChange={(event) =>
              onOptionsChange({
                ...options,
                zoom: { column: column.name, on: event.currentTarget.checked },
              })
            }
          />
        )}
        {canSplit && (
          <Switch
            size="xs"
            label={`Split by ${target}`}
            checked={options.split}
            onChange={(event) =>
              onOptionsChange({ ...options, split: event.currentTarget.checked })
            }
          />
        )}
        {regressionTarget && column.kind === 'numeric' && (
          <Button
            size="compact-sm"
            variant="subtle"
            leftSection={<IconChartDots size={14} />}
            onClick={() => onScatter(column.name, regressionTarget)}
          >
            Plot against {regressionTarget}
          </Button>
        )}
      </Group>
      <Text size="xs" c="dimmed">
        {KIND_LABELS[column.kind]}
        {column.integer && column.kind === 'numeric' ? ' (whole numbers)' : ''} ·{' '}
        {formatCount(column.count)} values ·{' '}
        {formatPercent(profile.n_rows ? column.missing / profile.n_rows : 0)} missing ·{' '}
        {formatCount(column.unique)} unique
        {isTarget ? ' · target' : ''}
      </Text>

      {column.kind === 'numeric' && (
        <NumericDistribution
          column={column}
          profile={profile}
          sample={sample}
          groups={split ? groups : null}
          log={options.log}
          zoom={zoom}
        />
      )}
      {column.kind === 'datetime' && (
        <DatetimeDistribution
          column={column}
          profile={profile}
          sample={sample}
          groups={split ? groups : null}
          log={options.log}
        />
      )}
      {(column.kind === 'categorical' || column.kind === 'boolean' || column.kind === 'text') && (
        <CategoryDistribution
          column={column}
          profile={profile}
          sample={sample}
          groups={split ? groups : null}
        />
      )}

      {warnings.length > 0 && (
        <Stack gap={6}>
          <Text size="sm" fw={600}>
            Warnings
          </Text>
          <WarningList warnings={warnings} resetKey={profile} onAddStep={onAddStep} />
        </Stack>
      )}
    </Stack>
  );
}

interface PartProps {
  column: ColumnProfile;
  profile: DataProfile;
  sample: DataSample;
  /** Split by these classes (from the sample); null for the whole column. */
  groups: RowGroups | null;
}

/** Histogram of the sample values between two bounds (and per group). */
function zoomedHistogram(
  all: number[],
  groupValues: number[][] | null,
  lo: number,
  hi: number,
  integer: boolean
) {
  const inside = all.filter((v) => v >= lo && v <= hi);
  const edges = autoEdges(inside, { integer, minBins: 8, maxBins: 40 });
  return {
    edges,
    inside: inside.length,
    outside: all.length - inside.length,
    all: countBins(inside, edges),
    groups: groupValues?.map((values) =>
      countBins(
        values.filter((v) => v >= lo && v <= hi),
        edges
      )
    ),
  };
}

function NumericDistribution({
  column,
  profile,
  sample,
  groups,
  log,
  zoom,
}: PartProps & { log: boolean; zoom: boolean }) {
  const col = sampleColumn(sample, column.name);
  const values = useMemo(() => (col && groups ? valuesByGroup(col, groups) : null), [col, groups]);
  const hist = column.histogram;
  const stats = column.stats;
  const zoomed = useMemo(
    () =>
      zoom && col && stats && stats.q01 !== null && stats.q99 !== null
        ? zoomedHistogram(numericValues(col), values, stats.q01, stats.q99, column.integer)
        : null,
    [zoom, col, stats, values, column.integer]
  );
  const series: HistogramSeries[] = zoomed
    ? zoomed.groups && groups
      ? groups.groups.map((g, i) => ({
          key: g.key,
          label: g.label,
          color: seriesColor(g.slot),
          counts: zoomed.groups![i],
        }))
      : [{ key: 'all', label: column.name, color: seriesColor(0), counts: zoomed.all }]
    : hist && values && groups
      ? classHistograms(hist.edges, values, groups)
      : hist
        ? [{ key: 'all', label: column.name, color: seriesColor(0), counts: hist.counts }]
        : [];
  const edges = zoomed ? zoomed.edges : (hist?.edges ?? []);
  const boxes: BoxGroup[] =
    values && groups
      ? groups.groups
          .map((g, i) => ({ g, box: boxFromValues(values[i]) }))
          .filter((b) => b.box)
          .map(({ g, box }) => ({
            key: g.key,
            label: g.label,
            color: seriesColor(g.slot),
            box: box!,
          }))
      : stats && boxFromStats(stats, column.count)
        ? [{ key: 'all', label: column.name, box: boxFromStats(stats, column.count)! }]
        : [];
  const split = !!(values && groups);

  return (
    <Stack gap="md">
      {edges.length > 1 ? (
        <Histogram
          title={split ? `${column.name} by ${groups!.column}` : `Distribution of ${column.name}`}
          subtitle={
            zoomed
              ? `${split ? "Share of each class's rows" : 'Rows'} per bin between ${num(stats!.q01)} and ${num(stats!.q99)} (1st–99th percentile; ${formatCount(zoomed.outside)} of ${formatCount(zoomed.inside + zoomed.outside)} sample values outside). ${sampleNote(sample)}`
              : split
                ? `Share of each class's rows per bin. ${sampleNote(sample)}`
                : `Rows per bin, all ${formatCount(profile.n_rows)} rows.`
          }
          edges={edges}
          series={series}
          integer={column.integer}
          log={log}
          normalize={split}
          xLabel={column.name}
          unit="rows"
        />
      ) : (
        <Text size="sm" c="dimmed">
          No values to show.
        </Text>
      )}
      {boxes.length > 0 && (
        <BoxPlot
          title={split ? 'Box plots by class' : 'Box plot'}
          subtitle={
            split
              ? sampleNote(sample)
              : 'Quartiles from all rows; whiskers at most 1.5 × IQR beyond the box.'
          }
          groups={boxes}
        />
      )}
      {stats && (
        <SimpleGrid type="container" cols={{ base: 1, '480px': 2 }} spacing="lg">
          <div>
            <Text size="sm" fw={600} mb={4}>
              Quantiles
            </Text>
            <KeyValueTable label={`Quantiles of ${column.name}`} rows={quantileRows(stats, num)} />
          </div>
          <div>
            <Text size="sm" fw={600} mb={4}>
              Statistics
            </Text>
            <KeyValueTable
              label={`Statistics of ${column.name}`}
              rows={numericFacts(column, profile.n_rows)}
            />
          </div>
        </SimpleGrid>
      )}
    </Stack>
  );
}

function DatetimeDistribution({
  column,
  profile,
  sample,
  groups,
  log,
}: PartProps & { log: boolean }) {
  const col = sampleColumn(sample, column.name);
  const values = useMemo(() => (col && groups ? valuesByGroup(col, groups) : null), [col, groups]);
  const calendar = useMemo(() => (col ? calendarCounts(numericValues(col)) : null), [col]);
  // Dates with a time of day (in the sample) are shown with it.
  const timed = useMemo(() => !!col && dateUnitForValues(numericValues(col)) !== 'day', [col]);
  const hist = column.histogram;
  const stats = column.stats;
  const split = !!(values && groups);
  const date = (v: number | null) => (timed ? formatDateTime(v) : formatDate(v));
  return (
    <Stack gap="md">
      {hist ? (
        <Histogram
          title={split ? `${column.name} by ${groups!.column}` : `${column.name} over time`}
          subtitle={
            split
              ? `Share of each class's rows per bin. ${sampleNote(sample)}`
              : `Rows per period, all ${formatCount(profile.n_rows)} rows.`
          }
          edges={hist.edges}
          series={
            split
              ? classHistograms(hist.edges, values!, groups!)
              : [{ key: 'all', label: column.name, color: seriesColor(0), counts: hist.counts }]
          }
          time
          log={log}
          normalize={split}
          xLabel={column.name}
          unit="rows"
        />
      ) : (
        <Text size="sm" c="dimmed">
          No values to show.
        </Text>
      )}
      {calendar && (
        <SimpleGrid type="container" cols={{ base: 1, '560px': 2 }} spacing="lg">
          <BarChart
            title="By weekday"
            subtitle={sampleNote(sample)}
            rows={WEEKDAYS.map((label, i) => ({
              key: label,
              label,
              values: [calendar.weekday[i]],
            }))}
            unit="rows"
          />
          <BarChart
            title="By month"
            subtitle={sampleNote(sample)}
            rows={MONTHS.map((label, i) => ({ key: label, label, values: [calendar.month[i]] }))}
            unit="rows"
          />
        </SimpleGrid>
      )}
      {stats && (
        <KeyValueTable
          label={`Statistics of ${column.name}`}
          rows={[
            ...columnFacts(column, profile.n_rows),
            ['First', date(stats.min)],
            ['Median', date(stats.median)],
            ['Last', date(stats.max)],
            [
              'Span',
              stats.min !== null && stats.max !== null
                ? formatDuration(stats.max - stats.min)
                : '–',
            ],
            ['Middle 50 %', `${date(stats.q25)} – ${date(stats.q75)}`],
          ]}
        />
      )}
    </Stack>
  );
}

function CategoryDistribution({ column, profile, sample, groups }: PartProps) {
  const col = sampleColumn(sample, column.name);
  const top = column.top ?? [];
  const total = column.count + column.missing;
  const rows: BarRow[] = [
    ...top.map((t) => ({
      key: `v:${t.value}`,
      label: valueLabel(t.value, column.kind),
      values: [t.count],
    })),
    ...(column.other_count
      ? [
          {
            key: 'other',
            label: `Other (${formatCount(Math.max(0, column.unique - top.length))} values)`,
            values: [column.other_count],
            muted: true,
          },
        ]
      : []),
    ...(column.missing
      ? [{ key: 'missing', label: 'Missing', values: [column.missing], muted: true }]
      : []),
  ];

  const split = useMemo(() => {
    if (!col || !groups) {
      return null;
    }
    const keys = top.slice(0, SPLIT_CATEGORIES).map((t) => t.value);
    const table = crossTab(col, groups, keys);
    const splitRows: BarRow[] = keys.map((key, i) => ({
      key: `v:${key}`,
      label: valueLabel(key, column.kind),
      values: table.rows[i],
    }));
    if (table.other.some((v) => v > 0)) {
      splitRows.push({ key: 'other', label: 'Other values', values: table.other });
    }
    if (table.missing.some((v) => v > 0)) {
      splitRows.push({ key: 'missing', label: 'Missing', values: table.missing });
    }
    const series: BarSeries[] = groups.groups.map((g) => ({
      key: g.key,
      label: g.label,
      color: g.slot >= 0 ? seriesColor(g.slot) : OTHER_COLOR,
    }));
    return { rows: splitRows, series };
  }, [col, groups, top, column.kind]);

  const lengths = useMemo(
    () => (column.kind === 'text' && col ? histogram(textLengths(col), { integer: true }) : null),
    [column.kind, col]
  );
  const sampleCounts = useMemo(
    () => (column.kind === 'text' && col ? countValues(col) : null),
    [column.kind, col]
  );

  return (
    <Stack gap="md">
      {split ? (
        <BarChart
          title={`${column.name} by ${groups!.column}`}
          subtitle={`Class mix per value. ${sampleNote(sample)}`}
          rows={split.rows}
          series={split.series}
          layout="stacked"
          normalize
          unit="rows"
        />
      ) : (
        <BarChart
          title={
            column.kind === 'text'
              ? `Most frequent texts in ${column.name}`
              : `Values of ${column.name}`
          }
          subtitle={`Rows per value, all ${formatCount(profile.n_rows)} rows${top.length >= 20 ? ' (the 20 most frequent values)' : ''}.`}
          rows={rows}
          tipLabel={(row) =>
            `${formatCount(row.values[0])} · ${formatPercent(total ? row.values[0] / total : 0)}`
          }
          unit="rows"
        />
      )}
      {lengths && lengths.edges.length > 1 && (
        <Histogram
          title="Text length"
          subtitle={`Characters per text. ${sampleNote(sample)}`}
          edges={lengths.edges}
          series={[
            { key: 'length', label: 'Length', color: seriesColor(0), counts: lengths.counts },
          ]}
          integer
          xLabel="Characters"
          unit="texts"
        />
      )}
      <KeyValueTable
        label={`Statistics of ${column.name}`}
        rows={[
          ...columnFacts(column, profile.n_rows),
          ...(column.kind === 'boolean' && typeof column.stats?.mean === 'number'
            ? ([['Share true', formatPercent(column.stats.mean)]] as [string, string][])
            : []),
          ...(column.mean_length !== null
            ? ([['Mean length', `${formatNumber(column.mean_length, 3)} characters`]] as [
                string,
                string,
              ][])
            : []),
          ...(top[0]
            ? ([
                [
                  'Most frequent',
                  `${valueLabel(top[0].value, column.kind)} (${formatPercent(column.count ? top[0].count / column.count : 0)})`,
                ],
              ] as [string, string][])
            : []),
          ...(sampleCounts
            ? ([
                [
                  'Distinct in the sample',
                  `${formatCount(sampleCounts.counts.length)} of ${formatCount(sample.rows - sampleCounts.missing)}`,
                ],
              ] as [string, string][])
            : []),
        ]}
      />
      {column.kind === 'text' && (
        <Note>
          Free text cannot be used as a feature directly. Drop it, or turn it into categories first.
        </Note>
      )}
    </Stack>
  );
}
