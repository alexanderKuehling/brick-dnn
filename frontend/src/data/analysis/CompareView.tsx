import { useEffect, useMemo, useRef } from 'react';
import { Badge, Group, Loader, Stack, Switch, Table, Text } from '@mantine/core';
import type { ColumnProfile, DataProfile } from '@/api/types';
import { BarChart, type BarRow, type BarSeries } from '@/components/charts/BarChart';
import { commonEdges, rebin, sameEdges } from '@/components/charts/binning';
import { BoxPlot, type BoxGroup } from '@/components/charts/BoxPlot';
import { OTHER_COLOR, seriesColor } from '@/components/charts/colors';
import { formatCount, formatDate, formatNumber } from '@/components/charts/format';
import { Histogram } from '@/components/charts/Histogram';
import { boxFromStats } from '@/components/charts/stats';
import { valueLabel } from './sampleData';
import { ColumnSelect, KIND_LABELS, Note } from './shared';
import classes from './AnalysisPanel.module.css';

export interface CompareViewProps {
  profile: DataProfile;
  before: DataProfile | null;
  onRequestCompare: () => void;
  column: string | null;
  onColumnChange: (column: string) => void;
  log: boolean;
  onLogChange: (log: boolean) => void;
}

const BEFORE = { key: 'before', label: 'Before', color: OTHER_COLOR };
const AFTER = { key: 'after', label: 'After', color: seriesColor(0) };
const TOP = 12;

function change(before: number | null | undefined, after: number | null | undefined): string {
  if (typeof before !== 'number' || typeof after !== 'number') {
    return '–';
  }
  const diff = after - before;
  if (diff === 0) {
    return 'same';
  }
  return `${diff > 0 ? '+' : '−'}${formatNumber(Math.abs(diff))}`;
}

function ComparisonTable({
  rows,
  label,
}: {
  rows: [string, string, string, string][];
  label: string;
}) {
  return (
    <Table.ScrollContainer minWidth={360} type="native">
      <Table fz="xs" verticalSpacing={3} aria-label={label} className={classes.kv}>
        <Table.Thead>
          <Table.Tr>
            <Table.Th />
            <Table.Th ta="right">Before</Table.Th>
            <Table.Th ta="right">After</Table.Th>
            <Table.Th ta="right">Change</Table.Th>
          </Table.Tr>
        </Table.Thead>
        <Table.Tbody>
          {rows.map(([name, a, b, c]) => (
            <Table.Tr key={name}>
              <Table.Td c="dimmed">{name}</Table.Td>
              <Table.Td ta="right" className={classes.number}>
                {a}
              </Table.Td>
              <Table.Td ta="right" className={classes.number}>
                {b}
              </Table.Td>
              <Table.Td ta="right" className={classes.number}>
                {c}
              </Table.Td>
            </Table.Tr>
          ))}
        </Table.Tbody>
      </Table>
    </Table.ScrollContainer>
  );
}

/** The data before the cleaning steps next to the data after them, for one column. */
export function CompareView({
  profile,
  before,
  onRequestCompare,
  column,
  onColumnChange,
  log,
  onLogChange,
}: CompareViewProps) {
  // Ask for the "before" profile whenever it is missing (also after it was dropped again).
  const request = useRef(onRequestCompare);
  useEffect(() => {
    request.current = onRequestCompare;
  });
  useEffect(() => {
    if (!before) {
      request.current();
    }
  }, [before]);

  const columns = useMemo(() => {
    if (!before) {
      return [];
    }
    const seen = new Set(profile.columns.map((c) => c.name));
    return [...profile.columns, ...before.columns.filter((c) => !seen.has(c.name))];
  }, [profile.columns, before]);

  if (!before) {
    return (
      <Group gap="sm" role="status">
        <Loader size="sm" />
        <Text size="sm" c="dimmed">
          Profiling the data before the cleaning steps…
        </Text>
      </Group>
    );
  }

  const beforeNames = new Set(before.columns.map((c) => c.name));
  const afterNames = new Set(profile.columns.map((c) => c.name));
  const added = profile.columns.filter((c) => !beforeNames.has(c.name)).map((c) => c.name);
  const removed = before.columns.filter((c) => !afterNames.has(c.name)).map((c) => c.name);
  const name =
    column && (afterNames.has(column) || beforeNames.has(column))
      ? column
      : (profile.columns.find((c) => beforeNames.has(c.name))?.name ?? columns[0]?.name ?? null);
  const a = before.columns.find((c) => c.name === name) ?? null;
  const b = profile.columns.find((c) => c.name === name) ?? null;

  return (
    <Stack gap="md">
      <ComparisonTable
        label="Data before and after the steps"
        rows={[
          [
            'Rows',
            formatCount(before.n_rows),
            formatCount(profile.n_rows),
            change(before.n_rows, profile.n_rows),
          ],
          [
            'Columns',
            formatCount(before.n_columns),
            formatCount(profile.n_columns),
            change(before.n_columns, profile.n_columns),
          ],
          [
            'Missing cells',
            formatCount(before.missing_cells),
            formatCount(profile.missing_cells),
            change(before.missing_cells, profile.missing_cells),
          ],
          [
            'Duplicate rows',
            formatCount(before.duplicate_rows),
            formatCount(profile.duplicate_rows),
            change(before.duplicate_rows, profile.duplicate_rows),
          ],
          [
            'Warnings',
            formatCount(before.warnings.length),
            formatCount(profile.warnings.length),
            change(before.warnings.length, profile.warnings.length),
          ],
        ]}
      />
      {(added.length > 0 || removed.length > 0) && (
        <Stack gap={4}>
          {added.length > 0 && <NameList label="Added" names={added} color="blue" />}
          {removed.length > 0 && <NameList label="Removed" names={removed} color="gray" />}
        </Stack>
      )}

      <Group gap="sm" align="flex-end" wrap="wrap">
        <ColumnSelect
          label="Column"
          columns={columns}
          value={name}
          onChange={(value) => value && onColumnChange(value)}
          w={240}
        />
        {a && b && (a.kind === 'numeric' || a.kind === 'datetime') && a.kind === b.kind && (
          <Switch
            size="xs"
            label="Log scale (counts)"
            checked={log}
            onChange={(e) => onLogChange(e.currentTarget.checked)}
          />
        )}
      </Group>

      {a && b ? (
        <ColumnComparison
          before={a}
          after={b}
          log={log}
          rowsBefore={before.n_rows}
          rowsAfter={profile.n_rows}
        />
      ) : null}
      {a && !b && (
        <Text size="sm" c="dimmed">
          {a.name} was removed by the steps ({KIND_LABELS[a.kind].toLowerCase()},{' '}
          {formatCount(a.missing)} missing).
        </Text>
      )}
      {!a && b && (
        <Text size="sm" c="dimmed">
          {b.name} was added by the steps ({KIND_LABELS[b.kind].toLowerCase()}).
        </Text>
      )}
      <Note>
        Before: the data after combining tables and assigning the split, before any cleaning step.
        After: the data at the previewed step. Both from all rows.
      </Note>
    </Stack>
  );
}

function NameList({ label, names, color }: { label: string; names: string[]; color: string }) {
  const shown = names.slice(0, 30);
  return (
    <Group gap={4} wrap="wrap">
      <Text size="xs" fw={600} mr={4}>
        {label} ({formatCount(names.length)}):
      </Text>
      {shown.map((n) => (
        <Badge key={n} size="xs" variant="light" color={color} tt="none">
          {n}
        </Badge>
      ))}
      {names.length > shown.length && (
        <Text size="xs" c="dimmed">
          and {formatCount(names.length - shown.length)} more
        </Text>
      )}
    </Group>
  );
}

function ColumnComparison({
  before,
  after,
  log,
  rowsBefore,
  rowsAfter,
}: {
  before: ColumnProfile;
  after: ColumnProfile;
  log: boolean;
  rowsBefore: number;
  rowsAfter: number;
}) {
  const time = after.kind === 'datetime';
  const fmt = (v: number | null | undefined) =>
    typeof v !== 'number' ? '–' : time ? formatDate(v) : formatNumber(v);
  const kindChanged = before.kind !== after.kind;
  const statRows: [string, string, string, string][] = [
    ['Type', KIND_LABELS[before.kind], KIND_LABELS[after.kind], kindChanged ? 'changed' : 'same'],
    [
      'Values',
      formatCount(before.count),
      formatCount(after.count),
      change(before.count, after.count),
    ],
    [
      'Missing',
      formatCount(before.missing),
      formatCount(after.missing),
      change(before.missing, after.missing),
    ],
    [
      'Unique',
      formatCount(before.unique),
      formatCount(after.unique),
      change(before.unique, after.unique),
    ],
  ];
  if (before.stats && after.stats && !kindChanged) {
    const keys = [
      ['Mean', 'mean'],
      ['Standard deviation', 'std'],
      ['Min', 'min'],
      ['Median', 'median'],
      ['Max', 'max'],
      ['Skewness', 'skew'],
    ] as const;
    for (const [label, key] of keys) {
      statRows.push([
        label,
        fmt(before.stats[key]),
        fmt(after.stats[key]),
        time ? '' : change(before.stats[key], after.stats[key]),
      ]);
    }
    statRows.push([
      'Outliers (1.5 × IQR)',
      formatCount(before.stats.iqr_outliers),
      formatCount(after.stats.iqr_outliers),
      change(before.stats.iqr_outliers, after.stats.iqr_outliers),
    ]);
  }

  const histograms = useMemo(() => {
    if (kindChanged || !before.histogram || !after.histogram) {
      return null;
    }
    const edges = commonEdges([before.histogram, after.histogram], 30);
    const exact =
      sameEdges(edges, before.histogram.edges) && sameEdges(edges, after.histogram.edges);
    return {
      edges,
      exact,
      series: [
        { ...BEFORE, counts: rebin(before.histogram, edges) },
        { ...AFTER, counts: rebin(after.histogram, edges) },
      ],
    };
  }, [before.histogram, after.histogram, kindChanged]);

  const boxes: BoxGroup[] = [];
  if (!kindChanged && before.stats && after.stats && before.kind !== 'boolean') {
    const x = boxFromStats(before.stats, before.count);
    const y = boxFromStats(after.stats, after.count);
    if (x && y) {
      boxes.push({ key: 'before', label: 'Before', color: OTHER_COLOR, box: x });
      boxes.push({ key: 'after', label: 'After', color: seriesColor(0), box: y });
    }
  }

  const bars = useMemo(() => {
    if (kindChanged || !before.top || !after.top || before.kind === 'text') {
      return null;
    }
    const keys: string[] = [];
    for (const t of [...after.top, ...before.top]) {
      if (!keys.includes(t.value) && keys.length < TOP) {
        keys.push(t.value);
      }
    }
    const count = (column: ColumnProfile, key: string) =>
      column.top?.find((t) => t.value === key)?.count ?? 0;
    const rows: BarRow[] = keys.map((key) => ({
      key,
      label: valueLabel(key, after.kind),
      values: [count(before, key), count(after, key)],
    }));
    if (before.missing || after.missing) {
      rows.push({ key: '__missing__', label: 'Missing', values: [before.missing, after.missing] });
    }
    return rows;
  }, [before, after, kindChanged]);
  const barSeries: BarSeries[] = [BEFORE, AFTER];

  return (
    <Stack gap="md">
      {histograms && (
        <Histogram
          title={`${after.name} before and after`}
          subtitle={
            histograms.exact
              ? `Rows per bin: before ${formatCount(rowsBefore)}, after ${formatCount(rowsAfter)} rows.`
              : 'Rows per bin on shared bins (counts spread evenly within the original bins, so heights are estimates).'
          }
          edges={histograms.edges}
          series={histograms.series}
          time={time}
          log={log}
          approximate={!histograms.exact}
          xLabel={after.name}
          unit="rows"
        />
      )}
      {boxes.length === 2 && (
        <BoxPlot
          title="Box plots"
          subtitle="Quartiles from all rows; whiskers at most 1.5 × IQR beyond the box."
          groups={boxes}
          time={time}
        />
      )}
      {bars && bars.length > 0 && (
        <BarChart
          title={`${after.name} before and after`}
          subtitle="Rows per value (the most frequent values)."
          rows={bars}
          series={barSeries}
          layout="grouped"
          unit="rows"
        />
      )}
      <ComparisonTable label={`${after.name} before and after`} rows={statRows} />
    </Stack>
  );
}
