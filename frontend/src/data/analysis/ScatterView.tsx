import { useMemo } from 'react';
import { Group, Stack, Switch, Text } from '@mantine/core';
import type { DataProfile, DataSample } from '@/api/types';
import { MAX_SCATTER_SERIES } from '@/components/charts/colors';
import { formatCoefficient, formatCount, formatDateTime } from '@/components/charts/format';
import {
  ScatterPlot,
  type ScatterCategory,
  type ScatterColor,
} from '@/components/charts/ScatterPlot';
import { spearman } from '@/components/charts/stats';
import {
  categoryKey,
  classGroups,
  isNumericKind,
  numericPairs,
  rowName,
  sampleColumn,
  sampleNote,
  valueLabel,
  type SampleColumn,
} from './sampleData';
import { ColumnSelect, Note } from './shared';

export interface ScatterOptions {
  x: string | null;
  y: string | null;
  /** Colour column, null for none. */
  color: string | null;
  trend: boolean;
  jitter: boolean;
}

export interface ScatterViewProps {
  profile: DataProfile;
  sample: DataSample;
  target: string | null;
  options: ScatterOptions;
  onOptionsChange: (options: ScatterOptions) => void;
}

/**
 * Default axes: the most strongly correlated pair of number columns (not yes/no columns and
 * not the target, which is better used as the colour), else the first two number columns.
 */
export function defaultScatterAxes(
  profile: DataProfile,
  sample: DataSample,
  target: string | null = null
): [string | null, string | null] {
  const pearson = profile.correlations.find((c) => c.method === 'pearson');
  const numeric = sample.columns.filter((c) => isNumericKind(c.kind) && c.kind !== 'boolean');
  const usable = (name: string) => name !== target && numeric.some((c) => c.name === name);
  if (pearson) {
    let best: [string, string] | null = null;
    let size = -1;
    for (let a = 0; a < pearson.columns.length; a += 1) {
      for (let b = a + 1; b < pearson.columns.length; b += 1) {
        const value = pearson.matrix[a]?.[b];
        const present = usable(pearson.columns[a]) && usable(pearson.columns[b]);
        if (typeof value === 'number' && Math.abs(value) > size && present) {
          size = Math.abs(value);
          best = [pearson.columns[a], pearson.columns[b]];
        }
      }
    }
    if (best) {
      return best;
    }
  }
  const features = numeric.filter((c) => c.name !== target);
  const pool = features.length > 0 ? features : numeric;
  return [pool[0]?.name ?? null, pool[1]?.name ?? pool[0]?.name ?? null];
}

function categoryColor(
  column: SampleColumn,
  rows: Int32Array,
  sample: DataSample,
  profile: DataProfile,
  target: string | null
): ScatterColor {
  // The target keeps its class colours (class order); other columns: most frequent first.
  const groups = column.name === target ? classGroups(sample, profile, target) : null;
  if (groups) {
    const shown = groups.groups.filter((g) => g.slot >= 0).slice(0, MAX_SCATTER_SERIES);
    const categories: ScatterCategory[] = shown.map((g, slot) => ({
      key: g.key,
      label: g.label,
      slot,
    }));
    const values = new Int32Array(rows.length);
    let other = false;
    rows.forEach((row, i) => {
      const g = groups.index[row];
      if (g >= 0 && g < shown.length) {
        values[i] = g;
      } else {
        other = true;
        values[i] = shown.length;
      }
    });
    if (other) {
      categories.push({
        key: '__other__',
        label: groups.groups.length > shown.length ? 'Other / missing' : 'Missing',
        slot: -1,
      });
    }
    return { type: 'category', label: column.name, values, categories };
  }
  const counts = new Map<string, number>();
  rows.forEach((row) => {
    const key = categoryKey(column.values[row]);
    if (key !== null) {
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
  });
  const keys = [...counts.entries()].sort((a, b) => b[1] - a[1]).map(([key]) => key);
  const shown = keys.slice(0, MAX_SCATTER_SERIES);
  const categories: ScatterCategory[] = shown.map((key, slot) => ({
    key,
    label: valueLabel(key, column.kind),
    slot,
  }));
  const position = new Map(shown.map((key, i) => [key, i]));
  const values = new Int32Array(rows.length);
  let other = false;
  rows.forEach((row, i) => {
    const key = categoryKey(column.values[row]);
    const p = key === null ? undefined : position.get(key);
    if (p === undefined) {
      other = true;
      values[i] = shown.length;
    } else {
      values[i] = p;
    }
  });
  if (other) {
    categories.push({
      key: '__other__',
      label: keys.length > shown.length ? 'Other / missing' : 'Missing',
      slot: -1,
    });
  }
  return { type: 'category', label: column.name, values, categories };
}

/** Two columns of the sample against each other, coloured by a third. */
export function ScatterView({
  profile,
  sample,
  target,
  options,
  onOptionsChange,
}: ScatterViewProps) {
  const axisColumns = sample.columns.filter((c) => isNumericKind(c.kind));
  const colorColumns = sample.columns.filter((c) => c.kind !== 'text');
  const xColumn = sampleColumn(sample, options.x);
  const yColumn = sampleColumn(sample, options.y);
  const colorColumn = sampleColumn(sample, options.color);
  const set = (patch: Partial<ScatterOptions>) => onOptionsChange({ ...options, ...patch });

  const data = useMemo(() => {
    if (!xColumn || !yColumn) {
      return null;
    }
    const pairs = numericPairs(xColumn, yColumn);
    let color: ScatterColor | undefined;
    if (colorColumn) {
      if (colorColumn.kind === 'numeric' || colorColumn.kind === 'datetime') {
        const values = new Float64Array(pairs.rows.length);
        pairs.rows.forEach((row, i) => {
          const v = colorColumn.values[row];
          values[i] = typeof v === 'number' ? v : NaN;
        });
        color = {
          type: 'numeric',
          label: colorColumn.name,
          values,
          format: colorColumn.kind === 'datetime' ? formatDateTime : undefined,
        };
      } else {
        color = categoryColor(colorColumn, pairs.rows, sample, profile, target);
      }
    }
    return { ...pairs, color, rho: spearman(pairs.x, pairs.y) };
  }, [xColumn, yColumn, colorColumn, sample, profile, target]);

  if (axisColumns.length === 0) {
    return (
      <Text size="sm" c="dimmed">
        A scatter plot needs number, yes/no or date columns.
      </Text>
    );
  }
  const discrete =
    !!xColumn &&
    !!yColumn &&
    [xColumn, yColumn].some(
      (c) => c.kind === 'boolean' || profile.columns.find((p) => p.name === c.name)?.integer
    );

  return (
    <Stack gap="md">
      <Group gap="sm" align="flex-end" wrap="wrap">
        <ColumnSelect
          label="X axis"
          columns={axisColumns}
          value={options.x}
          onChange={(x) => set({ x })}
          w={190}
        />
        <ColumnSelect
          label="Y axis"
          columns={axisColumns}
          value={options.y}
          onChange={(y) => set({ y })}
          w={190}
        />
        <ColumnSelect
          label="Colour"
          columns={colorColumns}
          value={options.color ?? '__none__'}
          onChange={(color) => set({ color: color === '__none__' ? null : color })}
          extra={[{ value: '__none__', label: 'None' }]}
          w={190}
        />
      </Group>
      <Group gap="md">
        <Switch
          size="xs"
          label="Trend line"
          checked={options.trend}
          onChange={(e) => set({ trend: e.currentTarget.checked })}
        />
        <Switch
          size="xs"
          label="Jitter"
          description={discrete ? 'Spreads points that sit on top of each other' : undefined}
          checked={options.jitter}
          onChange={(e) => set({ jitter: e.currentTarget.checked })}
        />
      </Group>

      {data && xColumn && yColumn ? (
        <ScatterPlot
          title={`${yColumn.name} vs ${xColumn.name}`}
          subtitle={`${formatCount(data.x.length)} of ${formatCount(sample.rows)} sample rows have both values. ${sampleNote(sample)}`}
          x={data.x}
          y={data.y}
          xLabel={xColumn.name}
          yLabel={yColumn.name}
          xTime={xColumn.kind === 'datetime'}
          yTime={yColumn.kind === 'datetime'}
          color={data.color}
          trend={options.trend}
          jitter={options.jitter}
          pointLabel={(i) => rowName(sample, data.rows[i])}
          caption={
            data.rho !== null
              ? `Spearman ρ (rank correlation) in the sample: ${formatCoefficient(data.rho)}.`
              : undefined
          }
        />
      ) : (
        <Text size="sm" c="dimmed">
          Pick two columns.
        </Text>
      )}
      <Note>
        The trend line is an ordinary least-squares fit; r is Pearson&apos;s correlation in the
        sample, R² the share of the variance of {yColumn?.name ?? 'y'} it explains. Correlation is
        not causation.
      </Note>
    </Stack>
  );
}
