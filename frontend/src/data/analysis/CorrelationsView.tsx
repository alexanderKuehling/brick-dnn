import { useMemo, useState } from 'react';
import { IconSearch } from '@tabler/icons-react';
import {
  Badge,
  Group,
  SegmentedControl,
  Stack,
  Text,
  TextInput,
  UnstyledButton,
} from '@mantine/core';
import type { DataProfile } from '@/api/types';
import { divergingColor, useChartScheme } from '@/components/charts/colors';
import { formatCoefficient } from '@/components/charts/format';
import { Heatmap, strongestPairs } from '@/components/charts/Heatmap';
import { analysisNote } from './sampleData';
import { Note } from './shared';
import classes from './AnalysisPanel.module.css';

export type CorrelationMethod = 'pearson' | 'spearman';

export interface CorrelationsViewProps {
  profile: DataProfile;
  method: CorrelationMethod;
  onMethodChange: (method: CorrelationMethod) => void;
  /** Open the scatter plot of two columns. */
  onScatter: (x: string, y: string) => void;
}

const METHOD_LABELS: Record<CorrelationMethod, string> = {
  pearson: 'Pearson',
  spearman: 'Spearman',
};

/** Correlation heatmap (Pearson or Spearman) and the strongest pairs; a click opens the scatter plot. */
export function CorrelationsView({
  profile,
  method,
  onMethodChange,
  onScatter,
}: CorrelationsViewProps) {
  const scheme = useChartScheme();
  const [query, setQuery] = useState('');
  const available = profile.correlations.map((c) => c.method);
  const matrix =
    profile.correlations.find((c) => c.method === method) ?? profile.correlations[0] ?? null;

  const filtered = useMemo(() => {
    if (!matrix) {
      return null;
    }
    const q = query.trim().toLowerCase();
    const keep = matrix.columns
      .map((name, i) => ({ name, i }))
      .filter(({ name }) => !q || name.toLowerCase().includes(q));
    return {
      columns: keep.map((k) => k.name),
      values: keep.map(({ i }) => keep.map(({ i: j }) => matrix.matrix[i]?.[j] ?? null)),
    };
  }, [matrix, query]);
  const pairs = useMemo(
    () => (matrix ? strongestPairs(matrix.columns, matrix.matrix, 10) : []),
    [matrix]
  );

  if (!matrix || matrix.columns.length < 2 || !filtered) {
    return (
      <Text size="sm" c="dimmed">
        Correlations need at least two number or yes/no columns.
      </Text>
    );
  }
  const label = METHOD_LABELS[matrix.method];
  const measure = matrix.method === 'pearson' ? 'Pearson r' : 'Spearman ρ';

  return (
    <Stack gap="md">
      <Group gap="sm" wrap="wrap">
        {available.length > 1 && (
          <SegmentedControl
            size="xs"
            aria-label="Correlation method"
            value={matrix.method}
            onChange={(value) => onMethodChange(value as CorrelationMethod)}
            data={available.map((m) => ({ value: m, label: METHOD_LABELS[m] }))}
          />
        )}
        {matrix.columns.length > 8 && (
          <TextInput
            size="xs"
            placeholder="Filter columns"
            aria-label="Filter the heatmap columns"
            leftSection={<IconSearch size={14} />}
            value={query}
            onChange={(event) => setQuery(event.currentTarget.value)}
            style={{ flex: '0 1 220px' }}
          />
        )}
      </Group>

      {filtered.columns.length > 0 ? (
        <Heatmap
          title={`${label} correlation`}
          subtitle={`${filtered.columns.length} number and yes/no columns. ${analysisNote(profile)}`}
          rows={filtered.columns}
          columns={filtered.columns}
          values={filtered.values}
          measure={measure}
          onCellClick={(r, c) => {
            if (r !== c) {
              onScatter(filtered.columns[c], filtered.columns[r]);
            }
          }}
          clickHint="open the scatter plot"
        />
      ) : (
        <Text size="sm" c="dimmed">
          No column matches the filter.
        </Text>
      )}

      {pairs.length > 0 && (
        <Stack gap={4}>
          <Text size="sm" fw={600}>
            Strongest pairs
          </Text>
          <ul className={classes.pairList}>
            {pairs.map((p) => {
              const a = matrix.columns[p.a];
              const b = matrix.columns[p.b];
              const size = Math.abs(p.value);
              return (
                <li key={`${a}|${b}`}>
                  <UnstyledButton
                    className={classes.pair}
                    onClick={() => onScatter(a, b)}
                    aria-label={`${a} and ${b}: ${measure} ${formatCoefficient(p.value)}. Open the scatter plot.`}
                  >
                    <svg width={64} height={8} aria-hidden="true" style={{ flexShrink: 0 }}>
                      <rect width={64} height={8} rx={4} style={{ fill: 'var(--chart-track)' }} />
                      <rect
                        width={Math.max(4, size * 64)}
                        height={8}
                        rx={4}
                        fill={divergingColor(p.value, scheme)}
                      />
                    </svg>
                    <Text size="xs" className={classes.number} w={40} ta="right">
                      {formatCoefficient(p.value)}
                    </Text>
                    <Text size="xs" truncate style={{ flex: 1 }}>
                      {a} × {b}
                    </Text>
                    {size > 0.95 && (
                      <Badge size="xs" variant="light" color="gray">
                        nearly redundant
                      </Badge>
                    )}
                  </UnstyledButton>
                </li>
              );
            })}
          </ul>
        </Stack>
      )}

      <Note>
        {matrix.method === 'pearson'
          ? 'Pearson r measures linear relations (sensitive to outliers).'
          : 'Spearman ρ is the correlation of ranks: it finds any monotonic relation and is robust to outliers.'}{' '}
        −1 = perfectly opposite, 0 = no relation, +1 = perfectly together. Constant columns have no
        correlation (–).
      </Note>
    </Stack>
  );
}
