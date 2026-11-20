import { memo, useMemo, type ReactNode } from 'react';
import { Badge, Group, SegmentedControl, Stack, Switch, Text, UnstyledButton } from '@mantine/core';
import type { DataProfile, DataSample } from '@/api/types';
import { divergingColor, useChartScheme } from '@/components/charts/colors';
import { formatCoefficient, formatCount } from '@/components/charts/format';
import { Heatmap, strongestPairs } from '@/components/charts/Heatmap';
import { i18n, useTranslation } from '@/i18n';
import { clusterOrder, sampleMatrix } from './relations';
import { analysisNote } from './sampleData';
import { Note } from './shared';
import classes from './AnalysisPanel.module.css';

export type CorrelationMethod = 'pearson' | 'spearman';

export interface CorrelationsViewProps {
  profile: DataProfile;
  sample: DataSample;
  method: CorrelationMethod;
  onMethodChange: (method: CorrelationMethod) => void;
  /** Open the scatter plot of two columns. */
  onScatter: (x: string, y: string) => void;
  /** The columns to show (null: those of the profile's correlation matrices). */
  columns?: string[] | null;
  /** Number and yes/no columns in all, for "showing N of M". */
  total?: number;
  /** Order related columns next to each other (hierarchical clustering). */
  clustered: boolean;
  onClusteredChange: (clustered: boolean) => void;
  /** Controls choosing the columns, shown above the heatmap. */
  chooser?: ReactNode;
}

const METHOD_LABELS: Record<CorrelationMethod, string> = {
  get pearson() {
    return i18n.t('ui.data.CorrelationsView.pearson');
  },
  get spearman() {
    return i18n.t('ui.data.CorrelationsView.spearman');
  },
};

interface ShownMatrix {
  columns: string[];
  values: (number | null)[][];
  /** Where the values come from, for the note. */
  basis: string;
}

/**
 * The correlation heatmap of chosen columns (Pearson or Spearman) and their strongest pairs;
 * a click opens the scatter plot. Columns of the profile's matrices use those (all analysis
 * rows); other choices are computed from the training rows of the preview sample.
 */
export const CorrelationsView = memo(
  ({
    profile,
    sample,
    method,
    onMethodChange,
    onScatter,
    columns = null,
    total,
    clustered,
    onClusteredChange,
    chooser,
  }: CorrelationsViewProps) => {
    useTranslation();
    const scheme = useChartScheme();
    const server = profile.correlations.find((c) => c.method === method) ?? null;

    const shown = useMemo((): ShownMatrix | null => {
      const wanted = columns ?? server?.columns ?? [];
      if (wanted.length < 2) {
        return null;
      }
      let matrix: ShownMatrix;
      const index = new Map((server?.columns ?? []).map((name, i) => [name, i]));
      if (server && wanted.every((name) => index.has(name))) {
        const at = wanted.map((name) => index.get(name)!);
        matrix = {
          columns: wanted,
          values: at.map((i) => at.map((j) => server.matrix[i]?.[j] ?? null)),
          basis: analysisNote(profile),
        };
      } else {
        const computed = sampleMatrix(sample, wanted, method);
        matrix = {
          columns: wanted,
          values: computed.values,
          basis: computed.trainingRows
            ? i18n.t('ui.data.CorrelationsView.computedOnTheTrainingRows', {
                rows: formatCount(computed.rows),
              })
            : i18n.t('ui.data.CorrelationsView.computedOnTheRows', {
                rows: formatCount(computed.rows),
              }),
        };
      }
      if (!clustered || matrix.columns.length < 3) {
        return matrix;
      }
      const order = clusterOrder(matrix.values);
      return {
        ...matrix,
        columns: order.map((i) => matrix.columns[i]),
        values: order.map((i) => order.map((j) => matrix.values[i][j])),
      };
    }, [columns, server, sample, method, clustered, profile]);

    const pairs = useMemo(
      () => (shown ? strongestPairs(shown.columns, shown.values, 10) : []),
      [shown]
    );

    const measure = method === 'pearson' ? 'Pearson r' : 'Spearman ρ';
    const all = total ?? server?.columns.length ?? shown?.columns.length ?? 0;

    return (
      <Stack gap="md">
        <Group gap="sm" wrap="wrap" align="flex-end">
          <SegmentedControl
            size="xs"
            aria-label={i18n.t('ui.data.CorrelationsView.correlationMethod')}
            value={method}
            onChange={(value) => onMethodChange(value as CorrelationMethod)}
            data={(['pearson', 'spearman'] as const).map((m) => ({
              value: m,
              label: METHOD_LABELS[m],
            }))}
          />
          <Switch
            size="xs"
            label={i18n.t('ui.data.CorrelationsView.groupRelatedColumns')}
            checked={clustered}
            onChange={(event) => onClusteredChange(event.currentTarget.checked)}
          />
          {chooser}
        </Group>

        {shown ? (
          <Heatmap
            title={i18n.t('ui.data.CorrelationsView.methodCorrelation', {
              method: METHOD_LABELS[method],
            })}
            subtitle={i18n.t('ui.data.CorrelationsView.showingOfNumberAndYes', {
              length: formatCount(shown.columns.length),
              all: formatCount(all),
              value: clustered ? i18n.t('ui.data.CorrelationsView.relatedOnesNextToEach') : '',
              basis: shown.basis,
            })}
            rows={shown.columns}
            columns={shown.columns}
            values={shown.values}
            measure={measure}
            onCellClick={(r, c) => {
              if (r !== c) {
                onScatter(shown.columns[c], shown.columns[r]);
              }
            }}
            clickHint={i18n.t('ui.data.CorrelationsView.openTheScatterPlot')}
          />
        ) : (
          <Text size="sm" c="dimmed">
            {columns
              ? i18n.t('ui.data.CorrelationsView.chooseAtLeastTwoColumns')
              : i18n.t('ui.data.CorrelationsView.correlationsNeedAtLeastTwo')}
          </Text>
        )}

        {pairs.length > 0 && shown && (
          <Stack gap={4}>
            <Text size="sm" fw={600}>
              {i18n.t('ui.data.CorrelationsView.strongestPairs')}
            </Text>
            <ul className={classes.pairList}>
              {pairs.map((p) => {
                const a = shown.columns[p.a];
                const b = shown.columns[p.b];
                const size = Math.abs(p.value);
                return (
                  <li key={`${a}|${b}`}>
                    <UnstyledButton
                      className={classes.pair}
                      onClick={() => onScatter(a, b)}
                      aria-label={i18n.t('ui.data.CorrelationsView.andOpenTheScatterPlot', {
                        a,
                        b,
                        measure,
                        value: formatCoefficient(p.value),
                      })}
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
                          {i18n.t('ui.data.CorrelationsView.nearlyRedundant')}
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
          {i18n.t('ui.data.CorrelationsView.text1PerfectlyOpposite0No', {
            value:
              method === 'pearson'
                ? i18n.t('ui.data.CorrelationsView.pearsonRMeasuresLinearRelations')
                : i18n.t('ui.data.CorrelationsView.spearmanIsTheCorrelationOf'),
          })}
        </Note>
      </Stack>
    );
  }
);
