import { useMemo } from 'react';
import { Alert, Group, Paper, Progress, Stack, Text } from '@mantine/core';
import type { RowAnalysis } from '@/api/types';
import { seriesColor } from '@/components/charts/colors';
import { formatCount, formatNumber, formatPercent } from '@/components/charts/format';
import { Histogram } from '@/components/charts/Histogram';
import { closePairs, histogramCounts, median, pairTargets, randomPairs } from './rowsData';
import classes from './RowsView.module.css';

export interface PairsViewProps {
  analysis: RowAnalysis;
  /** NN ratio below which a row and its nearest row are a close pair. */
  threshold: number;
}

/**
 * Close pairs (a row and its nearest row, ratio below the threshold) against random pairs of
 * rows: how much their targets differ. Uses the labels, so it is for understanding the
 * structure, not for choosing rows.
 */
export function PairsView({ analysis, threshold }: PairsViewProps) {
  const close = useMemo(() => closePairs(analysis, threshold), [analysis, threshold]);
  const random = useMemo(() => randomPairs(analysis), [analysis]);
  const closeTargets = useMemo(() => pairTargets(analysis, close), [analysis, close]);
  const randomTargets = useMemo(() => pairTargets(analysis, random), [analysis, random]);
  const target = analysis.target;

  const header = (
    <div>
      <Text fw={600} size="sm">
        Close pairs vs random pairs
      </Text>
      <Text size="xs" c="dimmed">
        A close pair is a row and its nearest row with an NN ratio below{' '}
        {formatNumber(threshold, 3)} ({formatCount(close.length)} pairs). This view uses the labels
        ({target ?? 'no target'}): use it to see what the near twins are, not to pick rows.
      </Text>
    </div>
  );

  if (!target || analysis.rows.target === null) {
    return (
      <Paper withBorder radius="md" p="sm" className={classes.card}>
        <Stack gap="xs">
          {header}
          <Text size="sm" c="dimmed">
            Choose a target in “Target &amp; split” to compare the target values of close and random
            pairs.
          </Text>
        </Stack>
      </Paper>
    );
  }
  if (close.length === 0) {
    return (
      <Paper withBorder radius="md" p="sm" className={classes.card}>
        <Stack gap="xs">
          {header}
          <Text size="sm" c="dimmed">
            No row has a near twin below this threshold.
          </Text>
        </Stack>
      </Paper>
    );
  }

  if (analysis.task === 'regression') {
    const all = [...closeTargets.diffs, ...randomTargets.diffs];
    const top = Math.max(1e-9, ...all);
    const bins = 24;
    const edges = Array.from({ length: bins + 1 }, (_, i) => (top * i) / bins);
    const closeMedian = median(closeTargets.diffs);
    const randomMedian = median(randomTargets.diffs);
    return (
      <Paper withBorder radius="md" p="sm" className={classes.card}>
        <Stack gap="xs">
          {header}
          <Group gap="lg" data-testid="pair-medians">
            <Text size="sm">
              Close pairs: median difference <b>{formatNumber(closeMedian, 3)}</b>
            </Text>
            <Text size="sm">
              Random pairs: median difference <b>{formatNumber(randomMedian, 3)}</b>
            </Text>
          </Group>
          <Histogram
            title={`Difference of ${target} within a pair`}
            edges={edges}
            normalize
            unit="pairs"
            xLabel={`|Δ ${target}|`}
            height={180}
            series={[
              {
                key: 'close',
                label: 'Close pairs',
                color: seriesColor(1),
                counts: histogramCounts(closeTargets.diffs, edges),
              },
              {
                key: 'random',
                label: 'Random pairs',
                color: seriesColor(0),
                counts: histogramCounts(randomTargets.diffs, edges),
              },
            ]}
          />
          {closeMedian !== null && randomMedian !== null && closeMedian < randomMedian * 0.5 && (
            <Alert color="yellow" p="xs">
              <Text size="sm">
                Near twins have very similar targets. If a row's twin is in the training data while
                the row is in validation or test, the scores look better than they are on new data:
                keep the groups together in the split.
              </Text>
            </Alert>
          )}
        </Stack>
      </Paper>
    );
  }

  const closeShare = closeTargets.known ? closeTargets.same / closeTargets.known : 0;
  const randomShare = randomTargets.known ? randomTargets.same / randomTargets.known : 0;
  return (
    <Paper withBorder radius="md" p="sm" className={classes.card}>
      <Stack gap="xs">
        {header}
        <Text size="sm">Same class within a pair</Text>
        <div>
          <Text size="xs">Close pairs: {formatPercent(closeShare)}</Text>
          <Progress value={closeShare * 100} color="orange" aria-label="Close pairs, same class" />
        </div>
        <div>
          <Text size="xs">Random pairs: {formatPercent(randomShare)}</Text>
          <Progress value={randomShare * 100} aria-label="Random pairs, same class" />
        </div>
      </Stack>
    </Paper>
  );
}
