import { useMemo } from 'react';
import { Alert, Group, Paper, Progress, Stack, Text } from '@mantine/core';
import type { RowAnalysis } from '@/api/types';
import { seriesColor } from '@/components/charts/colors';
import { formatCount, formatNumber, formatPercent } from '@/components/charts/format';
import { Histogram } from '@/components/charts/Histogram';
import { i18n, Trans, useTranslation } from '@/i18n';
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
  useTranslation();
  const close = useMemo(() => closePairs(analysis, threshold), [analysis, threshold]);
  const random = useMemo(() => randomPairs(analysis), [analysis]);
  const closeTargets = useMemo(() => pairTargets(analysis, close), [analysis, close]);
  const randomTargets = useMemo(() => pairTargets(analysis, random), [analysis, random]);
  const target = analysis.target;

  const header = (
    <div>
      <Text fw={600} size="sm">
        {i18n.t('ui.data.PairsView.closePairsVsRandomPairs')}
      </Text>
      <Text size="xs" c="dimmed">
        {i18n.t('ui.data.PairsView.aClosePairIsA', {
          threshold: formatNumber(threshold, 3),
          length: formatCount(close.length),
          value: target ?? i18n.t('ui.data.PairsView.noTarget'),
        })}
      </Text>
    </div>
  );

  if (!target || analysis.rows.target === null) {
    return (
      <Paper withBorder radius="md" p="sm" className={classes.card}>
        <Stack gap="xs">
          {header}
          <Text size="sm" c="dimmed">
            {i18n.t('ui.data.PairsView.chooseATargetInTarget')}
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
            {i18n.t('ui.data.PairsView.noRowHasANear')}
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
              <Trans
                i18nKey="ui.data.PairsView.closePairsMedianDifference"
                values={{ closeMedian: formatNumber(closeMedian, 3) }}
                components={[<b />]}
              />
            </Text>
            <Text size="sm">
              <Trans
                i18nKey="ui.data.PairsView.randomPairsMedianDifference"
                values={{ randomMedian: formatNumber(randomMedian, 3) }}
                components={[<b />]}
              />
            </Text>
          </Group>
          <Histogram
            title={i18n.t('ui.data.PairsView.differenceOfWithinAPair', { target })}
            edges={edges}
            normalize
            unit="pairs"
            xLabel={`|Δ ${target}|`}
            height={180}
            series={[
              {
                key: 'close',
                label: i18n.t('ui.data.PairsView.closePairs'),
                color: seriesColor(1),
                counts: histogramCounts(closeTargets.diffs, edges),
              },
              {
                key: 'random',
                label: i18n.t('ui.data.PairsView.randomPairs'),
                color: seriesColor(0),
                counts: histogramCounts(randomTargets.diffs, edges),
              },
            ]}
          />
          {closeMedian !== null && randomMedian !== null && closeMedian < randomMedian * 0.5 && (
            <Alert color="yellow" p="xs">
              <Text size="sm">{i18n.t('ui.data.PairsView.nearTwinsHaveVerySimilar')}</Text>
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
        <Text size="sm">{i18n.t('ui.data.PairsView.sameClassWithinAPair')}</Text>
        <div>
          <Text size="xs">
            {i18n.t('ui.data.PairsView.closePairs2', { closeShare: formatPercent(closeShare) })}
          </Text>
          <Progress
            value={closeShare * 100}
            color="orange"
            aria-label={i18n.t('ui.data.PairsView.closePairsSameClass')}
          />
        </div>
        <div>
          <Text size="xs">
            {i18n.t('ui.data.PairsView.randomPairs2', { randomShare: formatPercent(randomShare) })}
          </Text>
          <Progress
            value={randomShare * 100}
            aria-label={i18n.t('ui.data.PairsView.randomPairsSameClass')}
          />
        </div>
      </Stack>
    </Paper>
  );
}
