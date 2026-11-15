import { useMemo, useState } from 'react';
import { IconTrash } from '@tabler/icons-react';
import { Button, Group, NumberInput, SimpleGrid, Stack, Text } from '@mantine/core';
import type { DataProfile, DataSample, NewStep } from '@/api/types';
import { BarChart, type BarRow } from '@/components/charts/BarChart';
import { BoxPlot } from '@/components/charts/BoxPlot';
import { OTHER_COLOR, seriesColor } from '@/components/charts/colors';
import {
  formatCoefficient,
  formatCount,
  formatNumber,
  formatPercent,
} from '@/components/charts/format';
import { Histogram } from '@/components/charts/Histogram';
import { boxFromStats } from '@/components/charts/stats';
import { StatTile } from '@/components/charts/StatTile';
import { i18n, useTranslation } from '@/i18n';
import { usePrepAccess } from '../prepAccess';
import { ASSOCIATION_LABELS, SHORT_METHOD } from './ColumnsView';
import { KeyValueTable } from './DistributionView';
import { SPLIT_NAMES } from './OverviewView';
import { analysisNote, classGroups, classLabel, sampleNote } from './sampleData';
import { Note } from './shared';

export interface TargetViewProps {
  profile: DataProfile;
  sample: DataSample;
  target: string | null;
  onShowColumn: (column: string) => void;
  onAddStep?: (step: NewStep) => void;
}

const RANKING = 25;
/** The default of "Select features: little association with the target". */
const DEFAULT_MIN_ASSOCIATION = 0.05;
const LISTED = 8;

/** Class balance or the target's distribution, and the features ranked by association. */
export function TargetView({ profile, sample, target, onShowColumn, onAddStep }: TargetViewProps) {
  useTranslation();
  const [allFeatures, setAllFeatures] = useState(false);
  const [minAssociation, setMinAssociation] = useState(DEFAULT_MIN_ASSOCIATION);
  const [added, setAdded] = useState<string[]>([]);
  const { readOnly } = usePrepAccess();
  const info = profile.target;
  const ranking = useMemo(
    () =>
      profile.columns
        .filter((c) => c.target_association && c.name !== info?.column)
        .sort((a, b) => b.target_association!.value - a.target_association!.value),
    [profile.columns, info]
  );
  const groups = useMemo(
    () => (info?.task === 'classification' ? classGroups(sample, profile, info.column) : null),
    [info, sample, profile]
  );
  const splitMix = useMemo(() => {
    if (!groups || !sample.split) {
      return null;
    }
    const keys = ['train', 'val', 'test'] as const;
    const rows = keys.map(() => new Array<number>(groups.groups.length).fill(0));
    sample.split.forEach((part, row) => {
      const g = groups.index[row];
      if (g >= 0) {
        rows[keys.indexOf(part)][g] += 1;
      }
    });
    return rows
      .map((values, i): BarRow => ({ key: keys[i], label: SPLIT_NAMES[i], values }))
      .filter((row) => row.values.some((v) => v > 0));
  }, [groups, sample.split]);

  if (!info) {
    return (
      <Text size="sm" c="dimmed">
        {target
          ? i18n.t('ui.data.TargetView.theTargetColumnIsNot', { target })
          : i18n.t('ui.data.TargetView.chooseATargetColumnIn')}
      </Text>
    );
  }

  const shownRanking = allFeatures ? ranking : ranking.slice(0, RANKING);
  const targetKind = profile.columns.find((c) => c.name === info.column)?.kind ?? 'categorical';
  const className = (name: string) => classLabel(name, targetKind);
  const methods = [...new Set(ranking.map((c) => c.target_association!.method))];

  return (
    <Stack gap="md">
      {info.task === 'classification' && info.classes ? (
        <>
          {(() => {
            const total = info.classes.reduce((sum, c) => sum + c.count, 0);
            const largest = info.classes.reduce(
              (best, c) => (c.count > best.count ? c : best),
              info.classes[0]
            );
            return (
              <>
                <SimpleGrid type="container" cols={{ base: 2, '520px': 3 }} spacing="xs">
                  <StatTile
                    label={i18n.t('ui.data.TargetView.classes')}
                    value={formatCount(info.classes.length)}
                    hint={info.column}
                  />
                  <StatTile
                    label={i18n.t('ui.data.TargetView.imbalance')}
                    value={
                      info.imbalance_ratio ? `${formatNumber(info.imbalance_ratio, 3)} : 1` : '–'
                    }
                    hint={i18n.t('ui.data.TargetView.largestSmallestClass')}
                  />
                  <StatTile
                    label={i18n.t('ui.data.TargetView.majorityBaseline')}
                    value={largest ? formatPercent(total ? largest.count / total : 0) : '–'}
                    hint={
                      largest
                        ? i18n.t('ui.data.TargetView.accuracyOfAlways', {
                            className: className(largest.value),
                          })
                        : undefined
                    }
                  />
                </SimpleGrid>
                <BarChart
                  title={i18n.t('ui.data.TargetView.classesOf', { column: info.column })}
                  subtitle={i18n.t('ui.data.TargetView.rowsPerClassAllRows', {
                    n_rows: formatCount(profile.n_rows),
                  })}
                  rows={info.classes.map((c) => ({
                    key: c.value,
                    label: className(c.value),
                    values: [c.count],
                  }))}
                  tipLabel={(row) =>
                    `${formatCount(row.values[0])} · ${formatPercent(total ? row.values[0] / total : 0)}`
                  }
                  unit="rows"
                />
              </>
            );
          })()}
          {splitMix && groups && splitMix.length > 0 && (
            <BarChart
              title={i18n.t('ui.data.TargetView.classMixPerSplit')}
              subtitle={i18n.t('ui.data.TargetView.similarMixesMeanTheSplit', {
                sampleNote: sampleNote(sample),
              })}
              rows={splitMix}
              series={groups.groups.map((g) => ({
                key: g.key,
                label: g.label,
                color: g.slot >= 0 ? seriesColor(g.slot) : OTHER_COLOR,
              }))}
              layout="stacked"
              normalize
              unit="rows"
            />
          )}
        </>
      ) : (
        <>
          {info.histogram && (
            <Histogram
              title={i18n.t('ui.data.TargetView.distributionOf', { column: info.column })}
              subtitle={i18n.t('ui.data.TargetView.rowsPerBinAllRows', {
                n_rows: formatCount(profile.n_rows),
              })}
              edges={info.histogram.edges}
              series={[
                {
                  key: 'target',
                  label: info.column,
                  color: seriesColor(0),
                  counts: info.histogram.counts,
                },
              ]}
              xLabel={info.column}
              unit="rows"
            />
          )}
          {info.stats && boxFromStats(info.stats) && (
            <BoxPlot
              title={i18n.t('ui.data.TargetView.boxPlot')}
              groups={[{ key: 'target', label: info.column, box: boxFromStats(info.stats)! }]}
            />
          )}
          {info.stats && (
            <KeyValueTable
              label={i18n.t('ui.data.TargetView.statisticsOf', { column: info.column })}
              rows={[
                ['Mean', formatNumber(info.stats.mean)],
                ['Standard deviation', formatNumber(info.stats.std)],
                ['Median', formatNumber(info.stats.median)],
                ['Min – max', `${formatNumber(info.stats.min)} – ${formatNumber(info.stats.max)}`],
                ['Skewness', formatNumber(info.stats.skew)],
              ]}
            />
          )}
        </>
      )}

      {ranking.length > 0 ? (
        <>
          <BarChart
            title={i18n.t('ui.data.TargetView.featuresByAssociationWithThe')}
            subtitle={i18n.t('ui.data.TargetView.text0NoRelation1Perfect', {
              analysisNote: analysisNote(profile),
            })}
            rows={shownRanking.map((c) => ({
              key: c.name,
              label: c.name,
              values: [c.target_association!.value],
            }))}
            max={1}
            formatValue={formatCoefficient}
            tipLabel={(row) => {
              const column = ranking.find((c) => c.name === row.key);
              return `${formatCoefficient(row.values[0])} ${column ? SHORT_METHOD[column.target_association!.method] : ''}`;
            }}
            onSelect={onShowColumn}
            selectHint="show the column"
            unit="association"
          />
          {ranking.length > RANKING && (
            <Button
              variant="subtle"
              size="compact-sm"
              onClick={() => setAllFeatures((v) => !v)}
              style={{ alignSelf: 'flex-start' }}
            >
              {allFeatures
                ? i18n.t('ui.data.TargetView.showTheTop', { RANKING })
                : i18n.t('ui.data.TargetView.showAllFeatures', {
                    length: formatCount(ranking.length),
                  })}
            </Button>
          )}
          <WeakFeatures
            ranking={ranking.map((c) => ({ name: c.name, value: c.target_association!.value }))}
            threshold={minAssociation}
            onThresholdChange={setMinAssociation}
            canAdd={!readOnly && !!onAddStep}
            added={added}
            onAdd={(key, step) => {
              onAddStep?.(step);
              setAdded((keys) => [...keys, key]);
            }}
          />
          <Note>
            {i18n.t('ui.data.TargetView.measures')}{' '}
            {methods.map((m) => `${SHORT_METHOD[m]} = ${ASSOCIATION_LABELS[m]}`).join(', ')}.
            {methods.length > 1 && i18n.t('ui.data.TargetView.differentMeasuresAreOnlyRoughly')}{' '}
            {i18n.t('ui.data.TargetView.aLowValueDoesNot')}
          </Note>
        </>
      ) : (
        <Note>{i18n.t('ui.data.TargetView.noFeatureAssociationsYetThey')}</Note>
      )}
    </Stack>
  );
}

interface WeakFeaturesProps {
  ranking: { name: string; value: number }[];
  threshold: number;
  onThresholdChange: (value: number) => void;
  canAdd: boolean;
  added: string[];
  onAdd: (key: string, step: NewStep) => void;
}

/** Pick a minimum association and drop the features below it, as a step learned on the
 * training rows or as a fixed list of columns. */
function WeakFeatures({
  ranking,
  threshold,
  onThresholdChange,
  canAdd,
  added,
  onAdd,
}: WeakFeaturesProps) {
  useTranslation();
  const weak = ranking.filter((c) => c.value < threshold).map((c) => c.name);
  const listed = weak.slice(0, LISTED).join(', ');
  const more = weak.length > LISTED ? ` and ${formatCount(weak.length - LISTED)} more` : '';
  const stepKey = `select:${threshold}`;
  const listKey = `drop:${threshold}:${weak.length}`;
  return (
    <Stack gap={6} aria-label={i18n.t('ui.data.TargetView.dropWeaklyAssociatedFeatures')}>
      <Group gap="xs" wrap="wrap" align="flex-end">
        <NumberInput
          size="xs"
          label={i18n.t('ui.data.TargetView.dropBelowAssociation')}
          value={threshold}
          min={0}
          max={1}
          step={0.01}
          decimalScale={2}
          w={170}
          onChange={(value) => {
            if (typeof value === 'number' && Number.isFinite(value)) {
              onThresholdChange(Math.min(1, Math.max(0, value)));
            }
          }}
        />
        {canAdd && (
          <>
            <Button
              size="compact-sm"
              variant="filled"
              leftSection={<IconTrash size={14} />}
              disabled={weak.length === 0 || added.includes(stepKey)}
              onClick={() =>
                onAdd(stepKey, {
                  type: 'feature_selection',
                  params: { method: 'target_correlation', min_association: threshold },
                })
              }
            >
              {added.includes(stepKey)
                ? i18n.t('ui.data.TargetView.stepAdded')
                : i18n.t('ui.data.TargetView.addAStepDropBelow')}
            </Button>
            <Button
              size="compact-sm"
              variant="light"
              disabled={weak.length === 0 || added.includes(listKey)}
              onClick={() => onAdd(listKey, { type: 'drop_columns', params: { columns: weak } })}
            >
              {added.includes(listKey)
                ? i18n.t('ui.data.TargetView.columnsDropped')
                : i18n.t('ui.data.TargetView.dropTheseColumns', {
                    length: formatCount(weak.length),
                  })}
            </Button>
          </>
        )}
      </Group>
      <Text size="sm">
        {weak.length === 0
          ? i18n.t('ui.data.TargetView.noFeatureIsBelow', {
              threshold: formatCoefficient(threshold),
            })
          : i18n.t('ui.data.TargetView.ofFeaturesAreBelow', {
              length: formatCount(weak.length),
              length2: formatCount(ranking.length),
              threshold: formatCoefficient(threshold),
              listed,
              more,
            })}
      </Text>
      <Note>{i18n.t('ui.data.TargetView.addAStepAddsSelect')}</Note>
    </Stack>
  );
}
