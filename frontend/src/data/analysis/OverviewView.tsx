import { Paper, SimpleGrid, Stack, Text } from '@mantine/core';
import type { DataProfile, DataSample, NewStep } from '@/api/types';
import { seriesColor } from '@/components/charts/colors';
import { formatBytes, formatCount, formatPercent } from '@/components/charts/format';
import { SegmentedBar, StatTile } from '@/components/charts/StatTile';
import { i18n, useTranslation } from '@/i18n';
import { analysisNote, sampleNote } from './sampleData';
import { KIND_LABELS, KIND_ORDER, Note } from './shared';
import { groupWarnings } from './warningGroups';
import { WarningList } from './WarningList';

export const SPLIT_NAMES = ['Training', 'Validation', 'Test'];

export interface OverviewViewProps {
  profile: DataProfile;
  sample: DataSample;
  target: string | null;
  onAddStep: (step: NewStep) => void;
  onShowColumn: (column: string) => void;
}

/** "Warnings (3)", or "Warnings: 5 kinds, 1,062 in all" when many are grouped. */
export function warningHeading(profile: DataProfile): string {
  const total = profile.warnings.length;
  if (!total) {
    return 'Warnings';
  }
  const entries = groupWarnings(profile.warnings, profile).length;
  return entries < total
    ? i18n.t('ui.data.OverviewView.warningsInAll', {
        entries: formatCount(entries),
        value:
          entries === 1
            ? i18n.t('ui.data.OverviewView.entry')
            : i18n.t('ui.data.OverviewView.entries'),
        total: formatCount(total),
      })
    : i18n.t('ui.data.OverviewView.warnings', { total: formatCount(total) });
}

const lower = (text: string) => text.charAt(0).toLowerCase() + text.slice(1);

/** Headline numbers, column types and the warnings with one-click fixes. */
export function OverviewView({
  profile,
  sample,
  target,
  onAddStep,
  onShowColumn,
}: OverviewViewProps) {
  useTranslation();
  const cells = profile.n_rows * profile.n_columns;
  const split = profile.split_counts;
  const targetProfile = profile.target;
  const classes = targetProfile?.classes?.length ?? 0;
  const targetHint = targetProfile
    ? targetProfile.task === 'classification'
      ? `${formatCount(classes)} classes${targetProfile.imbalance_ratio ? ` · ${targetProfile.imbalance_ratio.toFixed(1)} : 1` : ''}`
      : i18n.t('ui.data.OverviewView.regression')
    : i18n.t('ui.data.OverviewView.chooseOneInTargetSplit');

  return (
    <Stack gap="md">
      <SimpleGrid type="container" cols={{ base: 2, '420px': 3, '1100px': 6 }} spacing="xs">
        <StatTile
          label={i18n.t('ui.data.OverviewView.rows')}
          value={formatCount(profile.n_rows)}
          hint={formatBytes(profile.memory_bytes)}
        />
        <StatTile
          label={i18n.t('ui.data.OverviewView.columns')}
          value={formatCount(profile.n_columns)}
          hint={
            targetProfile || target
              ? i18n.t('ui.data.OverviewView.featuresTarget', {
                  max: formatCount(Math.max(0, profile.n_columns - 1)),
                })
              : i18n.t('ui.data.OverviewView.noTargetYet')
          }
        />
        <StatTile
          label={i18n.t('ui.data.OverviewView.missingCells')}
          value={formatCount(profile.missing_cells)}
          hint={i18n.t('ui.data.OverviewView.ofAllCells', {
            value: formatPercent(cells ? profile.missing_cells / cells : 0),
          })}
        />
        <StatTile
          label={i18n.t('ui.data.OverviewView.duplicateRows')}
          value={formatCount(profile.duplicate_rows)}
          hint={
            profile.analysis_rows < profile.n_rows
              ? i18n.t('ui.data.OverviewView.inCheckedRows', {
                  analysis_rows: formatCount(profile.analysis_rows),
                })
              : i18n.t('ui.data.OverviewView.ofRows', {
                  value: formatPercent(
                    profile.n_rows ? profile.duplicate_rows / profile.n_rows : 0
                  ),
                })
          }
        />
        <StatTile
          label={i18n.t('ui.data.OverviewView.target')}
          value={targetProfile?.column ?? target ?? '–'}
          hint={targetHint}
        />
        <StatTile
          label={i18n.t('ui.data.OverviewView.split')}
          value={split ? split.map(formatCount).join(' / ') : '–'}
          hint={
            split
              ? i18n.t('ui.data.OverviewView.trainValidationTest')
              : i18n.t('ui.data.OverviewView.assignedAfterCombining')
          }
        >
          {split && (
            <SegmentedBar
              label={i18n.t('ui.data.OverviewView.rowsPerSplit')}
              showLegend={false}
              segments={split.map((value, i) => ({
                key: SPLIT_NAMES[i],
                label: SPLIT_NAMES[i],
                value,
                color: seriesColor(i),
              }))}
            />
          )}
        </StatTile>
      </SimpleGrid>

      <Paper withBorder p="sm" radius="sm">
        <Text size="sm" fw={600}>
          {i18n.t('ui.data.OverviewView.columnTypes')}
        </Text>
        <SegmentedBar
          label={i18n.t('ui.data.OverviewView.columnsPerType')}
          segments={KIND_ORDER.map((kind, i) => ({
            key: kind,
            label: KIND_LABELS[kind],
            value: profile.columns.filter((c) => c.kind === kind).length,
            color: seriesColor(i),
          }))}
        />
      </Paper>

      <Stack gap={6}>
        <Text size="sm" fw={600}>
          {warningHeading(profile)}
        </Text>
        <WarningList
          warnings={profile.warnings}
          profile={profile}
          resetKey={profile}
          onAddStep={onAddStep}
          onShowColumn={onShowColumn}
          emptyText={i18n.t('ui.data.OverviewView.noProblemsFound')}
        />
      </Stack>

      {/* What the figures are based on matters only when part of them come from a sample. */}
      {(analysisNote(profile) || sampleNote(sample)) && (
        <Note>
          {i18n.t('ui.data.OverviewView.columnStatisticsAreExactAll', {
            n_rows: formatCount(profile.n_rows),
            lower: lower(analysisNote(profile, true)),
            lower2: lower(sampleNote(sample, true)),
          })}
        </Note>
      )}
    </Stack>
  );
}
