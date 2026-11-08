import { Paper, SimpleGrid, Stack, Text } from '@mantine/core';
import type { DataProfile, DataSample, NewStep } from '@/api/types';
import { seriesColor } from '@/components/charts/colors';
import { formatBytes, formatCount, formatPercent } from '@/components/charts/format';
import { SegmentedBar, StatTile } from '@/components/charts/StatTile';
import { analysisNote, sampleNote } from './sampleData';
import { KIND_LABELS, KIND_ORDER, Note } from './shared';
import { WarningList } from './WarningList';

export const SPLIT_NAMES = ['Training', 'Validation', 'Test'];

export interface OverviewViewProps {
  profile: DataProfile;
  sample: DataSample;
  target: string | null;
  onAddStep: (step: NewStep) => void;
  onShowColumn: (column: string) => void;
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
  const cells = profile.n_rows * profile.n_columns;
  const split = profile.split_counts;
  const targetProfile = profile.target;
  const classes = targetProfile?.classes?.length ?? 0;
  const targetHint = targetProfile
    ? targetProfile.task === 'classification'
      ? `${formatCount(classes)} classes${targetProfile.imbalance_ratio ? ` · ${targetProfile.imbalance_ratio.toFixed(1)} : 1` : ''}`
      : 'Regression'
    : 'Choose one in Target & split';

  return (
    <Stack gap="md">
      <SimpleGrid type="container" cols={{ base: 2, '420px': 3, '1100px': 6 }} spacing="xs">
        <StatTile
          label="Rows"
          value={formatCount(profile.n_rows)}
          hint={formatBytes(profile.memory_bytes)}
        />
        <StatTile
          label="Columns"
          value={formatCount(profile.n_columns)}
          hint={
            targetProfile || target
              ? `${formatCount(Math.max(0, profile.n_columns - 1))} features + target`
              : 'No target yet'
          }
        />
        <StatTile
          label="Missing cells"
          value={formatCount(profile.missing_cells)}
          hint={`${formatPercent(cells ? profile.missing_cells / cells : 0)} of all cells`}
        />
        <StatTile
          label="Duplicate rows"
          value={formatCount(profile.duplicate_rows)}
          hint={
            profile.analysis_rows < profile.n_rows
              ? `in ${formatCount(profile.analysis_rows)} checked rows`
              : `${formatPercent(profile.n_rows ? profile.duplicate_rows / profile.n_rows : 0)} of rows`
          }
        />
        <StatTile label="Target" value={targetProfile?.column ?? target ?? '–'} hint={targetHint} />
        <StatTile
          label="Split"
          value={split ? split.map(formatCount).join(' / ') : '–'}
          hint={split ? 'train / validation / test' : 'Assigned after combining'}
        >
          {split && (
            <SegmentedBar
              label="Rows per split"
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
          Column types
        </Text>
        <SegmentedBar
          label="Columns per type"
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
          {profile.warnings.length ? `Warnings (${profile.warnings.length})` : 'Warnings'}
        </Text>
        <WarningList
          warnings={profile.warnings}
          resetKey={profile}
          onAddStep={onAddStep}
          onShowColumn={onShowColumn}
          emptyText="No problems found."
        />
      </Stack>

      <Note>
        Column statistics are exact (all {formatCount(profile.n_rows)} rows). Duplicates,
        correlations and target associations: {lower(analysisNote(profile))} Charts that split rows
        by class: {lower(sampleNote(sample))}
      </Note>
    </Stack>
  );
}
