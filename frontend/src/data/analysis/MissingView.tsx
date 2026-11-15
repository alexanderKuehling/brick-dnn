import { useMemo, useState } from 'react';
import { IconCircleCheck } from '@tabler/icons-react';
import { Button, Group, SimpleGrid, Stack, Switch, Text } from '@mantine/core';
import type { DataProfile, DataSample, NewStep } from '@/api/types';
import { BarChart } from '@/components/charts/BarChart';
import { formatCount, formatPercent } from '@/components/charts/format';
import { MissingMatrix } from '@/components/charts/MissingMatrix';
import { StatTile } from '@/components/charts/StatTile';
import { formatPercentTick } from '@/components/chartScale';
import { i18n, useTranslation } from '@/i18n';
import { usePrepAccess } from '../prepAccess';
import { sampleNote } from './sampleData';
import { WarningList } from './WarningList';

export interface MissingViewProps {
  profile: DataProfile;
  sample: DataSample;
  target: string | null;
  onAddStep: (step: NewStep) => void;
  onShowColumn: (column: string) => void;
}

const BARS = 30;

/** Missing values per column (exact) and where they are in the sample rows. */
export function MissingView({
  profile,
  sample,
  target,
  onAddStep,
  onShowColumn,
}: MissingViewProps) {
  useTranslation();
  const { readOnly } = usePrepAccess();
  const [allBars, setAllBars] = useState(false);
  const [allColumns, setAllColumns] = useState(false);
  const withMissing = useMemo(
    () => profile.columns.filter((c) => c.missing > 0).sort((a, b) => b.missing - a.missing),
    [profile.columns]
  );
  const rowStats = useMemo(() => {
    let incomplete = 0;
    for (let row = 0; row < sample.rows; row += 1) {
      if (sample.columns.some((c) => c.values[row] === null)) {
        incomplete += 1;
      }
    }
    return { incomplete };
  }, [sample]);

  const features = withMissing.filter((c) => c.name !== target);
  const numeric = features.filter((c) => c.kind === 'numeric').map((c) => c.name);
  const categorical = features
    .filter((c) => c.kind === 'categorical' || c.kind === 'boolean')
    .map((c) => c.name);
  const missingNames = new Set(withMissing.map((c) => c.name));
  const matrixColumns = sample.columns
    .filter((c) => allColumns || missingNames.has(c.name))
    .map((c) => ({ name: c.name, values: c.values }));
  const cells = profile.n_rows * profile.n_columns;
  const plural = (n: number, word: string) => `${formatCount(n)} ${word}${n === 1 ? '' : 's'}`;

  if (withMissing.length === 0) {
    return (
      <Group gap={6}>
        <IconCircleCheck size={18} color="var(--chart-status-good)" aria-hidden="true" />
        <Text size="sm">
          {i18n.t('ui.data.MissingView.noMissingValuesInRows', {
            n_rows: formatCount(profile.n_rows),
          })}
        </Text>
      </Group>
    );
  }

  return (
    <Stack gap="md">
      <SimpleGrid type="container" cols={{ base: 2, '520px': 3 }} spacing="xs">
        <StatTile
          label={i18n.t('ui.data.MissingView.missingCells')}
          value={formatCount(profile.missing_cells)}
          hint={i18n.t('ui.data.MissingView.ofAllCells', {
            value: formatPercent(cells ? profile.missing_cells / cells : 0),
          })}
        />
        <StatTile
          label={i18n.t('ui.data.MissingView.columnsWithGaps')}
          value={formatCount(withMissing.length)}
          hint={`of ${plural(profile.n_columns, 'column')}`}
        />
        <StatTile
          label={i18n.t('ui.data.MissingView.incompleteRows')}
          value={formatPercent(sample.rows ? rowStats.incomplete / sample.rows : 0)}
          hint={i18n.t('ui.data.MissingView.ofSampleRows', {
            incomplete: formatCount(rowStats.incomplete),
            rows: formatCount(sample.rows),
          })}
        />
      </SimpleGrid>

      {!readOnly && (numeric.length > 0 || categorical.length > 0) && (
        <Group gap="xs">
          {numeric.length > 0 && (
            <Button
              size="compact-sm"
              variant="light"
              onClick={() =>
                onAddStep({ type: 'impute', params: { columns: numeric, strategy: 'median' } })
              }
            >
              {i18n.t('ui.data.MissingView.fillMedian', {
                plural: plural(numeric.length, 'number column'),
              })}
            </Button>
          )}
          {categorical.length > 0 && (
            <Button
              size="compact-sm"
              variant="light"
              onClick={() =>
                onAddStep({
                  type: 'impute',
                  params: { columns: categorical, strategy: 'most_frequent' },
                })
              }
            >
              {i18n.t('ui.data.MissingView.fillMostFrequent', {
                plural: plural(categorical.length, 'category column'),
              })}
            </Button>
          )}
        </Group>
      )}

      <BarChart
        title={i18n.t('ui.data.MissingView.missingValuesPerColumn')}
        subtitle={i18n.t('ui.data.MissingView.shareOfAllRows', {
          n_rows: formatCount(profile.n_rows),
        })}
        rows={(allBars ? withMissing : withMissing.slice(0, BARS)).map((c) => ({
          key: c.name,
          label: c.name,
          values: [profile.n_rows ? c.missing / profile.n_rows : 0],
        }))}
        formatValue={formatPercent}
        formatTick={formatPercentTick}
        tipLabel={(row) => {
          const column = withMissing.find((c) => c.name === row.key);
          return `${formatPercent(row.values[0])} · ${formatCount(column?.missing ?? 0)}`;
        }}
        onSelect={onShowColumn}
        selectHint="show the column"
        unit="share missing"
      />
      {withMissing.length > BARS && (
        <Button
          variant="subtle"
          size="compact-sm"
          onClick={() => setAllBars((v) => !v)}
          style={{ alignSelf: 'flex-start' }}
        >
          {allBars
            ? i18n.t('ui.data.MissingView.showTheLargest', { BARS })
            : i18n.t('ui.data.MissingView.showAllColumns', {
                length: formatCount(withMissing.length),
              })}
        </Button>
      )}

      <MissingMatrix
        title={i18n.t('ui.data.MissingView.whereValuesAreMissing')}
        subtitle={i18n.t('ui.data.MissingView.sampleRowsFromTopTo', {
          sampleNote: sampleNote(sample),
        })}
        columns={matrixColumns}
        controls={
          <Switch
            size="xs"
            label={i18n.t('ui.data.MissingView.allColumns')}
            checked={allColumns}
            onChange={(event) => setAllColumns(event.currentTarget.checked)}
          />
        }
        caption={i18n.t('ui.data.MissingView.gapsInTheSameRows')}
      />

      <WarningList
        warnings={profile.warnings.filter((w) => w.code === 'high_missing')}
        profile={profile}
        resetKey={profile}
        onAddStep={onAddStep}
        onShowColumn={onShowColumn}
      />
    </Stack>
  );
}
