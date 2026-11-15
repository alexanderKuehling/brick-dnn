import { Alert, Anchor, Group, Loader, Paper, SimpleGrid, Stack, Table, Text } from '@mantine/core';
import type { PairColumn, RowAnalysis, SimilarRow, SimilarRows } from '@/api/types';
import { formatNumber } from '@/components/charts/format';
import { i18n, useTranslation } from '@/i18n';
import { rowName } from './rowsData';
import classes from './RowsView.module.css';

const SPLIT_LABELS = {
  get train() {
    return i18n.t('ui.data.SimilarPanel.train');
  },
  get val() {
    return i18n.t('ui.data.SimilarPanel.validation');
  },
  get test() {
    return i18n.t('ui.data.SimilarPanel.test');
  },
} as const;

export interface SimilarPanelProps {
  analysis: RowAnalysis;
  /** The chosen row (index into analysis.rows). */
  row: number;
  data: SimilarRows | null;
  loading: boolean;
  error: string | null;
  /** Compare the chosen row with another one (index), or with its nearest row (null). */
  onCompare: (other: number | null) => void;
  /** Make another row the chosen one. */
  onOpen: (index: number) => void;
}

function targetText(value: number | string | null | undefined): string {
  if (value === null || value === undefined) {
    return '–';
  }
  return typeof value === 'number' ? formatNumber(value, 4) : value;
}

/**
 * The rows most similar to the chosen row, with distances, target values and split parts side
 * by side, and for a pair of rows the columns where they agree and differ most.
 */
export function SimilarPanel({
  analysis,
  row,
  data,
  loading,
  error,
  onCompare,
  onOpen,
}: SimilarPanelProps) {
  useTranslation();
  const target = analysis.target;
  const neighbors = data?.neighbors ?? [];
  const describe = (item: SimilarRow) => rowName(analysis, item.index);
  return (
    <Paper
      withBorder
      radius="md"
      p="sm"
      className={classes.card}
      aria-label={i18n.t('ui.data.SimilarPanel.similarRows')}
    >
      <Stack gap="xs">
        <Group justify="space-between" wrap="nowrap">
          <Text fw={600} size="sm">
            {i18n.t('ui.data.SimilarPanel.rowsMostSimilarTo', { rowName: rowName(analysis, row) })}
          </Text>
          {loading && <Loader size={14} />}
        </Group>
        {error && (
          <Alert color="red" p="xs">
            <Text size="sm">{error}</Text>
          </Alert>
        )}
        {data && (
          <>
            <Text size="xs" c="dimmed">
              {i18n.t('ui.data.SimilarPanel.spreadARatioBelowAbout', {
                targetText: targetText(data.row.spread),
                value: target ? ` · ${target} ${targetText(data.row.target)}` : '',
                value2: data.row.split ? ` · ${SPLIT_LABELS[data.row.split]}` : '',
              })}
            </Text>
            <Table.ScrollContainer minWidth={320}>
              <Table className={classes.compare} striped highlightOnHover>
                <Table.Thead>
                  <Table.Tr>
                    <Table.Th>{i18n.t('ui.data.SimilarPanel.row')}</Table.Th>
                    <Table.Th className={classes.numeric}>
                      {i18n.t('ui.data.SimilarPanel.ratio')}
                    </Table.Th>
                    <Table.Th className={classes.numeric}>
                      {i18n.t('ui.data.SimilarPanel.distance')}
                    </Table.Th>
                    {target && <Table.Th>{target}</Table.Th>}
                    <Table.Th>{i18n.t('ui.data.SimilarPanel.split')}</Table.Th>
                    <Table.Th />
                  </Table.Tr>
                </Table.Thead>
                <Table.Tbody>
                  {neighbors.map((item) => (
                    <Table.Tr
                      key={item.index}
                      data-compared={data.pair?.b.index === item.index || undefined}
                    >
                      <Table.Td>
                        <Anchor
                          component="button"
                          type="button"
                          size="xs"
                          onClick={() => onOpen(item.index)}
                        >
                          {describe(item)}
                        </Anchor>
                      </Table.Td>
                      <Table.Td className={classes.numeric}>{formatNumber(item.ratio, 3)}</Table.Td>
                      <Table.Td className={classes.numeric}>
                        {formatNumber(item.distance, 3)}
                      </Table.Td>
                      {target && <Table.Td>{targetText(item.target)}</Table.Td>}
                      <Table.Td>{item.split ? SPLIT_LABELS[item.split] : '–'}</Table.Td>
                      <Table.Td>
                        {data.pair?.b.index === item.index ? (
                          <Text size="xs" c="dimmed">
                            {i18n.t('ui.data.SimilarPanel.compared')}
                          </Text>
                        ) : (
                          <Anchor
                            component="button"
                            type="button"
                            size="xs"
                            onClick={() => onCompare(item.index)}
                          >
                            {i18n.t('ui.data.SimilarPanel.compare')}
                          </Anchor>
                        )}
                      </Table.Td>
                    </Table.Tr>
                  ))}
                </Table.Tbody>
              </Table>
            </Table.ScrollContainer>
            {data.pair && (
              <Stack gap={4}>
                <Text size="sm" fw={600}>
                  {i18n.t('ui.data.SimilarPanel.andRatioOfColumnsCompared', {
                    describe: describe(data.pair.a),
                    describe2: describe(data.pair.b),
                    ratio: formatNumber(data.pair.ratio, 3),
                    shared: data.pair.shared,
                    features: data.pair.features,
                  })}
                </Text>
                <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
                  <PairTable
                    title={i18n.t('ui.data.SimilarPanel.mostSimilarColumns')}
                    columns={data.pair.closest}
                  />
                  <PairTable
                    title={i18n.t('ui.data.SimilarPanel.leastSimilarColumns')}
                    columns={data.pair.farthest}
                  />
                </SimpleGrid>
                <Text size="xs" c="dimmed">
                  {i18n.t('ui.data.SimilarPanel.zTheValueInRobust')}
                </Text>
              </Stack>
            )}
          </>
        )}
      </Stack>
    </Paper>
  );
}

function PairTable({ title, columns }: { title: string; columns: PairColumn[] }) {
  useTranslation();
  return (
    <div>
      <Text size="xs" fw={600} mb={2}>
        {title}
      </Text>
      <Table className={classes.compare} withRowBorders={false}>
        <Table.Thead>
          <Table.Tr>
            <Table.Th>{i18n.t('ui.data.SimilarPanel.column')}</Table.Th>
            <Table.Th className={classes.numeric}>A (z)</Table.Th>
            <Table.Th className={classes.numeric}>B (z)</Table.Th>
          </Table.Tr>
        </Table.Thead>
        <Table.Tbody>
          {columns.map((column) => (
            <Table.Tr key={column.name}>
              <Table.Td>{column.name}</Table.Td>
              <Table.Td className={classes.numeric}>
                {formatNumber(column.a, 4)} ({formatNumber(column.za, 2)})
              </Table.Td>
              <Table.Td className={classes.numeric}>
                {formatNumber(column.b, 4)} ({formatNumber(column.zb, 2)})
              </Table.Td>
            </Table.Tr>
          ))}
        </Table.Tbody>
      </Table>
    </div>
  );
}
