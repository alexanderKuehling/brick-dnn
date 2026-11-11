import { Alert, Anchor, Group, Loader, Paper, SimpleGrid, Stack, Table, Text } from '@mantine/core';
import type { PairColumn, RowAnalysis, SimilarRow, SimilarRows } from '@/api/types';
import { formatNumber } from '@/components/charts/format';
import { rowName } from './rowsData';
import classes from './RowsView.module.css';

const SPLIT_LABELS = { train: 'Train', val: 'Validation', test: 'Test' } as const;

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
  const target = analysis.target;
  const neighbors = data?.neighbors ?? [];
  const describe = (item: SimilarRow) => rowName(analysis, item.index);
  return (
    <Paper withBorder radius="md" p="sm" className={classes.card} aria-label="Similar rows">
      <Stack gap="xs">
        <Group justify="space-between" wrap="nowrap">
          <Text fw={600} size="sm">
            Rows most similar to {rowName(analysis, row)}
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
              Spread {targetText(data.row.spread)}
              {target ? ` · ${target} ${targetText(data.row.target)}` : ''}
              {data.row.split ? ` · ${SPLIT_LABELS[data.row.split]}` : ''}. A ratio below about 0.5
              means much closer than a typical pair of rows (1.0).
            </Text>
            <Table.ScrollContainer minWidth={320}>
              <Table className={classes.compare} striped highlightOnHover>
                <Table.Thead>
                  <Table.Tr>
                    <Table.Th>Row</Table.Th>
                    <Table.Th className={classes.numeric}>Ratio</Table.Th>
                    <Table.Th className={classes.numeric}>Distance</Table.Th>
                    {target && <Table.Th>{target}</Table.Th>}
                    <Table.Th>Split</Table.Th>
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
                            compared
                          </Text>
                        ) : (
                          <Anchor
                            component="button"
                            type="button"
                            size="xs"
                            onClick={() => onCompare(item.index)}
                          >
                            Compare
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
                  {describe(data.pair.a)} and {describe(data.pair.b)}: ratio{' '}
                  {formatNumber(data.pair.ratio, 3)}, {data.pair.shared} of {data.pair.features}{' '}
                  columns compared
                </Text>
                <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
                  <PairTable title="Most similar columns" columns={data.pair.closest} />
                  <PairTable title="Least similar columns" columns={data.pair.farthest} />
                </SimpleGrid>
                <Text size="xs" c="dimmed">
                  z: the value in robust standard deviations from the column median.
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
  return (
    <div>
      <Text size="xs" fw={600} mb={2}>
        {title}
      </Text>
      <Table className={classes.compare} withRowBorders={false}>
        <Table.Thead>
          <Table.Tr>
            <Table.Th>Column</Table.Th>
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
