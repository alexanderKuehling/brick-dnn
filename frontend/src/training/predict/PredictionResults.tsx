import { IconDownload } from '@tabler/icons-react';
import {
  Alert,
  Button,
  Group,
  List,
  Paper,
  Progress,
  ScrollArea,
  Stack,
  Table,
  Text,
} from '@mantine/core';
import type { Prediction, PredictionResult } from '@/api';
import { formatProbability, formatValue } from './rows';

/** One prediction: the class with the top classes as bars, or the value (± std). */
export function ResultCard({
  prediction,
  task,
  target,
  actual,
}: {
  prediction: Prediction;
  task: PredictionResult['task'];
  target?: string | null;
  /** The real target value of a sample row. */
  actual?: string | null;
}) {
  const classification = task === 'classification' || prediction.top !== undefined;
  return (
    <Paper withBorder p="sm" radius="sm" aria-label="Prediction">
      {classification ? (
        <Stack gap={6}>
          <Group justify="space-between" align="baseline" wrap="wrap" gap="xs">
            <Text size="sm" c="dimmed">
              Predicted {target ? target : 'class'}
            </Text>
            {actual !== undefined && actual !== null && actual !== '' && (
              <Text size="xs" c="dimmed">
                actual: {actual}
              </Text>
            )}
          </Group>
          <Text fw={700} size="xl" lh={1.2} style={{ overflowWrap: 'anywhere' }}>
            {prediction.class ?? '–'}{' '}
            <Text span size="sm" c="dimmed" fw={400}>
              {formatProbability(prediction.probability)}
            </Text>
          </Text>
          {typeof prediction.positive_probability === 'number' && (
            <Text size="xs" c="dimmed">
              Decided by the threshold on the probability of the positive class:{' '}
              {formatProbability(prediction.positive_probability)}
            </Text>
          )}
          {(prediction.top ?? []).map((item) => (
            <div key={item.class}>
              <Group justify="space-between" gap="xs" wrap="nowrap">
                <Text size="xs" style={{ overflowWrap: 'anywhere' }}>
                  {item.class}
                </Text>
                <Text size="xs" c="dimmed">
                  {formatProbability(item.probability)}
                </Text>
              </Group>
              <Progress value={item.probability * 100} size="sm" aria-label={item.class} />
            </div>
          ))}
        </Stack>
      ) : (
        <Stack gap={4}>
          <Group justify="space-between" align="baseline" wrap="wrap" gap="xs">
            <Text size="sm" c="dimmed">
              Predicted {target || 'value'}
            </Text>
            {actual !== undefined && actual !== null && actual !== '' && (
              <Text size="xs" c="dimmed">
                actual: {actual}
              </Text>
            )}
          </Group>
          <Text fw={700} size="xl" lh={1.2}>
            {formatValue(prediction.value)}
            {typeof prediction.std === 'number' && (
              <Text span size="sm" c="dimmed" fw={400}>
                {' '}
                ± {formatValue(prediction.std)}
              </Text>
            )}
          </Text>
          {typeof prediction.std === 'number' && (
            <Text size="xs" c="dimmed">
              ± the Gaussian process&apos;s standard deviation: how uncertain the prediction is.
            </Text>
          )}
        </Stack>
      )}
    </Paper>
  );
}

const SHOWN_ROWS = 100;

/** Predictions of many rows, with a download of all of them as a CSV file. */
export function ResultsTable({
  predictions,
  task,
  actual,
  onDownload,
}: {
  predictions: Prediction[];
  task: PredictionResult['task'];
  actual?: string[] | null;
  onDownload: () => void;
}) {
  const classification = task === 'classification';
  const withStd = predictions.some((p) => typeof p.std === 'number');
  return (
    <Stack gap="xs">
      <Group justify="space-between" gap="xs">
        <Text size="sm" fw={500}>
          {predictions.length} {predictions.length === 1 ? 'prediction' : 'predictions'}
        </Text>
        <Button
          size="compact-sm"
          variant="default"
          leftSection={<IconDownload size={14} />}
          onClick={onDownload}
        >
          Results (.csv)
        </Button>
      </Group>
      <ScrollArea.Autosize mah={320} type="auto">
        <Table striped withTableBorder fz="xs" aria-label="Predictions">
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Row</Table.Th>
              <Table.Th>Prediction</Table.Th>
              <Table.Th>{classification ? 'Probability' : withStd ? '± std' : ''}</Table.Th>
              {actual && <Table.Th>Actual</Table.Th>}
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {predictions.slice(0, SHOWN_ROWS).map((p, i) => (
              <Table.Tr key={i}>
                <Table.Td>{i + 1}</Table.Td>
                <Table.Td>{classification ? (p.class ?? '–') : formatValue(p.value)}</Table.Td>
                <Table.Td>
                  {classification
                    ? formatProbability(p.probability)
                    : typeof p.std === 'number'
                      ? formatValue(p.std)
                      : ''}
                </Table.Td>
                {actual && <Table.Td>{actual[i] ?? ''}</Table.Td>}
              </Table.Tr>
            ))}
          </Table.Tbody>
        </Table>
      </ScrollArea.Autosize>
      {predictions.length > SHOWN_ROWS && (
        <Text size="xs" c="dimmed">
          The first {SHOWN_ROWS} rows; the download has all {predictions.length}.
        </Text>
      )}
    </Stack>
  );
}

/** Problems of the inputs (from the server), or things to know about them. */
export function Messages({
  title,
  messages,
  color,
}: {
  title: string;
  messages: string[];
  color: 'red' | 'yellow';
}) {
  if (messages.length === 0) {
    return null;
  }
  return (
    <Alert color={color} variant="light" title={title} p="xs">
      {messages.length === 1 ? (
        <Text size="sm">{messages[0]}</Text>
      ) : (
        <List size="sm" spacing={2}>
          {messages.map((message, i) => (
            <List.Item key={i}>{message}</List.Item>
          ))}
        </List>
      )}
    </Alert>
  );
}
