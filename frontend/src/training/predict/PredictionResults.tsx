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
import type { MemberPredictions, Prediction, PredictionResult } from '@/api';
import { i18n, useTranslation } from '@/i18n';
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
  useTranslation();
  const classification = task === 'classification' || prediction.top !== undefined;
  return (
    <Paper
      withBorder
      p="sm"
      radius="sm"
      aria-label={i18n.t('ui.training.PredictionResults.prediction')}
    >
      {classification ? (
        <Stack gap={6}>
          <Group justify="space-between" align="baseline" wrap="wrap" gap="xs">
            <Text size="sm" c="dimmed">
              {i18n.t('ui.training.PredictionResults.predicted', {
                value: target ? target : i18n.t('ui.training.PredictionResults.class'),
              })}
            </Text>
            {actual !== undefined && actual !== null && actual !== '' && (
              <Text size="xs" c="dimmed">
                {i18n.t('ui.training.PredictionResults.actual', { actual })}
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
              {i18n.t('ui.training.PredictionResults.decidedByTheThresholdOn', {
                positive_probability: formatProbability(prediction.positive_probability),
              })}
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
              {i18n.t('ui.training.PredictionResults.predicted', {
                value: target || i18n.t('ui.training.PredictionResults.value'),
              })}
            </Text>
            {actual !== undefined && actual !== null && actual !== '' && (
              <Text size="xs" c="dimmed">
                {i18n.t('ui.training.PredictionResults.actual', { actual })}
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
              {i18n.t('ui.training.PredictionResults.theGaussianProcessSStandard')}
            </Text>
          )}
        </Stack>
      )}
    </Paper>
  );
}

/** A prediction in a cell: the class with its probability, or the value. */
function shortPrediction(prediction: Prediction | undefined, classification: boolean): string {
  if (!prediction) {
    return '–';
  }
  if (classification) {
    return `${prediction.class ?? '–'} (${formatProbability(prediction.probability)})`;
  }
  return formatValue(prediction.value);
}

/** What each member of an ensemble predicted for one sample. */
export function MemberResults({
  members,
  index,
  task,
}: {
  members: MemberPredictions[];
  index: number;
  task: PredictionResult['task'];
}) {
  useTranslation();
  const classification = task === 'classification';
  return (
    <Table
      fz="xs"
      withTableBorder
      aria-label={i18n.t('ui.training.PredictionResults.predictionsOfTheMembers')}
    >
      <Table.Thead>
        <Table.Tr>
          <Table.Th>{i18n.t('ui.training.PredictionResults.member')}</Table.Th>
          <Table.Th>{i18n.t('ui.training.PredictionResults.prediction')}</Table.Th>
          {classification && (
            <Table.Th>{i18n.t('ui.training.PredictionResults.pPositive')}</Table.Th>
          )}
        </Table.Tr>
      </Table.Thead>
      <Table.Tbody>
        {members.map((member) => {
          const prediction = member.predictions[index];
          return (
            <Table.Tr key={member.id}>
              <Table.Td>{member.name}</Table.Td>
              <Table.Td>{shortPrediction(prediction, classification)}</Table.Td>
              {classification && (
                <Table.Td>
                  {typeof prediction?.positive_probability === 'number'
                    ? formatProbability(prediction.positive_probability)
                    : ''}
                </Table.Td>
              )}
            </Table.Tr>
          );
        })}
      </Table.Tbody>
    </Table>
  );
}

const SHOWN_ROWS = 100;

/** Predictions of many rows, with a download of all of them as a CSV file. */
export function ResultsTable({
  predictions,
  task,
  actual,
  onDownload,
  members,
}: {
  predictions: Prediction[];
  task: PredictionResult['task'];
  actual?: string[] | null;
  onDownload: () => void;
  /** An ensemble's members: a column each. */
  members?: MemberPredictions[];
}) {
  useTranslation();
  const classification = task === 'classification';
  const withStd = predictions.some((p) => typeof p.std === 'number');
  return (
    <Stack gap="xs">
      <Group justify="space-between" gap="xs">
        <Text size="sm" fw={500}>
          {predictions.length}{' '}
          {predictions.length === 1
            ? i18n.t('ui.training.PredictionResults.prediction2')
            : i18n.t('ui.training.PredictionResults.predictions2')}
        </Text>
        <Button
          size="compact-sm"
          variant="default"
          leftSection={<IconDownload size={14} />}
          onClick={onDownload}
        >
          {i18n.t('ui.training.PredictionResults.resultsCsv')}
        </Button>
      </Group>
      <ScrollArea.Autosize mah={320} type="auto">
        <Table
          striped
          withTableBorder
          fz="xs"
          aria-label={i18n.t('ui.training.PredictionResults.predictions')}
        >
          <Table.Thead>
            <Table.Tr>
              <Table.Th>{i18n.t('ui.training.PredictionResults.row')}</Table.Th>
              <Table.Th>{i18n.t('ui.training.PredictionResults.prediction')}</Table.Th>
              <Table.Th>
                {classification
                  ? i18n.t('ui.training.PredictionResults.probability')
                  : withStd
                    ? i18n.t('ui.training.PredictionResults.std')
                    : ''}
              </Table.Th>
              {actual && <Table.Th>{i18n.t('ui.training.PredictionResults.actual2')}</Table.Th>}
              {members?.map((member) => (
                <Table.Th key={member.id}>{member.name}</Table.Th>
              ))}
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
                {members?.map((member) => (
                  <Table.Td key={member.id}>
                    {shortPrediction(member.predictions[i], classification)}
                  </Table.Td>
                ))}
              </Table.Tr>
            ))}
          </Table.Tbody>
        </Table>
      </ScrollArea.Autosize>
      {predictions.length > SHOWN_ROWS && (
        <Text size="xs" c="dimmed">
          {i18n.t('ui.training.PredictionResults.theFirstRowsTheDownload', {
            SHOWN_ROWS,
            length: predictions.length,
          })}
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
