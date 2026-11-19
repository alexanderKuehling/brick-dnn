import { useEffect, useId, useMemo, useRef, useState } from 'react';
import { IconDownload, IconFileUpload, IconWand } from '@tabler/icons-react';
import {
  Box,
  Button,
  FileButton,
  Group,
  MultiSelect,
  Progress,
  ScrollArea,
  SegmentedControl,
  Select,
  Stack,
  Table,
  Text,
  TextInput,
} from '@mantine/core';
import { api, type FileOutput, type FilePrediction, type PredictionForm } from '@/api';
import { downloadBlob } from '@/builder/download';
import { i18n, useTranslation } from '@/i18n';
import { formatInteger, formatNumber } from '@/i18n/format';
import { formatBytes, formatBytesOf } from '../format';
import { Messages } from './PredictionResults';
import { predictionProblem, type PredictionProblem } from './requests';
import { fileColumns, formatProbability, formatValue } from './rows';

const ACCEPT = '.csv,.tsv,.txt,text/csv,text/plain';
const DEFAULT_COLUMN = 'prediction';
/** Classes listed in the summary; the rest are counted together. */
const SHOWN_CLASSES = 6;

type Keep = 'all' | 'chosen';

/** The columns the model reads (they are no id to keep in a submission). */
function modelColumns(form: PredictionForm): Set<string> {
  return new Set([
    ...(form.prepared?.columns ?? []).map((c) => c.name),
    ...(form.raw?.columns ?? []).map((c) => c.name),
  ]);
}

function metricsText(result: FilePrediction): string | null {
  const metrics = result.metrics;
  if (!metrics) {
    return null;
  }
  const number = (value: number) => formatNumber(value, { maximumSignificantDigits: 4 });
  const parts: string[] = [];
  if (typeof metrics.accuracy === 'number') {
    const value = formatProbability(metrics.accuracy);
    parts.push(i18n.t('ui.training.FilePredict.accuracy', { value }));
  }
  if (typeof metrics.log_loss === 'number') {
    parts.push(i18n.t('ui.training.FilePredict.logLoss', { value: number(metrics.log_loss) }));
  }
  if (typeof metrics.roc_auc === 'number') {
    parts.push(i18n.t('ui.training.FilePredict.auc', { value: number(metrics.roc_auc) }));
  }
  if (typeof metrics.rmse === 'number') {
    parts.push(i18n.t('ui.training.FilePredict.rmse', { value: number(metrics.rmse) }));
  }
  if (typeof metrics.mae === 'number') {
    parts.push(i18n.t('ui.training.FilePredict.mae', { value: number(metrics.mae) }));
  }
  if (typeof metrics.r2 === 'number') {
    parts.push(i18n.t('ui.training.FilePredict.r2', { value: number(metrics.r2) }));
  }
  return i18n.t('ui.training.FilePredict.measuredOn', {
    target: result.target ?? '',
    rows: formatInteger(metrics.rows),
    metrics: parts.join(', '),
  });
}

function summaryText(result: FilePrediction): string | null {
  if (result.values) {
    const { min, max, mean } = result.values;
    const format = result.output === 'probability' ? formatProbability : formatValue;
    return i18n.t(
      result.output === 'probability'
        ? 'ui.training.FilePredict.probabilitiesFromTo'
        : 'ui.training.FilePredict.valuesFromTo',
      { min: format(min), max: format(max), mean: format(mean), class: result.class ?? '' }
    );
  }
  if (result.class_counts) {
    const counts = Object.entries(result.class_counts).sort((a, b) => b[1] - a[1]);
    const shown = counts.slice(0, SHOWN_CLASSES).map(([name, n]) => `${name} ${formatInteger(n)}`);
    const more = counts.length - shown.length;
    return i18n.t('ui.training.FilePredict.predictedClasses', {
      classes: shown.join(' · ') + (more > 0 ? ' · …' : ''),
    });
  }
  return null;
}

/** The first messages of the server, and how many more there are. */
function withMore(messages: string[], total: number): string[] {
  const more = total - messages.length;
  return more > 0
    ? [...messages, i18n.t('ui.training.FilePredict.andMore', { count: formatInteger(more) })]
    : messages;
}

/**
 * "Predict a file": the browser uploads a CSV file as it is, the server predicts every row
 * and keeps a result file to download. The options of the result (which columns, the name
 * of the prediction column, what a classifier writes) appear once a file is chosen.
 */
export function FilePredict({ runId, form }: { runId: string; form: PredictionForm }) {
  useTranslation();
  const limits = form.file;
  const classes = form.task === 'classification' ? (form.class_names ?? []) : [];
  const target = form.target ?? form.raw?.target ?? null;
  const [file, setFile] = useState<File | null>(null);
  const [columns, setColumns] = useState<string[]>([]);
  const [keep, setKeep] = useState<Keep>('all');
  const [chosen, setChosen] = useState<string[]>([]);
  const [name, setName] = useState(DEFAULT_COLUMN);
  const [output, setOutput] = useState<FileOutput>('label');
  const [ofClass, setOfClass] = useState<string | null>(classes[1] ?? classes[0] ?? null);
  const [dragging, setDragging] = useState(false);
  const [progress, setProgress] = useState<{ sent: number; total: number } | null>(null);
  const [busy, setBusy] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [result, setResult] = useState<FilePrediction | null>(null);
  const [problem, setProblem] = useState<PredictionProblem | null>(null);
  // The downloaded result (the server deletes its copy with the download), and the result
  // that still waits on the server.
  const downloaded = useRef<Blob | null>(null);
  const waiting = useRef<string | null>(null);
  const request = useRef<AbortController | null>(null);
  const used = useMemo(() => modelColumns(form), [form]);
  const columnsLabel = useId();

  const discard = () => {
    if (waiting.current) {
      void api.training.runs.discardPredictedFile(runId, waiting.current).catch(() => undefined);
      waiting.current = null;
    }
    downloaded.current = null;
  };

  // Leaving the panel: stop an upload, and do not leave a result on the server.
  useEffect(
    () => () => {
      request.current?.abort();
      if (waiting.current) {
        void api.training.runs.discardPredictedFile(runId, waiting.current).catch(() => undefined);
      }
    },
    [runId]
  );

  const choose = (next: File | null) => {
    if (!next) {
      return;
    }
    discard();
    setResult(null);
    setProblem(null);
    setKeep('all');
    setChosen([]);
    setColumns([]);
    setFile(next);
    if (limits && next.size > limits.max_mb * 1024 * 1024) {
      setProblem({
        message: i18n.t('ui.training.FilePredict.theFileIsTooLarge', {
          size: formatBytes(next.size),
          max: limits.max_mb,
        }),
        errors: [],
        budget: false,
      });
      return;
    }
    // Only the first kilobytes are read here, for the names of the columns.
    void fileColumns(next)
      .catch(() => [] as string[])
      .then((found) => {
        setColumns(found);
        setName(target && !found.includes(target) ? target : DEFAULT_COLUMN);
      });
  };

  const chooseKeep = (value: string) => {
    setKeep(value as Keep);
    if (value === 'chosen' && chosen.length === 0) {
      // Typically an id: the first column that is neither a feature nor the target.
      const id = columns.find((c) => !used.has(c) && c !== target);
      setChosen(id ? [id] : []);
    }
  };

  const predict = async () => {
    if (!file) {
      return;
    }
    discard();
    setResult(null);
    setProblem(null);
    setBusy(true);
    setProgress({ sent: 0, total: file.size });
    const controller = new AbortController();
    request.current = controller;
    try {
      const answer = await api.training.runs.predictFile(
        runId,
        file,
        {
          columns: keep === 'all' ? 'all' : chosen,
          prediction_column: name.trim() || DEFAULT_COLUMN,
          ...(classes.length > 0 ? { output } : {}),
          ...(output === 'probability' && ofClass ? { class: ofClass } : {}),
        },
        {
          signal: controller.signal,
          onUploadProgress: (sent, total) => setProgress({ sent, total }),
        }
      );
      waiting.current = answer.id;
      setResult(answer);
    } catch (error) {
      if ((error as Error).name !== 'AbortError') {
        setProblem(predictionProblem(error));
      }
    } finally {
      setBusy(false);
      setProgress(null);
    }
  };

  const download = async () => {
    if (!result) {
      return;
    }
    if (!downloaded.current) {
      setDownloading(true);
      try {
        downloaded.current = (await api.training.runs.downloadPredictedFile(runId, result.id)).blob;
        waiting.current = null;
      } catch (error) {
        setProblem(predictionProblem(error));
        return;
      } finally {
        setDownloading(false);
      }
    }
    downloadBlob(result.filename, downloaded.current);
  };

  const sent = progress !== null && progress.total > 0 && progress.sent >= progress.total;
  const tooLarge = Boolean(file && limits && file.size > limits.max_mb * 1024 * 1024);

  return (
    <Stack
      gap="xs"
      onDragOver={(event: React.DragEvent) => {
        event.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(event: React.DragEvent) => {
        event.preventDefault();
        setDragging(false);
        if (!busy) {
          choose(event.dataTransfer.files?.[0] ?? null);
        }
      }}
      style={{
        outline: dragging ? '1px dashed var(--mantine-color-blue-6)' : undefined,
        outlineOffset: 4,
        borderRadius: 'var(--mantine-radius-sm)',
      }}
    >
      <Group gap="xs" align="center">
        <FileButton accept={ACCEPT} onChange={choose} disabled={busy}>
          {(props) => (
            <Button {...props} variant="default" leftSection={<IconFileUpload size={16} />}>
              {file
                ? i18n.t('ui.training.FilePredict.chooseAnotherFile')
                : i18n.t('ui.training.FilePredict.predictAFile')}
            </Button>
          )}
        </FileButton>
        {file ? (
          <Text size="sm" style={{ overflowWrap: 'anywhere' }}>
            {file.name}{' '}
            <Text span size="xs" c="dimmed">
              {formatBytes(file.size)}
            </Text>
          </Text>
        ) : (
          <Text size="xs" c="dimmed">
            {i18n.t('ui.training.FilePredict.aCsvFileInThe')}
          </Text>
        )}
      </Group>

      {file && !tooLarge && (
        <Stack gap="xs">
          <Group gap="xs" align="flex-end" wrap="wrap">
            <Box>
              <Text size="sm" fw={500} mb={3} id={columnsLabel}>
                {i18n.t('ui.training.FilePredict.columnsOfTheResult')}
              </Text>
              <SegmentedControl
                size="xs"
                value={keep}
                onChange={chooseKeep}
                disabled={busy}
                aria-labelledby={columnsLabel}
                data={[
                  { value: 'all', label: i18n.t('ui.training.FilePredict.allColumns') },
                  { value: 'chosen', label: i18n.t('ui.training.FilePredict.onlyChosen') },
                ]}
              />
            </Box>
            {keep === 'chosen' && (
              <MultiSelect
                size="xs"
                label={i18n.t('ui.training.FilePredict.columnsToKeep')}
                placeholder={
                  chosen.length === 0 ? i18n.t('ui.training.FilePredict.onlyThePrediction') : ''
                }
                data={columns}
                value={chosen}
                onChange={setChosen}
                disabled={busy}
                searchable
                clearable
                style={{ flex: '1 1 220px' }}
              />
            )}
          </Group>
          <Group gap="xs" align="flex-end" wrap="wrap">
            {classes.length > 0 && (
              <Select
                size="xs"
                label={i18n.t('ui.training.FilePredict.whatIsWritten')}
                data={[
                  { value: 'label', label: i18n.t('ui.training.FilePredict.theClass') },
                  {
                    value: 'probability',
                    label: i18n.t('ui.training.FilePredict.theProbabilityOfOneClass'),
                  },
                  {
                    value: 'probabilities',
                    label: i18n.t('ui.training.FilePredict.theProbabilitiesOfAllClasses'),
                  },
                ]}
                value={output}
                onChange={(value) => setOutput((value as FileOutput | null) ?? 'label')}
                allowDeselect={false}
                disabled={busy}
                style={{ flex: '1 1 200px' }}
              />
            )}
            {output === 'probability' && classes.length > 0 && (
              <Select
                size="xs"
                label={i18n.t('ui.training.FilePredict.class')}
                data={classes}
                value={ofClass}
                onChange={setOfClass}
                allowDeselect={false}
                disabled={busy}
                searchable={classes.length > 10}
                style={{ flex: '1 1 140px' }}
              />
            )}
            {output !== 'probabilities' && (
              <TextInput
                size="xs"
                label={i18n.t('ui.training.FilePredict.nameOfThePredictionColumn')}
                value={name}
                onChange={(event) => setName(event.currentTarget.value)}
                maxLength={120}
                disabled={busy}
                style={{ flex: '1 1 180px' }}
              />
            )}
          </Group>
          <Group gap="xs" align="center">
            <Button
              leftSection={<IconWand size={16} />}
              onClick={() => void predict()}
              loading={busy}
            >
              {i18n.t('ui.training.FilePredict.predictTheFile')}
            </Button>
            {progress !== null && (
              <Text size="xs" c="dimmed" role="status">
                {sent
                  ? i18n.t('ui.training.FilePredict.predictingTheRows')
                  : i18n.t('ui.training.FilePredict.uploading', {
                      progress: formatBytesOf(progress.sent, progress.total),
                    })}
              </Text>
            )}
          </Group>
          {progress !== null && (
            <Progress
              value={sent ? 100 : (progress.sent / Math.max(1, progress.total)) * 100}
              size="sm"
              animated={sent}
              aria-label={i18n.t('ui.training.FilePredict.progress')}
            />
          )}
          {limits && (
            <Text size="xs" c="dimmed">
              {i18n.t('ui.training.FilePredict.theFileAndTheResult', {
                minutes: limits.keep_minutes,
              })}
            </Text>
          )}
        </Stack>
      )}

      {problem && (
        <Messages
          color={problem.budget ? 'yellow' : 'red'}
          title={
            problem.budget
              ? i18n.t('ui.training.FilePredict.noComputeLeft')
              : i18n.t('ui.training.FilePredict.cannotPredictTheFile')
          }
          messages={problem.errors.length > 0 ? problem.errors : [problem.message]}
        />
      )}

      {result && (
        <Stack gap="xs">
          <Group justify="space-between" gap="xs" align="center">
            <Text size="sm" fw={500}>
              {result.failed > 0
                ? i18n.t('ui.training.FilePredict.ofRowsPredicted', {
                    predicted: formatInteger(result.predicted),
                    rows: formatInteger(result.rows),
                  })
                : i18n.t('ui.training.FilePredict.rowsPredicted', {
                    rows: formatInteger(result.rows),
                  })}
            </Text>
            <Button
              leftSection={<IconDownload size={16} />}
              onClick={() => void download()}
              loading={downloading}
            >
              {i18n.t('ui.training.FilePredict.downloadTheResult')}
            </Button>
          </Group>
          {[summaryText(result), metricsText(result), result.metrics_note].map(
            (text, i) =>
              text && (
                <Text key={i} size="xs" c="dimmed">
                  {text}
                </Text>
              )
          )}
          <Messages
            color="yellow"
            title={
              result.failed === 1
                ? i18n.t('ui.training.FilePredict.rowWithoutAPrediction')
                : i18n.t('ui.training.FilePredict.rowsWithoutAPrediction', {
                    count: formatInteger(result.failed),
                  })
            }
            messages={withMore(result.problems, result.failed)}
          />
          <Messages
            color="yellow"
            title={i18n.t('ui.training.FilePredict.note')}
            messages={withMore(result.warnings, result.warning_count)}
          />
          <ScrollArea.Autosize mah={280} type="auto">
            <Table
              striped
              withTableBorder
              fz="xs"
              aria-label={i18n.t('ui.training.FilePredict.theFirstRowsOfThe')}
            >
              <Table.Thead>
                <Table.Tr>
                  {result.columns.map((column, i) => (
                    <Table.Th key={i}>{column}</Table.Th>
                  ))}
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {result.preview.map((row, i) => (
                  <Table.Tr key={i}>
                    {row.map((cell, j) => (
                      <Table.Td key={j}>{cell}</Table.Td>
                    ))}
                  </Table.Tr>
                ))}
              </Table.Tbody>
            </Table>
          </ScrollArea.Autosize>
          {result.rows > result.preview.length && (
            <Text size="xs" c="dimmed">
              {i18n.t('ui.training.FilePredict.theFirstRowsTheDownload', {
                shown: result.preview.length,
                rows: formatInteger(result.rows),
              })}
            </Text>
          )}
        </Stack>
      )}
    </Stack>
  );
}
