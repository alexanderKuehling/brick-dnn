import { useMemo, useState } from 'react';
import { IconArrowsShuffle, IconFileUpload, IconWand } from '@tabler/icons-react';
import {
  Autocomplete,
  Button,
  Collapse,
  FileButton,
  Group,
  Progress,
  SegmentedControl,
  Select,
  SimpleGrid,
  Stack,
  Text,
  Textarea,
  TextInput,
} from '@mantine/core';
import type { Prediction, PredictionForm, PredictionResult, PredictionSample } from '@/api';
import { downloadText } from '@/builder/download';
import { Messages, ResultCard, ResultsTable } from './PredictionResults';
import { isProblem, predictAll, type PredictionProblem } from './requests';
import {
  cellText,
  cellValue,
  csvRows,
  parseMessage,
  resultsCsv,
  type CsvRows,
  type InputColumn,
} from './rows';

type Mode = 'raw' | 'prepared';

interface TabularPredictProps {
  runId: string;
  form: PredictionForm;
  /** "Sales net" -> "sales-net-predictions.csv" */
  fileBase: string;
}

/** The input columns of a mode (raw: the columns of the source the model uses). */
function columnsOf(form: PredictionForm, mode: Mode): InputColumn[] {
  if (mode === 'raw' && form.raw) {
    return form.raw.columns.filter((c) => c.used !== false);
  }
  return (form.prepared?.columns ?? []).map((c) => ({ name: c.name, kind: 'numeric' as const }));
}

function samplesOf(form: PredictionForm, mode: Mode): PredictionSample[] {
  return (mode === 'raw' ? form.raw?.samples : form.prepared?.samples) ?? [];
}

function emptyValues(columns: InputColumn[]): Record<string, string> {
  return Object.fromEntries(columns.map((c) => [c.name, '']));
}

function rangeHint(column: InputColumn): string | undefined {
  if (column.min === undefined || column.min === null || column.max === undefined) {
    return undefined;
  }
  return `${column.min} … ${column.max}`;
}

/** One input of the form, by the kind of its column. */
function Field({
  column,
  value,
  error,
  onChange,
}: {
  column: InputColumn;
  value: string;
  error?: string;
  onChange: (value: string) => void;
}) {
  const common = {
    label: column.name,
    error,
    description: rangeHint(column),
    size: 'xs' as const,
  };
  if (column.kind === 'boolean') {
    return (
      <Select
        {...common}
        data={['true', 'false']}
        value={value || null}
        onChange={(next) => onChange(next ?? '')}
        placeholder="missing"
        clearable
      />
    );
  }
  if (column.kind === 'categorical' && column.categories && column.categories.length > 0) {
    return (
      <Autocomplete
        {...common}
        data={column.categories.slice(0, 200)}
        value={value}
        onChange={onChange}
        placeholder={column.categories[0]}
        limit={50}
      />
    );
  }
  if (column.kind === 'datetime') {
    const withTime = value.includes('T') || /T/.test(String(column.min ?? ''));
    return (
      <TextInput
        {...common}
        type={withTime ? 'datetime-local' : 'date'}
        value={value}
        onChange={(event) => onChange(event.currentTarget.value)}
      />
    );
  }
  return (
    <TextInput
      {...common}
      inputMode={column.kind === 'numeric' ? 'decimal' : undefined}
      value={value}
      onChange={(event) => onChange(event.currentTarget.value)}
      placeholder={column.kind === 'numeric' ? 'number' : undefined}
    />
  );
}

/** The prediction form of a run trained on a table: one row by hand, or many from CSV text. */
export function TabularPredict({ runId, form, fileBase }: TabularPredictProps) {
  const [mode, setMode] = useState<Mode>(form.raw ? 'raw' : 'prepared');
  const columns = useMemo(() => columnsOf(form, mode), [form, mode]);
  const samples = samplesOf(form, mode);
  const [values, setValues] = useState<Record<string, string>>(() => emptyValues(columns));
  const [sample, setSample] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [single, setSingle] = useState<{ prediction: Prediction; result: PredictionResult } | null>(
    null
  );
  const [problem, setProblem] = useState<PredictionProblem | null>(null);
  const [csvOpen, setCsvOpen] = useState(false);
  const [csvText, setCsvText] = useState('');
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null);
  const [many, setMany] = useState<{
    rows: CsvRows;
    result: PredictionResult;
  } | null>(null);
  const prepared = mode === 'prepared';
  const topK = Math.min(3, form.max_top_k);
  const unused = mode === 'raw' ? (form.raw?.columns ?? []).filter((c) => c.used === false) : [];

  const switchMode = (next: string) => {
    const nextMode = next as Mode;
    setMode(nextMode);
    setValues(emptyValues(columnsOf(form, nextMode)));
    setSample(null);
    setSingle(null);
    setProblem(null);
    setMany(null);
  };

  const takeSample = () => {
    if (samples.length === 0) {
      return;
    }
    const index = sample === null ? 0 : (sample + 1) % samples.length;
    const row = samples[index];
    setValues(Object.fromEntries(columns.map((c) => [c.name, cellText(row.values[c.name])])));
    setSample(index);
    setSingle(null);
    setProblem(null);
  };

  const fieldErrors = useMemo(() => {
    const map: Record<string, string> = {};
    for (const text of problem ? [problem.message, ...problem.errors] : []) {
      const message = parseMessage(text);
      if (message.column && !(message.column in map)) {
        map[message.column] = message.text;
      }
    }
    return map;
  }, [problem]);

  const send = async (rows: Record<string, string>[]) =>
    predictAll(
      runId,
      rows.map((row) =>
        Object.fromEntries(columns.map((c) => [c.name, cellValue(row[c.name] ?? '', c, prepared)]))
      ),
      {
        input_format: form.raw ? mode : undefined,
        top_k: form.task === 'classification' ? topK : undefined,
      },
      form.max_rows,
      rows.length > 1 ? (done) => setProgress({ done, total: rows.length }) : undefined
    );

  const predictOne = async () => {
    setBusy(true);
    setProblem(null);
    setSingle(null);
    try {
      const result = await send([values]);
      setSingle({ prediction: result.predictions[0], result });
    } catch (error) {
      setProblem(isProblem(error) ? error : { message: String(error), errors: [], budget: false });
    } finally {
      setBusy(false);
    }
  };

  const predictMany = async () => {
    setProblem(null);
    setMany(null);
    let rows: CsvRows;
    try {
      rows = csvRows(csvText, columns, { maxRows: form.max_form_rows, target: form.target });
    } catch (error) {
      setProblem({ message: (error as Error).message, errors: [], budget: false });
      return;
    }
    setBusy(true);
    setProgress({ done: 0, total: rows.rows.length });
    try {
      const result = await send(rows.rows);
      setMany({ rows, result });
    } catch (error) {
      setProblem(isProblem(error) ? error : { message: String(error), errors: [], budget: false });
    } finally {
      setBusy(false);
      setProgress(null);
    }
  };

  const download = () => {
    if (!many) {
      return;
    }
    const names = columns.map((c) => c.name);
    const text = resultsCsv(names, many.rows.rows, many.result.predictions, form.task, form.target);
    downloadText(`${fileBase}-predictions.csv`, text, 'text/csv');
  };

  const actual = sample !== null ? cellText(samples[sample]?.target) : null;
  const header = columns.map((c) => c.name).join(',');
  const warnings = (single ?? many)?.result.warnings ?? [];

  return (
    <Stack gap="sm">
      {form.raw && (
        <SegmentedControl
          size="xs"
          value={mode}
          onChange={switchMode}
          data={[
            { value: 'raw', label: 'Raw columns' },
            { value: 'prepared', label: 'Prepared columns' },
          ]}
          aria-label="Input columns"
        />
      )}
      <Text size="xs" c="dimmed">
        {mode === 'raw' && form.raw
          ? `The columns of “${form.raw.source ?? 'the source table'}” as imported. The server prepares them with ` +
            `the data preparation${form.raw.prep ? ` “${form.raw.prep.name}”` : ''} and the values it learned ` +
            'from the training rows, then predicts. Empty fields are missing values; the ranges are those of the training rows.'
          : `The feature columns of the training dataset${form.raw ? ', after the data preparation' : ''}, as numbers. ` +
            'The server standardises them like the training did.'}
      </Text>
      {mode === 'raw' && form.raw && form.raw.skipped_steps.length > 0 && (
        <Text size="xs" c="dimmed">
          Not applied to new rows (they only change training rows):{' '}
          {form.raw.skipped_steps.join(', ')}.
        </Text>
      )}
      {mode === 'raw' && form.raw && form.raw.joined.length > 0 && (
        <Text size="xs" c="dimmed">
          Joined with the stored copy of {form.raw.joined.map((name) => `“${name}”`).join(', ')} (as
          it was when the dataset was created).
        </Text>
      )}
      {unused.length > 0 && (
        <Text size="xs" c="dimmed">
          Not needed: {unused.map((c) => c.name).join(', ')} (the preparation drops{' '}
          {unused.length === 1 ? 'it' : 'them'}).
        </Text>
      )}

      <SimpleGrid cols={{ base: 1, xs: 2, sm: 3 }} spacing="xs" verticalSpacing={6}>
        {columns.map((column) => (
          <Field
            key={column.name}
            column={column}
            value={values[column.name] ?? ''}
            error={fieldErrors[column.name]}
            onChange={(value) => {
              setValues((current) => ({ ...current, [column.name]: value }));
              setSample(null);
            }}
          />
        ))}
      </SimpleGrid>

      <Group gap="xs">
        <Button
          leftSection={<IconWand size={16} />}
          loading={busy && progress === null}
          onClick={() => void predictOne()}
          disabled={busy}
        >
          Predict
        </Button>
        {samples.length > 0 && (
          <Button
            variant="default"
            leftSection={<IconArrowsShuffle size={16} />}
            onClick={takeSample}
            disabled={busy}
          >
            {sample === null ? 'Use a sample' : `Sample ${sample + 1} of ${samples.length}`}
          </Button>
        )}
        <Button variant="subtle" onClick={() => setCsvOpen((open) => !open)}>
          {csvOpen ? 'Hide CSV rows' : 'Paste CSV rows'}
        </Button>
      </Group>
      {samples.length > 0 && sample === null && (
        <Text size="xs" c="dimmed" mt={-6}>
          Samples are rows of the{' '}
          {form.prepared?.samples.length && !form.raw ? 'dataset' : 'validation data'}, with their
          actual value.
        </Text>
      )}

      <Collapse expanded={csvOpen}>
        <Stack gap="xs">
          <Textarea
            label="CSV rows"
            description={`A header line with the column names, then up to ${form.max_form_rows} rows (“,” or “;” separated).`}
            placeholder={`${header}\n…`}
            autosize
            minRows={3}
            maxRows={10}
            value={csvText}
            onChange={(event) => setCsvText(event.currentTarget.value)}
            styles={{ input: { fontFamily: 'var(--mantine-font-family-monospace)', fontSize: 12 } }}
          />
          <Group gap="xs">
            <Button
              size="xs"
              onClick={() => void predictMany()}
              disabled={busy || !csvText.trim()}
              loading={busy && progress !== null}
            >
              Predict rows
            </Button>
            <FileButton
              accept=".csv,.tsv,.txt,text/csv,text/plain"
              onChange={(file) => {
                if (file) {
                  void file.text().then(setCsvText);
                }
              }}
            >
              {(props) => (
                <Button
                  {...props}
                  size="xs"
                  variant="default"
                  leftSection={<IconFileUpload size={14} />}
                >
                  Upload a CSV file
                </Button>
              )}
            </FileButton>
          </Group>
          {progress !== null && many === null && (
            <Progress
              value={(progress.done / Math.max(1, progress.total)) * 100}
              size="sm"
              animated
              aria-label="Progress"
            />
          )}
        </Stack>
      </Collapse>

      {problem && (
        <Messages
          color={problem.budget ? 'yellow' : 'red'}
          title={problem.budget ? 'No compute left' : 'Cannot predict'}
          messages={problem.errors.length > 0 ? problem.errors : [problem.message]}
        />
      )}
      {single && (
        <ResultCard
          prediction={single.prediction}
          task={single.result.task}
          target={form.target}
          actual={actual}
        />
      )}
      {many && (
        <ResultsTable
          predictions={many.result.predictions}
          task={many.result.task}
          actual={many.rows.actual}
          onDownload={download}
        />
      )}
      {many && many.rows.ignored.length > 0 && (
        <Text size="xs" c="dimmed">
          Ignored columns: {many.rows.ignored.join(', ')}.
        </Text>
      )}
      <Messages color="yellow" title="Note" messages={warnings} />
    </Stack>
  );
}
