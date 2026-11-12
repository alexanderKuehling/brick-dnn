/**
 * The final evaluation of a finished run (state.json `evaluation`): the kept weights measured
 * once more on the validation and the test rows (a run without either: on its training rows,
 * marked as optimistic). Tiles with the main numbers (the splits side by side); for
 * classification a confusion matrix and the values per class, for regression predicted
 * against actual values. Runs without an evaluation show nothing.
 */
import { useMemo, useState } from 'react';
import { IconArrowDown, IconArrowUp, IconSelector } from '@tabler/icons-react';
import {
  Group,
  Paper,
  ScrollArea,
  SegmentedControl,
  SimpleGrid,
  Stack,
  Table,
  Text,
  UnstyledButton,
} from '@mantine/core';
import type {
  ClassificationEvaluation,
  ClassificationSplit,
  PerClassMetrics,
  RegressionEvaluation,
  RegressionSplit,
  RunEvaluation,
} from '@/api';
import { ChartFrame, GradientLegend } from '@/components/charts/ChartFrame';
import { SEQUENTIAL, sequentialColor, textOn, useChartScheme } from '@/components/charts/colors';
import { ScatterPlot } from '@/components/charts/ScatterPlot';
import { formatMetric, formatPercent } from './format';
import classes from './Training.module.css';

type SplitKey = 'val' | 'test' | 'training';

const SPLIT_LABELS: Record<SplitKey, string> = {
  val: 'Validation',
  test: 'Test',
  training: 'Training (optimistic)',
};
/** "120 validation rows". */
const ROW_LABELS: Record<SplitKey, string> = {
  val: 'validation',
  test: 'test',
  training: 'training',
};

/** F1 scores: 0.91234 → "0.912". */
export function formatScore(value: number | null | undefined): string {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(3) : '–';
}

// ---------------------------------------------------------------- tiles

interface TileValue {
  label: string;
  value: string;
}

/** A metric with its value per split side by side (train, validation, test). */
function SplitTile({ label, values, hint }: { label: string; values: TileValue[]; hint?: string }) {
  return (
    <Paper
      withBorder
      p="xs"
      radius="sm"
      role="group"
      aria-label={label}
      style={values.length > 2 ? { gridColumn: 'span 2' } : undefined}
    >
      <Text size="xs" c="dimmed">
        {label}
      </Text>
      <Group gap="md" wrap="nowrap" align="flex-start">
        {values.map((item) => (
          <div key={item.label}>
            <Text fw={600} size="md" lh={1.35} style={{ fontVariantNumeric: 'tabular-nums' }}>
              {item.value}
            </Text>
            <Text size="xs" c="dimmed" lh={1.2}>
              {item.label}
            </Text>
          </div>
        ))}
      </Group>
      {hint && (
        <Text size="xs" c="dimmed" mt={2}>
          {hint}
        </Text>
      )}
    </Paper>
  );
}

type SplitParts<T> = { key: SplitKey; split: T }[];

/**
 * The splits that were measured, in order: validation and test, or (a run without either) the
 * training rows.
 */
function splitsOf<T>(evaluation: {
  val: T | null;
  test: T | null;
  training?: T | null;
}): SplitParts<T> {
  const parts: SplitParts<T> = [];
  if (evaluation.val) {
    parts.push({ key: 'val', split: evaluation.val });
  }
  if (evaluation.test) {
    parts.push({ key: 'test', split: evaluation.test });
  }
  if (parts.length === 0 && evaluation.training) {
    parts.push({ key: 'training', split: evaluation.training });
  }
  return parts;
}

/** The split picked in the picker, or the first one. */
function pickSplit<T>(parts: SplitParts<T>, key: SplitKey): { key: SplitKey; split: T } {
  return parts.find((part) => part.key === key) ?? parts[0];
}

function tile<T>(
  parts: SplitParts<T>,
  label: string,
  pick: (split: T) => number | null | undefined,
  format: (value: number | null | undefined) => string,
  hint?: string
) {
  const values = parts.map(({ key, split }) => ({
    label: SPLIT_LABELS[key],
    value: format(pick(split)),
  }));
  if (values.every((v) => v.value === '–')) {
    return null;
  }
  return <SplitTile key={label} label={label} values={values} hint={hint} />;
}

function lossTile<T extends { loss: number | null }>(
  evaluation: RunEvaluation,
  parts: SplitParts<T>,
  label: string,
  hint: string
) {
  const values = [
    { label: 'Train', value: formatMetric(evaluation.train?.loss) },
    ...parts.map(({ key, split }) => ({
      label: SPLIT_LABELS[key],
      value: formatMetric(split.loss),
    })),
  ];
  return <SplitTile key="loss" label={label} values={values} hint={hint} />;
}

// ---------------------------------------------------------------- confusion matrix

/** The share of a cell in its row (the rows are the true classes). */
function rowShares(matrix: number[][]): number[][] {
  return matrix.map((row) => {
    const total = row.reduce((sum, count) => sum + count, 0);
    return row.map((count) => (total > 0 ? count / total : 0));
  });
}

/**
 * Counts of true (rows) against predicted classes (columns). The colour is the share of the
 * true class (row-normalised), so classes of different sizes read alike; the cells show the
 * counts or that share. The matrix scrolls in its own box when it is wider than the page.
 */
export function ConfusionMatrix({
  names,
  matrix,
  title = 'Confusion matrix',
  subtitle,
}: {
  names: string[];
  matrix: number[][];
  title?: string;
  subtitle?: string;
}) {
  const scheme = useChartScheme();
  const [mode, setMode] = useState<'counts' | 'percent'>('counts');
  const shares = useMemo(() => rowShares(matrix), [matrix]);
  const label = (index: number) => names[index] ?? String(index);
  return (
    <ChartFrame
      title={title}
      subtitle={subtitle}
      controls={
        <SegmentedControl
          size="xs"
          value={mode}
          onChange={(value) => setMode(value as 'counts' | 'percent')}
          data={[
            { value: 'counts', label: 'Counts' },
            { value: 'percent', label: '% of true class' },
          ]}
          aria-label="Values in the cells"
        />
      }
      legendExtra={
        <GradientLegend
          label="Colour: share of the true class, from 0% (light) to 100% (dark)"
          stops={[SEQUENTIAL[scheme].fillLow, SEQUENTIAL[scheme].high]}
          labels={['0%', '100% of the true class']}
        />
      }
      caption="Rows: true class · columns: predicted class · the diagonal is right. Hover a cell for both numbers."
    >
      <ScrollArea.Autosize mah={520} type="auto" offsetScrollbars>
        <table
          className={classes.matrix}
          aria-label={`${title}: rows true class, columns predicted class`}
        >
          <thead>
            <tr>
              <th className={classes.matrixCorner} scope="col">
                <span>true ↓ predicted →</span>
              </th>
              {names.map((name, column) => (
                <th
                  key={`${column}-${name}`}
                  scope="col"
                  className={classes.matrixColumn}
                  title={name}
                >
                  <span>{name}</span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, r) => (
              <tr key={`${r}-${label(r)}`}>
                <th scope="row" className={classes.matrixRow} title={label(r)}>
                  {label(r)}
                </th>
                {row.map((count, c) => {
                  const share = shares[r][c];
                  const fill = sequentialColor(share, scheme, true);
                  const text = `True ${label(r)}, predicted ${label(c)}: ${count.toLocaleString('en-US')} (${formatPercent(share)} of ${label(r)})`;
                  return (
                    <td
                      key={c}
                      className={classes.matrixCell}
                      data-diagonal={r === c || undefined}
                      style={{ backgroundColor: fill, color: textOn(fill) }}
                      title={text}
                    >
                      {mode === 'counts' ? count.toLocaleString('en-US') : formatPercent(share, 0)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </ScrollArea.Autosize>
    </ChartFrame>
  );
}

/** More classes than the matrix keeps: the most confused pairs instead. */
function ConfusedPairs({
  evaluation,
  split,
}: {
  evaluation: ClassificationEvaluation;
  split: ClassificationSplit;
}) {
  const names = evaluation.class_names;
  const support = split.per_class.support;
  return (
    <div>
      <Text size="sm" fw={600}>
        Most confused classes
      </Text>
      <Text size="xs" c="dimmed" mb={4}>
        No confusion matrix for {names.length} classes (it is kept up to{' '}
        {evaluation.matrix_classes_max}); the pairs mixed up most often instead.
      </Text>
      {split.most_confused.length === 0 ? (
        <Text size="sm" c="dimmed">
          Every row was predicted right.
        </Text>
      ) : (
        <ScrollArea type="auto">
          <Table fz="xs" aria-label="Most confused classes">
            <Table.Thead>
              <Table.Tr>
                <Table.Th>True class</Table.Th>
                <Table.Th>Predicted as</Table.Th>
                <Table.Th ta="right">Rows</Table.Th>
                <Table.Th ta="right">Of the true class</Table.Th>
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {split.most_confused.map((pair) => (
                <Table.Tr key={`${pair.true}-${pair.predicted}`}>
                  <Table.Td>{names[pair.true] ?? pair.true}</Table.Td>
                  <Table.Td>{names[pair.predicted] ?? pair.predicted}</Table.Td>
                  <Table.Td ta="right">{pair.count.toLocaleString('en-US')}</Table.Td>
                  <Table.Td ta="right">
                    {support[pair.true] ? formatPercent(pair.count / support[pair.true]) : '–'}
                  </Table.Td>
                </Table.Tr>
              ))}
            </Table.Tbody>
          </Table>
        </ScrollArea>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- per-class table

type SortKey = 'class' | 'precision' | 'recall' | 'f1' | 'support';

const COLUMNS: { key: Exclude<SortKey, 'class'>; label: string }[] = [
  { key: 'precision', label: 'Precision' },
  { key: 'recall', label: 'Recall' },
  { key: 'f1', label: 'F1' },
  { key: 'support', label: 'Support' },
];

/** Row order of the per-class table: by class (index) or by a column, nulls last. */
export function sortClasses(
  perClass: PerClassMetrics,
  count: number,
  key: SortKey,
  descending: boolean
): number[] {
  const indices = Array.from({ length: count }, (_, i) => i);
  if (key === 'class') {
    return descending ? indices.reverse() : indices;
  }
  const values = perClass[key] as (number | null)[];
  return indices.sort((a, b) => {
    const va = values[a];
    const vb = values[b];
    if (va === null || va === undefined) {
      return vb === null || vb === undefined ? a - b : 1;
    }
    if (vb === null || vb === undefined) {
      return -1;
    }
    return (descending ? vb - va : va - vb) || a - b;
  });
}

function SortHeader({
  label,
  active,
  descending,
  onClick,
  numeric,
}: {
  label: string;
  active: boolean;
  descending: boolean;
  onClick: () => void;
  numeric: boolean;
}) {
  const Icon = active ? (descending ? IconArrowDown : IconArrowUp) : IconSelector;
  return (
    <Table.Th
      ta={numeric ? 'right' : 'left'}
      aria-sort={active ? (descending ? 'descending' : 'ascending') : 'none'}
    >
      <UnstyledButton onClick={onClick} fz="xs" fw={700}>
        <Group gap={2} wrap="nowrap" justify={numeric ? 'flex-end' : 'flex-start'}>
          {label}
          <Icon size={12} aria-hidden />
        </Group>
      </UnstyledButton>
    </Table.Th>
  );
}

/** Precision, recall, F1 and support of every class; the columns sort on click. */
export function PerClassTable({ names, perClass }: { names: string[]; perClass: PerClassMetrics }) {
  const [sort, setSort] = useState<{ key: SortKey; descending: boolean }>({
    key: 'class',
    descending: false,
  });
  const order = useMemo(
    () => sortClasses(perClass, names.length, sort.key, sort.descending),
    [perClass, names.length, sort]
  );
  const toggle = (key: SortKey) =>
    setSort((current) =>
      current.key === key
        ? { key, descending: !current.descending }
        : { key, descending: key !== 'class' }
    );
  return (
    <div>
      <Text size="sm" fw={600} mb={4}>
        Per class
      </Text>
      <ScrollArea.Autosize mah={400} type="auto">
        <Table fz="xs" stickyHeader aria-label="Metrics per class">
          <Table.Thead>
            <Table.Tr>
              <SortHeader
                label="Class"
                active={sort.key === 'class'}
                descending={sort.descending}
                onClick={() => toggle('class')}
                numeric={false}
              />
              {COLUMNS.map((column) => (
                <SortHeader
                  key={column.key}
                  label={column.label}
                  active={sort.key === column.key}
                  descending={sort.descending}
                  onClick={() => toggle(column.key)}
                  numeric
                />
              ))}
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {order.map((index) => (
              <Table.Tr key={`${index}-${names[index]}`}>
                <Table.Td style={{ overflowWrap: 'anywhere' }}>{names[index]}</Table.Td>
                <Table.Td ta="right">{formatPercent(perClass.precision[index])}</Table.Td>
                <Table.Td ta="right">{formatPercent(perClass.recall[index])}</Table.Td>
                <Table.Td ta="right">{formatScore(perClass.f1[index])}</Table.Td>
                <Table.Td ta="right">
                  {(perClass.support[index] ?? 0).toLocaleString('en-US')}
                </Table.Td>
              </Table.Tr>
            ))}
          </Table.Tbody>
        </Table>
      </ScrollArea.Autosize>
      <Text size="xs" c="dimmed" mt={4}>
        Precision: of the rows predicted as the class, the share that is right. Recall: of the rows
        of the class, the share found. F1: their harmonic mean. Support: rows of the class. “–”: not
        defined (never predicted, or no rows).
      </Text>
    </div>
  );
}

// ---------------------------------------------------------------- sections

function SplitPicker({
  value,
  onChange,
  options,
}: {
  value: SplitKey;
  onChange: (value: SplitKey) => void;
  options: SplitKey[];
}) {
  if (options.length < 2) {
    return null;
  }
  return (
    <SegmentedControl
      size="xs"
      value={value}
      onChange={(next) => onChange(next as SplitKey)}
      data={options.map((key) => ({ value: key, label: SPLIT_LABELS[key] }))}
      aria-label="Rows of the matrix and the table"
    />
  );
}

function ClassificationResults({
  evaluation,
  trainHint,
}: {
  evaluation: ClassificationEvaluation;
  trainHint: string;
}) {
  const [splitKey, setSplitKey] = useState<SplitKey>('val');
  const parts = splitsOf(evaluation);
  const { key: shown, split } = pickSplit(parts, splitKey);
  const options = parts.map((part) => part.key);
  const names = evaluation.class_names;
  const tiles = [
    tile(parts, 'Accuracy', (s) => s.accuracy, formatPercent),
    tile(
      parts,
      'Balanced accuracy',
      (s) => s.balanced_accuracy,
      formatPercent,
      'mean recall per class'
    ),
    tile(parts, 'Macro F1', (s) => s.macro_f1, formatScore, 'mean over the classes'),
    tile(parts, 'Weighted F1', (s) => s.weighted_f1, formatScore, 'weighted by class size'),
    tile(parts, 'Top-3 accuracy', (s) => s.top_k_accuracy?.['3'], formatPercent),
    tile(parts, 'Top-5 accuracy', (s) => s.top_k_accuracy?.['5'], formatPercent),
  ].filter(Boolean);
  return (
    <>
      <SimpleGrid cols={{ base: 2, sm: 3 }} spacing="xs" verticalSpacing="xs">
        {tiles}
        {lossTile(evaluation, parts, 'Log loss (cross-entropy)', trainHint)}
      </SimpleGrid>
      <Group justify="space-between" gap="xs">
        <Text size="xs" c="dimmed">
          {split.size.toLocaleString('en-US')} {ROW_LABELS[shown]} rows · {names.length} classes
        </Text>
        <SplitPicker value={shown} onChange={setSplitKey} options={options} />
      </Group>
      {split.confusion_matrix ? (
        <ConfusionMatrix
          names={names}
          matrix={split.confusion_matrix}
          subtitle={`${SPLIT_LABELS[shown]} rows`}
        />
      ) : (
        <ConfusedPairs evaluation={evaluation} split={split} />
      )}
      <PerClassTable names={names} perClass={split.per_class} />
    </>
  );
}

/** Pairs of the sample where both values exist. */
function samplePoints(split: RegressionSplit): { actual: number[]; predicted: number[] } {
  const actual: number[] = [];
  const predicted: number[] = [];
  const sample = split.sample;
  if (sample) {
    sample.actual.forEach((value, i) => {
      const guess = sample.predicted[i];
      if (typeof value === 'number' && typeof guess === 'number') {
        actual.push(value);
        predicted.push(guess);
      }
    });
  }
  return { actual, predicted };
}

function RegressionResults({
  evaluation,
  trainHint,
}: {
  evaluation: RegressionEvaluation;
  trainHint: string;
}) {
  const [splitKey, setSplitKey] = useState<SplitKey>('val');
  const parts = splitsOf(evaluation);
  const { key: shown, split } = pickSplit(parts, splitKey);
  const options = parts.map((part) => part.key);
  const points = useMemo(() => samplePoints(split), [split]);
  const tiles = [
    tile(parts, 'RMSE', (s) => s.rmse, formatMetric, 'root mean squared error'),
    tile(parts, 'MAE', (s) => s.mae, formatMetric, 'mean absolute error'),
    tile(
      parts,
      'R²',
      (s) => s.r2,
      (v) => (typeof v === 'number' ? v.toFixed(3) : '–')
    ),
    tile(parts, 'MAPE', (s) => s.mape, formatPercent, 'mean absolute % error'),
    tile(parts, 'Median abs. error', (s) => s.median_absolute_error, formatMetric),
    tile(parts, 'Max error', (s) => s.max_error, formatMetric, 'largest absolute error'),
  ].filter(Boolean);
  const residuals = split.residuals;
  return (
    <>
      <SimpleGrid cols={{ base: 2, sm: 3 }} spacing="xs" verticalSpacing="xs">
        {tiles}
        {lossTile(evaluation, parts, 'Loss (MSE, standardised)', trainHint)}
      </SimpleGrid>
      {residuals && (
        <Text size="xs" c="dimmed">
          Residuals (predicted − actual), {ROW_LABELS[shown]} rows: 5% below{' '}
          {formatMetric(residuals.q05)}, half between {formatMetric(residuals.q25)} and{' '}
          {formatMetric(residuals.q75)} (median {formatMetric(residuals.median)}), 5% above{' '}
          {formatMetric(residuals.q95)}.
        </Text>
      )}
      {points.actual.length > 0 && (
        <ScatterPlot
          title="Predicted vs. actual"
          subtitle={`${SPLIT_LABELS[shown]} rows, in the units of the target`}
          controls={<SplitPicker value={shown} onChange={setSplitKey} options={options} />}
          x={points.actual}
          y={points.predicted}
          xLabel="Actual"
          yLabel="Predicted"
          diagonal="Predicted = actual"
          height={300}
          caption={
            points.actual.length < split.size
              ? `${points.actual.length.toLocaleString('en-US')} of ${split.size.toLocaleString('en-US')} rows, evenly spaced.`
              : `All ${split.size.toLocaleString('en-US')} rows.`
          }
        />
      )}
    </>
  );
}

/** The "Metrics" section of a finished run; nothing for runs without an evaluation. */
export function EvaluationResults({
  evaluation,
  classical = false,
}: {
  evaluation: RunEvaluation | null | undefined;
  classical?: boolean;
}) {
  if (!evaluation || splitsOf<unknown>(evaluation).length === 0) {
    return null;
  }
  // Without validation and test rows the model can only be measured on what it learned from.
  const optimistic = !evaluation.val && !evaluation.test;
  const trainHint = classical
    ? 'train: the fitted model on its training rows'
    : 'train: mean over the batches of the epoch';
  const kept = classical
    ? 'The fitted model'
    : [
        typeof evaluation.epoch === 'number'
          ? `Weights of epoch ${evaluation.epoch}`
          : 'The kept weights',
        typeof evaluation.trial === 'number' ? `trial ${evaluation.trial}` : null,
      ]
        .filter(Boolean)
        .join(' · ');
  return (
    <Stack gap="sm" component="section" aria-label="Metrics">
      <Group justify="space-between" gap="xs" align="baseline">
        <Text size="sm" fw={600}>
          Metrics
        </Text>
        <Text size="xs" c="dimmed">
          {kept}
        </Text>
      </Group>
      {optimistic && (
        <Text size="xs" c="orange" data-testid="optimistic-metrics">
          No validation or test rows: these numbers come from the training rows, which the model has
          learned from, so they are optimistic. They do not say how well it works on new data.
        </Text>
      )}
      {evaluation.task === 'classification' ? (
        <ClassificationResults evaluation={evaluation} trainHint={trainHint} />
      ) : (
        <RegressionResults evaluation={evaluation} trainHint={trainHint} />
      )}
    </Stack>
  );
}
