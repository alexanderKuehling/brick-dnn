/**
 * Binary classification: the decision threshold. On imbalanced data (say 2% fraud) the most
 * probable class (threshold 0.5) often finds few of the rare class; a lower threshold finds
 * more of it at the cost of more false alarms. A slider over the thresholds of the evaluation
 * shows precision, recall, F1, the false-positive rate and the confusion counts on every split,
 * the precision-recall and ROC curves mark the chosen point, and the owner of the run can save
 * it as the threshold its predictions use.
 */
import { useEffect, useMemo, useState } from 'react';
import { IconCheck } from '@tabler/icons-react';
import {
  Button,
  Group,
  Paper,
  ScrollArea,
  SimpleGrid,
  Slider,
  Stack,
  Table,
  Text,
} from '@mantine/core';
import { notifications } from '@mantine/notifications';
import type { ClassificationEvaluation, ThresholdCurve } from '@/api';
import { CurveChart } from '@/components/charts/CurveChart';
import { formatPercent } from './format';
import {
  DEFAULT_THRESHOLD,
  formatThreshold,
  nearestIndex,
  pointAt,
  prLine,
  rocLine,
} from './threshold';

type SplitKey = 'val' | 'test' | 'training';

const LABELS: Record<SplitKey, string> = {
  val: 'Validation',
  test: 'Test',
  training: 'Training (optimistic)',
};

const SOURCE_NOTES: Record<SplitKey, string> = {
  val: 'The thresholds and suggestions come from the validation rows; check them on the test rows.',
  test: 'No validation rows: the thresholds and suggestions come from the test rows, so a threshold picked here makes the test numbers optimistic.',
  training:
    'No validation or test rows: the thresholds and suggestions come from the training rows, which the model has learned from, so all numbers are optimistic.',
};

/** The splits with a threshold curve, in order. */
export function curvesOf(evaluation: ClassificationEvaluation): {
  key: SplitKey;
  curve: ThresholdCurve;
}[] {
  const found: { key: SplitKey; curve: ThresholdCurve }[] = [];
  for (const key of ['val', 'test', 'training'] as SplitKey[]) {
    const curve = evaluation[key]?.threshold_curve;
    if (curve && curve.points.threshold.length > 0) {
      found.push({ key, curve });
    }
  }
  return found;
}

function count(value: number): string {
  return value.toLocaleString('en-US');
}

export function ThresholdAnalysis({
  evaluation,
  savedThreshold,
  onSave,
}: {
  evaluation: ClassificationEvaluation;
  /** The threshold the run's predictions use (null or missing: the default 0.5). */
  savedThreshold?: number | null;
  /** Saves a new threshold; missing when the user may not (only who started the run may). */
  onSave?: (threshold: number) => Promise<void>;
}) {
  const curves = useMemo(() => curvesOf(evaluation), [evaluation]);
  const thresholds = useMemo(() => curves[0]?.curve.points.threshold ?? [], [curves]);
  const saved = savedThreshold ?? DEFAULT_THRESHOLD;
  const [index, setIndex] = useState(() => nearestIndex(thresholds, saved));
  const [saving, setSaving] = useState(false);
  // A threshold saved elsewhere (or the run reloaded): start from it again.
  useEffect(() => {
    setIndex(nearestIndex(thresholds, saved));
  }, [saved, thresholds]);
  const lines = useMemo(
    () => curves.map(({ key, curve }) => ({ key, pr: prLine(curve), roc: rocLine(curve) })),
    [curves]
  );
  if (curves.length === 0) {
    return null;
  }
  const binary = evaluation.binary;
  const names = evaluation.class_names;
  const positive = names[binary?.positive_class ?? 1] ?? 'class 1';
  const negative = names[1 - (binary?.positive_class ?? 1)] ?? 'class 0';
  const threshold = thresholds[index];
  const points = curves.map(({ key, curve }) => ({ key, point: pointAt(curve, index) }));
  const source: SplitKey = binary?.source ?? curves[0].key;
  const exact = Math.abs(threshold - saved) < 1e-12;
  const savedNearest = thresholds[nearestIndex(thresholds, saved)];

  const pick = (value: number) => setIndex(nearestIndex(thresholds, value));
  const save = async () => {
    if (!onSave) {
      return;
    }
    setSaving(true);
    try {
      await onSave(threshold);
      notifications.show({
        color: 'green',
        title: 'Threshold saved',
        message: `Predictions of this run are now “${positive}” from a probability of ${formatThreshold(threshold)}.`,
      });
    } catch (error) {
      notifications.show({
        color: 'red',
        title: 'Cannot save the threshold',
        message: (error as Error).message,
      });
    } finally {
      setSaving(false);
    }
  };

  const rows: { label: string; value: (p: (typeof points)[number]['point']) => string }[] = [
    { label: 'Precision', value: (p) => formatPercent(p.precision) },
    { label: 'Recall', value: (p) => formatPercent(p.recall) },
    { label: 'F1', value: (p) => (p.f1 === null ? '–' : p.f1.toFixed(3)) },
    { label: 'False-positive rate', value: (p) => formatPercent(p.fpr, 2) },
    { label: `True positives (${positive} found)`, value: (p) => count(p.tp) },
    { label: `False positives (false alarms)`, value: (p) => count(p.fp) },
    { label: `True negatives`, value: (p) => count(p.tn) },
    { label: `False negatives (${positive} missed)`, value: (p) => count(p.fn) },
  ];

  const suggestions = [
    binary?.best_f1 ? { label: 'Best F1', value: binary.best_f1.threshold } : null,
    binary?.recall
      ? {
          label: `Recall ≥ ${formatPercent(binary.recall_target, 0)}`,
          value: binary.recall.threshold,
        }
      : null,
    { label: 'Default', value: DEFAULT_THRESHOLD },
  ].filter((item): item is { label: string; value: number } => item !== null);

  const marks = [
    { value: nearestIndex(thresholds, DEFAULT_THRESHOLD), label: '0.5' },
    ...(exact ? [] : [{ value: nearestIndex(thresholds, saved), label: 'saved' }]),
  ];

  return (
    <Paper withBorder p="sm" radius="sm" component="section" aria-label="Decision threshold">
      <Stack gap="sm">
        <div>
          <Text size="sm" fw={600}>
            Decision threshold
          </Text>
          <Text size="xs" c="dimmed">
            A row is predicted “{positive}” when its probability of “{positive}” is at least the
            threshold, else “{negative}”. Lower finds more “{positive}” (recall) with more false
            alarms (precision falls).
          </Text>
        </div>
        <Group gap="lg" align="flex-end" wrap="wrap">
          <div>
            <Text size="xs" c="dimmed">
              Threshold
            </Text>
            <Text fw={700} size="xl" lh={1.2} data-testid="threshold-value">
              {formatThreshold(threshold)}
            </Text>
          </div>
          <Group gap={6} wrap="wrap">
            {suggestions.map((item) => (
              <Button
                key={item.label}
                size="compact-xs"
                variant="light"
                onClick={() => pick(item.value)}
                aria-label={`${item.label}: threshold ${formatThreshold(item.value)}`}
              >
                {item.label} · {formatThreshold(item.value)}
              </Button>
            ))}
          </Group>
        </Group>
        <Slider
          min={0}
          max={thresholds.length - 1}
          step={1}
          value={index}
          onChange={setIndex}
          label={(i) => formatThreshold(thresholds[i])}
          marks={marks}
          thumbLabel="Decision threshold"
          aria-valuetext={formatThreshold(threshold)}
          mb="md"
        />
        <ScrollArea type="auto">
          <Table fz="xs" aria-label="Results at the threshold" withRowBorders={false}>
            <Table.Thead>
              <Table.Tr>
                <Table.Th>At {formatThreshold(threshold)}</Table.Th>
                {points.map(({ key }) => (
                  <Table.Th key={key} ta="right">
                    {LABELS[key]}
                  </Table.Th>
                ))}
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {rows.map((row) => (
                <Table.Tr key={row.label}>
                  <Table.Td>{row.label}</Table.Td>
                  {points.map(({ key, point }) => (
                    <Table.Td key={key} ta="right" style={{ fontVariantNumeric: 'tabular-nums' }}>
                      {row.value(point)}
                    </Table.Td>
                  ))}
                </Table.Tr>
              ))}
            </Table.Tbody>
          </Table>
        </ScrollArea>
        <Text size="xs" c="dimmed">
          {SOURCE_NOTES[source]}
        </Text>
        <Group gap="xs" justify="space-between" wrap="wrap">
          <Text size="xs" c="dimmed" data-testid="saved-threshold">
            Predictions use {formatThreshold(saved)}
            {savedNearest !== undefined && Math.abs(savedNearest - saved) > 1e-12
              ? ` (the closest threshold measured: ${formatThreshold(savedNearest)})`
              : ''}
            .{!onSave && ' Only who started the run can change it.'}
          </Text>
          {onSave && (
            <Button
              size="xs"
              leftSection={<IconCheck size={14} />}
              disabled={exact}
              loading={saving}
              onClick={() => void save()}
            >
              Use this threshold for predictions
            </Button>
          )}
        </Group>
        <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
          <CurveChart
            title="Precision-recall curve"
            subtitle={`“${positive}” as the positive class; the dot is the threshold`}
            xLabel="Recall"
            yLabel="Precision"
            series={lines.map(({ key, pr }) => {
              const point = points.find((p) => p.key === key)!.point;
              return {
                key,
                label: LABELS[key],
                x: pr.x,
                y: pr.y,
                marker:
                  point.recall !== null && point.precision !== null
                    ? { x: point.recall, y: point.precision }
                    : null,
              };
            })}
            table={() => ({
              head: ['Split', 'Threshold', 'Recall', 'Precision'],
              rows: lines.flatMap(({ key, pr }) =>
                pr.x.map((x, i) => [
                  LABELS[key],
                  formatThreshold(pr.threshold[i]),
                  formatPercent(x),
                  formatPercent(pr.y[i]),
                ])
              ),
            })}
          />
          <CurveChart
            title="ROC curve"
            subtitle="Recall against the false-positive rate; the dot is the threshold"
            xLabel="False-positive rate"
            yLabel="Recall"
            diagonal="Guessing"
            series={lines.map(({ key, roc }) => {
              const point = points.find((p) => p.key === key)!.point;
              return {
                key,
                label: LABELS[key],
                x: roc.x,
                y: roc.y,
                marker:
                  point.recall !== null && point.fpr !== null
                    ? { x: point.fpr, y: point.recall }
                    : null,
              };
            })}
            table={() => ({
              head: ['Split', 'Threshold', 'False-positive rate', 'Recall'],
              rows: lines.flatMap(({ key, roc }) =>
                roc.x.map((x, i) => [
                  LABELS[key],
                  Number.isFinite(roc.threshold[i]) ? formatThreshold(roc.threshold[i]) : 'above 1',
                  formatPercent(x),
                  formatPercent(roc.y[i]),
                ])
              ),
            })}
          />
        </SimpleGrid>
      </Stack>
    </Paper>
  );
}
