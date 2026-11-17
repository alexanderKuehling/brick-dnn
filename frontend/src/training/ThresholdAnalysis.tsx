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
import { InfoHint } from '@/components/InfoHint';
import { currentLocale, i18n, useTranslation } from '@/i18n';
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
  get val() {
    return i18n.t('ui.training.ThresholdAnalysis.validation');
  },
  get test() {
    return i18n.t('ui.training.ThresholdAnalysis.test');
  },
  get training() {
    return i18n.t('ui.training.ThresholdAnalysis.trainingOptimistic');
  },
};

const SOURCE_NOTES: Record<SplitKey, string> = {
  get val() {
    return i18n.t('ui.training.ThresholdAnalysis.theThresholdsAndSuggestionsCome');
  },
  get test() {
    return i18n.t('ui.training.ThresholdAnalysis.noValidationRowsTheThresholds');
  },
  get training() {
    return i18n.t('ui.training.ThresholdAnalysis.noValidationOrTestRows');
  },
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
  return value.toLocaleString(currentLocale());
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
  useTranslation();
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
        title: i18n.t('ui.training.ThresholdAnalysis.thresholdSaved'),
        message: i18n.t('ui.training.ThresholdAnalysis.predictionsOfThisRunAre', {
          positive,
          threshold: formatThreshold(threshold),
        }),
      });
    } catch (error) {
      notifications.show({
        color: 'red',
        title: i18n.t('ui.training.ThresholdAnalysis.cannotSaveTheThreshold'),
        message: (error as Error).message,
      });
    } finally {
      setSaving(false);
    }
  };

  const rows: { label: string; value: (p: (typeof points)[number]['point']) => string }[] = [
    {
      label: i18n.t('ui.training.ThresholdAnalysis.precision'),
      value: (p) => formatPercent(p.precision),
    },
    {
      label: i18n.t('ui.training.ThresholdAnalysis.recall'),
      value: (p) => formatPercent(p.recall),
    },
    { label: 'F1', value: (p) => (p.f1 === null ? '–' : p.f1.toFixed(3)) },
    {
      label: i18n.t('ui.training.ThresholdAnalysis.falsePositiveRate'),
      value: (p) => formatPercent(p.fpr, 2),
    },
    {
      label: i18n.t('ui.training.ThresholdAnalysis.truePositivesFound', { positive }),
      value: (p) => count(p.tp),
    },
    {
      label: i18n.t('ui.training.ThresholdAnalysis.falsePositivesFalseAlarms'),
      value: (p) => count(p.fp),
    },
    { label: i18n.t('ui.training.ThresholdAnalysis.trueNegatives'), value: (p) => count(p.tn) },
    {
      label: i18n.t('ui.training.ThresholdAnalysis.falseNegativesMissed', { positive }),
      value: (p) => count(p.fn),
    },
  ];

  const suggestions = [
    binary?.best_f1
      ? { label: i18n.t('ui.training.ThresholdAnalysis.bestF1'), value: binary.best_f1.threshold }
      : null,
    binary?.recall
      ? {
          label: i18n.t('ui.training.ThresholdAnalysis.recall2', {
            recall_target: formatPercent(binary.recall_target, 0),
          }),
          value: binary.recall.threshold,
        }
      : null,
    { label: i18n.t('ui.training.ThresholdAnalysis.default'), value: DEFAULT_THRESHOLD },
  ].filter((item): item is { label: string; value: number } => item !== null);

  const marks = [
    { value: nearestIndex(thresholds, DEFAULT_THRESHOLD), label: '0.5' },
    ...(exact ? [] : [{ value: nearestIndex(thresholds, saved), label: 'saved' }]),
  ];

  return (
    <Paper
      withBorder
      p="sm"
      radius="sm"
      component="section"
      aria-label={i18n.t('ui.training.ThresholdAnalysis.decisionThreshold')}
    >
      <Stack gap="sm">
        <Group gap={4} wrap="nowrap">
          <Text size="sm" fw={600}>
            {i18n.t('ui.training.ThresholdAnalysis.decisionThreshold')}
          </Text>
          {/* How the threshold works, and where its numbers come from: on demand. */}
          <InfoHint topic={i18n.t('ui.training.ThresholdAnalysis.decisionThreshold')}>
            {i18n.t('ui.training.ThresholdAnalysis.aRowIsPredictedWhen', {
              positive,
              positive2: positive,
              negative,
              positive3: positive,
            })}
            {source === 'val' && ` ${SOURCE_NOTES.val}`}
          </InfoHint>
        </Group>
        <Group gap="lg" align="flex-end" wrap="wrap">
          <div>
            <Text size="xs" c="dimmed">
              {i18n.t('ui.training.ThresholdAnalysis.threshold')}
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
                aria-label={i18n.t('ui.training.ThresholdAnalysis.threshold2', {
                  label: item.label,
                  value: formatThreshold(item.value),
                })}
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
          // On the thumb (role slider), not on the root, which has no role.
          thumbProps={{ 'aria-valuetext': formatThreshold(threshold) }}
          mb="md"
        />
        <ScrollArea type="auto">
          <Table
            fz="xs"
            aria-label={i18n.t('ui.training.ThresholdAnalysis.resultsAtTheThreshold')}
            withRowBorders={false}
          >
            <Table.Thead>
              <Table.Tr>
                <Table.Th>
                  {i18n.t('ui.training.ThresholdAnalysis.at', {
                    threshold: formatThreshold(threshold),
                  })}
                </Table.Th>
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
        {/* Numbers that are not from the validation rows are optimistic: always said. */}
        {source !== 'val' && (
          <Text size="xs" c="dimmed">
            {SOURCE_NOTES[source]}
          </Text>
        )}
        <Group gap="xs" justify="space-between" wrap="wrap">
          <Text size="xs" c="dimmed" data-testid="saved-threshold">
            {i18n.t('ui.training.ThresholdAnalysis.predictionsUse')} {formatThreshold(saved)}
            {savedNearest !== undefined && Math.abs(savedNearest - saved) > 1e-12
              ? i18n.t('ui.training.ThresholdAnalysis.theClosestThresholdMeasured', {
                  savedNearest: formatThreshold(savedNearest),
                })
              : ''}
            .{!onSave && i18n.t('ui.training.ThresholdAnalysis.onlyWhoStartedTheRun')}
          </Text>
          {onSave && (
            <Button
              size="xs"
              leftSection={<IconCheck size={14} />}
              disabled={exact}
              loading={saving}
              onClick={() => void save()}
            >
              {i18n.t('ui.training.ThresholdAnalysis.useThisThresholdForPredictions')}
            </Button>
          )}
        </Group>
        <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
          <CurveChart
            title={i18n.t('ui.training.ThresholdAnalysis.precisionRecallCurve')}
            subtitle={i18n.t('ui.training.ThresholdAnalysis.asThePositiveClassThe', { positive })}
            xLabel={i18n.t('ui.training.ThresholdAnalysis.recall')}
            yLabel={i18n.t('ui.training.ThresholdAnalysis.precision')}
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
              head: [
                i18n.t('ui.training.ThresholdAnalysis.split'),
                i18n.t('ui.training.ThresholdAnalysis.threshold'),
                i18n.t('ui.training.ThresholdAnalysis.recall'),
                i18n.t('ui.training.ThresholdAnalysis.precision'),
              ],
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
            title={i18n.t('ui.training.ThresholdAnalysis.rocCurve')}
            subtitle={i18n.t('ui.training.ThresholdAnalysis.recallAgainstTheFalsePositive')}
            xLabel={i18n.t('ui.training.ThresholdAnalysis.falsePositiveRate')}
            yLabel={i18n.t('ui.training.ThresholdAnalysis.recall')}
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
