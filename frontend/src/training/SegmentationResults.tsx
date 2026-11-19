/**
 * What a segmentation run shows besides its curves: the IoU of every class, and a gallery of
 * validation images with their mask, the mask the network predicts and the prediction drawn
 * over the image (the preview images the training writes once the weights are saved).
 */
import { useEffect, useState } from 'react';
import { Alert, Box, Group, Loader, Paper, ScrollArea, Stack, Table, Text } from '@mantine/core';
import { api, type RunDetail, type RunPreviews, type SegmentationRunState } from '@/api';
import { i18n, useTranslation } from '@/i18n';
import { formatPercent } from './format';
import classes from './Training.module.css';

/** A square in the colour of a class (the name stands next to it: never colour alone). */
function Swatch({ color }: { color: string }) {
  return <span className={classes.swatch} style={{ backgroundColor: color }} aria-hidden />;
}

/** The classes with their colours, as the masks show them. */
export function ClassLegend({
  names,
  colors,
  ignoreColor = null,
}: {
  names: string[];
  colors: string[];
  /** Set if some pixels of the masks are ignored. */
  ignoreColor?: string | null;
}) {
  useTranslation();
  return (
    <Group
      gap="sm"
      style={{ rowGap: 4 }}
      role="list"
      aria-label={i18n.t('ui.training.SegmentationResults.classes')}
    >
      {names.map((name, index) => (
        <Group key={`${index}-${name}`} gap={6} wrap="nowrap" role="listitem">
          <Swatch color={colors[index] ?? '#888'} />
          <Text size="xs">{name}</Text>
        </Group>
      ))}
      {ignoreColor && (
        <Group gap={6} wrap="nowrap" role="listitem">
          <Swatch color={ignoreColor} />
          <Text size="xs" c="dimmed">
            {i18n.t('ui.training.SegmentationResults.ignored')}
          </Text>
        </Group>
      )}
    </Group>
  );
}

/** The IoU of every class for the kept weights: validation and, if measured, test. */
export function ClassTable({ results }: { results: SegmentationRunState }) {
  useTranslation();
  const { validation, test } = results;
  if (!validation && !test) {
    return null;
  }
  const epoch = validation?.epoch;
  return (
    <div>
      <Group justify="space-between" gap="xs" mb={4}>
        <Text size="sm" fw={600}>
          {i18n.t('ui.training.SegmentationResults.iouPerClass')}
        </Text>
        <Text size="xs" c="dimmed">
          {typeof epoch === 'number'
            ? i18n.t('ui.training.SegmentationResults.weightsOfEpoch2', { epoch })
            : i18n.t('ui.training.SegmentationResults.theKeptWeights')}
        </Text>
      </Group>
      <ScrollArea type="auto">
        <Table fz="xs" aria-label={i18n.t('ui.training.SegmentationResults.resultsPerClass')}>
          <Table.Thead>
            <Table.Tr>
              <Table.Th>{i18n.t('ui.training.SegmentationResults.class')}</Table.Th>
              {validation && (
                <Table.Th ta="right">{i18n.t('ui.training.SegmentationResults.valIou')}</Table.Th>
              )}
              {test && (
                <Table.Th ta="right">{i18n.t('ui.training.SegmentationResults.testIou')}</Table.Th>
              )}
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {results.class_names.map((name, index) => (
              <Table.Tr key={`${index}-${name}`}>
                <Table.Td>
                  <Group gap={6} wrap="nowrap">
                    <Swatch color={results.colors[index] ?? '#888'} />
                    {name}
                  </Group>
                </Table.Td>
                {validation && (
                  <Table.Td ta="right">{formatPercent(validation.class_iou[index])}</Table.Td>
                )}
                {test && <Table.Td ta="right">{formatPercent(test.class_iou[index])}</Table.Td>}
              </Table.Tr>
            ))}
            <Table.Tr fw={700}>
              <Table.Td>{i18n.t('ui.training.SegmentationResults.mean')}</Table.Td>
              {validation && <Table.Td ta="right">{formatPercent(validation.iou)}</Table.Td>}
              {test && <Table.Td ta="right">{formatPercent(test.iou)}</Table.Td>}
            </Table.Tr>
          </Table.Tbody>
        </Table>
      </ScrollArea>
      <Text size="xs" c="dimmed" mt={4}>
        {i18n.t('ui.training.SegmentationResults.iouTheOverlapOfThe')}
      </Text>
    </div>
  );
}

const PARTS = [
  {
    key: 'image',
    get label() {
      return i18n.t('ui.training.SegmentationResults.image2');
    },
  },
  {
    key: 'truth',
    get label() {
      return i18n.t('ui.training.SegmentationResults.mask');
    },
  },
  {
    key: 'prediction',
    get label() {
      return i18n.t('ui.training.SegmentationResults.prediction');
    },
  },
  {
    key: 'overlay',
    get label() {
      return i18n.t('ui.training.SegmentationResults.overlay');
    },
  },
] as const;

/**
 * Where the preview images come from: "validation", or "test" or "training" for a run without
 * validation data.
 */
function splitName(split: string | null | undefined): string {
  return split === 'test' || split === 'training' ? split : 'validation';
}

/** "8 validation images · weights of epoch 7". */
function galleryNote(previews: RunPreviews): string {
  const split = splitName(previews.split);
  const count = previews.samples.length;
  const images =
    split === 'training'
      ? i18n.t('ui.training.SegmentationResults.trainingImages', { count })
      : split === 'test'
        ? i18n.t('ui.training.SegmentationResults.testImages', { count })
        : i18n.t('ui.training.SegmentationResults.validationImages', { count });
  return typeof previews.epoch === 'number'
    ? i18n.t('ui.training.SegmentationResults.weightsOfEpoch', { images, epoch: previews.epoch })
    : images;
}

/** The preview images of a run, loaded once it has them. */
function usePreviews(runId: string, enabled: boolean) {
  const [previews, setPreviews] = useState<RunPreviews | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    if (!enabled) {
      setPreviews(null);
      setError(null);
      return undefined;
    }
    const controller = new AbortController();
    api.training.runs
      .previews(runId, controller.signal)
      .then((result) => {
        setPreviews(result);
        setError(null);
      })
      .catch((reason: Error) => {
        if (!controller.signal.aborted) {
          setError(reason.message);
        }
      });
    return () => controller.abort();
  }, [runId, enabled]);
  return { previews, error };
}

/**
 * Validation images (without validation data: test or training images) with their mask, the
 * predicted mask and the prediction over the image.
 */
export function PreviewGallery({ run }: { run: RunDetail }) {
  useTranslation();
  const count = run.state?.segmentation?.previews?.count ?? 0;
  const { previews, error } = usePreviews(run.id, run.has_weights && count > 0);
  if (count === 0 || !run.has_weights) {
    return null;
  }
  const split = splitName(previews?.split ?? run.state?.segmentation?.previews?.split);
  const kind = `${split[0].toUpperCase()}${split.slice(1)}`;
  const ignored = run.state?.segmentation?.ignore_index != null;
  return (
    <Stack
      gap="xs"
      component="section"
      aria-label={i18n.t('ui.training.SegmentationResults.predictions')}
    >
      <Group justify="space-between" gap="xs">
        <Text size="sm" fw={600}>
          {i18n.t('ui.training.SegmentationResults.predictions')}
        </Text>
        <Text size="xs" c="dimmed">
          {previews
            ? galleryNote(previews)
            : i18n.t('ui.training.SegmentationResults.images', { count, split })}
        </Text>
      </Group>
      {error && (
        <Alert
          color="red"
          title={i18n.t('ui.training.SegmentationResults.cannotLoadThePredictions')}
        >
          {error}
        </Alert>
      )}
      {!previews && !error && (
        <Group gap="xs">
          <Loader size="xs" />
          <Text size="sm" c="dimmed">
            {i18n.t('ui.training.SegmentationResults.loadingThePredictions')}
          </Text>
        </Group>
      )}
      {previews && (
        <>
          <ClassLegend
            names={previews.class_names}
            colors={previews.colors}
            ignoreColor={ignored ? previews.ignore_color : null}
          />
          <div className={classes.previewHead} aria-hidden>
            {PARTS.map((part) => (
              <Text key={part.key} size="xs" c="dimmed" ta="center">
                {part.label}
              </Text>
            ))}
          </div>
          <Stack gap="xs">
            {previews.samples.map((sample) => (
              <Paper
                key={sample.index}
                withBorder
                p={6}
                radius="sm"
                component="figure"
                m={0}
                aria-label={i18n.t('ui.training.SegmentationResults.image', {
                  kind,
                  value: sample.index + 1,
                })}
              >
                <div className={classes.previewRow}>
                  {PARTS.map((part) => (
                    <Box
                      key={part.key}
                      component="img"
                      className={classes.previewImage}
                      src={sample[part.key].url}
                      alt={i18n.t('ui.training.SegmentationResults.ofImage', {
                        label: part.label,
                        split,
                        value: sample.index + 1,
                      })}
                      width={previews.width}
                      height={previews.height}
                      loading="lazy"
                    />
                  ))}
                </div>
                <Text component="figcaption" size="xs" c="dimmed" mt={4}>
                  {i18n.t('ui.training.SegmentationResults.iouPixelAccuracy', {
                    iou: formatPercent(sample.iou),
                    pixel_accuracy: formatPercent(sample.pixel_accuracy),
                    map: sample.classes
                      .map((index) => previews.class_names[index] ?? index)
                      .join(', '),
                  })}
                </Text>
              </Paper>
            ))}
          </Stack>
        </>
      )}
    </Stack>
  );
}
