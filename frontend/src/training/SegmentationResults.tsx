/**
 * What a segmentation run shows besides its curves: the IoU of every class, and a gallery of
 * validation images with their mask, the mask the network predicts and the prediction drawn
 * over the image (the preview images the training writes once the weights are saved).
 */
import { useEffect, useState } from 'react';
import { Alert, Box, Group, Loader, Paper, ScrollArea, Stack, Table, Text } from '@mantine/core';
import { api, type RunDetail, type RunPreviews, type SegmentationRunState } from '@/api';
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
  return (
    <Group gap="sm" style={{ rowGap: 4 }} role="list" aria-label="Classes">
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
            ignored
          </Text>
        </Group>
      )}
    </Group>
  );
}

/** The IoU of every class for the kept weights: validation and, if measured, test. */
export function ClassTable({ results }: { results: SegmentationRunState }) {
  const { validation, test } = results;
  if (!validation && !test) {
    return null;
  }
  const epoch = validation?.epoch;
  return (
    <div>
      <Group justify="space-between" gap="xs" mb={4}>
        <Text size="sm" fw={600}>
          IoU per class
        </Text>
        <Text size="xs" c="dimmed">
          {typeof epoch === 'number' ? `Weights of epoch ${epoch}` : 'The kept weights'}
        </Text>
      </Group>
      <ScrollArea type="auto">
        <Table fz="xs" aria-label="Results per class">
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Class</Table.Th>
              {validation && <Table.Th ta="right">Val. IoU</Table.Th>}
              {test && <Table.Th ta="right">Test IoU</Table.Th>}
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
              <Table.Td>Mean</Table.Td>
              {validation && <Table.Td ta="right">{formatPercent(validation.iou)}</Table.Td>}
              {test && <Table.Td ta="right">{formatPercent(test.iou)}</Table.Td>}
            </Table.Tr>
          </Table.Tbody>
        </Table>
      </ScrollArea>
      <Text size="xs" c="dimmed" mt={4}>
        IoU: the overlap of the predicted and the true area of a class, divided by their union. “–”:
        the class does not occur.
      </Text>
    </div>
  );
}

const PARTS = [
  { key: 'image', label: 'Image' },
  { key: 'truth', label: 'Mask' },
  { key: 'prediction', label: 'Prediction' },
  { key: 'overlay', label: 'Overlay' },
] as const;

/** "8 validation images · weights of epoch 7". */
function galleryNote(previews: RunPreviews): string {
  const images = `${previews.samples.length} validation images`;
  return typeof previews.epoch === 'number'
    ? `${images} · weights of epoch ${previews.epoch}`
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

/** Validation images with their mask, the predicted mask and the prediction over the image. */
export function PreviewGallery({ run }: { run: RunDetail }) {
  const count = run.state?.segmentation?.previews?.count ?? 0;
  const { previews, error } = usePreviews(run.id, run.has_weights && count > 0);
  if (count === 0 || !run.has_weights) {
    return null;
  }
  const ignored = run.state?.segmentation?.ignore_index != null;
  return (
    <Stack gap="xs" component="section" aria-label="Predictions">
      <Group justify="space-between" gap="xs">
        <Text size="sm" fw={600}>
          Predictions
        </Text>
        <Text size="xs" c="dimmed">
          {previews ? galleryNote(previews) : `${count} validation images`}
        </Text>
      </Group>
      {error && (
        <Alert color="red" title="Cannot load the predictions">
          {error}
        </Alert>
      )}
      {!previews && !error && (
        <Group gap="xs">
          <Loader size="xs" />
          <Text size="sm" c="dimmed">
            Loading the predictions …
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
                aria-label={`Validation image ${sample.index + 1}`}
              >
                <div className={classes.previewRow}>
                  {PARTS.map((part) => (
                    <Box
                      key={part.key}
                      component="img"
                      className={classes.previewImage}
                      src={sample[part.key].url}
                      alt={`${part.label} of validation image ${sample.index + 1}`}
                      width={previews.width}
                      height={previews.height}
                      loading="lazy"
                    />
                  ))}
                </div>
                <Text component="figcaption" size="xs" c="dimmed" mt={4}>
                  IoU {formatPercent(sample.iou)} · pixel accuracy{' '}
                  {formatPercent(sample.pixel_accuracy)} ·{' '}
                  {sample.classes.map((index) => previews.class_names[index] ?? index).join(', ')}
                </Text>
              </Paper>
            ))}
          </Stack>
        </>
      )}
    </Stack>
  );
}
