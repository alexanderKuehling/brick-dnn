import { useEffect, useRef, useState } from 'react';
import {
  IconBrush,
  IconCheck,
  IconEraser,
  IconPencil,
  IconUpload,
  IconX,
} from '@tabler/icons-react';
import {
  Alert,
  Badge,
  Button,
  FileButton,
  Group,
  Loader,
  NumberInput,
  Paper,
  ScrollArea,
  SimpleGrid,
  Stack,
  Table,
  Text,
  Tooltip,
  UnstyledButton,
} from '@mantine/core';
import { formatValue } from './format';
import { canvasPixels, loadImage, type ImageFormat } from './pixels';
import type { ListedSample, PickedSample, SamplesAnswer } from './types';
import classes from './Explain.module.css';

interface PickerProps {
  samples: SamplesAnswer | null;
  loading: boolean;
  error: string | null;
  picked: PickedSample | null;
  onPick: (sample: PickedSample) => void;
}

function isPicked(picked: PickedSample | null, sample: ListedSample) {
  return picked?.kind === 'validation' && picked.index === sample.index;
}

function Correctness({ sample }: { sample: ListedSample }) {
  if (sample.correct === undefined) {
    return null;
  }
  return sample.correct ? (
    <IconCheck size={12} color="var(--mantine-color-teal-6)" aria-label="right" />
  ) : (
    <IconX size={12} color="var(--mantine-color-red-6)" aria-label="wrong" />
  );
}

/** Validation images (the model's guess under each), a drawing pad and an upload. */
export function ImageSamplePicker({
  samples,
  loading,
  error,
  picked,
  onPick,
  format,
}: PickerProps & { format: ImageFormat | null }) {
  const [drawing, setDrawing] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const upload = async (file: File | null) => {
    if (!file || !format) {
      return;
    }
    setUploadError(null);
    try {
      const image = await loadImage(file);
      const pixels = canvasPixels(image, format);
      if (!pixels) {
        throw new Error('This browser cannot read images.');
      }
      onPick({ kind: 'input', input: pixels, scale: 'raw', label: file.name });
    } catch (exc) {
      setUploadError((exc as Error).message);
    }
  };

  return (
    <Stack gap="xs">
      <Group justify="space-between" gap="xs">
        <Text size="sm" fw={600}>
          Pick an image
        </Text>
        {format && (
          <Group gap={6}>
            <Button
              size="compact-xs"
              variant="light"
              leftSection={<IconBrush size={14} />}
              onClick={() => setDrawing((open) => !open)}
            >
              Draw
            </Button>
            <FileButton onChange={(file) => void upload(file)} accept="image/*">
              {(props) => (
                <Button
                  size="compact-xs"
                  variant="light"
                  leftSection={<IconUpload size={14} />}
                  {...props}
                >
                  Upload
                </Button>
              )}
            </FileButton>
          </Group>
        )}
      </Group>
      {uploadError && (
        <Alert color="red" variant="light">
          {uploadError}
        </Alert>
      )}
      {drawing && format && (
        <DrawPad
          format={format}
          onUse={(pixels) => {
            onPick({ kind: 'input', input: pixels, scale: 'raw', label: 'your drawing' });
          }}
        />
      )}
      {loading && <Loader size="sm" />}
      {error && (
        <Text size="sm" c="dimmed">
          {error}
        </Text>
      )}
      {samples && (
        <>
          <SimpleGrid cols={{ base: 4, xs: 6, sm: 8 }} spacing={6} verticalSpacing={6}>
            {samples.samples.map((sample) => (
              <Tooltip
                key={sample.index}
                label={`True: ${sample.label} · model: ${sample.predicted} (${formatValue(
                  (sample.probability ?? 0) * 100,
                  2
                )} %)`}
              >
                <UnstyledButton
                  className={classes.thumb}
                  data-active={isPicked(picked, sample) || undefined}
                  data-wrong={sample.correct === false || undefined}
                  onClick={() =>
                    onPick({ kind: 'validation', index: sample.index, label: sample.label })
                  }
                  aria-label={`Sample ${sample.index}: ${sample.label}, predicted ${sample.predicted}`}
                >
                  <img src={sample.png} alt="" className={classes.pixelated} />
                  <Group gap={2} wrap="nowrap" justify="center">
                    <Correctness sample={sample} />
                    <Text size="10px" lh={1.2} truncate>
                      {sample.predicted}
                    </Text>
                  </Group>
                </UnstyledButton>
              </Tooltip>
            ))}
          </SimpleGrid>
          <Text size="xs" c="dimmed">
            Validation images the model did not train on, with its guess (✓ right, ✗ wrong). The
            last ones are mistakes: interesting to explain.
          </Text>
        </>
      )}
    </Stack>
  );
}

/** Draw a digit or shape with the mouse or a finger (white on black, like MNIST). */
export function DrawPad({
  format,
  onUse,
}: {
  format: ImageFormat;
  onUse: (pixels: number[][][]) => void;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const last = useRef<{ x: number; y: number } | null>(null);
  const size = 196;

  const clear = () => {
    const context = canvas.current?.getContext('2d');
    if (context) {
      context.fillStyle = '#000';
      context.fillRect(0, 0, size, size);
    }
  };
  useEffect(clear, []);

  const point = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const box = event.currentTarget.getBoundingClientRect();
    return {
      x: ((event.clientX - box.left) / box.width) * size,
      y: ((event.clientY - box.top) / box.height) * size,
    };
  };
  const stroke = (to: { x: number; y: number }) => {
    const context = canvas.current?.getContext('2d');
    if (!context) {
      return;
    }
    const from = last.current ?? to;
    context.strokeStyle = '#fff';
    context.lineWidth = size / 11;
    context.lineCap = 'round';
    context.lineJoin = 'round';
    context.beginPath();
    context.moveTo(from.x, from.y);
    context.lineTo(to.x, to.y);
    context.stroke();
    last.current = to;
  };

  return (
    <Paper withBorder p="xs">
      <Group align="flex-end" gap="sm">
        <canvas
          ref={canvas}
          width={size}
          height={size}
          className={classes.drawPad}
          aria-label="Drawing pad"
          onPointerDown={(event) => {
            event.currentTarget.setPointerCapture(event.pointerId);
            last.current = null;
            stroke(point(event));
          }}
          onPointerMove={(event) => {
            if (event.buttons > 0) {
              stroke(point(event));
            }
          }}
          onPointerUp={() => {
            last.current = null;
          }}
        />
        <Stack gap={6}>
          <Text size="xs" c="dimmed" maw={180}>
            Draw large and centred, like the digits of MNIST. It is scaled down to {format.width}×
            {format.height} pixels.
          </Text>
          <Group gap={6}>
            <Button
              size="compact-sm"
              onClick={() => {
                const pixels = canvas.current ? canvasPixels(canvas.current, format) : null;
                if (pixels) {
                  onUse(pixels);
                }
              }}
            >
              Explain it
            </Button>
            <Button
              size="compact-sm"
              variant="default"
              leftSection={<IconEraser size={14} />}
              onClick={clear}
            >
              Clear
            </Button>
          </Group>
        </Stack>
      </Group>
    </Paper>
  );
}

/** Validation rows (click one), or a row of your own values. */
export function TabularSamplePicker({ samples, loading, error, picked, onPick }: PickerProps) {
  const [editing, setEditing] = useState(false);
  const columns = samples?.columns ?? [];
  const selected = samples?.samples.find((sample) => isPicked(picked, sample));
  const [values, setValues] = useState<(number | string)[]>([]);

  const startEditing = () => {
    const source =
      picked?.kind === 'input' && Array.isArray(picked.input)
        ? (picked.input as number[])
        : (selected?.values ?? samples?.samples[0]?.values ?? []);
    setValues(columns.map((_, i) => source[i] ?? 0));
    setEditing(true);
  };

  return (
    <Stack gap="xs">
      <Group justify="space-between" gap="xs">
        <Text size="sm" fw={600}>
          Pick a row
        </Text>
        {samples && (
          <Button
            size="compact-xs"
            variant="light"
            leftSection={<IconPencil size={14} />}
            onClick={startEditing}
          >
            Your own values
          </Button>
        )}
      </Group>
      {editing && (
        <Paper withBorder p="xs">
          <SimpleGrid cols={{ base: 2, xs: 3, sm: 4 }} spacing="xs">
            {columns.map((column, i) => (
              <NumberInput
                key={column}
                label={column}
                size="xs"
                value={values[i]}
                onChange={(value) => setValues((old) => old.map((v, j) => (j === i ? value : v)))}
              />
            ))}
          </SimpleGrid>
          <Group gap={6} mt="xs">
            <Button
              size="compact-sm"
              onClick={() => {
                onPick({
                  kind: 'input',
                  input: values.map((v) => (typeof v === 'number' ? v : Number(v) || 0)),
                  label: 'your row',
                });
                setEditing(false);
              }}
            >
              Explain this row
            </Button>
            <Button size="compact-sm" variant="default" onClick={() => setEditing(false)}>
              Cancel
            </Button>
          </Group>
        </Paper>
      )}
      {loading && <Loader size="sm" />}
      {error && (
        <Text size="sm" c="dimmed">
          {error}
        </Text>
      )}
      {samples && (
        <ScrollArea.Autosize mah={260} type="auto">
          <Table
            aria-label="Validation rows"
            striped
            highlightOnHover
            withTableBorder
            fz="xs"
            miw={columns.length * 70 + 160}
          >
            <Table.Thead>
              <Table.Tr>
                <Table.Th>#</Table.Th>
                {columns.map((column) => (
                  <Table.Th key={column}>{column}</Table.Th>
                ))}
                <Table.Th>{samples.target ?? 'target'}</Table.Th>
                <Table.Th>model</Table.Th>
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {samples.samples.map((sample) => (
                <Table.Tr
                  key={sample.index}
                  className={classes.row}
                  data-active={isPicked(picked, sample) || undefined}
                  onClick={() => onPick({ kind: 'validation', index: sample.index })}
                  tabIndex={0}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' || event.key === ' ') {
                      onPick({ kind: 'validation', index: sample.index });
                    }
                  }}
                  aria-selected={isPicked(picked, sample)}
                >
                  <Table.Td>{sample.index}</Table.Td>
                  {(sample.values ?? []).map((value, i) => (
                    <Table.Td key={columns[i] ?? i}>{formatValue(value)}</Table.Td>
                  ))}
                  <Table.Td>{sample.label ?? formatValue(sample.value)}</Table.Td>
                  <Table.Td>
                    {sample.predicted !== undefined ? (
                      <Group gap={4} wrap="nowrap">
                        <Correctness sample={sample} />
                        {sample.predicted}
                      </Group>
                    ) : (
                      formatValue(sample.predicted_value)
                    )}
                  </Table.Td>
                </Table.Tr>
              ))}
            </Table.Tbody>
          </Table>
        </ScrollArea.Autosize>
      )}
      {picked?.kind === 'input' && (
        <Badge variant="light" color="grape">
          Explaining {picked.label}
        </Badge>
      )}
    </Stack>
  );
}
