import { useEffect, useRef, useState } from 'react';
import { IconEraser, IconPhotoUp, IconWand } from '@tabler/icons-react';
import { Box, Button, FileButton, Group, SegmentedControl, Stack, Text } from '@mantine/core';
import type { Prediction, PredictionForm, PredictionResult } from '@/api';
import { i18n, useTranslation } from '@/i18n';
import { imagePixels } from './loadImage';
import { fitDrawing, imageInput, rasterize, type Stroke } from './pixels';
import { Messages, ResultCard } from './PredictionResults';
import { isProblem, predictAll, type PredictionProblem } from './requests';

// The drawing pad is drawn at this resolution (4 pixels per pixel of a 28 × 28 image).
const PAD = 112;
const LINE_WIDTH = 0.075;

/** Draws `image` (channels × height × width, 0–255) into `canvas`, if the browser can. */
function paint(canvas: HTMLCanvasElement | null, image: number[][][]) {
  const context = canvas?.getContext('2d');
  if (!canvas || !context) {
    return;
  }
  const height = image[0].length;
  const width = image[0][0].length;
  canvas.width = width;
  canvas.height = height;
  const data = context.createImageData(width, height);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const i = (y * width + x) * 4;
      const [r, g, b] =
        image.length === 3 ? [image[0], image[1], image[2]] : [image[0], image[0], image[0]];
      data.data[i] = r[y][x];
      data.data[i + 1] = g[y][x];
      data.data[i + 2] = b[y][x];
      data.data[i + 3] = 255;
    }
  }
  context.putImageData(data, 0, 0);
}

/** A square pad to draw on with the mouse, a pen or a finger: white lines on black. */
function DrawPad({
  strokes,
  onChange,
}: {
  strokes: Stroke[];
  /** Takes an update (pointer events can come faster than renders). */
  onChange: (update: (strokes: Stroke[]) => Stroke[]) => void;
}) {
  useTranslation();
  const canvas = useRef<HTMLCanvasElement>(null);
  const drawing = useRef(false);

  useEffect(() => {
    const element = canvas.current;
    const context = element?.getContext('2d');
    if (!element || !context) {
      return;
    }
    const size = element.width;
    context.fillStyle = '#000';
    context.fillRect(0, 0, size, size);
    context.strokeStyle = '#fff';
    context.fillStyle = '#fff';
    context.lineWidth = LINE_WIDTH * size;
    context.lineCap = 'round';
    context.lineJoin = 'round';
    for (const stroke of strokes) {
      context.beginPath();
      stroke.forEach((p, i) =>
        i === 0 ? context.moveTo(p.x * size, p.y * size) : context.lineTo(p.x * size, p.y * size)
      );
      if (stroke.length === 1) {
        context.arc(
          stroke[0].x * size,
          stroke[0].y * size,
          (LINE_WIDTH * size) / 2,
          0,
          2 * Math.PI
        );
        context.fill();
      } else {
        context.stroke();
      }
    }
  }, [strokes]);

  const point = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const width = rect.width || 1;
    const height = rect.height || 1;
    return {
      x: Math.min(1, Math.max(0, (event.clientX - rect.left) / width)),
      y: Math.min(1, Math.max(0, (event.clientY - rect.top) / height)),
    };
  };

  return (
    <canvas
      ref={canvas}
      width={280}
      height={280}
      aria-label={i18n.t('ui.training.ImagePredict.drawingPad')}
      style={{
        width: '100%',
        maxWidth: 280,
        aspectRatio: '1',
        background: '#000',
        borderRadius: 4,
        touchAction: 'none',
        cursor: 'crosshair',
      }}
      onPointerDown={(event) => {
        drawing.current = true;
        event.currentTarget.setPointerCapture?.(event.pointerId);
        const first = point(event);
        onChange((current) => [...current, [first]]);
      }}
      onPointerMove={(event) => {
        if (!drawing.current) {
          return;
        }
        const next = point(event);
        onChange((current) =>
          current.length === 0
            ? current
            : [...current.slice(0, -1), [...current[current.length - 1], next]]
        );
      }}
      onPointerUp={() => {
        drawing.current = false;
      }}
      onPointerLeave={() => {
        drawing.current = false;
      }}
    />
  );
}

/** Predict the class of an uploaded or drawn image. */
export function ImagePredict({ runId, form }: { runId: string; form: PredictionForm }) {
  useTranslation();
  const image = form.image!;
  // Drawing makes sense for small grayscale images like the MNIST digits.
  const drawable = image.channels === 1 && image.width <= 64 && image.height <= 64;
  const [mode, setMode] = useState<'upload' | 'draw'>('upload');
  const [strokes, setStrokes] = useState<Stroke[]>([]);
  const [input, setInput] = useState<number[][][] | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState<PredictionProblem | null>(null);
  const [result, setResult] = useState<{ prediction: Prediction; result: PredictionResult } | null>(
    null
  );
  const preview = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (input) {
      paint(preview.current, input);
    }
  }, [input]);

  const predict = async (pixels: number[][][]) => {
    setBusy(true);
    setProblem(null);
    setResult(null);
    try {
      const answer = await predictAll(
        runId,
        [pixels],
        { input_scale: 'raw', top_k: Math.min(3, form.max_top_k) },
        1
      );
      setResult({ prediction: answer.predictions[0], result: answer });
    } catch (error) {
      setProblem(isProblem(error) ? error : { message: String(error), errors: [], budget: false });
    } finally {
      setBusy(false);
    }
  };

  const readFile = async (file: File | null) => {
    if (!file) {
      return;
    }
    setProblem(null);
    setResult(null);
    try {
      const rgba = await imagePixels(file, image.width, image.height);
      const pixels = imageInput(rgba, image.width, image.height, image.channels);
      setFileName(file.name);
      setInput(pixels);
      await predict(pixels);
    } catch (error) {
      setProblem({ message: (error as Error).message, errors: [], budget: false });
    }
  };

  const predictDrawing = async () => {
    const digit = fitDrawing(rasterize(strokes, PAD, LINE_WIDTH), PAD, image.width, image.height);
    if (!digit) {
      setProblem({
        message: i18n.t('ui.training.ImagePredict.drawSomethingFirst'),
        errors: [],
        budget: false,
      });
      return;
    }
    const pixels = [digit];
    setInput(pixels);
    await predict(pixels);
  };

  const size = `${image.width}×${image.height}`;
  return (
    <Stack gap="sm">
      {drawable && (
        <SegmentedControl
          size="xs"
          value={mode}
          onChange={(value) => {
            setMode(value as 'upload' | 'draw');
            setResult(null);
            setProblem(null);
            setInput(null);
          }}
          data={[
            { value: 'upload', label: i18n.t('ui.training.ImagePredict.uploadAnImage') },
            { value: 'draw', label: i18n.t('ui.training.ImagePredict.draw') },
          ]}
          aria-label={i18n.t('ui.training.ImagePredict.imageInput')}
        />
      )}
      <Text size="xs" c="dimmed">
        {mode === 'draw'
          ? i18n.t('ui.training.ImagePredict.drawADigitOrShape', { size })
          : i18n.t('ui.training.ImagePredict.anyImageFileItIs', {
              size,
              value:
                image.channels === 1 ? i18n.t('ui.training.ImagePredict.andTurnedGrayscale') : '',
            })}
      </Text>

      {mode === 'draw' ? (
        <Stack gap="xs" align="flex-start">
          <DrawPad strokes={strokes} onChange={setStrokes} />
          <Group gap="xs">
            <Button
              leftSection={<IconWand size={16} />}
              loading={busy}
              onClick={() => void predictDrawing()}
            >
              {i18n.t('ui.training.ImagePredict.predict')}
            </Button>
            <Button
              variant="default"
              leftSection={<IconEraser size={16} />}
              onClick={() => {
                setStrokes([]);
                setResult(null);
                setInput(null);
              }}
            >
              {i18n.t('ui.training.ImagePredict.clear')}
            </Button>
          </Group>
        </Stack>
      ) : (
        <FileButton accept="image/*" onChange={(file) => void readFile(file)}>
          {(props) => (
            <Box
              {...props}
              component="button"
              type="button"
              aria-label={i18n.t('ui.training.ImagePredict.uploadAnImage')}
              onDragOver={(event: React.DragEvent) => {
                event.preventDefault();
                setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={(event: React.DragEvent) => {
                event.preventDefault();
                setDragging(false);
                void readFile(event.dataTransfer.files?.[0] ?? null);
              }}
              style={{
                width: '100%',
                padding: 'var(--mantine-spacing-md)',
                border: `1px dashed var(--mantine-color-${dragging ? 'blue-6' : 'default-border'})`,
                borderRadius: 'var(--mantine-radius-sm)',
                background: dragging ? 'var(--mantine-color-blue-light)' : 'transparent',
                cursor: 'pointer',
                color: 'inherit',
                font: 'inherit',
              }}
            >
              <Group gap="xs" justify="center">
                <IconPhotoUp size={20} />
                <Text size="sm">
                  {busy
                    ? i18n.t('ui.training.ImagePredict.predicting')
                    : fileName
                      ? i18n.t('ui.training.ImagePredict.chooseAnother', { fileName })
                      : i18n.t('ui.training.ImagePredict.dropAnImageHereOr')}
                </Text>
              </Group>
            </Box>
          )}
        </FileButton>
      )}

      {input && (
        <Group gap="xs" align="center">
          <canvas
            ref={preview}
            aria-label={i18n.t('ui.training.ImagePredict.whatTheNetworkSees')}
            style={{
              width: 56,
              height: 56,
              imageRendering: 'pixelated',
              border: '1px solid var(--mantine-color-default-border)',
            }}
          />
          <Text size="xs" c="dimmed">
            {i18n.t('ui.training.ImagePredict.whatTheNetworkGets', { size })}
          </Text>
        </Group>
      )}
      {problem && (
        <Messages
          color={problem.budget ? 'yellow' : 'red'}
          title={
            problem.budget
              ? i18n.t('ui.training.ImagePredict.noComputeLeft')
              : i18n.t('ui.training.ImagePredict.cannotPredict')
          }
          messages={problem.errors.length > 0 ? problem.errors : [problem.message]}
        />
      )}
      {result && <ResultCard prediction={result.prediction} task={result.result.task} />}
    </Stack>
  );
}
