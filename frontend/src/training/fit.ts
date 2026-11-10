/**
 * Making the network fit the training data: the fixes the server suggests (see
 * `suggest_fixes` in backend/brickdnn/training/compat.py) and a first network for an empty
 * canvas.
 */
import type { CompatibilityFix, GraphDocument, LayerSpec, Params } from '@/api/types';
import { defaultParams, edgeId, LAYER_SPACING_X, newId } from '@/builder/graph';
import type { EffectiveSource } from './settings';

/** The parameters a fix may change without asking: the Input's data and the last layer's size. */
const AUTOMATIC: Record<string, string[]> = {
  Input: ['shape', 'dtype'],
  Linear: ['out_features'],
  // Segmentation: one output channel per class.
  Conv2d: ['out_channels'],
  ConvTranspose2d: ['out_channels'],
};

/** True if the fixes only change the Input's shape or data type and the number of outputs. */
export function onlyInputAndOutput(fixes: CompatibilityFix[]): boolean {
  return (
    fixes.length > 0 &&
    fixes.every((fix) => {
      const keys = Object.keys(fix.params);
      const allowed = AUTOMATIC[fix.type] ?? [];
      return keys.length > 0 && keys.every((key) => allowed.includes(key));
    })
  );
}

const plural = (n: number, word: string) =>
  `${n.toLocaleString('en-US')} ${word}${n === 1 ? '' : 's'}`;

/** "4 features", "3,072 values" (flat images), "1×28×28". */
export function describeShape(shape: number[], images: boolean): string {
  if (shape.length === 1) {
    return images ? plural(shape[0], 'value') : plural(shape[0], 'feature');
  }
  return shape.join('×');
}

/**
 * What the fixes change, for a notification: "Input set to 4 features, last layer to
 * 3 outputs". `images`: the data are images (flat Input shapes are pixel values).
 */
export function describeFixes(fixes: CompatibilityFix[], images: boolean): string {
  const input = fixes.find((fix) => fix.type === 'Input')?.params;
  const channels = fixes.find((fix) => typeof fix.params.out_channels === 'number')?.params
    .out_channels as number | undefined;
  const outputs = fixes.find((fix) => typeof fix.params.out_features === 'number')?.params
    .out_features as number | undefined;
  const inputText =
    input &&
    `Input set to ${[
      Array.isArray(input.shape) ? describeShape(input.shape as number[], images) : null,
      typeof input.dtype === 'string' ? `data type ${input.dtype}` : null,
    ]
      .filter(Boolean)
      .join(', ')}`;
  const outputText =
    channels !== undefined
      ? plural(channels, 'output channel')
      : outputs === undefined
        ? null
        : plural(outputs, 'output');
  if (inputText && outputText) {
    return `${inputText}, last layer to ${outputText}`;
  }
  return inputText || (outputText ? `Last layer set to ${outputText}` : 'Network adapted');
}

/** True if `source` holds images (built-in datasets, image folders, images with masks). */
export function isImageSource(source: EffectiveSource): boolean {
  return (
    source.ref?.kind === 'builtin' ||
    source.upload?.kind === 'images' ||
    source.upload?.kind === 'segmentation'
  );
}

/** True if `source` has a class for every pixel: Shapes or an upload of images with masks. */
export function isSegmentationSource(source: EffectiveSource | null): boolean {
  if (source?.ref?.kind === 'builtin') {
    return source.builtin?.task === 'segmentation';
  }
  return source?.ref?.kind === 'upload' && source.upload?.kind === 'segmentation';
}

/**
 * A first segmentation network: two convolutions that keep the image size and a 1×1
 * convolution with one channel per class.
 */
function segmentationLayers(shape: number[], classes: number): [string, Params][] {
  return [
    ['Input', { shape }],
    ['Conv2d', { out_channels: 16 }],
    ['Activation', { function: 'relu' }],
    ['Conv2d', { out_channels: 16 }],
    ['Activation', { function: 'relu' }],
    ['Conv2d', { out_channels: classes, kernel_size: [1, 1] }],
  ];
}

/** The layers of a first network for `source`, from its known shape and classes. */
function starterLayers(source: EffectiveSource): [string, Params][] | null {
  const head = (classes: number): [string, Params][] => [
    ['Linear', { out_features: 64 }],
    ['Activation', { function: 'relu' }],
    ['Linear', { out_features: classes }],
  ];
  if (source.ref?.kind === 'builtin' && source.builtin) {
    const { sample_shape: shape, num_classes: classes } = source.builtin;
    if (source.builtin.task === 'segmentation') {
      return segmentationLayers(shape, classes);
    }
    return [['Input', { shape }], ['Flatten', {}], ...head(classes)];
  }
  const info = source.ref?.kind === 'upload' ? source.upload?.info : null;
  if (!info) {
    return null;
  }
  if (info.kind === 'segmentation') {
    const channels = info.sample_shape?.[0] === 1 ? 1 : 3;
    return segmentationLayers([channels, 64, 64], info.num_classes ?? 2);
  }
  if (info.kind === 'images') {
    return [['Input', { shape: [3, 64, 64] }], ['Flatten', {}], ...head(info.num_classes ?? 2)];
  }
  if (info.kind !== 'csv' || !info.num_features) {
    return null;
  }
  // Same rules as the server: prepared data (Data tab) only keeps class names for
  // classification, text targets are classes, numbers are a regression.
  const prepared = Boolean(info.predefined_split);
  const classes = prepared
    ? (info.class_names?.length ?? 1)
    : info.target_kind === 'text'
      ? (info.num_classes ?? 1)
      : 1;
  return [['Input', { shape: [info.num_features] }], ...head(classes)];
}

/**
 * A small network that fits `source`, for an empty canvas: Input → (Flatten →) Linear → ReLU →
 * Linear with one output per class (1 for a regression); for segmentation data convolutions
 * that end in one channel per class. Null if the data or a layer type is not known.
 */
export function starterNetwork(
  source: EffectiveSource,
  specs: Record<string, LayerSpec>
): GraphDocument | null {
  const layers = starterLayers(source);
  if (!layers || layers.some(([type]) => !specs[type])) {
    return null;
  }
  const ids = layers.map(() => newId());
  return {
    version: 1,
    nodes: layers.map(([type, params], index) => ({
      id: ids[index],
      type,
      params: { ...defaultParams(specs[type]), ...params },
      position: { x: index * LAYER_SPACING_X, y: 0 },
    })),
    edges: ids.slice(1).map((id, index) => ({
      id: edgeId(ids[index], id),
      source: ids[index],
      target: id,
    })),
  };
}

/** Equal parameter values (as JSON: shapes are lists). */
export function sameParams(a: Params, b: Params): boolean {
  const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
  return [...keys].every((key) => JSON.stringify(a[key]) === JSON.stringify(b[key]));
}
