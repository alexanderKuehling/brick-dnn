/**
 * Making the network fit the training data: the fixes the server suggests (see
 * `suggest_fixes` in backend/brickdnn/training/compat.py) and a first network for an empty
 * canvas.
 */
import type { CompatibilityFix, GraphDocument, LayerSpec, Params } from '@/api/types';
import { defaultParams, edgeId, LAYER_SPACING_X, newId } from '@/builder/graph';
import { currentLocale, i18n } from '@/i18n';
import type { EffectiveSource } from './settings';

/** The parameters a fix may change without asking: the Input's data and the last layer's size. */
const AUTOMATIC: Record<string, string[]> = {
  Input: ['shape', 'dtype'],
  Linear: ['out_features'],
  // Segmentation: one output channel per class.
  Conv2d: ['out_channels'],
  ConvTranspose2d: ['out_channels'],
  // Texts: the Embedding knows every word of the vocabulary.
  Embedding: ['num_embeddings'],
};

/**
 * True if the fixes only change the Input's shape or data type, the number of outputs and, for
 * texts, the vocabulary size of the Embedding.
 */
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

/**
 * What a flat Input shape counts: pixel values of images, tokens of texts, else the feature
 * columns of a table.
 */
export type InputUnit = 'images' | 'tokens' | 'features';

/** "4 features", "3,072 values" (flat images), "24 tokens" (texts), "1×28×28". */
export function describeShape(shape: number[], unit: InputUnit): string {
  if (shape.length === 1) {
    const n = shape[0].toLocaleString(currentLocale());
    const count = shape[0];
    return unit === 'images'
      ? i18n.t('ui.training.fit.values', { count, n })
      : unit === 'tokens'
        ? i18n.t('ui.training.fit.tokens', { count, n })
        : i18n.t('ui.training.fit.features', { count, n });
  }
  return shape.join('×');
}

/**
 * What the fixes change, for a notification: "Input set to 4 features, last layer to
 * 3 outputs". `unit`: what the values of a flat Input are (see InputUnit).
 */
export function describeFixes(fixes: CompatibilityFix[], unit: InputUnit): string {
  const input = fixes.find((fix) => fix.type === 'Input')?.params;
  const channels = fixes.find((fix) => typeof fix.params.out_channels === 'number')?.params
    .out_channels as number | undefined;
  const outputs = fixes.find((fix) => typeof fix.params.out_features === 'number')?.params
    .out_features as number | undefined;
  const inputText =
    input &&
    i18n.t('ui.training.fit.inputSetTo', {
      filter: [
        Array.isArray(input.shape) ? describeShape(input.shape as number[], unit) : null,
        typeof input.dtype === 'string'
          ? i18n.t('ui.training.fit.dataType', { dtype: input.dtype })
          : null,
      ]
        .filter(Boolean)
        .join(', '),
    });
  const outputText =
    channels !== undefined
      ? i18n.t('ui.training.fit.outputChannels', {
          count: channels,
          n: channels.toLocaleString(currentLocale()),
        })
      : outputs === undefined
        ? null
        : i18n.t('ui.training.fit.outputs', {
            count: outputs,
            n: outputs.toLocaleString(currentLocale()),
          });
  const words = fixes.find((fix) => typeof fix.params.num_embeddings === 'number')?.params
    .num_embeddings as number | undefined;
  const text =
    inputText && outputText
      ? i18n.t('ui.training.fit.lastLayerTo', { inputText, outputText })
      : inputText ||
        (outputText
          ? i18n.t('ui.training.fit.lastLayerSetTo', { outputText })
          : words === undefined
            ? i18n.t('ui.training.fit.networkAdapted')
            : null);
  const vocabulary =
    words === undefined
      ? null
      : i18n.t('ui.training.fit.vocabularySetTo', {
          words: words.toLocaleString(currentLocale()),
        });
  return [text, vocabulary].filter(Boolean).join(', ');
}

/** True if `source` holds texts as token ids (the built-in Sentiment, Key lookup, Topics). */
export function isTextSource(source: EffectiveSource | null): boolean {
  return source?.ref?.kind === 'builtin' && source.builtin?.input_kind === 'tokens';
}

/** True if `source` holds images (built-in datasets, image folders, images with masks). */
export function isImageSource(source: EffectiveSource): boolean {
  return (
    (source.ref?.kind === 'builtin' && !isTextSource(source)) ||
    source.upload?.kind === 'images' ||
    source.upload?.kind === 'segmentation'
  );
}

/** What the values of a flat Input are for `source`. */
export function inputUnit(source: EffectiveSource): InputUnit {
  return isTextSource(source) ? 'tokens' : isImageSource(source) ? 'images' : 'features';
}

/**
 * True if `source` has a class for every pixel: Shapes, Cells or an upload of images with masks.
 */
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

/**
 * A first text network: token ids → Embedding → a Transformer encoder with learned positions →
 * the mean over the tokens → one output per class.
 */
function textLayers(length: number, words: number, classes: number): [string, Params][] {
  return [
    ['Input', { shape: [length], dtype: 'long' }],
    ['Embedding', { num_embeddings: words, embedding_dim: 64 }],
    ['PositionalEmbedding', { max_len: length }],
    ['TransformerEncoder', { nhead: 4, dim_feedforward: 128, num_layers: 2 }],
    ['SequencePool', { mode: 'mean' }],
    ['Linear', { out_features: classes }],
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
    if (source.builtin.input_kind === 'tokens') {
      return textLayers(shape[0], source.builtin.vocab_size ?? 2, classes);
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
 * that end in one channel per class; for texts a small Transformer. Null if the data or a
 * layer type is not known.
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
