/**
 * Explanations of trained models: `/api/training/runs/<id>/explain/` (backend
 * `brickdnn.training.explain`, docs/explain.md). Pictures are PNG data URLs.
 */

export type ExplainMethodId =
  'samples' | 'image' | 'tabular' | 'permutation' | 'boundary' | 'activations' | 'weights';

/** The tabs of the Explain dialog. */
export type ExplainTab = 'prediction' | 'importance' | 'boundary' | 'inside';

export interface ExplainMethod {
  id: ExplainMethodId;
  label: string;
  available: boolean;
  /** Why it does not work for this run. */
  reason: string | null;
}

/** GET: which methods work for a run. */
export interface ExplainMethods {
  family: 'neural' | 'classical';
  /** How the inputs were prepared; null if unknown. */
  kind: 'image' | 'tabular' | 'demo' | null;
  /** Why nothing works (no weights, no PyTorch, training disabled). */
  unavailable: string | null;
  methods: ExplainMethod[];
  demo: boolean;
}

/** POST body. */
export interface ExplainRequest {
  method: ExplainMethodId;
  /** A validation sample (its index in the answer of `samples`). */
  sample?: number;
  /** Or one sample of your own: a row of raw values, or an image (C×H×W or H×W, 0–255). */
  input?: unknown;
  input_scale?: 'raw' | 'unit' | 'normalized';
  /** The class to explain (default: the predicted one). */
  target?: number;
  /** Grad-CAM: the brick to look at. */
  layer?: string;
  /** Activations: only this brick. */
  node?: string;
}

export interface TopClass {
  class: string;
  index: number;
  probability: number | null;
}

export interface Prediction {
  class?: string;
  index?: number;
  probability?: number | null;
  top?: TopClass[];
  value?: number | null;
  /** Gaussian process: predictive standard deviation. */
  std?: number | null;
}

export interface SampleInfo {
  source: 'validation' | 'input' | 'random';
  index: number | null;
  label?: string;
  label_index?: number;
  value?: number | null;
  /** Tabular: the raw feature values. */
  values?: (number | null)[];
}

interface Answer {
  method: ExplainMethodId;
  notes?: string[];
  seconds?: number;
  cached?: boolean;
  run?: { id: string; name: string };
}

export interface ListedSample {
  index: number;
  /** Images: a thumbnail. */
  png?: string;
  /** Tabular: the raw feature values. */
  values?: (number | null)[];
  label?: string;
  label_index?: number;
  predicted?: string;
  predicted_index?: number;
  probability?: number | null;
  correct?: boolean;
  /** Regression. */
  value?: number | null;
  predicted_value?: number | null;
}

export interface SamplesAnswer extends Answer {
  kind: 'image' | 'tabular';
  source: 'validation';
  total: number;
  pool_accuracy?: number;
  columns?: string[];
  target?: string | null;
  samples: ListedSample[];
}

export interface HeatMap {
  png?: string;
  grid?: number[][];
  unavailable?: string;
  note?: string;
}

export interface LayerChoice {
  id: string;
  name: string;
  type: string;
  shape: number[];
}

export interface ImageAnswer extends Answer {
  kind: 'image';
  sample: SampleInfo;
  image: string;
  size: [number, number];
  prediction: Prediction;
  target: { index: number; class: string };
  saliency: HeatMap & { samples: number; noise: number };
  gradcam: HeatMap & { layer?: LayerChoice; layers?: LayerChoice[] };
  occlusion: HeatMap & { patch?: number; stride?: number; largest_drop?: number };
}

export interface FeatureAttribution {
  name: string;
  value: number | null;
  shapley: number | null;
  integrated_gradients?: number | null;
}

export interface TabularAnswer extends Answer {
  kind: 'tabular';
  sample: SampleInfo;
  prediction: Prediction;
  explains: string;
  target: { index: number; class: string } | null;
  base_value: number | null;
  value: number | null;
  baseline: string;
  exact: boolean;
  orders: number | null;
  features: FeatureAttribution[];
}

export interface PermutationAnswer extends Answer {
  kind: 'tabular';
  metric: 'accuracy' | 'r2';
  baseline: number;
  rows: number;
  repeats: number;
  features: { name: string; mean: number | null; std: number | null }[];
}

export interface BoundaryAnswer extends Answer {
  kind: 'tabular';
  features: [string, string];
  x: [number, number];
  y: [number, number];
  resolution: number;
  png: string;
  classes?: string[];
  colors?: string[];
  value_range?: [number, number];
  target?: string | null;
  grid: number[][];
  /** Training points: [x, y, class index or value]. */
  points: [number, number, number][];
  std?: { png: string; range: [number, number] };
}

export interface Histogram {
  edges: number[];
  counts: number[];
  count: number;
  mean?: number | null;
  std?: number | null;
  min?: number | null;
  max?: number | null;
}

export type ActivationView =
  | {
      kind: 'maps';
      png: string;
      channels: number;
      shown: number;
      columns: number;
      tile: [number, number];
      channel_means: (number | null)[];
    }
  | { kind: 'matrix' | 'image'; png: string; rows: number; cols: number; shown: [number, number] }
  | { kind: 'vector'; values: (number | null)[]; total: number }
  | { kind: 'scalar'; value: number | null };

export interface NodeActivation {
  id: string;
  type: string;
  name: string;
  shape: number[];
  stats: {
    mean: number | null;
    std: number | null;
    min: number | null;
    max: number | null;
    zeros: number | null;
  };
  histogram: Histogram;
  view: ActivationView;
}

export interface ActivationsAnswer extends Answer {
  kind: 'image' | 'tabular' | 'demo';
  sample: SampleInfo;
  image?: string;
  prediction?: Prediction;
  nodes: NodeActivation[];
}

export interface WeightsAnswer extends Answer {
  filters: {
    node: string;
    name: string;
    shape: [number, number, number, number];
    count: number;
    columns: number;
    rgb: boolean;
    png: string;
    note: string | null;
  } | null;
  layers: {
    id: string;
    type: string;
    name: string;
    params: { name: string; shape: number[]; histogram: Histogram; norm: number | null }[];
  }[];
}

/** The sample everything is explained for. */
export type PickedSample =
  | { kind: 'validation'; index: number; label?: string }
  | { kind: 'input'; input: unknown; scale?: 'raw' | 'unit'; label: string }
  | { kind: 'random' };
