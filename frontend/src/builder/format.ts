import type { Analysis, NodeAnalysis, Params, ParamValue } from '@/api/types';
import { compactKernel } from './kinds';

const ACTIVATION_NAMES: Record<string, string> = {
  relu: 'ReLU',
  leaky_relu: 'LeakyReLU',
  gelu: 'GELU',
  elu: 'ELU',
  selu: 'SELU',
  silu: 'SiLU',
  mish: 'Mish',
  sigmoid: 'Sigmoid',
  tanh: 'Tanh',
  softplus: 'Softplus',
  softmax: 'Softmax',
  log_softmax: 'LogSoftmax',
};

/** Labels of select options whose values are code (the backend sends the raw values). */
const OPTION_LABELS: Record<string, Record<string, string>> = {
  function: ACTIVATION_NAMES,
  // Kernel bricks: how the Gaussian process treats their hyperparameters.
  hyperparameters: { optimize: 'Optimise', 'custom bounds': 'Custom bounds', fixed: 'Fixed' },
  // The own kernel of SVR and SVC.
  kernel: { linear: 'Linear', poly: 'Polynomial', rbf: 'RBF', sigmoid: 'Sigmoid' },
  class_weight: { none: 'None', balanced: 'Balanced' },
};

export function optionLabel(param: string, value: string): string {
  return OPTION_LABELS[param]?.[value] ?? value;
}

export function formatShape(shape: number[] | null | undefined): string {
  if (!shape) {
    return '?';
  }
  return shape.join(' × ');
}

export function formatCount(count: number | null | undefined): string {
  if (count === null || count === undefined) {
    return '–';
  }
  if (count >= 1e9) {
    return `${(count / 1e9).toFixed(1)}B`;
  }
  if (count >= 1e6) {
    return `${(count / 1e6).toFixed(1)}M`;
  }
  if (count >= 1e4) {
    return `${(count / 1e3).toFixed(1)}k`;
  }
  return count.toLocaleString('en-US');
}

/** [3, 3] → "3×3", [5] → "5", null → "auto". */
function size(value: ParamValue): string {
  if (Array.isArray(value)) {
    return value.join('×');
  }
  return value === null || value === undefined ? 'auto' : String(value);
}

function strideText(stride: ParamValue): string {
  if (!Array.isArray(stride) || stride.every((s) => s === 1)) {
    return '';
  }
  return ` · stride ${new Set(stride).size === 1 ? stride[0] : stride.join('×')}`;
}

function num(value: ParamValue): string {
  return value === null || value === undefined || value === '' ? '?' : String(value);
}

/** A float as Python writes it: 1 → "1.0", 0.1 → "0.1", 1e-10 → "1e-10". */
export function pyFloat(value: ParamValue): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return num(value);
  }
  return Number.isInteger(value) && Math.abs(value) < 1e16 ? value.toFixed(1) : String(value);
}

/** Kernel bricks: their scikit-learn name with the starting values of the hyperparameters. */
const KERNEL_FORMULAS: Record<string, (p: Params) => string> = {
  RBF: (p) => `RBF(ℓ=${pyFloat(p.length_scale)})`,
  Matern: (p) => `Matérn(ℓ=${pyFloat(p.length_scale)}, ν=${num(p.nu)})`,
  RationalQuadratic: (p) => `RQ(ℓ=${pyFloat(p.length_scale)}, α=${pyFloat(p.alpha)})`,
  ExpSineSquared: (p) => `Periodic(ℓ=${pyFloat(p.length_scale)}, p=${pyFloat(p.periodicity)})`,
  DotProduct: (p) => `Dot(σ₀=${pyFloat(p.sigma_0)})`,
  WhiteKernel: (p) => `White(noise=${pyFloat(p.noise_level)})`,
  ConstantKernel: (p) => `Constant(${pyFloat(p.constant_value)})`,
};

const HYPERPARAMETER_NOTES: Record<string, string> = {
  fixed: ' · fixed',
  'custom bounds': ' · bounded',
};

/** The SVM's own kernel: "RBF kernel", "poly (degree 3)". */
function svmKernel(p: Params): string {
  if (p.kernel === 'poly') {
    return `Polynomial kernel (degree ${num(p.degree)})`;
  }
  return `${optionLabel('kernel', String(p.kernel ?? 'rbf'))} kernel`;
}

/** What the analysis adds to a node's summary (kernels of Gaussian processes). */
export interface SummaryContext {
  info?: NodeAnalysis | null;
  /** A Gaussian process gets a kernel from kernel bricks (otherwise it uses its default). */
  kernelConnected?: boolean;
}

/** One-line description of a layer's configuration, shown on the canvas node. */
export function summarize(type: string, p: Params, context: SummaryContext = {}): string {
  const kernel = KERNEL_FORMULAS[type];
  if (kernel) {
    return `${kernel(p)}${HYPERPARAMETER_NOTES[String(p.hyperparameters)] ?? ''}`;
  }
  switch (type) {
    case 'Input':
      return `${formatShape(p.shape as number[])}${p.dtype === 'long' ? ' · token ids' : ''}`;
    case 'Linear':
      return `${num(p.in_features)} → ${num(p.out_features)}${p.bias === false ? ' · no bias' : ''}`;
    case 'Conv1d':
    case 'Conv2d':
    case 'ConvTranspose2d':
      return `${num(p.out_channels)} filters · ${size(p.kernel_size)}${strideText(p.stride)}`;
    case 'MaxPool1d':
    case 'MaxPool2d':
    case 'AvgPool2d':
      return `${size(p.kernel_size)}${strideText(p.stride)}`;
    case 'AdaptiveAvgPool2d':
      return `→ ${size(p.output_size)}`;
    case 'Flatten':
      return 'to a vector';
    case 'Reshape':
      return `→ ${formatShape(p.shape as number[])}`;
    case 'Dropout':
      return `p = ${num(p.p)}`;
    case 'Activation':
      return optionLabel('function', String(p.function));
    case 'BatchNorm1d':
    case 'BatchNorm2d':
      return `${num(p.num_features)} channels`;
    case 'LayerNorm':
      return `${num(p.normalized_shape)} features`;
    case 'Embedding':
      return `${num(p.num_embeddings)} tokens → ${num(p.embedding_dim)}`;
    case 'PositionalEmbedding':
      return `max length ${num(p.max_len)}`;
    case 'RNN':
    case 'LSTM':
    case 'GRU': {
      const parts = [`hidden ${num(p.hidden_size)}`];
      if (Number(p.num_layers) > 1) {
        parts.push(`${p.num_layers} layers`);
      }
      if (p.bidirectional) {
        parts.push('bi');
      }
      if (p.return_sequences) {
        parts.push('sequences');
      }
      return parts.join(' · ');
    }
    case 'TransformerEncoder':
      return `${num(p.num_layers)} × ${num(p.nhead)} heads · ff ${num(p.dim_feedforward)}`;
    case 'SequencePool':
      return `${p.mode} over sequence`;
    case 'Add':
      return 'element-wise sum';
    case 'Concat':
      return `along dim ${num(p.dim)}`;
    case 'KernelSum':
    case 'KernelProduct': {
      const code = context.info?.kernel;
      return code
        ? compactKernel(code)
        : type === 'KernelSum'
          ? 'sum of kernels'
          : 'product of kernels';
    }
    case 'GaussianProcessRegressor':
    case 'GaussianProcessClassifier': {
      const code = context.info?.kernel;
      if (!context.kernelConnected) {
        return 'default kernel';
      }
      return code ? compactKernel(code) : 'kernel';
    }
    case 'SVR':
    case 'SVC': {
      const parts = [svmKernel(p), `C = ${num(p.C)}`];
      if (type === 'SVR') {
        parts.push(`ε = ${num(p.epsilon)}`);
      } else if (p.class_weight === 'balanced') {
        parts.push('balanced');
      }
      return parts.join(' · ');
    }
    default:
      return '';
  }
}

/**
 * The output of a node for the canvas and the inspector: its shape, "kernel" for kernel
 * bricks, "classes from data" for classifiers (their classes come from the training data).
 */
export function formatOutput(info: NodeAnalysis | null | undefined, kernel = false): string {
  if (kernel || info?.output_kind === 'kernel') {
    return 'kernel';
  }
  if (info && !info.output_shape && info.task === 'classification') {
    return 'classes from data';
  }
  return formatShape(info?.output_shape);
}

/** "1 hyperparameter", "3 hyperparameters". */
export function hyperparameters(count: number): string {
  return `${count.toLocaleString('en-US')} hyperparameter${count === 1 ? '' : 's'}`;
}

/**
 * The size of a valid network for the toolbar: "1.2k params"; for a Gaussian process the
 * kernel hyperparameters the fit optimises; an SVM has nothing to count before the fit.
 */
export function networkSize(analysis: Analysis): string {
  if (analysis.family !== 'classical') {
    return `${formatCount(analysis.total_params)} params`;
  }
  if (analysis.total_params === null) {
    return analysis.classical?.task === 'classification' ? 'SVM classifier' : 'SVM regression';
  }
  return hyperparameters(analysis.total_params);
}
