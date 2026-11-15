import type { Analysis, NodeAnalysis, Params, ParamValue } from '@/api/types';
import { currentLocale, i18n } from '@/i18n';
import { formatNumber } from '@/i18n/format';
import { compactKernel, isBoosting, isEnsemble } from './kinds';

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

/**
 * Labels of select options whose values are code (the backend sends the raw values): the
 * activation functions by name, the others in the current language (`builder.options`):
 * kernel bricks (how the Gaussian process treats their hyperparameters), the own kernel of
 * SVR and SVC, class weights, early stopping of gradient boosting, voting.
 */
export function optionLabel(param: string, value: string): string {
  if (param === 'function') {
    return ACTIVATION_NAMES[value] ?? value;
  }
  const key = `builder.options.${param}.${value}`;
  return i18n.exists(key) ? String(i18n.t(key as 'builder.options.kernel.rbf')) : value;
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
  const one = (value: number) =>
    formatNumber(value, { minimumFractionDigits: 1, maximumFractionDigits: 1 });
  if (count >= 1e9) {
    return i18n.t('builder.format.billions', { value: one(count / 1e9) });
  }
  if (count >= 1e6) {
    return i18n.t('builder.format.millions', { value: one(count / 1e6) });
  }
  if (count >= 1e4) {
    return i18n.t('builder.format.thousands', { value: one(count / 1e3) });
  }
  return count.toLocaleString(currentLocale());
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
  return ` · ${i18n.t('builder.format.stride', { stride: new Set(stride).size === 1 ? stride[0] : stride.join('×') })}`;
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

function hyperparameterNote(value: string): string {
  if (value === 'fixed') {
    return ` · ${i18n.t('builder.format.fixed')}`;
  }
  return value === 'custom bounds' ? ` · ${i18n.t('builder.format.bounded')}` : '';
}

/** The SVM's own kernel: "RBF kernel", "poly (degree 3)". */
function svmKernel(p: Params): string {
  if (p.kernel === 'poly') {
    return i18n.t('builder.format.polyKernel', { degree: num(p.degree) });
  }
  return i18n.t('builder.format.kernel', {
    kernel: optionLabel('kernel', String(p.kernel ?? 'rbf')),
  });
}

/** The models of an ensemble by name, in the order of their connections ("gb, svc, gp"). */
function memberNames(context: SummaryContext): string | null {
  const members = context.info?.members;
  return members?.length ? members.map((member) => member.name).join(', ') : null;
}

/** "weights 2:1:1", or null without weights. */
function weightsText(weights: ParamValue): string | null {
  return Array.isArray(weights) && weights.length > 0
    ? i18n.t('builder.format.weights', { weights: weights.join(':') })
    : null;
}

/** What the analysis adds to a node's summary (kernels of Gaussian processes, models of ensembles). */
export interface SummaryContext {
  info?: NodeAnalysis | null;
  /** A Gaussian process gets a kernel from kernel bricks (otherwise it uses its default). */
  kernelConnected?: boolean;
}

/** One-line description of a layer's configuration, shown on the canvas node. */
export function summarize(type: string, p: Params, context: SummaryContext = {}): string {
  const kernel = KERNEL_FORMULAS[type];
  if (kernel) {
    return `${kernel(p)}${hyperparameterNote(String(p.hyperparameters))}`;
  }
  switch (type) {
    case 'Input':
      return `${formatShape(p.shape as number[])}${p.dtype === 'long' ? ` · ${i18n.t('builder.format.tokenIds')}` : ''}`;
    case 'Linear':
      return `${num(p.in_features)} → ${num(p.out_features)}${p.bias === false ? ` · ${i18n.t('builder.format.noBias')}` : ''}`;
    case 'Conv1d':
    case 'Conv2d':
    case 'ConvTranspose2d':
      return `${i18n.t('builder.format.filters', { count: Number(p.out_channels), value: num(p.out_channels) })} · ${size(p.kernel_size)}${strideText(p.stride)}`;
    case 'MaxPool1d':
    case 'MaxPool2d':
    case 'AvgPool2d':
      return `${size(p.kernel_size)}${strideText(p.stride)}`;
    case 'AdaptiveAvgPool2d':
      return `→ ${size(p.output_size)}`;
    case 'Flatten':
      return i18n.t('builder.format.toVector');
    case 'Reshape':
      return `→ ${formatShape(p.shape as number[])}`;
    case 'Dropout':
      return `p = ${num(p.p)}`;
    case 'Activation':
      return optionLabel('function', String(p.function));
    case 'BatchNorm1d':
    case 'BatchNorm2d':
      return i18n.t('builder.format.channels', { value: num(p.num_features) });
    case 'LayerNorm':
      return i18n.t('builder.format.features', { value: num(p.normalized_shape) });
    case 'Embedding':
      return i18n.t('builder.format.tokens', {
        tokens: num(p.num_embeddings),
        dim: num(p.embedding_dim),
      });
    case 'PositionalEmbedding':
      return i18n.t('builder.format.maxLength', { value: num(p.max_len) });
    case 'RNN':
    case 'LSTM':
    case 'GRU': {
      const parts = [i18n.t('builder.format.hidden', { value: num(p.hidden_size) })];
      if (Number(p.num_layers) > 1) {
        parts.push(i18n.t('builder.format.layers', { value: String(p.num_layers) }));
      }
      if (p.bidirectional) {
        parts.push(i18n.t('builder.format.bi'));
      }
      if (p.return_sequences) {
        parts.push(i18n.t('builder.format.sequences'));
      }
      return parts.join(' · ');
    }
    case 'TransformerEncoder':
      return i18n.t('builder.format.transformer', {
        layers: num(p.num_layers),
        heads: num(p.nhead),
        ff: num(p.dim_feedforward),
      });
    case 'SequencePool':
      return i18n.t('builder.format.overSequence', { mode: String(p.mode) });
    case 'Add':
      return i18n.t('builder.format.elementwiseSum');
    case 'Concat':
      return i18n.t('builder.format.alongDim', { dim: num(p.dim) });
    case 'KernelSum':
    case 'KernelProduct': {
      const code = context.info?.kernel;
      return code
        ? compactKernel(code)
        : type === 'KernelSum'
          ? i18n.t('builder.format.sumOfKernels')
          : i18n.t('builder.format.productOfKernels');
    }
    case 'GaussianProcessRegressor':
    case 'GaussianProcessClassifier': {
      const code = context.info?.kernel;
      if (!context.kernelConnected) {
        return i18n.t('builder.format.defaultKernel');
      }
      return code ? compactKernel(code) : i18n.t('builder.format.kernelOutput');
    }
    case 'SVR':
    case 'SVC': {
      const parts = [svmKernel(p), `C = ${num(p.C)}`];
      if (type === 'SVR') {
        parts.push(`ε = ${num(p.epsilon)}`);
      } else if (p.class_weight === 'balanced') {
        parts.push(i18n.t('builder.format.balanced'));
      }
      return parts.join(' · ');
    }
    case 'HistGradientBoostingRegressor':
    case 'HistGradientBoostingClassifier': {
      const parts = [
        i18n.t('builder.format.iterations', { value: num(p.max_iter) }),
        i18n.t('builder.format.rate', { value: num(p.learning_rate) }),
        i18n.t('builder.format.leaves', { value: num(p.max_leaf_nodes) }),
      ];
      if (typeof p.max_depth === 'number') {
        parts.push(i18n.t('builder.format.depth', { value: p.max_depth }));
      }
      if (p.class_weight === 'balanced') {
        parts.push(i18n.t('builder.format.balanced'));
      }
      return parts.join(' · ');
    }
    case 'VotingClassifier':
    case 'VotingRegressor': {
      const how =
        type === 'VotingClassifier'
          ? i18n.t('builder.format.voting', { voting: String(p.voting ?? 'soft') })
          : i18n.t('builder.format.average');
      const parts = [how, weightsText(p.weights), memberNames(context)];
      return parts.filter(Boolean).join(' · ');
    }
    case 'StackingClassifier':
    case 'StackingRegressor': {
      const final =
        type === 'StackingClassifier'
          ? i18n.t('builder.format.logistic')
          : i18n.t('builder.format.ridge');
      const parts = [
        i18n.t('builder.format.onFolds', { final, folds: num(p.cv) }),
        p.passthrough ? i18n.t('builder.format.plusFeatures') : null,
        memberNames(context),
      ];
      return parts.filter(Boolean).join(' · ');
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
    return i18n.t('builder.format.kernelOutput');
  }
  if (info && !info.output_shape && info.task === 'classification') {
    return i18n.t('builder.format.classesFromData');
  }
  return formatShape(info?.output_shape);
}

/** "1 hyperparameter", "3 hyperparameters". */
export function hyperparameters(count: number): string {
  return i18n.t('builder.format.hyperparameters', {
    count,
    value: count.toLocaleString(currentLocale()),
  });
}

/**
 * The size of a valid network for the toolbar: "1.2k params"; for a Gaussian process the
 * kernel hyperparameters the fit optimises; an SVM or gradient boosting has nothing to count
 * before the fit, an ensemble its models.
 */
export function networkSize(analysis: Analysis): string {
  if (analysis.family !== 'classical') {
    return i18n.t('builder.format.params', { value: formatCount(analysis.total_params) });
  }
  if (isEnsemble(analysis.classical?.type)) {
    const count = analysis.classical?.members?.length ?? 0;
    return i18n.t('builder.format.ensembleOf', { count });
  }
  if (analysis.total_params === null) {
    const what = isBoosting(analysis.classical?.type)
      ? i18n.t('builder.format.gradientBoosting')
      : 'SVM';
    return analysis.classical?.task === 'classification'
      ? i18n.t('builder.format.classifier', { what })
      : i18n.t('builder.format.regression', { what });
  }
  return hyperparameters(analysis.total_params);
}
