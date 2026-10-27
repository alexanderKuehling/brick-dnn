import type { Params, ParamValue } from '@/api/types';

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

export function optionLabel(param: string, value: string): string {
  if (param === 'function') {
    return ACTIVATION_NAMES[value] ?? value;
  }
  return value;
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

/** One-line description of a layer's configuration, shown on the canvas node. */
export function summarize(type: string, p: Params): string {
  switch (type) {
    case 'Input':
      return `${formatShape(p.shape as number[])}${p.dtype === 'long' ? ' · token ids' : ''}`;
    case 'Linear':
      return `${num(p.in_features)} → ${num(p.out_features)}${p.bias === false ? ' · no bias' : ''}`;
    case 'Conv1d':
    case 'Conv2d':
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
    default:
      return '';
  }
}
