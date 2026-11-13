/**
 * Learning aids that need no training: how a brick's parameters are counted, how far a
 * convolution "sees" (its receptive field) and the steps of the data flow. They only use
 * the analysis (shapes and filled-in settings).
 */
import type { Analysis, NodeAnalysis, Params } from '@/api/types';

export interface Link {
  source: string;
  target: string;
}

const n = (value: number) => value.toLocaleString('en-US');

function ints(value: unknown): number[] {
  if (Array.isArray(value)) {
    return value.map(Number);
  }
  return typeof value === 'number' ? [value] : [];
}

function num(params: Params, name: string): number {
  return Number(params[name]);
}

const RECURRENT_GATES: Record<string, number> = { RNN: 1, GRU: 3, LSTM: 4 };

/**
 * How the parameters of a brick add up, e.g. "3×3×32×64 + 64 = 18,496". Null when there is
 * no simple formula, or when it would not give the count of the analysis.
 */
export function paramFormula(
  info: NodeAnalysis | undefined
): { formula: string; parts: string } | null {
  const params = info?.params;
  const total = info?.num_params;
  if (!info || !params || total === null || total === undefined || total === 0) {
    return null;
  }
  const bias = params.bias !== false;
  let formula: string | null = null;
  let value = 0;
  let parts = '';
  switch (info.type) {
    case 'Linear': {
      const [i, o] = [num(params, 'in_features'), num(params, 'out_features')];
      value = i * o + (bias ? o : 0);
      formula = `${n(i)}×${n(o)}${bias ? ` + ${n(o)}` : ''}`;
      parts = `${n(i)} inputs × ${n(o)} outputs weights${bias ? `, ${n(o)} biases` : ''}`;
      break;
    }
    case 'Conv1d':
    case 'Conv2d':
    case 'ConvTranspose2d': {
      const kernel = ints(params.kernel_size);
      const [i, o] = [num(params, 'in_channels'), num(params, 'out_channels')];
      value = kernel.reduce((a, b) => a * b, 1) * i * o + (bias ? o : 0);
      formula = `${kernel.join('×')}×${n(i)}×${n(o)}${bias ? ` + ${n(o)}` : ''}`;
      parts = `a ${kernel.join('×')} kernel for each of ${n(i)} input and ${n(o)} output channels${
        bias ? `, ${n(o)} biases` : ''
      }`;
      break;
    }
    case 'BatchNorm1d':
    case 'BatchNorm2d': {
      const f = num(params, 'num_features');
      value = 2 * f;
      formula = `2×${n(f)}`;
      parts = `a scale γ and a shift β per channel (the running mean and variance are not trained)`;
      break;
    }
    case 'LayerNorm': {
      const f = num(params, 'normalized_shape');
      value = 2 * f;
      formula = `2×${n(f)}`;
      parts = 'a scale γ and a shift β per feature';
      break;
    }
    case 'Embedding': {
      const [v, d] = [num(params, 'num_embeddings'), num(params, 'embedding_dim')];
      value = v * d;
      formula = `${n(v)}×${n(d)}`;
      parts = `a vector of ${n(d)} numbers for each of ${n(v)} tokens`;
      break;
    }
    case 'PositionalEmbedding': {
      const [l, d] = [num(params, 'max_len'), num(params, 'embedding_dim')];
      value = l * d;
      formula = `${n(l)}×${n(d)}`;
      parts = `a vector of ${n(d)} numbers for each of ${n(l)} positions`;
      break;
    }
    case 'RNN':
    case 'GRU':
    case 'LSTM': {
      if (num(params, 'num_layers') !== 1) {
        return null;
      }
      const gates = RECURRENT_GATES[info.type];
      const [i, h] = [num(params, 'input_size'), num(params, 'hidden_size')];
      const directions = params.bidirectional ? 2 : 1;
      value = directions * (gates * h * (i + h) + 2 * gates * h);
      formula = `${directions > 1 ? '2×(' : ''}${gates}×${n(h)}×(${n(i)}+${n(h)}) + 2×${gates}×${n(h)}${
        directions > 1 ? ')' : ''
      }`;
      parts = `${gates} gate${gates > 1 ? 's' : ''}, each with weights for the input (${n(i)}) and the previous hidden state (${n(h)}) and two biases${
        directions > 1 ? ', in both directions' : ''
      }`;
      break;
    }
    default:
      return null;
  }
  if (value !== total) {
    return null;
  }
  return { formula: `${formula} = ${n(total)}`, parts };
}

// --------------------------------------------------------------------------- receptive field

export interface ReceptiveField {
  /** Input pixels (or steps) per spatial dimension that one output value depends on. */
  size: number[];
  /** Input pixels between neighbouring output values. */
  jump: number[];
  /** The spatial size of the network's input. */
  input: number[];
  /** True if the value is an estimate (transposed convolutions, adaptive pooling). */
  approximate: boolean;
}

const ELEMENTWISE = new Set(['Activation', 'BatchNorm1d', 'BatchNorm2d', 'Dropout']);
const WINDOWED = new Set(['Conv1d', 'Conv2d', 'MaxPool1d', 'MaxPool2d', 'AvgPool2d']);

/**
 * The receptive field of every brick whose output still has spatial dimensions (images,
 * signals): with kernel k and stride s, rf' = rf + (k - 1)·jump and jump' = jump·s.
 * Bricks after a Flatten, Linear or recurrent layer depend on the whole input (not listed).
 */
export function receptiveFields(analysis: Analysis, links: Link[]): Record<string, ReceptiveField> {
  const inputs: Record<string, string[]> = {};
  for (const link of links) {
    (inputs[link.target] ??= []).push(link.source);
  }
  const result: Record<string, ReceptiveField | null> = {};
  const visiting = new Set<string>();

  const compute = (id: string): ReceptiveField | null => {
    if (id in result) {
      return result[id];
    }
    if (visiting.has(id)) {
      return null; // a cycle (the analysis reports it)
    }
    visiting.add(id);
    const info = analysis.nodes[id];
    const shape = info?.output_shape ?? null;
    let field: ReceptiveField | null = null;
    if (info && shape) {
      const sources = (inputs[id] ?? []).map(compute);
      const first = sources[0] ?? null;
      const params = info.params ?? {};
      if (info.type === 'Input') {
        if (shape.length >= 2 && shape.length <= 3 && info.dtype !== 'long') {
          const spatial = shape.slice(1);
          field = {
            size: spatial.map(() => 1),
            jump: spatial.map(() => 1),
            input: spatial,
            approximate: false,
          };
        }
      } else if (first && sources.every((s) => s && s.size.length === first.size.length)) {
        const all = sources as ReceptiveField[];
        const merged: ReceptiveField = {
          size: first.size.map((_, d) => Math.max(...all.map((s) => s.size[d]))),
          jump: first.jump.map((_, d) => Math.max(...all.map((s) => s.jump[d]))),
          input: first.input,
          approximate: all.some((s) => s.approximate),
        };
        if (ELEMENTWISE.has(info.type) || info.type === 'Add' || info.type === 'Concat') {
          field = shape.length - 1 === merged.size.length ? merged : null;
        } else if (WINDOWED.has(info.type)) {
          const kernel = ints(params.kernel_size);
          const stride = ints(params.stride ?? params.kernel_size);
          if (kernel.length === merged.size.length) {
            field = {
              ...merged,
              size: merged.size.map((s, d) => s + (kernel[d] - 1) * merged.jump[d]),
              jump: merged.jump.map((j, d) => j * (stride[d] ?? 1)),
            };
          }
        } else if (info.type === 'ConvTranspose2d') {
          const kernel = ints(params.kernel_size);
          const stride = ints(params.stride);
          field = {
            ...merged,
            size: merged.size.map(
              (s, d) => s + (Math.ceil(kernel[d] / (stride[d] || 1)) - 1) * merged.jump[d]
            ),
            jump: merged.jump.map((j, d) => j / (stride[d] || 1)),
            approximate: true,
          };
        } else if (info.type === 'AdaptiveAvgPool2d') {
          const before = analysis.nodes[(inputs[id] ?? [])[0]]?.output_shape?.slice(1) ?? [];
          const after = shape.slice(1);
          if (before.length === merged.size.length) {
            field = {
              ...merged,
              size: merged.size.map(
                (s, d) => s + (Math.ceil(before[d] / after[d]) - 1) * merged.jump[d]
              ),
              jump: merged.jump.map((j, d) => j * Math.max(1, Math.floor(before[d] / after[d]))),
              approximate: before.some((b, d) => b % after[d] !== 0),
            };
          }
        }
      }
    }
    visiting.delete(id);
    result[id] = field;
    return field;
  };

  for (const id of Object.keys(analysis.nodes)) {
    compute(id);
  }
  return Object.fromEntries(
    Object.entries(result).filter((entry): entry is [string, ReceptiveField] => entry[1] !== null)
  );
}

/** "7×7 pixels" (capped at the input size: padding adds nothing to see). */
export function describeField(field: ReceptiveField): string {
  const seen = field.size.map((s, d) => Math.min(Math.round(s), field.input[d]));
  const unit = field.size.length === 2 ? 'pixels' : 'steps';
  return `${seen.join('×')} ${unit}`;
}

export function coversInput(field: ReceptiveField): boolean {
  return field.size.every((s, d) => s >= field.input[d]);
}

// --------------------------------------------------------------------------- data flow

export interface FlowStep {
  id: string;
  type: string;
  /** Shapes coming in (one per incoming connection) and going out. */
  inputs: (number[] | null)[];
  output: number[] | null;
  params: number | null;
}

/** The bricks in the order the data flows through them (inputs first). */
export function flowSteps(analysis: Analysis, links: Link[]): FlowStep[] {
  const incoming: Record<string, string[]> = {};
  const outgoing: Record<string, string[]> = {};
  for (const link of links) {
    if (analysis.nodes[link.source] && analysis.nodes[link.target]) {
      (incoming[link.target] ??= []).push(link.source);
      (outgoing[link.source] ??= []).push(link.target);
    }
  }
  const ids = Object.keys(analysis.nodes);
  const waiting = Object.fromEntries(ids.map((id) => [id, (incoming[id] ?? []).length]));
  const queue = [...analysis.inputs.filter((id) => waiting[id] === 0)];
  for (const id of ids) {
    if (waiting[id] === 0 && !queue.includes(id)) {
      queue.push(id);
    }
  }
  const order: string[] = [];
  while (queue.length > 0) {
    const id = queue.shift() as string;
    order.push(id);
    for (const next of outgoing[id] ?? []) {
      waiting[next] -= 1;
      if (waiting[next] === 0) {
        queue.push(next);
      }
    }
  }
  return order.map((id) => {
    const info = analysis.nodes[id];
    return {
      id,
      type: info.type,
      inputs: (incoming[id] ?? []).map((source) => analysis.nodes[source]?.output_shape ?? null),
      output: info.output_shape,
      params: info.num_params,
    };
  });
}

/** What a brick does to the shape, in words: "28×28 → 14×14: halves height and width". */
export function shapeChange(step: FlowStep): string | null {
  const before = step.inputs[0];
  const after = step.output;
  if (!before || !after || step.type === 'Input') {
    return null;
  }
  if (step.inputs.length > 1) {
    return step.type === 'Concat'
      ? 'joins its inputs along the channels or features'
      : 'combines its inputs';
  }
  if (before.join() === after.join()) {
    return 'keeps the shape';
  }
  if (before.length === 3 && after.length === 3) {
    const notes = [];
    if (before[0] !== after[0]) {
      notes.push(`${n(before[0])} → ${n(after[0])} channels`);
    }
    if (before[1] !== after[1] || before[2] !== after[2]) {
      notes.push(`${before[1]}×${before[2]} → ${after[1]}×${after[2]} pixels`);
    }
    return notes.join(', ');
  }
  const total = (shape: number[]) => shape.reduce((a, b) => a * b, 1);
  if (after.length === 1 && before.length > 1 && total(before) === after[0]) {
    return `lays out all ${n(after[0])} numbers in one row`;
  }
  return `${before.join('×')} → ${after.join('×')}`;
}
