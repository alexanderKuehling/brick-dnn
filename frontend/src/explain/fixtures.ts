/** A small CNN and its analysis, for the learning aids. */
import type { Analysis, NodeAnalysis } from '@/api/types';

export function node(
  type: string,
  shape: number[],
  params: Record<string, unknown> = {},
  count = 0
): NodeAnalysis {
  return { type, output_shape: shape, dtype: 'float', num_params: count, params } as NodeAnalysis;
}

/** Input 1×28×28 → Conv 3×3 → ReLU → MaxPool 2 → Conv 3×3 → MaxPool 2 → Flatten → Linear. */
export const CNN: Analysis = {
  valid: true,
  errors: [],
  warnings: [],
  inputs: ['in'],
  outputs: ['fc'],
  total_params: 0,
  family: 'neural',
  nodes: {
    in: node('Input', [1, 28, 28], { shape: [1, 28, 28] }),
    c1: node(
      'Conv2d',
      [32, 28, 28],
      { in_channels: 1, out_channels: 32, kernel_size: [3, 3], stride: [1, 1], bias: true },
      320
    ),
    r1: node('Activation', [32, 28, 28], { function: 'relu' }),
    p1: node('MaxPool2d', [32, 14, 14], { kernel_size: [2, 2], stride: [2, 2] }),
    c2: node(
      'Conv2d',
      [64, 14, 14],
      { in_channels: 32, out_channels: 64, kernel_size: [3, 3], stride: [1, 1], bias: true },
      18496
    ),
    p2: node('MaxPool2d', [64, 7, 7], { kernel_size: [2, 2], stride: [2, 2] }),
    fl: node('Flatten', [3136]),
    fc: node('Linear', [10], { in_features: 3136, out_features: 10, bias: true }, 31370),
  },
} as unknown as Analysis;
export const LINKS = [
  ['in', 'c1'],
  ['c1', 'r1'],
  ['r1', 'p1'],
  ['p1', 'c2'],
  ['c2', 'p2'],
  ['p2', 'fl'],
  ['fl', 'fc'],
].map(([source, target]) => ({ source, target }));
