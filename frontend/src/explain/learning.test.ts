import type { Analysis } from '@/api/types';
import { CNN, LINKS, node } from './fixtures';
import {
  coversInput,
  describeField,
  flowSteps,
  paramFormula,
  receptiveFields,
  shapeChange,
} from './learning';

describe('parameter formulas', () => {
  it('add up like the analysis counts', () => {
    expect(paramFormula(CNN.nodes.c2)?.formula).toBe('3×3×32×64 + 64 = 18,496');
    expect(paramFormula(CNN.nodes.fc)?.formula).toBe('3,136×10 + 10 = 31,370');
    expect(paramFormula(node('BatchNorm2d', [8, 4, 4], { num_features: 8 }, 16))?.formula).toBe(
      '2×8 = 16'
    );
    const lstm = node(
      'LSTM',
      [128],
      { input_size: 64, hidden_size: 32, num_layers: 1, bidirectional: true },
      2 * (4 * 32 * (64 + 32) + 2 * 4 * 32)
    );
    expect(paramFormula(lstm)?.formula).toBe('2×(4×32×(64+32) + 2×4×32) = 25,088');
  });

  it('are left out when they would not match or there is nothing to count', () => {
    expect(paramFormula({ ...CNN.nodes.c2, num_params: 1 })).toBeNull();
    expect(paramFormula(CNN.nodes.r1)).toBeNull();
    expect(paramFormula(node('TransformerEncoder', [8, 16], {}, 999))).toBeNull();
  });
});

describe('receptive fields', () => {
  it('grow with every convolution and pooling', () => {
    const fields = receptiveFields(CNN, LINKS);
    expect(fields.c1.size).toEqual([3, 3]);
    expect(fields.r1.size).toEqual([3, 3]);
    expect(fields.p1).toMatchObject({ size: [4, 4], jump: [2, 2] });
    expect(fields.c2).toMatchObject({ size: [8, 8], jump: [2, 2] });
    expect(fields.p2).toMatchObject({ size: [10, 10], jump: [4, 4] });
    expect(fields.fl).toBeUndefined(); // after Flatten: everything
    expect(describeField(fields.p2)).toBe('10×10 pixels');
    expect(coversInput(fields.p2)).toBe(false);
  });

  it('are capped at the input and take the larger branch', () => {
    const wide: Analysis = {
      ...CNN,
      nodes: {
        in: node('Input', [1, 6, 6]),
        a: node('Conv2d', [4, 6, 6], { kernel_size: [5, 5], stride: [1, 1] }),
        b: node('Conv2d', [4, 6, 6], { kernel_size: [3, 3], stride: [1, 1] }),
        sum: node('Add', [4, 6, 6]),
        c: node('Conv2d', [4, 6, 6], { kernel_size: [3, 3], stride: [1, 1] }),
      },
    } as unknown as Analysis;
    const links = [
      ['in', 'a'],
      ['in', 'b'],
      ['a', 'sum'],
      ['b', 'sum'],
      ['sum', 'c'],
    ].map(([source, target]) => ({ source, target }));
    const fields = receptiveFields(wide, links);
    expect(fields.sum.size).toEqual([5, 5]);
    expect(fields.c.size).toEqual([7, 7]);
    expect(describeField(fields.c)).toBe('6×6 pixels');
    expect(coversInput(fields.c)).toBe(true);
  });
});

describe('the data flow', () => {
  it('goes through the bricks in order and says what changes', () => {
    const steps = flowSteps(CNN, LINKS);
    expect(steps.map((s) => s.id)).toEqual(['in', 'c1', 'r1', 'p1', 'c2', 'p2', 'fl', 'fc']);
    expect(steps[1].inputs).toEqual([[1, 28, 28]]);
    expect(shapeChange(steps[0])).toBeNull();
    expect(shapeChange(steps[1])).toBe('1 → 32 channels');
    expect(shapeChange(steps[2])).toBe('keeps the shape');
    expect(shapeChange(steps[3])).toBe('28×28 → 14×14 pixels');
    expect(shapeChange(steps[6])).toBe('lays out all 3,136 numbers in one row');
    expect(shapeChange(steps[7])).toBe('3136 → 10');
  });
});
