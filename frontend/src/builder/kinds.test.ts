import type { Edge } from '@xyflow/react';
import { SPECS } from './classicalFixtures';
import { decorateEdges } from './flow';
import {
  compactKernel,
  connectionProblem,
  inputSlots,
  kernelLines,
  kernelTerms,
  outputKind,
  slotFor,
} from './kinds';

describe('input slots', () => {
  it('come from the catalogue', () => {
    expect(inputSlots(SPECS.GaussianProcessRegressor).map((slot) => slot.name)).toEqual([
      'features',
      'kernel',
    ]);
    expect(inputSlots(SPECS.RBF)).toEqual([]);
    expect(outputKind(SPECS.RBF)).toBe('kernel');
    expect(outputKind(SPECS.SVC)).toBe('tensor');
  });

  it('default to one tensor input for catalogues of older servers', () => {
    const { input_slots: _, family: __, output_kind: ___, ...old } = SPECS.Linear;
    expect(inputSlots(old)).toEqual([
      { name: 'input', label: 'Input', kind: 'tensor', min: 1, max: 1 },
    ]);
    expect(inputSlots({ ...old, max_inputs: 0 })).toEqual([]);
    expect(outputKind(old)).toBe('tensor');
  });

  it('pick the slot of a handle or of the kind', () => {
    const slots = inputSlots(SPECS.GaussianProcessRegressor);
    expect(slotFor(slots, 'kernel')?.name).toBe('kernel');
    expect(slotFor(slots, 'tensor')?.name).toBe('features');
    expect(slotFor(slots, 'tensor', 'kernel')?.name).toBe('kernel');
    expect(slotFor(inputSlots(SPECS.KernelSum), 'tensor')).toBeNull();
  });
});

describe('connections of the wrong kind', () => {
  it('refuses a kernel into a data input and data into a kernel input', () => {
    expect(connectionProblem(SPECS.RBF, SPECS.Linear)).toBe('Linear cannot take a kernel.');
    expect(connectionProblem(SPECS.RBF, SPECS.SVC)).toBe(
      'Support vector classification cannot take a kernel.'
    );
    expect(connectionProblem(SPECS.Input, SPECS.KernelSum)).toBe('Kernel sum only takes kernels.');
    expect(connectionProblem(SPECS.Input, SPECS.RBF)).toBe('RBF kernel takes no inputs.');
  });

  it('checks the handle of a Gaussian process', () => {
    const gp = SPECS.GaussianProcessRegressor;
    expect(connectionProblem(SPECS.Input, gp, 'features')).toBeNull();
    expect(connectionProblem(SPECS.KernelSum, gp, 'kernel')).toBeNull();
    expect(connectionProblem(SPECS.RBF, gp, 'features')).toBe(
      'The features input of Gaussian process regression takes data, not a kernel.'
    );
    expect(connectionProblem(SPECS.Input, gp, 'kernel')).toBe(
      'The kernel input of Gaussian process regression only takes a kernel.'
    );
    // Without a handle (e.g. appended from the palette) the slot follows the kind.
    expect(connectionProblem(SPECS.RBF, gp)).toBeNull();
  });

  it('leaves neural layers and unknown types alone', () => {
    expect(connectionProblem(SPECS.Input, SPECS.Linear)).toBeNull();
    expect(connectionProblem(SPECS.Linear, SPECS.Linear, 'whatever')).toBeNull();
    expect(connectionProblem(undefined, SPECS.Linear)).toBeNull();
  });
});

describe('edges on the canvas', () => {
  const types: Record<string, string> = {
    in: 'Input',
    rbf: 'RBF',
    sum: 'KernelSum',
    gp: 'GaussianProcessRegressor',
    fc: 'Linear',
  };
  const edge = (source: string, target: string): Edge => ({
    id: `e-${source}-${target}`,
    source,
    target,
  });

  it('end at the input that takes what they carry, kernels dashed', () => {
    const edges = [edge('in', 'gp'), edge('sum', 'gp'), edge('rbf', 'sum'), edge('in', 'fc')];
    const [features, kernel, intoSum, neural] = decorateEdges(edges, (id) => types[id], SPECS);
    expect(features.targetHandle).toBe('features');
    expect(features.className).toBeUndefined();
    expect(kernel.targetHandle).toBe('kernel');
    expect(kernel.className).toMatch(/kernelEdge/);
    expect(intoSum.targetHandle).toBeUndefined();
    expect(intoSum.className).toMatch(/kernelEdge/);
    // Neural connections stay as they are.
    expect(neural).toBe(edges[3]);
  });
});

describe('kernels in words', () => {
  const code =
    'ConstantKernel(constant_value=1.0) * RBF(length_scale=1.0, length_scale_bounds=(0.01, 100.0)) + WhiteKernel(noise_level=0.1)';

  it('drops the arguments for the canvas', () => {
    expect(compactKernel(code)).toBe('Constant × RBF + White');
    expect(
      compactKernel('(RBF(length_scale=1.0) + DotProduct(sigma_0=1.0)) * Matern(nu=1.5)')
    ).toBe('(RBF + Dot) × Matérn');
  });

  it('breaks lines between the terms of a sum', () => {
    expect(kernelTerms(code)).toEqual([
      'ConstantKernel(constant_value=1.0) * RBF(length_scale=1.0, length_scale_bounds=(0.01, 100.0))',
      '+ WhiteKernel(noise_level=0.1)',
    ]);
    expect(kernelTerms('(RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)) * RBF()')).toEqual([
      '(RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)) * RBF()',
    ]);
  });

  it('splits the kernels a classifier fitted per class', () => {
    expect(
      kernelLines('setosa: 1**2 * RBF(length_scale=0.5); versicolor: RBF(length_scale=2)')
    ).toEqual(['setosa: 1**2 * RBF(length_scale=0.5)', 'versicolor: RBF(length_scale=2)']);
    expect(kernelLines('RBF(length_scale=1)')).toEqual(['RBF(length_scale=1)']);
  });
});
