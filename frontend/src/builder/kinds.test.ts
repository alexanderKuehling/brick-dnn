import type { Edge } from '@xyflow/react';
import { SPECS } from './classicalFixtures';
import { decorateEdges } from './flow';
import {
  compactKernel,
  connectionProblem,
  inputSlots,
  isBoosting,
  isEnsemble,
  isFeatureEdge,
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
    // A classical model passes itself on (into an ensemble).
    expect(outputKind(SPECS.SVC)).toBe('estimator');
    expect(inputSlots(SPECS.VotingClassifier)).toEqual([
      { name: 'models', label: 'Models', kind: 'estimator', min: 2, max: null },
    ]);
    expect(isEnsemble('StackingRegressor')).toBe(true);
    expect(isEnsemble('SVC')).toBe(false);
    expect(inputSlots(SPECS.HistGradientBoostingClassifier).map((slot) => slot.name)).toEqual([
      'features',
    ]);
    expect(isBoosting('HistGradientBoostingClassifier')).toBe(true);
    expect(isBoosting('SVC')).toBe(false);
    expect(isBoosting(undefined)).toBe(false);
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
    expect(connectionProblem(SPECS.RBF, SPECS.HistGradientBoostingClassifier)).toBe(
      'Gradient boosting classification cannot take a kernel.'
    );
    expect(connectionProblem(SPECS.Input, SPECS.HistGradientBoostingClassifier)).toBeNull();
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

  it('connects classical models only to ensembles', () => {
    expect(connectionProblem(SPECS.SVC, SPECS.VotingClassifier)).toBeNull();
    expect(connectionProblem(SPECS.GaussianProcessRegressor, SPECS.StackingRegressor)).toBeNull();
    expect(connectionProblem(SPECS.SVC, SPECS.Linear)).toBe(
      'Nothing can follow Support vector classification except an ensemble (voting or stacking).'
    );
    expect(connectionProblem(SPECS.SVC, SPECS.GaussianProcessRegressor, 'features')).toBe(
      'The features input of Gaussian process regression takes data, not a model.'
    );
    expect(connectionProblem(SPECS.Input, SPECS.VotingClassifier)).toBe(
      'Voting classifier only takes classical models: connect the Input to the models.'
    );
    expect(connectionProblem(SPECS.RBF, SPECS.VotingClassifier)).toBe(
      'Voting classifier cannot take a kernel.'
    );
  });

  it('connects a neural layer into a classical model: a hybrid network', () => {
    // The layers learn the features of the model; the features input takes their data.
    expect(connectionProblem(SPECS.Linear, SPECS.HistGradientBoostingClassifier)).toBeNull();
    expect(connectionProblem(SPECS.Linear, SPECS.SVC)).toBeNull();
    expect(connectionProblem(SPECS.Linear, SPECS.GaussianProcessRegressor, 'features')).toBeNull();
    expect(connectionProblem(SPECS.Linear, SPECS.GaussianProcessRegressor, 'kernel')).toBe(
      'The kernel input of Gaussian process regression only takes a kernel.'
    );
    expect(isFeatureEdge(SPECS.Linear, SPECS.SVC)).toBe(true);
    expect(isFeatureEdge(SPECS.Input, SPECS.SVC)).toBe(false);
    expect(isFeatureEdge(SPECS.Linear, SPECS.Linear)).toBe(false);
    expect(isFeatureEdge(SPECS.SVC, SPECS.VotingClassifier)).toBe(false);
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

  it('draw the models that go into an ensemble dotted', () => {
    const kinds: Record<string, string> = { ...types, svc: 'SVC', vote: 'VotingClassifier' };
    const edges = [edge('in', 'svc'), edge('svc', 'vote')];
    const [features, model] = decorateEdges(edges, (id) => kinds[id], SPECS);
    expect(features.className).toBeUndefined();
    expect(model.className).toMatch(/estimatorEdge/);
    expect(model.targetHandle).toBeUndefined(); // an ensemble has one input
    expect(model.markerEnd).toMatchObject({ color: 'var(--mantine-color-grape-6)' });
    // Unchanged edges keep their identity.
    expect(decorateEdges([model], (id) => kinds[id], SPECS)[0]).toBe(model);
  });

  it('mark the boundary of a hybrid network: the features into a classical model', () => {
    const kinds: Record<string, string> = { ...types, svc: 'SVC', fc2: 'Linear' };
    const edges = [edge('in', 'fc'), edge('fc', 'fc2'), edge('fc2', 'svc')];
    const [first, between, features] = decorateEdges(edges, (id) => kinds[id], SPECS);
    expect(first).toBe(edges[0]);
    expect(between).toBe(edges[1]);
    expect(features.className).toMatch(/featuresEdge/);
    expect(features.label).toBe('features');
    expect(features.markerEnd).toMatchObject({ color: 'var(--mantine-color-teal-6)' });
    expect(decorateEdges([features], (id) => kinds[id], SPECS)[0]).toBe(features);
    // A classical model fed straight by the Input has no boundary.
    const [direct] = decorateEdges([edge('in', 'svc')], (id) => kinds[id], SPECS);
    expect(direct.className).toBeUndefined();
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
