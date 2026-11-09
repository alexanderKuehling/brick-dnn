import type { CompatibilityFix, Dataset, LayerSpec } from '@/api/types';
import {
  describeFixes,
  describeShape,
  onlyInputAndOutput,
  sameParams,
  starterNetwork,
} from './fit';
import { dataset, STATUS } from './fixtures';
import { effectiveSource } from './settings';

const INPUT_FIX: CompatibilityFix = {
  node: 'in',
  type: 'Input',
  params: { shape: [4] },
  previous: { shape: [784] },
  summary: 'Input shape 784 → 4',
};
const OUTPUT_FIX: CompatibilityFix = {
  node: 'fc',
  type: 'Linear',
  params: { out_features: 3 },
  previous: { out_features: 10 },
  summary: 'Outputs of the last layer 10 → 3',
};

const spec = (type: string, params: [string, unknown][] = []): LayerSpec =>
  ({
    type,
    label: type,
    category: 'Basic',
    description: '',
    params: params.map(([name, value]) => ({ name, default: value, auto: false })),
    min_inputs: type === 'Input' ? 0 : 1,
    max_inputs: type === 'Input' ? 0 : 1,
  }) as unknown as LayerSpec;

const SPECS: Record<string, LayerSpec> = {
  Input: spec('Input', [
    ['shape', [784]],
    ['dtype', 'float'],
  ]),
  Linear: spec('Linear', [
    ['out_features', 64],
    ['bias', true],
  ]),
  Activation: spec('Activation', [['function', 'relu']]),
  Flatten: spec('Flatten'),
};

function upload(info: Partial<Dataset['info']>, kind: Dataset['kind'] = 'csv'): Dataset {
  const base = dataset({ id: 7, name: 'iris' });
  return { ...base, kind, info: { ...base.info, kind, ...info } };
}

describe('fixes', () => {
  it('only changes the Input and the number of outputs without asking', () => {
    expect(onlyInputAndOutput([INPUT_FIX, OUTPUT_FIX])).toBe(true);
    expect(onlyInputAndOutput([{ ...INPUT_FIX, params: { dtype: 'float' } }])).toBe(true);
    expect(onlyInputAndOutput([])).toBe(false);
    expect(onlyInputAndOutput([{ ...OUTPUT_FIX, params: { bias: false } }])).toBe(false);
    expect(onlyInputAndOutput([{ ...OUTPUT_FIX, type: 'Conv2d' }])).toBe(false);
    expect(onlyInputAndOutput([{ ...INPUT_FIX, params: {} }])).toBe(false);
  });

  it('describes what changes', () => {
    expect(describeFixes([INPUT_FIX], false)).toBe('Input set to 4 features');
    expect(describeFixes([INPUT_FIX, OUTPUT_FIX], false)).toBe(
      'Input set to 4 features, last layer to 3 outputs'
    );
    expect(describeFixes([OUTPUT_FIX], true)).toBe('Last layer set to 3 outputs');
    expect(describeFixes([{ ...INPUT_FIX, params: { shape: [3072] } }], true)).toBe(
      'Input set to 3,072 values'
    );
    expect(describeFixes([{ ...INPUT_FIX, params: { shape: [1, 28, 28] } }], true)).toBe(
      'Input set to 1×28×28'
    );
    expect(describeFixes([{ ...OUTPUT_FIX, params: { out_features: 1 } }], false)).toBe(
      'Last layer set to 1 output'
    );
    expect(describeShape([1], false)).toBe('1 feature');
  });

  it('compares parameters by value', () => {
    expect(sameParams({ shape: [4], dtype: 'float' }, { dtype: 'float', shape: [4] })).toBe(true);
    expect(sameParams({ shape: [4] }, { shape: [5] })).toBe(false);
    expect(sameParams({ shape: [4] }, { shape: [4], dtype: 'long' })).toBe(false);
  });
});

describe('starterNetwork', () => {
  const layers = (network: ReturnType<typeof starterNetwork>) =>
    network?.nodes.map((node) => [node.type, node.params]);

  it('builds an MLP for a CSV file', () => {
    const iris = upload({
      num_features: 4,
      target_kind: 'text',
      class_names: ['a', 'b', 'c'],
      num_classes: 3,
    });
    const source = effectiveSource(STATUS, { tab: 'upload', builtin: null, upload: 7 }, [iris]);
    const network = starterNetwork(source, SPECS);
    expect(layers(network)).toEqual([
      ['Input', { shape: [4], dtype: 'float' }],
      ['Linear', { out_features: 64, bias: true }],
      ['Activation', { function: 'relu' }],
      ['Linear', { out_features: 3, bias: true }],
    ]);
    expect(network!.edges.map((e) => [e.source, e.target])).toEqual(
      network!.nodes.slice(1).map((node, i) => [network!.nodes[i].id, node.id])
    );

    // Numbers are a regression; prepared data (Data tab) only names classes for classification.
    const prices = upload({ num_features: 2, target_kind: 'number', class_names: null });
    const regression = effectiveSource(STATUS, { tab: 'upload', builtin: null, upload: 7 }, [
      prices,
    ]);
    expect(layers(starterNetwork(regression, SPECS))?.at(-1)).toEqual([
      'Linear',
      { out_features: 1, bias: true },
    ]);
    const prepared = upload({
      num_features: 2,
      target_kind: 'integer',
      class_names: ['low', 'high'],
      predefined_split: [8, 1, 1],
    });
    const classes = effectiveSource(STATUS, { tab: 'upload', builtin: null, upload: 7 }, [
      prepared,
    ]);
    expect(layers(starterNetwork(classes, SPECS))?.at(-1)).toEqual([
      'Linear',
      { out_features: 2, bias: true },
    ]);
  });

  it('flattens images first', () => {
    const mnist = effectiveSource(STATUS, { tab: 'builtin', builtin: 'mnist', upload: null }, []);
    expect(layers(starterNetwork(mnist, SPECS))?.slice(0, 2)).toEqual([
      ['Input', { shape: [1, 28, 28], dtype: 'float' }],
      ['Flatten', {}],
    ]);
    const pets = upload({ class_names: ['cats', 'dogs'], num_classes: 2 }, 'images');
    const folder = effectiveSource(STATUS, { tab: 'upload', builtin: null, upload: 7 }, [pets]);
    const network = starterNetwork(folder, SPECS);
    expect(layers(network)?.[0]).toEqual(['Input', { shape: [3, 64, 64], dtype: 'float' }]);
    expect(layers(network)?.at(-1)).toEqual(['Linear', { out_features: 2, bias: true }]);
  });

  it('needs the layer types and the data', () => {
    const mnist = effectiveSource(STATUS, { tab: 'builtin', builtin: 'mnist', upload: null }, []);
    expect(starterNetwork(mnist, { Input: SPECS.Input })).toBeNull();
    const demo = effectiveSource(STATUS, { tab: 'random', builtin: null, upload: null }, []);
    expect(starterNetwork(demo, SPECS)).toBeNull();
  });
});
