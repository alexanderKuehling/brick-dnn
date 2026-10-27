import type { GraphDocument, LayerSpec } from '@/api/types';
import {
  autoLayout,
  defaultParams,
  freePosition,
  fromDocument,
  isParamVisible,
  parseGraphFile,
  semanticKey,
  slugify,
  toDocument,
  wouldCreateCycle,
  type LayerFlowNode,
} from './graph';

const linearSpec: LayerSpec = {
  type: 'Linear',
  label: 'Linear',
  category: 'Basic',
  description: '',
  min_inputs: 1,
  max_inputs: 1,
  params: [
    {
      name: 'in_features',
      label: 'Input features',
      kind: 'int',
      default: null,
      help: '',
      options: [],
      min: 1,
      max: null,
      auto: true,
      length: null,
      show_if: null,
    },
    {
      name: 'out_features',
      label: 'Output features',
      kind: 'int',
      default: 64,
      help: '',
      options: [],
      min: 1,
      max: null,
      auto: false,
      length: null,
      show_if: null,
    },
  ],
};

const doc: GraphDocument = {
  version: 1,
  nodes: [
    { id: 'in', type: 'Input', params: { shape: [4] }, position: { x: 0, y: 0 } },
    {
      id: 'fc',
      type: 'Linear',
      params: { out_features: 2 },
      name: 'head',
      position: { x: 300, y: 0 },
    },
  ],
  edges: [{ id: 'e1', source: 'in', target: 'fc' }],
};

describe('graph conversion', () => {
  it('round-trips a document through React Flow state', () => {
    const { nodes, edges } = fromDocument(doc);
    expect(nodes[1].data).toEqual({ type: 'Linear', params: { out_features: 2 }, name: 'head' });
    expect(toDocument(nodes, edges)).toEqual(doc);
  });

  it('fills in missing positions and edge ids', () => {
    const { nodes, edges } = fromDocument({
      version: 1,
      nodes: [
        { id: 'a', type: 'Input', params: {} },
        { id: 'b', type: 'Linear', params: {} },
      ],
      edges: [{ source: 'a', target: 'b' }],
    });
    expect(nodes[1].position).toEqual({ x: 300, y: 0 });
    expect(edges[0].id).toBe('e-a-b');
  });

  it('ignores layout in the semantic key', () => {
    const moved = { ...doc, nodes: doc.nodes.map((n) => ({ ...n, position: { x: 1, y: 1 } })) };
    expect(semanticKey(moved)).toBe(semanticKey(doc));
    const changed = {
      ...doc,
      nodes: [doc.nodes[0], { ...doc.nodes[1], params: { out_features: 3 } }],
    };
    expect(semanticKey(changed)).not.toBe(semanticKey(doc));
  });
});

describe('defaultParams', () => {
  it('leaves auto params empty and uses defaults otherwise', () => {
    expect(defaultParams(linearSpec)).toEqual({ in_features: null, out_features: 64 });
  });
});

describe('isParamVisible', () => {
  it('respects show_if', () => {
    const param = { ...linearSpec.params[0], show_if: { param: 'padding', value: 'custom' } };
    expect(isParamVisible(param, { padding: 'same' })).toBe(false);
    expect(isParamVisible(param, { padding: 'custom' })).toBe(true);
  });
});

describe('wouldCreateCycle', () => {
  const edges = [
    { id: '1', source: 'a', target: 'b' },
    { id: '2', source: 'b', target: 'c' },
  ];
  it('detects cycles', () => {
    expect(wouldCreateCycle(edges, 'c', 'a')).toBe(true);
    expect(wouldCreateCycle(edges, 'a', 'a')).toBe(true);
  });
  it('allows forward and parallel edges', () => {
    expect(wouldCreateCycle(edges, 'a', 'c')).toBe(false);
    expect(wouldCreateCycle(edges, 'x', 'a')).toBe(false);
  });
});

describe('parseGraphFile', () => {
  it('accepts a bare graph and a saved project', () => {
    expect(parseGraphFile(JSON.stringify(doc)).doc.nodes).toHaveLength(2);
    const project = parseGraphFile(JSON.stringify({ name: 'My net', graph: doc }));
    expect(project.name).toBe('My net');
  });
  it('rejects invalid files', () => {
    expect(() => parseGraphFile('not json')).toThrow('not valid JSON');
    expect(() => parseGraphFile('{"foo": 1}')).toThrow('missing "nodes"');
    expect(() => parseGraphFile('{"nodes": [{}], "edges": []}')).toThrow('malformed');
  });
});

describe('helpers', () => {
  it('finds a free position', () => {
    const nodes = fromDocument(doc).nodes as LayerFlowNode[];
    expect(freePosition(nodes, { x: 300, y: 0 })).toEqual({ x: 300, y: 170 });
    expect(freePosition(nodes, { x: 900, y: 0 })).toEqual({ x: 900, y: 0 });
  });
  it('slugifies names', () => {
    expect(slugify('My CNN (v2)!')).toBe('my_cnn_v2');
    expect(slugify('   ')).toBe('model');
  });
});

describe('autoLayout', () => {
  const chain = (n: number) => {
    const nodes = Array.from({ length: n }, (_, i) => ({
      id: `n${i}`,
      type: 'layer' as const,
      position: { x: 0, y: 0 },
      data: { type: 'Linear', params: {} },
    }));
    const edges = nodes
      .slice(1)
      .map((node, i) => ({ id: `e${i}`, source: `n${i}`, target: node.id }));
    return { nodes, edges };
  };

  it('wraps long chains into rows', () => {
    const { nodes, edges } = chain(6);
    const laid = autoLayout(nodes, edges, 4);
    expect(laid.map((n) => n.position)).toEqual([
      { x: 0, y: 0 },
      { x: 280, y: 0 },
      { x: 560, y: 0 },
      { x: 840, y: 0 },
      { x: 0, y: 230 },
      { x: 280, y: 230 },
    ]);
  });

  it('aligns parallel branches and stacks them', () => {
    const nodes = ['a', 'b', 'c', 'd'].map((id) => ({
      id,
      type: 'layer' as const,
      position: { x: 0, y: 0 },
      data: { type: 'Linear', params: {} },
    }));
    // a → b → d and c → d: c is placed in the same column as b.
    const edges = [
      { id: '1', source: 'a', target: 'b' },
      { id: '2', source: 'b', target: 'd' },
      { id: '3', source: 'c', target: 'd' },
    ];
    const laid = Object.fromEntries(autoLayout(nodes, edges).map((n) => [n.id, n.position]));
    expect(laid.b).toEqual({ x: 280, y: 0 });
    expect(laid.c).toEqual({ x: 280, y: 190 });
    expect(laid.d).toEqual({ x: 560, y: 0 });
  });

  it('survives cycles', () => {
    const { nodes, edges } = chain(3);
    expect(() =>
      autoLayout(nodes, [...edges, { id: 'x', source: 'n2', target: 'n0' }])
    ).not.toThrow();
  });
});
