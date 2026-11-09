import type { Edge } from '@xyflow/react';
import type { GraphDocument } from '@/api/types';
import { makeEdge } from '@/builder/flow';
import { fromDocument, toDocument, type LayerFlowNode } from '@/builder/graph';
import { appendOps, applyMeta, applyOps, diffGraph, mergeEdges, mergeNodes } from './ops';
import type { LiveOp } from './protocol';

const BASE: GraphDocument = {
  version: 1,
  nodes: [
    { id: 'in', type: 'Input', params: { shape: [4] }, position: { x: 0, y: 0 } },
    {
      id: 'fc',
      type: 'Linear',
      params: { out_features: 2, bias: true },
      position: { x: 300, y: 0 },
    },
    { id: 'act', type: 'Activation', params: { function: 'relu' }, position: { x: 600, y: 0 } },
  ],
  edges: [
    { source: 'in', target: 'fc' },
    { source: 'fc', target: 'act' },
  ],
};

function canvas(graph = BASE) {
  const { nodes, edges } = fromDocument(graph);
  return { nodes, edges: edges.map((e) => makeEdge(e.source, e.target)) };
}

const patch = (nodes: LayerFlowNode[], id: string, change: (n: LayerFlowNode) => LayerFlowNode) =>
  nodes.map((node) => (node.id === id ? change(node) : node));

describe('diffGraph: local changes become ops', () => {
  it('finds nothing in an unchanged canvas', () => {
    const { nodes, edges } = canvas();
    expect(diffGraph(BASE, 'MLP', nodes, edges, 'MLP').ops).toEqual([]);
  });

  it('sends a new layer with its connection', () => {
    const { nodes, edges } = canvas();
    const added: LayerFlowNode = {
      id: 'drop',
      type: 'layer',
      position: { x: 900.4, y: 10.6 },
      data: { type: 'Dropout', params: { p: 0.5 } },
    };
    const { ops } = diffGraph(
      BASE,
      'MLP',
      [...nodes, added],
      [...edges, makeEdge('act', 'drop')],
      'MLP'
    );
    expect(ops).toEqual([
      {
        op: 'add_node',
        node: { id: 'drop', type: 'Dropout', params: { p: 0.5 }, position: { x: 900, y: 11 } },
      },
      { op: 'add_edge', source: 'act', target: 'drop' },
    ]);
  });

  it('sends changed parameters only, a removed one as null, and names', () => {
    const { nodes, edges } = canvas();
    const changed = patch(nodes, 'fc', (n) => ({
      ...n,
      data: { ...n.data, params: { out_features: 10 }, name: 'head' },
    }));
    const { ops } = diffGraph(BASE, 'MLP', changed, edges, 'MLP');
    expect(ops).toEqual([
      {
        op: 'update_node',
        id: 'fc',
        set: { name: 'head', params: { out_features: 10, bias: null } },
      },
    ]);
  });

  it('sends a moved layer once the drag has ended', () => {
    const { nodes, edges } = canvas();
    const dragging = patch(nodes, 'act', (n) => ({
      ...n,
      position: { x: 650, y: 80 },
      dragging: true,
    }));
    const during = diffGraph(BASE, 'MLP', dragging, edges, 'MLP');
    expect(during.ops).toEqual([]);
    // The base keeps the old position, so the end of the drag is a change.
    const dropped = patch(dragging, 'act', (n) => ({
      ...n,
      position: { x: 660.2, y: 90 },
      dragging: false,
    }));
    expect(diffGraph(during.graph, 'MLP', dropped, edges, 'MLP').ops).toEqual([
      { op: 'update_node', id: 'act', set: { position: { x: 660, y: 90 } } },
    ]);
  });

  it('removes a layer with its connections in one op', () => {
    const { nodes, edges } = canvas();
    const { ops } = diffGraph(
      BASE,
      'MLP',
      nodes.filter((n) => n.id !== 'fc'),
      edges.filter((e) => e.source !== 'fc' && e.target !== 'fc'),
      'MLP'
    );
    expect(ops).toEqual([{ op: 'remove_node', id: 'fc' }]);
  });

  it('replaces the connection of a single-input layer', () => {
    const { nodes, edges } = canvas();
    const rewired: Edge[] = [...edges.filter((e) => e.target !== 'act'), makeEdge('in', 'act')];
    expect(diffGraph(BASE, 'MLP', nodes, rewired, 'MLP').ops).toEqual([
      { op: 'remove_edge', source: 'fc', target: 'act' },
      { op: 'add_edge', source: 'in', target: 'act' },
    ]);
  });

  it('renames the network', () => {
    const { nodes, edges } = canvas();
    expect(diffGraph(BASE, 'MLP', nodes, edges, 'Classifier').ops).toEqual([
      { op: 'set_meta', name: 'Classifier' },
    ]);
  });

  it('turns the base into the canvas when the ops are applied', () => {
    const { nodes } = canvas();
    const changed = [
      ...patch(nodes, 'fc', (n) => ({
        ...n,
        data: { ...n.data, params: { ...n.data.params, out_features: 7 } },
      })),
      {
        id: 'x',
        type: 'layer' as const,
        position: { x: 1, y: 2 },
        data: { type: 'Flatten', params: {} },
      },
    ].filter((n) => n.id !== 'in');
    const newEdges = [makeEdge('act', 'x')];
    const { ops, graph } = diffGraph(BASE, 'MLP', changed, newEdges, 'MLP');
    const applied = applyOps(BASE, ops);
    expect(toDocument(fromDocument(applied).nodes, [])).toEqual(
      toDocument(fromDocument(graph).nodes, [])
    );
    expect(applied.edges).toEqual([{ source: 'act', target: 'x' }]);
  });
});

describe('applyOps: ops from others', () => {
  it('applies them like the server and skips what does not apply', () => {
    const ops: LiveOp[] = [
      { op: 'update_node', id: 'fc', set: { params: { out_features: 5, bias: null } } },
      { op: 'update_node', id: 'gone', set: { params: { p: 1 } } },
      { op: 'add_node', node: { id: 'fc', type: 'Linear', params: {} } },
      { op: 'add_edge', source: 'in', target: 'act' },
      { op: 'add_edge', source: 'in', target: 'act' },
      { op: 'add_edge', source: 'in', target: 'gone' },
      { op: 'remove_node', id: 'fc' },
      { op: 'set_meta', name: 'Renamed' },
    ];
    const result = applyOps(BASE, ops);
    expect(result.nodes.map((n) => n.id)).toEqual(['in', 'act']);
    expect(result.edges).toEqual([{ source: 'in', target: 'act' }]);
    expect(applyMeta('MLP', ops)).toBe('Renamed');
    // Cleared parameters stay as null (the backend treats null and a missing key alike).
    const updated = applyOps(BASE, ops.slice(0, 1)).nodes[1];
    expect(updated.params).toEqual({ out_features: 5, bias: null });
    expect(BASE.nodes[1].params).toEqual({ out_features: 2, bias: true });
  });

  it('applies our pending ops on top again, as the server will', () => {
    const mine: LiveOp[] = [
      { op: 'update_node', id: 'fc', set: { params: { out_features: 64 } } },
      { op: 'add_edge', source: 'act', target: 'fc2' },
    ];
    const theirs: LiveOp[] = [
      { op: 'update_node', id: 'fc', set: { params: { out_features: 8, bias: false } } },
      { op: 'remove_node', id: 'act' },
    ];
    const result = applyOps(applyOps(BASE, mine), [...theirs, ...mine]);
    expect(result.nodes.find((n) => n.id === 'fc')?.params).toEqual({
      out_features: 64,
      bias: false,
    });
    expect(result.edges).toEqual([{ source: 'in', target: 'fc' }]);
  });
});

describe('appendOps', () => {
  it('merges changes of a layer into the earlier change or its creation', () => {
    const queue = appendOps(
      [{ op: 'add_node', node: { id: 'x', type: 'Linear', params: { out_features: 1 } } }],
      [
        { op: 'update_node', id: 'x', set: { params: { out_features: 12 }, name: 'head' } },
        { op: 'update_node', id: 'fc', set: { params: { out_features: 3 } } },
        { op: 'add_edge', source: 'fc', target: 'x' },
        { op: 'update_node', id: 'fc', set: { position: { x: 1, y: 1 }, params: { bias: false } } },
        { op: 'remove_node', id: 'fc' },
        { op: 'update_node', id: 'fc', set: { params: { out_features: 4 } } },
      ]
    );
    expect(queue).toEqual([
      {
        op: 'add_node',
        node: { id: 'x', type: 'Linear', params: { out_features: 12 }, name: 'head' },
      },
      {
        op: 'update_node',
        id: 'fc',
        set: { params: { out_features: 3, bias: false }, position: { x: 1, y: 1 } },
      },
      { op: 'add_edge', source: 'fc', target: 'x' },
      { op: 'remove_node', id: 'fc' },
      { op: 'update_node', id: 'fc', set: { params: { out_features: 4 } } },
    ]);
  });
});

describe('mergeNodes / mergeEdges: the canvas from the base', () => {
  it('keeps selection and unchanged layers, and the position of a dragged one', () => {
    const { nodes, edges } = canvas();
    const current = nodes.map((n) => ({
      ...n,
      selected: n.id === 'fc',
      dragging: n.id === 'act',
      measured: { width: 220, height: 90 },
    }));
    const next = applyOps(BASE, [
      { op: 'update_node', id: 'fc', set: { params: { out_features: 9 } } },
      { op: 'update_node', id: 'act', set: { position: { x: 5, y: 5 } } },
      {
        op: 'add_node',
        node: { id: 'new', type: 'Flatten', params: {}, position: { x: 1, y: 1 } },
      },
    ]);
    const merged = mergeNodes(current, next);
    expect(merged[0]).toBe(current[0]);
    expect(merged[1]).toMatchObject({ selected: true, data: { params: { out_features: 9 } } });
    expect(merged[1].measured).toEqual({ width: 220, height: 90 });
    expect(merged[2].position).toEqual({ x: 600, y: 0 });
    expect(merged[3]).toMatchObject({ id: 'new', position: { x: 1, y: 1 } });
    expect(mergeEdges(edges, next)).toEqual(edges);
  });

  it('drops layers others removed, but keeps new local ones', () => {
    const { nodes } = canvas();
    const local: LayerFlowNode = {
      id: 'mine',
      type: 'layer',
      position: { x: 0, y: 0 },
      data: { type: 'Flatten', params: {} },
    };
    const next = applyOps(BASE, [{ op: 'remove_node', id: 'fc' }]);
    expect(mergeNodes([...nodes, local], next, BASE).map((n) => n.id)).toEqual([
      'in',
      'act',
      'mine',
    ]);
    expect(mergeNodes([...nodes, local], next).map((n) => n.id)).toEqual(['in', 'act']);
  });
});
