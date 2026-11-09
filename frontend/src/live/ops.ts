/**
 * Turning canvas changes into live ops and applying ops to a graph.
 *
 * The builder keeps a "base": the graph as the server will have it once all of our pending
 * ops are applied. Local edits are found by comparing the canvas with the base (`diffGraph`);
 * remote ops are applied to the base and the canvas is rebuilt from it (`mergeNodes`,
 * `mergeEdges`), keeping selection, sizes and layers that are being dragged.
 */
import type { Edge, XYPosition } from '@xyflow/react';
import type { GraphDocument, GraphEdge, GraphNode, Params, ParamValue } from '@/api/types';
import { makeEdge } from '@/builder/flow';
import { LAYER_SPACING_X, type LayerFlowNode, type LayerNodeData } from '@/builder/graph';
import type { LiveOp, NodeChange } from './protocol';

const edgeKey = (source: string, target: string) => JSON.stringify([source, target]);

const samePosition = (a?: XYPosition, b?: XYPosition) =>
  a === b || (a !== undefined && b !== undefined && a.x === b.x && a.y === b.y);

function sameValue(a: ParamValue | undefined, b: ParamValue | undefined): boolean {
  if (a === b) {
    return true;
  }
  if (Array.isArray(a) && Array.isArray(b)) {
    return a.length === b.length && a.every((value, index) => value === b[index]);
  }
  return false;
}

const rounded = (position: XYPosition): XYPosition => ({
  x: Math.round(position.x),
  y: Math.round(position.y),
});

/** The changes from `before` to `after` of one layer, or null if there are none. */
export function nodeChange(before: GraphNode, after: GraphNode): NodeChange | null {
  const set: NodeChange = {};
  if (before.type !== after.type) {
    set.type = after.type;
  }
  if ((before.name || null) !== (after.name || null)) {
    set.name = after.name || null;
  }
  if (before.params !== after.params) {
    const params: Params = {};
    for (const [key, value] of Object.entries(after.params)) {
      if (!sameValue(before.params[key], value)) {
        params[key] = value;
      }
    }
    for (const key of Object.keys(before.params)) {
      if (!(key in after.params)) {
        params[key] = null;
      }
    }
    if (Object.keys(params).length > 0) {
      set.params = params;
    }
  }
  if (after.position && !samePosition(before.position, after.position)) {
    set.position = after.position;
  }
  return Object.keys(set).length > 0 ? set : null;
}

export interface DiffResult {
  ops: LiveOp[];
  /** The new base: the canvas, except that layers being dragged keep their old position. */
  graph: GraphDocument;
  name: string;
}

/**
 * The ops that turn `base` into the canvas. Layers that are being dragged are compared
 * without their position, so a drag is sent once, when it ends.
 */
export function diffGraph(
  base: GraphDocument,
  baseName: string,
  nodes: LayerFlowNode[],
  edges: Edge[],
  name: string
): DiffResult {
  const before = new Map(base.nodes.map((node) => [String(node.id), node]));
  const ids = new Set(nodes.map((node) => node.id));
  const added: LiveOp[] = [];
  const updated: LiveOp[] = [];
  const graphNodes: GraphNode[] = nodes.map((node) => {
    const old = before.get(node.id);
    const position = old && node.dragging ? old.position : rounded(node.position);
    const current: GraphNode = {
      id: node.id,
      type: node.data.type,
      params: node.data.params,
      ...(node.data.name ? { name: node.data.name } : {}),
      ...(position ? { position } : {}),
    };
    if (!old) {
      added.push({ op: 'add_node', node: current });
    } else {
      const set = nodeChange(old, current);
      if (set) {
        updated.push({ op: 'update_node', id: node.id, set });
      }
    }
    return current;
  });
  const removedNodes: LiveOp[] = base.nodes
    .filter((node) => !ids.has(String(node.id)))
    .map((node) => ({ op: 'remove_node', id: String(node.id) }));

  const oldEdges = new Set(base.edges.map((e) => edgeKey(String(e.source), String(e.target))));
  const newEdges = new Set(edges.map((e) => edgeKey(e.source, e.target)));
  // Connections of removed layers go with them (remove_node takes care of that).
  const removedEdges: LiveOp[] = base.edges
    .filter((e) => !newEdges.has(edgeKey(String(e.source), String(e.target))))
    .filter((e) => ids.has(String(e.source)) && ids.has(String(e.target)))
    .map((e) => ({ op: 'remove_edge', source: String(e.source), target: String(e.target) }));
  const addedEdges: LiveOp[] = edges
    .filter((e) => !oldEdges.has(edgeKey(e.source, e.target)))
    .map((e) => ({ op: 'add_edge', source: e.source, target: e.target }));

  const meta: LiveOp[] = name !== baseName ? [{ op: 'set_meta', name }] : [];
  return {
    ops: [...added, ...updated, ...removedEdges, ...removedNodes, ...addedEdges, ...meta],
    graph: {
      version: 1,
      nodes: graphNodes,
      edges: edges.map((e) => ({ source: e.source, target: e.target })),
    },
    name,
  };
}

function applyChange(node: GraphNode, set: NodeChange): GraphNode {
  const next: GraphNode = { ...node };
  if (set.params) {
    // A null value clears a parameter. The backend treats null and a missing key alike; null
    // is kept here so that everyone sees the cleared field like the person who cleared it.
    next.params = { ...node.params, ...set.params };
  }
  if (set.position) {
    next.position = set.position;
  }
  if ('name' in set) {
    if (set.name) {
      next.name = set.name;
    } else {
      delete next.name;
    }
  }
  if (set.type) {
    next.type = set.type;
  }
  return next;
}

/** Apply ops to a graph like the server does (ops that do not apply are skipped). */
export function applyOps(graph: GraphDocument, ops: LiveOp[]): GraphDocument {
  let nodes = graph.nodes;
  let edges = graph.edges;
  const has = (id: string) => nodes.some((node) => String(node.id) === id);
  for (const op of ops) {
    switch (op.op) {
      case 'add_node':
        if (!has(op.node.id)) {
          nodes = [...nodes, { ...op.node, params: { ...(op.node.params ?? {}) } }];
        }
        break;
      case 'update_node':
        nodes = nodes.map((node) => (String(node.id) === op.id ? applyChange(node, op.set) : node));
        break;
      case 'remove_node':
        nodes = nodes.filter((node) => String(node.id) !== op.id);
        edges = edges.filter((e) => String(e.source) !== op.id && String(e.target) !== op.id);
        break;
      case 'add_edge': {
        const exists = edges.some(
          (e) => String(e.source) === op.source && String(e.target) === op.target
        );
        if (!exists && op.source !== op.target && has(op.source) && has(op.target)) {
          edges = [...edges, { source: op.source, target: op.target }];
        }
        break;
      }
      case 'remove_edge':
        edges = edges.filter(
          (e) => !(String(e.source) === op.source && String(e.target) === op.target)
        );
        break;
      default:
        break; // set_meta: see applyMeta
    }
  }
  return nodes === graph.nodes && edges === graph.edges ? graph : { version: 1, nodes, edges };
}

/** The network name after the ops. */
export function applyMeta(name: string, ops: LiveOp[]): string {
  let result = name;
  for (const op of ops) {
    if (op.op === 'set_meta' && typeof op.name === 'string') {
      result = op.name;
    }
  }
  return result;
}

function mergeChange(target: LiveOp, set: NodeChange): LiveOp {
  if (target.op === 'add_node') {
    return { ...target, node: applyChange(target.node, set) };
  }
  if (target.op === 'update_node') {
    const merged: NodeChange = { ...target.set, ...set };
    if (target.set.params && set.params) {
      merged.params = { ...target.set.params, ...set.params };
    }
    return { ...target, set: merged };
  }
  return target;
}

/**
 * Append ops to a queue that has not been sent yet. A change to a layer is merged into an
 * earlier change (or the creation) of the same layer, so typing a value sends one op; the
 * same for renaming the network.
 */
export function appendOps(queue: LiveOp[], ops: LiveOp[]): LiveOp[] {
  const result = [...queue];
  for (const op of ops) {
    if (op.op === 'update_node') {
      let index = -1;
      for (let i = result.length - 1; i >= 0; i -= 1) {
        const earlier = result[i];
        if (earlier.op === 'remove_node' && earlier.id === op.id) {
          break;
        }
        if (
          (earlier.op === 'update_node' && earlier.id === op.id) ||
          (earlier.op === 'add_node' && earlier.node.id === op.id)
        ) {
          index = i;
          break;
        }
      }
      if (index >= 0) {
        result[index] = mergeChange(result[index], op.set);
        continue;
      }
    }
    if (op.op === 'set_meta') {
      // The name does not depend on the graph: one rename is enough.
      const index = result.findIndex((earlier) => earlier.op === 'set_meta');
      if (index >= 0) {
        result[index] = { ...result[index], ...op };
        continue;
      }
    }
    result.push(op);
  }
  return result;
}

function sameData(a: LayerNodeData, b: GraphNode): boolean {
  if (a.type !== b.type || (a.name || null) !== (b.name || null)) {
    return false;
  }
  if (a.params === b.params) {
    return true;
  }
  const keys = new Set([...Object.keys(a.params), ...Object.keys(b.params)]);
  return [...keys].every((key) => sameValue(a.params[key], b.params[key]));
}

/**
 * The canvas layers for `graph`, reusing the current ones (selection, measured size; a layer
 * that is being dragged keeps its position). Layers that are not in `graph` are dropped,
 * unless they are missing from `previous` too (added here, not compared with the base yet).
 */
export function mergeNodes(
  current: LayerFlowNode[],
  graph: GraphDocument,
  previous?: GraphDocument
): LayerFlowNode[] {
  const byId = new Map(current.map((node) => [node.id, node]));
  const result: LayerFlowNode[] = graph.nodes.map((doc, index) => {
    const id = String(doc.id);
    const existing = byId.get(id);
    const position =
      existing?.dragging || !doc.position
        ? (existing?.position ?? { x: index * LAYER_SPACING_X, y: 0 })
        : doc.position;
    if (existing && sameData(existing.data, doc) && samePosition(existing.position, position)) {
      return existing;
    }
    const data: LayerNodeData = {
      type: doc.type,
      params: doc.params ?? {},
      ...(doc.name ? { name: doc.name } : {}),
    };
    return existing ? { ...existing, position, data } : { id, type: 'layer', position, data };
  });
  if (previous) {
    const known = new Set([...graph.nodes, ...previous.nodes].map((node) => String(node.id)));
    result.push(...current.filter((node) => !known.has(node.id)));
  }
  return result;
}

/** The canvas connections for `graph`, reusing the current ones (their selection). */
export function mergeEdges(current: Edge[], graph: GraphDocument): Edge[] {
  const byKey = new Map(current.map((edge) => [edgeKey(edge.source, edge.target), edge]));
  const ids = new Set(graph.nodes.map((node) => String(node.id)));
  return graph.edges
    .filter((e: GraphEdge) => ids.has(String(e.source)) && ids.has(String(e.target)))
    .map(
      (e: GraphEdge) =>
        byKey.get(edgeKey(String(e.source), String(e.target))) ??
        makeEdge(String(e.source), String(e.target))
    );
}

/** The ids of layers that ops remove. */
export function removedIds(ops: LiveOp[]): string[] {
  return ops.flatMap((op) => (op.op === 'remove_node' ? [op.id] : []));
}
