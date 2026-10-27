import type { Edge, Node, XYPosition } from '@xyflow/react';
import type { GraphDocument, GraphEdge, LayerSpec, Params, ParamSpec } from '@/api/types';

export type LayerNodeData = {
  type: string;
  params: Params;
  /** Optional attribute name used in the generated code. */
  name?: string;
};

export type LayerFlowNode = Node<LayerNodeData, 'layer'>;

/** Horizontal distance between consecutive layers. */
export const LAYER_SPACING_X = 300;
export const LAYER_SPACING_Y = 170;

export function defaultParams(spec: LayerSpec): Params {
  return Object.fromEntries(spec.params.map((p) => [p.name, p.auto ? null : p.default]));
}

let counter = 0;
export function newId(prefix = 'n'): string {
  counter += 1;
  return `${prefix}${Date.now().toString(36)}${counter.toString(36)}`;
}

export function edgeId(source: string, target: string): string {
  return `e-${source}-${target}`;
}

export function createNode(spec: LayerSpec, position: XYPosition): LayerFlowNode {
  return {
    id: newId(),
    type: 'layer',
    position,
    data: { type: spec.type, params: defaultParams(spec) },
  };
}

export function isParamVisible(param: ParamSpec, params: Params): boolean {
  return !param.show_if || params[param.show_if.param] === param.show_if.value;
}

// --------------------------------------------------------------------------
// Conversion between React Flow state and the JSON graph document
// --------------------------------------------------------------------------

export function toDocument(nodes: LayerFlowNode[], edges: Edge[]): GraphDocument {
  return {
    version: 1,
    nodes: nodes.map((node) => ({
      id: node.id,
      type: node.data.type,
      params: node.data.params,
      ...(node.data.name ? { name: node.data.name } : {}),
      position: { x: Math.round(node.position.x), y: Math.round(node.position.y) },
    })),
    edges: edges.map((edge) => ({ id: edge.id, source: edge.source, target: edge.target })),
  };
}

export function fromDocument(doc: GraphDocument): { nodes: LayerFlowNode[]; edges: Edge[] } {
  const nodes: LayerFlowNode[] = doc.nodes.map((node, index) => ({
    id: String(node.id),
    type: 'layer',
    position: node.position ?? { x: index * LAYER_SPACING_X, y: 0 },
    data: {
      type: node.type,
      params: { ...(node.params ?? {}) },
      ...(node.name ? { name: node.name } : {}),
    },
  }));
  const edges: Edge[] = doc.edges.map((edge: GraphEdge) => ({
    id: edge.id ?? edgeId(String(edge.source), String(edge.target)),
    source: String(edge.source),
    target: String(edge.target),
  }));
  return { nodes, edges };
}

/** Everything the backend needs, without layout. Used to skip re-analysis while dragging. */
export function semanticKey(doc: GraphDocument): string {
  return JSON.stringify({
    version: doc.version,
    nodes: doc.nodes.map(({ id, type, params, name }) => ({ id, type, params, name })),
    edges: doc.edges.map(({ source, target }) => ({ source, target })),
  });
}

/** Parse a graph file. Accepts a bare graph or a saved project ({name, graph}). */
export function parseGraphFile(text: string): { doc: GraphDocument; name?: string } {
  let data: unknown;
  try {
    data = JSON.parse(text);
  } catch {
    throw new Error('The file is not valid JSON.');
  }
  const record = (data ?? {}) as Record<string, unknown>;
  const graph = (
    record.graph && typeof record.graph === 'object' ? record.graph : record
  ) as Partial<GraphDocument>;
  if (!Array.isArray(graph.nodes) || !Array.isArray(graph.edges)) {
    throw new Error('The file does not contain a Brick DNN graph (missing "nodes" or "edges").');
  }
  const ok =
    graph.nodes.every((n) => n && typeof n === 'object' && 'id' in n && 'type' in n) &&
    graph.edges.every((e) => e && typeof e === 'object' && 'source' in e && 'target' in e);
  if (!ok) {
    throw new Error('The graph contains malformed layers or connections.');
  }
  return {
    doc: { version: 1, nodes: graph.nodes, edges: graph.edges },
    name: typeof record.name === 'string' ? record.name : undefined,
  };
}

// --------------------------------------------------------------------------
// Graph helpers
// --------------------------------------------------------------------------

/** Would adding source → target close a cycle? */
export function wouldCreateCycle(edges: Edge[], source: string, target: string): boolean {
  if (source === target) {
    return true;
  }
  const outgoing = new Map<string, string[]>();
  for (const edge of edges) {
    outgoing.set(edge.source, [...(outgoing.get(edge.source) ?? []), edge.target]);
  }
  const stack = [target];
  const seen = new Set<string>();
  while (stack.length > 0) {
    const current = stack.pop()!;
    if (current === source) {
      return true;
    }
    if (!seen.has(current)) {
      seen.add(current);
      stack.push(...(outgoing.get(current) ?? []));
    }
  }
  return false;
}

/** Find a free spot near `position` so new layers do not stack on top of each other. */
export function freePosition(nodes: LayerFlowNode[], position: XYPosition): XYPosition {
  const taken = (p: XYPosition) =>
    nodes.some((n) => Math.abs(n.position.x - p.x) < 120 && Math.abs(n.position.y - p.y) < 80);
  let candidate = position;
  for (let i = 0; i < 20 && taken(candidate); i += 1) {
    candidate = { x: position.x, y: position.y + (i + 1) * LAYER_SPACING_Y };
  }
  return candidate;
}

export function slugify(text: string): string {
  return (
    text
      .trim()
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '_')
      .replace(/^_+|_+$/g, '')
      .slice(0, 60) || 'model'
  );
}

// --------------------------------------------------------------------------
// Automatic layout
// --------------------------------------------------------------------------

export const LAYOUT_PER_ROW = 4;
const LAYOUT_DX = 280;
const LAYOUT_DY = 190;

/**
 * Layered layout that wraps long networks into rows of `perRow` columns.
 *
 * Each layer gets a column from its longest distance to an output, so parallel
 * branches line up before they merge. Layers in the same column are stacked.
 */
export function autoLayout(
  nodes: LayerFlowNode[],
  edges: Edge[],
  perRow = LAYOUT_PER_ROW
): LayerFlowNode[] {
  const ids = new Set(nodes.map((n) => n.id));
  const outgoing = new Map<string, string[]>();
  for (const edge of edges) {
    if (ids.has(edge.source) && ids.has(edge.target)) {
      outgoing.set(edge.source, [...(outgoing.get(edge.source) ?? []), edge.target]);
    }
  }

  // Longest path to a sink (memoised DFS; cycles are cut).
  const height = new Map<string, number>();
  const visiting = new Set<string>();
  const heightOf = (id: string): number => {
    const known = height.get(id);
    if (known !== undefined) {
      return known;
    }
    if (visiting.has(id)) {
      return 0;
    }
    visiting.add(id);
    const h = Math.max(-1, ...(outgoing.get(id) ?? []).map(heightOf)) + 1;
    visiting.delete(id);
    height.set(id, h);
    return h;
  };
  const maxHeight = Math.max(0, ...nodes.map((n) => heightOf(n.id)));

  const columns = new Map<number, string[]>();
  for (const node of nodes) {
    const column = maxHeight - heightOf(node.id);
    columns.set(column, [...(columns.get(column) ?? []), node.id]);
  }

  const positions = new Map<string, XYPosition>();
  let top = 0;
  for (let band = 0; band * perRow <= maxHeight; band += 1) {
    let rows = 1;
    for (let offset = 0; offset < perRow; offset += 1) {
      const column = band * perRow + offset;
      const members = columns.get(column) ?? [];
      members.forEach((id, row) =>
        positions.set(id, { x: offset * LAYOUT_DX, y: top + row * LAYOUT_DY })
      );
      rows = Math.max(rows, members.length);
    }
    top += rows * LAYOUT_DY + 40;
  }

  return nodes.map((node) => ({ ...node, position: positions.get(node.id) ?? node.position }));
}
