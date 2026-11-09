import { useMemo } from 'react';
import { MarkerType, type Edge, type Node } from '@xyflow/react';
import type { MantineTheme } from '@mantine/core';
import type { LayerSpec } from '@/api/types';
import { categoryColor } from '@/theme';
import { edgeId, type LayerFlowNode, type LayerNodeData } from './graph';
import { inputSlots, outputKind, slotFor } from './kinds';
import { LayerNode } from './LayerNode';
import classes from './Flow.module.css';

/** The builder and read-only views (share links) draw layers and connections alike. */
export const nodeTypes = { layer: LayerNode };

export const EDGE_DEFAULTS = {
  type: 'smoothstep',
  markerEnd: { type: MarkerType.ArrowClosed, width: 18, height: 18 },
} as const;

/** The arrow of connections that carry a kernel, in the colour of the kernels. */
const KERNEL_MARKER = {
  type: MarkerType.ArrowClosed,
  width: 18,
  height: 18,
  color: 'var(--mantine-color-pink-5)',
} as const;

export function makeEdge(source: string, target: string): Edge {
  return { id: edgeId(source, target), source, target, ...EDGE_DEFAULTS };
}

export function withEdgeStyle(edges: Edge[]): Edge[] {
  return edges.map((edge) => ({ ...edge, ...EDGE_DEFAULTS }));
}

/**
 * Edges as the canvas draws them: a connection into a brick with several inputs (a Gaussian
 * process: features and kernel) ends at the handle of the input that takes what its source
 * puts out, and kernel connections are dashed. The graph document has no handles; they
 * follow from the bricks (see kinds.ts). Unchanged edges keep their identity.
 */
export function decorateEdges(
  edges: Edge[],
  typeOf: (id: string) => string | undefined,
  specs: Record<string, LayerSpec>
): Edge[] {
  return edges.map((edge) => {
    const kind = outputKind(specs[typeOf(edge.source) ?? '']);
    const slots = inputSlots(specs[typeOf(edge.target) ?? '']);
    const handle = slots.length > 1 ? (slotFor(slots, kind) ?? slots[0]).name : undefined;
    const kernel = kind === 'kernel';
    if (
      (edge.targetHandle ?? undefined) === handle &&
      (edge.className === classes.kernelEdge) === kernel
    ) {
      return edge;
    }
    return {
      ...edge,
      targetHandle: handle,
      className: kernel ? classes.kernelEdge : undefined,
      markerEnd: kernel ? KERNEL_MARKER : EDGE_DEFAULTS.markerEnd,
    };
  });
}

/** `decorateEdges` for a canvas; only recomputed when the edges or the layer types change. */
export function useDecoratedEdges(
  edges: Edge[],
  nodes: LayerFlowNode[],
  specs: Record<string, LayerSpec>
): Edge[] {
  // Moving layers changes `nodes` all the time, their types rarely.
  const typesKey = JSON.stringify(nodes.map((node) => [node.id, node.data.type]));
  return useMemo(() => {
    const types = new Map(JSON.parse(typesKey) as [string, string][]);
    return decorateEdges(edges, (id) => types.get(id), specs);
  }, [edges, typesKey, specs]);
}

/** Colours layers in the minimap by category, like on the canvas. */
export function miniMapColor(theme: MantineTheme, specs: Record<string, LayerSpec>) {
  return (node: Node) =>
    theme.colors[categoryColor(specs[(node.data as LayerNodeData).type]?.category)]?.[6] ??
    theme.colors.gray[6];
}
