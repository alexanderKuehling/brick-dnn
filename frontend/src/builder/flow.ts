import { MarkerType, type Edge, type Node } from '@xyflow/react';
import type { MantineTheme } from '@mantine/core';
import type { LayerSpec } from '@/api/types';
import { categoryColor } from '@/theme';
import { edgeId, type LayerNodeData } from './graph';
import { LayerNode } from './LayerNode';

/** The builder and read-only views (share links) draw layers and connections alike. */
export const nodeTypes = { layer: LayerNode };

export const EDGE_DEFAULTS = {
  type: 'smoothstep',
  markerEnd: { type: MarkerType.ArrowClosed, width: 18, height: 18 },
} as const;

export function makeEdge(source: string, target: string): Edge {
  return { id: edgeId(source, target), source, target, ...EDGE_DEFAULTS };
}

export function withEdgeStyle(edges: Edge[]): Edge[] {
  return edges.map((edge) => ({ ...edge, ...EDGE_DEFAULTS }));
}

/** Colours layers in the minimap by category, like on the canvas. */
export function miniMapColor(theme: MantineTheme, specs: Record<string, LayerSpec>) {
  return (node: Node) =>
    theme.colors[categoryColor(specs[(node.data as LayerNodeData).type]?.category)]?.[6] ??
    theme.colors.gray[6];
}
