/**
 * The canvas for keyboards and screen readers (docs/accessibility.md).
 *
 * React Flow already makes layers and connections focusable (Tab), selectable (Enter, Space),
 * movable (arrow keys) and deletable (Delete). This adds what it leaves to the app: names for
 * layers and connections ("Linear, fully connected", "Connection from Input to Linear"), its
 * labels and descriptions in the app's language, and a polite live region that says what an
 * action changed. Connecting layers without a mouse happens in the properties panel
 * (Inspector, "Connections").
 */
import { useCallback, useMemo, useRef, useState } from 'react';
import type { Edge } from '@xyflow/react';
import type { AriaLabelConfig } from '@xyflow/system';
import { VisuallyHidden } from '@mantine/core';
import type { LayerSpec } from '@/api/types';
import { i18n, useLanguage } from '@/i18n';
import type { LayerFlowNode } from './graph';

/** What a layer is called in texts: its name in code, else the label of its type. */
export function layerName(
  node: Pick<LayerFlowNode, 'data'> | undefined,
  specs: Record<string, LayerSpec>
): string {
  if (!node) {
    return '';
  }
  return node.data.name || specs[node.data.type]?.label || node.data.type;
}

/** React Flow's labels and descriptions in the app's language. */
export function useCanvasLabels(readOnly = false): Partial<AriaLabelConfig> {
  const language = useLanguage();
  return useMemo(() => {
    const node = readOnly
      ? i18n.t('a11y.canvas.nodeDescriptionReadOnly')
      : i18n.t('a11y.canvas.nodeDescription');
    return {
      // React Flow shows the "keyboardDisabled" text while the keyboard is enabled (and the
      // other way round), so both get the same text.
      'node.a11yDescription.default': node,
      'node.a11yDescription.keyboardDisabled': node,
      'node.a11yDescription.ariaLiveMessage': ({ direction, x, y }) =>
        i18n.t('a11y.canvas.moved', {
          direction: i18n.t(
            `a11y.canvas.directions.${direction as 'left' | 'right' | 'up' | 'down'}`
          ),
          x,
          y,
        }),
      'edge.a11yDescription.default': i18n.t('a11y.canvas.edgeDescription'),
      'controls.ariaLabel': i18n.t('a11y.canvas.controls'),
      'controls.zoomIn.ariaLabel': i18n.t('a11y.canvas.zoomIn'),
      'controls.zoomOut.ariaLabel': i18n.t('a11y.canvas.zoomOut'),
      'controls.fitView.ariaLabel': i18n.t('a11y.canvas.fitView'),
      'controls.interactive.ariaLabel': i18n.t('a11y.canvas.interactive'),
      'minimap.ariaLabel': i18n.t('a11y.canvas.minimap'),
      'handle.ariaLabel': i18n.t('a11y.canvas.handle'),
    };
    // `language`: the texts change with it.
  }, [language, readOnly]);
}

/**
 * The nodes with an accessible name each ("conv1, Conv2d, has problems"). A node keeps its
 * identity while it and its name stay the same, so React Flow does not redo unchanged nodes
 * (e.g. while one of them is dragged).
 */
export function useLabelledNodes(
  nodes: LayerFlowNode[],
  specs: Record<string, LayerSpec>,
  errorsByNode: Record<string, string[]>
): LayerFlowNode[] {
  const language = useLanguage();
  const cache = useRef(new WeakMap<LayerFlowNode, LayerFlowNode>());
  return useMemo(
    () =>
      nodes.map((node) => {
        const type = specs[node.data.type]?.label ?? node.data.type;
        const name = node.data.name || type;
        const ariaLabel = errorsByNode[node.id]?.length
          ? i18n.t('a11y.canvas.nodeWithErrors', { name, type })
          : i18n.t('a11y.canvas.node', { name, type });
        const cached = cache.current.get(node);
        if (cached?.ariaLabel === ariaLabel) {
          return cached;
        }
        const labelled = { ...node, ariaLabel };
        cache.current.set(node, labelled);
        return labelled;
      }),
    // `language`: the texts change with it.
    [nodes, specs, errorsByNode, language]
  );
}

/** The edges with a name each ("Connection from Input to Linear"), cached like the nodes. */
export function useLabelledEdges(
  edges: Edge[],
  nodes: LayerFlowNode[],
  specs: Record<string, LayerSpec>
): Edge[] {
  const language = useLanguage();
  const cache = useRef(new WeakMap<Edge, Edge>());
  // Moving layers changes `nodes` all the time, their names rarely.
  const namesKey = JSON.stringify(nodes.map((node) => [node.id, layerName(node, specs)]));
  return useMemo(() => {
    const names = new Map(JSON.parse(namesKey) as [string, string][]);
    return edges.map((edge) => {
      const ariaLabel = i18n.t('a11y.canvas.edge', {
        source: names.get(edge.source) ?? edge.source,
        target: names.get(edge.target) ?? edge.target,
      });
      const cached = cache.current.get(edge);
      if (cached?.ariaLabel === ariaLabel) {
        return cached;
      }
      const labelled = { ...edge, ariaLabel };
      cache.current.set(edge, labelled);
      return labelled;
    });
    // `language`: the texts change with it.
  }, [edges, namesKey, language]);
}

/** The canvas element of a layer (React Flow's node wrapper), if it is rendered. */
export function layerElement(id: string): HTMLElement | null {
  return document.querySelector<HTMLElement>(`.react-flow__node[data-id="${CSS.escape(id)}"]`);
}

/** Moves the keyboard focus to a layer on the canvas, once React has drawn it. */
export function focusLayer(id: string): void {
  window.setTimeout(() => layerElement(id)?.focus({ preventScroll: true }), 0);
}

/**
 * A polite live region and the function that fills it. The same message twice in a row is
 * read again (a zero-width space makes the text differ).
 */
export function useAnnouncer(): [string, (message: string) => void] {
  const [message, setMessage] = useState('');
  const announce = useCallback(
    (next: string) => setMessage((current) => (current === next ? `${next}\u200b` : next)),
    []
  );
  return [message, announce];
}

export function LiveRegion({ message }: { message: string }) {
  return (
    <VisuallyHidden aria-live="polite" aria-atomic="true" data-testid="announcer">
      {message}
    </VisuallyHidden>
  );
}
