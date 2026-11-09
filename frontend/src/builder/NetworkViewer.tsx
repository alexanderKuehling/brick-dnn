import { useCallback, useEffect, useMemo, useRef } from 'react';
import { IconAdjustmentsHorizontal, IconPlugConnectedX } from '@tabler/icons-react';
import {
  Background,
  Controls,
  MiniMap,
  Panel,
  ReactFlow,
  ReactFlowProvider,
  useNodesState,
  useReactFlow,
  type Edge,
} from '@xyflow/react';
import {
  Box,
  Button,
  Center,
  Drawer,
  Loader,
  ScrollArea,
  Stack,
  Text,
  useComputedColorScheme,
  useMantineTheme,
} from '@mantine/core';
import { useDisclosure, useMediaQuery } from '@mantine/hooks';
import type { GraphDocument } from '@/api';
import { BuilderContext, groupIssues, type BuilderContextValue } from './BuilderContext';
import { EDGE_DEFAULTS, miniMapColor, nodeTypes, withEdgeStyle } from './flow';
import { fromDocument, type LayerFlowNode } from './graph';
import { Inspector } from './Inspector';
import { useAnalysis } from './useAnalysis';
import { useCatalog } from './useCatalog';

interface NetworkViewerProps {
  doc: GraphDocument;
  /** Tips in the inspector while no layer is selected (default: those for shared networks). */
  tips?: React.ReactNode[];
}

/**
 * A network on the canvas, read only: layers can be selected to see their settings, but not
 * moved or changed. Draws layers like the builder (same nodes, analysis and inspector).
 */
export function NetworkViewer({ doc, tips }: NetworkViewerProps) {
  return (
    <ReactFlowProvider>
      <Viewer doc={doc} tips={tips} />
    </ReactFlowProvider>
  );
}

const noop = () => undefined;

function Viewer({ doc, tips }: NetworkViewerProps) {
  const { catalog, error, reload } = useCatalog();
  const { analysis } = useAnalysis(doc);
  const flow = useReactFlow<LayerFlowNode, Edge>();
  const theme = useMantineTheme();
  const colorScheme = useComputedColorScheme('light');
  const wideScreen = useMediaQuery(`(min-width: ${theme.breakpoints.lg})`);
  const mediumScreen = useMediaQuery(`(min-width: ${theme.breakpoints.md})`);
  const [inspectorOpened, inspector] = useDisclosure(false);
  const canvasRef = useRef<HTMLDivElement>(null);

  const initial = useMemo(() => fromDocument(doc), [doc]);
  const [nodes, setNodes, onNodesChange] = useNodesState<LayerFlowNode>(initial.nodes);
  const edges = useMemo(() => withEdgeStyle(initial.edges), [initial]);
  const specs = useMemo(
    () => Object.fromEntries((catalog?.layers ?? []).map((spec) => [spec.type, spec])),
    [catalog]
  );
  const selected = nodes.find((node) => node.selected) ?? null;

  // Pinching zooms the network, not the page (Safari's gesture events, see the builder).
  const hasCanvas = Boolean(catalog);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return undefined;
    }
    const prevent = (event: Event) => event.preventDefault();
    canvas.addEventListener('gesturestart', prevent);
    canvas.addEventListener('gesturechange', prevent);
    return () => {
      canvas.removeEventListener('gesturestart', prevent);
      canvas.removeEventListener('gesturechange', prevent);
    };
  }, [hasCanvas]);

  const focusNode = useCallback(
    (id: string) => {
      const node = nodes.find((n) => n.id === id);
      if (!node) {
        return;
      }
      setNodes((current) => current.map((n) => ({ ...n, selected: n.id === id })));
      flow.setCenter(node.position.x + 110, node.position.y + 45, {
        zoom: Math.max(flow.getZoom(), 1),
        duration: 400,
      });
      inspector.close();
    },
    [nodes, setNodes, flow, inspector]
  );

  const context: BuilderContextValue = useMemo(
    () => ({
      specs,
      analysis,
      errorsByNode: groupIssues(analysis?.errors),
      warningsByNode: groupIssues(analysis?.warnings),
      readOnly: true,
      updateNode: noop,
      deleteNode: noop,
      duplicateNode: noop,
      focusNode,
    }),
    [specs, analysis, focusNode]
  );

  if (!catalog) {
    return (
      <Center style={{ flex: 1 }}>
        {error ? (
          <Stack align="center" maw={420} ta="center" p="md">
            <IconPlugConnectedX size={40} color="var(--mantine-color-red-6)" />
            <Text c="dimmed" size="sm">
              {error}
            </Text>
            <Button onClick={reload}>Try again</Button>
          </Stack>
        ) : (
          <Loader />
        )}
      </Center>
    );
  }

  const inspectorView = <Inspector node={selected} nodes={nodes} tips={tips} />;

  return (
    <BuilderContext.Provider value={context}>
      <Box style={{ flex: 1, display: 'flex', minHeight: 0 }}>
        <Box ref={canvasRef} style={{ flex: 1, position: 'relative', minWidth: 0 }}>
          <ReactFlow<LayerFlowNode, Edge>
            nodes={nodes}
            edges={edges}
            nodeTypes={nodeTypes}
            onNodesChange={onNodesChange}
            nodesDraggable={false}
            nodesConnectable={false}
            deleteKeyCode={null}
            defaultEdgeOptions={EDGE_DEFAULTS}
            colorMode={colorScheme}
            minZoom={0.15}
            maxZoom={2}
            fitView
            fitViewOptions={{ padding: 0.15, maxZoom: 1 }}
          >
            <Background gap={20} />
            <Controls showInteractive={false} />
            {mediumScreen && <MiniMap pannable zoomable nodeColor={miniMapColor(theme, specs)} />}
            <Panel position="top-right">
              <Button
                hiddenFrom="lg"
                variant="default"
                size="xs"
                leftSection={<IconAdjustmentsHorizontal size={14} />}
                onClick={inspector.open}
              >
                Properties
              </Button>
            </Panel>
          </ReactFlow>
        </Box>
        <Box
          visibleFrom="lg"
          w={330}
          style={{ borderLeft: '1px solid var(--mantine-color-default-border)', flexShrink: 0 }}
        >
          <ScrollArea h="100%">{inspectorView}</ScrollArea>
        </Box>
      </Box>
      <Drawer
        opened={inspectorOpened && !wideScreen}
        onClose={inspector.close}
        position="right"
        title="Properties"
        size={340}
      >
        {inspectorView}
      </Drawer>
    </BuilderContext.Provider>
  );
}
