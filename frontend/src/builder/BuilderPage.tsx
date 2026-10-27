import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { IconPlugConnectedX, IconSchool } from '@tabler/icons-react';
import {
  Background,
  Controls,
  MarkerType,
  MiniMap,
  Panel,
  ReactFlow,
  ReactFlowProvider,
  useEdgesState,
  useNodesState,
  useReactFlow,
  type Connection,
  type Edge,
  type XYPosition,
} from '@xyflow/react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Button,
  Center,
  Drawer,
  Group,
  Loader,
  Paper,
  ScrollArea,
  Stack,
  Text,
  Title,
  useComputedColorScheme,
  useMantineTheme,
} from '@mantine/core';
import { useDisclosure, useHotkeys, useMediaQuery } from '@mantine/hooks';
import { modals } from '@mantine/modals';
import { notifications } from '@mantine/notifications';
import {
  api,
  ApiError,
  type Example,
  type GenerateOptions,
  type GraphDocument,
  type LayerSpec,
} from '@/api';
import { useAuth } from '@/auth/AuthContext';
import { HEADER_HEIGHT } from '@/components/AppHeader';
import { categoryColor } from '@/theme';
import { BuilderContext, groupIssues, type BuilderContextValue } from './BuilderContext';
import { downloadText } from './download';
import { loadDraft, saveDraft, type Draft } from './draft';
import {
  autoLayout,
  createNode,
  edgeId,
  freePosition,
  fromDocument,
  LAYER_SPACING_X,
  newId,
  parseGraphFile,
  semanticKey,
  slugify,
  toDocument,
  wouldCreateCycle,
  type LayerFlowNode,
  type LayerNodeData,
} from './graph';
import { Inspector } from './Inspector';
import { LayerNode } from './LayerNode';
import { DRAG_MIME, Palette } from './Palette';
import { Toolbar } from './Toolbar';
import { useAnalysis } from './useAnalysis';
import { useCatalog } from './useCatalog';

const nodeTypes = { layer: LayerNode };

// The syntax highlighter is large, so the code preview is loaded on first use.
const CodeModal = lazy(() => import('./CodeModal'));

const EDGE_DEFAULTS = {
  type: 'smoothstep',
  markerEnd: { type: MarkerType.ArrowClosed, width: 18, height: 18 },
} as const;

function makeEdge(source: string, target: string): Edge {
  return { id: edgeId(source, target), source, target, ...EDGE_DEFAULTS };
}

function withEdgeStyle(edges: Edge[]): Edge[] {
  return edges.map((edge) => ({ ...edge, ...EDGE_DEFAULTS }));
}

export function BuilderPage() {
  return (
    <ReactFlowProvider>
      <Builder />
    </ReactFlowProvider>
  );
}

function Builder() {
  const { catalog, examples, error: catalogError, reload } = useCatalog();
  const { user } = useAuth();
  const navigate = useNavigate();
  const flow = useReactFlow<LayerFlowNode, Edge>();
  const theme = useMantineTheme();
  const colorScheme = useComputedColorScheme('light');
  const wideScreen = useMediaQuery(`(min-width: ${theme.breakpoints.lg})`);
  const mediumScreen = useMediaQuery(`(min-width: ${theme.breakpoints.md})`);

  // ---------------------------------------------------------------- state
  const initial = useMemo(() => {
    const draft = loadDraft();
    return draft ? { ...draft, ...fromDocument(draft.doc) } : null;
  }, []);
  const [nodes, setNodes, onNodesChange] = useNodesState<LayerFlowNode>(initial?.nodes ?? []);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>(withEdgeStyle(initial?.edges ?? []));
  const [name, setName] = useState(initial?.name ?? 'Untitled network');
  const [projectId, setProjectId] = useState<number | null>(initial?.projectId ?? null);
  const [savedKey, setSavedKey] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [codeOptions, setCodeOptions] = useState<GenerateOptions>({
    class_name: 'NeuralNetwork',
    include_main: true,
    include_training: true,
  });
  const [codeOpened, code] = useDisclosure(false);
  const [paletteOpened, palette] = useDisclosure(false);
  const [inspectorOpened, inspector] = useDisclosure(false);
  const [fitRequest, setFitRequest] = useState(0);
  const fileInput = useRef<HTMLInputElement>(null);
  const canvasRef = useRef<HTMLDivElement>(null);
  // Guards against double saves and against a save finishing after another network was loaded.
  const savingRef = useRef(false);
  const documentGeneration = useRef(0);
  const seeded = useRef(Boolean(initial));

  const specs = useMemo(
    () => Object.fromEntries((catalog?.layers ?? []).map((spec) => [spec.type, spec])),
    [catalog]
  );
  const doc = useMemo(() => toDocument(nodes, edges), [nodes, edges]);
  const { analysis, loading: analyzing, error: analysisError } = useAnalysis(doc);
  const selected = nodes.find((node) => node.selected) ?? null;
  const filename = slugify(name);
  const currentKey = `${name}\n${semanticKey(doc)}`;

  // ---------------------------------------------------------------- persistence
  const pendingDraft = useRef<Draft | null>(null);
  useEffect(() => {
    if (!seeded.current && nodes.length === 0) {
      return undefined;
    }
    pendingDraft.current = { doc, name, projectId };
    const timer = window.setTimeout(() => {
      saveDraft({ doc, name, projectId });
      pendingDraft.current = null;
    }, 400);
    return () => window.clearTimeout(timer);
  }, [doc, name, projectId]);

  // Write a pending draft immediately when leaving the page or closing the tab.
  useEffect(() => {
    const flush = () => {
      if (pendingDraft.current) {
        saveDraft(pendingDraft.current);
        pendingDraft.current = null;
      }
    };
    window.addEventListener('pagehide', flush);
    return () => {
      window.removeEventListener('pagehide', flush);
      flush();
    };
  }, []);

  useEffect(() => {
    if (fitRequest === 0) {
      return undefined;
    }
    const timer = window.setTimeout(
      () => flow.fitView({ padding: 0.15, maxZoom: 1, duration: 300 }),
      60
    );
    return () => window.clearTimeout(timer);
  }, [fitRequest, flow]);

  const loadDocument = useCallback(
    (graph: GraphDocument, newName: string, newProjectId: number | null) => {
      documentGeneration.current += 1;
      const loaded = fromDocument(graph);
      setNodes(loaded.nodes);
      setEdges(withEdgeStyle(loaded.edges));
      setName(newName);
      setProjectId(newProjectId);
      setSavedKey(newProjectId ? `${newName}\n${semanticKey(graph)}` : null);
      setFitRequest((n) => n + 1);
    },
    [setNodes, setEdges]
  );

  // First visit: show an example so there is something to play with.
  useEffect(() => {
    if (!seeded.current && examples.length > 0) {
      seeded.current = true;
      loadDocument(examples[0].graph, examples[0].name, null);
    }
  }, [examples, loadDocument]);

  const confirmReplace = (action: () => void) => {
    if (nodes.length === 0) {
      action();
      return;
    }
    modals.openConfirmModal({
      title: 'Replace the current network?',
      children: (
        <Text size="sm">
          The canvas will be replaced. Save first if you want to keep your current network.
        </Text>
      ),
      labels: { confirm: 'Replace', cancel: 'Cancel' },
      onConfirm: action,
    });
  };

  // ---------------------------------------------------------------- editing
  const selectOnly = useCallback(
    (id: string | null) =>
      setNodes((current) => current.map((node) => ({ ...node, selected: node.id === id }))),
    [setNodes]
  );

  const addLayer = useCallback(
    (spec: LayerSpec, dropPosition?: XYPosition) => {
      const anchor = dropPosition
        ? null
        : (nodes.find((node) => node.selected) ?? nodes[nodes.length - 1] ?? null);
      let position: XYPosition;
      if (dropPosition) {
        position = dropPosition;
      } else if (spec.max_inputs === 0) {
        const left = Math.min(0, ...nodes.map((node) => node.position.x));
        position = freePosition(nodes, { x: left, y: 0 });
      } else if (anchor) {
        position = freePosition(nodes, {
          x: anchor.position.x + LAYER_SPACING_X,
          y: anchor.position.y,
        });
      } else {
        position = { x: 0, y: 0 };
      }
      const node = { ...createNode(spec, position), selected: true };
      setNodes((current) => [...current.map((n) => ({ ...n, selected: false })), node]);
      if (anchor && !dropPosition && spec.max_inputs !== 0) {
        setEdges((current) => [...current, makeEdge(anchor.id, node.id)]);
      }
      palette.close();

      // Keep the new layer in view when it was appended outside the visible area.
      const rect = canvasRef.current?.getBoundingClientRect();
      if (!dropPosition && rect) {
        const topLeft = flow.flowToScreenPosition(position);
        const bottomRight = flow.flowToScreenPosition({ x: position.x + 220, y: position.y + 90 });
        const visible =
          topLeft.x >= rect.left &&
          topLeft.y >= rect.top &&
          bottomRight.x <= rect.right &&
          bottomRight.y <= rect.bottom;
        if (!visible) {
          flow.setCenter(position.x + 110, position.y + 45, {
            zoom: flow.getZoom(),
            duration: 300,
          });
        }
      }
    },
    [nodes, setNodes, setEdges, palette, flow]
  );

  const updateNode = useCallback(
    (id: string, patch: Partial<LayerNodeData>) =>
      setNodes((current) =>
        current.map((node) =>
          node.id === id ? { ...node, data: { ...node.data, ...patch } } : node
        )
      ),
    [setNodes]
  );

  const deleteNode = useCallback(
    (id: string) => {
      setNodes((current) => current.filter((node) => node.id !== id));
      setEdges((current) => current.filter((edge) => edge.source !== id && edge.target !== id));
    },
    [setNodes, setEdges]
  );

  const duplicateNode = useCallback(
    (id: string) => {
      const original = nodes.find((node) => node.id === id);
      if (!original) {
        return;
      }
      const copy: LayerFlowNode = {
        ...original,
        id: newId(),
        selected: true,
        position: freePosition(nodes, { x: original.position.x, y: original.position.y + 170 }),
        data: { ...original.data, params: { ...original.data.params }, name: undefined },
      };
      setNodes((current) => [...current.map((n) => ({ ...n, selected: false })), copy]);
    },
    [nodes, setNodes]
  );

  const focusNode = useCallback(
    (id: string) => {
      const node = nodes.find((n) => n.id === id);
      if (!node) {
        return;
      }
      selectOnly(id);
      flow.setCenter(node.position.x + 110, node.position.y + 45, {
        zoom: Math.max(flow.getZoom(), 1),
        duration: 400,
      });
      inspector.close();
    },
    [nodes, flow, selectOnly, inspector]
  );

  const onConnect = useCallback(
    (connection: Connection) => {
      const { source, target } = connection;
      if (!source || !target) {
        return;
      }
      setEdges((current) => {
        let next = current.filter((e) => !(e.source === source && e.target === target));
        // Layers with a single input: a new connection replaces the old one.
        if (specs[nodes.find((n) => n.id === target)?.data.type ?? '']?.max_inputs === 1) {
          next = next.filter((e) => e.target !== target);
        }
        return [...next, makeEdge(source, target)];
      });
    },
    [setEdges, specs, nodes]
  );

  const isValidConnection = useCallback(
    (connection: Edge | Connection) =>
      connection.source !== connection.target &&
      !wouldCreateCycle(edges, connection.source, connection.target),
    [edges]
  );

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      const spec = specs[event.dataTransfer.getData(DRAG_MIME)];
      if (spec) {
        const point = flow.screenToFlowPosition({ x: event.clientX, y: event.clientY });
        addLayer(spec, { x: point.x - 110, y: point.y - 40 });
      }
    },
    [specs, flow, addLayer]
  );

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  // ---------------------------------------------------------------- actions
  const showSummary = () => {
    selectOnly(null);
    if (!wideScreen) {
      inspector.open();
    }
  };

  const save = async () => {
    if (!user) {
      notifications.show({
        title: 'Log in to save',
        message: 'Saving networks needs a free account. Your work is kept while you log in.',
      });
      navigate('/login', { state: { from: '/' } });
      return;
    }
    if (savingRef.current) {
      return;
    }
    savingRef.current = true;
    const generation = documentGeneration.current;
    setSaving(true);
    const payload = { name: name.trim() || 'Untitled network', graph: doc };
    try {
      let project;
      try {
        project = projectId
          ? await api.projects.update(projectId, payload)
          : await api.projects.create(payload);
      } catch (error) {
        if (error instanceof ApiError && error.status === 404) {
          project = await api.projects.create(payload); // deleted elsewhere: save as new
        } else {
          throw error;
        }
      }
      if (generation === documentGeneration.current) {
        setProjectId(project.id);
        setSavedKey(`${name}\n${semanticKey(doc)}`);
      }
      notifications.show({ color: 'green', message: `Saved “${project.name}”.` });
    } catch (error) {
      notifications.show({
        color: 'red',
        title: 'Saving failed',
        message: (error as Error).message,
      });
    } finally {
      savingRef.current = false;
      setSaving(false);
    }
  };

  const exportPython = async () => {
    try {
      const result = await api.generate(doc, { ...codeOptions, filename });
      downloadText(result.filename, result.code, 'text/x-python');
    } catch (error) {
      if (
        error instanceof ApiError &&
        error.status === 400 &&
        (error.data as { analysis?: unknown })?.analysis
      ) {
        notifications.show({
          color: 'red',
          title: 'The network has problems',
          message: 'Fix the highlighted layers first. The list is shown in the side panel.',
        });
        showSummary();
      } else {
        notifications.show({
          color: 'red',
          title: 'Export failed',
          message: (error as Error).message,
        });
      }
    }
  };

  const exportJson = () =>
    downloadText(
      `${filename}.json`,
      `${JSON.stringify({ name, graph: doc }, null, 2)}\n`,
      'application/json'
    );

  const importJson = async (file: File) => {
    try {
      const parsed = parseGraphFile(await file.text());
      confirmReplace(() =>
        loadDocument(parsed.doc, parsed.name ?? file.name.replace(/\.json$/i, ''), null)
      );
    } catch (error) {
      notifications.show({
        color: 'red',
        title: 'Cannot open file',
        message: (error as Error).message,
      });
    }
  };

  const tidyUp = () => {
    setNodes((current) => autoLayout(current, edges));
    setFitRequest((n) => n + 1);
  };

  const loadExample = (example: Example) =>
    confirmReplace(() => loadDocument(example.graph, example.name, null));

  const newNetwork = () =>
    confirmReplace(() =>
      loadDocument({ version: 1, nodes: [], edges: [] }, 'Untitled network', null)
    );

  useHotkeys([
    ['mod+S', () => void save()],
    ['mod+E', () => void exportPython()],
  ]);

  // ---------------------------------------------------------------- render
  const context: BuilderContextValue = useMemo(
    () => ({
      specs,
      analysis,
      errorsByNode: groupIssues(analysis?.errors),
      warningsByNode: groupIssues(analysis?.warnings),
      updateNode,
      deleteNode,
      duplicateNode,
      focusNode,
    }),
    [specs, analysis, updateNode, deleteNode, duplicateNode, focusNode]
  );

  if (!catalog) {
    return (
      <Center h={`calc(100dvh - ${HEADER_HEIGHT}px)`}>
        {catalogError ? (
          <Stack align="center" maw={420} ta="center">
            <IconPlugConnectedX size={40} color="var(--mantine-color-red-6)" />
            <Title order={3}>Cannot reach the backend</Title>
            <Text c="dimmed" size="sm">
              {catalogError} If you run Brick DNN locally, start the API with{' '}
              <code>python manage.py runserver</code> in the <code>backend</code> folder.
            </Text>
            <Button onClick={reload}>Try again</Button>
          </Stack>
        ) : (
          <Loader />
        )}
      </Center>
    );
  }

  const paletteView = <Palette catalog={catalog} onAdd={(spec) => addLayer(spec)} />;
  const inspectorView = <Inspector node={selected} nodes={nodes} />;

  return (
    <BuilderContext.Provider value={context}>
      <Box style={{ display: 'flex', height: `calc(100dvh - ${HEADER_HEIGHT}px)` }}>
        <Box
          visibleFrom="md"
          w={230}
          style={{ borderRight: '1px solid var(--mantine-color-default-border)', flexShrink: 0 }}
        >
          {paletteView}
        </Box>

        <Box style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
          <Toolbar
            name={name}
            onNameChange={setName}
            analysis={analysis}
            analyzing={analyzing}
            analysisError={analysisError}
            empty={nodes.length === 0}
            examples={examples}
            saving={saving}
            isSaved={projectId !== null && savedKey === currentKey}
            onLoadExample={loadExample}
            onNew={newNetwork}
            onImport={() => fileInput.current?.click()}
            onExportJson={exportJson}
            onTidy={tidyUp}
            onSave={() => void save()}
            onShowCode={code.open}
            onExportPython={() => void exportPython()}
            onShowSummary={showSummary}
            onOpenPalette={palette.open}
            onOpenInspector={inspector.open}
          />
          <Box
            ref={canvasRef}
            style={{ flex: 1, position: 'relative' }}
            onDrop={onDrop}
            onDragOver={onDragOver}
          >
            <ReactFlow<LayerFlowNode, Edge>
              nodes={nodes}
              edges={edges}
              nodeTypes={nodeTypes}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onConnect={onConnect}
              isValidConnection={isValidConnection}
              defaultEdgeOptions={EDGE_DEFAULTS}
              deleteKeyCode={['Backspace', 'Delete']}
              colorMode={colorScheme}
              minZoom={0.15}
              maxZoom={2}
              fitView
              fitViewOptions={{ padding: 0.15, maxZoom: 1 }}
            >
              <Background gap={20} />
              <Controls />
              {mediumScreen && (
                <MiniMap
                  pannable
                  zoomable
                  nodeColor={(node) =>
                    theme.colors[
                      categoryColor(specs[(node.data as LayerNodeData).type]?.category)
                    ]?.[6] ?? theme.colors.gray[6]
                  }
                />
              )}
              {nodes.length === 0 && (
                <Panel position="top-center">
                  <Paper withBorder shadow="sm" p="lg" mt={60} maw={420} ta="center">
                    <Title order={4}>Start building</Title>
                    <Text size="sm" c="dimmed" mt={6}>
                      Drag an <b>Input</b> layer from the palette onto the canvas, then add layers
                      by clicking them. Or start from an example:
                    </Text>
                    <Group justify="center" mt="md" gap="xs">
                      {examples.slice(0, 3).map((example) => (
                        <Button
                          key={example.id}
                          size="xs"
                          variant="light"
                          leftSection={<IconSchool size={14} />}
                          onClick={() => loadExample(example)}
                        >
                          {example.name}
                        </Button>
                      ))}
                    </Group>
                  </Paper>
                </Panel>
              )}
            </ReactFlow>
          </Box>
        </Box>

        <Box
          visibleFrom="lg"
          w={330}
          style={{ borderLeft: '1px solid var(--mantine-color-default-border)', flexShrink: 0 }}
        >
          <ScrollArea h="100%">{inspectorView}</ScrollArea>
        </Box>
      </Box>

      <Drawer opened={paletteOpened} onClose={palette.close} title="Layers" size={280}>
        <Box h="calc(100dvh - 90px)">{paletteView}</Box>
      </Drawer>
      <Drawer
        opened={inspectorOpened && !wideScreen}
        onClose={inspector.close}
        position="right"
        title="Properties"
        size={340}
      >
        {inspectorView}
      </Drawer>
      {codeOpened && (
        <Suspense fallback={null}>
          <CodeModal
            opened={codeOpened}
            onClose={code.close}
            doc={doc}
            filename={filename}
            options={codeOptions}
            onOptionsChange={setCodeOptions}
          />
        </Suspense>
      )}
      <input
        ref={fileInput}
        type="file"
        accept="application/json,.json"
        hidden
        onChange={(event) => {
          const file = event.currentTarget.files?.[0];
          event.currentTarget.value = '';
          if (file) {
            void importJson(file);
          }
        }}
      />
    </BuilderContext.Provider>
  );
}
