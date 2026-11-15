import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { IconPlugConnectedX } from '@tabler/icons-react';
import {
  Background,
  Controls,
  MiniMap,
  Panel,
  ReactFlow,
  ReactFlowProvider,
  useEdgesState,
  useNodesState,
  useReactFlow,
  type Connection,
  type Edge,
  type FinalConnectionState,
  type XYPosition,
} from '@xyflow/react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  Box,
  Button,
  Center,
  Drawer,
  Loader,
  ScrollArea,
  Stack,
  Text,
  Title,
  useComputedColorScheme,
  useMantineTheme,
} from '@mantine/core';
import { useDisclosure, useHotkeys, useMediaQuery, useReducedMotion } from '@mantine/hooks';
import { modals } from '@mantine/modals';
import { notifications } from '@mantine/notifications';
import {
  api,
  ApiError,
  type Example,
  type GenerateOptions,
  type GraphDocument,
  type LayerSpec,
  type Params,
  type Project,
  type ProjectConflict,
} from '@/api';
import { useAuth } from '@/auth/AuthContext';
import { Collaboration } from '@/collab/Collaboration';
import { CollabProvider } from '@/collab/context';
import { HEADER_HEIGHT } from '@/components/AppHeader';
import { DataFlowTour } from '@/explain/DataFlowTour';
import { i18n, Trans, useTranslation } from '@/i18n';
import { useLiveBuilder } from '@/live/useLiveBuilder';
import { ShareDialog } from '@/sharing/ShareDialog';
import { useDatasetFit } from '@/training/useDatasetFit';
import { useTraining } from '@/training/useTraining';
import {
  focusLayer,
  layerName,
  LiveRegion,
  useAnnouncer,
  useCanvasLabels,
  useLabelledEdges,
  useLabelledNodes,
} from './a11y';
import { BuilderContext, groupIssues, type BuilderContextValue } from './BuilderContext';
import { ConflictDialog, type ConflictChoice } from './ConflictDialog';
import { downloadText } from './download';
import { loadDraft, saveDraft, type Draft } from './draft';
import {
  EDGE_DEFAULTS,
  makeEdge,
  miniMapColor,
  nodeTypes,
  useDecoratedEdges,
  withEdgeStyle,
} from './flow';
import {
  autoLayout,
  copySelection,
  createNode,
  freePosition,
  fromDocument,
  LAYER_SPACING_X,
  LAYER_SPACING_Y,
  newId,
  parseGraphFile,
  pasteClipboard,
  semanticKey,
  slugify,
  toDocument,
  wouldCreateCycle,
  type Clipboard,
  type LayerFlowNode,
  type LayerNodeData,
} from './graph';
import { Inspector } from './Inspector';
import { connectionProblem, inputSlots, networkFamily, outputKind, slotFor } from './kinds';
import { EmptyCanvas, guideSeen, guideStartsByItself, OnboardingGuide } from './Onboarding';
import { Palette } from './Palette';
import type { Point } from './pointerDrag';
import { Toolbar } from './Toolbar';
import { useAnalysis } from './useAnalysis';
import { useCatalog } from './useCatalog';
import { openProjectOf, useOpenProject, type OpenProject } from './useOpenProject';

// The syntax highlighter is large, so the code preview is loaded on first use.
const CodeModal = lazy(() => import('./CodeModal'));
// Same for the training panel (charts, forms); it stays mounted once it was opened.
const TrainingDrawer = lazy(() => import('@/training/TrainingDrawer'));

/** Name and graph without layout: equal keys mean there is nothing to save. */
function contentKey(name: string, graph: GraphDocument): string {
  const { nodes, edges } = fromDocument(graph);
  return `${name}\n${semanticKey(toDocument(nodes, edges))}`;
}

interface SaveOptions {
  /** Save over this version (after a conflict) instead of the one the canvas is based on. */
  baseVersion?: number;
  /** Save as a new network named "<name> (copy)". */
  asCopy?: boolean;
}

export function BuilderPage() {
  return (
    <ReactFlowProvider>
      {/* History and comments of the open network, for the canvas, inspector and panels. */}
      <CollabProvider>
        <Builder />
      </CollabProvider>
    </ReactFlowProvider>
  );
}

function Builder() {
  const { t } = useTranslation();
  const { catalog, examples, error: catalogError, reload } = useCatalog();
  const { user, ready } = useAuth();
  const navigate = useNavigate();
  const flow = useReactFlow<LayerFlowNode, Edge>();
  const theme = useMantineTheme();
  const colorScheme = useComputedColorScheme('light');
  const wideScreen = useMediaQuery(`(min-width: ${theme.breakpoints.lg})`);
  const mediumScreen = useMediaQuery(`(min-width: ${theme.breakpoints.md})`);
  // Without animated zooming when the system asks for less motion.
  const reduceMotion = useReducedMotion();
  // What an action changed, for screen readers (a polite live region).
  const [announcement, announce] = useAnnouncer();

  // ---------------------------------------------------------------- state
  const initial = useMemo(() => {
    const draft = loadDraft();
    if (!draft) {
      return null;
    }
    const loaded = fromDocument(draft.doc);
    const project: OpenProject | null =
      draft.projectId === null
        ? null
        : {
            id: draft.projectId,
            version: draft.version ?? null,
            role: draft.role ?? null,
            info: null,
          };
    // Live changes the server had not confirmed when the page was closed.
    const seed = project && draft.pending ? { projectId: project.id, ops: draft.pending } : null;
    return { ...draft, ...loaded, key: contentKey(draft.name, draft.doc), project, seed };
  }, []);
  const [nodes, setNodes, onNodesChange] = useNodesState<LayerFlowNode>(initial?.nodes ?? []);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>(withEdgeStyle(initial?.edges ?? []));
  const [name, setName] = useState(initial?.name ?? i18n.t('builder.untitled'));
  // The saved project the canvas belongs to: its version, the user's role and the sharing.
  const { project, setProject, attach, detach, members, applySharing, readOnly } = useOpenProject(
    initial?.project ?? null
  );
  const projectId = project?.id ?? null;
  const [savedKey, setSavedKey] = useState<string | null>(initial?.saved ? initial.key : null);
  const [saving, setSaving] = useState(false);
  const [conflict, setConflict] = useState<Project | null>(null);
  const [shareOpened, setShareOpened] = useState(false);
  const [codeOptions, setCodeOptions] = useState<GenerateOptions>({
    class_name: 'NeuralNetwork',
    include_main: true,
    include_training: true,
  });
  const [codeOpened, code] = useDisclosure(false);
  const [paletteOpened, palette] = useDisclosure(false);
  const [inspectorOpened, inspector] = useDisclosure(false);
  const [fitRequest, setFitRequest] = useState(0);
  const [guideOpened, setGuideOpened] = useState(false);
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
  // Connections end at the input that takes what they carry; kernel connections are dashed.
  const displayEdges = useDecoratedEdges(edges, nodes, specs);
  const { analysis, loading: analyzing, error: analysisError } = useAnalysis(doc);
  // Trainings of a saved network belong to it: its members see them (viewers cannot train).
  const training = useTraining(Boolean(user), {
    username: user?.username ?? null,
    project: project && {
      id: project.id,
      role: project.role,
      shared: Boolean(project.info?.shared),
    },
  });
  const [trainingLoaded, setTrainingLoaded] = useState(false);
  const location = useLocation();
  // "Train with this data" in the Data tab opens the training panel here, with the exported
  // dataset selected (also when the builder is open already, e.g. from a notification); the
  // agent activity page opens a run (`run`) or selects a dataset the same way. The state is
  // cleared, so going back or reloading does not open it again.
  const { setSource: setTrainingSource, reloadDatasets } = training.data;
  useEffect(() => {
    const state = location.state as {
      openTraining?: boolean;
      dataset?: unknown;
      run?: unknown;
    } | null;
    if (!state?.openTraining) {
      return;
    }
    const { dataset, run } = state;
    if (typeof dataset === 'number') {
      setTrainingSource((current) => ({ ...current, tab: 'upload', upload: dataset }));
      reloadDatasets();
    }
    if (typeof run === 'string') {
      training.openRun(run);
    } else {
      training.open();
    }
    navigate(location.pathname, { replace: true, state: null });
  }, [
    location.state,
    location.pathname,
    training.open,
    training.openRun,
    setTrainingSource,
    reloadDatasets,
    navigate,
  ]);
  useEffect(() => {
    if (training.opened) {
      setTrainingLoaded(true);
    }
  }, [training.opened]);
  const selected = nodes.find((node) => node.selected) ?? null;
  const filename = slugify(name);
  const currentKey = `${name}\n${semanticKey(doc)}`;
  const isSaved = projectId !== null && savedKey === currentKey;
  const currentKeyRef = useRef(currentKey);
  currentKeyRef.current = currentKey;

  // Live editing of a saved project: changes go out as they happen, others' changes come in.
  const live = useLiveBuilder({
    enabled: ready && Boolean(user),
    user,
    project,
    setProject,
    detach,
    readOnly,
    nodes,
    edges,
    name,
    setNodes,
    setEdges,
    setName,
    contentKey: currentKey,
    isSaved,
    seed: initial?.seed,
    onSelectedRemoved: (by) => {
      inspector.close();
      notifications.show({
        message: i18n.t('builder.page.selectedRemoved', { user: by.username }),
      });
    },
  });

  // ---------------------------------------------------------------- persistence
  const pendingDraft = useRef<(() => Draft) | null>(null);
  const version = project?.version ?? null;
  const role = project?.role ?? null;
  const { tracking, pendingCount, pendingOps } = live;
  useEffect(() => {
    if (!seeded.current && nodes.length === 0) {
      return undefined;
    }
    // Live, the draft is saved once the server has confirmed every change; the changes it
    // has not confirmed yet are kept with it (read when it is written, they change often).
    const draft = (): Draft => {
      const pending = tracking ? pendingOps() : [];
      return {
        doc,
        name,
        projectId,
        version,
        role,
        saved: tracking ? pending.length === 0 : isSaved,
        ...(pending.length > 0 ? { pending } : {}),
      };
    };
    pendingDraft.current = draft;
    const timer = window.setTimeout(() => {
      saveDraft(draft());
      pendingDraft.current = null;
    }, 400);
    return () => window.clearTimeout(timer);
  }, [doc, name, projectId, version, role, isSaved, tracking, pendingCount, pendingOps]);

  // Live, what the server has confirmed is saved (e.g. when the user logs out afterwards).
  useEffect(() => {
    if (tracking && pendingCount === 0) {
      setSavedKey(currentKey);
    }
  }, [tracking, pendingCount, currentKey]);

  // Write a pending draft immediately when leaving the page or closing the tab.
  useEffect(() => {
    const flush = () => {
      if (pendingDraft.current) {
        saveDraft(pendingDraft.current());
        pendingDraft.current = null;
      }
    };
    window.addEventListener('pagehide', flush);
    return () => {
      window.removeEventListener('pagehide', flush);
      flush();
    };
  }, []);

  // Pinching the canvas zooms the network. Safari's own gesture events would zoom the page as
  // well (touch-action on the canvas is not enough for older iOS versions).
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

  useEffect(() => {
    if (fitRequest === 0) {
      return undefined;
    }
    const timer = window.setTimeout(
      () => flow.fitView({ padding: 0.15, maxZoom: 1, duration: reduceMotion ? 0 : 300 }),
      60
    );
    return () => window.clearTimeout(timer);
  }, [fitRequest, flow, reduceMotion]);

  /** Replace the canvas. `source` is the saved project it shows (null: a new network). */
  const loadDocument = useCallback(
    (graph: GraphDocument, newName: string, source: Project | null) => {
      documentGeneration.current += 1;
      const loaded = fromDocument(graph);
      setNodes(loaded.nodes);
      setEdges(withEdgeStyle(loaded.edges));
      setName(newName);
      if (source) {
        attach(source);
      } else {
        detach();
      }
      setSavedKey(source ? contentKey(newName, graph) : null);
      setFitRequest((n) => n + 1);
    },
    [setNodes, setEdges, attach, detach]
  );

  const loadProject = useCallback(
    (source: Project) => loadDocument(source.graph, source.name, source),
    [loadDocument]
  );

  // First visit: show an example so there is something to play with.
  useEffect(() => {
    if (!seeded.current && examples.length > 0) {
      seeded.current = true;
      loadDocument(examples[0].graph, examples[0].name, null);
    }
  }, [examples, loadDocument]);

  // First visit in this browser: the guide, once the builder is on the screen.
  const guideStarted = useRef(false);
  useEffect(() => {
    if (guideStarted.current || !catalog || readOnly || !guideStartsByItself() || guideSeen()) {
      return;
    }
    const timer = window.setTimeout(() => {
      guideStarted.current = true;
      setGuideOpened(true);
    }, 700);
    return () => window.clearTimeout(timer);
  }, [catalog, readOnly]);

  // A saved project from the draft (opened in "My networks", or after a reload): get the
  // user's role, the sharing state and the latest version from the server.
  const refreshed = useRef(false);
  useEffect(() => {
    const opened = initial?.project;
    if (refreshed.current || !ready || !user || !initial || !opened) {
      return;
    }
    refreshed.current = true;
    const generation = documentGeneration.current;
    const stillOpen = () => generation === documentGeneration.current;
    api.projects
      .get(opened.id)
      .then((server) => {
        if (!stillOpen()) {
          return;
        }
        if (live.isTracking()) {
          // Live already: the connection brought the latest state, keep only the details.
          const { info } = openProjectOf(server);
          setProject((open) => (open?.id === server.id ? { ...open, info } : open));
          return;
        }
        const current = currentKeyRef.current;
        if (current === contentKey(server.name, server.graph)) {
          attach(server);
          setSavedKey(current);
          return;
        }
        const changedBy = server.updated_by ?? i18n.t('builder.page.someone');
        const newer = opened.version !== null && opened.version !== server.version;
        if (initial.saved && current === initial.key) {
          // No local changes: show the latest version.
          loadProject(server);
          if (newer) {
            notifications.show({
              message: i18n.t('builder.page.latestVersion', { user: changedBy }),
            });
          }
        } else if (server.role === 'viewer') {
          detach();
          notifications.show({
            title: i18n.t('builder.page.viewOnlyNow', { name: server.name }),
            message: i18n.t('builder.page.unsavedStayOpen'),
          });
        } else {
          // Keep the unsaved changes, with the version they are based on.
          setProject({ ...openProjectOf(server), version: opened.version ?? server.version });
          if (newer) {
            notifications.show({
              title: i18n.t('builder.page.changedBy', { user: changedBy, name: server.name }),
              message: i18n.t('builder.page.olderVersion'),
            });
          }
        }
      })
      .catch((error) => {
        if (stillOpen() && error instanceof ApiError && [403, 404].includes(error.status)) {
          detach();
          notifications.show({
            id: 'no-access',
            color: 'yellow',
            title: i18n.t('builder.page.noAccess'),
            message: i18n.t('builder.page.noAccessText', { name: initial.name }),
          });
        }
      });
  }, [ready, user, initial, attach, detach, setProject, loadProject, live.isTracking]);

  const confirmReplace = (action: () => void) => {
    // Viewers cannot have unsaved changes; live, all changes are saved (unless offline).
    if (nodes.length === 0 || readOnly || live.indicator === 'live') {
      action();
      return;
    }
    modals.openConfirmModal({
      title: i18n.t('builder.page.replaceTitle'),
      children: <Text size="sm">{i18n.t('builder.page.replaceText')}</Text>,
      labels: { confirm: i18n.t('builder.page.replace'), cancel: i18n.t('common.cancel') },
      onConfirm: action,
    });
  };

  // ---------------------------------------------------------------- editing
  const selectOnly = useCallback(
    (id: string | null) =>
      setNodes((current) => current.map((node) => ({ ...node, selected: node.id === id }))),
    [setNodes]
  );

  /** The type of a layer on the canvas ('' if there is none with this id). */
  const typeOf = useCallback(
    (id: string) => nodes.find((node) => node.id === id)?.data.type ?? '',
    [nodes]
  );

  /** Whether `target` can take one more kernel (a Gaussian process takes one). */
  const takesAnotherKernel = useCallback(
    (target: LayerFlowNode) => {
      const slot = slotFor(inputSlots(specs[target.data.type]), 'kernel');
      if (!slot || slot.kind !== 'kernel') {
        return false;
      }
      const kernels = edges.filter(
        (edge) => edge.target === target.id && outputKind(specs[typeOf(edge.source)]) === 'kernel'
      );
      return slot.max === null || kernels.length < slot.max;
    },
    [edges, specs, typeOf]
  );

  const addLayer = useCallback(
    (spec: LayerSpec, dropPosition?: XYPosition) => {
      const anchor = dropPosition
        ? null
        : (nodes.find((node) => node.selected) ?? nodes[nodes.length - 1] ?? null);
      // A new layer is appended after the anchor if it takes what the anchor puts out. A
      // kernel has no inputs: it goes into the anchor if that takes kernels (a kernel sum,
      // a Gaussian process without a kernel).
      const appends =
        anchor !== null &&
        inputSlots(spec).length > 0 &&
        connectionProblem(specs[anchor.data.type], spec) === null;
      const feeds =
        anchor !== null && !appends && outputKind(spec) === 'kernel' && takesAnotherKernel(anchor);
      let position: XYPosition;
      if (dropPosition) {
        position = dropPosition;
      } else if (anchor && feeds) {
        position = freePosition(nodes, {
          x: anchor.position.x - LAYER_SPACING_X,
          y: anchor.position.y + LAYER_SPACING_Y,
        });
      } else if (inputSlots(spec).length === 0) {
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
      if (anchor && appends) {
        setEdges((current) => [...current, makeEdge(anchor.id, node.id)]);
      } else if (anchor && feeds) {
        setEdges((current) => [...current, makeEdge(node.id, anchor.id)]);
      }
      palette.close();
      if (anchor && appends) {
        announce(
          i18n.t('a11y.live.addedAfter', { name: spec.label, after: layerName(anchor, specs) })
        );
      } else if (anchor && feeds) {
        announce(
          i18n.t('a11y.live.addedInto', { name: spec.label, into: layerName(anchor, specs) })
        );
      } else {
        announce(i18n.t('a11y.live.added', { name: spec.label }));
      }

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
            duration: reduceMotion ? 0 : 300,
          });
        }
      }
    },
    [nodes, specs, takesAnotherKernel, setNodes, setEdges, palette, flow, announce, reduceMotion]
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

  const patchParams = useCallback(
    (id: string, type: string, patch: Params): Params | null => {
      const node = nodes.find((n) => n.id === id && n.data.type === type);
      if (!node || readOnly) {
        return null;
      }
      updateNode(id, { params: { ...node.data.params, ...patch } });
      return node.data.params;
    },
    [nodes, readOnly, updateNode]
  );

  const deleteNode = useCallback(
    (id: string) => {
      setNodes((current) => current.filter((node) => node.id !== id));
      setEdges((current) => current.filter((edge) => edge.source !== id && edge.target !== id));
    },
    [setNodes, setEdges]
  );

  /**
   * Deletes a layer the user chose to delete (properties panel), says so and puts the focus
   * on a neighbour on the canvas, so that it is not lost with the panel's delete button.
   */
  const removeLayer = useCallback(
    (id: string) => {
      const node = nodes.find((n) => n.id === id);
      if (!node) {
        return;
      }
      const neighbour =
        edges.find((edge) => edge.target === id)?.source ??
        edges.find((edge) => edge.source === id)?.target ??
        nodes.find((n) => n.id !== id)?.id;
      deleteNode(id);
      announce(i18n.t('a11y.live.deleted', { name: layerName(node, specs) }));
      if (neighbour) {
        focusLayer(neighbour);
      }
    },
    [nodes, edges, specs, deleteNode, announce]
  );

  /** Deleted on the canvas (the Delete key, or cut): say what went. */
  const onCanvasDelete = useCallback(
    ({ nodes: gone, edges: cut }: { nodes: LayerFlowNode[]; edges: Edge[] }) => {
      const name = (id: string) =>
        layerName(
          nodes.find((n) => n.id === id),
          specs
        );
      if (gone.length === 1) {
        announce(i18n.t('a11y.live.deleted', { name: layerName(gone[0], specs) }));
      } else if (gone.length > 1) {
        announce(i18n.t('a11y.live.deletedMany', { count: gone.length }));
      } else if (cut.length === 1) {
        announce(
          i18n.t('a11y.live.disconnected', {
            source: name(cut[0].source),
            target: name(cut[0].target),
          })
        );
      }
      // The focus was on what is gone: on to a neighbour that stays.
      const removed = new Set(gone.map((n) => n.id));
      const neighbour = cut
        .flatMap((edge) => [edge.source, edge.target])
        .find((id) => !removed.has(id));
      if (neighbour) {
        focusLayer(neighbour);
      }
    },
    [nodes, specs, announce]
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

  // Ctrl+C / Ctrl+X / Ctrl+V: copy, cut and paste the selected bricks with the connections
  // between them. The clipboard lives in this tab; each paste lands a bit further down.
  const clipboard = useRef<{ clip: Clipboard; pastes: number } | null>(null);
  const textSelected = () => Boolean(window.getSelection()?.toString());
  const copyBricks = useCallback(
    (event?: KeyboardEvent) => {
      if (textSelected()) {
        return false; // the browser copies the selected text
      }
      const clip = copySelection(nodes, edges);
      if (!clip) {
        return false;
      }
      event?.preventDefault();
      clipboard.current = { clip, pastes: 0 };
      return true;
    },
    [nodes, edges]
  );
  const cutBricks = useCallback(
    (event?: KeyboardEvent) => {
      if (readOnly || !copyBricks(event)) {
        return;
      }
      const cut = new Set(clipboard.current!.clip.nodes.map((node) => node.id));
      setNodes((current) => current.filter((node) => !cut.has(node.id)));
      setEdges((current) =>
        current.filter((edge) => !cut.has(edge.source) && !cut.has(edge.target))
      );
    },
    [readOnly, copyBricks, setNodes, setEdges]
  );
  const pasteBricks = useCallback(
    (event?: KeyboardEvent) => {
      const stored = clipboard.current;
      if (readOnly || !stored) {
        return;
      }
      event?.preventDefault();
      stored.pastes += 1;
      const step = 40 * stored.pastes;
      const pasted = pasteClipboard(stored.clip, { x: step, y: step });
      setNodes((current) => [
        ...current.map((node) => (node.selected ? { ...node, selected: false } : node)),
        ...pasted.nodes,
      ]);
      setEdges((current) => [
        ...current.map((edge) => (edge.selected ? { ...edge, selected: false } : edge)),
        ...pasted.edges,
      ]);
    },
    [readOnly, setNodes, setEdges]
  );

  /** Put a network on the (empty) canvas without leaving the open project. */
  const startNetwork = useCallback(
    (graph: GraphDocument) => {
      const loaded = fromDocument(graph);
      setNodes(loaded.nodes);
      setEdges(withEdgeStyle(loaded.edges));
      setFitRequest((n) => n + 1);
    },
    [setNodes, setEdges]
  );

  // The Input layer follows the dataset chosen for training (not for viewers).
  const datasetFit = useDatasetFit({
    training,
    loggedIn: Boolean(user),
    readOnly,
    doc,
    nodes,
    analysis,
    analyzing,
    specs,
    // On a first visit an example is placed on the canvas once the catalogue is loaded.
    canvasReady: Boolean(catalog) && seeded.current,
    selectedInput: selected?.data.type === 'Input' ? selected.id : null,
    updateNode,
    deleteNode,
    startNetwork,
  });

  const focusNode = useCallback(
    (id: string) => {
      const node = nodes.find((n) => n.id === id);
      if (!node) {
        return;
      }
      selectOnly(id);
      flow.setCenter(node.position.x + 110, node.position.y + 45, {
        zoom: Math.max(flow.getZoom(), 1),
        duration: reduceMotion ? 0 : 400,
      });
      inspector.close();
      // Keyboard users go on from the layer (the list they came from may be gone).
      focusLayer(id);
    },
    [nodes, flow, selectOnly, inspector, reduceMotion]
  );

  const onConnect = useCallback(
    (connection: Connection) => {
      const { source, target } = connection;
      if (!source || !target) {
        return;
      }
      const slots = inputSlots(specs[typeOf(target)]);
      const kind = outputKind(specs[typeOf(source)]);
      const slot = slotFor(slots, kind, slots.length > 1 ? connection.targetHandle : null);
      setEdges((current) => {
        let next = current.filter((e) => !(e.source === source && e.target === target));
        // An input that takes one connection (e.g. of most layers, or the features of a
        // Gaussian process): a new connection replaces the old one.
        if (slot?.max === 1) {
          next = next.filter(
            (e) =>
              e.target !== target ||
              (slots.length > 1 && outputKind(specs[typeOf(e.source)]) !== slot.kind)
          );
        }
        return [...next, makeEdge(source, target)];
      });
      const name = (id: string) =>
        layerName(
          nodes.find((n) => n.id === id),
          specs
        );
      announce(i18n.t('a11y.live.connected', { source: name(source), target: name(target) }));
    },
    [setEdges, specs, typeOf, nodes, announce]
  );

  const isValidConnection = useCallback(
    (connection: Edge | Connection) =>
      connection.source !== connection.target &&
      !wouldCreateCycle(edges, connection.source, connection.target) &&
      // A kernel into a data input or data into a kernel input (the analysis checks the rest).
      connectionProblem(
        specs[typeOf(connection.source)],
        specs[typeOf(connection.target)],
        connection.targetHandle
      ) === null,
    [edges, specs, typeOf]
  );

  /** Whether `source` can be connected to `target` (not yet connected, the right kinds). */
  const canConnect = useCallback(
    (source: string, target: string) =>
      !edges.some((edge) => edge.source === source && edge.target === target) &&
      isValidConnection({ source, target, sourceHandle: null, targetHandle: null }),
    [edges, isValidConnection]
  );

  /** Connect two layers without dragging (the properties panel's "Connections"). */
  const connectLayers = useCallback(
    (source: string, target: string) => {
      if (!readOnly && canConnect(source, target)) {
        onConnect({ source, target, sourceHandle: null, targetHandle: null });
      }
    },
    [readOnly, canConnect, onConnect]
  );

  const disconnectLayers = useCallback(
    (source: string, target: string) => {
      if (readOnly) {
        return;
      }
      setEdges((current) =>
        current.filter((edge) => !(edge.source === source && edge.target === target))
      );
      const name = (id: string) =>
        layerName(
          nodes.find((n) => n.id === id),
          specs
        );
      announce(i18n.t('a11y.live.disconnected', { source: name(source), target: name(target) }));
    },
    [readOnly, setEdges, nodes, specs, announce]
  );

  /** Says why a connection of the wrong kind was refused. */
  const onConnectEnd = useCallback(
    (_event: MouseEvent | TouchEvent, state: FinalConnectionState) => {
      const { fromNode, toNode, fromHandle, toHandle } = state;
      if (state.isValid !== false || !fromNode || !toNode || !fromHandle || !toHandle) {
        return;
      }
      const fromSource = fromHandle.type === 'source';
      const source = fromSource ? fromNode : toNode;
      const target = fromSource ? toNode : fromNode;
      const problem = connectionProblem(
        specs[typeOf(source.id)],
        specs[typeOf(target.id)],
        (fromSource ? toHandle : fromHandle).id
      );
      if (problem) {
        notifications.show({ id: `connection:${problem}`, color: 'yellow', message: problem });
      }
    },
    [specs, typeOf]
  );

  const dropLayer = useCallback(
    (spec: LayerSpec, point: Point) => {
      // Only drops on the canvas count (not on the sidebars, the toolbar or a drawer).
      const target = document.elementFromPoint(point.x, point.y);
      if (!target || !canvasRef.current?.contains(target)) {
        return;
      }
      const position = flow.screenToFlowPosition(point);
      addLayer(spec, { x: position.x - 110, y: position.y - 40 });
    },
    [flow, addLayer]
  );

  // ---------------------------------------------------------------- actions
  const showSummary = () => {
    selectOnly(null);
    if (!wideScreen) {
      inspector.open();
    }
  };

  /** Save to the account. Resolves to the saved project, or null if it was not saved. */
  const save = async ({ baseVersion, asCopy = false }: SaveOptions = {}) => {
    if (live.tracking && !asCopy) {
      live.flush(); // Live, changes are saved as they happen.
      return null;
    }
    if (!user) {
      notifications.show({
        title: i18n.t('builder.page.loginToSave'),
        message: i18n.t('builder.page.loginToSaveText'),
      });
      navigate('/login', { state: { from: '/' } });
      return null;
    }
    if (readOnly && !asCopy) {
      notifications.show({
        title: i18n.t('builder.page.viewOnly'),
        message: i18n.t('builder.page.viewOnlyText'),
      });
      return null;
    }
    if (savingRef.current) {
      return null;
    }
    savingRef.current = true;
    const generation = documentGeneration.current;
    setSaving(true);
    const trimmed = name.trim() || i18n.t('builder.untitled');
    const payload = { name: asCopy ? `${trimmed} (copy)` : trimmed, graph: doc };
    let savedAsNew = false;
    try {
      let saved: Project;
      if (project && !asCopy) {
        const base = baseVersion ?? project.version;
        try {
          saved = await api.projects.update(
            project.id,
            base === null ? payload : { ...payload, version: base }
          );
        } catch (error) {
          if (!(error instanceof ApiError) || ![403, 404, 409].includes(error.status)) {
            throw error;
          }
          if (error.status === 409) {
            // Someone else saved in the meantime: let the user choose.
            const current =
              (error.data as Partial<ProjectConflict> | null)?.project ??
              (await api.projects.get(project.id));
            if (generation === documentGeneration.current) {
              setConflict(current);
            }
            return null;
          }
          if (error.status === 403) {
            // Only allowed to view now: the changes can still be saved as a new network.
            if (generation === documentGeneration.current) {
              detach();
            }
            notifications.show({
              color: 'yellow',
              title: i18n.t('builder.page.viewOnlyNow', { name: project.info?.name ?? trimmed }),
              message: i18n.t('builder.page.changesStayOpen'),
            });
            return null;
          }
          // Deleted, or no longer shared with this user: save as a new network.
          saved = await api.projects.create(payload);
          savedAsNew = true;
        }
      } else {
        saved = await api.projects.create(payload);
      }
      if (generation === documentGeneration.current) {
        attach(saved);
        if (asCopy) {
          setName(saved.name);
        }
        setSavedKey(`${asCopy ? saved.name : name}\n${semanticKey(doc)}`);
      }
      notifications.show({
        color: 'green',
        message: savedAsNew
          ? i18n.t('builder.page.savedAsNew', { name: saved.name })
          : i18n.t('builder.page.saved', { name: saved.name }),
      });
      return saved;
    } catch (error) {
      notifications.show({
        color: 'red',
        title: i18n.t('builder.page.savingFailed'),
        message: (error as Error).message,
      });
      return null;
    } finally {
      savingRef.current = false;
      setSaving(false);
    }
  };

  const resolveConflict = (choice: ConflictChoice) => {
    const current = conflict;
    setConflict(null);
    if (!current) {
      return;
    }
    if (choice === 'theirs') {
      loadProject(current);
    } else if (choice === 'mine') {
      // Over the version just seen: if it changed yet again, the user is asked again.
      void save({ baseVersion: current.version });
    } else {
      void save({ asCopy: true });
    }
  };

  /** Opens the share dialog; a network that is not saved yet is saved first. */
  const share = async () => {
    if (project?.info) {
      setShareOpened(true);
      return;
    }
    if (project && user) {
      // Opened from a draft and the details are not loaded yet.
      try {
        const server = await api.projects.get(project.id);
        setProject((current) =>
          current?.id === server.id
            ? { ...openProjectOf(server), version: current.version }
            : current
        );
        setShareOpened(true);
      } catch (error) {
        notifications.show({ color: 'red', message: (error as Error).message });
      }
      return;
    }
    if (await save()) {
      setShareOpened(true);
    }
  };

  /** Viewers: copy the project into their own account and continue with the copy. */
  const duplicateProject = async () => {
    if (!project) {
      return;
    }
    try {
      const copy = await api.projects.duplicate(project.id);
      loadProject(copy);
      notifications.show({
        color: 'green',
        title: i18n.t('builder.page.duplicated', { name: copy.name }),
        message: i18n.t('builder.page.duplicatedText'),
      });
    } catch (error) {
      notifications.show({
        color: 'red',
        title: i18n.t('builder.page.duplicateFailed'),
        message: (error as Error).message,
      });
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
          title: i18n.t('builder.page.hasProblems'),
          message: i18n.t('builder.page.hasProblemsText'),
        });
        showSummary();
      } else {
        notifications.show({
          color: 'red',
          title: i18n.t('builder.page.exportFailed'),
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
        title: i18n.t('builder.page.cannotOpen'),
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
      loadDocument({ version: 1, nodes: [], edges: [] }, i18n.t('builder.untitled'), null)
    );

  const loadTrainedNetwork = (graph: GraphDocument, runName: string) =>
    confirmReplace(() => {
      loadDocument(graph, runName, null);
      training.close();
    });

  const showProblemsFromTraining = () => {
    training.close();
    showSummary();
  };

  const loginToTrain = () => {
    training.close();
    navigate('/login', { state: { from: '/' } });
  };

  useHotkeys([
    ['mod+S', () => void save()],
    ['mod+E', () => void exportPython()],
    ['mod+Enter', () => !readOnly && training.open()],
    // Not inside text fields (Mantine skips them), and Ctrl+C copies selected text as usual.
    ['mod+C', (event) => void copyBricks(event), { preventDefault: false }],
    ['mod+X', (event) => cutBricks(event), { preventDefault: false }],
    ['mod+V', (event) => pasteBricks(event), { preventDefault: false }],
  ]);

  // ---------------------------------------------------------------- render
  const presence = live.tracking ? live.presence : null;
  const context: BuilderContextValue = useMemo(
    () => ({
      specs,
      analysis,
      errorsByNode: groupIssues(analysis?.errors),
      warningsByNode: groupIssues(analysis?.warnings),
      readOnly,
      updateNode,
      patchParams,
      deleteNode: removeLayer,
      duplicateNode,
      focusNode,
      presence,
      inputFit: datasetFit.inputFit,
      links: edges,
      canConnect,
      connect: connectLayers,
      disconnect: disconnectLayers,
    }),
    [
      edges,
      specs,
      analysis,
      readOnly,
      updateNode,
      patchParams,
      removeLayer,
      duplicateNode,
      focusNode,
      presence,
      datasetFit.inputFit,
      canConnect,
      connectLayers,
      disconnectLayers,
    ]
  );
  const canvasNodes = useLabelledNodes(live.displayNodes, specs, context.errorsByNode);
  const canvasEdges = useLabelledEdges(displayEdges, nodes, specs);
  const canvasLabels = useCanvasLabels(readOnly);

  if (!catalog) {
    return (
      <Center h={`calc(100dvh - ${HEADER_HEIGHT}px)`}>
        {catalogError ? (
          <Stack align="center" maw={420} ta="center">
            <IconPlugConnectedX size={40} color="var(--mantine-color-red-6)" />
            <Title order={3}>{t('builder.page.noBackend')}</Title>
            <Text c="dimmed" size="sm">
              {catalogError}{' '}
              <Trans i18nKey="builder.page.noBackendText" components={{ code: <code /> }} />
            </Text>
            <Button onClick={reload}>{t('common.retry')}</Button>
          </Stack>
        ) : (
          <Loader />
        )}
      </Center>
    );
  }

  const paletteView = (
    <Palette
      catalog={catalog}
      onAdd={(spec) => addLayer(spec)}
      onDrop={dropLayer}
      // On small screens the palette is a drawer over the canvas: get it out of the way.
      onDragStart={palette.close}
    />
  );
  const inspectorView = <Inspector node={selected} nodes={nodes} />;
  // Live: who has the network open, me included (one entry per person).
  const here =
    live.tracking && user
      ? [...new Set([user.username, ...live.people.map((person) => person.username)])]
      : null;
  const people = members
    ? [members.owner.username, ...members.members.map((member) => member.user.username)]
    : here && here.length > 1
      ? here
      : null;

  return (
    <BuilderContext.Provider value={context}>
      <Box style={{ display: 'flex', height: `calc(100dvh - ${HEADER_HEIGHT}px)` }}>
        {/* Viewers cannot add layers. */}
        {!readOnly && (
          <Box
            visibleFrom="md"
            w={230}
            style={{ borderRight: '1px solid var(--mantine-color-default-border)', flexShrink: 0 }}
            data-tour="palette"
          >
            {paletteView}
          </Box>
        )}

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
            isSaved={isSaved}
            readOnly={readOnly}
            owner={project?.info?.owner.username ?? null}
            people={people}
            here={here}
            live={live.indicator}
            onLoadExample={loadExample}
            onNew={newNetwork}
            onImport={() => fileInput.current?.click()}
            onExportJson={exportJson}
            onTidy={tidyUp}
            onSave={() => void save()}
            onShare={() => void share()}
            onDuplicate={() => void duplicateProject()}
            onShowCode={code.open}
            onExportPython={() => void exportPython()}
            onTrain={training.open}
            trainingRun={training.activeRun}
            onShowSummary={showSummary}
            onOpenPalette={palette.open}
            onOpenInspector={inspector.open}
            onShowGuide={() => setGuideOpened(true)}
          />
          <Box ref={canvasRef} style={{ flex: 1, position: 'relative' }} data-tour="canvas">
            <ReactFlow<LayerFlowNode, Edge>
              nodes={canvasNodes}
              edges={canvasEdges}
              nodeTypes={nodeTypes}
              aria-label={t('a11y.canvas.label')}
              ariaLabelConfig={canvasLabels}
              onDelete={onCanvasDelete}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onConnect={readOnly ? undefined : onConnect}
              isValidConnection={isValidConnection}
              onConnectEnd={readOnly ? undefined : onConnectEnd}
              nodesDraggable={!readOnly}
              nodesConnectable={!readOnly}
              defaultEdgeOptions={EDGE_DEFAULTS}
              deleteKeyCode={readOnly ? null : ['Backspace', 'Delete']}
              colorMode={colorScheme}
              minZoom={0.15}
              maxZoom={2}
              fitView
              fitViewOptions={{ padding: 0.15, maxZoom: 1 }}
            >
              <Background gap={20} />
              <Controls showInteractive={!readOnly} />
              {mediumScreen && <MiniMap pannable zoomable nodeColor={miniMapColor(theme, specs)} />}
              <Collaboration
                projectId={projectId}
                role={role}
                subscribe={live.subscribe}
                live={live.status === 'live'}
                unsaved={!isSaved}
                // Live, the server's `replaced` message brings the restored state.
                onRestored={(restored) => !live.isTracking() && loadProject(restored)}
              />
              {/* For learners: step a sample through the network (src/explain). */}
              <DataFlowTour analysis={analysis} specs={specs} links={edges} focusNode={focusNode} />
              {nodes.length === 0 && !readOnly && (
                <Panel position="top-center">
                  <EmptyCanvas examples={examples} onLoadExample={loadExample} />
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

      {!readOnly && (
        <Drawer
          opened={paletteOpened}
          onClose={palette.close}
          title={t('builder.toolbar.layers')}
          size={280}
        >
          <Box h="calc(100dvh - 90px - env(safe-area-inset-bottom))">{paletteView}</Box>
        </Drawer>
      )}
      <Drawer
        opened={inspectorOpened && !wideScreen}
        onClose={inspector.close}
        position="right"
        title={t('builder.page.properties')}
        size={340}
      >
        {inspectorView}
      </Drawer>
      <ConflictDialog
        project={conflict}
        onChoose={resolveConflict}
        onClose={() => setConflict(null)}
      />
      {shareOpened && project?.info && (
        <ShareDialog
          project={project.info}
          onClose={() => setShareOpened(false)}
          onChange={applySharing}
          // The network stays open here, as a new network that is not saved.
          onLeave={detach}
        />
      )}
      {codeOpened && (
        <Suspense fallback={null}>
          <CodeModal
            opened={codeOpened}
            onClose={code.close}
            doc={doc}
            filename={filename}
            options={codeOptions}
            onOptionsChange={setCodeOptions}
            family={networkFamily(analysis)}
          />
        </Suspense>
      )}
      {(trainingLoaded || training.opened) && (
        <Suspense fallback={null}>
          <TrainingDrawer
            training={training}
            loggedIn={Boolean(user)}
            doc={doc}
            networkName={name}
            analysis={analysis}
            onLogin={loginToTrain}
            onShowProblems={showProblemsFromTraining}
            onLoadNetwork={loadTrainedNetwork}
            fit={datasetFit.controller}
          />
        </Suspense>
      )}
      <OnboardingGuide opened={guideOpened} onClose={() => setGuideOpened(false)} />
      <LiveRegion message={announcement} />
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
