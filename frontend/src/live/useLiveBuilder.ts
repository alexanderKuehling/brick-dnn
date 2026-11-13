/**
 * Live editing in the builder: keeps the canvas in sync with the open project.
 *
 * "Base" is the project as the server will have it once our pending ops are applied. After
 * every local change the canvas is compared with the base and the differences go out as ops;
 * ops from others are applied to the base (with our pending ops on top again, as the server
 * will apply them after) and the canvas is rebuilt from it.
 *
 * The canvas is "tracked" once it matches the server. Until then (e.g. unsaved changes based
 * on an older version, or no connection) the builder works as without live editing: the user
 * saves and resolves conflicts in the save dialog.
 */
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type Dispatch,
  type SetStateAction,
} from 'react';
import type { Edge } from '@xyflow/react';
import { notifications } from '@mantine/notifications';
import type { GraphDocument, ProjectRole, UserRef } from '@/api/types';
import { fromDocument, semanticKey, toDocument, type LayerFlowNode } from '@/builder/graph';
import type { OpenProject } from '@/builder/useOpenProject';
import { applyMeta, applyOps, diffGraph, mergeEdges, mergeNodes, removedIds } from './ops';
import { usePeople, useRemoteDrags } from './presence';
import type { LiveOp, LiveProjectState, PresenceState, ServerMessage } from './protocol';
import { useLiveProject, type PendingSeed } from './useLiveProject';

/** What the toolbar shows instead of the save button while the canvas is live. */
export type LiveIndicator = 'live' | 'saving' | 'offline';

interface Base {
  graph: GraphDocument;
  name: string;
  version: number;
}

export interface LiveBuilderOptions {
  /** Logged in. */
  enabled: boolean;
  user: UserRef | null;
  project: OpenProject | null;
  setProject: Dispatch<SetStateAction<OpenProject | null>>;
  /** Continue with the canvas as a new network that is not saved. */
  detach: () => void;
  readOnly: boolean;
  nodes: LayerFlowNode[];
  edges: Edge[];
  name: string;
  setNodes: Dispatch<SetStateAction<LayerFlowNode[]>>;
  setEdges: Dispatch<SetStateAction<Edge[]>>;
  setName: (name: string) => void;
  /** Name and graph without layout (equal keys: same content). */
  contentKey: string;
  /** The canvas has no unsaved changes (outside live editing). */
  isSaved: boolean;
  /** Ops from the draft that the server had not confirmed when the page was closed. */
  seed?: PendingSeed | null;
  /** Someone deleted the layer that is selected here. */
  onSelectedRemoved?: (by: UserRef) => void;
}

/** The server's graph as the canvas would have it (string ids, positions, valid edges). */
function normalized(graph: GraphDocument): GraphDocument {
  const { nodes, edges } = fromDocument(graph);
  return toDocument(nodes, edges);
}

const keyOf = (base: Base) => `${base.name}\n${semanticKey(base.graph)}`;

function rounded(node: LayerFlowNode) {
  return {
    id: node.id,
    position: { x: Math.round(node.position.x), y: Math.round(node.position.y) },
  };
}

export function useLiveBuilder(options: LiveBuilderOptions) {
  const projectId = options.project?.id ?? null;
  const live = useLiveProject(projectId, options.enabled, options.seed);
  const { client, getClient } = live;
  const [tracking, setTracking] = useState(false);
  const trackingRef = useRef(false);
  const base = useRef<Base | null>(null);
  // The handlers below run outside of renders: they read the latest options from here.
  const latest = useRef(options);
  latest.current = options;
  const setTracked = useCallback((value: boolean) => {
    trackingRef.current = value;
    setTracking(value);
  }, []);

  // Another project (or none): start over.
  useEffect(
    () => () => {
      base.current = null;
      setTracked(false);
    },
    [projectId, setTracked]
  );

  const patchProject = useCallback(
    (patch: { version?: number; role?: ProjectRole; name?: string }) => {
      latest.current.setProject((current) => {
        if (!current || current.id !== latest.current.project?.id) {
          return current;
        }
        const { name, ...rest } = patch;
        const info = current.info && {
          ...current.info,
          ...rest,
          ...(name !== undefined ? { name } : {}),
        };
        return { ...current, ...rest, info };
      });
    },
    []
  );

  /** Show `next` on the canvas. `full`: drop layers that are not in it (a new state). */
  const load = useCallback((next: Base, full: boolean) => {
    const previous = base.current?.graph;
    base.current = next;
    const { setNodes, setEdges, setName } = latest.current;
    setNodes((current) => mergeNodes(current, next.graph, full ? undefined : previous));
    setEdges((current) => mergeEdges(current, next.graph));
    setName(next.name);
  }, []);

  /** Turn the differences between the canvas and the base into ops and send them. */
  const sync = useCallback(() => {
    const current = base.current;
    const live = getClient();
    if (!current || !live || !trackingRef.current) {
      return;
    }
    const { nodes, edges, name, readOnly } = latest.current;
    const result = diffGraph(current.graph, current.name, nodes, edges, name);
    base.current = { ...current, graph: result.graph, name: result.name };
    if (result.ops.length > 0 && !readOnly) {
      live.sendOps(result.ops);
    }
  }, [getClient]);

  /** Start following the server if the canvas has nothing to lose. */
  const tryTrack = useCallback(() => {
    const current = base.current;
    const { isSaved, contentKey, project } = latest.current;
    if (!current || trackingRef.current || getClient()?.status !== 'live') {
      return;
    }
    // Just saved through the API: wait until that version arrives here too.
    if (project?.version != null && current.version < project.version) {
      return;
    }
    if (isSaved || contentKey === keyOf(current)) {
      load(current, true);
      patchProject({ version: current.version });
      setTracked(true);
    }
  }, [getClient, load, patchProject, setTracked]);

  const leaveAsCopy = useCallback(
    (title: string) => {
      getClient()?.dropPending();
      latest.current.detach();
      notifications.show({
        id: 'live-left',
        color: 'yellow',
        title,
        message: 'Your unsaved changes stay open here as a new network. Save it to keep them.',
      });
    },
    [getClient]
  );

  const hello = useCallback(
    (message: Extract<ServerMessage, { type: 'hello' }>) => {
      const live = getClient();
      if (!live) {
        return;
      }
      const server: LiveProjectState = message.project;
      const pending = live.pendingOps();
      const graph = normalized(server.graph);
      const { isSaved, project } = latest.current;
      patchProject({ role: message.role });
      if (message.role === 'viewer' && (pending.length > 0 || (!trackingRef.current && !isSaved))) {
        leaveAsCopy(`You can only view “${server.name}” now`);
        return;
      }
      if (trackingRef.current || pending.length > 0 || isSaved) {
        // Our changes that the server has not confirmed go on top of its state again.
        load(
          {
            graph: applyOps(graph, pending),
            name: applyMeta(server.name, pending),
            version: server.version,
          },
          true
        );
        patchProject({ version: server.version, name: server.name });
        setTracked(true);
      } else if (project?.version === server.version) {
        // Unsaved changes of the version the server still has: send them.
        base.current = { graph, name: server.name, version: server.version };
        setTracked(true);
        sync();
      } else {
        // Unsaved changes of an older version: saving lets the user choose (as without live).
        base.current = { graph, name: server.name, version: server.version };
      }
    },
    [getClient, load, patchProject, setTracked, sync, leaveAsCopy]
  );

  const applyRemote = useCallback(
    (ops: LiveOp[], version: number, by: UserRef) => {
      const current = base.current;
      const live = getClient();
      if (!current || !live) {
        return;
      }
      // The server applies our pending ops after these, so they go on top again.
      const all = [...ops, ...live.pendingOps()];
      const next = {
        graph: applyOps(current.graph, all),
        name: applyMeta(current.name, all),
        version,
      };
      if (!trackingRef.current) {
        base.current = next;
        tryTrack();
        return;
      }
      const removed = removedIds(ops);
      const selected = latest.current.nodes.find((node) => node.selected);
      load(next, false);
      patchProject({ version });
      if (selected && removed.includes(selected.id)) {
        latest.current.onSelectedRemoved?.(by);
      }
    },
    [getClient, load, patchProject, tryTrack]
  );

  const replaced = useCallback(
    (message: Extract<ServerMessage, { type: 'replaced' }>) => {
      const live = getClient();
      if (!live || !base.current) {
        return;
      }
      const pending = live.pendingOps();
      const next = {
        graph: applyOps(normalized(message.project.graph), pending),
        name: applyMeta(message.project.name, pending),
        version: message.project.version,
      };
      if (!trackingRef.current) {
        base.current = next;
        tryTrack();
        return;
      }
      load(next, true);
      patchProject({ version: next.version, name: message.project.name });
      const own = message.by.id === latest.current.user?.id;
      notifications.show({
        id: 'live-replaced', // A restore from the history shows its own message instead.
        title: own ? 'The network was replaced' : `${message.by.username} replaced the network`,
        message: own
          ? 'You saved it from another window or app. The new version is shown here.'
          : 'Their version is shown here now.',
      });
    },
    [getClient, load, patchProject, tryTrack]
  );

  const access = useCallback(
    (role: ProjectRole | null) => {
      if (role === null) {
        return; // The server closes the connection: handled with the status below.
      }
      patchProject({ role });
      const name = latest.current.name;
      if (role === 'viewer') {
        // Changes that were not confirmed cannot be saved any more: get the current state.
        getClient()?.dropPending();
        getClient()?.resync();
        notifications.show({
          title: `You can only view “${name}” now`,
          message: 'You still see the changes of the others as they happen.',
        });
      } else {
        notifications.show({ message: `You can edit “${name}” now.` });
      }
    },
    [getClient, patchProject]
  );

  // Messages from the server.
  useEffect(
    () =>
      live.subscribe((message) => {
        const current = getClient();
        switch (message.type) {
          case 'hello':
            hello(message);
            break;
          case 'ops':
            if (current && !current.isOwn(message)) {
              applyRemote(message.ops, message.version, message.by);
            } else if (base.current) {
              base.current = { ...base.current, version: message.version };
              if (trackingRef.current) {
                patchProject({ version: message.version });
              }
            }
            break;
          case 'replaced':
            replaced(message);
            break;
          case 'access':
            access(message.role);
            break;
          default:
            break;
        }
      }),
    [live.subscribe, getClient, hello, applyRemote, replaced, access, patchProject]
  );

  // No connection (logged out) or it ended for good.
  const { status, endReason } = live;
  useEffect(() => {
    if (status !== 'ended' && status !== 'off') {
      return;
    }
    base.current = null;
    setTracked(false);
    const name = latest.current.name;
    const keep = 'It stays open here as an unsaved network. Save it to keep it.';
    if (endReason === 'deleted') {
      notifications.show({
        id: 'live-left',
        color: 'yellow',
        title: `“${name}” was deleted`,
        message: keep,
      });
    } else if (endReason === 'forbidden') {
      notifications.show({
        id: 'live-left',
        color: 'yellow',
        title: `You no longer have access to “${name}”`,
        message: keep,
      });
    } else if (endReason === 'not_found') {
      notifications.show({
        id: 'no-access',
        color: 'yellow',
        title: 'No access any more',
        message: `“${name}” was deleted or is no longer shared with you. ${keep}`,
      });
    } else {
      return;
    }
    latest.current.detach();
  }, [status, endReason, setTracked]);

  // Local changes become ops.
  const { nodes, edges, name, isSaved, contentKey } = options;
  useEffect(() => {
    if (trackingRef.current) {
      sync();
    }
  }, [nodes, edges, name, tracking, sync]);

  // Not tracked yet: start once the canvas matches the server (e.g. after a save).
  useEffect(() => {
    if (!trackingRef.current) {
      tryTrack();
    }
  }, [isSaved, contentKey, tryTrack]);

  // Our selection and the layers we drag, for the others. When a drag ends, its final
  // position goes out first (it was just turned into an op above).
  const presenceKey = useRef('');
  const wasDragging = useRef(false);
  useEffect(() => {
    if (!client || !tracking) {
      presenceKey.current = '';
      return;
    }
    const dragging = nodes.filter((node) => node.dragging).map(rounded);
    const state: PresenceState = {
      selected: nodes.filter((node) => node.selected).map((node) => node.id),
      dragging: dragging.length === 0 ? null : dragging.length === 1 ? dragging[0] : dragging,
    };
    const key = JSON.stringify(state);
    if (key === presenceKey.current) {
      return;
    }
    if (wasDragging.current && dragging.length === 0) {
      client.flush();
    }
    wasDragging.current = dragging.length > 0;
    presenceKey.current = key;
    client.sendPresence(state);
  }, [nodes, client, tracking]);

  // "Saving…" only when the server takes a moment to confirm.
  const pending = live.pendingCount > 0;
  const [slow, setSlow] = useState(false);
  useEffect(() => {
    if (!pending) {
      setSlow(false);
      return undefined;
    }
    const timer = window.setTimeout(() => setSlow(true), 300);
    return () => window.clearTimeout(timer);
  }, [pending]);

  // Layers that others drag right now are shown where they are (without changing them).
  const drags = useRemoteDrags(live.presence);
  const displayNodes = useMemo(
    () =>
      Object.keys(drags).length === 0
        ? nodes
        : nodes.map((node) =>
            drags[node.id] && !node.dragging ? { ...node, position: drags[node.id] } : node
          ),
    [nodes, drags]
  );
  const people = usePeople(live.presence);

  let indicator: LiveIndicator | null = null;
  if (tracking && status === 'live') {
    indicator = slow ? 'saving' : 'live';
  } else if (tracking && (status === 'offline' || status === 'connecting')) {
    indicator = 'offline';
  }

  const pendingOps = useCallback(
    () => (trackingRef.current ? (getClient()?.pendingOps() ?? []) : []),
    [getClient]
  );
  const isTracking = useCallback(() => trackingRef.current, []);
  const flush = useCallback(() => getClient()?.flush(), [getClient]);

  return {
    /** The canvas follows the live project (no save button needed). */
    tracking,
    isTracking,
    status,
    indicator,
    pendingCount: live.pendingCount,
    pendingOps,
    flush,
    /** Others here now, one entry per person. */
    people,
    presence: live.presence,
    /** The canvas layers, with the ones others drag at their current position. */
    displayNodes,
    /** All messages of the live connection (comments, history, …). */
    subscribe: live.subscribe,
  };
}
