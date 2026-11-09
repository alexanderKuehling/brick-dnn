import { useCallback, useEffect, useRef, useState } from 'react';
import { apiBaseUrl, freshAccessToken, type ProjectRole } from '@/api';
import { LiveClient, type EndReason, type LiveStatus } from './LiveClient';
import { PresenceStore } from './presence';
import type { LiveOp, PresenceState, ServerMessage } from './protocol';

/** The WebSocket address of a project, next to the API (wss:// for https://). */
export function liveUrl(projectId: number): string {
  const url = new URL(apiBaseUrl());
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${url.href.replace(/\/+$/, '')}/api/ws/projects/${projectId}/`;
}

export interface LiveState {
  /** 'off': no connection (no saved project open, logged out or the online demo). */
  status: LiveStatus | 'off';
  endReason: EndReason | null;
  /** Number of change batches the server has not confirmed yet. */
  pendingCount: number;
  role: ProjectRole | null;
}

const OFF: LiveState = { status: 'off', endReason: null, pendingCount: 0, role: null };

export interface LiveProject extends LiveState {
  /** The connection of the open project; null while live editing is off. */
  client: LiveClient | null;
  /** The same, also before the next render (for message handlers). */
  getClient: () => LiveClient | null;
  /** Who is here and what they have selected or drag. */
  presence: PresenceStore;
  /** Changes for the server (batched; ignored while live editing is off). */
  sendOps: (ops: LiveOp[]) => void;
  /** Our selection etc. for the others (throttled). */
  sendPresence: (state: PresenceState) => void;
  /** Messages from the server, also across reconnects and project changes. */
  subscribe: (listener: (message: ServerMessage) => void) => () => void;
}

/** Ops of a project that were not confirmed before the page was closed (from the draft). */
export interface PendingSeed {
  projectId: number;
  ops: LiveOp[];
}

/**
 * Keeps a live connection to the project while `enabled` (logged in, not the online demo).
 * Status, role and pending changes are React state; peers go to the presence store.
 */
export function useLiveProject(
  projectId: number | null,
  enabled: boolean,
  seed?: PendingSeed | null
): LiveProject {
  const [client, setClient] = useState<LiveClient | null>(null);
  const [state, setState] = useState<LiveState>(OFF);
  const [presence] = useState(() => new PresenceStore());
  const listeners = useRef(new Set<(message: ServerMessage) => void>());
  const current = useRef<LiveClient | null>(null);
  const seedRef = useRef(seed ?? null);

  useEffect(() => {
    if (seedRef.current && seedRef.current.projectId !== projectId) {
      seedRef.current = null; // Another project was opened: the seed is not for it.
    }
    if (!enabled || projectId === null) {
      return undefined;
    }
    const live = new LiveClient({
      url: liveUrl(projectId),
      getToken: freshAccessToken,
      pending: seedRef.current?.ops,
    });
    const update = () => {
      presence.update(live.peers);
      const next: LiveState = {
        status: live.status,
        endReason: live.endReason,
        pendingCount: live.pendingCount,
        role: live.role,
      };
      setState((current) =>
        current.status === next.status &&
        current.endReason === next.endReason &&
        current.pendingCount === next.pendingCount &&
        current.role === next.role
          ? current
          : next
      );
    };
    const offChange = live.onChange(update);
    const offMessages = live.subscribe((message) =>
      listeners.current.forEach((listener) => listener(message))
    );
    current.current = live;
    setClient(live);
    update();
    live.start();
    return () => {
      offChange();
      offMessages();
      live.stop();
      current.current = null;
      setClient(null);
      setState(OFF);
      presence.update([]);
    };
  }, [projectId, enabled, presence]);

  const subscribe = useCallback((listener: (message: ServerMessage) => void) => {
    listeners.current.add(listener);
    return () => {
      listeners.current.delete(listener);
    };
  }, []);

  const getClient = useCallback(() => current.current, []);
  const sendOps = useCallback((ops: LiveOp[]) => current.current?.sendOps(ops), []);
  const sendPresence = useCallback(
    (presenceState: PresenceState) => current.current?.sendPresence(presenceState),
    []
  );

  return { ...state, client, getClient, presence, subscribe, sendOps, sendPresence };
}
