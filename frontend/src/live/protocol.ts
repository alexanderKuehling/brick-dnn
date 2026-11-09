/**
 * Messages of the live editing WebSocket (`/api/ws/projects/<id>/`, see docs/api.md, "Live
 * editing"). The server holds the truth: clients send small operations ("ops"), the server
 * applies them in order and sends them to everyone in the project, the sender included.
 */
import type { GraphDocument, GraphNode, Params, ProjectRole, UserRef } from '@/api/types';

/** What `update_node` may change. `params` are merged key by key; null clears a key. */
export interface NodeChange {
  params?: Params;
  position?: { x: number; y: number };
  /** null (or empty) removes the name. */
  name?: string | null;
  type?: string;
}

export type LiveOp =
  | { op: 'add_node'; node: GraphNode }
  | { op: 'update_node'; id: string; set: NodeChange }
  | { op: 'remove_node'; id: string }
  | { op: 'add_edge'; source: string; target: string }
  | { op: 'remove_edge'; source: string; target: string }
  | { op: 'set_meta'; name?: string; description?: string };

/** The project as the live server describes it. */
export interface LiveProjectState {
  id: number;
  name: string;
  description: string;
  graph: GraphDocument;
  version: number;
}

/** A layer being moved by someone right now (passed on, never stored). */
export interface DragPresence {
  id: string;
  position: { x: number; y: number };
}

/** What a client tells the others about itself: its selection and what it is dragging. */
export interface PresenceState {
  selected?: string[];
  /** One layer, or several when a selection is dragged together; null: nothing. */
  dragging?: DragPresence | DragPresence[] | null;
  [key: string]: unknown;
}

/** Someone else in the project (one per browser tab). */
export interface Peer {
  clientId: string;
  user: UserRef;
  state: PresenceState;
}

export type ServerMessage =
  | {
      type: 'hello';
      project: LiveProjectState;
      role: ProjectRole;
      client_id: string;
      user: UserRef;
    }
  | {
      type: 'ops';
      ops: LiveOp[];
      version: number;
      by: UserRef;
      client_id: string;
      id: string | null;
    }
  | { type: 'rejected'; id: string | null; version: number }
  | { type: 'joined'; client_id: string; user: UserRef }
  | { type: 'left'; client_id: string }
  | { type: 'presence'; client_id: string; user: UserRef; state: PresenceState }
  | { type: 'replaced'; project: LiveProjectState; by: UserRef }
  | { type: 'access'; role: ProjectRole | null }
  | { type: 'deleted' }
  | { type: 'error'; id?: string | null; detail: string }
  | { type: 'comment'; action: 'created' | 'updated' | 'deleted'; comment: unknown }
  | { type: 'history' }
  | { type: 'pong' };

export type ClientMessage =
  | { type: 'auth'; token: string }
  | { type: 'ops'; id: string; ops: LiveOp[] }
  | { type: 'presence'; state: PresenceState }
  | { type: 'ping' };

/** Close codes of the server. */
export const CLOSE_UNAUTHORIZED = 4401;
export const CLOSE_FORBIDDEN = 4403;
export const CLOSE_NOT_FOUND = 4404;
export const CLOSE_TOO_FAST = 4429;

/** At most this many ops per message (the server's limit). */
export const MAX_OPS_PER_MESSAGE = 500;
