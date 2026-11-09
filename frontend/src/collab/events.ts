import type { ProjectComment } from '@/api/types';

/**
 * A message of the live editing connection of a project (docs/api.md, "Live editing"). Only
 * the messages below are read here; everything else is ignored, so the live connection can
 * pass on all of its messages as they are.
 */
export interface ProjectEvent {
  readonly type: string;
}

/**
 * Registers a handler for the messages of the open project and returns a function that
 * removes it. The live connection (`useLiveProject(...).subscribe`) has this shape.
 */
export type SubscribeToEvents = (handler: (event: ProjectEvent) => void) => () => void;

export type CommentAction = 'created' | 'updated' | 'deleted';

/** `{"type": "comment", action, comment}`: someone created, changed or deleted a comment. */
export interface CommentEvent {
  type: 'comment';
  action: CommentAction;
  /** The comment; only `{id}` when it was deleted (its replies are deleted with it). */
  comment: ProjectComment | { id: number };
}

const ACTIONS: readonly string[] = ['created', 'updated', 'deleted'];

/** The event as a comment change, or null if it is something else (or malformed). */
export function asCommentEvent(event: ProjectEvent): CommentEvent | null {
  if (event.type !== 'comment') {
    return null;
  }
  const { action, comment } = event as Partial<CommentEvent>;
  if (typeof action !== 'string' || !ACTIONS.includes(action)) {
    return null;
  }
  if (!comment || typeof comment !== 'object' || typeof comment.id !== 'number') {
    return null;
  }
  if (action !== 'deleted' && typeof (comment as Partial<ProjectComment>).text !== 'string') {
    return null;
  }
  return { type: 'comment', action, comment };
}

/** `{"type": "history"}`: a new entry in the history (a named or restored version, …). */
export function isHistoryEvent(event: ProjectEvent): boolean {
  return event.type === 'history';
}

/**
 * `ops` or `replaced`: the network changed. The server sends `history` only for new entries,
 * so an open history refreshes its latest (automatic) entry on these.
 */
export function isChangeEvent(event: ProjectEvent): boolean {
  return event.type === 'ops' || event.type === 'replaced';
}

/**
 * `{"type": "hello"}`: the live connection was (re)established. Messages sent while it was
 * down are lost, so lists are loaded again.
 */
export function isConnectedEvent(event: ProjectEvent): boolean {
  return event.type === 'hello';
}
