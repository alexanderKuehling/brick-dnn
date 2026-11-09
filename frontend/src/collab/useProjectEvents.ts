import { useEffect, useRef } from 'react';
import {
  asCommentEvent,
  isChangeEvent,
  isConnectedEvent,
  isHistoryEvent,
  type CommentEvent,
  type SubscribeToEvents,
} from './events';

export interface ProjectEventHandlers {
  /** A comment was created, changed or deleted (by anyone, the current user included). */
  onComment?: (event: CommentEvent) => void;
  /** A new entry in the history: load it again. */
  onHistory?: () => void;
  /** The network changed (which may update the latest entry of the history). */
  onChange?: () => void;
  /** The live connection was (re)established: messages may have been missed. */
  onConnected?: () => void;
}

/**
 * Passes the live messages of the open project to the handlers.
 *
 * Integration point for live editing: `subscribe` is the live connection's subscription
 * (`useLiveBuilder(...).subscribe` in the builder, see docs/api.md, "Live editing"). It stays
 * subscribed across reconnects, so the `hello` of a reconnect arrives too. While there is no
 * connection nothing arrives and the callers poll instead. The handlers may change on every
 * render; the subscription stays.
 */
export function useProjectEvents(
  subscribe: SubscribeToEvents | null | undefined,
  handlers: ProjectEventHandlers
): void {
  const current = useRef(handlers);
  current.current = handlers;

  useEffect(() => {
    if (!subscribe) {
      return undefined;
    }
    return subscribe((event) => {
      const comment = asCommentEvent(event);
      if (comment) {
        current.current.onComment?.(comment);
      } else if (isHistoryEvent(event)) {
        current.current.onHistory?.();
      } else if (isChangeEvent(event)) {
        current.current.onChange?.();
      } else if (isConnectedEvent(event)) {
        current.current.onConnected?.();
      }
    });
  }, [subscribe]);
}

/**
 * Calls `callback` every `interval` milliseconds while `enabled` and the page is visible (the
 * fallback while there is no live connection).
 */
export function usePolling(callback: () => void, interval: number, enabled: boolean): void {
  const current = useRef(callback);
  current.current = callback;

  useEffect(() => {
    if (!enabled) {
      return undefined;
    }
    const timer = window.setInterval(() => {
      if (document.visibilityState !== 'hidden') {
        current.current();
      }
    }, interval);
    return () => window.clearInterval(timer);
  }, [interval, enabled]);
}

/**
 * Returns a function that calls `callback` once `delay` milliseconds have passed without
 * another call (e.g. to reload after a burst of changes).
 */
export function useDebounced(callback: () => void, delay: number): () => void {
  const current = useRef(callback);
  current.current = callback;
  const timer = useRef<number | null>(null);

  useEffect(
    () => () => {
      if (timer.current !== null) {
        window.clearTimeout(timer.current);
      }
    },
    []
  );

  const debounced = useRef(() => {
    if (timer.current !== null) {
      window.clearTimeout(timer.current);
    }
    timer.current = window.setTimeout(() => {
      timer.current = null;
      current.current();
    }, delay);
  });
  return debounced.current;
}
