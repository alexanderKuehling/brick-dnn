import { useEffect, useRef, useState } from 'react';
import { explainApi } from './api';
import type { ExplainRequest } from './types';

export interface Explanation<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
  retry: () => void;
}

/** A stable text for a request (the cache key on this page). */
export function requestKey(body: ExplainRequest | null): string {
  return body ? JSON.stringify(body) : '';
}

/**
 * Ask the server for one explanation (null: nothing yet). Answers are kept for the page, so
 * going back to a sample or tab shows it at once (the server caches them, too).
 */
export function useExplanation<T>(runId: string, body: ExplainRequest | null): Explanation<T> {
  const cache = useRef(new Map<string, unknown>());
  const key = requestKey(body);
  const [state, setState] = useState<{ key: string; data: T | null; error: string | null }>({
    key: '',
    data: null,
    error: null,
  });
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    if (!body) {
      return undefined;
    }
    const cached = cache.current.get(key);
    if (cached !== undefined) {
      setState({ key, data: cached as T, error: null });
      return undefined;
    }
    const controller = new AbortController();
    setState({ key, data: null, error: null });
    explainApi
      .run<T>(runId, body, controller.signal)
      .then((data) => {
        cache.current.set(key, data);
        setState({ key, data, error: null });
      })
      .catch((error: Error) => {
        if (error.name !== 'AbortError') {
          setState({ key, data: null, error: error.message });
        }
      });
    return () => controller.abort();
    // `key` stands for `body` (a new object on every render).
  }, [runId, key, attempt]);

  const current = state.key === key;
  return {
    data: current ? state.data : null,
    loading: Boolean(body) && (!current || (state.data === null && state.error === null)),
    error: current ? state.error : null,
    retry: () => setAttempt((n) => n + 1),
  };
}
