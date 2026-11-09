import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { api, type AgentAction, type AgentActivityQuery, type AgentKey } from '@/api';

/** How often the page asks for new entries while it is visible. */
export const POLL_INTERVAL = 3000;
const PAGE_SIZE = 50;
const POLL_SIZE = 100;

export interface ActivityFilters {
  agent: AgentKey | null;
  errorsOnly: boolean;
  project: number | null;
}

/** "live": polling; "paused": the page is hidden; "offline": the last poll failed. */
export type LiveState = 'live' | 'paused' | 'offline';

export interface AgentActivity {
  /** Newest first; null while loading. */
  entries: AgentAction[] | null;
  error: string | null;
  /** How long the server keeps entries (0: no log); null until loaded. */
  retentionDays: number | null;
  hasMore: boolean;
  loadingMore: boolean;
  loadMore: () => void;
  reload: () => void;
  /** Entries that arrived while the page was open (they animate in). */
  fresh: ReadonlySet<number>;
  live: LiveState;
}

function toQuery(filters: ActivityFilters): AgentActivityQuery {
  return {
    agent: filters.agent ?? undefined,
    outcome: filters.errorsOnly ? 'error' : undefined,
    project: filters.project ?? undefined,
  };
}

/** `incoming` (newest first) on top of `current`, without duplicates. */
function merge(incoming: AgentAction[], current: AgentAction[]): AgentAction[] {
  const known = new Set(current.map((entry) => entry.id));
  return [...incoming.filter((entry) => !known.has(entry.id)), ...current];
}

/**
 * The activity log with `filters`: the newest page, older pages on demand, and new entries
 * every `interval` milliseconds while the page is visible (`?after=` the newest one known).
 */
export function useAgentActivity(
  filters: ActivityFilters,
  interval: number = POLL_INTERVAL
): AgentActivity {
  const { agent, errorsOnly, project } = filters;
  const query = useMemo(
    () => toQuery({ agent, errorsOnly, project }),
    [agent, errorsOnly, project]
  );
  const [entries, setEntries] = useState<AgentAction[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [retentionDays, setRetentionDays] = useState<number | null>(null);
  const [hasMore, setHasMore] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [fresh, setFresh] = useState<ReadonlySet<number>>(new Set());
  const [live, setLive] = useState<LiveState>('live');
  const [request, setRequest] = useState(0);
  // The newest entry of the user the page knows about (the next `after`).
  const cursor = useRef(0);
  // Answers for an earlier query (or before a reload) are dropped.
  const generation = useRef(0);
  const ready = entries !== null;

  useEffect(() => {
    const current = ++generation.current;
    const controller = new AbortController();
    setEntries(null);
    setError(null);
    setFresh(new Set());
    api.agentActivity
      .list({ ...query, limit: PAGE_SIZE }, controller.signal)
      .then((page) => {
        if (current !== generation.current) {
          return;
        }
        cursor.current = page.latest_id;
        setRetentionDays(page.retention_days);
        setHasMore(page.has_more);
        setEntries(page.results);
      })
      .catch((err: Error) => {
        if (!controller.signal.aborted && current === generation.current) {
          setError(err.message);
        }
      });
    return () => controller.abort();
  }, [query, request]);

  const reload = useCallback(() => setRequest((n) => n + 1), []);

  // New entries while the page is visible; one request at a time.
  useEffect(() => {
    if (!ready) {
      return undefined;
    }
    const current = generation.current;
    let timer: number | undefined;
    let busy = false;
    let stopped = false;

    const poll = async () => {
      window.clearTimeout(timer);
      if (stopped || busy) {
        return;
      }
      if (document.visibilityState === 'hidden') {
        setLive('paused');
        return; // visibilitychange polls again
      }
      busy = true;
      try {
        const page = await api.agentActivity.list({
          ...query,
          after: cursor.current,
          limit: POLL_SIZE,
        });
        if (stopped || current !== generation.current) {
          return;
        }
        setLive('live');
        if (page.has_more) {
          reload(); // more new entries than one page: start over
          return;
        }
        cursor.current = Math.max(cursor.current, page.latest_id);
        setRetentionDays(page.retention_days);
        if (page.results.length > 0) {
          setEntries((list) => merge(page.results, list ?? []));
          setFresh((ids) => new Set([...ids, ...page.results.map((entry) => entry.id)]));
        }
      } catch {
        if (!stopped) {
          setLive('offline');
        }
      } finally {
        busy = false;
        if (!stopped) {
          timer = window.setTimeout(poll, interval);
        }
      }
    };

    const onVisibility = () => {
      if (document.visibilityState === 'hidden') {
        window.clearTimeout(timer);
        setLive('paused');
      } else {
        poll();
      }
    };

    timer = window.setTimeout(poll, interval);
    document.addEventListener('visibilitychange', onVisibility);
    return () => {
      stopped = true;
      window.clearTimeout(timer);
      document.removeEventListener('visibilitychange', onVisibility);
    };
  }, [ready, query, interval, reload]);

  const loadMore = useCallback(() => {
    const oldest = entries?.[entries.length - 1];
    if (!oldest || loadingMore) {
      return;
    }
    const current = generation.current;
    setLoadingMore(true);
    api.agentActivity
      .list({ ...query, before: oldest.id, limit: PAGE_SIZE })
      .then((page) => {
        if (current === generation.current) {
          setEntries((list) => [...(list ?? []), ...page.results]);
          setHasMore(page.has_more);
        }
      })
      .catch((err: Error) => current === generation.current && setError(err.message))
      .finally(() => setLoadingMore(false));
  }, [entries, loadingMore, query]);

  return {
    entries,
    error,
    retentionDays,
    hasMore,
    loadingMore,
    loadMore,
    reload,
    fresh,
    live,
  };
}

/** The current time, updated every `interval` milliseconds (for "2 min ago"). */
export function useNow(interval = 30_000): Date {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const timer = window.setInterval(() => setNow(new Date()), interval);
    return () => window.clearInterval(timer);
  }, [interval]);
  return now;
}
