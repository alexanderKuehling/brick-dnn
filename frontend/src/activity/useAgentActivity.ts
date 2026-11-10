import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  api,
  type AgentAction,
  type AgentActivityPage,
  type AgentActivityQuery,
  type AgentKey,
  type ApprovalRequest,
} from '@/api';
import { getPendingCount, setPendingCount } from '@/approvals/pending';

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
  /** Requests waiting for the user's approval (with every poll); null while loading. */
  approvals: ApprovalRequest[] | null;
  /** A request was decided on the page: it leaves the list at once. */
  decided: (approval: ApprovalRequest) => void;
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
  const [approvals, setApprovals] = useState<ApprovalRequest[] | null>(null);
  const [request, setRequest] = useState(0);
  // Requests decided here, until the server's list no longer has them.
  const gone = useRef(new Set<number>());
  // The newest entry of the user the page knows about (the next `after`).
  const cursor = useRef(0);
  // Answers for an earlier query (or before a reload) are dropped.
  const generation = useRef(0);
  const ready = entries !== null;

  const takeApprovals = useCallback((page: AgentActivityPage) => {
    if (!page.approvals) {
      return; // an older server
    }
    const waiting = page.approvals.pending.filter((item) => !gone.current.has(item.id));
    setApprovals(waiting);
    setPendingCount(
      page.approvals.pending_count - (page.approvals.pending.length - waiting.length)
    );
  }, []);

  const decided = useCallback((approval: ApprovalRequest) => {
    gone.current.add(approval.id);
    setApprovals((list) => list?.filter((item) => item.id !== approval.id) ?? null);
    setPendingCount(Math.max(0, getPendingCount() - 1));
  }, []);

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
        takeApprovals(page);
      })
      .catch((err: Error) => {
        if (!controller.signal.aborted && current === generation.current) {
          setError(err.message);
        }
      });
    return () => controller.abort();
  }, [query, request, takeApprovals]);

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
        takeApprovals(page);
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
  }, [ready, query, interval, reload, takeApprovals]);

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
    approvals,
    decided,
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
