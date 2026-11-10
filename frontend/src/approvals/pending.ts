import { useEffect, useSyncExternalStore } from 'react';
import { api } from '@/api';

/** How often the header asks for the number of waiting approvals while the page is visible. */
export const PENDING_POLL_INTERVAL = 60_000;

// The number of requests that wait for the user's approval, shared by the header and the
// activity page (which learns it with every poll and passes it on).
let pendingCount = 0;
const listeners = new Set<() => void>();

export function setPendingCount(count: number): void {
  if (count !== pendingCount) {
    pendingCount = count;
    listeners.forEach((listener) => listener());
  }
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

const current = () => pendingCount;

export function getPendingCount(): number {
  return pendingCount;
}

/** The number of waiting approvals, as last heard by any part of the app. */
export function usePendingCount(): number {
  return useSyncExternalStore(subscribe, current, () => 0);
}

/**
 * The number of waiting approvals; asks the server every `interval` milliseconds while
 * `enabled` (logged in) and the page is visible, and when it becomes visible again.
 */
export function usePendingApprovals(
  enabled: boolean,
  interval: number = PENDING_POLL_INTERVAL
): number {
  const count = usePendingCount();
  useEffect(() => {
    if (!enabled) {
      setPendingCount(0);
      return undefined;
    }
    let stopped = false;
    let timer: number | undefined;
    const check = () => {
      window.clearTimeout(timer);
      if (document.visibilityState !== 'hidden') {
        Promise.resolve()
          .then(() => api.approvals.list({ status: 'pending', limit: 0 }))
          .then((list) => !stopped && setPendingCount(list.pending_count))
          .catch(() => undefined); // The badge keeps its last count.
      }
      timer = window.setTimeout(check, interval);
    };
    const onVisibility = () => document.visibilityState === 'visible' && check();
    check();
    document.addEventListener('visibilitychange', onVisibility);
    return () => {
      stopped = true;
      window.clearTimeout(timer);
      document.removeEventListener('visibilitychange', onVisibility);
    };
  }, [enabled, interval]);
  return count;
}
