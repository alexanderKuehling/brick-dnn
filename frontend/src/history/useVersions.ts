import { useCallback, useEffect, useRef, useState } from 'react';
import { api, type ProjectVersion } from '@/api';
import type { SubscribeToEvents } from '@/collab/events';
import { useDebounced, usePolling, useProjectEvents } from '@/collab/useProjectEvents';

/** Without a live connection, the open history is loaded again this often. */
export const HISTORY_POLL_MS = 30_000;
/** Live, the open history is loaded again this long after the last change of the network. */
export const HISTORY_SETTLE_MS = 3_000;

/**
 * The history of a project while it is shown: loaded when mounted (the panel opens) and again
 * when a live `history` message arrives, shortly after changes (they update the latest
 * automatic entry without a message), or every 30 seconds while not `live`.
 */
export function useVersions(
  projectId: number,
  subscribe?: SubscribeToEvents | null,
  live: boolean = false
) {
  const [versions, setVersions] = useState<ProjectVersion[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const latest = useRef(0);

  const reload = useCallback(async () => {
    const request = ++latest.current;
    try {
      const result = await api.projects.versions.list(projectId);
      if (request === latest.current) {
        setVersions(result);
        setError(null);
      }
    } catch (err) {
      if (request === latest.current) {
        setError((err as Error).message);
      }
    }
  }, [projectId]);

  useEffect(() => {
    setVersions(null);
    setError(null);
    void reload();
    return () => {
      latest.current += 1; // Answers for another project (or after closing) are dropped.
    };
  }, [reload]);

  const settled = useDebounced(() => void reload(), HISTORY_SETTLE_MS);
  useProjectEvents(subscribe, {
    onHistory: () => void reload(),
    onChange: settled,
    onConnected: () => void reload(),
  });
  usePolling(() => void reload(), HISTORY_POLL_MS, !live);

  /** Shows a new or changed entry right away (before the list is loaded again). */
  const replace = useCallback(
    (entry: ProjectVersion) =>
      setVersions((current) =>
        current?.some((v) => v.id === entry.id)
          ? current.map((v) => (v.id === entry.id ? entry : v))
          : [entry, ...(current ?? [])]
      ),
    []
  );

  return { versions, error, reload, replace };
}
