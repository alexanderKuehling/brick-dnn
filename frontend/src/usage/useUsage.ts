import { useCallback, useEffect, useState } from 'react';
import { api, type Api, type Usage } from '@/api';

/** The API, unless this test has none. */
function usageApi(): Api['account']['usage'] | undefined {
  const account = (api as { account?: Partial<Api['account']> }).account;
  return typeof account?.usage === 'function' ? account.usage : undefined;
}

/**
 * This month's usage and limits of the account, for the places that show them besides the
 * account page. Loaded when `enabled`, again with `reload` and whenever the tab gets the
 * focus back (the plan may have been changed in another tab). Null while loading and when
 * the server does not answer: the usage is only shown, never needed.
 */
export function useUsage(enabled: boolean): { usage: Usage | null; reload: () => void } {
  const [usage, setUsage] = useState<Usage | null>(null);
  const [request, setRequest] = useState(0);
  const reload = useCallback(() => setRequest((n) => n + 1), []);

  useEffect(() => {
    const load = usageApi();
    if (!enabled || !load) {
      setUsage(null);
      return undefined;
    }
    const controller = new AbortController();
    Promise.resolve()
      .then(() => load(controller.signal))
      .then((result) => !controller.signal.aborted && setUsage(result ?? null))
      .catch(() => undefined);
    return () => controller.abort();
  }, [enabled, request]);

  useEffect(() => {
    if (!enabled) {
      return undefined;
    }
    window.addEventListener('focus', reload);
    return () => window.removeEventListener('focus', reload);
  }, [enabled, reload]);

  return { usage, reload };
}
