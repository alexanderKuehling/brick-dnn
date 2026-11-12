import { useEffect, useSyncExternalStore } from 'react';
import { api, type Api, type LegalInfo, type LegalPageName } from '@/api';
import { DEMO } from '@/demo';

/** undefined while loading; null if there is nothing to show (the demo, or no answer). */
type State = LegalInfo | null | undefined;

let state: State;
let loading = false;
const listeners = new Set<() => void>();

/** The API, unless this build or test has none: the demo has no server. */
function legalApi(): Api['legal'] | undefined {
  const legal = DEMO ? undefined : (api as { legal?: Api['legal'] }).legal;
  return typeof legal?.info === 'function' ? legal : undefined;
}

function load(): void {
  const legal = legalApi();
  if (loading || state !== undefined || !legal) {
    return;
  }
  loading = true;
  legal
    .info()
    // Without an answer the links are left out until the page is loaded again.
    .catch(() => null)
    .then((info) => {
      state = info;
      loading = false;
      listeners.forEach((listener) => listener());
    });
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

function snapshot(): State {
  return legalApi() ? state : null;
}

/** Which legal pages the server has, loaded once per page load. */
export function useLegalInfo(): State {
  const info = useSyncExternalStore(subscribe, snapshot);
  useEffect(load, []);
  return info;
}

/** The legal pages the server has, in menu order. */
export function legalPages(info: State): LegalPageName[] {
  return info
    ? (['imprint', 'privacy', 'terms', 'refund'] as const).filter((page) => info.pages[page])
    : [];
}

export const LEGAL_LABELS: Record<LegalPageName, string> = {
  imprint: 'Imprint',
  privacy: 'Privacy policy',
  terms: 'Terms of use',
  refund: 'Refund policy',
};

/** For tests: forget what was loaded. */
export function resetLegalInfo(): void {
  state = undefined;
  loading = false;
}
