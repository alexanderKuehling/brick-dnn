import { useEffect, useSyncExternalStore } from 'react';
import { api, type Api, type BillingEnabled, type BillingInfo } from '@/api';
import { DEMO } from '@/demo';

/** undefined while loading; null if the server sells no plans (or did not answer). */
type State = BillingEnabled | null | undefined;

let state: State;
let loading = false;
const listeners = new Set<() => void>();

/** The API, unless this build or test has none: the demo has no server. */
function billingApi(): Api['billing'] | undefined {
  const billing = DEMO ? undefined : (api as { billing?: Api['billing'] }).billing;
  return typeof billing?.info === 'function' ? billing : undefined;
}

function load(): void {
  const billing = billingApi();
  if (loading || state !== undefined || !billing) {
    return;
  }
  loading = true;
  billing
    .info()
    // Without an answer nothing of billing is shown until the page is loaded again.
    .catch((): BillingInfo => ({ enabled: false }))
    .then((info) => {
      state = info?.enabled ? info : null;
      loading = false;
      listeners.forEach((listener) => listener());
    });
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

function snapshot(): State {
  return billingApi() ? state : null;
}

/**
 * The server's paid plans and what the checkout needs, loaded once per page load. Null when
 * the server sells none: then nothing of billing is shown anywhere.
 */
export function useBillingInfo(): State {
  const info = useSyncExternalStore(subscribe, snapshot);
  useEffect(load, []);
  return info;
}

/** For tests: forget what was loaded. */
export function resetBillingInfo(): void {
  state = undefined;
  loading = false;
}
