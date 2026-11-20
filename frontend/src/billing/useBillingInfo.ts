import { useEffect, useSyncExternalStore } from 'react';
import { api, type Api, type BillingEnabled, type BillingInfo } from '@/api';

/** undefined while loading; null if the server sells no plans (or did not answer). */
type State = BillingEnabled | null | undefined;

let state: State;
let loading = false;
/** The server did not answer: unlike "sells no plans", asking again may help. */
let failed = false;
const listeners = new Set<() => void>();

/** The API, unless this test has none. */
function billingApi(): Api['billing'] | undefined {
  const billing = (api as { billing?: Api['billing'] }).billing;
  return typeof billing?.info === 'function' ? billing : undefined;
}

function load(): void {
  const billing = billingApi();
  if (loading || state !== undefined || !billing) {
    return;
  }
  loading = true;
  Promise.resolve()
    .then(() => billing.info())
    .then(
      (info): [BillingInfo, boolean] => [info, false],
      // Without an answer nothing of billing is shown until it is asked again.
      (): [BillingInfo, boolean] => [{ enabled: false }, true]
    )
    .then(([info, error]) => {
      state = info?.enabled ? info : null;
      failed = error;
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

/** Ask the server again (after it did not answer). */
export function retryBillingInfo(): void {
  if (loading) {
    return;
  }
  state = undefined;
  failed = false;
  listeners.forEach((listener) => listener());
  load();
}

/**
 * `useBillingInfo` for the pricing page, which says why there are no plans: `failed` when
 * the server did not answer (with `retry`), otherwise it sells none.
 */
export function useBillingStatus(): { info: State; failed: boolean; retry: () => void } {
  const info = useBillingInfo();
  const error = useSyncExternalStore(subscribe, () => failed);
  return { info, failed: info === null && error, retry: retryBillingInfo };
}

/** For tests: forget what was loaded. */
export function resetBillingInfo(): void {
  state = undefined;
  loading = false;
  failed = false;
}
