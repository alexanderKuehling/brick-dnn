/**
 * Learning tips: one short sentence with a link to a guide, shown where a network or a run
 * gives a reason for it (tips.ts has the conditions, LearningTip.tsx shows them). This file
 * keeps what the user decided about them:
 *
 * - **All tips on or off.** Kept in the browser (localStorage) and, when logged in, on the
 *   account (`UserProfile.preferences`, `PATCH /api/auth/account/preferences/`), so it follows
 *   the user to other devices. After a login the account's choice wins. The switch is on the
 *   account page; "Don't show tips" in a tip turns them off.
 * - **Tips hidden one by one**, each for one network or run (`dismissTip`). Kept in the browser
 *   only, the newest 200.
 *
 * Without storage (private mode) both hold until the page is reloaded. Errors and warnings of
 * the checks are not tips and never go through here.
 */
import { useSyncExternalStore } from 'react';
import { api, getTokens, type User } from '@/api';

/** `"off"` while learning tips are turned off in this browser. */
export const LEARNING_TIPS_KEY = 'brickdnn:learning-tips';
/** The tips hidden one by one: a JSON list of keys, the newest last. */
export const DISMISSED_TIPS_KEY = 'brickdnn:dismissed-tips';
const MAX_DISMISSED = 200;

// What could not be stored, and tips hidden for this visit only.
let memoryEnabled: boolean | null = null;
const memoryDismissed = new Set<string>();

let version = 0;
const listeners = new Set<() => void>();

function changed(): void {
  version += 1;
  listeners.forEach((listener) => listener());
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

/** Whether learning tips are shown at all. */
export function learningTipsEnabled(): boolean {
  if (memoryEnabled !== null) {
    return memoryEnabled;
  }
  try {
    return localStorage.getItem(LEARNING_TIPS_KEY) !== 'off';
  } catch {
    return true;
  }
}

function storeEnabled(on: boolean): void {
  try {
    if (on) {
      localStorage.removeItem(LEARNING_TIPS_KEY);
    } else {
      localStorage.setItem(LEARNING_TIPS_KEY, 'off');
    }
    memoryEnabled = null;
  } catch {
    memoryEnabled = on; // Storage unavailable: the choice holds until the page is reloaded.
  }
}

/**
 * Turns all learning tips on or off: in this browser and, when logged in, on the account
 * (`save: false` only follows what the account says). A failed save only costs the sync to
 * other devices.
 */
export function setLearningTips(on: boolean, { save = true }: { save?: boolean } = {}): void {
  storeEnabled(on);
  changed();
  const send = (api as { account?: { setPreferences?: unknown } }).account?.setPreferences;
  if (save && typeof send === 'function' && getTokens()) {
    void Promise.resolve(api.account.setPreferences({ learning_tips: on })).catch(() => undefined);
  }
}

/** After a login: the choice saved on the account wins over the browser's. */
export function adoptLearningTips(user: User): void {
  if (typeof user.learning_tips === 'boolean' && user.learning_tips !== learningTipsEnabled()) {
    setLearningTips(user.learning_tips, { save: false });
  }
}

function storedDismissed(): string[] {
  try {
    const data: unknown = JSON.parse(localStorage.getItem(DISMISSED_TIPS_KEY) ?? '[]');
    return Array.isArray(data) ? data.filter((key): key is string => typeof key === 'string') : [];
  } catch {
    return [];
  }
}

/** Whether the tip with this key (a tip and the network or run it was shown for) is hidden. */
export function isTipDismissed(key: string): boolean {
  return memoryDismissed.has(key) || storedDismissed().includes(key);
}

/**
 * Hides one tip for one network or run. `persist: false` hides it for this visit only (a
 * network that is not saved has nothing to remember it by).
 */
export function dismissTip(key: string, { persist = true }: { persist?: boolean } = {}): void {
  memoryDismissed.add(key);
  if (persist) {
    try {
      const keys = storedDismissed().filter((other) => other !== key);
      keys.push(key);
      localStorage.setItem(DISMISSED_TIPS_KEY, JSON.stringify(keys.slice(-MAX_DISMISSED)));
    } catch {
      // Storage unavailable: hidden until the page is reloaded.
    }
  }
  changed();
}

/** For tests: forget everything decided about tips. */
export function resetLearningTips(): void {
  memoryEnabled = null;
  memoryDismissed.clear();
  try {
    localStorage.removeItem(LEARNING_TIPS_KEY);
    localStorage.removeItem(DISMISSED_TIPS_KEY);
  } catch {
    // No storage.
  }
  changed();
}

// Another tab changed the choice: follow along.
if (typeof window !== 'undefined') {
  window.addEventListener('storage', (event) => {
    if (event.key === LEARNING_TIPS_KEY || event.key === DISMISSED_TIPS_KEY || event.key === null) {
      changed();
    }
  });
}

/** Re-renders when tips are turned on or off or one is hidden; returns whether they are on. */
export function useLearningTips(): boolean {
  useSyncExternalStore(
    subscribe,
    () => version,
    () => version
  );
  return learningTipsEnabled();
}
