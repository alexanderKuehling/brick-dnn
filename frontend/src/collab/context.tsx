import { createContext, useContext, useState, useSyncExternalStore } from 'react';
import { CollabStore, type CollabState } from './store';

/**
 * History and comments of the network open in the builder. null outside the builder and in
 * read-only views of other versions: layers and the inspector show no comments there.
 */
export const CollabContext = createContext<CollabStore | null>(null);

/** Provides the store to everything below it (the builder: canvas, inspector and panels). */
export function CollabProvider({ children }: { children: React.ReactNode }) {
  const [store] = useState(() => new CollabStore());
  return <CollabContext.Provider value={store}>{children}</CollabContext.Provider>;
}

/** Hides comments from the views below (e.g. the preview of an old version). */
export function WithoutCollab({ children }: { children: React.ReactNode }) {
  return <CollabContext.Provider value={null}>{children}</CollabContext.Provider>;
}

export function useCollabStore(): CollabStore | null {
  return useContext(CollabContext);
}

/** The whole state of a store, updated when it changes. */
export function useStoreState(store: CollabStore): CollabState {
  return useSyncExternalStore(store.subscribe, store.getState);
}

const noSubscription = () => () => undefined;

/**
 * A value from the state, updated when it changes; `fallback` without a store. `select` must
 * return the same value for the same state (a part of it or a primitive, nothing new).
 */
export function useCollab<T>(select: (state: CollabState) => T, fallback: T): T {
  const store = useContext(CollabContext);
  return useSyncExternalStore(store?.subscribe ?? noSubscription, () =>
    store ? select(store.getState()) : fallback
  );
}
