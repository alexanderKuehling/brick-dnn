/**
 * Who else is here and what they do: the layers they have selected and the ones they drag.
 * Kept in a small store outside React state, so that a layer only re-renders when its own
 * remote selection changes, and the builder only while someone drags.
 */
import { useSyncExternalStore } from 'react';
import type { XYPosition } from '@xyflow/react';
import type { DragPresence, Peer } from './protocol';

/** Someone in the project, with the colour that marks their selection. */
export interface RemotePerson {
  clientId: string;
  userId: number;
  username: string;
  /** A Mantine colour name. */
  color: string;
}

const COLORS = ['grape', 'teal', 'orange', 'pink', 'cyan', 'indigo', 'lime', 'red', 'violet'];

/** The same colour for a person everywhere (avatars, selection outlines). */
export function userColor(username: string): string {
  let hash = 0;
  for (const char of username) {
    hash = (hash * 31 + char.charCodeAt(0)) | 0;
  }
  return COLORS[Math.abs(hash) % COLORS.length];
}

const EMPTY: RemotePerson[] = [];
const NO_DRAGS: Record<string, XYPosition> = {};

function personOf(peer: Peer): RemotePerson {
  return {
    clientId: peer.clientId,
    userId: peer.user.id,
    username: peer.user.username,
    color: userColor(peer.user.username),
  };
}

const samePeople = (a: RemotePerson[], b: RemotePerson[]) =>
  a.length === b.length && a.every((person, index) => person.clientId === b[index].clientId);

function isPosition(value: unknown): value is XYPosition {
  const position = value as XYPosition | null;
  return (
    typeof position?.x === 'number' &&
    typeof position?.y === 'number' &&
    Number.isFinite(position.x) &&
    Number.isFinite(position.y)
  );
}

/** The layers a peer drags, from its presence (one or a list; malformed ones are skipped). */
function dragsOf(peer: Peer): DragPresence[] {
  const { dragging } = peer.state;
  const list = Array.isArray(dragging) ? dragging : dragging ? [dragging] : [];
  return list.filter(
    (item): item is DragPresence => typeof item?.id === 'string' && isPosition(item.position)
  );
}

function selectedOf(peer: Peer): string[] {
  const { selected } = peer.state;
  return Array.isArray(selected) ? selected.filter((id) => typeof id === 'string') : [];
}

export class PresenceStore {
  private peers: Peer[] = [];
  private selections = new Map<string, RemotePerson[]>();
  private drags: Record<string, XYPosition> = NO_DRAGS;
  private people: RemotePerson[] = EMPTY;
  private readonly listeners = new Set<() => void>();

  update(peers: Peer[]): void {
    if (peers === this.peers) {
      return;
    }
    this.peers = peers;

    const selections = new Map<string, RemotePerson[]>();
    const drags: Record<string, XYPosition> = {};
    const people: RemotePerson[] = [];
    for (const peer of peers) {
      const person = personOf(peer);
      if (!people.some((known) => known.userId === person.userId)) {
        people.push(person);
      }
      for (const id of selectedOf(peer)) {
        selections.set(id, [...(selections.get(id) ?? []), person]);
      }
      for (const drag of dragsOf(peer)) {
        drags[drag.id] = drag.position;
      }
    }
    // Unchanged lists keep their identity, so that their layers do not re-render.
    for (const [id, list] of selections) {
      const old = this.selections.get(id);
      if (old && samePeople(old, list)) {
        selections.set(id, old);
      }
    }
    this.selections = selections;
    const dragIds = Object.keys(drags);
    const sameDrags =
      dragIds.length === Object.keys(this.drags).length &&
      dragIds.every((id) => this.drags[id]?.x === drags[id].x && this.drags[id]?.y === drags[id].y);
    if (!sameDrags) {
      this.drags = dragIds.length > 0 ? drags : NO_DRAGS;
    }
    if (!samePeople(this.people, people)) {
      this.people = people.length > 0 ? people : EMPTY;
    }
    this.listeners.forEach((listener) => listener());
  }

  /** The people who have this layer selected. */
  selectionOf = (id: string): RemotePerson[] => this.selections.get(id) ?? EMPTY;

  /** Positions of layers that others are dragging right now. */
  getDrags = (): Record<string, XYPosition> => this.drags;

  /** Everyone here (one entry per person, even with several tabs open). */
  getPeople = (): RemotePerson[] => this.people;

  /** Everyone here with their presence state (one entry per browser tab). */
  getPeers = (): Peer[] => this.peers;

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  };
}

const noSubscribe = () => () => undefined;

/** The people who have selected the layer `id` (for its outline and name tag). */
export function useRemoteSelection(store: PresenceStore | null, id: string): RemotePerson[] {
  return useSyncExternalStore(
    store?.subscribe ?? noSubscribe,
    () => store?.selectionOf(id) ?? EMPTY
  );
}

export function useRemoteDrags(store: PresenceStore): Record<string, XYPosition> {
  return useSyncExternalStore(store.subscribe, store.getDrags);
}

export function usePeople(store: PresenceStore): RemotePerson[] {
  return useSyncExternalStore(store.subscribe, store.getPeople);
}
