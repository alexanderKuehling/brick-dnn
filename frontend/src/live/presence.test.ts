import { vi } from 'vitest';
import { PresenceStore, userColor } from './presence';
import type { Peer } from './protocol';

const MARIA = { id: 2, username: 'maria' };
const TOM = { id: 3, username: 'tom' };

describe('PresenceStore', () => {
  it('collects selections, drags and people from the peers', () => {
    const store = new PresenceStore();
    const listener = vi.fn();
    store.subscribe(listener);
    const peers: Peer[] = [
      { clientId: 'a', user: MARIA, state: { selected: ['fc'], dragging: null } },
      {
        clientId: 'b',
        user: TOM,
        state: { selected: ['fc', 'in'], dragging: { id: 'act', position: { x: 5, y: 6 } } },
      },
      // The same person in a second tab, dragging two layers (one malformed).
      {
        clientId: 'c',
        user: MARIA,
        state: {
          dragging: [
            { id: 'x', position: { x: 1, y: 2 } },
            { id: 'y', position: { x: 'no' } } as never,
          ],
        },
      },
    ];
    store.update(peers);
    expect(listener).toHaveBeenCalledTimes(1);
    expect(store.selectionOf('fc').map((p) => p.username)).toEqual(['maria', 'tom']);
    expect(store.selectionOf('fc')[0].color).toBe(userColor('maria'));
    expect(store.selectionOf('nothing')).toEqual([]);
    expect(store.getDrags()).toEqual({ act: { x: 5, y: 6 }, x: { x: 1, y: 2 } });
    expect(store.getPeople().map((p) => p.username)).toEqual(['maria', 'tom']);

    // Unchanged parts keep their identity (so their layers do not re-render).
    const fc = store.selectionOf('fc');
    const drags = store.getDrags();
    store.update([...peers.slice(0, 2), { ...peers[2], state: {} }]);
    expect(store.selectionOf('fc')).toBe(fc);
    expect(store.getDrags()).not.toBe(drags);
    expect(store.getDrags()).toEqual({ act: { x: 5, y: 6 } });
  });

  it('gives everyone the same colour everywhere', () => {
    expect(userColor('maria')).toBe(userColor('maria'));
    expect(typeof userColor('')).toBe('string');
  });
});
