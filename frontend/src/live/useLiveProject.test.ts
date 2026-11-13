import { act, FakeWebSocket, renderHook, waitFor } from '@test-utils';
import { afterEach, beforeEach, vi } from 'vitest';
import { setTokens } from '@/api';
import type { ServerMessage } from './protocol';
import { liveUrl, OFFLINE_GRACE_MS, useLiveProject } from './useLiveProject';

const hello = (id: number) => ({
  type: 'hello',
  project: {
    id,
    name: 'MLP',
    description: '',
    graph: { version: 1, nodes: [], edges: [] },
    version: 1,
  },
  role: 'editor',
  client_id: 'me',
  user: { id: 1, username: 'alex' },
});

beforeEach(() => {
  FakeWebSocket.reset();
  vi.stubGlobal('WebSocket', FakeWebSocket);
  setTokens({ access: 'access-token', refresh: 'refresh-token' });
});

afterEach(() => {
  vi.unstubAllGlobals();
  setTokens(null);
});

describe('useLiveProject', () => {
  it('derives the WebSocket address from the API address', () => {
    expect(liveUrl(7)).toBe('ws://localhost:3000/api/ws/projects/7/');
  });

  it('connects, reports its status and passes messages on', async () => {
    const received: ServerMessage[] = [];
    const { result } = renderHook(() => useLiveProject(5, true));
    expect(result.current.status).toBe('connecting');
    const off = result.current.subscribe((message) => received.push(message));

    await waitFor(() => expect(FakeWebSocket.instances).toHaveLength(1));
    const socket = FakeWebSocket.last;
    act(() => {
      socket.open();
      socket.receive(hello(5));
    });
    expect(result.current.status).toBe('live');
    expect(result.current.role).toBe('editor');
    expect(received.map((message) => message.type)).toEqual(['hello']);

    act(() => {
      socket.receive({ type: 'joined', client_id: 'p1', user: { id: 2, username: 'maria' } });
      result.current.sendPresence({ selected: ['fc'] });
      result.current.sendOps([{ op: 'remove_node', id: 'fc' }]);
    });
    expect(result.current.presence.getPeople().map((person) => person.username)).toEqual(['maria']);
    expect(socket.messages('presence')).toEqual([
      { type: 'presence', state: { selected: ['fc'] } },
    ]);
    expect(result.current.pendingCount).toBe(1);
    off();
  });

  it('shows a dropped connection as offline only when it stays down', async () => {
    const { result } = renderHook(() => useLiveProject(5, true));
    await waitFor(() => expect(FakeWebSocket.instances).toHaveLength(1));
    const first = FakeWebSocket.last;
    act(() => {
      first.open();
      first.receive(hello(5));
    });
    expect(result.current.status).toBe('live');

    vi.useFakeTimers();
    try {
      // A short drop: reconnected before the grace period ends, the badge never says offline.
      act(() => first.drop(1006));
      expect(result.current.status).toBe('live');
      await act(async () => {
        await vi.advanceTimersByTimeAsync(1500);
      });
      expect(FakeWebSocket.instances).toHaveLength(2);
      const second = FakeWebSocket.last;
      act(() => {
        second.open();
        second.receive(hello(5));
      });
      await act(async () => {
        await vi.advanceTimersByTimeAsync(OFFLINE_GRACE_MS);
      });
      expect(result.current.status).toBe('live');

      // A drop that lasts: offline once the grace period is over.
      act(() => second.drop(1006));
      await act(async () => {
        await vi.advanceTimersByTimeAsync(OFFLINE_GRACE_MS - 100);
      });
      expect(result.current.status).toBe('live');
      await act(async () => {
        await vi.advanceTimersByTimeAsync(200);
      });
      expect(result.current.status).toBe('offline');
    } finally {
      vi.useRealTimers();
    }
  });

  it('stays off without a project or when disabled, and switches projects', async () => {
    const { result, rerender } = renderHook(
      ({ id, enabled }: { id: number | null; enabled: boolean }) => useLiveProject(id, enabled),
      { initialProps: { id: null as number | null, enabled: true } }
    );
    expect(result.current.status).toBe('off');
    rerender({ id: 5, enabled: false });
    expect(result.current.status).toBe('off');
    expect(FakeWebSocket.instances).toHaveLength(0);

    rerender({ id: 5, enabled: true });
    await waitFor(() => expect(FakeWebSocket.instances).toHaveLength(1));
    const first = FakeWebSocket.last;
    rerender({ id: 6, enabled: true });
    expect(first.closedByClient).toBe(1000);
    await waitFor(() => expect(FakeWebSocket.instances).toHaveLength(2));
    expect(FakeWebSocket.last.url).toBe('ws://localhost:3000/api/ws/projects/6/');
  });
});
