import { act, renderHook } from '@test-utils';
import { afterEach, vi } from 'vitest';
import type { ProjectEvent } from './events';
import { comment } from './fixtures';
import { useDebounced, usePolling, useProjectEvents } from './useProjectEvents';

function fakeConnection() {
  const handlers = new Set<(event: ProjectEvent) => void>();
  return {
    handlers,
    subscribe: (handler: (event: ProjectEvent) => void) => {
      handlers.add(handler);
      return () => {
        handlers.delete(handler);
      };
    },
    send: (event: object) => handlers.forEach((handler) => handler(event as ProjectEvent)),
  };
}

afterEach(() => {
  vi.useRealTimers();
});

describe('useProjectEvents', () => {
  it('passes comment, history, change and connection messages on', () => {
    const connection = fakeConnection();
    const handlers = {
      onComment: vi.fn(),
      onHistory: vi.fn(),
      onChange: vi.fn(),
      onConnected: vi.fn(),
    };
    const { unmount } = renderHook(() => useProjectEvents(connection.subscribe, handlers));

    const created = comment({ id: 4, text: 'Hi' });
    connection.send({ type: 'comment', action: 'created', comment: created });
    connection.send({ type: 'comment', action: 'deleted', comment: { id: 4 } });
    expect(handlers.onComment).toHaveBeenNthCalledWith(1, {
      type: 'comment',
      action: 'created',
      comment: created,
    });
    expect(handlers.onComment).toHaveBeenNthCalledWith(2, {
      type: 'comment',
      action: 'deleted',
      comment: { id: 4 },
    });

    connection.send({ type: 'history' });
    connection.send({ type: 'ops', ops: [], version: 4 });
    connection.send({ type: 'replaced', project: {} });
    connection.send({ type: 'hello', project: {} });
    connection.send({ type: 'presence', state: {} });
    expect(handlers.onHistory).toHaveBeenCalledTimes(1);
    expect(handlers.onChange).toHaveBeenCalledTimes(2);
    expect(handlers.onConnected).toHaveBeenCalledTimes(1);

    unmount();
    expect(connection.handlers.size).toBe(0);
  });

  it('ignores malformed comment messages', () => {
    const connection = fakeConnection();
    const onComment = vi.fn();
    renderHook(() => useProjectEvents(connection.subscribe, { onComment }));
    connection.send({ type: 'comment', action: 'moved', comment: { id: 1 } });
    connection.send({ type: 'comment', action: 'created', comment: { id: 1 } });
    connection.send({ type: 'comment', action: 'deleted' });
    expect(onComment).not.toHaveBeenCalled();
  });

  it('uses the latest handlers without subscribing again', () => {
    const connection = fakeConnection();
    const subscribe = vi.fn(connection.subscribe);
    const first = vi.fn();
    const second = vi.fn();
    const { rerender } = renderHook(({ onHistory }) => useProjectEvents(subscribe, { onHistory }), {
      initialProps: { onHistory: first },
    });
    rerender({ onHistory: second });
    connection.send({ type: 'history' });
    expect(first).not.toHaveBeenCalled();
    expect(second).toHaveBeenCalledTimes(1);
    expect(subscribe).toHaveBeenCalledTimes(1);
  });

  it('does nothing without a connection', () => {
    expect(() => renderHook(() => useProjectEvents(null, { onHistory: vi.fn() }))).not.toThrow();
  });
});

describe('usePolling', () => {
  it('calls back at the interval while enabled', () => {
    vi.useFakeTimers();
    const callback = vi.fn();
    const { rerender, unmount } = renderHook(
      ({ enabled }) => usePolling(callback, 15_000, enabled),
      { initialProps: { enabled: true } }
    );
    act(() => vi.advanceTimersByTime(15_000));
    expect(callback).toHaveBeenCalledTimes(1);
    act(() => vi.advanceTimersByTime(30_000));
    expect(callback).toHaveBeenCalledTimes(3);

    rerender({ enabled: false }); // E.g. the live connection is up again.
    act(() => vi.advanceTimersByTime(60_000));
    expect(callback).toHaveBeenCalledTimes(3);

    rerender({ enabled: true });
    unmount();
    act(() => vi.advanceTimersByTime(60_000));
    expect(callback).toHaveBeenCalledTimes(3);
  });

  it('waits while the page is hidden', () => {
    vi.useFakeTimers();
    const callback = vi.fn();
    const visibility = vi.spyOn(document, 'visibilityState', 'get').mockReturnValue('hidden');
    renderHook(() => usePolling(callback, 1_000, true));
    act(() => vi.advanceTimersByTime(5_000));
    expect(callback).not.toHaveBeenCalled();
    visibility.mockReturnValue('visible');
    act(() => vi.advanceTimersByTime(1_000));
    expect(callback).toHaveBeenCalledTimes(1);
    visibility.mockRestore();
  });
});

describe('useDebounced', () => {
  it('calls back once after a burst', () => {
    vi.useFakeTimers();
    const callback = vi.fn();
    const { result } = renderHook(() => useDebounced(callback, 3_000));
    act(() => {
      result.current();
      vi.advanceTimersByTime(2_000);
      result.current();
      vi.advanceTimersByTime(2_000);
    });
    expect(callback).not.toHaveBeenCalled();
    act(() => vi.advanceTimersByTime(1_000));
    expect(callback).toHaveBeenCalledTimes(1);
  });
});
