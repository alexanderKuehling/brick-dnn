import { act, render, renderHook } from '@testing-library/react';
import { afterEach, beforeEach, vi } from 'vitest';
import { MantineProvider } from '@mantine/core';
import { notifications } from '@mantine/notifications';
import { api, type RunSummary } from '@/api';
import { SUPPORT_URL } from '@/links';
import { detail, STATUS, summary } from './fixtures';
import { SUPPORT_HINT_KEY } from './supportHint';
import { DETAIL_INTERVAL, LIST_INTERVAL, useTraining } from './useTraining';

vi.mock('@/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api')>();
  return {
    ...actual,
    api: {
      training: {
        status: vi.fn(),
        runs: { list: vi.fn(), get: vi.fn() },
      },
    },
  };
});

const flush = () => act(() => vi.advanceTimersByTimeAsync(0));
const wait = (ms: number) => act(() => vi.advanceTimersByTimeAsync(ms));

beforeEach(() => {
  vi.useFakeTimers();
  vi.clearAllMocks();
  vi.spyOn(notifications, 'show').mockImplementation(() => 'id');
  vi.mocked(api.training.status).mockResolvedValue(STATUS);
  // The coffee hint after the first training has its own tests below.
  localStorage.setItem(SUPPORT_HINT_KEY, 'shown');
});

afterEach(() => {
  vi.useRealTimers();
});

describe('useTraining', () => {
  it('watches an active run while the panel is closed and reports the result', async () => {
    vi.mocked(api.training.runs.list)
      .mockResolvedValueOnce([summary()])
      .mockResolvedValueOnce([summary({ epochs_done: 5 })])
      .mockResolvedValue([summary({ status: 'completed', epochs_done: 10 })]);
    const { result } = renderHook(() => useTraining(true));
    await flush();
    expect(result.current.activeRun?.id).toBe('r1');

    await wait(LIST_INTERVAL);
    expect(result.current.activeRun?.epochs_done).toBe(5);
    expect(notifications.show).not.toHaveBeenCalled();

    await wait(LIST_INTERVAL);
    expect(result.current.activeRun).toBeNull();
    expect(notifications.show).toHaveBeenCalledWith(
      expect.objectContaining({ title: 'Training finished', color: 'green' })
    );

    // Nothing is active any more, so polling stops.
    const calls = vi.mocked(api.training.runs.list).mock.calls.length;
    await wait(LIST_INTERVAL * 4);
    expect(api.training.runs.list).toHaveBeenCalledTimes(calls);
  });

  it('polls the shown run every second and stops when it ends', async () => {
    vi.mocked(api.training.runs.list).mockResolvedValue([summary()]);
    vi.mocked(api.training.runs.get)
      .mockResolvedValueOnce(detail())
      .mockResolvedValue(detail({ status: 'completed', epochs_done: 10 }));
    const { result } = renderHook(() => useTraining(true));
    await flush();

    act(() => result.current.open());
    await flush();
    expect(result.current.view).toBe('run');
    expect(result.current.detail?.status).toBe('running');
    expect(api.training.runs.get).toHaveBeenCalledTimes(1);

    await wait(DETAIL_INTERVAL);
    expect(result.current.detail?.status).toBe('completed');
    expect(api.training.runs.get).toHaveBeenCalledTimes(2);
    // The user watched it finish: no notification needed.
    expect(notifications.show).not.toHaveBeenCalled();

    await wait(DETAIL_INTERVAL * 5);
    expect(api.training.runs.get).toHaveBeenCalledTimes(2);
  });

  it('does nothing for visitors', async () => {
    const { result } = renderHook(() => useTraining(false));
    await flush();
    expect(result.current.status).toEqual(STATUS);
    expect(result.current.runs).toBeNull();
    expect(api.training.runs.list).not.toHaveBeenCalled();
  });

  it('keeps runs (and their polling) when PyTorch is missing on the server', async () => {
    vi.mocked(api.training.status).mockResolvedValue({
      ...STATUS,
      available: false,
      reason: 'PyTorch is not installed on the server.',
    });
    vi.mocked(api.training.runs.list).mockResolvedValue([summary()]);
    const { result } = renderHook(() => useTraining(true));
    await flush();
    expect(result.current.activeRun?.id).toBe('r1');
    await wait(LIST_INTERVAL);
    expect(api.training.runs.list).toHaveBeenCalledTimes(2);
  });

  it('does nothing when training is disabled', async () => {
    vi.mocked(api.training.status).mockResolvedValue({ ...STATUS, enabled: false });
    renderHook(() => useTraining(true));
    await wait(LIST_INTERVAL * 2);
    expect(api.training.runs.list).not.toHaveBeenCalled();
  });

  it('keeps the run list current while one run is watched', async () => {
    vi.mocked(api.training.runs.list).mockResolvedValue([summary()]);
    vi.mocked(api.training.runs.get).mockResolvedValue(detail());
    const { result } = renderHook(() => useTraining(true));
    await flush();
    act(() => result.current.open());
    await flush();
    const before = vi.mocked(api.training.runs.list).mock.calls.length;
    await wait(LIST_INTERVAL);
    expect(api.training.runs.list).toHaveBeenCalledTimes(before + 1);
    expect(vi.mocked(api.training.runs.get).mock.calls.length).toBeGreaterThanOrEqual(5);
  });

  it('does not let an older list response undo a finished run', async () => {
    let resolveStale!: (list: RunSummary[]) => void;
    vi.mocked(api.training.runs.list)
      .mockResolvedValueOnce([summary()])
      // The refresh when the panel opens is slow and answers with what was true back then.
      .mockReturnValueOnce(new Promise((resolve) => (resolveStale = resolve)))
      .mockResolvedValue([summary({ status: 'completed', epochs_done: 10 })]);
    vi.mocked(api.training.runs.get).mockResolvedValue(
      detail({ status: 'completed', epochs_done: 10 })
    );
    const { result } = renderHook(() => useTraining(true));
    await flush();

    act(() => result.current.open());
    await flush();
    expect(result.current.runs?.[0].status).toBe('completed');

    await act(async () => resolveStale([summary({ epochs_done: 4 })]));
    expect(result.current.runs?.[0].status).toBe('completed');
    expect(result.current.activeRun).toBeNull();

    // Closing the panel must not report the run as finished a second time.
    act(() => result.current.close());
    act(() => result.current.refreshRuns());
    await wait(LIST_INTERVAL);
    expect(notifications.show).not.toHaveBeenCalled();
  });

  it('does not bring back a deleted run', async () => {
    let resolveStale!: (list: RunSummary[]) => void;
    vi.mocked(api.training.runs.list)
      .mockResolvedValueOnce([summary({ status: 'completed' })])
      .mockReturnValueOnce(new Promise((resolve) => (resolveStale = resolve)));
    const { result } = renderHook(() => useTraining(true));
    await flush();
    act(() => result.current.refreshRuns());
    await flush();
    act(() => result.current.forgetRun('r1'));
    await act(async () => resolveStale([summary({ status: 'completed' })]));
    expect(result.current.runs).toEqual([]);
  });

  it('follows runs of the open network, not those of others elsewhere', async () => {
    const elsewhere = summary({ id: 'eve', started_by: 'eve', project: 9 });
    const network = summary({ id: 'bob', started_by: 'bob', project: 5 });
    vi.mocked(api.training.runs.list).mockResolvedValue([elsewhere, network]);
    const { result, rerender } = renderHook(
      ({ project }: { project: number | null }) =>
        useTraining(true, {
          username: 'ada',
          project: project === null ? null : { id: project, role: 'viewer', shared: true },
        }),
      { initialProps: { project: 5 as number | null } }
    );
    await flush();
    expect(result.current.activeRun?.id).toBe('bob');
    expect(result.current.scope.project).toEqual({ id: 5, role: 'viewer', shared: true });

    rerender({ project: null });
    expect(result.current.activeRun).toBeNull();
  });

  it('reports finished runs of the network with who started them', async () => {
    const network = summary({ id: 'bob', name: 'Bob’s run', started_by: 'bob', project: 5 });
    const elsewhere = summary({ id: 'eve', started_by: 'eve', project: 9 });
    vi.mocked(api.training.runs.list)
      .mockResolvedValueOnce([network, elsewhere])
      .mockResolvedValue([
        { ...network, status: 'completed' },
        { ...elsewhere, status: 'completed' },
      ]);
    renderHook(() =>
      useTraining(true, { username: 'ada', project: { id: 5, role: 'editor', shared: true } })
    );
    await flush();
    await wait(LIST_INTERVAL);
    expect(notifications.show).toHaveBeenCalledTimes(1);
    const { message } = vi.mocked(notifications.show).mock.calls[0][0];
    const { container } = render(<MantineProvider>{message}</MantineProvider>);
    expect(container).toHaveTextContent('“Bob’s run” by bob');
  });
});

describe('coffee hint after the first successful training', () => {
  const titles = () => vi.mocked(notifications.show).mock.calls.map(([n]) => n.title);

  beforeEach(() => localStorage.removeItem(SUPPORT_HINT_KEY));

  it('asks once, also when the user watches the run', async () => {
    vi.mocked(api.training.runs.list).mockResolvedValue([summary()]);
    vi.mocked(api.training.runs.get)
      .mockResolvedValueOnce(detail())
      .mockResolvedValue(detail({ status: 'completed', epochs_done: 10 }));
    const { result } = renderHook(() => useTraining(true));
    await flush();
    act(() => result.current.openRun('r1'));
    await wait(DETAIL_INTERVAL * 2);
    expect(titles()).toEqual(['Your first training is done!']);
    expect(localStorage.getItem(SUPPORT_HINT_KEY)).not.toBeNull();
    const { message } = vi.mocked(notifications.show).mock.calls[0][0];
    const { getByRole } = render(<MantineProvider>{message}</MantineProvider>);
    expect(getByRole('link', { name: /buy me a coffee/i })).toHaveAttribute('href', SUPPORT_URL);

    // Never again in this browser.
    vi.mocked(notifications.show).mockClear();
    vi.mocked(api.training.runs.list).mockResolvedValue([summary({ id: 'r2' })]);
    vi.mocked(api.training.runs.get)
      .mockResolvedValueOnce(detail({ id: 'r2' }))
      .mockResolvedValue(detail({ id: 'r2', status: 'completed', epochs_done: 10 }));
    act(() => result.current.openRun('r2'));
    await wait(DETAIL_INTERVAL * 2);
    expect(titles()).toEqual([]);
  });

  it('waits for a run that completes, not one that fails or is stopped', async () => {
    vi.mocked(api.training.runs.list)
      .mockResolvedValueOnce([summary({ id: 'a' }), summary({ id: 'b' })])
      .mockResolvedValue([
        summary({ id: 'a', status: 'failed' }),
        summary({ id: 'b', status: 'stopped' }),
      ]);
    renderHook(() => useTraining(true));
    await flush();
    await wait(LIST_INTERVAL);
    expect(titles()).toEqual(['Training failed', 'Training stopped']);
    expect(localStorage.getItem(SUPPORT_HINT_KEY)).toBeNull();
  });

  it('is not shown for runs that others started in the network', async () => {
    const network = summary({ id: 'bob', started_by: 'bob', project: 5 });
    vi.mocked(api.training.runs.list)
      .mockResolvedValueOnce([network])
      .mockResolvedValue([{ ...network, status: 'completed' }]);
    renderHook(() =>
      useTraining(true, { username: 'ada', project: { id: 5, role: 'editor', shared: true } })
    );
    await flush();
    await wait(LIST_INTERVAL);
    expect(titles()).toEqual(['Training finished']);
  });
});
