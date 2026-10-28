import { act, renderHook } from '@testing-library/react';
import { afterEach, beforeEach, vi } from 'vitest';
import { notifications } from '@mantine/notifications';
import { api, type RunSummary } from '@/api';
import { detail, STATUS, summary } from './fixtures';
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
});
