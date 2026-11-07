import { act, renderHook, waitFor } from '@testing-library/react';
import { beforeEach, vi } from 'vitest';
import { api, ApiError, type DataSource } from '@/api';
import { CATALOG, DATA_STATUS, OTHER_SOURCE, PREP, SOURCE } from './fixtures';
import { useDataResources } from './useDataResources';

vi.mock('@/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api')>();
  return {
    ...actual,
    api: {
      data: {
        status: vi.fn(),
        steps: vi.fn(),
        sources: { list: vi.fn() },
        preps: { list: vi.fn() },
      },
    },
  };
});

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(api.data.status).mockResolvedValue(DATA_STATUS);
  vi.mocked(api.data.steps).mockResolvedValue(CATALOG);
  vi.mocked(api.data.sources.list).mockResolvedValue([SOURCE]);
  vi.mocked(api.data.preps.list).mockResolvedValue([PREP]);
});

describe('useDataResources', () => {
  it('loads the capabilities, the steps and the lists of a logged-in user', async () => {
    const { result } = renderHook(() => useDataResources(true));
    await waitFor(() => expect(result.current.preps).toEqual([PREP]));
    expect(result.current.status).toEqual(DATA_STATUS);
    expect(result.current.catalog).toBe(CATALOG);
    expect(result.current.sources).toEqual([SOURCE]);
  });

  it('does not load lists for visitors or when the server cannot prepare data', async () => {
    const visitor = renderHook(() => useDataResources(false));
    await waitFor(() => expect(visitor.result.current.catalog).toBe(CATALOG));
    expect(api.data.sources.list).not.toHaveBeenCalled();
    expect(visitor.result.current.preps).toBeNull();

    vi.mocked(api.data.status).mockResolvedValue({
      ...DATA_STATUS,
      available: false,
      reason: 'pandas is not installed on the server.',
    });
    vi.mocked(api.data.steps).mockClear();
    const unavailable = renderHook(() => useDataResources(true));
    await waitFor(() => expect(unavailable.result.current.status?.available).toBe(false));
    expect(api.data.steps).not.toHaveBeenCalled();
    expect(api.data.preps.list).not.toHaveBeenCalled();
  });

  it('loads the lists again when they changed while loading', async () => {
    let finish!: (sources: DataSource[]) => void;
    vi.mocked(api.data.sources.list)
      .mockReturnValueOnce(new Promise((done) => (finish = done)))
      .mockResolvedValueOnce([OTHER_SOURCE, SOURCE]);
    const { result } = renderHook(() => useDataResources(true));
    await waitFor(() => expect(api.data.sources.list).toHaveBeenCalledTimes(1));

    // An import finishes before the first list arrives: the list would not contain it.
    act(() => result.current.putSource(OTHER_SOURCE));
    await act(async () => finish([SOURCE]));
    await waitFor(() => expect(api.data.sources.list).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(result.current.sources).toEqual([OTHER_SOURCE, SOURCE]));
  });

  it('reports errors and loads again on retry', async () => {
    vi.mocked(api.data.preps.list).mockRejectedValueOnce(
      new ApiError(0, null, 'Cannot reach the Brick DNN server. Is the backend running?')
    );
    const { result } = renderHook(() => useDataResources(true));
    await waitFor(() => expect(result.current.listsError).toMatch(/Cannot reach/));
    act(() => result.current.retry());
    await waitFor(() => expect(result.current.preps).toEqual([PREP]));
    expect(result.current.listsError).toBeNull();
  });

  it('keeps lists up to date after changes', async () => {
    const { result } = renderHook(() => useDataResources(true));
    await waitFor(() => expect(result.current.preps).toEqual([PREP]));
    act(() => result.current.putPrep({ ...PREP, name: 'Renamed' }));
    expect(result.current.preps?.map((p) => p.name)).toEqual(['Renamed']);
    act(() => result.current.putPrep({ ...PREP, id: 99, name: 'New' }));
    expect(result.current.preps?.map((p) => p.id)).toEqual([99, PREP.id]);
    act(() => result.current.dropPrep(PREP.id));
    act(() => result.current.dropSource(SOURCE.id));
    expect(result.current.preps?.map((p) => p.id)).toEqual([99]);
    expect(result.current.sources).toEqual([]);
  });
});
