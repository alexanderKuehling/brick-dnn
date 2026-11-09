import { act, renderHook } from '@testing-library/react';
import { afterEach, beforeEach, vi } from 'vitest';
import { useUploadProgress } from './useUploadProgress';

describe('useUploadProgress', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('shows an upload only when it takes a moment', () => {
    const { result } = renderHook(() => useUploadProgress());
    act(() => result.current.start(1000));
    act(() => result.current.report(400, 1000));
    act(() => vi.advanceTimersByTime(299));
    expect(result.current.bytes).toBeNull();
    act(() => vi.advanceTimersByTime(1));
    expect(result.current.bytes).toEqual({ sent: 400, total: 1000 });
  });

  it('never shows an upload that is done at once', () => {
    const { result } = renderHook(() => useUploadProgress());
    act(() => result.current.start(1000));
    act(() => result.current.report(1000, 1000));
    act(() => result.current.stop());
    act(() => vi.advanceTimersByTime(1000));
    expect(result.current.bytes).toBeNull();
  });

  it('passes on progress at most every 200 ms, and the end at once', () => {
    const { result } = renderHook(() => useUploadProgress());
    act(() => result.current.start(1000));
    act(() => vi.advanceTimersByTime(300));
    expect(result.current.bytes).toEqual({ sent: 0, total: 1000 });

    act(() => result.current.report(100, 1000));
    act(() => result.current.report(200, 1000));
    expect(result.current.bytes).toEqual({ sent: 0, total: 1000 });
    // The numbers held back are shown when the interval is over.
    act(() => vi.advanceTimersByTime(200));
    expect(result.current.bytes).toEqual({ sent: 200, total: 1000 });

    act(() => result.current.report(1000, 1000));
    expect(result.current.bytes).toEqual({ sent: 1000, total: 1000 });
  });

  it('ignores progress after the upload ended', () => {
    const { result } = renderHook(() => useUploadProgress());
    act(() => result.current.start(1000));
    act(() => vi.advanceTimersByTime(300));
    act(() => result.current.stop());
    act(() => result.current.report(500, 1000));
    act(() => vi.advanceTimersByTime(1000));
    expect(result.current.bytes).toBeNull();
  });
});
