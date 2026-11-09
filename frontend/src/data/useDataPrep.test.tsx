import { act, renderHook } from '@testing-library/react';
import { afterEach, beforeEach, vi } from 'vitest';
import { notifications } from '@mantine/notifications';
import { api, ApiError, type DataPrep, type PreviewResult } from '@/api';
import { CATALOG, PREP, PREVIEW, STEP_REPORTS } from './fixtures';
import { AUTOSAVE_DELAY, PREVIEW_DELAY, useDataPrep } from './useDataPrep';

vi.mock('@/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api')>();
  return {
    ...actual,
    api: {
      data: {
        preps: {
          get: vi.fn(),
          update: vi.fn(),
          preview: vi.fn(),
        },
      },
    },
  };
});

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((done, fail) => {
    resolve = done;
    reject = fail;
  });
  return { promise, resolve, reject };
}

const OTHER_PREP: DataPrep = {
  ...PREP,
  id: 11,
  name: 'Other',
  steps: [{ id: 's1', type: 'drop_columns', enabled: true, params: { columns: ['notes'] } }],
};

const get = () => vi.mocked(api.data.preps.get);
const update = () => vi.mocked(api.data.preps.update);
const preview = () => vi.mocked(api.data.preps.preview);

/** Let pending promises and timers up to `ms` run. */
const advance = (ms = 0) => act(() => vi.advanceTimersByTimeAsync(ms));

async function mountLoaded(prepId = 10) {
  const hook = renderHook(({ id }) => useDataPrep(id, { catalog: CATALOG }), {
    initialProps: { id: prepId },
  });
  await advance();
  return hook;
}

beforeEach(() => {
  vi.useFakeTimers();
  vi.clearAllMocks();
  vi.spyOn(notifications, 'show').mockImplementation(() => 'id');
  get().mockImplementation(async (id: number) => (id === 11 ? OTHER_PREP : PREP));
  update().mockImplementation(async (id: number, data) => ({
    ...(id === 11 ? OTHER_PREP : PREP),
    ...data,
  }));
  preview().mockResolvedValue(PREVIEW);
});

afterEach(() => {
  vi.useRealTimers();
});

describe('useDataPrep', () => {
  it('loads the preparation and previews it right away', async () => {
    const { result } = await mountLoaded();
    expect(get()).toHaveBeenCalledWith(10, expect.any(AbortSignal));
    expect(result.current.name).toBe('Churn model data');
    expect(result.current.steps).toEqual(PREP.steps);
    expect(preview()).toHaveBeenCalledWith(
      10,
      { steps: PREP.steps, settings: PREP.settings, upto: null },
      expect.any(AbortSignal)
    );
    expect(result.current.preview).toBe(PREVIEW);
    expect(result.current.previewing).toBe(false);
    expect(result.current.reports.s2.status).toBe('error');
    expect(result.current.saveStatus).toBe('saved');
    await advance(AUTOSAVE_DELAY * 2);
    expect(update()).not.toHaveBeenCalled();
  });

  it('debounces previews, ignores outdated answers and keeps the last result meanwhile', async () => {
    const { result } = await mountLoaded();
    const first = deferred<PreviewResult>();
    const second = deferred<PreviewResult>();
    preview().mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise);

    act(() => result.current.setSteps(PREP.steps.slice(1)));
    await advance(PREVIEW_DELAY - 1);
    expect(preview()).toHaveBeenCalledTimes(1);
    expect(result.current.previewing).toBe(true);
    await advance(1);
    expect(preview()).toHaveBeenCalledTimes(2);
    const firstSignal = preview().mock.calls[1][2]!;

    act(() => result.current.setSteps(PREP.steps.slice(2)));
    expect(firstSignal.aborted).toBe(true);
    await advance(PREVIEW_DELAY);
    expect(preview()).toHaveBeenLastCalledWith(
      10,
      expect.objectContaining({ steps: PREP.steps.slice(2) }),
      expect.any(AbortSignal)
    );

    // The answer to the aborted request arrives late and is ignored.
    await act(async () => first.resolve({ ...PREVIEW, duration_ms: 1 }));
    expect(result.current.preview).toBe(PREVIEW);
    expect(result.current.previewing).toBe(true);

    const latest = { ...PREVIEW, duration_ms: 2 };
    await act(async () => second.resolve(latest));
    expect(result.current.preview).toBe(latest);
    expect(result.current.previewing).toBe(false);
  });

  it('shows preview errors next to the last result', async () => {
    const { result } = await mountLoaded();
    preview().mockRejectedValueOnce(new ApiError(400, null, 'Validation share: at most 0.5.'));
    act(() =>
      result.current.setSettings({
        ...PREP.settings,
        split: { ...PREP.settings.split, val: 0.8 },
      })
    );
    await advance(PREVIEW_DELAY);
    expect(result.current.previewError).toBe('Validation share: at most 0.5.');
    expect(result.current.preview).toBe(PREVIEW);

    act(() => result.current.retryPreview());
    await advance(PREVIEW_DELAY);
    expect(result.current.previewError).toBeNull();
  });

  it('saves changes after a pause, once', async () => {
    const { result } = await mountLoaded();
    act(() => result.current.setName('Churn v2'));
    expect(result.current.saveStatus).toBe('unsaved');
    await advance(AUTOSAVE_DELAY - 1);
    expect(update()).not.toHaveBeenCalled();
    // Renaming does not run the preview again.
    expect(preview()).toHaveBeenCalledTimes(1);
    await advance(1);
    expect(update()).toHaveBeenCalledWith(10, {
      name: 'Churn v2',
      steps: PREP.steps,
      settings: PREP.settings,
    });
    expect(result.current.saveStatus).toBe('saved');
    expect(result.current.prep?.name).toBe('Churn v2');
    await advance(AUTOSAVE_DELAY * 2);
    expect(update()).toHaveBeenCalledTimes(1);
  });

  it('saves again after a save when something changed meanwhile', async () => {
    const { result } = await mountLoaded();
    const slow = deferred<DataPrep>();
    update().mockReturnValueOnce(slow.promise);
    act(() => result.current.setName('A'));
    await advance(AUTOSAVE_DELAY);
    expect(result.current.saveStatus).toBe('saving');

    act(() => result.current.setName('B'));
    await advance(AUTOSAVE_DELAY);
    // The second save waits for the first one.
    expect(update()).toHaveBeenCalledTimes(1);
    await act(async () => slow.resolve({ ...PREP, name: 'A' }));
    await advance();
    expect(update()).toHaveBeenCalledTimes(2);
    expect(update()).toHaveBeenLastCalledWith(10, expect.objectContaining({ name: 'B' }));
    expect(result.current.saveStatus).toBe('saved');
    expect(result.current.name).toBe('B');
  });

  it('saves again when an edit is undone while it is being saved', async () => {
    const { result } = await mountLoaded();
    const slow = deferred<DataPrep>();
    update().mockReturnValueOnce(slow.promise);
    act(() => result.current.setName('Churn model dataX'));
    await advance(AUTOSAVE_DELAY);
    expect(result.current.saveStatus).toBe('saving');

    // Backspace: back to what the server had before the running save.
    act(() => result.current.setName('Churn model data'));
    await act(async () => slow.resolve({ ...PREP, name: 'Churn model dataX' }));
    expect(result.current.saveStatus).toBe('unsaved');
    await advance(AUTOSAVE_DELAY);
    expect(update()).toHaveBeenCalledTimes(2);
    expect(update()).toHaveBeenLastCalledWith(
      10,
      expect.objectContaining({ name: 'Churn model data' })
    );
    expect(result.current.saveStatus).toBe('saved');
    expect(result.current.prep?.name).toBe('Churn model data');
  });

  it('saves an edit undone during a save also when the page is left meanwhile', async () => {
    const { result, unmount } = await mountLoaded();
    const slow = deferred<DataPrep>();
    update().mockReturnValueOnce(slow.promise);
    act(() => result.current.setName('Churn model dataX'));
    await advance(AUTOSAVE_DELAY);
    act(() => result.current.setName('Churn model data'));
    unmount();
    await act(async () => slow.resolve({ ...PREP, name: 'Churn model dataX' }));
    await advance();
    expect(update()).toHaveBeenCalledTimes(2);
    expect(update()).toHaveBeenLastCalledWith(
      10,
      expect.objectContaining({ name: 'Churn model data' })
    );
  });

  it('saves at once on flush and reports failures', async () => {
    const { result } = await mountLoaded();
    update().mockRejectedValueOnce(new ApiError(400, null, 'Step 2: unknown column.'));
    act(() => result.current.setSteps(PREP.steps.slice(0, 1)));

    let failure: unknown = null;
    await act(() => result.current.flush().catch((error: unknown) => (failure = error)));
    expect(update()).toHaveBeenCalledTimes(1);
    expect((failure as Error).message).toBe('Step 2: unknown column.');
    expect(result.current.saveStatus).toBe('error');
    expect(result.current.saveError).toBe('Step 2: unknown column.');

    act(() => result.current.retrySave());
    await advance();
    expect(update()).toHaveBeenCalledTimes(2);
    expect(result.current.saveStatus).toBe('saved');
    expect(result.current.saveError).toBeNull();
    // Nothing left to save.
    await act(() => result.current.flush());
    expect(update()).toHaveBeenCalledTimes(2);
  });

  it('saves pending changes when the page is left', async () => {
    const { result, unmount } = await mountLoaded();
    act(() => result.current.setName('Left behind'));
    unmount();
    await advance();
    expect(update()).toHaveBeenCalledWith(10, expect.objectContaining({ name: 'Left behind' }));
  });

  it('opens another preparation without losing the changes of the first one', async () => {
    const { result, rerender } = await mountLoaded();
    const slow = deferred<DataPrep>();
    update().mockReturnValueOnce(slow.promise);
    act(() => result.current.setName('Changed'));

    rerender({ id: 11 });
    // Nothing of the first preparation is shown for the second one.
    expect(result.current.prep).toBeNull();
    expect(result.current.steps).toEqual([]);
    expect(result.current.preview).toBeNull();
    await advance();
    expect(update()).toHaveBeenCalledWith(10, expect.objectContaining({ name: 'Changed' }));

    // The second one is loaded once the save of the first has finished.
    expect(get()).not.toHaveBeenCalledWith(11, expect.anything());
    await act(async () => slow.resolve({ ...PREP, name: 'Changed' }));
    await advance();
    expect(get()).toHaveBeenCalledWith(11, expect.any(AbortSignal));
    expect(result.current.name).toBe('Other');
    expect(result.current.saveStatus).toBe('saved');
    expect(preview()).toHaveBeenLastCalledWith(
      11,
      expect.objectContaining({ steps: OTHER_PREP.steps }),
      expect.any(AbortSignal)
    );
  });

  it('previews up to a step and with a comparison', async () => {
    const { result } = await mountLoaded();
    act(() => result.current.setUpto('s2'));
    await advance(PREVIEW_DELAY);
    expect(preview()).toHaveBeenLastCalledWith(
      10,
      expect.objectContaining({ upto: 's2' }),
      expect.any(AbortSignal)
    );

    act(() => result.current.requestCompare());
    await advance(PREVIEW_DELAY);
    expect(preview()).toHaveBeenLastCalledWith(
      10,
      expect.objectContaining({ upto: 's2', compare: true }),
      expect.any(AbortSignal)
    );

    // Removing the previewed step shows the end of the pipeline again.
    act(() => result.current.setSteps(PREP.steps.filter((step) => step.id !== 's2')));
    expect(result.current.upto).toBeNull();
  });

  it('adds steps: combine steps first, others after the previewed step', async () => {
    const { result } = await mountLoaded();
    act(() => {
      result.current.addStep({ type: 'join', params: { source: 2 } });
    });
    expect(result.current.steps[0]).toMatchObject({
      type: 'join',
      params: { source: 2, how: 'left', suffix: '_right' },
    });

    act(() => result.current.setUpto('s1'));
    let added: (string | null)[] = [];
    act(() => {
      // Several fixes in one event build on each other.
      added = [
        result.current.addStep({ type: 'drop_columns', params: { columns: ['notes'] } }),
        result.current.addStep({ type: 'drop_columns', params: { columns: ['customer_id'] } }),
      ];
    });
    const ids = result.current.steps.map((step) => step.id);
    expect(ids).toEqual(['s4', 's1', added[0], added[1], 's2', 's3']);
    expect(result.current.upto).toBe(added[1]);
  });

  it('keeps the reports of the steps after a previewed step from the last full preview', async () => {
    const { result } = await mountLoaded();
    const notRun = (report: (typeof STEP_REPORTS)[number]) => ({
      ...report,
      status: 'not_run' as const,
      error: null,
      columns: STEP_REPORTS[0].columns.slice(0, 2),
    });
    const upto: PreviewResult = {
      ...PREVIEW,
      steps: [STEP_REPORTS[0], notRun(STEP_REPORTS[1]), notRun(STEP_REPORTS[2])],
    };
    preview().mockResolvedValueOnce(upto);
    act(() => result.current.setUpto('s1'));
    await advance(PREVIEW_DELAY);
    expect(result.current.preview).toBe(upto);
    expect(result.current.reports.s1).toBe(STEP_REPORTS[0]);
    // Not run in this preview: the reports (and columns) of the last preview of all steps.
    expect(result.current.reports.s2).toBe(STEP_REPORTS[1]);
    expect(result.current.reports.s3).toBe(STEP_REPORTS[2]);

    const full: PreviewResult = { ...PREVIEW, steps: STEP_REPORTS.map((r) => ({ ...r })) };
    preview().mockResolvedValueOnce(full);
    act(() => result.current.setUpto(null));
    await advance(PREVIEW_DELAY);
    expect(result.current.reports.s2).toBe(full.steps[1]);
  });

  it('puts quick fixes in front of a failed step and adds them once', async () => {
    const { result } = await mountLoaded();
    // Step 2 failed in the preview; the preview shows the data after step 3.
    act(() => result.current.setUpto('s3'));
    const fix = { type: 'drop_columns', params: { columns: ['notes'] } };
    let first: string | null = null;
    let second: string | null = 'not called';
    act(() => {
      first = result.current.addStep(fix);
    });
    act(() => {
      second = result.current.addStep(structuredClone(fix));
    });
    expect(result.current.steps.map((step) => step.id)).toEqual(['s1', first, 's2', 's3']);
    expect(second).toBeNull();
    // It runs before the previewed step, so the preview stays there.
    expect(result.current.upto).toBe('s3');

    // Once a preview ran it, the same step can be added again on purpose.
    await advance(PREVIEW_DELAY);
    act(() => {
      second = result.current.addStep(fix);
    });
    expect(second).not.toBeNull();
    expect(result.current.steps).toHaveLength(5);
  });

  it('never gives a new step the id of a deleted one', async () => {
    const { result } = await mountLoaded();
    act(() => result.current.setSteps(PREP.steps.slice(0, 2)));
    // The next preview has no report of the deleted step s3 any more.
    preview().mockResolvedValueOnce({ ...PREVIEW, steps: STEP_REPORTS.slice(0, 2) });
    await advance(PREVIEW_DELAY);
    expect(result.current.reports.s3).toBeUndefined();
    let id: string | null = null;
    act(() => {
      id = result.current.addStep({ type: 'drop_columns', params: { columns: ['notes'] } });
    });
    expect(id).toBe('s4');
  });

  it('says when the preparation does not exist', async () => {
    get().mockRejectedValue(new ApiError(404, null, 'Not found.'));
    const { result } = await mountLoaded();
    expect(result.current.notFound).toBe(true);
    expect(result.current.loadError).toBe('This preparation does not exist (any more).');
    expect(preview()).not.toHaveBeenCalled();
  });

  it('forgets pending changes when discarded', async () => {
    const { result, unmount } = await mountLoaded();
    act(() => result.current.setName('Doomed'));
    act(() => result.current.discard());
    expect(result.current.saveStatus).toBe('saved');
    unmount();
    await advance(AUTOSAVE_DELAY * 2);
    expect(update()).not.toHaveBeenCalled();
  });

  it('changes and saves nothing of a preparation shared to look at', async () => {
    get().mockResolvedValue({ ...PREP, owner: 'bob', role: 'viewer' });
    const { result } = await mountLoaded();
    expect(result.current.readOnly).toBe(true);
    act(() => result.current.setSteps([]));
    act(() => result.current.setName('Mine now'));
    let added: string | null = 'x';
    act(() => {
      added = result.current.addStep({ type: 'drop_columns', params: { columns: ['notes'] } });
    });
    expect(added).toBeNull();
    expect(result.current.steps).toEqual(PREP.steps);
    expect(result.current.name).toBe('Churn model data');
    await act(() => result.current.flush());
    await advance(AUTOSAVE_DELAY * 2);
    expect(update()).not.toHaveBeenCalled();
    expect(result.current.saveStatus).toBe('saved');
    // The preview still works, e.g. up to a step.
    act(() => result.current.setUpto('s1'));
    await advance(PREVIEW_DELAY);
    expect(preview()).toHaveBeenLastCalledWith(
      10,
      expect.objectContaining({ upto: 's1' }),
      expect.any(AbortSignal)
    );
  });

  it('saves the changes of an editor of a shared preparation', async () => {
    get().mockResolvedValue({ ...PREP, owner: 'bob', role: 'editor' });
    const { result } = await mountLoaded();
    expect(result.current.readOnly).toBe(false);
    act(() => result.current.setSteps([]));
    await advance(AUTOSAVE_DELAY);
    expect(update()).toHaveBeenCalledWith(10, expect.objectContaining({ steps: [] }));
  });
});
