import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { notifications } from '@mantine/notifications';
import {
  api,
  ApiError,
  type DataPrep,
  type NewStep,
  type PipelineStep,
  type PrepSettings,
  type PreviewRequest,
  type PreviewResult,
  type StepCatalog,
  type StepReport,
} from '@/api';
import { prepRole } from './prepAccess';
import { insertStep, mergeReports } from './steps';

/** Wait this long after the last change before running the preview again (ms). */
export const PREVIEW_DELAY = 400;
/** Wait this long after the last change before saving it (ms). */
export const AUTOSAVE_DELAY = 800;

export const UNTITLED = 'Untitled preparation';

/**
 * saved: the server has the shown state · unsaved: a save is scheduled · saving: a save is
 * running · error: the last save failed (`saveError` says why).
 */
export type SaveStatus = 'saved' | 'unsaved' | 'saving' | 'error';

/** The editable part of a preparation. */
interface Draft {
  name: string;
  steps: PipelineStep[];
  settings: PrepSettings;
}

const draftKey = (draft: Draft) => JSON.stringify([draft.name, draft.steps, draft.settings]);

/**
 * Saves of all hook instances, one after the other. A preparation that is opened again (also
 * after leaving the page) is only loaded once the saves started before have finished, so it
 * never comes back without its latest changes.
 */
let saveQueue: Promise<unknown> = Promise.resolve();

export interface DataPrepController {
  /** The preparation as last loaded or saved; null while loading. */
  prep: DataPrep | null;
  loadError: string | null;
  /** The preparation does not exist (any more). */
  notFound: boolean;
  reload: () => void;
  /**
   * Shared with the user to look at (viewer): the setters do nothing and nothing is saved,
   * previews and downloads still work.
   */
  readOnly: boolean;

  /** Local state, saved automatically. Empty / null until loaded. */
  name: string;
  steps: PipelineStep[];
  settings: PrepSettings | null;
  setName: (name: string) => void;
  setSteps: (steps: PipelineStep[]) => void;
  setSettings: (settings: PrepSettings) => void;
  /**
   * Add a step (quick fixes, column actions). While the preview shows the data after a step,
   * the new step goes right after that one and the preview moves on to it. It goes in front of
   * a step that failed, where it can run. Returns its id, or null if nothing was added (also
   * when the same step was just added and no preview has run it yet: a double click).
   */
  addStep: (step: NewStep) => string | null;

  /** Step whose output is previewed (null: the end of the pipeline). */
  upto: string | null;
  setUpto: (id: string | null) => void;
  /** Whether previews also profile the data before the cleaning steps. */
  compare: boolean;
  requestCompare: () => void;

  /** The last successful preview; kept while a new one is loading. */
  preview: PreviewResult | null;
  /** A preview is scheduled or running. */
  previewing: boolean;
  previewError: string | null;
  retryPreview: () => void;
  /**
   * Reports of the last preview by step id. While the preview stops at a step (`upto`), the
   * steps after it keep their reports of the last preview of all steps (for their columns).
   */
  reports: Record<string, StepReport>;

  saveStatus: SaveStatus;
  saveError: string | null;
  /** Save pending changes now; resolves when the server has them, rejects if saving fails. */
  flush: () => Promise<void>;
  retrySave: () => void;
  /** Forget pending changes (before deleting the preparation). */
  discard: () => void;
  /** Record a training dataset exported from the preparation. */
  addDataset: (dataset: { id: number; name: string; created_at: string }) => void;
}

interface Options {
  /** Needed by `addStep` (combine steps go first, defaults of the step type). */
  catalog?: StepCatalog | null;
  /** Called with the server's version after every save, e.g. to update a list. */
  onSaved?: (prep: DataPrep) => void;
}

interface Pending {
  id: number;
  draft: Draft;
}

const SAVED = { status: 'saved', error: null } as const;

/**
 * State of the open preparation: loads it, keeps the edits locally, previews them (debounced,
 * older requests are aborted and their late answers ignored) and saves them automatically.
 * Pending changes are saved when another preparation is opened, when the page is left and
 * before export (`flush`).
 */
export function useDataPrep(prepId: number | null, options: Options = {}): DataPrepController {
  const { catalog = null } = options;
  const [shownId, setShownId] = useState(prepId);
  const [prep, setPrep] = useState<DataPrep | null>(null);
  const [draft, setDraft] = useState<Draft | null>(null);
  const [loadError, setLoadError] = useState<{ message: string; notFound: boolean } | null>(null);
  const [loadRequest, setLoadRequest] = useState(0);
  const [upto, setUptoState] = useState<string | null>(null);
  const [compare, setCompare] = useState(false);
  const [preview, setPreview] = useState<PreviewResult | null>(null);
  /** `upto` of the request that `preview` answers. */
  const [previewUpto, setPreviewUpto] = useState<string | null>(null);
  /** Step reports of the last preview of all steps (without `upto`). */
  const [fullReports, setFullReports] = useState<StepReport[] | null>(null);
  const [previewing, setPreviewing] = useState(false);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [previewRequest, setPreviewRequest] = useState(0);
  const [save, setSave] = useState<{ status: SaveStatus; error: string | null }>(SAVED);

  // Another preparation: start from scratch in this render. Pending changes of the previous
  // one are saved by the cleanup of the effect below.
  if (shownId !== prepId) {
    setShownId(prepId);
    setPrep(null);
    setDraft(null);
    setLoadError(null);
    setUptoState(null);
    setCompare(false);
    setPreview(null);
    setPreviewUpto(null);
    setFullReports(null);
    setPreviewing(false);
    setPreviewError(null);
    setSave(SAVED);
  }

  // The preparation on screen, for async work that may finish after another one was opened.
  const active = useRef(prepId);
  active.current = prepId;
  const onSaved = useRef(options.onSaved);
  onSaved.current = options.onSaved;
  /** Key of the state the server has, per preparation. */
  const savedKeys = useRef(new Map<number, string>());
  /** The latest local state (set after every change is committed). */
  const latest = useRef<Pending | null>(null);
  const saveTimer = useRef(0);
  const discarded = useRef(new Set<number>());
  const previewed = useRef<number | null>(null);
  const mounted = useRef(true);
  /** Preparations the user may only look at: they are never saved. */
  const viewOnly = useRef(new Set<number>());
  /** Every step id of the open preparation so far: new steps never reuse one. */
  const usedIds = useRef(new Set<string>());
  /** Steps added by `addStep` that no preview has run yet: step key → id. */
  const unpreviewed = useRef(new Map<string, string>());

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  // ---------------------------------------------------------------- loading
  useEffect(() => {
    if (prepId === null) {
      return undefined;
    }
    const controller = new AbortController();
    setLoadError(null);
    usedIds.current = new Set();
    unpreviewed.current = new Map();
    const load = () => api.data.preps.get(prepId, controller.signal);
    saveQueue
      .then(load, load)
      .then((loaded) => {
        if (controller.signal.aborted) {
          return;
        }
        const initial = { name: loaded.name, steps: loaded.steps, settings: loaded.settings };
        loaded.steps.forEach((step) => usedIds.current.add(step.id));
        savedKeys.current.set(prepId, draftKey(initial));
        if (prepRole(loaded) === 'viewer') {
          viewOnly.current.add(prepId);
        } else {
          viewOnly.current.delete(prepId);
        }
        discarded.current.delete(prepId);
        setPrep(loaded);
        setDraft(initial);
      })
      .catch((error: Error) => {
        if (controller.signal.aborted) {
          return;
        }
        const notFound = error instanceof ApiError && error.status === 404;
        setLoadError({
          message: notFound ? 'This preparation does not exist (any more).' : error.message,
          notFound,
        });
      });
    return () => controller.abort();
  }, [prepId, loadRequest]);

  // ---------------------------------------------------------------- saving
  /**
   * Save the latest state of preparation `id` (or `fallback` if another one is open by now),
   * after all saves that started before. Does nothing if the server has it already.
   */
  const persistRef = useRef<(id: number, fallback: Pending) => Promise<DataPrep | null>>(null);

  /** Save `pending` once more: after a pause while it is on screen, else right away. */
  const saveAgain = useCallback((id: number, pending: Pending) => {
    const save = () => persistRef.current?.(id, pending) ?? Promise.resolve(null);
    if (mounted.current && active.current === id) {
      window.clearTimeout(saveTimer.current);
      saveTimer.current = window.setTimeout(() => {
        save().catch(() => undefined);
      }, AUTOSAVE_DELAY);
    } else {
      save().catch((error: Error) => notifyNotSaved(pending, error));
    }
  }, []);

  const persist = useCallback(
    (id: number, fallback: Pending): Promise<DataPrep | null> => {
      const task = async () => {
        const current = latest.current?.id === id ? latest.current : fallback;
        const key = draftKey(current.draft);
        if (
          discarded.current.has(id) ||
          viewOnly.current.has(id) ||
          savedKeys.current.get(id) === key
        ) {
          return null;
        }
        if (active.current === id) {
          setSave({ status: 'saving', error: null });
        }
        try {
          const saved = await api.data.preps.update(id, {
            name: current.draft.name.trim() || UNTITLED,
            steps: current.draft.steps,
            settings: current.draft.settings,
          });
          savedKeys.current.set(id, key);
          // Changed while saving, maybe back to what the server had before: the autosave does
          // not run for that (the server had it when it was typed), so save again here.
          const now = latest.current?.id === id ? latest.current : null;
          const changed =
            now !== null && draftKey(now.draft) !== key && !discarded.current.has(id) ? now : null;
          if (active.current === id) {
            setPrep(saved);
            setSave(changed ? { status: 'unsaved', error: null } : SAVED);
          }
          onSaved.current?.(saved);
          if (changed) {
            saveAgain(id, changed);
          }
          return saved;
        } catch (error) {
          if (active.current === id) {
            setSave({ status: 'error', error: (error as Error).message });
          }
          throw error;
        }
      };
      const run = saveQueue.then(task, task);
      saveQueue = run.catch(() => undefined);
      return run;
    },
    [saveAgain]
  );
  persistRef.current = persist;

  const key = draft ? draftKey(draft) : null;
  useEffect(() => {
    if (prepId === null || draft === null || key === null) {
      return undefined;
    }
    latest.current = { id: prepId, draft };
    if (savedKeys.current.get(prepId) === key) {
      // Changed back to what the server has.
      setSave((current) => (current.status === 'saving' ? current : SAVED));
      return undefined;
    }
    setSave((current) =>
      current.status === 'saving' ? current : { status: 'unsaved', error: current.error }
    );
    const pending = latest.current;
    saveTimer.current = window.setTimeout(() => {
      persist(prepId, pending).catch(() => undefined);
    }, AUTOSAVE_DELAY);
    return () => window.clearTimeout(saveTimer.current);
    // `key` stands for `draft`.
  }, [prepId, key, persist]);

  // Save pending changes when another preparation is opened or the page is left.
  useEffect(() => {
    if (prepId === null) {
      return undefined;
    }
    return () => {
      window.clearTimeout(saveTimer.current);
      const pending = latest.current;
      if (
        pending &&
        pending.id === prepId &&
        !discarded.current.has(prepId) &&
        savedKeys.current.get(prepId) !== draftKey(pending.draft)
      ) {
        persist(prepId, pending).catch((error: Error) => notifyNotSaved(pending, error));
      }
    };
  }, [prepId, persist]);

  // Closing the tab: try to save, and let the browser ask while changes are not saved.
  const unsaved = save.status !== 'saved';
  useEffect(() => {
    if (!unsaved) {
      return undefined;
    }
    const onBeforeUnload = (event: BeforeUnloadEvent) => {
      const pending = latest.current;
      if (pending && !discarded.current.has(pending.id)) {
        window.clearTimeout(saveTimer.current);
        persist(pending.id, pending).catch(() => undefined);
        event.preventDefault();
      }
    };
    window.addEventListener('beforeunload', onBeforeUnload);
    return () => window.removeEventListener('beforeunload', onBeforeUnload);
  }, [unsaved, persist]);

  const flush = useCallback(async () => {
    const pending = latest.current;
    if (prepId === null || !pending || pending.id !== prepId) {
      return;
    }
    window.clearTimeout(saveTimer.current);
    await persist(prepId, pending);
  }, [prepId, persist]);

  const retrySave = useCallback(() => {
    flush().catch(() => undefined);
  }, [flush]);

  const discard = useCallback(() => {
    if (prepId === null) {
      return;
    }
    discarded.current.add(prepId);
    window.clearTimeout(saveTimer.current);
    setSave(SAVED);
  }, [prepId]);

  // ---------------------------------------------------------------- preview
  const previewKey = draft ? JSON.stringify([draft.steps, draft.settings, upto, compare]) : null;
  useEffect(() => {
    if (prepId === null || draft === null || previewKey === null) {
      return undefined;
    }
    // The first preview of a preparation starts right away, later ones wait for a pause.
    const immediate = previewed.current !== prepId;
    previewed.current = prepId;
    const body: PreviewRequest = {
      steps: draft.steps,
      settings: draft.settings,
      upto,
      ...(compare ? { compare: true } : {}),
    };
    const controller = new AbortController();
    setPreviewing(true);
    const start = () => {
      api.data.preps
        .preview(prepId, body, controller.signal)
        .then((result) => {
          if (controller.signal.aborted) {
            return;
          }
          setPreview(result);
          setPreviewUpto(body.upto ?? null);
          if (!body.upto) {
            setFullReports(result.steps);
          }
          const ran = new Set(body.steps?.map((step) => step.id));
          for (const [stepKey, id] of unpreviewed.current) {
            if (ran.has(id)) {
              unpreviewed.current.delete(stepKey);
            }
          }
          setPreviewError(null);
          setPreviewing(false);
        })
        .catch((error: Error) => {
          if (controller.signal.aborted) {
            return;
          }
          setPreviewError(
            error instanceof ApiError && error.status === 404
              ? 'This preparation does not exist (any more).'
              : error.message
          );
          setPreviewing(false);
        });
    };
    let timer = 0;
    if (immediate) {
      start();
    } else {
      timer = window.setTimeout(start, PREVIEW_DELAY);
    }
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
    // `previewKey` stands for the steps, settings, upto and compare.
  }, [prepId, previewKey, previewRequest]);

  // ---------------------------------------------------------------- editing
  // Changes are applied to these first, so several changes in one event (e.g. "Fix all"
  // adding one step per problem) build on each other.
  const draftNow = useRef(draft);
  draftNow.current = draft;
  const uptoNow = useRef(upto);
  uptoNow.current = upto;

  const change = useCallback((update: (current: Draft) => Draft) => {
    if (draftNow.current && !viewOnly.current.has(active.current ?? -1)) {
      draftNow.current = update(draftNow.current);
      draftNow.current.steps.forEach((step) => usedIds.current.add(step.id));
      setDraft(draftNow.current);
    }
  }, []);
  const changeUpto = useCallback((id: string | null) => {
    uptoNow.current = id;
    setUptoState(id);
  }, []);

  const setName = useCallback((name: string) => change((d) => ({ ...d, name })), [change]);
  const setSettings = useCallback(
    (settings: PrepSettings) => change((d) => ({ ...d, settings })),
    [change]
  );
  const setSteps = useCallback(
    (steps: PipelineStep[]) => {
      change((d) => ({ ...d, steps }));
      // A removed step cannot be previewed any more.
      const shown = uptoNow.current;
      if (shown !== null && !steps.some((step) => step.id === shown)) {
        changeUpto(null);
      }
    },
    [change, changeUpto]
  );

  const reports = useMemo(
    () => mergeReports(preview?.steps, previewUpto, fullReports),
    [preview, previewUpto, fullReports]
  );
  const reportsNow = useRef(reports);
  reportsNow.current = reports;

  const addStep = useCallback(
    (step: NewStep): string | null => {
      const current = draftNow.current;
      if (!catalog || !current || viewOnly.current.has(active.current ?? -1)) {
        return null;
      }
      const after = uptoNow.current;
      const next = insertStep(catalog, current.steps, step, after, {
        reports: reportsNow.current,
        reserved: [...usedIds.current, ...Object.keys(reportsNow.current)],
      });
      const known = new Set(current.steps.map((s) => s.id));
      const index = next.findIndex((s) => !known.has(s.id));
      if (index < 0) {
        return null;
      }
      const created = next[index];
      // The same step again before a preview ran the first one: a double click, or the same
      // fix offered in two places. Ignore it.
      const stepKey = JSON.stringify([created.type, created.params]);
      const twin = unpreviewed.current.get(stepKey);
      if (twin !== undefined && known.has(twin)) {
        return null;
      }
      unpreviewed.current.set(stepKey, created.id);
      change((d) => ({ ...d, steps: next }));
      // Added behind the previewed step: show the data after the new step.
      if (after !== null && index > next.findIndex((s) => s.id === after)) {
        changeUpto(created.id);
      }
      return created.id;
    },
    [catalog, change, changeUpto]
  );

  const setUpto = useCallback((id: string | null) => changeUpto(id), [changeUpto]);
  const requestCompare = useCallback(() => setCompare(true), []);
  const retryPreview = useCallback(() => setPreviewRequest((n) => n + 1), []);
  const reload = useCallback(() => setLoadRequest((n) => n + 1), []);
  const addDataset = useCallback(
    (dataset: { id: number; name: string; created_at: string }) =>
      setPrep(
        (current) =>
          current && {
            ...current,
            datasets: [
              { id: dataset.id, name: dataset.name, created_at: dataset.created_at },
              ...current.datasets.filter((d) => d.id !== dataset.id),
            ],
          }
      ),
    []
  );

  return {
    prep,
    loadError: loadError?.message ?? null,
    notFound: Boolean(loadError?.notFound),
    reload,
    readOnly: prep !== null && prepRole(prep) === 'viewer',
    name: draft?.name ?? '',
    steps: draft?.steps ?? NO_STEPS,
    settings: draft?.settings ?? null,
    setName,
    setSteps,
    setSettings,
    addStep,
    upto,
    setUpto,
    compare,
    requestCompare,
    preview,
    previewing,
    previewError,
    retryPreview,
    reports,
    saveStatus: save.status,
    saveError: save.error,
    flush,
    retrySave,
    discard,
    addDataset,
  };
}

const NO_STEPS: PipelineStep[] = [];

function notifyNotSaved(pending: Pending, error: Error): void {
  notifications.show({
    color: 'red',
    title: 'Changes not saved',
    message: `“${pending.draft.name.trim() || UNTITLED}”: ${error.message}`,
  });
}
