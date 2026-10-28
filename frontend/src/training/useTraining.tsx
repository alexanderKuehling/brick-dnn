import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Button, Text } from '@mantine/core';
import { notifications } from '@mantine/notifications';
import { api, ApiError, type RunDetail, type RunSummary, type TrainingStatus } from '@/api';
import { isActive, resultText } from './format';
import { useDatasets, type DataController } from './useDatasets';

export type TrainingView = 'setup' | 'history' | 'run';

/** Poll the shown run this often while it is active (ms). */
export const DETAIL_INTERVAL = 1000;
/** Poll the run list this often while a run is active (ms). */
export const LIST_INTERVAL = 5000;

function summaryOf(run: RunSummary | RunDetail): RunSummary {
  const { graph: _graph, plan: _plan, state: _state, ...summary } = run as RunDetail;
  return summary;
}

/**
 * Whether `run`, from a response to a request stamped `stamp`, may replace what is known:
 * no later request updated the run yet, and a finished run never becomes active again.
 */
function accepts(
  stamps: Map<string, number>,
  run: RunSummary,
  stamp: number,
  existing: RunSummary | undefined
): boolean {
  return (
    stamp >= (stamps.get(run.id) ?? -Infinity) &&
    !(existing && !isActive(existing.status) && isActive(run.status))
  );
}

export interface TrainingController {
  /** Server capabilities; null while loading. */
  status: TrainingStatus | null;
  statusError: string | null;
  reloadStatus: () => void;
  /** The user's runs, newest first; null until loaded (or when logged out). */
  runs: RunSummary[] | null;
  runsError: string | null;
  refreshRuns: () => void;
  /** The newest active run of this user, if any. */
  activeRun: RunSummary | null;

  opened: boolean;
  open: () => void;
  close: () => void;
  view: TrainingView;
  showSetup: () => void;
  showHistory: () => void;
  openRun: (id: string) => void;
  selectedRunId: string | null;
  /** Details of the selected run (null while loading). */
  detail: RunDetail | null;
  detailError: string | null;
  retryDetail: () => void;
  /** Use a run returned by the API (after starting or stopping it). */
  applyRun: (run: RunDetail) => void;
  forgetRun: (id: string) => void;
  /** Data source, uploaded datasets and uploads in progress. */
  data: DataController;
}

/**
 * Training state of the builder: server capabilities, the user's runs and the polling that
 * keeps them up to date. Shows a notification when a run finishes while it is not on screen.
 *
 * Past runs stay available whenever training is enabled on the server, even if PyTorch is
 * missing there right now: only starting new trainings needs it.
 */
export function useTraining(loggedIn: boolean): TrainingController {
  const [status, setStatus] = useState<TrainingStatus | null>(null);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [statusRequest, setStatusRequest] = useState(0);
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [runsError, setRunsError] = useState<string | null>(null);
  const [runsRequest, setRunsRequest] = useState(0);
  const [opened, setOpened] = useState(false);
  const [view, setView] = useState<TrainingView>('setup');
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [detailError, setDetailError] = useState<string | null>(null);

  const enabled = Boolean(status?.enabled);
  const active = loggedIn && enabled;
  const activeRun = useMemo(() => runs?.find((run) => isActive(run.status)) ?? null, [runs]);

  // Latest values for callbacks that outlive a render (polling, notifications, uploads).
  const shown = useRef({ opened, view, selectedRunId });
  shown.current = { opened, view, selectedRunId };
  const lastStatus = useRef(new Map<string, RunSummary['status']>());

  const setupVisible = useCallback(
    () => shown.current.opened && shown.current.view === 'setup',
    []
  );
  const data = useDatasets(active, status?.max_upload_mb ?? null, setupVisible);

  // ---------------------------------------------------------------- open / navigate
  const openRun = useCallback((id: string) => {
    setOpened(true);
    setView('run');
    setDetailError(null);
    setDetail((current) => (current?.id === id ? current : null));
    setSelectedRunId(id);
  }, []);

  const notifyFinished = useCallback(
    (run: RunSummary) => {
      const { opened: isOpen, view: currentView, selectedRunId: shownId } = shown.current;
      if (isOpen && currentView === 'run' && shownId === run.id) {
        return; // The user is watching it.
      }
      const id = `training-${run.id}`;
      const result = resultText(run);
      const titles = {
        completed: 'Training finished',
        stopped: 'Training stopped',
        failed: 'Training failed',
      } as const;
      const status = run.status as keyof typeof titles;
      notifications.show({
        id,
        color: status === 'failed' ? 'red' : status === 'completed' ? 'green' : 'orange',
        title: titles[status] ?? 'Training ended',
        autoClose: status === 'failed' ? false : 15000,
        message: (
          <>
            <Text size="sm">
              “{run.name}”
              {status === 'failed'
                ? `: ${run.message || 'Something went wrong.'}`
                : result
                  ? `: ${result}`
                  : '.'}
            </Text>
            <Button
              size="compact-xs"
              variant="light"
              mt={6}
              onClick={() => {
                notifications.hide(id);
                openRun(run.id);
              }}
            >
              {status === 'failed' ? 'Show details' : 'Show results'}
            </Button>
          </>
        ),
      });
    },
    [openRun]
  );

  /** Remember statuses and report runs that just ended. */
  const track = useCallback(
    (list: RunSummary[]) => {
      for (const run of list) {
        const previous = lastStatus.current.get(run.id);
        if (previous && isActive(previous) && !isActive(run.status)) {
          notifyFinished(run);
        }
        lastStatus.current.set(run.id, run.status);
      }
    },
    [notifyFinished]
  );

  // ---------------------------------------------------------------- merging responses
  // Every request takes a stamp when it starts (see `accepts`), so a slow list response
  // cannot undo a newer detail, a stop or a deletion.
  const runsRef = useRef<RunSummary[] | null>(null);
  const stamps = useRef(new Map<string, number>());
  const counter = useRef(0);
  const nextStamp = useCallback(() => {
    counter.current += 1;
    return counter.current;
  }, []);

  const commitRuns = useCallback((next: RunSummary[] | null) => {
    runsRef.current = next;
    setRuns(next);
  }, []);

  const applyList = useCallback(
    (list: RunSummary[], stamp: number) => {
      const current = runsRef.current ?? [];
      const known = new Map(current.map((run) => [run.id, run]));
      const listed = new Set(list.map((run) => run.id));
      // Runs learned about after this request started (e.g. just created) are kept.
      const next = current.filter(
        (run) => !listed.has(run.id) && (stamps.current.get(run.id) ?? -Infinity) > stamp
      );
      const accepted: RunSummary[] = [];
      for (const run of list) {
        const existing = known.get(run.id);
        if (accepts(stamps.current, run, stamp, existing)) {
          stamps.current.set(run.id, stamp);
          accepted.push(run);
          next.push(run);
        } else if (existing) {
          next.push(existing);
        }
      }
      commitRuns(next);
      track(accepted);
    },
    [commitRuns, track]
  );

  /** Merge one run into the list. Returns false if newer information is already known. */
  const applyRunData = useCallback(
    (run: RunSummary | RunDetail, stamp: number): boolean => {
      const summary = summaryOf(run);
      const current = runsRef.current ?? [];
      const existing = current.find((r) => r.id === summary.id);
      if (!accepts(stamps.current, summary, stamp, existing)) {
        return false;
      }
      stamps.current.set(summary.id, stamp);
      commitRuns(
        existing ? current.map((r) => (r.id === summary.id ? summary : r)) : [summary, ...current]
      );
      track([summary]);
      return true;
    },
    [commitRuns, track]
  );

  const forgetRun = useCallback(
    (id: string) => {
      // The stamp also keeps an older list response from bringing the run back.
      stamps.current.set(id, nextStamp());
      lastStatus.current.delete(id);
      commitRuns(runsRef.current?.filter((r) => r.id !== id) ?? null);
      setSelectedRunId((current) => (current === id ? null : current));
      setDetail((current) => (current?.id === id ? null : current));
      setView((current) => (current === 'run' ? 'history' : current));
    },
    [commitRuns, nextStamp]
  );

  // ---------------------------------------------------------------- capabilities
  useEffect(() => {
    const controller = new AbortController();
    setStatusError(null);
    api.training
      .status(controller.signal)
      .then(setStatus)
      .catch((error: Error) => {
        if (controller.signal.aborted) {
          return;
        }
        // Keep the last known capabilities: a failed refresh should not hide the runs.
        setStatusError(
          error instanceof ApiError && error.status === 404
            ? 'This server does not support training. Update Brick DNN to use it.'
            : error.message
        );
      });
    return () => controller.abort();
  }, [statusRequest]);

  // ---------------------------------------------------------------- run list
  useEffect(() => {
    if (!active) {
      commitRuns(null);
      setDetail(null);
      setSelectedRunId(null);
      lastStatus.current.clear();
      stamps.current.clear();
      return undefined;
    }
    const controller = new AbortController();
    const stamp = nextStamp();
    api.training.runs
      .list(controller.signal)
      .then((list) => {
        if (!controller.signal.aborted) {
          applyList(list, stamp);
          setRunsError(null);
        }
      })
      .catch((error: Error) => {
        if (!controller.signal.aborted) {
          setRunsError(error.message);
        }
      });
    return () => controller.abort();
  }, [active, runsRequest, applyList, commitRuns, nextStamp]);

  // ---------------------------------------------------------------- polling
  const watchingDetail =
    opened &&
    view === 'run' &&
    selectedRunId !== null &&
    (detail?.id !== selectedRunId ? detailError === null : isActive(detail.status));
  const mode: 'detail' | 'list' | 'idle' = !active
    ? 'idle'
    : watchingDetail
      ? 'detail'
      : activeRun
        ? 'list'
        : 'idle';

  useEffect(() => {
    if (mode === 'idle' || (mode === 'detail' && !selectedRunId)) {
      return undefined;
    }
    const controller = new AbortController();
    const cleanups: (() => void)[] = [];

    /** Call `poll` every `interval` ms until it returns false or the effect ends. */
    const repeat = (poll: () => Promise<boolean>, interval: number, immediate: boolean) => {
      let timer = 0;
      const tick = async () => {
        if (controller.signal.aborted) {
          return;
        }
        const again = await poll();
        if (again && !controller.signal.aborted) {
          timer = window.setTimeout(tick, interval);
        }
      };
      cleanups.push(() => window.clearTimeout(timer));
      if (immediate) {
        void tick();
      } else {
        timer = window.setTimeout(tick, interval);
      }
    };

    // The list keeps the toolbar and other runs current, also while one run is watched.
    repeat(
      async () => {
        const stamp = nextStamp();
        try {
          const list = await api.training.runs.list(controller.signal);
          if (controller.signal.aborted) {
            return false;
          }
          applyList(list, stamp);
          setRunsError(null);
        } catch {
          // Keep the last list; the next round tries again.
        }
        return true;
      },
      LIST_INTERVAL,
      false
    );

    if (mode === 'detail') {
      const id = selectedRunId!;
      repeat(
        async () => {
          const stamp = nextStamp();
          try {
            const run = await api.training.runs.get(id, controller.signal);
            if (controller.signal.aborted) {
              return false;
            }
            if (applyRunData(run, stamp)) {
              setDetail(run);
            }
            setDetailError(null);
          } catch (error) {
            if (controller.signal.aborted) {
              return false;
            }
            if (error instanceof ApiError && error.status === 404) {
              forgetRun(id);
              return false;
            }
            setDetailError((error as Error).message);
          }
          return true;
        },
        DETAIL_INTERVAL,
        true
      );
    }

    return () => {
      controller.abort();
      cleanups.forEach((cleanup) => cleanup());
    };
  }, [mode, selectedRunId, applyList, applyRunData, forgetRun, nextStamp]);

  // ---------------------------------------------------------------- public API
  const refreshRuns = useCallback(() => setRunsRequest((n) => n + 1), []);
  const activeRef = useRef(activeRun);
  activeRef.current = activeRun;

  const open = useCallback(() => {
    setOpened(true);
    refreshRuns();
    const current = activeRef.current;
    if (current) {
      openRun(current.id);
    } else {
      setView((view) => (view === 'run' && !shown.current.selectedRunId ? 'setup' : view));
    }
  }, [openRun, refreshRuns]);

  const applyRun = useCallback(
    (run: RunDetail) => {
      if (applyRunData(run, nextStamp())) {
        setDetail(run);
      }
      setSelectedRunId(run.id);
      setDetailError(null);
      setView('run');
    },
    [applyRunData, nextStamp]
  );

  return {
    status,
    statusError,
    reloadStatus: useCallback(() => setStatusRequest((n) => n + 1), []),
    runs,
    runsError,
    refreshRuns,
    activeRun,
    opened,
    open,
    close: useCallback(() => setOpened(false), []),
    view,
    showSetup: useCallback(() => setView('setup'), []),
    showHistory: useCallback(() => {
      setView('history');
      refreshRuns();
    }, [refreshRuns]),
    openRun,
    selectedRunId,
    detail: detail && detail.id === selectedRunId ? detail : null,
    detailError,
    retryDetail: useCallback(() => setDetailError(null), []),
    applyRun,
    forgetRun,
    data,
  };
}
