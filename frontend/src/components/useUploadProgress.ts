import { useCallback, useEffect, useRef, useState } from 'react';

/** Bytes of a running upload. All sent: the server is working on the file. */
export interface UploadBytes {
  sent: number;
  total: number;
}

/** The progress is shown only after this many ms, so that small files do not flash it. */
export const SHOW_PROGRESS_AFTER = 300;

/** The shown progress changes at most this often (ms); the bar glides in between. */
export const PROGRESS_INTERVAL = 200;

/**
 * The progress of an upload for <UploadProgress>. `start(size)` begins a new upload, `report`
 * takes its progress events and `stop` ends it. `bytes` stays null for the first 300 ms and
 * then changes at most every 200 ms (at once when everything is sent), so a long upload does
 * not re-render the page all the time.
 */
export function useUploadProgress() {
  const [bytes, setBytes] = useState<UploadBytes | null>(null);
  // The latest numbers; null while no upload runs.
  const latest = useRef<UploadBytes | null>(null);
  const shown = useRef(false);
  const lastUpdate = useRef(0);
  // Shows the progress first, then the numbers held back by the interval.
  const timer = useRef<number | undefined>(undefined);

  useEffect(() => () => window.clearTimeout(timer.current), []);

  const show = useCallback(() => {
    window.clearTimeout(timer.current);
    timer.current = undefined;
    shown.current = true;
    lastUpdate.current = Date.now();
    setBytes(latest.current);
  }, []);

  const stop = useCallback(() => {
    window.clearTimeout(timer.current);
    timer.current = undefined;
    latest.current = null;
    shown.current = false;
    setBytes(null);
  }, []);

  const start = useCallback(
    (size: number) => {
      stop();
      latest.current = { sent: 0, total: size };
      timer.current = window.setTimeout(show, SHOW_PROGRESS_AFTER);
    },
    [show, stop]
  );

  const report = useCallback(
    (sent: number, total: number) => {
      if (!latest.current) {
        return;
      }
      latest.current = { sent, total };
      if (!shown.current) {
        return;
      }
      const wait = sent < total ? lastUpdate.current + PROGRESS_INTERVAL - Date.now() : 0;
      if (wait <= 0) {
        show();
      } else if (timer.current === undefined) {
        timer.current = window.setTimeout(show, wait);
      }
    },
    [show]
  );

  return { bytes, start, report, stop };
}
