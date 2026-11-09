import { useCallback, useEffect, useRef, useState } from 'react';
import { notifications } from '@mantine/notifications';
import { api, type Dataset } from '@/api';
import { useUploadProgress, type UploadBytes } from '@/components/useUploadProgress';
import { formatBytes } from './format';
import { loadSource, saveSource, type SourceChoice } from './settings';

export interface DataController {
  /** The chosen data source (remembered per browser). */
  source: SourceChoice;
  /**
   * Choose a data source. Every call counts as a selection (see `selection`): picking a
   * dataset, a finished upload, "Train with this data" in the Data tab.
   */
  setSource: (update: SourceChoice | ((current: SourceChoice) => SourceChoice)) => void;
  /** Counts the selections, so the builder can adapt the network once per selection. */
  selection: number;
  /** The user's uploaded datasets, newest first; null until loaded. */
  datasets: Dataset[] | null;
  datasetsError: string | null;
  reloadDatasets: () => void;
  /** Name of the file being uploaded. */
  uploading: string | null;
  /** How much of it is sent; null for the first moment of an upload. */
  uploadProgress: UploadBytes | null;
  uploadError: string | null;
  dismissUploadError: () => void;
  upload: (file: File) => void;
  /** Stop the running upload (without an error). */
  cancelUpload: () => void;
  /** Delete an uploaded dataset; throws if the server refuses. */
  remove: (dataset: Dataset) => Promise<void>;
}

/**
 * Uploaded datasets and the data source of the training panel. Lives as long as the builder,
 * so an upload keeps going (and reports its result) when the panel is closed or another tab is
 * shown. `setupVisible` tells whether the result can be seen in the panel right now.
 */
export function useDatasets(
  active: boolean,
  maxUploadMb: number | null,
  setupVisible: () => boolean
): DataController {
  const [source, setSourceState] = useState<SourceChoice>(loadSource);
  const [selection, setSelection] = useState(0);
  const [datasets, setDatasets] = useState<Dataset[] | null>(null);
  const [datasetsError, setDatasetsError] = useState<string | null>(null);
  const [uploading, setUploading] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  // Changes whenever the session ends, so late responses of a former session are ignored.
  const session = useRef(0);
  // Bumped by every change of the list, so a list request that started before is ignored.
  const version = useRef(0);
  const busy = useRef(false);
  const running = useRef<AbortController | null>(null);
  const {
    bytes: uploadProgress,
    start: startProgress,
    report: reportProgress,
    stop: stopProgress,
  } = useUploadProgress();

  useEffect(() => {
    session.current += 1;
    version.current += 1;
    if (!active) {
      setDatasets(null);
      setDatasetsError(null);
      setUploading(null);
      setUploadError(null);
      busy.current = false;
      // The upload belonged to the account that logged out.
      running.current?.abort();
      running.current = null;
      stopProgress();
    }
  }, [active, stopProgress]);

  const setSource = useCallback(
    (update: SourceChoice | ((current: SourceChoice) => SourceChoice)) => {
      setSourceState((current) => (typeof update === 'function' ? update(current) : update));
      setSelection((n) => n + 1);
    },
    []
  );
  useEffect(() => saveSource(source), [source]);

  const reloadDatasets = useCallback(() => {
    if (!active) {
      return;
    }
    const started = version.current;
    setDatasetsError(null);
    api.training.datasets
      .list()
      .then((list) => {
        if (version.current === started) {
          setDatasets(list);
        }
      })
      .catch((error: Error) => {
        if (version.current === started) {
          setDatasetsError(error.message);
        }
      });
  }, [active]);

  const upload = useCallback(
    (file: File) => {
      if (!active || busy.current) {
        return;
      }
      setUploadError(null);
      if (maxUploadMb !== null && file.size > maxUploadMb * 1024 * 1024) {
        setUploadError(
          `The file is ${formatBytes(file.size)}; the server accepts up to ${maxUploadMb} MB.`
        );
        return;
      }
      const started = session.current;
      const controller = new AbortController();
      running.current = controller;
      busy.current = true;
      setUploading(file.name);
      startProgress(file.size);
      api.training.datasets
        .upload(file, undefined, {
          signal: controller.signal,
          onUploadProgress: (sent, total) => {
            if (session.current === started) {
              reportProgress(sent, total);
            }
          },
        })
        .then((dataset) => {
          if (session.current !== started) {
            return;
          }
          version.current += 1;
          setDatasets((list) => [dataset, ...(list ?? []).filter((d) => d.id !== dataset.id)]);
          // Select it, without undoing a change of the data source made meanwhile.
          setSource((current) => ({ ...current, upload: dataset.id }));
          if (!setupVisible()) {
            notifications.show({
              color: 'green',
              title: 'Upload finished',
              message: `“${dataset.name}” is ready for training.`,
            });
          }
        })
        .catch((error: Error) => {
          if (session.current !== started || controller.signal.aborted) {
            return;
          }
          setUploadError(error.message);
          if (!setupVisible()) {
            notifications.show({
              color: 'red',
              title: `Uploading “${file.name}” failed`,
              message: error.message,
            });
          }
        })
        .finally(() => {
          if (session.current === started) {
            busy.current = false;
            running.current = null;
            setUploading(null);
            stopProgress();
          }
        });
    },
    [active, maxUploadMb, setSource, setupVisible, startProgress, reportProgress, stopProgress]
  );

  const remove = useCallback(async (dataset: Dataset) => {
    await api.training.datasets.remove(dataset.id);
    version.current += 1;
    setDatasets((list) => list?.filter((d) => d.id !== dataset.id) ?? null);
  }, []);

  return {
    source,
    setSource,
    selection,
    datasets,
    datasetsError,
    reloadDatasets,
    uploading,
    uploadProgress,
    uploadError,
    dismissUploadError: useCallback(() => setUploadError(null), []),
    upload,
    cancelUpload: useCallback(() => running.current?.abort(), []),
    remove,
  };
}
