import { useCallback, useEffect, useRef, useState } from 'react';
import { notifications } from '@mantine/notifications';
import { api, type Dataset } from '@/api';
import { formatBytes } from './format';
import { loadSource, saveSource, type SourceChoice } from './settings';

export interface DataController {
  /** The chosen data source (remembered per browser). */
  source: SourceChoice;
  setSource: (update: SourceChoice | ((current: SourceChoice) => SourceChoice)) => void;
  /** The user's uploaded datasets, newest first; null until loaded. */
  datasets: Dataset[] | null;
  datasetsError: string | null;
  reloadDatasets: () => void;
  /** Name of the file being uploaded. */
  uploading: string | null;
  uploadError: string | null;
  dismissUploadError: () => void;
  upload: (file: File) => void;
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
  const [datasets, setDatasets] = useState<Dataset[] | null>(null);
  const [datasetsError, setDatasetsError] = useState<string | null>(null);
  const [uploading, setUploading] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  // Changes whenever the session ends, so late responses of a former session are ignored.
  const session = useRef(0);
  // Bumped by every change of the list, so a list request that started before is ignored.
  const version = useRef(0);
  const busy = useRef(false);

  useEffect(() => {
    session.current += 1;
    version.current += 1;
    if (!active) {
      setDatasets(null);
      setDatasetsError(null);
      setUploading(null);
      setUploadError(null);
      busy.current = false;
    }
  }, [active]);

  const setSource = useCallback(
    (update: SourceChoice | ((current: SourceChoice) => SourceChoice)) =>
      setSourceState((current) => (typeof update === 'function' ? update(current) : update)),
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
      busy.current = true;
      setUploading(file.name);
      api.training.datasets
        .upload(file)
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
          if (session.current !== started) {
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
            setUploading(null);
          }
        });
    },
    [active, maxUploadMb, setSource, setupVisible]
  );

  const remove = useCallback(async (dataset: Dataset) => {
    await api.training.datasets.remove(dataset.id);
    version.current += 1;
    setDatasets((list) => list?.filter((d) => d.id !== dataset.id) ?? null);
  }, []);

  return {
    source,
    setSource,
    datasets,
    datasetsError,
    reloadDatasets,
    uploading,
    uploadError,
    dismissUploadError: useCallback(() => setUploadError(null), []),
    upload,
    remove,
  };
}
