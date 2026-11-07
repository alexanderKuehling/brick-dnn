import { useCallback, useEffect, useRef, useState } from 'react';
import {
  api,
  ApiError,
  type DataPrep,
  type DataSource,
  type DataStatus,
  type StepCatalog,
} from '@/api';

export interface DataResources {
  /** What the server can do; null while loading. */
  status: DataStatus | null;
  statusError: string | null;
  /** Ask the server again what it can do (e.g. after installing pandas). */
  reloadStatus: () => void;
  /** All step types; null while loading (and when data preparation is unavailable). */
  catalog: StepCatalog | null;
  catalogError: string | null;
  /** The user's imported tables; null until loaded (or when logged out). */
  sources: DataSource[] | null;
  /** The user's preparations; null until loaded (or when logged out). */
  preps: DataPrep[] | null;
  listsError: string | null;
  /** Load whatever failed again (status, catalog, lists). */
  retry: () => void;
  /** Add a table, or replace it if it is listed. */
  putSource: (source: DataSource) => void;
  dropSource: (id: number) => void;
  /** Add a preparation (at the top), or replace it in place if it is listed. */
  putPrep: (prep: DataPrep) => void;
  dropPrep: (id: number) => void;
}

function put<T extends { id: number }>(list: T[] | null, item: T): T[] {
  const current = list ?? [];
  return current.some((entry) => entry.id === item.id)
    ? current.map((entry) => (entry.id === item.id ? item : entry))
    : [item, ...current];
}

/**
 * Everything the Data page needs besides the open preparation: server capabilities, the step
 * catalog, and the user's tables and preparations. Lists are only loaded when logged in and
 * the server supports data preparation.
 */
export function useDataResources(loggedIn: boolean): DataResources {
  const [status, setStatus] = useState<DataStatus | null>(null);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [statusRequest, setStatusRequest] = useState(0);
  const [catalog, setCatalog] = useState<StepCatalog | null>(null);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [catalogRequest, setCatalogRequest] = useState(0);
  const [sources, setSources] = useState<DataSource[] | null>(null);
  const [preps, setPreps] = useState<DataPrep[] | null>(null);
  const [listsError, setListsError] = useState<string | null>(null);
  const [listsRequest, setListsRequest] = useState(0);
  // Bumped by every local change of the lists: a list request that started before it would
  // undo the change, so its response is replaced by a new request.
  const version = useRef(0);

  const available = Boolean(status?.available);
  const active = loggedIn && available;

  useEffect(() => {
    const controller = new AbortController();
    setStatusError(null);
    api.data
      .status(controller.signal)
      .then(setStatus)
      .catch((error: Error) => {
        if (controller.signal.aborted) {
          return;
        }
        setStatusError(
          error instanceof ApiError && error.status === 404
            ? 'This server does not support data preparation. Update Brick DNN to use it.'
            : error.message
        );
      });
    return () => controller.abort();
  }, [statusRequest]);

  useEffect(() => {
    if (!available) {
      return undefined;
    }
    const controller = new AbortController();
    setCatalogError(null);
    api.data
      .steps(controller.signal)
      .then(setCatalog)
      .catch((error: Error) => {
        if (!controller.signal.aborted) {
          setCatalogError(error.message);
        }
      });
    return () => controller.abort();
  }, [available, catalogRequest]);

  useEffect(() => {
    version.current += 1;
    if (!active) {
      setSources(null);
      setPreps(null);
      setListsError(null);
      return undefined;
    }
    const controller = new AbortController();
    const started = version.current;
    setListsError(null);
    Promise.all([api.data.sources.list(controller.signal), api.data.preps.list(controller.signal)])
      .then(([sourceList, prepList]) => {
        if (controller.signal.aborted) {
          return;
        }
        if (version.current !== started) {
          setListsRequest((n) => n + 1);
          return;
        }
        setSources(sourceList);
        setPreps(prepList);
      })
      .catch((error: Error) => {
        if (!controller.signal.aborted) {
          setListsError(error.message);
        }
      });
    return () => controller.abort();
  }, [active, listsRequest]);

  const reloadStatus = useCallback(() => setStatusRequest((n) => n + 1), []);
  const retry = useCallback(() => {
    if (statusError) {
      setStatusRequest((n) => n + 1);
    }
    if (catalogError) {
      setCatalogRequest((n) => n + 1);
    }
    if (listsError) {
      setListsRequest((n) => n + 1);
    }
  }, [statusError, catalogError, listsError]);

  const putSource = useCallback((source: DataSource) => {
    version.current += 1;
    setSources((list) => put(list, source));
  }, []);
  const dropSource = useCallback((id: number) => {
    version.current += 1;
    setSources((list) => list?.filter((source) => source.id !== id) ?? null);
  }, []);
  const putPrep = useCallback((prep: DataPrep) => {
    version.current += 1;
    setPreps((list) => put(list, prep));
  }, []);
  const dropPrep = useCallback((id: number) => {
    version.current += 1;
    setPreps((list) => list?.filter((prep) => prep.id !== id) ?? null);
  }, []);

  return {
    status,
    statusError,
    reloadStatus,
    catalog,
    catalogError,
    sources,
    preps,
    listsError,
    retry,
    putSource,
    dropSource,
    putPrep,
    dropPrep,
  };
}
