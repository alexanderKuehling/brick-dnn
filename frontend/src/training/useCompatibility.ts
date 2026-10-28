import { useEffect, useMemo, useState } from 'react';
import { api, type Compatibility, type DatasetRef, type GraphDocument } from '@/api';
import { semanticKey } from '@/builder/graph';

export interface CompatibilityState {
  result: Compatibility | null;
  loading: boolean;
  error: string | null;
}

/**
 * Asks the server whether the network can be trained on the dataset, whenever either changes
 * (debounced). Moving layers around does not trigger a request. The last result is kept
 * while a new check is running.
 */
export function useCompatibility(
  doc: GraphDocument,
  dataset: DatasetRef | null,
  enabled: boolean,
  delay = 400
): CompatibilityState {
  const key = useMemo(() => semanticKey(doc), [doc]);
  const datasetKey = dataset ? JSON.stringify(dataset) : null;
  const [state, setState] = useState<CompatibilityState>({
    result: null,
    loading: false,
    error: null,
  });

  useEffect(() => {
    if (!enabled || !datasetKey) {
      setState({ result: null, loading: false, error: null });
      return undefined;
    }
    const controller = new AbortController();
    setState((s) => ({ ...s, loading: true }));
    const timer = window.setTimeout(() => {
      api.training
        .check(
          JSON.parse(key) as GraphDocument,
          JSON.parse(datasetKey) as DatasetRef,
          controller.signal
        )
        .then((result) => setState({ result, loading: false, error: null }))
        .catch((error: Error) => {
          if (!controller.signal.aborted) {
            setState({ result: null, loading: false, error: error.message });
          }
        });
    }, delay);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [key, datasetKey, enabled, delay]);

  return state;
}
