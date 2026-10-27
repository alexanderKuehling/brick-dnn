import { useEffect, useMemo, useState } from 'react';
import { api, type Analysis, type GraphDocument } from '@/api';
import { semanticKey } from './graph';

interface AnalysisState {
  analysis: Analysis | null;
  loading: boolean;
  error: string | null;
}

/**
 * Validates the graph on the server whenever it changes (debounced).
 * Moving layers around does not trigger a request: only the semantic content counts.
 */
export function useAnalysis(doc: GraphDocument, delay = 250): AnalysisState {
  const key = useMemo(() => semanticKey(doc), [doc]);
  const [state, setState] = useState<AnalysisState>({
    analysis: null,
    loading: false,
    error: null,
  });

  useEffect(() => {
    const graph = JSON.parse(key) as GraphDocument;
    if (graph.nodes.length === 0) {
      setState({ analysis: null, loading: false, error: null });
      return undefined;
    }
    const controller = new AbortController();
    setState((s) => ({ ...s, loading: true }));
    const timer = window.setTimeout(() => {
      api
        .analyze(graph, controller.signal)
        .then((analysis) => setState({ analysis, loading: false, error: null }))
        .catch((error: Error) => {
          if (!controller.signal.aborted) {
            setState((s) => ({ ...s, loading: false, error: error.message }));
          }
        });
    }, delay);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [key, delay]);

  return state;
}
