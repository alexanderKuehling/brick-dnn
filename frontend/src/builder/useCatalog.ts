import { useCallback, useEffect, useState } from 'react';
import { api, type Example, type LayerCatalog } from '@/api';

interface CatalogState {
  catalog: LayerCatalog | null;
  examples: Example[];
  error: string | null;
}

/** Loads the layer catalogue and the example networks from the API. */
export function useCatalog() {
  const [state, setState] = useState<CatalogState>({ catalog: null, examples: [], error: null });

  const load = useCallback(() => {
    setState((s) => ({ ...s, error: null }));
    Promise.all([api.layers(), api.examples()])
      .then(([catalog, examples]) => setState({ catalog, examples, error: null }))
      .catch((error: Error) => setState((s) => ({ ...s, error: error.message })));
  }, []);

  useEffect(load, [load]);
  return { ...state, reload: load };
}
