import { useCallback, useEffect, useState } from 'react';
import { api, type Example, type LayerCatalog } from '@/api';
import { useLanguage } from '@/i18n';

interface CatalogState {
  catalog: LayerCatalog | null;
  examples: Example[];
  error: string | null;
}

/** Loads the layer catalogue and the example networks from the API. */
export function useCatalog() {
  const [state, setState] = useState<CatalogState>({ catalog: null, examples: [], error: null });
  // The descriptions come in the app's language: load them again when it changes.
  const language = useLanguage();

  const load = useCallback(() => {
    setState((s) => ({ ...s, error: null }));
    Promise.all([api.layers(), api.examples()])
      .then(([catalog, examples]) => setState({ catalog, examples, error: null }))
      .catch((error: Error) => setState((s) => ({ ...s, error: error.message })));
  }, []);

  useEffect(load, [load, language]);
  return { ...state, reload: load };
}
