import { useEffect, useMemo, useState } from 'react';
import {
  api,
  type BudgetCheck,
  type CostEstimate,
  type DatasetRef,
  type GraphDocument,
  type QueuePreview,
  type TrainingConfig,
  type TuningInput,
} from '@/api';
import { semanticKey } from '@/builder/graph';

export interface CostEstimateState {
  estimate: CostEstimate | null;
  budget: BudgetCheck | null;
  /** Servers with a training queue: where the run would wait and for how long. */
  queue: QueuePreview | null;
  loading: boolean;
  /** The answer belongs to the current network, data and settings. */
  fresh: boolean;
}

interface Answer {
  estimate: CostEstimate | null;
  budget: BudgetCheck | null;
  queue: QueuePreview | null;
  key: string | null;
}

const NONE: Answer = { estimate: null, budget: null, queue: null, key: null };

/**
 * The estimated CPU time of the planned training and whether the budgets allow it, from the
 * training check with the current settings (debounced). Servers without budgets answer
 * without them: then both stay null.
 */
export function useCostEstimate(
  doc: GraphDocument,
  dataset: DatasetRef | null,
  config: Partial<TrainingConfig>,
  tuning: TuningInput | null,
  enabled: boolean,
  delay = 600
): CostEstimateState {
  const graphKey = useMemo(() => semanticKey(doc), [doc]);
  const key = dataset ? JSON.stringify([graphKey, dataset, config, tuning]) : null;
  const [answer, setAnswer] = useState<Answer>(NONE);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!enabled || key === null) {
      setLoading(false);
      return undefined;
    }
    const controller = new AbortController();
    setLoading(true);
    const timer = window.setTimeout(() => {
      const [graph, ref, settings, search] = JSON.parse(key) as [
        string,
        DatasetRef,
        Partial<TrainingConfig>,
        TuningInput | null,
      ];
      Promise.resolve()
        .then(() =>
          api.training.estimate(
            JSON.parse(graph) as GraphDocument,
            ref,
            settings,
            search,
            controller.signal
          )
        )
        .then((result) => {
          setAnswer({
            estimate: result.estimate ?? null,
            budget: result.budget ?? null,
            queue: result.queue ?? null,
            key,
          });
          setLoading(false);
        })
        .catch(() => {
          if (!controller.signal.aborted) {
            // Only informative: the start itself checks the budgets again.
            setAnswer({ ...NONE, key });
            setLoading(false);
          }
        });
    }, delay);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [key, enabled, delay]);

  const fresh = enabled && !loading && answer.key === key;
  return {
    estimate: answer.estimate,
    budget: answer.budget,
    queue: answer.queue,
    loading,
    fresh,
  };
}
