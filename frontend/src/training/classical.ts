/**
 * Classical models (Gaussian processes, SVMs, gradient boosting) train with scikit-learn
 * instead of PyTorch:
 * fitted once, without epochs, on tabular data (see docs/training.md#classical-models).
 */
import type { Analysis, RunSummary, TrainingStatus } from '@/api/types';
import { networkFamily } from '@/builder/kinds';

export type Family = 'neural' | 'classical';

/** What the open network needs to train: PyTorch, or scikit-learn for a classical model. */
export function trainingFamily(analysis: Analysis | null | undefined): Family {
  return networkFamily(analysis);
}

/**
 * Whether new trainings of a `family` network can start on this server. `available` and
 * `reason` of the status are about PyTorch; classical models only need scikit-learn.
 */
export function canStartTraining(status: TrainingStatus, family: Family): boolean {
  if (!status.enabled) {
    return false;
  }
  return family === 'classical' ? Boolean(status.classical?.available) : status.available;
}

/** Why trainings of a `family` network cannot start (null: they can, or unknown). */
export function unavailableReason(status: TrainingStatus, family: Family): string | null {
  if (family === 'classical') {
    return status.classical?.reason ?? null;
  }
  return status.reason;
}

/**
 * The most training rows the classical model of the network is fitted on (larger training
 * sets are subsampled); null for neural networks or when the server does not say.
 */
export function classicalMaxRows(
  status: TrainingStatus | null,
  analysis: Analysis | null
): number | null {
  const type = analysis?.classical?.type;
  if (!status || !type || networkFamily(analysis) !== 'classical') {
    return null;
  }
  return status.classical?.max_rows[type] ?? null;
}

/** A run of a classical model (fitted once with scikit-learn). */
export function isClassicalRun(run: Pick<RunSummary, 'family'> & { state?: unknown }): boolean {
  if (run.family) {
    return run.family === 'classical';
  }
  const state = run.state as { family?: string } | null | undefined;
  return state?.family === 'classical';
}

/** "Gaussian process regression", or what the model is called in the catalogue. */
export const MODEL_KINDS: Record<string, string> = {
  GaussianProcessRegressor: 'Gaussian process regression',
  GaussianProcessClassifier: 'Gaussian process classification',
  SVR: 'Support vector regression',
  SVC: 'Support vector classification',
  HistGradientBoostingRegressor: 'Gradient boosting regression',
  HistGradientBoostingClassifier: 'Gradient boosting classification',
};
