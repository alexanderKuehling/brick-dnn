/**
 * Classical models (Gaussian processes, SVMs, gradient boosting and ensembles of them) train
 * with scikit-learn instead of PyTorch:
 * fitted once, without epochs, on tabular data (see docs/training.md#classical-models).
 * Hybrid networks need both: neural layers learn the features (PyTorch, with epochs), then a
 * classical model is fitted on them (see docs/training.md#hybrid-networks).
 */
import type { Analysis, RunSummary, TrainingStatus } from '@/api/types';
import { networkFamily } from '@/builder/kinds';
import { i18n } from '@/i18n';

export type Family = 'neural' | 'classical' | 'hybrid';

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
  if (family === 'hybrid') {
    return status.available && Boolean(status.classical?.available);
  }
  return family === 'classical' ? Boolean(status.classical?.available) : status.available;
}

/** Why trainings of a `family` network cannot start (null: they can, or unknown). */
export function unavailableReason(status: TrainingStatus, family: Family): string | null {
  if (family === 'classical') {
    return status.classical?.reason ?? null;
  }
  if (family === 'hybrid') {
    return status.reason ?? status.classical?.reason ?? null;
  }
  return status.reason;
}

/**
 * The most training rows the classical model of the network is fitted on (larger training
 * sets are subsampled); for an ensemble the strictest limit of its models. Null for neural
 * networks or when the server does not say.
 */
export function classicalMaxRows(
  status: TrainingStatus | null,
  analysis: Analysis | null
): number | null {
  const classical = analysis?.classical;
  if (!status || !classical || networkFamily(analysis) === 'neural') {
    return null;
  }
  const types = classical.members ? classical.members.map((m) => m.type) : [classical.type];
  const limits = types.map((type) => status.classical?.max_rows[type]);
  if (limits.length === 0 || limits.some((limit) => limit === undefined)) {
    return null;
  }
  return Math.min(...(limits as number[]));
}

/** A hybrid run: neural layers learned the features, a classical model was fitted on them. */
export function isHybridRun(run: Pick<RunSummary, 'family'> & { state?: unknown }): boolean {
  const state = run.state as { family?: string } | null | undefined;
  return (run.family ?? state?.family) === 'hybrid';
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
  get GaussianProcessRegressor() {
    return i18n.t('ui.training.classical.gaussianProcessRegression');
  },
  get GaussianProcessClassifier() {
    return i18n.t('ui.training.classical.gaussianProcessClassification');
  },
  get SVR() {
    return i18n.t('ui.training.classical.supportVectorRegression');
  },
  get SVC() {
    return i18n.t('ui.training.classical.supportVectorClassification');
  },
  get HistGradientBoostingRegressor() {
    return i18n.t('ui.training.classical.gradientBoostingRegression');
  },
  get HistGradientBoostingClassifier() {
    return i18n.t('ui.training.classical.gradientBoostingClassification');
  },
  get XGBRegressor() {
    return i18n.t('ui.training.classical.xgboostRegression');
  },
  get XGBClassifier() {
    return i18n.t('ui.training.classical.xgboostClassification');
  },
  get LGBMRegressor() {
    return i18n.t('ui.training.classical.lightgbmRegression');
  },
  get LGBMClassifier() {
    return i18n.t('ui.training.classical.lightgbmClassification');
  },
  get CatBoostRegressor() {
    return i18n.t('ui.training.classical.catboostRegression');
  },
  get CatBoostClassifier() {
    return i18n.t('ui.training.classical.catboostClassification');
  },
  get VotingClassifier() {
    return i18n.t('ui.training.classical.votingClassifier');
  },
  get VotingRegressor() {
    return i18n.t('ui.training.classical.votingRegression');
  },
  get StackingClassifier() {
    return i18n.t('ui.training.classical.stackingClassifier');
  },
  get StackingRegressor() {
    return i18n.t('ui.training.classical.stackingRegression');
  },
};
