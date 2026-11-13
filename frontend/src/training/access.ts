import type { Dataset, ProjectRole, RunSummary } from '@/api';

/** The saved network open in the builder, as far as training is concerned. */
export interface TrainingProject {
  id: number;
  /** The user's role; null until the server has answered. */
  role: ProjectRole | null;
  /** Shared with other people (members or a view link). */
  shared: boolean;
}

/** Who trains, and in which saved network (runs started there belong to it). */
export interface TrainingScope {
  /** The logged-in user; null if unknown (then every run counts as their own). */
  username: string | null;
  project: TrainingProject | null;
}

export const NO_SCOPE: TrainingScope = { username: null, project: null };

/** Which runs the history shows: the open network's (by everyone) or the user's own. */
export type RunFilter = 'network' | 'mine';

export function isOwnRun(run: RunSummary, scope: TrainingScope): boolean {
  return scope.username === null || !run.started_by || run.started_by === scope.username;
}

/** Who started a run, if it was someone else. */
export function otherStarter(run: RunSummary, scope: TrainingScope): string | null {
  return isOwnRun(run, scope) ? null : run.started_by;
}

export function isNetworkRun(run: RunSummary, scope: TrainingScope): boolean {
  return scope.project !== null && run.project === scope.project.id;
}

/** Runs the builder follows (toolbar, notifications): the user's and the open network's. */
export function isRelevantRun(run: RunSummary, scope: TrainingScope): boolean {
  return isOwnRun(run, scope) || isNetworkRun(run, scope);
}

/** Viewers of a network can look at its runs but not start trainings in it. */
export function canTrain(scope: TrainingScope): boolean {
  return scope.project?.role !== 'viewer';
}

/** New trainings belong to the open network if the user owns or edits it. */
export function trainingProject(scope: TrainingScope): TrainingProject | null {
  const project = scope.project;
  return project && (project.role === 'owner' || project.role === 'editor') ? project : null;
}

/**
 * Who started a run and the owner and editors of its network may stop it. For runs of other
 * networks the role is not known here: the server decides (and answers 403).
 */
export function canStopRun(run: RunSummary, scope: TrainingScope): boolean {
  if (isOwnRun(run, scope)) {
    return true;
  }
  if (isNetworkRun(run, scope)) {
    return scope.project!.role === 'owner' || scope.project!.role === 'editor';
  }
  return run.project !== null;
}

/** Who started a run and the owner of its network may delete it (see `canStopRun`). */
export function canDeleteRun(run: RunSummary, scope: TrainingScope): boolean {
  if (isOwnRun(run, scope)) {
    return true;
  }
  if (isNetworkRun(run, scope)) {
    return scope.project!.role === 'owner';
  }
  return run.project !== null;
}

/** Only who started a run may change its decision threshold (binary classification). */
export function canSetThreshold(run: RunSummary, scope: TrainingScope): boolean {
  return isOwnRun(run, scope);
}

/** The runs of one view of the history, in the order of the list (newest first). */
export function filterRuns(
  runs: RunSummary[],
  scope: TrainingScope,
  filter: RunFilter
): RunSummary[] {
  if (filter === 'network' && scope.project) {
    return runs.filter((run) => isNetworkRun(run, scope));
  }
  return runs.filter((run) => isOwnRun(run, scope));
}

/** Uploaded and exported datasets of the user (older servers only list those). */
export function isOwnDataset(dataset: Dataset): boolean {
  return dataset.mine !== false;
}
