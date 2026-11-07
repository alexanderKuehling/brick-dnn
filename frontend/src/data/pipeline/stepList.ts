/** Editing the step list: moving, duplicating, removing, and the status shown per step. */
import type { PipelineStep, StepCatalog, StepReport } from '@/api/types';
import { newStepId, stepSpec } from '../steps';

/** Same limit as the server (`pipeline.MAX_STEPS`). */
export const MAX_STEPS = 100;

export function isCombineStep(catalog: StepCatalog, step: PipelineStep): boolean {
  return Boolean(stepSpec(catalog, step.type)?.combine);
}

/**
 * Whether the step at `index` can move by `delta` (-1 up, +1 down). Combine steps (join,
 * append) must stay in front of all other steps, so the two kinds never swap.
 */
export function canMoveStep(
  catalog: StepCatalog,
  steps: PipelineStep[],
  index: number,
  delta: -1 | 1
): boolean {
  const target = index + delta;
  if (index < 0 || index >= steps.length || target < 0 || target >= steps.length) {
    return false;
  }
  return isCombineStep(catalog, steps[index]) === isCombineStep(catalog, steps[target]);
}

export function moveStep(
  catalog: StepCatalog,
  steps: PipelineStep[],
  index: number,
  delta: -1 | 1
): PipelineStep[] {
  if (!canMoveStep(catalog, steps, index, delta)) {
    return steps;
  }
  const next = [...steps];
  [next[index], next[index + delta]] = [next[index + delta], next[index]];
  return next;
}

/**
 * A copy of the step right after it (with a new id and its own params). `reserved`: ids the
 * copy must not get (see `newStepId`).
 */
export function duplicateStep(
  steps: PipelineStep[],
  index: number,
  reserved: Iterable<string> = []
): PipelineStep[] {
  const original = steps[index];
  if (!original) {
    return steps;
  }
  const copy: PipelineStep = {
    ...original,
    id: newStepId(steps, reserved),
    params: structuredClone(original.params),
  };
  return [...steps.slice(0, index + 1), copy, ...steps.slice(index + 1)];
}

/**
 * Put a removed step back at (about) its old place, keeping combine steps first. It keeps its
 * id unless another step has it by now; `reserved` as for `duplicateStep`.
 */
export function restoreStep(
  catalog: StepCatalog,
  steps: PipelineStep[],
  step: PipelineStep,
  index: number,
  reserved: Iterable<string> = []
): PipelineStep[] {
  const combineCount = steps.filter((s) => isCombineStep(catalog, s)).length;
  const position = isCombineStep(catalog, step)
    ? Math.min(index, combineCount)
    : Math.min(Math.max(index, combineCount), steps.length);
  const restored = steps.some((s) => s.id === step.id)
    ? { ...step, id: newStepId(steps, reserved) }
    : step;
  return [...steps.slice(0, position), restored, ...steps.slice(position)];
}

export type StepDisplayStatus = StepReport['status'] | 'waiting';

export const STATUS_LABELS: Record<StepDisplayStatus, string> = {
  ok: 'OK',
  error: 'Error',
  skipped: 'Skipped',
  disabled: 'Off',
  not_run: 'Not run',
  waiting: 'Waiting',
};

export const STATUS_COLORS: Record<StepDisplayStatus, string> = {
  ok: 'green',
  error: 'red',
  skipped: 'gray',
  disabled: 'gray',
  not_run: 'gray',
  waiting: 'gray',
};

export const STATUS_HINTS: Record<StepDisplayStatus, string> = {
  ok: 'The step ran without problems.',
  error: 'The step failed; the data after it is the data before it.',
  skipped: 'Not run because an earlier step failed.',
  disabled: 'Turned off: the data passes through unchanged.',
  not_run: 'Not run: the preview shows the data of an earlier step.',
  waiting: 'Waiting for the preview.',
};

/** The status to show: the step's own switch wins over an older report. */
export function stepStatus(step: PipelineStep, report: StepReport | undefined): StepDisplayStatus {
  if (!step.enabled) {
    return 'disabled';
  }
  if (!report || report.type !== step.type || report.status === 'disabled') {
    return 'waiting';
  }
  return report.status;
}
