/** Helpers for pipeline step lists, shared by the Data page and the pipeline editor. */
import type {
  ColumnInfo,
  NewStep,
  PipelineStep,
  StepCatalog,
  StepReport,
  StepSpec,
  StepStatus,
} from '@/api/types';

/** The spec of a step type, or undefined for types the server does not know (any more). */
export function stepSpec(catalog: StepCatalog, type: string): StepSpec | undefined {
  return catalog.steps.find((spec) => spec.type === type);
}

/** The number of a generated id ("s7" → 7), 0 for other ids. */
function idNumber(id: string): number {
  const match = /^s(\d+)$/.exec(id);
  return match ? Number(match[1]) : 0;
}

/**
 * A new step id ("s1", "s2", …), higher than every id in `steps` and in `reserved` (e.g. ids
 * of deleted steps and of the last reports). So a new step never gets the id of a step that
 * was deleted, and never shows that step's old report.
 */
export function newStepId(steps: PipelineStep[], reserved: Iterable<string> = []): string {
  let highest = steps.length;
  for (const step of steps) {
    highest = Math.max(highest, idNumber(step.id));
  }
  for (const id of reserved) {
    highest = Math.max(highest, idNumber(id));
  }
  return `s${highest + 1}`;
}

/** Default values of all params of a step type (copies, so they can be changed). */
export function defaultParams(spec: StepSpec): Record<string, unknown> {
  return Object.fromEntries(
    spec.params.map((param) => [param.name, structuredClone(param.default)])
  );
}

/** A complete step: defaults of the type, overridden by `step.params`. */
export function createStep(
  catalog: StepCatalog,
  step: NewStep,
  steps: PipelineStep[],
  reserved: Iterable<string> = []
): PipelineStep {
  const spec = stepSpec(catalog, step.type);
  const params = spec ? { ...defaultParams(spec), ...step.params } : { ...step.params };
  return { id: newStepId(steps, reserved), type: step.type, enabled: true, params };
}

/** The report of a step if it belongs to it (same type), else undefined. */
function ownReport(
  step: PipelineStep,
  reports: Record<string, StepReport>
): StepReport | undefined {
  const report = reports[step.id];
  return report && report.type === step.type ? report : undefined;
}

/** Index of the first switched-on step whose report says it failed, or -1. */
export function firstFailedStep(
  steps: PipelineStep[],
  reports: Record<string, StepReport>
): number {
  return steps.findIndex((step) => step.enabled && ownReport(step, reports)?.status === 'error');
}

export interface InsertOptions {
  /**
   * Reports of the last preview. Steps after a failed step do not run, so a normal step goes
   * in front of the first step that failed (where it can run and fix the data).
   */
  reports?: Record<string, StepReport>;
  /** Ids the new step must not get (see `newStepId`). */
  reserved?: Iterable<string>;
}

/**
 * The stage of a step: combine steps (join, append) come first, then the steps that run before
 * the split (remove atypical rows, find near-duplicate groups), then all others.
 */
export function stepStage(catalog: StepCatalog, type: string): 0 | 1 | 2 {
  const spec = stepSpec(catalog, type);
  return spec?.combine ? 0 : spec?.pre_split ? 1 : 2;
}

/**
 * Add a step: combine steps (join, append) go after the last combine step, because they must
 * come first; steps that run before the split go after the last combine or pre-split step.
 * All others go to the end, or right after `after` (a step id) if given, but in front of the
 * first step that failed (see `InsertOptions.reports`) and never in front of an earlier stage.
 */
export function insertStep(
  catalog: StepCatalog,
  steps: PipelineStep[],
  step: NewStep,
  after: string | null = null,
  options: InsertOptions = {}
): PipelineStep[] {
  const created = createStep(catalog, step, steps, options.reserved);
  const stage = stepStage(catalog, step.type);
  // End of the steps of this stage and of all earlier ones.
  let stageEnd = 0;
  let earlyEnd = 0;
  steps.forEach((s, index) => {
    const other = stepStage(catalog, s.type);
    if (other <= stage) {
      stageEnd = index + 1;
    }
    if (other < 2) {
      earlyEnd = index + 1;
    }
  });
  let position = stageEnd;
  if (stage === 2) {
    const anchor = after === null ? -1 : steps.findIndex((s) => s.id === after);
    position = anchor >= 0 ? anchor + 1 : steps.length;
    const failed = options.reports ? firstFailedStep(steps, options.reports) : -1;
    if (failed >= 0) {
      position = Math.min(position, failed);
    }
    position = Math.max(position, earlyEnd);
  }
  return [...steps.slice(0, position), created, ...steps.slice(position)];
}

/** Steps with these statuses did not run: their report lists the columns before them. */
const DID_NOT_RUN: ReadonlySet<StepStatus> = new Set<StepStatus>(['not_run', 'skipped']);

/**
 * The columns a step at `index` works on: the columns after the closest earlier step that
 * has a report of a run (steps that did not run are passed over), otherwise the columns of
 * the source.
 */
export function columnsBefore(
  steps: PipelineStep[],
  index: number,
  reports: Record<string, StepReport>,
  sourceColumns: ColumnInfo[]
): ColumnInfo[] {
  for (let i = index - 1; i >= 0; i -= 1) {
    const report = ownReport(steps[i], reports);
    if (report && !DID_NOT_RUN.has(report.status)) {
      return report.columns;
    }
  }
  return sourceColumns;
}

/** The columns at the end of the pipeline (for the target picker and the data preview). */
export function finalColumns(
  steps: PipelineStep[],
  reports: Record<string, StepReport>,
  sourceColumns: ColumnInfo[]
): ColumnInfo[] {
  return columnsBefore(steps, steps.length, reports, sourceColumns);
}

/** Reports keyed by step id. */
export function reportsById(reports: StepReport[] | undefined): Record<string, StepReport> {
  return Object.fromEntries((reports ?? []).map((report) => [report.id, report]));
}

/**
 * The reports of a preview by step id. A preview up to a step (`upto`) does not run the steps
 * after it: their reports are 'not_run' and list the columns at the previewed step. For
 * those, the reports of the last preview of all steps (`full`) are used if they have them, so
 * the columns of later steps and the target stay known.
 */
export function mergeReports(
  current: StepReport[] | undefined,
  upto: string | null,
  full: StepReport[] | null
): Record<string, StepReport> {
  const merged = reportsById(current);
  if (upto === null || !current || !full) {
    return merged;
  }
  const previewed = current.findIndex((report) => report.id === upto);
  if (previewed < 0) {
    return merged;
  }
  const older = reportsById(full);
  for (const report of current.slice(previewed + 1)) {
    const known = older[report.id];
    if (report.status === 'not_run' && known && known.type === report.type) {
      merged[report.id] = known;
    }
  }
  return merged;
}
