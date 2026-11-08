import type { ColumnInfo, PipelineStep, StepReport } from '@/api/types';
import { CATALOG, STEP_REPORTS } from './fixtures';
import {
  columnsBefore,
  finalColumns,
  firstFailedStep,
  insertStep,
  mergeReports,
  newStepId,
  reportsById,
} from './steps';

const SOURCE_COLUMNS: ColumnInfo[] = [
  { name: 'age', kind: 'numeric' },
  { name: 'income', kind: 'numeric' },
];
const RENAMED: ColumnInfo[] = [
  { name: 'years', kind: 'numeric' },
  { name: 'income', kind: 'numeric' },
];

// impute → rename age to years → clip years
const STEPS: PipelineStep[] = [
  { id: 's1', type: 'impute', enabled: true, params: { columns: [], strategy: 'median' } },
  { id: 's2', type: 'rename_columns', enabled: true, params: { mapping: { age: 'years' } } },
  { id: 's3', type: 'clip', enabled: true, params: { columns: ['years'], mode: 'quantiles' } },
];

function report(step: PipelineStep, status: StepReport['status'], columns: ColumnInfo[]) {
  return { ...STEP_REPORTS[0], id: step.id, type: step.type, status, columns };
}

/** The last preview of all steps. */
const FULL = [
  report(STEPS[0], 'ok', SOURCE_COLUMNS),
  report(STEPS[1], 'ok', RENAMED),
  report(STEPS[2], 'ok', RENAMED),
];
/** The preview up to step 1: the later steps did not run and list the columns after step 1. */
const UPTO_S1 = [
  report(STEPS[0], 'ok', SOURCE_COLUMNS),
  report(STEPS[1], 'not_run', SOURCE_COLUMNS),
  report(STEPS[2], 'not_run', SOURCE_COLUMNS),
];

const names = (columns: ColumnInfo[]) => columns.map((column) => column.name);

describe('columns of a preview up to a step', () => {
  it('keeps the columns of later steps from the last preview of all steps', () => {
    const reports = mergeReports(UPTO_S1, 's1', FULL);
    expect(reports.s1).toBe(UPTO_S1[0]);
    expect(reports.s2).toBe(FULL[1]);
    expect(reports.s3).toBe(FULL[2]);
    expect(names(columnsBefore(STEPS, 2, reports, SOURCE_COLUMNS))).toEqual(['years', 'income']);
    expect(names(finalColumns(STEPS, reports, SOURCE_COLUMNS))).toEqual(['years', 'income']);
  });

  it('uses the reports as they are without a preview point or an earlier full preview', () => {
    expect(mergeReports(FULL, null, null)).toEqual(reportsById(FULL));
    expect(mergeReports(UPTO_S1, 's1', null)).toEqual(reportsById(UPTO_S1));
    // A step that was not in the full preview, or had another type then, keeps its report.
    const older = [FULL[0], { ...FULL[2], type: 'impute' }];
    const reports = mergeReports(UPTO_S1, 's1', older);
    expect(reports.s2).toBe(UPTO_S1[1]);
    expect(reports.s3).toBe(UPTO_S1[2]);
  });

  it('passes over steps that did not run when looking for the columns', () => {
    // A step added after the full preview has no older report: the closest earlier step that
    // ran tells the columns, not the columns at the previewed step.
    const added: PipelineStep = { id: 's4', type: 'drop_columns', enabled: true, params: {} };
    const steps = [...STEPS.slice(0, 2), added, STEPS[2]];
    const reports = {
      ...reportsById(FULL.slice(0, 2)),
      s4: report(added, 'not_run', SOURCE_COLUMNS),
    };
    expect(names(columnsBefore(steps, 3, reports, SOURCE_COLUMNS))).toEqual(['years', 'income']);
    // Skipped steps (after a failure) list the columns before them, too.
    const skipped = { ...reports, s4: report(added, 'skipped', SOURCE_COLUMNS) };
    expect(names(columnsBefore(steps, 3, skipped, SOURCE_COLUMNS))).toEqual(['years', 'income']);
    // Reports of another step type (an old step with that id) do not count.
    const other = { s2: { ...FULL[1], type: 'clip' } };
    expect(names(columnsBefore(STEPS, 2, other, SOURCE_COLUMNS))).toEqual(['age', 'income']);
  });
});

describe('insertStep', () => {
  const failed = reportsById([
    report(STEPS[0], 'ok', SOURCE_COLUMNS),
    report(STEPS[1], 'error', SOURCE_COLUMNS),
    report(STEPS[2], 'skipped', SOURCE_COLUMNS),
  ]);
  const fix = { type: 'drop_columns', params: { columns: ['notes'] } };
  const ids = (steps: PipelineStep[]) => steps.map((step) => step.id);

  it('puts a step in front of the first step that failed, where it can run', () => {
    expect(firstFailedStep(STEPS, failed)).toBe(1);
    expect(ids(insertStep(CATALOG, STEPS, fix, null, { reports: failed }))).toEqual([
      's1',
      's4',
      's2',
      's3',
    ]);
    // Also when the preview shows a later step.
    expect(ids(insertStep(CATALOG, STEPS, fix, 's3', { reports: failed }))).toEqual([
      's1',
      's4',
      's2',
      's3',
    ]);
    // Right after the previewed step if that comes first.
    expect(ids(insertStep(CATALOG, STEPS, fix, 's1', { reports: failed }))).toEqual([
      's1',
      's4',
      's2',
      's3',
    ]);
    // Without a failure at the end.
    expect(ids(insertStep(CATALOG, STEPS, fix, null, { reports: reportsById(FULL) }))).toEqual([
      's1',
      's2',
      's3',
      's4',
    ]);
  });

  it('ignores failures of steps that are off or have other reports, and keeps combine steps first', () => {
    const off = STEPS.map((step) => (step.id === 's2' ? { ...step, enabled: false } : step));
    expect(firstFailedStep(off, failed)).toBe(-1);
    const join: PipelineStep = { id: 'j1', type: 'join', enabled: true, params: {} };
    const steps = [join, ...STEPS];
    const joinFailed = { ...failed, j1: { ...report(join, 'error', SOURCE_COLUMNS) } };
    expect(ids(insertStep(CATALOG, steps, fix, null, { reports: joinFailed }))).toEqual([
      'j1',
      's5',
      's1',
      's2',
      's3',
    ]);
  });
});

describe('newStepId', () => {
  it('never reuses the id of a deleted step', () => {
    expect(newStepId(STEPS)).toBe('s4');
    // s3 was deleted, but its report is still shown, or it was used before.
    expect(newStepId(STEPS.slice(0, 2), ['s3'])).toBe('s4');
    expect(newStepId(STEPS.slice(0, 1), ['s1', 's2', 's7'])).toBe('s8');
    // Ids made elsewhere ("j1") count as steps, not as numbers.
    expect(newStepId([{ ...STEPS[0], id: 'j1' }])).toBe('s2');
  });
});
