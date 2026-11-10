import type { PipelineStep } from '@/api/types';
import { CATALOG, STEP_REPORTS } from '../fixtures';
import { insertStep } from '../steps';
import { displayedSplit, percentToShare, shareToPercent, splitError, splitRows } from './split';
import { canMoveStep, duplicateStep, moveStep, restoreStep, stepStatus } from './stepList';

const step = (id: string, type: string): PipelineStep => ({
  id,
  type,
  enabled: true,
  params: { columns: [id] },
});

const STEPS = [step('j1', 'join'), step('j2', 'join'), step('a', 'impute'), step('b', 'clip')];

describe('moving steps', () => {
  it('keeps combine steps in front', () => {
    expect(canMoveStep(CATALOG, STEPS, 1, 1)).toBe(false); // join below impute
    expect(canMoveStep(CATALOG, STEPS, 2, -1)).toBe(false); // impute above join
    expect(canMoveStep(CATALOG, STEPS, 1, -1)).toBe(true);
    expect(canMoveStep(CATALOG, STEPS, 2, 1)).toBe(true);
    expect(canMoveStep(CATALOG, STEPS, 3, 1)).toBe(false);
    expect(canMoveStep(CATALOG, STEPS, 0, -1)).toBe(false);
  });

  it('swaps neighbours', () => {
    expect(moveStep(CATALOG, STEPS, 2, 1).map((s) => s.id)).toEqual(['j1', 'j2', 'b', 'a']);
    expect(moveStep(CATALOG, STEPS, 1, -1).map((s) => s.id)).toEqual(['j2', 'j1', 'a', 'b']);
    expect(moveStep(CATALOG, STEPS, 2, -1)).toBe(STEPS);
  });

  it('treats unknown step types like normal steps', () => {
    const steps = [step('x', 'no_such_step'), step('a', 'impute')];
    expect(canMoveStep(CATALOG, steps, 0, 1)).toBe(true);
  });
});

describe('duplicateStep', () => {
  it('inserts an independent copy after the original', () => {
    const next = duplicateStep(STEPS, 2);
    expect(next.map((s) => s.id)).toEqual(['j1', 'j2', 'a', 's5', 'b']);
    expect(next[3].params).toEqual(STEPS[2].params);
    expect(next[3].params).not.toBe(STEPS[2].params);
  });
});

describe('restoreStep', () => {
  it('puts a step back at its place without breaking the combine order', () => {
    const without = STEPS.filter((s) => s.id !== 'a');
    expect(restoreStep(CATALOG, without, STEPS[2], 2).map((s) => s.id)).toEqual([
      'j1',
      'j2',
      'a',
      'b',
    ]);
    // Normal steps never go in front of combine steps, combine steps never behind normal ones.
    expect(restoreStep(CATALOG, without, STEPS[2], 0).map((s) => s.id)).toEqual([
      'j1',
      'j2',
      'a',
      'b',
    ]);
    const noJoin = STEPS.filter((s) => s.id !== 'j1');
    expect(restoreStep(CATALOG, noJoin, STEPS[0], 3).map((s) => s.id)).toEqual([
      'j2',
      'j1',
      'a',
      'b',
    ]);
  });
});

describe('stepStatus', () => {
  it('prefers the step switch and ignores reports of other step types', () => {
    const impute = { id: 's1', type: 'impute', enabled: true, params: {} };
    expect(stepStatus(impute, STEP_REPORTS[0])).toBe('ok');
    expect(stepStatus({ ...impute, enabled: false }, STEP_REPORTS[0])).toBe('disabled');
    expect(stepStatus(impute, undefined)).toBe('waiting');
    expect(stepStatus({ ...impute, type: 'clip' }, STEP_REPORTS[0])).toBe('waiting');
    expect(stepStatus(impute, { ...STEP_REPORTS[0], status: 'disabled' })).toBe('waiting');
  });
});

describe('split', () => {
  it('converts shares and percent without float noise', () => {
    expect(shareToPercent(0.15)).toBe(15);
    expect(shareToPercent(0.7000000000000001)).toBe(70);
    expect(percentToShare(15)).toBe(0.15);
    expect(percentToShare(12.5)).toBe(0.125);
  });

  it('validates the shares', () => {
    expect(splitError(0.15, 0.15)).toBeNull();
    expect(splitError(0.5, 0.4)).toBeNull();
    expect(splitError(0, 0.15)).toEqual({ val: 'Use more than 0 % for validation.' });
    expect(splitError(0.5, 0.5)).toEqual({
      test: 'Validation and test together can be at most 90 %.',
    });
  });

  it('shows preview counts while they match, else estimates', () => {
    expect(splitRows(1200, 0.15, 0.15)).toEqual([840, 180, 180]);
    expect(displayedSplit([840, 180, 180], 0.15, 0.15)).toEqual({
      counts: [840, 180, 180],
      estimated: false,
    });
    expect(displayedSplit([840, 180, 180], 0.2, 0.1)).toEqual({
      counts: [840, 240, 120],
      estimated: true,
    });
    expect(displayedSplit(null, 0.2, 0.1)).toBeNull();
  });
});

describe('steps that run before the split', () => {
  const groups = { ...CATALOG.steps[0], type: 'find_groups', combine: false, pre_split: true };
  const catalog = { ...CATALOG, steps: [...CATALOG.steps, groups] };
  const steps = [step('j1', 'join'), step('g', 'find_groups'), step('a', 'impute')];

  it('stay between the combine steps and the others', () => {
    expect(canMoveStep(catalog, steps, 1, -1)).toBe(false);
    expect(canMoveStep(catalog, steps, 1, 1)).toBe(false);
    expect(canMoveStep(catalog, steps, 2, -1)).toBe(false);
    const next = insertStep(catalog, steps, { type: 'find_groups', params: {} });
    expect(next.map((s) => s.type)).toEqual(['join', 'find_groups', 'find_groups', 'impute']);
    const after = insertStep(catalog, [step('g', 'find_groups')], { type: 'impute', params: {} });
    expect(after.map((s) => s.type)).toEqual(['find_groups', 'impute']);
    const back = restoreStep(catalog, [step('j1', 'join'), step('a', 'impute')], steps[1], 5);
    expect(back.map((s) => s.id)).toEqual(['j1', 'g', 'a']);
  });
});
