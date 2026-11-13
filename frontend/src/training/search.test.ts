import type { TrainingPlan } from '@/api/types';
import {
  availableMetrics,
  CLASSICAL_TUNABLES,
  classicalCombinations,
  classicalSpace,
  formatMetricValue,
  metricDirection,
  metricInSentence,
  paramStates,
  parseList,
  planMetrics,
  rangeError,
  type ParamState,
} from './search';

const names = (list: { name: string }[] | null) => list?.map((metric) => metric.name);

function plan(targets: TrainingPlan['targets'], numClasses: number | null): TrainingPlan {
  return {
    task: 'classification',
    summary: '',
    inputs: [],
    targets,
    image: null,
    num_classes: numClasses,
    class_names: null,
    notes: [],
  };
}

const tunable = (model: string, name: string) =>
  CLASSICAL_TUNABLES[model].find((item) => item.name === name)!;

describe('objective metrics', () => {
  it('fit the task and the number of classes', () => {
    expect(names(availableMetrics('classification', 2))).toEqual([
      'loss',
      'accuracy',
      'balanced_accuracy',
      'macro_f1',
      'roc_auc',
      'pr_auc',
    ]);
    expect(names(availableMetrics('classification', 3))).not.toContain('roc_auc');
    expect(names(availableMetrics('regression', null))).toEqual(['loss', 'rmse', 'mae', 'r2']);
    expect(names(availableMetrics('segmentation', 2))).toEqual(['loss']);
    expect(names(planMetrics(plan([{ kind: 'class', num_classes: 2 }], 2)))).toContain('pr_auc');
    expect(names(planMetrics(plan([{ kind: 'value', shape: [1] }], null)))).toContain('mae');
    expect(names(planMetrics(plan([{ kind: 'value', shape: [3] }], null)))).toEqual(['loss']);
    expect(planMetrics(null)).toBeNull();
  });

  it('say their direction and format their values', () => {
    expect(metricDirection('loss')).toBe('lower is better');
    expect(metricDirection('r2')).toBe('higher is better');
    expect(metricInSentence('balanced_accuracy')).toBe('balanced accuracy');
    expect(metricInSentence('roc_auc')).toBe('ROC AUC');
    expect(metricInSentence('loss')).toBe('the loss');
    expect(formatMetricValue('accuracy', 0.9312)).toBe('93.1%');
    expect(formatMetricValue('roc_auc', 0.91234)).toBe('0.9123');
    expect(formatMetricValue('mae', null)).toBe('–');
  });
});

describe('classical search space', () => {
  it('starts from the defaults of each model', () => {
    const states = paramStates('SVC', undefined);
    expect(Object.keys(states)).toEqual(['C', 'gamma', 'kernel', 'class_weight']);
    expect(states.C).toMatchObject({ on: true, mode: 'range', min: 0.1, max: 100, scale: 'log' });
    expect(states.kernel).toMatchObject({ on: false, mode: 'list', choices: ['rbf', 'linear'] });
    expect(classicalSpace('SVC', states, false)).toEqual({
      C: { min: 0.1, max: 100 },
      gamma: { min: 0.001, max: 1 },
    });
    // A saved choice overrides the default.
    const saved = paramStates('SVC', { kernel: { on: true } });
    expect(classicalSpace('SVC', saved, false).kernel).toEqual(['linear', 'rbf']);
  });

  it('reads lists of numbers and names', () => {
    const state = (text: string): ParamState => ({
      ...paramStates('HistGradientBoostingClassifier', undefined).max_depth,
      text,
    });
    const depth = tunable('HistGradientBoostingClassifier', 'max_depth');
    expect(parseList(depth, state('none, 3, 6, 3'))).toEqual([null, 3, 6]);
    expect(parseList(depth, state('2.5'))).toBe('2.5: whole numbers');
    expect(parseList(depth, state(''))).toBe('Enter at least one value');
    const gamma = tunable('SVC', 'gamma');
    expect(parseList(gamma, { ...state('scale, 0.1'), text: 'scale, 0.1' })).toEqual([
      'scale',
      0.1,
    ]);
    expect(parseList(gamma, state('0'))).toBe('0: greater than 0, at most 10,000');
  });

  it('checks ranges', () => {
    const states = paramStates('SVR', undefined);
    const epsilon = tunable('SVR', 'epsilon');
    expect(rangeError(epsilon, states.epsilon)).toBeNull();
    expect(rangeError(epsilon, { ...states.epsilon, min: 0 })).toMatch(/log range needs a minimum/);
    expect(rangeError(epsilon, { ...states.epsilon, min: 0, scale: 'linear' })).toBeNull();
    expect(rangeError(epsilon, { ...states.epsilon, min: 2 })).toMatch(/larger than the maximum/);
    expect(rangeError(epsilon, { ...states.epsilon, max: '' })).toBe('Enter both ends');
    const linear = { ...states.epsilon, on: true, min: 0, scale: 'linear' as const };
    expect(classicalSpace('SVR', { ...states, epsilon: linear }, false).epsilon).toEqual({
      min: 0,
      max: 1,
      scale: 'linear',
    });
  });

  it('counts the combinations of a grid', () => {
    const states = paramStates('SVC', {
      C: { mode: 'list', text: '0.1, 1, 10' },
      gamma: { on: false },
      kernel: { on: true },
    });
    expect(classicalCombinations('SVC', states, true)).toBe(6);
    expect(classicalSpace('SVC', states, true)).toEqual({
      C: [0.1, 1, 10],
      kernel: ['linear', 'rbf'],
    });
    // A random search with a range has no limit.
    expect(classicalCombinations('SVC', paramStates('SVC', undefined), false)).toBeNull();
  });
});
