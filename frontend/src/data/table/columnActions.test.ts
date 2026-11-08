import { PROFILE } from '../fixtures';
import { castStep, columnStepActions, renameError, renameStep } from './columnActions';

const profile = (name: string) => PROFILE.columns.find((column) => column.name === name);

describe('columnStepActions', () => {
  it('offers numeric actions with prefilled params', () => {
    const actions = columnStepActions({ name: 'age', kind: 'numeric' }, profile('age'), 120);
    expect(actions.map((action) => action.key)).toEqual(['impute', 'clip', 'log', 'drop']);
    expect(actions[0].step).toEqual({
      type: 'impute',
      params: { columns: ['age'], strategy: 'median' },
    });
    expect(actions[1].step).toEqual({
      type: 'clip',
      params: { columns: ['age'], mode: 'quantiles', lower_q: 0.01, upper_q: 0.99 },
    });
    expect(actions[2].step).toEqual({
      type: 'transform',
      params: { columns: ['age'], function: 'log1p' },
    });
  });

  it('uses Yeo-Johnson for columns with negative values', () => {
    const base = profile('age')!;
    const negative = { ...base, stats: { ...base.stats!, min: -5 } };
    const log = columnStepActions({ name: 'age', kind: 'numeric' }, negative, 0).find(
      (action) => action.key === 'log'
    );
    expect(log?.step.params.function).toBe('yeo_johnson');
  });

  it('fills categories with the most frequent value and offers one-hot encoding', () => {
    const actions = columnStepActions({ name: 'city', kind: 'categorical' }, profile('city'), 96);
    expect(actions.map((action) => action.key)).toEqual(['impute', 'onehot', 'drop']);
    expect(actions[0].step.params.strategy).toBe('most_frequent');
    expect(actions[1].step).toEqual({
      type: 'encode',
      params: { columns: ['city'], method: 'onehot' },
    });
  });

  it('offers date parts for dates and nothing to fill for complete columns', () => {
    const actions = columnStepActions({ name: 'signup', kind: 'datetime' }, profile('signup'), 0);
    expect(actions.map((action) => action.key)).toEqual(['date_parts', 'drop']);
    expect(actions[0].step).toEqual({ type: 'datetime_features', params: { columns: ['signup'] } });
  });
});

describe('columnStepActions for the target', () => {
  it('offers no encoding, filling or dropping, only dropping rows without a target', () => {
    const churn = { name: 'churn', kind: 'categorical' as const };
    const actions = columnStepActions(churn, profile('churn'), 12, true);
    expect(actions).toEqual([
      {
        key: 'drop_rows',
        label: 'Drop rows without a target value',
        step: { type: 'drop_missing', params: { columns: ['churn'], how: 'any' } },
      },
    ]);
    expect(columnStepActions(churn, profile('churn'), 0, true)).toEqual([]);
    const age = columnStepActions({ name: 'age', kind: 'numeric' }, profile('age'), 120, true);
    expect(age.map((action) => action.key)).toEqual(['drop_rows']);
  });
});

describe('rename and cast', () => {
  it('builds the steps', () => {
    expect(renameStep('age', ' years ')).toEqual({
      type: 'rename_columns',
      params: { mapping: { age: 'years' } },
    });
    expect(castStep('plan', 'numeric')).toEqual({
      type: 'cast',
      params: { columns: ['plan'], to: 'numeric' },
    });
  });

  it('checks new names', () => {
    expect(renameError('age', '', ['age', 'city'])).toBe('Enter a name.');
    expect(renameError('age', 'age', ['age', 'city'])).toBe('That is the current name.');
    expect(renameError('age', 'city', ['age', 'city'])).toBe('There is already a column “city”.');
    expect(renameError('age', 'years', ['age', 'city'])).toBeNull();
  });
});
