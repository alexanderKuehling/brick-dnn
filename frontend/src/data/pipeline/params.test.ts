import type { StepReport } from '@/api/types';
import { CATALOG, OTHER_SOURCE, STEP_REPORTS } from '../fixtures';
import { stepSpec } from '../steps';
import {
  allowedColumns,
  columnsProblem,
  errorField,
  formatCellValue,
  isParamVisible,
  paramSummary,
  paramValue,
  parseCellValue,
} from './params';

const spec = (type: string) => stepSpec(CATALOG, type)!;
const param = (type: string, name: string) => spec(type).params.find((p) => p.name === name)!;

const COLUMNS = [
  { name: 'age', kind: 'numeric' as const },
  { name: 'city', kind: 'categorical' as const },
  { name: 'signup', kind: 'datetime' as const },
];

describe('paramValue', () => {
  it('falls back to the default', () => {
    expect(paramValue(param('impute', 'strategy'), {})).toBe('median');
    expect(paramValue(param('impute', 'strategy'), { strategy: 'mean' })).toBe('mean');
    expect(paramValue(param('impute', 'fill_value'), { fill_value: null })).toBeNull();
  });
});

describe('isParamVisible', () => {
  const filter = spec('filter_rows');
  const value2 = param('filter_rows', 'value2');

  it('compares the controlling param, using its default when unset', () => {
    // Default operator ">=": value is shown, value2 only for "between".
    expect(isParamVisible(param('filter_rows', 'value'), {}, filter)).toBe(true);
    expect(isParamVisible(value2, {}, filter)).toBe(false);
    expect(isParamVisible(value2, { operator: 'between' }, filter)).toBe(true);
    expect(isParamVisible(param('filter_rows', 'column'), {}, filter)).toBe(true);
  });

  it('works with boolean conditions', () => {
    const append = {
      ...filter,
      params: [
        { ...value2, name: 'add', kind: 'bool' as const, default: false, visible_if: undefined },
        { ...value2, name: 'name', visible_if: { param: 'add', values: [true] } },
      ],
    };
    expect(isParamVisible(append.params[1], {}, append)).toBe(false);
    expect(isParamVisible(append.params[1], { add: true }, append)).toBe(true);
  });
});

describe('parseCellValue / formatCellValue', () => {
  it('sends numbers as numbers and text as text', () => {
    expect(parseCellValue('')).toBeNull();
    expect(parseCellValue('  ')).toBeNull();
    expect(parseCellValue('42')).toBe(42);
    expect(parseCellValue(' -0.5 ')).toBe(-0.5);
    expect(parseCellValue('1e3')).toBe(1000);
    expect(parseCellValue('.5')).toBe(0.5);
    expect(parseCellValue('unknown')).toBe('unknown');
    expect(parseCellValue('1,5')).toBe('1,5');
    expect(parseCellValue('2021-01-01')).toBe('2021-01-01');
    expect(parseCellValue('"007"')).toBe('007');
  });

  it('round-trips values', () => {
    for (const value of [42, -0.5, 'unknown', '007', ' padded ', '"quoted"', true]) {
      const text = formatCellValue(value);
      expect(parseCellValue(text)).toEqual(value === true ? 'true' : value);
    }
    expect(formatCellValue(null)).toBe('');
    expect(formatCellValue('007')).toBe('"007"');
  });
});

describe('columns', () => {
  it('filters by column kinds', () => {
    expect(allowedColumns(param('clip', 'columns'), COLUMNS).map((c) => c.name)).toEqual(['age']);
    expect(allowedColumns(param('drop_columns', 'columns'), COLUMNS)).toHaveLength(3);
  });

  it('explains columns that cannot be used', () => {
    const clipColumns = param('clip', 'columns');
    expect(columnsProblem(clipColumns, ['age'], COLUMNS)).toBeNull();
    expect(columnsProblem(clipColumns, ['city'], COLUMNS)).toBe(
      '“city” is a categorical column; this needs numeric columns.'
    );
    expect(columnsProblem(clipColumns, ['gone'], COLUMNS)).toBe(
      '“gone” does not exist at this step.'
    );
    expect(columnsProblem(clipColumns, ['a', 'b'], COLUMNS)).toBe(
      '“a”, “b” do not exist at this step.'
    );
  });
});

describe('paramSummary', () => {
  it('lists the choices of the visible params', () => {
    expect(paramSummary(spec('impute'), { columns: [], strategy: 'median' })).toBe(
      'All columns with missing values · Median'
    );
    expect(paramSummary(spec('clip'), { columns: ['age', 'income', 'x', 'y'] })).toBe(
      'age, income, x +1 · Quantiles'
    );
    expect(
      paramSummary(spec('join'), { source: 2, how: 'inner', left_on: ['city'] }, [OTHER_SOURCE])
    ).toBe('regions · Only matching rows (inner join) · city');
    expect(paramSummary(spec('rename_columns'), { mapping: { a: 'b' } })).toBe('a → b');
    expect(paramSummary(spec('datetime_features'), {})).toBe(
      'All date columns · Year, Month, Weekday'
    );
  });
});

describe('errorField', () => {
  it('finds the field named at the start of a step error', () => {
    expect(errorField(spec('clip'), STEP_REPORTS[1])).toEqual({
      param: 'columns',
      message: "'city' is a categorical column; numeric columns needed.",
    });
    const other: StepReport = { ...STEP_REPORTS[1], error: 'Something else broke.' };
    expect(errorField(spec('clip'), other)).toBeNull();
    expect(errorField(spec('impute'), STEP_REPORTS[0])).toBeNull();
  });
});
