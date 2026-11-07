import { nextSort, sortOrder } from './sort';
import { columnOffsets, columnRange, rowRange } from './virtual';

describe('rowRange', () => {
  it('covers the viewport plus overscan, clamped to the rows', () => {
    expect(rowRange(0, 300, 30, 5000, 5)).toEqual({ start: 0, end: 15 });
    expect(rowRange(3000, 300, 30, 5000, 5)).toEqual({ start: 95, end: 115 });
    expect(rowRange(149_900, 300, 30, 5000, 5)).toEqual({ start: 4991, end: 5000 });
    expect(rowRange(0, 300, 30, 0, 5)).toEqual({ start: 0, end: 0 });
  });
});

describe('columnRange', () => {
  const offsets = columnOffsets([100, 200, 100, 300, 100]);

  it('builds prefix sums', () => {
    expect(offsets).toEqual([0, 100, 300, 400, 700, 800]);
  });

  it('finds the columns that intersect the viewport', () => {
    expect(columnRange(offsets, 0, 250, 0)).toEqual({ start: 0, end: 2 });
    expect(columnRange(offsets, 350, 100, 0)).toEqual({ start: 2, end: 4 });
    expect(columnRange(offsets, 350, 100, 1)).toEqual({ start: 1, end: 5 });
    expect(columnRange(offsets, 0, 5000, 2)).toEqual({ start: 0, end: 5 });
    expect(columnRange([0], 0, 100, 1)).toEqual({ start: 0, end: 0 });
  });
});

describe('sortOrder', () => {
  it('sorts numbers and text, missing values last in both directions', () => {
    const values = [3, null, 1, 2, null];
    expect(sortOrder(values, 'ascending')).toEqual([2, 3, 0, 1, 4]);
    expect(sortOrder(values, 'descending')).toEqual([0, 3, 2, 1, 4]);
    expect(sortOrder(['item 10', 'item 2', 'Apple'], 'ascending')).toEqual([2, 1, 0]);
  });

  it('cycles through the sort states', () => {
    expect(nextSort(null)).toBe('ascending');
    expect(nextSort('ascending')).toBe('descending');
    expect(nextSort('descending')).toBeNull();
  });
});
