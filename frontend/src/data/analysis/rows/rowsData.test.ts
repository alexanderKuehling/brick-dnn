import { rowAnalysis } from './fixtures';
import {
  closePairs,
  countBelow,
  dropRowsStep,
  filterRows,
  freeName,
  histogramCounts,
  histogramEdges,
  median,
  pairTargets,
  randomPairs,
  sortRows,
} from './rowsData';

const thresholds = { atypical: 0.25, twins: 0.32 };

describe('rowsData', () => {
  it('counts rows below a threshold and bins values', () => {
    expect(countBelow([0.1, null, 0.3, 0.5], 0.4)).toBe(2);
    const edges = histogramEdges([0.2, 0.9, null], 4, 2);
    expect(edges).toEqual([0, 0.25, 0.5, 0.75, 1]);
    expect(histogramCounts([0.2, 0.9, 5, null], edges)).toEqual([1, 0, 0, 2]);
    expect(median([3, 1, 2])).toBe(2);
    expect(median([])).toBeNull();
  });

  it('finds close pairs once and compares their targets', () => {
    const analysis = rowAnalysis();
    const pairs = closePairs(analysis, 0.32);
    expect(pairs).toEqual([
      [0, 1],
      [2, 3],
      [4, 5],
    ]);
    expect(pairTargets(analysis, pairs).diffs).toEqual([1, 1.5, 28]);
    expect(pairTargets(analysis, randomPairs(analysis)).known).toBe(5);
  });

  it('filters, searches and sorts the rows', () => {
    const analysis = rowAnalysis();
    expect(filterRows(analysis, 'atypical', thresholds)).toEqual([4, 5]);
    expect(filterRows(analysis, 'twins', thresholds)).toEqual([0, 1, 2, 3, 4, 5]);
    expect(filterRows(analysis, 'group', thresholds, { group: 1 })).toEqual([2, 3]);
    expect(filterRows(analysis, 'selected', thresholds, { selected: new Set([7]) })).toEqual([7]);
    expect(filterRows(analysis, 'all', thresholds, { search: '105' })).toEqual([5]);
    const all = filterRows(analysis, 'all', thresholds);
    expect(sortRows(analysis, all, { key: 'spread', descending: false }).slice(0, 2)).toEqual([
      4, 5,
    ]);
    expect(sortRows(analysis, all, { key: 'target', descending: true })[0]).toBe(7);
  });

  it('drops rows by id, or by row number without ids', () => {
    expect(dropRowsStep(rowAnalysis(), [4, 5])).toEqual({
      type: 'drop_rows',
      params: { by: 'column', column: 'id', values: [104, 105] },
    });
    const noIds = rowAnalysis({ id_column: null });
    noIds.rows = { ...noIds.rows, id: null };
    expect(dropRowsStep(noIds, [4]).params).toEqual({ by: 'position', positions: [4] });
    expect(freeName(['a', 'group', 'group_2'])).toBe('group_3');
  });
});
