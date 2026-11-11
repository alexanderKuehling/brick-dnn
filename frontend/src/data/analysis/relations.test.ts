import type { DataProfile, DataSample } from '@/api/types';
import { PROFILE, RELATIONS } from '@/data/fixtures';
import {
  clusterOrder,
  countBelow,
  filterRows,
  parseColumnList,
  relationRows,
  sampleMatrix,
  sortRows,
  strongestPairColumns,
} from './relations';

const profile: DataProfile = {
  ...PROFILE,
  n_rows: 100,
  columns: [
    {
      ...PROFILE.columns[0],
      name: 'a',
      missing: 10,
      target_association: { method: 'pearson', value: 0.4 },
    },
    { ...PROFILE.columns[0], name: 'noise1', missing: 0, target_association: null },
  ],
};

describe('relation rows', () => {
  const rows = relationRows(RELATIONS, profile);

  it('describe every analysed column by its strongest partner', () => {
    expect(rows.map((r) => r.name)).toEqual(RELATIONS.columns);
    expect(rows[0]).toMatchObject({
      partner: 'b',
      rho: 0.9,
      strength: 0.9,
      r: 0.88,
      partners: 2,
      group: 0,
      target: 0.4,
      missing: 0.1,
    });
    expect(rows[2].strength).toBe(0.6);
    expect(rows[3]).toMatchObject({ partner: 'c', target: null, missing: 0, group: -1 });
  });

  it('filters by name, threshold, partner count and group', () => {
    const base = {
      query: '',
      strength: 'all' as const,
      threshold: 0.15,
      minPartners: null,
      group: null,
    };
    expect(filterRows(rows, { ...base, strength: 'below' }).map((r) => r.name)).toEqual([
      'noise1',
      'noise2',
    ]);
    expect(filterRows(rows, { ...base, strength: 'above' })).toHaveLength(3);
    expect(filterRows(rows, { ...base, query: 'NOI' })).toHaveLength(2);
    expect(filterRows(rows, { ...base, minPartners: 1 }).map((r) => r.name)).toEqual([
      'a',
      'b',
      'c',
    ]);
    expect(filterRows(rows, { ...base, group: -1 })).toHaveLength(2);
  });

  it('sorts by strongest partner (noise first), and in the clustered order', () => {
    expect(sortRows(rows, 'strength', false, RELATIONS.order).map((r) => r.name)).toEqual([
      'noise1',
      'noise2',
      'c',
      'a',
      'b',
    ]);
    expect(sortRows(rows, 'order', false, RELATIONS.order).map((r) => r.name)).toEqual([
      'noise1',
      'a',
      'b',
      'c',
      'noise2',
    ]);
    // no target association: last in both directions
    const byTarget = sortRows(rows, 'target', true, null).map((r) => r.name);
    expect(byTarget[0]).toBe('a');
  });

  it('counts the columns below a threshold', () => {
    expect(countBelow(RELATIONS, 0.15)).toBe(2);
    expect(countBelow(RELATIONS, 0.07)).toBe(1);
    expect(countBelow(RELATIONS, 0.95)).toBe(5);
  });

  it('chooses the columns of the strongest pairs', () => {
    expect(strongestPairColumns(RELATIONS, 3)).toEqual(['a', 'b', 'c']);
  });
});

describe('matrices from the sample', () => {
  const sample: DataSample = {
    n_total: 6,
    rows: 6,
    seed: 0,
    index: [0, 1, 2, 3, 4, 5],
    split: ['train', 'train', 'train', 'train', 'val', 'train'],
    columns: [
      { name: 'x', kind: 'numeric', values: [1, 2, 3, 4, 100, 5] },
      { name: 'y', kind: 'numeric', values: [2, 4, 6, 8, -100, 100] },
      { name: 'z', kind: 'numeric', values: [5, 4, 3, null, 1, 1] },
      { name: 'k', kind: 'numeric', values: [1, 1, 1, 1, 1, 1] },
    ],
  };

  it('uses the training rows and pairs where both have a value', () => {
    const spearman = sampleMatrix(sample, ['x', 'y', 'z', 'k'], 'spearman');
    expect(spearman.rows).toBe(5);
    expect(spearman.trainingRows).toBe(true);
    expect(spearman.values[0][1]).toBe(1); // monotonic
    expect(spearman.values[0][2]).toBe(-1);
    expect(spearman.values[3][0]).toBeNull(); // constant
    const pearson = sampleMatrix(sample, ['x', 'y'], 'pearson');
    expect(pearson.values[0][1]).toBeLessThan(0.9);
  });

  it('orders related columns next to each other', () => {
    const values = [
      [1, 0.9, 0, 0],
      [0.9, 1, 0, 0.1],
      [0, 0, 1, 0.8],
      [0, 0.1, 0.8, 1],
    ];
    const order = clusterOrder(values);
    expect(order).toHaveLength(4);
    const at = (i: number) => order.indexOf(i);
    expect(Math.abs(at(0) - at(1))).toBe(1);
    expect(Math.abs(at(2) - at(3))).toBe(1);
  });
});

describe('pasted column lists', () => {
  const available = ['age', 'income', 'home city', 'x1'];

  it('splits by commas, spaces and new lines, and keeps names with spaces', () => {
    expect(parseColumnList('age, income\nx1', available)).toEqual({
      found: ['age', 'income', 'x1'],
      unknown: [],
    });
    expect(parseColumnList("['age', 'home city']", available).found).toEqual(['age', 'home city']);
    expect(parseColumnList('age income x2 age', available)).toEqual({
      found: ['age', 'income'],
      unknown: ['x2'],
    });
    expect(parseColumnList('"x1";\tage', available).found).toEqual(['x1', 'age']);
  });
});
