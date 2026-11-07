import type { DataProfile, DataSample } from '@/api/types';
import { PROFILE, SAMPLE } from '@/data/fixtures';
import { needsZoom } from './DistributionView';
import {
  calendarCounts,
  classGroups,
  classKey,
  countValues,
  crossTab,
  numericPairs,
  rowName,
  sampleColumn,
  sampleNote,
  valueLabel,
  valuesByGroup,
} from './sampleData';

describe('sample helpers', () => {
  it('groups the sample rows by target class in class order', () => {
    const groups = classGroups(SAMPLE, PROFILE, 'churn')!;
    expect(groups.groups.map((g) => [g.key, g.slot])).toEqual([
      ['no', 0],
      ['yes', 1],
    ]);
    expect(groups.groups[0].count + groups.groups[1].count).toBe(SAMPLE.rows);
    const income = valuesByGroup(sampleColumn(SAMPLE, 'income')!, groups);
    expect(income[0].length + income[1].length).toBe(
      SAMPLE.columns[2].values.filter((v) => v !== null).length
    );
    expect(classGroups(SAMPLE, PROFILE, null)).toBeNull();
  });

  it('folds classes beyond the palette into "Other" and matches numeric labels', () => {
    const labels = Array.from({ length: 12 }, (_, i) => i);
    const sample: DataSample = {
      ...SAMPLE,
      rows: 24,
      columns: [{ name: 'digit', kind: 'numeric', values: [...labels, ...labels] }],
    };
    const profile = {
      ...PROFILE,
      target: {
        column: 'digit',
        task: 'classification' as const,
        classes: labels.map((i) => ({ value: `${i}.0`, count: 2 })),
        imbalance_ratio: 1,
        histogram: null,
        stats: null,
      },
    };
    const groups = classGroups(sample, profile, 'digit')!;
    expect(groups.groups).toHaveLength(8);
    expect(groups.groups[7]).toMatchObject({ label: 'Other (5 classes)', slot: -1, count: 10 });
    expect(groups.index[3]).toBe(3);
    expect(groups.index[11]).toBe(7);
  });

  it('matches yes/no targets to the class names "false"/"true" (and the old "0"/"1")', () => {
    const values = [0, 1, 1, null, 0, 1];
    const sample: DataSample = {
      ...SAMPLE,
      rows: values.length,
      columns: [{ name: 'flag', kind: 'boolean', values }],
    };
    const profileWith = (names: string[]): DataProfile => ({
      ...PROFILE,
      target: {
        column: 'flag',
        task: 'classification',
        classes: names.map((value) => ({ value, count: 3 })),
        imbalance_ratio: 1,
        histogram: null,
        stats: null,
      },
    });
    for (const names of [
      ['false', 'true'],
      ['0', '1'],
    ]) {
      const groups = classGroups(sample, profileWith(names), 'flag')!;
      expect(groups.groups.map((g) => [g.key, g.label, g.slot, g.count])).toEqual([
        ['false', 'false', 0, 2],
        ['true', 'true', 1, 3],
      ]);
      expect(Array.from(groups.index)).toEqual([0, 1, 1, -1, 0, 1]);
    }
  });

  it('matches dates and numbers to the class names of the backend', () => {
    const dates = [Date.UTC(2021, 0, 1), Date.UTC(2021, 0, 1, 12, 30), Date.UTC(2021, 0, 1)];
    const numbers = [3, 2.5, 3, 1e-7];
    const sample: DataSample = {
      ...SAMPLE,
      rows: 4,
      columns: [
        { name: 'day', kind: 'datetime', values: [...dates, null] },
        { name: 'grade', kind: 'numeric', values: numbers },
      ],
    };
    const target = (column: string, names: string[]): DataProfile => ({
      ...PROFILE,
      target: {
        column,
        task: 'classification',
        classes: names.map((value) => ({ value, count: 1 })),
        imbalance_ratio: 1,
        histogram: null,
        stats: null,
      },
    });
    const days = classGroups(sample, target('day', ['2021-01-01', '2021-01-01 12:30:00']), 'day')!;
    expect(days.groups.map((g) => [g.label, g.count])).toEqual([
      ['2021-01-01', 2],
      ['2021-01-01 12:30:00', 1],
    ]);
    expect(Array.from(days.index)).toEqual([0, 1, 0, -1]);
    const grades = classGroups(sample, target('grade', ['1e-07', '2.5', '3']), 'grade')!;
    expect(grades.groups.map((g) => [g.label, g.count])).toEqual([
      ['1e-07', 1],
      ['2.5', 1],
      ['3', 2],
    ]);
    // Without a class list the sample decides, named like the backend names classes.
    const named = classGroups(sample, { ...PROFILE, target: null }, 'grade')!;
    expect(named.groups.map((g) => g.label)).toEqual(['3', '2.5', '1e-07']);
    expect(classKey(Date.UTC(2021, 0, 1, 0, 0, 0, 500), 'datetime')).toBe('2021-01-01');
    expect(classKey(1e16, 'numeric')).toBe('1e+16');
    expect(classKey(1e15, 'numeric')).toBe('1000000000000000.0');
    expect(classKey(-0, 'numeric')).toBe('0');
    expect(classKey(0.0001, 'numeric')).toBe('0.0001');
    expect(classKey(1, 'boolean')).toBe('true');
  });

  it('groups many classes quickly and only once per sample and profile', () => {
    const rows = 20_000;
    const values = Array.from({ length: rows }, (_, i) => `class ${i}`);
    const sample: DataSample = {
      ...SAMPLE,
      rows,
      columns: [{ name: 'label', kind: 'categorical', values }],
    };
    const profile: DataProfile = {
      ...PROFILE,
      target: {
        column: 'label',
        task: 'classification',
        classes: values.map((value) => ({ value, count: 1 })),
        imbalance_ratio: 1,
        histogram: null,
        stats: null,
      },
    };
    const start = performance.now();
    const groups = classGroups(sample, profile, 'label')!;
    expect(performance.now() - start).toBeLessThan(250);
    expect(groups.groups.at(-1)).toMatchObject({ slot: -1, count: rows - 7 });
    expect(classGroups(sample, profile, 'label')).toBe(groups);
    expect(classGroups({ ...sample }, profile, 'label')).not.toBe(groups);
  });

  it('counts values, cross tables and calendar parts', () => {
    const city = sampleColumn(SAMPLE, 'city')!;
    const { counts, missing } = countValues(city);
    expect(counts.reduce((sum, c) => sum + c.count, 0) + missing).toBe(SAMPLE.rows);
    const groups = classGroups(SAMPLE, PROFILE, 'churn')!;
    const table = crossTab(city, groups, ['Berlin']);
    const total =
      table.rows[0].reduce((a, b) => a + b, 0) +
      table.other.reduce((a, b) => a + b, 0) +
      table.missing.reduce((a, b) => a + b, 0);
    expect(total).toBe(SAMPLE.rows);
    // 2021-01-04 was a Monday.
    expect(calendarCounts([Date.UTC(2021, 0, 4), Date.UTC(2021, 0, 10)]).weekday).toEqual([
      1, 0, 0, 0, 0, 0, 1,
    ]);
  });

  it('pairs rows where both columns have a number', () => {
    const pairs = numericPairs(sampleColumn(SAMPLE, 'age')!, sampleColumn(SAMPLE, 'income')!);
    expect(pairs.x.length).toBe(pairs.rows.length);
    expect(Array.from(pairs.x).every(Number.isFinite)).toBe(true);
    expect(pairs.x.length).toBeLessThan(SAMPLE.rows);
  });

  it('labels values and samples honestly', () => {
    expect(valueLabel('1', 'boolean')).toBe('true');
    expect(valueLabel('', 'categorical')).toBe('(empty)');
    expect(sampleNote(SAMPLE)).toBe('Based on a random sample of 60 of 1,200 rows.');
    expect(sampleNote({ ...SAMPLE, n_total: 60 })).toBe('Based on all 60 rows.');
    expect(rowName(SAMPLE, 2)).toBe('Row 41');
  });

  it('zooms only histograms that a few extreme values squeeze into one bin', () => {
    const income = PROFILE.columns.find((c) => c.name === 'income')!;
    const age = PROFILE.columns.find((c) => c.name === 'age')!;
    expect(needsZoom(income)).toBe(true);
    expect(needsZoom(age)).toBe(false);
  });
});
