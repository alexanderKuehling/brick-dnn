import { innerTicks, logAxis, logScale, timeTicks, valueAxis } from './axes';

describe('axes', () => {
  it('keeps ticks inside a fixed domain', () => {
    const axis = innerTicks(22_100, 950_000, 5);
    expect(axis.domain).toEqual([22_100, 950_000]);
    expect(axis.ticks.every((t) => t >= 22_100 && t <= 950_000)).toBe(true);
    expect(axis.ticks.length).toBeGreaterThanOrEqual(3);
    expect(axis.format(200_000)).toBe('200K');
    expect(innerTicks(17.5, 78.5, 6, true).ticks.every(Number.isInteger)).toBe(true);
  });

  it('starts value axes on round numbers', () => {
    const axis = valueAxis(0, 1137, 5, [0, Infinity]);
    expect(axis.domain[0]).toBe(0);
    expect(axis.domain[1]).toBeGreaterThanOrEqual(1137);
    expect(axis.ticks[0]).toBe(0);
  });

  it('makes log axes from powers of ten', () => {
    const axis = logAxis(1, 1137);
    expect(axis.domain).toEqual([0.5, 10_000]);
    expect(axis.ticks).toEqual([1, 10, 100, 1000, 10_000]);
    expect(axis.format(1000)).toBe('1K');
    // Narrow ranges get 2 and 5 in between.
    expect(logAxis(3, 80).ticks).toEqual([1, 2, 5, 10, 20, 50, 100]);
    const shares = logAxis(0.001, 0.6, true);
    expect(shares.format(0.01)).toBe('1%');
    const scale = logScale([1, 100], [100, 0]);
    expect(scale(10)).toBeCloseTo(50, 10);
    expect(scale(0)).toBe(100);
  });

  it('places time ticks on calendar boundaries (UTC)', () => {
    const years = timeTicks(Date.UTC(2021, 0, 1), Date.UTC(2024, 0, 1), 4);
    expect(years.unit).toBe('year');
    expect(years.ticks).toEqual([2021, 2022, 2023, 2024].map((y) => Date.UTC(y, 0, 1)));
    expect(years.format(years.ticks[0])).toBe('2021');

    const months = timeTicks(Date.UTC(2021, 0, 15), Date.UTC(2021, 11, 20), 5);
    expect(months.unit).toBe('month');
    expect(months.ticks.every((t) => new Date(t).getUTCDate() === 1)).toBe(true);
    expect(months.format(Date.UTC(2021, 3, 1))).toBe('Apr 2021');

    const days = timeTicks(Date.UTC(2021, 0, 1), Date.UTC(2021, 0, 8), 7);
    expect(days.unit).toBe('day');
    expect(days.ticks).toHaveLength(8);

    const hours = timeTicks(Date.UTC(2021, 0, 1, 0), Date.UTC(2021, 0, 1, 12), 4);
    expect(hours.ticks).toEqual([0, 3, 6, 9, 12].map((h) => Date.UTC(2021, 0, 1, h)));
  });
});
