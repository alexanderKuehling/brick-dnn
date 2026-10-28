import {
  extent,
  formatPercentTick,
  formatTick,
  integerTicks,
  linearScale,
  linePath,
  nearestIndex,
  niceScale,
  niceStep,
  stepDecimals,
} from './chartScale';

describe('niceStep', () => {
  it('rounds up to 1, 2 or 5 times a power of ten', () => {
    expect(niceStep(0.7)).toBe(1);
    expect(niceStep(1.3)).toBe(2);
    expect(niceStep(3)).toBe(5);
    expect(niceStep(7)).toBe(10);
    expect(niceStep(0.023)).toBeCloseTo(0.05);
    expect(niceStep(180)).toBe(200);
    expect(niceStep(0)).toBe(1);
    expect(niceStep(Number.NaN)).toBe(1);
  });

  it('knows how many decimals a step needs', () => {
    expect(stepDecimals(1)).toBe(0);
    expect(stepDecimals(50)).toBe(0);
    expect(stepDecimals(0.5)).toBe(1);
    expect(stepDecimals(0.05)).toBe(2);
  });
});

describe('niceScale', () => {
  it('extends the domain to round numbers', () => {
    const scale = niceScale(0.12, 2.31, 5);
    expect(scale.step).toBe(1);
    expect(scale.min).toBe(0);
    expect(scale.max).toBe(3);
    expect(scale.ticks).toEqual([0, 1, 2, 3]);
  });

  it('avoids floating point noise in the ticks', () => {
    const scale = niceScale(0.1, 0.3, 5);
    expect(scale.ticks).toEqual([0.1, 0.15, 0.2, 0.25, 0.3]);
  });

  it('gives a single value some room', () => {
    const scale = niceScale(0.42, 0.42, 5);
    expect(scale.min).toBeLessThan(0.42);
    expect(scale.max).toBeGreaterThan(0.42);
    expect(niceScale(0, 0).ticks.length).toBeGreaterThan(1);
  });

  it('respects bounds', () => {
    const accuracy = niceScale(0.97, 1, 5, [0, 1]);
    expect(accuracy.max).toBe(1);
    expect(accuracy.ticks[accuracy.ticks.length - 1]).toBe(1);
    const loss = niceScale(0, 0, 5, [0, Infinity]);
    expect(loss.min).toBe(0);
    expect(loss.max).toBeGreaterThan(0);
  });

  it('copes with missing values', () => {
    const scale = niceScale(Number.NaN, Number.NaN);
    expect(scale.ticks.length).toBeGreaterThan(1);
  });
});

describe('niceScale with awkward numbers', () => {
  it('finishes for values that differ only by rounding noise', () => {
    // Two losses that differ in the last bit only.
    const scale = niceScale(2.3025850929940455, 2.302585092994046, 4, [0, Infinity]);
    expect(scale.ticks.length).toBeGreaterThan(1);
    expect(scale.ticks.length).toBeLessThanOrEqual(50);
    expect(scale.min).toBeLessThan(2.3025850929940455);
    expect(scale.max).toBeGreaterThan(2.302585092994046);
    for (const rel of [1e-12, 1e-13, 1e-15]) {
      expect(niceScale(2.3, 2.3 * (1 + rel), 4).ticks.length).toBeLessThanOrEqual(50);
    }
    expect(niceScale(0.1, 0.10000000000000002, 4, [0, 1]).ticks.length).toBeGreaterThan(1);
  });

  it('keeps tiny but real differences', () => {
    const scale = niceScale(2.3025, 2.3025001, 4);
    expect(scale.min).toBeLessThanOrEqual(2.3025);
    expect(scale.max).toBeGreaterThanOrEqual(2.3025001);
    expect(new Set(scale.ticks).size).toBe(scale.ticks.length);
  });

  it('keeps the bottom tick far from zero', () => {
    for (const [lo, hi] of [
      // With an absolute tolerance, rounding noise in min / step dropped the first tick here.
      [1000, 1000.00001],
      [1000.00000111, 1000.00004111],
      [1e9 + 0.3, 1e9 + 0.9],
      [123456.71, 123456.79],
    ]) {
      const scale = niceScale(lo, hi, 4);
      expect(scale.min).toBeLessThanOrEqual(lo);
      expect(scale.max).toBeGreaterThanOrEqual(hi);
      expect(scale.ticks[0]).toBe(scale.min);
      expect(scale.ticks[scale.ticks.length - 1]).toBe(scale.max);
    }
  });
});

describe('integerTicks', () => {
  it('returns whole numbers only', () => {
    expect(integerTicks(1, 1)).toEqual([1]);
    expect(integerTicks(1, 3, 6)).toEqual([1, 2, 3]);
    expect(integerTicks(1, 10, 6)).toEqual([2, 4, 6, 8, 10]);
    expect(integerTicks(1, 500, 6)).toEqual([100, 200, 300, 400, 500]);
    expect(integerTicks(5, 1)).toEqual([]);
  });
});

describe('linearScale', () => {
  it('maps the domain onto the range', () => {
    const scale = linearScale([0, 10], [100, 0]);
    expect(scale(0)).toBe(100);
    expect(scale(5)).toBe(50);
    expect(scale(10)).toBe(0);
  });

  it('puts a zero-width domain in the middle', () => {
    expect(linearScale([3, 3], [0, 200])(3)).toBe(100);
  });
});

describe('formatting', () => {
  it('formats ticks with the decimals of the step', () => {
    expect(formatTick(0.25, 0.05)).toBe('0.25');
    expect(formatTick(2, 0.5)).toBe('2.0');
    expect(formatTick(-0, 1)).toBe('0');
    expect(formatTick(2500, 500)).toBe('2.5K');
    expect(formatTick(1_000_000, 500_000)).toBe('1M');
  });

  it('keeps large tick labels distinct', () => {
    const labels = [1200, 1210, 1220].map((tick) => formatTick(tick, 10));
    expect(labels).toEqual(['1,200', '1,210', '1,220']);
    expect([1_050_000, 1_100_000].map((tick) => formatTick(tick, 50_000))).toEqual([
      '1,050,000',
      '1,100,000',
    ]);
    expect([1000, 1500].map((tick) => formatTick(tick, 500))).toEqual(['1K', '1.5K']);
  });

  it('formats percent ticks', () => {
    expect(formatPercentTick(0.95, 0.05)).toBe('95%');
    expect(formatPercentTick(0.975, 0.005)).toBe('97.5%');
  });
});

describe('helpers', () => {
  it('finds the extent of finite values', () => {
    expect(extent([3, null, 1, undefined, Number.NaN, 2])).toEqual([1, 3]);
    expect(extent([null])).toBeNull();
  });

  it('finds the nearest index', () => {
    const xs = [1, 2, 3, 4, 5];
    expect(nearestIndex(xs, 0)).toBe(0);
    expect(nearestIndex(xs, 2.4)).toBe(1);
    expect(nearestIndex(xs, 2.6)).toBe(2);
    expect(nearestIndex(xs, 99)).toBe(4);
    expect(nearestIndex([7], 1)).toBe(0);
    expect(nearestIndex([], 1)).toBe(-1);
  });

  it('breaks lines at missing values', () => {
    expect(linePath([[0, 0], [10, 5], null, [20, 1]])).toBe('M0.0,0.0L10.0,5.0M20.0,1.0');
    expect(linePath([])).toBe('');
  });
});
