import {
  boxFromStats,
  boxFromValues,
  describe as describeValues,
  ols,
  pearson,
  quantileSorted,
  ranks,
  sortedFinite,
  spearman,
} from './stats';

// Reference values computed with pandas 2.3 (Series.mean/std/skew/kurt/quantile/corr).
const VALUES = [2, 4, 4, 4, 5, 5, 7, 9, 30, -1];

describe('stats', () => {
  it('sorts the finite numbers only', () => {
    expect(sortedFinite([3, null, 'a', NaN, 1, Infinity, 2])).toEqual([1, 2, 3]);
  });

  it('interpolates quantiles linearly like pandas', () => {
    const sorted = sortedFinite(VALUES);
    expect(quantileSorted(sorted, 0.01)).toBeCloseTo(-0.73, 10);
    expect(quantileSorted(sorted, 0.05)).toBeCloseTo(0.35, 10);
    expect(quantileSorted(sorted, 0.25)).toBe(4);
    expect(quantileSorted(sorted, 0.5)).toBe(4.5);
    expect(quantileSorted(sorted, 0.75)).toBe(6.5);
    expect(quantileSorted(sorted, 0.95)).toBeCloseTo(20.55, 10);
    expect(quantileSorted(sorted, 0.99)).toBeCloseTo(28.11, 10);
    expect(quantileSorted([], 0.5)).toBeNaN();
  });

  it('describes values like pandas (sample std, bias-corrected skew and kurtosis)', () => {
    const stats = describeValues([...VALUES, null, 'x']);
    expect(stats.mean).toBeCloseTo(6.9, 12);
    expect(stats.std).toBeCloseTo(8.543353491977777, 12);
    expect(stats.skew).toBeCloseTo(2.592171390963774, 12);
    expect(stats.kurtosis).toBeCloseTo(7.519778757448292, 12);
    expect(stats.min).toBe(-1);
    expect(stats.max).toBe(30);
    expect(stats.negative).toBe(1);
    expect(stats.zeros).toBe(0);
    // Fences: 4 − 1.5·2.5 = 0.25 and 6.5 + 3.75 = 10.25 → −1 and 30 are outliers.
    expect(stats.iqr_outliers).toBe(2);
    expect(stats.z_outliers).toBe(0);
  });

  it('leaves undefined statistics empty', () => {
    const constant = describeValues([5, 5, 5, 5]);
    expect(constant.std).toBe(0);
    expect(constant.skew).toBeNull();
    expect(constant.kurtosis).toBeNull();
    const empty = describeValues([]);
    expect(empty.mean).toBeNull();
    expect(empty.min).toBeNull();
    expect(empty.median).toBeNull();
  });

  it('fits a least-squares line', () => {
    const fit = ols([1, 2, 3, 4, 5, 6], [2, 1, 4, 3, 7, 8])!;
    expect(fit.slope).toBeCloseTo(1.34285714, 7);
    expect(fit.intercept).toBeCloseTo(-0.53333333, 7);
    expect(fit.r).toBeCloseTo(0.9014600868406593, 12);
    expect(fit.r2).toBeCloseTo(0.9014600868406593 ** 2, 12);
    expect(fit.n).toBe(6);
    expect(ols([1, 1, 1], [1, 2, 3])).toBeNull();
    expect(ols([1], [1])).toBeNull();
  });

  it('keeps its precision for large numbers such as timestamps', () => {
    const base = Date.UTC(2024, 0, 1);
    const x = [0, 1, 2, 3].map((d) => base + d * 86_400_000);
    const fit = ols(x, [10, 12, 14, 16])!;
    expect(fit.r).toBeCloseTo(1, 10);
    expect(fit.slope * 86_400_000).toBeCloseTo(2, 8);
  });

  it('computes Pearson and Spearman correlations', () => {
    expect(pearson([1, 2, 3, 4, 5, 6], [2, 1, 4, 3, 7, 8])).toBeCloseTo(0.90146008684, 10);
    expect(spearman([1, 2, 3, 4, 5, 6], [2, 1, 4, 3, 7, 8])).toBeCloseTo(0.8857142857, 10);
    // Ties get their average rank.
    expect(ranks([1, 2, 2, 3])).toEqual([1, 2.5, 2.5, 4]);
    expect(spearman([1, 2, 2, 3], [1, 3, 2, 2])).toBeCloseTo(0.5, 10);
    expect(pearson([1, 2, 3], [4, 4, 4])).toBeNull();
    expect(pearson([1, NaN, 3, 4], [1, 5, 3, 4])).toBeCloseTo(1, 10);
  });

  it('builds box plots from values with Tukey whiskers', () => {
    const box = boxFromValues(VALUES)!;
    expect(box.q1).toBe(4);
    expect(box.median).toBe(4.5);
    expect(box.q3).toBe(6.5);
    expect(box.low).toBe(2);
    expect(box.high).toBe(9);
    expect(box.outliers).toEqual([-1, 30]);
    expect(box.outlierCount).toBe(2);
    expect(box.exact).toBe(true);
    expect(boxFromValues([])).toBeNull();
  });

  it('estimates whiskers from profile statistics', () => {
    const stats = describeValues(VALUES);
    const box = boxFromStats(stats, 10)!;
    expect(box.low).toBe(0.25);
    expect(box.high).toBe(10.25);
    expect(box.exact).toBe(false);
    expect(box.outlierCount).toBe(2);
    const tight = boxFromStats(describeValues([1, 2, 3, 4]))!;
    expect(tight.low).toBe(1);
    expect(tight.high).toBe(4);
    expect(tight.exact).toBe(true);
    expect(boxFromStats(describeValues([]))).toBeNull();
  });
});
