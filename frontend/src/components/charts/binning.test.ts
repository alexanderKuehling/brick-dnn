import {
  autoEdges,
  binCount,
  binIndex,
  commonEdges,
  countBins,
  histogram,
  rebin,
  roundEdges,
  sameEdges,
} from './binning';

describe('binning', () => {
  const edges = [0, 10, 20, 30];

  it('finds the bin of a value; the last bin includes its end', () => {
    expect(binIndex(edges, 0)).toBe(0);
    expect(binIndex(edges, 9.99)).toBe(0);
    expect(binIndex(edges, 10)).toBe(1);
    expect(binIndex(edges, 30)).toBe(2);
    expect(binIndex(edges, -0.1)).toBe(-1);
    expect(binIndex(edges, 30.1)).toBe(-1);
    expect(binIndex(edges, NaN)).toBe(-1);
    expect(binIndex([5], 5)).toBe(-1);
  });

  it('counts numbers per bin and ignores everything else', () => {
    expect(countBins([0, 5, 10, 25, 30, 31, null, 'a', NaN], edges)).toEqual([2, 1, 2]);
  });

  it('picks the number of bins with the Freedman–Diaconis rule', () => {
    const uniform = Array.from({ length: 1000 }, (_, i) => i);
    // IQR ≈ 500 → width 2·500/10 = 100 → 10 bins.
    expect(binCount(uniform)).toBe(10);
    expect(binCount(uniform, 20)).toBe(20);
    expect(binCount([1, 1, 1])).toBe(1);
    // IQR 0 falls back to Sturges.
    expect(binCount([0, 0, 0, 0, 0, 0, 0, 100])).toBe(5);
  });

  it('puts edges on round numbers that cover all values', () => {
    const e = roundEdges(3.2, 97.5, 10);
    expect(e[0]).toBe(0);
    expect(e[e.length - 1]).toBeGreaterThanOrEqual(97.5);
    expect(e[1] - e[0]).toBe(10);
    expect(roundEdges(0.012, 0.087, 8)).toEqual([
      0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
    ]);
    expect(roundEdges(5, 5, 10)).toEqual([4.75, 5.25]);
  });

  it('gives whole numbers one bin each when there are few', () => {
    expect(autoEdges([1, 2, 2, 3], { integer: true })).toEqual([0.5, 1.5, 2.5, 3.5]);
    const wide = autoEdges(
      Array.from({ length: 500 }, (_, i) => i * 7),
      { integer: true }
    );
    expect(wide.length).toBeGreaterThan(2);
    expect(wide.every((edge) => Number.isInteger(edge))).toBe(true);
    expect(autoEdges([])).toEqual([]);
  });

  it('makes a histogram whose counts add up', () => {
    const values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, null];
    const h = histogram(values);
    expect(h.counts.reduce((a, b) => a + b, 0)).toBe(10);
    expect(h.edges.length).toBe(h.counts.length + 1);
  });

  it('moves counts onto other bins and keeps the total', () => {
    const h = { edges: [0, 10, 20], counts: [10, 20] };
    expect(rebin(h, [0, 10, 20])).toEqual([10, 20]);
    expect(rebin(h, [0, 5, 10, 15, 20])).toEqual([5, 5, 10, 10]);
    expect(rebin(h, [0, 20])).toEqual([30]);
    const shifted = rebin(h, [-10, 0, 10, 20, 30]);
    expect(shifted).toEqual([0, 10, 20, 0]);
  });

  it('shares edges between histograms', () => {
    const a = { edges: [0, 1, 2], counts: [1, 1] };
    const b = { edges: [0, 1, 2], counts: [2, 0] };
    expect(commonEdges([a, b])).toBe(a.edges);
    const c = { edges: [0, 50, 100], counts: [1, 1] };
    const shared = commonEdges([a, c], 10);
    expect(shared[0]).toBe(0);
    expect(shared[shared.length - 1]).toBeGreaterThanOrEqual(100);
    expect(sameEdges(shared, c.edges)).toBe(false);
    expect(commonEdges([])).toEqual([]);
  });
});
