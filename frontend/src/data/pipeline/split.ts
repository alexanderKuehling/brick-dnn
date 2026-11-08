/** Train / validation / test split settings: percent fields, validation, row counts. */

/** Largest seed the server accepts (2**31 − 1). */
export const MAX_SEED = 2 ** 31 - 1;
/** Validation + test may use at most this share of the rows. */
export const MAX_HELD_OUT = 0.9;

/** 0.15 → 15 (one decimal, without float noise). */
export function shareToPercent(share: number): number {
  return Math.round(share * 1000) / 10;
}

/** 15 → 0.15. */
export function percentToShare(percent: number): number {
  return Math.round(percent * 10) / 1000;
}

/** Plain-language problem with the shares, or null if they are fine. */
export function splitError(val: number, test: number): { val?: string; test?: string } | null {
  if (!(val > 0)) {
    return { val: 'Use more than 0 % for validation.' };
  }
  if (test < 0) {
    return { test: 'Use 0 % or more.' };
  }
  if (val + test > MAX_HELD_OUT + 1e-9) {
    return { test: 'Validation and test together can be at most 90 %.' };
  }
  return null;
}

/** Rows per part for `total` rows (train gets the rest, like the server). */
export function splitRows(total: number, val: number, test: number): [number, number, number] {
  const valRows = Math.round(total * val);
  const testRows = Math.min(total - valRows, Math.round(total * test));
  return [total - valRows - testRows, valRows, testRows];
}

/**
 * Row counts to show next to the shares: the counts of the last preview while they match
 * the shares, otherwise an estimate from the shares (`estimated` true).
 */
export function displayedSplit(
  splitCounts: number[] | null,
  val: number,
  test: number
): { counts: [number, number, number]; estimated: boolean } | null {
  if (!splitCounts || splitCounts.length < 3) {
    return null;
  }
  const total = splitCounts[0] + splitCounts[1] + splitCounts[2];
  const estimate = splitRows(total, val, test);
  const tolerance = Math.max(1, Math.round(total * 0.01));
  const matches = estimate.every((count, i) => Math.abs(count - splitCounts[i]) <= tolerance);
  return matches
    ? { counts: [splitCounts[0], splitCounts[1], splitCounts[2]], estimated: false }
    : { counts: estimate, estimated: true };
}
