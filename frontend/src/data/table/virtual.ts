/**
 * Windowing for the data table: only the rows and columns in (and near) the visible part of
 * the scroll container are rendered, so samples with many thousand rows stay fast.
 */

/** Half-open index range [start, end). */
export interface IndexRange {
  start: number;
  end: number;
}

/** Rows of equal height that intersect the viewport, plus `overscan` rows on each side. */
export function rowRange(
  scrollTop: number,
  viewportHeight: number,
  rowHeight: number,
  count: number,
  overscan: number
): IndexRange {
  if (count <= 0 || rowHeight <= 0) {
    return { start: 0, end: 0 };
  }
  const first = Math.floor(Math.max(0, scrollTop) / rowHeight);
  const last = Math.ceil((Math.max(0, scrollTop) + Math.max(0, viewportHeight)) / rowHeight);
  return {
    start: Math.min(count, Math.max(0, first - overscan)),
    end: Math.min(count, Math.max(0, last + overscan)),
  };
}

/** Left edge of every column and the total width at the end: [0, w0, w0 + w1, …]. */
export function columnOffsets(widths: number[]): number[] {
  const offsets = [0];
  for (const width of widths) {
    offsets.push(offsets[offsets.length - 1] + width);
  }
  return offsets;
}

/** Index of the column that contains x (offsets from `columnOffsets`). */
function columnAt(offsets: number[], x: number): number {
  let low = 0;
  let high = offsets.length - 2;
  while (low < high) {
    const middle = Math.ceil((low + high) / 2);
    if (offsets[middle] <= x) {
      low = middle;
    } else {
      high = middle - 1;
    }
  }
  return Math.max(0, low);
}

/** Columns of different widths that intersect [scrollLeft, scrollLeft + width). */
export function columnRange(
  offsets: number[],
  scrollLeft: number,
  viewportWidth: number,
  overscan: number
): IndexRange {
  const count = offsets.length - 1;
  if (count <= 0) {
    return { start: 0, end: 0 };
  }
  const left = Math.max(0, scrollLeft);
  const first = columnAt(offsets, left);
  const last = columnAt(offsets, left + Math.max(0, viewportWidth));
  return {
    start: Math.max(0, first - overscan),
    end: Math.min(count, last + 1 + overscan),
  };
}
