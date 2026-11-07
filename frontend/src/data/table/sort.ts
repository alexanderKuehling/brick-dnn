/** Sorting the sample in the table (view only, the data itself is not changed). */

/**
 * Row positions ordered by `values`: numbers numerically, text alphabetically (natural
 * order, so "item 2" < "item 10"), missing values always last; ties keep table order.
 */
export function sortOrder(
  values: (number | string | null)[],
  direction: 'ascending' | 'descending'
): number[] {
  const sign = direction === 'ascending' ? 1 : -1;
  const collator = new Intl.Collator('en', { numeric: true, sensitivity: 'base' });
  return values
    .map((_, index) => index)
    .sort((a, b) => {
      const x = values[a];
      const y = values[b];
      const xMissing = x === null || x === undefined || (typeof x === 'number' && Number.isNaN(x));
      const yMissing = y === null || y === undefined || (typeof y === 'number' && Number.isNaN(y));
      if (xMissing || yMissing) {
        return xMissing === yMissing ? a - b : xMissing ? 1 : -1;
      }
      const order =
        typeof x === 'number' && typeof y === 'number'
          ? x - y
          : collator.compare(String(x), String(y));
      return order === 0 ? a - b : sign * order;
    });
}

/** No sort → ascending → descending → no sort. */
export function nextSort(
  current: 'ascending' | 'descending' | null
): 'ascending' | 'descending' | null {
  if (current === null) {
    return 'ascending';
  }
  return current === 'ascending' ? 'descending' : null;
}
