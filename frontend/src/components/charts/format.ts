/** Number and date formatting for charts and statistics (English, UTC dates). */

const isNumber = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value);

/** 1234567 → "1,234,567". */
export function formatCount(value: number | null | undefined): string {
  return isNumber(value) ? Math.round(value).toLocaleString('en-US') : '–';
}

/**
 * A measured value with about four significant digits: 41200 → "41,200", 0.4183 → "0.4183",
 * 1.5e-7 → "1.5e-7", 3.2e18 → "3.2e+18".
 */
export function formatNumber(value: number | null | undefined, digits = 4): string {
  if (!isNumber(value)) {
    return '–';
  }
  const size = Math.abs(value);
  if (size !== 0 && (size < 1e-4 || size >= 1e15)) {
    return value.toExponential(Math.max(0, digits - 2)).replace(/\.?0+e/, 'e');
  }
  if (size >= 10 ** digits) {
    return Math.round(value).toLocaleString('en-US');
  }
  const text = value.toLocaleString('en-US', { maximumSignificantDigits: digits });
  return text === '-0' ? '0' : text;
}

const compact = new Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 1 });

/** 12900 → "12.9K", 4200000 → "4.2M"; small numbers like formatNumber. */
export function formatCompact(value: number | null | undefined): string {
  if (!isNumber(value)) {
    return '–';
  }
  return Math.abs(value) >= 10_000 ? compact.format(value) : formatNumber(value);
}

/**
 * A share (0 … 1) as percent with enough digits to tell it from 0 and 100:
 * 0.5 → "50%", 0.042 → "4.2%", 0.0004 → "<0.1%", 0.9996 → ">99.9%".
 */
export function formatPercent(share: number | null | undefined): string {
  if (!isNumber(share)) {
    return '–';
  }
  if (share === 0 || share === 1) {
    return `${share * 100}%`;
  }
  const size = Math.abs(share);
  if (size < 0.001) {
    return share < 0 ? '>-0.1%' : '<0.1%';
  }
  if (share > 0.999 && share < 1) {
    return '>99.9%';
  }
  const digits = size < 0.1 || share > 0.99 ? 1 : 0;
  // "5.0%" reads as noise; keep the decimal only where it says something.
  return `${(share * 100).toFixed(digits).replace(/\.0$/, '')}%`;
}

/** Correlations and other coefficients: two decimals and a real minus sign. */
export function formatCoefficient(value: number | null | undefined): string {
  if (!isNumber(value)) {
    return '–';
  }
  const text = value.toFixed(2);
  return text === '-0.00' ? '0.00' : text.replace('-', '−');
}

export function formatBytes(bytes: number | null | undefined): string {
  if (!isNumber(bytes)) {
    return '–';
  }
  const units = ['bytes', 'KB', 'MB', 'GB', 'TB'];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return unit === 0 ? `${value} bytes` : `${value.toFixed(value < 10 ? 1 : 0)} ${units[unit]}`;
}

export type TimeUnit = 'year' | 'month' | 'day' | 'hour' | 'minute' | 'second';

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const pad = (n: number) => String(n).padStart(2, '0');

/** A timestamp (ms since the epoch, UTC) at the given precision: "2021", "Mar 2021", "2021-03-04". */
export function formatDate(ms: number | null | undefined, unit: TimeUnit = 'day'): string {
  if (!isNumber(ms)) {
    return '–';
  }
  const d = new Date(ms);
  if (Number.isNaN(d.getTime())) {
    return '–';
  }
  const date = `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())}`;
  switch (unit) {
    case 'year':
      return String(d.getUTCFullYear());
    case 'month':
      return `${MONTHS[d.getUTCMonth()]} ${d.getUTCFullYear()}`;
    case 'day':
      return date;
    case 'hour':
    case 'minute':
      return `${date} ${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}`;
    default:
      return `${date} ${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}:${pad(d.getUTCSeconds())}`;
  }
}

const DAY = 86_400_000;

/**
 * A timestamp in full (UTC), for tooltips and labels: "2021-03-04" at midnight, else
 * "2021-03-04 13:05:09" (and ".250" when it has milliseconds).
 */
export function formatDateTime(ms: number | null | undefined): string {
  if (!isNumber(ms) || Number.isNaN(new Date(ms).getTime())) {
    return '–';
  }
  const d = new Date(ms);
  const day = formatDate(ms, 'day');
  if (((ms % DAY) + DAY) % DAY === 0) {
    return day;
  }
  const time = `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}:${pad(d.getUTCSeconds())}`;
  const millis = d.getUTCMilliseconds();
  return `${day} ${time}${millis ? `.${String(millis).padStart(3, '0')}` : ''}`;
}

/** Precision that tells timestamps `step` ms apart from each other. */
export function timeUnitFor(step: number): TimeUnit {
  if (step >= 365 * DAY) {
    return 'year';
  }
  if (step >= 28 * DAY) {
    return 'month';
  }
  if (step >= DAY) {
    return 'day';
  }
  if (step >= 60_000) {
    return 'minute';
  }
  return 'second';
}

/** Precision for a value of a datetime column: days unless the data has a time of day. */
export function dateUnitForValues(values: Iterable<number>): TimeUnit {
  let unit: TimeUnit = 'day';
  for (const value of values) {
    if (value % 1000 !== 0) {
      return 'second';
    }
    if (value % 60_000 !== 0) {
      unit = 'second';
    } else if (unit === 'day' && value % DAY !== 0) {
      unit = 'minute';
    }
  }
  return unit;
}

/** A duration in ms: "3 days", "5 h", "2.5 years". */
export function formatDuration(ms: number | null | undefined): string {
  if (!isNumber(ms)) {
    return '–';
  }
  const size = Math.abs(ms);
  const units: [number, string][] = [
    [365.25 * DAY, 'year'],
    [30.44 * DAY, 'month'],
    [DAY, 'day'],
    [3_600_000, 'hour'],
    [60_000, 'minute'],
    [1000, 'second'],
  ];
  for (const [length, name] of units) {
    if (size >= length) {
      const value = Number((ms / length).toPrecision(2));
      return `${value} ${name}${value === 1 ? '' : 's'}`;
    }
  }
  return `${formatNumber(ms)} ms`;
}
