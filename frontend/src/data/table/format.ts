/** Formatting of sample values and column facts for the data table. */
import type { ColumnKind } from '@/api/types';

const NBSP = ' ';

/** 1234567 → "1,234,567". */
export function formatCount(value: number): string {
  return value.toLocaleString('en-US');
}

/**
 * Numbers for table cells: thousands separators, up to six significant digits for
 * fractions, exponent notation for very large or tiny values.
 */
export function formatNumber(value: number): string {
  if (!Number.isFinite(value)) {
    return String(value);
  }
  const magnitude = Math.abs(value);
  if (magnitude !== 0 && (magnitude >= 1e15 || magnitude < 1e-6)) {
    return value.toExponential(3).replace(/\.?0+e/, 'e');
  }
  if (Number.isInteger(value)) {
    return value.toLocaleString('en-US');
  }
  return value.toLocaleString('en-US', {
    maximumSignificantDigits: Math.max(6, Math.floor(Math.log10(magnitude)) + 3),
  });
}

/**
 * Milliseconds since the epoch (UTC) → ISO 8601: "2021-03-05" at midnight,
 * "2021-03-05 14:30:00" otherwise (milliseconds only when there are any).
 */
export function formatDateTime(ms: number): string {
  const date = new Date(ms);
  if (Number.isNaN(date.getTime())) {
    return String(ms);
  }
  const iso = date.toISOString(); // 2021-03-05T14:30:00.000Z
  const day = iso.slice(0, 10);
  const time = iso.slice(11, 23);
  if (time === '00:00:00.000') {
    return day;
  }
  return `${day} ${time.endsWith('.000') ? time.slice(0, 8) : time}`;
}

export interface FormattedCell {
  text: string;
  missing: boolean;
  align: 'left' | 'right' | 'center';
}

/** One sample value as shown in the table. */
export function formatCell(
  value: number | string | null | undefined,
  kind: ColumnKind
): FormattedCell {
  if (value === null || value === undefined || (typeof value === 'number' && Number.isNaN(value))) {
    return { text: 'missing', missing: true, align: kind === 'numeric' ? 'right' : 'left' };
  }
  if (typeof value === 'number') {
    if (kind === 'datetime') {
      return { text: formatDateTime(value), missing: false, align: 'left' };
    }
    if (kind === 'boolean') {
      return { text: value ? 'true' : 'false', missing: false, align: 'center' };
    }
    return {
      text: formatNumber(value),
      missing: false,
      align: kind === 'numeric' ? 'right' : 'left',
    };
  }
  return { text: value, missing: false, align: 'left' };
}

/**
 * A share as a percentage for column headers: "0 %", "<0.1 %", "4.2 %", "57 %".
 */
export function formatShare(part: number, total: number): string {
  if (total <= 0 || part <= 0) {
    return `0${NBSP}%`;
  }
  const percent = (part / total) * 100;
  if (percent < 0.1) {
    return `<0.1${NBSP}%`;
  }
  if (percent >= 99.95 && part < total) {
    return `>99.9${NBSP}%`;
  }
  const digits = percent < 10 ? 1 : 0;
  return `${Number(percent.toFixed(digits))}${NBSP}%`;
}

/** "Showing 5,000 random rows of 1,234,567" / "Showing all 1,200 rows". */
export function sampleCaption(rows: number, total: number): string {
  if (rows >= total) {
    return `Showing all ${formatCount(total)} ${total === 1 ? 'row' : 'rows'}`;
  }
  return `Showing ${formatCount(rows)} random ${rows === 1 ? 'row' : 'rows'} of ${formatCount(total)}`;
}
