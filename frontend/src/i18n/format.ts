/**
 * Numbers and dates as the current language writes them (`Intl`): 1,234.5 and Nov 15, 2026
 * in English, 1.234,5 and 15.11.2026 in German.
 *
 * Only for display. Values sent to the API, put into generated code or into form fields that
 * the code parses keep JavaScript's own format (`String(value)`).
 */
import { currentLocale } from './index';

export function formatNumber(value: number, options?: Intl.NumberFormatOptions): string {
  return value.toLocaleString(currentLocale(), options);
}

/** A whole number with digit grouping: 12,345 / 12.345. */
export function formatInteger(value: number): string {
  return Math.round(value).toLocaleString(currentLocale());
}

export function formatDate(
  value: string | number | Date,
  options: Intl.DateTimeFormatOptions = { dateStyle: 'medium' }
): string {
  return new Date(value).toLocaleDateString(currentLocale(), options);
}

export function formatDateTime(
  value: string | number | Date,
  options: Intl.DateTimeFormatOptions = { dateStyle: 'medium', timeStyle: 'short' }
): string {
  return new Date(value).toLocaleString(currentLocale(), options);
}

export function formatTime(
  value: string | number | Date,
  options: Intl.DateTimeFormatOptions = { hour: '2-digit', minute: '2-digit' }
): string {
  return new Date(value).toLocaleTimeString(currentLocale(), options);
}

/** A list joined as the language does: "a, b and c" / "a, b und c". */
export function formatList(
  items: string[],
  type: 'conjunction' | 'disjunction' = 'conjunction'
): string {
  return new Intl.ListFormat(currentLocale(), { style: 'long', type }).format(items);
}
