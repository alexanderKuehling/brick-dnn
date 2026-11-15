import { i18n } from '@/i18n';

/**
 * The name of a layer category in the current language. The server's category names
 * ("Basic", "Classical models") are identifiers (colours, grouping) and stay English.
 */
export function categoryLabel(category: string): string {
  const key = `builder.categories.${category}`;
  return i18n.exists(key) ? String(i18n.t(key as 'builder.categories.Basic')) : category;
}
