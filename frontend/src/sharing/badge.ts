import type { Language } from '@/i18n';

/**
 * The "Open in Brick DNN" badge (frontend/public/badge/, docs/badge.md): an SVG image this
 * site serves, for READMEs and course pages that link into the editor or to a shared network.
 */
const BADGE_FILES: Record<Language, string> = {
  en: '/badge/open-in-brick-dnn.svg',
  de: '/badge/open-in-brick-dnn.de.svg',
};
const BADGE_ALT: Record<Language, string> = {
  en: 'Open in Brick DNN',
  de: 'In Brick DNN öffnen',
};

/** The address of the badge on this site, in the language of its text. */
export function badgeUrl(language: Language): string {
  return new URL(BADGE_FILES[language], window.location.origin).href;
}

/** Markdown for a README or notebook: the badge as a link to `target`. */
export function badgeMarkdown(target: string, language: Language): string {
  return `[![${BADGE_ALT[language]}](${badgeUrl(language)})](${target})`;
}

/** The same as HTML, for pages that take no Markdown. */
export function badgeHtml(target: string, language: Language): string {
  return `<a href="${target}"><img src="${badgeUrl(language)}" alt="${BADGE_ALT[language]}" height="28"></a>`;
}
