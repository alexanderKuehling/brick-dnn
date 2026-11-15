import type { Language } from '@/i18n';

/** The headings of the language sections of a bilingual page ("## English", "## Deutsch"). */
const SECTION_HEADINGS: Record<Language, string> = { en: 'English', de: 'Deutsch' };

/**
 * A page written in both languages, with a section `## English` and a section `## Deutsch`,
 * shows only the section of the app's language (and what comes before the first section,
 * e.g. the title). null if the page has no such sections. The text itself is not changed.
 */
export function languageSection(markdown: string, language: Language): string | null {
  const lines = markdown.split('\n');
  const headings = new Map<number, Language>();
  lines.forEach((line, index) => {
    const match = /^##\s+(.+?)\s*#*\s*$/.exec(line);
    const name = match?.[1];
    const found = (Object.keys(SECTION_HEADINGS) as Language[]).find(
      (code) => SECTION_HEADINGS[code] === name
    );
    if (found) {
      headings.set(index, found);
    }
  });
  const languages = new Set(headings.values());
  if (languages.size < 2 || !languages.has(language)) {
    return null;
  }
  const starts = [...headings.keys()];
  const intro = lines.slice(0, starts[0]);
  const kept: string[] = [];
  starts.forEach((start, position) => {
    if (headings.get(start) === language) {
      const end = starts[position + 1] ?? lines.length;
      kept.push(...lines.slice(start, end));
    }
  });
  return [...intro, ...kept].join('\n').trim();
}
