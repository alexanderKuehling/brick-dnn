import { languageSection } from './languages';

const PAGE = [
  '# Refund policy / Erstattung',
  '',
  'Last updated / Stand: 15 November 2026',
  '',
  '## English',
  '',
  'Full refund within 14 days.',
  '',
  '## Deutsch',
  '',
  'Volle Erstattung innerhalb von 14 Tagen.',
].join('\n');

describe('bilingual legal pages', () => {
  it('keep the title and the section of the language', () => {
    const german = languageSection(PAGE, 'de')!;
    expect(german).toContain('# Refund policy / Erstattung');
    expect(german).toContain('Volle Erstattung');
    expect(german).not.toContain('Full refund');
    const english = languageSection(PAGE, 'en')!;
    expect(english).toContain('Full refund');
    expect(english).not.toContain('Volle Erstattung');
  });

  it('leave pages in one language alone', () => {
    expect(languageSection('# Datenschutzerklärung\n\n## 1. Verantwortlicher', 'en')).toBeNull();
    expect(languageSection('# Terms\n\n## English\n\nText', 'de')).toBeNull();
  });
});
