import { render as plainRender, screen, userEvent, waitFor } from '@test-utils';
import { MemoryRouter } from 'react-router-dom';
import { NumberInput } from '@mantine/core';
import { formatCount, summarize } from '@/builder/format';
import { NotFoundPage } from '@/pages/NotFoundPage';
import { formatDate, formatInteger, formatNumber } from './format';
import {
  currentLanguage,
  detectLanguage,
  i18n,
  initialLanguage,
  LANGUAGE_STORAGE_KEY,
  onLanguageChosen,
  setLanguage,
} from './index';
import { LanguageSwitch } from './LanguageSwitch';
import { de } from './locales/de';
import { en } from './locales/en';
import { LocalizedMantineProvider } from './mantine';

type Tree = { [key: string]: string | Tree };

/** Every key of a resource tree with its text: "builder.toolbar.train" → "Train". */
function flatten(tree: Tree, prefix = ''): Map<string, string> {
  const result = new Map<string, string>();
  for (const [key, value] of Object.entries(tree)) {
    const path = prefix ? `${prefix}.${key}` : key;
    if (typeof value === 'string') {
      result.set(path, value);
    } else {
      flatten(value, path).forEach((text, sub) => result.set(sub, text));
    }
  }
  return result;
}

const english = flatten(en as unknown as Tree);
const german = flatten(de as unknown as Tree);

/** {{placeholders}} and <tags> of a text, sorted. */
function markers(text: string): string[] {
  return [...text.matchAll(/\{\{\s*\w+\s*\}\}|<\/?\w+\s*\/?>/g)].map((m) => m[0]).sort();
}

describe('translations', () => {
  it('has every English key in German and no others', () => {
    const missing = [...english.keys()].filter((key) => !german.has(key));
    const extra = [...german.keys()].filter((key) => !english.has(key));
    expect(missing).toEqual([]);
    expect(extra).toEqual([]);
  });

  it('has no empty texts', () => {
    const empty = [...english, ...german].filter(([, text]) => !text.trim()).map(([k]) => k);
    expect(empty).toEqual([]);
  });

  it('keeps placeholders and tags in German', () => {
    const differ = [...english]
      .filter(([key, text]) => markers(text).join() !== markers(german.get(key) ?? '').join())
      .map(([key]) => key);
    expect(differ).toEqual([]);
  });

  it('uses the plural forms both languages have', () => {
    for (const key of english.keys()) {
      if (key.endsWith('_one')) {
        expect(english.has(key.replace(/_one$/, '_other'))).toBe(true);
      }
    }
  });
});

describe('the language of the browser', () => {
  it('is German for de and its regions, otherwise English', () => {
    expect(detectLanguage(['de-DE', 'en-US'])).toBe('de');
    expect(detectLanguage(['de-AT'])).toBe('de');
    expect(detectLanguage(['DE'])).toBe('de');
    expect(detectLanguage(['en-US', 'de-DE'])).toBe('en');
    expect(detectLanguage(['fr-FR', 'de'])).toBe('en');
    expect(detectLanguage([], 'de-CH')).toBe('de');
    expect(detectLanguage(undefined, 'it')).toBe('en');
    expect(detectLanguage(undefined, undefined)).toBe('en');
  });

  it('decides the first language unless one was chosen', () => {
    const languages = vi.spyOn(navigator, 'languages', 'get').mockReturnValue(['de-DE', 'en']);
    expect(initialLanguage()).toBe('de');
    localStorage.setItem(LANGUAGE_STORAGE_KEY, 'en');
    expect(initialLanguage()).toBe('en');
    localStorage.setItem(LANGUAGE_STORAGE_KEY, 'xx');
    expect(initialLanguage()).toBe('de');
    languages.mockRestore();
  });

  it('is English in the tests', () => {
    expect(currentLanguage()).toBe('en');
    expect(document.documentElement.lang).toBe('en');
  });
});

describe('switching the language', () => {
  it('switches the texts, <html lang> and remembers the choice', async () => {
    const chosen = vi.fn();
    const stop = onLanguageChosen(chosen);
    await setLanguage('de');
    expect(i18n.t('builder.toolbar.train')).toBe('Trainieren');
    expect(document.documentElement.lang).toBe('de');
    expect(localStorage.getItem(LANGUAGE_STORAGE_KEY)).toBe('de');
    expect(chosen).toHaveBeenCalledWith('de');

    // Following the account's language is not a new choice.
    chosen.mockClear();
    await setLanguage('en', { save: false });
    expect(document.documentElement.lang).toBe('en');
    expect(localStorage.getItem(LANGUAGE_STORAGE_KEY)).toBe('en');
    expect(chosen).not.toHaveBeenCalled();
    stop();
  });

  it('has a switch EN | DE', async () => {
    plainRender(
      <MemoryRouter>
        <LanguageSwitch />
        <NotFoundPage />
      </MemoryRouter>
    );
    expect(screen.getByText('Page not found')).toBeInTheDocument();
    await userEvent.click(screen.getByText('DE'));
    expect(await screen.findByText('Seite nicht gefunden')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Zurück zum Editor' })).toBeInTheDocument();
    expect(document.documentElement.lang).toBe('de');
    expect(localStorage.getItem(LANGUAGE_STORAGE_KEY)).toBe('de');
    await userEvent.click(screen.getByText('EN'));
    expect(await screen.findByText('Page not found')).toBeInTheDocument();
    expect(document.documentElement.lang).toBe('en');
  });

  it('plural forms follow the count', async () => {
    expect(i18n.t('builder.toolbar.problems', { count: 1 })).toBe('1 problem');
    expect(i18n.t('builder.toolbar.problems', { count: 3 })).toBe('3 problems');
    await setLanguage('de', { save: false });
    expect(i18n.t('builder.toolbar.problems', { count: 1 })).toBe('1 Problem');
    expect(i18n.t('builder.toolbar.problems', { count: 3 })).toBe('3 Probleme');
  });
});

describe('numbers and dates', () => {
  it('are written as the language writes them', async () => {
    expect(formatNumber(1234.5)).toBe('1,234.5');
    expect(formatInteger(12345)).toBe('12,345');
    expect(formatCount(12345)).toBe('12.3k');
    expect(formatDate('2026-11-15T12:00:00Z', { dateStyle: 'long', timeZone: 'UTC' })).toBe(
      'November 15, 2026'
    );
    await setLanguage('de', { save: false });
    expect(formatNumber(1234.5)).toBe('1.234,5');
    expect(formatInteger(12345)).toBe('12.345');
    expect(formatCount(12345)).toBe('12,3 Tsd.');
    expect(formatDate('2026-11-15T12:00:00Z', { dateStyle: 'long', timeZone: 'UTC' })).toBe(
      '15. November 2026'
    );
  });

  it('keep the point in values for code (layer summaries show what the code gets)', async () => {
    await setLanguage('de', { save: false });
    // The dropout probability is a parameter value, written as in the generated code.
    expect(summarize('Dropout', { p: 0.25 })).toBe('p = 0.25');
    expect(String(0.25)).toBe('0.25');
  });

  it('number fields show a decimal comma in German and still return numbers', async () => {
    await setLanguage('de', { save: false });
    const changed = vi.fn();
    plainRender(
      <LocalizedMantineProvider>
        <NumberInput label="Lernrate" defaultValue={0.5} onChange={changed} />
      </LocalizedMantineProvider>
    );
    const input = screen.getByLabelText('Lernrate');
    expect(input).toHaveValue('0,5');
    await userEvent.clear(input);
    await userEvent.type(input, '0,25');
    await waitFor(() => expect(changed).toHaveBeenLastCalledWith(0.25));
  });
});
