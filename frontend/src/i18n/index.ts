/**
 * The languages of the app (docs/translations.md): English and German.
 *
 * i18next holds the texts (`locales/en`, `locales/de`); components read them with
 * `useTranslation()` from react-i18next, plain functions with `i18n.t`. The keys are typed
 * (`i18next.d.ts`), and the German texts must have exactly the keys of the English ones (the
 * compiler and `i18n.test.ts` check it).
 *
 * The first language comes from the browser (German for `de*`, otherwise English); a choice
 * made with the switch is kept in localStorage and, when logged in, on the account
 * (`AuthContext`), so it follows the user to other devices and decides the language of the
 * e-mails the server sends. `<html lang>` follows the language, and requests to the API carry
 * it as `Accept-Language`.
 */
import i18next from 'i18next';
import { initReactI18next, useTranslation } from 'react-i18next';
import { de } from './locales/de';
import { en } from './locales/en';

export const LANGUAGES = ['en', 'de'] as const;
export type Language = (typeof LANGUAGES)[number];

export const DEFAULT_LANGUAGE: Language = 'en';

/** The name of each language in itself, for the switch. */
export const LANGUAGE_NAMES: Record<Language, string> = { en: 'English', de: 'Deutsch' };

export const LANGUAGE_STORAGE_KEY = 'brickdnn:language';

export function isLanguage(value: unknown): value is Language {
  return typeof value === 'string' && (LANGUAGES as readonly string[]).includes(value);
}

/** The browser's preferred language: German for `de`, `de-AT` …, otherwise English. */
export function detectLanguage(
  languages: readonly string[] | undefined = typeof navigator === 'undefined'
    ? undefined
    : navigator.languages,
  fallback: string | undefined = typeof navigator === 'undefined' ? undefined : navigator.language
): Language {
  const first = languages?.find(Boolean) ?? fallback ?? '';
  return first.toLowerCase().startsWith('de') ? 'de' : 'en';
}

export function storedLanguage(): Language | null {
  try {
    const value = localStorage.getItem(LANGUAGE_STORAGE_KEY);
    return isLanguage(value) ? value : null;
  } catch {
    return null; // Storage may be unavailable (private mode).
  }
}

function storeLanguage(language: Language): void {
  try {
    localStorage.setItem(LANGUAGE_STORAGE_KEY, language);
  } catch {
    // Then the choice lasts until the page is reloaded.
  }
}

/** The language the app starts with: the saved choice, else the browser's. */
export function initialLanguage(): Language {
  return storedLanguage() ?? detectLanguage();
}

/** The language of the app now. */
export function currentLanguage(): Language {
  const language = i18next.resolvedLanguage ?? i18next.language;
  return isLanguage(language) ? language : DEFAULT_LANGUAGE;
}

/**
 * The BCP 47 locale numbers and dates are formatted with (`Intl`): `de-DE` writes 1.234,5 and
 * 15.11.2026. Only for what is shown: values sent to the API and generated code keep `.`.
 */
export function currentLocale(): string {
  return currentLanguage() === 'de' ? 'de-DE' : 'en-US';
}

/** Dates written out ("1 December 2026", "1. Dezember 2026"): British order in English. */
export function longDateLocale(): string {
  return currentLanguage() === 'de' ? 'de-DE' : 'en-GB';
}

function followDocument(language: string): void {
  if (typeof document !== 'undefined') {
    document.documentElement.lang = isLanguage(language) ? language : DEFAULT_LANGUAGE;
  }
}

type Listener = (language: Language) => void;
const choiceListeners = new Set<Listener>();

/**
 * Switch the language because the user chose it: kept in localStorage, and listeners (the
 * account, when logged in) are told. `save: false` only switches, e.g. to the language saved
 * on the account after logging in.
 */
export async function setLanguage(
  language: Language,
  { save = true }: { save?: boolean } = {}
): Promise<void> {
  storeLanguage(language);
  if (currentLanguage() !== language || i18next.language !== language) {
    await i18next.changeLanguage(language);
  }
  if (save) {
    choiceListeners.forEach((listener) => listener(language));
  }
}

/** Called when the user picks a language with the switch (not when the app follows one). */
export function onLanguageChosen(listener: Listener): () => void {
  choiceListeners.add(listener);
  return () => choiceListeners.delete(listener);
}

/** The current language; components re-render when it changes. */
export function useLanguage(): Language {
  const { i18n } = useTranslation();
  const language = i18n.resolvedLanguage ?? i18n.language;
  return isLanguage(language) ? language : DEFAULT_LANGUAGE;
}

export const resources = { en: { translation: en }, de: { translation: de } } as const;

i18next.on('languageChanged', followDocument);

void i18next.use(initReactI18next).init({
  resources,
  lng: initialLanguage(),
  fallbackLng: DEFAULT_LANGUAGE,
  supportedLngs: [...LANGUAGES],
  // React escapes already.
  interpolation: { escapeValue: false },
  returnNull: false,
  // The resources are bundled: ready right away, no Suspense.
  initAsync: false,
  react: { useSuspense: false },
});
followDocument(i18next.language);

export const i18n = i18next;
export { useTranslation, Trans } from 'react-i18next';
