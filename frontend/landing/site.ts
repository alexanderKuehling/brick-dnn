/**
 * Facts the landing pages (/ and /de/) share: addresses, plans and the inline script.
 *
 * The pages are static HTML made at build time (plugin.ts), so search engines and AI
 * assistants read them without JavaScript. Everything here must be true for brickdnn.com:
 * the plans are those of deploy/sites/brickdnn.com/settings.env (landing.test.ts compares them).
 */
import { createHash } from 'node:crypto';

export const SITE_URL = 'https://brickdnn.com';
/** Where the editor lives (the React app's route; `/` inside the app leads there too). */
export const EDITOR_PATH = '/app';
export const CONTACT_EMAIL = 'alexander.kuehling@brickdnn.com';
export const REPOSITORY_URL = 'https://github.com/alexanderKuehling/brick-dnn';
export const DOCS_URL = `${REPOSITORY_URL}/tree/main/docs`;
export const LICENSE_URL = 'https://www.apache.org/licenses/LICENSE-2.0';

export type Lang = 'en' | 'de';

/** The path of each language's page. */
export const PAGE_PATHS: Record<Lang, string> = { en: '/', de: '/de/' };

/**
 * The static content pages (pages.ts), made by the same generator: the example gallery, the
 * guides and the German pricing page. Every path ends with a slash; nginx.conf serves these
 * folders with the landing pages' strict Content-Security-Policy and redirects a path without
 * the slash to the one with it.
 */
export const EXAMPLES_PATHS: Record<Lang, string> = { en: '/examples/', de: '/de/beispiele/' };
export const GUIDES_PATHS: Record<Lang, string> = { en: '/guides/', de: '/de/anleitungen/' };
/**
 * The German pricing page. English has none of its own: /pricing is the app's page (where
 * plans are bought), so the English landing page links there.
 */
export const PRICING_DE_PATH = '/de/preise/';
/** The folders of the content pages (nginx.conf has a location for exactly these). */
export const CONTENT_FOLDERS = [
  EXAMPLES_PATHS.en,
  GUIDES_PATHS.en,
  EXAMPLES_PATHS.de,
  GUIDES_PATHS.de,
  PRICING_DE_PATH,
] as const;

/** Where each language's visitors find the prices. */
export function pricingPath(lang: Lang): string {
  return lang === 'de' ? PRICING_DE_PATH : '/pricing';
}

/** The author of the guides and example pages (JSON-LD). */
export const AUTHOR = { name: 'Alexander Kühling', url: 'https://github.com/alexanderKuehling' };

/** The editor with a built-in example loaded (the app reads ?example=, BuilderPage.tsx). */
export function editorExampleUrl(id: string): string {
  return `${EDITOR_PATH}?example=${encodeURIComponent(id)}`;
}

/** Pages of the React app that the landing pages link to and the sitemap lists. */
export const APP_PAGES = [
  '/pricing',
  '/imprint',
  '/privacy',
  '/terms',
  '/refund',
  '/accessibility',
] as const;

export interface PlanFacts {
  key: string;
  name: string;
  /** EUR per month incl. VAT. */
  price: number;
  cpuHoursPerMonth: number;
  storageGb: number;
  cpuMinutesPerRun: number;
  concurrentTrainings: number;
}

/** brickdnn.com's plans (BRICKDNN_PLANS in deploy/sites/brickdnn.com/settings.env). */
export const PLANS: PlanFacts[] = [
  {
    key: 'free',
    name: 'Free',
    price: 0,
    cpuHoursPerMonth: 0.5,
    storageGb: 0.5,
    cpuMinutesPerRun: 15,
    concurrentTrainings: 1,
  },
  {
    key: 'starter',
    name: 'Starter',
    price: 5,
    cpuHoursPerMonth: 10,
    storageGb: 3,
    cpuMinutesPerRun: 60,
    concurrentTrainings: 1,
  },
  {
    key: 'plus',
    name: 'Plus',
    price: 14,
    cpuHoursPerMonth: 30,
    storageGb: 10,
    cpuMinutesPerRun: 180,
    concurrentTrainings: 1,
  },
  {
    key: 'pro',
    name: 'Pro',
    price: 45,
    cpuHoursPerMonth: 120,
    storageGb: 50,
    cpuMinutesPerRun: 480,
    concurrentTrainings: 2,
  },
];
export const CURRENCY = 'EUR';
/** Compute beyond a paid plan's allowance, EUR per CPU hour (each account switches it on). */
export const OVERAGE_PRICE = 0.4;

/**
 * The ids of the landing page's sections. A link to `/#<one of these>` stays on the page;
 * any other `#…` on `/` is an old link into the editor and goes on to /app.
 */
export const SECTION_IDS = [
  'main',
  'at-a-glance',
  'use-cases',
  'how-it-works',
  'features',
  'privacy',
  'teaching',
  'pricing',
  'faq',
] as const;

/**
 * The only script of the landing pages, inline in <head> so that it runs before anything is
 * shown. It sends old links into the editor on `/` on to the editor (/app, with query and
 * hash): `/?…` (nginx redirects those already) or a `#…` that is not a section of the page.
 * Everybody else stays on the landing page, logged in or not; its header leads to the editor.
 * It reads and stores nothing.
 *
 * nginx.conf allows exactly this script by its hash (Content-Security-Policy of `/` and `/de/`);
 * landing.test.ts checks that the hash there is this script's.
 */
export const INLINE_SCRIPT = `(function(){var l=location,h=l.hash.slice(1),s=${JSON.stringify([
  ...SECTION_IDS,
])},go=l.pathname=='/'&&(l.search||h&&s.indexOf(h)<0);if(go)l.replace('${EDITOR_PATH}'+l.search+l.hash)})();`;

/** The CSP source that allows INLINE_SCRIPT, e.g. `'sha256-…'`. */
export function inlineScriptHash(): string {
  return `'sha256-${createHash('sha256').update(INLINE_SCRIPT).digest('base64')}'`;
}

/** The absolute address of a path on the site. */
export function absolute(path: string): string {
  return `${SITE_URL}${path}`;
}
