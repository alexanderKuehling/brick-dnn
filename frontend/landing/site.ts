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

/** Pages of the React app that the landing pages link to and the sitemap lists. */
export const APP_PAGES = ['/pricing', '/imprint', '/privacy', '/terms', '/refund'] as const;

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
    storageGb: 1,
    cpuMinutesPerRun: 15,
    concurrentTrainings: 1,
  },
  {
    key: 'starter',
    name: 'Starter',
    price: 5,
    cpuHoursPerMonth: 10,
    storageGb: 5,
    cpuMinutesPerRun: 60,
    concurrentTrainings: 1,
  },
  {
    key: 'plus',
    name: 'Plus',
    price: 14,
    cpuHoursPerMonth: 30,
    storageGb: 20,
    cpuMinutesPerRun: 180,
    concurrentTrainings: 1,
  },
  {
    key: 'pro',
    name: 'Pro',
    price: 45,
    cpuHoursPerMonth: 120,
    storageGb: 100,
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
 * shown. It sends to the editor (/app, with query and hash):
 * - old links into the editor on `/` (`/?…` – nginx redirects those already – or a `#…` that
 *   is not a section of the page);
 * - users who are logged in: the app keeps their tokens in localStorage (`brickdnn:tokens`,
 *   src/api/client.ts), and the refresh token has not expired yet. Only read in the browser:
 *   no cookies, nothing is sent anywhere.
 *
 * nginx.conf allows exactly this script by its hash (Content-Security-Policy of `/` and `/de/`);
 * landing.test.ts checks that the hash there is this script's.
 */
export const INLINE_SCRIPT = `(function(){var l=location,h=l.hash.slice(1),s=${JSON.stringify([
  ...SECTION_IDS,
])},go=l.pathname=='/'&&(l.search||h&&s.indexOf(h)<0);if(!go)try{var t=JSON.parse(localStorage.getItem('brickdnn:tokens')).refresh.split('.')[1].replace(/-/g,'+').replace(/_/g,'/');go=JSON.parse(atob(t)).exp*1e3>Date.now()}catch(e){}if(go)l.replace('${EDITOR_PATH}'+l.search+l.hash)})();`;

/** The CSP source that allows INLINE_SCRIPT, e.g. `'sha256-…'`. */
export function inlineScriptHash(): string {
  return `'sha256-${createHash('sha256').update(INLINE_SCRIPT).digest('base64')}'`;
}

/** The absolute address of a path on the site. */
export function absolute(path: string): string {
  return `${SITE_URL}${path}`;
}
