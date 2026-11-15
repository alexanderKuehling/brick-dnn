#!/usr/bin/env node
/**
 * Accessibility smoke check of a running Brick DNN (docs/accessibility.md): opens the main
 * pages in Chromium, in English and German and in the light and dark theme, and runs axe-core
 * (WCAG 2.0, 2.1 and 2.2, levels A and AA) on each. It needs Playwright with Chromium (not a
 * dependency of the app; e.g. `npm install -g playwright && npx playwright install chromium`)
 * and an app with its API, e.g. `make backend` and `npm run dev`.
 *
 *   node scripts/axe-smoke.mjs                       # http://localhost:5173, logged out
 *   BASE_URL=http://127.0.0.1:5917 A11Y_USER=alice A11Y_PASSWORD=… node scripts/axe-smoke.mjs
 *
 * With A11Y_USER and A11Y_PASSWORD it also checks the pages behind the login (my networks,
 * account, the training panel and a finished run, if there is one). Screenshots and a JSON
 * report go to A11Y_OUT (default: a new folder in the system's temp folder). The exit code is
 * 1 if a page has a violation with the impact "serious" or "critical".
 *
 * Not part of CI: it needs a browser and a running server. The component tests (vitest,
 * `src/a11y`) run axe on the main components on every push.
 */
import { mkdirSync, mkdtempSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { AxeBuilder } from '@axe-core/playwright';

const require = createRequire(import.meta.url);

/** Playwright from this project or, failing that, a global installation (NODE_PATH). */
function loadPlaywright() {
  const places = [undefined, ...(process.env.NODE_PATH ?? '').split(':').filter(Boolean)];
  for (const place of places) {
    try {
      return place
        ? require(require.resolve('playwright', { paths: [place] }))
        : require('playwright');
    } catch {
      // Try the next place.
    }
  }
  console.error('Playwright is missing: npm install -g playwright, then set NODE_PATH.');
  process.exit(2);
}

const BASE_URL = (process.env.BASE_URL ?? 'http://localhost:5173').replace(/\/$/, '');
const USER = process.env.A11Y_USER;
const PASSWORD = process.env.A11Y_PASSWORD;
const OUT = process.env.A11Y_OUT ?? mkdtempSync(join(tmpdir(), 'brickdnn-a11y-'));
const LANGUAGES = (process.env.A11Y_LANGUAGES ?? 'en,de').split(',');
const SCHEMES = (process.env.A11Y_SCHEMES ?? 'light,dark').split(',');
const TAGS = ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22aa'];

/** Pages without a login. `prepare` brings the page into the state to check. */
const PUBLIC_PAGES = [
  { name: 'login', path: '/login' },
  { name: 'register', path: '/register' },
  { name: 'forgot-password', path: '/forgot-password' },
  { name: 'editor', path: '/app' },
  { name: 'pricing', path: '/pricing' },
  { name: 'imprint', path: '/imprint' },
  { name: 'privacy', path: '/privacy' },
  { name: 'terms', path: '/terms' },
  { name: 'refund', path: '/refund' },
  { name: 'accessibility', path: '/accessibility' },
  { name: 'not-found', path: '/no-such-page' },
];

const PRIVATE_PAGES = [
  { name: 'editor-logged-in', path: '/app' },
  {
    name: 'training-panel',
    path: '/app',
    prepare: async (page) => {
      await page.keyboard.press('Control+Enter');
      await page.getByRole('dialog').waitFor({ timeout: 10000 });
      await page.waitForTimeout(800);
    },
  },
  {
    name: 'run-view',
    path: '/app',
    prepare: async (page) => {
      await page.keyboard.press('Control+Enter');
      const dialog = page.getByRole('dialog');
      await dialog.waitFor({ timeout: 10000 });
      await dialog.getByRole('tab').last().click();
      // All of the user's runs, not only those of the network on the canvas.
      const all = dialog.getByText(/All my runs|Alle meine Trainings|Alle eigenen/);
      if (await all.count()) {
        await all.first().click();
      }
      const run = dialog.locator('[data-run-id], button:has-text("MLP")').first();
      await run.click({ timeout: 5000 });
      await page.waitForTimeout(1500);
    },
  },
  { name: 'data', path: '/data' },
  { name: 'projects', path: '/projects' },
  { name: 'account', path: '/account' },
  { name: 'tokens', path: '/tokens' },
  { name: 'activity', path: '/activity' },
];

async function login(context) {
  const response = await context.request.post(`${BASE_URL}/api/auth/token/`, {
    data: { username: USER, password: PASSWORD },
  });
  if (!response.ok()) {
    throw new Error(`Login failed: HTTP ${response.status()}`);
  }
  const tokens = await response.json();
  await context.addInitScript((value) => {
    localStorage.setItem('brickdnn:tokens', value);
  }, JSON.stringify(tokens));
}

async function check(browser, page, { language, scheme, loggedIn }) {
  const context = await browser.newContext({
    colorScheme: scheme,
    locale: language === 'de' ? 'de-DE' : 'en-US',
    viewport: { width: 1440, height: 900 },
  });
  await context.addInitScript(
    ({ language }) => {
      localStorage.setItem('brickdnn:language', language);
      // No guide over the editor.
      localStorage.setItem('brickdnn:guide-seen', new Date().toISOString());
    },
    { language }
  );
  if (loggedIn) {
    await login(context);
  }
  const tab = await context.newPage();
  const label = `${page.name} [${language}, ${scheme}]`;
  try {
    await tab.goto(`${BASE_URL}${page.path}`, { waitUntil: 'networkidle' });
    await tab.waitForTimeout(600);
    if (page.prepare) {
      await page.prepare(tab);
    }
    const result = await new AxeBuilder({ page: tab }).withTags(TAGS).analyze();
    await tab.screenshot({ path: join(OUT, `${page.name}-${language}-${scheme}.png`) });
    return { label, page: page.name, language, scheme, violations: result.violations };
  } catch (error) {
    return { label, page: page.name, language, scheme, error: String(error), violations: [] };
  } finally {
    await context.close();
  }
}

async function main() {
  mkdirSync(OUT, { recursive: true });
  const { chromium } = loadPlaywright();
  const browser = await chromium.launch();
  const results = [];
  const runs = [
    ...PUBLIC_PAGES.map((page) => ({ page, loggedIn: false })),
    ...(USER && PASSWORD ? PRIVATE_PAGES.map((page) => ({ page, loggedIn: true })) : []),
  ];
  for (const { page, loggedIn } of runs) {
    for (const language of LANGUAGES) {
      for (const scheme of SCHEMES) {
        const result = await check(browser, page, { language, scheme, loggedIn });
        results.push(result);
        const counts = result.violations.map((v) => `${v.id} (${v.impact}, ${v.nodes.length})`);
        console.log(
          `${result.error ? 'ERROR' : counts.length ? 'FAIL ' : 'ok   '} ${result.label}` +
            (result.error ? `: ${result.error.split('\n')[0]}` : '') +
            (counts.length ? `: ${counts.join(', ')}` : '')
        );
      }
    }
  }
  await browser.close();
  writeFileSync(join(OUT, 'axe-report.json'), JSON.stringify(results, null, 2));
  const serious = results.filter((result) =>
    result.violations.some((v) => v.impact === 'serious' || v.impact === 'critical')
  );
  console.log(`\n${results.length} checks, ${serious.length} with serious or critical violations.`);
  console.log(`Report and screenshots: ${OUT}`);
  process.exit(serious.length > 0 ? 1 : 0);
}

await main();
