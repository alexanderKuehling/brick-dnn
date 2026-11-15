/**
 * Checks a build (dist/) for what the landing pages need, run at the end of every build
 * (plugin.ts) and by landing.test.ts. Returns the problems; empty means fine.
 */
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { absolute, INLINE_SCRIPT, PAGE_PATHS, type Lang } from './site.ts';

const FILES: Record<Lang, string> = { en: 'index.html', de: 'de/index.html' };

/** The @types of a page's JSON-LD (its @graph), or null if there is none or it is invalid. */
export function jsonLdTypes(html: string): string[] | null {
  const match = html.match(/<script type="application\/ld\+json">([\s\S]*?)<\/script>/);
  if (!match) {
    return null;
  }
  try {
    const data = JSON.parse(match[1]) as { '@graph'?: { '@type': string }[] };
    return (data['@graph'] ?? []).map((node) => node['@type']);
  } catch {
    return null;
  }
}

export function checkLandingHtml(lang: Lang, html: string): string[] {
  const problems: string[] = [];
  const need = (ok: boolean, what: string) => ok || problems.push(`${lang}: ${what}`);
  need(html.includes(`<html lang="${lang}"`), `<html lang="${lang}">`);
  need(
    html.includes(`<link rel="canonical" href="${absolute(PAGE_PATHS[lang])}">`),
    'canonical link'
  );
  for (const [hreflang, path] of [
    ['en', PAGE_PATHS.en],
    ['de', PAGE_PATHS.de],
    ['x-default', PAGE_PATHS.en],
  ]) {
    need(
      html.includes(`<link rel="alternate" hreflang="${hreflang}" href="${absolute(path)}">`),
      `hreflang ${hreflang}`
    );
  }
  const types = jsonLdTypes(html);
  need(types !== null, 'valid JSON-LD');
  for (const type of ['SoftwareApplication', 'FAQPage', 'Organization']) {
    need(Boolean(types?.includes(type)), `JSON-LD ${type}`);
  }
  need((html.match(/<h1[\s>]/g) ?? []).length === 1, 'exactly one <h1>');
  need(html.includes(`<script>${INLINE_SCRIPT}</script>`), 'the inline script of site.ts');
  need(html.includes('href="/imprint"') && html.includes('href="/privacy"'), 'legal links');
  return problems;
}

export function checkDist(dir: string): string[] {
  const problems: string[] = [];
  for (const lang of ['en', 'de'] as const) {
    const file = join(dir, FILES[lang]);
    if (!existsSync(file)) {
      problems.push(`${FILES[lang]} is missing`);
      continue;
    }
    const html = readFileSync(file, 'utf8');
    problems.push(...checkLandingHtml(lang, html));
    // Every file of this site that the page refers to is in the build.
    for (const [, path] of html.matchAll(
      /(?:src|href|srcset|content)="(\/[^"#?\s]+\.[a-z0-9]+)/g
    )) {
      if (!existsSync(join(dir, path))) {
        problems.push(`${lang}: ${path} is missing`);
      }
    }
    for (const [, list] of html.matchAll(/srcset="([^"]+)"/g)) {
      for (const entry of list.split(',')) {
        const path = entry.trim().split(' ')[0];
        if (!existsSync(join(dir, path))) {
          problems.push(`${lang}: ${path} is missing`);
        }
      }
    }
  }
  for (const file of ['app.html', 'robots.txt', 'sitemap.xml', 'llms.txt']) {
    if (!existsSync(join(dir, file))) {
      problems.push(`${file} is missing`);
    }
  }
  return problems;
}
