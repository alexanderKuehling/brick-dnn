/**
 * Checks a build (dist/) for what the static pages need, run at the end of every build
 * (plugin.ts) and by landing.test.ts. Returns the problems; empty means fine.
 */
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { allPages, type PageMeta } from './pages.ts';
import { absolute, INLINE_SCRIPT, PAGE_PATHS, SITE_URL, type Lang } from './site.ts';

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

/** The hreflang links of a page: language → path. */
export function hreflangs(html: string): Record<string, string> {
  const links: Record<string, string> = {};
  for (const [, lang, href] of html.matchAll(
    /<link rel="alternate" hreflang="([^"]+)" href="([^"]+)">/g
  )) {
    links[lang] = href.startsWith(SITE_URL) ? href.slice(SITE_URL.length) : href;
  }
  return links;
}

function unescape(text: string): string {
  return text
    .replace(/&quot;/g, '"')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&amp;/g, '&');
}

/** What every static page needs: language, title, description, canonical, hreflang, JSON-LD. */
export function checkPageHtml(page: PageMeta, html: string): string[] {
  const problems: string[] = [];
  const need = (ok: boolean, what: string) => ok || problems.push(`${page.path}: ${what}`);
  need(html.includes(`<html lang="${page.lang}"`), `<html lang="${page.lang}">`);
  const title = html.match(/<title>([^<]*)<\/title>/)?.[1];
  need(Boolean(title) && unescape(title!) === page.title, 'its <title>');
  const description = html.match(/<meta name="description" content="([^"]*)">/)?.[1];
  need(Boolean(description) && unescape(description!) === page.description, 'its meta description');
  need(html.includes(`<link rel="canonical" href="${absolute(page.path)}">`), 'canonical link');
  const links = hreflangs(html);
  if (page.alternates.en && page.alternates.de) {
    for (const [hreflang, path] of [
      ['en', page.alternates.en],
      ['de', page.alternates.de],
      ['x-default', page.alternates.en],
    ]) {
      need(links[hreflang] === path, `hreflang ${hreflang}`);
    }
    need(page.alternates[page.lang] === page.path, 'hreflang of itself');
  } else {
    need(Object.keys(links).length === 0, 'no hreflang without a counterpart');
  }
  const types = jsonLdTypes(html);
  need(types !== null, 'valid JSON-LD');
  for (const type of page.types) {
    need(Boolean(types?.includes(type)), `JSON-LD ${type}`);
  }
  need((html.match(/<h1[\s>]/g) ?? []).length === 1, 'exactly one <h1>');
  need(html.includes('href="/imprint"') && html.includes('href="/privacy"'), 'legal links');
  // Only the landing pages have a script (the one nginx.conf allows by its hash).
  const scripts = [...html.matchAll(/<script(?: type="([^"]*)")?>/g)].filter(
    ([, type]) => type !== 'application/ld+json'
  );
  need(scripts.length === (page.kind === 'home' ? 1 : 0), 'no other script');
  return [...new Set(problems)];
}

/** The landing pages' own checks on top of checkPageHtml. */
export function checkLandingHtml(lang: Lang, html: string): string[] {
  const page = allPages().find((candidate) => candidate.path === PAGE_PATHS[lang])!;
  const problems = checkPageHtml(page, html).map((problem) =>
    problem.replace(`${page.path}: `, `${lang}: `)
  );
  const need = (ok: boolean, what: string) => ok || problems.push(`${lang}: ${what}`);
  need(html.includes(`<script>${INLINE_SCRIPT}</script>`), 'the inline script of site.ts');
  return [...new Set(problems)];
}

/** The pages listed in sitemap.xml. */
export function sitemapLocs(xml: string): string[] {
  return [...xml.matchAll(/<loc>([^<]+)<\/loc>/g)].map(([, loc]) => loc);
}

export function checkDist(dir: string, pages: PageMeta[] = allPages()): string[] {
  const problems: string[] = [];
  const found = new Map<string, Record<string, string>>();
  for (const page of pages) {
    const file = join(dir, page.file);
    if (!existsSync(file)) {
      problems.push(`${page.file} is missing`);
      continue;
    }
    const html = readFileSync(file, 'utf8');
    problems.push(
      ...(page.kind === 'home' ? checkLandingHtml(page.lang, html) : checkPageHtml(page, html))
    );
    found.set(page.path, hreflangs(html));
    // Every file of this site that the page refers to is in the build.
    for (const [, path] of html.matchAll(
      /(?:src|href|srcset|content)="(\/[^"#?\s]+\.[a-z0-9]+)/g
    )) {
      if (!existsSync(join(dir, path))) {
        problems.push(`${page.path}: ${path} is missing`);
      }
    }
    for (const [, list] of html.matchAll(/srcset="([^"]+)"/g)) {
      for (const entry of list.split(',')) {
        const path = entry.trim().split(' ')[0];
        if (!existsSync(join(dir, path))) {
          problems.push(`${page.path}: ${path} is missing`);
        }
      }
    }
  }
  // The language versions of a page point at each other.
  for (const [path, links] of found) {
    for (const lang of ['en', 'de']) {
      const other = links[lang];
      if (other && other !== path && found.has(other)) {
        const back = Object.values(found.get(other)!);
        if (!back.includes(path)) {
          problems.push(`${path}: ${other} has no hreflang link back to it`);
        }
      }
    }
  }
  for (const file of ['app.html', 'robots.txt', 'sitemap.xml', 'llms.txt']) {
    if (!existsSync(join(dir, file))) {
      problems.push(`${file} is missing`);
    }
  }
  if (existsSync(join(dir, 'sitemap.xml'))) {
    const locs = new Set(sitemapLocs(readFileSync(join(dir, 'sitemap.xml'), 'utf8')));
    for (const page of pages) {
      if (!locs.has(absolute(page.path))) {
        problems.push(`sitemap.xml lacks ${page.path}`);
      }
    }
  }
  return [...new Set(problems)];
}
