/**
 * The parts every page of the site shares: the header with the navigation, the footer, the
 * language switch and the breadcrumbs. The landing pages (render.ts) and the content pages
 * (pages.ts) both use them, so the navigation is the same everywhere.
 */
import { COMMON } from './content/common.ts';
import { de } from './content/de.ts';
import { en } from './content/en.ts';
import type { LandingContent } from './content/types.ts';
import { escapeHtml } from './html.ts';
import {
  absolute,
  CONTACT_EMAIL,
  EDITOR_PATH,
  EXAMPLES_PATHS,
  GUIDES_PATHS,
  PAGE_PATHS,
  pricingPath,
  REPOSITORY_URL,
  type Lang,
} from './site.ts';

const LANDING: Record<Lang, LandingContent> = { en, de };

/** The part of the site a page belongs to, marked in the navigation. */
export type Area = 'home' | 'examples' | 'guides' | 'pricing';

export interface Chrome {
  lang: Lang;
  /** The address of this page in each language, for the language switch; a language without
   * one leads to `fallback` (or its landing page). */
  alternates: Partial<Record<Lang, string>>;
  fallback?: Partial<Record<Lang, string>>;
  area: Area;
  favicon: string;
}

/** The links of the language switch: the page in the other language, or its nearest page. */
export function switchTargets(chrome: Chrome): Record<Lang, string> {
  const pick = (lang: Lang) =>
    chrome.alternates[lang] ?? chrome.fallback?.[lang] ?? PAGE_PATHS[lang];
  return { en: pick('en'), de: pick('de') };
}

export function brand(lang: Lang, favicon: string): string {
  return `<a class="brand" href="${PAGE_PATHS[lang]}"><img src="${favicon}" width="28" height="28" alt=""><span>Brick DNN</span></a>`;
}

export function languageSwitch(lang: Lang, label: string, targets: Record<Lang, string>): string {
  const links = (['en', 'de'] as const).map((other) => {
    const name = LANDING[other].languageName;
    const current = other === lang ? ' aria-current="page"' : '';
    return `<li><a href="${targets[other]}" hreflang="${other}" lang="${other}"${current}><span aria-hidden="true">${other.toUpperCase()}</span><span class="sr-only">${name}</span></a></li>`;
  });
  return `<nav class="lang" aria-label="${escapeHtml(label)}"><ul>${links.join('')}</ul></nav>`;
}

/** The pages of the site that every page links to (in the header and the footer). */
export function siteLinks(lang: Lang): { area: Area; href: string; label: string }[] {
  const t = COMMON[lang].nav;
  return [
    { area: 'examples', href: EXAMPLES_PATHS[lang], label: t.examples },
    { area: 'guides', href: GUIDES_PATHS[lang], label: t.guides },
    { area: 'pricing', href: pricingPath(lang), label: t.pricing },
  ];
}

/**
 * The header. `local` are links to sections of the same page (the landing page's), shown on
 * wide screens before the site's own links, which are always shown.
 */
export function siteHeader(chrome: Chrome, local: { href: string; label: string }[] = []): string {
  const c = LANDING[chrome.lang];
  const t = COMMON[chrome.lang];
  // The landing page has its own pricing section; the site's pricing link is for the others.
  const site = siteLinks(chrome.lang).filter((link) => !(local.length && link.area === 'pricing'));
  const items = [
    ...local.map(
      (link) => `<li class="local"><a href="${link.href}">${escapeHtml(link.label)}</a></li>`
    ),
    ...site.map((link) => {
      const current = link.area === chrome.area ? ' aria-current="page"' : '';
      return `<li><a href="${link.href}"${current}>${escapeHtml(link.label)}</a></li>`;
    }),
  ];
  return `<header class="site-header"><div class="wrap bar">
${brand(chrome.lang, chrome.favicon)}
<nav class="sections" aria-label="${escapeHtml(t.nav.label)}"><ul>
${items.join('\n')}
</ul></nav>
<div class="actions">
${languageSwitch(chrome.lang, c.footer.language, switchTargets(chrome))}
<a class="button small" href="${EDITOR_PATH}">${escapeHtml(c.nav.editor)}</a>
</div>
</div></header>`;
}

export function siteFooter(chrome: Chrome): string {
  const c = LANDING[chrome.lang];
  const t = COMMON[chrome.lang];
  const legal = c.footer.legal;
  return `<footer class="site-footer"><div class="wrap">
<div class="about">${brand(chrome.lang, chrome.favicon)}<p>${escapeHtml(c.footer.tagline)}</p></div>
<nav aria-label="${escapeHtml(t.nav.siteLabel)}"><ul>
${siteLinks(chrome.lang)
  .map((link) => `<li><a href="${link.href}">${escapeHtml(link.label)}</a></li>`)
  .join('\n')}
<li><a href="${EDITOR_PATH}">${escapeHtml(t.nav.editor)}</a></li>
</ul></nav>
<nav aria-label="${escapeHtml(c.footer.legalLabel)}"><ul>
<li><a href="/imprint">${escapeHtml(legal.imprint)}</a></li>
<li><a href="/privacy">${escapeHtml(legal.privacy)}</a></li>
<li><a href="/terms">${escapeHtml(legal.terms)}</a></li>
<li><a href="/refund">${escapeHtml(legal.refund)}</a></li>
<li><a href="/accessibility">${escapeHtml(legal.accessibility)}</a></li>
</ul>${c.footer.legalNote ? `<p class="note">${escapeHtml(c.footer.legalNote)}</p>` : ''}</nav>
<div class="links"><ul>
<li><a href="${REPOSITORY_URL}">${escapeHtml(c.footer.source)}</a></li>
<li><a href="mailto:${CONTACT_EMAIL}">${escapeHtml(c.footer.contact)}: ${CONTACT_EMAIL}</a></li>
</ul>
${languageSwitch(chrome.lang, c.footer.language, switchTargets(chrome))}
</div>
</div></footer>`;
}

/**
 * The hreflang links of a page that exists in both languages (English is the x-default);
 * nothing for a page without a counterpart.
 */
export function alternateLinks(alternates: Partial<Record<Lang, string>>): string {
  if (!alternates.en || !alternates.de) {
    return '';
  }
  return [
    `<link rel="alternate" hreflang="en" href="${absolute(alternates.en)}">`,
    `<link rel="alternate" hreflang="de" href="${absolute(alternates.de)}">`,
    `<link rel="alternate" hreflang="x-default" href="${absolute(alternates.en)}">`,
  ].join('\n');
}

export interface Crumb {
  name: string;
  /** The page's path; the last crumb is the page itself. */
  path: string;
}

/** The visible breadcrumbs (the JSON-LD BreadcrumbList has the same entries). */
export function breadcrumbs(lang: Lang, crumbs: Crumb[]): string {
  const items = crumbs.map((crumb, index) =>
    index === crumbs.length - 1
      ? `<li><a href="${crumb.path}" aria-current="page">${escapeHtml(crumb.name)}</a></li>`
      : `<li><a href="${crumb.path}">${escapeHtml(crumb.name)}</a></li>`
  );
  return `<nav class="crumbs" aria-label="${escapeHtml(COMMON[lang].nav.breadcrumbs)}"><ol>${items.join('')}</ol></nav>`;
}
