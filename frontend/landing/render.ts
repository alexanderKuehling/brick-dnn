/**
 * The landing pages and the files for crawlers (robots.txt, sitemap.xml, llms.txt) as text.
 * Pure functions: plugin.ts writes their output into the build and serves it in development.
 */
import { de } from './content/de.ts';
import { en } from './content/en.ts';
import type { Item, LandingContent } from './content/types.ts';
import { escapeHtml, picture, plainText, scriptJson, type Assets } from './html.ts';
import { offers, organization, ORGANIZATION_ID, SOFTWARE_ID, WEBSITE_ID } from './jsonld.ts';
import { alternateLinks, siteFooter, siteHeader, type Chrome } from './layout.ts';
import { allPages, llmsSections } from './pages.ts';
import {
  absolute,
  APP_PAGES,
  CONTACT_EMAIL,
  DOCS_URL,
  EDITOR_PATH,
  INLINE_SCRIPT,
  LICENSE_URL,
  PAGE_PATHS,
  PLANS,
  pricingPath,
  REPOSITORY_URL,
  SITE_URL,
  type Lang,
} from './site.ts';

export { escapeHtml, picture, plainText, type Assets, type ImageAsset } from './html.ts';

export const CONTENT: Record<Lang, LandingContent> = { en, de };

/** The screenshot ids a page shows: the hero's and those of the sections' items. */
export function shotIds(c: LandingContent): string[] {
  const ids = [c.hero.image.id];
  for (const item of [...c.useCases.items, ...c.steps.items, ...c.features.items]) {
    if (item.image) {
      ids.push(item.image.id);
    }
  }
  return [...new Set(ids)];
}

const other = (lang: Lang): Lang => (lang === 'en' ? 'de' : 'en');

export function structuredData(c: LandingContent, assets: Assets): object {
  const page = absolute(PAGE_PATHS[c.lang]);
  const website = WEBSITE_ID;
  const software = SOFTWARE_ID;
  return {
    '@context': 'https://schema.org',
    '@graph': [
      organization(assets),
      {
        '@type': 'WebSite',
        '@id': website,
        url: `${SITE_URL}/`,
        name: 'Brick DNN',
        inLanguage: ['en', 'de'],
        publisher: { '@id': ORGANIZATION_ID },
      },
      {
        '@type': 'SoftwareApplication',
        '@id': software,
        name: 'Brick DNN',
        url: absolute(EDITOR_PATH),
        description: c.appDescription,
        applicationCategory: 'DeveloperApplication',
        applicationSubCategory: 'Machine learning',
        operatingSystem: 'Web browser',
        inLanguage: 'en',
        isAccessibleForFree: true,
        license: LICENSE_URL,
        codeRepository: REPOSITORY_URL,
        screenshot: shotIds(c).map((id) => absolute(assets.images[id].sources.at(-1)!.src)),
        featureList: c.featureList,
        publisher: { '@id': ORGANIZATION_ID },
        offers: offers(),
      },
      {
        '@type': 'FAQPage',
        '@id': page,
        url: page,
        name: c.title,
        description: c.description,
        inLanguage: c.lang,
        isPartOf: { '@id': website },
        about: { '@id': software },
        mainEntity: c.faq.items.map((item) => ({
          '@type': 'Question',
          name: item.question,
          acceptedAnswer: { '@type': 'Answer', text: plainText(item.html) },
        })),
      },
    ],
  };
}

function head(c: LandingContent, assets: Assets): string {
  const page = absolute(PAGE_PATHS[c.lang]);
  const alt = CONTENT[other(c.lang)];
  const og = absolute(assets.ogImage.src);
  return `<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<script>${INLINE_SCRIPT}</script>
<title>${escapeHtml(c.title)}</title>
<meta name="description" content="${escapeHtml(c.description)}">
<link rel="canonical" href="${page}">
${alternateLinks(PAGE_PATHS)}
<meta name="color-scheme" content="light dark">
<link rel="icon" type="image/svg+xml" href="${assets.favicon}">
<link rel="stylesheet" href="${assets.css}">
<meta property="og:type" content="website">
<meta property="og:site_name" content="Brick DNN">
<meta property="og:title" content="${escapeHtml(c.title)}">
<meta property="og:description" content="${escapeHtml(c.description)}">
<meta property="og:url" content="${page}">
<meta property="og:locale" content="${c.locale}">
<meta property="og:locale:alternate" content="${alt.locale}">
<meta property="og:image" content="${og}">
<meta property="og:image:type" content="image/jpeg">
<meta property="og:image:width" content="${assets.ogImage.width}">
<meta property="og:image:height" content="${assets.ogImage.height}">
<meta property="og:image:alt" content="${escapeHtml(c.ogImageAlt)}">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="${escapeHtml(c.title)}">
<meta name="twitter:description" content="${escapeHtml(c.description)}">
<meta name="twitter:image" content="${og}">
<meta name="twitter:image:alt" content="${escapeHtml(c.ogImageAlt)}">
<script type="application/ld+json">${scriptJson(structuredData(c, assets))}</script>
</head>`;
}

function section(id: string, title: string, body: string, extraClass = ''): string {
  const cls = extraClass ? ` class="${extraClass}"` : '';
  return `<section id="${id}" aria-labelledby="${id}-title"${cls}><div class="wrap">
<h2 id="${id}-title">${escapeHtml(title)}</h2>
${body}
</div></section>`;
}

function items(list: Item[], cls: string, tag = 'ul'): string {
  return `<${tag} class="${cls}">${list
    .map((item) => `<li><h3>${escapeHtml(item.title)}</h3><p>${item.html}</p></li>`)
    .join('\n')}</${tag}>`;
}

/**
 * Items with a screenshot each: `rows` (text and image side by side, alternating), `steps`
 * (the image under the text) or `tiles` (the image above).
 */
function illustrated(list: Item[], kind: 'rows' | 'steps' | 'tiles', assets: Assets): string {
  const tag = kind === 'steps' ? 'ol' : 'ul';
  const sizes = kind === 'rows' ? 'row' : kind === 'steps' ? 'step' : 'tile';
  const display = kind === 'rows' ? 592 : kind === 'steps' ? 336 : 528;
  return `<${tag} class="${kind}">${list
    .map((item) => {
      const image = item.image
        ? `<figure class="frame">${picture(assets.images[item.image.id], item.image.alt, sizes, display)}</figure>`
        : '';
      const text = `<h3>${escapeHtml(item.title)}</h3><p>${item.html}</p>`;
      return kind === 'tiles'
        ? `<li>${image}${text}</li>`
        : `<li><div class="text">${text}</div>${image}</li>`;
    })
    .join('\n')}</${tag}>`;
}

function body(c: LandingContent, assets: Assets): string {
  const withImage = c.features.items.filter((item) => item.image);
  const withoutImage = c.features.items.filter((item) => !item.image);
  const pricingRows = PLANS.map((plan) => {
    const [name, ...cells] = c.pricing.row(plan);
    return `<tr><th scope="row">${escapeHtml(name)}</th>${cells
      .map((cell) => `<td>${escapeHtml(cell)}</td>`)
      .join('')}</tr>`;
  }).join('\n');
  const chrome: Chrome = {
    lang: c.lang,
    alternates: PAGE_PATHS,
    area: 'home',
    favicon: assets.favicon,
  };

  return `<body>
<a class="skip" href="#main">${escapeHtml(c.skipLink)}</a>
${siteHeader(chrome, [
  { href: '#use-cases', label: c.nav.useCases },
  { href: '#features', label: c.nav.features },
  { href: '#pricing', label: c.nav.pricing },
  { href: '#faq', label: c.nav.faq },
])}

<main id="main" tabindex="-1">
<section class="hero" aria-labelledby="hero-title"><div class="wrap">
<h1 id="hero-title">${escapeHtml(c.hero.title)}</h1>
<p class="lead">${escapeHtml(c.hero.lead)}</p>
<p class="cta"><a class="button" href="${EDITOR_PATH}">${escapeHtml(c.hero.primary)}</a> <a class="button secondary" href="${pricingPath(c.lang)}">${escapeHtml(c.hero.secondary)}</a></p>
<ul class="facts">${c.hero.facts.map((fact) => `<li>${escapeHtml(fact)}</li>`).join('')}</ul>
<figure class="shot">
${picture(assets.images[c.hero.image.id], c.hero.image.alt, 'hero', 1280, true)}
<figcaption>${escapeHtml(c.hero.caption)}</figcaption>
</figure>
</div></section>

${section(
  'at-a-glance',
  c.glance.title,
  `<dl class="glance">${c.glance.items
    .map((item) => `<div><dt>${escapeHtml(item.term)}</dt><dd>${item.html}</dd></div>`)
    .join('\n')}</dl>`
)}

${section(
  'use-cases',
  c.useCases.title,
  `<p class="intro">${escapeHtml(c.useCases.intro)}</p>
${illustrated(c.useCases.items, 'rows', assets)}`
)}

${section('how-it-works', c.steps.title, illustrated(c.steps.items, 'steps', assets))}

${section(
  'features',
  c.features.title,
  `<p class="intro">${escapeHtml(c.features.intro)}</p>
${illustrated(withImage, 'tiles', assets)}
${items(withoutImage, 'grid')}`
)}

${section(
  'privacy',
  c.privacy.title,
  `<p class="intro">${escapeHtml(c.privacy.intro)}</p>
${items(c.privacy.items, 'grid two')}
<p class="more"><a href="/privacy">${escapeHtml(c.privacy.more)}</a></p>`
)}

${section(
  'teaching',
  c.teaching.title,
  `<ul class="checks">${c.teaching.items.map((item) => `<li>${escapeHtml(item)}</li>`).join('\n')}</ul>
<p class="discount">${c.teaching.discount}</p>`,
  'tinted'
)}

${section(
  'pricing',
  c.pricing.title,
  `<p class="intro">${escapeHtml(c.pricing.intro)}</p>
<div class="table" role="region" aria-labelledby="pricing-caption" tabindex="0">
<table>
<caption id="pricing-caption">${escapeHtml(c.pricing.caption)}</caption>
<thead><tr>${c.pricing.columns.map((column) => `<th scope="col">${escapeHtml(column)}</th>`).join('')}</tr></thead>
<tbody>
${pricingRows}
</tbody>
</table>
</div>
<ul class="notes">${c.pricing.notes.map((note) => `<li>${escapeHtml(note)}</li>`).join('\n')}</ul>
<p class="more"><a class="button secondary" href="${pricingPath(c.lang)}">${escapeHtml(c.pricing.more)}</a></p>`
)}

${section(
  'faq',
  c.faq.title,
  `<div class="faq">${c.faq.items
    .map((item) => `<div><h3>${escapeHtml(item.question)}</h3><p>${item.html}</p></div>`)
    .join('\n')}</div>`
)}
</main>

${siteFooter(chrome)}
</body>`;
}

export function renderPage(lang: Lang, assets: Assets): string {
  const c = CONTENT[lang];
  return `<!doctype html>
<html lang="${lang}" dir="ltr">
${head(c, assets)}
${body(c, assets)}
</html>
`;
}

export function renderRobots(): string {
  return `# Brick DNN (frontend/landing/render.ts)
User-agent: *
Disallow: /api/
Disallow: /admin/
Disallow: /oauth/
Disallow: /share/
Disallow: /account
Disallow: /tokens
Disallow: /activity
Disallow: /reset-password
Disallow: /verify-email

Sitemap: ${absolute('/sitemap.xml')}
`;
}

/** Every page of the site, each with its other language where there is one, and the app's. */
export function renderSitemap(): string {
  const pages = allPages().map((page) => {
    const links = alternateLinks(page.alternates)
      .replace(/<link rel="alternate"/g, '<xhtml:link rel="alternate"')
      .replace(/">/g, '"/>');
    return `<url><loc>${absolute(page.path)}</loc>${links ? `\n${links}\n` : ''}</url>`;
  });
  const app = APP_PAGES.map((path) => `<url><loc>${absolute(path)}</loc></url>`);
  return `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9" xmlns:xhtml="http://www.w3.org/1999/xhtml">
${[...pages, ...app].join('\n')}
</urlset>
`;
}

/** A summary for AI assistants and search (https://llmstxt.org), from the English page. */
export function renderLlms(): string {
  const c = en;
  const list = (entries: { title: string; html: string }[]) =>
    entries.map((item) => `- **${item.title}**: ${plainText(item.html)}`).join('\n');
  const columns = c.pricing.columns;
  const plans = PLANS.map((plan) => {
    const [name, price, compute, storage, longest, parallel] = c.pricing.row(plan);
    return `- **${name}**: ${price} per month; ${compute} of compute per month, ${storage} storage, ${columns[4].toLowerCase()} ${longest}, ${parallel} ${parallel === '1' ? 'training' : 'trainings'} at a time`;
  }).join('\n');
  return `# Brick DNN

> ${c.appDescription}

${c.glance.items.map((item) => `- **${item.term}**: ${plainText(item.html)}`).join('\n')}

## Use cases

${list(c.useCases.items)}

## How it works

${c.steps.items.map((item, index) => `${index + 1}. **${item.title}**: ${plainText(item.html)}`).join('\n')}

## Features

${list(c.features.items)}

## Pricing

${plainText(c.pricing.intro)} Prices in EUR per month including VAT.

${plans}

${c.pricing.notes.map((note) => `- ${note}`).join('\n')}

## Privacy and data location

${list(c.privacy.items)}

## Teaching and research

${c.teaching.items.map((item) => `- ${item}`).join('\n')}
- ${plainText(c.teaching.discount)}

## FAQ

${c.faq.items.map((item) => `### ${item.question}\n\n${plainText(item.html)}`).join('\n\n')}

${llmsSections()}

## Links

- [Editor](${absolute(EDITOR_PATH)}): the web app
- [Pricing](${absolute('/pricing')})
- [Landing page in German](${absolute(PAGE_PATHS.de)})
- [Pricing in German](${absolute(pricingPath('de'))})
- [Documentation](${DOCS_URL}): training, data preparation, MCP server, REST API
- [MCP server](${REPOSITORY_URL}/blob/main/docs/mcp.md): connecting AI assistants
- [Source code](${REPOSITORY_URL}) (Apache 2.0)
- [Privacy policy](${absolute('/privacy')}) (German), [imprint](${absolute('/imprint')}), [terms](${absolute('/terms')})
- Contact: ${CONTACT_EMAIL}
`;
}
