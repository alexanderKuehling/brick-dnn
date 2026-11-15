/**
 * The landing pages and the files for crawlers (robots.txt, sitemap.xml, llms.txt) as text.
 * Pure functions: plugin.ts writes their output into the build and serves it in development.
 */
import { de } from './content/de.ts';
import { en } from './content/en.ts';
import type { Item, LandingContent } from './content/types.ts';
import {
  absolute,
  APP_PAGES,
  CONTACT_EMAIL,
  CURRENCY,
  DOCS_URL,
  EDITOR_PATH,
  INLINE_SCRIPT,
  LICENSE_URL,
  PAGE_PATHS,
  PLANS,
  REPOSITORY_URL,
  SITE_URL,
  type Lang,
} from './site.ts';

export const CONTENT: Record<Lang, LandingContent> = { en, de };

/** The addresses of the files a page refers to (all on this site). */
export interface Assets {
  css: string;
  favicon: string;
  /** The PNG logo for the Organization in the JSON-LD. */
  logo: { src: string; width: number; height: number };
  /** The screenshots by id (see Shot), each in several widths. */
  images: Record<string, ImageAsset>;
  ogImage: { src: string; width: number; height: number };
}

export interface ImageAsset {
  /** Narrowest first. */
  sources: { src: string; width: number; height: number }[];
}

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

/** Where an image is shown, for `sizes`: the width it takes on the page. */
const SIZES = {
  hero: '(min-width: 72rem) 68rem, calc(100vw - 2.5rem)',
  row: '(min-width: 72rem) 37rem, (min-width: 60rem) 55vw, calc(100vw - 2.5rem)',
  tile: '(min-width: 72rem) 33rem, (min-width: 52rem) 45vw, calc(100vw - 2.5rem)',
  step: '(min-width: 60rem) 21rem, calc(100vw - 2.5rem)',
};

/**
 * An <img> of a screenshot with all its widths. The width and height attributes are those of
 * the file closest to `display` (they give the proportions, CSS sets the size). Below the fold
 * it loads lazily.
 */
export function picture(
  asset: ImageAsset,
  alt: string,
  sizes: keyof typeof SIZES,
  display: number,
  eager = false
): string {
  const srcset = asset.sources.map((s) => `${s.src} ${s.width}w`).join(', ');
  const base =
    asset.sources.find((s) => s.width >= display) ?? asset.sources[asset.sources.length - 1];
  const loading = eager ? 'fetchpriority="high"' : 'loading="lazy"';
  return `<img src="${base.src}" srcset="${srcset}" sizes="${SIZES[sizes]}" width="${base.width}" height="${base.height}" alt="${escapeHtml(alt)}" ${loading} decoding="async">`;
}

export function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/** The plain text of an HTML snippet (for meta data and the JSON-LD). */
export function plainText(html: string): string {
  return html
    .replace(/<q>/g, '“')
    .replace(/<\/q>/g, '”')
    .replace(/<[^>]+>/g, '')
    .replace(/&quot;/g, '"')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&amp;/g, '&')
    .replace(/\s+/g, ' ')
    .trim();
}

const other = (lang: Lang): Lang => (lang === 'en' ? 'de' : 'en');

function alternates(): string {
  return [
    `<link rel="alternate" hreflang="en" href="${absolute(PAGE_PATHS.en)}">`,
    `<link rel="alternate" hreflang="de" href="${absolute(PAGE_PATHS.de)}">`,
    `<link rel="alternate" hreflang="x-default" href="${absolute(PAGE_PATHS.en)}">`,
  ].join('\n');
}

/** JSON for a <script> element: `<` escaped, so the text can never end the element. */
function scriptJson(value: unknown): string {
  return JSON.stringify(value).replace(/</g, '\\u003c');
}

export function structuredData(c: LandingContent, assets: Assets): object {
  const page = absolute(PAGE_PATHS[c.lang]);
  const organization = `${SITE_URL}/#organization`;
  const website = `${SITE_URL}/#website`;
  const software = `${SITE_URL}/#software`;
  return {
    '@context': 'https://schema.org',
    '@graph': [
      {
        '@type': 'Organization',
        '@id': organization,
        name: 'Brick DNN',
        url: `${SITE_URL}/`,
        logo: {
          '@type': 'ImageObject',
          url: absolute(assets.logo.src),
          width: assets.logo.width,
          height: assets.logo.height,
        },
        email: CONTACT_EMAIL,
        sameAs: [REPOSITORY_URL],
      },
      {
        '@type': 'WebSite',
        '@id': website,
        url: `${SITE_URL}/`,
        name: 'Brick DNN',
        inLanguage: ['en', 'de'],
        publisher: { '@id': organization },
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
        publisher: { '@id': organization },
        offers: PLANS.map((plan) => ({
          '@type': 'Offer',
          name: plan.name,
          price: String(plan.price),
          priceCurrency: CURRENCY,
          url: absolute('/pricing'),
          priceSpecification: {
            '@type': 'UnitPriceSpecification',
            price: String(plan.price),
            priceCurrency: CURRENCY,
            billingDuration: 'P1M',
            unitText: 'month',
            valueAddedTaxIncluded: true,
          },
        })),
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
${alternates()}
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

function brand(c: LandingContent, assets: Assets): string {
  return `<a class="brand" href="${PAGE_PATHS[c.lang]}"><img src="${assets.favicon}" width="28" height="28" alt=""><span>Brick DNN</span></a>`;
}

function languageSwitch(c: LandingContent, label: string): string {
  const links = (['en', 'de'] as const).map((lang) => {
    const name = CONTENT[lang].languageName;
    const current = lang === c.lang ? ' aria-current="page"' : '';
    return `<li><a href="${PAGE_PATHS[lang]}" hreflang="${lang}" lang="${lang}"${current}><span aria-hidden="true">${lang.toUpperCase()}</span><span class="sr-only">${name}</span></a></li>`;
  });
  return `<nav class="lang" aria-label="${escapeHtml(label)}"><ul>${links.join('')}</ul></nav>`;
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
  const legal = c.footer.legal;

  return `<body>
<a class="skip" href="#main">${escapeHtml(c.skipLink)}</a>
<header class="site-header"><div class="wrap bar">
${brand(c, assets)}
<nav class="sections" aria-label="${escapeHtml(c.nav.label)}"><ul>
<li><a href="#use-cases">${escapeHtml(c.nav.useCases)}</a></li>
<li><a href="#features">${escapeHtml(c.nav.features)}</a></li>
<li><a href="#pricing">${escapeHtml(c.nav.pricing)}</a></li>
<li><a href="#faq">${escapeHtml(c.nav.faq)}</a></li>
</ul></nav>
<div class="actions">
${languageSwitch(c, c.footer.language)}
<a class="button small" href="${EDITOR_PATH}">${escapeHtml(c.nav.editor)}</a>
</div>
</div></header>

<main id="main" tabindex="-1">
<section class="hero" aria-labelledby="hero-title"><div class="wrap">
<h1 id="hero-title">${escapeHtml(c.hero.title)}</h1>
<p class="lead">${escapeHtml(c.hero.lead)}</p>
<p class="cta"><a class="button" href="${EDITOR_PATH}">${escapeHtml(c.hero.primary)}</a> <a class="button secondary" href="/pricing">${escapeHtml(c.hero.secondary)}</a></p>
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
<p class="more"><a class="button secondary" href="/pricing">${escapeHtml(c.pricing.more)}</a></p>`
)}

${section(
  'faq',
  c.faq.title,
  `<div class="faq">${c.faq.items
    .map((item) => `<div><h3>${escapeHtml(item.question)}</h3><p>${item.html}</p></div>`)
    .join('\n')}</div>`
)}
</main>

<footer class="site-footer"><div class="wrap">
<div class="about">${brand(c, assets)}<p>${escapeHtml(c.footer.tagline)}</p></div>
<nav aria-label="${escapeHtml(c.footer.legalLabel)}"><ul>
<li><a href="/imprint">${escapeHtml(legal.imprint)}</a></li>
<li><a href="/privacy">${escapeHtml(legal.privacy)}</a></li>
<li><a href="/terms">${escapeHtml(legal.terms)}</a></li>
<li><a href="/refund">${escapeHtml(legal.refund)}</a></li>
<li><a href="/accessibility">${escapeHtml(legal.accessibility)}</a></li>
</ul>${c.footer.legalNote ? `<p class="note">${escapeHtml(c.footer.legalNote)}</p>` : ''}</nav>
<ul class="links">
<li><a href="${REPOSITORY_URL}">${escapeHtml(c.footer.source)}</a></li>
<li><a href="mailto:${CONTACT_EMAIL}">${escapeHtml(c.footer.contact)}: ${CONTACT_EMAIL}</a></li>
</ul>
${languageSwitch(c, c.footer.language)}
</div></footer>
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

export function renderSitemap(): string {
  const links = alternates().replace(/<link rel="alternate"/g, '<xhtml:link rel="alternate"');
  const landing = (['en', 'de'] as const).map(
    (lang) =>
      `<url><loc>${absolute(PAGE_PATHS[lang])}</loc>\n${links.replace(/">/g, '"/>')}\n</url>`
  );
  const pages = APP_PAGES.map((path) => `<url><loc>${absolute(path)}</loc></url>`);
  return `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9" xmlns:xhtml="http://www.w3.org/1999/xhtml">
${[...landing, ...pages].join('\n')}
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

## Links

- [Editor](${absolute(EDITOR_PATH)}): the web app
- [Pricing](${absolute('/pricing')})
- [Landing page in German](${absolute(PAGE_PATHS.de)})
- [Documentation](${DOCS_URL}): training, data preparation, MCP server, REST API
- [MCP server](${REPOSITORY_URL}/blob/main/docs/mcp.md): connecting AI assistants
- [Source code](${REPOSITORY_URL}) (Apache 2.0)
- [Privacy policy](${absolute('/privacy')}) (German), [imprint](${absolute('/imprint')}), [terms](${absolute('/terms')})
- Contact: ${CONTACT_EMAIL}
`;
}
