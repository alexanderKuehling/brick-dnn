/**
 * The static content pages, in English and German, made by the same generator as the landing
 * pages (plugin.ts writes them into the build):
 *
 * - the example gallery (/examples/, /de/beispiele/) and one page per built-in example, with its
 *   diagram, its layers with shapes and parameter counts and the generated code, all from
 *   examples/examples.json (made by the backend from the real example graphs and generator);
 * - the guides (/guides/, /de/anleitungen/);
 * - the German pricing page (/de/preise/). English has none: /pricing is the app's page.
 *
 * Like the landing pages they are plain HTML and CSS without any script, and load nothing from
 * elsewhere.
 */
import { COMMON } from './content/common.ts';
import { de as landingDe } from './content/de.ts';
import { en as landingEn } from './content/en.ts';
import { examplesDe } from './content/examples.de.ts';
import { examplesEn } from './content/examples.en.ts';
import { guidesDe } from './content/guides.de.ts';
import { guidesEn } from './content/guides.en.ts';
import { pricingDe } from './content/pricing.de.ts';
import type {
  ExampleCopy,
  ExampleGallery,
  Guide,
  GuideIndex,
  LandingContent,
} from './content/types.ts';
import {
  exampleById,
  EXAMPLES,
  exampleSlug,
  trainableParams,
  type ExampleData,
} from './examples/data.ts';
import { describeGraph, diagramSvg, formatCount, formatShape } from './examples/diagram.ts';
import { highlightPython } from './highlight.ts';
import { escapeHtml, picture, plainText, scriptJson, type Assets } from './html.ts';
import {
  author,
  breadcrumbList,
  offers,
  organization,
  ORGANIZATION_ID,
  SOFTWARE_ID,
  WEBSITE_ID,
} from './jsonld.ts';
import {
  alternateLinks,
  breadcrumbs,
  siteFooter,
  siteHeader,
  type Area,
  type Chrome,
  type Crumb,
} from './layout.ts';
import {
  absolute,
  AUTHOR,
  EDITOR_PATH,
  editorExampleUrl,
  EXAMPLES_PATHS,
  GUIDES_PATHS,
  LICENSE_URL,
  PAGE_PATHS,
  PLANS,
  PRICING_DE_PATH,
  type Lang,
} from './site.ts';

const LANGS = ['en', 'de'] as const;
const LANDING: Record<Lang, LandingContent> = { en: landingEn, de: landingDe };
export const EXAMPLE_COPY: Record<Lang, { gallery: ExampleGallery; examples: ExampleCopy[] }> = {
  en: examplesEn,
  de: examplesDe,
};
export const GUIDE_COPY: Record<Lang, { index: GuideIndex; guides: Guide[] }> = {
  en: guidesEn,
  de: guidesDe,
};
const LOCALES: Record<Lang, string> = { en: 'en_US', de: 'de_DE' };

/** When the example pages were written (their JSON-LD); the facts follow examples.json. */
export const EXAMPLES_DATE = '2026-11-16';

export type PageKind = 'home' | 'examples' | 'example' | 'guides' | 'guide' | 'pricing';

/** A page of the site: what the sitemap, llms.txt, the build check and the tests need. */
export interface PageMeta {
  kind: PageKind;
  path: string;
  lang: Lang;
  title: string;
  description: string;
  /** The page in each language; empty for a page without a counterpart (no hreflang). */
  alternates: Partial<Record<Lang, string>>;
  /** The @types its JSON-LD must have. */
  types: string[];
  /** The file in dist/. */
  file: string;
  /** The example or guide it shows. */
  id?: string;
}

export function fileOf(path: string): string {
  return `${path.slice(1)}index.html`;
}

export function examplePath(lang: Lang, id: string): string {
  return `${EXAMPLES_PATHS[lang]}${exampleSlug(id)}/`;
}

function guide(lang: Lang, id: string): Guide | undefined {
  return GUIDE_COPY[lang].guides.find((candidate) => candidate.id === id);
}

export function guidePath(lang: Lang, entry: Guide): string {
  return `${GUIDES_PATHS[lang]}${entry.slug}/`;
}

function exampleCopy(lang: Lang, id: string): ExampleCopy {
  const copy = EXAMPLE_COPY[lang].examples.find((candidate) => candidate.id === id);
  if (!copy) {
    throw new Error(`No ${lang} copy for the example “${id}” (content/examples.${lang}.ts)`);
  }
  return copy;
}

/** The content pages (without the landing pages). */
export function contentPages(): PageMeta[] {
  const pages: PageMeta[] = [];
  for (const lang of LANGS) {
    const gallery = EXAMPLE_COPY[lang].gallery;
    pages.push({
      kind: 'examples',
      path: EXAMPLES_PATHS[lang],
      lang,
      title: gallery.title,
      description: gallery.description,
      alternates: EXAMPLES_PATHS,
      types: ['CollectionPage', 'BreadcrumbList'],
      file: fileOf(EXAMPLES_PATHS[lang]),
    });
    for (const example of EXAMPLES) {
      const copy = exampleCopy(lang, example.id);
      const path = examplePath(lang, example.id);
      pages.push({
        kind: 'example',
        path,
        lang,
        title: copy.title,
        description: copy.description,
        alternates: { en: examplePath('en', example.id), de: examplePath('de', example.id) },
        types: ['TechArticle', 'BreadcrumbList'],
        file: fileOf(path),
        id: example.id,
      });
    }
    const index = GUIDE_COPY[lang].index;
    pages.push({
      kind: 'guides',
      path: GUIDES_PATHS[lang],
      lang,
      title: index.title,
      description: index.description,
      alternates: GUIDES_PATHS,
      types: ['CollectionPage', 'BreadcrumbList'],
      file: fileOf(GUIDES_PATHS[lang]),
    });
    for (const entry of GUIDE_COPY[lang].guides) {
      const path = guidePath(lang, entry);
      const pair: Partial<Record<Lang, string>> = {};
      for (const other of LANGS) {
        const counterpart = guide(other, entry.id);
        if (counterpart) {
          pair[other] = guidePath(other, counterpart);
        }
      }
      pages.push({
        kind: 'guide',
        path,
        lang,
        title: entry.title,
        description: entry.description,
        alternates: pair.en && pair.de ? pair : {},
        types:
          entry.type === 'HowTo'
            ? ['Article', 'HowTo', 'BreadcrumbList']
            : ['Article', 'BreadcrumbList'],
        file: fileOf(path),
        id: entry.id,
      });
    }
  }
  pages.push({
    kind: 'pricing',
    path: PRICING_DE_PATH,
    lang: 'de',
    title: pricingDe.title,
    description: pricingDe.description,
    alternates: {},
    types: ['FAQPage', 'BreadcrumbList'],
    file: fileOf(PRICING_DE_PATH),
  });
  return pages;
}

/** Every page of the site: the landing pages first. */
export function allPages(): PageMeta[] {
  const home = LANGS.map((lang): PageMeta => ({
    kind: 'home',
    path: PAGE_PATHS[lang],
    lang,
    title: LANDING[lang].title,
    description: LANDING[lang].description,
    alternates: PAGE_PATHS,
    types: ['Organization', 'WebSite', 'SoftwareApplication', 'FAQPage'],
    file: fileOf(PAGE_PATHS[lang]),
  }));
  return [...home, ...contentPages()];
}

/** The screenshots the content pages of a language show (the guides'). */
export function contentShotIds(lang: Lang): string[] {
  const ids = GUIDE_COPY[lang].guides.flatMap((entry) =>
    entry.sections.flatMap((section) => (section.image ? [section.image.id] : []))
  );
  return [...new Set(ids)];
}

// ------------------------------------------------------------------------------ the frame

function head(page: PageMeta, assets: Assets, jsonLd: object, ogType: string): string {
  const url = absolute(page.path);
  const og = absolute(assets.ogImage.src);
  const other = page.lang === 'en' ? 'de' : 'en';
  const hreflang = alternateLinks(page.alternates);
  const localeAlternate = page.alternates[other]
    ? `\n<meta property="og:locale:alternate" content="${LOCALES[other]}">`
    : '';
  return `<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${escapeHtml(page.title)}</title>
<meta name="description" content="${escapeHtml(page.description)}">
<link rel="canonical" href="${url}">${hreflang ? `\n${hreflang}` : ''}
<meta name="color-scheme" content="light dark">
<link rel="icon" type="image/svg+xml" href="${assets.favicon}">
<link rel="stylesheet" href="${assets.css}">
<meta property="og:type" content="${ogType}">
<meta property="og:site_name" content="Brick DNN">
<meta property="og:title" content="${escapeHtml(page.title)}">
<meta property="og:description" content="${escapeHtml(page.description)}">
<meta property="og:url" content="${url}">
<meta property="og:locale" content="${LOCALES[page.lang]}">${localeAlternate}
<meta property="og:image" content="${og}">
<meta property="og:image:type" content="image/jpeg">
<meta property="og:image:width" content="${assets.ogImage.width}">
<meta property="og:image:height" content="${assets.ogImage.height}">
<meta property="og:image:alt" content="${escapeHtml(LANDING[page.lang].ogImageAlt)}">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="${escapeHtml(page.title)}">
<meta name="twitter:description" content="${escapeHtml(page.description)}">
<meta name="twitter:image" content="${og}">
<script type="application/ld+json">${scriptJson(jsonLd)}</script>
</head>`;
}

function shell(
  page: PageMeta,
  assets: Assets,
  area: Area,
  main: string,
  graph: object[],
  ogType = 'article',
  fallback?: Partial<Record<Lang, string>>
): string {
  const chrome: Chrome = {
    lang: page.lang,
    alternates: page.alternates,
    fallback,
    area,
    favicon: assets.favicon,
  };
  const jsonLd = { '@context': 'https://schema.org', '@graph': [organization(assets), ...graph] };
  return `<!doctype html>
<html lang="${page.lang}" dir="ltr">
${head(page, assets, jsonLd, ogType)}
<body>
<a class="skip" href="#main">${escapeHtml(COMMON[page.lang].skipLink)}</a>
${siteHeader(chrome)}
<main id="main" tabindex="-1" class="page"><div class="wrap">
${main}
</div></main>
${siteFooter(chrome)}
</body>
</html>
`;
}

function section(id: string, title: string, body: string): string {
  return `<section id="${id}" aria-labelledby="${id}-title">
<h2 id="${id}-title">${escapeHtml(title)}</h2>
${body}
</section>`;
}

function homeCrumb(lang: Lang): Crumb {
  return { name: COMMON[lang].nav.home, path: PAGE_PATHS[lang] };
}

function webPage(page: PageMeta, type: string, extra: object = {}): object {
  return {
    '@type': type,
    '@id': `${absolute(page.path)}#page`,
    url: absolute(page.path),
    name: page.title,
    description: page.description,
    inLanguage: page.lang,
    isPartOf: { '@id': WEBSITE_ID },
    publisher: { '@id': ORGANIZATION_ID },
    ...extra,
  };
}

// ------------------------------------------------------------------------------ examples

function frameworkOf(example: ExampleData): 'pytorch' | 'scikit-learn' | 'pytorch+scikit-learn' {
  return example.framework;
}

/** A parameter value as the table shows it: `[3, 3]` → `3×3`. */
function formatValue(value: unknown): string {
  if (Array.isArray(value)) {
    return value.every((item) => typeof item === 'number')
      ? value.join('×')
      : JSON.stringify(value);
  }
  if (value === null) {
    return 'None';
  }
  return String(value);
}

function settingsCell(params: Record<string, unknown>, lang: Lang): string {
  const entries = Object.entries(params);
  if (!entries.length) {
    return `<span class="muted">${escapeHtml(COMMON[lang].example.defaults)}</span>`;
  }
  return entries
    .map(([key, value]) => `<code>${escapeHtml(key)}=${escapeHtml(formatValue(value))}</code>`)
    .join(' ');
}

function paramsCell(node: ExampleData['nodes'][number], lang: Lang): string {
  const t = COMMON[lang].example.diagram;
  if (!node.num_params) {
    return node.num_params === 0 ? '0' : '–';
  }
  return node.output_kind === 'tensor'
    ? formatCount(node.num_params, lang)
    : escapeHtml(t.hyperparameters(node.num_params));
}

function totalCell(example: ExampleData, lang: Lang): string {
  const t = COMMON[lang].example;
  if (example.family === 'neural' && example.total_params !== null) {
    return formatCount(example.total_params, lang);
  }
  if (example.family === 'hybrid' && example.neural_params !== null) {
    return escapeHtml(`${t.diagram.params(example.neural_params)} + ${example.estimator}`);
  }
  if (example.total_params) {
    return escapeHtml(t.diagram.hyperparameters(example.total_params));
  }
  return escapeHtml(t.noParams);
}

function layerTable(example: ExampleData, copy: ExampleCopy, lang: Lang): string {
  const t = COMMON[lang].example;
  const sources = new Set(example.edges.map((edge) => edge.source));
  const rows = example.nodes.map((node) => {
    const output =
      node.output_kind === 'kernel'
        ? t.diagram.kernel
        : node.output_kind === 'estimator' && sources.has(node.id)
          ? t.diagram.model
          : node.output_shape
            ? formatShape(node.output_shape)
            : t.diagram.classes;
    return `<tr><th scope="row"><code>${escapeHtml(node.id)}</code></th><td>${escapeHtml(node.type)}</td><td class="settings">${settingsCell(node.params, lang)}</td><td class="shape">${escapeHtml(output)}</td><td class="num">${paramsCell(node, lang)}</td></tr>`;
  });
  return `<div class="table" role="region" aria-labelledby="layers-caption" tabindex="0">
<table class="layers">
<caption id="layers-caption">${escapeHtml(t.tableCaption(copy.name))}</caption>
<thead><tr>${t.columns.map((column, index) => `<th scope="col"${index === 4 ? ' class="num"' : ''}>${escapeHtml(column)}</th>`).join('')}</tr></thead>
<tbody>
${rows.join('\n')}
</tbody>
<tfoot><tr><th scope="row" colspan="4">${escapeHtml(t.total)}</th><td class="num">${totalCell(example, lang)}</td></tr></tfoot>
</table>
</div>`;
}

function exampleMeta(example: ExampleData, lang: Lang): string {
  const t = COMMON[lang].example;
  const params = trainableParams(example);
  const parts = [t.frameworks[frameworkOf(example)]];
  if (params !== null) {
    parts.push(t.diagram.params(params));
  } else if (example.estimator) {
    parts.push(example.estimator);
  }
  return parts.map(escapeHtml).join(' · ');
}

function exampleCards(ids: string[], lang: Lang): string {
  return `<ul class="cards">${ids
    .map((id) => {
      const copy = exampleCopy(lang, id);
      const example = exampleById(id);
      return `<li><h3><a href="${examplePath(lang, id)}">${escapeHtml(copy.name)}</a></h3><p>${copy.summary}</p><p class="meta">${exampleMeta(example, lang)}</p></li>`;
    })
    .join('\n')}</ul>`;
}

function guideCards(entries: Guide[], lang: Lang, level = 3): string {
  return `<ul class="cards">${entries
    .map(
      (entry) =>
        `<li><h${level}><a href="${guidePath(lang, entry)}">${escapeHtml(entry.h1)}</a></h${level}><p>${entry.summary}</p></li>`
    )
    .join('\n')}</ul>`;
}

function inputShapes(example: ExampleData): string {
  return example.nodes
    .filter((node) => node.type === 'Input')
    .map((node) => formatShape(node.output_shape ?? []))
    .join(' + ');
}

function renderExample(page: PageMeta, assets: Assets): string {
  const lang = page.lang;
  const t = COMMON[lang].example;
  const example = exampleById(page.id!);
  const copy = exampleCopy(lang, example.id);
  const framework = frameworkOf(example);
  const crumbs: Crumb[] = [
    homeCrumb(lang),
    { name: COMMON[lang].nav.examples, path: EXAMPLES_PATHS[lang] },
    { name: copy.name, path: page.path },
  ];
  const params = trainableParams(example);
  const facts: [string, string][] = [
    [t.facts.framework, t.frameworks[framework]],
    [t.facts.task, t.tasks[copy.task]],
    [t.facts.input, inputShapes(example)],
    [t.facts.layers, String(example.nodes.length)],
    params !== null
      ? [t.facts.params, formatCount(params, lang)]
      : [t.facts.model, example.estimator ?? ''],
  ];
  const diagram = diagramSvg(
    example,
    t.diagram,
    t.diagramTitle(copy.name),
    describeGraph(example, t.diagram)
  );
  const wide = diagram.width > 1088;
  const filename = `${example.id}.py`;
  const csvLoop = /^def train\(/m.test(example.code);
  const guides = GUIDE_COPY[lang].guides.filter((entry) => entry.examples.includes(example.id));

  const main = `${breadcrumbs(lang, crumbs)}
<article class="doc">
<header class="page-head">
<p class="kicker">${escapeHtml(COMMON[lang].nav.examples)} · ${escapeHtml(t.frameworks[framework])}</p>
<h1>${escapeHtml(copy.h1)}</h1>
<p class="lead">${copy.lead}</p>
<p class="cta"><a class="button" href="${editorExampleUrl(example.id)}">${escapeHtml(t.openEditor)}</a> <a class="button secondary" href="#code">${escapeHtml(t.toCode)}</a></p>
<dl class="facts-list">${facts.map(([term, value]) => `<div><dt>${escapeHtml(term)}</dt><dd>${escapeHtml(value)}</dd></div>`).join('')}</dl>
</header>
${section('what', t.whatTitle, `<div class="prose">${copy.what.map((html) => `<p>${html}</p>`).join('\n')}</div>`)}
${section('when', t.whenTitle, `<ul class="checks prose">${copy.when.map((html) => `<li>${html}</li>`).join('\n')}</ul>`)}
${section(
  'network',
  t.networkTitle,
  `<figure class="diagram${wide ? ' wide' : ''}">
<div class="scroll" role="region" tabindex="0" aria-labelledby="diagram-caption">${diagram.svg}</div>
<figcaption id="diagram-caption">${escapeHtml(t.diagramCaption)}${wide ? ` ${escapeHtml(t.diagramScroll)}` : ''}</figcaption>
</figure>
${layerTable(example, copy, lang)}`
)}
${section('train', t.trainTitle, `<div class="prose"><p>${copy.train}</p></div>`)}
${section(
  'code',
  t.codeTitle,
  `<div class="prose"><p>${t.codeIntro(framework, filename, csvLoop)}</p></div>
<pre class="code" tabindex="0" aria-label="${escapeHtml(`${t.codeLabel}: ${filename}`)}"><code>${highlightPython(example.code)}</code></pre>
<p class="cta"><a class="button" href="${editorExampleUrl(example.id)}">${escapeHtml(t.openEditor)}</a></p>`
)}
${section('related', t.relatedTitle, exampleCards(copy.related, lang))}
${guides.length ? section('guides', t.guidesTitle, guideCards(guides, lang)) : ''}
</article>`;

  const url = absolute(page.path);
  const graph = [
    {
      '@type': 'TechArticle',
      '@id': `${url}#article`,
      headline: copy.h1,
      name: copy.title,
      description: copy.description,
      url,
      mainEntityOfPage: url,
      inLanguage: lang,
      author: author(),
      publisher: { '@id': ORGANIZATION_ID },
      isPartOf: { '@id': WEBSITE_ID },
      datePublished: EXAMPLES_DATE,
      dateModified: EXAMPLES_DATE,
      image: absolute(assets.ogImage.src),
      about: { '@id': SOFTWARE_ID },
      proficiencyLevel: 'Beginner',
      hasPart: {
        '@type': 'SoftwareSourceCode',
        name: filename,
        programmingLanguage: 'Python',
        runtimePlatform: t.frameworks[framework],
        codeSampleType: 'full solution',
        license: LICENSE_URL,
      },
    },
    breadcrumbList(crumbs),
  ];
  return shell(page, assets, 'examples', main, graph);
}

function renderGallery(page: PageMeta, assets: Assets): string {
  const lang = page.lang;
  const gallery = EXAMPLE_COPY[lang].gallery;
  const crumbs: Crumb[] = [homeCrumb(lang), { name: COMMON[lang].nav.examples, path: page.path }];
  const listed = gallery.groups.flatMap((group) => group.ids);
  const main = `${breadcrumbs(lang, crumbs)}
<header class="page-head">
<h1>${escapeHtml(gallery.h1)}</h1>
<p class="lead">${gallery.intro}</p>
</header>
${gallery.groups
  .map((group, index) =>
    section(
      `group-${index + 1}`,
      group.title,
      `<p class="intro">${group.intro}</p>\n${exampleCards(group.ids, lang)}`
    )
  )
  .join('\n')}
<p class="note">${gallery.note}</p>`;
  const graph = [
    webPage(page, 'CollectionPage', {
      mainEntity: {
        '@type': 'ItemList',
        itemListElement: listed.map((id, index) => ({
          '@type': 'ListItem',
          position: index + 1,
          url: absolute(examplePath(lang, id)),
          name: exampleCopy(lang, id).h1,
        })),
      },
    }),
    breadcrumbList(crumbs),
  ];
  return shell(page, assets, 'examples', main, graph, 'website');
}

// ------------------------------------------------------------------------------ guides

function renderGuide(page: PageMeta, assets: Assets): string {
  const lang = page.lang;
  const t = COMMON[lang];
  const entry = guide(lang, page.id!)!;
  const crumbs: Crumb[] = [
    homeCrumb(lang),
    { name: t.nav.guides, path: GUIDES_PATHS[lang] },
    { name: entry.h1, path: page.path },
  ];
  const url = absolute(page.path);
  const sections = entry.sections.map((part) => {
    const steps = part.steps
      ? `<ol class="howto">${part.steps.map((step) => `<li><h3>${escapeHtml(step.title)}</h3><p>${step.html}</p></li>`).join('\n')}</ol>`
      : '';
    const image = part.image
      ? `<figure class="frame">${picture(assets.images[part.image.id], part.image.alt, 'article', 736)}<figcaption>${escapeHtml(part.image.caption)}</figcaption></figure>`
      : '';
    return section(
      part.id,
      part.title,
      `<div class="prose">${part.html}</div>${steps}${image}${part.after ? `<div class="prose">${part.after}</div>` : ''}`
    );
  });
  const others = GUIDE_COPY[lang].guides.filter((other) => other.id !== entry.id);
  const main = `${breadcrumbs(lang, crumbs)}
<article class="doc guide">
<header class="page-head">
<p class="kicker">${escapeHtml(t.nav.guides)}</p>
<h1>${escapeHtml(entry.h1)}</h1>
<p class="lead">${entry.lead}</p>
<p class="byline">${escapeHtml(t.by)} ${escapeHtml(AUTHOR.name)} · ${escapeHtml(t.updated)} <time datetime="${entry.modified}">${escapeHtml(t.date(entry.modified))}</time></p>
</header>
<nav class="toc" aria-labelledby="toc-title"><h2 id="toc-title">${escapeHtml(t.guide.contents)}</h2><ol>${entry.sections
    .map((part) => `<li><a href="#${part.id}">${escapeHtml(part.title)}</a></li>`)
    .join('')}</ol></nav>
${sections.join('\n')}
${section('examples', t.guide.relatedExamples, exampleCards(entry.examples, lang))}
${section('more', t.guide.moreGuides, guideCards(others, lang))}
<p class="cta"><a class="button" href="${EDITOR_PATH}">${escapeHtml(t.guide.tryIt)}</a></p>
</article>`;

  const images = contentImages(entry, assets);
  const article = {
    '@type': 'Article',
    '@id': `${url}#article`,
    headline: entry.h1,
    name: entry.title,
    description: entry.description,
    url,
    mainEntityOfPage: url,
    inLanguage: lang,
    author: author(),
    publisher: { '@id': ORGANIZATION_ID },
    isPartOf: { '@id': WEBSITE_ID },
    about: { '@id': SOFTWARE_ID },
    datePublished: entry.published,
    dateModified: entry.modified,
    image: images.length ? images : [absolute(assets.ogImage.src)],
  };
  const graph: object[] = [article];
  if (entry.type === 'HowTo') {
    graph.push({
      '@type': 'HowTo',
      '@id': `${url}#howto`,
      name: entry.h1,
      description: entry.description,
      url,
      inLanguage: lang,
      author: author(),
      datePublished: entry.published,
      dateModified: entry.modified,
      ...(entry.totalTime ? { totalTime: entry.totalTime } : {}),
      tool: [{ '@type': 'HowToTool', name: 'Brick DNN' }],
      step: entry.sections.flatMap((part) =>
        (part.steps ?? []).map((step, index) => ({
          '@type': 'HowToStep',
          name: step.title,
          text: plainText(step.html),
          url: `${url}#${part.id}`,
          position: index + 1,
        }))
      ),
    });
  }
  graph.push(breadcrumbList(crumbs));
  return shell(page, assets, 'guides', main, graph);
}

function contentImages(entry: Guide, assets: Assets): string[] {
  return entry.sections.flatMap((part) =>
    part.image ? [absolute(assets.images[part.image.id].sources.at(-1)!.src)] : []
  );
}

function renderGuideIndex(page: PageMeta, assets: Assets): string {
  const lang = page.lang;
  const copy = GUIDE_COPY[lang];
  const crumbs: Crumb[] = [homeCrumb(lang), { name: COMMON[lang].nav.guides, path: page.path }];
  const main = `${breadcrumbs(lang, crumbs)}
<header class="page-head">
<h1>${escapeHtml(copy.index.h1)}</h1>
<p class="lead">${copy.index.intro}</p>
</header>
${guideCards(copy.guides, lang, 2)}`;
  const graph = [
    webPage(page, 'CollectionPage', {
      mainEntity: {
        '@type': 'ItemList',
        itemListElement: copy.guides.map((entry, index) => ({
          '@type': 'ListItem',
          position: index + 1,
          url: absolute(guidePath(lang, entry)),
          name: entry.h1,
        })),
      },
    }),
    breadcrumbList(crumbs),
  ];
  return shell(page, assets, 'guides', main, graph, 'website');
}

// ------------------------------------------------------------------------------ pricing

function renderPricing(page: PageMeta, assets: Assets): string {
  const c = pricingDe;
  const crumbs: Crumb[] = [homeCrumb('de'), { name: COMMON.de.nav.pricing, path: page.path }];
  const rows = PLANS.map((plan) => {
    const [name, ...cells] = c.row(plan);
    return `<tr><th scope="row">${escapeHtml(name)}</th>${cells.map((cell) => `<td>${escapeHtml(cell)}</td>`).join('')}</tr>`;
  }).join('\n');
  const main = `${breadcrumbs('de', crumbs)}
<article class="doc">
<header class="page-head">
<h1>${escapeHtml(c.h1)}</h1>
<p class="lead">${c.lead}</p>
</header>
<div class="table" role="region" aria-labelledby="plans-caption" tabindex="0">
<table>
<caption id="plans-caption">${escapeHtml(c.caption)}</caption>
<thead><tr>${c.columns.map((column) => `<th scope="col">${escapeHtml(column)}</th>`).join('')}</tr></thead>
<tbody>
${rows}
</tbody>
</table>
</div>
${section('buchen', c.buy.title, `<div class="prose"><p>${c.buy.html}</p></div>\n<p class="cta"><a class="button" href="/pricing">${escapeHtml(c.buy.button)}</a></p>`)}
${c.sections.map((part) => section(part.id, part.title, `<div class="prose">${part.html}</div>`)).join('\n')}
${section(
  'fragen',
  c.faq.title,
  `<div class="faq">${c.faq.items
    .map((item) => `<div><h3>${escapeHtml(item.question)}</h3><p>${item.html}</p></div>`)
    .join('\n')}</div>`
)}
</article>`;
  const graph = [
    webPage(page, 'FAQPage', {
      about: { '@id': SOFTWARE_ID },
      mainEntity: c.faq.items.map((item) => ({
        '@type': 'Question',
        name: item.question,
        acceptedAnswer: { '@type': 'Answer', text: plainText(item.html) },
      })),
    }),
    {
      '@type': 'SoftwareApplication',
      '@id': SOFTWARE_ID,
      name: 'Brick DNN',
      url: absolute(EDITOR_PATH),
      applicationCategory: 'DeveloperApplication',
      operatingSystem: 'Web browser',
      offers: offers(),
    },
    breadcrumbList(crumbs),
  ];
  return shell(page, assets, 'pricing', main, graph, 'website', { en: '/#pricing' });
}

/** The HTML of a content page. */
export function renderContentPage(page: PageMeta, assets: Assets): string {
  switch (page.kind) {
    case 'example':
      return renderExample(page, assets);
    case 'examples':
      return renderGallery(page, assets);
    case 'guide':
      return renderGuide(page, assets);
    case 'guides':
      return renderGuideIndex(page, assets);
    case 'pricing':
      return renderPricing(page, assets);
    default:
      throw new Error(`${page.path} is not a content page`);
  }
}

/** The content pages in llms.txt (Markdown). */
export function llmsSections(): string {
  const examples = EXAMPLES.map((example) => {
    const copy = exampleCopy('en', example.id);
    return `- [${copy.h1}](${absolute(examplePath('en', example.id))}): ${plainText(copy.summary)}`;
  });
  const guides = GUIDE_COPY.en.guides.map(
    (entry) => `- [${entry.h1}](${absolute(guidePath('en', entry))}): ${plainText(entry.summary)}`
  );
  const german = [
    `- [Beispiele](${absolute(EXAMPLES_PATHS.de)}): the example gallery in German`,
    `- [Anleitungen](${absolute(GUIDES_PATHS.de)}): the guides in German`,
    `- [Preise](${absolute(PRICING_DE_PATH)}): plans, limits and billing in German`,
  ];
  return `## Examples

Every built-in example has a page with its layers, their output shapes and parameter counts, a diagram and the generated code (${absolute(EXAMPLES_PATHS.en)}). Each opens in the editor at ${absolute(EDITOR_PATH)}?example= followed by the id of the example, e.g. ${absolute(editorExampleUrl('unet'))}.

${examples.join('\n')}

## Guides

${guides.join('\n')}

## Pages in German

${german.join('\n')}`;
}
