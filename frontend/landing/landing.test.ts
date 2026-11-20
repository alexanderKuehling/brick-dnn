import { spawnSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { checkDist, checkLandingHtml, checkPageHtml, hreflangs, jsonLdTypes } from './check.ts';
import { COMMON } from './content/common.ts';
import { EXAMPLES, EXAMPLES_FILE, type ExampleData } from './examples/data.ts';
import { titleLines } from './examples/diagram.ts';
import { guideArt, hasOwnGuideArt } from './guideArt.ts';
import { IMAGE_DIR, imageSize, LOGO, OG_IMAGE, readImage, shotFiles } from './images.ts';
import {
  allPages,
  contentPages,
  contentShotIds,
  EXAMPLE_COPY,
  examplePath,
  GUIDE_COPY,
  readingMinutes,
} from './pages.ts';
import { minifyCss, renderStatic, staticPage } from './plugin.ts';
import {
  CONTENT,
  plainText,
  renderLlms,
  renderPage,
  renderRobots,
  renderSitemap,
  shotIds,
  type Assets,
} from './render.ts';
import {
  APP_PAGES,
  CONTENT_FOLDERS,
  EDITOR_PATH,
  EXAMPLES_PATHS,
  GUIDES_PATHS,
  INLINE_SCRIPT,
  inlineScriptHash,
  OVERAGE_PRICE,
  PLANS,
  pricingPath,
  SECTION_IDS,
} from './site.ts';

const FRONTEND = join(import.meta.dirname, '..');
const REPO = join(FRONTEND, '..');

/** Every screenshot a page of either language shows. */
const ALL_SHOT_IDS = [
  ...new Set([
    ...shotIds(CONTENT.en),
    ...shotIds(CONTENT.de),
    ...contentShotIds('en'),
    ...contentShotIds('de'),
  ]),
];

const ASSETS: Assets = {
  css: '/assets/landing-abc.css',
  favicon: '/favicon.svg',
  logo: { src: '/assets/logo-512-abc.png', width: 512, height: 512 },
  images: Object.fromEntries(
    ALL_SHOT_IDS.map((id) => [
      id,
      {
        sources: [480, 1056, 2112].map((width) => ({
          src: `/assets/${id}-${width}-abc.webp`,
          width,
          height: Math.round(width / 1.6),
        })),
      },
    ])
  ),
  ogImage: { src: '/assets/og-image-abc.jpg', width: 1200, height: 630 },
};

const LANGS = ['en', 'de'] as const;

function parse(html: string): Document {
  return new DOMParser().parseFromString(html, 'text/html');
}

function ldJson(doc: Document): { '@graph': Record<string, any>[] } {
  return JSON.parse(doc.querySelector('script[type="application/ld+json"]')!.textContent!);
}

describe.each(LANGS)('landing page %s', (lang) => {
  const html = renderPage(lang, ASSETS);
  const doc = parse(html);
  const c = CONTENT[lang];

  test('passes the build check (lang, canonical, hreflang, JSON-LD, one h1)', () => {
    expect(checkLandingHtml(lang, html)).toEqual([]);
  });

  test('has its own title and description of a sensible length', () => {
    expect(doc.title).toBe(c.title);
    expect(c.title.length).toBeLessThanOrEqual(75);
    const description = doc.querySelector('meta[name="description"]')!.getAttribute('content')!;
    expect(description.length).toBeGreaterThan(80);
    expect(description.length).toBeLessThanOrEqual(170);
    const otherLang = lang === 'en' ? 'de' : 'en';
    expect(c.title).not.toBe(CONTENT[otherLang].title);
    expect(description).not.toBe(CONTENT[otherLang].description);
  });

  test('has Open Graph and Twitter meta with absolute addresses', () => {
    const meta = (name: string) =>
      doc.querySelector(`meta[property="${name}"], meta[name="${name}"]`)?.getAttribute('content');
    expect(meta('og:image')).toBe(`https://brickdnn.com${ASSETS.ogImage.src}`);
    expect(meta('og:url')).toBe(
      lang === 'en' ? 'https://brickdnn.com/' : 'https://brickdnn.com/de/'
    );
    expect(meta('og:locale')).toBe(c.locale);
    expect(meta('twitter:card')).toBe('summary_large_image');
    expect(meta('twitter:image')).toBe(meta('og:image'));
  });

  test('JSON-LD: the real prices and the visible FAQ', () => {
    const graph = ldJson(doc)['@graph'];
    expect(jsonLdTypes(html)).toEqual([
      'Organization',
      'WebSite',
      'SoftwareApplication',
      'FAQPage',
    ]);
    const app = graph.find((node) => node['@type'] === 'SoftwareApplication')!;
    expect(app.offers.map((offer: any) => [offer.name, offer.price, offer.priceCurrency])).toEqual(
      PLANS.map((plan) => [plan.name, String(plan.price), 'EUR'])
    );
    expect(app.url).toBe(`https://brickdnn.com${EDITOR_PATH}`);
    expect(app.inLanguage).toEqual(['en', 'de']);
    for (const node of [app, graph.find((node) => node['@type'] === 'Organization')!]) {
      expect(node.areaServed).toEqual({ '@type': 'Continent', name: 'Europe' });
    }
    const faq = graph.find((node) => node['@type'] === 'FAQPage')!;
    const visible = [...doc.querySelectorAll('#faq h3')].map((h3) => h3.textContent);
    expect(faq.mainEntity.map((q: any) => q.name)).toEqual(visible);
    for (const [index, question] of faq.mainEntity.entries()) {
      expect(question.acceptedAnswer.text).toBe(
        doc.querySelectorAll('#faq h3 + p')[index].textContent!.replace(/\s+/g, ' ').trim()
      );
    }
  });

  test('semantic structure: sections with headings, one h1, a skip link', () => {
    expect(doc.documentElement.lang).toBe(lang);
    expect(doc.querySelectorAll('h1')).toHaveLength(1);
    for (const section of doc.querySelectorAll('section')) {
      const heading = doc.getElementById(section.getAttribute('aria-labelledby')!);
      expect(heading?.tagName).toMatch(/^H[12]$/);
    }
    expect(doc.querySelector('a.skip')!.getAttribute('href')).toBe('#main');
    expect(doc.querySelector('main#main')).not.toBeNull();
  });

  test('in-page links point to sections the inline script knows', () => {
    for (const id of SECTION_IDS) {
      expect(doc.getElementById(id), id).not.toBeNull();
    }
    for (const link of doc.querySelectorAll('a[href^="#"]')) {
      expect(SECTION_IDS).toContain(link.getAttribute('href')!.slice(1));
    }
  });

  test('loads nothing from elsewhere and needs no inline styles or other scripts', () => {
    for (const element of doc.querySelectorAll('[src], link[rel="stylesheet"], link[rel="icon"]')) {
      const url = element.getAttribute('src') ?? element.getAttribute('href')!;
      expect(url.startsWith('/')).toBe(true);
    }
    expect(doc.querySelectorAll('[style], style')).toHaveLength(0);
    const scripts = [...doc.querySelectorAll('script')];
    expect(scripts.filter((s) => !s.type).map((s) => s.textContent)).toEqual([INLINE_SCRIPT]);
    expect(scripts.filter((s) => s.type && s.type !== 'application/ld+json')).toHaveLength(0);
  });

  test('images have alt text and a size; screenshots several widths, lazy below the fold', () => {
    for (const img of doc.querySelectorAll('img')) {
      expect(img.hasAttribute('alt')).toBe(true);
      expect(Number(img.getAttribute('width'))).toBeGreaterThan(0);
      expect(Number(img.getAttribute('height'))).toBeGreaterThan(0);
    }
    const shots = [...doc.querySelectorAll('img[srcset]')];
    expect(shots).toHaveLength(shotIds(c).length);
    expect(shotIds(c).length).toBeGreaterThanOrEqual(12);
    for (const [index, img] of shots.entries()) {
      expect(img.getAttribute('alt')!.length, img.getAttribute('src')!).toBeGreaterThan(60);
      expect(img.getAttribute('srcset')!.split(',').length).toBeGreaterThanOrEqual(2);
      expect(img.getAttribute('sizes')).toBeTruthy();
      expect(img.getAttribute('decoding')).toBe('async');
      // Only the first (the hero) loads at once and first.
      expect(img.getAttribute('loading')).toBe(index === 0 ? null : 'lazy');
      expect(img.getAttribute('fetchpriority')).toBe(index === 0 ? 'high' : null);
    }
    expect(doc.querySelectorAll('#use-cases .rows img')).toHaveLength(c.useCases.items.length);
    expect(doc.querySelectorAll('#how-it-works .steps img')).toHaveLength(c.steps.items.length);
    expect(doc.querySelectorAll('#features .tiles img').length).toBeGreaterThanOrEqual(4);
  });

  test('alt texts are written for this language', () => {
    const otherLang = lang === 'en' ? 'de' : 'en';
    const alts = (content: typeof c) =>
      [...content.useCases.items, ...content.steps.items, ...content.features.items]
        .filter((item) => item.image)
        .map((item) => item.image!.alt);
    const mine = alts(c);
    for (const alt of mine) {
      expect(alts(CONTENT[otherLang])).not.toContain(alt);
    }
    expect(new Set(mine).size).toBe(mine.length);
  });

  test('links to the editor, pricing, the legal pages and the other language', () => {
    const hrefs = [...doc.querySelectorAll('a')].map((a) => a.getAttribute('href'));
    // The German page leads to its own pricing page (/de/preise/), which links to /pricing.
    const app = APP_PAGES.filter((path) => lang === 'en' || path !== '/pricing');
    for (const path of [EDITOR_PATH, ...app, '/', '/de/', pricingPath(lang)]) {
      expect(hrefs).toContain(path);
    }
    expect(hrefs).toContain(EXAMPLES_PATHS[lang]);
    expect(hrefs).toContain(GUIDES_PATHS[lang]);
    expect(html).toContain('mailto:alexander.kuehling@brickdnn.com');
    const other = doc.querySelector(`.lang a[hreflang="${lang === 'en' ? 'de' : 'en'}"]`)!;
    expect(other.getAttribute('lang')).toBe(lang === 'en' ? 'de' : 'en');
  });

  test('the facts under the hero: each with a check mark that is not read out', () => {
    const facts = [...doc.querySelectorAll('.hero .facts li')];
    expect(facts.map((fact) => fact.textContent)).toEqual(c.hero.facts);
    for (const fact of facts) {
      const ticks = fact.querySelectorAll('svg.tick');
      expect(ticks).toHaveLength(1);
      expect(ticks[0].getAttribute('aria-hidden')).toBe('true');
    }
  });

  test('the pricing table lists every plan', () => {
    const rows = [...doc.querySelectorAll('#pricing tbody tr')];
    expect(rows.map((row) => row.querySelector('th')!.textContent)).toEqual(
      PLANS.map((plan) => plan.name)
    );
  });
});

test('the plans are those of brickdnn.com (deploy/sites/brickdnn.com/settings.env)', () => {
  const settings = readFileSync(join(REPO, 'deploy/sites/brickdnn.com/settings.env'), 'utf8');
  const plans = JSON.parse(settings.match(/^BRICKDNN_PLANS=(.*)$/m)![1]);
  expect(settings).toMatch(/^BRICKDNN_PRICE_CURRENCY=EUR$/m);
  expect(
    Object.entries(plans).map(([key, plan]: [string, any]) => ({
      key,
      name: plan.name,
      price: plan.price,
      cpuHoursPerMonth: plan.cpu_hours_per_month,
      storageGb: plan.storage_gb,
      cpuMinutesPerRun: plan.cpu_minutes_per_run,
      concurrentTrainings: plan.concurrent_trainings,
    }))
  ).toEqual(PLANS);
  for (const plan of Object.values(plans) as any[]) {
    if (plan.price) {
      expect(plan.overage_price_per_cpu_hour).toBe(OVERAGE_PRICE);
    }
  }
});

test('nginx.conf allows exactly the inline script on both landing pages', () => {
  const nginx = readFileSync(join(FRONTEND, 'nginx.conf'), 'utf8');
  const hashes = nginx.match(/'sha256-[^']+'/g);
  expect(hashes).toEqual([inlineScriptHash(), inlineScriptHash()]);
  expect(nginx).toMatch(/location = \/ \{[^}]*return 302 \/app\?\$args;/);
  expect(nginx).toContain('try_files $uri /app.html;');
});

describe('the inline script', () => {
  const base64url = (value: object) => Buffer.from(JSON.stringify(value)).toString('base64url');
  const token = (exp: number) => `${base64url({ alg: 'HS256' })}.${base64url({ exp })}.sig`;
  const now = Date.now() / 1000;

  function run(url: string, tokens?: object | string): string | null {
    const parsed = new URL(url, 'https://brickdnn.com');
    let target: string | null = null;
    const location = {
      pathname: parsed.pathname,
      search: parsed.search,
      hash: parsed.hash,
      replace: (to: string) => {
        target = to;
      },
    };
    const storage = {
      getItem: (key: string) =>
        key === 'brickdnn:tokens' && tokens !== undefined
          ? typeof tokens === 'string'
            ? tokens
            : JSON.stringify(tokens)
          : null,
    };
    new Function('location', 'localStorage', INLINE_SCRIPT)(location, storage);
    return target;
  }

  test('leaves visitors on the landing page', () => {
    expect(run('/')).toBeNull();
    expect(run('/de/')).toBeNull();
    expect(run('/#faq')).toBeNull();
    expect(run('/#pricing')).toBeNull();
  });

  test('sends old links into the editor on to /app', () => {
    expect(run('/?project=5')).toBe('/app?project=5');
    expect(run('/?a=1#b')).toBe('/app?a=1#b');
    expect(run('/#token=abc')).toBe('/app#token=abc');
    // Only on /: the German page had no old links.
    expect(run('/de/#token=abc')).toBeNull();
  });

  test('leaves logged-in users on the landing page', () => {
    const valid = { access: token(now - 10), refresh: token(now + 3600) };
    expect(run('/', valid)).toBeNull();
    expect(run('/de/', valid)).toBeNull();
    expect(run('/#faq', valid)).toBeNull();
    expect(run('/?project=5', valid)).toBe('/app?project=5');
  });
});

describe('the build check', () => {
  let dir: string;

  function write(path: string, content: string | Buffer = '') {
    mkdirSync(dirname(join(dir, path)), { recursive: true });
    writeFileSync(join(dir, path), content);
  }

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), 'landing-'));
    for (const page of allPages()) {
      write(page.file, renderStatic(page, ASSETS));
    }
    write('app.html', '<!doctype html><div id="root"></div>');
    write('robots.txt', renderRobots());
    write('sitemap.xml', renderSitemap());
    write('llms.txt', renderLlms());
    for (const path of [
      ASSETS.css,
      ASSETS.favicon,
      ASSETS.logo.src,
      ASSETS.ogImage.src,
      ...Object.values(ASSETS.images).flatMap((image) => image.sources.map((s) => s.src)),
    ]) {
      write(path);
    }
  });

  afterEach(() => {
    rmSync(dir, { recursive: true, force: true });
  });

  test('accepts a complete build', () => {
    expect(checkDist(dir)).toEqual([]);
  });

  test('finds a missing page, hreflang, JSON-LD or asset', () => {
    rmSync(join(dir, 'de/index.html'));
    write('index.html', renderPage('en', ASSETS).replace('hreflang="de"', 'hreflang="fr"'));
    rmSync(join(dir, 'app.html'));
    // A screenshot of a guide, a German example page and an English hreflang link of a guide.
    const image = ASSETS.images.segmentation.sources[0].src;
    rmSync(join(dir, image.slice(1)));
    rmSync(join(dir, 'de/beispiele/unet/index.html'));
    const guide = staticPage('/guides/imbalanced-classes/')!;
    write(
      guide.file,
      renderStatic(guide, ASSETS).replace(
        'hreflang="de" href="https://brickdnn.com/de/anleitungen/unausgewogene-klassen/"',
        'hreflang="de" href="https://brickdnn.com/de/anleitungen/"'
      )
    );
    expect(checkDist(dir)).toEqual([
      'en: hreflang de',
      `/: ${image} is missing`,
      'de/index.html is missing',
      `/guides/image-segmentation-u-net/: ${image} is missing`,
      '/guides/imbalanced-classes/: hreflang de',
      'de/beispiele/unet/index.html is missing',
      `/de/anleitungen/bildsegmentierung-u-net/: ${image} is missing`,
      '/guides/imbalanced-classes/: /de/anleitungen/ has no hreflang link back to it',
      '/de/anleitungen/unausgewogene-klassen/: /guides/imbalanced-classes/ has no hreflang link back to it',
      'app.html is missing',
    ]);
    // A page the sitemap does not list.
    write('sitemap.xml', renderSitemap().replace('<loc>https://brickdnn.com/de/preise/</loc>', ''));
    expect(checkDist(dir)).toContain('sitemap.xml lacks /de/preise/');
    write(
      'index.html',
      renderPage('en', ASSETS).replace(/<script type="application\/ld\+json">/, '<script>')
    );
    expect(checkDist(dir)).toContain('en: valid JSON-LD');
  });
});

test('robots.txt, sitemap.xml and llms.txt', () => {
  expect(renderRobots()).toContain('Sitemap: https://brickdnn.com/sitemap.xml');
  const sitemap = parse(renderSitemap());
  const locs = [...sitemap.querySelectorAll('loc')].map((loc) => loc.textContent);
  expect(locs).toEqual([
    ...allPages().map((page) => `https://brickdnn.com${page.path}`),
    ...APP_PAGES.map((path) => `https://brickdnn.com${path}`),
  ]);
  expect(locs).toContain('https://brickdnn.com/examples/unet/');
  expect(locs).toContain('https://brickdnn.com/de/preise/');
  expect(renderSitemap()).toContain(
    '<xhtml:link rel="alternate" hreflang="de" href="https://brickdnn.com/de/"/>'
  );
  const llms = renderLlms();
  expect(llms.startsWith('# Brick DNN\n\n> ')).toBe(true);
  for (const plan of PLANS) {
    expect(llms).toContain(`**${plan.name}**: €${plan.price} per month`);
  }
  expect(llms).toContain('Switzerland');
  expect(llms).not.toMatch(/<[a-z]/);
});

test('plainText drops markup', () => {
  expect(plainText('An <code>nn.Module</code> &amp; <q>quoted</q>')).toBe(
    'An nn.Module & “quoted”'
  );
});

test('minifyCss keeps selectors and media queries intact', () => {
  expect(
    minifyCss(
      '/* x */\na:hover ,\n.b > .c {\n  color: red;\n}\n@media (min-width: 54rem) {\n  .d { margin: 0 auto; }\n}'
    )
  ).toBe('a:hover,.b>.c{color:red}@media (min-width:54rem){.d{margin:0 auto}}');
});

test('every screenshot exists in several widths of the same proportions, and no others', () => {
  const ids = ALL_SHOT_IDS;
  for (const lang of LANGS) {
    for (const id of new Set([...shotIds(CONTENT[lang]), ...contentShotIds(lang)])) {
      const files = shotFiles(id, lang);
      expect(files.length, id).toBeGreaterThanOrEqual(2);
      // At least twice the width the page shows it at (1x and 2x screens).
      expect(files.at(-1)!.width, id).toBeGreaterThanOrEqual(640);
      for (const file of files) {
        expect(
          Math.abs(file.height / file.width - files[0].height / files[0].width),
          id
        ).toBeLessThan(0.01);
        // Optimised: no single file is large.
        expect(readImage(file.name).data.length, file.name).toBeLessThan(260_000);
      }
    }
  }
  const unused = readdirSync(IMAGE_DIR).filter(
    (name) =>
      name.endsWith('.webp') &&
      !ids.some((id) => new RegExp(`^${id}(\\.de)?-\\d+\\.webp$`).test(name))
  );
  expect(unused).toEqual([]);
  expect(() => shotFiles('no-such-shot', 'en')).toThrow();
});

test('the Open Graph image and the logo', () => {
  expect(readImage(OG_IMAGE).size).toEqual({ width: 1200, height: 630 });
  expect(readImage(LOGO).size).toEqual({ width: 512, height: 512 });
  expect(() => imageSize(new Uint8Array(32))).toThrow();
});

// ------------------------------------------------------------------------- content pages

const PAGES = contentPages();
const RENDERED = new Map(PAGES.map((page) => [page.path, renderStatic(page, ASSETS)]));
const KNOWN_PATHS = new Set([...allPages().map((page) => page.path), EDITOR_PATH, ...APP_PAGES]);

describe.each(PAGES.map((page) => [page.path, page] as const))('content page %s', (_, page) => {
  const html = RENDERED.get(page.path)!;
  const doc = parse(html);

  test('passes the build check: title, description, canonical, hreflang, JSON-LD, one h1', () => {
    expect(checkPageHtml(page, html)).toEqual([]);
    expect(doc.title).toBe(page.title);
    expect(page.title.length).toBeLessThanOrEqual(75);
    expect(page.description.length).toBeGreaterThan(80);
    expect(page.description.length).toBeLessThanOrEqual(170);
  });

  test('its language versions point at each other', () => {
    const links = hreflangs(html);
    if (!page.alternates.en) {
      expect(links).toEqual({});
      return;
    }
    expect(links[page.lang]).toBe(page.path);
    for (const lang of LANGS) {
      const other = RENDERED.get(links[lang])!;
      expect(parse(other).documentElement.lang).toBe(lang);
      expect(hreflangs(other)).toEqual(links);
      // The language switch leads there too.
      expect(doc.querySelector(`header .lang a[hreflang="${lang}"]`)!.getAttribute('href')).toBe(
        links[lang]
      );
    }
  });

  test('the JSON-LD is valid and its breadcrumbs are the visible ones', () => {
    const graph = ldJson(doc)['@graph'];
    const crumbs = graph.find((node) => node['@type'] === 'BreadcrumbList')!;
    const visible = [...doc.querySelectorAll('.crumbs a')];
    expect(crumbs.itemListElement.map((item: any) => [item.name, item.item])).toEqual(
      visible.map((a) => [a.textContent, `https://brickdnn.com${a.getAttribute('href')}`])
    );
    expect(crumbs.itemListElement.at(-1).item).toBe(`https://brickdnn.com${page.path}`);
    for (const node of graph) {
      if (['TechArticle', 'Article', 'HowTo'].includes(node['@type'])) {
        expect(node.author).toEqual({
          '@type': 'Person',
          name: 'Alexander Kühling',
          url: 'https://github.com/alexanderKuehling',
        });
        expect(node.inLanguage).toBe(page.lang);
        expect(node.datePublished).toMatch(/^\d{4}-\d{2}-\d{2}$/);
        expect(node.dateModified).toMatch(/^\d{4}-\d{2}-\d{2}$/);
      }
    }
  });

  test('every link leads somewhere that exists', () => {
    for (const a of doc.querySelectorAll('a[href]')) {
      const href = a.getAttribute('href')!;
      if (href.startsWith('#')) {
        expect(doc.getElementById(href.slice(1)), href).not.toBeNull();
      } else if (href.startsWith('/')) {
        const [path, query] = href.split('#')[0].split('?');
        expect(KNOWN_PATHS.has(path), href).toBe(true);
        if (query) {
          expect(path).toBe(EDITOR_PATH);
          const id = new URLSearchParams(query).get('example');
          expect(EXAMPLES.map((example) => example.id)).toContain(id);
        }
      } else {
        expect(href, href).toMatch(/^(https:\/\/github\.com\/alexanderKuehling\/|mailto:)/);
      }
    }
  });

  test('accessible: lang, headings in order, alt texts, labelled regions, no script', () => {
    expect(doc.documentElement.lang).toBe(page.lang);
    const levels = [...doc.querySelectorAll('main h1, main h2, main h3')].map((h) =>
      Number(h.tagName[1])
    );
    expect(levels[0]).toBe(1);
    for (const [index, level] of levels.entries()) {
      expect(level - (levels[index - 1] ?? 0)).toBeLessThanOrEqual(1);
    }
    for (const img of doc.querySelectorAll('main img')) {
      expect(img.getAttribute('alt')!.length).toBeGreaterThan(40);
      expect(img.getAttribute('loading')).toBe('lazy');
    }
    for (const region of doc.querySelectorAll('[tabindex="0"]')) {
      const name = region.getAttribute('aria-label') ?? region.getAttribute('aria-labelledby');
      expect(name, region.outerHTML.slice(0, 80)).toBeTruthy();
    }
    for (const section of doc.querySelectorAll('section')) {
      expect(doc.getElementById(section.getAttribute('aria-labelledby')!)?.tagName).toBe('H2');
    }
    expect([...doc.querySelectorAll('script')].map((script) => script.type)).toEqual([
      'application/ld+json',
    ]);
    expect(doc.querySelectorAll('[style], style')).toHaveLength(0);
    for (const element of doc.querySelectorAll('[src], link[href]')) {
      const url = element.getAttribute('src') ?? element.getAttribute('href')!;
      expect(url.startsWith('/') || url.startsWith('https://brickdnn.com/'), url).toBe(true);
    }
  });
});

test('the content pages: every example and guide in both languages, unique titles', () => {
  const ids = EXAMPLES.map((example) => example.id);
  for (const lang of LANGS) {
    const copy = EXAMPLE_COPY[lang];
    expect(copy.examples.map((example) => example.id).sort()).toEqual([...ids].sort());
    // The gallery lists every example once.
    expect(copy.gallery.groups.flatMap((group) => group.ids).sort()).toEqual([...ids].sort());
    for (const example of copy.examples) {
      expect(example.related.length).toBeGreaterThanOrEqual(2);
      expect(example.related).not.toContain(example.id);
      for (const id of example.related) {
        expect(ids).toContain(id);
      }
    }
    for (const guide of GUIDE_COPY[lang].guides) {
      for (const id of guide.examples) {
        expect(ids).toContain(id);
      }
    }
  }
  expect(GUIDE_COPY.en.guides.map((guide) => guide.id)).toEqual(
    GUIDE_COPY.de.guides.map((guide) => guide.id)
  );
  expect(GUIDE_COPY.en.guides.length).toBeGreaterThanOrEqual(3);
  const titles = allPages().map((page) => page.title);
  expect(new Set(titles).size).toBe(titles.length);
  const descriptions = allPages().map((page) => page.description);
  expect(new Set(descriptions).size).toBe(descriptions.length);
  for (const page of PAGES) {
    expect(
      CONTENT_FOLDERS.some((folder) => page.path.startsWith(folder)),
      page.path
    ).toBe(true);
  }
});

describe.each(LANGS)('the guides overview in %s', (lang) => {
  const guides = GUIDE_COPY[lang].guides;
  const doc = parse(RENDERED.get(GUIDES_PATHS[lang])!);
  const cards = [...doc.querySelectorAll('.guide-grid > li.guide-card')];

  test('has a card for every guide, in their order, each one link named by its title', () => {
    expect(cards).toHaveLength(guides.length);
    expect(doc.querySelectorAll('.guide-card')).toHaveLength(guides.length);
    for (const [index, card] of cards.entries()) {
      const guide = guides[index];
      // One link, in the heading: landing.css stretches it over the card.
      const links = card.querySelectorAll('a');
      expect(links).toHaveLength(1);
      expect(links[0].getAttribute('href')).toBe(`${GUIDES_PATHS[lang]}${guide.slug}/`);
      expect(links[0].textContent).toBe(guide.short ?? guide.h1);
      expect(links[0].parentElement!.tagName).toBe('H2');
      expect(links[0].hasAttribute('aria-label')).toBe(false);
      expect(card.querySelector('.summary')!.textContent).toBe(plainText(guide.summary));
      expect(card.querySelector('.kicker')?.textContent).toBe(guide.topic);
      expect(card.querySelector('.foot')!.textContent).toContain(
        COMMON[lang].guide.readingTime(readingMinutes(guide))
      );
    }
  });

  test('the drawings and “Read guide” are decoration: hidden, and no images to load', () => {
    for (const card of cards) {
      const art = card.querySelectorAll('svg');
      expect(art).toHaveLength(1);
      expect(art[0].getAttribute('class')).toBe('art');
      expect(art[0].getAttribute('aria-hidden')).toBe('true');
      expect(art[0].querySelector('text, image, use, a')).toBeNull();
      const go = card.querySelector('.go')!;
      expect(go.textContent).toBe(COMMON[lang].guide.read);
      expect(go.getAttribute('aria-hidden')).toBe('true');
    }
    expect(doc.querySelectorAll('main img')).toHaveLength(0);
  });

  test('opens with one sentence that names deep learning', () => {
    const lead = doc.querySelector('.page-head .lead')!.textContent!;
    expect(lead).toMatch(/deep[- ]learning/i);
    expect(lead.match(/[.!?](\s|$)/g)).toHaveLength(1);
    expect(lead.length).toBeLessThan(180);
  });

  test('fills its rows on a wide screen: three cards, or two wide ones', () => {
    const wide = cards.filter((card) => card.classList.contains('wide')).length;
    expect(wide % 2).toBe(0);
    expect((cards.length - wide) % 3).toBe(0);
    // The wide ones come first.
    expect(cards.slice(0, wide).every((card) => card.classList.contains('wide'))).toBe(true);
    // Two in a row on a tablet: the first alone if their number is odd.
    expect(cards[0].classList.contains('first')).toBe(cards.length % 2 === 1);
  });
});

test('the cards of the guides: reading time from the text, a drawing for any guide', () => {
  for (const lang of LANGS) {
    for (const guide of GUIDE_COPY[lang].guides) {
      expect(readingMinutes(guide), guide.id).toBeGreaterThanOrEqual(2);
      expect(readingMinutes(guide), guide.id).toBeLessThanOrEqual(30);
      // The card is one link: its sentence has none of its own.
      expect(guide.summary).not.toMatch(/<a\b/);
    }
  }
  const guide = GUIDE_COPY.en.guides[0];
  expect(readingMinutes({ ...guide, lead: 'Short.', sections: [] })).toBe(1);
  for (const id of ['no-code', 'tabular', 'segmentation', 'imbalanced', 'gdpr']) {
    expect(hasOwnGuideArt(id), id).toBe(true);
  }
  // Guides written later: by a word of their id, else the general drawing.
  expect(hasOwnGuideArt('activation-functions')).toBe(true);
  expect(guideArt('activation-functions')).not.toBe(guideArt('no-code'));
  expect(hasOwnGuideArt('something-new')).toBe(false);
  const drawings = [
    ...GUIDE_COPY.en.guides.map((entry) => entry.id),
    'activation-functions',
    'parameters-features-data-points',
    'something-new',
  ].map(guideArt);
  expect(new Set(drawings).size).toBe(drawings.length);
  for (const svg of drawings) {
    expect(svg).toMatch(/^<svg class="art" viewBox="0 0 320 160"/);
    expect(svg).not.toMatch(/style=|<style|href=|url\(/);
  }
});

test('the German pages show screenshots of the German app, the English ones of the English', () => {
  for (const id of new Set([...shotIds(CONTENT.de), ...contentShotIds('de')])) {
    expect(
      shotFiles(id, 'de').every((file) => file.name.startsWith(`${id}.de-`)),
      id
    ).toBe(true);
  }
  for (const id of new Set([...shotIds(CONTENT.en), ...contentShotIds('en')])) {
    expect(
      shotFiles(id, 'en').every((file) => /^[a-z-]+-\d+\.webp$/.test(file.name)),
      id
    ).toBe(true);
  }
});

test('the pages say where the server is, and never that it is in the EU', () => {
  // brickdnn.com's server is in Switzerland (Infomaniak, Geneva), which is in Europe but not
  // in the EU; the privacy policy (deploy/sites/brickdnn.com/legal/privacy.md) says the same.
  const texts = [
    ...LANGS.map((lang) => renderPage(lang, ASSETS)),
    ...RENDERED.values(),
    renderLlms(),
  ].map((html) => plainText(html.replace(/<script[\s\S]*?<\/script>/g, '')));
  const wrong =
    /\b(hosted in the EU|EU[- ]servers?|servers? in the EU|EU[- ]hosting)\b|in der EU gehostet|Servern? in der EU|EU-Server/i;
  for (const text of texts) {
    expect(text).not.toMatch(wrong);
  }
  for (const lang of LANGS) {
    const home = parse(renderPage(lang, ASSETS));
    expect(home.querySelector('#privacy')!.textContent).toMatch(/Switzerland|Schweiz/);
    // The section on data in Europe leads to the privacy policy and to the GDPR guide.
    const links = [...home.querySelectorAll('#privacy a')].map((a) => a.getAttribute('href'));
    const gdpr = GUIDE_COPY[lang].guides.find((guide) => guide.id === 'gdpr')!;
    expect(links).toEqual(
      expect.arrayContaining(['/privacy', `${GUIDES_PATHS[lang]}${gdpr.slug}/`])
    );
  }
});

/** Numbers that a page may state about an example: its parameters, shapes and settings. */
function facts(example: ExampleData): Set<number> {
  const found = new Set<number>();
  const add = (value: unknown) => {
    if (typeof value === 'number') {
      found.add(value);
    }
  };
  add(example.total_params);
  add(example.neural_params);
  // The parameters of all layers of a type, e.g. of all convolutions.
  const byType = new Map<string, number>();
  for (const node of example.nodes) {
    byType.set(node.type, (byType.get(node.type) ?? 0) + (node.num_params ?? 0));
  }
  byType.forEach(add);
  for (const node of example.nodes) {
    add(node.num_params);
    add(node.output_shape?.reduce((a, b) => a * b, 1));
    const numbers = Object.values(node.params)
      .flat()
      .filter((value): value is number => typeof value === 'number');
    numbers.forEach(add);
    for (const a of numbers) {
      for (const b of numbers) {
        add(a * b);
      }
    }
  }
  return found;
}

/** Facts about the data and the server's limits, not about an example itself. */
const DATA_FACTS = new Set([2_000, 5_000, 10_000, 50_000, 60_000, 70_000, 1_000_000]);

describe.each(EXAMPLES.map((example) => [example.id, example] as const))(
  'example %s',
  (id, example) => {
    test.each(LANGS)('the numbers in the %s text are those of the example', (lang) => {
      const copy = EXAMPLE_COPY[lang].examples.find((candidate) => candidate.id === id)!;
      const text = plainText(
        [copy.description, copy.lead, ...copy.what, ...copy.when, copy.train].join(' ')
      );
      const separator = lang === 'de' ? '.' : ',';
      const allowed = facts(example);
      for (const [number] of text.matchAll(new RegExp(`\\d{1,3}(?:\\${separator}\\d{3})+`, 'g'))) {
        const value = Number(number.split(separator).join(''));
        expect(allowed.has(value) || DATA_FACTS.has(value), `${number} (${lang})`).toBe(true);
      }
    });

    test.each(LANGS)('the %s page shows the layers, the diagram and the code', (lang) => {
      const doc = parse(RENDERED.get(examplePath(lang, id))!);
      const rows = [...doc.querySelectorAll('table.layers tbody tr')];
      expect(rows.map((row) => row.querySelector('th')!.textContent)).toEqual(
        example.nodes.map((node) => node.id)
      );
      expect(doc.querySelectorAll('svg.network .node')).toHaveLength(example.nodes.length);
      expect(doc.querySelectorAll('svg.network path.edge')).toHaveLength(example.edges.length);
      expect(doc.querySelector('svg.network title')!.textContent).toBeTruthy();
      // The code on the page is exactly the generator's (examples.json).
      expect(doc.querySelector('pre.code code')!.textContent).toBe(example.code);
      expect(doc.querySelector('.page-head a.button')!.getAttribute('href')).toBe(
        `/app?example=${id}`
      );
    });
  }
);

test('examples.json: parameter counts add up, every script is PyTorch or scikit-learn', () => {
  for (const example of EXAMPLES) {
    if (example.family === 'neural') {
      expect(example.total_params).toBe(
        example.nodes.reduce((sum, node) => sum + (node.num_params ?? 0), 0)
      );
    }
    expect(example.code).toMatch(/^# Built with Brick DNN/);
    expect(example.code).toMatch(
      example.framework === 'pytorch' ? /^import torch$/m : /^from sklearn\./m
    );
  }
});

// The backend makes examples.json (python -m brickdnn.examples.export), and its tests check
// that the file is current. This test does the same when Python is there (as in CI): the
// code generator needs nothing but Python. It writes nothing.
const python = spawnSync('python3', ['--version']).status === 0;
test.skipIf(!python)('examples.json is what the backend’s generator makes now', () => {
  const result = spawnSync('python3', ['-m', 'brickdnn.examples.export', '--stdout'], {
    cwd: join(REPO, 'backend'),
    encoding: 'utf8',
    maxBuffer: 64 * 1024 * 1024,
  });
  expect(result.stderr).toBe('');
  expect(result.stdout === readFileSync(EXAMPLES_FILE, 'utf8')).toBe(true);
});

test('nginx.conf serves the content pages with the strict CSP and redirects to the slash', () => {
  const nginx = readFileSync(join(FRONTEND, 'nginx.conf'), 'utf8');
  const folders = CONTENT_FOLDERS.map((folder) => folder.slice(1, -1)).join('|');
  expect(nginx).toContain(
    `location ~ ^/(${folders})(/[a-z0-9-]+)?$ {\n        return 301 /$1$2/$is_args$args;`
  );
  const block = nginx.split(`location ~ ^/(${folders})/([a-z0-9-]+/)?$ {`)[1].split('\n    }')[0];
  expect(block).toContain(
    `Content-Security-Policy "default-src 'none'; style-src 'self'; img-src 'self'; connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"`
  );
  expect(block).toContain(`try_files \${uri}index.html /app.html;`);
  // Every page's address is one that location serves.
  const pattern = new RegExp(`^/(${folders})/([a-z0-9-]+/)?$`);
  for (const page of PAGES) {
    expect(page.path).toMatch(pattern);
  }
});

test('long type names in the diagram wrap at their capitals', () => {
  expect(titleLines('Conv2d')).toEqual(['Conv2d']);
  expect(titleLines('HistGradientBoostingClassifier')).toEqual([
    'HistGradient',
    'Boosting',
    'Classifier',
  ]);
  expect(titleLines('AdaptiveAvgPool2d')).toEqual(['AdaptiveAvg', 'Pool2d']);
});
