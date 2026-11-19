import { spawnSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import {
  checkDist,
  checkLandingHtml,
  checkPageHtml,
  hreflangs,
  jsonLdTypes,
  LINK_PREVIEW_INCLUDE,
} from './check.ts';
import { BADGE_MARKDOWN, CITATION_BIBTEX, CITATION_TEXT } from './content/info.en.ts';
import { EXAMPLES, EXAMPLES_FILE, type ExampleData } from './examples/data.ts';
import { titleLines } from './examples/diagram.ts';
import { IMAGE_DIR, imageSize, LOGO, OG_IMAGE, readImage, shotFiles } from './images.ts';
import {
  allPages,
  contentPages,
  contentShotIds,
  EXAMPLE_COPY,
  examplePath,
  GUIDE_COPY,
  INFO_COPY,
} from './pages.ts';
import { minifyCss, renderStatic, staticPage } from './plugin.ts';
import {
  CONTENT,
  plainText,
  renderLlms,
  renderPage,
  renderRobots,
  renderSecurityTxt,
  renderSitemap,
  shotIds,
  type Assets,
} from './render.ts';
import {
  ABOUT_PATHS,
  APP_PAGES,
  BADGE_PATHS,
  CHANGELOG_URL,
  CONTENT_FOLDERS,
  DATA_PROTECTION_PATHS,
  EDITOR_PATH,
  EXAMPLES_PATHS,
  GUIDES_PATHS,
  INLINE_SCRIPT,
  inlineScriptHash,
  OVERAGE_PRICE,
  PLANS,
  pricingPath,
  RELEASE,
  REPOSITORY_URL,
  SECTION_IDS,
  SECURITY_PATHS,
  SECURITY_TXT_PATH,
  TEACHING_PATHS,
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
  /** The app's page as the build writes it: index.html with the address of the site. */
  const APP_HTML = readFileSync(join(FRONTEND, 'index.html'), 'utf8').replaceAll(
    '%SITE_URL%',
    'https://brickdnn.com'
  );

  function write(path: string, content: string | Buffer = '') {
    mkdirSync(dirname(join(dir, path)), { recursive: true });
    writeFileSync(join(dir, path), content);
  }

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), 'landing-'));
    for (const page of allPages()) {
      write(page.file, renderStatic(page, ASSETS));
    }
    write('app.html', APP_HTML);
    write('robots.txt', renderRobots());
    write('sitemap.xml', renderSitemap());
    write('llms.txt', renderLlms());
    write(SECURITY_TXT_PATH.slice(1), renderSecurityTxt());
    for (const path of [
      ASSETS.css,
      ASSETS.favicon,
      BADGE_PATHS.en,
      BADGE_PATHS.de,
      '/brand/link-preview.png',
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

  test('finds an app page without the link preview of shared links, or without its image', () => {
    write('app.html', APP_HTML.replace(LINK_PREVIEW_INCLUDE, ''));
    rmSync(join(dir, 'brand/link-preview.png'));
    expect(checkDist(dir)).toEqual([
      'app.html lacks the include for the link preview of shared links',
      'app.html: /brand/link-preview.png is missing',
    ]);
    rmSync(join(dir, SECURITY_TXT_PATH.slice(1)));
    expect(checkDist(dir)).toContain('.well-known/security.txt is missing');
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
const KNOWN_PATHS = new Set([
  ...allPages().map((page) => page.path),
  EDITOR_PATH,
  ...APP_PAGES,
  SECURITY_TXT_PATH,
]);

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

// ------------------------------------------------------- branding: previews, badge, trust

test('every page has the meta tags of a link preview, with the preview image', () => {
  for (const page of allPages()) {
    const doc = parse(renderStatic(page, ASSETS));
    const meta = (name: string) =>
      doc.querySelector(`meta[property="${name}"], meta[name="${name}"]`)?.getAttribute('content');
    expect(meta('og:title'), page.path).toBe(page.title);
    expect(meta('og:description'), page.path).toBe(page.description);
    expect(meta('og:url'), page.path).toBe(`https://brickdnn.com${page.path}`);
    expect(meta('og:site_name'), page.path).toBe('Brick DNN');
    expect(meta('og:image'), page.path).toBe(`https://brickdnn.com${ASSETS.ogImage.src}`);
    expect(meta('og:image:width'), page.path).toBe('1200');
    expect(meta('og:image:height'), page.path).toBe('630');
    expect(meta('og:image:alt')!.length, page.path).toBeGreaterThan(20);
    expect(meta('twitter:card'), page.path).toBe('summary_large_image');
    expect(meta('twitter:title'), page.path).toBe(page.title);
    expect(meta('twitter:image'), page.path).toBe(meta('og:image'));
  }
});

test('one line about Brick DNN: the same tagline on the landing page, the app and the README', () => {
  const tagline = 'See every shape. Keep the code.';
  expect(CONTENT.en.hero.title).toBe(tagline);
  expect(CONTENT.de.hero.title).toBe('Jede Form im Blick. Der Code gehört Ihnen.');
  const app = readFileSync(join(FRONTEND, 'index.html'), 'utf8');
  expect(app).toContain(`<title>Brick DNN · ${tagline}</title>`);
  expect(readFileSync(join(REPO, 'README.md'), 'utf8')).toContain(tagline);
  expect(readFileSync(join(FRONTEND, 'scripts/brand-images.mjs'), 'utf8')).toContain(tagline);
  // Deep learning stays in what search engines read.
  for (const lang of LANGS) {
    expect(`${CONTENT[lang].title} ${CONTENT[lang].description}`).toMatch(/deep[- ]learning/i);
    expect(CONTENT[lang].title).toContain('PyTorch');
  }
  // No claim that somebody uses it: there is nobody to name yet.
  const texts = allPages().map((page) => plainText(renderStatic(page, ASSETS)));
  for (const text of texts) {
    expect(text).not.toMatch(/is used (for|in) teaching|wird in der Lehre eingesetzt/i);
  }
});

test('the logo is flat and the drawn images have their sizes', () => {
  const logo = readFileSync(join(FRONTEND, 'public/favicon.svg'), 'utf8');
  expect(logo).not.toMatch(/ellipse|polygon/);
  expect(new Set(logo.match(/#[0-9a-f]{3,6}/gi))).toEqual(new Set(['#3d36db', '#fff']));
  expect(readImage(LOGO).size).toEqual({ width: 512, height: 512 });
  const size = (path: string) => imageSize(readFileSync(join(REPO, path)));
  expect(size('frontend/public/brand/link-preview.png')).toEqual({ width: 1200, height: 630 });
  expect(size('docs/images/social-preview.png')).toEqual({ width: 1280, height: 640 });
  for (const file of ['frontend/public/brand/link-preview.png', 'docs/images/social-preview.png']) {
    expect(readFileSync(join(REPO, file)).length, file).toBeLessThan(300_000);
  }
});

describe('the pages about Brick DNN', () => {
  const pages = PAGES.filter((page) => page.kind === 'info');
  const html = (path: string) => RENDERED.get(path)!;
  const text = (path: string) => parse(html(path)).querySelector('main')!.textContent!;

  test('exist in both languages, in the sitemap, and every page links to them', () => {
    expect(pages.map((page) => page.path).sort()).toEqual(
      [TEACHING_PATHS, ABOUT_PATHS, SECURITY_PATHS, DATA_PROTECTION_PATHS]
        .flatMap((paths) => [paths.en, paths.de])
        .sort()
    );
    expect(INFO_COPY.en.map((page) => page.id)).toEqual(INFO_COPY.de.map((page) => page.id));
    const sitemap = renderSitemap();
    for (const page of pages) {
      expect(sitemap).toContain(`<loc>https://brickdnn.com${page.path}</loc>`);
      expect(page.alternates[page.lang]).toBe(page.path);
    }
    for (const page of allPages()) {
      const footer = parse(renderStatic(page, ASSETS)).querySelector('footer')!;
      const hrefs = [...footer.querySelectorAll('a')].map((a) => a.getAttribute('href'));
      for (const paths of [TEACHING_PATHS, ABOUT_PATHS, SECURITY_PATHS]) {
        expect(hrefs, page.path).toContain(paths[page.lang]);
      }
    }
    for (const lang of LANGS) {
      const home = parse(renderPage(lang, ASSETS));
      expect(home.querySelector('#teaching a[href^="/"]')!.getAttribute('href')).toBe(
        TEACHING_PATHS[lang]
      );
    }
    expect(renderLlms()).toContain(`https://brickdnn.com${TEACHING_PATHS.en}`);
  });

  test('teaching: what a course relies on, the badge, the discount and the citation', () => {
    for (const lang of LANGS) {
      const page = text(TEACHING_PATHS[lang]);
      for (const dataset of ['MNIST', 'Sentiment', 'Key lookup', 'Topics', 'Shapes', 'Cells']) {
        expect(page).toContain(dataset);
      }
      expect(page).toMatch(/Docker Compose/);
      expect(page).toMatch(lang === 'en' ? /Geneva/ : /Genf/);
      expect(page).toContain('alexander.kuehling@brickdnn.com');
      expect(page).toContain(CITATION_TEXT);
      expect(page).toContain(CITATION_BIBTEX);
      // The discount in the words of the landing page.
      expect(html(TEACHING_PATHS[lang])).toContain(CONTENT[lang].teaching.discount);
      const doc = parse(html(TEACHING_PATHS[lang]));
      expect(doc.querySelector('main img')!.getAttribute('src')).toBe(BADGE_PATHS[lang]);
      expect(doc.querySelector('.cta a')!.getAttribute('href')).toBe(EDITOR_PATH);
    }
    expect(text(TEACHING_PATHS.en)).toContain(BADGE_MARKDOWN);
    // The same snippet in the docs and the README.
    expect(readFileSync(join(REPO, 'docs/badge.md'), 'utf8')).toContain(BADGE_MARKDOWN);
    expect(readFileSync(join(REPO, 'README.md'), 'utf8')).toContain(BADGE_MARKDOWN);
    expect(readFileSync(join(REPO, 'docs/badge.md'), 'utf8')).toContain(CITATION_TEXT);
  });

  test('the citation is the one of CITATION.cff and of the released version', () => {
    const cff = readFileSync(join(REPO, 'CITATION.cff'), 'utf8');
    expect(cff).toContain(`version: "${RELEASE.version}"`);
    expect(cff).toContain(`date-released: "${RELEASE.date}"`);
    expect(cff).toContain(`repository-code: "${REPOSITORY_URL}"`);
    expect(cff).toMatch(/family-names: "Kühling"\n\s+given-names: "Alexander"/);
    expect(readFileSync(join(REPO, 'backend/pyproject.toml'), 'utf8')).toContain(
      `version = "${RELEASE.version}"`
    );
    expect(readFileSync(join(REPO, 'CHANGELOG.md'), 'utf8')).toContain(
      `## ${RELEASE.version} (${RELEASE.date})`
    );
    expect(CITATION_TEXT).toContain(RELEASE.version);
  });

  test('about: one person, the repository, the changelog, and what happens if it ends', () => {
    for (const lang of LANGS) {
      const doc = parse(html(ABOUT_PATHS[lang]));
      const page = text(ABOUT_PATHS[lang]);
      expect(page).toMatch(lang === 'en' ? /one person/ : /einer Person/);
      expect(page).toContain('Alexander Kühling');
      expect(page).toMatch(lang === 'en' ? /Geneva, Switzerland/ : /Genf, Schweiz/);
      const hrefs = [...doc.querySelectorAll('main a')].map((a) => a.getAttribute('href'));
      expect(hrefs).toEqual(
        expect.arrayContaining([
          REPOSITORY_URL,
          `${REPOSITORY_URL}/blob/main/LICENSE`,
          CHANGELOG_URL,
          '/imprint',
          '/privacy',
          '/refund',
          '/terms',
          SECURITY_PATHS[lang],
          'mailto:alexander.kuehling@brickdnn.com',
        ])
      );
      // No postal address: that is the imprint's.
      expect(page).not.toMatch(/\b\d{5}\b|straße|strasse|street/i);
      const end = doc.querySelector(lang === 'en' ? '#continuity' : '#fortbestand')!.textContent!;
      expect(end).toMatch(/90/);
      expect(end).toMatch(/Apache/);
      expect(end).toMatch(lang === 'en' ? /refunded in proportion/ : /erstatte ich anteilig/);
    }
    // English visitors are told that the legal texts are in German.
    const about = html(ABOUT_PATHS.en);
    for (const page of ['imprint', 'privacy', 'terms']) {
      expect(about).toMatch(new RegExp(`href="/${page}">[^<]+</a> \\(in German\\)`));
    }
    // The German pricing page has the commitments too.
    expect(text('/de/preise/')).toMatch(/mindestens 90 Tage/);
  });

  test('security: what is in place, how to report, and security.txt', () => {
    for (const lang of LANGS) {
      const doc = parse(html(SECURITY_PATHS[lang]));
      const page = text(SECURITY_PATHS[lang]);
      for (const fact of ['HTTPS', 'HSTS', 'Swiss Backup', '35', 'ZIP']) {
        expect(page).toContain(fact);
      }
      const hrefs = [...doc.querySelectorAll('main a')].map((a) => a.getAttribute('href'));
      expect(hrefs).toEqual(
        expect.arrayContaining([
          `${REPOSITORY_URL}/security/advisories/new`,
          'mailto:alexander.kuehling@brickdnn.com',
          SECURITY_TXT_PATH,
          `${REPOSITORY_URL}/actions/workflows/uptime.yml`,
        ])
      );
    }
    const txt = renderSecurityTxt(new Date('2026-11-19T10:00:00Z'));
    expect(txt).toContain('Contact: mailto:alexander.kuehling@brickdnn.com\n');
    expect(txt).toContain(`Contact: ${REPOSITORY_URL}/security/advisories/new\n`);
    expect(txt).toContain('Expires: 2027-11-19T00:00:00.000Z\n');
    expect(txt).toContain('Canonical: https://brickdnn.com/.well-known/security.txt\n');
    expect(txt).toContain(`Policy: https://brickdnn.com${SECURITY_PATHS.en}\n`);
    // What the page says is what SECURITY.md says.
    const policy = readFileSync(join(REPO, 'SECURITY.md'), 'utf8');
    expect(policy).toContain(`${REPOSITORY_URL}/security/advisories/new`);
    expect(policy).toContain('never from Python supplied by users');
    const nginx = readFileSync(join(FRONTEND, 'nginx.conf'), 'utf8');
    expect(nginx).toContain('\\.well-known/security\\.txt)$');
  });

  test('the fact sheet says what a course without accounts leaves on the server', () => {
    for (const lang of LANGS) {
      const page = text(DATA_PROTECTION_PATHS[lang]);
      for (const fact of ['Sentry', 'Brevo', 'Infomaniak', 'Paddle', '180', '35', 'Art. 28']) {
        expect(page).toContain(fact);
      }
      expect(page).toMatch(lang === 'en' ? /not stored/ : /nicht gespeichert/);
      expect(page).toMatch(lang === 'en' ? /No cookies/ : /Keine Cookies/);
      const crumbs = [...parse(html(DATA_PROTECTION_PATHS[lang])).querySelectorAll('.crumbs a')];
      expect(crumbs.map((a) => a.getAttribute('href'))).toEqual([
        lang === 'en' ? '/' : '/de/',
        TEACHING_PATHS[lang],
        DATA_PROTECTION_PATHS[lang],
      ]);
      expect(html(TEACHING_PATHS[lang])).toContain(`href="${DATA_PROTECTION_PATHS[lang]}"`);
    }
  });
});

test('the badge files are small SVG images with their text as the accessible name', () => {
  for (const [lang, alt] of [
    ['en', 'Open in Brick DNN'],
    ['de', 'In Brick DNN öffnen'],
  ] as const) {
    const svg = readFileSync(join(FRONTEND, 'public', BADGE_PATHS[lang].slice(1)), 'utf8');
    expect(svg.startsWith('<svg xmlns="http://www.w3.org/2000/svg"')).toBe(true);
    expect(svg).toContain(`role="img" aria-label="${alt}"`);
    expect(svg).toContain(`<title>${alt}</title>`);
    expect(svg).toContain('height="28"');
    // Nothing that loads or runs: GitHub and other sites show it through an image proxy.
    expect(svg).not.toMatch(/<script|href=|<image|<style|@import|url\(http/);
    expect(svg.length).toBeLessThan(1500);
  }
});

test('shared links: the app page has general tags and the include, nginx fills it', () => {
  const app = readFileSync(join(FRONTEND, 'index.html'), 'utf8');
  // The block is what every other page gets; the include follows it and names it.
  const block = app.split('<!--# block name="preview" -->')[1].split('<!--# endblock -->')[0];
  for (const tag of ['<title>', 'name="description"', 'og:title', 'og:image', 'twitter:card']) {
    expect(block).toContain(tag);
  }
  expect(block).toContain('content="%SITE_URL%/brand/link-preview.png"');
  expect(app.split('<!--# endblock -->')[1].trimStart().startsWith(LINK_PREVIEW_INCLUDE)).toBe(
    true
  );
  // Nothing else in the page is a server-side include.
  expect(app.match(/<!--#/g)).toHaveLength(3);

  const nginx = readFileSync(join(FRONTEND, 'nginx.conf'), 'utf8');
  const shared = nginx
    .split('location ~ ^/(?<preview_kind>share|compare)/(?<preview_token>[A-Za-z0-9_-]+)/?$ {')[1]
    .split('\n    }')[0];
  for (const line of [
    'ssi on;',
    'ssi_silent_errors on;',
    `add_header Content-Security-Policy "frame-ancestors 'none'";`,
    'add_header X-Robots-Tag "noindex";',
    'try_files /app.html =404;',
  ]) {
    expect(shared).toContain(line);
  }
  const include = nginx.split('location ^~ /api/link-preview/ {')[1].split('\n    }')[0];
  for (const line of ['internal;', 'proxy_intercept_errors on;', 'proxy_read_timeout 3s;']) {
    expect(include).toContain(line);
  }
  expect(include).toMatch(/error_page [\d ]*\b404\b[\d ]*\b500\b[\d ]* = @no_link_preview;/);
  // Crawlers may read shared links (previews need that); the header keeps them out of indexes.
  expect(renderRobots()).not.toMatch(/Disallow: \/(share|compare)/);
});
