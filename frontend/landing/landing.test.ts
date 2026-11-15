import { mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { checkDist, checkLandingHtml, jsonLdTypes } from './check.ts';
import { IMAGE_DIR, imageSize, LOGO, OG_IMAGE, readImage, shotFiles } from './images.ts';
import { minifyCss } from './plugin.ts';
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
  EDITOR_PATH,
  INLINE_SCRIPT,
  inlineScriptHash,
  OVERAGE_PRICE,
  PLANS,
  SECTION_IDS,
} from './site.ts';

const FRONTEND = join(import.meta.dirname, '..');
const REPO = join(FRONTEND, '..');

const ASSETS: Assets = {
  css: '/assets/landing-abc.css',
  favicon: '/favicon.svg',
  logo: { src: '/assets/logo-512-abc.png', width: 512, height: 512 },
  images: Object.fromEntries(
    [...new Set([...shotIds(CONTENT.en), ...shotIds(CONTENT.de)])].map((id) => [
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
    for (const path of [EDITOR_PATH, ...APP_PAGES, '/', '/de/']) {
      expect(hrefs).toContain(path);
    }
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

  test('sends logged-in users to the editor', () => {
    const valid = { access: token(now - 10), refresh: token(now + 3600) };
    expect(run('/', valid)).toBe('/app');
    expect(run('/de/', valid)).toBe('/app');
    expect(run('/#faq', valid)).toBe('/app#faq');
  });

  test('ignores expired or broken logins', () => {
    expect(run('/', { access: token(now - 10), refresh: token(now - 10) })).toBeNull();
    expect(run('/', 'not json')).toBeNull();
    expect(run('/', { refresh: 'no-dots' })).toBeNull();
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
    write('index.html', renderPage('en', ASSETS));
    write('de/index.html', renderPage('de', ASSETS));
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
    rmSync(join(dir, ASSETS.css.slice(1)));
    rmSync(join(dir, 'app.html'));
    expect(checkDist(dir)).toEqual([
      'en: hreflang de',
      `en: ${ASSETS.css} is missing`,
      'de/index.html is missing',
      'app.html is missing',
    ]);
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
    'https://brickdnn.com/',
    'https://brickdnn.com/de/',
    ...APP_PAGES.map((path) => `https://brickdnn.com${path}`),
  ]);
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
  const ids = [...new Set([...shotIds(CONTENT.en), ...shotIds(CONTENT.de)])];
  for (const lang of LANGS) {
    for (const id of shotIds(CONTENT[lang])) {
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
