#!/usr/bin/env node
/**
 * The screenshots of the landing pages, taken with Playwright from a running Brick DNN with
 * real data (setup.mjs) and written to ../images as WebP files in several widths. See README.md.
 *
 *   node landing/screenshots/shoot.mjs [--app http://127.0.0.1:4173] [--lang en|de]
 *        [--only editor,data-prep] [--raw /tmp/raw] [--out ../images]
 */
import { mkdir, readdir, readFile, rm, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { setup } from './setup.mjs';
import { prepareContext, SHOTS } from './shots.mjs';

const QUALITY = 0.8;

const { values: args } = parseArgs({
  options: {
    app: { type: 'string', default: 'http://127.0.0.1:4173' },
    lang: { type: 'string', default: 'en' },
    only: { type: 'string' },
    raw: { type: 'string' },
    out: { type: 'string', default: join(import.meta.dirname, '..', 'images') },
    'no-og': { type: 'boolean', default: false },
  },
});

async function loadPlaywright() {
  const candidates = ['playwright', process.env.PLAYWRIGHT_MODULE].filter(Boolean);
  for (const name of candidates) {
    try {
      const module = await import(name);
      return module.chromium ? module : module.default;
    } catch {
      // try the next one
    }
  }
  throw new Error(
    'Playwright is missing: `npm install --no-save playwright && npx playwright install chromium` ' +
      '(or set PLAYWRIGHT_MODULE to the path of its index.mjs).'
  );
}

/** The widths written for a shot: a phone width, the width shown on the page and twice that. */
function widthsOf(shot, rawWidth) {
  const wanted = shot.widths ?? [Math.min(480, shot.display), shot.display, shot.display * 2];
  return [...new Set(wanted.map((w) => Math.min(w, rawWidth)))].sort((a, b) => a - b);
}

/** PNG → WebP in the given widths, encoded by Chromium itself (no image library needed). */
async function encode(page, png, widths, type = 'image/webp', quality = QUALITY) {
  return page.evaluate(
    async ({ data, widths, type, quality }) => {
      const blob = await (await fetch(`data:image/png;base64,${data}`)).blob();
      const bitmap = await createImageBitmap(blob);
      const results = [];
      for (const width of widths) {
        const height = Math.round((bitmap.height * width) / bitmap.width);
        // Halve step by step, then scale to the width: sharper than one big step.
        let source = bitmap;
        let w = bitmap.width;
        let h = bitmap.height;
        while (w / 2 >= width) {
          const half = new OffscreenCanvas(Math.round(w / 2), Math.round(h / 2));
          const context = half.getContext('2d');
          context.imageSmoothingQuality = 'high';
          context.drawImage(source, 0, 0, half.width, half.height);
          source = half;
          w = half.width;
          h = half.height;
        }
        const canvas = new OffscreenCanvas(width, height);
        const context = canvas.getContext('2d');
        context.imageSmoothingQuality = 'high';
        context.drawImage(source, 0, 0, width, height);
        const out = await canvas.convertToBlob({ type, quality });
        const url = await new Promise((resolve) => {
          const reader = new FileReader();
          reader.onload = () => resolve(reader.result);
          reader.readAsDataURL(out);
        });
        results.push({ width, height, data: url.split(',')[1] });
      }
      return results;
    },
    { data: png.toString('base64'), widths, type, quality }
  );
}

/** A browser page of the app, logged in as `user`, with the guide already seen. */
async function appPage(browser, shot, ctx) {
  const [width, height] = shot.viewport ?? [1440, 900];
  const context = await browser.newContext({
    viewport: { width, height },
    deviceScaleFactor: 2,
    colorScheme: 'light',
    locale: args.lang === 'de' ? 'de-DE' : 'en-GB',
  });
  const api = ctx.users[shot.user ?? 'main'];
  const tokens = await api.login();
  const storage = {
    'brickdnn:tokens': JSON.stringify(tokens),
    'brickdnn:guide-seen': '2026-01-01T00:00:00.000Z',
    'brickdnn:support-hint': '2026-01-01T00:00:00.000Z',
    'brickdnn:language': args.lang,
    ...(shot.storage?.(ctx) ?? {}),
  };
  await context.addInitScript(
    ([storage, state]) => {
      for (const [key, value] of Object.entries(storage)) {
        localStorage.setItem(key, value);
      }
      // A navigation state as the app's own links set it (e.g. open a run in the panel), once.
      if (state && !sessionStorage.getItem('landing-state')) {
        sessionStorage.setItem('landing-state', '1');
        history.replaceState({ usr: state, key: 'landing', idx: 0 }, '');
      }
    },
    [storage, shot.state?.(ctx) ?? null]
  );
  const page = await context.newPage();
  page.setDefaultTimeout(15000);
  return page;
}

async function main() {
  const OUT = args.out;
  const playwright = await loadPlaywright();
  const only = args.only?.split(',');
  const shots = SHOTS.filter((shot) => !only || only.includes(shot.id));
  console.log(`Setting up the data on ${args.app}`);
  const ctx = await setup(args.app);
  await prepareContext(ctx, args.app);
  ctx.lang = args.lang;
  // The language saved on an account wins over the browser's (AuthContext): set it too.
  for (const api of Object.values(ctx.users)) {
    await api.request('PATCH', '/api/auth/account/language/', { language: args.lang });
  }

  const browser = await playwright.chromium.launch({ args: ['--no-sandbox'] });
  const encoder = await browser.newPage();
  await mkdir(OUT, { recursive: true });
  if (args.raw) {
    await mkdir(args.raw, { recursive: true });
  }
  const suffix = args.lang === 'en' ? '' : `.${args.lang}`;
  const files = await readdir(OUT);
  const raws = {};

  for (const shot of shots) {
    console.log(`Shot ${shot.id}`);
    const page = await appPage(browser, shot, ctx);
    await page.goto(args.app + shot.path(ctx), { waitUntil: 'networkidle' });
    await page.waitForTimeout(1200);
    // A shot can open more pages, e.g. a second person in a shared network.
    const others = [];
    const open = async (spec) => {
      const other = await appPage(browser, spec, ctx);
      others.push(other);
      return other;
    };
    const wanted = await shot.take(page, ctx, open);
    // Kept inside the window: near the end of a page the part moves up instead of shrinking.
    const viewport = page.viewportSize();
    const clip = {
      ...wanted,
      y: Math.max(0, Math.min(wanted.y, viewport.height - wanted.height)),
    };
    await page.waitForTimeout(400);
    const png = await page.screenshot({ clip, animations: 'disabled', caret: 'hide' });
    for (const other of [page, ...others]) {
      await other.context().close();
    }
    raws[shot.id] = png;
    if (args.raw) {
      await writeFile(join(args.raw, `${shot.id}${suffix}.png`), png);
    }
    const old = new RegExp(`^${shot.id}${suffix.replace('.', '\\.')}-\\d+\\.webp$`);
    for (const file of files.filter((name) => old.test(name))) {
      await rm(join(OUT, file));
    }
    for (const image of await encode(encoder, png, widthsOf(shot, clip.width * 2))) {
      const name = `${shot.id}${suffix}-${image.width}.webp`;
      await writeFile(join(OUT, name), Buffer.from(image.data, 'base64'));
      console.log(`  ${name} ${image.width}×${image.height}`);
    }
  }

  // The Open Graph image: the editor next to the name of the site, 1200 × 630.
  if (!args['no-og'] && raws.editor && args.lang === 'en') {
    const og = await browser.newPage({ viewport: { width: 1200, height: 630 } });
    const logo = await readFile(join(import.meta.dirname, '..', '..', 'public', 'favicon.svg'));
    await og.setContent(ogHtml(raws.editor.toString('base64'), logo.toString('base64')));
    await og.waitForTimeout(300);
    const png = await og.screenshot();
    const [jpeg] = await encode(encoder, png, [1200], 'image/jpeg', 0.88);
    await writeFile(join(OUT, 'og-image.jpg'), Buffer.from(jpeg.data, 'base64'));
    console.log('  og-image.jpg');
  }
  await browser.close();
}

function ogHtml(editor, logo) {
  return `<!doctype html><html><head><style>
  body { margin: 0; width: 1200px; height: 630px; overflow: hidden; position: relative; background: #fff;
    color: #212529; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }
  .text { position: absolute; left: 72px; top: 72px; width: 520px; }
  .brand { display: flex; align-items: center; gap: 14px; font-size: 30px; font-weight: 800;
    letter-spacing: -0.02em; }
  .brand img { width: 48px; height: 48px; }
  h1 { margin: 56px 0 0; font-size: 50px; line-height: 1.1; letter-spacing: -0.028em; }
  p { margin: 28px 0 0; font-size: 23px; line-height: 1.45; color: #495057; }
  .url { position: absolute; left: 72px; bottom: 64px; font-size: 22px; font-weight: 600; color: #2f29c2; }
  .shot { position: absolute; left: 650px; top: 72px; width: 760px; height: 520px; border: 1px solid #dee2e6;
    border-radius: 14px; overflow: hidden; box-shadow: 0 20px 50px -24px rgba(16, 24, 40, .35); }
  .shot img { width: 1440px; margin-left: -255px; margin-top: -175px; }
  .bar { position: absolute; left: 0; right: 0; bottom: 0; height: 8px; background: #3d36db; }
  </style></head><body>
  <div class="text"><div class="brand"><img src="data:image/svg+xml;base64,${logo}" alt="">Brick DNN</div>
  <h1>Build neural networks from layers. Export the PyTorch code.</h1>
  <p>Live shape checks, training on servers in Switzerland, readable PyTorch and scikit-learn code.</p></div>
  <div class="url">brickdnn.com</div>
  <div class="shot"><img src="data:image/png;base64,${editor}" alt=""></div><div class="bar"></div>
  </body></html>`;
}

await main();
