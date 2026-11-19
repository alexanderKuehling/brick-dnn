/**
 * The checked-in brand images that are drawn, not photographed, all from the logo
 * (public/favicon.svg) and the one line the site says about itself:
 *
 * - public/brand/link-preview.png (1200 × 630): the preview image of a shared link
 *   (/share/…, /compare/…) and of the app's own pages in chats and social networks
 *   (index.html, backend/brick_dnn_core/link_previews.py). It names no address, so it is
 *   right on every server.
 * - landing/images/og-image.jpg (1200 × 630): the same for the landing pages of brickdnn.com.
 * - landing/images/logo-512.png: the logo as PNG, for the JSON-LD of the landing pages.
 * - ../docs/images/social-preview.png (1280 × 640): the repository's social preview, to upload
 *   under GitHub → Settings → Social preview.
 *
 * They are pages of HTML rendered by Chromium, so changing a word or a colour means changing
 * it below and running this again. Playwright is not a dependency of the app:
 *
 *   npm install --no-save playwright && npx playwright install chromium
 *   node scripts/brand-images.mjs
 *
 * (or a global installation: NODE_PATH="$(npm root -g)" node scripts/brand-images.mjs)
 *
 * The font is Inter; without it installed the system's sans-serif is used (look at the result).
 */
import { readFile, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { join } from 'node:path';

const FRONTEND = join(import.meta.dirname, '..');
const require = createRequire(import.meta.url);

/** Playwright from this project or, failing that, a global installation (NODE_PATH). */
function loadPlaywright() {
  const places = [undefined, ...(process.env.NODE_PATH ?? '').split(':').filter(Boolean)];
  for (const place of places) {
    try {
      return place
        ? require(require.resolve('playwright', { paths: [place] }))
        : require('playwright');
    } catch {
      // Try the next place.
    }
  }
  console.error('Playwright is missing: npm install --no-save playwright, or set NODE_PATH.');
  process.exit(2);
}

/** The layers of the drawing: a small CNN as the editor shows it (name, settings, shape). */
const LAYERS = [
  { name: 'Input', detail: '1 × 28 × 28', shape: '1 × 28 × 28', color: '#12b886' },
  { name: 'Conv2d', detail: '32 filters · 3×3', shape: '32 × 28 × 28', color: '#7950f2' },
  { name: 'MaxPool2d', detail: '2×2 · stride 2', shape: '32 × 14 × 14', color: '#7950f2' },
  { name: 'Flatten', detail: '', shape: '6272', color: '#4c6ef5' },
  { name: 'Linear', detail: '6272 → 10', shape: '10', color: '#4c6ef5' },
];

function layer({ name, detail, shape, color }) {
  return `<div class="layer"><div class="head"><i style="background:${color}"></i>${name}</div>
<div class="body"><span>${detail || '&nbsp;'}</span><b>${shape}</b></div></div>`;
}

function page({ width, height, logo, title, lines, foot }) {
  return `<!doctype html><meta charset="utf-8"><style>
  * { box-sizing: border-box; }
  html, body { margin: 0; }
  body { position: relative; overflow: hidden; width: ${width}px; height: ${height}px;
    background: #fff; color: #212529;
    font-family: Inter, ui-sans-serif, system-ui, sans-serif; }
  .text { position: absolute; top: 0; bottom: 14px; left: 84px; width: 600px;
    display: flex; flex-direction: column; justify-content: center; }
  .brand { display: flex; align-items: center; gap: 16px; font-size: 34px; font-weight: 800;
    letter-spacing: -0.02em; }
  .brand img { width: 60px; height: 60px; }
  h1 { margin: 52px 0 0; font-size: 54px; font-weight: 700; line-height: 1.12;
    letter-spacing: -0.03em; }
  p { margin: 28px 0 0; color: #495057; font-size: 25px; line-height: 1.45; }
  .foot { margin-top: 44px; color: #2f29c2; font-size: 22px; font-weight: 600; }
  .canvas { position: absolute; top: 0; right: 0; bottom: 14px; width: 440px;
    display: flex; flex-direction: column; justify-content: center; gap: 22px;
    padding: 0 84px 0 0; }
  .canvas::before { position: absolute; z-index: 0; top: 80px; bottom: 80px; left: 178px;
    width: 2px; background: #ced4da; content: ''; }
  .layer { position: relative; z-index: 1; overflow: hidden; border: 1.5px solid #dee2e6;
    border-radius: 10px; background: #fff;
    box-shadow: 0 1px 2px rgb(16 24 40 / 0.05), 0 10px 24px -16px rgb(16 24 40 / 0.3); }
  .head { display: flex; align-items: center; gap: 10px; padding: 9px 16px;
    background: #f3f2ff; font-size: 19px; font-weight: 700; }
  .head i { width: 11px; height: 11px; border-radius: 3px; }
  .body { display: flex; justify-content: space-between; gap: 12px; padding: 9px 16px 10px;
    color: #868e96; font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: 15px; }
  .body b { color: #212529; font-weight: 500; }
  .bar { position: absolute; right: 0; bottom: 0; left: 0; height: 14px; background: #3d36db; }
</style>
<div class="text">
<div class="brand"><img src="data:image/svg+xml;base64,${logo}" alt="">Brick DNN</div>
<h1>${title}</h1>
<p>${lines}</p>
${foot ? `<div class="foot">${foot}</div>` : ''}
</div>
<div class="canvas">${LAYERS.map(layer).join('\n')}</div>
<div class="bar"></div>`;
}

const TAGLINE = 'See every shape. Keep the code.';
const LINE = 'Build neural networks from bricks. Read the PyTorch code they become.';

const IMAGES = [
  {
    file: join(FRONTEND, 'public/brand/link-preview.png'),
    width: 1200,
    height: 630,
    title: TAGLINE,
    lines: LINE,
    foot: '',
  },
  {
    file: join(FRONTEND, 'landing/images/og-image.jpg'),
    width: 1200,
    height: 630,
    title: TAGLINE,
    lines: LINE,
    foot: 'brickdnn.com',
  },
  {
    file: join(FRONTEND, '../docs/images/social-preview.png'),
    width: 1280,
    height: 640,
    title: TAGLINE,
    lines: 'The deep-learning lab for courses: build networks from bricks in the browser, export readable PyTorch code.',
    foot: 'Open source · Apache 2.0',
  },
];

const svg = await readFile(join(FRONTEND, 'public/favicon.svg'));
const logo = svg.toString('base64');
const browser = await loadPlaywright().chromium.launch({ args: ['--no-sandbox'] });
try {
  for (const image of IMAGES) {
    const tab = await browser.newPage({
      viewport: { width: image.width, height: image.height },
      deviceScaleFactor: 1,
    });
    await tab.setContent(page({ ...image, logo }));
    await tab.evaluate(() => document.fonts.ready);
    const type = image.file.endsWith('.jpg') ? { type: 'jpeg', quality: 90 } : { type: 'png' };
    await writeFile(image.file, await tab.screenshot(type));
    await tab.close();
    console.log(`${image.file} ${image.width}×${image.height}`);
  }
  // The logo alone, 512 × 512 with a transparent background.
  const tab = await browser.newPage({ viewport: { width: 512, height: 512 }, deviceScaleFactor: 1 });
  await tab.setContent(
    `<body style="margin:0"><img width="512" height="512" src="data:image/svg+xml;base64,${logo}" alt="">`
  );
  const file = join(FRONTEND, 'landing/images/logo-512.png');
  await writeFile(file, await tab.screenshot({ type: 'png', omitBackground: true }));
  console.log(`${file} 512×512`);
} finally {
  await browser.close();
}
