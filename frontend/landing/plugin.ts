/**
 * Vite plugin for the static landing pages: `/` (English) and `/de/` (German) are plain HTML
 * made at build time, without the React app; the editor and every other page of the app are
 * served from app.html.
 *
 * - Build: index.html (the app's page) is written as app.html; the landing pages, their
 *   stylesheet and images (hashed, under assets/), robots.txt, sitemap.xml and llms.txt are
 *   added, and the result is checked (check.ts): a broken landing page fails the build.
 * - `vite` (development) and `vite preview` serve the same addresses as nginx.conf does in
 *   production: the landing pages at / and /de/, /?… redirected to /app?…, the app everywhere else.
 */
import { readFileSync } from 'node:fs';
import type { IncomingMessage, ServerResponse } from 'node:http';
import { join } from 'node:path';
import type { Plugin } from 'vite';
import { checkDist } from './check.ts';
import { IMAGE_DIR, LOGO, OG_IMAGE, readImage, SCREENSHOTS } from './images.ts';
import { renderLlms, renderPage, renderRobots, renderSitemap, type Assets } from './render.ts';
import { EDITOR_PATH, type Lang } from './site.ts';

const CSS_FILE = join(import.meta.dirname, 'landing.css');
/** Where development serves the stylesheet and images (the build hashes them instead). */
const DEV_PREFIX = '/@landing/';

/** Comments and needless white space out of the stylesheet (it has no strings or url()). */
export function minifyCss(css: string): string {
  return css
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/\s+/g, ' ')
    .replace(/\s*([{};,>])\s*/g, '$1')
    .replace(/:\s+/g, ':')
    .replace(/;}/g, '}')
    .replace(/\(\s+/g, '(')
    .replace(/\s+\)/g, ')')
    .trim();
}

function assets(url: (name: string, data: Buffer | string) => string): Assets {
  const shots = SCREENSHOTS.map((name) => ({ name, ...readImage(name) }));
  const og = readImage(OG_IMAGE);
  const logo = readImage(LOGO);
  const css = readFileSync(CSS_FILE, 'utf8');
  return {
    css: url('landing.css', css),
    favicon: '/favicon.svg',
    logo: { src: url(LOGO, logo.data), ...logo.size },
    screenshot: {
      sources: shots.map((shot) => ({ src: url(shot.name, shot.data), width: shot.size.width })),
      width: shots[0].size.width,
      height: shots[0].size.height,
    },
    ogImage: { src: url(OG_IMAGE, og.data), ...og.size },
  };
}

const TEXT_FILES: Record<string, { type: string; render: () => string }> = {
  'robots.txt': { type: 'text/plain', render: renderRobots },
  'sitemap.xml': { type: 'application/xml', render: renderSitemap },
  'llms.txt': { type: 'text/plain', render: renderLlms },
};

const CONTENT_TYPES: Record<string, string> = {
  css: 'text/css',
  png: 'image/png',
  webp: 'image/webp',
};

function send(res: ServerResponse, type: string, body: string | Buffer): void {
  res.statusCode = 200;
  res.setHeader('Content-Type', type.startsWith('image/') ? type : `${type}; charset=utf-8`);
  res.setHeader('Cache-Control', 'no-cache');
  res.end(body);
}

function redirect(res: ServerResponse, status: number, location: string): void {
  res.statusCode = status;
  res.setHeader('Location', location);
  res.end();
}

/** Which landing page a path is, if any. */
function landingPage(path: string): Lang | null {
  return path === '/' ? 'en' : path === '/de/' ? 'de' : null;
}

/**
 * The redirects of nginx.conf: `/?…` → `/app?…` (old links into the editor), `/de` → `/de/`.
 * Returns whether it answered.
 */
function redirects(req: IncomingMessage, res: ServerResponse): boolean {
  const [path, query] = (req.url ?? '/').split(/\?(.*)/s);
  if (path === '/' && query) {
    redirect(res, 302, `${EDITOR_PATH}?${query}`);
    return true;
  }
  if (path === '/de') {
    redirect(res, 301, '/de/');
    return true;
  }
  return false;
}

/** Requests for a page (not a file, the API or Vite's own modules). */
function isPageRequest(req: IncomingMessage, path: string): boolean {
  return (
    (req.method === 'GET' || req.method === 'HEAD') &&
    !/\.[a-z0-9]+$/i.test(path) &&
    !/^\/(api|admin|static|\.well-known|@|node_modules|src)(\/|$)/.test(path)
  );
}

export function landingPages(): Plugin {
  return {
    name: 'brickdnn:landing',
    enforce: 'post',

    configureServer(server) {
      // Before Vite's own middlewares, so that / is not the app's index.html.
      server.middlewares.use((req, res, next) => {
        const path = (req.url ?? '/').split('?')[0];
        if (path.startsWith(DEV_PREFIX)) {
          const name = path.slice(DEV_PREFIX.length);
          const extension = name.split('.').pop() ?? '';
          const file = name === 'landing.css' ? CSS_FILE : join(IMAGE_DIR, name);
          try {
            send(res, CONTENT_TYPES[extension] ?? 'application/octet-stream', readFileSync(file));
          } catch {
            next();
          }
          return;
        }
        if (redirects(req, res)) {
          return;
        }
        const lang = landingPage(path);
        if (lang) {
          send(
            res,
            'text/html',
            renderPage(
              lang,
              assets((name) => `${DEV_PREFIX}${name}`)
            )
          );
          return;
        }
        const text = TEXT_FILES[path.slice(1)];
        if (text) {
          send(res, text.type, text.render());
          return;
        }
        next(); // Everything else is the app (Vite serves index.html for its pages).
      });
    },

    configurePreviewServer(server) {
      // dist/ as nginx serves it: the app's pages from app.html.
      server.middlewares.use((req, res, next) => {
        const [path, query] = (req.url ?? '/').split(/\?(.*)/s);
        if (redirects(req, res)) {
          return;
        }
        if (!landingPage(path) && isPageRequest(req, path)) {
          req.url = `/app.html${query ? `?${query}` : ''}`;
        }
        next();
      });
    },

    generateBundle: {
      order: 'post',
      handler(_options, bundle) {
        const app = bundle['index.html'];
        if (!app || app.type !== 'asset') {
          this.error('index.html (the app) is not in the build');
        }
        delete bundle['index.html'];
        this.emitFile({ type: 'asset', fileName: 'app.html', source: app.source });

        const files = assets((name, data) => {
          const source = name.endsWith('.css') ? minifyCss(String(data)) : data;
          const ref = this.emitFile({ type: 'asset', name, source });
          return `/${this.getFileName(ref)}`;
        });
        this.emitFile({ type: 'asset', fileName: 'index.html', source: renderPage('en', files) });
        this.emitFile({
          type: 'asset',
          fileName: 'de/index.html',
          source: renderPage('de', files),
        });
        for (const [fileName, text] of Object.entries(TEXT_FILES)) {
          this.emitFile({ type: 'asset', fileName, source: text.render() });
        }
      },
    },

    writeBundle(options) {
      const problems = options.dir ? checkDist(options.dir) : ['no output directory'];
      if (problems.length) {
        this.error(`Landing pages: ${problems.join('; ')}`);
      }
    },
  };
}
