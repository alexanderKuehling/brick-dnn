/**
 * Vite plugin for the static pages: the landing pages `/` (English) and `/de/` (German) and the
 * content pages (pages.ts: examples, guides, the German pricing page) are plain HTML made at
 * build time, without the React app; the editor and every other page of the app are served
 * from app.html.
 *
 * - Build: index.html (the app's page) is written as app.html; the static pages, their
 *   stylesheet and images (hashed, under assets/), robots.txt, sitemap.xml and llms.txt are
 *   added, and the result is checked (check.ts): a broken page fails the build.
 * - `vite` (development) and `vite preview` serve the same addresses as nginx.conf does in
 *   production: the static pages at their paths (a path without its trailing slash redirected
 *   to the one with it), /?… redirected to /app?…, the app everywhere else.
 */
import { readFileSync } from 'node:fs';
import type { IncomingMessage, ServerResponse } from 'node:http';
import { join } from 'node:path';
import type { Plugin } from 'vite';
import { checkDist } from './check.ts';
import { IMAGE_DIR, LOGO, OG_IMAGE, readImage, shotFiles } from './images.ts';
import { allPages, contentShotIds, renderContentPage, type PageMeta } from './pages.ts';
import {
  CONTENT,
  renderLlms,
  renderPage,
  renderRobots,
  renderSitemap,
  shotIds,
  type Assets,
} from './render.ts';
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

function assets(lang: Lang, url: (name: string, data: Buffer | string) => string): Assets {
  const og = readImage(OG_IMAGE);
  const logo = readImage(LOGO);
  const css = readFileSync(CSS_FILE, 'utf8');
  const images: Assets['images'] = {};
  for (const id of new Set([...shotIds(CONTENT[lang]), ...contentShotIds(lang)])) {
    images[id] = {
      sources: shotFiles(id, lang).map((file) => ({
        src: url(file.name, readImage(file.name).data),
        width: file.width,
        height: file.height,
      })),
    };
  }
  return {
    css: url('landing.css', css),
    favicon: '/favicon.svg',
    logo: { src: url(LOGO, logo.data), ...logo.size },
    images,
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
  jpg: 'image/jpeg',
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

const PAGES = new Map(allPages().map((page) => [page.path, page]));

/** The static page at a path, if any. */
export function staticPage(path: string): PageMeta | null {
  return PAGES.get(path) ?? null;
}

/** The HTML of a static page. */
export function renderStatic(page: PageMeta, pageAssets: Assets): string {
  return page.kind === 'home'
    ? renderPage(page.lang, pageAssets)
    : renderContentPage(page, pageAssets);
}

/**
 * The redirects of nginx.conf: `/?…` → `/app?…` (old links into the editor), a static page's
 * path without its trailing slash → with it (`/de` → `/de/`, `/examples/unet` →
 * `/examples/unet/`, keeping the query). Returns whether it answered.
 */
function redirects(req: IncomingMessage, res: ServerResponse): boolean {
  const [path, query] = (req.url ?? '/').split(/\?(.*)/s);
  if (path === '/' && query) {
    redirect(res, 302, `${EDITOR_PATH}?${query}`);
    return true;
  }
  if (!path.endsWith('/') && PAGES.has(`${path}/`)) {
    redirect(res, 301, `${path}/${query ? `?${query}` : ''}`);
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
        const page = staticPage(path);
        if (page) {
          send(
            res,
            'text/html',
            renderStatic(
              page,
              assets(page.lang, (name) => `${DEV_PREFIX}${name}`)
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
        const page = staticPage(path);
        if (page) {
          // The file of the page, as nginx serves it (try_files …/index.html).
          req.url = `/${page.file}${query ? `?${query}` : ''}`;
        } else if (isPageRequest(req, path)) {
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

        // Every file once, also when both pages use it.
        const emitted = new Map<string, string>();
        const url = (name: string, data: Buffer | string) => {
          if (!emitted.has(name)) {
            const source = name.endsWith('.css') ? minifyCss(String(data)) : data;
            const ref = this.emitFile({ type: 'asset', name, source });
            emitted.set(name, `/${this.getFileName(ref)}`);
          }
          return emitted.get(name)!;
        };
        for (const page of allPages()) {
          this.emitFile({
            type: 'asset',
            fileName: page.file,
            source: renderStatic(page, assets(page.lang, url)),
          });
        }
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
