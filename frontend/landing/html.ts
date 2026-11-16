/** Small helpers for writing the pages' HTML, shared by render.ts, layout.ts and pages.ts. */

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

/** JSON for a <script> element: `<` escaped, so the text can never end the element. */
export function scriptJson(value: unknown): string {
  return JSON.stringify(value).replace(/</g, '\\u003c');
}

/** Where an image is shown, for `sizes`: the width it takes on the page. */
export const SIZES = {
  hero: '(min-width: 72rem) 68rem, calc(100vw - 2.5rem)',
  row: '(min-width: 72rem) 37rem, (min-width: 60rem) 55vw, calc(100vw - 2.5rem)',
  tile: '(min-width: 72rem) 33rem, (min-width: 52rem) 45vw, calc(100vw - 2.5rem)',
  step: '(min-width: 60rem) 21rem, calc(100vw - 2.5rem)',
  /** In the text column of a guide. */
  article: '(min-width: 50rem) 46rem, calc(100vw - 2.5rem)',
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
