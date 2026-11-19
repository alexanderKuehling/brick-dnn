/**
 * The images of the landing pages (frontend/landing/images/): screenshots of the app made by
 * screenshots/shoot.mjs as `<id>-<width>.webp` (German ones, once the app is translated, as
 * `<id>.de-<width>.webp`), and the Open Graph image and the PNG logo, which
 * scripts/brand-images.mjs draws.
 */
import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import type { Lang } from './site.ts';

export const IMAGE_DIR = join(import.meta.dirname, 'images');

/** Open Graph / Twitter card, 1200 × 630. */
export const OG_IMAGE = 'og-image.jpg';
/** The logo as PNG for the Organization in the JSON-LD (search engines want a raster image). */
export const LOGO = 'logo-512.png';

export interface Size {
  width: number;
  height: number;
}

/** One file of a screenshot. */
export interface ShotFile extends Size {
  name: string;
}

/** Width and height of a PNG, JPEG or WebP file, from its header. */
export function imageSize(data: Uint8Array): Size {
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  const ascii = (start: number, length: number) =>
    String.fromCharCode(...data.subarray(start, start + length));
  if (ascii(1, 3) === 'PNG') {
    return { width: view.getUint32(16), height: view.getUint32(20) };
  }
  if (data[0] === 0xff && data[1] === 0xd8) {
    // JPEG: the first start-of-frame marker has the size.
    let at = 2;
    while (at + 9 < data.length) {
      const marker = data[at + 1];
      const length = view.getUint16(at + 2);
      if (marker >= 0xc0 && marker <= 0xcf && ![0xc4, 0xc8, 0xcc].includes(marker)) {
        return { width: view.getUint16(at + 7), height: view.getUint16(at + 5) };
      }
      at += 2 + length;
    }
  }
  if (ascii(0, 4) === 'RIFF' && ascii(8, 4) === 'WEBP') {
    const chunk = ascii(12, 4);
    if (chunk === 'VP8X') {
      const u24 = (at: number) => data[at] | (data[at + 1] << 8) | (data[at + 2] << 16);
      return { width: u24(24) + 1, height: u24(27) + 1 };
    }
    if (chunk === 'VP8L') {
      const bits = view.getUint32(21, true);
      return { width: (bits & 0x3fff) + 1, height: ((bits >> 14) & 0x3fff) + 1 };
    }
    if (chunk === 'VP8 ') {
      return {
        width: view.getUint16(26, true) & 0x3fff,
        height: view.getUint16(28, true) & 0x3fff,
      };
    }
  }
  throw new Error('Not a PNG, JPEG or WebP image');
}

export function readImage(name: string): { data: Buffer; size: Size } {
  const data = readFileSync(join(IMAGE_DIR, name));
  return { data, size: imageSize(data) };
}

/**
 * The files of a screenshot, narrowest first: the language's own if there are any (`de`),
 * otherwise the English ones. Throws if there are none.
 */
export function shotFiles(id: string, lang: Lang, dir = IMAGE_DIR): ShotFile[] {
  const names = readdirSync(dir);
  const pick = (suffix: string) => {
    const pattern = new RegExp(`^${id}${suffix.replace('.', '\\.')}-(\\d+)\\.webp$`);
    return names
      .filter((name) => pattern.test(name))
      .map((name) => ({ name, ...imageSize(readFileSync(join(dir, name))) }))
      .sort((a, b) => a.width - b.width);
  };
  const own = lang === 'en' ? [] : pick(`.${lang}`);
  const result = own.length ? own : pick('');
  if (!result.length) {
    throw new Error(`No image files for the screenshot “${id}” in ${dir}`);
  }
  return result;
}
