/** The images of the landing pages (frontend/landing/images/) and their sizes in pixels. */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

export const IMAGE_DIR = join(import.meta.dirname, 'images');

/** The editor screenshot in two widths (Playwright, deviceScaleFactor 2), smallest first. */
export const SCREENSHOTS = ['editor-1280.webp', 'editor-2560.webp'];
/** Open Graph / Twitter card, 1200 × 630. */
export const OG_IMAGE = 'og-image.png';
/** The logo as PNG for the Organization in the JSON-LD (search engines want a raster image). */
export const LOGO = 'logo-512.png';

export interface Size {
  width: number;
  height: number;
}

/** Width and height of a PNG or WebP file, from its header. */
export function imageSize(data: Uint8Array): Size {
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  const ascii = (start: number, length: number) =>
    String.fromCharCode(...data.subarray(start, start + length));
  if (ascii(1, 3) === 'PNG') {
    return { width: view.getUint32(16), height: view.getUint32(20) };
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
  throw new Error('Not a PNG or WebP image');
}

export function readImage(name: string): { data: Buffer; size: Size } {
  const data = readFileSync(join(IMAGE_DIR, name));
  return { data, size: imageSize(data) };
}
