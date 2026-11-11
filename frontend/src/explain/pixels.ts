/** Turning drawings and image files into the pixel arrays an image network takes. */

export interface ImageFormat {
  channels: number;
  height: number;
  width: number;
}

/** The image format of a run from its preprocessing (null for other runs). */
export function imageFormat(
  preprocessing: Record<string, unknown> | null | undefined
): ImageFormat | null {
  if (!preprocessing || preprocessing.kind !== 'image') {
    return null;
  }
  const { channels, height, width } = preprocessing as Record<string, unknown>;
  if (typeof channels !== 'number' || typeof height !== 'number' || typeof width !== 'number') {
    return null;
  }
  return { channels, height, width };
}

/**
 * RGBA pixels (as from a canvas) -> channels × height × width values 0–255. Grayscale uses
 * the luminance; transparent pixels count as black (a drawing on an empty canvas).
 */
export function toChannels(rgba: ArrayLike<number>, format: ImageFormat): number[][][] {
  const { channels, height, width } = format;
  const planes: number[][][] = Array.from({ length: channels }, () =>
    Array.from({ length: height }, () => new Array<number>(width).fill(0))
  );
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const i = (y * width + x) * 4;
      const alpha = rgba[i + 3] / 255;
      const [r, g, b] = [rgba[i] * alpha, rgba[i + 1] * alpha, rgba[i + 2] * alpha];
      if (channels === 1) {
        planes[0][y][x] = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
      } else {
        planes[0][y][x] = Math.round(r);
        planes[1][y][x] = Math.round(g);
        planes[2][y][x] = Math.round(b);
      }
    }
  }
  return planes;
}

/** Draw `source` scaled into a canvas of the network's size and read its pixels. */
export function canvasPixels(
  source: CanvasImageSource,
  format: ImageFormat,
  invert = false
): number[][][] | null {
  const canvas = document.createElement('canvas');
  canvas.width = format.width;
  canvas.height = format.height;
  const context = canvas.getContext('2d');
  if (!context) {
    return null;
  }
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = 'high';
  context.drawImage(source, 0, 0, format.width, format.height);
  const data = context.getImageData(0, 0, format.width, format.height).data;
  const planes = toChannels(data, format);
  return invert ? planes.map((plane) => plane.map((row) => row.map((v) => 255 - v))) : planes;
}

/** Read an image file the browser can show. */
export function loadImage(file: Blob): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const image = new Image();
    image.onload = () => {
      URL.revokeObjectURL(url);
      resolve(image);
    };
    image.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error('This file is not an image the browser can read.'));
    };
    image.src = url;
  });
}

/** A short hash of a value, to tell inputs apart. */
export function hashOf(value: unknown): string {
  const text = JSON.stringify(value) ?? '';
  let hash = 5381;
  for (let i = 0; i < text.length; i += 1) {
    hash = (hash * 33 + text.charCodeAt(i)) >>> 0;
  }
  return hash.toString(36);
}
