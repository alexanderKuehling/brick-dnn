/**
 * Images for predictions: pixels as the server takes them (input_scale "raw": 0–255,
 * channels × height × width; the server normalises them like the training did), and drawn
 * digits turned into an image like the MNIST digits (centred, with a margin).
 */

/** Grayscale like Pillow's `convert("L")`, which the training uses: (299 R + 587 G + 114 B) / 1000. */
export function grayOf(r: number, g: number, b: number): number {
  return Math.floor((r * 299 + g * 587 + b * 114) / 1000);
}

/** RGBA pixels (canvas ImageData) as a nested array channels × height × width of 0–255. */
export function imageInput(
  rgba: ArrayLike<number>,
  width: number,
  height: number,
  channels: 1 | 3
): number[][][] {
  const planes: number[][][] = Array.from({ length: channels }, () => []);
  for (let y = 0; y < height; y += 1) {
    const rows = planes.map(() => [] as number[]);
    for (let x = 0; x < width; x += 1) {
      const i = (y * width + x) * 4;
      if (channels === 1) {
        rows[0].push(grayOf(rgba[i], rgba[i + 1], rgba[i + 2]));
      } else {
        rows[0].push(rgba[i]);
        rows[1].push(rgba[i + 1]);
        rows[2].push(rgba[i + 2]);
      }
    }
    rows.forEach((row, c) => planes[c].push(row));
  }
  return planes;
}

/** A stroke of the drawing pad: points in 0–1 coordinates. */
export type Stroke = { x: number; y: number }[];

/**
 * The strokes as a `size` × `size` grayscale image (0–255, white on black); `width` is the
 * line width as a share of the size.
 */
export function rasterize(strokes: Stroke[], size: number, width = 0.08): Float32Array {
  const pixels = new Float32Array(size * size);
  const radius = (width * size) / 2;
  const paint = (ax: number, ay: number, bx: number, by: number) => {
    const x0 = Math.max(0, Math.floor(Math.min(ax, bx) - radius - 1));
    const x1 = Math.min(size - 1, Math.ceil(Math.max(ax, bx) + radius + 1));
    const y0 = Math.max(0, Math.floor(Math.min(ay, by) - radius - 1));
    const y1 = Math.min(size - 1, Math.ceil(Math.max(ay, by) + radius + 1));
    const dx = bx - ax;
    const dy = by - ay;
    const length = dx * dx + dy * dy;
    for (let y = y0; y <= y1; y += 1) {
      for (let x = x0; x <= x1; x += 1) {
        const px = x + 0.5;
        const py = y + 0.5;
        const t =
          length > 0 ? Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / length)) : 0;
        const distance = Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
        // Soft edges, one pixel wide.
        const value = Math.max(0, Math.min(1, radius + 0.5 - distance)) * 255;
        const at = y * size + x;
        pixels[at] = Math.max(pixels[at], value);
      }
    }
  };
  for (const stroke of strokes) {
    const points = stroke.map((p) => ({ x: p.x * size, y: p.y * size }));
    if (points.length === 1) {
      paint(points[0].x, points[0].y, points[0].x, points[0].y);
    }
    for (let i = 1; i < points.length; i += 1) {
      paint(points[i - 1].x, points[i - 1].y, points[i].x, points[i].y);
    }
  }
  return pixels;
}

/** Resize a grayscale image by averaging the source area of every target pixel. */
export function resizeArea(
  source: Float32Array,
  sourceWidth: number,
  sourceHeight: number,
  width: number,
  height: number
): Float32Array {
  const target = new Float32Array(width * height);
  const sx = sourceWidth / width;
  const sy = sourceHeight / height;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      let total = 0;
      let weight = 0;
      const left = x * sx;
      const top = y * sy;
      for (let v = Math.floor(top); v < Math.ceil(top + sy); v += 1) {
        const wy = Math.min(v + 1, top + sy) - Math.max(v, top);
        for (let u = Math.floor(left); u < Math.ceil(left + sx); u += 1) {
          const wx = Math.min(u + 1, left + sx) - Math.max(u, left);
          if (wx > 0 && wy > 0 && u < sourceWidth && v < sourceHeight) {
            total += source[v * sourceWidth + u] * wx * wy;
            weight += wx * wy;
          }
        }
      }
      target[y * width + x] = weight > 0 ? total / weight : 0;
    }
  }
  return target;
}

/**
 * A drawn digit like an MNIST digit: the drawing's bounding box scaled into the inner
 * 20/28 of a `width` × `height` image and centred by its centre of mass. null if nothing is
 * drawn. Returns rows of whole pixel values 0–255.
 */
export function fitDrawing(
  pixels: Float32Array,
  size: number,
  width: number,
  height: number
): number[][] | null {
  let minX = size;
  let minY = size;
  let maxX = -1;
  let maxY = -1;
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      if (pixels[y * size + x] > 25) {
        minX = Math.min(minX, x);
        maxX = Math.max(maxX, x);
        minY = Math.min(minY, y);
        maxY = Math.max(maxY, y);
      }
    }
  }
  if (maxX < 0) {
    return null;
  }
  // A square box around the drawing, so that the digit keeps its proportions.
  const side = Math.max(maxX - minX, maxY - minY) + 1;
  const cx = (minX + maxX + 1) / 2;
  const cy = (minY + maxY + 1) / 2;
  const box = new Float32Array(side * side);
  for (let y = 0; y < side; y += 1) {
    for (let x = 0; x < side; x += 1) {
      const u = Math.floor(cx - side / 2 + x);
      const v = Math.floor(cy - side / 2 + y);
      if (u >= 0 && v >= 0 && u < size && v < size) {
        box[y * side + x] = pixels[v * size + u];
      }
    }
  }
  const inner = Math.max(1, Math.round((Math.min(width, height) * 20) / 28));
  const digit = resizeArea(box, side, side, inner, inner);
  // Centre of mass of the digit, placed at the centre of the image.
  let mass = 0;
  let mx = 0;
  let my = 0;
  for (let y = 0; y < inner; y += 1) {
    for (let x = 0; x < inner; x += 1) {
      const value = digit[y * inner + x];
      mass += value;
      mx += (x + 0.5) * value;
      my += (y + 0.5) * value;
    }
  }
  const offsetX = Math.round(width / 2 - (mass > 0 ? mx / mass : inner / 2));
  const offsetY = Math.round(height / 2 - (mass > 0 ? my / mass : inner / 2));
  const image = Array.from({ length: height }, () => new Array<number>(width).fill(0));
  for (let y = 0; y < inner; y += 1) {
    for (let x = 0; x < inner; x += 1) {
      const tx = x + offsetX;
      const ty = y + offsetY;
      if (tx >= 0 && ty >= 0 && tx < width && ty < height) {
        image[ty][tx] = Math.round(Math.min(255, digit[y * inner + x]));
      }
    }
  }
  return image;
}
