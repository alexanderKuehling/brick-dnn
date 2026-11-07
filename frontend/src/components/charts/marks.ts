/** SVG paths for bars: rounded at the data end (4px), square at the baseline. */

/** A column growing up from `base` to `top`, rounded at the top. */
export function columnPath(x: number, width: number, top: number, base: number, radius = 4) {
  const height = base - top;
  if (!(height > 0) || !(width > 0)) {
    return '';
  }
  const r = Math.min(radius, width / 2, height);
  return (
    `M${x},${base}V${top + r}Q${x},${top} ${x + r},${top}` +
    `H${x + width - r}Q${x + width},${top} ${x + width},${top + r}V${base}Z`
  );
}

/** A bar growing right from `start` to `end`, rounded at the right end when `rounded`. */
export function barPath(start: number, end: number, y: number, height: number, rounded = true) {
  const width = end - start;
  if (!(width > 0) || !(height > 0)) {
    return '';
  }
  const r = rounded ? Math.min(4, height / 2, width) : 0;
  return (
    `M${start},${y}H${end - r}Q${end},${y} ${end},${y + r}` +
    `V${y + height - r}Q${end},${y + height} ${end - r},${y + height}H${start}Z`
  );
}

/** Label truncated to about `maxChars` characters (the full text goes into a title). */
export function truncate(text: string, maxChars: number): string {
  return text.length > maxChars ? `${text.slice(0, Math.max(1, maxChars - 1))}…` : text;
}

/** Rough width of a label in the 11–12px UI font, for layout without measuring. */
export function textWidth(text: string, size = 11): number {
  return text.length * size * 0.6;
}
