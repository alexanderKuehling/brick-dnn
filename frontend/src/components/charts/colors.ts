/**
 * Chart colours by job: categorical (identity), sequential (magnitude), diverging (polarity).
 * SVG marks use the CSS tokens of `global.css` (they follow light/dark by themselves); canvas
 * drawing and continuous scales need real colours, so the values are mirrored here per mode
 * (a test keeps both in sync).
 */
import { useComputedColorScheme } from '@mantine/core';

export type ChartScheme = 'light' | 'dark';

/** The categorical slots in their fixed order, stepped per mode. */
export const CATEGORICAL: Record<ChartScheme, readonly string[]> = {
  light: ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948'],
  dark: ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767'],
};

/** Most categories with their own colour; more fold into "Other". */
export const MAX_SERIES = CATEGORICAL.light.length;
/** Scatter plots: any two colours can touch, and only three slots stay apart for everyone. */
export const MAX_SCATTER_SERIES = 3;

/** De-emphasis gray ("Other", "Before"). Same in both modes. */
export const OTHER_HEX = '#898781';
export const OTHER_COLOR = 'var(--chart-other)';

/** Neutral cell colour (present values in the missing matrix). */
export const TRACK: Record<ChartScheme, string> = { light: '#e9ecef', dark: '#3b3b3b' };

/** Sequential ends: `marks` keep ≥ 2:1 on the surface, `fill` may fade into it. */
export const SEQUENTIAL: Record<ChartScheme, { low: string; fillLow: string; high: string }> = {
  light: { low: '#86b6ef', fillLow: '#cde2fb', high: '#0d366b' },
  dark: { low: '#1c5cab', fillLow: '#0d366b', high: '#cde2fb' },
};

export const DIVERGING: Record<
  ChartScheme,
  { negative: string; midpoint: string; positive: string }
> = {
  light: { negative: '#8e2526', midpoint: '#f0efec', positive: '#184f95' },
  dark: { negative: '#de6f69', midpoint: '#383835', positive: '#5598e7' },
};

/** CSS colour of categorical slot `index` (0-based) – an SVG fill or stroke. */
export function seriesColor(index: number): string {
  return index >= 0 && index < MAX_SERIES ? `var(--chart-series-${index + 1})` : OTHER_COLOR;
}

/** Hex colour of categorical slot `index` for canvas drawing. */
export function seriesHex(index: number, scheme: ChartScheme): string {
  return CATEGORICAL[scheme][index] ?? OTHER_HEX;
}

/** The colour scheme the charts are drawn in. */
export function useChartScheme(): ChartScheme {
  return useComputedColorScheme('light', { getInitialValueInEffect: false });
}

// ---- OKLab interpolation ---------------------------------------------------------------

type Lab = [number, number, number];

const toLinear = (c: number) => {
  const v = c / 255;
  return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
};
const toByte = (c: number) => {
  const v = c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055;
  return Math.round(Math.min(1, Math.max(0, v)) * 255);
};

function parseHex(hex: string): [number, number, number] {
  const value = hex.replace('#', '');
  const full =
    value.length === 3
      ? value
          .split('')
          .map((c) => c + c)
          .join('')
      : value;
  const n = parseInt(full, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

export function hexToOklab(hex: string): Lab {
  const [r, g, b] = parseHex(hex).map(toLinear);
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  return [
    0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
    1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s,
  ];
}

export function oklabToHex([L, a, b]: Lab): string {
  const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3;
  const m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3;
  const s = (L - 0.0894841775 * a - 1.291485548 * b) ** 3;
  const rgb = [
    4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s,
  ];
  return `#${rgb.map((c) => toByte(c).toString(16).padStart(2, '0')).join('')}`;
}

/** Colour between `from` (t = 0) and `to` (t = 1), interpolated in OKLab. */
export function mixColors(from: string, to: string, t: number): string {
  const k = Math.min(1, Math.max(0, Number.isFinite(t) ? t : 0));
  const a = hexToOklab(from);
  const b = hexToOklab(to);
  return oklabToHex([a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k, a[2] + (b[2] - a[2]) * k]);
}

/** Sequential colour for t in [0, 1]; `fill` lets the low end fade into the surface. */
export function sequentialColor(t: number, scheme: ChartScheme, fill = false): string {
  const ramp = SEQUENTIAL[scheme];
  return mixColors(fill ? ramp.fillLow : ramp.low, ramp.high, t);
}

/** Diverging colour for a value in [-1, 1]: negative → red, 0 → gray, positive → blue. */
export function divergingColor(value: number, scheme: ChartScheme): string {
  const ramp = DIVERGING[scheme];
  const v = Math.min(1, Math.max(-1, Number.isFinite(value) ? value : 0));
  return mixColors(ramp.midpoint, v < 0 ? ramp.negative : ramp.positive, Math.abs(v));
}

/** WCAG relative luminance of a hex colour. */
export function luminance(hex: string): number {
  const [r, g, b] = parseHex(hex).map(toLinear);
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

/** WCAG contrast ratio of two hex colours (1 … 21). */
export function contrastRatio(a: string, b: string): number {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
}

/** Label colour inside a filled mark: white or near-black, whichever reads better. */
export function textOn(fill: string): string {
  return contrastRatio(fill, '#ffffff') >= contrastRatio(fill, '#0b0b0b') ? '#ffffff' : '#0b0b0b';
}
