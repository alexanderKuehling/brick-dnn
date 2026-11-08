import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import {
  CATEGORICAL,
  contrastRatio,
  DIVERGING,
  divergingColor,
  hexToOklab,
  mixColors,
  oklabToHex,
  OTHER_HEX,
  SEQUENTIAL,
  sequentialColor,
  seriesColor,
  seriesHex,
  textOn,
  TRACK,
} from './colors';

const css = (file: string) => readFileSync(resolve(__dirname, file), 'utf8');

/** The `light-dark(a, b)` values of a custom property in a stylesheet. */
function token(source: string, name: string): [string, string] | string | null {
  const match = source.match(new RegExp(`${name}:\\s*([^;]+);`));
  if (!match) {
    return null;
  }
  const value = match[1].trim();
  const pair = value.match(/^light-dark\(\s*(#[0-9a-f]+)\s*,\s*(#[0-9a-f]+)\s*\)$/i);
  return pair ? [pair[1].toLowerCase(), pair[2].toLowerCase()] : value.toLowerCase();
}

describe('chart colours', () => {
  it('keeps the CSS tokens and the canvas colours in sync', () => {
    const global = css('../../global.css');
    CATEGORICAL.light.forEach((light, i) => {
      expect(token(global, `--chart-series-${i + 1}`)).toEqual([light, CATEGORICAL.dark[i]]);
    });
    expect(token(global, '--chart-other')).toBe(OTHER_HEX);
    expect(token(global, '--chart-track')).toEqual([TRACK.light, TRACK.dark]);
    expect(token(global, '--chart-sequential-low')).toEqual([
      SEQUENTIAL.light.low,
      SEQUENTIAL.dark.low,
    ]);
    expect(token(global, '--chart-sequential-high')).toEqual([
      SEQUENTIAL.light.high,
      SEQUENTIAL.dark.high,
    ]);
    expect(token(global, '--chart-sequential-fill-low')).toEqual([
      SEQUENTIAL.light.fillLow,
      SEQUENTIAL.dark.fillLow,
    ]);
    for (const part of ['negative', 'midpoint', 'positive'] as const) {
      expect(token(global, `--chart-diverging-${part}`)).toEqual([
        DIVERGING.light[part],
        DIVERGING.dark[part],
      ]);
    }
  });

  it('uses the LineChart colours for series 1 and 2', () => {
    const line = css('../LineChart.module.css');
    expect(token(line, '--chart-series-1')).toEqual([CATEGORICAL.light[0], CATEGORICAL.dark[0]]);
    expect(token(line, '--chart-series-2')).toEqual([CATEGORICAL.light[1], CATEGORICAL.dark[1]]);
  });

  it('assigns slots in fixed order and folds the rest into gray', () => {
    expect(seriesColor(0)).toBe('var(--chart-series-1)');
    expect(seriesColor(7)).toBe('var(--chart-series-8)');
    expect(seriesColor(8)).toBe('var(--chart-other)');
    expect(seriesColor(-1)).toBe('var(--chart-other)');
    expect(seriesHex(2, 'dark')).toBe('#199e70');
    expect(seriesHex(9, 'light')).toBe(OTHER_HEX);
  });

  it('converts to OKLab and back', () => {
    for (const hex of [...CATEGORICAL.light, '#000000', '#ffffff', '#f0efec']) {
      expect(oklabToHex(hexToOklab(hex))).toBe(hex);
    }
    expect(hexToOklab('#ffffff')[0]).toBeCloseTo(1, 4);
  });

  it('interpolates scales between their ends', () => {
    expect(mixColors('#000000', '#ffffff', 0)).toBe('#000000');
    expect(mixColors('#000000', '#ffffff', 1)).toBe('#ffffff');
    expect(mixColors('#000000', '#ffffff', 2)).toBe('#ffffff');
    expect(sequentialColor(0, 'light')).toBe(SEQUENTIAL.light.low);
    expect(sequentialColor(1, 'dark')).toBe(SEQUENTIAL.dark.high);
    expect(sequentialColor(0, 'light', true)).toBe(SEQUENTIAL.light.fillLow);
    expect(divergingColor(0, 'light')).toBe(DIVERGING.light.midpoint);
    expect(divergingColor(1, 'light')).toBe(DIVERGING.light.positive);
    expect(divergingColor(-1, 'dark')).toBe(DIVERGING.dark.negative);
    expect(divergingColor(-5, 'dark')).toBe(DIVERGING.dark.negative);
  });

  it('gets lighter towards the midpoint on both arms (light mode)', () => {
    const lightness = (v: number) => hexToOklab(divergingColor(v, 'light'))[0];
    expect(lightness(-1)).toBeLessThan(lightness(-0.5));
    expect(lightness(-0.5)).toBeLessThan(lightness(0));
    expect(lightness(1)).toBeLessThan(lightness(0.5));
    expect(lightness(0.5)).toBeLessThan(lightness(0));
    // Equal arms: both poles have the same lightness.
    expect(lightness(-1)).toBeCloseTo(lightness(1), 2);
  });

  it('keeps marks visible on both surfaces', () => {
    expect(contrastRatio(SEQUENTIAL.light.low, '#ffffff')).toBeGreaterThanOrEqual(2);
    expect(contrastRatio(SEQUENTIAL.dark.low, '#242424')).toBeGreaterThanOrEqual(2);
    for (const hex of CATEGORICAL.dark) {
      expect(contrastRatio(hex, '#242424')).toBeGreaterThanOrEqual(3);
    }
  });

  it('picks white or near-black text inside a fill', () => {
    expect(textOn(DIVERGING.light.positive)).toBe('#ffffff');
    expect(textOn(DIVERGING.light.midpoint)).toBe('#0b0b0b');
    expect(textOn('#eda100')).toBe('#0b0b0b');
  });
});
