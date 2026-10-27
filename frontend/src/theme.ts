import { createTheme, type MantineColorsTuple } from '@mantine/core';

/** Navy/indigo of the Brick DNN logo. */
const brick: MantineColorsTuple = [
  '#eeedff',
  '#d9d7fb',
  '#b0acf1',
  '#857fe8',
  '#615ae0',
  '#4a42dc',
  '#3d36db',
  '#2f29c2',
  '#2724ae',
  '#1c1d9a',
];

export const theme = createTheme({
  primaryColor: 'brick',
  colors: { brick },
  defaultRadius: 'md',
  fontFamily:
    'Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
  fontFamilyMonospace:
    'ui-monospace, "JetBrains Mono", "SFMono-Regular", Menlo, Consolas, monospace',
});

/** Colour of each layer category on the canvas and in the palette. */
export const CATEGORY_COLORS: Record<string, string> = {
  Input: 'teal',
  Basic: 'brick',
  Convolution: 'violet',
  Normalization: 'cyan',
  Sequence: 'green',
  Merge: 'orange',
};

export function categoryColor(category: string | undefined): string {
  return (category && CATEGORY_COLORS[category]) || 'gray';
}
