import {
  Anchor,
  createTheme,
  Drawer,
  Modal,
  Notification,
  type CSSVariablesResolver,
  type MantineColorsTuple,
} from '@mantine/core';

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

/**
 * Colours Mantine sets that are too faint for WCAG 2.2 AA (docs/accessibility.md): dimmed
 * text has at least 4.5:1 on the page, on gray-0 panels and on light badges (Mantine's
 * gray-6 / dark-2 have 3.3:1 and 4.0:1), and links in the dark theme 4.5:1 on dark-7 and
 * dark-6 (brick-4 has 3.0:1).
 */
export const DIMMED_LIGHT = '#555c63';
export const DIMMED_DARK = '#a6a6a6';
export const ANCHOR_DARK = '#9a95ec';

/**
 * Text colours of the light theme with at least 4.5:1 on white and on the colour's own light
 * background (shades 1 and 2: light badges, alerts and their hover). Mantine draws coloured
 * text (`c="red"`), outline buttons and light variants in shade 6 or 9, which for most colours
 * is below that (green-6 on white 2.4:1, yellow-9 on yellow-1 2.7:1).
 */
export const LIGHT_INK: Record<string, string> = {
  red: '#ad2424',
  pink: '#a61e4d',
  grape: '#862e9c',
  violet: '#5939b8',
  indigo: '#3249b7',
  blue: '#165a9a',
  cyan: '#0a697a',
  teal: '#077050',
  green: '#237133',
  lime: '#48730a',
  yellow: '#a15300',
  orange: '#a9380c',
};

export const cssVariablesResolver: CSSVariablesResolver = () => ({
  variables: {},
  light: {
    '--mantine-color-dimmed': DIMMED_LIGHT,
    ...Object.fromEntries(
      Object.entries(LIGHT_INK).flatMap(([color, ink]) => [
        [`--mantine-color-${color}-text`, ink],
        [`--mantine-color-${color}-light-color`, ink],
        [`--mantine-color-${color}-outline`, ink],
      ])
    ),
    // Red buttons (deleting) with white text: red-8 has 4.5:1, red-6 3.3:1.
    '--mantine-color-red-filled': 'var(--mantine-color-red-8)',
    '--mantine-color-red-filled-hover': 'var(--mantine-color-red-9)',
  },
  dark: { '--mantine-color-dimmed': DIMMED_DARK, '--mantine-color-anchor': ANCHOR_DARK },
});

export const theme = createTheme({
  primaryColor: 'brick',
  // Transitions of Mantine's components (modals, menus, drawers) are left out for users who
  // asked their system for less motion; global.css does the same for the app's own CSS.
  respectReducedMotion: true,
  // Filled buttons and badges get black or white text, whichever has more contrast (white on
  // yellow or green filled badges had less than 3:1).
  autoContrast: true,
  colors: { brick },
  defaultRadius: 'md',
  fontFamily:
    'Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
  fontFamilyMonospace:
    'ui-monospace, "JetBrains Mono", "SFMono-Regular", Menlo, Consolas, monospace',
  // index.html uses viewport-fit=cover, so a screen edge can be under the iPhone notch or home
  // indicator. Drawers and full-screen modals keep their content clear of them.
  components: {
    // Links in text are underlined, not only coloured (WCAG 1.4.1, axe: link-in-text-block).
    Anchor: Anchor.extend({ defaultProps: { underline: 'always' } }),
    // Notifications are status messages: read out when the screen reader is idle, without
    // interrupting (WCAG 4.1.3). Mantine's default is role="alert".
    Notification: Notification.extend({ defaultProps: { role: 'status' } }),
    Drawer: Drawer.extend({
      styles: (_theme, { position = 'left' }) => ({
        content: {
          paddingLeft: position === 'right' ? undefined : 'env(safe-area-inset-left)',
          paddingRight: position === 'left' ? undefined : 'env(safe-area-inset-right)',
          paddingBottom: 'env(safe-area-inset-bottom)',
        },
      }),
    }),
    Modal: Modal.extend({
      styles: (_theme, { fullScreen }) => ({
        content: fullScreen
          ? {
              padding:
                'env(safe-area-inset-top) env(safe-area-inset-right) env(safe-area-inset-bottom) env(safe-area-inset-left)',
            }
          : {},
      }),
    }),
  },
});

/** Colour of each layer category on the canvas and in the palette. */
export const CATEGORY_COLORS: Record<string, string> = {
  Input: 'teal',
  Basic: 'brick',
  Convolution: 'violet',
  Normalization: 'cyan',
  Sequence: 'green',
  Merge: 'orange',
  Kernels: 'pink',
  'Classical models': 'grape',
  Ensembles: 'indigo',
};

export function categoryColor(category: string | undefined): string {
  return (category && CATEGORY_COLORS[category]) || 'gray';
}
