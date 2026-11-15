import { render as testingLibraryRender } from '@testing-library/react';
import { LocalizedMantineProvider } from '../src/i18n/mantine';

/** Mantine with the theme of the app in the current language (English unless a test switches). */
export function render(ui: React.ReactNode) {
  return testingLibraryRender(ui, {
    wrapper: ({ children }: { children: React.ReactNode }) => (
      <LocalizedMantineProvider env="test">{children}</LocalizedMantineProvider>
    ),
  });
}
