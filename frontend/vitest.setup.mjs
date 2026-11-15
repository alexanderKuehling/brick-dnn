import '@testing-library/jest-dom/vitest';
import { afterEach, vi } from 'vitest';
import { i18n, LANGUAGE_STORAGE_KEY } from './src/i18n';

const { getComputedStyle } = window;
window.getComputedStyle = (elt) => getComputedStyle(elt);
window.HTMLElement.prototype.scrollIntoView = () => {};

Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: vi.fn().mockImplementation((query) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
});

class ResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

window.ResizeObserver = ResizeObserver;

// Textareas that grow with their text listen for loaded fonts.
if (!document.fonts) {
  Object.defineProperty(document, 'fonts', {
    value: { addEventListener() {}, removeEventListener() {} },
  });
}

// Tests run in English (jsdom's browser language) unless they switch; each starts over.
afterEach(async () => {
  try {
    localStorage.removeItem(LANGUAGE_STORAGE_KEY);
  } catch {
    // No storage in this test.
  }
  if (i18n.language !== 'en') {
    await i18n.changeLanguage('en');
  }
});
