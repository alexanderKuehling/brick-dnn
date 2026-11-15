import 'i18next';

import type { en } from './locales/en';

// Typed keys: `t('builder.toolbar.save')` must name a text of the English resources.
declare module 'i18next' {
  interface CustomTypeOptions {
    defaultNS: 'translation';
    resources: { translation: typeof en };
  }
}
