import { useTranslation } from '@/i18n';

/** The id of the main region (AppShell.Main), the target of the skip link. */
export const MAIN_ID = 'main';

/**
 * "Skip to content": the first stop of the Tab key, shown only while it has the focus, so
 * keyboard users get past the header in one step (WCAG 2.4.1). It moves the focus without
 * changing the address (the hash).
 */
export function SkipLink() {
  const { t } = useTranslation();
  return (
    <a
      className="skip-link"
      href={`#${MAIN_ID}`}
      onClick={(event) => {
        const main = document.getElementById(MAIN_ID);
        if (main) {
          event.preventDefault();
          main.focus();
        }
      }}
    >
      {t('a11y.skipToContent')}
    </a>
  );
}
