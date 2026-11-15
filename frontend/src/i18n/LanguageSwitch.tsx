import { SegmentedControl, Tooltip, type MantineSize } from '@mantine/core';
import { LANGUAGE_NAMES, LANGUAGES, setLanguage, useLanguage, useTranslation } from './index';

/** EN | DE: switches the app's language (kept in the browser and on the account). */
export function LanguageSwitch({ size = 'xs' }: { size?: MantineSize }) {
  const { t } = useTranslation();
  const language = useLanguage();
  return (
    <Tooltip label={t('common.language.tooltip')}>
      <SegmentedControl
        size={size}
        value={language}
        onChange={(value) => {
          if (value === 'en' || value === 'de') {
            void setLanguage(value);
          }
        }}
        aria-label={t('common.language.label')}
        data={LANGUAGES.map((code) => ({
          value: code,
          label: (
            <span lang={code} title={LANGUAGE_NAMES[code]}>
              {code.toUpperCase()}
            </span>
          ),
        }))}
      />
    </Tooltip>
  );
}
