/**
 * Mantine in the app's language. Mantine has no locale of its own (and the app uses no
 * @mantine/dates); what depends on the language is set here:
 *
 * - number fields write decimals with a comma in German (`NumberInput` returns numbers, so
 *   what is sent to the API keeps its point);
 * - the buttons of confirmation dialogs (`modals.openConfirmModal`).
 */
import { useMemo } from 'react';
import {
  MantineProvider,
  mergeThemeOverrides,
  NumberInput,
  type MantineThemeOverride,
} from '@mantine/core';
import { ModalsProvider } from '@mantine/modals';
import { cssVariablesResolver, theme } from '@/theme';
import { useLanguage, useTranslation, type Language } from './index';

export function numberSeparators(language: Language): {
  decimalSeparator: string;
  thousandSeparator: string;
} {
  return language === 'de'
    ? { decimalSeparator: ',', thousandSeparator: '.' }
    : { decimalSeparator: '.', thousandSeparator: ',' };
}

export function localizedTheme(language: Language): MantineThemeOverride {
  return mergeThemeOverrides(theme, {
    components: {
      NumberInput: NumberInput.extend({
        defaultProps: { decimalSeparator: numberSeparators(language).decimalSeparator },
      }),
    },
  });
}

export function LocalizedMantineProvider({
  children,
  ...props
}: Omit<React.ComponentProps<typeof MantineProvider>, 'theme'>) {
  const language = useLanguage();
  const localized = useMemo(() => localizedTheme(language), [language]);
  return (
    <MantineProvider theme={localized} cssVariablesResolver={cssVariablesResolver} {...props}>
      {children}
    </MantineProvider>
  );
}

export function LocalizedModalsProvider({ children }: { children: React.ReactNode }) {
  const { t } = useTranslation();
  return (
    <ModalsProvider labels={{ confirm: t('common.confirm'), cancel: t('common.cancel') }}>
      {children}
    </ModalsProvider>
  );
}
