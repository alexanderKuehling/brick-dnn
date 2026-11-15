import { IconMoon, IconSun } from '@tabler/icons-react';
import { ActionIcon, Tooltip, useComputedColorScheme, useMantineColorScheme } from '@mantine/core';
import { useTranslation } from '@/i18n';

export function ColorSchemeToggle() {
  const { t } = useTranslation();
  const { setColorScheme } = useMantineColorScheme();
  const computed = useComputedColorScheme('light', { getInitialValueInEffect: true });
  const next = computed === 'dark' ? 'light' : 'dark';

  return (
    <Tooltip label={t(`common.colorScheme.${next}`)}>
      <ActionIcon
        variant="default"
        size="lg"
        aria-label={t(`common.colorScheme.${next}`)}
        onClick={() => setColorScheme(next)}
      >
        {computed === 'dark' ? <IconSun size={18} /> : <IconMoon size={18} />}
      </ActionIcon>
    </Tooltip>
  );
}
