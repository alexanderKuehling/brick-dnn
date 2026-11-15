import { Group, Text } from '@mantine/core';
import { i18n, useTranslation } from '@/i18n';

export function Logo({ size = 30 }: { size?: number }) {
  useTranslation();
  return (
    <Group gap={8} wrap="nowrap">
      <img src={`${import.meta.env.BASE_URL}favicon.svg`} width={size} height={size} alt="" />
      <Text fw={800} size="lg" style={{ letterSpacing: '-0.02em', whiteSpace: 'nowrap' }}>
        {i18n.t('ui.components.Logo.brickDnn')}
      </Text>
    </Group>
  );
}
