import { IconShieldLock } from '@tabler/icons-react';
import { Group, Text } from '@mantine/core';
import { i18n, useTranslation } from '@/i18n';
import { useLegalInfo } from '@/legal/useLegalInfo';

/** Shown where files are uploaded: what not to upload, and how long the server keeps it. */
export function UploadNotice() {
  useTranslation();
  const days = useLegalInfo()?.retention.unused_uploads_days ?? 0;
  return (
    <Group gap={6} wrap="nowrap" align="flex-start">
      <IconShieldLock
        size={16}
        color="var(--mantine-color-dimmed)"
        style={{ flexShrink: 0, marginTop: 1 }}
        aria-hidden
      />
      <Text size="xs" c="dimmed">
        {i18n.t('ui.components.UploadNotice.pleaseDonTUploadPersonal', {
          value: days > 0 ? i18n.t('ui.components.UploadNotice.orNobodyHasUsedThem', { days }) : '',
        })}
      </Text>
    </Group>
  );
}
