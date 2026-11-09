import { IconShieldLock } from '@tabler/icons-react';
import { Group, Text } from '@mantine/core';
import { useLegalInfo } from '@/legal/useLegalInfo';

/** Shown where files are uploaded: what not to upload, and how long the server keeps it. */
export function UploadNotice() {
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
        Please don’t upload personal or confidential data. Uploads are stored on this server until
        you delete them{days > 0 ? ` or nobody has used them for ${days} days` : ''}.
      </Text>
    </Group>
  );
}
