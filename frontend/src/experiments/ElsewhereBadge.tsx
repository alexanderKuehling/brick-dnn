import { IconWorldUpload } from '@tabler/icons-react';
import { Badge, Tooltip } from '@mantine/core';
import type { ExternalRunInfo } from '@/api';
import { useTranslation } from '@/i18n';
import { formatDate } from '@/training/format';

/** "Ran elsewhere": a run that a script on another machine logged here. */
export function ElsewhereBadge({ external }: { external: ExternalRunInfo }) {
  const { t } = useTranslation();
  return (
    <Tooltip
      label={t('experiments.elsewhere.tooltip', {
        count: external.points,
        date: formatDate(external.reported_at),
      })}
      multiline
      maw={280}
    >
      <Badge
        size="xs"
        variant="outline"
        color="teal"
        tt="none"
        leftSection={<IconWorldUpload size={10} aria-hidden />}
        style={{ flexShrink: 0 }}
      >
        {t('experiments.elsewhere.badge')}
      </Badge>
    </Tooltip>
  );
}
