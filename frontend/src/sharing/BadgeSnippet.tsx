import { IconCheck, IconCopy } from '@tabler/icons-react';
import { Button, CopyButton, Group, Text } from '@mantine/core';
import { useLanguage, useTranslation } from '@/i18n';
import { badgeHtml, badgeMarkdown, badgeUrl } from './badge';
import classes from './Sharing.module.css';

/**
 * The "Open in Brick DNN" badge for a share link: what it looks like, and its snippet to copy
 * as Markdown (READMEs, notebooks) or HTML (course pages).
 */
export function BadgeSnippet({ url }: { url: string }) {
  const { t } = useTranslation();
  const language = useLanguage();
  const snippets = [
    { label: t('brand.badge.copyMarkdown'), value: badgeMarkdown(url, language) },
    { label: t('brand.badge.copyHtml'), value: badgeHtml(url, language) },
  ];
  return (
    <div className={classes.badge}>
      <Text fw={600} size="sm">
        {t('brand.badge.title')}
      </Text>
      <Text size="sm" c="dimmed">
        {t('brand.badge.hint')}
      </Text>
      <Group gap="xs" mt="xs">
        <img
          className={classes.badgeImage}
          src={badgeUrl(language)}
          alt={t('brand.badge.alt')}
          height={28}
        />
        {snippets.map((snippet) => (
          <CopyButton key={snippet.label} value={snippet.value} timeout={2000}>
            {({ copied, copy }) => (
              <Button
                variant="default"
                size="xs"
                onClick={copy}
                leftSection={copied ? <IconCheck size={14} /> : <IconCopy size={14} />}
              >
                {copied ? t('brand.badge.copied') : snippet.label}
              </Button>
            )}
          </CopyButton>
        ))}
      </Group>
    </div>
  );
}
