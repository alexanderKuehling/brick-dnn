import { useEffect, useState } from 'react';
import { IconAlertCircle } from '@tabler/icons-react';
import { Alert, Anchor, Center, Container, Loader, Paper, Text, Title } from '@mantine/core';
import { api, ApiError, type LegalPageName, type LegalPage as Page } from '@/api';
import { useLanguage, useTranslation } from '@/i18n';
import { formatDateTime } from '@/i18n/format';
import { languageSection } from './languages';
import { LegalLinks } from './LegalLinks';
import { Markdown, parseMarkdown } from './markdown';

type State =
  | { status: 'loading' }
  | { status: 'ready'; page: Page }
  | { status: 'missing' }
  | { status: 'error'; message: string };

/** /imprint, /privacy, /terms and /refund: Markdown files the operator put on the server. */
export function LegalPage({ page }: { page: LegalPageName }) {
  const { t } = useTranslation();
  const language = useLanguage();
  const [state, setState] = useState<State>({ status: 'loading' });
  const [allLanguages, setAllLanguages] = useState(false);

  useEffect(() => {
    let current = true;
    setState({ status: 'loading' });
    api.legal.page(page).then(
      (loaded) => current && setState({ status: 'ready', page: loaded }),
      (error) => {
        if (!current) {
          return;
        }
        if (error instanceof ApiError && error.status === 404) {
          setState({ status: 'missing' });
        } else {
          setState({ status: 'error', message: (error as Error).message });
        }
      }
    );
    return () => {
      current = false;
    };
    // The server sends the file of the app's language if it has one (`privacy.de.md`).
  }, [page, language]);

  if (state.status === 'loading') {
    return (
      <Center h={300}>
        <Loader />
      </Center>
    );
  }

  const full = state.status === 'ready' ? state.page.content : '';
  // A page in both languages ("## English", "## Deutsch") shows the section of the app's.
  const section = languageSection(full, language);
  const content = section !== null && !allLanguages ? section : full;
  // The file usually starts with its own title; otherwise the page gets one.
  const first = parseMarkdown(content)[0];
  const titled = first?.type === 'heading' && first.level === 1;

  return (
    <Container size="md" py="xl">
      {!titled && (
        <Title order={1} mb="md">
          {t(`legal.labels.${page}`)}
        </Title>
      )}
      {state.status === 'missing' && <Text c="dimmed">{t('legal.missing')}</Text>}
      {state.status === 'error' && (
        <Alert color="red" icon={<IconAlertCircle size={18} />}>
          {state.message}
        </Alert>
      )}
      {state.status === 'ready' && (
        <>
          <Paper component="article" p={{ base: 'md', sm: 'xl' }} withBorder radius="md">
            <Markdown text={content} />
          </Paper>
          <Text size="xs" c="dimmed" mt="sm">
            {t('legal.lastChanged', { date: formatDateTime(state.page.updated_at) })}
            {section !== null && !allLanguages && (
              <>
                {' · '}
                <Anchor
                  component="button"
                  type="button"
                  inherit
                  onClick={() => setAllLanguages(true)}
                >
                  {t('legal.allLanguages')}
                </Anchor>
              </>
            )}
          </Text>
        </>
      )}
      <LegalLinks mt="xl" />
    </Container>
  );
}
