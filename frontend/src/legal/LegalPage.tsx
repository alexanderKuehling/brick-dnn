import { useEffect, useState } from 'react';
import { IconAlertCircle } from '@tabler/icons-react';
import { Alert, Center, Container, Loader, Paper, Text, Title } from '@mantine/core';
import { api, ApiError, type LegalPageName, type LegalPage as Page } from '@/api';
import { formatDate } from '@/training/format';
import { LegalLinks } from './LegalLinks';
import { Markdown, parseMarkdown } from './markdown';
import { LEGAL_LABELS } from './useLegalInfo';

type State =
  | { status: 'loading' }
  | { status: 'ready'; page: Page }
  | { status: 'missing' }
  | { status: 'error'; message: string };

/** /imprint, /privacy and /terms: Markdown files the operator put on the server. */
export function LegalPage({ page }: { page: LegalPageName }) {
  const [state, setState] = useState<State>({ status: 'loading' });

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
  }, [page]);

  if (state.status === 'loading') {
    return (
      <Center h={300}>
        <Loader />
      </Center>
    );
  }

  const content = state.status === 'ready' ? state.page.content : '';
  // The file usually starts with its own title; otherwise the page gets one.
  const first = parseMarkdown(content)[0];
  const titled = first?.type === 'heading' && first.level === 1;

  return (
    <Container size="md" py="xl">
      {!titled && (
        <Title order={1} mb="md">
          {LEGAL_LABELS[page]}
        </Title>
      )}
      {state.status === 'missing' && (
        <Text c="dimmed">This page has not been set up on this server.</Text>
      )}
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
            Last changed {formatDate(state.page.updated_at)}
          </Text>
        </>
      )}
      <LegalLinks mt="xl" />
    </Container>
  );
}
