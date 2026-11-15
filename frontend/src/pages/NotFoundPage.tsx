import { Link } from 'react-router-dom';
import { Button, Container, Text, Title } from '@mantine/core';
import { useTranslation } from '@/i18n';

export function NotFoundPage() {
  const { t } = useTranslation();
  return (
    <Container size="sm" py={80} ta="center">
      <Title order={2}>{t('auth.notFound.title')}</Title>
      <Text c="dimmed" mt="sm">
        {t('auth.notFound.text')}
      </Text>
      <Button component={Link} to="/" mt="xl">
        {t('auth.notFound.back')}
      </Button>
    </Container>
  );
}
