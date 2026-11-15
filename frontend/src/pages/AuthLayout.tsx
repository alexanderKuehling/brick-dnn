import { Container, Group, Paper, Text, Title } from '@mantine/core';
import { LanguageSwitch } from '@/i18n/LanguageSwitch';
import { LegalLinks } from '@/legal/LegalLinks';

export function AuthLayout({
  title,
  subtitle,
  children,
}: {
  title: React.ReactNode;
  subtitle?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <Container size={440} py={60}>
      <Title ta="center" order={2}>
        {title}
      </Title>
      {subtitle && (
        <Text c="dimmed" size="sm" ta="center" mt={6}>
          {subtitle}
        </Text>
      )}
      <Paper withBorder shadow="sm" p="xl" mt="xl" radius="md">
        {children}
      </Paper>
      {/* The language can be chosen before logging in (the header hides it on phones). */}
      <Group justify="center" mt="lg">
        <LanguageSwitch />
      </Group>
      <LegalLinks mt="sm" />
    </Container>
  );
}
