import { Container, Paper, Text, Title } from '@mantine/core';

export function AuthLayout({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <Container size={440} py={60}>
      <Title ta="center" order={2}>
        {title}
      </Title>
      <Text c="dimmed" size="sm" ta="center" mt={6}>
        {subtitle}
      </Text>
      <Paper withBorder shadow="sm" p="xl" mt="xl" radius="md">
        {children}
      </Paper>
    </Container>
  );
}
