import { Link } from 'react-router-dom';
import { Button, Container, Text, Title } from '@mantine/core';

export function NotFoundPage() {
  return (
    <Container size="sm" py={80} ta="center">
      <Title order={2}>Page not found</Title>
      <Text c="dimmed" mt="sm">
        This page does not exist.
      </Text>
      <Button component={Link} to="/" mt="xl">
        Back to the builder
      </Button>
    </Container>
  );
}
