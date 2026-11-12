import { IconAlertTriangle, IconRefresh } from '@tabler/icons-react';
import { Alert, Button, Group, Loader, Stack, Text } from '@mantine/core';
import type { Explanation } from './useExplanation';

/** Loading, error (with a retry) or the content of one explanation, with its notes. */
export function Pending<T>({
  result,
  children,
  what,
}: {
  result: Explanation<T>;
  children: (data: T) => React.ReactNode;
  what: string;
}) {
  if (result.error) {
    return (
      <Alert color="red" variant="light" icon={<IconAlertTriangle size={18} />}>
        <Stack gap={6} align="flex-start">
          <Text size="sm">{result.error}</Text>
          <Button
            size="compact-xs"
            variant="light"
            leftSection={<IconRefresh size={14} />}
            onClick={result.retry}
          >
            Try again
          </Button>
        </Stack>
      </Alert>
    );
  }
  if (result.loading || !result.data) {
    return (
      <Group gap="xs" wrap="nowrap">
        <Loader size="sm" />
        <Text size="sm" c="dimmed">
          Computing {what} on the server …
        </Text>
      </Group>
    );
  }
  return (
    <>
      {children(result.data)}
      <Notes notes={(result.data as { notes?: string[] }).notes} />
    </>
  );
}

function Notes({ notes }: { notes?: string[] }) {
  if (!notes || notes.length === 0) {
    return null;
  }
  return (
    <Stack gap={2} mt="xs">
      {notes.map((note) => (
        <Text key={note} size="xs" c="orange">
          {note}
        </Text>
      ))}
    </Stack>
  );
}
