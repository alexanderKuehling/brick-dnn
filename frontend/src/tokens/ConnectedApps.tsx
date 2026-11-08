import { useEffect, useState } from 'react';
import { IconPlugConnectedX } from '@tabler/icons-react';
import { Alert, Badge, Button, Center, Group, Loader, Text, Title } from '@mantine/core';
import { modals } from '@mantine/modals';
import { notifications } from '@mantine/notifications';
import { api, type OAuthConnection } from '@/api';
import { urlHost } from '@/oauth/url';
import { formatDate } from '@/training/format';
import classes from './Tokens.module.css';

function ConnectionItem({
  connection,
  onDisconnect,
}: {
  connection: OAuthConnection;
  onDisconnect: () => void;
}) {
  const host = urlHost(connection.client_uri);
  return (
    <li className={classes.item} aria-label={connection.client_name}>
      <Group justify="space-between" align="flex-start" wrap="nowrap" gap="sm">
        <div className={classes.itemTitle}>
          <Group gap={8} wrap="wrap">
            <Text fw={600} className={classes.name}>
              {connection.client_name}
            </Text>
            <Badge variant="light" color={connection.read_only ? 'gray' : 'brick'}>
              {connection.read_only ? 'Read only' : 'Full access'}
            </Badge>
          </Group>
          {host && (
            <Text size="sm" c="dimmed" mt={2} className={classes.name}>
              {host}
            </Text>
          )}
        </div>
        <Button
          variant="light"
          color="red"
          size="xs"
          leftSection={<IconPlugConnectedX size={14} />}
          className={classes.itemAction}
          onClick={onDisconnect}
          aria-label={`Disconnect “${connection.client_name}”`}
        >
          Disconnect
        </Button>
      </Group>
      <dl className={classes.meta}>
        <div>
          <dt>Connected</dt>
          <dd>{formatDate(connection.created_at)}</dd>
        </div>
        <div>
          <dt>Last used</dt>
          <dd>{connection.last_used_at ? formatDate(connection.last_used_at) : 'Never'}</dd>
        </div>
      </dl>
    </li>
  );
}

/** Apps the user allowed to use their account by signing in (OAuth), e.g. the Claude apps. */
export function ConnectedApps() {
  const [connections, setConnections] = useState<OAuthConnection[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.connections
      .list()
      .then(setConnections)
      .catch((err: Error) => setError(err.message));
  }, []);

  const disconnect = (connection: OAuthConnection) =>
    modals.openConfirmModal({
      title: `Disconnect “${connection.client_name}”?`,
      children: (
        <Text size="sm">
          {connection.client_name} loses access immediately. To use it again, connect it again in
          the app.
        </Text>
      ),
      labels: { confirm: 'Disconnect', cancel: 'Cancel' },
      confirmProps: { color: 'red' },
      onConfirm: async () => {
        try {
          await api.connections.revoke(connection.id);
          setConnections((list) => list?.filter((c) => c.id !== connection.id) ?? null);
        } catch (err) {
          notifications.show({
            color: 'red',
            title: 'Could not disconnect the app',
            message: (err as Error).message,
          });
        }
      },
    });

  return (
    <section className={classes.section} aria-labelledby="apps-title">
      <Title order={3} id="apps-title" mb="md">
        Connected apps
      </Title>
      {error && <Alert color="red">{error}</Alert>}
      {!connections && !error && (
        <Center h={80}>
          <Loader size="sm" />
        </Center>
      )}
      {connections?.length === 0 && (
        <Text c="dimmed" size="sm">
          No apps connected yet. Apps like Claude ask for access when you add Brick DNN as a
          connector.
        </Text>
      )}
      {connections && connections.length > 0 && (
        <ul className={classes.list} aria-labelledby="apps-title">
          {connections.map((connection) => (
            <ConnectionItem
              key={connection.id}
              connection={connection}
              onDisconnect={() => disconnect(connection)}
            />
          ))}
        </ul>
      )}
    </section>
  );
}
