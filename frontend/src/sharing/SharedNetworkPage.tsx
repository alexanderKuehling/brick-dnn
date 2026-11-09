import { lazy, Suspense, useEffect, useState } from 'react';
import { IconCode, IconCopyPlus, IconEye, IconLinkOff } from '@tabler/icons-react';
import { Link, useLocation, useNavigate, useParams } from 'react-router-dom';
import {
  Badge,
  Box,
  Button,
  Center,
  Container,
  Group,
  Loader,
  Text,
  ThemeIcon,
  Title,
} from '@mantine/core';
import { useDisclosure } from '@mantine/hooks';
import { notifications } from '@mantine/notifications';
import { api, ApiError, type GenerateOptions, type SharedNetwork } from '@/api';
import { useAuth } from '@/auth/AuthContext';
import { projectDraft } from '@/builder/draft';
import { slugify } from '@/builder/graph';
import { NetworkViewer } from '@/builder/NetworkViewer';
import { useOpenInBuilder } from '@/builder/useOpenInBuilder';
import { HEADER_HEIGHT } from '@/components/AppHeader';
import { formatDate } from '@/training/format';
import classes from './Sharing.module.css';

const CodeModal = lazy(() => import('@/builder/CodeModal'));

interface LoadError {
  notFound: boolean;
  message: string;
}

/** A network opened through its share link: read only, no login needed. */
export function SharedNetworkPage() {
  const { token = '' } = useParams<{ token: string }>();
  const { user, ready } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const openInBuilder = useOpenInBuilder();
  const [network, setNetwork] = useState<SharedNetwork | null>(null);
  const [error, setError] = useState<LoadError | null>(null);
  const [attempt, setAttempt] = useState(0);
  const [duplicating, setDuplicating] = useState(false);
  const [codeOpened, code] = useDisclosure(false);
  const [codeOptions, setCodeOptions] = useState<GenerateOptions>({
    class_name: 'NeuralNetwork',
    include_main: true,
    include_training: true,
  });

  useEffect(() => {
    let cancelled = false;
    setNetwork(null);
    setError(null);
    api.shared
      .get(token)
      .then((result) => !cancelled && setNetwork(result))
      .catch((err: Error) => {
        if (!cancelled) {
          setError({
            notFound: err instanceof ApiError && err.status === 404,
            message: err.message,
          });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [token, attempt]);

  const duplicate = async () => {
    if (!user) {
      notifications.show({
        title: 'Log in to duplicate',
        message: 'A copy of your own needs a free account. You come back here afterwards.',
      });
      // The login page sends the user back here, with the state to continue.
      navigate('/login', {
        state: { from: location.pathname, returnState: { duplicate: true } },
      });
      return;
    }
    setDuplicating(true);
    try {
      const copy = await api.shared.duplicate(token);
      notifications.show({
        color: 'green',
        title: `Saved “${copy.name}” to your networks`,
        message: 'This copy is yours: you can change it.',
      });
      openInBuilder(projectDraft(copy));
    } catch (err) {
      notifications.show({
        color: 'red',
        title: 'Could not duplicate the network',
        message: (err as Error).message,
      });
    } finally {
      setDuplicating(false);
    }
  };

  // Back from logging in to duplicate: continue (once; the state is cleared).
  const continueDuplicate = Boolean((location.state as { duplicate?: boolean } | null)?.duplicate);
  useEffect(() => {
    if (continueDuplicate && ready && user && network) {
      navigate(location.pathname, { replace: true, state: null });
      void duplicate();
    }
  }, [continueDuplicate, ready, user, network]);

  if (error) {
    return (
      <Container size="sm" py={80} ta="center">
        <ThemeIcon variant="light" color="gray" size={56} radius="xl" aria-hidden>
          <IconLinkOff size={30} />
        </ThemeIcon>
        <Title order={2} mt="md">
          {error.notFound
            ? 'This link does not exist or was turned off'
            : 'Cannot open the network'}
        </Title>
        <Text c="dimmed" mt="sm">
          {error.notFound ? 'Ask the person who shared the network for a new link.' : error.message}
        </Text>
        {error.notFound ? (
          <Button component={Link} to="/" mt="xl">
            Go to the builder
          </Button>
        ) : (
          <Button mt="xl" onClick={() => setAttempt((n) => n + 1)}>
            Try again
          </Button>
        )}
      </Container>
    );
  }

  if (!network) {
    return (
      <Center h={`calc(100dvh - ${HEADER_HEIGHT}px)`}>
        <Loader />
      </Center>
    );
  }

  return (
    <Box
      style={{
        display: 'flex',
        flexDirection: 'column',
        height: `calc(100dvh - ${HEADER_HEIGHT}px)`,
      }}
    >
      <div className={classes.sharedBar}>
        <div className={classes.sharedTitle}>
          <Group gap="xs" wrap="nowrap">
            <Title order={1} size="h4" lineClamp={1} className={classes.sharedName}>
              {network.name}
            </Title>
            <Badge
              color="gray"
              variant="outline"
              tt="none"
              leftSection={<IconEye size={12} />}
              style={{ flexShrink: 0 }}
            >
              View only
            </Badge>
          </Group>
          <Text size="sm" c="dimmed">
            by {network.owner} · updated {formatDate(network.updated_at)}
          </Text>
        </div>
        <Group gap="xs" className={classes.sharedActions}>
          <Button variant="default" leftSection={<IconCode size={16} />} onClick={code.open}>
            View code
          </Button>
          <Button
            leftSection={<IconCopyPlus size={16} />}
            loading={duplicating}
            onClick={() => void duplicate()}
          >
            Duplicate to my networks
          </Button>
        </Group>
        {network.description && (
          <Text size="sm" c="dimmed" className={classes.sharedDescription}>
            {network.description}
          </Text>
        )}
      </div>
      <NetworkViewer key={token} doc={network.graph} />
      {codeOpened && (
        <Suspense fallback={null}>
          <CodeModal
            opened={codeOpened}
            onClose={code.close}
            doc={network.graph}
            filename={slugify(network.name)}
            options={codeOptions}
            onOptionsChange={setCodeOptions}
          />
        </Suspense>
      )}
    </Box>
  );
}
