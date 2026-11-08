import { useEffect, useState } from 'react';
import { IconAlertCircle, IconCheck } from '@tabler/icons-react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import {
  Alert,
  Anchor,
  Avatar,
  Button,
  Center,
  Divider,
  Group,
  Loader,
  Stack,
  Switch,
  Text,
  ThemeIcon,
} from '@mantine/core';
import { notifications } from '@mantine/notifications';
import { api, ApiError, type OAuthRequest, type OAuthRequestError } from '@/api';
import { useAuth } from '@/auth/AuthContext';
import { AuthLayout } from '@/pages/AuthLayout';
import { urlHost } from './url';
import classes from './AuthorizePage.module.css';

type PageState =
  | { status: 'loading' }
  | { status: 'ready'; request: OAuthRequest }
  /** The app or its redirect URI is unknown: the error is shown here, never sent anywhere. */
  | { status: 'invalid'; description: string }
  /** The request could not be checked (no connection, server error); it can be retried. */
  | { status: 'failed'; message: string }
  /** The browser is on its way back to the app. */
  | { status: 'leaving'; outcome: Outcome; host: string | null; clientName: string | null };

/** Why the browser goes back to the app: with a code, after a denial, or with an error. */
type Outcome = 'allowed' | 'denied' | 'error';

const LEAVING_TITLES: Record<Outcome, string> = {
  allowed: 'Access allowed',
  denied: 'Access denied',
  error: 'Returning to the app',
};

/** The OAuth error of a rejected request (HTTP 400), or null for any other failure. */
function requestError(error: unknown): OAuthRequestError | null {
  if (!(error instanceof ApiError) || error.status !== 400) {
    return null;
  }
  const data = error.data as Partial<OAuthRequestError> | null;
  return data && typeof data.error === 'string' ? (data as OAuthRequestError) : null;
}

/**
 * Addresses that would run code on this site instead of leaving it. The server only accepts
 * registered redirect URIs; this is a second line of defence, since an app's own scheme (e.g.
 * for a desktop app) can be anything else.
 */
// eslint-disable-next-line no-script-url -- listed to be refused, not used.
const UNSAFE_PROTOCOLS = new Set(['javascript:', 'data:', 'vbscript:', 'blob:', 'file:']);

function isSafeRedirect(url: string): boolean {
  try {
    return !UNSAFE_PROTOCOLS.has(new URL(url).protocol);
  } catch {
    return false;
  }
}

/** Shows that the browser leaves for the app, and sends it there (with the code or an error). */
function leave(
  setState: (state: PageState) => void,
  redirectTo: string,
  outcome: Outcome,
  clientName: string | null
) {
  if (!isSafeRedirect(redirectTo)) {
    setState({
      status: 'invalid',
      description: 'The app wants to return to an address that is not allowed.',
    });
    return;
  }
  setState({ status: 'leaving', outcome, host: urlHost(redirectTo), clientName });
  window.location.assign(redirectTo);
}

/**
 * The page an app (e.g. Claude) sends the browser to when it connects to the MCP server: it
 * shows who asks for what, and the user allows or denies it. The server checks the request
 * and creates the authorization code; this page passes its query string on unchanged.
 */
export function AuthorizePage() {
  const { user, logout } = useAuth();
  const { pathname, search } = useLocation();
  const navigate = useNavigate();
  const query = search.replace(/^\?/, '');

  const [state, setState] = useState<PageState>({ status: 'loading' });
  // Increased by "Try again", to check the request once more.
  const [attempt, setAttempt] = useState(0);
  const [allowWrite, setAllowWrite] = useState(true);
  const [busy, setBusy] = useState<'allow' | 'deny' | null>(null);

  useEffect(() => {
    let active = true;
    api.oauth
      .validate(query)
      .then((request) => {
        if (active) {
          setState({ status: 'ready', request });
        }
      })
      .catch((error: Error) => {
        if (!active) {
          return;
        }
        const rejected = requestError(error);
        if (rejected?.redirect_to) {
          leave(setState, rejected.redirect_to, 'error', null);
        } else if (rejected) {
          setState({ status: 'invalid', description: rejected.error_description });
        } else {
          setState({ status: 'failed', message: error.message });
        }
      });
    return () => {
      active = false;
    };
  }, [query, attempt]);

  if (state.status === 'loading') {
    return (
      <Center h={300}>
        <Loader aria-label="Checking the sign-in request" />
      </Center>
    );
  }

  if (state.status === 'invalid') {
    return (
      <AuthLayout title="This sign-in link is not valid">
        <Stack>
          <Alert color="red" icon={<IconAlertCircle size={18} />}>
            {state.description || 'The app sent a sign-in request that Brick DNN cannot accept.'}
          </Alert>
          <Text size="sm">
            Go back to the app and connect it again. If this keeps happening, the app may not
            support this Brick DNN server.
          </Text>
          <Button component={Link} to="/" variant="default" fullWidth>
            Go to Brick DNN
          </Button>
        </Stack>
      </AuthLayout>
    );
  }

  if (state.status === 'failed') {
    return (
      <AuthLayout title="Could not check the sign-in request">
        <Stack>
          <Alert color="red" icon={<IconAlertCircle size={18} />}>
            {state.message}
          </Alert>
          <Button
            fullWidth
            onClick={() => {
              setState({ status: 'loading' });
              setAttempt((count) => count + 1);
            }}
          >
            Try again
          </Button>
        </Stack>
      </AuthLayout>
    );
  }

  if (state.status === 'leaving') {
    const app = state.clientName ?? 'the app';
    return (
      <AuthLayout title={LEAVING_TITLES[state.outcome]}>
        <Stack align="center" ta="center" gap="sm">
          <Loader size="sm" />
          <Text>
            Taking you back to{' '}
            {state.host ? <strong className={classes.wrap}>{state.host}</strong> : app}…
          </Text>
          <Text size="xs" c="dimmed">
            If nothing happens, go back to {app}. You can close this page.
          </Text>
        </Stack>
      </AuthLayout>
    );
  }

  const { request } = state;
  const name = request.client.name;
  const clientHost = urlHost(request.client.uri);
  const write = request.write_requested && allowWrite;

  const decide = async (approve: boolean) => {
    setBusy(approve ? 'allow' : 'deny');
    try {
      const { redirect_to } = await api.oauth.decide(query, approve, approve && write);
      leave(setState, redirect_to, approve ? 'allowed' : 'denied', name);
    } catch (error) {
      const rejected = requestError(error);
      if (rejected?.redirect_to) {
        leave(setState, rejected.redirect_to, 'error', name);
      } else if (rejected) {
        setState({ status: 'invalid', description: rejected.error_description });
      } else {
        notifications.show({
          color: 'red',
          title: 'Could not send your answer',
          message: (error as Error).message,
        });
        setBusy(null);
      }
    }
  };

  const switchAccount = () => {
    navigate('/login', { replace: true, state: { from: `${pathname}${search}` } });
    logout();
  };

  return (
    <AuthLayout
      title={
        <>
          Allow <span className={classes.wrap}>{name}</span> to use your Brick DNN account?
        </>
      }
      subtitle={clientHost && <span className={classes.wrap}>{clientHost}</span>}
    >
      <Stack gap="lg">
        <Group gap="xs" justify="space-between" wrap="wrap">
          <Group gap="xs" wrap="nowrap" miw={0}>
            <Avatar size={26} color="brick" radius="xl" aria-hidden>
              {user?.username.slice(0, 1).toUpperCase()}
            </Avatar>
            <Text size="sm" className={classes.wrap}>
              Signed in as <strong>{user?.username}</strong>
            </Text>
          </Group>
          <Text size="sm" c="dimmed">
            Not you?{' '}
            <Anchor component="button" type="button" size="sm" onClick={switchAccount}>
              Log out
            </Anchor>
          </Text>
        </Group>

        <Divider />

        <div>
          <Text size="sm" fw={600} mb="xs">
            <span className={classes.wrap}>{name}</span> will be able to:
          </Text>
          <Group gap="sm" align="flex-start" wrap="nowrap">
            <ThemeIcon size={20} radius="xl" variant="light" color="teal" mt={1} aria-hidden>
              <IconCheck size={13} />
            </ThemeIcon>
            <Text size="sm">
              Look at your networks, layer types and training runs, analyse networks and generate
              code
            </Text>
          </Group>
          {request.write_requested ? (
            <Switch
              mt="md"
              checked={allowWrite}
              onChange={(event) => setAllowWrite(event.currentTarget.checked)}
              disabled={busy !== null}
              label="Also allow changes: save networks, start and stop trainings"
              description={
                allowWrite
                  ? 'Turn this off for read-only access.'
                  : 'Read only: it changes nothing.'
              }
            />
          ) : (
            <Text size="sm" c="dimmed" mt="xs">
              It asks for read-only access and cannot change anything.
            </Text>
          )}
        </div>

        <Group grow gap="sm">
          <Button
            variant="default"
            onClick={() => void decide(false)}
            loading={busy === 'deny'}
            disabled={busy === 'allow'}
          >
            Deny
          </Button>
          <Button
            onClick={() => void decide(true)}
            loading={busy === 'allow'}
            disabled={busy === 'deny'}
          >
            Allow
          </Button>
        </Group>

        <Stack gap={4}>
          <Text size="xs" c="dimmed">
            After you decide, you will return to{' '}
            <strong className={classes.wrap}>{request.redirect_host}</strong>.
          </Text>
          <Text size="xs" c="dimmed">
            You can disconnect it at any time on the API tokens page.
          </Text>
        </Stack>
      </Stack>
    </AuthLayout>
  );
}
