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
import { i18n, Trans, useTranslation } from '@/i18n';
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
  get allowed() {
    return i18n.t('ui.oauth.AuthorizePage.accessAllowed');
  },
  get denied() {
    return i18n.t('ui.oauth.AuthorizePage.accessDenied');
  },
  get error() {
    return i18n.t('ui.oauth.AuthorizePage.returningToTheApp');
  },
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
      description: i18n.t('ui.oauth.AuthorizePage.theAppWantsToReturn'),
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
  useTranslation();
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
        <Loader aria-label={i18n.t('ui.oauth.AuthorizePage.checkingTheSignInRequest')} />
      </Center>
    );
  }

  if (state.status === 'invalid') {
    return (
      <AuthLayout title={i18n.t('ui.oauth.AuthorizePage.thisSignInLinkIs')}>
        <Stack>
          <Alert color="red" icon={<IconAlertCircle size={18} />}>
            {state.description || i18n.t('ui.oauth.AuthorizePage.theAppSentASign')}
          </Alert>
          <Text size="sm">{i18n.t('ui.oauth.AuthorizePage.goBackToTheApp')}</Text>
          <Button component={Link} to="/" variant="default" fullWidth>
            {i18n.t('ui.oauth.AuthorizePage.goToBrickDnn')}
          </Button>
        </Stack>
      </AuthLayout>
    );
  }

  if (state.status === 'failed') {
    return (
      <AuthLayout title={i18n.t('ui.oauth.AuthorizePage.couldNotCheckTheSign')}>
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
            {i18n.t('ui.oauth.AuthorizePage.tryAgain')}
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
            {i18n.t('ui.oauth.AuthorizePage.takingYouBackTo')}{' '}
            {state.host ? <strong className={classes.wrap}>{state.host}</strong> : app}…
          </Text>
          <Text size="xs" c="dimmed">
            {i18n.t('ui.oauth.AuthorizePage.ifNothingHappensGoBack', { app })}
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
          title: i18n.t('ui.oauth.AuthorizePage.couldNotSendYourAnswer'),
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
          {i18n.t('ui.oauth.AuthorizePage.allow')} <span className={classes.wrap}>{name}</span>{' '}
          {i18n.t('ui.oauth.AuthorizePage.toUseYourBrickDnn')}
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
              <Trans
                i18nKey="ui.oauth.AuthorizePage.signedInAs"
                values={{ username: user?.username }}
                components={[<strong />]}
              />
            </Text>
          </Group>
          <Text size="sm" c="dimmed">
            <Trans
              i18nKey="ui.oauth.AuthorizePage.notYouLogOut"
              components={[
                <Anchor component="button" type="button" size="sm" onClick={switchAccount} />,
              ]}
            />
          </Text>
        </Group>

        <Divider />

        <div>
          <Text size="sm" fw={600} mb="xs">
            <span className={classes.wrap}>{name}</span>{' '}
            {i18n.t('ui.oauth.AuthorizePage.willBeAbleTo')}
          </Text>
          <Group gap="sm" align="flex-start" wrap="nowrap">
            <ThemeIcon size={20} radius="xl" variant="light" color="teal" mt={1} aria-hidden>
              <IconCheck size={13} />
            </ThemeIcon>
            <Text size="sm">{i18n.t('ui.oauth.AuthorizePage.lookAtYourNetworksLayer')}</Text>
          </Group>
          {request.write_requested ? (
            <Switch
              mt="md"
              checked={allowWrite}
              onChange={(event) => setAllowWrite(event.currentTarget.checked)}
              disabled={busy !== null}
              label={i18n.t('ui.oauth.AuthorizePage.alsoAllowChangesSaveNetworks')}
              description={
                allowWrite
                  ? i18n.t('ui.oauth.AuthorizePage.turnThisOffForRead')
                  : i18n.t('ui.oauth.AuthorizePage.readOnlyItChangesNothing')
              }
            />
          ) : (
            <Text size="sm" c="dimmed" mt="xs">
              {i18n.t('ui.oauth.AuthorizePage.itAsksForReadOnly')}
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
            {i18n.t('ui.oauth.AuthorizePage.deny')}
          </Button>
          <Button
            onClick={() => void decide(true)}
            loading={busy === 'allow'}
            disabled={busy === 'deny'}
          >
            {i18n.t('ui.oauth.AuthorizePage.allow')}
          </Button>
        </Group>

        <Stack gap={4}>
          <Text size="xs" c="dimmed">
            <Trans
              i18nKey="ui.oauth.AuthorizePage.afterYouDecideYouWill"
              values={{ redirect_host: request.redirect_host }}
              components={[<strong className={classes.wrap} />]}
            />
          </Text>
          <Text size="xs" c="dimmed">
            {i18n.t('ui.oauth.AuthorizePage.youCanDisconnectItAt')}
          </Text>
        </Stack>
      </Stack>
    </AuthLayout>
  );
}
