import { useState } from 'react';
import { IconAlertCircle } from '@tabler/icons-react';
import { Link, Navigate, useLocation, useNavigate } from 'react-router-dom';
import { Alert, Anchor, Button, Group, PasswordInput, Stack, Text, TextInput } from '@mantine/core';
import { ApiError } from '@/api';
import { useAuth, useLoginWithCode } from '@/auth/AuthContext';
import { Trans, useTranslation } from '@/i18n';
import { AuthLayout } from './AuthLayout';

/** Where to go after logging in, and the router state to pass on to that page. */
export interface ReturnTo {
  from?: string;
  returnState?: unknown;
}

/** The second step of a login with two-factor authentication. */
function CodeStep({
  challenge,
  onDone,
  onRestart,
}: {
  challenge: string;
  onDone: () => void;
  onRestart: () => void;
}) {
  const { t } = useTranslation();
  const loginWithCode = useLoginWithCode();
  const [code, setCode] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [recovery, setRecovery] = useState(false);

  const onSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await loginWithCode(challenge, code.trim());
      onDone();
    } catch (err) {
      const expired =
        err instanceof ApiError &&
        (err.data as { code?: unknown } | null)?.code === 'challenge_expired';
      if (expired) {
        onRestart();
        return;
      }
      setError(err instanceof Error ? err.message : t('auth.login.codeRejected'));
      setCode('');
    } finally {
      setBusy(false);
    }
  };

  return (
    <form onSubmit={onSubmit}>
      <Stack>
        <Text size="sm">
          {recovery ? t('auth.login.enterRecovery') : t('auth.login.enterCode')}
        </Text>
        {error && (
          <Alert color="red" icon={<IconAlertCircle size={18} />}>
            {error}
          </Alert>
        )}
        <TextInput
          label={recovery ? t('auth.login.recoveryCode') : t('auth.login.code')}
          autoComplete="one-time-code"
          inputMode={recovery ? 'text' : 'numeric'}
          autoCapitalize="none"
          spellCheck={false}
          required
          data-autofocus
          autoFocus
          value={code}
          onChange={(e) => setCode(e.currentTarget.value)}
          placeholder={recovery ? 'xxxx-xxxx-xxxx-xxxx' : '123456'}
          maxLength={recovery ? 40 : 6}
        />
        <Button type="submit" loading={busy} fullWidth mt="sm" disabled={!code.trim()}>
          {t('auth.login.verify')}
        </Button>
        <Group justify="space-between" gap="xs">
          <Anchor
            component="button"
            type="button"
            size="sm"
            onClick={() => {
              setRecovery(!recovery);
              setCode('');
              setError(null);
            }}
          >
            {recovery ? t('auth.login.useApp') : t('auth.login.useRecovery')}
          </Anchor>
          <Anchor component="button" type="button" size="sm" c="dimmed" onClick={onRestart}>
            {t('common.back')}
          </Anchor>
        </Group>
      </Stack>
    </form>
  );
}

export function LoginPage() {
  const { t } = useTranslation();
  const { user, login } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const { from = '/', returnState } = (location.state as ReturnTo | null) ?? {};
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [challenge, setChallenge] = useState<string | null>(null);

  if (user) {
    return <Navigate to={from} replace state={returnState} />;
  }

  const done = () => navigate(from, { replace: true, state: returnState });

  const onSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const step = await login(username.trim(), password);
      if (step.done) {
        done();
      } else {
        setChallenge(step.challenge);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : t('auth.login.failed'));
    } finally {
      setBusy(false);
    }
  };

  if (challenge) {
    return (
      <AuthLayout
        title={t('auth.login.twoFactorTitle')}
        subtitle={t('auth.login.loggingInAs', { username: username.trim() })}
      >
        <CodeStep
          challenge={challenge}
          onDone={done}
          onRestart={() => {
            setChallenge(null);
            setPassword('');
            setError(t('auth.login.expired'));
          }}
        />
      </AuthLayout>
    );
  }

  return (
    <AuthLayout
      title={t('auth.login.title')}
      subtitle={
        <Trans
          i18nKey="auth.login.noAccount"
          components={{
            anchor: (
              <Anchor component={Link} to="/register" state={{ from, returnState }} size="sm" />
            ),
          }}
        />
      }
    >
      <form onSubmit={onSubmit}>
        <Stack>
          {error && (
            <Alert color="red" icon={<IconAlertCircle size={18} />}>
              {error}
            </Alert>
          )}
          <TextInput
            label={t('auth.username')}
            autoComplete="username"
            required
            value={username}
            onChange={(e) => setUsername(e.currentTarget.value)}
          />
          <PasswordInput
            label={t('auth.password')}
            autoComplete="current-password"
            required
            value={password}
            onChange={(e) => setPassword(e.currentTarget.value)}
          />
          <Anchor component={Link} to="/forgot-password" size="sm" ta="right" mt={-8}>
            {t('auth.login.forgot')}
          </Anchor>
          <Button type="submit" loading={busy} fullWidth mt="xs">
            {t('auth.logIn')}
          </Button>
        </Stack>
      </form>
    </AuthLayout>
  );
}
