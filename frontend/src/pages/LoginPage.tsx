import { useState } from 'react';
import { IconAlertCircle } from '@tabler/icons-react';
import { Link, Navigate, useLocation, useNavigate } from 'react-router-dom';
import { Alert, Anchor, Button, Group, PasswordInput, Stack, Text, TextInput } from '@mantine/core';
import { ApiError } from '@/api';
import { useAuth, useLoginWithCode } from '@/auth/AuthContext';
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
      setError(err instanceof Error ? err.message : 'The code was not accepted.');
      setCode('');
    } finally {
      setBusy(false);
    }
  };

  return (
    <form onSubmit={onSubmit}>
      <Stack>
        <Text size="sm">
          {recovery
            ? 'Enter one of your recovery codes. Each code works once.'
            : 'Enter the 6-digit code from your authenticator app.'}
        </Text>
        {error && (
          <Alert color="red" icon={<IconAlertCircle size={18} />}>
            {error}
          </Alert>
        )}
        <TextInput
          label={recovery ? 'Recovery code' : 'Code'}
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
          Verify
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
            {recovery ? 'Use the authenticator app' : 'Use a recovery code'}
          </Anchor>
          <Anchor component="button" type="button" size="sm" c="dimmed" onClick={onRestart}>
            Back
          </Anchor>
        </Group>
      </Stack>
    </form>
  );
}

export function LoginPage() {
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
      setError(err instanceof Error ? err.message : 'Login failed.');
    } finally {
      setBusy(false);
    }
  };

  if (challenge) {
    return (
      <AuthLayout title="Two-factor authentication" subtitle={`Logging in as ${username.trim()}`}>
        <CodeStep
          challenge={challenge}
          onDone={done}
          onRestart={() => {
            setChallenge(null);
            setPassword('');
            setError('The login has expired or was cancelled. Enter your password again.');
          }}
        />
      </AuthLayout>
    );
  }

  return (
    <AuthLayout
      title="Welcome back"
      subtitle={
        <>
          No account yet?{' '}
          <Anchor component={Link} to="/register" state={{ from, returnState }} size="sm">
            Create one
          </Anchor>
        </>
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
            label="Username"
            autoComplete="username"
            required
            value={username}
            onChange={(e) => setUsername(e.currentTarget.value)}
          />
          <PasswordInput
            label="Password"
            autoComplete="current-password"
            required
            value={password}
            onChange={(e) => setPassword(e.currentTarget.value)}
          />
          <Anchor component={Link} to="/forgot-password" size="sm" ta="right" mt={-8}>
            Forgot password?
          </Anchor>
          <Button type="submit" loading={busy} fullWidth mt="xs">
            Log in
          </Button>
        </Stack>
      </form>
    </AuthLayout>
  );
}
