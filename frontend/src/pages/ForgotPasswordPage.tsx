import { useEffect, useState } from 'react';
import { IconAlertCircle, IconMailCheck } from '@tabler/icons-react';
import { Link } from 'react-router-dom';
import { Alert, Anchor, Button, Center, Loader, Stack, Text, TextInput } from '@mantine/core';
import { api } from '@/api';
import { AuthLayout } from './AuthLayout';

export const RESET_UNAVAILABLE =
  'Password reset is not available on this server: it sends no e-mail. Ask its operator for help.';

/** /forgot-password: ask for a link to choose a new password. */
export function ForgotPasswordPage() {
  const [available, setAvailable] = useState<boolean | null>(null);
  const [login, setLogin] = useState('');
  const [sent, setSent] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api.passwordReset
      .available()
      .then(({ available: yes }) => setAvailable(yes))
      .catch(() => setAvailable(true)); // let the request itself tell what is wrong
  }, []);

  const onSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const { detail } = await api.passwordReset.request(login.trim());
      setSent(detail);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'The request failed.');
    } finally {
      setBusy(false);
    }
  };

  const back = (
    <Anchor component={Link} to="/login" size="sm">
      Back to logging in
    </Anchor>
  );

  return (
    <AuthLayout title="Forgot your password?" subtitle={back}>
      {available === null ? (
        <Center h={80}>
          <Loader size="sm" />
        </Center>
      ) : !available ? (
        <Alert color="yellow" icon={<IconAlertCircle size={18} />}>
          {RESET_UNAVAILABLE}
        </Alert>
      ) : sent ? (
        <Stack>
          <Alert color="green" icon={<IconMailCheck size={18} />} title="Check your e-mail">
            {sent}
          </Alert>
          <Text size="sm" c="dimmed">
            Nothing arrived? Check the spam folder. Accounts whose e-mail address was never
            confirmed cannot get a link; ask the operator of this site for help.
          </Text>
        </Stack>
      ) : (
        <form onSubmit={onSubmit}>
          <Stack>
            <Text size="sm">
              Enter your username or e-mail address. We send a link to choose a new password to the
              confirmed e-mail address of your account.
            </Text>
            {error && (
              <Alert color="red" icon={<IconAlertCircle size={18} />}>
                {error}
              </Alert>
            )}
            <TextInput
              label="Username or e-mail address"
              autoComplete="username"
              autoCapitalize="none"
              spellCheck={false}
              required
              value={login}
              onChange={(e) => setLogin(e.currentTarget.value)}
            />
            <Button type="submit" loading={busy} fullWidth mt="sm" disabled={!login.trim()}>
              Send me a link
            </Button>
          </Stack>
        </form>
      )}
    </AuthLayout>
  );
}
