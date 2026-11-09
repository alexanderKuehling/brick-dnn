import { useState } from 'react';
import { IconAlertCircle } from '@tabler/icons-react';
import { Link, Navigate, useLocation, useNavigate } from 'react-router-dom';
import { Alert, Anchor, Button, PasswordInput, Stack, Text, TextInput } from '@mantine/core';
import { useAuth } from '@/auth/AuthContext';
import { AuthLayout } from './AuthLayout';
import type { ReturnTo } from './LoginPage';

export function RegisterPage() {
  const { user, register } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const { from = '/', returnState } = (location.state as ReturnTo | null) ?? {};
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  if (user) {
    return <Navigate to={from} replace state={returnState} />;
  }

  const onSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await register(username.trim(), email.trim(), password);
      navigate(from, { replace: true, state: returnState });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Registration failed.');
    } finally {
      setBusy(false);
    }
  };

  return (
    <AuthLayout
      title="Create an account"
      subtitle={
        <>
          Already registered?{' '}
          <Anchor component={Link} to="/login" state={{ from, returnState }} size="sm">
            Log in
          </Anchor>
        </>
      }
    >
      <form onSubmit={onSubmit}>
        <Stack>
          <Text size="sm" c="dimmed">
            An account lets you save networks on this server. Building and exporting code works
            without one.
          </Text>
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
          <TextInput
            label="Email"
            description="Other people can invite you to their networks with it."
            type="email"
            autoComplete="email"
            required
            value={email}
            onChange={(e) => setEmail(e.currentTarget.value)}
          />
          <PasswordInput
            label="Password"
            description="At least 8 characters, not too common."
            autoComplete="new-password"
            required
            value={password}
            onChange={(e) => setPassword(e.currentTarget.value)}
          />
          <Button type="submit" loading={busy} fullWidth mt="sm">
            Create account
          </Button>
        </Stack>
      </form>
    </AuthLayout>
  );
}
