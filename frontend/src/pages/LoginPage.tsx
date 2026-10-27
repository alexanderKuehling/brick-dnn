import { useState } from 'react';
import { IconAlertCircle } from '@tabler/icons-react';
import { Link, Navigate, useLocation, useNavigate } from 'react-router-dom';
import { Alert, Anchor, Button, PasswordInput, Stack, TextInput } from '@mantine/core';
import { useAuth } from '@/auth/AuthContext';
import { AuthLayout } from './AuthLayout';

export function LoginPage() {
  const { user, login } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const from = (location.state as { from?: string } | null)?.from ?? '/';
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  if (user) {
    return <Navigate to={from} replace />;
  }

  const onSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await login(username.trim(), password);
      navigate(from, { replace: true });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Login failed.');
    } finally {
      setBusy(false);
    }
  };

  return (
    <AuthLayout
      title="Welcome back"
      subtitle={
        <>
          No account yet?{' '}
          <Anchor component={Link} to="/register" state={{ from }} size="sm">
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
          <Button type="submit" loading={busy} fullWidth mt="sm">
            Log in
          </Button>
        </Stack>
      </form>
    </AuthLayout>
  );
}
