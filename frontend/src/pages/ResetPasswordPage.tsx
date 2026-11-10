import { useEffect, useState } from 'react';
import { IconAlertCircle, IconCircleCheck } from '@tabler/icons-react';
import { Link } from 'react-router-dom';
import { Alert, Anchor, Button, Center, Loader, PasswordInput, Stack, Text } from '@mantine/core';
import { fieldError } from '@/account/fieldError';
import { api, ApiError } from '@/api';
import { AuthLayout } from './AuthLayout';
import { useLinkParams } from './linkParams';

type LinkState = { checked: false } | { checked: true; valid: boolean; username: string | null };

/** /reset-password#uid=…&token=…: the link from the e-mail; choose a new password. */
export function ResetPasswordPage() {
  const params = useLinkParams();
  const uid = params.get('uid') ?? '';
  const token = params.get('token') ?? '';
  const [link, setLink] = useState<LinkState>({ checked: false });
  const [password, setPassword] = useState('');
  const [repeat, setRepeat] = useState('');
  const [errors, setErrors] = useState<{ password?: string; other?: string }>({});
  const [busy, setBusy] = useState(false);
  const [done, setDone] = useState(false);

  useEffect(() => {
    if (!uid || !token) {
      setLink({ checked: true, valid: false, username: null });
      return;
    }
    api.passwordReset
      .check(uid, token)
      .then(({ valid, username }) => setLink({ checked: true, valid, username }))
      .catch(() => setLink({ checked: true, valid: false, username: null }));
  }, [uid, token]);

  const onSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (password !== repeat) {
      setErrors({ password: 'The two passwords are not the same.' });
      return;
    }
    setBusy(true);
    setErrors({});
    try {
      await api.passwordReset.confirm(uid, token, password);
      setDone(true);
    } catch (err) {
      const field = fieldError(err, 'new_password');
      if (field) {
        setErrors({ password: field });
      } else if (err instanceof ApiError && err.status === 400) {
        setLink({ checked: true, valid: false, username: null });
      } else {
        setErrors({ other: err instanceof Error ? err.message : 'Failed.' });
      }
    } finally {
      setBusy(false);
    }
  };

  let content: React.ReactNode;
  if (!link.checked) {
    content = (
      <Center h={80}>
        <Loader size="sm" />
      </Center>
    );
  } else if (done) {
    content = (
      <Stack>
        <Alert color="green" icon={<IconCircleCheck size={18} />} title="Password changed">
          Your password was changed and you were logged out everywhere. Log in with the new one.
        </Alert>
        <Button component={Link} to="/login" fullWidth>
          Log in
        </Button>
      </Stack>
    );
  } else if (!link.valid) {
    content = (
      <Stack>
        <Alert color="red" icon={<IconAlertCircle size={18} />} title="This link does not work">
          The link is invalid, has expired (it is valid for one hour) or was used already.
        </Alert>
        <Button component={Link} to="/forgot-password" variant="light" fullWidth>
          Ask for a new link
        </Button>
      </Stack>
    );
  } else {
    content = (
      <form onSubmit={onSubmit}>
        <Stack>
          {link.username && (
            <Text size="sm">
              Choose a new password for <b>{link.username}</b>.
            </Text>
          )}
          {errors.other && (
            <Alert color="red" icon={<IconAlertCircle size={18} />}>
              {errors.other}
            </Alert>
          )}
          <PasswordInput
            label="New password"
            description="At least 8 characters, not too common."
            autoComplete="new-password"
            required
            value={password}
            onChange={(e) => setPassword(e.currentTarget.value)}
            error={errors.password}
          />
          <PasswordInput
            label="Repeat the new password"
            autoComplete="new-password"
            required
            value={repeat}
            onChange={(e) => setRepeat(e.currentTarget.value)}
          />
          <Button type="submit" loading={busy} fullWidth mt="sm" disabled={!password || !repeat}>
            Change password
          </Button>
        </Stack>
      </form>
    );
  }

  return (
    <AuthLayout
      title="Choose a new password"
      subtitle={
        <Anchor component={Link} to="/login" size="sm">
          Back to logging in
        </Anchor>
      }
    >
      {content}
    </AuthLayout>
  );
}
