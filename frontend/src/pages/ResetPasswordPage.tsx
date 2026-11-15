import { useEffect, useState } from 'react';
import { IconAlertCircle, IconCircleCheck } from '@tabler/icons-react';
import { Link } from 'react-router-dom';
import { Alert, Anchor, Button, Center, Loader, PasswordInput, Stack, Text } from '@mantine/core';
import { fieldError } from '@/account/fieldError';
import { api, ApiError } from '@/api';
import { Trans, useTranslation } from '@/i18n';
import { AuthLayout } from './AuthLayout';
import { useLinkParams } from './linkParams';

type LinkState = { checked: false } | { checked: true; valid: boolean; username: string | null };

/** /reset-password#uid=…&token=…: the link from the e-mail; choose a new password. */
export function ResetPasswordPage() {
  const { t } = useTranslation();
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
      setErrors({ password: t('auth.reset.mismatch') });
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
        setErrors({ other: err instanceof Error ? err.message : t('common.failed') });
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
        <Alert
          color="green"
          icon={<IconCircleCheck size={18} />}
          title={t('auth.reset.changedTitle')}
        >
          {t('auth.reset.changed')}
        </Alert>
        <Button component={Link} to="/login" fullWidth>
          {t('auth.logIn')}
        </Button>
      </Stack>
    );
  } else if (!link.valid) {
    content = (
      <Stack>
        <Alert
          color="red"
          icon={<IconAlertCircle size={18} />}
          title={t('auth.reset.invalidTitle')}
        >
          {t('auth.reset.invalid')}
        </Alert>
        <Button component={Link} to="/forgot-password" variant="light" fullWidth>
          {t('auth.reset.newLink')}
        </Button>
      </Stack>
    );
  } else {
    content = (
      <form onSubmit={onSubmit}>
        <Stack>
          {link.username && (
            <Text size="sm">
              <Trans
                i18nKey="auth.reset.chooseFor"
                values={{ username: link.username }}
                components={{ b: <b /> }}
              />
            </Text>
          )}
          {errors.other && (
            <Alert color="red" icon={<IconAlertCircle size={18} />}>
              {errors.other}
            </Alert>
          )}
          <PasswordInput
            label={t('auth.reset.newPassword')}
            description={t('auth.passwordHint')}
            autoComplete="new-password"
            required
            value={password}
            onChange={(e) => setPassword(e.currentTarget.value)}
            error={errors.password}
          />
          <PasswordInput
            label={t('auth.reset.repeat')}
            autoComplete="new-password"
            required
            value={repeat}
            onChange={(e) => setRepeat(e.currentTarget.value)}
          />
          <Button type="submit" loading={busy} fullWidth mt="sm" disabled={!password || !repeat}>
            {t('auth.reset.submit')}
          </Button>
        </Stack>
      </form>
    );
  }

  return (
    <AuthLayout
      title={t('auth.reset.title')}
      subtitle={
        <Anchor component={Link} to="/login" size="sm">
          {t('auth.backToLogin')}
        </Anchor>
      }
    >
      {content}
    </AuthLayout>
  );
}
