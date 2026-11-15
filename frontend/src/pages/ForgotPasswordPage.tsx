import { useEffect, useState } from 'react';
import { IconAlertCircle, IconMailCheck } from '@tabler/icons-react';
import { Link } from 'react-router-dom';
import { Alert, Anchor, Button, Center, Loader, Stack, Text, TextInput } from '@mantine/core';
import { api } from '@/api';
import { useTranslation } from '@/i18n';
import { AuthLayout } from './AuthLayout';

/** /forgot-password: ask for a link to choose a new password. */
export function ForgotPasswordPage() {
  const { t } = useTranslation();
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
      setError(err instanceof Error ? err.message : t('common.errors.requestFailed'));
    } finally {
      setBusy(false);
    }
  };

  const back = (
    <Anchor component={Link} to="/login" size="sm">
      {t('auth.backToLogin')}
    </Anchor>
  );

  return (
    <AuthLayout title={t('auth.forgot.title')} subtitle={back}>
      {available === null ? (
        <Center h={80}>
          <Loader size="sm" />
        </Center>
      ) : !available ? (
        <Alert color="yellow" icon={<IconAlertCircle size={18} />}>
          {t('auth.forgot.unavailable')}
        </Alert>
      ) : sent ? (
        <Stack>
          <Alert
            color="green"
            icon={<IconMailCheck size={18} />}
            title={t('auth.forgot.checkMail')}
          >
            {sent}
          </Alert>
          <Text size="sm" c="dimmed">
            {t('auth.forgot.nothingArrived')}
          </Text>
        </Stack>
      ) : (
        <form onSubmit={onSubmit}>
          <Stack>
            <Text size="sm">{t('auth.forgot.intro')}</Text>
            {error && (
              <Alert color="red" icon={<IconAlertCircle size={18} />}>
                {error}
              </Alert>
            )}
            <TextInput
              label={t('auth.forgot.login')}
              autoComplete="username"
              autoCapitalize="none"
              spellCheck={false}
              required
              value={login}
              onChange={(e) => setLogin(e.currentTarget.value)}
            />
            <Button type="submit" loading={busy} fullWidth mt="sm" disabled={!login.trim()}>
              {t('auth.forgot.submit')}
            </Button>
          </Stack>
        </form>
      )}
    </AuthLayout>
  );
}
