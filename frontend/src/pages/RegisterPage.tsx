import { useRef, useState } from 'react';
import { IconAlertCircle } from '@tabler/icons-react';
import { Link, Navigate, useLocation, useNavigate } from 'react-router-dom';
import {
  Alert,
  Anchor,
  Button,
  Checkbox,
  PasswordInput,
  Stack,
  Text,
  TextInput,
} from '@mantine/core';
import { fieldError } from '@/account/fieldError';
import { useAuth } from '@/auth/AuthContext';
import { Trans, useTranslation } from '@/i18n';
import { useLegalInfo } from '@/legal/useLegalInfo';
import { AuthLayout } from './AuthLayout';
import type { ReturnTo } from './LoginPage';

/** The fields the API names in its errors, in the order of the form. */
const FIELDS = ['username', 'email', 'password'] as const;
type Field = (typeof FIELDS)[number];

export function RegisterPage() {
  const { t } = useTranslation();
  const { user, register } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const { from = '/', returnState } = (location.state as ReturnTo | null) ?? {};
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [accepted, setAccepted] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Errors of single fields: shown at the field and linked to it (aria-describedby), and the
  // first one gets the focus, so that a screen reader reads it with the field.
  const [errors, setErrors] = useState<Partial<Record<Field, string>>>({});
  const inputs = useRef<Partial<Record<Field, HTMLInputElement | null>>>({});
  const [busy, setBusy] = useState(false);
  const legal = useLegalInfo();
  const termsRequired = Boolean(legal?.accept_terms_required);

  if (user) {
    return <Navigate to={from} replace state={returnState} />;
  }

  const onSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setError(null);
    setErrors({});
    try {
      await register(username.trim(), email.trim(), password, termsRequired && accepted);
      navigate(from, { replace: true, state: returnState });
    } catch (err) {
      const found = Object.fromEntries(
        FIELDS.map((field) => [field, fieldError(err, field)]).filter(([, message]) => message)
      ) as Partial<Record<Field, string>>;
      const first = FIELDS.find((field) => found[field]);
      if (first) {
        setErrors(found);
        window.setTimeout(() => inputs.current[first]?.focus(), 0);
      } else {
        setError(err instanceof Error ? err.message : t('auth.register.failed'));
      }
    } finally {
      setBusy(false);
    }
  };

  return (
    <AuthLayout
      title={t('auth.register.title')}
      subtitle={
        <Trans
          i18nKey="auth.register.already"
          components={{
            anchor: <Anchor component={Link} to="/login" state={{ from, returnState }} size="sm" />,
          }}
        />
      }
    >
      <form onSubmit={onSubmit}>
        <Stack>
          <Text size="sm" c="dimmed">
            {t('auth.register.intro')}
          </Text>
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
            error={errors.username}
            ref={(element) => {
              inputs.current.username = element;
            }}
          />
          <TextInput
            label={t('auth.email')}
            description={t('auth.register.emailHint')}
            type="email"
            autoComplete="email"
            required
            value={email}
            onChange={(e) => setEmail(e.currentTarget.value)}
            error={errors.email}
            ref={(element) => {
              inputs.current.email = element;
            }}
          />
          <PasswordInput
            label={t('auth.password')}
            description={t('auth.passwordHint')}
            autoComplete="new-password"
            required
            value={password}
            onChange={(e) => setPassword(e.currentTarget.value)}
            error={errors.password}
            ref={(element) => {
              inputs.current.password = element;
            }}
          />
          {termsRequired ? (
            <Checkbox
              required
              checked={accepted}
              onChange={(e) => setAccepted(e.currentTarget.checked)}
              label={
                <Trans
                  i18nKey={
                    legal?.pages.privacy
                      ? 'auth.register.acceptTermsAndPrivacy'
                      : 'auth.register.acceptTerms'
                  }
                  components={{
                    terms: <Anchor component={Link} to="/terms" target="_blank" inherit />,
                    privacy: <Anchor component={Link} to="/privacy" target="_blank" inherit />,
                  }}
                />
              }
            />
          ) : (
            legal?.pages.privacy && (
              <Text size="xs" c="dimmed">
                <Trans
                  i18nKey="auth.register.privacyNote"
                  components={{
                    privacy: <Anchor component={Link} to="/privacy" target="_blank" inherit />,
                  }}
                />
              </Text>
            )
          )}
          <Button type="submit" loading={busy} fullWidth mt="sm">
            {t('auth.register.submit')}
          </Button>
        </Stack>
      </form>
    </AuthLayout>
  );
}
