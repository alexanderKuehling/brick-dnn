import { useEffect, useRef, useState } from 'react';
import { IconAlertCircle, IconCircleCheck } from '@tabler/icons-react';
import { Link } from 'react-router-dom';
import { Alert, Button, Center, Loader, Stack } from '@mantine/core';
import { api, ApiError } from '@/api';
import { useAuth } from '@/auth/AuthContext';
import { AuthLayout } from './AuthLayout';
import { useLinkParams } from './linkParams';

type Result = { ok: true; email: string } | { ok: false; message: string; otherAccount: boolean };

/**
 * /verify-email#token=…: the link from the e-mail. Only the account the link was sent to can
 * confirm its address, so the page is shown after logging in (RequireAuth).
 */
export function VerifyEmailPage() {
  const params = useLinkParams();
  const token = params.get('token') ?? '';
  const { user, logout } = useAuth();
  const [result, setResult] = useState<Result | null>(null);
  const sent = useRef(false);

  useEffect(() => {
    if (sent.current) {
      return; // once, also in React's strict mode
    }
    sent.current = true;
    api.account
      .verifyEmail(token)
      .then((data) => setResult({ ok: true, email: data.email ?? '' }))
      .catch((err: unknown) => {
        const code =
          err instanceof ApiError ? (err.data as { code?: unknown } | null)?.code : undefined;
        setResult({
          ok: false,
          message: err instanceof Error ? err.message : 'The address could not be confirmed.',
          otherAccount: code === 'other_account',
        });
      });
  }, [token]);

  let content: React.ReactNode;
  if (!result) {
    content = (
      <Center h={80}>
        <Loader size="sm" />
      </Center>
    );
  } else if (result.ok) {
    content = (
      <Stack>
        <Alert color="green" icon={<IconCircleCheck size={18} />} title="Thank you">
          {result.email ? `${result.email} is confirmed` : 'Your e-mail address is confirmed'}. If
          you ever forget your password, you can reset it with a link sent to it.
        </Alert>
        <Button component={Link} to="/account" fullWidth>
          Go to your account
        </Button>
      </Stack>
    );
  } else {
    content = (
      <Stack>
        <Alert color="red" icon={<IconAlertCircle size={18} />} title="Not confirmed">
          {result.message}
        </Alert>
        {result.otherAccount ? (
          <Button variant="light" onClick={() => logout()} fullWidth>
            Log out of {user?.username ?? 'this account'}
          </Button>
        ) : (
          <Button component={Link} to="/account" variant="light" fullWidth>
            Send a new link from your account
          </Button>
        )}
      </Stack>
    );
  }

  return <AuthLayout title="Confirm your e-mail address">{content}</AuthLayout>;
}
