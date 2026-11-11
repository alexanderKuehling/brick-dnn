import { useEffect, useRef, useState } from 'react';
import {
  IconAlertCircle,
  IconAlertTriangle,
  IconCircleCheck,
  IconDownload,
  IconMailQuestion,
  IconTrash,
} from '@tabler/icons-react';
import { useNavigate } from 'react-router-dom';
import {
  Alert,
  Badge,
  Button,
  Center,
  Container,
  Group,
  List,
  Loader,
  Modal,
  Paper,
  PasswordInput,
  Stack,
  Text,
  TextInput,
  Title,
} from '@mantine/core';
import { notifications } from '@mantine/notifications';
import { api, setTokens, type Account } from '@/api';
import { useAuth } from '@/auth/AuthContext';
import { PlanSection } from '@/billing/PlanSection';
import { useBillingInfo } from '@/billing/useBillingInfo';
import { downloadBlob } from '@/builder/download';
import { LegalLinks } from '@/legal/LegalLinks';
import { useLegalInfo } from '@/legal/useLegalInfo';
import { formatBytes, formatDate } from '@/training/format';
import { UsageSection } from '@/usage/UsageSection';
import { fieldError } from './fieldError';
import { TwoFactorSection } from './TwoFactorSection';

function plural(count: number, word: string, words = `${word}s`): string {
  return `${count.toLocaleString('en-US')} ${count === 1 ? word : words}`;
}

/** "3 networks, 2 comments, 1 dataset …": what the server keeps for the account. */
export function storedSummary(stored: Account['stored']): string[] {
  const lines = [
    stored.projects > 0 && plural(stored.projects, 'network'),
    stored.memberships > 0 && `access to ${plural(stored.memberships, 'network')} of other people`,
    stored.comments > 0 && plural(stored.comments, 'comment'),
    stored.data_sources > 0 && plural(stored.data_sources, 'imported table'),
    stored.data_preps > 0 && plural(stored.data_preps, 'data preparation'),
    stored.datasets > 0 && plural(stored.datasets, 'training dataset'),
    stored.training_runs > 0 && plural(stored.training_runs, 'training run'),
    stored.api_tokens > 0 && plural(stored.api_tokens, 'API token'),
    stored.connected_apps > 0 && plural(stored.connected_apps, 'connected app'),
    stored.agent_actions > 0 &&
      `${plural(stored.agent_actions, 'entry', 'entries')} in the agent activity log`,
  ];
  return lines.filter((line): line is string => Boolean(line));
}

function Section({
  title,
  danger = false,
  children,
}: {
  title: string;
  danger?: boolean;
  children: React.ReactNode;
}) {
  return (
    <Paper
      component="section"
      withBorder
      radius="md"
      p={{ base: 'md', sm: 'lg' }}
      style={danger ? { borderColor: 'var(--mantine-color-red-filled)' } : undefined}
      aria-label={title}
    >
      <Title order={3} size="h4" mb="sm" c={danger ? 'red' : undefined}>
        {title}
      </Title>
      {children}
    </Paper>
  );
}

/** Whether the address is confirmed, and sending the link again. */
function EmailStatus({ account }: { account: Account }) {
  const [busy, setBusy] = useState(false);
  const [sent, setSent] = useState(false);

  if (!account.email || !account.email_sending) {
    return null;
  }
  if (account.email_verified) {
    return (
      <Group gap={6}>
        <Badge color="green" variant="light" leftSection={<IconCircleCheck size={12} />}>
          Confirmed
        </Badge>
        {account.email_verified_at && (
          <Text size="xs" c="dimmed">
            {formatDate(account.email_verified_at)}
          </Text>
        )}
      </Group>
    );
  }

  const resend = async () => {
    setBusy(true);
    try {
      const { detail } = await api.account.resendVerification();
      setSent(true);
      notifications.show({ color: 'green', message: detail });
    } catch (err) {
      notifications.show({
        color: 'red',
        title: 'Could not send the link',
        message: (err as Error).message,
      });
    } finally {
      setBusy(false);
    }
  };

  return (
    <Alert color="yellow" icon={<IconMailQuestion size={18} />} title="Not confirmed yet">
      <Stack gap="xs">
        <Text size="sm">
          {account.password_reset_available
            ? 'Open the link we sent to this address to confirm it.'
            : 'Open the link we sent to this address to confirm it. Until then, a forgotten password cannot be reset by e-mail.'}
        </Text>
        <Group>
          <Button size="xs" variant="light" color="yellow" loading={busy} onClick={resend}>
            {sent ? 'Send the link again' : 'Send me a new link'}
          </Button>
        </Group>
      </Stack>
    </Alert>
  );
}

function EmailForm({ account, onSaved }: { account: Account; onSaved: (a: Account) => void }) {
  const [email, setEmail] = useState(account.email);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const changed = email.trim() !== account.email;

  const save = async (event: React.FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const updated = await api.account.changeEmail(email.trim());
      onSaved(updated);
      setEmail(updated.email);
      notifications.show({
        color: 'green',
        message: updated.email_sending
          ? `Your e-mail address was changed. We sent a link to ${updated.email} to confirm it.`
          : 'Your e-mail address was changed.',
      });
    } catch (err) {
      setError(fieldError(err, 'email') ?? (err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <form onSubmit={save}>
      <Stack gap="sm">
        {!account.email && (
          <Alert color="yellow" icon={<IconAlertTriangle size={18} />}>
            Your account has no e-mail address yet. Add one, so that people can invite you to their
            networks by e-mail{account.email_sending && ' and you can reset a forgotten password'}.
          </Alert>
        )}
        <TextInput
          label="Username"
          value={account.username}
          readOnly
          description="Usernames cannot be changed."
        />
        <TextInput
          label="E-mail address"
          type="email"
          autoComplete="email"
          required
          value={email}
          onChange={(e) => setEmail(e.currentTarget.value)}
          error={error}
        />
        {!changed && <EmailStatus account={account} />}
        <Group justify="flex-end">
          <Button type="submit" loading={busy} disabled={!changed}>
            Save e-mail address
          </Button>
        </Group>
      </Stack>
    </form>
  );
}

function PasswordForm() {
  const [current, setCurrent] = useState('');
  const [next, setNext] = useState('');
  const [errors, setErrors] = useState<{ current?: string; next?: string; other?: string }>({});
  const [busy, setBusy] = useState(false);

  const save = async (event: React.FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setErrors({});
    try {
      // The other sessions end; this one goes on with the new tokens.
      setTokens(await api.account.changePassword(current, next));
      setCurrent('');
      setNext('');
      notifications.show({
        color: 'green',
        message: 'Your password was changed. You were logged out everywhere else.',
      });
    } catch (err) {
      const fields = {
        current: fieldError(err, 'current_password') ?? undefined,
        next: fieldError(err, 'new_password') ?? undefined,
      };
      setErrors(
        fields.current || fields.next ? fields : { other: (err as Error).message ?? 'Failed.' }
      );
    } finally {
      setBusy(false);
    }
  };

  return (
    <form onSubmit={save}>
      <Stack gap="sm">
        {errors.other && (
          <Alert color="red" icon={<IconAlertCircle size={18} />}>
            {errors.other}
          </Alert>
        )}
        <PasswordInput
          label="Current password"
          autoComplete="current-password"
          required
          value={current}
          onChange={(e) => setCurrent(e.currentTarget.value)}
          error={errors.current}
        />
        <PasswordInput
          label="New password"
          description="At least 8 characters, not too common."
          autoComplete="new-password"
          required
          value={next}
          onChange={(e) => setNext(e.currentTarget.value)}
          error={errors.next}
        />
        <Group justify="space-between" align="center" gap="sm">
          <Text size="xs" c="dimmed" style={{ flex: '1 1 12rem' }}>
            You stay logged in here; other devices are logged out.
          </Text>
          <Button type="submit" loading={busy} disabled={!current || !next}>
            Change password
          </Button>
        </Group>
      </Stack>
    </form>
  );
}

function DeleteDialog({
  account,
  opened,
  onClose,
}: {
  account: Account;
  opened: boolean;
  onClose: () => void;
}) {
  const { logout } = useAuth();
  const navigate = useNavigate();
  const [password, setPassword] = useState('');
  const [username, setUsername] = useState('');
  const [errors, setErrors] = useState<{ password?: string; username?: string; other?: string }>(
    {}
  );
  const [busy, setBusy] = useState(false);
  const confirmed = username === account.username && password !== '';
  const deleted = useRef(false);

  // Forget the login only once this page is gone: logged out on /account, the login page would
  // be shown instead of the home page.
  useEffect(
    () => () => {
      if (deleted.current) {
        logout(false); // the server ended every login already
      }
    },
    [logout]
  );

  const remove = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!confirmed) {
      return;
    }
    setBusy(true);
    setErrors({});
    try {
      await api.account.remove(password, username);
    } catch (err) {
      const fields = {
        password: fieldError(err, 'password') ?? undefined,
        username: fieldError(err, 'username') ?? undefined,
      };
      setErrors(fields.password || fields.username ? fields : { other: (err as Error).message });
      setBusy(false);
      return;
    }
    deleted.current = true;
    notifications.show({
      color: 'green',
      title: 'Your account was deleted',
      message: 'Everything you stored on this server was deleted with it.',
    });
    navigate('/', { replace: true });
  };

  return (
    <Modal
      opened={opened}
      onClose={onClose}
      title="Delete your account?"
      closeOnClickOutside={!busy}
    >
      <form onSubmit={remove}>
        <Stack gap="sm">
          <Text size="sm">
            This deletes your account and everything you stored on this server, for good:
          </Text>
          <List size="sm" spacing={4}>
            <List.Item>
              your networks with their history and comments (people you shared them with lose
              access)
            </List.Item>
            <List.Item>
              your imported tables, data preparations, datasets and trainings with their files
            </List.Item>
            <List.Item>
              your comments, API tokens, connected apps, the log of their activity and usage records
            </List.Item>
          </List>
          <Text size="sm">
            Networks of other people you worked on stay with them. Download your data first if you
            want to keep it.
          </Text>
          {errors.other && (
            <Alert color="red" icon={<IconAlertCircle size={18} />}>
              {errors.other}
            </Alert>
          )}
          <PasswordInput
            label="Your password"
            autoComplete="current-password"
            required
            value={password}
            onChange={(e) => setPassword(e.currentTarget.value)}
            error={errors.password}
            data-autofocus
          />
          <TextInput
            label={`Type your username “${account.username}” to confirm`}
            autoComplete="off"
            autoCapitalize="none"
            spellCheck={false}
            required
            value={username}
            onChange={(e) => setUsername(e.currentTarget.value)}
            error={errors.username}
          />
          <Group justify="flex-end" gap="sm">
            <Button variant="default" onClick={onClose} disabled={busy}>
              Cancel
            </Button>
            <Button
              type="submit"
              color="red"
              loading={busy}
              disabled={!confirmed}
              leftSection={<IconTrash size={16} />}
            >
              Delete my account
            </Button>
          </Group>
        </Stack>
      </form>
    </Modal>
  );
}

/** /account: the own account, its data, and deleting it. */
export function AccountPage() {
  const legal = useLegalInfo();
  const billing = useBillingInfo();
  const [account, setAccount] = useState<Account | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [exporting, setExporting] = useState(false);
  const [deleting, setDeleting] = useState(false);

  useEffect(() => {
    api.account
      .get()
      .then(setAccount)
      .catch((err: Error) => setError(err.message));
  }, []);

  const exportData = async () => {
    setExporting(true);
    try {
      const { blob, filename } = await api.account.exportData();
      downloadBlob(filename ?? 'brick-dnn-data.zip', blob);
    } catch (err) {
      notifications.show({
        color: 'red',
        title: 'Could not export your data',
        message: (err as Error).message,
      });
    } finally {
      setExporting(false);
    }
  };

  if (error) {
    return (
      <Container size="sm" py="xl">
        <Alert color="red" icon={<IconAlertCircle size={18} />}>
          {error}
        </Alert>
      </Container>
    );
  }
  if (!account) {
    return (
      <Center h={300}>
        <Loader />
      </Center>
    );
  }

  const stored = storedSummary(account.stored);
  const uploadDays = legal?.retention.unused_uploads_days ?? 0;
  const accountDays = legal?.retention.inactive_accounts_days ?? 0;

  return (
    <Container size="sm" py="xl">
      <Title order={2}>Account</Title>
      <Text c="dimmed" size="sm" mt={4}>
        Signed up {formatDate(account.date_joined)}
        {account.terms_accepted_at &&
          ` · accepted the terms of use ${formatDate(account.terms_accepted_at)}`}
      </Text>

      <Stack gap="lg" mt="xl">
        <Section title="Profile">
          <EmailForm account={account} onSaved={setAccount} />
        </Section>

        <Section title="Password">
          <PasswordForm />
        </Section>

        <Section title="Two-factor authentication">
          <TwoFactorSection
            username={account.username}
            state={account.two_factor}
            onChange={(twoFactor) => setAccount({ ...account, two_factor: twoFactor })}
          />
        </Section>

        {billing && (
          <Section title="Plan & billing">
            <PlanSection info={billing} />
          </Section>
        )}

        <Section title="Usage & budget">
          <UsageSection />
        </Section>

        <Section title="Your data">
          <Stack gap="sm">
            <Text size="sm">
              {stored.length > 0
                ? 'Besides your username and e-mail address, this server stores:'
                : 'This server stores your username and e-mail address, nothing else yet.'}
            </Text>
            {stored.length > 0 && (
              <List size="sm" spacing={2}>
                {stored.map((line) => (
                  <List.Item key={line}>{line}</List.Item>
                ))}
                {account.stored.upload_bytes > 0 && (
                  <List.Item>
                    {formatBytes(account.stored.upload_bytes)} of uploaded files
                  </List.Item>
                )}
              </List>
            )}
            {(uploadDays > 0 || accountDays > 0) && (
              <Text size="sm" c="dimmed">
                {uploadDays > 0 &&
                  `Uploads and finished trainings nobody has used for ${uploadDays} days are deleted automatically. `}
                {accountDays > 0 &&
                  `Accounts nobody has logged in to for ${accountDays} days are deleted with everything in them.`}
              </Text>
            )}
            <Group justify="space-between" align="center" gap="sm">
              <Text size="xs" c="dimmed" style={{ flex: '1 1 14rem' }}>
                A ZIP file with your account, networks, comments, data preparations, trainings and
                uploaded files.
              </Text>
              <Button
                variant="light"
                leftSection={<IconDownload size={16} />}
                loading={exporting}
                onClick={exportData}
              >
                Download my data
              </Button>
            </Group>
          </Stack>
        </Section>

        <Section title="Delete account" danger>
          <Group justify="space-between" align="center" gap="sm">
            <Text size="sm" style={{ flex: '1 1 14rem' }}>
              Delete your account and everything you stored on this server. This cannot be undone.
            </Text>
            <Button
              color="red"
              variant="light"
              leftSection={<IconTrash size={16} />}
              onClick={() => setDeleting(true)}
            >
              Delete account…
            </Button>
          </Group>
        </Section>
      </Stack>

      <LegalLinks mt="xl" />
      {deleting && (
        <DeleteDialog account={account} opened={deleting} onClose={() => setDeleting(false)} />
      )}
    </Container>
  );
}
