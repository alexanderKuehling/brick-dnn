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
import { currentLocale, i18n, useTranslation } from '@/i18n';
import { LegalLinks } from '@/legal/LegalLinks';
import { useLegalInfo } from '@/legal/useLegalInfo';
import { formatBytes, formatDate } from '@/training/format';
import { UsageSection } from '@/usage/UsageSection';
import { fieldError } from './fieldError';
import { TwoFactorSection } from './TwoFactorSection';

function plural(count: number, word: string, words = `${word}s`): string {
  return `${count.toLocaleString(currentLocale())} ${count === 1 ? word : words}`;
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
  useTranslation();
  const [busy, setBusy] = useState(false);
  const [sent, setSent] = useState(false);

  if (!account.email || !account.email_sending) {
    return null;
  }
  if (account.email_verified) {
    return (
      <Group gap={6}>
        <Badge color="green" variant="light" leftSection={<IconCircleCheck size={12} />}>
          {i18n.t('ui.account.AccountPage.confirmed')}
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
        title: i18n.t('ui.account.AccountPage.couldNotSendTheLink'),
        message: (err as Error).message,
      });
    } finally {
      setBusy(false);
    }
  };

  return (
    <Alert
      color="yellow"
      icon={<IconMailQuestion size={18} />}
      title={i18n.t('ui.account.AccountPage.notConfirmedYet')}
    >
      <Stack gap="xs">
        <Text size="sm">
          {account.password_reset_available
            ? i18n.t('ui.account.AccountPage.openTheLinkWeSent')
            : i18n.t('ui.account.AccountPage.openTheLinkWeSent2')}
        </Text>
        <Group>
          <Button size="xs" variant="light" color="yellow" loading={busy} onClick={resend}>
            {sent
              ? i18n.t('ui.account.AccountPage.sendTheLinkAgain')
              : i18n.t('ui.account.AccountPage.sendMeANewLink')}
          </Button>
        </Group>
      </Stack>
    </Alert>
  );
}

function EmailForm({ account, onSaved }: { account: Account; onSaved: (a: Account) => void }) {
  useTranslation();
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
          ? i18n.t('ui.account.AccountPage.yourEMailAddressWas', { email: updated.email })
          : i18n.t('ui.account.AccountPage.yourEMailAddressWas2'),
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
            {i18n.t('ui.account.AccountPage.yourAccountHasNoE')}
            {account.email_sending && i18n.t('ui.account.AccountPage.andYouCanResetA')}.
          </Alert>
        )}
        <TextInput
          label={i18n.t('ui.account.AccountPage.username')}
          value={account.username}
          readOnly
          description={i18n.t('ui.account.AccountPage.usernamesCannotBeChanged')}
        />
        <TextInput
          label={i18n.t('ui.account.AccountPage.eMailAddress')}
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
            {i18n.t('ui.account.AccountPage.saveEMailAddress')}
          </Button>
        </Group>
      </Stack>
    </form>
  );
}

function PasswordForm() {
  useTranslation();
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
        message: i18n.t('ui.account.AccountPage.yourPasswordWasChangedYou'),
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
          label={i18n.t('ui.account.AccountPage.currentPassword')}
          autoComplete="current-password"
          required
          value={current}
          onChange={(e) => setCurrent(e.currentTarget.value)}
          error={errors.current}
        />
        <PasswordInput
          label={i18n.t('ui.account.AccountPage.newPassword')}
          description={i18n.t('ui.account.AccountPage.atLeast8CharactersNot')}
          autoComplete="new-password"
          required
          value={next}
          onChange={(e) => setNext(e.currentTarget.value)}
          error={errors.next}
        />
        <Group justify="space-between" align="center" gap="sm">
          <Text size="xs" c="dimmed" style={{ flex: '1 1 12rem' }}>
            {i18n.t('ui.account.AccountPage.youStayLoggedInHere')}
          </Text>
          <Button type="submit" loading={busy} disabled={!current || !next}>
            {i18n.t('ui.account.AccountPage.changePassword')}
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
  useTranslation();
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
      title: i18n.t('ui.account.AccountPage.yourAccountWasDeleted'),
      message: i18n.t('ui.account.AccountPage.everythingYouStoredOnThis'),
    });
    navigate('/', { replace: true });
  };

  return (
    <Modal
      opened={opened}
      onClose={onClose}
      title={i18n.t('ui.account.AccountPage.deleteYourAccount')}
      closeOnClickOutside={!busy}
    >
      <form onSubmit={remove}>
        <Stack gap="sm">
          <Text size="sm">{i18n.t('ui.account.AccountPage.thisDeletesYourAccountAnd')}</Text>
          <List size="sm" spacing={4}>
            <List.Item>{i18n.t('ui.account.AccountPage.yourNetworksWithTheirHistory')}</List.Item>
            <List.Item>
              {i18n.t('ui.account.AccountPage.yourImportedTablesDataPreparations')}
            </List.Item>
            <List.Item>{i18n.t('ui.account.AccountPage.yourCommentsApiTokensConnected')}</List.Item>
          </List>
          <Text size="sm">{i18n.t('ui.account.AccountPage.networksOfOtherPeopleYou')}</Text>
          {errors.other && (
            <Alert color="red" icon={<IconAlertCircle size={18} />}>
              {errors.other}
            </Alert>
          )}
          <PasswordInput
            label={i18n.t('ui.account.AccountPage.yourPassword')}
            autoComplete="current-password"
            required
            value={password}
            onChange={(e) => setPassword(e.currentTarget.value)}
            error={errors.password}
            data-autofocus
          />
          <TextInput
            label={i18n.t('ui.account.AccountPage.typeYourUsernameToConfirm', {
              username: account.username,
            })}
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
              {i18n.t('ui.account.AccountPage.cancel')}
            </Button>
            <Button
              type="submit"
              color="red"
              loading={busy}
              disabled={!confirmed}
              leftSection={<IconTrash size={16} />}
            >
              {i18n.t('ui.account.AccountPage.deleteMyAccount')}
            </Button>
          </Group>
        </Stack>
      </form>
    </Modal>
  );
}

/** /account: the own account, its data, and deleting it. */
export function AccountPage() {
  useTranslation();
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
        title: i18n.t('ui.account.AccountPage.couldNotExportYourData'),
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
      <Title order={2}>{i18n.t('ui.account.AccountPage.account')}</Title>
      <Text c="dimmed" size="sm" mt={4}>
        {i18n.t('ui.account.AccountPage.signedUp', {
          date_joined: formatDate(account.date_joined),
          value:
            account.terms_accepted_at &&
            i18n.t('ui.account.AccountPage.acceptedTheTermsOfUse', {
              terms_accepted_at: formatDate(account.terms_accepted_at),
            }),
        })}
      </Text>

      <Stack gap="lg" mt="xl">
        <Section title={i18n.t('ui.account.AccountPage.profile')}>
          <EmailForm account={account} onSaved={setAccount} />
        </Section>

        <Section title={i18n.t('ui.account.AccountPage.password')}>
          <PasswordForm />
        </Section>

        <Section title={i18n.t('ui.account.AccountPage.twoFactorAuthentication')}>
          <TwoFactorSection
            username={account.username}
            state={account.two_factor}
            onChange={(twoFactor) => setAccount({ ...account, two_factor: twoFactor })}
          />
        </Section>

        {billing && (
          <Section title={i18n.t('ui.account.AccountPage.planBilling')}>
            <PlanSection info={billing} />
          </Section>
        )}

        <Section title={i18n.t('ui.account.AccountPage.usageBudget')}>
          <UsageSection />
        </Section>

        <Section title={i18n.t('ui.account.AccountPage.yourData')}>
          <Stack gap="sm">
            <Text size="sm">
              {stored.length > 0
                ? i18n.t('ui.account.AccountPage.besidesYourUsernameAndE')
                : i18n.t('ui.account.AccountPage.thisServerStoresYourUsername')}
            </Text>
            {stored.length > 0 && (
              <List size="sm" spacing={2}>
                {stored.map((line) => (
                  <List.Item key={line}>{line}</List.Item>
                ))}
                {account.stored.upload_bytes > 0 && (
                  <List.Item>
                    {i18n.t('ui.account.AccountPage.ofUploadedFiles', {
                      upload_bytes: formatBytes(account.stored.upload_bytes),
                    })}
                  </List.Item>
                )}
              </List>
            )}
            {(uploadDays > 0 || accountDays > 0) && (
              <Text size="sm" c="dimmed">
                {uploadDays > 0 &&
                  i18n.t('ui.account.AccountPage.uploadsAndFinishedTrainingsNobody', {
                    uploadDays,
                  })}
                {accountDays > 0 &&
                  i18n.t('ui.account.AccountPage.accountsNobodyHasLoggedIn', { accountDays })}
              </Text>
            )}
            <Group justify="space-between" align="center" gap="sm">
              <Text size="xs" c="dimmed" style={{ flex: '1 1 14rem' }}>
                {i18n.t('ui.account.AccountPage.aZipFileWithYour')}
              </Text>
              <Button
                variant="light"
                leftSection={<IconDownload size={16} />}
                loading={exporting}
                onClick={exportData}
              >
                {i18n.t('ui.account.AccountPage.downloadMyData')}
              </Button>
            </Group>
          </Stack>
        </Section>

        <Section title={i18n.t('ui.account.AccountPage.deleteAccount')} danger>
          <Group justify="space-between" align="center" gap="sm">
            <Text size="sm" style={{ flex: '1 1 14rem' }}>
              {i18n.t('ui.account.AccountPage.deleteYourAccountAndEverything')}
            </Text>
            <Button
              color="red"
              variant="light"
              leftSection={<IconTrash size={16} />}
              onClick={() => setDeleting(true)}
            >
              {i18n.t('ui.account.AccountPage.deleteAccount2')}
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
