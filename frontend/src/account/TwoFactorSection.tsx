import { useState } from 'react';
import {
  IconAlertCircle,
  IconCheck,
  IconCopy,
  IconDownload,
  IconShieldCheck,
  IconShieldOff,
} from '@tabler/icons-react';
import {
  Alert,
  Badge,
  Button,
  Checkbox,
  Code,
  CopyButton,
  Group,
  Modal,
  PasswordInput,
  SimpleGrid,
  Stack,
  Text,
  TextInput,
} from '@mantine/core';
import { notifications } from '@mantine/notifications';
import { api, setTokens, type TwoFactorSetup, type TwoFactorState } from '@/api';
import { downloadText } from '@/builder/download';
import { formatDate } from '@/training/format';
import { fieldError } from './fieldError';
import { QrCode } from './QrCode';

/** The secret in groups of four, easier to type into an app. */
export function groupSecret(secret: string): string {
  return secret.replace(/(.{4})/g, '$1 ').trim();
}

function errorsOf(err: unknown, fields: string[]): Record<string, string | undefined> {
  const found = Object.fromEntries(
    fields.map((field) => [field, fieldError(err, field) ?? undefined])
  );
  if (Object.values(found).some(Boolean)) {
    return found;
  }
  return { other: err instanceof Error ? err.message : 'Failed.' };
}

function RecoveryCodes({ codes, username }: { codes: string[]; username: string }) {
  const text = codes.join('\n');
  return (
    <Stack gap="xs">
      <SimpleGrid cols={2} spacing={4} verticalSpacing={4} aria-label="Recovery codes">
        {codes.map((code) => (
          <Code key={code} style={{ textAlign: 'center', fontSize: 'var(--mantine-font-size-sm)' }}>
            {code}
          </Code>
        ))}
      </SimpleGrid>
      <Group gap="xs" grow>
        <CopyButton value={text} timeout={1500}>
          {({ copied, copy }) => (
            <Button
              variant="light"
              size="xs"
              onClick={copy}
              leftSection={copied ? <IconCheck size={14} /> : <IconCopy size={14} />}
            >
              {copied ? 'Copied' : 'Copy'}
            </Button>
          )}
        </CopyButton>
        <Button
          variant="light"
          size="xs"
          leftSection={<IconDownload size={14} />}
          onClick={() =>
            downloadText(
              `brick-dnn-recovery-codes-${username}.txt`,
              `Brick DNN recovery codes for ${username}\nEach code works once.\n\n${text}\n`
            )
          }
        >
          Download
        </Button>
      </Group>
    </Stack>
  );
}

/** Turning it on: password, then the QR code and a code from the app, then the recovery codes. */
function SetupDialog({
  username,
  onClose,
  onEnabled,
}: {
  username: string;
  onClose: () => void;
  onEnabled: (state: TwoFactorState) => void;
}) {
  const [password, setPassword] = useState('');
  const [setup, setSetup] = useState<TwoFactorSetup | null>(null);
  const [code, setCode] = useState('');
  const [codes, setCodes] = useState<string[] | null>(null);
  const [saved, setSaved] = useState(false);
  const [errors, setErrors] = useState<Record<string, string | undefined>>({});
  const [busy, setBusy] = useState(false);

  const start = async (event: React.FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setErrors({});
    try {
      setSetup(await api.twoFactor.setup(password));
      setPassword('');
    } catch (err) {
      setErrors(errorsOf(err, ['password']));
    } finally {
      setBusy(false);
    }
  };

  const confirm = async (event: React.FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setErrors({});
    try {
      const result = await api.twoFactor.enable(code.trim());
      // Every other login ended; this one goes on with the new tokens.
      setTokens({ access: result.access, refresh: result.refresh });
      setCodes(result.recovery_codes);
      onEnabled(result.state);
    } catch (err) {
      setErrors(errorsOf(err, ['code']));
      setCode('');
    } finally {
      setBusy(false);
    }
  };

  let body: React.ReactNode;
  if (codes) {
    body = (
      <Stack gap="sm">
        <Alert color="green" icon={<IconShieldCheck size={18} />}>
          Two-factor authentication is on. You were logged out on your other devices.
        </Alert>
        <Text size="sm">
          Save these recovery codes somewhere safe, e.g. in your password manager. If you lose your
          phone, each code lets you log in once. They are shown only now.
        </Text>
        <RecoveryCodes codes={codes} username={username} />
        <Checkbox
          label="I have saved my recovery codes"
          checked={saved}
          onChange={(e) => setSaved(e.currentTarget.checked)}
        />
        <Group justify="flex-end">
          <Button onClick={onClose} disabled={!saved}>
            Done
          </Button>
        </Group>
      </Stack>
    );
  } else if (setup) {
    body = (
      <form onSubmit={confirm}>
        <Stack gap="sm">
          <Text size="sm">
            Scan this QR code with an authenticator app (e.g. Google Authenticator, Microsoft
            Authenticator, 1Password, Aegis or FreeOTP):
          </Text>
          <Group justify="center">
            <QrCode
              value={setup.otpauth_uri}
              size={192}
              label="QR code for your authenticator app"
            />
          </Group>
          <Text size="xs" c="dimmed">
            Or type this key into the app (time-based, 6 digits, 30 seconds):
          </Text>
          <Group gap="xs" wrap="nowrap">
            <Code
              data-testid="two-factor-secret"
              style={{ flex: 1, fontSize: 'var(--mantine-font-size-sm)', wordBreak: 'break-all' }}
            >
              {groupSecret(setup.secret)}
            </Code>
            <CopyButton value={setup.secret} timeout={1500}>
              {({ copied, copy }) => (
                <Button variant="subtle" size="xs" onClick={copy}>
                  {copied ? 'Copied' : 'Copy'}
                </Button>
              )}
            </CopyButton>
          </Group>
          {errors.other && (
            <Alert color="red" icon={<IconAlertCircle size={18} />}>
              {errors.other}
            </Alert>
          )}
          <TextInput
            label="Code from the app"
            autoComplete="one-time-code"
            inputMode="numeric"
            maxLength={6}
            required
            data-autofocus
            value={code}
            onChange={(e) => setCode(e.currentTarget.value)}
            error={errors.code}
          />
          <Group justify="flex-end" gap="sm">
            <Button variant="default" onClick={onClose} disabled={busy}>
              Cancel
            </Button>
            <Button type="submit" loading={busy} disabled={code.trim().length !== 6}>
              Turn on
            </Button>
          </Group>
        </Stack>
      </form>
    );
  } else {
    body = (
      <form onSubmit={start}>
        <Stack gap="sm">
          <Text size="sm">
            After your password, logging in will ask for a code from an app on your phone. Enter
            your password to start.
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
            data-autofocus
            value={password}
            onChange={(e) => setPassword(e.currentTarget.value)}
            error={errors.password}
          />
          <Group justify="flex-end" gap="sm">
            <Button variant="default" onClick={onClose} disabled={busy}>
              Cancel
            </Button>
            <Button type="submit" loading={busy} disabled={!password}>
              Continue
            </Button>
          </Group>
        </Stack>
      </form>
    );
  }

  return (
    <Modal
      opened
      onClose={onClose}
      title="Turn on two-factor authentication"
      closeOnClickOutside={false}
      withCloseButton={!codes}
      closeOnEscape={!codes}
    >
      {body}
    </Modal>
  );
}

/** Password and a code: to turn it off, or to get new recovery codes. */
function ConfirmDialog({
  action,
  username,
  onClose,
  onChanged,
}: {
  action: 'disable' | 'codes';
  username: string;
  onClose: () => void;
  onChanged: (state: TwoFactorState) => void;
}) {
  const [password, setPassword] = useState('');
  const [code, setCode] = useState('');
  const [codes, setCodes] = useState<string[] | null>(null);
  const [errors, setErrors] = useState<Record<string, string | undefined>>({});
  const [busy, setBusy] = useState(false);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setErrors({});
    try {
      if (action === 'disable') {
        onChanged(await api.twoFactor.disable(password, code.trim()));
        notifications.show({ color: 'green', message: 'Two-factor authentication is off.' });
        onClose();
      } else {
        const result = await api.twoFactor.recoveryCodes(password, code.trim());
        setCodes(result.recovery_codes);
        onChanged(result.state);
      }
    } catch (err) {
      setErrors(errorsOf(err, ['password', 'code']));
    } finally {
      setBusy(false);
    }
  };

  const title = action === 'disable' ? 'Turn off two-factor authentication?' : 'New recovery codes';
  return (
    <Modal opened onClose={onClose} title={title} closeOnClickOutside={!busy && !codes}>
      {codes ? (
        <Stack gap="sm">
          <Text size="sm">Your old recovery codes no longer work. Save these:</Text>
          <RecoveryCodes codes={codes} username={username} />
          <Group justify="flex-end">
            <Button onClick={onClose}>Done</Button>
          </Group>
        </Stack>
      ) : (
        <form onSubmit={submit}>
          <Stack gap="sm">
            <Text size="sm">
              {action === 'disable'
                ? 'Logging in will need only your password again.'
                : 'This replaces all your recovery codes.'}
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
              data-autofocus
              value={password}
              onChange={(e) => setPassword(e.currentTarget.value)}
              error={errors.password}
            />
            <TextInput
              label="Code from the app or a recovery code"
              autoComplete="one-time-code"
              autoCapitalize="none"
              spellCheck={false}
              required
              value={code}
              onChange={(e) => setCode(e.currentTarget.value)}
              error={errors.code}
            />
            <Group justify="flex-end" gap="sm">
              <Button variant="default" onClick={onClose} disabled={busy}>
                Cancel
              </Button>
              <Button
                type="submit"
                color={action === 'disable' ? 'red' : undefined}
                loading={busy}
                disabled={!password || !code.trim()}
              >
                {action === 'disable' ? 'Turn off' : 'Make new codes'}
              </Button>
            </Group>
          </Stack>
        </form>
      )}
    </Modal>
  );
}

/** The section "Two-factor authentication" of the account page. */
export function TwoFactorSection({
  username,
  state,
  onChange,
}: {
  username: string;
  state: TwoFactorState;
  onChange: (state: TwoFactorState) => void;
}) {
  const [dialog, setDialog] = useState<'setup' | 'disable' | 'codes' | null>(null);
  const close = () => setDialog(null);

  return (
    <Stack gap="sm">
      {state.enabled ? (
        <>
          <Group gap="xs">
            <Badge color="green" leftSection={<IconShieldCheck size={14} />}>
              On
            </Badge>
            <Text size="sm" c="dimmed">
              {state.enabled_at && `since ${formatDate(state.enabled_at)} · `}
              {state.recovery_codes_left} recovery code{state.recovery_codes_left === 1 ? '' : 's'}{' '}
              left
            </Text>
          </Group>
          {state.recovery_codes_left <= 3 && (
            <Alert color="yellow" icon={<IconAlertCircle size={18} />}>
              Few recovery codes are left. Make new ones and keep them safe.
            </Alert>
          )}
          <Text size="sm">
            Logging in asks for a code from your authenticator app after your password. API tokens
            and connected apps work as before.
          </Text>
          <Group justify="flex-end" gap="sm">
            <Button variant="default" onClick={() => setDialog('codes')}>
              New recovery codes…
            </Button>
            <Button
              color="red"
              variant="light"
              leftSection={<IconShieldOff size={16} />}
              onClick={() => setDialog('disable')}
            >
              Turn off…
            </Button>
          </Group>
        </>
      ) : (
        <>
          <Text size="sm">
            Protect your account with a second step: after your password, logging in asks for a code
            from an authenticator app on your phone. Recommended.
          </Text>
          <Group justify="flex-end">
            <Button leftSection={<IconShieldCheck size={16} />} onClick={() => setDialog('setup')}>
              Turn on…
            </Button>
          </Group>
        </>
      )}
      {dialog === 'setup' && (
        <SetupDialog username={username} onClose={close} onEnabled={onChange} />
      )}
      {(dialog === 'disable' || dialog === 'codes') && (
        <ConfirmDialog action={dialog} username={username} onClose={close} onChanged={onChange} />
      )}
    </Stack>
  );
}
