import { useState } from 'react';
import { IconAlertTriangle, IconCheck, IconCopy } from '@tabler/icons-react';
import {
  Alert,
  Button,
  CopyButton,
  Divider,
  Group,
  Modal,
  Radio,
  Select,
  Stack,
  Text,
  TextInput,
  Title,
} from '@mantine/core';
import { notifications } from '@mantine/notifications';
import { api, ApiError, type ApiToken, type ApiTokenExpiry, type NewApiToken } from '@/api';
import { McpSetup } from './McpSetup';
import classes from './Tokens.module.css';

export const NAME_MAX_LENGTH = 100;

const EXPIRY_OPTIONS: { value: string; label: string }[] = [
  { value: '7', label: '7 days' },
  { value: '30', label: '30 days' },
  { value: '90', label: '90 days' },
  { value: '365', label: '1 year' },
  { value: 'never', label: 'Never' },
];

const DEFAULT_EXPIRY = '90';

type Access = 'full' | 'read';
type Field = 'name' | 'read_only' | 'expires_in_days';
type FieldErrors = Partial<Record<Field, string>>;

const FIELDS: readonly Field[] = ['name', 'read_only', 'expires_in_days'];

function expiryDays(value: string): ApiTokenExpiry | null {
  return value === 'never' ? null : (Number(value) as ApiTokenExpiry);
}

/** "Expires on 6 Feb 2027." for the chosen expiry, counted from now. */
function expiryHint(value: string): string {
  const days = expiryDays(value);
  if (days === null) {
    return 'The token works until you revoke it.';
  }
  const date = new Date(Date.now() + days * 24 * 60 * 60 * 1000);
  return `Expires on ${date.toLocaleDateString(undefined, { dateStyle: 'medium' })}.`;
}

/** Splits a 400 answer into messages for the form fields and the rest (if any). */
function splitErrors(error: unknown): { fields: FieldErrors; other: string | null } {
  if (!(error instanceof ApiError) || error.status !== 400 || !error.data) {
    return { fields: {}, other: (error as Error).message };
  }
  const data = error.data as Record<string, unknown>;
  const fields: FieldErrors = {};
  const rest: string[] = [];
  for (const [key, value] of Object.entries(data)) {
    const messages = (Array.isArray(value) ? value : [value]).map(String).join(' ');
    if ((FIELDS as readonly string[]).includes(key)) {
      fields[key as Field] = messages;
    } else {
      rest.push(messages);
    }
  }
  const other =
    rest.length > 0 ? rest.join(' ') : Object.keys(fields).length ? null : error.message;
  return { fields, other };
}

interface CreateTokenModalProps {
  opened: boolean;
  onClose: () => void;
  /** Called with the new token (without the secret) as soon as it exists. */
  onCreated: (token: ApiToken) => void;
}

/**
 * Asks for the name, access and expiry of a new token, then shows the token once, with the
 * setup snippets filled in. Mount it with a new `key` for each token, so the form starts empty.
 */
export function CreateTokenModal({ opened, onClose, onCreated }: CreateTokenModalProps) {
  const [name, setName] = useState('');
  const [access, setAccess] = useState<Access>('full');
  const [expiry, setExpiry] = useState(DEFAULT_EXPIRY);
  const [errors, setErrors] = useState<FieldErrors>({});
  const [busy, setBusy] = useState(false);
  const [created, setCreated] = useState<NewApiToken | null>(null);

  const submit = async () => {
    const trimmed = name.trim();
    if (!trimmed) {
      setErrors({ name: 'Give the token a name.' });
      return;
    }
    setBusy(true);
    setErrors({});
    try {
      const result = await api.apiTokens.create({
        name: trimmed,
        read_only: access === 'read',
        expires_in_days: expiryDays(expiry),
      });
      const { token: _secret, ...token } = result;
      setCreated(result);
      onCreated(token);
    } catch (error) {
      const { fields, other } = splitErrors(error);
      setErrors(fields);
      if (other) {
        notifications.show({ color: 'red', title: 'Could not create the token', message: other });
      }
    } finally {
      setBusy(false);
    }
  };

  return (
    <Modal
      opened={opened}
      onClose={onClose}
      title={created ? 'Token created' : 'Create an API token'}
      // Wider for the token and the setup snippets.
      size={created ? 'xl' : 'lg'}
      // Once the token is shown, only the buttons close it, so it is not lost by a stray click.
      closeOnClickOutside={!created && !busy}
      closeOnEscape={!created}
    >
      {created ? (
        <Stack gap="md">
          <Alert
            color="yellow"
            icon={<IconAlertTriangle size={18} />}
            title="Copy your token now. It will not be shown again."
          >
            Store it somewhere safe, like a password manager. If you lose it, revoke it and create a
            new one.
          </Alert>
          <div className={classes.secret}>
            <Text className={classes.secretValue} data-testid="new-token">
              {created.token}
            </Text>
            <CopyButton value={created.token} timeout={2000}>
              {({ copied, copy }) => (
                <Button
                  onClick={copy}
                  color={copied ? 'teal' : undefined}
                  leftSection={copied ? <IconCheck size={16} /> : <IconCopy size={16} />}
                  className={classes.secretCopy}
                >
                  {copied ? 'Copied' : 'Copy token'}
                </Button>
              )}
            </CopyButton>
          </div>
          <Divider />
          <Title order={4}>Connect an AI assistant with it</Title>
          <McpSetup token={created.token} />
          <Group justify="flex-end">
            <Button variant="default" onClick={onClose}>
              Done
            </Button>
          </Group>
        </Stack>
      ) : (
        <form
          onSubmit={(event) => {
            event.preventDefault();
            void submit();
          }}
        >
          <Stack gap="md">
            <TextInput
              data-autofocus
              label="Name"
              description="So that you can recognise the token later."
              placeholder="e.g. Claude on my laptop"
              required
              maxLength={NAME_MAX_LENGTH}
              value={name}
              onChange={(event) => setName(event.currentTarget.value)}
              error={errors.name}
              disabled={busy}
            />
            <Radio.Group
              label="Access"
              value={access}
              onChange={(value) => setAccess(value as Access)}
              error={errors.read_only}
            >
              <Stack gap="sm" mt={6}>
                <Radio
                  value="full"
                  label="Full access"
                  description="Read and change your networks, start and stop trainings."
                  disabled={busy}
                />
                <Radio
                  value="read"
                  label="Read only"
                  description="Read your networks and training runs, analyse networks and generate code. Changes nothing."
                  disabled={busy}
                />
              </Stack>
            </Radio.Group>
            <Select
              label="Expires after"
              data={EXPIRY_OPTIONS}
              value={expiry}
              onChange={(value) => setExpiry(value ?? DEFAULT_EXPIRY)}
              allowDeselect={false}
              description={expiryHint(expiry)}
              inputWrapperOrder={['label', 'input', 'description', 'error']}
              error={errors.expires_in_days}
              disabled={busy}
            />
            <Group justify="flex-end" mt="xs">
              <Button variant="default" onClick={onClose} disabled={busy}>
                Cancel
              </Button>
              <Button type="submit" loading={busy}>
                Create token
              </Button>
            </Group>
          </Stack>
        </form>
      )}
    </Modal>
  );
}
