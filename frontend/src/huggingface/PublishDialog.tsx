import { useEffect, useRef, useState } from 'react';
import {
  IconAlertTriangle,
  IconChevronDown,
  IconChevronRight,
  IconCircleCheck,
  IconExternalLink,
  IconWorld,
} from '@tabler/icons-react';
import {
  Alert,
  Anchor,
  Button,
  Checkbox,
  Code,
  Collapse,
  Group,
  Loader,
  Modal,
  PasswordInput,
  Radio,
  ScrollArea,
  Select,
  Stack,
  Table,
  Text,
  TextInput,
} from '@mantine/core';
import {
  api,
  type HubCapabilities,
  type HubPublishPreview,
  type HubPublishResult,
  type RunSummary,
} from '@/api';
import { i18n, useTranslation } from '@/i18n';
import { formatBytes } from '@/training/format';
import { hubErrorMessage, licenseLabel, validRepoName } from './errors';

interface PublishDialogProps {
  opened: boolean;
  onClose: () => void;
  run: Pick<RunSummary, 'id' | 'name'>;
  hub: HubCapabilities;
}

/**
 * Publish a finished run to the Hugging Face Hub: the user pastes an access token (sent with
 * this one request, never kept), chooses the repository, its visibility and the licence, and
 * sees the files and the model card before anything is sent. Mount it with a new `key` per
 * opening, so the token never outlives the dialog.
 */
export function PublishDialog({ opened, onClose, run, hub }: PublishDialogProps) {
  useTranslation();
  const [preview, setPreview] = useState<HubPublishPreview | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [token, setToken] = useState('');
  const [repo, setRepo] = useState('');
  const [isPrivate, setIsPrivate] = useState(true);
  const [license, setLicense] = useState(hub.default_license);
  const [updateExisting, setUpdateExisting] = useState(false);
  const [cardOpened, setCardOpened] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<HubPublishResult | null>(null);
  /** The first preview fills in the default name; later ones wait until typing pauses. */
  const loaded = useRef(false);

  // What would be uploaded; again for the model card when the name changes.
  const name = repo.trim();
  const shownName = name && validRepoName(name) ? name : '';
  useEffect(() => {
    if (!opened) {
      return undefined;
    }
    const controller = new AbortController();
    const timer = window.setTimeout(
      () => {
        api.training.runs
          .hubPreview(run.id, shownName, controller.signal)
          .then((data) => {
            setPreview(data);
            setPreviewError(null);
            if (!loaded.current) {
              loaded.current = true;
              setRepo((current) => current || data.default_repo);
            }
          })
          .catch((err: unknown) => {
            if (!controller.signal.aborted) {
              setPreviewError((err as Error).message);
            }
          });
      },
      loaded.current ? 400 : 0
    );
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [opened, run.id, shownName]);

  const repoProblem =
    name && !validRepoName(name) ? i18n.t('huggingface.publish.repoInvalid') : null;
  const blocked = !preview || preview.too_large || !preview.can_publish;
  const canSubmit = !busy && !blocked && token.trim() !== '' && name !== '' && !repoProblem;

  const submit = async () => {
    if (!canSubmit) {
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const published = await api.training.runs.publishToHub(run.id, {
        hf_token: token.trim(),
        repo: name,
        private: isPrivate,
        license,
        update_existing: updateExisting,
      });
      setToken('');
      setResult(published);
    } catch (err) {
      setError(hubErrorMessage(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Modal
      opened={opened}
      onClose={onClose}
      title={result ? i18n.t('huggingface.publish.done') : i18n.t('huggingface.publish.title')}
      size="lg"
      closeOnClickOutside={!busy}
      closeButtonProps={{ 'aria-label': i18n.t('huggingface.publish.close') }}
    >
      {result ? (
        <Stack gap="md">
          <Alert color="green" icon={<IconCircleCheck size={18} />}>
            <Text size="sm">
              {i18n.t('huggingface.publish.doneText', {
                repo: result.repo_id,
                count: result.files.length,
              })}
            </Text>
            {!result.created && (
              <Text size="sm" mt={4}>
                {i18n.t('huggingface.publish.updated')}
              </Text>
            )}
          </Alert>
          <Group justify="flex-end" gap="xs">
            <Button
              component="a"
              href={result.url}
              target="_blank"
              rel="noopener noreferrer"
              variant="default"
              leftSection={<IconExternalLink size={16} />}
            >
              {i18n.t('huggingface.publish.open')}
            </Button>
            <Button onClick={onClose} data-autofocus>
              {i18n.t('huggingface.publish.finish')}
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
            <Text size="sm">{i18n.t('huggingface.publish.intro')}</Text>
            <PasswordInput
              label={i18n.t('huggingface.publish.token')}
              description={
                <>
                  {i18n.t('huggingface.publish.tokenHelp')}{' '}
                  <Anchor
                    href={`${hub.url}/settings/tokens`}
                    target="_blank"
                    rel="noopener noreferrer"
                    size="xs"
                  >
                    {i18n.t('huggingface.publish.createToken')}
                  </Anchor>
                </>
              }
              placeholder="hf_…"
              value={token}
              onChange={(event) => setToken(event.currentTarget.value)}
              autoComplete="off"
              spellCheck={false}
              disabled={busy}
              required
              data-autofocus
            />
            <TextInput
              label={i18n.t('huggingface.publish.repo')}
              description={i18n.t('huggingface.publish.repoHelp')}
              value={repo}
              onChange={(event) => setRepo(event.currentTarget.value)}
              error={repoProblem}
              maxLength={200}
              autoComplete="off"
              spellCheck={false}
              disabled={busy}
              required
            />
            <Radio.Group
              label={i18n.t('huggingface.publish.visibility')}
              value={isPrivate ? 'private' : 'public'}
              onChange={(value) => setIsPrivate(value === 'private')}
            >
              <Stack gap={6} mt={6}>
                <Radio
                  value="private"
                  label={i18n.t('huggingface.publish.private')}
                  disabled={busy}
                />
                <Radio
                  value="public"
                  label={
                    <Group gap={4} wrap="nowrap" component="span">
                      <IconWorld size={14} aria-hidden />
                      <span>{i18n.t('huggingface.publish.public')}</span>
                    </Group>
                  }
                  disabled={busy}
                />
              </Stack>
            </Radio.Group>
            <Select
              label={i18n.t('huggingface.publish.license')}
              data={hub.licenses.map((value) => ({ value, label: licenseLabel(value) }))}
              value={license}
              onChange={(value) => value && setLicense(value)}
              allowDeselect={false}
              disabled={busy}
            />
            <Checkbox
              label={i18n.t('huggingface.publish.updateExisting')}
              description={i18n.t('huggingface.publish.updateExistingHelp')}
              checked={updateExisting}
              onChange={(event) => setUpdateExisting(event.currentTarget.checked)}
              disabled={busy}
            />

            {preview ? (
              <FileList preview={preview} />
            ) : previewError ? (
              <Alert
                color="red"
                icon={<IconAlertTriangle size={18} />}
                title={i18n.t('huggingface.publish.previewFailed')}
              >
                {previewError}
              </Alert>
            ) : (
              <Group gap="xs">
                <Loader size="xs" />
                <Text size="sm" c="dimmed">
                  {i18n.t('huggingface.publish.loading')}
                </Text>
              </Group>
            )}
            {preview && (
              <div>
                <Button
                  variant="subtle"
                  color="gray"
                  size="compact-sm"
                  leftSection={
                    cardOpened ? <IconChevronDown size={14} /> : <IconChevronRight size={14} />
                  }
                  onClick={() => setCardOpened((value) => !value)}
                  aria-expanded={cardOpened}
                >
                  {cardOpened
                    ? i18n.t('huggingface.publish.hideCard')
                    : i18n.t('huggingface.publish.showCard')}
                </Button>
                <Collapse expanded={cardOpened}>
                  {/* A region that the keyboard can reach and scroll. */}
                  <ScrollArea.Autosize
                    mah={320}
                    type="auto"
                    mt="xs"
                    viewportProps={{
                      tabIndex: 0,
                      role: 'region',
                      'aria-label': i18n.t('huggingface.publish.cardLabel'),
                    }}
                  >
                    <Code block style={{ whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>
                      {preview.model_card}
                    </Code>
                  </ScrollArea.Autosize>
                </Collapse>
              </div>
            )}
            {preview?.too_large && (
              <Alert color="yellow" icon={<IconAlertTriangle size={18} />}>
                {i18n.t('huggingface.publish.tooLarge', { max: preview.max_upload_mb })}
              </Alert>
            )}
            {preview && !preview.can_publish && (
              <Alert color="yellow" icon={<IconAlertTriangle size={18} />}>
                {i18n.t('huggingface.publish.notAllowed')}
              </Alert>
            )}
            <Text size="xs" c="dimmed">
              {i18n.t('huggingface.publish.usa')}
            </Text>
            {error && (
              <Alert
                color="red"
                icon={<IconAlertTriangle size={18} />}
                title={i18n.t('huggingface.publish.failed')}
              >
                {error}
              </Alert>
            )}
            {busy && (
              <Group gap="xs" role="status">
                <Loader size="xs" />
                <Text size="sm">{i18n.t('huggingface.publish.uploading')}</Text>
              </Group>
            )}
            <Group justify="flex-end" gap="xs">
              <Button variant="default" onClick={onClose} disabled={busy}>
                {i18n.t('huggingface.publish.cancel')}
              </Button>
              <Button type="submit" loading={busy} disabled={!canSubmit}>
                {i18n.t('huggingface.publish.submit')}
              </Button>
            </Group>
          </Stack>
        </form>
      )}
    </Modal>
  );
}

function FileList({ preview }: { preview: HubPublishPreview }) {
  return (
    <div>
      <Text size="sm" fw={600} mb={4}>
        {i18n.t('huggingface.publish.files', { size: formatBytes(preview.size) })}
      </Text>
      <Table withTableBorder={false} verticalSpacing={2} fz="sm">
        <Table.Tbody>
          {preview.files.map((file) => (
            <Table.Tr key={file.path}>
              <Table.Td>
                <Code>{file.path}</Code>
              </Table.Td>
              <Table.Td ta="right" c="dimmed">
                {file.size === null ? '' : formatBytes(file.size)}
              </Table.Td>
            </Table.Tr>
          ))}
        </Table.Tbody>
      </Table>
      <Text size="xs" c="dimmed" mt={4}>
        {i18n.t('huggingface.publish.approximate')}
      </Text>
    </div>
  );
}
