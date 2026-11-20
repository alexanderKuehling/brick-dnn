import { useEffect, useMemo, useRef, useState } from 'react';
import {
  IconAlertTriangle,
  IconExternalLink,
  IconInfoCircle,
  IconSearch,
} from '@tabler/icons-react';
import {
  Alert,
  Anchor,
  Button,
  Group,
  Loader,
  PasswordInput,
  Radio,
  ScrollArea,
  Select,
  Stack,
  Text,
  TextInput,
} from '@mantine/core';
import { api, type HubDatasetFile, type HubDatasetFiles } from '@/api';
import { i18n, Trans, useTranslation } from '@/i18n';
import { formatBytes } from '@/training/format';
import { hubErrorMessage } from './errors';

/** What the import dialog sends: the chosen file and the token, if one was entered. */
export interface HubChoice {
  file: HubDatasetFile | null;
  token: string;
}

export const EXAMPLE_DATASET = 'scikit-learn/iris';

/** "hf://datasets/…" or a huggingface.co dataset page: something for this panel. */
export function isHubDatasetLink(text: string): boolean {
  const value = text.trim();
  return (
    /^hf:\/\/datasets\/\S+/i.test(value) ||
    /^https?:\/\/(www\.)?huggingface\.co\/datasets\/[^/\s]+\/[^/\s]+\/?$/i.test(value) ||
    /^https?:\/\/(www\.)?huggingface\.co\/datasets\/[^/\s]+\/[^/\s]+\/(viewer|tree)\//i.test(value)
  );
}

const ALL = '*';

interface HubImportPanelProps {
  maxUploadMb: number;
  busy: boolean;
  /** A dataset to list right away (e.g. a hf:// link pasted into the link tab). */
  initialDataset?: string;
  onChange: (choice: HubChoice) => void;
}

/**
 * Import a file of a dataset on the Hugging Face Hub: name the dataset (hf://datasets/…,
 * owner/name or its page), optionally enter an access token for a gated or private one, list
 * its CSV, JSON and Parquet files, filter them by config and split and choose one. The import
 * itself is a link import on the server.
 */
export function HubImportPanel({
  maxUploadMb,
  busy,
  initialDataset = '',
  onChange,
}: HubImportPanelProps) {
  useTranslation();
  const [dataset, setDataset] = useState(initialDataset);
  const [token, setToken] = useState('');
  const [listing, setListing] = useState<HubDatasetFiles | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [config, setConfig] = useState(ALL);
  const [split, setSplit] = useState(ALL);
  const [path, setPath] = useState<string | null>(null);
  const request = useRef<AbortController | null>(null);

  useEffect(() => () => request.current?.abort(), []);

  const list = async (name: string = dataset) => {
    if (!name.trim() || loading) {
      return;
    }
    request.current?.abort();
    const controller = new AbortController();
    request.current = controller;
    setLoading(true);
    setError(null);
    setListing(null);
    setPath(null);
    onChange({ file: null, token });
    try {
      const found = await api.data.hubFiles(name.trim(), token.trim() || null, controller.signal);
      if (controller.signal.aborted) {
        return;
      }
      setListing(found);
      setConfig(found.config ?? (found.configs.length > 1 ? found.configs[0] : ALL));
      setSplit(found.split ?? ALL);
    } catch (err) {
      if (!controller.signal.aborted) {
        setError(hubErrorMessage(err));
      }
    } finally {
      if (!controller.signal.aborted) {
        setLoading(false);
      }
    }
  };

  useEffect(() => {
    if (initialDataset) {
      void list(initialDataset);
    }
    // Only for the dataset the panel was opened with (it is mounted with a key per dataset).
  }, []);

  const shown = useMemo(
    () =>
      (listing?.files ?? []).filter(
        (file) =>
          (config === ALL || file.config === config) && (split === ALL || file.split === split)
      ),
    [listing, config, split]
  );

  const choose = (next: string | null) => {
    setPath(next);
    onChange({ file: listing?.files.find((file) => file.path === next) ?? null, token });
  };

  return (
    <Stack gap="xs">
      <Group gap="xs" align="flex-end" wrap="nowrap">
        <TextInput
          style={{ flex: 1 }}
          label={i18n.t('huggingface.import.dataset')}
          placeholder="hf://datasets/owner/name"
          value={dataset}
          onChange={(event) => setDataset(event.currentTarget.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter') {
              event.preventDefault(); // not the import: list the files first
              void list();
            }
          }}
          autoComplete="off"
          spellCheck={false}
          disabled={busy}
          data-autofocus
        />
        <Button
          variant="light"
          leftSection={<IconSearch size={14} />}
          onClick={() => void list()}
          loading={loading}
          disabled={busy || !dataset.trim()}
        >
          {i18n.t('huggingface.import.list')}
        </Button>
      </Group>
      <Text size="xs" c="dimmed">
        {i18n.t('huggingface.import.datasetHelp', { maxUploadMb })}
      </Text>
      <PasswordInput
        label={i18n.t('huggingface.import.token')}
        description={i18n.t('huggingface.import.tokenHelp')}
        placeholder="hf_…"
        value={token}
        onChange={(event) => {
          const value = event.currentTarget.value;
          setToken(value);
          onChange({
            file: listing?.files.find((file) => file.path === path) ?? null,
            token: value,
          });
        }}
        autoComplete="off"
        spellCheck={false}
        disabled={busy}
      />
      {!listing && !loading && (
        <Text size="xs" c="dimmed">
          <Trans
            i18nKey="huggingface.import.example"
            components={[
              <Anchor
                component="button"
                type="button"
                size="xs"
                onClick={() => {
                  setDataset(EXAMPLE_DATASET);
                  void list(EXAMPLE_DATASET);
                }}
                disabled={busy}
              />,
            ]}
          />
        </Text>
      )}
      {loading && (
        <Group gap="xs" role="status">
          <Loader size="xs" />
          <Text size="sm">{i18n.t('huggingface.import.listing')}</Text>
        </Group>
      )}
      {error && (
        <Alert
          color="red"
          icon={<IconAlertTriangle size={18} />}
          title={i18n.t('huggingface.import.listFailed')}
        >
          {error}
        </Alert>
      )}
      {listing && (
        <DatasetFiles
          listing={listing}
          shown={shown}
          config={config}
          split={split}
          path={path}
          busy={busy}
          onConfig={setConfig}
          onSplit={setSplit}
          onChoose={choose}
        />
      )}
      <Text size="xs" c="dimmed">
        {i18n.t('huggingface.import.privacy')}
      </Text>
    </Stack>
  );
}

function filterOptions(values: string[]) {
  return [
    { value: ALL, label: i18n.t('huggingface.import.all') },
    ...values.map((value) => ({ value, label: value })),
  ];
}

/** The files of a listed dataset, filtered by config and split, to choose one. */
function DatasetFiles({
  listing,
  shown,
  config,
  split,
  path,
  busy,
  onConfig,
  onSplit,
  onChoose,
}: {
  listing: HubDatasetFiles;
  shown: HubDatasetFile[];
  config: string;
  split: string;
  path: string | null;
  busy: boolean;
  onConfig: (value: string) => void;
  onSplit: (value: string) => void;
  onChoose: (value: string | null) => void;
}) {
  const notes = [
    listing.gated ? i18n.t('huggingface.import.gated') : null,
    listing.private ? i18n.t('huggingface.import.private') : null,
    listing.converted ? i18n.t('huggingface.import.converted') : null,
    listing.truncated
      ? i18n.t('huggingface.import.truncated', { count: listing.files.length })
      : null,
  ].filter((note): note is string => note !== null);
  return (
    <Stack gap="xs">
      <Group justify="space-between" gap="xs">
        <Text size="sm" fw={600} style={{ overflowWrap: 'anywhere' }}>
          {listing.repo_id}
          {listing.revision !== 'main' ? ` @ ${listing.revision}` : ''}
        </Text>
        <Anchor href={listing.page} target="_blank" rel="noopener noreferrer" size="xs">
          <Group gap={4} wrap="nowrap" component="span">
            {i18n.t('huggingface.import.open')}
            <IconExternalLink size={12} aria-hidden />
          </Group>
        </Anchor>
      </Group>
      {notes.map((note) => (
        <Alert key={note} color="blue" variant="light" icon={<IconInfoCircle size={16} />} p="xs">
          <Text size="xs">{note}</Text>
        </Alert>
      ))}
      {listing.files.length === 0 ? (
        <Text size="sm" c="dimmed">
          {i18n.t('huggingface.import.noFiles')}
        </Text>
      ) : (
        <>
          {(listing.configs.length > 1 || listing.splits.length > 1) && (
            <Group grow gap="xs">
              {listing.configs.length > 1 && (
                <Select
                  label={i18n.t('huggingface.import.config')}
                  data={filterOptions(listing.configs)}
                  value={config}
                  onChange={(value) => value && onConfig(value)}
                  allowDeselect={false}
                  disabled={busy}
                />
              )}
              {listing.splits.length > 1 && (
                <Select
                  label={i18n.t('huggingface.import.split')}
                  data={filterOptions(listing.splits)}
                  value={split}
                  onChange={(value) => value && onSplit(value)}
                  allowDeselect={false}
                  disabled={busy}
                />
              )}
            </Group>
          )}
          <Radio.Group value={path} onChange={onChoose} label={i18n.t('huggingface.import.file')}>
            <ScrollArea.Autosize mah={240} type="auto" offsetScrollbars>
              <Stack gap={8} mt={6}>
                {shown.length === 0 && (
                  <Text size="sm" c="dimmed">
                    {i18n.t('huggingface.import.noMatch')}
                  </Text>
                )}
                {shown.map((file) => (
                  <Radio
                    key={file.path}
                    value={file.path}
                    label={file.path}
                    disabled={busy}
                    description={i18n.t('huggingface.import.fileDetails', {
                      config: file.config,
                      split: file.split ?? i18n.t('huggingface.import.splitNone'),
                      size:
                        file.size === null
                          ? i18n.t('huggingface.import.unknownSize')
                          : formatBytes(file.size),
                    })}
                    styles={{ label: { overflowWrap: 'anywhere' } }}
                  />
                ))}
              </Stack>
            </ScrollArea.Autosize>
          </Radio.Group>
        </>
      )}
    </Stack>
  );
}
