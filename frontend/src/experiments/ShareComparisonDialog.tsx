import { useEffect, useState } from 'react';
import { IconCheck, IconCopy, IconTrash } from '@tabler/icons-react';
import {
  ActionIcon,
  Alert,
  Button,
  Checkbox,
  CopyButton,
  Group,
  List,
  Modal,
  Stack,
  Text,
  TextInput,
  Tooltip,
} from '@mantine/core';
import { notifications } from '@mantine/notifications';
import { api, type ComparisonOptions, type RunComparison, type RunRow } from '@/api';
import { useTranslation } from '@/i18n';
import { formatDate } from '@/training/format';

export function comparisonUrl(token: string): string {
  return new URL(`/compare/${encodeURIComponent(token)}`, window.location.origin).href;
}

/**
 * Share the selected runs read-only with a link: says what the link shows (and what not),
 * creates it, and lists the links shared before so they can be turned off. A link takes a
 * limited number of runs: `rows` are the ones it gets, `selectedCount` how many are selected.
 */
export function ShareComparisonDialog({
  opened,
  onClose,
  rows,
  selectedCount = rows.length,
  columns,
  options,
}: {
  opened: boolean;
  onClose: () => void;
  rows: RunRow[];
  selectedCount?: number;
  columns: string[];
  options: ComparisonOptions;
}) {
  const { t } = useTranslation();
  const [name, setName] = useState('');
  const [includeNotes, setIncludeNotes] = useState(false);
  const [creating, setCreating] = useState(false);
  const [created, setCreated] = useState<RunComparison | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [existing, setExisting] = useState<RunComparison[] | null>(null);

  useEffect(() => {
    if (!opened) {
      return;
    }
    setCreated(null);
    setError(null);
    api.training.comparisons
      .list()
      .then(setExisting)
      .catch(() => setExisting([]));
  }, [opened]);

  const foreign = rows.filter((row) => row.mine === false);
  const create = async () => {
    setCreating(true);
    setError(null);
    try {
      const comparison = await api.training.comparisons.create({
        name: name.trim() || undefined,
        runs: rows.map((row) => row.id),
        columns,
        options,
        include_notes: includeNotes,
      });
      setCreated(comparison);
      setExisting((list) => [comparison, ...(list ?? [])]);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setCreating(false);
    }
  };
  const remove = async (comparison: RunComparison) => {
    try {
      await api.training.comparisons.remove(comparison.id);
      setExisting((list) => (list ?? []).filter((c) => c.id !== comparison.id));
      if (created?.id === comparison.id) {
        setCreated(null);
      }
      notifications.show({ message: t('experiments.share.turnedOff', { name: comparison.name }) });
    } catch (err) {
      notifications.show({ color: 'red', message: (err as Error).message });
    }
  };

  return (
    <Modal opened={opened} onClose={onClose} title={t('experiments.share.title')} size="lg">
      <Stack gap="sm">
        <Text size="sm">
          {t('experiments.share.intro', { count: rows.length })}
          {selectedCount > rows.length &&
            ` ${t('experiments.share.firstOnly', { shown: rows.length, total: selectedCount })}`}
        </Text>
        <List size="sm" spacing={2}>
          <List.Item>{t('experiments.share.shows')}</List.Item>
          <List.Item>{t('experiments.share.hides')}</List.Item>
        </List>
        {foreign.length > 0 && (
          <Alert color="yellow">
            {t('experiments.share.others', { names: foreign.map((row) => row.name).join(', ') })}
          </Alert>
        )}
        <TextInput
          label={t('experiments.share.name')}
          placeholder={t('experiments.share.namePlaceholder')}
          value={name}
          maxLength={120}
          onChange={(event) => setName(event.currentTarget.value)}
        />
        <Checkbox
          label={t('experiments.share.includeNotes')}
          checked={includeNotes}
          onChange={(event) => setIncludeNotes(event.currentTarget.checked)}
        />
        {error && <Alert color="red">{error}</Alert>}
        {created ? (
          <Group gap="xs" wrap="nowrap">
            <TextInput
              style={{ flex: 1 }}
              aria-label={t('experiments.share.link')}
              value={comparisonUrl(created.token)}
              readOnly
              onFocus={(event) => event.currentTarget.select()}
            />
            <CopyButton value={comparisonUrl(created.token)} timeout={2000}>
              {({ copied, copy }) => (
                <Button
                  onClick={copy}
                  color={copied ? 'teal' : undefined}
                  leftSection={copied ? <IconCheck size={16} /> : <IconCopy size={16} />}
                >
                  {copied ? t('experiments.share.copied') : t('experiments.share.copy')}
                </Button>
              )}
            </CopyButton>
          </Group>
        ) : (
          <Group justify="flex-end">
            <Button onClick={create} loading={creating} disabled={rows.length === 0}>
              {t('experiments.share.create')}
            </Button>
          </Group>
        )}
        {existing && existing.length > 0 && (
          <div>
            <Text fw={600} size="sm" mt="sm" mb={4}>
              {t('experiments.share.existing')}
            </Text>
            <Stack gap={4}>
              {existing.map((comparison) => (
                <Group key={comparison.id} justify="space-between" wrap="nowrap" gap="xs">
                  <div style={{ minWidth: 0 }}>
                    <Text size="sm" truncate>
                      {comparison.name}
                    </Text>
                    <Text size="xs" c="dimmed">
                      {t('experiments.share.existingLine', {
                        count: comparison.runs.length,
                        date: formatDate(comparison.created_at),
                      })}
                    </Text>
                  </div>
                  <Group gap={4} wrap="nowrap">
                    <CopyButton value={comparisonUrl(comparison.token)} timeout={2000}>
                      {({ copied, copy }) => (
                        <Tooltip
                          label={
                            copied ? t('experiments.share.copied') : t('experiments.share.copy')
                          }
                        >
                          <ActionIcon
                            variant="subtle"
                            color={copied ? 'teal' : 'gray'}
                            onClick={copy}
                            aria-label={t('experiments.share.copyNamed', { name: comparison.name })}
                          >
                            {copied ? <IconCheck size={16} /> : <IconCopy size={16} />}
                          </ActionIcon>
                        </Tooltip>
                      )}
                    </CopyButton>
                    <Tooltip label={t('experiments.share.turnOff')}>
                      <ActionIcon
                        variant="subtle"
                        color="red"
                        onClick={() => void remove(comparison)}
                        aria-label={t('experiments.share.turnOffNamed', { name: comparison.name })}
                      >
                        <IconTrash size={16} />
                      </ActionIcon>
                    </Tooltip>
                  </Group>
                </Group>
              ))}
            </Stack>
          </div>
        )}
      </Stack>
    </Modal>
  );
}
