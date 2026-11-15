import { useState } from 'react';
import {
  IconBookmark,
  IconBookmarkPlus,
  IconDots,
  IconEye,
  IconPencil,
  IconRestore,
} from '@tabler/icons-react';
import {
  ActionIcon,
  Alert,
  Badge,
  Button,
  Center,
  Group,
  Loader,
  Menu,
  Stack,
  Text,
  TextInput,
  ThemeIcon,
  UnstyledButton,
  VisuallyHidden,
} from '@mantine/core';
import { notifications } from '@mantine/notifications';
import { api, ApiError, type Project, type ProjectRole, type ProjectVersion } from '@/api';
import { useAuth } from '@/auth/AuthContext';
import type { SubscribeToEvents } from '@/collab/events';
import { fullDateTime, groupByDay, timeInDay } from '@/collab/time';
import { i18n, useTranslation } from '@/i18n';
import { KIND_LABELS, layerCount, versionAuthor, versionTitle } from './labels';
import { useVersions } from './useVersions';
import { VersionPreview } from './VersionPreview';
import classes from './History.module.css';

export interface HistoryPanelProps {
  projectId: number;
  /** The user's role: owner and editors name and restore versions, viewers look. */
  role: ProjectRole | null;
  /** Live messages of the project (see useProjectEvents); without them, the list polls. */
  subscribe?: SubscribeToEvents | null;
  /** The live connection is up (messages arrive); otherwise the list is polled. */
  live?: boolean;
  /** The canvas has changes that are not saved (not part of named or restored versions). */
  unsaved?: boolean;
  /** A version was restored: the project as it is now, for the canvas. */
  onRestored: (project: Project) => void;
}

/** The server's message for the name, or the general one. */
function labelError(error: unknown): string {
  if (error instanceof ApiError) {
    const label = (error.data as { label?: unknown } | null)?.label;
    if (Array.isArray(label) && label.length > 0) {
      return label.map(String).join(' ');
    }
  }
  return (error as Error).message;
}

interface LabelFormProps {
  label: string;
  initial?: string;
  submitLabel: string;
  description?: string;
  onSubmit: (label: string) => Promise<unknown>;
  onCancel: () => void;
}

/** A name for a version, saved with Enter. */
function LabelForm({
  label,
  initial = '',
  submitLabel,
  description,
  onSubmit,
  onCancel,
}: LabelFormProps) {
  useTranslation();
  const [value, setValue] = useState(initial);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  const submit = async () => {
    if (!value.trim()) {
      setError(i18n.t('ui.history.HistoryPanel.giveTheVersionAName'));
      return;
    }
    setSaving(true);
    setError(null);
    try {
      await onSubmit(value.trim());
    } catch (err) {
      setError(labelError(err));
      setSaving(false);
    }
  };

  return (
    <form
      onSubmit={(event) => {
        event.preventDefault();
        void submit();
      }}
    >
      <TextInput
        label={label}
        description={description}
        placeholder={i18n.t('ui.history.HistoryPanel.eGBaseline')}
        value={value}
        onChange={(event) => setValue(event.currentTarget.value)}
        onKeyDown={(event) => {
          if (event.key === 'Escape') {
            onCancel();
          }
        }}
        data-mantine-stop-propagation="true"
        error={error}
        maxLength={120}
        data-autofocus
        autoFocus
      />
      <Group justify="flex-end" gap="xs" mt="xs">
        <Button variant="subtle" color="gray" size="xs" onClick={onCancel}>
          {i18n.t('ui.history.HistoryPanel.cancel')}
        </Button>
        <Button type="submit" size="xs" loading={saving}>
          {submitLabel}
        </Button>
      </Group>
    </form>
  );
}

/** The id of the live editing notice "… replaced the network" (useLiveBuilder). */
const REPLACED_NOTIFICATION = 'live-replaced';

const KIND_ICONS = {
  auto: { icon: IconPencil, color: 'gray' },
  named: { icon: IconBookmark, color: 'blue' },
  restore: { icon: IconRestore, color: 'orange' },
} as const;

/**
 * The history of a saved network, newest first and grouped by day. Everyone with access
 * can preview earlier versions; the owner and editors name, rename and restore them.
 */
export function HistoryPanel({
  projectId,
  role,
  subscribe,
  live = false,
  unsaved = false,
  onRestored,
}: HistoryPanelProps) {
  useTranslation();
  const { user } = useAuth();
  const me = user?.username ?? null;
  const { versions, error, reload, replace } = useVersions(projectId, subscribe, live);
  const canEdit = role === 'owner' || role === 'editor';
  const [naming, setNaming] = useState(false);
  const [renaming, setRenaming] = useState<number | null>(null);
  const [preview, setPreview] = useState<{ entry: ProjectVersion; confirm: boolean } | null>(null);
  const currentVersion = versions?.[0]?.version ?? null;
  const now = new Date();

  const nameCurrent = async (label: string) => {
    replace(await api.projects.versions.create(projectId, label));
    setNaming(false);
    void reload();
  };

  const rename = async (entry: ProjectVersion, label: string) => {
    replace(await api.projects.versions.rename(projectId, entry.id, label));
    setRenaming(null);
  };

  const restore = async (entry: ProjectVersion) => {
    const project = await api.projects.versions.restore(projectId, entry.id);
    onRestored(project);
    setPreview(null);
    void reload();
    // Live, the server also announces the new state as a replacement: one message is enough.
    notifications.hide(REPLACED_NOTIFICATION);
    notifications.show({
      id: REPLACED_NOTIFICATION,
      color: 'green',
      title: i18n.t('ui.history.HistoryPanel.restored', { versionTitle: versionTitle(entry, me) }),
      message: i18n.t('ui.history.HistoryPanel.itIsTheCurrentState'),
    });
  };

  const previewEntry = preview?.entry;

  return (
    <Stack gap="xs">
      {canEdit ? (
        naming ? (
          <LabelForm
            label={i18n.t('ui.history.HistoryPanel.nameTheCurrentVersion')}
            description={
              unsaved
                ? i18n.t('ui.history.HistoryPanel.yourUnsavedChangesAreNot')
                : i18n.t('ui.history.HistoryPanel.namedVersionsAreKeptFor')
            }
            submitLabel="Save name"
            onSubmit={nameCurrent}
            onCancel={() => setNaming(false)}
          />
        ) : (
          <Button
            variant="light"
            leftSection={<IconBookmarkPlus size={16} />}
            onClick={() => setNaming(true)}
            style={{ alignSelf: 'flex-start' }}
          >
            {i18n.t('ui.history.HistoryPanel.nameCurrentVersion')}
          </Button>
        )
      ) : (
        <Text size="sm" c="dimmed">
          {i18n.t('ui.history.HistoryPanel.youCanLookAtEarlier')}
        </Text>
      )}
      <Text size="xs" c="dimmed">
        {i18n.t('ui.history.HistoryPanel.everyChangeIsKeptChanges')}
      </Text>

      {error && (
        <Alert
          color="red"
          variant="light"
          title={i18n.t('ui.history.HistoryPanel.couldNotLoadTheHistory')}
        >
          {error}{' '}
          <Button variant="subtle" size="compact-xs" onClick={() => void reload()}>
            {i18n.t('ui.history.HistoryPanel.tryAgain')}
          </Button>
        </Alert>
      )}
      {!versions && !error && (
        <Center py="lg">
          <Loader size="sm" />
        </Center>
      )}
      {versions?.length === 0 && (
        <Text size="sm" c="dimmed" ta="center" py="lg">
          {i18n.t('ui.history.HistoryPanel.noHistoryYet')}
        </Text>
      )}

      {versions &&
        groupByDay(versions, (entry) => entry.updated_at, now).map((group) => (
          <section key={group.key} aria-label={group.label}>
            <Text component="h3" className={classes.day}>
              {group.label}
            </Text>
            <ul className={classes.list}>
              {group.items.map((entry) => {
                const title = versionTitle(entry, me);
                const author = versionAuthor(entry, me);
                const isCurrent = entry.version === currentVersion;
                const latest = entry === versions[0];
                const { icon: Icon, color } = KIND_ICONS[entry.kind];
                return (
                  <li key={entry.id} className={classes.entry} data-current={latest || undefined}>
                    {renaming === entry.id ? (
                      <div className={classes.rename}>
                        <LabelForm
                          label={
                            entry.kind === 'named'
                              ? i18n.t('ui.history.HistoryPanel.renameVersion')
                              : i18n.t('ui.history.HistoryPanel.nameThisVersion')
                          }
                          initial={entry.kind === 'auto' ? '' : entry.label}
                          submitLabel="Save name"
                          onSubmit={(label) => rename(entry, label)}
                          onCancel={() => setRenaming(null)}
                        />
                      </div>
                    ) : (
                      <>
                        <UnstyledButton
                          className={classes.entryMain}
                          onClick={() => setPreview({ entry, confirm: false })}
                        >
                          <ThemeIcon
                            variant="light"
                            color={color}
                            size="md"
                            radius="xl"
                            title={KIND_LABELS[entry.kind]}
                          >
                            <Icon size={14} />
                          </ThemeIcon>
                          <div className={classes.entryText}>
                            <Group gap={6}>
                              <Text size="sm" fw={600} className={classes.entryTitle}>
                                <VisuallyHidden>{KIND_LABELS[entry.kind]}:</VisuallyHidden> {title}
                              </Text>
                              {latest && (
                                <Badge size="xs" variant="light" tt="none">
                                  {i18n.t('ui.history.HistoryPanel.current')}
                                </Badge>
                              )}
                            </Group>
                            <Text size="xs" c="dimmed" title={fullDateTime(entry.updated_at)}>
                              {[
                                author,
                                timeInDay(entry.updated_at, now),
                                layerCount(entry.num_layers),
                              ]
                                .filter(Boolean)
                                .join(' · ')}
                            </Text>
                          </div>
                        </UnstyledButton>
                        <Menu position="bottom-end" withinPortal>
                          <Menu.Target>
                            <ActionIcon
                              variant="subtle"
                              color="gray"
                              className={classes.entryMenu}
                              aria-label={i18n.t('ui.history.HistoryPanel.actionsFor', { title })}
                            >
                              <IconDots size={16} />
                            </ActionIcon>
                          </Menu.Target>
                          <Menu.Dropdown>
                            <Menu.Item
                              leftSection={<IconEye size={14} />}
                              onClick={() => setPreview({ entry, confirm: false })}
                            >
                              {i18n.t('ui.history.HistoryPanel.preview')}
                            </Menu.Item>
                            {canEdit && (
                              <Menu.Item
                                leftSection={<IconBookmark size={14} />}
                                onClick={() => setRenaming(entry.id)}
                              >
                                {entry.kind === 'named'
                                  ? i18n.t('ui.history.HistoryPanel.rename')
                                  : i18n.t('ui.history.HistoryPanel.nameThisVersion')}
                              </Menu.Item>
                            )}
                            {canEdit && !isCurrent && (
                              <Menu.Item
                                leftSection={<IconRestore size={14} />}
                                onClick={() => setPreview({ entry, confirm: true })}
                              >
                                {i18n.t('ui.history.HistoryPanel.restore')}
                              </Menu.Item>
                            )}
                          </Menu.Dropdown>
                        </Menu>
                      </>
                    )}
                  </li>
                );
              })}
            </ul>
          </section>
        ))}

      {preview && previewEntry && (
        <VersionPreview
          projectId={projectId}
          entry={previewEntry}
          title={versionTitle(previewEntry, me)}
          current={previewEntry.version === currentVersion}
          canRestore={canEdit}
          unsaved={unsaved}
          confirmFirst={preview.confirm}
          onRestore={() => restore(previewEntry)}
          onClose={() => setPreview(null)}
        />
      )}
    </Stack>
  );
}
