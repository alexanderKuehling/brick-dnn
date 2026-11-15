import { IconDots, IconPencil, IconPlus, IconTable, IconTrash } from '@tabler/icons-react';
import { ActionIcon, Group, Menu, Text, TextInput, ThemeIcon, Tooltip } from '@mantine/core';
import { modals } from '@mantine/modals';
import { notifications } from '@mantine/notifications';
import { api, ApiError, type DataSource } from '@/api';
import { i18n, useTranslation } from '@/i18n';
import { formatBytes, formatDate } from '@/training/format';
import { describeReport, tableSize } from './ImportDialog';
import classes from './DataPage.module.css';

interface SourcesPanelProps {
  /** Null while loading. */
  sources: DataSource[] | null;
  /** The main table of the open preparation (marked in the list). */
  currentSourceId: number | null;
  onNewPrep: (source: DataSource) => void;
  onRenamed: (source: DataSource) => void;
  onDeleted: (id: number) => void;
}

/** The user's imported tables with rename and delete. */
export function SourcesPanel({
  sources,
  currentSourceId,
  onNewPrep,
  onRenamed,
  onDeleted,
}: SourcesPanelProps) {
  useTranslation();
  const rename = (source: DataSource) => {
    let value = source.name;
    modals.openConfirmModal({
      title: i18n.t('ui.data.SourcesPanel.renameTable'),
      children: (
        <TextInput
          data-autofocus
          label={i18n.t('ui.data.SourcesPanel.name')}
          defaultValue={source.name}
          maxLength={120}
          onChange={(event) => {
            value = event.currentTarget.value;
          }}
        />
      ),
      labels: {
        confirm: i18n.t('ui.data.SourcesPanel.save'),
        cancel: i18n.t('ui.data.SourcesPanel.cancel'),
      },
      onConfirm: async () => {
        const name = value.trim();
        if (!name || name === source.name) {
          return;
        }
        try {
          onRenamed(await api.data.sources.rename(source.id, name));
        } catch (error) {
          notifications.show({
            color: 'red',
            title: i18n.t('ui.data.SourcesPanel.cannotRenameTheTable'),
            message: (error as Error).message,
          });
        }
      },
    });
  };

  const remove = (source: DataSource) =>
    modals.openConfirmModal({
      title: i18n.t('ui.data.SourcesPanel.delete', { name: source.name }),
      children: <Text size="sm">{i18n.t('ui.data.SourcesPanel.theImportedTableIsRemoved')}</Text>,
      labels: {
        confirm: i18n.t('ui.data.SourcesPanel.delete2'),
        cancel: i18n.t('ui.data.SourcesPanel.cancel'),
      },
      confirmProps: { color: 'red' },
      onConfirm: async () => {
        try {
          await api.data.sources.remove(source.id);
          onDeleted(source.id);
        } catch (error) {
          notifications.show({
            color: 'red',
            title: i18n.t('ui.data.SourcesPanel.cannotDelete', { name: source.name }),
            message:
              error instanceof ApiError && error.status === 404
                ? i18n.t('ui.data.SourcesPanel.itWasDeletedAlready')
                : (error as Error).message,
          });
          if (error instanceof ApiError && error.status === 404) {
            onDeleted(source.id);
          }
        }
      },
    });

  if (sources === null) {
    return null;
  }
  if (sources.length === 0) {
    return (
      <Text size="xs" c="dimmed" px="sm" py={4}>
        {i18n.t('ui.data.SourcesPanel.noTablesYetImportA')}
      </Text>
    );
  }
  return (
    <ul
      aria-label={i18n.t('ui.data.SourcesPanel.importedTables')}
      style={{ listStyle: 'none', margin: 0, padding: 0 }}
    >
      {sources.map((source) => (
        <li key={source.id} className={classes.row}>
          <Tooltip
            label={
              <>
                {source.filename} · {formatBytes(source.size_bytes)}
                <br />
                {describeReport(source.report)}
                <br />
                {i18n.t('ui.data.SourcesPanel.imported')} {formatDate(source.created_at)}
              </>
            }
            multiline
            w={260}
            openDelay={400}
            position="right"
            events={{ hover: true, focus: true, touch: false }}
          >
            {/* Focusable, so the details can be read with the keyboard, too. */}
            <Group
              gap={8}
              wrap="nowrap"
              className={`${classes.rowStatic} mantine-focus-auto`}
              tabIndex={0}
              aria-describedby={`source-details-${source.id}`}
            >
              <ThemeIcon
                variant="light"
                color={source.id === currentSourceId ? 'brick' : 'gray'}
                size="sm"
                aria-hidden
              >
                <IconTable size={13} />
              </ThemeIcon>
              <div style={{ minWidth: 0 }}>
                <Text size="sm" truncate>
                  {source.name}
                </Text>
                <Text size="xs" c="dimmed" truncate>
                  {tableSize(source.n_rows, source.n_columns)} · {formatBytes(source.size_bytes)}
                </Text>
              </div>
              <span id={`source-details-${source.id}`} hidden>
                {i18n.t('ui.data.SourcesPanel.imported2', {
                  filename: source.filename,
                  describeReport: describeReport(source.report),
                  created_at: formatDate(source.created_at),
                })}
              </span>
            </Group>
          </Tooltip>
          <Menu position="bottom-end" withinPortal>
            <Menu.Target>
              <ActionIcon
                variant="subtle"
                color="gray"
                size="sm"
                mr={4}
                aria-label={i18n.t('ui.data.SourcesPanel.actionsFor', { name: source.name })}
              >
                <IconDots size={14} />
              </ActionIcon>
            </Menu.Target>
            <Menu.Dropdown>
              <Menu.Item leftSection={<IconPlus size={14} />} onClick={() => onNewPrep(source)}>
                {i18n.t('ui.data.SourcesPanel.newPreparation')}
              </Menu.Item>
              <Menu.Item leftSection={<IconPencil size={14} />} onClick={() => rename(source)}>
                {i18n.t('ui.data.SourcesPanel.rename')}
              </Menu.Item>
              <Menu.Item
                color="red"
                leftSection={<IconTrash size={14} />}
                onClick={() => remove(source)}
              >
                {i18n.t('ui.data.SourcesPanel.delete2')}
              </Menu.Item>
            </Menu.Dropdown>
          </Menu>
        </li>
      ))}
    </ul>
  );
}
