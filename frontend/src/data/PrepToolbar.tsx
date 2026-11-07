import {
  IconBrandPython,
  IconChevronDown,
  IconCloudCheck,
  IconCloudX,
  IconDatabaseExport,
  IconDownload,
  IconFileCode,
  IconFileSpreadsheet,
  IconLayoutSidebar,
  IconPlayerPlay,
  IconTable,
} from '@tabler/icons-react';
import {
  ActionIcon,
  Badge,
  Box,
  Button,
  Group,
  Loader,
  Menu,
  Text,
  TextInput,
  Tooltip,
} from '@mantine/core';
import type { DataSource, PrepDownloadFormat } from '@/api';
import { UNTITLED, type SaveStatus } from './useDataPrep';

export type ToolbarAction = 'export' | 'train' | PrepDownloadFormat;

interface PrepToolbarProps {
  name: string;
  onNameChange: (name: string) => void;
  /** The main table; null while the tables are loading. */
  source: DataSource | null;
  sourceName: string;
  saveStatus: SaveStatus;
  saveError: string | null;
  onRetrySave: () => void;
  /** The action that is running, if any. */
  busy: ToolbarAction | null;
  onDownload: (format: PrepDownloadFormat) => void;
  onExport: () => void;
  onTrain: () => void;
  /** Opens the sidebar; null when the sidebar is always visible. */
  onOpenSidebar: (() => void) | null;
}

const DOWNLOADS: {
  format: PrepDownloadFormat;
  label: string;
  description: string;
  icon: React.ReactNode;
}[] = [
  {
    format: 'csv',
    label: 'Prepared data (.csv)',
    description: 'All rows after the steps, with a split column',
    icon: <IconFileSpreadsheet size={16} />,
  },
  {
    format: 'recipe',
    label: 'Recipe (.json)',
    description: 'Tables, steps, settings and the learned values',
    icon: <IconFileCode size={16} />,
  },
  {
    format: 'python',
    label: 'Python script (.py)',
    description: 'pandas code that prepares new data the same way',
    icon: <IconBrandPython size={16} />,
  },
];

function SaveState({
  status,
  error,
  onRetry,
}: {
  status: SaveStatus;
  error: string | null;
  onRetry: () => void;
}) {
  if (status === 'error') {
    return (
      <Tooltip label={`${error ?? 'Saving failed.'} Click to try again.`} multiline w={260}>
        <Button
          size="compact-xs"
          variant="light"
          color="red"
          leftSection={<IconCloudX size={14} />}
          onClick={onRetry}
          style={{ flexShrink: 0 }}
        >
          Not saved
        </Button>
      </Tooltip>
    );
  }
  const saving = status !== 'saved';
  const label = saving ? 'Saving…' : 'Saved';
  // Below `sm` only the icon is shown: it carries the label for screen readers.
  return (
    <Group gap={6} wrap="nowrap" style={{ flexShrink: 0 }} role="img" aria-label={label}>
      {saving ? (
        <Loader size={12} color="gray" aria-hidden />
      ) : (
        <IconCloudCheck size={16} color="var(--mantine-color-dimmed)" aria-hidden />
      )}
      <Text size="xs" c="dimmed" visibleFrom="sm" aria-hidden>
        {label}
      </Text>
    </Group>
  );
}

/** Name, main table, save state, downloads and the export actions of a preparation. */
export function PrepToolbar(props: PrepToolbarProps) {
  const { name, source, busy } = props;
  const downloading = busy === 'csv' || busy === 'recipe' || busy === 'python';

  return (
    <Group
      justify="space-between"
      wrap="wrap"
      px="sm"
      py={6}
      gap="xs"
      style={{ borderBottom: '1px solid var(--mantine-color-default-border)' }}
    >
      <Group gap="xs" wrap="nowrap" style={{ minWidth: 0, flex: '1 1 auto' }}>
        {props.onOpenSidebar && (
          <Tooltip label="Preparations and tables">
            <ActionIcon
              variant="default"
              size="lg"
              onClick={props.onOpenSidebar}
              aria-label="Show preparations and tables"
            >
              <IconLayoutSidebar size={18} />
            </ActionIcon>
          </Tooltip>
        )}
        <TextInput
          aria-label="Preparation name"
          value={name}
          onChange={(event) => props.onNameChange(event.currentTarget.value)}
          onBlur={() => {
            if (!name.trim()) {
              props.onNameChange(UNTITLED);
            }
          }}
          maxLength={120}
          w={{ base: 140, sm: 220 }}
          styles={{ input: { fontWeight: 600 } }}
        />
        <Tooltip
          label={
            source
              ? `Main table: ${source.filename}, ${source.n_rows.toLocaleString('en-US')} rows × ${source.n_columns.toLocaleString('en-US')} columns`
              : 'Main table'
          }
        >
          <Badge
            variant="light"
            color="gray"
            tt="none"
            size="lg"
            radius="sm"
            fw={500}
            leftSection={<IconTable size={14} />}
            visibleFrom="md"
            style={{ flexShrink: 1, minWidth: 0 }}
          >
            {props.sourceName}
            {source &&
              ` · ${source.n_rows.toLocaleString('en-US')} × ${source.n_columns.toLocaleString('en-US')}`}
          </Badge>
        </Tooltip>
        <SaveState status={props.saveStatus} error={props.saveError} onRetry={props.onRetrySave} />
      </Group>

      <Group gap="xs" wrap="nowrap" ml="auto">
        <Menu position="bottom-end" withinPortal>
          <Menu.Target>
            <Button
              variant="default"
              leftSection={<IconDownload size={16} />}
              rightSection={<IconChevronDown size={14} />}
              loading={downloading}
              disabled={busy !== null && !downloading}
              aria-label="Download"
              px={{ base: 'xs', sm: 'md' }}
            >
              <Box component="span" visibleFrom="sm">
                Download
              </Box>
            </Button>
          </Menu.Target>
          <Menu.Dropdown maw={320}>
            {DOWNLOADS.map((item) => (
              <Menu.Item
                key={item.format}
                leftSection={item.icon}
                onClick={() => props.onDownload(item.format)}
              >
                <Text size="sm">{item.label}</Text>
                <Text size="xs" c="dimmed">
                  {item.description}
                </Text>
              </Menu.Item>
            ))}
            <Menu.Divider hiddenFrom="md" />
            <Menu.Item
              hiddenFrom="md"
              leftSection={<IconDatabaseExport size={16} />}
              onClick={props.onExport}
            >
              <Text size="sm">Create training dataset</Text>
              <Text size="xs" c="dimmed">
                Listed under “Your data” in the training panel
              </Text>
            </Menu.Item>
          </Menu.Dropdown>
        </Menu>
        <Tooltip label="Save the prepared data as a training dataset. It is listed under “Your data” in the training panel.">
          <Button
            variant="default"
            leftSection={<IconDatabaseExport size={16} />}
            onClick={props.onExport}
            loading={busy === 'export'}
            disabled={busy !== null && busy !== 'export'}
            visibleFrom="md"
          >
            Create training dataset
          </Button>
        </Tooltip>
        <Tooltip label="Create the training dataset and open the training panel with it">
          <Button
            leftSection={<IconPlayerPlay size={16} />}
            onClick={props.onTrain}
            loading={busy === 'train'}
            disabled={busy !== null && busy !== 'train'}
          >
            {/* One inline span: the button label is a flex box, which would drop the space. */}
            <span>
              Train{' '}
              <Box component="span" visibleFrom="sm">
                with this data
              </Box>
            </span>
          </Button>
        </Tooltip>
      </Group>
    </Group>
  );
}
