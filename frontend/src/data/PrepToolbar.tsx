import {
  IconBrandPython,
  IconChevronDown,
  IconCloudCheck,
  IconCloudX,
  IconDatabaseExport,
  IconEye,
  IconFileCode,
  IconFileSpreadsheet,
  IconLayoutSidebar,
  IconPlayerPlay,
  IconTable,
  IconUsers,
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
import type { DataSource, PrepDownloadFormat, ProjectRef, ProjectRole } from '@/api';
import { currentLocale, i18n, useTranslation } from '@/i18n';
import { untitled, type SaveStatus } from './useDataPrep';
import classes from './DataPage.module.css';

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
  /** A preparation of someone else, shared with the user in their networks. */
  shared?: SharedPrep | null;
}

export interface SharedPrep {
  owner: string;
  role: Exclude<ProjectRole, 'owner'>;
  projects: ProjectRef[];
}

/** "“Churn” and “Fraud”" */
function projectNames(projects: ProjectRef[]): string {
  const names = projects.map((project) => `“${project.name}”`);
  return names.length > 1
    ? i18n.t('ui.data.PrepToolbar.and', { slice: names.slice(0, -1).join(', '), at: names.at(-1) })
    : names.join('');
}

const DOWNLOADS: {
  format: PrepDownloadFormat;
  label: string;
  description: string;
  icon: React.ReactNode;
}[] = [
  {
    format: 'csv',
    get label() {
      return i18n.t('ui.data.PrepToolbar.preparedDataCsv');
    },
    get description() {
      return i18n.t('ui.data.PrepToolbar.allRowsAfterTheSteps');
    },
    icon: <IconFileSpreadsheet size={16} />,
  },
  {
    format: 'recipe',
    get label() {
      return i18n.t('ui.data.PrepToolbar.recipeJson');
    },
    get description() {
      return i18n.t('ui.data.PrepToolbar.tablesStepsSettingsAndThe');
    },
    icon: <IconFileCode size={16} />,
  },
  {
    format: 'python',
    get label() {
      return i18n.t('ui.data.PrepToolbar.pythonScriptPy');
    },
    get description() {
      return i18n.t('ui.data.PrepToolbar.pandasCodeThatPreparesNew');
    },
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
  useTranslation();
  if (status === 'error') {
    return (
      <Tooltip
        label={i18n.t('ui.data.PrepToolbar.clickToTryAgain', {
          value: error ?? i18n.t('ui.data.PrepToolbar.savingFailed'),
        })}
        multiline
        w={260}
      >
        <Button
          size="compact-xs"
          variant="light"
          color="red"
          leftSection={<IconCloudX size={14} />}
          onClick={onRetry}
          style={{ flexShrink: 0 }}
        >
          {i18n.t('ui.data.PrepToolbar.notSaved')}
        </Button>
      </Tooltip>
    );
  }
  const saving = status !== 'saved';
  const label = saving ? i18n.t('ui.data.PrepToolbar.saving') : i18n.t('ui.data.PrepToolbar.saved');
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
  useTranslation();
  const { name, source, busy, shared = null } = props;
  const downloading = busy === 'csv' || busy === 'recipe' || busy === 'python';
  const readOnly = shared?.role === 'viewer';

  return (
    <>
      <div className={classes.toolbar}>
        <Group gap="sm" wrap="nowrap" className={classes.toolbarStart}>
          {props.onOpenSidebar && (
            <Tooltip label={i18n.t('ui.data.PrepToolbar.preparationsAndTables')}>
              <ActionIcon
                variant="default"
                size="lg"
                onClick={props.onOpenSidebar}
                aria-label={i18n.t('ui.data.PrepToolbar.showPreparationsAndTables')}
              >
                <IconLayoutSidebar size={18} />
              </ActionIcon>
            </Tooltip>
          )}
          <TextInput
            aria-label={i18n.t('ui.data.PrepToolbar.preparationName')}
            value={name}
            readOnly={readOnly}
            variant={readOnly ? 'filled' : 'default'}
            onChange={(event) => props.onNameChange(event.currentTarget.value)}
            onBlur={() => {
              if (!name.trim() && !readOnly) {
                props.onNameChange(untitled());
              }
            }}
            maxLength={120}
            className={classes.nameInput}
            styles={{ input: { fontWeight: 600 } }}
          />
          <Tooltip
            label={
              source
                ? i18n.t('ui.data.PrepToolbar.mainTableRowsColumns', {
                    filename: source.filename,
                    n_rows: source.n_rows.toLocaleString(currentLocale()),
                    n_columns: source.n_columns.toLocaleString(currentLocale()),
                  })
                : i18n.t('ui.data.PrepToolbar.mainTable')
            }
          >
            <Text size="sm" c="dimmed" visibleFrom="md" truncate className={classes.sourceName}>
              <IconTable size={14} aria-hidden />
              <span>
                {props.sourceName}
                {source &&
                  ` · ${source.n_rows.toLocaleString(currentLocale())} × ${source.n_columns.toLocaleString(currentLocale())}`}
              </span>
            </Text>
          </Tooltip>
          {readOnly ? (
            <Badge
              variant="light"
              color="gray"
              tt="none"
              leftSection={<IconEye size={12} />}
              style={{ flexShrink: 0 }}
            >
              <span className={classes.viewOnlyLabel}>
                {i18n.t('ui.data.PrepToolbar.viewOnly')}
              </span>
            </Badge>
          ) : (
            <SaveState
              status={props.saveStatus}
              error={props.saveError}
              onRetry={props.onRetrySave}
            />
          )}
        </Group>

        <Group gap="xs" wrap="nowrap" className={classes.toolbarEnd}>
          {/* Everything that is not training right away: the dataset and the files. */}
          <Menu position="bottom-end" withinPortal>
            <Menu.Target>
              <Button
                variant="default"
                leftSection={<IconDatabaseExport size={16} />}
                rightSection={<IconChevronDown size={14} />}
                loading={downloading || busy === 'export'}
                disabled={busy !== null && !downloading && busy !== 'export'}
                aria-label={i18n.t('ui.data.PrepToolbar.export')}
                px={{ base: 'xs', sm: 'md' }}
              >
                <Box component="span" visibleFrom="sm">
                  {i18n.t('ui.data.PrepToolbar.export')}
                </Box>
              </Button>
            </Menu.Target>
            <Menu.Dropdown maw={340}>
              {!readOnly && (
                <>
                  <Menu.Item
                    leftSection={<IconDatabaseExport size={16} />}
                    onClick={props.onExport}
                  >
                    <Text size="sm">{i18n.t('ui.data.PrepToolbar.createTrainingDataset')}</Text>
                    <Text size="xs" c="dimmed">
                      {i18n.t('ui.data.PrepToolbar.listedUnderYourDataIn')}
                    </Text>
                  </Menu.Item>
                  <Menu.Divider />
                </>
              )}
              <Menu.Label>{i18n.t('ui.data.PrepToolbar.download')}</Menu.Label>
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
            </Menu.Dropdown>
          </Menu>
          {!readOnly && (
            <Tooltip label={i18n.t('ui.data.PrepToolbar.createTheTrainingDatasetAnd')}>
              <Button
                leftSection={<IconPlayerPlay size={16} />}
                onClick={props.onTrain}
                loading={busy === 'train'}
                disabled={busy !== null && busy !== 'train'}
              >
                {/* One inline span: the button label is a flex box, which would drop the space. */}
                <span>
                  {i18n.t('ui.data.PrepToolbar.train')}{' '}
                  <Box component="span" visibleFrom="sm">
                    {i18n.t('ui.data.PrepToolbar.withThisData')}
                  </Box>
                </span>
              </Button>
            </Tooltip>
          )}
        </Group>
      </div>
      {shared && (
        <Group gap={6} wrap="nowrap" align="flex-start" className={classes.sharedNote}>
          <IconUsers
            size={14}
            color="var(--mantine-color-dimmed)"
            style={{ flexShrink: 0, marginTop: 2 }}
            aria-hidden
          />
          <Text size="xs" c="dimmed">
            {i18n.t('ui.data.PrepToolbar.sharedBy')} <b>{shared.owner}</b>
            {shared.projects.length > 0 && (
              <>
                {' '}
                {i18n.t('ui.data.PrepToolbar.in', { projectNames: projectNames(shared.projects) })}
              </>
            )}
            .{' '}
            {readOnly
              ? i18n.t('ui.data.PrepToolbar.youCanLookAtThe')
              : i18n.t('ui.data.PrepToolbar.youCanChangeTheSteps')}
          </Text>
        </Group>
      )}
    </>
  );
}
