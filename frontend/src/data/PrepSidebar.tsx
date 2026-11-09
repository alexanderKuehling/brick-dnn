import { useId } from 'react';
import {
  IconCopy,
  IconDots,
  IconEye,
  IconFileImport,
  IconPencil,
  IconPlus,
  IconTable,
  IconTrash,
  IconUsersPlus,
} from '@tabler/icons-react';
import { Link } from 'react-router-dom';
import {
  ActionIcon,
  Alert,
  Button,
  Group,
  Loader,
  Menu,
  ScrollArea,
  Text,
  Tooltip,
} from '@mantine/core';
import type { DataPrep, DataSource } from '@/api';
import { ROLE_LABELS } from '@/sharing/roles';
import { prepRole } from './prepAccess';
import { SourcesPanel } from './SourcesPanel';
import classes from './DataPage.module.css';

interface PrepSidebarProps {
  /** Null while loading. */
  preps: DataPrep[] | null;
  sources: DataSource[] | null;
  error: string | null;
  onRetry: () => void;
  currentId: number | null;
  /** Live name of the open preparation (it is saved with a delay). */
  currentName: string | null;
  currentSourceId: number | null;
  /** Called when a preparation is opened (e.g. to close the drawer). */
  onNavigate?: () => void;
  onImport: () => void;
  onNewPrep: (source: DataSource) => void;
  onRenamePrep: (prep: DataPrep) => void;
  onDuplicatePrep: (prep: DataPrep) => void;
  onDeletePrep: (prep: DataPrep) => void;
  /** Share one of the user's preparations in their networks. */
  onSharePrep?: (prep: DataPrep) => void;
  onSourceRenamed: (source: DataSource) => void;
  onSourceDeleted: (id: number) => void;
}

function stepCount(count: number): string {
  return count === 0 ? 'no steps' : count === 1 ? '1 step' : `${count} steps`;
}

/** Preparations and imported tables of the user, and preparations shared with them. */
export function PrepSidebar(props: PrepSidebarProps) {
  const { preps, sources, error } = props;
  const own = preps?.filter((prep) => prepRole(prep) === 'owner') ?? null;
  const shared = preps?.filter((prep) => prepRole(prep) !== 'owner') ?? [];
  const sharedHeading = useId();

  return (
    <>
      <Group p="sm" pb={4} wrap="nowrap">
        <Button fullWidth leftSection={<IconFileImport size={16} />} onClick={props.onImport}>
          Import data
        </Button>
      </Group>
      <ScrollArea style={{ flex: 1, minHeight: 0 }} type="auto">
        <nav aria-label="Preparations">
          <Group justify="space-between" wrap="nowrap" className={classes.sectionTitle}>
            <Text size="xs" fw={700} c="dimmed" tt="uppercase">
              Preparations
            </Text>
            <NewPrepMenu sources={sources} onNewPrep={props.onNewPrep} onImport={props.onImport} />
          </Group>
          {error && (
            <Alert color="red" mx="sm" p="xs">
              <Text size="xs">{error}</Text>
              <Button size="compact-xs" variant="light" color="red" mt={6} onClick={props.onRetry}>
                Try again
              </Button>
            </Alert>
          )}
          {preps === null && !error && <Loader size="sm" mx="sm" my="xs" />}
          {own?.length === 0 && (
            <Text size="xs" c="dimmed" px="sm" py={4}>
              No preparations yet.
            </Text>
          )}
          {own && own.length > 0 && <PrepList {...props} preps={own} />}
          {shared.length > 0 && (
            <>
              <Text
                size="xs"
                fw={700}
                c="dimmed"
                tt="uppercase"
                className={classes.sectionTitle}
                id={sharedHeading}
              >
                Shared with you
              </Text>
              <PrepList {...props} preps={shared} labelledBy={sharedHeading} />
            </>
          )}
        </nav>

        <section aria-label="Imported tables">
          <Text size="xs" fw={700} c="dimmed" tt="uppercase" className={classes.sectionTitle}>
            Imported tables
          </Text>
          <SourcesPanel
            sources={sources}
            currentSourceId={props.currentSourceId}
            onNewPrep={props.onNewPrep}
            onRenamed={props.onSourceRenamed}
            onDeleted={props.onSourceDeleted}
          />
        </section>
      </ScrollArea>
    </>
  );
}

/** Preparations with their actions (as far as the user's role allows them). */
function PrepList({
  preps,
  currentId,
  currentName,
  labelledBy,
  ...props
}: PrepSidebarProps & { preps: DataPrep[]; labelledBy?: string }) {
  return (
    <ul style={{ listStyle: 'none', margin: 0, padding: 0 }} aria-labelledby={labelledBy}>
      {preps.map((prep) => {
        const current = prep.id === currentId;
        const name = (current && currentName?.trim()) || prep.name;
        const role = prepRole(prep);
        return (
          <li key={prep.id} className={classes.row} data-active={current || undefined}>
            <Link
              to={`/data/${prep.id}`}
              className={classes.rowLink}
              aria-current={current ? 'page' : undefined}
              onClick={props.onNavigate}
            >
              <Text size="sm" fw={current ? 600 : 500} truncate>
                {name}
              </Text>
              <Text size="xs" c="dimmed" truncate>
                {role === 'owner' ? (
                  <>
                    {prep.source_name} · {stepCount(prep.steps.length)}
                    {prep.datasets.length > 0 && ' · exported'}
                    {prep.shared_in.length > 0 && ' · shared'}
                  </>
                ) : (
                  <>
                    by {prep.owner} · {ROLE_LABELS[role].toLowerCase()} ·{' '}
                    {stepCount(prep.steps.length)}
                  </>
                )}
              </Text>
            </Link>
            {role === 'viewer' ? (
              <Tooltip label={`Shared by ${prep.owner}: you can view it`}>
                <span className={classes.rowIcon}>
                  <IconEye size={14} color="var(--mantine-color-dimmed)" aria-hidden />
                </span>
              </Tooltip>
            ) : (
              <Menu position="bottom-end" withinPortal>
                <Menu.Target>
                  <ActionIcon
                    variant="subtle"
                    color="gray"
                    size="sm"
                    mr={4}
                    aria-label={`Actions for ${name}`}
                  >
                    <IconDots size={14} />
                  </ActionIcon>
                </Menu.Target>
                <Menu.Dropdown>
                  <Menu.Item
                    leftSection={<IconPencil size={14} />}
                    onClick={() => props.onRenamePrep(prep)}
                  >
                    Rename
                  </Menu.Item>
                  {role === 'owner' && (
                    <>
                      <Menu.Item
                        leftSection={<IconCopy size={14} />}
                        onClick={() => props.onDuplicatePrep(prep)}
                      >
                        Duplicate
                      </Menu.Item>
                      {props.onSharePrep && (
                        <Menu.Item
                          leftSection={<IconUsersPlus size={14} />}
                          onClick={() => props.onSharePrep?.(prep)}
                        >
                          Share in network…
                        </Menu.Item>
                      )}
                      <Menu.Item
                        color="red"
                        leftSection={<IconTrash size={14} />}
                        onClick={() => props.onDeletePrep(prep)}
                      >
                        Delete
                      </Menu.Item>
                    </>
                  )}
                </Menu.Dropdown>
              </Menu>
            )}
          </li>
        );
      })}
    </ul>
  );
}

/** "+" menu: a new preparation from one of the tables, or from a new file. */
function NewPrepMenu({
  sources,
  onNewPrep,
  onImport,
}: Pick<PrepSidebarProps, 'sources' | 'onNewPrep' | 'onImport'>) {
  if (!sources || sources.length === 0) {
    return (
      <Tooltip label="New preparation">
        <ActionIcon
          variant="subtle"
          color="gray"
          size="sm"
          onClick={onImport}
          aria-label="New preparation"
          disabled={sources === null}
        >
          <IconPlus size={14} />
        </ActionIcon>
      </Tooltip>
    );
  }
  return (
    <Menu position="bottom-end" withinPortal>
      <Tooltip label="New preparation">
        <Menu.Target>
          <ActionIcon variant="subtle" color="gray" size="sm" aria-label="New preparation">
            <IconPlus size={14} />
          </ActionIcon>
        </Menu.Target>
      </Tooltip>
      <Menu.Dropdown maw={280}>
        <Menu.Label>New preparation from</Menu.Label>
        {sources.map((source) => (
          <Menu.Item
            key={source.id}
            leftSection={<IconTable size={14} />}
            onClick={() => onNewPrep(source)}
          >
            <Text size="sm" truncate>
              {source.name}
            </Text>
          </Menu.Item>
        ))}
        <Menu.Divider />
        <Menu.Item leftSection={<IconFileImport size={14} />} onClick={onImport}>
          A new file or link…
        </Menu.Item>
      </Menu.Dropdown>
    </Menu>
  );
}
