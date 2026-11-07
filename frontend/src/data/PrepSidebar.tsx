import {
  IconCopy,
  IconDots,
  IconFileImport,
  IconPencil,
  IconPlus,
  IconTable,
  IconTrash,
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
  onSourceRenamed: (source: DataSource) => void;
  onSourceDeleted: (id: number) => void;
}

function stepCount(count: number): string {
  return count === 0 ? 'no steps' : count === 1 ? '1 step' : `${count} steps`;
}

/** Preparations and imported tables of the user. */
export function PrepSidebar(props: PrepSidebarProps) {
  const { preps, sources, error, currentId, currentName } = props;

  return (
    <>
      <Group p="sm" pb={4} wrap="nowrap">
        <Button fullWidth leftSection={<IconFileImport size={16} />} onClick={props.onImport}>
          Import CSV file
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
          {preps?.length === 0 && (
            <Text size="xs" c="dimmed" px="sm" py={4}>
              No preparations yet.
            </Text>
          )}
          {preps && preps.length > 0 && (
            <ul style={{ listStyle: 'none', margin: 0, padding: 0 }}>
              {preps.map((prep) => {
                const current = prep.id === currentId;
                const name = (current && currentName?.trim()) || prep.name;
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
                        {prep.source_name} · {stepCount(prep.steps.length)}
                        {prep.datasets.length > 0 && ' · exported'}
                      </Text>
                    </Link>
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
                        <Menu.Item
                          leftSection={<IconCopy size={14} />}
                          onClick={() => props.onDuplicatePrep(prep)}
                        >
                          Duplicate
                        </Menu.Item>
                        <Menu.Item
                          color="red"
                          leftSection={<IconTrash size={14} />}
                          onClick={() => props.onDeletePrep(prep)}
                        >
                          Delete
                        </Menu.Item>
                      </Menu.Dropdown>
                    </Menu>
                  </li>
                );
              })}
            </ul>
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
          A new CSV file…
        </Menu.Item>
      </Menu.Dropdown>
    </Menu>
  );
}
