import { useCallback, useEffect, useState } from 'react';
import {
  IconCopyPlus,
  IconDots,
  IconEdit,
  IconEye,
  IconFolder,
  IconFolderOpen,
  IconLogout,
  IconPencil,
  IconPlus,
  IconTrash,
  IconUsers,
  IconUsersPlus,
} from '@tabler/icons-react';
import { useSearchParams } from 'react-router-dom';
import {
  ActionIcon,
  Alert,
  Badge,
  Button,
  Card,
  Center,
  Container,
  Group,
  Loader,
  Menu,
  SimpleGrid,
  Tabs,
  Text,
  TextInput,
  Title,
} from '@mantine/core';
import { modals } from '@mantine/modals';
import { notifications } from '@mantine/notifications';
import { api, type Project } from '@/api';
import { useAuth } from '@/auth/AuthContext';
import { projectDraft } from '@/builder/draft';
import { useOpenInBuilder } from '@/builder/useOpenInBuilder';
import { leaveProject, ROLE_LABELS } from '@/sharing/roles';
import { ShareDialog } from '@/sharing/ShareDialog';

type Tab = 'mine' | 'shared';

function formatDate(iso: string): string {
  return new Date(iso).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });
}

export function ProjectsPage() {
  const { user } = useAuth();
  const openInBuilder = useOpenInBuilder();
  const [searchParams, setSearchParams] = useSearchParams();
  const tab: Tab = searchParams.get('tab') === 'shared' ? 'shared' : 'mine';
  const [projects, setProjects] = useState<Project[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sharing, setSharing] = useState<Project | null>(null);

  const load = useCallback(() => {
    api.projects
      .list()
      .then(setProjects)
      .catch((err: Error) => setError(err.message));
  }, []);

  useEffect(load, [load]);

  const replaceProject = (updated: Partial<Project> & { id: number }) =>
    setProjects(
      (list) => list?.map((p) => (p.id === updated.id ? { ...p, ...updated } : p)) ?? null
    );
  const dropProject = (id: number) =>
    setProjects((list) => list?.filter((p) => p.id !== id) ?? null);
  const failed = (title: string, err: unknown) =>
    notifications.show({ color: 'red', title, message: (err as Error).message });

  const open = (project: Project) => openInBuilder(projectDraft(project));

  const createNew = () =>
    openInBuilder({
      doc: { version: 1, nodes: [], edges: [] },
      name: 'Untitled network',
      projectId: null,
    });

  const rename = (project: Project) => {
    let value = project.name;
    modals.openConfirmModal({
      title: 'Rename network',
      children: (
        <TextInput
          data-autofocus
          label="Name"
          defaultValue={project.name}
          onChange={(e) => {
            value = e.currentTarget.value;
          }}
        />
      ),
      labels: { confirm: 'Save', cancel: 'Cancel' },
      onConfirm: async () => {
        try {
          replaceProject(
            await api.projects.update(project.id, { name: value.trim() || project.name })
          );
        } catch (err) {
          failed('Could not rename the network', err);
        }
      },
    });
  };

  const duplicate = async (project: Project) => {
    try {
      const copy = await api.projects.duplicate(project.id);
      setProjects((list) => [copy, ...(list ?? [])]);
      notifications.show({
        color: 'green',
        message:
          tab === 'mine'
            ? `Created “${copy.name}”.`
            : `Saved “${copy.name}” to your own networks (“Mine”).`,
      });
    } catch (err) {
      failed('Could not duplicate the network', err);
    }
  };

  const remove = (project: Project) =>
    modals.openConfirmModal({
      title: `Delete "${project.name}"?`,
      children: (
        <Text size="sm">
          {project.shared && 'Everyone you shared it with loses access too. '}This cannot be undone.
        </Text>
      ),
      labels: { confirm: 'Delete', cancel: 'Cancel' },
      confirmProps: { color: 'red' },
      onConfirm: async () => {
        try {
          await api.projects.remove(project.id);
          dropProject(project.id);
        } catch (err) {
          failed('Could not delete the network', err);
        }
      },
    });

  const leave = (project: Project) =>
    modals.openConfirmModal({
      title: `Leave “${project.name}”?`,
      children: (
        <Text size="sm">
          You lose access to this network. Only {project.owner.username} can invite you again.
        </Text>
      ),
      labels: { confirm: 'Leave', cancel: 'Cancel' },
      confirmProps: { color: 'red' },
      onConfirm: async () => {
        if (!user) {
          return;
        }
        try {
          await leaveProject(project.id, user.id);
          dropProject(project.id);
          notifications.show({ message: `You left “${project.name}”.` });
        } catch (err) {
          failed('Could not leave the network', err);
        }
      },
    });

  const lists: Record<Tab, Project[] | null> = {
    mine: projects?.filter((p) => p.role === 'owner') ?? null,
    shared: projects?.filter((p) => p.role !== 'owner') ?? null,
  };

  const panel = (value: Tab) => {
    const list = lists[value];
    return (
      <Tabs.Panel value={value} pt="lg">
        {list?.length === 0 && (
          <Card withBorder p="xl" ta="center">
            <Text c="dimmed">
              {value === 'mine'
                ? 'You have not saved any networks yet. Build one in the editor and press “Save”.'
                : 'Nobody has shared a network with you yet. When someone invites you, it appears here.'}
            </Text>
          </Card>
        )}
        <SimpleGrid cols={{ base: 1, sm: 2, md: 3 }}>
          {list?.map((project) => (
            <ProjectCard
              key={project.id}
              project={project}
              username={user?.username ?? null}
              onOpen={() => open(project)}
              onShare={() => setSharing(project)}
              onRename={() => rename(project)}
              onDuplicate={() => void duplicate(project)}
              onDelete={() => remove(project)}
              onLeave={() => leave(project)}
            />
          ))}
        </SimpleGrid>
      </Tabs.Panel>
    );
  };

  const count = (value: Tab) =>
    lists[value] && (
      <Badge size="sm" variant="light" color="gray" circle={lists[value].length < 10}>
        {lists[value].length}
      </Badge>
    );

  return (
    <Container size="lg" py="xl">
      <Group justify="space-between" mb="md">
        <Title order={2}>My networks</Title>
        <Button leftSection={<IconPlus size={16} />} onClick={createNew}>
          New network
        </Button>
      </Group>

      {error && <Alert color="red">{error}</Alert>}
      {!projects && !error && (
        <Center h={200}>
          <Loader />
        </Center>
      )}

      {projects && (
        <Tabs
          value={tab}
          onChange={(value) =>
            setSearchParams(value === 'shared' ? { tab: 'shared' } : {}, { replace: true })
          }
        >
          <Tabs.List>
            <Tabs.Tab
              value="mine"
              leftSection={<IconFolder size={16} />}
              rightSection={count('mine')}
            >
              Mine
            </Tabs.Tab>
            <Tabs.Tab
              value="shared"
              leftSection={<IconUsers size={16} />}
              rightSection={count('shared')}
            >
              Shared with me
            </Tabs.Tab>
          </Tabs.List>
          {panel('mine')}
          {panel('shared')}
        </Tabs>
      )}

      {sharing && (
        <ShareDialog
          project={sharing}
          onClose={() => setSharing(null)}
          onChange={({ shared, link_access, share_token }) =>
            replaceProject({ id: sharing.id, shared, link_access, share_token })
          }
          onLeave={() => dropProject(sharing.id)}
        />
      )}
    </Container>
  );
}

interface ProjectCardProps {
  project: Project;
  /** The current user: "Updated … by" only names other people. */
  username: string | null;
  onOpen: () => void;
  onShare: () => void;
  onRename: () => void;
  onDuplicate: () => void;
  onDelete: () => void;
  onLeave: () => void;
}

function ProjectCard({ project, username, ...actions }: ProjectCardProps) {
  const isOwner = project.role === 'owner';
  const changedBy = project.updated_by !== username ? project.updated_by : null;
  return (
    <Card withBorder padding="lg" radius="md" component="article" aria-label={project.name}>
      <Group justify="space-between" wrap="nowrap" align="flex-start">
        <Text fw={600} lineClamp={2} style={{ overflowWrap: 'anywhere' }}>
          {project.name}
        </Text>
        <Menu position="bottom-end" withinPortal>
          <Menu.Target>
            <ActionIcon variant="subtle" color="gray" aria-label="Network actions">
              <IconDots size={16} />
            </ActionIcon>
          </Menu.Target>
          <Menu.Dropdown>
            <Menu.Item leftSection={<IconFolderOpen size={14} />} onClick={actions.onOpen}>
              Open
            </Menu.Item>
            <Menu.Item leftSection={<IconUsersPlus size={14} />} onClick={actions.onShare}>
              Share…
            </Menu.Item>
            {project.role !== 'viewer' && (
              <Menu.Item leftSection={<IconPencil size={14} />} onClick={actions.onRename}>
                Rename
              </Menu.Item>
            )}
            <Menu.Item leftSection={<IconCopyPlus size={14} />} onClick={actions.onDuplicate}>
              Duplicate
            </Menu.Item>
            <Menu.Divider />
            {isOwner ? (
              <Menu.Item
                color="red"
                leftSection={<IconTrash size={14} />}
                onClick={actions.onDelete}
              >
                Delete
              </Menu.Item>
            ) : (
              <Menu.Item
                color="red"
                leftSection={<IconLogout size={14} />}
                onClick={actions.onLeave}
              >
                Leave
              </Menu.Item>
            )}
          </Menu.Dropdown>
        </Menu>
      </Group>
      {!isOwner && (
        <Text size="sm" c="dimmed">
          by {project.owner.username}
        </Text>
      )}
      <Group gap="xs" mt="xs">
        {!isOwner && (
          <Badge
            variant="light"
            color={project.role === 'editor' ? 'brick' : 'gray'}
            leftSection={project.role === 'editor' ? <IconEdit size={12} /> : <IconEye size={12} />}
            tt="none"
          >
            {ROLE_LABELS[project.role]}
          </Badge>
        )}
        <Badge variant="light" color={isOwner ? undefined : 'gray'}>
          {project.num_layers} layers
        </Badge>
        {isOwner && project.shared && (
          <Badge variant="outline" leftSection={<IconUsers size={12} />} tt="none">
            Shared
          </Badge>
        )}
      </Group>
      <Text size="xs" c="dimmed" mt="sm">
        Updated {formatDate(project.updated_at)}
        {changedBy && ` by ${changedBy}`}
      </Text>
      <Button variant="light" fullWidth mt="md" onClick={actions.onOpen}>
        Open in builder
      </Button>
    </Card>
  );
}
