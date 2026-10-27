import { useCallback, useEffect, useState } from 'react';
import { IconDots, IconPencil, IconPlus, IconTrash } from '@tabler/icons-react';
import { useNavigate } from 'react-router-dom';
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
  Text,
  TextInput,
  Title,
} from '@mantine/core';
import { modals } from '@mantine/modals';
import { notifications } from '@mantine/notifications';
import { api, type Project } from '@/api';
import { loadDraft, saveDraft, type Draft } from '@/builder/draft';

function formatDate(iso: string): string {
  return new Date(iso).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });
}

export function ProjectsPage() {
  const navigate = useNavigate();
  const [projects, setProjects] = useState<Project[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    api.projects
      .list()
      .then(setProjects)
      .catch((err: Error) => setError(err.message));
  }, []);

  useEffect(load, [load]);

  /** Ask before replacing a different network that is open in the builder. */
  const replaceDraft = (draft: Draft) => {
    const current = loadDraft();
    const replace = () => {
      saveDraft(draft);
      navigate('/');
    };
    const sameProject = draft.projectId !== null && current?.projectId === draft.projectId;
    if (!current || current.doc.nodes.length === 0 || sameProject) {
      replace();
      return;
    }
    modals.openConfirmModal({
      title: 'Replace the network in the builder?',
      children: (
        <Text size="sm">
          “{current.name}” is currently open in the builder. Unsaved changes to it will be lost.
        </Text>
      ),
      labels: { confirm: 'Replace', cancel: 'Cancel' },
      onConfirm: replace,
    });
  };

  const open = (project: Project) =>
    replaceDraft({ doc: project.graph, name: project.name, projectId: project.id });

  const createNew = () =>
    replaceDraft({
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
          await api.projects.update(project.id, { name: value.trim() || project.name });
          load();
        } catch (err) {
          notifications.show({ color: 'red', message: (err as Error).message });
        }
      },
    });
  };

  const remove = (project: Project) =>
    modals.openConfirmModal({
      title: `Delete "${project.name}"?`,
      children: <Text size="sm">This cannot be undone.</Text>,
      labels: { confirm: 'Delete', cancel: 'Cancel' },
      confirmProps: { color: 'red' },
      onConfirm: async () => {
        try {
          await api.projects.remove(project.id);
          setProjects((list) => list?.filter((p) => p.id !== project.id) ?? null);
        } catch (err) {
          notifications.show({ color: 'red', message: (err as Error).message });
        }
      },
    });

  return (
    <Container size="lg" py="xl">
      <Group justify="space-between" mb="lg">
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
      {projects?.length === 0 && (
        <Card withBorder p="xl" ta="center">
          <Text c="dimmed">
            You have not saved any networks yet. Build one in the editor and press “Save”.
          </Text>
        </Card>
      )}

      <SimpleGrid cols={{ base: 1, sm: 2, md: 3 }}>
        {projects?.map((project) => (
          <Card key={project.id} withBorder padding="lg" radius="md">
            <Group justify="space-between" wrap="nowrap" align="flex-start">
              <Text fw={600} lineClamp={2}>
                {project.name}
              </Text>
              <Menu position="bottom-end" withinPortal>
                <Menu.Target>
                  <ActionIcon variant="subtle" color="gray" aria-label="Network actions">
                    <IconDots size={16} />
                  </ActionIcon>
                </Menu.Target>
                <Menu.Dropdown>
                  <Menu.Item leftSection={<IconPencil size={14} />} onClick={() => rename(project)}>
                    Rename
                  </Menu.Item>
                  <Menu.Item
                    color="red"
                    leftSection={<IconTrash size={14} />}
                    onClick={() => remove(project)}
                  >
                    Delete
                  </Menu.Item>
                </Menu.Dropdown>
              </Menu>
            </Group>
            <Group gap="xs" mt="xs">
              <Badge variant="light">{project.num_layers} layers</Badge>
            </Group>
            <Text size="xs" c="dimmed" mt="sm">
              Updated {formatDate(project.updated_at)}
            </Text>
            <Button variant="light" fullWidth mt="md" onClick={() => open(project)}>
              Open in builder
            </Button>
          </Card>
        ))}
      </SimpleGrid>
    </Container>
  );
}
