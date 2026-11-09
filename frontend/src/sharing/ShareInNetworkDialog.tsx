import { useEffect, useState } from 'react';
import { Alert, Button, Center, Checkbox, Group, Loader, Modal, Switch, Text } from '@mantine/core';
import { notifications } from '@mantine/notifications';
import { api, type Project, type ProjectDataKind, type ProjectRef } from '@/api';
import { ROLE_LABELS } from './roles';
import classes from './Sharing.module.css';

interface ShareInNetworkDialogProps {
  kind: ProjectDataKind;
  /** One of the user's own datasets or preparations. */
  item: { id: number; name: string; shared_in: ProjectRef[] };
  onClose: () => void;
  /** The networks it is shared in now, by name. */
  onChange: (sharedIn: ProjectRef[]) => void;
  /** Preparations: the training datasets made from it, offered to be shared along. */
  datasets?: { id: number; name: string }[];
}

const byName = (a: ProjectRef, b: ProjectRef) => a.name.localeCompare(b.name);

/**
 * Share one of the user's datasets or preparations in their networks (as owner or editor),
 * or stop sharing it there. Mount it while it is open.
 */
export function ShareInNetworkDialog({
  kind,
  item,
  onClose,
  onChange,
  datasets = [],
}: ShareInNetworkDialogProps) {
  const [projects, setProjects] = useState<Project[] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [sharedIn, setSharedIn] = useState<ProjectRef[]>(item.shared_in);
  /** The change on its way: the switch shows it right away. */
  const [busy, setBusy] = useState<{ id: number; on: boolean } | null>(null);
  const [withDatasets, setWithDatasets] = useState(true);

  useEffect(() => {
    let cancelled = false;
    api.projects
      .list()
      .then((list) => !cancelled && setProjects(list))
      .catch((error: Error) => !cancelled && setLoadError(error.message));
    return () => {
      cancelled = true;
    };
  }, []);

  const shared = new Set(sharedIn.map((project) => project.id));
  // Networks the user can share in, and those it is shared in already (the owner of an item
  // can always stop sharing it, also after becoming a viewer there).
  const listed = (projects ?? []).filter(
    (project) => project.role !== 'viewer' || shared.has(project.id)
  );

  const toggle = async (project: Project, on: boolean) => {
    setBusy({ id: project.id, on });
    try {
      let next: ProjectRef[];
      if (on) {
        await api.projects.data.add(project.id, kind, item.id);
        next = [...sharedIn, { id: project.id, name: project.name }].sort(byName);
        if (withDatasets && datasets.length > 0) {
          await Promise.all(
            datasets.map((dataset) => api.projects.data.add(project.id, 'dataset', dataset.id))
          );
        }
      } else {
        await api.projects.data.remove(project.id, kind, item.id);
        next = sharedIn.filter((entry) => entry.id !== project.id);
      }
      setSharedIn(next);
      onChange(next);
    } catch (error) {
      notifications.show({
        color: 'red',
        title: on ? `Could not share it in “${project.name}”` : 'Could not stop sharing it',
        message: (error as Error).message,
      });
    } finally {
      setBusy(null);
    }
  };

  return (
    <Modal opened onClose={onClose} title={`Share “${item.name}” in a network`}>
      <Text size="sm" c="dimmed" mb="md">
        {kind === 'prep'
          ? 'Everyone in the network can open it in their Data tab; its editors can also change the steps and create training datasets from it. Training datasets you create from it are shared there, too. The imported table stays yours.'
          : 'Everyone in the network can train on it. Only you can delete it.'}
      </Text>
      {datasets.length > 0 && (
        <Checkbox
          mb="md"
          label={
            datasets.length === 1
              ? `Also share the training dataset “${datasets[0].name}” made from it`
              : `Also share the ${datasets.length} training datasets made from it`
          }
          checked={withDatasets}
          onChange={(event) => setWithDatasets(event.currentTarget.checked)}
        />
      )}
      {loadError && <Alert color="red">{loadError}</Alert>}
      {!projects && !loadError && (
        <Center h={80}>
          <Loader size="sm" />
        </Center>
      )}
      {projects && listed.length === 0 && (
        <Text size="sm">
          You have no networks to share it in yet. Save a network (or ask to edit someone else’s),
          then share it there.
        </Text>
      )}
      {listed.length > 0 && (
        <ul className={classes.people} aria-label="Networks">
          {listed.map((project) => (
            <li key={project.id} className={classes.person}>
              <Switch
                label={project.name}
                description={
                  project.role === 'owner'
                    ? 'Your network'
                    : `${project.owner.username}’s network · ${ROLE_LABELS[project.role]}`
                }
                checked={busy?.id === project.id ? busy.on : shared.has(project.id)}
                disabled={busy !== null || (project.role === 'viewer' && !shared.has(project.id))}
                onChange={(event) => void toggle(project, event.currentTarget.checked)}
                styles={{ body: { alignItems: 'flex-start' }, labelWrapper: { minWidth: 0 } }}
              />
            </li>
          ))}
        </ul>
      )}
      <Group justify="flex-end" mt="xl">
        <Button variant="default" onClick={onClose}>
          Done
        </Button>
      </Group>
    </Modal>
  );
}
