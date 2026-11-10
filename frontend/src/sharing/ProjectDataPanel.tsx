import { useEffect, useId, useMemo, useState } from 'react';
import {
  IconFileSpreadsheet,
  IconLayersIntersect,
  IconPhoto,
  IconTableOptions,
  IconX,
} from '@tabler/icons-react';
import { Link, useInRouterContext } from 'react-router-dom';
import {
  ActionIcon,
  Alert,
  Anchor,
  Button,
  Center,
  Loader,
  Select,
  Text,
  ThemeIcon,
  Tooltip,
} from '@mantine/core';
import { notifications } from '@mantine/notifications';
import {
  api,
  type ProjectData,
  type ProjectDataItem,
  type ProjectDataKind,
  type ProjectRole,
} from '@/api';
import { useAuth } from '@/auth/AuthContext';
import { prepRole } from '@/data/prepAccess';
import { isOwnDataset } from '@/training/access';
import { formatBytes } from '@/training/format';
import classes from './Sharing.module.css';

interface ProjectDataPanelProps {
  project: { id: number; name: string; role: ProjectRole };
}

/** One of the user's own datasets or preparations that could be shared. */
interface OwnItem {
  kind: ProjectDataKind;
  id: number;
  name: string;
}

const KIND_TITLES: Record<ProjectDataKind, string> = {
  dataset: 'Training datasets',
  prep: 'Data preparations',
};

const itemKey = (kind: ProjectDataKind, id: number) => `${kind}:${id}`;

/** What a shared training dataset holds. */
const DATASET_LABELS = { csv: 'CSV', images: 'Images', segmentation: 'Segmentation' } as const;
const DATASET_COLORS = { csv: 'teal', images: 'violet', segmentation: 'grape' } as const;

/**
 * Training datasets and data preparations shared in a project: everyone with access sees
 * them, owners and editors share their own, the item's owner and the project's owner remove
 * them. Mount it while it is shown.
 */
export function ProjectDataPanel({ project }: ProjectDataPanelProps) {
  const { user } = useAuth();
  const canAdd = project.role === 'owner' || project.role === 'editor';
  const [data, setData] = useState<ProjectData | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [own, setOwn] = useState<OwnItem[] | null>(null);
  const [choice, setChoice] = useState<string | null>(null);
  const [adding, setAdding] = useState(false);
  const [removing, setRemoving] = useState<string | null>(null);
  const headingId = useId();

  useEffect(() => {
    const controller = new AbortController();
    api.projects.data
      .list(project.id, controller.signal)
      .then((result) => !controller.signal.aborted && setData(result))
      .catch((error: Error) => !controller.signal.aborted && setLoadError(error.message));
    return () => controller.abort();
  }, [project.id]);

  // The user's own items to choose from. Training or data preparation may be switched off on
  // the server: then that kind is simply missing.
  useEffect(() => {
    if (!canAdd) {
      return undefined;
    }
    const controller = new AbortController();
    void Promise.allSettled([
      api.training.datasets.list(controller.signal),
      api.data.preps.list(controller.signal),
    ]).then(([datasets, preps]) => {
      if (controller.signal.aborted) {
        return;
      }
      setOwn([
        ...(datasets.status === 'fulfilled' ? datasets.value : [])
          .filter(isOwnDataset)
          .map((dataset) => ({ kind: 'dataset' as const, id: dataset.id, name: dataset.name })),
        ...(preps.status === 'fulfilled' ? preps.value : [])
          .filter((prep) => prepRole(prep) === 'owner')
          .map((prep) => ({ kind: 'prep' as const, id: prep.id, name: prep.name })),
      ]);
    });
    return () => controller.abort();
  }, [canAdd]);

  const options = useMemo(() => {
    if (!own || !data) {
      return [];
    }
    const shared = new Set([
      ...data.datasets.map((item) => itemKey('dataset', item.id)),
      ...data.preps.map((item) => itemKey('prep', item.id)),
    ]);
    const groups: { group: string; items: { value: string; label: string }[] }[] = [];
    for (const kind of ['dataset', 'prep'] as const) {
      const items = own
        .filter((item) => item.kind === kind && !shared.has(itemKey(kind, item.id)))
        .map((item) => ({ value: itemKey(kind, item.id), label: item.name }));
      if (items.length > 0) {
        groups.push({ group: KIND_TITLES[kind], items });
      }
    }
    return groups;
  }, [own, data]);

  const add = async () => {
    if (!choice) {
      return;
    }
    const [kind, id] = choice.split(':');
    setAdding(true);
    try {
      setData(await api.projects.data.add(project.id, kind as ProjectDataKind, Number(id)));
      setChoice(null);
    } catch (error) {
      notifications.show({
        color: 'red',
        title: 'Could not share it',
        message: (error as Error).message,
      });
    } finally {
      setAdding(false);
    }
  };

  const remove = async (kind: ProjectDataKind, item: ProjectDataItem) => {
    const key = itemKey(kind, item.id);
    setRemoving(key);
    try {
      await api.projects.data.remove(project.id, kind, item.id);
      const field = kind === 'dataset' ? 'datasets' : 'preps';
      setData(
        (current) =>
          current && { ...current, [field]: current[field].filter((entry) => entry.id !== item.id) }
      );
    } catch (error) {
      notifications.show({
        color: 'red',
        title: `Could not stop sharing “${item.name}”`,
        message: (error as Error).message,
      });
    } finally {
      setRemoving(null);
    }
  };

  const mayRemove = (item: ProjectDataItem) =>
    project.role === 'owner' || (user !== null && item.owner === user.username);

  if (loadError) {
    return <Alert color="red">{loadError}</Alert>;
  }
  if (!data) {
    return (
      <Center h={80}>
        <Loader size="sm" />
      </Center>
    );
  }
  const empty = data.datasets.length === 0 && data.preps.length === 0;

  return (
    <>
      <Text size="sm" c="dimmed" mb="sm">
        Everyone in this network can train on the datasets shared here and open the preparations in
        their Data tab; editors can also change the preparations.
      </Text>

      {canAdd && (
        <form
          className={classes.invite}
          onSubmit={(event) => {
            event.preventDefault();
            void add();
          }}
        >
          <Select
            className={classes.inviteUser}
            aria-label="Dataset or preparation to share"
            placeholder={
              own === null
                ? 'Loading your data …'
                : options.length === 0
                  ? 'Nothing else of yours to share'
                  : 'Choose one of your datasets or preparations'
            }
            data={options}
            value={choice}
            onChange={setChoice}
            disabled={options.length === 0}
            searchable
            nothingFoundMessage="Nothing found"
            comboboxProps={{ withinPortal: true }}
          />
          <div className={classes.inviteControls}>
            <Button type="submit" loading={adding} disabled={!choice}>
              Share
            </Button>
          </div>
        </form>
      )}

      {empty && (
        <Text size="sm" c="dimmed" mt="md">
          Nothing is shared here yet.
          {!canAdd && ' The owner and the editors of the network can share their data here.'}
        </Text>
      )}
      {(['dataset', 'prep'] as const).map((kind) => {
        const items = kind === 'dataset' ? data.datasets : data.preps;
        if (items.length === 0) {
          return null;
        }
        return (
          <section key={kind}>
            <Text fw={600} size="sm" mt="lg" mb={4} id={`${headingId}-${kind}`}>
              {KIND_TITLES[kind]}
            </Text>
            <ul className={classes.people} aria-labelledby={`${headingId}-${kind}`}>
              {items.map((item) => (
                <DataItemRow
                  key={item.id}
                  kind={kind}
                  item={item}
                  you={user !== null && item.owner === user.username}
                  onRemove={mayRemove(item) ? () => void remove(kind, item) : undefined}
                  removing={removing === itemKey(kind, item.id)}
                />
              ))}
            </ul>
          </section>
        );
      })}
    </>
  );
}

function DataItemRow({
  kind,
  item,
  you,
  onRemove,
  removing,
}: {
  kind: ProjectDataKind;
  item: ProjectDataItem;
  you: boolean;
  onRemove?: () => void;
  removing: boolean;
}) {
  const inRouter = useInRouterContext();
  const meta = [
    you ? 'by you' : `by ${item.owner}`,
    kind === 'dataset' && DATASET_LABELS[item.kind ?? 'csv'],
    kind === 'dataset' && formatBytes(item.size_bytes),
  ]
    .filter(Boolean)
    .join(' · ');
  const icon =
    kind === 'prep' ? (
      <IconTableOptions size={16} />
    ) : item.kind === 'segmentation' ? (
      <IconLayersIntersect size={16} />
    ) : item.kind === 'images' ? (
      <IconPhoto size={16} />
    ) : (
      <IconFileSpreadsheet size={16} />
    );
  return (
    <li className={classes.person} aria-label={item.name}>
      <div className={classes.personName}>
        <ThemeIcon
          variant="light"
          color={kind === 'prep' ? 'blue' : DATASET_COLORS[item.kind ?? 'csv']}
          aria-hidden
        >
          {icon}
        </ThemeIcon>
        <div style={{ minWidth: 0 }}>
          {kind === 'prep' && inRouter ? (
            <Anchor component={Link} to={`/data/${item.id}`} size="sm" fw={500} truncate="end">
              {item.name}
            </Anchor>
          ) : (
            <Text size="sm" fw={500} truncate>
              {item.name}
            </Text>
          )}
          <Text size="xs" c="dimmed" truncate>
            {meta}
          </Text>
        </div>
      </div>
      {onRemove && (
        <div className={classes.personActions}>
          <Tooltip label="Stop sharing it here">
            <ActionIcon
              variant="subtle"
              color="gray"
              onClick={onRemove}
              loading={removing}
              aria-label={`Stop sharing ${item.name}`}
            >
              <IconX size={16} />
            </ActionIcon>
          </Tooltip>
        </div>
      )}
    </li>
  );
}
