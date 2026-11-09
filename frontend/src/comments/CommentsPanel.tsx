import { useMemo, useState } from 'react';
import { IconMessageCircle, IconStack2, IconX } from '@tabler/icons-react';
import {
  Alert,
  Badge,
  Button,
  Center,
  Group,
  Loader,
  SegmentedControl,
  Stack,
  Text,
} from '@mantine/core';
import { useStoreState } from '@/collab/context';
import type { CollabStore } from '@/collab/store';
import { CommentComposer } from './CommentComposer';
import { CommentThread } from './CommentThread';
import { buildThreads, filterThreads, type CommentViewer, type ThreadFilter } from './threads';

export interface CommentsPanelProps {
  store: CollabStore;
  /** The name of a layer on the canvas; null if there is no such layer (any more). */
  layerName: (id: string) => string | null;
  /** Selects a layer and shows it on the canvas. */
  onFocusLayer: (id: string) => void;
}

const EMPTY_TEXT: Record<ThreadFilter, string> = {
  all: 'No comments yet.',
  open: 'No open comments.',
  resolved: 'No resolved comments.',
};

/**
 * The comments of a saved network: threads on the network and on its layers, filtered by
 * state (and layer), and a field for a new comment. Everyone with access may comment.
 */
export function CommentsPanel({ store, layerName, onFocusLayer }: CommentsPanelProps) {
  const state = useStoreState(store);
  const { comments, loaded, error, layer } = state;
  const [filter, setFilter] = useState<ThreadFilter>('open');
  const viewer: CommentViewer = { userId: state.userId, role: state.role };

  const threads = useMemo(() => buildThreads(comments), [comments]);
  const inLayer = filterThreads(threads, 'all', layer);
  const shown = filterThreads(inLayer, filter);
  const counts = {
    all: inLayer.length,
    open: inLayer.filter((thread) => !thread.root.resolved).length,
    resolved: inLayer.filter((thread) => thread.root.resolved).length,
  };
  const layerLabel = layer === null ? null : (layerName(layer) ?? 'Deleted layer');

  const segment = (value: ThreadFilter, label: string) => ({
    value,
    label: counts[value] > 0 ? `${label} · ${counts[value]}` : label,
  });

  return (
    <Stack gap="sm">
      {layerLabel !== null && (
        <Group gap={6} wrap="nowrap" justify="space-between">
          <Badge
            variant="light"
            color="gray"
            tt="none"
            size="lg"
            leftSection={<IconStack2 size={14} />}
            style={{ minWidth: 0 }}
          >
            On {layerLabel}
          </Badge>
          <Button
            variant="subtle"
            size="compact-sm"
            rightSection={<IconX size={14} />}
            onClick={() => store.setLayer(null)}
          >
            All comments
          </Button>
        </Group>
      )}

      <CommentComposer
        key={layer ?? ''}
        label={layerLabel === null ? 'Comment on the network' : `Comment on ${layerLabel}`}
        placeholder={layerLabel === null ? 'Comment on the network…' : `Comment on ${layerLabel}…`}
        submitLabel="Comment"
        onSubmit={async (text) => {
          await store.add(text, layer);
          setFilter((current) => (current === 'resolved' ? 'open' : current));
        }}
      />
      {layer === null && (
        <Text size="xs" c="dimmed" mt={-6}>
          To comment on a layer, select it and use “Add comment” in its properties.
        </Text>
      )}

      <SegmentedControl
        fullWidth
        size="xs"
        value={filter}
        onChange={(value) => setFilter(value as ThreadFilter)}
        data={[segment('all', 'All'), segment('open', 'Open'), segment('resolved', 'Resolved')]}
        aria-label="Show comments"
      />

      {error && !loaded && (
        <Alert color="red" variant="light" title="Could not load the comments">
          {error}{' '}
          <Button variant="subtle" size="compact-xs" onClick={() => void store.load()}>
            Try again
          </Button>
        </Alert>
      )}
      {!loaded && !error && (
        <Center py="lg">
          <Loader size="sm" />
        </Center>
      )}
      {loaded && shown.length === 0 && (
        <Stack align="center" gap={4} py="lg">
          <IconMessageCircle size={28} color="var(--mantine-color-dimmed)" />
          <Text size="sm" c="dimmed">
            {EMPTY_TEXT[filter]}
          </Text>
        </Stack>
      )}
      {shown.map((thread) => {
        const nodeId = thread.root.node_id;
        return (
          <CommentThread
            key={thread.root.id}
            thread={thread}
            viewer={viewer}
            store={store}
            layer={
              nodeId && layer === null
                ? { name: layerName(nodeId), onFocus: () => onFocusLayer(nodeId) }
                : null
            }
          />
        );
      })}
    </Stack>
  );
}
