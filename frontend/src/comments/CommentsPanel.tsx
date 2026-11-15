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
import { i18n, useTranslation } from '@/i18n';
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
  get all() {
    return i18n.t('ui.comments.CommentsPanel.noCommentsYet');
  },
  get open() {
    return i18n.t('ui.comments.CommentsPanel.noOpenComments');
  },
  get resolved() {
    return i18n.t('ui.comments.CommentsPanel.noResolvedComments');
  },
};

/**
 * The comments of a saved network: threads on the network and on its layers, filtered by
 * state (and layer), and a field for a new comment. Everyone with access may comment.
 */
export function CommentsPanel({ store, layerName, onFocusLayer }: CommentsPanelProps) {
  useTranslation();
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
  const layerLabel =
    layer === null ? null : (layerName(layer) ?? i18n.t('ui.comments.CommentsPanel.deletedLayer'));

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
            {i18n.t('ui.comments.CommentsPanel.on', { layerLabel })}
          </Badge>
          <Button
            variant="subtle"
            size="compact-sm"
            rightSection={<IconX size={14} />}
            onClick={() => store.setLayer(null)}
          >
            {i18n.t('ui.comments.CommentsPanel.allComments')}
          </Button>
        </Group>
      )}

      <CommentComposer
        key={layer ?? ''}
        label={
          layerLabel === null
            ? i18n.t('ui.comments.CommentsPanel.commentOnTheNetwork')
            : i18n.t('ui.comments.CommentsPanel.commentOn', { layerLabel })
        }
        placeholder={
          layerLabel === null
            ? i18n.t('ui.comments.CommentsPanel.commentOnTheNetwork2')
            : i18n.t('ui.comments.CommentsPanel.commentOn2', { layerLabel })
        }
        submitLabel="Comment"
        onSubmit={async (text) => {
          await store.add(text, layer);
          setFilter((current) => (current === 'resolved' ? 'open' : current));
        }}
      />
      {layer === null && (
        <Text size="xs" c="dimmed" mt={-6}>
          {i18n.t('ui.comments.CommentsPanel.toCommentOnALayer')}
        </Text>
      )}

      <SegmentedControl
        fullWidth
        size="xs"
        value={filter}
        onChange={(value) => setFilter(value as ThreadFilter)}
        data={[segment('all', 'All'), segment('open', 'Open'), segment('resolved', 'Resolved')]}
        aria-label={i18n.t('ui.comments.CommentsPanel.showComments')}
      />

      {error && !loaded && (
        <Alert
          color="red"
          variant="light"
          title={i18n.t('ui.comments.CommentsPanel.couldNotLoadTheComments')}
        >
          {error}{' '}
          <Button variant="subtle" size="compact-xs" onClick={() => void store.load()}>
            {i18n.t('ui.comments.CommentsPanel.tryAgain')}
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
