import { useMemo, useState } from 'react';
import { IconMessagePlus } from '@tabler/icons-react';
import { Anchor, Button, Divider, Stack } from '@mantine/core';
import { useCollab, useCollabStore } from '@/collab/context';
import type { CollabState } from '@/collab/store';
import { CommentComposer } from './CommentComposer';
import { CommentThread } from './CommentThread';
import { buildThreads, filterThreads } from './threads';

const whole = (state: CollabState) => state;

/**
 * The open comment threads on a layer and a button to add one, for the inspector. Nothing
 * outside the builder or for networks that are not saved. Viewers may comment too.
 */
export function LayerComments({ nodeId }: { nodeId: string }) {
  const store = useCollabStore();
  const state = useCollab<CollabState | null>(whole, null);
  const [composing, setComposing] = useState(false);
  const comments = state?.comments;
  const threads = useMemo(
    () => filterThreads(buildThreads(comments ?? []), 'all', nodeId),
    [comments, nodeId]
  );

  if (!store || !state || state.projectId === null) {
    return null;
  }
  const open = filterThreads(threads, 'open');
  const resolved = threads.length - open.length;
  const viewer = { userId: state.userId, role: state.role };

  return (
    <Stack gap="xs">
      <Divider label="Comments" labelPosition="left" />
      {open.map((thread) => (
        <CommentThread key={thread.root.id} thread={thread} viewer={viewer} store={store} />
      ))}
      {resolved > 0 && (
        <Anchor
          component="button"
          type="button"
          size="xs"
          ta="left"
          onClick={() => store.openPanel('comments', nodeId)}
        >
          {resolved === 1 ? '1 resolved comment' : `${resolved} resolved comments`}: show in the
          comments panel
        </Anchor>
      )}
      {composing ? (
        <CommentComposer
          label="Comment on this layer"
          placeholder="Comment on this layer…"
          submitLabel="Comment"
          autoFocus
          onSubmit={async (text) => {
            await store.add(text, nodeId);
            setComposing(false);
          }}
          onCancel={() => setComposing(false)}
        />
      ) : (
        <Button
          variant="light"
          size="xs"
          leftSection={<IconMessagePlus size={14} />}
          onClick={() => setComposing(true)}
          style={{ alignSelf: 'flex-start' }}
        >
          Add comment
        </Button>
      )}
    </Stack>
  );
}
