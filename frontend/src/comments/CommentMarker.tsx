import { IconMessageCircle } from '@tabler/icons-react';
import { Tooltip, UnstyledButton } from '@mantine/core';
import { useCollab, useCollabStore } from '@/collab/context';
import classes from './Comments.module.css';

/**
 * The number of open comment threads on a layer, in its header on the canvas (nothing if
 * there are none or outside the builder). Clicking it shows them in the comments panel.
 */
export function CommentMarker({ nodeId }: { nodeId: string }) {
  const store = useCollabStore();
  const count = useCollab((state) => state.openByLayer[nodeId] ?? 0, 0);
  if (!store || count === 0) {
    return null;
  }
  const label = count === 1 ? '1 open comment' : `${count} open comments`;
  return (
    <Tooltip label={label} withArrow>
      <UnstyledButton
        className={classes.marker}
        onClick={() => store.openPanel('comments', nodeId)}
        aria-label={`${label}. Show them`}
        data-testid="comment-marker"
      >
        <IconMessageCircle size={12} stroke={2.5} />
        {count}
      </UnstyledButton>
    </Tooltip>
  );
}
