import { IconMessageCircle } from '@tabler/icons-react';
import { Tooltip, UnstyledButton } from '@mantine/core';
import { useCollab, useCollabStore } from '@/collab/context';
import { i18n, useTranslation } from '@/i18n';
import classes from './Comments.module.css';

/**
 * The number of open comment threads on a layer, in its header on the canvas (nothing if
 * there are none or outside the builder). Clicking it shows them in the comments panel.
 */
export function CommentMarker({ nodeId }: { nodeId: string }) {
  useTranslation();
  const store = useCollabStore();
  const count = useCollab((state) => state.openByLayer[nodeId] ?? 0, 0);
  if (!store || count === 0) {
    return null;
  }
  const label =
    count === 1
      ? i18n.t('ui.comments.CommentMarker.text1OpenComment')
      : i18n.t('ui.comments.CommentMarker.openComments', { count });
  return (
    <Tooltip label={label} withArrow>
      <UnstyledButton
        className={classes.marker}
        onClick={() => store.openPanel('comments', nodeId)}
        aria-label={i18n.t('ui.comments.CommentMarker.showThem', { label })}
        data-testid="comment-marker"
      >
        <IconMessageCircle size={12} stroke={2.5} />
        {count}
      </UnstyledButton>
    </Tooltip>
  );
}
