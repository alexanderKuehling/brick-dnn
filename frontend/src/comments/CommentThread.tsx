import { useState } from 'react';
import {
  IconCircleCheck,
  IconDots,
  IconPencil,
  IconRefresh,
  IconStack2,
  IconTrash,
} from '@tabler/icons-react';
import {
  ActionIcon,
  Avatar,
  Button,
  Group,
  Menu,
  Text,
  Tooltip,
  UnstyledButton,
} from '@mantine/core';
import { notifications } from '@mantine/notifications';
import type { ProjectComment } from '@/api';
import type { CollabStore } from '@/collab/store';
import { fullDateTime, relativeTime } from '@/collab/time';
import { CommentComposer } from './CommentComposer';
import {
  canDeleteComment,
  canEditComment,
  canResolveThread,
  type CommentViewer,
  type Thread,
} from './threads';
import classes from './Comments.module.css';

/** The layer a thread is about, as shown above it. */
export interface ThreadLayer {
  /** The layer's name on the canvas; null if it was deleted. */
  name: string | null;
  /** Selects the layer and shows it on the canvas. */
  onFocus?: () => void;
}

interface CommentThreadProps {
  thread: Thread;
  viewer: CommentViewer;
  store: CollabStore;
  /** The layer of a thread on a layer (left out where the layer is clear, e.g. the inspector). */
  layer?: ThreadLayer | null;
  highlight?: boolean;
}

const failed = (title: string) => (error: unknown) => {
  notifications.show({ color: 'red', title, message: (error as Error).message });
};

function LayerChip({ layer }: { layer: ThreadLayer }) {
  if (layer.name === null) {
    return (
      <Text span className={classes.layer} c="dimmed">
        <IconStack2 size={12} />
        <span className={classes.layerName}>Deleted layer</span>
      </Text>
    );
  }
  return (
    <Tooltip label="Show this layer on the canvas">
      <UnstyledButton
        className={classes.layer}
        onClick={layer.onFocus}
        aria-label={`Show layer ${layer.name}`}
      >
        <IconStack2 size={12} />
        <span className={classes.layerName}>{layer.name}</span>
      </UnstyledButton>
    </Tooltip>
  );
}

interface CommentItemProps {
  comment: ProjectComment;
  viewer: CommentViewer;
  store: CollabStore;
  /** The first comment of its thread, with the resolve button. */
  root?: boolean;
  /** Number of replies (deleting the first comment deletes them). */
  replyCount?: number;
}

function CommentItem({ comment, viewer, store, root = false, replyCount = 0 }: CommentItemProps) {
  const [editing, setEditing] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [busy, setBusy] = useState(false);
  const canEdit = canEditComment(comment, viewer);
  const canDelete = canDeleteComment(comment, viewer);
  const canResolve = root && canResolveThread(comment, viewer);
  const author = comment.author.username;
  const isMine = viewer.userId === comment.author.id;

  const run = async (action: () => Promise<unknown>, title: string) => {
    setBusy(true);
    try {
      await action();
    } catch (error) {
      failed(title)(error);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className={classes.comment}>
      <Avatar name={author} color="initials" size={26} mt={2} />
      <div className={classes.commentBody}>
        <div className={classes.commentHeader}>
          <Text size="sm" className={classes.author}>
            {isMine ? `${author} (you)` : author}
          </Text>
          <Tooltip label={fullDateTime(comment.created_at)}>
            <Text size="xs" c="dimmed" className={classes.time}>
              {relativeTime(comment.created_at)}
            </Text>
          </Tooltip>
          <div className={classes.headerActions}>
            {canResolve && !comment.resolved && (
              <Tooltip label="Resolve: mark as done">
                <ActionIcon
                  variant="subtle"
                  color="green"
                  size="sm"
                  loading={busy}
                  onClick={() =>
                    void run(() => store.setResolved(comment.id, true), 'Could not resolve it')
                  }
                  aria-label="Resolve thread"
                >
                  <IconCircleCheck size={16} />
                </ActionIcon>
              </Tooltip>
            )}
            {(canEdit || canDelete) && !editing && (
              <Menu position="bottom-end" withinPortal>
                <Menu.Target>
                  <ActionIcon
                    variant="subtle"
                    color="gray"
                    size="sm"
                    aria-label={`Actions for the comment by ${author}`}
                  >
                    <IconDots size={16} />
                  </ActionIcon>
                </Menu.Target>
                <Menu.Dropdown>
                  {canEdit && (
                    <Menu.Item
                      leftSection={<IconPencil size={14} />}
                      onClick={() => setEditing(true)}
                    >
                      Edit
                    </Menu.Item>
                  )}
                  {canDelete && (
                    <Menu.Item
                      color="red"
                      leftSection={<IconTrash size={14} />}
                      onClick={() => setConfirmDelete(true)}
                    >
                      Delete
                    </Menu.Item>
                  )}
                </Menu.Dropdown>
              </Menu>
            )}
          </div>
        </div>

        {editing ? (
          <CommentComposer
            label="Edit comment"
            submitLabel="Save"
            initialText={comment.text}
            autoFocus
            onSubmit={async (text) => {
              await store.edit(comment.id, text);
              setEditing(false);
            }}
            onCancel={() => setEditing(false)}
          />
        ) : (
          <Text size="sm" className={classes.text}>
            {comment.text}
          </Text>
        )}

        {confirmDelete && (
          <Group gap="xs" mt={6} wrap="wrap">
            <Text size="xs" c="dimmed" style={{ flex: '1 1 10rem' }}>
              {root && replyCount > 0
                ? `Delete this comment and its ${replyCount === 1 ? 'reply' : `${replyCount} replies`}?`
                : 'Delete this comment?'}
            </Text>
            <Button variant="default" size="xs" onClick={() => setConfirmDelete(false)}>
              Cancel
            </Button>
            <Button
              color="red"
              size="xs"
              loading={busy}
              onClick={() =>
                void run(async () => {
                  await store.remove(comment.id);
                  setConfirmDelete(false);
                }, 'Could not delete the comment')
              }
            >
              Delete
            </Button>
          </Group>
        )}
      </div>
    </div>
  );
}

/** A comment with its replies: reply, edit, delete, resolve and reopen as allowed. */
export function CommentThread({ thread, viewer, store, layer, highlight }: CommentThreadProps) {
  const { root, replies } = thread;
  const [replying, setReplying] = useState(false);
  const [reopening, setReopening] = useState(false);
  const canResolve = canResolveThread(root, viewer);

  const reopen = async () => {
    setReopening(true);
    try {
      await store.setResolved(root.id, false);
    } catch (error) {
      failed('Could not reopen it')(error);
    } finally {
      setReopening(false);
    }
  };

  return (
    <article
      className={classes.thread}
      data-resolved={root.resolved || undefined}
      data-highlight={highlight || undefined}
      aria-label={`Comment by ${root.author.username}`}
    >
      {layer && <LayerChip layer={layer} />}
      <CommentItem comment={root} viewer={viewer} store={store} root replyCount={replies.length} />
      {replies.length > 0 && (
        <div className={classes.replies}>
          {replies.map((reply) => (
            <CommentItem key={reply.id} comment={reply} viewer={viewer} store={store} />
          ))}
        </div>
      )}

      {root.resolved ? (
        <Group gap="xs" mt="xs" justify="space-between" wrap="wrap">
          <Text size="xs" c="dimmed">
            <IconCircleCheck
              size={14}
              style={{ verticalAlign: 'text-bottom' }}
              color="var(--mantine-color-green-6)"
            />{' '}
            Resolved{root.resolved_by ? ` by ${root.resolved_by}` : ''}
          </Text>
          {canResolve && (
            <Button
              variant="subtle"
              size="compact-xs"
              leftSection={<IconRefresh size={12} />}
              loading={reopening}
              onClick={() => void reopen()}
              aria-label="Reopen thread"
            >
              Reopen
            </Button>
          )}
        </Group>
      ) : replying ? (
        <div style={{ marginTop: 'var(--mantine-spacing-xs)' }}>
          <CommentComposer
            label="Reply"
            placeholder="Write a reply…"
            submitLabel="Reply"
            autoFocus
            onSubmit={async (text) => {
              await store.reply(root.id, text);
              setReplying(false);
            }}
            onCancel={() => setReplying(false)}
          />
        </div>
      ) : (
        <Button
          variant="subtle"
          size="compact-xs"
          mt={6}
          onClick={() => setReplying(true)}
          aria-label={`Reply to ${root.author.username}`}
        >
          Reply
        </Button>
      )}
    </article>
  );
}
