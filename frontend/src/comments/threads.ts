import type { ProjectComment, ProjectRole } from '@/api/types';

/** A comment with its replies (one level, oldest first). */
export interface Thread {
  root: ProjectComment;
  replies: ProjectComment[];
}

export type ThreadFilter = 'all' | 'open' | 'resolved';

/** Who is looking: the user's id and role in the project (null: unknown). */
export interface CommentViewer {
  userId: number | null;
  role: ProjectRole | null;
}

/** Comments in creation order (ids grow with time). */
export function sortComments(comments: ProjectComment[]): ProjectComment[] {
  return [...comments].sort((a, b) => a.id - b.id);
}

/** Threads, newest first; replies whose thread is unknown are left out. */
export function buildThreads(comments: ProjectComment[]): Thread[] {
  const threads = new Map<number, Thread>();
  for (const comment of comments) {
    if (comment.parent === null) {
      threads.set(comment.id, { root: comment, replies: [] });
    }
  }
  for (const comment of comments) {
    if (comment.parent !== null) {
      threads.get(comment.parent)?.replies.push(comment);
    }
  }
  return [...threads.values()].reverse();
}

export function filterThreads(
  threads: Thread[],
  filter: ThreadFilter,
  layer: string | null = null
): Thread[] {
  return threads.filter(
    ({ root }) =>
      (layer === null || root.node_id === layer) &&
      (filter === 'all' || (filter === 'open') !== root.resolved)
  );
}

/** Open (not resolved) threads per layer id. */
export function openThreadsByLayer(comments: ProjectComment[]): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const comment of comments) {
    if (comment.parent === null && !comment.resolved && comment.node_id) {
      counts[comment.node_id] = (counts[comment.node_id] ?? 0) + 1;
    }
  }
  return counts;
}

export function openThreadCount(comments: ProjectComment[]): number {
  return comments.filter((comment) => comment.parent === null && !comment.resolved).length;
}

const isAuthor = (comment: ProjectComment, viewer: CommentViewer) =>
  viewer.userId !== null && comment.author.id === viewer.userId;

/** Only the author changes the text of a comment. */
export function canEditComment(comment: ProjectComment, viewer: CommentViewer): boolean {
  return isAuthor(comment, viewer);
}

/** The author and the owner of the project delete comments (with their replies). */
export function canDeleteComment(comment: ProjectComment, viewer: CommentViewer): boolean {
  return isAuthor(comment, viewer) || viewer.role === 'owner';
}

/** The author of the thread, editors and the owner resolve and reopen it. */
export function canResolveThread(root: ProjectComment, viewer: CommentViewer): boolean {
  return isAuthor(root, viewer) || viewer.role === 'owner' || viewer.role === 'editor';
}
