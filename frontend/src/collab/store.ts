import { api, type ProjectComment, type ProjectRole } from '@/api';
import { openThreadCount, openThreadsByLayer, sortComments } from '@/comments/threads';
import type { CommentEvent } from './events';

/** The side panels of a saved network. */
export type CollabPanel = 'history' | 'comments';

export interface CollabState {
  /** The saved project open in the builder; null: none (nothing is shown then). */
  projectId: number | null;
  /** The current user's id and role in it, for what they may do. */
  userId: number | null;
  role: ProjectRole | null;
  /** All comments of the project, oldest first. */
  comments: ProjectComment[];
  /** Open threads per layer id (the markers on the canvas). */
  openByLayer: Record<string, number>;
  /** Open threads in total (the badge of the comments button). */
  openCount: number;
  /** The comments were loaded (they are kept up to date afterwards). */
  loaded: boolean;
  loading: boolean;
  /** The last load failed. */
  error: string | null;
  panel: CollabPanel | null;
  /** The comments panel shows only the threads on this layer. */
  layer: string | null;
}

const EMPTY: CollabState = {
  projectId: null,
  userId: null,
  role: null,
  comments: [],
  openByLayer: {},
  openCount: 0,
  loaded: false,
  loading: false,
  error: null,
  panel: null,
  layer: null,
};

type Change = { kind: 'upsert'; comment: ProjectComment } | { kind: 'remove'; id: number };

function applyChange(comments: ProjectComment[], change: Change): ProjectComment[] {
  if (change.kind === 'remove') {
    // Replies go with their thread.
    return comments.filter((c) => c.id !== change.id && c.parent !== change.id);
  }
  const { comment } = change;
  const index = comments.findIndex((c) => c.id === comment.id);
  if (index >= 0) {
    const next = [...comments];
    next[index] = comment;
    return next;
  }
  return sortComments([...comments, comment]);
}

/**
 * Comments of the open project and which side panel is open, shared by the builder's panels,
 * the layers on the canvas and the inspector (read with `useCollab`).
 *
 * Changes of the current user are applied when the server confirms them; changes of others
 * arrive as live messages (`applyEvent`) or by loading again. Both are idempotent, so the own
 * changes may also arrive as messages.
 */
export class CollabStore {
  private state: CollabState = EMPTY;
  private listeners = new Set<() => void>();
  /** Goes up when another project is opened: answers for the old one are dropped. */
  private generation = 0;
  private inFlight = false;
  private loadAgain = false;
  /** Changes during a load, applied again on top of its (possibly older) result. */
  private changesDuringLoad: Change[] | null = null;

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  };

  getState = (): CollabState => this.state;

  private set(patch: Partial<CollabState>): void {
    let next = { ...this.state, ...patch };
    if (patch.comments) {
      next = {
        ...next,
        openByLayer: openThreadsByLayer(patch.comments),
        openCount: openThreadCount(patch.comments),
      };
    }
    this.state = next;
    this.listeners.forEach((listener) => listener());
  }

  /** The open project and the user's role in it; loads the comments of a new project. */
  configure({ projectId, userId, role }: Pick<CollabState, 'projectId' | 'userId' | 'role'>): void {
    if (projectId !== this.state.projectId) {
      this.generation += 1;
      this.inFlight = false;
      this.loadAgain = false;
      this.changesDuringLoad = null;
      this.set({ ...EMPTY, projectId, userId, role });
      if (projectId !== null) {
        void this.load();
      }
    } else if (userId !== this.state.userId || role !== this.state.role) {
      this.set({ userId, role });
    }
  }

  /** (Re)load the comments; a load while one is running runs once more afterwards. */
  async load(): Promise<void> {
    const { projectId } = this.state;
    if (projectId === null) {
      return;
    }
    if (this.inFlight) {
      this.loadAgain = true;
      return;
    }
    const generation = this.generation;
    this.inFlight = true;
    this.changesDuringLoad = [];
    this.set({ loading: true });
    try {
      let comments = sortComments(await api.projects.comments.list(projectId));
      if (generation !== this.generation) {
        return;
      }
      for (const change of this.changesDuringLoad ?? []) {
        comments = applyChange(comments, change);
      }
      this.set({ comments, loaded: true, loading: false, error: null });
    } catch (error) {
      if (generation === this.generation) {
        this.set({ loading: false, error: (error as Error).message });
      }
    } finally {
      if (generation === this.generation) {
        this.inFlight = false;
        this.changesDuringLoad = null;
        if (this.loadAgain) {
          this.loadAgain = false;
          void this.load();
        }
      }
    }
  }

  private change(projectId: number, change: Change): void {
    if (projectId !== this.state.projectId) {
      return; // Another project was opened in the meantime.
    }
    this.changesDuringLoad?.push(change);
    this.set({ comments: applyChange(this.state.comments, change) });
  }

  private requireProject(): number {
    const { projectId } = this.state;
    if (projectId === null) {
      throw new Error('No saved network is open.');
    }
    return projectId;
  }

  /** A new thread on the network, or on a layer. */
  async add(text: string, nodeId?: string | null): Promise<ProjectComment> {
    const projectId = this.requireProject();
    const comment = await api.projects.comments.create(
      projectId,
      nodeId ? { text, node_id: nodeId } : { text }
    );
    this.change(projectId, { kind: 'upsert', comment });
    return comment;
  }

  async reply(threadId: number, text: string): Promise<ProjectComment> {
    const projectId = this.requireProject();
    const comment = await api.projects.comments.create(projectId, { text, parent: threadId });
    this.change(projectId, { kind: 'upsert', comment });
    return comment;
  }

  async edit(id: number, text: string): Promise<ProjectComment> {
    const projectId = this.requireProject();
    const comment = await api.projects.comments.update(projectId, id, { text });
    this.change(projectId, { kind: 'upsert', comment });
    return comment;
  }

  async setResolved(id: number, resolved: boolean): Promise<ProjectComment> {
    const projectId = this.requireProject();
    const comment = await api.projects.comments.update(projectId, id, { resolved });
    this.change(projectId, { kind: 'upsert', comment });
    return comment;
  }

  /** Deletes a comment; deleting the first comment of a thread deletes the thread. */
  async remove(id: number): Promise<void> {
    const projectId = this.requireProject();
    await api.projects.comments.remove(projectId, id);
    this.change(projectId, { kind: 'remove', id });
  }

  /** A live message: someone created, changed or deleted a comment. */
  applyEvent(event: CommentEvent): void {
    const { projectId } = this.state;
    if (projectId === null) {
      return;
    }
    this.change(
      projectId,
      event.action === 'deleted'
        ? { kind: 'remove', id: event.comment.id }
        : { kind: 'upsert', comment: event.comment as ProjectComment }
    );
  }

  /** Opens a side panel (the comments panel optionally with the threads of one layer). */
  openPanel(panel: CollabPanel, layer: string | null = null): void {
    this.set({ panel, layer: panel === 'comments' ? layer : null });
  }

  closePanel(): void {
    this.set({ panel: null, layer: null });
  }

  /** The comments panel shows all threads again (or those of another layer). */
  setLayer(layer: string | null): void {
    this.set({ layer });
  }
}
