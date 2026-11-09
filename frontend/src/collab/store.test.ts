import { waitFor } from '@test-utils';
import { beforeEach, vi } from 'vitest';
import { api, type ProjectComment } from '@/api';
import { ALICE, BOB, comment } from './fixtures';
import { CollabStore } from './store';

vi.mock('@/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api')>();
  return {
    ...actual,
    api: {
      projects: {
        comments: { list: vi.fn(), create: vi.fn(), update: vi.fn(), remove: vi.fn() },
      },
    },
  };
});

const THREAD = comment({ id: 1, node_id: 'fc', author: ALICE });
const REPLY = comment({ id: 2, parent: 1, node_id: 'fc', author: BOB });
const GENERAL = comment({ id: 3, author: BOB });

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((r) => {
    resolve = r;
  });
  return { promise, resolve };
}

async function openStore(comments: ProjectComment[] = [THREAD, REPLY, GENERAL]) {
  vi.mocked(api.projects.comments.list).mockResolvedValue(comments);
  const store = new CollabStore();
  store.configure({ projectId: 5, userId: ALICE.id, role: 'editor' });
  await waitFor(() => expect(store.getState().loaded).toBe(true));
  return store;
}

beforeEach(() => {
  vi.clearAllMocks();
});

describe('CollabStore', () => {
  it('loads the comments of the open project and counts open threads', async () => {
    const store = await openStore();
    expect(api.projects.comments.list).toHaveBeenCalledWith(5);
    const state = store.getState();
    expect(state.comments.map((c) => c.id)).toEqual([1, 2, 3]);
    expect(state.openCount).toBe(2);
    expect(state.openByLayer).toEqual({ fc: 1 });
  });

  it('applies live messages: created, updated and deleted (with the replies)', async () => {
    const store = await openStore();
    const listener = vi.fn();
    store.subscribe(listener);

    const created = comment({ id: 4, node_id: 'in', author: BOB });
    store.applyEvent({ type: 'comment', action: 'created', comment: created });
    expect(store.getState().openByLayer).toEqual({ fc: 1, in: 1 });

    store.applyEvent({
      type: 'comment',
      action: 'updated',
      comment: { ...THREAD, resolved: true, resolved_by: 'bob' },
    });
    expect(store.getState().openByLayer).toEqual({ in: 1 });
    expect(store.getState().openCount).toBe(2);

    store.applyEvent({ type: 'comment', action: 'deleted', comment: { id: 1 } });
    expect(store.getState().comments.map((c) => c.id)).toEqual([3, 4]);
    expect(listener).toHaveBeenCalledTimes(3);
  });

  it('takes own changes and their live messages once', async () => {
    const store = await openStore();
    const created = comment({ id: 9, text: 'Mine', author: ALICE });
    vi.mocked(api.projects.comments.create).mockResolvedValue(created);

    await store.add('Mine');
    store.applyEvent({ type: 'comment', action: 'created', comment: created });
    expect(api.projects.comments.create).toHaveBeenCalledWith(5, { text: 'Mine' });
    expect(store.getState().comments.filter((c) => c.id === 9)).toHaveLength(1);

    vi.mocked(api.projects.comments.create).mockResolvedValue({ ...created, id: 10, parent: 9 });
    await store.reply(9, 'Again');
    expect(api.projects.comments.create).toHaveBeenLastCalledWith(5, { text: 'Again', parent: 9 });

    vi.mocked(api.projects.comments.update).mockResolvedValue({ ...created, text: 'Edited' });
    await store.edit(9, 'Edited');
    expect(api.projects.comments.update).toHaveBeenCalledWith(5, 9, { text: 'Edited' });
    expect(store.getState().comments.find((c) => c.id === 9)?.text).toBe('Edited');

    vi.mocked(api.projects.comments.update).mockResolvedValue({ ...created, resolved: true });
    await store.setResolved(9, true);
    expect(api.projects.comments.update).toHaveBeenLastCalledWith(5, 9, { resolved: true });

    vi.mocked(api.projects.comments.remove).mockResolvedValue(undefined);
    await store.remove(9);
    store.applyEvent({ type: 'comment', action: 'deleted', comment: { id: 9 } });
    expect(store.getState().comments.map((c) => c.id)).toEqual([1, 2, 3]);
  });

  it('comments on layers', async () => {
    const store = await openStore();
    vi.mocked(api.projects.comments.create).mockResolvedValue(
      comment({ id: 7, node_id: 'in', author: ALICE })
    );
    await store.add('On the input', 'in');
    expect(api.projects.comments.create).toHaveBeenCalledWith(5, {
      text: 'On the input',
      node_id: 'in',
    });
    expect(store.getState().openByLayer.in).toBe(1);
  });

  it('keeps changes that arrive while the comments load', async () => {
    const store = await openStore();
    const pending = deferred<ProjectComment[]>();
    vi.mocked(api.projects.comments.list).mockReturnValueOnce(pending.promise);
    const loading = store.load();

    const live = comment({ id: 8, author: BOB });
    store.applyEvent({ type: 'comment', action: 'created', comment: live });
    store.applyEvent({ type: 'comment', action: 'deleted', comment: { id: 3 } });
    // The answer was made before these changes.
    pending.resolve([THREAD, REPLY, GENERAL]);
    await loading;
    expect(store.getState().comments.map((c) => c.id)).toEqual([1, 2, 8]);
  });

  it('drops answers for a project that is no longer open', async () => {
    const store = await openStore();
    const pending = deferred<ProjectComment[]>();
    vi.mocked(api.projects.comments.list).mockReturnValueOnce(pending.promise);
    void store.load();
    vi.mocked(api.projects.comments.list).mockResolvedValue([]);
    store.configure({ projectId: 6, userId: ALICE.id, role: 'viewer' });
    pending.resolve([THREAD]);
    await waitFor(() => expect(store.getState().loaded).toBe(true));
    expect(store.getState()).toMatchObject({ projectId: 6, comments: [], role: 'viewer' });

    store.configure({ projectId: null, userId: ALICE.id, role: null });
    expect(store.getState()).toMatchObject({ projectId: null, comments: [], loaded: false });
  });

  it('reports a failed load', async () => {
    vi.mocked(api.projects.comments.list).mockRejectedValue(new Error('Server down'));
    const store = new CollabStore();
    store.configure({ projectId: 5, userId: 1, role: 'owner' });
    await waitFor(() => expect(store.getState().error).toBe('Server down'));
    expect(store.getState().loaded).toBe(false);
  });

  it('knows which panel is open', async () => {
    const store = await openStore();
    store.openPanel('comments', 'fc');
    expect(store.getState()).toMatchObject({ panel: 'comments', layer: 'fc' });
    store.setLayer(null);
    expect(store.getState().layer).toBeNull();
    store.openPanel('history', 'fc');
    expect(store.getState()).toMatchObject({ panel: 'history', layer: null });
    store.closePanel();
    expect(store.getState().panel).toBeNull();
  });
});
