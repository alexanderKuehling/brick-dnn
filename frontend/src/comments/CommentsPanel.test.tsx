import { act, render, screen, userEvent, waitFor, within } from '@test-utils';
import { beforeEach, vi } from 'vitest';
import { notifications } from '@mantine/notifications';
import { api, ApiError, type ProjectComment, type ProjectRole } from '@/api';
import { CollabContext } from '@/collab/context';
import { ALICE, BOB, CAROL, comment } from '@/collab/fixtures';
import { CollabStore } from '@/collab/store';
import { CommentMarker } from './CommentMarker';
import { CommentsPanel } from './CommentsPanel';
import { LayerComments } from './LayerComments';

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

const ON_LINEAR = comment({ id: 1, node_id: 'fc', author: BOB, text: 'Is 10 enough?' });
const REPLY = comment({ id: 2, parent: 1, node_id: 'fc', author: ALICE, text: 'Try 64.' });
const GENERAL = comment({ id: 3, author: ALICE, text: 'Nice network.' });
const RESOLVED = comment({
  id: 4,
  node_id: 'in',
  author: CAROL,
  text: 'Shape is wrong.',
  resolved: true,
  resolved_by: 'bob',
});
const ON_DELETED = comment({ id: 5, node_id: 'gone', author: BOB, text: 'Remove this?' });
const COMMENTS = [ON_LINEAR, REPLY, GENERAL, RESOLVED, ON_DELETED];

const LAYERS: Record<string, string> = { in: 'Input', fc: 'Linear' };

async function openStore(role: ProjectRole, comments: ProjectComment[] = COMMENTS) {
  vi.mocked(api.projects.comments.list).mockResolvedValue(comments);
  const store = new CollabStore();
  store.configure({ projectId: 5, userId: ALICE.id, role });
  await waitFor(() => expect(store.getState().loaded).toBe(true));
  return store;
}

async function renderPanel(role: ProjectRole = 'editor') {
  const store = await openStore(role);
  const onFocusLayer = vi.fn();
  render(
    <CommentsPanel
      store={store}
      layerName={(id) => LAYERS[id] ?? null}
      onFocusLayer={onFocusLayer}
    />
  );
  return { store, onFocusLayer };
}

/** The thread with this text. */
const thread = (text: string) => screen.getByText(text).closest('article') as HTMLElement;

beforeEach(() => {
  vi.clearAllMocks();
  vi.spyOn(notifications, 'show').mockImplementation(() => 'id');
});

describe('CommentsPanel', () => {
  it('shows open threads with replies and filters by state', async () => {
    await renderPanel();
    expect(screen.getByRole('radio', { name: 'Open · 3' })).toBeChecked();
    expect(screen.getByText('Is 10 enough?')).toBeInTheDocument();
    expect(within(thread('Is 10 enough?')).getByText('Try 64.')).toBeInTheDocument();
    expect(screen.getByText('Nice network.')).toBeInTheDocument();
    expect(screen.queryByText('Shape is wrong.')).not.toBeInTheDocument();

    await userEvent.click(screen.getByText('Resolved · 1'));
    expect(screen.getByText('Shape is wrong.')).toBeInTheDocument();
    expect(screen.getByText('Resolved by bob')).toBeInTheDocument();
    expect(screen.queryByText('Nice network.')).not.toBeInTheDocument();

    await userEvent.click(screen.getByText('All · 4'));
    expect(screen.getAllByRole('article')).toHaveLength(4);
  });

  it('shows the layer of a thread and opens it on the canvas', async () => {
    const { onFocusLayer } = await renderPanel();
    await userEvent.click(
      within(thread('Is 10 enough?')).getByRole('button', { name: 'Show layer Linear' })
    );
    expect(onFocusLayer).toHaveBeenCalledWith('fc');
    expect(within(thread('Remove this?')).getByText('Deleted layer')).toBeInTheDocument();
    // Comments on the whole network have no layer.
    expect(within(thread('Nice network.')).queryByText(/layer/i)).not.toBeInTheDocument();
  });

  it('adds a comment on the network', async () => {
    const created = comment({ id: 9, author: ALICE, text: 'Ship it' });
    vi.mocked(api.projects.comments.create).mockResolvedValue(created);
    await renderPanel('viewer');
    const field = screen.getByRole('textbox', { name: 'Comment on the network' });
    expect(screen.getByRole('button', { name: 'Comment' })).toBeDisabled();
    await userEvent.type(field, '  Ship it ');
    await userEvent.click(screen.getByRole('button', { name: 'Comment' }));
    expect(api.projects.comments.create).toHaveBeenCalledWith(5, { text: 'Ship it' });
    expect(await screen.findByText('Ship it')).toBeInTheDocument();
    expect(field).toHaveValue('');
  });

  it('shows why a comment was rejected and keeps the text', async () => {
    vi.mocked(api.projects.comments.create).mockRejectedValue(
      new ApiError(400, { text: ['At most 5000 characters.'] }, 'Text: At most 5000 characters.')
    );
    await renderPanel();
    const field = screen.getByRole('textbox', { name: 'Comment on the network' });
    await userEvent.type(field, 'Too long{Control>}{Enter}{/Control}');
    expect(await screen.findByText('At most 5000 characters.')).toBeInTheDocument();
    expect(field).toHaveValue('Too long');
  });

  it('replies to a thread', async () => {
    vi.mocked(api.projects.comments.create).mockResolvedValue(
      comment({ id: 10, parent: 1, node_id: 'fc', author: ALICE, text: 'Or 128.' })
    );
    await renderPanel('viewer');
    await userEvent.click(within(thread('Is 10 enough?')).getByRole('button', { name: /Reply/ }));
    await userEvent.type(screen.getByRole('textbox', { name: 'Reply' }), 'Or 128.');
    await userEvent.click(within(thread('Is 10 enough?')).getByRole('button', { name: 'Reply' }));
    expect(api.projects.comments.create).toHaveBeenCalledWith(5, { text: 'Or 128.', parent: 1 });
    expect(await within(thread('Is 10 enough?')).findByText('Or 128.')).toBeInTheDocument();
    expect(screen.queryByRole('textbox', { name: 'Reply' })).not.toBeInTheDocument();
  });

  it('edits own comments', async () => {
    vi.mocked(api.projects.comments.update).mockResolvedValue({ ...REPLY, text: 'Try 256.' });
    await renderPanel();
    // The reply of alice in bob's thread.
    await userEvent.click(
      within(thread('Is 10 enough?')).getByRole('button', {
        name: 'Actions for the comment by alice',
      })
    );
    await userEvent.click(await screen.findByRole('menuitem', { name: 'Edit' }));
    const field = screen.getByRole('textbox', { name: 'Edit comment' });
    expect(field).toHaveValue('Try 64.');
    await userEvent.clear(field);
    await userEvent.type(field, 'Try 256.');
    await userEvent.click(screen.getByRole('button', { name: 'Save' }));
    expect(api.projects.comments.update).toHaveBeenCalledWith(5, 2, { text: 'Try 256.' });
    expect(await screen.findByText('Try 256.')).toBeInTheDocument();
  });

  it('deletes own comments after asking', async () => {
    vi.mocked(api.projects.comments.remove).mockResolvedValue(undefined);
    await renderPanel('viewer');
    const general = thread('Nice network.');
    await userEvent.click(
      within(general).getByRole('button', { name: 'Actions for the comment by alice' })
    );
    expect(screen.queryByRole('menuitem', { name: 'Edit' })).toBeInTheDocument();
    await userEvent.click(await screen.findByRole('menuitem', { name: 'Delete' }));
    expect(within(general).getByText('Delete this comment?')).toBeInTheDocument();
    await userEvent.click(within(general).getByRole('button', { name: 'Delete' }));
    expect(api.projects.comments.remove).toHaveBeenCalledWith(5, 3);
    await waitFor(() => expect(screen.queryByText('Nice network.')).not.toBeInTheDocument());
  });

  it('lets the owner delete comments of others with their replies, not edit them', async () => {
    vi.mocked(api.projects.comments.remove).mockResolvedValue(undefined);
    await renderPanel('owner');
    const linear = thread('Is 10 enough?');
    await userEvent.click(
      within(linear).getByRole('button', { name: 'Actions for the comment by bob' })
    );
    expect(await screen.findByRole('menuitem', { name: 'Delete' })).toBeInTheDocument();
    expect(screen.queryByRole('menuitem', { name: 'Edit' })).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('menuitem', { name: 'Delete' }));
    expect(within(linear).getByText('Delete this comment and its reply?')).toBeInTheDocument();
    await userEvent.click(within(linear).getByRole('button', { name: 'Delete' }));
    await waitFor(() => expect(screen.queryByText('Try 64.')).not.toBeInTheDocument());
  });

  it('lets editors resolve and reopen threads of others', async () => {
    vi.mocked(api.projects.comments.update).mockResolvedValue({
      ...ON_LINEAR,
      resolved: true,
      resolved_by: 'alice',
    });
    await renderPanel('editor');
    // Editors cannot delete comments of others.
    expect(
      within(thread('Is 10 enough?')).queryByRole('button', {
        name: 'Actions for the comment by bob',
      })
    ).not.toBeInTheDocument();
    await userEvent.click(
      within(thread('Is 10 enough?')).getByRole('button', { name: 'Resolve thread' })
    );
    expect(api.projects.comments.update).toHaveBeenCalledWith(5, 1, { resolved: true });
    // Resolved threads leave the "Open" list.
    await waitFor(() => expect(screen.queryByText('Is 10 enough?')).not.toBeInTheDocument());

    vi.mocked(api.projects.comments.update).mockResolvedValue({ ...RESOLVED, resolved: false });
    await userEvent.click(screen.getByText('Resolved · 2'));
    await userEvent.click(
      within(thread('Shape is wrong.')).getByRole('button', { name: 'Reopen thread' })
    );
    expect(api.projects.comments.update).toHaveBeenLastCalledWith(5, 4, { resolved: false });
  });

  it('lets viewers resolve only their own threads', async () => {
    await renderPanel('viewer');
    expect(
      within(thread('Is 10 enough?')).queryByRole('button', { name: 'Resolve thread' })
    ).not.toBeInTheDocument();
    expect(
      within(thread('Nice network.')).getByRole('button', { name: 'Resolve thread' })
    ).toBeInTheDocument();
    await userEvent.click(screen.getByText('Resolved · 1'));
    expect(screen.queryByRole('button', { name: 'Reopen thread' })).not.toBeInTheDocument();
  });

  it('reports failed actions', async () => {
    vi.mocked(api.projects.comments.update).mockRejectedValue(new Error('No access.'));
    await renderPanel('editor');
    await userEvent.click(
      within(thread('Is 10 enough?')).getByRole('button', { name: 'Resolve thread' })
    );
    await waitFor(() =>
      expect(notifications.show).toHaveBeenCalledWith(
        expect.objectContaining({ color: 'red', message: 'No access.' })
      )
    );
    expect(screen.getByText('Is 10 enough?')).toBeInTheDocument();
  });

  it('shows the threads of one layer and comments on it', async () => {
    vi.mocked(api.projects.comments.create).mockResolvedValue(
      comment({ id: 11, node_id: 'fc', author: ALICE, text: 'Add dropout after it' })
    );
    const { store } = await renderPanel();
    act(() => store.openPanel('comments', 'fc'));
    expect(screen.getByText('On Linear')).toBeInTheDocument();
    expect(screen.getAllByRole('article')).toHaveLength(1);
    // The layer is clear: no layer button on the thread.
    expect(screen.queryByRole('button', { name: 'Show layer Linear' })).not.toBeInTheDocument();

    await userEvent.type(
      screen.getByRole('textbox', { name: 'Comment on Linear' }),
      'Add dropout after it'
    );
    await userEvent.click(screen.getByRole('button', { name: 'Comment' }));
    expect(api.projects.comments.create).toHaveBeenCalledWith(5, {
      text: 'Add dropout after it',
      node_id: 'fc',
    });
    expect(await screen.findAllByRole('article')).toHaveLength(2);

    await userEvent.click(screen.getByRole('button', { name: 'All comments' }));
    expect(screen.getAllByRole('article')).toHaveLength(4);
  });

  it('applies changes of others as they arrive', async () => {
    const { store } = await renderPanel();
    act(() =>
      store.applyEvent({
        type: 'comment',
        action: 'created',
        comment: comment({ id: 12, author: BOB, text: 'Hello from bob' }),
      })
    );
    expect(screen.getByText('Hello from bob')).toBeInTheDocument();
    act(() =>
      store.applyEvent({
        type: 'comment',
        action: 'updated',
        comment: { ...GENERAL, text: 'Nice network!' },
      })
    );
    expect(screen.getByText('Nice network!')).toBeInTheDocument();
    act(() => store.applyEvent({ type: 'comment', action: 'deleted', comment: { id: 1 } }));
    expect(screen.queryByText('Is 10 enough?')).not.toBeInTheDocument();
    expect(screen.queryByText('Try 64.')).not.toBeInTheDocument();
  });

  it('says when there is nothing to show', async () => {
    vi.mocked(api.projects.comments.list).mockResolvedValue([]);
    const store = new CollabStore();
    store.configure({ projectId: 5, userId: ALICE.id, role: 'owner' });
    render(<CommentsPanel store={store} layerName={() => null} onFocusLayer={vi.fn()} />);
    expect(await screen.findByText('No open comments.')).toBeInTheDocument();
  });
});

describe('Comments on the canvas and in the inspector', () => {
  function withStore(store: CollabStore | null, ui: React.ReactNode) {
    return render(<CollabContext.Provider value={store}>{ui}</CollabContext.Provider>);
  }

  it('marks layers with open threads and opens them in the panel', async () => {
    const store = await openStore('viewer');
    withStore(
      store,
      <>
        <CommentMarker nodeId="fc" />
        <CommentMarker nodeId="in" />
      </>
    );
    // "in" only has a resolved thread.
    expect(screen.getAllByTestId('comment-marker')).toHaveLength(1);
    const marker = screen.getByRole('button', { name: '1 open comment. Show them' });
    expect(marker).toHaveTextContent('1');

    act(() =>
      store.applyEvent({
        type: 'comment',
        action: 'created',
        comment: comment({ id: 13, node_id: 'fc', author: BOB }),
      })
    );
    expect(screen.getByRole('button', { name: '2 open comments. Show them' })).toBeInTheDocument();

    await userEvent.click(screen.getByTestId('comment-marker'));
    expect(store.getState()).toMatchObject({ panel: 'comments', layer: 'fc' });
  });

  it('shows nothing outside the builder', () => {
    withStore(
      null,
      <>
        <CommentMarker nodeId="fc" />
        <LayerComments nodeId="fc" />
      </>
    );
    expect(screen.queryByRole('button')).not.toBeInTheDocument();
    expect(screen.queryByText('Comments')).not.toBeInTheDocument();
  });

  it('lists the open threads of a layer in the inspector and adds one', async () => {
    vi.mocked(api.projects.comments.create).mockResolvedValue(
      comment({ id: 14, node_id: 'in', author: ALICE, text: 'Use 28×28' })
    );
    const store = await openStore('viewer');
    const { rerender } = withStore(store, <LayerComments nodeId="fc" />);
    expect(screen.getByText('Is 10 enough?')).toBeInTheDocument();
    expect(screen.queryByText('Nice network.')).not.toBeInTheDocument();

    rerender(
      <CollabContext.Provider value={store}>
        <LayerComments nodeId="in" />
      </CollabContext.Provider>
    );
    expect(screen.queryByText('Shape is wrong.')).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: /1 resolved comment/ }));
    expect(store.getState()).toMatchObject({ panel: 'comments', layer: 'in' });

    await userEvent.click(screen.getByRole('button', { name: 'Add comment' }));
    await userEvent.type(
      screen.getByRole('textbox', { name: 'Comment on this layer' }),
      'Use 28×28'
    );
    await userEvent.click(screen.getByRole('button', { name: 'Comment' }));
    expect(api.projects.comments.create).toHaveBeenCalledWith(5, {
      text: 'Use 28×28',
      node_id: 'in',
    });
    expect(await screen.findByText('Use 28×28')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Add comment' })).toBeInTheDocument();
  });
});
